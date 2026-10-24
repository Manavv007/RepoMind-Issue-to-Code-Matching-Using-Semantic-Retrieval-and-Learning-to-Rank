"""Tests for the eval-set builder: labeling, validation, leakage, and finalization."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from app.config import Settings
from app.storage.index_store import IndexStore
from evaluation.build_eval_set import (
    build_case,
    build_manifest,
    finalize_cases,
    patch_applies,
    select,
)
from evaluation.leakage import sha256_text
from evaluation.schema import (
    EvalCase,
    GroundTruth,
    IssueInput,
    LeakageFlags,
    Provenance,
)
from evaluation.swebench import SweInstance
from tests.conftest import GitRepoFactory, run_git, write_files

BASE_SRC = """\
import os


def alpha(x):
    return x + 1


def beta(items):
    total = 0
    for i in items:
        total += i
    return total
"""
FIXED_SRC = BASE_SRC.replace("    return total\n", "    return total or 0\n")


def _diff(repo: Path, base: str) -> str:
    return subprocess.run(
        ["git", "diff", base, "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _instance(repo: Path, base: str, patch: str, **overrides) -> SweInstance:
    data = {
        "instance_id": "o__r-1",
        "repo": "o/r",
        "base_commit": base,
        "patch": patch,
        "problem_statement": "beta() returns None-ish results for empty input.",
        "fail_to_pass": ("tests/test_x.py::test_beta",),
        "created_at": "2024-01-01T00:00:00Z",
        "version": "1.0",
        "hf_split": "test",
        **overrides,
    }
    return SweInstance(**data)  # type: ignore[arg-type]


@pytest.fixture
def fix_repo(tmp_path: Path, git_repo: GitRepoFactory) -> tuple[Path, str, str]:
    """A repo with a base commit and a fix commit; returns (path, base, patch)."""
    repo = tmp_path / "upstream"
    base = git_repo(repo, {"pkg/m.py": BASE_SRC, "README.md": "# r\n"}, "base")
    git_repo(repo, {"pkg/m.py": FIXED_SRC}, "fix")
    return repo, base, _diff(repo, base)


def _build(
    inst: SweInstance, repo: Path, settings: Settings, tmp_path: Path
) -> EvalCase:
    return build_case(
        inst,
        settings=settings,
        store=IndexStore(tmp_path / "b.db"),
        revision="rev123",
        source_override=str(repo),
    )


# ------------------------------------------------------------------- build_case
def test_build_case_labels_pre_fix_function(fix_repo, settings, tmp_path):
    repo, base, patch = fix_repo
    case = _build(_instance(repo, base, patch), repo, settings, tmp_path)

    assert case.suitable and case.unsuitable_reasons == ()
    assert case.base_commit == base
    assert case.ground_truth.files == ("pkg/m.py",)
    assert [u.qualified_name for u in case.ground_truth.units] == ["beta"]
    unit = case.ground_truth.units[0]
    assert (unit.start_line, unit.end_line) == (8, 12)  # pre-fix line numbers
    assert case.provenance.patch_applies_cleanly is True
    assert case.provenance.patch_sha256 == sha256_text(patch)
    assert case.provenance.dataset_revision == "rev123"
    assert case.fail_to_pass == ("tests/test_x.py::test_beta",)


def test_the_gold_patch_text_is_never_stored_in_the_case(fix_repo, settings, tmp_path):
    repo, base, patch = fix_repo
    case = _build(_instance(repo, base, patch), repo, settings, tmp_path)
    blob = case.model_dump_json()
    assert "diff --git" not in blob
    assert "return total or 0" not in blob
    assert '"patch":' not in blob


def test_leakage_flag_when_issue_quotes_the_fix(fix_repo, settings, tmp_path):
    repo, base, _ = fix_repo
    # Short added lines are ignored by design, so the fix needs a distinctive line.
    long_fix = BASE_SRC.replace(
        "    return total\n",
        "    return total if total else compute_default_total(items)\n",
    )
    write_files(repo, {"pkg/m.py": long_fix})
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "fix2")
    leaky = _instance(
        repo,
        base,
        _diff(repo, base),
        problem_statement=(
            "Use: return total if total else compute_default_total(items)"
        ),
    )
    case = _build(leaky, repo, settings, tmp_path)
    assert case.leakage.issue_contains_patch_lines
    assert case.leakage.patch_lines_in_issue == 1
    assert case.suitable  # flagged, not dropped, by default


def test_issue_mentioning_gold_file_is_flagged_informationally(
    fix_repo, settings, tmp_path
):
    repo, base, patch = fix_repo
    inst = _instance(repo, base, patch, problem_statement="Traceback in pkg/m.py beta")
    case = _build(inst, repo, settings, tmp_path)
    assert case.leakage.issue_mentions_gold_file and case.suitable


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("readme_only", "no_python_file_modified"),
        ("empty_issue", "empty_issue"),
        ("bad_context", "patch_does_not_apply"),
        ("test_file", "gold_in_test_path"),
        ("syntax_error", "gold_file_has_syntax_errors"),
    ],
)
def test_unsuitable_reasons(scenario, expected, tmp_path, settings, git_repo):
    repo = tmp_path / "up"
    files = {
        "pkg/m.py": BASE_SRC,
        "README.md": "one\n",
        "tests/test_m.py": "def test_a():\n    return 1\n",
    }
    if scenario == "syntax_error":
        files["pkg/m.py"] = "def broken(:\n    return 1\n"
    base = git_repo(repo, files)
    overrides = {}
    if scenario == "readme_only":
        git_repo(repo, {"README.md": "two\n"})
        patch = _diff(repo, base)
    elif scenario == "test_file":
        git_repo(repo, {"tests/test_m.py": "def test_a():\n    return 2\n"})
        patch = _diff(repo, base)
    elif scenario == "syntax_error":
        git_repo(repo, {"pkg/m.py": "def broken(:\n    return 2\n"})
        patch = _diff(repo, base)
    elif scenario == "bad_context":
        patch = (
            "--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -4,1 +4,1 @@\n"
            "-    return nothing_like_this\n+    return 1\n"
        )
    else:  # empty_issue
        git_repo(repo, {"pkg/m.py": FIXED_SRC})
        patch = _diff(repo, base)
        overrides["problem_statement"] = "   \n"
    case = _build(_instance(repo, base, patch, **overrides), repo, settings, tmp_path)
    assert not case.suitable
    assert expected in case.unsuitable_reasons


def test_unknown_base_commit_is_reported_not_raised(fix_repo, settings, tmp_path):
    repo, _, patch = fix_repo
    inst = _instance(repo, "f" * 40, patch)
    case = _build(inst, repo, settings, tmp_path)
    assert not case.suitable
    assert case.unsuitable_reasons == ("index_failed:repository_inaccessible",)
    assert case.ground_truth == GroundTruth()


def test_unparseable_patch_is_unsuitable(fix_repo, settings, tmp_path):
    repo, base, _ = fix_repo
    inst = _instance(
        repo, base, "--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1,9 +1,9 @@\n x\n"
    )
    case = _build(inst, repo, settings, tmp_path)
    assert case.unsuitable_reasons == ("patch_unparseable",)


def test_patch_applies_check_on_checkout(fix_repo):
    repo, base, patch = fix_repo
    run_git(repo, "-c", "core.autocrlf=false", "checkout", "-q", base)
    assert patch_applies(repo, patch) is True
    assert patch_applies(repo, patch.replace("total = 0", "total = 99")) is False
    assert (
        not (repo / "pkg" / "m.py").read_text().count("or 0")
    )  # --check did not apply it
    run_git(repo, "-c", "core.autocrlf=false", "checkout", "-q", "-")


# ------------------------------------------------------------------ finalization
def _case(cid: str, repo: str, text: str, patch: str = "p", **kw) -> EvalCase:
    return EvalCase(
        case_id=cid,
        repo=repo,
        repo_url=f"https://github.com/{repo}",
        base_commit="a" * 40,
        issue=IssueInput(body=text),
        ground_truth=GroundTruth(files=("a.py",)),
        suitable=kw.pop("suitable", True),
        leakage=kw.pop("leakage", LeakageFlags()),
        provenance=Provenance(
            dataset="d",
            instance_id=cid,
            license_note="n",
            patch_sha256=sha256_text(patch + cid),
        ),
    )


def _finalize(cases, **kw):
    hashes = {c.case_id: c.provenance.patch_sha256 for c in cases}
    defaults = {
        "heldout_repos": None,
        "heldout_fraction": 0.4,
        "seed": 0,
        "exclude_patch_leaks": False,
        "dedup_threshold": 0.8,
        **kw,
    }
    return finalize_cases(cases, hashes, **defaults)


def test_finalize_assigns_repo_aware_splits_and_dedups():
    dup_text = (
        "Requests drops the Authorization header when a redirect changes host names"
    )
    cases = [
        _case("a-1", "o/a", dup_text),
        _case("a-2", "o/a", dup_text),
        _case(
            "b-1",
            "o/b",
            "A totally different report about memory growth in the cache layer",
        ),
        _case(
            "c-1",
            "o/c",
            "Yet another unrelated issue regarding unicode handling in headers",
        ),
    ]
    final = {c.case_id: c for c in _finalize(cases, heldout_repos=["o/b"])}
    assert final["a-2"].leakage.duplicate_of == "a-1"
    assert not final["a-2"].suitable
    assert final["a-2"].unsuitable_reasons == ("duplicate_of:a-1",)
    assert final["a-1"].suitable
    assert {c.split for c in final.values() if c.repo == "o/a"} == {"dev"}
    assert final["b-1"].split == "test"
    assert all(c.split in ("dev", "test") for c in final.values())


def test_finalize_can_exclude_cases_whose_issue_quotes_the_fix():
    leaky = _case(
        "a-1",
        "o/a",
        "text one",
        leakage=LeakageFlags(issue_contains_patch_lines=True, patch_lines_in_issue=2),
    )
    clean = _case(
        "b-1", "o/b", "something wholly different here to avoid dedup entirely"
    )
    keep = {c.case_id: c.suitable for c in _finalize([leaky, clean])}
    drop = {c.case_id: c for c in _finalize([leaky, clean], exclude_patch_leaks=True)}
    assert keep == {"a-1": True, "b-1": True}
    assert not drop["a-1"].suitable
    assert "issue_contains_patch_lines" in drop["a-1"].unsuitable_reasons


def test_finalize_keeps_unsuitable_cases_for_auditing():
    bad = _case("a-1", "o/a", "text", suitable=False)
    final = _finalize(
        [bad, _case("b-1", "o/b", "other words entirely different from before")]
    )
    assert [c.case_id for c in final] == ["a-1", "b-1"]
    assert not final[0].suitable


def test_manifest_documents_provenance_and_counts():
    cases = _finalize(
        [
            _case("a-1", "o/a", "first issue about parsing trailing commas in lists"),
            _case(
                "b-1",
                "o/b",
                "second issue about caching of stale values",
                suitable=False,
            ),
        ]
    )
    args = argparse.Namespace(
        repos=["o/a"],
        instances=None,
        limit=None,
        heldout_repos=None,
        heldout_fraction=0.4,
        seed=0,
        dedup_threshold=0.8,
        exclude_patch_leaks=False,
    )
    manifest = build_manifest(cases, args, "rev1", Settings())
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["dataset_revision"] == "rev1"
    assert manifest["counts"]["processed"] == 2
    assert manifest["counts"]["suitable"] == 1
    assert (
        manifest["excluded_cases"] == {"b-1": []} or "b-1" in manifest["excluded_cases"]
    )
    assert manifest["builder"]["annotation_method"] == "patch-hunk-to-function/v1"
    assert "license_note" in manifest


def test_select_filters_by_repo_instance_and_limit():
    base = {
        "patch": "",
        "problem_statement": "x",
        "fail_to_pass": (),
        "created_at": "",
        "version": "",
        "hf_split": "test",
        "base_commit": "a" * 40,
    }
    insts = [
        SweInstance(instance_id=f"{r.replace('/', '__')}-{n}", repo=r, **base)
        for r in ("o/a", "o/b")
        for n in (1, 2)
    ]

    def ns(**kw):
        return argparse.Namespace(
            **{"repos": None, "instances": None, "limit": None, **kw}
        )

    assert len(select(insts, ns())) == 4
    assert {i.repo for i in select(insts, ns(repos=["o/b"]))} == {"o/b"}
    assert [i.instance_id for i in select(insts, ns(instances=["o__a-2"]))] == [
        "o__a-2"
    ]
    assert len(select(insts, ns(limit=3))) == 3


def test_checkpoint_resume_skips_built_cases_and_tolerates_truncation(tmp_path):
    from evaluation.build_eval_set import append_checkpoint, load_checkpoint

    ckpt = tmp_path / "ckpt.jsonl"
    a, b = _case("a-1", "o/a", "first"), _case("b-1", "o/b", "second")
    append_checkpoint(ckpt, a)
    append_checkpoint(ckpt, b)
    with ckpt.open("a", encoding="utf-8") as handle:
        handle.write(a.model_dump_json()[:40])  # crash mid-write
    assert load_checkpoint(ckpt, {"a-1", "b-1"}) == {"a-1": a, "b-1": b}
    assert set(load_checkpoint(ckpt, {"b-1"})) == {"b-1"}  # other selections ignored
    assert load_checkpoint(tmp_path / "missing.jsonl", {"a-1"}) == {}
    assert load_checkpoint(None, {"a-1"}) == {}
