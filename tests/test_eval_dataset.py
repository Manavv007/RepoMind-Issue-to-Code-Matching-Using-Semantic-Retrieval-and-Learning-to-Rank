"""Tests for the SWE-bench loader, dataset I/O, and the committed dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT
from evaluation import swebench
from evaluation.dataset import file_sha256, read_cases, write_cases
from evaluation.leakage import repos_spanning_splits

DATASETS = PROJECT_ROOT / "evaluation" / "datasets"
COMMITTED = DATASETS / "swebench_lite_subset.jsonl"
MANIFEST = DATASETS / "swebench_lite_subset.manifest.json"

ROW = {
    "instance_id": "o__r-1",
    "repo": "o/r",
    "base_commit": "a" * 40,
    "patch": "diff",
    "problem_statement": "title\nbody",
    "FAIL_TO_PASS": '["t1", "t2"]',
    "created_at": "2020-01-01T00:00:00Z",
    "version": "1.2",
}


# ----------------------------------------------------------------- SWE-bench loader
def test_parse_row_handles_json_string_and_list_fail_to_pass():
    assert swebench.parse_row(ROW, "test").fail_to_pass == ("t1", "t2")
    as_list = swebench.parse_row({**ROW, "FAIL_TO_PASS": ["x"]}, "dev")
    assert as_list.fail_to_pass == ("x",) and as_list.hf_split == "dev"
    assert swebench.parse_row(
        {**ROW, "FAIL_TO_PASS": "not json"}, "test"
    ).fail_to_pass == ("not json",)


def test_raw_cache_roundtrip(tmp_path: Path):
    insts = [
        swebench.parse_row({**ROW, "instance_id": i}, "test")
        for i in ("o__r-2", "o__r-1")
    ]
    path = tmp_path / "raw" / "cache.jsonl"
    swebench.save_raw(path, insts)
    loaded = swebench.load_raw(path)
    assert [i.instance_id for i in loaded] == ["o__r-1", "o__r-2"]  # sorted
    assert loaded[0] == swebench.parse_row({**ROW, "instance_id": "o__r-1"}, "test")


def test_fetch_instances_pages_through_all_rows(monkeypatch):
    rows = [{**ROW, "instance_id": f"o__r-{n}"} for n in range(5)]
    monkeypatch.setattr(swebench, "_PAGE_SIZE", 2)
    calls = []

    def fake_get(url, **_):
        from urllib.parse import parse_qs, urlparse

        q = parse_qs(urlparse(url).query)
        offset, length = int(q["offset"][0]), int(q["length"][0])
        calls.append((q["split"][0], offset))
        page = rows[offset : offset + length]
        return {"rows": [{"row": r} for r in page], "num_rows_total": len(rows)}

    monkeypatch.setattr(swebench, "_get_json", fake_get)
    out = swebench.fetch_instances(splits=("test",))
    assert [i.instance_id for i in out] == [f"o__r-{n}" for n in range(5)]
    assert calls == [("test", 0), ("test", 2), ("test", 4)]


def test_fetch_refuses_unexpected_hosts():
    with pytest.raises(ValueError, match="unexpected host"):
        swebench._get_json("https://evil.example/rows")


# --------------------------------------------------------------------- dataset IO
def test_duplicate_case_ids_and_bad_lines_are_rejected(tmp_path: Path):
    from tests.test_eval_runner import make_case, make_pool

    case = make_case("c1", make_pool(1, 3))
    path = tmp_path / "d.jsonl"
    path.write_text(case.model_dump_json() + "\n" + case.model_dump_json() + "\n")
    with pytest.raises(ValueError, match="duplicate case_id"):
        read_cases(path)

    path.write_text(case.model_dump_json() + "\n{not json}\n")
    with pytest.raises(ValueError, match=r"d\.jsonl:2"):
        read_cases(path)


def test_write_is_sorted_and_stable(tmp_path: Path):
    from tests.test_eval_runner import make_case, make_pool

    pool = make_pool(1, 3)
    cases = [make_case(i, pool) for i in ("b", "a", "c")]
    p1, p2 = tmp_path / "1.jsonl", tmp_path / "2.jsonl"
    write_cases(p1, cases)
    write_cases(p2, list(reversed(cases)))
    assert p1.read_bytes() == p2.read_bytes()
    assert [c.case_id for c in read_cases(p1)] == ["a", "b", "c"]
    assert file_sha256(p1) == file_sha256(p2)


# ---------------------------------------------------- committed dataset integrity
pytestmark_dataset = pytest.mark.skipif(
    not COMMITTED.exists(), reason="evaluation subset has not been built"
)


@pytestmark_dataset
def test_committed_dataset_is_well_formed():
    cases = read_cases(COMMITTED)
    assert len(cases) >= 30
    text = COMMITTED.read_text(encoding="utf-8")
    assert '"patch":' not in text  # gold patches are never stored

    for case in cases:
        assert re.fullmatch(r"[0-9a-f]{40}", case.base_commit)
        assert case.split in ("dev", "test")
        assert case.provenance.dataset == "princeton-nlp/SWE-bench_Lite"
        assert len(case.provenance.patch_sha256) == 64
        if not case.suitable:
            assert case.unsuitable_reasons
            continue
        gt = case.ground_truth
        assert gt.files, case.case_id
        assert case.provenance.patch_applies_cleanly is True
        assert case.issue.body.strip()
        for unit in gt.units:
            assert unit.file_path in gt.files
            assert unit.unit_id.startswith(f"{unit.file_path}::")
            assert 1 <= unit.start_line <= unit.end_line
            assert unit.kind in ("function", "method")


@pytestmark_dataset
def test_committed_dataset_has_no_repo_split_overlap_or_duplicates():
    cases = read_cases(COMMITTED)
    assert repos_spanning_splits(cases) == []
    dev_repos = {c.repo for c in cases if c.split == "dev"}
    test_repos = {c.repo for c in cases if c.split == "test"}
    assert dev_repos and test_repos and not (dev_repos & test_repos)
    for case in cases:
        if case.leakage.duplicate_of:
            assert not case.suitable


@pytestmark_dataset
def test_manifest_matches_dataset():
    cases = read_cases(COMMITTED)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["counts"]["processed"] == len(cases)
    assert manifest["counts"]["suitable"] == sum(c.suitable for c in cases)
    assert manifest["dataset"] == "princeton-nlp/SWE-bench_Lite"
    assert manifest["dataset_revision"]
    assert manifest["builder"]["annotation_method"] == "patch-hunk-to-function/v1"
    assert set(manifest["excluded_cases"]) == {
        c.case_id for c in cases if not c.suitable
    }
