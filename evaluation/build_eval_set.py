"""Build an evaluation set from SWE-bench Lite.

    python -m evaluation.build_eval_set --repos psf/requests pallets/flask \\
        --output evaluation/datasets/swebench_lite_subset.jsonl

For every selected instance the builder:

1. indexes the repository at the **pre-fix** ``base_commit`` (never the fix),
2. derives file/function labels from the gold patch against that index,
3. checks the gold patch applies cleanly at ``base_commit`` (`git apply --check`),
4. records leakage diagnostics (patch lines quoted in the issue, duplicates),
5. assigns a repository-aware split,

and writes the cases plus a manifest documenting provenance, exclusions, and
configuration. Unsuitable cases are kept in the file with reasons, so exclusions
are auditable; the evaluation runner skips them.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, Settings, get_settings
from app.indexing.pipeline import index_repository
from app.ingestion.errors import IngestionError
from app.storage.index_store import IndexStore, sqlite_path_from_url
from evaluation.candidates import POOL_RULE, is_test_path
from evaluation.dataset import write_cases
from evaluation.env_info import platform_summary
from evaluation.labeler import label_patch
from evaluation.leakage import (
    DedupItem,
    assign_splits,
    count_patch_lines_in_issue,
    find_duplicates,
    mentions_gold_file,
    repos_spanning_splits,
    sha256_text,
)
from evaluation.patch_parser import PatchParseError, parse_unified_diff
from evaluation.schema import (
    EvalCase,
    GroundTruth,
    IssueInput,
    LeakageFlags,
    Provenance,
)
from evaluation.swebench import (
    DATASET_ID,
    DATASET_LICENSE_NOTE,
    SweInstance,
    fetch_dataset_revision,
    fetch_instances,
    load_raw,
    save_raw,
)

DEFAULT_RAW = PROJECT_ROOT / "evaluation" / "datasets" / "raw" / "swebench_lite.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "datasets" / "swebench_lite_subset.jsonl"


def patch_applies(checkout: Path, patch: str) -> bool | None:
    """Does ``git apply --check`` accept the patch on the checkout? (None = unknown)."""
    try:
        proc = subprocess.run(
            ["git", "-c", "core.autocrlf=false", "apply", "--check", "-"],
            cwd=checkout,
            input=patch.encode("utf-8"),  # bytes: avoid newline translation on Windows
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.returncode == 0


def _unsuitable_case(
    inst: SweInstance, revision: str | None, reasons: Sequence[str]
) -> EvalCase:
    return EvalCase(
        case_id=inst.instance_id,
        repo=inst.repo,
        repo_url=f"https://github.com/{inst.repo}",
        base_commit=inst.base_commit,
        issue=IssueInput(body=inst.problem_statement),
        ground_truth=GroundTruth(),
        fail_to_pass=inst.fail_to_pass,
        suitable=False,
        unsuitable_reasons=tuple(reasons),
        provenance=_provenance(inst, revision, None),
    )


def _provenance(
    inst: SweInstance, revision: str | None, applies: bool | None
) -> Provenance:
    return Provenance(
        dataset=DATASET_ID,
        dataset_revision=revision,
        instance_id=inst.instance_id,
        license_note=DATASET_LICENSE_NOTE,
        patch_sha256=sha256_text(inst.patch),
        patch_applies_cleanly=applies,
    )


def build_case(
    inst: SweInstance,
    *,
    settings: Settings,
    store: IndexStore,
    revision: str | None,
    source_override: str | None = None,
) -> EvalCase:
    """Index the pre-fix revision and derive a labeled, validated case.

    ``source_override`` indexes a local mirror instead of GitHub (for tests and
    offline use); the case still records the canonical GitHub URL.
    """
    repo_url = f"https://github.com/{inst.repo}"
    try:
        result = index_repository(
            source_override or repo_url,
            commit=inst.base_commit,
            settings=settings,
            store=store,
        )
    except IngestionError as exc:
        return _unsuitable_case(inst, revision, [f"index_failed:{exc.code}"])

    repo_id = result.repository.repo_id
    if result.repository.commit != inst.base_commit:
        return _unsuitable_case(inst, revision, ["base_commit_mismatch"])

    units = store.list_units(repo_id)
    file_records = {f.path: f for f in store.list_files(repo_id)}
    units_by_file: dict[str, list[Any]] = {}
    for unit in units:
        units_by_file.setdefault(unit.file_path, []).append(unit)

    try:
        truth = label_patch(inst.patch, units_by_file, set(file_records))
        added_lines = [
            line for fp in parse_unified_diff(inst.patch) for line in fp.added_lines
        ]
    except PatchParseError:
        return _unsuitable_case(inst, revision, ["patch_unparseable"])

    applies = patch_applies(result.checkout_path, inst.patch)

    reasons: list[str] = []
    if not inst.problem_statement.strip():
        reasons.append("empty_issue")
    if not truth.has_file_labels:
        reasons.append("no_python_file_modified")
    if any(file_records[f].has_syntax_errors for f in truth.files):
        reasons.append("gold_file_has_syntax_errors")
    if any(is_test_path(f) for f in truth.files):
        reasons.append("gold_in_test_path")
    if applies is False:
        reasons.append("patch_does_not_apply")

    hints = count_patch_lines_in_issue(inst.problem_statement, added_lines)
    leakage = LeakageFlags(
        patch_lines_in_issue=hints,
        issue_contains_patch_lines=hints > 0,
        issue_mentions_gold_file=mentions_gold_file(
            inst.problem_statement, truth.files
        ),
    )
    return EvalCase(
        case_id=inst.instance_id,
        repo=inst.repo,
        repo_url=repo_url,
        base_commit=inst.base_commit,
        issue=IssueInput(body=inst.problem_statement),
        ground_truth=truth,
        fail_to_pass=inst.fail_to_pass,
        suitable=not reasons,
        unsuitable_reasons=tuple(reasons),
        leakage=leakage,
        provenance=_provenance(inst, revision, applies),
    )


def finalize_cases(
    cases: list[EvalCase],
    patch_hashes: dict[str, str],
    *,
    heldout_repos: Sequence[str] | None,
    heldout_fraction: float,
    seed: int,
    exclude_patch_leaks: bool,
    dedup_threshold: float,
) -> list[EvalCase]:
    """Apply dedup, optional patch-leak exclusion, and repo-aware splits."""
    items = [
        DedupItem(c.case_id, c.issue.body, patch_hashes[c.case_id])
        for c in cases
        if c.suitable
    ]
    duplicates = find_duplicates(items, threshold=dedup_threshold)
    splits = assign_splits(
        (c.repo for c in cases),
        heldout_repos=heldout_repos,
        heldout_fraction=heldout_fraction,
        seed=seed,
    )

    final: list[EvalCase] = []
    for case in cases:
        reasons = list(case.unsuitable_reasons)
        leakage = case.leakage
        if case.case_id in duplicates:
            reasons.append(f"duplicate_of:{duplicates[case.case_id]}")
            leakage = leakage.model_copy(
                update={"duplicate_of": duplicates[case.case_id]}
            )
        if exclude_patch_leaks and leakage.issue_contains_patch_lines:
            reasons.append("issue_contains_patch_lines")
        final.append(
            case.model_copy(
                update={
                    "split": splits[case.repo],
                    "leakage": leakage,
                    "unsuitable_reasons": tuple(reasons),
                    "suitable": case.suitable and not reasons,
                }
            )
        )
    overlap = repos_spanning_splits(final)
    if overlap:  # defensive: assign_splits is per-repo, so this should never trigger
        raise RuntimeError(f"Repositories span multiple splits: {overlap}")
    return final


def build_manifest(
    cases: Sequence[EvalCase],
    args: argparse.Namespace,
    revision: str | None,
    settings: Settings,
) -> dict[str, Any]:
    suitable = [c for c in cases if c.suitable]
    reasons = Counter(r.split(":")[0] for c in cases for r in c.unsuitable_reasons)
    by_split: dict[str, Counter[str]] = {"dev": Counter(), "test": Counter()}
    for c in suitable:
        if c.split:
            by_split[c.split][c.repo] += 1
    return {
        "dataset": DATASET_ID,
        "dataset_revision": revision,
        "license_note": DATASET_LICENSE_NOTE,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "builder": {
            "repos": args.repos,
            "instances": args.instances,
            "limit": args.limit,
            "heldout_repos": args.heldout_repos,
            "heldout_fraction": args.heldout_fraction,
            "seed": args.seed,
            "dedup_jaccard_threshold": args.dedup_threshold,
            "exclude_patch_leaks": args.exclude_patch_leaks,
            "annotation_method": "patch-hunk-to-function/v1",
            "candidate_pool": POOL_RULE,
            "index": {
                "languages": ["python"],
                "include_nested_functions": False,
                "max_file_size_kb": settings.max_file_size_kb,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform_summary(),
        },
        "counts": {
            "processed": len(cases),
            "suitable": len(suitable),
            "unsuitable": len(cases) - len(suitable),
            "unsuitable_reasons": dict(reasons),
            "with_function_labels": sum(
                c.ground_truth.has_function_labels for c in suitable
            ),
            "issue_contains_patch_lines": sum(
                c.leakage.issue_contains_patch_lines for c in suitable
            ),
            "issue_mentions_gold_file": sum(
                c.leakage.issue_mentions_gold_file for c in suitable
            ),
            "mean_gold_units": (
                sum(len(c.ground_truth.units) for c in suitable) / len(suitable)
                if suitable
                else 0
            ),
            "suitable_by_split_and_repo": {k: dict(v) for k, v in by_split.items()},
        },
        "excluded_cases": {
            c.case_id: list(c.unsuitable_reasons) for c in cases if not c.suitable
        },
    }


def load_instances(args: argparse.Namespace) -> tuple[list[SweInstance], str | None]:
    raw_path: Path = args.raw
    revision: str | None = None
    if raw_path.exists() and not args.refresh:
        instances = load_raw(raw_path)
        print(
            f"Loaded {len(instances)} cached instances from {raw_path}", file=sys.stderr
        )
    else:
        print("Downloading SWE-bench Lite from Hugging Face ...", file=sys.stderr)
        instances = fetch_instances()
        save_raw(raw_path, instances)
        revision = fetch_dataset_revision()
        (raw_path.parent / "swebench_lite.revision.txt").write_text(
            revision or "", encoding="utf-8"
        )
    rev_file = raw_path.parent / "swebench_lite.revision.txt"
    if revision is None and rev_file.exists():
        revision = rev_file.read_text(encoding="utf-8").strip() or None
    return instances, revision


def select(
    instances: Sequence[SweInstance], args: argparse.Namespace
) -> list[SweInstance]:
    chosen = list(instances)
    if args.repos:
        wanted = set(args.repos)
        chosen = [i for i in chosen if i.repo in wanted]
    if args.instances:
        wanted_ids = set(args.instances)
        chosen = [i for i in chosen if i.instance_id in wanted_ids]
    if args.limit:
        chosen = chosen[: args.limit]
    return chosen


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evaluation.build_eval_set",
        description="Build a labeled localization eval set from SWE-bench Lite.",
    )
    p.add_argument("--repos", nargs="+", help="Only these 'owner/repo' (default: all)")
    p.add_argument("--instances", nargs="+", help="Only these instance ids")
    p.add_argument("--limit", type=int, help="Process at most N instances")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--raw", type=Path, default=DEFAULT_RAW, help="Local cache of raw rows"
    )
    p.add_argument("--refresh", action="store_true", help="Re-download the raw rows")
    p.add_argument("--db", type=Path, help="SQLite index path (default: settings)")
    p.add_argument("--heldout-repos", nargs="+", help="Repos forming the test split")
    p.add_argument("--heldout-fraction", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--checkpoint",
        type=Path,
        help="Append each built case here and skip cases already in it (resume)",
    )
    p.add_argument("--dedup-threshold", type=float, default=0.8)
    p.add_argument(
        "--exclude-patch-leaks",
        action="store_true",
        help="Mark cases whose issue quotes lines of the fix as unsuitable",
    )
    return p


def load_checkpoint(path: Path | None, wanted: set[str]) -> dict[str, EvalCase]:
    """Cases already built by an interrupted run (pre-finalization, one per line).

    A truncated last line (crash mid-write) is ignored and rebuilt.
    """
    if path is None or not path.exists():
        return {}
    done: dict[str, EvalCase] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                case = EvalCase.model_validate_json(line)
            except ValueError:
                continue
            if case.case_id in wanted:
                done[case.case_id] = case
    return done


def append_checkpoint(path: Path | None, case: EvalCase) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(case.model_dump_json() + "\n")
        handle.flush()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    store = IndexStore(args.db or sqlite_path_from_url(settings.database_url))

    instances, revision = load_instances(args)
    chosen = select(instances, args)
    if not chosen:
        print("error: no instances matched the selection", file=sys.stderr)
        return 2

    cases: list[EvalCase] = []
    patch_hashes: dict[str, str] = {}
    done = load_checkpoint(args.checkpoint, {i.instance_id for i in chosen})
    if done:
        print(
            f"Resuming: {len(done)} cases loaded from {args.checkpoint}",
            file=sys.stderr,
        )
    for n, inst in enumerate(chosen, start=1):
        case = done.get(inst.instance_id)
        if case is None:
            print(
                f"[{n}/{len(chosen)}] {inst.instance_id}", file=sys.stderr, flush=True
            )
            case = build_case(inst, settings=settings, store=store, revision=revision)
            append_checkpoint(args.checkpoint, case)
        cases.append(case)
        patch_hashes[case.case_id] = case.provenance.patch_sha256

    final = finalize_cases(
        cases,
        patch_hashes,
        heldout_repos=args.heldout_repos,
        heldout_fraction=args.heldout_fraction,
        seed=args.seed,
        exclude_patch_leaks=args.exclude_patch_leaks,
        dedup_threshold=args.dedup_threshold,
    )
    write_cases(args.output, final)
    manifest = build_manifest(final, args, revision, settings)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    counts = manifest["counts"]
    print(
        f"\nWrote {len(final)} cases to {args.output}\n"
        f"  suitable: {counts['suitable']}  unsuitable: {counts['unsuitable']} "
        f"{counts['unsuitable_reasons']}\n"
        f"  with function labels: {counts['with_function_labels']}\n"
        f"  manifest: {manifest_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
