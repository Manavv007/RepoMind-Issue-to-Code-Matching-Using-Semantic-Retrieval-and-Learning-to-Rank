"""Run a retriever over an evaluation set and write JSON + Markdown reports.

    python -m evaluation.run_retrieval_eval --dataset evaluation/datasets/<set>.jsonl \\
        --retriever random --split test

Protocol (identical for every retriever so results are comparable):

1. Select suitable cases from the requested split.
2. For each case, index the repository at its **pre-fix** ``base_commit`` and build
   the label-independent candidate pool (leaf units outside test files).
3. Check that every gold unit exists in that pool; otherwise the case is excluded
   and reported (``label_mismatch``).
4. Give the retriever *only* the issue text and the pool, then validate that every
   returned candidate is a real, unique pool member with consecutive ranks.
5. Score file-level and function-level rankings; aggregate with bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, Settings, get_settings
from app.indexing.pipeline import index_repository, make_repo_id
from app.retrieval.base import RankedCandidate, RetrievalQuery, Retriever
from app.retrieval.dense import DenseUnavailableError
from app.retrieval.registry import (
    available_retrievers,
    create_retriever,
    load_params,
)
from app.schemas.code_unit import CodeUnit
from app.storage.index_store import IndexStore, sqlite_path_from_url
from evaluation import metrics as m
from evaluation.candidates import CANDIDATE_KINDS, POOL_RULE, candidate_units
from evaluation.dataset import file_sha256, read_cases
from evaluation.env_info import platform_summary
from evaluation.leakage import sha256_text
from evaluation.schema import EvalCase

UnitsProvider = Callable[[EvalCase], list[CodeUnit]]

DEFINITIONS = {
    "hit@k": "1 if at least one relevant item is in the top k (project.md 'Recall@k').",
    "recall@k": "Fraction of the case's relevant items found in the top k.",
    "acc@k": "1 if all of the case's relevant items are in the top k "
    "(Acc@k in Agentless/LocAgent/SweRank).",
    "mrr": "Reciprocal rank of the first relevant item in the returned list "
    "(0 if absent), averaged over cases.",
    "ndcg@k": "Binary-relevance nDCG over the top k.",
    "file level": "The ranked units are collapsed to distinct files (first "
    "occurrence kept); relevant files are the Python files the fix modifies.",
    "function level": "Relevant units are the functions/methods the fix modifies, "
    "evaluated only on cases that have at least one such unit.",
}


# ------------------------------------------------------------------ unit provider
class IndexedUnitsProvider:
    """Indexes each case's pre-fix revision (cached in SQLite) and returns the pool."""

    def __init__(self, settings: Settings, store: IndexStore) -> None:
        self._settings = settings
        self._store = store
        self._last: tuple[tuple[str, str], list[CodeUnit]] | None = None

    def __call__(self, case: EvalCase) -> list[CodeUnit]:
        key = (case.repo, case.base_commit)
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        repo_id = make_repo_id(f"github.com/{case.repo}".lower(), case.base_commit)
        if self._store.get_repository(repo_id) is None:
            result = index_repository(
                case.repo_url,
                commit=case.base_commit,
                settings=self._settings,
                store=self._store,
            )
            repo_id = result.repository.repo_id
        pool = candidate_units(self._store.list_units(repo_id, kinds=CANDIDATE_KINDS))
        self._last = (key, pool)
        return pool


# ----------------------------------------------------------------------- scoring
def validate_candidates(
    candidates: Sequence[RankedCandidate], pool_ids: set[str], depth: int
) -> None:
    """Reject results that are not real, unique, correctly ranked pool members."""
    if len(candidates) > depth:
        raise ValueError(f"retriever returned {len(candidates)} > depth {depth}")
    seen: set[str] = set()
    for expected_rank, cand in enumerate(candidates, start=1):
        if cand.rank != expected_rank:
            raise ValueError(f"rank {cand.rank} at position {expected_rank}")
        if cand.unit_id not in pool_ids:
            raise ValueError(f"candidate {cand.unit_id!r} is not in the indexed pool")
        if cand.unit_id in seen:
            raise ValueError(f"duplicate candidate {cand.unit_id!r}")
        seen.add(cand.unit_id)


def _pool_digest(pool: Sequence[CodeUnit]) -> str:
    material = "\n".join(sorted(f"{u.unit_id}:{u.content_hash}" for u in pool))
    return sha256_text(material)[:16]


def _evaluate_case(
    case: EvalCase,
    pool: list[CodeUnit],
    retriever: Retriever,
    *,
    prepared_key: tuple[str, str] | None,
    depth: int,
    cutoffs: Sequence[int],
    ndcg_k: int,
) -> tuple[dict[str, Any], tuple[str, str]]:
    key = (case.repo, case.base_commit)
    prepare_seconds = 0.0
    if key != prepared_key:
        start = time.perf_counter()
        retriever.prepare(pool)
        prepare_seconds = time.perf_counter() - start

    query = RetrievalQuery(
        query_id=case.case_id, issue_title=case.issue.title, issue_body=case.issue.body
    )
    start = time.perf_counter()
    candidates = retriever.retrieve(query, depth)
    retrieve_seconds = time.perf_counter() - start

    validate_candidates(candidates, {u.unit_id for u in pool}, depth)

    gold = case.ground_truth
    ranked_units = [c.unit_id for c in candidates]
    ranked_files = m.dedupe_preserve_order([c.file_path for c in candidates])
    pool_files = {u.file_path for u in pool}

    result: dict[str, Any] = {
        "case_id": case.case_id,
        "repo": case.repo,
        "split": case.split,
        "status": "ok",
        "pool_size": len(pool),
        "pool_sha": _pool_digest(pool),
        "gold_files": len(gold.files),
        "gold_units": len(gold.units),
        "gold_files_without_candidates": sum(
            1 for f in gold.files if f not in pool_files
        ),
        "retrieve_seconds": retrieve_seconds,
        "prepare_seconds": prepare_seconds,
        "first_relevant_file_rank": m.rank_of_first(ranked_files, gold.files),
        "first_relevant_unit_rank": (
            m.rank_of_first(ranked_units, [u.unit_id for u in gold.units])
            if gold.units
            else None
        ),
        "file_metrics": m.score_ranking(
            ranked_files, gold.files, cutoffs=cutoffs, ndcg_k=ndcg_k
        ),
        "function_metrics": (
            m.score_ranking(
                ranked_units,
                [u.unit_id for u in gold.units],
                cutoffs=cutoffs,
                ndcg_k=ndcg_k,
            )
            if gold.units
            else None
        ),
        "top_candidates": [
            {"rank": c.rank, "unit_id": c.unit_id, "score": c.score}
            for c in candidates[:10]
        ],
    }
    return result, key


def aggregate(
    results: Sequence[dict[str, Any]], *, n_resamples: int, seed: int
) -> dict[str, Any]:
    """Aggregate per-case metrics with bootstrap CIs and explicit denominators."""
    ok = [r for r in results if r["status"] == "ok"]
    out: dict[str, Any] = {}
    for level, field in (("file", "file_metrics"), ("function", "function_metrics")):
        rows = [r[field] for r in ok if r[field] is not None]
        level_out: dict[str, Any] = {"n_cases": len(rows), "metrics": {}}
        if rows:
            for name in rows[0]:
                summary = m.summarize(
                    [row[name] for row in rows], n_resamples=n_resamples, seed=seed
                )
                level_out["metrics"][name] = summary.to_dict() if summary else None
        out[level] = level_out
    out["latency_retrieve_seconds"] = m.latency_summary(
        [r["retrieve_seconds"] for r in ok]
    )
    prepares = [r["prepare_seconds"] for r in ok if r["prepare_seconds"] > 0]
    out["latency_prepare_seconds"] = m.latency_summary(prepares)
    sizes = [float(r["pool_size"]) for r in ok]
    out["pool_size"] = (
        {"mean": sum(sizes) / len(sizes), "median": m.percentile(sizes, 50)}
        if sizes
        else None
    )
    out["file_level_cases_with_unreachable_gold_file"] = sum(
        1 for r in ok if r["gold_files_without_candidates"] > 0
    )
    return out


def run_evaluation(
    cases: Sequence[EvalCase],
    retriever: Retriever,
    units_provider: UnitsProvider,
    *,
    split: str = "all",
    cutoffs: Sequence[int] = (5, 10),
    ndcg_k: int = 10,
    depth: int = 100,
    n_resamples: int = 1000,
    seed: int = 0,
    limit: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate ``retriever`` on ``cases`` and return the report dictionary.

    ``progress(done, total, case_id)`` is called before each case (logging only).
    """
    exclusions: Counter[str] = Counter()
    selected: list[EvalCase] = []
    for case in sorted(cases, key=lambda c: (c.repo, c.base_commit, c.case_id)):
        if not case.suitable:
            exclusions["unsuitable"] += 1
        elif split != "all" and case.split != split:
            exclusions["other_split"] += 1
        else:
            selected.append(case)
    if limit is not None:
        selected = selected[:limit]

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    prepared_key: tuple[str, str] | None = None
    for n, case in enumerate(selected):
        if progress is not None:
            progress(n, len(selected), case.case_id)
        try:
            pool = units_provider(case)
            pool_ids = {u.unit_id for u in pool}
            missing = [
                u.unit_id for u in case.ground_truth.units if u.unit_id not in pool_ids
            ]
            if missing:
                exclusions["label_mismatch"] += 1
                failures.append(
                    {
                        "case_id": case.case_id,
                        "reason": "label_mismatch",
                        "detail": missing[0],
                    }
                )
                continue
            if not pool:
                exclusions["empty_pool"] += 1
                failures.append(
                    {"case_id": case.case_id, "reason": "empty_pool", "detail": ""}
                )
                continue
            result, prepared_key = _evaluate_case(
                case,
                pool,
                retriever,
                prepared_key=prepared_key,
                depth=depth,
                cutoffs=cutoffs,
                ndcg_k=ndcg_k,
            )
            results.append(result)
        except Exception as exc:  # one bad case must not abort the whole run
            exclusions["error"] += 1
            failures.append(
                {
                    "case_id": case.case_id,
                    "reason": "error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": _environment(),
        "config": {
            "retriever": retriever.name,
            "retriever_params": retriever.describe(),
            "seed": seed,
            "split": split,
            "cutoffs": list(cutoffs),
            "ndcg_k": ndcg_k,
            "depth": depth,
            "bootstrap_resamples": n_resamples,
            "candidate_pool": POOL_RULE,
            "limit": limit,
        },
        "counts": {
            "cases_in_dataset": len(cases),
            "selected": len(selected),
            "evaluated": len(results),
            "excluded": dict(exclusions),
        },
        "failures": failures,
        "aggregate": aggregate(results, n_resamples=n_resamples, seed=seed),
        "definitions": DEFINITIONS,
        "cases": results,
    }


def _environment() -> dict[str, Any]:
    commit: str | None = None
    dirty: bool | None = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform_summary(),
        "repomind_commit": commit,
        "repomind_dirty": dirty,
    }


# ---------------------------------------------------------------------- markdown
def _fmt(summary: dict[str, Any] | None) -> tuple[str, str, str]:
    if summary is None:
        return "-", "-", "0"
    return (
        f"{summary['mean']:.3f}",
        f"[{summary['ci_low']:.3f}, {summary['ci_high']:.3f}]",
        str(summary["n"]),
    )


def render_markdown(report: dict[str, Any], dataset_name: str = "") -> str:
    cfg, counts, agg = report["config"], report["counts"], report["aggregate"]
    params_json = json.dumps(cfg.get("retriever_params", {}), sort_keys=True)
    lines = [
        f"# Retrieval evaluation: `{cfg['retriever']}`",
        "",
        f"- Dataset: `{dataset_name}`" if dataset_name else "- Dataset: (unnamed)",
        f"- Split: `{cfg['split']}`  |  seed: {cfg['seed']}  |  depth: {cfg['depth']}",
        f"- Candidate pool: {cfg['candidate_pool']}",
        f"- Retriever parameters: `{params_json}`",
        f"- Cases: {counts['evaluated']} evaluated of {counts['selected']} selected "
        f"({counts['cases_in_dataset']} in dataset); "
        f"excluded: {counts['excluded'] or 'none'}",
        f"- Generated: {report['generated_at']}  |  RepoMind commit: "
        f"`{(report['environment'].get('repomind_commit') or 'unknown')[:10]}`",
    ]
    if agg["pool_size"]:
        lines.append(
            f"- Pool size per case: mean {agg['pool_size']['mean']:.0f}, "
            f"median {agg['pool_size']['median']:.0f} (chance level scales with this)"
        )
    lines += [
        "",
        "## Results (mean, 95% bootstrap CI over cases, n = cases scored)",
        "",
    ]
    lines += ["| Level | Metric | Mean | 95% CI | n |", "|---|---|---|---|---|"]
    for level in ("file", "function"):
        for name, summary in agg[level]["metrics"].items():
            mean, ci, n = _fmt(summary)
            lines.append(f"| {level} | {name} | {mean} | {ci} | {n} |")
    lat = agg["latency_retrieve_seconds"]
    if lat:
        lines += [
            "",
            "## Latency (retrieval only, seconds per query)",
            "",
            f"mean {lat['mean']:.4f}, median {lat['median']:.4f}, "
            f"p95 {lat['p95']:.4f}, max {lat['max']:.4f} (n = {lat['n']})",
        ]
    prep = agg["latency_prepare_seconds"]
    if prep:
        lines.append(
            f"\nPer-revision `prepare` (index build): mean {prep['mean']:.4f}s "
            f"(n = {prep['n']})"
        )
    unreachable = agg["file_level_cases_with_unreachable_gold_file"]
    if unreachable:
        lines.append(
            f"\n{unreachable} case(s) have a gold file with no candidate units in the "
            "pool (e.g. module-level-only code); they stay in the file-level "
            "denominator."
        )
    if report["failures"]:
        lines += ["", "## Excluded or failed cases", ""]
        lines += [
            f"- `{f['case_id']}`: {f['reason']} {f['detail']}".rstrip()
            for f in report["failures"]
        ]
    lines += ["", "## Metric definitions", ""]
    lines += [f"- **{k}**: {v}" for k, v in report["definitions"].items()]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_retrieval_eval",
        description="Evaluate a retriever on a RepoMind evaluation set.",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL eval set")
    parser.add_argument("--retriever", default="random", choices=available_retrievers())
    parser.add_argument(
        "--retriever-config",
        help="Retriever parameters: an inline JSON object or a path to a JSON file",
    )
    parser.add_argument(
        "--tag",
        help="Label appended to the report file name (e.g. 'classic', 'tuned')",
    )
    parser.add_argument("--split", default="all", choices=["all", "dev", "test"])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument(
        "--depth", type=int, default=100, help="Results retrieved per case"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="Bootstrap resamples"
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases")
    parser.add_argument("--db", type=Path, help="SQLite index path (default: settings)")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "reports"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:  # legacy console code pages cannot print all text
        reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    db_path = args.db or sqlite_path_from_url(settings.database_url)
    cases = read_cases(args.dataset)
    try:
        retriever = create_retriever(
            args.retriever, seed=args.seed, params=load_params(args.retriever_config)
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except DenseUnavailableError as exc:
        print(
            f"error [dense_unavailable]: {exc}\n"
            "Dense retrieval is unavailable on this machine; BM25 still works "
            "(--retriever bm25).",
            file=sys.stderr,
        )
        return 3
    provider = IndexedUnitsProvider(settings, IndexStore(db_path))

    report = run_evaluation(
        cases,
        retriever,
        provider,
        split=args.split,
        cutoffs=args.cutoffs,
        ndcg_k=args.ndcg_k,
        depth=args.depth,
        n_resamples=args.bootstrap,
        seed=args.seed,
        limit=args.limit,
        progress=lambda n, total, cid: print(
            f"[{n + 1}/{total}] {cid}", file=sys.stderr, flush=True
        ),
    )
    report["dataset"] = {"path": args.dataset.name, "sha256": file_sha256(args.dataset)}

    run_name = args.retriever
    if args.tag:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
            print(
                "error: --tag may only contain letters, digits, '_', '.', '-'",
                file=sys.stderr,
            )
            return 2
        run_name = f"{args.retriever}-{args.tag}"
    stem = f"{args.dataset.stem}__{run_name}__{args.split}__seed{args.seed}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{stem}.json"
    md_path = args.output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(report, args.dataset.name)
    md_path.write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    sys.stdout.write(f"\nWrote {json_path}\nWrote {md_path}\n")
    return 0 if report["counts"]["evaluated"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
