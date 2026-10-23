"""Tune BM25 hyperparameters on the ``dev`` split only.

    python -m evaluation.tune_bm25 \\
        --dataset evaluation/datasets/swebench_lite_subset.jsonl

The held-out ``test`` split is never read: cases are filtered to ``split == "dev"``
before anything else happens. Selection criterion, fixed in advance of seeing
results: **function-level MRR** (primary), file-level MRR (tie-break), then grid
order. The full grid, the textbook configuration, and per-repository results for
the winner are written to ``evaluation/reports/bm25_tuning_dev.{json,md}`` so the
choice is auditable and its sensitivity visible.

With ~20 dev cases the grid is deliberately small and the chosen values should be
treated as a mild improvement to be confirmed on ``test``, not as a precise optimum.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, get_settings
from app.retrieval.bm25 import BM25Config, BM25Retriever
from app.schemas.code_unit import CodeUnit
from app.storage.index_store import IndexStore, sqlite_path_from_url
from evaluation.dataset import file_sha256, read_cases
from evaluation.run_retrieval_eval import IndexedUnitsProvider, run_evaluation
from evaluation.schema import EvalCase

GRID: dict[str, list[Any]] = {
    "k1": [0.9, 1.5],
    "b": [0.5, 0.75, 0.9],
    "w_name": [1.0, 3.0],
    "w_path": [1.0, 2.0],
    "query_tf": ["binary", "log", "raw"],
}
CLASSIC = BM25Config(  # textbook, explicit so it never follows the tuned defaults
    k1=1.2,
    b=0.75,
    w_name=1.0,
    w_path=1.0,
    w_doc=1.0,
    w_body=1.0,
    query_tf="raw",
)
SPLIT = "dev"


def grid_configs() -> list[dict[str, Any]]:
    keys = list(GRID)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*GRID.values())
    ]


def evaluate_config(
    cases: Sequence[EvalCase],
    pools: dict[str, list[CodeUnit]],
    params: dict[str, Any],
    *,
    n_resamples: int = 10,
) -> dict[str, Any]:
    """Score one configuration on the given (dev) cases."""
    report = run_evaluation(
        cases,
        BM25Retriever(BM25Config(**{**CLASSIC.model_dump(), **params})),
        lambda case: pools[case.case_id],
        split=SPLIT,
        n_resamples=n_resamples,
    )
    metrics = report["aggregate"]["function"]["metrics"]
    file_metrics = report["aggregate"]["file"]["metrics"]
    per_repo: dict[str, list[float]] = defaultdict(list)
    for row in report["cases"]:
        if row["function_metrics"] is not None:
            per_repo[row["repo"]].append(row["function_metrics"]["mrr"])
    return {
        "params": params,
        "function": {
            k: metrics[k]["mean"] for k in ("hit@5", "hit@10", "mrr", "ndcg@10")
        },
        "file": {
            k: file_metrics[k]["mean"] for k in ("hit@5", "hit@10", "mrr", "ndcg@10")
        },
        "function_mrr_by_repo": {
            r: sum(v) / len(v) for r, v in sorted(per_repo.items())
        },
        "n_function_cases": metrics["mrr"]["n"],
        "n_file_cases": file_metrics["mrr"]["n"],
        "evaluated": report["counts"]["evaluated"],
    }


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by the pre-registered criterion; ties keep grid order (stable sort)."""
    return sorted(
        results,
        key=lambda r: (-round(r["function"]["mrr"], 12), -round(r["file"]["mrr"], 12)),
    )


def sensitivity(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean function MRR over all configs sharing each hyperparameter value."""
    out: dict[str, dict[str, float]] = {}
    for key in GRID:
        buckets: dict[str, list[float]] = defaultdict(list)
        for r in results:
            buckets[str(r["params"][key])].append(r["function"]["mrr"])
        out[key] = {value: sum(v) / len(v) for value, v in buckets.items()}
    return out


def render_markdown(summary: dict[str, Any]) -> str:
    best, classic = summary["best"], summary["classic"]
    lines = [
        "# BM25 hyperparameter tuning (dev split only)",
        "",
        f"- Dataset: `{summary['dataset']}` "
        f"(sha256 `{summary['dataset_sha256'][:12]}...`)",
        f"- Cases: {summary['n_cases']} `dev` cases; held-out `test` was **not** read",
        f"- Grid: {summary['n_configs']} configurations: "
        + ", ".join(f"{k} in {v}" for k, v in summary["grid"].items()),
        "- Selection (fixed in advance): function-level MRR, then file-level MRR, "
        "then grid order",
        "",
        "## Chosen configuration",
        "",
        f"`{json.dumps(best['params'], sort_keys=True)}`",
        "",
        "| Config | fn MRR | fn hit@5 | fn hit@10 | fn nDCG@10 | file MRR | "
        "file hit@10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, r in (("chosen", best), ("classic (k1=1.2, b=0.75, raw)", classic)):
        f, fl = r["function"], r["file"]
        lines.append(
            f"| {label} | {f['mrr']:.3f} | {f['hit@5']:.3f} | {f['hit@10']:.3f} | "
            f"{f['ndcg@10']:.3f} | {fl['mrr']:.3f} | {fl['hit@10']:.3f} |"
        )
    lines += ["", "Function-level MRR by repository (dev):", ""]
    lines += ["| Repo | chosen | classic |", "|---|---|---|"]
    for repo in best["function_mrr_by_repo"]:
        lines.append(
            f"| {repo} | {best['function_mrr_by_repo'][repo]:.3f} | "
            f"{classic['function_mrr_by_repo'].get(repo, float('nan')):.3f} |"
        )
    lines += ["", "## Sensitivity (mean function MRR over configs sharing a value)", ""]
    for key, values in summary["sensitivity"].items():
        lines.append(
            f"- **{key}**: " + ", ".join(f"{v} -> {m:.3f}" for v, m in values.items())
        )
    spread = summary["spread"]
    lines += [
        "",
        f"Function MRR across the grid: best {spread['best']:.3f}, median "
        f"{spread['median']:.3f}, worst {spread['worst']:.3f}.",
        "",
        "## Top 10 configurations",
        "",
        "| # | params | fn MRR | fn hit@10 | file MRR |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(summary["top"], start=1):
        lines.append(
            f"| {i} | `{json.dumps(r['params'], sort_keys=True)}` | "
            f"{r['function']['mrr']:.3f} | {r['function']['hit@10']:.3f} | "
            f"{r['file']['mrr']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _utf8_stdout() -> None:
    """Avoid UnicodeEncodeError on consoles with legacy code pages."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.tune_bm25", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--db", type=Path, help="SQLite index path (default: settings)")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "reports"
    )
    args = parser.parse_args(argv)
    _utf8_stdout()

    settings = get_settings()
    store = IndexStore(args.db or sqlite_path_from_url(settings.database_url))
    # Only dev cases are ever loaded past this point.
    cases = [c for c in read_cases(args.dataset) if c.suitable and c.split == SPLIT]
    if not cases:
        print("error: no suitable dev cases in the dataset", file=sys.stderr)
        return 2

    provider = IndexedUnitsProvider(settings, store)
    pools = {c.case_id: list(provider(c)) for c in cases}

    configs = grid_configs()
    results = []
    for n, params in enumerate(configs, start=1):
        results.append(evaluate_config(cases, pools, params))
        if n % 10 == 0 or n == len(configs):
            print(
                f"[{n}/{len(configs)}] configs evaluated", file=sys.stderr, flush=True
            )

    ranked = rank_results(results)
    classic = evaluate_config(cases, pools, CLASSIC.model_dump())
    mrrs = sorted(r["function"]["mrr"] for r in results)
    summary = {
        "dataset": args.dataset.name,
        "dataset_sha256": file_sha256(args.dataset),
        "split": SPLIT,
        "n_cases": len(cases),
        "n_configs": len(configs),
        "grid": GRID,
        "selection_criterion": "function MRR, then file MRR, then grid order",
        "best": ranked[0],
        "classic": classic,
        "top": ranked[:10],
        "sensitivity": sensitivity(results),
        "spread": {"best": mrrs[-1], "median": mrrs[len(mrrs) // 2], "worst": mrrs[0]},
        "all_results": ranked,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bm25_tuning_dev.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(summary)
    (args.output_dir / "bm25_tuning_dev.md").write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
