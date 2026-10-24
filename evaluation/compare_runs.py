"""Paired comparison of two evaluation reports.

    python -m evaluation.compare_runs --a <baseline.json> --b <candidate.json>

Both reports must come from the same dataset file and split. For every metric the
per-case difference ``b - a`` is computed on the cases both runs scored, then
summarized with a paired percentile bootstrap CI over cases and win/tie/loss counts.
A paired test is much more sensitive than comparing two overlapping unpaired CIs,
but with few cases it still cannot show small effects: read ``n`` and the CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluation.metrics import bootstrap_ci

LEVELS = ("function", "file")
METRICS = ("hit@5", "hit@10", "acc@5", "acc@10", "mrr", "ndcg@10")
_FIELD = {"function": "function_metrics", "file": "file_metrics"}


def load_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return report


def check_comparable(a: dict[str, Any], b: dict[str, Any]) -> None:
    """Refuse to compare runs over different data, splits, or candidate pools."""
    problems = []
    if a["dataset"]["sha256"] != b["dataset"]["sha256"]:
        problems.append("different dataset files (sha256 differs)")
    for key in ("split", "candidate_pool", "depth"):
        if a["config"][key] != b["config"][key]:
            problems.append(
                f"different {key}: {a['config'][key]!r} vs {b['config'][key]!r}"
            )
    if problems:
        raise ValueError("Reports are not comparable: " + "; ".join(problems))


def paired_differences(
    a: dict[str, Any], b: dict[str, Any], level: str, metric: str
) -> tuple[list[float], list[str]]:
    """Per-case ``b - a`` on cases scored in both runs (and labeled at ``level``)."""
    field = _FIELD[level]
    rows_a = {r["case_id"]: r for r in a["cases"] if r["status"] == "ok"}
    diffs: list[float] = []
    repos: list[str] = []
    for row in b["cases"]:
        other = rows_a.get(row["case_id"])
        if row["status"] != "ok" or other is None:
            continue
        if row[field] is None or other[field] is None:
            continue
        if metric not in row[field] or metric not in other[field]:
            continue  # metric added after one of the reports was written
        diffs.append(row[field][metric] - other[field][metric])
        repos.append(row["repo"])
    return diffs, repos


def compare(
    a: dict[str, Any], b: dict[str, Any], *, n_resamples: int = 2000, seed: int = 0
) -> dict[str, Any]:
    check_comparable(a, b)
    result: dict[str, Any] = {
        "a": {
            "retriever": a["config"]["retriever"],
            "params": a["config"].get("retriever_params"),
        },
        "b": {
            "retriever": b["config"]["retriever"],
            "params": b["config"].get("retriever_params"),
        },
        "split": a["config"]["split"],
        "levels": {},
    }
    for level in LEVELS:
        level_out: dict[str, Any] = {}
        for metric in METRICS:
            diffs, repos = paired_differences(a, b, level, metric)
            if not diffs:
                level_out[metric] = None
                continue
            low, high = bootstrap_ci(diffs, n_resamples=n_resamples, seed=seed)
            by_repo: dict[str, list[float]] = defaultdict(list)
            for repo, d in zip(repos, diffs, strict=True):
                by_repo[repo].append(d)
            level_out[metric] = {
                "n": len(diffs),
                "mean_diff": sum(diffs) / len(diffs),
                "ci_low": low,
                "ci_high": high,
                "wins": sum(d > 1e-12 for d in diffs),
                "ties": sum(abs(d) <= 1e-12 for d in diffs),
                "losses": sum(d < -1e-12 for d in diffs),
                "by_repo": {r: sum(v) / len(v) for r, v in sorted(by_repo.items())},
                "significant": low > 0 or high < 0,
            }
        result["levels"][level] = level_out
    return result


def _params(side: dict[str, Any]) -> str:
    return json.dumps(side["params"], sort_keys=True)


def render_markdown(result: dict[str, Any]) -> str:
    a, b = result["a"], result["b"]
    lines = [
        f"# Paired comparison on the `{result['split']}` split",
        "",
        f"- A (baseline): `{a['retriever']}` {_params(a)}",
        f"- B (candidate): `{b['retriever']}` {_params(b)}",
        "- Difference = B - A per case; CI is a paired percentile bootstrap over "
        "cases.",
        "  `excludes 0` means the interval does not contain zero (not a multiple-"
        "comparison-corrected claim).",
        "",
        "| Level | Metric | n | Mean diff | 95% CI | W/T/L | Excludes 0 |",
        "|---|---|---|---|---|---|---|",
    ]
    for level, metrics in result["levels"].items():
        for metric, s in metrics.items():
            if s is None:
                lines.append(f"| {level} | {metric} | 0 | - | - | - | - |")
                continue
            lines.append(
                f"| {level} | {metric} | {s['n']} | {s['mean_diff']:+.3f} | "
                f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] | "
                f"{s['wins']}/{s['ties']}/{s['losses']} | "
                f"{'yes' if s['significant'] else 'no'} |"
            )
    mrr = result["levels"]["function"].get("mrr")
    if mrr:
        lines += [
            "",
            "Function-level MRR difference by repository: "
            + ", ".join(f"{repo} {d:+.3f}" for repo, d in mrr["by_repo"].items()),
        ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.compare_runs", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--a", type=Path, required=True, help="Baseline report JSON")
    parser.add_argument("--b", type=Path, required=True, help="Candidate report JSON")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Also write the Markdown here")
    args = parser.parse_args(argv)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
    try:
        result = compare(
            load_report(args.a),
            load_report(args.b),
            n_resamples=args.bootstrap,
            seed=args.seed,
        )
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    markdown = render_markdown(result)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
