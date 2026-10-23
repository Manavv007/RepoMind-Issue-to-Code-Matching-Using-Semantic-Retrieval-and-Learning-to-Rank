"""Ranking metrics, latency summaries, and bootstrap confidence intervals.

Definitions (also written into every report):

* **hit@k** — 1 if at least one relevant item is in the top *k*, else 0. This is the
  "Recall@k" of project.md §11 (a case counts if *any* known location is found).
* **recall@k** — fraction of the case's relevant items found in the top *k*.
* **MRR** — reciprocal rank of the first relevant item within the returned list
  (0 if none is returned); averaged over cases.
* **nDCG@k** — binary-relevance nDCG: DCG of the ranking over the ideal DCG for
  ``min(|relevant|, k)`` relevant items, with gain ``1 / log2(rank + 1)``.

All functions take plain id sequences, so they are independent of retriever
internals. Cases with no relevant items must be excluded by the caller; each
aggregate reports its own denominator ``n``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass


def dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    """Drop repeats, keeping the first occurrence of each item."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def rank_of_first(ranked: Sequence[str], relevant: Collection[str]) -> int | None:
    """1-based rank of the first relevant item, or None if absent."""
    rel = set(relevant)
    for i, item in enumerate(ranked, start=1):
        if item in rel:
            return i
    return None


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    rel = set(relevant)
    return 1.0 if any(item in rel for item in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("recall@k is undefined without relevant items.")
    return len(rel & set(ranked[:k])) / len(rel)


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    rank = rank_of_first(ranked, relevant)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("nDCG is undefined without relevant items.")
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, item in enumerate(ranked[:k], start=1)
        if item in rel
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal


def score_ranking(
    ranked: Sequence[str],
    relevant: Collection[str],
    *,
    cutoffs: Sequence[int] = (5, 10),
    ndcg_k: int = 10,
) -> dict[str, float]:
    """All per-case metrics for one ranked list."""
    scores: dict[str, float] = {}
    for k in cutoffs:
        scores[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
        scores[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
    scores["mrr"] = reciprocal_rank(ranked, relevant)
    scores[f"ndcg@{ndcg_k}"] = ndcg_at_k(ranked, relevant, ndcg_k)
    return scores


# --------------------------------------------------------------------- statistics
def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling cases with replacement."""
    if not values:
        raise ValueError("bootstrap_ci of empty sequence")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    return percentile(means, alpha * 100), percentile(means, (1.0 - alpha) * 100)


@dataclass(frozen=True)
class MetricSummary:
    """Mean of a per-case metric with its bootstrap CI and denominator."""

    mean: float
    ci_low: float
    ci_high: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
        }


def summarize(
    values: Sequence[float], *, n_resamples: int = 1000, seed: int = 0
) -> MetricSummary | None:
    """Mean and 95% bootstrap CI; None when there are no cases to average."""
    if not values:
        return None
    mean = sum(values) / len(values)
    low, high = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
    return MetricSummary(mean=mean, ci_low=low, ci_high=high, n=len(values))


def latency_summary(seconds: Sequence[float]) -> dict[str, float | int] | None:
    """Mean/median/p95/max of per-query latencies (seconds)."""
    if not seconds:
        return None
    return {
        "n": len(seconds),
        "mean": sum(seconds) / len(seconds),
        "median": percentile(seconds, 50),
        "p95": percentile(seconds, 95),
        "max": max(seconds),
    }
