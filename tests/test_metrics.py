"""Tests for evaluation metrics (hand-computed expected values)."""

from __future__ import annotations

import math

import pytest

from evaluation import metrics as m

RANKED = ["a", "b", "c", "d", "e"]


def test_hit_at_k_is_any_relevant_in_top_k():
    assert m.hit_at_k(RANKED, {"c", "z"}, 2) == 0.0
    assert m.hit_at_k(RANKED, {"c", "z"}, 3) == 1.0
    assert m.hit_at_k(RANKED, {"z"}, 5) == 0.0
    assert m.hit_at_k([], {"a"}, 5) == 0.0


def test_recall_at_k_is_fraction_of_relevant_found():
    assert m.recall_at_k(RANKED, {"c", "z"}, 3) == 0.5
    assert m.recall_at_k(RANKED, {"a", "b"}, 2) == 1.0
    assert m.recall_at_k(RANKED, {"a", "z", "y", "x"}, 10) == 0.25


def test_reciprocal_rank():
    assert m.reciprocal_rank(RANKED, {"c"}) == pytest.approx(1 / 3)
    assert m.reciprocal_rank(RANKED, {"c", "a"}) == 1.0  # first relevant wins
    assert m.reciprocal_rank(RANKED, {"zzz"}) == 0.0
    assert m.rank_of_first(RANKED, {"d", "e"}) == 4
    assert m.rank_of_first(RANKED, {"q"}) is None


def test_ndcg_matches_hand_computation():
    ranked = ["x", "r1", "y", "r2"]
    relevant = {"r1", "r2"}
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert m.ndcg_at_k(ranked, relevant, 4) == pytest.approx(dcg / ideal)


def test_ndcg_bounds_and_cutoff():
    assert m.ndcg_at_k(["r", "a"], {"r"}, 10) == 1.0
    assert m.ndcg_at_k(["a", "b"], {"r"}, 10) == 0.0
    # Relevant item beyond the cutoff contributes nothing.
    assert m.ndcg_at_k(["a", "b", "r"], {"r"}, 2) == 0.0
    # Ideal DCG is capped at k relevant items.
    assert m.ndcg_at_k(["r1", "r2"], {"r1", "r2", "r3", "r4"}, 2) == 1.0


def test_metrics_reject_empty_relevant_sets():
    with pytest.raises(ValueError):
        m.recall_at_k(RANKED, set(), 5)
    with pytest.raises(ValueError):
        m.ndcg_at_k(RANKED, set(), 5)


def test_score_ranking_keys_and_values():
    scores = m.score_ranking(RANKED, {"b"}, cutoffs=(1, 5), ndcg_k=10)
    assert set(scores) == {"hit@1", "recall@1", "hit@5", "recall@5", "mrr", "ndcg@10"}
    assert scores["hit@1"] == 0.0
    assert scores["hit@5"] == 1.0
    assert scores["mrr"] == 0.5
    assert scores["ndcg@10"] == pytest.approx(1 / math.log2(3))


def test_dedupe_preserve_order():
    assert m.dedupe_preserve_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_percentile_interpolates():
    assert m.percentile([1, 2, 3, 4], 50) == 2.5
    assert m.percentile([1, 2, 3, 4], 0) == 1
    assert m.percentile([1, 2, 3, 4], 100) == 4
    assert m.percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    with pytest.raises(ValueError):
        m.percentile([], 50)


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    low, high = m.bootstrap_ci(values, n_resamples=500, seed=7)
    assert (low, high) == m.bootstrap_ci(values, n_resamples=500, seed=7)
    assert low <= sum(values) / len(values) <= high
    assert 0.0 <= low < high <= 1.0


def test_bootstrap_ci_degenerate_and_wider_with_less_data():
    assert m.bootstrap_ci([1.0] * 8, n_resamples=100) == (1.0, 1.0)
    small = [0.0, 1.0, 0.0, 1.0]
    big = small * 25
    s_low, s_high = m.bootstrap_ci(small, n_resamples=500, seed=1)
    b_low, b_high = m.bootstrap_ci(big, n_resamples=500, seed=1)
    assert (s_high - s_low) > (b_high - b_low)


def test_summarize_reports_denominator_and_handles_empty():
    summary = m.summarize([1.0, 0.0, 1.0, 1.0], n_resamples=200, seed=0)
    assert summary is not None
    assert summary.n == 4
    assert summary.mean == 0.75
    assert summary.ci_low <= summary.mean <= summary.ci_high
    assert m.summarize([]) is None


def test_latency_summary():
    stats = m.latency_summary([0.1, 0.2, 0.3, 0.4])
    assert stats is not None
    assert stats["n"] == 4
    assert stats["mean"] == pytest.approx(0.25)
    assert stats["median"] == pytest.approx(0.25)
    assert stats["max"] == 0.4
    assert m.latency_summary([]) is None
