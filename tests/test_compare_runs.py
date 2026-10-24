"""Tests for paired report comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.compare_runs import (
    check_comparable,
    compare,
    main,
    paired_differences,
    render_markdown,
)


def report(
    mrrs: dict[str, float | None], *, sha="s1", split="test", repo=None, retriever="x"
):
    cases = []
    for cid, value in mrrs.items():
        metrics = (
            None
            if value is None
            else {"hit@5": value, "hit@10": value, "mrr": value, "ndcg@10": value}
        )
        cases.append(
            {
                "case_id": cid,
                "repo": (repo or {}).get(cid, "o/r"),
                "status": "ok",
                "function_metrics": metrics,
                "file_metrics": {
                    "hit@5": 1.0,
                    "hit@10": 1.0,
                    "mrr": 1.0,
                    "ndcg@10": 1.0,
                },
            }
        )
    return {
        "dataset": {"sha256": sha},
        "config": {
            "split": split,
            "candidate_pool": "pool",
            "depth": 100,
            "retriever": retriever,
            "retriever_params": {},
        },
        "cases": cases,
    }


def test_paired_differences_only_use_shared_labeled_cases():
    a = report({"c1": 0.5, "c2": 0.2, "c3": None, "c4": 1.0})
    b = report({"c1": 1.0, "c2": 0.2, "c3": 0.9, "c5": 0.3})
    diffs, _ = paired_differences(a, b, "function", "mrr")
    assert diffs == [0.5, 0.0]  # c3 has no labels in A, c4/c5 are unshared


def test_improvement_is_detected_with_win_tie_loss_counts():
    ids = [f"c{i}" for i in range(20)]
    a = report({i: 0.2 for i in ids})
    b = report({i: 0.6 for i in ids[:15]} | {i: 0.2 for i in ids[15:]})
    result = compare(a, b, n_resamples=300)
    s = result["levels"]["function"]["mrr"]
    assert s["n"] == 20
    assert s["mean_diff"] == pytest.approx(0.3)
    assert (s["wins"], s["ties"], s["losses"]) == (15, 5, 0)
    assert s["significant"] and s["ci_low"] > 0
    assert result["levels"]["file"]["mrr"]["mean_diff"] == 0


def test_noise_is_not_reported_as_significant():
    ids = [f"c{i}" for i in range(10)]
    a = report({i: (0.9 if n % 2 else 0.1) for n, i in enumerate(ids)})
    b = report({i: (0.1 if n % 2 else 0.9) for n, i in enumerate(ids)})
    s = compare(a, b, n_resamples=300)["levels"]["function"]["mrr"]
    assert s["mean_diff"] == pytest.approx(0.0)
    assert not s["significant"]


def test_by_repo_breakdown():
    repo = {"c1": "o/a", "c2": "o/b"}
    a = report({"c1": 0.0, "c2": 0.5}, repo=repo)
    b = report({"c1": 1.0, "c2": 0.5}, repo=repo)
    s = compare(a, b, n_resamples=50)["levels"]["function"]["mrr"]
    assert s["by_repo"] == {"o/a": 1.0, "o/b": 0.0}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"sha": "other"}, "dataset"), ({"split": "dev"}, "split")],
)
def test_incomparable_reports_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        check_comparable(report({"c": 1.0}), report({"c": 1.0}, **kwargs))


def test_markdown_and_cli(tmp_path: Path, capsys):
    a, b = report({"c1": 0.1, "c2": 0.2}), report({"c1": 0.5, "c2": 0.6}, retriever="y")
    pa, pb, out = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "out.md"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    assert "| function | mrr | 2 | +0.400" in render_markdown(
        compare(a, b, n_resamples=50)
    )
    assert (
        main(
            ["--a", str(pa), "--b", str(pb), "--bootstrap", "50", "--output", str(out)]
        )
        == 0
    )
    assert out.read_text().startswith("# Paired comparison")
    assert "W/T/L" in capsys.readouterr().out

    pb.write_text(json.dumps(report({"c1": 1.0}, sha="different")))
    assert main(["--a", str(pa), "--b", str(pb)]) == 2
    assert main(["--a", str(tmp_path / "missing.json"), "--b", str(pb)]) == 2


def test_reports_without_a_newer_metric_still_compare():
    a, b = report({"c1": 0.0, "c2": 1.0}), report({"c1": 1.0, "c2": 1.0})
    result = compare(a, b, n_resamples=50)
    assert result["levels"]["function"]["acc@5"] is None  # absent in both reports
    assert result["levels"]["function"]["mrr"]["n"] == 2
