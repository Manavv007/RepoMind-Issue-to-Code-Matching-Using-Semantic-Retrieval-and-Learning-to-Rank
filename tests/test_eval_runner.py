"""Tests for the evaluation runner, report generation, and retriever contract."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.config import Settings
from app.indexing.pipeline import index_repository
from app.parsing.symbol_extractor import extract_file
from app.retrieval.base import (
    RankedCandidate,
    RetrievalQuery,
    Retriever,
    make_candidate,
)
from app.retrieval.random_baseline import RandomRetriever
from app.retrieval.registry import available_retrievers, create_retriever
from app.schemas.code_unit import CodeUnit
from app.storage.index_store import IndexStore
from evaluation import run_retrieval_eval as runner
from evaluation.dataset import read_cases, write_cases
from evaluation.run_retrieval_eval import (
    render_markdown,
    run_evaluation,
    validate_candidates,
)
from evaluation.schema import (
    EvalCase,
    GoldUnit,
    GroundTruth,
    IssueInput,
    Provenance,
)
from tests.conftest import GitRepoFactory


def make_pool(n_files: int = 3, n_funcs: int = 20) -> list[CodeUnit]:
    units: list[CodeUnit] = []
    for fi in range(n_files):
        src = "\n\n".join(
            f"def func_{fi}_{i}():\n    return {i}\n" for i in range(n_funcs)
        )
        units += extract_file(f"pkg/m{fi}.py", "python", src.encode()).units
    return sorted(units, key=lambda u: u.unit_id)


def make_case(
    case_id: str,
    pool: Sequence[CodeUnit],
    gold_idx: Sequence[int] = (0,),
    *,
    repo: str = "o/r",
    commit: str = "a" * 40,
    split: str | None = "test",
    suitable: bool = True,
    with_units: bool = True,
) -> EvalCase:
    gold = [pool[i] for i in gold_idx]
    truth = GroundTruth(
        files=tuple(sorted({u.file_path for u in gold})),
        units=(
            tuple(
                GoldUnit(
                    unit_id=u.unit_id,
                    file_path=u.file_path,
                    qualified_name=u.qualified_name,
                    kind=u.kind,
                    start_line=u.start_line,
                    end_line=u.end_line,
                )
                for u in gold
            )
            if with_units
            else ()
        ),
    )
    return EvalCase(
        case_id=case_id,
        repo=repo,
        repo_url=f"https://github.com/{repo}",
        base_commit=commit,
        issue=IssueInput(body=f"issue text for {case_id}"),
        ground_truth=truth,
        suitable=suitable,
        split=split,  # type: ignore[arg-type]
        provenance=Provenance(
            dataset="test", instance_id=case_id, license_note="n", patch_sha256="h"
        ),
    )


class ListRetriever(Retriever):
    """Test retriever returning units in a caller-chosen order per query."""

    name = "listed"

    def __init__(self, order_for: dict[str, list[str]]) -> None:
        self._order_for = order_for
        self._by_id: dict[str, CodeUnit] = {}
        self.prepare_calls = 0
        self.queries: list[RetrievalQuery] = []

    def prepare(self, units: Sequence[CodeUnit]) -> None:
        self.prepare_calls += 1
        self._by_id = {u.unit_id: u for u in units}

    def retrieve(self, query: RetrievalQuery, k: int) -> list[RankedCandidate]:
        self.queries.append(query)
        ids = self._order_for[query.query_id][:k]
        return [
            make_candidate(self._by_id[i], rank=n, score=1.0 / n, method=self.name)
            for n, i in enumerate(ids, start=1)
        ]


def _evaluate(cases, retriever, pool, **kwargs):
    kwargs.setdefault("n_resamples", 50)
    return run_evaluation(cases, retriever, lambda case: list(pool), **kwargs)


# --------------------------------------------------------------- perfect / worst
def test_oracle_ranking_scores_perfectly():
    pool = make_pool()
    cases = [make_case("c1", pool, [3]), make_case("c2", pool, [25, 41])]
    order = {
        c.case_id: [u.unit_id for u in c.ground_truth.units]
        + [
            u.unit_id
            for u in pool
            if u.unit_id not in {g.unit_id for g in c.ground_truth.units}
        ]
        for c in cases
    }
    report = _evaluate(cases, ListRetriever(order), pool)

    fn = report["aggregate"]["function"]["metrics"]
    assert report["counts"]["evaluated"] == 2
    for name in ("hit@5", "hit@10", "recall@10", "mrr", "ndcg@10"):
        assert fn[name]["mean"] == pytest.approx(1.0), name
    assert report["aggregate"]["file"]["metrics"]["hit@5"]["mean"] == 1.0


def test_worst_case_ranking_scores_zero_within_cutoffs():
    pool = make_pool()
    case = make_case("c1", pool, [0])
    gold_id = case.ground_truth.units[0].unit_id
    worst = [u.unit_id for u in pool if u.unit_id != gold_id] + [gold_id]
    report = _evaluate([case], ListRetriever({"c1": worst}), pool, depth=100)

    fn = report["aggregate"]["function"]["metrics"]
    assert fn["hit@10"]["mean"] == 0.0
    assert fn["ndcg@10"]["mean"] == 0.0
    assert fn["mrr"]["mean"] == pytest.approx(1 / len(pool))
    assert report["cases"][0]["first_relevant_unit_rank"] == len(pool)


def test_file_level_collapses_functions_to_distinct_files():
    pool = make_pool(n_files=3, n_funcs=20)
    case = make_case("c1", pool, [45])  # a function in the last file (pkg/m2.py)
    gold_file = case.ground_truth.files[0]
    # Put 15 functions from two other files first, then the gold file's functions.
    other_files = sorted({u.file_path for u in pool if u.file_path != gold_file})
    assert len(other_files) == 2
    first = [u.unit_id for u in pool if u.file_path == other_files[0]][:8]
    second = [u.unit_id for u in pool if u.file_path == other_files[1]][:7]
    mine = [u.unit_id for u in pool if u.file_path == gold_file]
    report = _evaluate([case], ListRetriever({"c1": first + second + mine}), pool)

    row = report["cases"][0]
    # 15 functions come from just two files, so the gold file is file-rank 3.
    assert row["first_relevant_file_rank"] == 3
    assert row["file_metrics"]["hit@5"] == 1.0
    assert row["function_metrics"]["hit@10"] == 0.0


# --------------------------------------------------------------- denominators
def test_function_level_denominator_excludes_cases_without_unit_labels():
    pool = make_pool()
    cases = [
        make_case("c1", pool, [0]),
        make_case("c2", pool, [20], with_units=False),
    ]
    report = _evaluate(cases, create_retriever("random", seed=1), pool)
    assert report["aggregate"]["file"]["n_cases"] == 2
    assert report["aggregate"]["function"]["n_cases"] == 1
    assert report["aggregate"]["function"]["metrics"]["mrr"]["n"] == 1
    assert report["cases"][1]["function_metrics"] is None


def test_selection_filters_and_exclusion_counts():
    pool = make_pool()
    cases = [
        make_case("a", pool, split="dev"),
        make_case("b", pool, split="test"),
        make_case("c", pool, split="test", suitable=False),
    ]
    report = _evaluate(cases, RandomRetriever(0), pool, split="test")
    assert [r["case_id"] for r in report["cases"]] == ["b"]
    assert report["counts"]["excluded"] == {"unsuitable": 1, "other_split": 1}
    assert report["counts"]["cases_in_dataset"] == 3

    everything = _evaluate(cases, RandomRetriever(0), pool, split="all")
    assert {r["case_id"] for r in everything["cases"]} == {"a", "b"}


def test_label_mismatch_excludes_case_instead_of_scoring_it():
    pool = make_pool()
    good = make_case("good", pool, [0])
    stale = make_case("stale", pool, [1])
    stale = stale.model_copy(
        update={
            "ground_truth": stale.ground_truth.model_copy(
                update={
                    "units": (
                        stale.ground_truth.units[0].model_copy(
                            update={"unit_id": "pkg/m0.py::renamed:1"}
                        ),
                    )
                }
            )
        }
    )
    report = _evaluate([good, stale], RandomRetriever(0), pool)
    assert report["counts"]["evaluated"] == 1
    assert report["counts"]["excluded"] == {"label_mismatch": 1}
    assert report["failures"][0]["case_id"] == "stale"


# ------------------------------------------------------------- retriever contract
def test_retriever_only_receives_issue_text_never_labels():
    assert set(RetrievalQuery.model_fields) == {"query_id", "issue_title", "issue_body"}
    pool = make_pool()
    case = make_case("c1", pool, [7])
    spy = ListRetriever({"c1": [u.unit_id for u in pool]})
    _evaluate([case], spy, pool)
    assert len(spy.queries) == 1
    query = spy.queries[0]
    assert query.issue_body == "issue text for c1"
    dumped = json.dumps(query.model_dump())
    assert case.ground_truth.units[0].unit_id not in dumped


def test_prepare_runs_once_per_revision_not_per_case():
    pool = make_pool()
    cases = [
        make_case("a", pool, commit="a" * 40),
        make_case("b", pool, commit="a" * 40),
        make_case("c", pool, commit="b" * 40),
    ]
    spy = ListRetriever({c.case_id: [u.unit_id for u in pool] for c in cases})
    _evaluate(cases, spy, pool)
    assert spy.prepare_calls == 2


def test_misbehaving_retrievers_are_rejected_without_aborting_the_run():
    pool = make_pool()
    ids = [u.unit_id for u in pool]

    class Cheater(ListRetriever):
        def retrieve(self, query, k):
            real = super().retrieve(query, k)
            if query.query_id == "bad-unknown":
                fake = real[0].model_copy(update={"unit_id": "not/in/pool.py::x:1"})
                return [fake, *real[1:]]
            if query.query_id == "bad-dup":
                return [real[0], real[0].model_copy(update={"rank": 2})]
            if query.query_id == "bad-rank":
                return [real[0].model_copy(update={"rank": 5})]
            return real

    cases = [
        make_case("ok", pool),
        make_case("bad-unknown", pool),
        make_case("bad-dup", pool),
        make_case("bad-rank", pool),
    ]
    report = _evaluate(cases, Cheater({c.case_id: ids for c in cases}), pool)
    assert report["counts"]["evaluated"] == 1
    assert report["counts"]["excluded"] == {"error": 3}
    assert {f["case_id"] for f in report["failures"]} == {
        "bad-unknown",
        "bad-dup",
        "bad-rank",
    }


def test_validate_candidates_rejects_too_many_results():
    pool = make_pool(1, 5)
    spy = ListRetriever({"q": [u.unit_id for u in pool]})
    spy.prepare(pool)
    results = spy.retrieve(RetrievalQuery(query_id="q", issue_body="x"), 5)
    with pytest.raises(ValueError, match="depth"):
        validate_candidates(results, {u.unit_id for u in pool}, depth=3)


# ---------------------------------------------------------------- random baseline
def test_random_baseline_is_reproducible_and_seed_dependent():
    pool = make_pool()
    case = make_case("c1", pool)
    a = _evaluate([case], RandomRetriever(0), pool)["cases"][0]["top_candidates"]
    b = _evaluate([case], RandomRetriever(0), pool)["cases"][0]["top_candidates"]
    c = _evaluate([case], RandomRetriever(1), pool)["cases"][0]["top_candidates"]
    assert a == b
    assert a != c


def test_random_baseline_ignores_input_order_and_is_per_query():
    pool = make_pool()
    r1, r2 = RandomRetriever(5), RandomRetriever(5)
    r1.prepare(pool)
    r2.prepare(list(reversed(pool)))
    q = RetrievalQuery(query_id="x", issue_body="b")
    assert [c.unit_id for c in r1.retrieve(q, 10)] == [
        c.unit_id for c in r2.retrieve(q, 10)
    ]
    other = RetrievalQuery(query_id="y", issue_body="b")
    assert [c.unit_id for c in r1.retrieve(other, 10)] != [
        c.unit_id for c in r1.retrieve(q, 10)
    ]


def test_random_baseline_is_near_chance_on_large_sample():
    pool = make_pool(n_files=4, n_funcs=50)  # 200 units, 1 gold each
    cases = [make_case(f"c{i}", pool, [i]) for i in range(150)]
    report = _evaluate(cases, RandomRetriever(3), pool, n_resamples=200)
    hit10 = report["aggregate"]["function"]["metrics"]["hit@10"]
    # Expected chance rate is 10/200 = 0.05 with SD ~0.018 at n=150; allow 4 SD.
    assert abs(hit10["mean"] - 10 / 200) < 0.07
    assert hit10["n"] == 150


def test_registry():
    assert "random" in available_retrievers()
    assert create_retriever("random", seed=4).name == "random"
    with pytest.raises(ValueError, match="Unknown retriever"):
        create_retriever("nope")


# ---------------------------------------------------------------------- reports
def test_report_contains_denominators_ci_latency_and_definitions():
    pool = make_pool()
    report = _evaluate([make_case("c1", pool)], RandomRetriever(0), pool)
    assert json.loads(json.dumps(report)) == report  # JSON-serializable
    assert report["config"]["candidate_pool"].startswith("leaf units")
    assert report["aggregate"]["latency_retrieve_seconds"]["n"] == 1
    assert report["aggregate"]["pool_size"]["mean"] == len(pool)

    markdown = render_markdown(report, "unit.jsonl")
    assert "95% bootstrap CI" in markdown
    assert "| function | hit@10 |" in markdown
    assert "## Metric definitions" in markdown
    assert "unit.jsonl" in markdown


# ------------------------------------------------------------------------- CLI
def test_cli_end_to_end_with_real_index(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory, monkeypatch, capsys
):
    repo = tmp_path / "src_repo"
    source = "\n\n".join(f"def work_{i}():\n    return {i}\n" for i in range(30))
    sha = git_repo(
        repo, {"lib/core.py": source, "tests/test_core.py": "def test_a():\n    pass\n"}
    )

    store = IndexStore(tmp_path / "eval.db")
    indexed = index_repository(str(repo), commit=sha, settings=settings, store=store)
    pool = runner.candidate_units(store.list_units(indexed.repository.repo_id))
    assert pool and all(
        not u.file_path.startswith("tests/") for u in pool
    )  # tests excluded

    case = make_case("c1", pool, [4], repo="o/r", commit=sha)
    dataset = tmp_path / "mini.jsonl"
    write_cases(dataset, [case])
    assert read_cases(dataset)[0] == case

    real_index = runner.index_repository
    monkeypatch.setattr(
        runner, "index_repository", lambda url, **kw: real_index(str(repo), **kw)
    )
    monkeypatch.setattr(runner, "get_settings", lambda: settings)

    out = tmp_path / "reports"
    code = runner.main(
        [
            "--dataset",
            str(dataset),
            "--retriever",
            "random",
            "--db",
            str(tmp_path / "eval2.db"),
            "--output-dir",
            str(out),
            "--bootstrap",
            "50",
        ]
    )
    assert code == 0
    report = json.loads((out / "mini__random__all__seed0.json").read_text())
    assert report["counts"]["evaluated"] == 1
    assert report["dataset"]["path"] == "mini.jsonl"
    assert len(report["dataset"]["sha256"]) == 64
    assert (
        (out / "mini__random__all__seed0.md")
        .read_text()
        .startswith("# Retrieval evaluation")
    )
    assert "Wrote" in capsys.readouterr().out


def test_reports_never_query_wmi(monkeypatch):
    """platform.platform()/uname() hit WMI on Windows and can hang; never call them."""
    import platform as _platform

    def boom(*_a, **_k):
        raise AssertionError("WMI-backed platform call made during evaluation")

    monkeypatch.setattr(_platform, "platform", boom)
    monkeypatch.setattr(_platform, "uname", boom)
    pool = make_pool(1, 5)
    report = _evaluate([make_case("c1", pool)], RandomRetriever(0), pool)
    assert report["environment"]["platform"]
