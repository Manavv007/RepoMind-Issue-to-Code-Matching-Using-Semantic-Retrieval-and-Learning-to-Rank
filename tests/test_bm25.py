"""Tests for BM25: index math vs a brute-force reference, and retriever behavior."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.parsing.symbol_extractor import extract_file
from app.retrieval import bm25 as bm25_module
from app.retrieval.base import RetrievalQuery
from app.retrieval.bm25 import (
    BM25Config,
    BM25Index,
    BM25Retriever,
    field_tokens,
    query_weights,
    weighted_document,
)
from app.retrieval.registry import create_retriever, load_params
from evaluation import run_retrieval_eval as runner
from evaluation.run_retrieval_eval import run_evaluation
from evaluation.schema import IssueInput
from tests.conftest import GitRepoFactory
from tests.test_eval_runner import make_case, make_pool

# ------------------------------------------------------------ reference BM25
K1, B = 1.2, 0.75


def brute_force_scores(docs, query, k1=K1, b=B):
    """Textbook BM25 written independently of BM25Index (no inverted index)."""
    n = len(docs)
    lengths = [sum(d.values()) for d in docs]
    avgdl = sum(lengths) / n
    scores = {}
    for i, doc in enumerate(docs):
        total = 0.0
        for term, qw in query.items():
            tf = doc.get(term, 0.0)
            if tf == 0:
                continue
            df = sum(1 for d in docs if d.get(term, 0.0) > 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            total += (
                qw * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[i] / avgdl))
            )
        if total > 0:
            scores[i] = total
    return scores


def random_corpus(seed: int, n_docs: int = 60, vocab: int = 40):
    rng = random.Random(seed)
    terms = [f"t{i}" for i in range(vocab)]
    docs = []
    for _ in range(n_docs):
        size = rng.randint(1, 15)
        doc: dict[str, float] = {}
        for _ in range(size):
            term = rng.choice(terms[: rng.randint(5, vocab)])
            doc[term] = doc.get(term, 0.0) + rng.choice([1.0, 1.0, 2.0, 3.0, 0.5])
        docs.append(doc)
    return docs, terms


# ------------------------------------------------------------------ index math
def test_hand_computed_single_term_score():
    docs = [{"a": 2.0, "b": 1.0}, {"b": 1.0}]
    index = BM25Index(docs, k1=1.2, b=0.75)
    # N=2, df(a)=1 -> idf = ln(1 + 1.5/1.5) = ln 2; |d0|=3, avgdl=2.
    norm = 1.2 * (1 - 0.75 + 0.75 * 3 / 2)
    expected = math.log(2) * 2 * 2.2 / (2 + norm)
    assert index.score({"a": 1.0}) == {0: pytest.approx(expected)}
    assert index.idf("a") == pytest.approx(math.log(2))
    assert index.df("b") == 2 and index.df("zzz") == 0


@pytest.mark.parametrize("seed", range(8))
def test_inverted_index_matches_brute_force(seed: int):
    docs, terms = random_corpus(seed)
    rng = random.Random(1000 + seed)
    index = BM25Index(docs, k1=K1, b=B)
    for _ in range(10):
        query = {
            t: rng.choice([1.0, 2.0, 1.5]) for t in rng.sample(terms, rng.randint(1, 8))
        }
        expected = brute_force_scores(docs, query)
        got = index.score(query)
        assert set(got) == set(expected)
        for doc_id, score in expected.items():
            assert got[doc_id] == pytest.approx(score, rel=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_top_k_is_the_brute_force_ranking_with_deterministic_ties(seed: int):
    docs, terms = random_corpus(seed, n_docs=80)
    query = {t: 1.0 for t in terms[:6]}
    expected = sorted(
        brute_force_scores(docs, query).items(), key=lambda x: (-x[1], x[0])
    )
    got = BM25Index(docs, k1=K1, b=B).top_k(query, 10)
    assert [d for d, _ in got] == [d for d, _ in expected[:10]]
    assert [s for _, s in got] == pytest.approx([s for _, s in expected[:10]])


def test_scoring_properties():
    # Same length (via filler term), increasing tf -> increasing, saturating score.
    docs = [{"q": float(tf), "pad": 20.0 - tf} for tf in (1, 2, 4, 8, 16)]
    docs += [{"pad": 20.0}] * 3
    scores = BM25Index(docs, k1=K1, b=B).score({"q": 1.0})
    seq = [scores[i] for i in range(5)]
    assert seq == sorted(seq)
    gaps = [b - a for a, b in zip(seq, seq[1:], strict=False)]
    assert gaps[0] > gaps[-1]  # diminishing returns
    idf = BM25Index(docs, k1=K1, b=B).idf("q")
    assert max(seq) < idf * (K1 + 1)  # bounded by the saturation ceiling


def test_length_normalization_controlled_by_b():
    short, long_ = {"q": 1.0, "x": 1.0}, {"q": 1.0, "x": 30.0}
    filler = [{"y": 5.0}] * 4
    no_norm = BM25Index([short, long_, *filler], k1=K1, b=0.0).score({"q": 1.0})
    full_norm = BM25Index([short, long_, *filler], k1=K1, b=1.0).score({"q": 1.0})
    assert no_norm[0] == pytest.approx(no_norm[1])
    assert full_norm[0] > full_norm[1]


def test_rare_terms_outweigh_common_terms():
    docs = [{"common": 1.0, "rare": 1.0}] + [{"common": 1.0}] * 20
    index = BM25Index(docs, k1=K1, b=B)
    assert index.idf("rare") > index.idf("common") > 0
    scores = index.score({"common": 1.0, "rare": 1.0})
    assert max(scores, key=lambda d: scores[d]) == 0


def test_query_weights_and_unknown_terms():
    docs = [{"a": 1.0}, {"b": 1.0}]
    index = BM25Index(docs, k1=K1, b=B)
    assert index.score({"nope": 1.0}) == {}
    assert index.score({"a": 0.0}) == {}
    doubled = index.score({"a": 2.0})[0]
    assert doubled == pytest.approx(2 * index.score({"a": 1.0})[0])


def test_empty_index_and_empty_documents():
    assert BM25Index([], k1=K1, b=B).score({"a": 1.0}) == {}
    index = BM25Index([{}, {"a": 1.0}], k1=K1, b=B)
    assert list(index.score({"a": 1.0})) == [1]
    assert BM25Index([{}], k1=K1, b=B).top_k({"a": 1.0}, 5) == []


def test_explain_contributions_sum_to_score():
    docs, terms = random_corpus(3)
    index = BM25Index(docs, k1=K1, b=B)
    query = {t: 1.0 for t in terms[:8]}
    doc_id, score = index.top_k(query, 1)[0]
    parts = index.explain(query, doc_id, docs[doc_id], top=100)
    assert sum(c for _, c in parts) == pytest.approx(score)
    assert [c for _, c in parts] == sorted((c for _, c in parts), reverse=True)


def test_query_weight_modes():
    tokens = ["a", "a", "a", "b"]
    assert query_weights(tokens, "binary") == {"a": 1.0, "b": 1.0}
    assert query_weights(tokens, "raw") == {"a": 3.0, "b": 1.0}
    log = query_weights(tokens, "log")
    assert log["a"] == pytest.approx(1 + math.log(3)) and log["b"] == 1.0


# ---------------------------------------------------------------- retriever
CORPUS = {
    "pkg/auth.py": '''
def restore_session(token):
    """Restore the user session from a stored token."""
    return load_token(token)


def logout_user(session):
    """Clear the session and redirect to login."""
    return redirect("login")
''',
    "pkg/views.py": '''
def render_dashboard(user):
    """Render the dashboard page for the signed in user."""
    return template("dashboard", user)
''',
    "pkg/billing.py": """
def compute_invoice(items):
    return sum(items)


def format_currency(amount):
    return f"{amount:.2f}"
""",
}


def corpus_units():
    units = []
    for path, source in CORPUS.items():
        units += [
            u
            for u in extract_file(path, "python", source.encode()).units
            if u.kind in ("function", "method")
        ]
    return units


def prepared(config: BM25Config | None = None) -> BM25Retriever:
    retriever = BM25Retriever(config)
    retriever.prepare(corpus_units())
    return retriever


def q(text: str, qid: str = "q") -> RetrievalQuery:
    return RetrievalQuery(query_id=qid, issue_body=text)


def names(results):
    return [r.qualified_name for r in results]


def test_relevant_function_is_ranked_first():
    results = prepared().retrieve(
        q("After refreshing the dashboard, users are redirected to login"), 5
    )
    assert names(results)[0] in {"render_dashboard", "logout_user"}
    only_dash = prepared().retrieve(q("dashboard page rendering"), 5)
    assert names(only_dash)[0] == "render_dashboard"


def test_camel_case_in_issue_matches_snake_case_in_code():
    results = prepared().retrieve(q("restoreSession fails with a stored token"), 3)
    assert names(results)[0] == "restore_session"


def test_file_path_terms_match_when_path_field_is_weighted():
    results = prepared().retrieve(q("billing"), 5)
    assert {r.file_path for r in results} == {"pkg/billing.py"}


def test_result_contract_ranks_scores_and_signals():
    results = prepared().retrieve(q("session token user dashboard login"), 10)
    assert results
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True) and all(s > 0 for s in scores)
    for r in results:
        assert r.method == "bm25"
        assert r.signals == {"bm25": r.score}
        assert r.start_line >= 1 and r.end_line >= r.start_line


def test_k_limits_results_and_no_match_returns_empty():
    retriever = prepared()
    assert len(retriever.retrieve(q("session token user dashboard login"), 2)) == 2
    assert retriever.retrieve(q("zzzzqqqq nonexistent"), 5) == []
    assert retriever.retrieve(q(""), 5) == []
    assert retriever.retrieve(q("the and of is"), 5) == []  # only stopwords


def test_only_units_with_lexical_evidence_are_returned():
    results = prepared().retrieve(q("invoice"), 10)
    assert names(results) == ["compute_invoice"]


def test_ties_are_broken_deterministically_by_unit_id():
    src = "def helper():\n    return shared_marker\n"
    a = extract_file("a/mod.py", "python", src.encode()).units
    b = extract_file("b/mod.py", "python", src.encode()).units
    config = BM25Config(w_path=0.0, w_name=0.0)
    forward, backward = BM25Retriever(config), BM25Retriever(config)
    forward.prepare([*a, *b])
    backward.prepare([*b, *a])
    ids = [r.unit_id for r in forward.retrieve(q("shared_marker"), 5)]
    assert ids == [r.unit_id for r in backward.retrieve(q("shared_marker"), 5)]
    assert ids == sorted(ids)
    assert len(ids) == 2


def test_name_weight_promotes_symbol_name_matches():
    src = (
        "def render_report(a):\n    return a\n\n\n"
        "def other(a):\n    return a, 'render report render report'\n"
    )
    units = extract_file("m.py", "python", src.encode()).units
    flat = BM25Retriever(BM25Config(w_name=1.0, w_path=0.0))
    boosted = BM25Retriever(BM25Config(w_name=6.0, w_path=0.0))
    flat.prepare(units)
    boosted.prepare(units)
    assert names(boosted.retrieve(q("render report"), 2))[0] == "render_report"
    gap_flat = _gap(flat.retrieve(q("render report"), 2))
    gap_boosted = _gap(boosted.retrieve(q("render report"), 2))
    assert gap_boosted > gap_flat


def _gap(results):
    by_name = {r.qualified_name: r.score for r in results}
    return by_name["render_report"] - by_name["other"]


def test_stopword_toggle_changes_matching():
    units = corpus_units()
    on, off = BM25Retriever(BM25Config()), BM25Retriever(
        BM25Config(use_stopwords=False)
    )
    on.prepare(units)
    off.prepare(units)
    assert on.retrieve(q("the of and"), 3) == []
    assert off.retrieve(q("the of and"), 3) != []


def test_retrieve_before_prepare_is_an_error():
    with pytest.raises(RuntimeError, match="prepare"):
        BM25Retriever().retrieve(q("x"), 1)
    with pytest.raises(RuntimeError, match="prepare"):
        BM25Retriever().explain(q("x"), "u")


def test_empty_pool_returns_nothing():
    retriever = BM25Retriever()
    retriever.prepare([])
    assert retriever.retrieve(q("anything at all"), 5) == []


def test_prepare_replaces_previous_state():
    retriever = prepared()
    retriever.prepare(
        extract_file("only.py", "python", b"def lonely():\n    return 1\n").units
    )
    assert names(retriever.retrieve(q("lonely session"), 5)) == ["lonely"]


def test_explain_reports_matching_terms():
    retriever = prepared()
    query = q("restoreSession token")
    top = retriever.retrieve(query, 1)[0]
    terms = dict(retriever.explain(query, top.unit_id))
    assert {"restore", "session"} & set(terms)
    assert all(c > 0 for c in terms.values())
    with pytest.raises(KeyError):
        retriever.explain(query, "missing")


def test_field_tokens_are_cached_by_content():
    unit = corpus_units()[0]
    assert field_tokens(unit) is field_tokens(unit)
    assert field_tokens(unit, use_stopwords=False) is not field_tokens(unit)


def test_cache_is_cleared_when_full(monkeypatch):
    monkeypatch.setattr(bm25_module, "_FIELD_CACHE_MAX", 2)
    monkeypatch.setattr(bm25_module, "_FIELD_CACHE", {})
    for unit in corpus_units():
        field_tokens(unit)
    assert len(bm25_module._FIELD_CACHE) <= 2


def test_weighted_document_applies_field_weights():
    unit = extract_file(
        "pkg/zeta.py", "python", b'def alpha_beta():\n    """gamma"""\n    return 1\n'
    ).units[0]
    fields = field_tokens(unit)
    base = weighted_document(fields, BM25Config(w_name=1.0, w_path=1.0))
    doubled = weighted_document(fields, BM25Config(w_name=2.0, w_path=1.0))
    assert doubled["alpha"] == pytest.approx(base["alpha"] + fields.name["alpha"])
    assert "zeta" in base  # path tokens are indexed
    no_path = weighted_document(fields, BM25Config(w_name=1.0, w_path=0.0))
    assert "zeta" not in no_path


def test_config_validation_and_describe():
    assert BM25Retriever(BM25Config(k1=2.0)).describe()["k1"] == 2.0
    for bad in ({"k1": 0}, {"b": 1.5}, {"b": -0.1}, {"w_name": -1}, {"query_tf": "x"}):
        with pytest.raises(ValidationError):
            BM25Config(**bad)
    with pytest.raises(ValidationError):
        BM25Config(unknown_option=1)  # type: ignore[call-arg]


def test_classic_config_file_matches_documented_textbook_values():
    from evaluation.tune_bm25 import CLASSIC

    path = Path(__file__).resolve().parent.parent / "configs" / "bm25_classic.json"
    cfg = BM25Config(**json.loads(path.read_text()))
    assert (cfg.k1, cfg.b) == (1.2, 0.75)
    assert (cfg.w_name, cfg.w_path, cfg.w_doc, cfg.w_body) == (1.0, 1.0, 1.0, 1.0)
    assert cfg.query_tf == "raw"
    assert cfg == CLASSIC  # the tuning baseline and the config file cannot drift


def test_frozen_defaults_are_the_dev_tuning_winner():
    """The defaults must match the committed dev-only tuning report."""
    report = Path(__file__).resolve().parent.parent / "evaluation" / "reports"
    tuning = report / "bm25_tuning_dev.json"
    if not tuning.exists():
        pytest.skip("tuning report not generated")
    best = json.loads(tuning.read_text())["best"]["params"]
    defaults = BM25Config().model_dump()
    assert {k: defaults[k] for k in best} == best


# ------------------------------------------------------------------- registry
def test_registry_builds_bm25_with_params():
    retriever = create_retriever("bm25", params={"k1": 1.9, "b": 0.4})
    assert isinstance(retriever, BM25Retriever)
    assert retriever.describe()["k1"] == 1.9
    assert create_retriever("bm25").describe() == BM25Config().model_dump()


def test_registry_rejects_bad_params():
    with pytest.raises(ValueError):
        create_retriever("bm25", params={"nope": 1})
    with pytest.raises(ValueError):
        create_retriever("bm25", params={"b": 7})
    with pytest.raises(ValueError, match="no parameters"):
        create_retriever("random", params={"x": 1})


def test_load_params_inline_file_and_errors(tmp_path: Path):
    assert load_params(None) == {} and load_params("") == {}
    assert load_params('{"k1": 2}') == {"k1": 2}
    f = tmp_path / "c.json"
    f.write_text('{"b": 0.5}')
    assert load_params(str(f)) == {"b": 0.5}
    for bad in ("{not json", str(tmp_path / "missing.json")):
        with pytest.raises(ValueError):
            load_params(bad)
    f.write_text("[1, 2]")
    with pytest.raises(ValueError, match="object"):
        load_params(str(f))


# --------------------------------------------------------- evaluation harness
def test_bm25_through_the_evaluation_runner():
    units = corpus_units()
    case = make_case("c1", units, [names_index(units, "restore_session")])
    case = case.model_copy(
        update={
            "issue": IssueInput(body="restoreSession does not restore the stored token")
        }
    )
    report = run_evaluation(
        [case], BM25Retriever(), lambda c: list(units), n_resamples=20, depth=10
    )
    row = report["cases"][0]
    assert report["config"]["retriever"] == "bm25"
    assert report["config"]["retriever_params"]["k1"] == BM25Config().k1
    assert report["config"]["retriever_params"] == BM25Config().model_dump()
    assert row["first_relevant_unit_rank"] == 1
    assert report["aggregate"]["function"]["metrics"]["mrr"]["mean"] == 1.0


def names_index(units, name):
    return next(i for i, u in enumerate(units) if u.qualified_name == name)


def test_bm25_beats_random_on_a_synthetic_benchmark():
    pool = make_pool(n_files=3, n_funcs=20)
    cases = []
    for i in range(0, 60, 3):
        gold = pool[i]
        marker = gold.symbol_name.replace("func_", "target_")
        case = make_case(f"c{i}", pool, [i])
        cases.append(
            case.model_copy(
                update={
                    "issue": IssueInput(
                        body=f"failure in {gold.symbol_name} ({marker})"
                    )
                }
            )
        )

    def mrr(retriever):
        report = run_evaluation(cases, retriever, lambda c: list(pool), n_resamples=20)
        return report["aggregate"]["function"]["metrics"]["mrr"]["mean"]

    assert mrr(BM25Retriever()) > 0.8
    assert mrr(create_retriever("random", seed=0)) < 0.2


def test_cli_with_bm25_config_and_tag(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory, monkeypatch
):
    repo = tmp_path / "src_repo"
    source = "\n\n".join(f"def work_{i}():\n    return {i}\n" for i in range(12))
    sha = git_repo(repo, {"lib/core.py": source})

    from app.indexing.pipeline import index_repository
    from app.storage.index_store import IndexStore
    from evaluation.dataset import write_cases

    store = IndexStore(tmp_path / "eval.db")
    result = index_repository(str(repo), commit=sha, settings=settings, store=store)
    pool = runner.candidate_units(store.list_units(result.repository.repo_id))
    case = make_case("c1", pool, [4], commit=sha)
    case = case.model_copy(update={"issue": IssueInput(body="work_4 fails")})
    dataset = tmp_path / "mini.jsonl"
    write_cases(dataset, [case])

    real_index = runner.index_repository
    monkeypatch.setattr(
        runner, "index_repository", lambda url, **kw: real_index(str(repo), **kw)
    )
    monkeypatch.setattr(runner, "get_settings", lambda: settings)

    out = tmp_path / "reports"
    base = [
        "--dataset",
        str(dataset),
        "--retriever",
        "bm25",
        "--db",
        str(tmp_path / "e2.db"),
        "--output-dir",
        str(out),
        "--bootstrap",
        "20",
    ]
    code = runner.main([*base, "--retriever-config", '{"k1": 2.0}', "--tag", "k2"])
    assert code == 0
    report = json.loads((out / "mini__bm25-k2__all__seed0.json").read_text())
    assert report["config"]["retriever_params"]["k1"] == 2.0
    assert report["cases"][0]["first_relevant_unit_rank"] == 1
    assert "Retriever parameters" in (out / "mini__bm25-k2__all__seed0.md").read_text()

    assert runner.main([*base, "--retriever-config", '{"bogus": 1}']) == 2
    assert runner.main([*base, "--tag", "../evil"]) == 2
