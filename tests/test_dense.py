"""Tests for dense retrieval, hybrid fusion, and the unavailable-backend fallback.

No real model is loaded: a deterministic fake encoder stands in, and the
sentence-transformers module is replaced by a fake to exercise failure paths.
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
import types
from collections.abc import Sequence

import pytest

from app.parsing.symbol_extractor import extract_file
from app.retrieval.base import RetrievalQuery
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import (
    DenseConfig,
    DenseRetriever,
    DenseUnavailableError,
    HybridConfig,
    HybridRetriever,
    SentenceTransformerEncoder,
    build_hybrid,
    rrf_fuse,
    unit_text,
)
from app.retrieval.registry import create_retriever
from app.retrieval.tokenizer import tokenize
from evaluation import run_retrieval_eval as runner
from evaluation.run_retrieval_eval import run_evaluation
from evaluation.schema import IssueInput
from tests.test_eval_runner import make_case

DIM = 64


class HashingEncoder:
    """Bag-of-words hashed into DIM buckets, L2-normalized. Deterministic."""

    def __init__(self, synonyms: dict[str, str] | None = None) -> None:
        self.synonyms = synonyms or {}
        self.calls: list[tuple[int, bool]] = []

    def encode(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        self.calls.append((len(texts), is_query))
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for tok in tokenize(text):
                tok = self.synonyms.get(tok, tok)
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


SRC = {
    "pkg/auth.py": (
        "def restore_session(token):\n    return load_token(token)\n\n\n"
        "def logout_user(session):\n    return redirect('login')\n"
    ),
    "pkg/billing.py": "def compute_invoice(items):\n    return sum(items)\n",
}


def units():
    out = []
    for path, src in SRC.items():
        out += extract_file(path, "python", src.encode()).units
    return out


def q(text: str) -> RetrievalQuery:
    return RetrievalQuery(query_id="q", issue_body=text)


# ------------------------------------------------------------------ dense
def test_dense_ranks_by_cosine_and_reports_signals():
    retriever = DenseRetriever(encoder=HashingEncoder())
    retriever.prepare(units())
    results = retriever.retrieve(q("invoice items sum"), 3)
    assert results[0].qualified_name == "compute_invoice"
    assert [r.rank for r in results] == [1, 2, 3]
    assert all(r.method == "dense" and r.signals == {"dense": r.score} for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_bridges_vocabulary_mismatch_that_bm25_cannot():
    # The issue says "bill"; the code says "invoice". BM25 has no shared term.
    query = q("bill")
    lexical = BM25Retriever()
    lexical.prepare(units())
    assert all(
        r.qualified_name != "compute_invoice" for r in lexical.retrieve(query, 3)
    )
    dense = DenseRetriever(encoder=HashingEncoder(synonyms={"bill": "invoice"}))
    dense.prepare(units())
    assert dense.retrieve(query, 1)[0].qualified_name == "compute_invoice"


def test_dense_embedding_cache_reencodes_only_changed_units():
    encoder = HashingEncoder()
    retriever = DenseRetriever(encoder=encoder)
    pool = units()
    retriever.prepare(pool)
    retriever.prepare(pool)  # same revision: no new document encoding
    doc_calls = [n for n, is_q in encoder.calls if not is_q]
    assert doc_calls == [len(pool)]
    changed = extract_file(
        "pkg/billing.py", "python", b"def compute_invoice(items):\n    return 0\n"
    ).units
    retriever.prepare([*pool[:2], *changed])
    assert [n for n, is_q in encoder.calls if not is_q] == [len(pool), 1]


def test_dense_is_deterministic_independent_of_input_order():
    a, b = DenseRetriever(encoder=HashingEncoder()), DenseRetriever(
        encoder=HashingEncoder()
    )
    a.prepare(units())
    b.prepare(list(reversed(units())))
    query = q("session token login")
    assert [r.unit_id for r in a.retrieve(query, 3)] == [
        r.unit_id for r in b.retrieve(query, 3)
    ]


def test_dense_empty_pool_and_describe():
    retriever = DenseRetriever(DenseConfig(model_name="x/y"), encoder=HashingEncoder())
    retriever.prepare([])
    assert retriever.retrieve(q("anything"), 5) == []
    assert retriever.describe()["model_name"] == "x/y"
    assert retriever.describe()["trust_remote_code"] is False


def test_unit_text_includes_path_and_name_by_default():
    unit = units()[0]
    assert unit_text(unit).startswith("pkg/auth.py restore_session\n")
    assert unit_text(unit, include_path=False) == unit.source


# -------------------------------------------------------------- RRF / hybrid
def test_rrf_matches_hand_computation_and_weights():
    fused = dict(rrf_fuse([(1.0, ["a", "b"]), (1.0, ["b", "c"])], rrf_k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)
    order = [u for u, _ in rrf_fuse([(1.0, ["a", "b"]), (1.0, ["b", "c"])], rrf_k=60)]
    assert order == ["b", "a", "c"]
    only_second = [u for u, _ in rrf_fuse([(0.0, ["a"]), (1.0, ["c"])], rrf_k=60)]
    assert only_second == ["c"]


def test_rrf_ties_break_by_id():
    assert [u for u, _ in rrf_fuse([(1.0, ["z"]), (1.0, ["a"])], rrf_k=10)] == [
        "a",
        "z",
    ]


def test_hybrid_combines_both_and_keeps_raw_signals():
    hybrid = HybridRetriever(
        BM25Retriever(),
        DenseRetriever(encoder=HashingEncoder(synonyms={"bill": "invoice"})),
    )
    hybrid.prepare(units())
    results = hybrid.retrieve(q("bill for restore_session"), 3)
    names = [r.qualified_name for r in results]
    assert {"restore_session", "compute_invoice"} <= set(names)
    top = results[0]
    assert top.method == "hybrid" and "rrf" in top.signals
    assert any("bm25" in r.signals and "dense" in r.signals for r in results)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_hybrid_with_zero_dense_weight_equals_bm25_order():
    pool = units()
    bm25 = BM25Retriever()
    bm25.prepare(pool)
    hybrid = HybridRetriever(
        BM25Retriever(),
        DenseRetriever(encoder=HashingEncoder()),
        HybridConfig(w_dense=0.0),
    )
    hybrid.prepare(pool)
    query = q("session token login invoice")
    assert [r.unit_id for r in hybrid.retrieve(query, 5)] == [
        r.unit_id for r in bm25.retrieve(query, 5)
    ]


def test_build_hybrid_from_params_and_validation():
    hybrid = build_hybrid(
        {"rrf_k": 10, "bm25": {"k1": 1.5}, "dense": {"model_name": "m"}},
        encoder=HashingEncoder(),
    )
    desc = hybrid.describe()
    assert desc["fusion"]["rrf_k"] == 10
    assert desc["bm25"]["k1"] == 1.5 and desc["dense"]["model_name"] == "m"
    with pytest.raises(ValueError):
        build_hybrid({"bogus": 1}, encoder=HashingEncoder())


def test_hybrid_through_evaluation_runner():
    pool = [u for u in units() if u.kind == "function"]
    gold = next(i for i, u in enumerate(pool) if u.qualified_name == "compute_invoice")
    case = make_case("c1", pool, [gold])
    case = case.model_copy(update={"issue": IssueInput(body="the bill total is wrong")})
    hybrid = HybridRetriever(
        BM25Retriever(),
        DenseRetriever(encoder=HashingEncoder(synonyms={"bill": "invoice"})),
    )
    report = run_evaluation(
        [case], hybrid, lambda c: list(pool), n_resamples=10, depth=10
    )
    assert report["counts"]["evaluated"] == 1
    assert report["cases"][0]["first_relevant_unit_rank"] == 1
    assert report["config"]["retriever_params"]["dense"]["model_name"]


# ------------------------------------------------------- unavailable fallback
def _fake_st_module(monkeypatch, cls):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_model_load_error_becomes_dense_unavailable(monkeypatch):
    class Broken:
        def __init__(self, *a, **k):
            raise OSError(
                "The paging file is too small for this operation (os error 1455)"
            )

    _fake_st_module(monkeypatch, Broken)
    with pytest.raises(DenseUnavailableError, match="1455"):
        SentenceTransformerEncoder(DenseConfig(model_name="m"))


def test_hanging_model_load_times_out(monkeypatch):
    class Slow:
        def __init__(self, *a, **k):
            time.sleep(5)

    _fake_st_module(monkeypatch, Slow)
    start = time.monotonic()
    with pytest.raises(DenseUnavailableError, match="did not finish"):
        SentenceTransformerEncoder(
            DenseConfig(model_name="m", load_timeout_seconds=0.2)
        )
    assert time.monotonic() - start < 3


def test_working_backend_encodes_with_query_prefix(monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, name, **kwargs):
            seen["init"] = (name, kwargs)
            self.max_seq_length = 0

        def encode(self, texts, **kwargs):
            seen["texts"] = list(texts)
            return [[1.0, 0.0] for _ in texts]

    _fake_st_module(monkeypatch, Fake)
    enc = SentenceTransformerEncoder(
        DenseConfig(
            model_name="m", query_prefix="Q: ", max_seq_length=128, revision="abc"
        )
    )
    assert enc.encode(["hello"], is_query=True) == [[1.0, 0.0]]
    assert seen["texts"] == ["Q: hello"]
    enc.encode(["code"], is_query=False)
    assert seen["texts"] == ["code"]
    assert seen["init"][1]["revision"] == "abc"
    assert seen["init"][1]["trust_remote_code"] is False
    assert enc._model.max_seq_length == 128
    assert enc.encode([], is_query=False) == []


def test_registry_surfaces_unavailable_backend(monkeypatch):
    class Broken:
        def __init__(self, *a, **k):
            raise OSError("boom")

    _fake_st_module(monkeypatch, Broken)
    with pytest.raises(DenseUnavailableError):
        create_retriever("dense")
    with pytest.raises(DenseUnavailableError):
        create_retriever("hybrid")
    with pytest.raises(ValueError):
        create_retriever("dense", params={"not_a_field": 1})


def test_cli_reports_dense_unavailable_with_exit_code_3(monkeypatch, tmp_path, capsys):
    class Broken:
        def __init__(self, *a, **k):
            raise OSError("paging file is too small (os error 1455)")

    _fake_st_module(monkeypatch, Broken)
    dataset = tmp_path / "d.jsonl"
    dataset.write_text("")
    code = runner.main(
        [
            "--dataset",
            str(dataset),
            "--retriever",
            "dense",
            "--db",
            str(tmp_path / "x.db"),
        ]
    )
    assert code == 3
    err = capsys.readouterr().err
    assert "dense_unavailable" in err and "--retriever bm25" in err
