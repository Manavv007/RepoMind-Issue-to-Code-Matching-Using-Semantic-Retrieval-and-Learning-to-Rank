"""BM25 lexical retrieval over code units (own implementation, inverted index).

Scoring (Robertson/Lucene BM25 with a non-negative IDF):

    score(q, d) = sum_t  w_q(t) * idf(t) * tf(t,d) * (k1 + 1)
                                 / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

``tf(t, d)`` is a *field-weighted* term frequency (BM25F with a shared length
normalization): occurrences in the symbol name, file path, docstring, and body are
multiplied by their field weight and summed. ``|d|`` is the weighted document length.

Only documents sharing at least one term with the query are scored, so a result
list may be shorter than ``k``; documents with no lexical evidence are not
returned. Ties are broken by document order, which is sorted by ``unit_id``, so
rankings are fully deterministic.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.base import (
    RankedCandidate,
    RetrievalQuery,
    Retriever,
    make_candidate,
)
from app.retrieval.tokenizer import tokenize
from app.schemas.code_unit import CodeUnit

QueryTf = Literal["binary", "log", "raw"]


class BM25Config(BaseModel):
    """BM25 hyperparameters and field weights.

    The defaults are the **frozen** configuration chosen on the ``dev`` split (see
    ``evaluation/reports/bm25_tuning_dev.md``). The textbook configuration lives in
    ``configs/bm25_classic.json``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    k1: float = Field(default=0.9, gt=0, description="Term-frequency saturation.")
    b: float = Field(default=0.75, ge=0, le=1, description="Length normalization.")
    w_name: float = Field(default=3.0, ge=0, description="Qualified symbol name.")
    w_path: float = Field(default=2.0, ge=0, description="File path.")
    w_doc: float = Field(default=1.0, ge=0, description="Docstring (extra boost).")
    w_body: float = Field(default=1.0, ge=0, description="Full source text.")
    query_tf: QueryTf = Field(
        default="log",
        description="How repeated query terms count: 1, 1+ln(n), or n.",
    )
    use_stopwords: bool = True


# ------------------------------------------------------------------------ index
class BM25Index:
    """Inverted index over weighted term-frequency documents."""

    def __init__(
        self, docs: Sequence[Mapping[str, float]], *, k1: float, b: float
    ) -> None:
        self.k1 = k1
        self.b = b
        self.num_docs = len(docs)
        self.doc_lengths = [float(sum(d.values())) for d in docs]
        self.avgdl = sum(self.doc_lengths) / self.num_docs if self.num_docs else 0.0

        postings: dict[str, list[tuple[int, float]]] = {}
        for doc_id, doc in enumerate(docs):
            for term, tf in doc.items():
                if tf > 0:
                    postings.setdefault(term, []).append((doc_id, float(tf)))
        self._postings = postings

        n = self.num_docs
        self._idf = {
            term: math.log(1.0 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in postings.items()
        }
        self._norm = [self._length_norm(dl) for dl in self.doc_lengths]

    def _length_norm(self, doc_length: float) -> float:
        if self.avgdl <= 0:
            return self.k1 * (1.0 - self.b)
        return self.k1 * (1.0 - self.b + self.b * doc_length / self.avgdl)

    # ---- statistics
    def df(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    def idf(self, term: str) -> float:
        return self._idf.get(term, 0.0)

    # ---- scoring
    def score(self, query: Mapping[str, float]) -> dict[int, float]:
        """Scores for every document sharing at least one query term."""
        scores: dict[int, float] = {}
        k1p1 = self.k1 + 1.0
        for term, weight in query.items():
            postings = self._postings.get(term)
            if postings is None or weight <= 0:
                continue
            factor = weight * self._idf[term] * k1p1
            for doc_id, tf in postings:
                scores[doc_id] = scores.get(doc_id, 0.0) + factor * tf / (
                    tf + self._norm[doc_id]
                )
        return scores

    def top_k(self, query: Mapping[str, float], k: int) -> list[tuple[int, float]]:
        """Best ``k`` ``(doc_id, score)`` pairs; score desc, ties by doc id."""
        scored = [(d, s) for d, s in self.score(query).items() if s > 0]
        return heapq.nsmallest(k, scored, key=lambda item: (-item[1], item[0]))

    def explain(
        self,
        query: Mapping[str, float],
        doc_id: int,
        doc: Mapping[str, float],
        top: int = 5,
    ) -> list[tuple[str, float]]:
        """Per-term contributions to ``doc``'s score, largest first."""
        out: list[tuple[str, float]] = []
        for term, weight in query.items():
            tf = doc.get(term, 0.0)
            if tf <= 0 or term not in self._idf or weight <= 0:
                continue
            contribution = (
                weight
                * self._idf[term]
                * tf
                * (self.k1 + 1.0)
                / (tf + self._norm[doc_id])
            )
            out.append((term, contribution))
        out.sort(key=lambda item: (-item[1], item[0]))
        return out[:top]


# ---------------------------------------------------------------- unit documents
@dataclass(frozen=True)
class FieldTokens:
    """Token counts of one code unit, per field (independent of field weights)."""

    name: Counter[str]
    path: Counter[str]
    doc: Counter[str]
    body: Counter[str]


_FIELD_CACHE: dict[tuple[str, str, str, bool], FieldTokens] = {}
_FIELD_CACHE_MAX = 200_000


def field_tokens(unit: CodeUnit, *, use_stopwords: bool = True) -> FieldTokens:
    """Tokenize a unit's fields, cached by content.

    Units that are unchanged between revisions of a repository share a cache entry,
    which makes indexing many revisions and hyperparameter sweeps cheap.
    """
    key = (unit.file_path, unit.qualified_name, unit.content_hash, use_stopwords)
    cached = _FIELD_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_FIELD_CACHE) >= _FIELD_CACHE_MAX:
        _FIELD_CACHE.clear()
    tokens = FieldTokens(
        name=Counter(tokenize(unit.qualified_name, use_stopwords=use_stopwords)),
        path=Counter(tokenize(unit.file_path, use_stopwords=use_stopwords)),
        doc=Counter(tokenize(unit.docstring or "", use_stopwords=use_stopwords)),
        body=Counter(tokenize(unit.source, use_stopwords=use_stopwords)),
    )
    _FIELD_CACHE[key] = tokens
    return tokens


def weighted_document(fields: FieldTokens, config: BM25Config) -> dict[str, float]:
    """Combine field counts into one field-weighted term-frequency map."""
    doc: dict[str, float] = {}
    for counts, weight in (
        (fields.name, config.w_name),
        (fields.path, config.w_path),
        (fields.doc, config.w_doc),
        (fields.body, config.w_body),
    ):
        if weight == 0:
            continue
        for term, count in counts.items():
            doc[term] = doc.get(term, 0.0) + weight * count
    return doc


def query_weights(tokens: Sequence[str], mode: QueryTf) -> dict[str, float]:
    """Per-term query weights from repeated occurrences."""
    counts = Counter(tokens)
    if mode == "binary":
        return {t: 1.0 for t in counts}
    if mode == "log":
        return {t: 1.0 + math.log(c) for t, c in counts.items()}
    return {t: float(c) for t, c in counts.items()}


# ---------------------------------------------------------------------- retriever
class BM25Retriever(Retriever):
    """Field-weighted BM25 over the code units of one repository revision."""

    name = "bm25"

    def __init__(self, config: BM25Config | None = None) -> None:
        self.config = config or BM25Config()
        self._units: list[CodeUnit] = []
        self._docs: list[dict[str, float]] = []
        self._index: BM25Index | None = None

    def describe(self) -> dict[str, Any]:
        return self.config.model_dump()

    def prepare(self, units: Sequence[CodeUnit]) -> None:
        cfg = self.config
        self._units = sorted(units, key=lambda u: u.unit_id)
        self._docs = [
            weighted_document(field_tokens(u, use_stopwords=cfg.use_stopwords), cfg)
            for u in self._units
        ]
        self._index = BM25Index(self._docs, k1=cfg.k1, b=cfg.b)

    def _query(self, query: RetrievalQuery) -> dict[str, float]:
        tokens = tokenize(query.text, use_stopwords=self.config.use_stopwords)
        return query_weights(tokens, self.config.query_tf)

    def retrieve(self, query: RetrievalQuery, k: int) -> list[RankedCandidate]:
        if self._index is None:
            raise RuntimeError("prepare() must be called before retrieve().")
        hits = self._index.top_k(self._query(query), k)
        return [
            make_candidate(
                self._units[doc_id],
                rank=rank,
                score=score,
                method=self.name,
                signals={"bm25": score},
            )
            for rank, (doc_id, score) in enumerate(hits, start=1)
        ]

    def explain(
        self, query: RetrievalQuery, unit_id: str, top: int = 5
    ) -> list[tuple[str, float]]:
        """The query terms contributing most to a unit's BM25 score."""
        if self._index is None:
            raise RuntimeError("prepare() must be called before explain().")
        for doc_id, unit in enumerate(self._units):
            if unit.unit_id == unit_id:
                return self._index.explain(
                    self._query(query), doc_id, self._docs[doc_id], top
                )
        raise KeyError(unit_id)
