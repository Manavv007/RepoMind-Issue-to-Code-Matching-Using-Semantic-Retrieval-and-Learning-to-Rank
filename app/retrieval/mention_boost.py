"""BM25 boosted by code the issue names explicitly (symbols, files, modules).

    score(u) = bm25(u) / max_bm25 + w_mention * [issue names u's symbol]
                                   + w_path    * [issue names u's file]

The default weights were chosen on the ``dev`` split (48-configuration grid,
``experiments/bm25_signals.py``; report ``evaluation/reports/exp_bm25_signals.json``).
On held-out ``test`` they changed function MRR by +0.045 [-0.032, +0.133] and
hit@5 by 0, so treat this as a small ordering refinement, not a recall gain.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.issue_understanding.entities import extract_entities, mentions_file
from app.retrieval.base import (
    RankedCandidate,
    RetrievalQuery,
    Retriever,
    make_candidate,
)
from app.retrieval.bm25 import BM25Config, BM25Retriever
from app.schemas.code_unit import CodeUnit


class MentionBoostConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    w_mention: float = Field(default=0.25, ge=0)
    w_path: float = Field(default=0.5, ge=0)
    bm25: BM25Config = Field(default_factory=BM25Config)


class MentionBoostRetriever(Retriever):
    name = "bm25-mentions"

    def __init__(self, config: MentionBoostConfig | None = None) -> None:
        self.config = config or MentionBoostConfig()
        self.bm25 = BM25Retriever(self.config.bm25)
        self._units: list[CodeUnit] = []

    def describe(self) -> dict[str, Any]:
        return self.config.model_dump()

    def prepare(self, units: Sequence[CodeUnit]) -> None:
        self._units = sorted(units, key=lambda u: u.unit_id)
        self.bm25.prepare(self._units)

    def retrieve(self, query: RetrievalQuery, k: int) -> list[RankedCandidate]:
        if not self._units:
            return []
        lexical = self.bm25.score_all(query)  # no full candidate list (memory)
        top = max(lexical.values(), default=0.0) or 1.0
        entities = extract_entities(query.text)
        file_hit: dict[str, bool] = {}
        scored: list[tuple[float, str, CodeUnit]] = []
        for unit in self._units:
            if unit.file_path not in file_hit:
                file_hit[unit.file_path] = mentions_file(unit.file_path, entities)
            mention = 1.0 if unit.symbol_name in entities.names else 0.0
            path = 1.0 if file_hit[unit.file_path] else 0.0
            bm25 = lexical.get(unit.unit_id, 0.0)
            score = (
                bm25 / top + self.config.w_mention * mention + self.config.w_path * path
            )
            if score > 0:
                scored.append((score, unit.unit_id, unit))
        best = heapq.nsmallest(k, scored, key=lambda item: (-item[0], item[1]))
        return [
            make_candidate(
                unit,
                rank=rank,
                score=score,
                method=self.name,
                signals={
                    "bm25": lexical.get(unit.unit_id, 0.0),
                    "mention": float(unit.symbol_name in entities.names),
                    "path_mention": float(file_hit[unit.file_path]),
                },
            )
            for rank, (score, _, unit) in enumerate(best, start=1)
        ]
