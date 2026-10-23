"""Random-ranking baseline: the floor every real retriever must beat."""

from __future__ import annotations

import random
from collections.abc import Sequence

from app.retrieval.base import (
    RankedCandidate,
    RetrievalQuery,
    Retriever,
    make_candidate,
)
from app.schemas.code_unit import CodeUnit


class RandomRetriever(Retriever):
    """Returns a uniformly random ordering of the repository's code units.

    The ordering is a pure function of ``(seed, query_id)`` and of the *set* of
    units, so results are reproducible and independent of evaluation order.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._units: list[CodeUnit] = []

    def prepare(self, units: Sequence[CodeUnit]) -> None:
        # Sort so the ordering does not depend on how units were loaded.
        self._units = sorted(units, key=lambda u: u.unit_id)

    def retrieve(self, query: RetrievalQuery, k: int) -> list[RankedCandidate]:
        rng = random.Random(f"{self._seed}:{query.query_id}")
        n = min(k, len(self._units))
        chosen = rng.sample(self._units, n)
        return [
            make_candidate(
                unit,
                rank=i,
                score=1.0 - (i - 1) / max(len(self._units), 1),
                method=self.name,
            )
            for i, unit in enumerate(chosen, start=1)
        ]
