"""Retriever interface shared by the product pipeline and the evaluation harness.

Every retrieval method (random baseline, BM25, dense, hybrid, reranked, ...)
implements :class:`Retriever`. A retriever only ever sees the issue text and the
code units of the pre-fix revision: never the gold patch or ground-truth labels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.code_unit import CodeUnit, UnitKind


class RetrievalQuery(BaseModel):
    """Retrieval input: the bug report and nothing else."""

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(description="Stable id (used for deterministic seeding).")
    issue_title: str | None = None
    issue_body: str

    @property
    def text(self) -> str:
        """Title and body joined, as originally written."""
        if self.issue_title:
            return f"{self.issue_title}\n{self.issue_body}"
        return self.issue_body


class RankedCandidate(BaseModel):
    """A retrieved code unit with its rank and the evidence behind it.

    ``score`` is a prioritization signal for ordering, not a calibrated
    probability. ``signals`` retains each method's raw score so methods can be
    evaluated and fused independently later.
    """

    model_config = ConfigDict(frozen=True)

    unit_id: str
    file_path: str
    qualified_name: str
    kind: UnitKind
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    rank: int = Field(ge=1, description="1-based position in the result list.")
    score: float
    method: str = Field(description="Retriever that produced this ranking.")
    signals: Mapping[str, float] = Field(default_factory=dict)


def make_candidate(
    unit: CodeUnit,
    *,
    rank: int,
    score: float,
    method: str,
    signals: Mapping[str, float] | None = None,
) -> RankedCandidate:
    """Build a :class:`RankedCandidate` from an indexed unit."""
    return RankedCandidate(
        unit_id=unit.unit_id,
        file_path=unit.file_path,
        qualified_name=unit.qualified_name,
        kind=unit.kind,
        start_line=unit.start_line,
        end_line=unit.end_line,
        rank=rank,
        score=score,
        method=method,
        signals=dict(signals or {}),
    )


class Retriever(ABC):
    """Ranks the code units of one repository revision for a bug report."""

    #: Short identifier used in reports (e.g. ``"random"``, ``"bm25"``).
    name: str = "retriever"

    def describe(self) -> dict[str, Any]:
        """JSON-serializable parameters, recorded in every evaluation report."""
        return {}

    @abstractmethod
    def prepare(self, units: Sequence[CodeUnit]) -> None:
        """Build any per-revision state (e.g. an inverted index) from ``units``."""

    @abstractmethod
    def retrieve(self, query: RetrievalQuery, k: int) -> list[RankedCandidate]:
        """Return up to ``k`` candidates ordered best-first with ranks 1..n."""
