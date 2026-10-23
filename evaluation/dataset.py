"""Read/write evaluation cases as JSONL (one :class:`EvalCase` per line)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from evaluation.schema import EvalCase


def write_cases(path: Path, cases: Iterable[EvalCase]) -> int:
    """Write cases sorted by ``case_id`` with stable formatting; returns the count."""
    ordered = sorted(cases, key=lambda c: c.case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in ordered:
            handle.write(case.model_dump_json() + "\n")
    return len(ordered)


def read_cases(path: Path) -> list[EvalCase]:
    """Load and validate cases; raises ``ValueError`` naming the bad line."""
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(EvalCase.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid eval case: {exc}") from exc
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate case_id values")
    return cases


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
