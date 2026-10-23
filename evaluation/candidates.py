"""Candidate pool definition for bug-localization evaluation.

The pool is the set of code units a retriever ranks. It is defined **without
reference to any label**: leaf units (functions and methods) outside test code.
Excluding tests mirrors how bugs are localized in practice and keeps the pool
from being inflated by thousands of test functions. The rule is recorded in every
report so results stay comparable.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.schemas.code_unit import CodeUnit, UnitKind

CANDIDATE_KINDS: tuple[UnitKind, ...] = ("function", "method")
TEST_DIR_NAMES = frozenset({"tests", "test", "testing"})
POOL_RULE = "leaf units (function, method) excluding test files"


def is_test_path(path: str) -> bool:
    """True for files under test directories or named like test modules."""
    parts = path.split("/")
    name = parts[-1]
    if any(part in TEST_DIR_NAMES for part in parts[:-1]):
        return True
    return (
        name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
    )


def candidate_units(units: Iterable[CodeUnit]) -> list[CodeUnit]:
    """Filter indexed units down to the evaluation candidate pool."""
    return [
        u for u in units if u.kind in CANDIDATE_KINDS and not is_test_path(u.file_path)
    ]
