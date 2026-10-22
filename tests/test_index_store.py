"""Tests for the SQLite index store."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import PROJECT_ROOT
from app.parsing.symbol_extractor import extract_file
from app.schemas.code_unit import RepositoryRecord
from app.storage.index_store import IndexStore, sqlite_path_from_url

SOURCE = (
    b"import os\n\n"
    b"def f(a):\n    '''doc'''\n    return os.getcwd()\n\n"
    b"class C:\n    def m(self):\n        self.x()\n"
)


def _record(repo_id: str = "r1", units: int = 0, files: int = 1) -> RepositoryRecord:
    return RepositoryRecord(
        repo_id=repo_id,
        source="github.com/o/r",
        source_type="github",
        commit="a" * 40,
        languages=("python",),
        num_files=files,
        num_units=units,
        indexed_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture
def store(tmp_path: Path) -> IndexStore:
    return IndexStore(tmp_path / "db" / "index.db")


def test_roundtrip_preserves_all_fields(store: IndexStore):
    extracted = extract_file("pkg/m.py", "python", SOURCE)
    store.replace_repository(
        _record(units=len(extracted.units)), [extracted.record], extracted.units
    )

    assert store.list_units("r1") == sorted(
        extracted.units, key=lambda u: (u.file_path, u.start_line, u.qualified_name)
    )
    assert store.list_files("r1") == [extracted.record]
    repo = store.get_repository("r1")
    assert repo is not None
    assert repo.commit == "a" * 40
    assert repo.languages == ("python",)


def test_get_unit_and_kind_filter(store: IndexStore):
    extracted = extract_file("m.py", "python", SOURCE)
    store.replace_repository(_record(), [extracted.record], extracted.units)

    unit = store.get_unit("r1", "m.py::C.m:8")
    assert unit is not None and unit.kind == "method"
    assert unit.calls[0].expression == "self.x"
    assert store.get_unit("r1", "missing") is None
    assert [u.kind for u in store.list_units("r1", kinds=["class"])] == ["class"]


def test_replace_is_idempotent_and_does_not_duplicate(store: IndexStore):
    extracted = extract_file("m.py", "python", SOURCE)
    for _ in range(3):
        store.replace_repository(_record(), [extracted.record], extracted.units)
    assert len(store.list_units("r1")) == len(extracted.units)
    assert len(store.list_files("r1")) == 1
    assert len(store.list_repositories()) == 1


def test_repositories_are_isolated(store: IndexStore):
    a = extract_file("a.py", "python", b"def a(): pass\n")
    b = extract_file("b.py", "python", b"def b(): pass\n")
    store.replace_repository(_record("r1"), [a.record], a.units)
    store.replace_repository(_record("r2"), [b.record], b.units)
    assert [u.symbol_name for u in store.list_units("r1")] == ["a"]
    assert [u.symbol_name for u in store.list_units("r2")] == ["b"]
    store.delete_repository("r1")
    assert store.list_units("r1") == []
    assert [u.symbol_name for u in store.list_units("r2")] == ["b"]


def test_failed_replace_rolls_back_and_keeps_old_data(store: IndexStore):
    good = extract_file("a.py", "python", b"def a(): pass\n")
    store.replace_repository(_record(), [good.record], good.units)

    duplicate = [good.units[0], good.units[0]]  # violates (repo_id, unit_id) PK
    with pytest.raises(Exception):  # noqa: B017 - sqlite3.IntegrityError
        store.replace_repository(_record(), [good.record], duplicate)
    assert [u.symbol_name for u in store.list_units("r1")] == ["a"]


def test_sql_injection_in_ids_is_harmless(store: IndexStore):
    assert store.get_unit("r1' OR '1'='1", "x") is None
    assert store.get_repository("'; DROP TABLE repositories; --") is None
    assert store.list_repositories() == []


def test_sqlite_path_from_url(tmp_path: Path):
    abs_path = (tmp_path / "x.db").as_posix()
    assert sqlite_path_from_url(f"sqlite:///{abs_path}") == Path(abs_path)
    rel = sqlite_path_from_url("sqlite:///./data/x.db")
    assert rel == (PROJECT_ROOT / "data" / "x.db").resolve()
    with pytest.raises(ValueError):
        sqlite_path_from_url("postgresql://u@h/db")
