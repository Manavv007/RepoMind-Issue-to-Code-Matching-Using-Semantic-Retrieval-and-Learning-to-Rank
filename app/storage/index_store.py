"""SQLite store for repository index metadata, files, and code units.

Uses the standard-library ``sqlite3`` with parameterized queries only. A
repository's rows are replaced atomically in one transaction, so a failed
re-index never leaves a half-written index.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from app.config import PROJECT_ROOT
from app.schemas.code_unit import (
    CallRef,
    CodeUnit,
    FileRecord,
    RepositoryRecord,
    UnitKind,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repositories (
    repo_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_type TEXT NOT NULL,
    commit_hash TEXT NOT NULL,
    dirty INTEGER NOT NULL,
    languages TEXT NOT NULL,
    num_files INTEGER NOT NULL,
    num_units INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    repo_id TEXT NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    num_lines INTEGER NOT NULL,
    has_syntax_errors INTEGER NOT NULL,
    imports TEXT NOT NULL,
    PRIMARY KEY (repo_id, path)
);

CREATE TABLE IF NOT EXISTS code_units (
    repo_id TEXT NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    unit_id TEXT NOT NULL,
    language TEXT NOT NULL,
    file_path TEXT NOT NULL,
    kind TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    parent TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    signature TEXT NOT NULL,
    docstring TEXT,
    source TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    is_async INTEGER NOT NULL,
    decorators TEXT NOT NULL,
    calls TEXT NOT NULL,
    PRIMARY KEY (repo_id, unit_id)
);

CREATE INDEX IF NOT EXISTS idx_units_file ON code_units(repo_id, file_path);
CREATE INDEX IF NOT EXISTS idx_units_kind ON code_units(repo_id, kind);
"""


def sqlite_path_from_url(database_url: str) -> Path:
    """Convert ``sqlite:///relative/or/abs.db`` into a filesystem path.

    Relative paths are resolved against the project root, not the CWD, so the
    same database is used regardless of where a command is launched.
    """
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError(f"Only sqlite:/// URLs are supported, got: {database_url!r}")
    path = Path(database_url[len(prefix) :])
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class IndexStore:
    """Persistence layer for indexed repositories."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:  # commit on success, rollback on exception
                yield conn
        finally:
            conn.close()

    # --------------------------------------------------------------- writes
    def replace_repository(
        self,
        record: RepositoryRecord,
        files: Sequence[FileRecord],
        units: Sequence[CodeUnit],
    ) -> None:
        """Atomically replace everything stored for ``record.repo_id``."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM repositories WHERE repo_id = ?", (record.repo_id,)
            )
            conn.execute(
                "INSERT INTO repositories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.repo_id,
                    record.source,
                    record.source_type,
                    record.commit,
                    int(record.dirty),
                    json.dumps(list(record.languages)),
                    record.num_files,
                    record.num_units,
                    record.indexed_at,
                ),
            )
            conn.executemany(
                "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        record.repo_id,
                        f.path,
                        f.language,
                        f.sha256,
                        f.size_bytes,
                        f.num_lines,
                        int(f.has_syntax_errors),
                        json.dumps(list(f.imports)),
                    )
                    for f in files
                ],
            )
            conn.executemany(
                "INSERT INTO code_units VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        record.repo_id,
                        u.unit_id,
                        u.language,
                        u.file_path,
                        u.kind,
                        u.symbol_name,
                        u.qualified_name,
                        u.parent,
                        u.start_line,
                        u.end_line,
                        u.signature,
                        u.docstring,
                        u.source,
                        u.content_hash,
                        int(u.is_async),
                        json.dumps(list(u.decorators)),
                        json.dumps([c.model_dump() for c in u.calls]),
                    )
                    for u in units
                ],
            )

    def delete_repository(self, repo_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM repositories WHERE repo_id = ?", (repo_id,))

    # ---------------------------------------------------------------- reads
    def get_repository(self, repo_id: str) -> RepositoryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM repositories WHERE repo_id = ?", (repo_id,)
            ).fetchone()
        if row is None:
            return None
        return RepositoryRecord(
            repo_id=row["repo_id"],
            source=row["source"],
            source_type=row["source_type"],
            commit=row["commit_hash"],
            dirty=bool(row["dirty"]),
            languages=tuple(json.loads(row["languages"])),
            num_files=row["num_files"],
            num_units=row["num_units"],
            indexed_at=row["indexed_at"],
        )

    def list_repositories(self) -> list[RepositoryRecord]:
        with self._connect() as conn:
            ids = [
                r["repo_id"]
                for r in conn.execute(
                    "SELECT repo_id FROM repositories ORDER BY repo_id"
                )
            ]
        records = (self.get_repository(i) for i in ids)
        return [r for r in records if r is not None]

    def list_files(self, repo_id: str) -> list[FileRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM files WHERE repo_id = ? ORDER BY path", (repo_id,)
            ).fetchall()
        return [
            FileRecord(
                path=r["path"],
                language=r["language"],
                sha256=r["sha256"],
                size_bytes=r["size_bytes"],
                num_lines=r["num_lines"],
                has_syntax_errors=bool(r["has_syntax_errors"]),
                imports=tuple(json.loads(r["imports"])),
            )
            for r in rows
        ]

    def list_units(
        self, repo_id: str, kinds: Sequence[UnitKind] | None = None
    ) -> list[CodeUnit]:
        """Return units ordered by file path then start line (deterministic)."""
        query = "SELECT * FROM code_units WHERE repo_id = ?"
        params: list[object] = [repo_id]
        if kinds:
            query += f" AND kind IN ({','.join('?' for _ in kinds)})"
            params.extend(kinds)
        query += " ORDER BY file_path, start_line, qualified_name"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def get_unit(self, repo_id: str, unit_id: str) -> CodeUnit | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM code_units WHERE repo_id = ? AND unit_id = ?",
                (repo_id, unit_id),
            ).fetchone()
        return self._row_to_unit(row) if row is not None else None

    @staticmethod
    def _row_to_unit(row: sqlite3.Row) -> CodeUnit:
        return CodeUnit(
            unit_id=row["unit_id"],
            language=row["language"],
            file_path=row["file_path"],
            kind=row["kind"],
            symbol_name=row["symbol_name"],
            qualified_name=row["qualified_name"],
            parent=row["parent"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            signature=row["signature"],
            docstring=row["docstring"],
            source=row["source"],
            content_hash=row["content_hash"],
            is_async=bool(row["is_async"]),
            decorators=tuple(json.loads(row["decorators"])),
            calls=tuple(CallRef(**c) for c in json.loads(row["calls"])),
        )
