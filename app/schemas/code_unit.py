"""Core data models for repository indexing.

These models are the contract between the parser, the index store, and (in
later phases) the retrieval and evaluation code. Line numbers are 1-based and
inclusive. File paths are POSIX-style and relative to the repository root.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

UnitKind = Literal["function", "method", "class"]


class CallRef(BaseModel):
    """A call expression found inside a code unit.

    Calls are recorded *syntactically*: ``expression`` is the callee text as
    written (e.g. ``self.session.refresh``) and is **not** resolved to a
    definition. Resolution is a later-phase concern, so every call carries
    ``provenance="syntactic"`` to keep known and inferred links distinguishable.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Final identifier of the callee, e.g. 'refresh'.")
    expression: str = Field(description="Callee text as written in source.")
    line: int = Field(ge=1, description="1-based line of the call.")
    provenance: Literal["syntactic"] = "syntactic"


class CodeUnit(BaseModel):
    """A function, method, or class extracted from a source file."""

    model_config = ConfigDict(frozen=True)

    unit_id: str = Field(
        description="Stable id: '<file_path>::<qualified_name>:<start_line>'."
    )
    language: str
    file_path: str = Field(description="POSIX path relative to the repository root.")
    kind: UnitKind
    symbol_name: str
    qualified_name: str = Field(description="Dotted path, e.g. 'Session.refresh'.")
    parent: str | None = Field(
        default=None, description="Qualified name of the enclosing class, if any."
    )
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    signature: str
    docstring: str | None = None
    source: str = Field(description="Exact source text of the unit (LF line endings).")
    content_hash: str = Field(description="SHA-256 of ``source``.")
    is_async: bool = False
    decorators: tuple[str, ...] = ()
    calls: tuple[CallRef, ...] = ()


class FileRecord(BaseModel):
    """Per-file metadata produced during indexing."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(description="POSIX path relative to the repository root.")
    language: str
    sha256: str = Field(description="SHA-256 of the raw file bytes.")
    size_bytes: int = Field(ge=0)
    num_lines: int = Field(ge=0)
    has_syntax_errors: bool = False
    imports: tuple[str, ...] = Field(
        default=(), description="Module-level import statements, whitespace-normalized."
    )


class RepositoryRecord(BaseModel):
    """Identity and provenance of an indexed repository revision."""

    model_config = ConfigDict(frozen=True)

    repo_id: str = Field(description="Deterministic id derived from source + commit.")
    source: str = Field(
        description="Canonical source: GitHub 'owner/repo' or local path."
    )
    source_type: Literal["github", "local"]
    commit: str = Field(
        description="Resolved commit hash, or 'local:<hash>' for non-git directories."
    )
    dirty: bool = Field(
        default=False,
        description="True if the working tree had uncommitted changes when indexed.",
    )
    languages: tuple[str, ...] = ()
    num_files: int = 0
    num_units: int = 0
    indexed_at: str = Field(default="", description="UTC ISO-8601 timestamp.")
