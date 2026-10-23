"""Evaluation case schema.

An :class:`EvalCase` separates what a retriever may see (:class:`IssueInput`) from
the ground truth used only for scoring (:class:`GroundTruth`). The gold patch is
never stored in a case, only its hash, so it cannot leak into retrieval input.

Line numbers refer to the **pre-fix** revision (``base_commit``). File paths are
POSIX-style and relative to the repository root.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.code_unit import UnitKind

SCHEMA_VERSION = 1
Split = Literal["dev", "test"]


class IssueInput(BaseModel):
    """The bug report exactly as written: the only text a retriever receives."""

    model_config = ConfigDict(frozen=True)

    title: str | None = None
    body: str


class GoldUnit(BaseModel):
    """A function/method at the pre-fix revision that the fix modified."""

    model_config = ConfigDict(frozen=True)

    unit_id: str = Field(description="Matches CodeUnit.unit_id in the pre-fix index.")
    file_path: str
    qualified_name: str
    kind: UnitKind
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class GroundTruth(BaseModel):
    """Relevant locations derived from the fix patch (never shown to retrievers).

    Relevance definition:

    * **File level**: a Python file that exists at ``base_commit`` and is modified
      (or deleted) by the fix patch.
    * **Function level**: a function or method (leaf unit) whose line range
      contains a line removed or changed by the patch, or into whose body the
      patch inserts lines. Edits that fall outside any function (module level,
      class bodies) are counted in ``unlocated_changes`` and produce no unit label.
    """

    model_config = ConfigDict(frozen=True)

    files: tuple[str, ...] = ()
    units: tuple[GoldUnit, ...] = ()
    new_files: tuple[str, ...] = Field(
        default=(),
        description="Python files created by the patch (no pre-fix location).",
    )
    non_python_files: tuple[str, ...] = ()
    unlocated_changes: int = Field(
        default=0, ge=0, description="Change groups not attributable to a function."
    )
    annotation_method: str = "patch-hunk-to-function/v1"

    @property
    def has_file_labels(self) -> bool:
        return bool(self.files)

    @property
    def has_function_labels(self) -> bool:
        return bool(self.units)


class LeakageFlags(BaseModel):
    """Leakage diagnostics computed at build time."""

    model_config = ConfigDict(frozen=True)

    patch_lines_in_issue: int = Field(
        default=0,
        ge=0,
        description="Distinct non-trivial added patch lines quoted in the issue.",
    )
    issue_contains_patch_lines: bool = False
    issue_mentions_gold_file: bool = Field(
        default=False,
        description="Informational: the issue names a gold file path or module.",
    )
    duplicate_of: str | None = Field(
        default=None, description="case_id of a near-identical earlier case."
    )


class Provenance(BaseModel):
    """Where the case and its labels came from."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    dataset_revision: str | None = None
    instance_id: str
    license_note: str
    patch_sha256: str = Field(description="Hash of the gold patch (patch not stored).")
    patch_applies_cleanly: bool | None = Field(
        default=None, description="`git apply --check` result at base_commit."
    )


class EvalCase(BaseModel):
    """One issue-to-code localization case."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    case_id: str
    repo: str = Field(description="GitHub 'owner/repo'.")
    repo_url: str
    base_commit: str = Field(min_length=40, max_length=40)
    issue: IssueInput
    ground_truth: GroundTruth
    fail_to_pass: tuple[str, ...] = Field(
        default=(), description="Tests that fail before and pass after the fix."
    )
    suitable: bool = Field(description="Usable for localization evaluation.")
    unsuitable_reasons: tuple[str, ...] = ()
    leakage: LeakageFlags = LeakageFlags()
    split: Split | None = None
    provenance: Provenance
