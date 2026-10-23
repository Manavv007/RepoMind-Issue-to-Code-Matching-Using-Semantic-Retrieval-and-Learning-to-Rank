"""Derive ground-truth labels by mapping a fix patch onto the pre-fix index.

Annotation rules (version ``patch-hunk-to-function/v1``):

1. A **removed or rewritten** old line labels the function/method whose line range
   contains it.
2. An **insertion** after old line ``N`` labels the function containing both ``N``
   and ``N+1`` (it lands inside the body). If ``N`` is a function's last line, the
   insertion belongs to that function only when the first added line is indented
   deeper than the ``def`` (it extends the body rather than starting a sibling).
3. Edits outside any function (module level, class bodies, new functions) add no
   unit label; they are tallied in ``unlocated_changes``.
4. Only **leaf units** (functions and methods) are labeled; class units are not,
   because a class span would trivially contain every one of its methods.

File labels are the Python files that exist at the base commit and are modified
or deleted by the patch.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.schemas.code_unit import CodeUnit
from evaluation.patch_parser import FilePatch, Insertion, parse_unified_diff
from evaluation.schema import GoldUnit, GroundTruth

LEAF_KINDS = ("function", "method")


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip(" \t"))


def _containing_leaf(units: Sequence[CodeUnit], line: int) -> CodeUnit | None:
    for unit in units:
        if unit.start_line <= line <= unit.end_line:
            return unit
    return None


def _def_indent(unit: CodeUnit) -> int:
    """Indentation of the definition line (skipping decorator lines)."""
    for text in unit.source.split("\n"):
        stripped = text.lstrip()
        if stripped.startswith(("def ", "async def ")):
            return _indent(text)
    return _indent(unit.source.split("\n", 1)[0])


def _unit_for_insertion(units: Sequence[CodeUnit], ins: Insertion) -> CodeUnit | None:
    after = ins.after_old_line
    before_unit = _containing_leaf(units, after) if after >= 1 else None
    after_unit = _containing_leaf(units, after + 1)
    if before_unit is not None and before_unit is after_unit:
        return before_unit  # strictly inside one function body
    if before_unit is not None and after == before_unit.end_line:
        first_added = next((t for t in ins.lines if t.strip()), None)
        if first_added is not None and _indent(first_added) > _def_indent(before_unit):
            return before_unit  # extends the end of the function body
    return None


def label_file_patch(
    file_patch: FilePatch, units: Sequence[CodeUnit]
) -> tuple[list[CodeUnit], int]:
    """Return ``(labeled leaf units, number of unlocated change groups)``."""
    leaves = sorted(
        (u for u in units if u.kind in LEAF_KINDS), key=lambda u: u.start_line
    )
    hit: dict[str, CodeUnit] = {}
    unlocated = 0
    deleted = sorted(file_patch.deleted_old_lines)

    # Each run of consecutive removed lines is one change group.
    for run in _runs(deleted):
        located = False
        for line in run:
            unit = _containing_leaf(leaves, line)
            if unit is not None:
                hit[unit.unit_id] = unit
                located = True
        if not located:
            unlocated += 1

    deleted_set = set(deleted)
    for ins in file_patch.insertions:
        unit = _unit_for_insertion(leaves, ins)
        if unit is not None:
            hit[unit.unit_id] = unit
        elif ins.after_old_line not in deleted_set:
            # Pure insertion outside any function. (An insertion right after
            # removed lines is a rewrite already counted above.)
            unlocated += 1

    ordered = sorted(hit.values(), key=lambda u: (u.start_line, u.qualified_name))
    return ordered, unlocated


def _runs(sorted_lines: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for line in sorted_lines:
        if runs and line == runs[-1][-1] + 1:
            runs[-1].append(line)
        else:
            runs.append([line])
    return runs


def label_patch(
    patch: str,
    units_by_file: dict[str, list[CodeUnit]],
    indexed_files: set[str],
) -> GroundTruth:
    """Compute :class:`GroundTruth` for ``patch`` against the pre-fix index.

    Args:
        patch: Unified diff of the fix (source files only).
        units_by_file: Pre-fix code units grouped by file path.
        indexed_files: Paths of all files present in the pre-fix index.
    """
    files: list[str] = []
    new_files: list[str] = []
    non_python: list[str] = []
    gold: dict[str, GoldUnit] = {}
    unlocated = 0

    for fp in parse_unified_diff(patch):
        path = fp.path
        if not path.endswith(".py"):
            non_python.append(path)
            continue
        if fp.is_new:
            new_files.append(path)
            continue
        if path not in indexed_files:
            # Exists upstream but was not indexed (excluded dir, too large, ...).
            non_python.append(path)
            continue
        files.append(path)
        if fp.is_deleted:
            continue
        labeled, loose = label_file_patch(fp, units_by_file.get(path, []))
        unlocated += loose
        for unit in labeled:
            gold[unit.unit_id] = GoldUnit(
                unit_id=unit.unit_id,
                file_path=unit.file_path,
                qualified_name=unit.qualified_name,
                kind=unit.kind,
                start_line=unit.start_line,
                end_line=unit.end_line,
            )

    return GroundTruth(
        files=tuple(sorted(set(files))),
        units=tuple(sorted(gold.values(), key=lambda g: (g.file_path, g.start_line))),
        new_files=tuple(sorted(set(new_files))),
        non_python_files=tuple(sorted(set(non_python))),
        unlocated_changes=unlocated,
    )
