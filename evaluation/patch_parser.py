"""Unified-diff parser producing old-side (pre-fix) change locations.

Only what labeling needs is extracted: for each file, the old line numbers that
the patch removes or rewrites, and the points where it inserts new lines.

Hunk parsing is driven by the line counts in each ``@@`` header, so file content
that happens to start with ``--- `` or ``diff `` cannot be mistaken for headers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DEVNULL = "/dev/null"


class PatchParseError(ValueError):
    """The patch text is not a well-formed unified diff."""


@dataclass(frozen=True)
class Insertion:
    """Lines added after old line ``after_old_line`` (0 = start of file)."""

    after_old_line: int
    lines: tuple[str, ...]


@dataclass
class FilePatch:
    """Changes to one file."""

    old_path: str | None  # None for newly created files
    new_path: str | None  # None for deleted files
    deleted_old_lines: list[int] = field(default_factory=list)
    insertions: list[Insertion] = field(default_factory=list)
    added_lines: list[str] = field(default_factory=list)
    hunk_count: int = 0

    @property
    def is_new(self) -> bool:
        return self.old_path is None

    @property
    def is_deleted(self) -> bool:
        return self.new_path is None

    @property
    def path(self) -> str:
        """Path to report: pre-fix path when it exists, otherwise the new path."""
        return self.old_path or self.new_path or ""


def _strip_prefix(raw: str, prefix: str) -> str | None:
    """Extract a path from a ``---``/``+++`` line; None for /dev/null."""
    path = raw[4:].split("\t", 1)[0].strip()
    if path == _DEVNULL:
        return None
    if path.startswith(prefix):
        path = path[len(prefix) :]
    return path


def parse_unified_diff(patch: str) -> list[FilePatch]:
    """Parse a (git-style) unified diff.

    Raises:
        PatchParseError: on malformed hunks (counts that do not match content).
    """
    lines = patch.replace("\r\n", "\n").split("\n")
    files: list[FilePatch] = []
    current: FilePatch | None = None
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        if line.startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
            old_path = _strip_prefix(line, "a/")
            new_path = _strip_prefix(lines[i + 1], "b/")
            current = FilePatch(old_path=old_path, new_path=new_path)
            files.append(current)
            i += 2
            continue

        match = _HUNK_RE.match(line)
        if match is not None:
            if current is None:
                raise PatchParseError("Hunk header before any file header.")
            old_start = int(match.group(1))
            old_count = int(match.group(2) if match.group(2) is not None else 1)
            new_count = int(match.group(4) if match.group(4) is not None else 1)
            # With zero old lines the header number is the line *before* the edit.
            old_line = old_start if old_count > 0 else old_start + 1
            i += 1
            i = _consume_hunk(lines, i, old_line, old_count, new_count, current)
            current.hunk_count += 1
            continue

        i += 1

    return files


def _consume_hunk(
    lines: list[str],
    i: int,
    old_line: int,
    old_count: int,
    new_count: int,
    target: FilePatch,
) -> int:
    old_left, new_left = old_count, new_count
    pending: list[str] = []
    pending_after = old_line - 1

    def flush() -> None:
        nonlocal pending
        if pending:
            target.insertions.append(Insertion(pending_after, tuple(pending)))
            target.added_lines.extend(pending)
            pending = []

    while (old_left > 0 or new_left > 0) and i < len(lines):
        text = lines[i]
        if text.startswith("\\"):  # "\ No newline at end of file"
            i += 1
            continue
        tag = text[:1]
        if tag == "-":
            flush()
            target.deleted_old_lines.append(old_line)
            old_line += 1
            old_left -= 1
        elif tag == "+":
            if not pending:
                pending_after = old_line - 1
            pending.append(text[1:])
            new_left -= 1
        elif tag == " " or text == "":  # empty line = context line stripped of space
            flush()
            old_line += 1
            old_left -= 1
            new_left -= 1
        else:
            raise PatchParseError(f"Unexpected line inside hunk: {text[:60]!r}")
        i += 1

    flush()
    if old_left != 0 or new_left != 0:
        raise PatchParseError("Hunk ended before its declared line counts were met.")
    # Swallow trailing "\ No newline" markers belonging to this hunk.
    while i < len(lines) and lines[i].startswith("\\"):
        i += 1
    return i
