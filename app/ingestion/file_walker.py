"""Repository file walker.

Produces a deterministic, sorted list of indexable source files while:

* pruning dependency, build, VCS, and cache directories,
* never following symlinks (prevents escaping the repository root),
* enforcing per-file size, total size, and file-count limits,
* reporting why files were skipped.

The walker only inspects names and sizes; it never opens or executes files.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.ingestion.errors import RepositoryTooLargeError

# Extension -> language. Phase 1 supports Python only.
LANGUAGE_EXTENSIONS: dict[str, str] = {".py": "python"}

DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".eggs",
        "site-packages",
        "build",
        "dist",
        "htmlcov",
        ".idea",
        ".vscode",
    }
)


@dataclass(frozen=True)
class SourceFile:
    """A file selected for indexing."""

    rel_path: str  # POSIX, relative to repo root
    abs_path: Path
    size_bytes: int
    language: str


@dataclass
class WalkResult:
    """Outcome of walking a repository."""

    files: list[SourceFile] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)
    total_bytes: int = 0  # bytes of all non-excluded regular files seen


def walk_repository(
    root: Path,
    *,
    languages: frozenset[str] = frozenset({"python"}),
    max_file_size_bytes: int = 1024 * 1024,
    max_total_bytes: int = 500 * 1024 * 1024,
    max_files: int = 20000,
    excluded_dirs: frozenset[str] = DEFAULT_EXCLUDED_DIRS,
) -> WalkResult:
    """Walk ``root`` and return indexable files in sorted path order.

    Raises:
        RepositoryTooLargeError: total size or candidate file count exceeded.
    """
    root = root.resolve()
    extensions = {
        ext: lang for ext, lang in LANGUAGE_EXTENSIONS.items() if lang in languages
    }
    result = WalkResult()

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(dirpath)

        kept: list[str] = []
        for name in sorted(dirnames):
            child = current / name
            if name in excluded_dirs or name.endswith(".egg-info"):
                result.skipped["excluded_dir"] += 1
            elif child.is_symlink():
                result.skipped["symlink"] += 1
            else:
                kept.append(name)
        dirnames[:] = kept  # prune in place; os.walk honors this

        for name in sorted(filenames):
            path = current / name
            if path.is_symlink():
                result.skipped["symlink"] += 1
                continue
            try:
                size = path.stat().st_size
            except OSError:
                result.skipped["unreadable"] += 1
                continue

            result.total_bytes += size
            if result.total_bytes > max_total_bytes:
                raise RepositoryTooLargeError(
                    "Repository exceeds the size limit of "
                    f"{max_total_bytes // (1024 * 1024)} MB."
                )

            language = extensions.get(path.suffix.lower())
            if language is None:
                result.skipped["unsupported_extension"] += 1
                continue
            if size > max_file_size_bytes:
                result.skipped["too_large"] += 1
                continue
            if not _is_within(path, root):
                result.skipped["outside_root"] += 1
                continue

            if len(result.files) >= max_files:
                raise RepositoryTooLargeError(
                    f"Repository has more than {max_files} indexable files."
                )
            result.files.append(
                SourceFile(
                    rel_path=path.relative_to(root).as_posix(),
                    abs_path=path,
                    size_bytes=size,
                    language=language,
                )
            )

    result.files.sort(key=lambda f: f.rel_path)
    return result


def _is_within(path: Path, root: Path) -> bool:
    """Return True if ``path`` resolves to a location inside ``root``."""
    try:
        path.resolve().relative_to(root)
    except (ValueError, OSError):
        return False
    return True
