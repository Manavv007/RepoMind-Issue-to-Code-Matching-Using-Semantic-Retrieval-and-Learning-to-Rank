"""Tests for the repository file walker."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.ingestion.errors import RepositoryTooLargeError
from app.ingestion.file_walker import walk_repository
from tests.conftest import write_files


def _paths(result) -> list[str]:
    return [f.rel_path for f in result.files]


def test_collects_python_files_sorted_and_posix(tmp_path: Path):
    write_files(
        tmp_path,
        {
            "z.py": "x = 1\n",
            "a.py": "x = 1\n",
            "pkg/b.py": "x = 1\n",
            "pkg/sub/c.py": "x = 1\n",
            "README.md": "# hi\n",
        },
    )
    result = walk_repository(tmp_path)
    assert _paths(result) == ["a.py", "pkg/b.py", "pkg/sub/c.py", "z.py"]
    assert result.skipped["unsupported_extension"] == 1


def test_excluded_directories_are_pruned(tmp_path: Path):
    write_files(
        tmp_path,
        {
            "keep.py": "x = 1\n",
            "node_modules/dep/index.py": "x = 1\n",
            ".venv/lib/site.py": "x = 1\n",
            "build/gen.py": "x = 1\n",
            "__pycache__/m.py": "x = 1\n",
            ".git/hooks/pre-commit.py": "x = 1\n",
            "src/pkg.egg-info/PKG.py": "x = 1\n",
        },
    )
    assert _paths(walk_repository(tmp_path)) == ["keep.py"]


def test_oversized_files_are_skipped(tmp_path: Path):
    write_files(tmp_path, {"small.py": "x = 1\n", "big.py": "# " + "x" * 5000 + "\n"})
    result = walk_repository(tmp_path, max_file_size_bytes=1000)
    assert _paths(result) == ["small.py"]
    assert result.skipped["too_large"] == 1


def test_symlinks_are_never_followed(tmp_path: Path):
    outside = tmp_path / "outside"
    repo = tmp_path / "repo"
    write_files(outside, {"secret.py": "SECRET = 1\n"})
    write_files(repo, {"real.py": "x = 1\n"})
    try:
        os.symlink(outside / "secret.py", repo / "link_file.py")
        os.symlink(outside, repo / "link_dir", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform")

    result = walk_repository(repo)
    assert _paths(result) == ["real.py"]
    assert result.skipped["symlink"] == 2


def test_too_many_files_raises(tmp_path: Path):
    write_files(tmp_path, {f"f{i}.py": "x = 1\n" for i in range(3)})
    with pytest.raises(RepositoryTooLargeError):
        walk_repository(tmp_path, max_files=2)


def test_total_size_limit_raises(tmp_path: Path):
    write_files(tmp_path, {"a.py": "x" * 600, "b.py": "x" * 600})
    with pytest.raises(RepositoryTooLargeError):
        walk_repository(tmp_path, max_total_bytes=1000)


def test_walk_is_deterministic(tmp_path: Path):
    write_files(tmp_path, {f"d{i % 3}/f{i}.py": "x = 1\n" for i in range(12)})
    assert _paths(walk_repository(tmp_path)) == _paths(walk_repository(tmp_path))
