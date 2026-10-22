"""Shared test fixtures: isolated settings and a throwaway git repo builder."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from app.config import Settings

GitRepoFactory = Callable[..., str]


def run_git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that keep all data (checkouts, DB) inside ``tmp_path``."""
    return Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{(tmp_path / 'index.db').as_posix()}",
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> GitRepoFactory:
    """Create a git repo with one commit; returns the full commit SHA.

    Usage: ``sha = git_repo(path, {"a.py": "..."})``. Calling again with the same
    path adds another commit and returns its SHA.
    """

    def _make(path: Path, files: dict[str, str], message: str = "commit") -> str:
        path.mkdir(parents=True, exist_ok=True)
        if not (path / ".git").exists():
            run_git(path, "init", "-q")
        write_files(path, files)
        run_git(path, "add", "-A")
        run_git(path, "commit", "-q", "-m", message)
        return run_git(path, "rev-parse", "HEAD")

    return _make
