"""Tests for the repository loader (commit tracking and input validation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.ingestion.errors import InvalidSourceError, RepositoryAccessError
from app.ingestion.repository_loader import (
    RepositoryLoader,
    parse_github_url,
    validate_commit,
)
from tests.conftest import GitRepoFactory


def test_local_git_repo_records_head_commit(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory
):
    repo = tmp_path / "repo"
    sha = git_repo(repo, {"a.py": "x = 1\n"})
    loaded = RepositoryLoader(settings).load(str(repo))
    assert loaded.commit == sha
    assert loaded.source_type == "local"
    assert not loaded.dirty
    assert loaded.root == repo.resolve()


def test_dirty_working_tree_is_flagged(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory
):
    repo = tmp_path / "repo"
    git_repo(repo, {"a.py": "x = 1\n"})
    (repo / "a.py").write_text("x = 2\n")
    loaded = RepositoryLoader(settings).load(str(repo))
    assert loaded.dirty
    assert any("uncommitted" in w for w in loaded.warnings)


def test_non_git_directory_has_no_commit(tmp_path: Path, settings: Settings):
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "a.py").write_text("x = 1\n")
    loaded = RepositoryLoader(settings).load(str(tmp_path / "plain"))
    assert loaded.commit is None
    assert loaded.warnings


def test_pinned_commit_checks_out_that_revision(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory
):
    repo = tmp_path / "repo"
    first = git_repo(repo, {"a.py": "VERSION = 1\n"})
    second = git_repo(repo, {"a.py": "VERSION = 2\n"})
    assert first != second

    loader = RepositoryLoader(settings)
    old = loader.load(str(repo), commit=first[:10])  # abbreviated hash accepted
    assert old.commit == first  # resolved to the full hash
    assert (old.root / "a.py").read_text().strip() == "VERSION = 1"
    assert first in old.root.name

    new = loader.load(str(repo), commit=second)
    assert (new.root / "a.py").read_text().strip() == "VERSION = 2"
    # The original working tree is never modified.
    assert (repo / "a.py").read_text().strip() == "VERSION = 2"


def test_same_commit_reuses_checkout(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory
):
    repo = tmp_path / "repo"
    sha = git_repo(repo, {"a.py": "x = 1\n"})
    loader = RepositoryLoader(settings)
    assert (
        loader.load(str(repo), commit=sha).root
        == loader.load(str(repo), commit=sha).root
    )
    leftovers = [
        p for p in loader.checkouts_dir.iterdir() if p.name.startswith(".tmp-")
    ]
    assert leftovers == []


def test_unknown_commit_raises_access_error(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory
):
    repo = tmp_path / "repo"
    git_repo(repo, {"a.py": "x = 1\n"})
    with pytest.raises(RepositoryAccessError):
        RepositoryLoader(settings).load(str(repo), commit="deadbeef" * 5)
    leftovers = list(RepositoryLoader(settings).checkouts_dir.glob(".tmp-*"))
    assert leftovers == []


@pytest.mark.parametrize(
    "bad", ["main", "HEAD~1", "--upload-pack=evil", "abc", "zz" * 10, "a" * 41, ""]
)
def test_invalid_commit_values_rejected(bad: str):
    with pytest.raises(InvalidSourceError):
        validate_commit(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "https://evil.com/owner/repo",
        "http://github.com/owner/repo",
        "https://github.com/owner",
        "https://github.com/owner/repo/extra",
        "https://github.com/../repo",
        "ext::sh -c id",
        "git@github.com:owner/repo.git",
        "file:///etc",
    ],
)
def test_non_github_urls_rejected(bad: str, settings: Settings):
    with pytest.raises(InvalidSourceError):
        RepositoryLoader(settings).load(bad)


def test_valid_github_urls_parse():
    assert parse_github_url("https://github.com/psf/requests") == ("psf", "requests")
    assert parse_github_url("https://github.com/psf/requests.git") == (
        "psf",
        "requests",
    )
    assert parse_github_url("https://github.com/psf/requests/") == ("psf", "requests")


def test_missing_path_and_empty_source(settings: Settings, tmp_path: Path):
    loader = RepositoryLoader(settings)
    with pytest.raises(RepositoryAccessError):
        loader.load(str(tmp_path / "nope"))
    with pytest.raises(InvalidSourceError):
        loader.load("   ")


def test_commit_on_non_git_directory_is_invalid(tmp_path: Path, settings: Settings):
    (tmp_path / "plain").mkdir()
    with pytest.raises(InvalidSourceError):
        RepositoryLoader(settings).load(str(tmp_path / "plain"), commit="a" * 40)
