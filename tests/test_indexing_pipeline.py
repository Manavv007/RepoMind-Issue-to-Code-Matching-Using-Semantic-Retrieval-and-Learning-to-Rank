"""End-to-end tests for the indexing pipeline and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.indexing.cli import main as cli_main
from app.indexing.pipeline import index_repository
from app.ingestion.errors import (
    InvalidSourceError,
    NoSupportedFilesError,
    RepositoryTooLargeError,
    UnsupportedLanguageError,
)
from app.storage.index_store import IndexStore, sqlite_path_from_url
from tests.conftest import GitRepoFactory, write_files

FILES = {
    "app/auth.py": (
        "import os\n\n\n"
        "def restore_session(token):\n"
        '    """Restore a session."""\n'
        "    return load(token)\n\n\n"
        "class Store:\n"
        "    def load(self, key):\n"
        "        return os.environ.get(key)\n"
    ),
    "app/util.py": "def helper():\n    return 1\n",
    "README.md": "# docs\n",
}


def _store(settings: Settings) -> IndexStore:
    return IndexStore(sqlite_path_from_url(settings.database_url))


def test_end_to_end_index_of_git_repo(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory
):
    repo = tmp_path / "repo"
    sha = git_repo(repo, FILES)
    store = _store(settings)

    result = index_repository(str(repo), settings=settings, store=store)

    record = result.repository
    assert record.commit == sha  # exact commit is recorded
    assert record.num_files == 2
    assert record.num_units == 4  # restore_session, Store, Store.load, helper
    assert result.units_by_kind == {"function": 2, "class": 1, "method": 1}
    assert store.get_repository(record.repo_id) == record

    names = {u.qualified_name for u in store.list_units(record.repo_id)}
    assert names == {"restore_session", "Store", "Store.load", "helper"}


def test_every_indexed_location_maps_to_real_source_lines(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory
):
    repo = tmp_path / "repo"
    git_repo(repo, FILES)
    store = _store(settings)
    result = index_repository(str(repo), settings=settings, store=store)

    for unit in store.list_units(result.repository.repo_id):
        file_lines = (
            (result.checkout_path / unit.file_path)
            .read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .split("\n")
        )
        assert 1 <= unit.start_line <= unit.end_line <= len(file_lines)
        assert unit.source == "\n".join(file_lines[unit.start_line - 1 : unit.end_line])


def test_reindexing_same_commit_is_deterministic(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory
):
    repo = tmp_path / "repo"
    git_repo(repo, FILES)
    store = _store(settings)

    first = index_repository(str(repo), settings=settings, store=store)
    units_1 = store.list_units(first.repository.repo_id)
    second = index_repository(str(repo), settings=settings, store=store)
    units_2 = store.list_units(second.repository.repo_id)

    assert first.repository.repo_id == second.repository.repo_id
    assert units_1 == units_2
    assert len(store.list_repositories()) == 1


def test_different_commits_get_distinct_indexes(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory
):
    repo = tmp_path / "repo"
    first_sha = git_repo(repo, {"a.py": "def old(): pass\n"})
    git_repo(repo, {"a.py": "def new(): pass\n"})
    store = _store(settings)

    old = index_repository(str(repo), commit=first_sha, settings=settings, store=store)
    new = index_repository(str(repo), settings=settings, store=store)

    assert old.repository.repo_id != new.repository.repo_id
    assert [u.symbol_name for u in store.list_units(old.repository.repo_id)] == ["old"]
    assert [u.symbol_name for u in store.list_units(new.repository.repo_id)] == ["new"]


def test_non_git_directory_uses_content_derived_commit(
    tmp_path: Path, settings: Settings
):
    repo = tmp_path / "plain"
    write_files(repo, {"a.py": "def a(): pass\n"})
    store = _store(settings)
    first = index_repository(str(repo), settings=settings, store=store)
    assert first.repository.commit.startswith("local:")
    assert first.warnings

    (repo / "a.py").write_text("def a(): return 2\n")
    second = index_repository(str(repo), settings=settings, store=store)
    assert second.repository.commit != first.repository.commit


def test_dirty_tree_gets_its_own_repo_id(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory
):
    repo = tmp_path / "repo"
    git_repo(repo, {"a.py": "def a(): pass\n"})
    store = _store(settings)
    clean = index_repository(str(repo), settings=settings, store=store)
    (repo / "a.py").write_text("def a(): return 1\n")
    dirty = index_repository(str(repo), settings=settings, store=store)
    assert dirty.repository.dirty
    assert dirty.repository.commit == clean.repository.commit
    assert dirty.repository.repo_id != clean.repository.repo_id


def test_repository_code_is_never_executed(tmp_path: Path, settings: Settings):
    marker = tmp_path / "executed.txt"
    repo = tmp_path / "evil"
    write_files(
        repo,
        {
            "evil.py": (
                f"open({str(marker)!r}, 'w').write('pwned')\n" "def f():\n    pass\n"
            ),
            "setup.py": f"open({str(marker)!r}, 'w').write('pwned')\n",
            "conftest.py": f"open({str(marker)!r}, 'w').write('pwned')\n",
        },
    )
    index_repository(str(repo), settings=settings, store=_store(settings))
    assert not marker.exists()


def test_syntax_error_files_are_reported_not_fatal(tmp_path: Path, settings: Settings):
    repo = tmp_path / "r"
    write_files(repo, {"ok.py": "def ok(): pass\n", "bad.py": "def broken(:\n"})
    result = index_repository(str(repo), settings=settings, store=_store(settings))
    assert result.files_with_syntax_errors == 1
    assert any("syntax errors" in w for w in result.warnings)
    assert result.repository.num_files == 2


def test_binary_generated_and_large_files_are_skipped(
    tmp_path: Path, settings: Settings
):
    repo = tmp_path / "r"
    write_files(
        repo,
        {
            "ok.py": "def ok(): pass\n",
            "gen.py": "# Generated by protoc. DO NOT EDIT.\ndef g(): pass\n",
            "huge.py": "# " + "x" * 2_000_000 + "\n",
        },
    )
    (repo / "blob.py").write_bytes(b"def f():\x00\x01\x02\n")
    result = index_repository(str(repo), settings=settings, store=_store(settings))
    assert result.skipped["generated"] == 1
    assert result.skipped["binary"] == 1
    assert result.skipped["too_large"] == 1
    assert result.repository.num_files == 1


def test_no_supported_files_is_a_clear_error(tmp_path: Path, settings: Settings):
    repo = tmp_path / "r"
    write_files(repo, {"main.js": "console.log(1)\n", "README.md": "# x\n"})
    with pytest.raises(NoSupportedFilesError) as exc:
        index_repository(str(repo), settings=settings, store=_store(settings))
    assert exc.value.code == "no_supported_files"


def test_unsupported_language_is_a_clear_error(tmp_path: Path, settings: Settings):
    with pytest.raises(UnsupportedLanguageError) as exc:
        index_repository(
            str(tmp_path), languages=["rust"], settings=settings, store=_store(settings)
        )
    assert exc.value.code == "unsupported_language"


def test_inaccessible_and_invalid_sources(tmp_path: Path, settings: Settings):
    store = _store(settings)
    with pytest.raises(InvalidSourceError):
        index_repository("https://evil.example/x/y", settings=settings, store=store)
    from app.ingestion.errors import RepositoryAccessError

    with pytest.raises(RepositoryAccessError):
        index_repository(str(tmp_path / "missing"), settings=settings, store=store)


def test_file_count_limit_is_enforced(tmp_path: Path, settings: Settings):
    repo = tmp_path / "r"
    write_files(repo, {f"f{i}.py": "x = 1\n" for i in range(3)})
    limited = settings.model_copy(update={"max_files": 2})
    with pytest.raises(RepositoryTooLargeError):
        index_repository(str(repo), settings=limited, store=_store(limited))


def test_nested_function_option_flows_through(tmp_path: Path, settings: Settings):
    repo = tmp_path / "r"
    write_files(repo, {"a.py": "def outer():\n    def inner():\n        pass\n"})
    store = _store(settings)
    default = index_repository(str(repo), settings=settings, store=store)
    assert default.repository.num_units == 1
    nested = index_repository(
        str(repo), settings=settings, store=store, include_nested_functions=True
    )
    assert nested.repository.num_units == 2


def test_cli_success_prints_json_summary(
    tmp_path: Path, settings: Settings, git_repo: GitRepoFactory, capsys, monkeypatch
):
    repo = tmp_path / "repo"
    sha = git_repo(repo, FILES)
    monkeypatch.setattr("app.indexing.cli.get_settings", lambda: settings)

    code = cli_main([str(repo), "--db", str(tmp_path / "cli.db")])

    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["commit"] == sha
    assert summary["units"] == 4
    assert IndexStore(tmp_path / "cli.db").list_repositories()[0].commit == sha


def test_cli_reports_errors_with_code_and_nonzero_exit(
    tmp_path: Path, settings: Settings, capsys, monkeypatch
):
    monkeypatch.setattr("app.indexing.cli.get_settings", lambda: settings)
    code = cli_main([str(tmp_path / "nope"), "--db", str(tmp_path / "cli.db")])
    assert code == 2
    assert "repository_inaccessible" in capsys.readouterr().err
