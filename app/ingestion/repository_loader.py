"""Repository loader: resolve a GitHub URL or local path to a directory on disk.

Responsibilities (project.md §6.1):

* Accept a public GitHub URL or a local path.
* Optionally pin an exact commit and always record the resolved full hash.
* Never execute repository code. Git is invoked with argument lists (no shell),
  with an empty template (no hooks), ``ext::`` transports disabled, and symlink
  checkout disabled.

Checkouts are stored under ``<data_dir>/checkouts/<slug>__<full_sha>`` so the
same revision is reused rather than re-cloned.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import stat
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.config import Settings, get_settings
from app.ingestion.errors import InvalidSourceError, RepositoryAccessError

logger = logging.getLogger(__name__)

_GITHUB_URL_RE = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://|^git@|^[A-Za-z0-9.-]+@[^/]+:")


@dataclass(frozen=True)
class LoadedRepository:
    """A repository materialized on disk, ready to be walked."""

    root: Path
    source: str
    source_type: Literal["github", "local"]
    commit: str | None
    dirty: bool = False
    warnings: tuple[str, ...] = ()


def parse_github_url(url: str) -> tuple[str, str]:
    """Return ``(owner, repo)`` for a valid ``https://github.com/...`` URL."""
    match = _GITHUB_URL_RE.match(url.strip())
    if match is None:
        raise InvalidSourceError(
            "Only https://github.com/<owner>/<repo> URLs are supported."
        )
    owner, repo = match.group("owner"), match.group("repo")
    if set(owner) == {"."} or set(repo) == {"."}:
        raise InvalidSourceError("Invalid GitHub owner or repository name.")
    return owner, repo


def validate_commit(commit: str) -> str:
    """Validate a user-supplied commit hash (7-40 hex characters)."""
    commit = commit.strip()
    if not _COMMIT_RE.match(commit):
        raise InvalidSourceError(
            "Commit must be a 7-40 character hexadecimal hash "
            "(branch and tag names are not accepted)."
        )
    return commit.lower()


def _force_rmtree(path: Path) -> None:
    """Remove a directory tree, clearing read-only bits (git objects on Windows)."""
    for dirpath, dirnames, filenames in os.walk(path):
        for name in (*dirnames, *filenames):
            try:
                os.chmod(os.path.join(dirpath, name), stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


class RepositoryLoader:
    """Resolves repository sources into on-disk directories with commit info."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def checkouts_dir(self) -> Path:
        return self._settings.data_dir / "checkouts"

    # ------------------------------------------------------------------ API
    def load(self, source: str, commit: str | None = None) -> LoadedRepository:
        """Resolve ``source`` (GitHub URL or local path) to a directory.

        Raises:
            InvalidSourceError: malformed/disallowed source or commit.
            RepositoryAccessError: clone/checkout failed or path missing.
        """
        source = source.strip()
        if not source:
            raise InvalidSourceError("Repository source must not be empty.")
        pinned = validate_commit(commit) if commit else None

        if "::" in source:
            raise InvalidSourceError(
                "Git transport syntax (e.g. 'ext::') is not allowed as a source."
            )
        if _URL_SCHEME_RE.match(source):
            return self._load_github(source, pinned)
        return self._load_local(source, pinned)

    # --------------------------------------------------------------- GitHub
    def _load_github(self, url: str, commit: str | None) -> LoadedRepository:
        owner, repo = parse_github_url(url)
        clone_url = f"https://github.com/{owner}/{repo}.git"
        canonical = f"github.com/{owner}/{repo}".lower()
        root, full_sha = self._materialize(
            clone_url, commit, slug=f"{owner}__{repo}", shallow=commit is None
        )
        return LoadedRepository(
            root=root, source=canonical, source_type="github", commit=full_sha
        )

    # ---------------------------------------------------------------- local
    def _load_local(self, raw_path: str, commit: str | None) -> LoadedRepository:
        path = Path(raw_path).expanduser()
        if not path.is_dir():
            raise RepositoryAccessError(f"Local path is not a directory: {raw_path}")
        root = path.resolve()
        canonical = root.as_posix()
        is_git = self._is_git_worktree(root)

        if commit is not None:
            if not is_git:
                raise InvalidSourceError(
                    "A commit was requested but the local path is not a git repository."
                )
            checkout, full_sha = self._materialize(
                str(root), commit, slug=f"local-{root.name}", shallow=False
            )
            return LoadedRepository(
                root=checkout, source=canonical, source_type="local", commit=full_sha
            )

        if not is_git:
            return LoadedRepository(
                root=root,
                source=canonical,
                source_type="local",
                commit=None,
                warnings=(
                    "Not a git repository; commit is derived from file contents.",
                ),
            )

        head = self._git(["rev-parse", "HEAD"], cwd=root).strip()
        status = self._git(["status", "--porcelain"], cwd=root)
        dirty = bool(status.strip())
        warnings: tuple[str, ...] = ()
        if dirty:
            warnings = (
                "Working tree has uncommitted changes; the index reflects the "
                "working tree, not the recorded commit exactly.",
            )
        return LoadedRepository(
            root=root,
            source=canonical,
            source_type="local",
            commit=head,
            dirty=dirty,
            warnings=warnings,
        )

    def _is_git_worktree(self, root: Path) -> bool:
        try:
            out = self._git(["rev-parse", "--is-inside-work-tree"], cwd=root)
        except RepositoryAccessError:
            return False
        return out.strip() == "true"

    # ------------------------------------------------------------ git helpers
    def _materialize(
        self, clone_source: str, commit: str | None, *, slug: str, shallow: bool
    ) -> tuple[Path, str]:
        """Clone ``clone_source`` (optionally at ``commit``) into the cache.

        Returns ``(checkout_dir, full_sha)``. A checkout for the same full SHA is
        reused when it already exists.
        """
        self.checkouts_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.checkouts_dir / f".tmp-{uuid.uuid4().hex}"
        try:
            clone_args = [
                "-c",
                "core.symlinks=false",
                "clone",
                "--quiet",
                "--template=",
            ]
            if commit is not None:
                clone_args.append("--no-checkout")
            elif shallow:
                clone_args.extend(["--depth", "1"])
            clone_args.extend(["--", clone_source, str(tmp)])
            self._git(clone_args, timeout=self._settings.clone_timeout_seconds)

            if commit is not None:
                resolved = self._git(
                    ["rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"],
                    cwd=tmp,
                    error=f"Commit {commit} was not found in the repository.",
                ).strip()
                self._git(
                    [
                        "-c",
                        "core.symlinks=false",
                        "checkout",
                        "--quiet",
                        "--detach",
                        resolved,
                    ],
                    cwd=tmp,
                    timeout=self._settings.clone_timeout_seconds,
                )

            full_sha = self._git(["rev-parse", "HEAD"], cwd=tmp).strip()
            dest = self.checkouts_dir / f"{slug}__{full_sha}"
            if dest.exists():
                _force_rmtree(tmp)
            else:
                shutil.move(str(tmp), str(dest))
            return dest, full_sha
        except BaseException:
            _force_rmtree(tmp)
            raise

    def _git(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 60,
        error: str | None = None,
    ) -> str:
        """Run git without a shell and return stdout."""
        env = {
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ALLOW_PROTOCOL": "file:https",
        }
        token = self._settings.github_token
        if token:
            # Passed via environment (not argv) so it is not visible in process lists.
            basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
            env.update(
                GIT_CONFIG_COUNT="1",
                GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
                GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {basic}",
            )
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["git", *args],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RepositoryAccessError("The 'git' executable was not found.") from exc
        except subprocess.TimeoutExpired as exc:
            raise RepositoryAccessError(
                f"git {args[0] if args else ''} timed out after {timeout}s."
            ) from exc

        if proc.returncode != 0:
            detail = proc.stderr.strip().splitlines()
            tail = detail[-1] if detail else "unknown error"
            if token:
                tail = tail.replace(token, "***")
            logger.debug("git failed: %s", tail)
            raise RepositoryAccessError(error or f"git command failed: {tail}")
        return proc.stdout
