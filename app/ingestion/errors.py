"""Errors raised by repository ingestion and indexing.

Each error carries a stable machine-readable ``code`` so callers (CLI now, API
later) can report unsupported languages or inaccessible repositories clearly
instead of crashing.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for ingestion/indexing failures."""

    code: str = "ingestion_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message


class InvalidSourceError(IngestionError):
    """The repository source (URL/path/commit) is malformed or disallowed."""

    code = "invalid_source"


class RepositoryAccessError(IngestionError):
    """The repository could not be cloned, found, or checked out."""

    code = "repository_inaccessible"


class UnsupportedLanguageError(IngestionError):
    """A requested language is not supported."""

    code = "unsupported_language"


class RepositoryTooLargeError(IngestionError):
    """The repository exceeds configured size or file-count limits."""

    code = "repository_too_large"


class NoSupportedFilesError(IngestionError):
    """No indexable source files were found."""

    code = "no_supported_files"


class IndexingTimeoutError(IngestionError):
    """Indexing exceeded the configured time limit."""

    code = "indexing_timeout"
