"""Indexing: orchestrates loading, parsing, and storing a repository index."""

from app.indexing.pipeline import IndexResult, index_repository, make_repo_id

__all__ = ["IndexResult", "index_repository", "make_repo_id"]
