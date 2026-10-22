"""Centralized configuration for RepoMind.

Settings are loaded from environment variables and an optional ``.env`` file
using ``pydantic-settings``. Keep secrets out of source control: real values
belong in ``.env`` (git-ignored) or the process environment, not here.

See ``.env.example`` for the documented set of variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = two levels up from this file (app/config.py -> app -> root).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings.

    Environment variables are prefixed with ``REPOMIND_`` (case-insensitive).
    For example, ``REPOMIND_LOG_LEVEL=DEBUG``.
    """

    model_config = SettingsConfigDict(
        env_prefix="REPOMIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General -----------------------------------------------------------
    app_name: str = "RepoMind"
    environment: str = Field(
        default="development",
        description="Deployment environment: development | test | production.",
    )
    log_level: str = Field(default="INFO", description="Python logging level.")

    # --- Storage / data paths ---------------------------------------------
    data_dir: Path = Field(
        default=PROJECT_ROOT / "data",
        description="Root directory for indexes, caches, and checkouts.",
    )
    database_url: str = Field(
        default="sqlite:///./data/repomind.db",
        description="SQLAlchemy-style database URL for metadata/analysis records.",
    )

    # --- Ingestion ---------------------------------------------------------
    github_token: str | None = Field(
        default=None,
        description="Optional GitHub token for higher rate limits / private repos.",
    )
    max_repo_size_mb: int = Field(
        default=500,
        description="Reject repositories larger than this (safety limit).",
    )
    max_file_size_kb: int = Field(
        default=1024,
        description="Skip individual files larger than this during indexing.",
    )
    max_files: int = Field(
        default=20000,
        description="Reject repositories with more candidate files than this.",
    )
    clone_timeout_seconds: int = Field(
        default=600, description="Timeout for each git clone/checkout operation."
    )
    max_index_seconds: int = Field(
        default=900, description="Abort an indexing run that exceeds this duration."
    )

    # --- Retrieval ---------------------------------------------------------
    default_top_k: int = Field(
        default=10, description="Default number of candidates returned."
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Default dense embedding model id.",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Default cross-encoder reranker model id.",
    )

    # --- Optional LLM provider --------------------------------------------
    llm_provider: str | None = Field(
        default=None,
        description="Optional LLM provider for diagnosis/patch (e.g. 'openai').",
    )
    llm_api_key: str | None = Field(
        default=None, description="API key for the configured LLM provider."
    )

    def ensure_data_dir(self) -> Path:
        """Create the data directory if it does not exist and return it."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Cached so the ``.env`` file and environment are read once per process.
    """
    return Settings()
