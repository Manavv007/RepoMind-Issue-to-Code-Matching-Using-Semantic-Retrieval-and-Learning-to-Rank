"""Phase 0 smoke tests.

These verify the scaffold is wired correctly: the package imports, settings
load, and the FastAPI app serves its health check. They intentionally avoid
heavy optional dependencies (parsing/retrieval) so the base install stays fast.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import __version__
from app.config import Settings, get_settings
from app.main import create_app


def test_version_is_defined() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_settings_load_with_defaults() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.app_name == "RepoMind"
    assert settings.default_top_k >= 1


def test_settings_env_override(monkeypatch) -> None:
    # Environment variables use the REPOMIND_ prefix.
    monkeypatch.setenv("REPOMIND_APP_NAME", "RepoMindTest")
    monkeypatch.setenv("REPOMIND_DEFAULT_TOP_K", "25")
    # Bypass the lru_cache by constructing a fresh Settings instance.
    settings = Settings()
    assert settings.app_name == "RepoMindTest"
    assert settings.default_top_k == 25


def test_health_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
