"""RepoMind FastAPI application entry point.

This is a minimal scaffold for Phase 0. Routers for repositories, issues, and
patches are added in later phases (see ``plan.md``). For now it exposes a
health check so the service and its wiring can be smoke-tested.
"""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.config import get_settings


def create_app() -> FastAPI:
    """Application factory.

    Using a factory keeps construction testable and lets later phases attach
    routers, middleware, and lifespan handlers in one place.
    """
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "An NLP-based framework for semantic code retrieval and "
            "bug localization."
        ),
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        """Liveness probe used by smoke tests and deployment checks."""
        return {
            "status": "ok",
            "app": settings.app_name,
            "version": __version__,
            "environment": settings.environment,
        }

    return app


app = create_app()
