"""Registry of retriever factories, so the eval runner can select one by name.

Later phases register their retrievers here (``dense``, ``hybrid``, ...).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from app.retrieval.base import Retriever
from app.retrieval.bm25 import BM25Config, BM25Retriever
from app.retrieval.random_baseline import RandomRetriever

#: ``factory(seed, params) -> Retriever``; ``params`` are retriever-specific options.
RetrieverFactory = Callable[[int, Mapping[str, Any]], Retriever]


def _random(seed: int, params: Mapping[str, Any]) -> Retriever:
    if params:
        raise ValueError(f"'random' takes no parameters, got: {sorted(params)}")
    return RandomRetriever(seed=seed)


def _bm25(seed: int, params: Mapping[str, Any]) -> Retriever:
    return BM25Retriever(BM25Config(**params))  # rejects unknown keys


def _dense(seed: int, params: Mapping[str, Any]) -> Retriever:
    from app.retrieval.dense import DenseConfig, DenseRetriever

    # May raise DenseUnavailableError (model cannot be loaded); callers report it.
    return DenseRetriever(DenseConfig(**params))


def _hybrid(seed: int, params: Mapping[str, Any]) -> Retriever:
    from app.retrieval.dense import build_hybrid

    return build_hybrid(dict(params))


RETRIEVER_FACTORIES: dict[str, RetrieverFactory] = {
    "random": _random,
    "bm25": _bm25,
    "dense": _dense,
    "hybrid": _hybrid,
}


def available_retrievers() -> list[str]:
    return sorted(RETRIEVER_FACTORIES)


def create_retriever(
    name: str, *, seed: int = 0, params: Mapping[str, Any] | None = None
) -> Retriever:
    """Instantiate a registered retriever.

    Raises:
        ValueError: if ``name`` is not registered or ``params`` are invalid.
    """
    try:
        factory = RETRIEVER_FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown retriever '{name}'. "
            f"Available: {', '.join(available_retrievers())}."
        ) from None
    try:
        return factory(seed, dict(params or {}))
    except ValueError:
        raise
    except RuntimeError:
        raise  # e.g. DenseUnavailableError: a backend problem, not bad parameters
    except Exception as exc:
        raise ValueError(f"Invalid parameters for '{name}': {exc}") from exc


def load_params(spec: str | None) -> dict[str, Any]:
    """Parse ``--retriever-config``: an inline JSON object or a path to a JSON file."""
    if not spec:
        return {}
    text = spec.strip()
    if not text.startswith("{"):
        path = Path(text)
        if not path.is_file():
            raise ValueError(f"Retriever config file not found: {spec}")
        text = path.read_text(encoding="utf-8")
    try:
        params = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Retriever config is not valid JSON: {exc}") from exc
    if not isinstance(params, dict):
        raise ValueError("Retriever config must be a JSON object.")
    return params
