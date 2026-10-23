"""Dense (embedding) retrieval and BM25 + dense hybrid fusion.

The embedding model sits behind a small :class:`Encoder` interface, so the
retrievers are testable without downloading a model and the backend can be swapped
by configuration. ``sentence-transformers`` (and ``numpy``) are imported lazily: if
they are missing or the model cannot be loaded, construction fails with
:class:`DenseUnavailableError` and BM25 keeps working.

Security note: ``trust_remote_code`` executes Python shipped with a model repository.
It is off by default; enable it only for models you trust, and pin ``revision``.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.config import PROJECT_ROOT
from app.retrieval.base import (
    RankedCandidate,
    RetrievalQuery,
    Retriever,
    make_candidate,
)
from app.retrieval.bm25 import BM25Config, BM25Retriever
from app.retrieval.embedding_store import (
    EmbeddingStore,
    EmbeddingStoreError,
    check_compatible,
)
from app.schemas.code_unit import CodeUnit


class DenseUnavailableError(RuntimeError):
    """The embedding backend cannot be used (missing package or model load failure)."""


class DenseConfig(BaseModel):
    """Embedding backend configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str | None = Field(default=None, description="Pin a model revision.")
    code_revision: str | None = Field(
        default=None,
        description="Pin the revision of remote modeling code hosted in another repo.",
    )
    weights_sha256: str | None = Field(
        default=None,
        description="Expected SHA-256 of model.safetensors; verified before loading.",
    )
    query_prefix: str = Field(
        default="", description="Instruction prefix some models require on queries."
    )
    max_seq_length: int = Field(default=256, ge=16, le=8192)
    batch_size: int = Field(default=32, ge=1)
    trust_remote_code: bool = False
    device: str = "cpu"
    include_path: bool = Field(
        default=True, description="Prepend the file path and symbol name to code text."
    )
    load_timeout_seconds: float = Field(
        default=180.0,
        gt=0,
        description="Raise DenseUnavailableError if loading takes longer.",
    )
    embeddings_path: str | None = Field(
        default=None,
        description="Precomputed artifacts directory (relative to the project root).",
    )
    on_missing: Literal["encode", "error"] = Field(
        default="encode",
        description="Units absent from the artifacts: encode them here, or fail.",
    )


class Encoder(Protocol):
    """Maps texts to L2-normalized vectors (one list of floats per text)."""

    def encode(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]: ...


class SentenceTransformerEncoder:
    """Encoder backed by ``sentence-transformers``.

    Import and model loading run in a daemon thread bounded by
    ``config.load_timeout_seconds``; on timeout or any error the constructor raises
    :class:`DenseUnavailableError` instead of hanging the caller. A timed-out load
    thread cannot be killed and is left running in the background until exit.
    """

    def __init__(self, config: DenseConfig) -> None:
        self.config = config
        outcome: dict[str, Any] = {}

        def load() -> None:
            try:
                from sentence_transformers import SentenceTransformer

                source, revision = resolve_model_source(config)
                pin: dict[str, Any] = {}
                if config.code_revision:
                    pin = {
                        "model_kwargs": {"code_revision": config.code_revision},
                        "config_kwargs": {"code_revision": config.code_revision},
                    }
                outcome["model"] = SentenceTransformer(
                    source,
                    revision=revision,
                    trust_remote_code=config.trust_remote_code,
                    device=config.device,
                    **pin,
                )
            except BaseException as exc:  # report every failure to the caller
                outcome["error"] = exc

        thread = threading.Thread(target=load, name="dense-model-load", daemon=True)
        thread.start()
        thread.join(config.load_timeout_seconds)
        if thread.is_alive():
            raise DenseUnavailableError(
                f"Loading embedding model '{config.model_name}' did not finish within "
                f"{config.load_timeout_seconds:g}s."
            )
        if "error" in outcome:
            exc = outcome["error"]
            raise DenseUnavailableError(
                f"Could not load embedding model '{config.model_name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._model = outcome["model"]
        self._model.max_seq_length = config.max_seq_length

    def encode(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        if not texts:
            return []
        prefix = self.config.query_prefix if is_query else ""
        vectors = self._model.encode(
            [prefix + t for t in texts],
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vectors]


#: Version of the text built by :func:`unit_text`; recorded in embedding artifacts
#: so vectors are never mixed with a different text format.
UNIT_TEXT_VERSION = "path-name-source/v1"


def unit_text(unit: CodeUnit, include_path: bool = True) -> str:
    """Text embedded for a code unit (format :data:`UNIT_TEXT_VERSION`)."""
    if include_path:
        return f"{unit.file_path} {unit.qualified_name}\n{unit.source}"
    return unit.source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_source(config: DenseConfig) -> tuple[str, str | None]:
    """Return ``(source, revision)`` to pass to ``SentenceTransformer``.

    Remote modeling code may download weights itself and ignore ``revision``
    (SweRankEmbed's does). With remote code and a pinned revision, the pinned
    snapshot is downloaded first and loaded from disk, and ``weights_sha256`` (if
    set) is checked against it, so the exact pinned weights are used.
    """
    if not (config.trust_remote_code and config.revision):
        if config.weights_sha256:
            raise DenseUnavailableError(
                "weights_sha256 requires trust_remote_code and a pinned revision"
            )
        return config.model_name, config.revision
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(config.model_name, revision=config.revision))
    if config.weights_sha256:
        weights = local / "model.safetensors"
        if not weights.is_file():
            raise DenseUnavailableError(f"{weights} not found in the pinned snapshot")
        actual = sha256_file(weights)
        if actual != config.weights_sha256:
            raise DenseUnavailableError(
                f"weights hash mismatch for {config.model_name}@{config.revision}: "
                f"expected {config.weights_sha256}, got {actual}"
            )
    return str(local), None


class MissingEmbeddingsError(EmbeddingStoreError):
    """Units in the pool have no precomputed embedding and ``on_missing='error'``."""


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _as_matrix(vectors: Sequence[Sequence[float]]) -> Any:
    """Stack vectors into a float32 matrix, or None without numpy (pure-Python path)."""
    if not vectors:
        return None
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy ships with the retrieval extra
        return None
    return np.asarray(vectors, dtype=np.float32)


class DenseRetriever(Retriever):
    """Cosine-similarity retrieval over unit embeddings.

    Embeddings are cached by ``(file path, qualified name, content hash)``, so
    revisions of the same repository only re-encode units that changed. With
    ``embeddings_path`` set, unit vectors come from precomputed GPU artifacts
    (validated against this config) and only queries are encoded locally.
    """

    name = "dense"

    def __init__(
        self,
        config: DenseConfig | None = None,
        *,
        encoder: Encoder | None = None,
        encoder_factory: Callable[[DenseConfig], Encoder] = SentenceTransformerEncoder,
    ) -> None:
        self.config = config or DenseConfig()
        self.store: EmbeddingStore | None = None
        if self.config.embeddings_path:
            path = Path(self.config.embeddings_path)
            self.store = EmbeddingStore.load(
                path if path.is_absolute() else PROJECT_ROOT / path
            )
            check_compatible(self.store.manifest, self.config)
        self._encoder = encoder if encoder is not None else encoder_factory(self.config)
        self._cache: dict[tuple[str, str, str], Sequence[float]] = {}
        self._units: list[CodeUnit] = []
        self._vectors: list[Sequence[float]] = []
        self._matrix: Any = None

    def describe(self) -> dict[str, Any]:
        return self.config.model_dump()

    @staticmethod
    def _key(unit: CodeUnit) -> tuple[str, str, str]:
        return (unit.file_path, unit.qualified_name, unit.content_hash)

    def _lookup(self, unit: CodeUnit) -> Sequence[float] | None:
        key = self._key(unit)
        vector = self._cache.get(key)
        if vector is None and self.store is not None:
            vector = self.store.get(key)
        return vector

    def prepare(self, units: Sequence[CodeUnit]) -> None:
        self._units = sorted(units, key=lambda u: u.unit_id)
        missing = [u for u in self._units if self._lookup(u) is None]
        if missing and self.config.on_missing == "error":
            raise MissingEmbeddingsError(
                f"{len(missing)} of {len(self._units)} units have no precomputed "
                f"embedding (e.g. {missing[0].unit_id}); re-export and re-embed, or "
                "set on_missing='encode'."
            )
        if missing:
            texts = [unit_text(u, self.config.include_path) for u in missing]
            vectors = self._encoder.encode(texts, is_query=False)
            for unit, vector in zip(missing, vectors, strict=True):
                self._cache[self._key(unit)] = vector
        self._vectors = [self._lookup(u) for u in self._units]  # type: ignore[misc]
        self._matrix = _as_matrix(self._vectors)

    def scores(self, query: RetrievalQuery) -> list[float]:
        (qvec,) = self._encoder.encode([query.text], is_query=True)
        if self._matrix is not None:
            import numpy as np  # present whenever _matrix was built

            return [float(s) for s in self._matrix @ np.asarray(qvec, dtype=np.float32)]
        return [_dot(qvec, v) for v in self._vectors]

    def retrieve(self, query: RetrievalQuery, k: int) -> list[RankedCandidate]:
        if not self._units:
            return []
        scored = self.scores(query)
        order = sorted(range(len(scored)), key=lambda i: (-scored[i], i))[:k]
        return [
            make_candidate(
                self._units[i],
                rank=rank,
                score=scored[i],
                method=self.name,
                signals={"dense": scored[i]},
            )
            for rank, i in enumerate(order, start=1)
        ]


class HybridConfig(BaseModel):
    """Reciprocal rank fusion of BM25 and dense rankings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rrf_k: int = Field(default=60, ge=1, description="RRF smoothing constant.")
    w_bm25: float = Field(default=1.0, ge=0)
    w_dense: float = Field(default=1.0, ge=0)
    depth: int = Field(
        default=200, ge=1, description="Candidates taken from each list."
    )


def rrf_fuse(
    rankings: Sequence[tuple[float, Sequence[str]]], rrf_k: int
) -> list[tuple[str, float]]:
    """Weighted RRF: ``score(d) = sum_i w_i / (rrf_k + rank_i(d))``; ties by id."""
    scores: dict[str, float] = {}
    for weight, ids in rankings:
        if weight == 0:
            continue
        for rank, unit_id in enumerate(ids, start=1):
            scores[unit_id] = scores.get(unit_id, 0.0) + weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever(Retriever):
    """BM25 + dense, combined with reciprocal rank fusion; keeps both raw scores."""

    name = "hybrid"

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        config: HybridConfig | None = None,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.config = config or HybridConfig()
        self._by_id: dict[str, CodeUnit] = {}

    def describe(self) -> dict[str, Any]:
        return {
            "fusion": self.config.model_dump(),
            "bm25": self.bm25.describe(),
            "dense": self.dense.describe(),
        }

    def prepare(self, units: Sequence[CodeUnit]) -> None:
        self.bm25.prepare(units)
        self.dense.prepare(units)
        self._by_id = {u.unit_id: u for u in units}

    def retrieve(self, query: RetrievalQuery, k: int) -> list[RankedCandidate]:
        depth = max(self.config.depth, k)
        lexical = self.bm25.retrieve(query, depth)
        semantic = self.dense.retrieve(query, depth)
        fused = rrf_fuse(
            [
                (self.config.w_bm25, [c.unit_id for c in lexical]),
                (self.config.w_dense, [c.unit_id for c in semantic]),
            ],
            self.config.rrf_k,
        )[:k]
        bm25_by_id = {c.unit_id: c for c in lexical}
        dense_by_id = {c.unit_id: c for c in semantic}
        results: list[RankedCandidate] = []
        for rank, (unit_id, score) in enumerate(fused, start=1):
            signals: dict[str, float] = {"rrf": score}
            if unit_id in bm25_by_id:
                signals["bm25"] = bm25_by_id[unit_id].score
                signals["bm25_rank"] = float(bm25_by_id[unit_id].rank)
            if unit_id in dense_by_id:
                signals["dense"] = dense_by_id[unit_id].score
                signals["dense_rank"] = float(dense_by_id[unit_id].rank)
            results.append(
                make_candidate(
                    self._by_id[unit_id],
                    rank=rank,
                    score=score,
                    method=self.name,
                    signals=signals,
                )
            )
        return results


def build_hybrid(
    params: dict[str, Any], *, encoder: Encoder | None = None
) -> HybridRetriever:
    """Build a hybrid from ``{"bm25": {...}, "dense": {...}, **fusion_params}``."""
    params = dict(params)
    bm25_params = params.pop("bm25", {})
    dense_params = params.pop("dense", {})
    return HybridRetriever(
        BM25Retriever(BM25Config(**bm25_params)),
        DenseRetriever(DenseConfig(**dense_params), encoder=encoder),
        HybridConfig(**params),
    )
