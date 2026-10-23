"""Precomputed embedding artifacts: format, loading, and validation.

Artifacts are produced once on a GPU by ``notebooks/kaggle_swerank_embeddings.ipynb``
from an export written by ``python -m evaluation.gpu_embeddings export``, then used
on CPU by :class:`app.retrieval.dense.DenseRetriever` (``embeddings_path``), which
only has to encode queries.

Directory layout (format ``repomind-embeddings/v1``)::

    manifest.json    model pins, text format, export hash, runtime info
    embeddings.npy   float32 [rows, dim], L2-normalized, row i <-> ids.jsonl row i
    ids.jsonl        {"row": i, "key": [file_path, qualified_name, content_hash],
                      "text_sha256": "..."}

The key is the same content-addressed key :class:`DenseRetriever` caches by, so a
function shared by many revisions is embedded once and matched in all of them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from app.retrieval.dense import DenseConfig

FORMAT_VERSION = "repomind-embeddings/v1"
EXPORT_FORMAT_VERSION = "repomind-embedding-export/v1"
MANIFEST_FILE = "manifest.json"
EMBEDDINGS_FILE = "embeddings.npy"
IDS_FILE = "ids.jsonl"
NORM_TOLERANCE = 0.02

Key = tuple[str, str, str]


class EmbeddingStoreError(ValueError):
    """Artifacts are missing, corrupt, misaligned, or incompatible with the config."""


def _numpy() -> Any:
    try:
        import numpy as np
    except (
        ImportError
    ) as exc:  # pragma: no cover - numpy ships with the retrieval extra
        raise EmbeddingStoreError(
            "numpy is required to load embedding artifacts"
        ) from exc
    return np


class EmbeddingStore:
    """Read-only, validated view of an embedding artifact directory."""

    def __init__(
        self, path: Path, manifest: dict[str, Any], matrix: np.ndarray, ids: list[dict]
    ) -> None:
        self.path = path
        self.manifest = manifest
        self.matrix = matrix
        self.ids = ids
        self._row: dict[Key, int] = {
            (r["key"][0], r["key"][1], r["key"][2]): r["row"] for r in ids
        }

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def get(self, key: Key) -> np.ndarray | None:
        row = self._row.get(key)
        return None if row is None else self.matrix[row]

    def keys(self) -> Iterator[Key]:
        return iter(self._row)

    @classmethod
    def load(cls, path: str | Path, *, mmap: bool = True) -> EmbeddingStore:
        """Load and fully validate an artifact directory."""
        np = _numpy()
        root = Path(path)
        for name in (MANIFEST_FILE, EMBEDDINGS_FILE, IDS_FILE):
            if not (root / name).is_file():
                raise EmbeddingStoreError(f"missing {name} in {root}")
        try:
            manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EmbeddingStoreError(
                f"{MANIFEST_FILE} is not valid JSON: {exc}"
            ) from exc
        if manifest.get("format_version") != FORMAT_VERSION:
            raise EmbeddingStoreError(
                f"unsupported format {manifest.get('format_version')!r}; "
                f"expected {FORMAT_VERSION!r}"
            )
        matrix = np.load(root / EMBEDDINGS_FILE, mmap_mode="r" if mmap else None)
        ids = []
        with (root / IDS_FILE).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    ids.append(json.loads(line))
        _validate(manifest, matrix, ids, np)
        return cls(root, manifest, matrix, ids)


def _validate(manifest: dict[str, Any], matrix: Any, ids: list[dict], np: Any) -> None:
    if matrix.ndim != 2:
        raise EmbeddingStoreError(f"embeddings must be 2-D, got shape {matrix.shape}")
    rows, dim = matrix.shape
    if str(matrix.dtype) != "float32":
        raise EmbeddingStoreError(
            f"embeddings dtype must be float32, got {matrix.dtype}"
        )
    if manifest.get("rows") != rows or manifest.get("dim") != dim:
        raise EmbeddingStoreError(
            f"manifest says {manifest.get('rows')}x{manifest.get('dim')}, "
            f"embeddings.npy is {rows}x{dim}"
        )
    if len(ids) != rows:
        raise EmbeddingStoreError(
            f"ids.jsonl has {len(ids)} rows, embeddings have {rows}"
        )
    seen: set[tuple[str, ...]] = set()
    for expected, record in enumerate(ids):
        if record.get("row") != expected:
            raise EmbeddingStoreError(
                f"ids.jsonl row {expected} is labeled {record.get('row')}"
            )
        key = record.get("key")
        if not (isinstance(key, list) and len(key) == 3):
            raise EmbeddingStoreError(f"ids.jsonl row {expected} has a malformed key")
        if tuple(key) in seen:
            raise EmbeddingStoreError(f"duplicate key at row {expected}: {key}")
        seen.add(tuple(key))
    if rows:
        if not bool(np.isfinite(matrix).all()):
            raise EmbeddingStoreError("embeddings contain NaN or infinite values")
        norms = np.linalg.norm(np.asarray(matrix, dtype=np.float64), axis=1)
        bad = int(np.sum(np.abs(norms - 1.0) > NORM_TOLERANCE))
        if bad:
            raise EmbeddingStoreError(f"{bad} embeddings are not L2-normalized")


def check_compatible(manifest: dict[str, Any], config: DenseConfig) -> None:
    """Raise if the artifacts were not produced with the model/text the config uses."""
    from app.retrieval.dense import UNIT_TEXT_VERSION

    model = manifest.get("model", {})
    export = manifest.get("export", {})
    expected = {
        "model.name": (model.get("name"), config.model_name),
        "model.revision": (model.get("revision"), config.revision),
        "model.code_revision": (model.get("code_revision"), config.code_revision),
        "model.weights_sha256": (model.get("weights_sha256"), config.weights_sha256),
        "model.query_prefix": (model.get("query_prefix"), config.query_prefix),
        "export.include_path": (export.get("include_path"), config.include_path),
        "export.unit_text_version": (
            export.get("unit_text_version"),
            UNIT_TEXT_VERSION,
        ),
    }
    problems = [
        f"{field}: artifacts={got!r} config={want!r}"
        for field, (got, want) in expected.items()
        if got != want
    ]
    if not manifest.get("normalized", False):
        problems.append("artifacts are not marked as L2-normalized")
    if problems:
        raise EmbeddingStoreError(
            "embedding artifacts do not match the dense config: " + "; ".join(problems)
        )
