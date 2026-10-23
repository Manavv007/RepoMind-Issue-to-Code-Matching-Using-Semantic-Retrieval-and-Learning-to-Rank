"""Export functions for GPU embedding, and verify the artifacts that come back.

    python -m evaluation.gpu_embeddings export --dataset <eval.jsonl> --output <dir>
    python -m evaluation.gpu_embeddings verify --artifacts <dir> --export <dir> \\
        --dense-config configs/dense_swerank_small.json [--spot-check 4]

``export`` reuses the evaluation pipeline exactly: the same indexed pre-fix
revisions (``IndexedUnitsProvider``), the same label-independent candidate pool
(``candidate_units``), and the same text builder (``app.retrieval.dense.unit_text``).
Units are de-duplicated by ``(file_path, qualified_name, content_hash)``, the key
the dense retriever looks vectors up by. The GPU notebook therefore never parses
code; it only turns ``text`` into vectors, row by row.

``verify`` checks a downloaded artifact directory: format and normalization, row and
key alignment with the export, compatibility with the dense config, coverage of
every evaluation pool, and (optionally) agreement of a few rows with fresh CPU
embeddings from the pinned model.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, get_settings
from app.retrieval.dense import (
    UNIT_TEXT_VERSION,
    DenseConfig,
    DenseUnavailableError,
    SentenceTransformerEncoder,
    unit_text,
)
from app.retrieval.embedding_store import (
    EXPORT_FORMAT_VERSION,
    EmbeddingStore,
    EmbeddingStoreError,
    check_compatible,
)
from app.retrieval.registry import load_params
from app.schemas.code_unit import CodeUnit
from app.storage.index_store import IndexStore, sqlite_path_from_url
from evaluation.candidates import POOL_RULE
from evaluation.dataset import file_sha256, read_cases
from evaluation.run_retrieval_eval import IndexedUnitsProvider
from evaluation.schema import EvalCase

EXPORT_UNITS_FILE = "units.jsonl.gz"
EXPORT_MANIFEST_FILE = "export_manifest.json"
SPOT_CHECK_MIN_COSINE = 0.99

UnitsProvider = Callable[[EvalCase], list[CodeUnit]]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_sha256(records: Iterable[dict[str, Any]]) -> str:
    """Identity of an export: hash over (row, key, text_sha256) of every row.

    The notebook computes the same value; artifacts record it so they can only be
    matched with the export they were produced from.
    """
    digest = hashlib.sha256()
    for r in records:
        line = json.dumps(
            [r["row"], list(r["key"]), r["text_sha256"]], separators=(",", ":")
        )
        digest.update(line.encode("utf-8") + b"\n")
    return digest.hexdigest()


# ----------------------------------------------------------------------- export
def build_export(
    cases: Sequence[EvalCase],
    provider: UnitsProvider,
    *,
    include_path: bool,
    max_units: int | None = None,
) -> list[dict[str, Any]]:
    """Unique pool units of ``cases`` as rows ``{row, key, text, text_sha256}``."""
    unique: dict[tuple[str, str, str], CodeUnit] = {}
    for case in sorted(cases, key=lambda c: (c.repo, c.base_commit, c.case_id)):
        for unit in provider(case):
            unique.setdefault(
                (unit.file_path, unit.qualified_name, unit.content_hash), unit
            )
    keys = sorted(unique)
    if max_units is not None:
        keys = keys[:max_units]
    rows = []
    for i, key in enumerate(keys):
        text = unit_text(unique[key], include_path)
        rows.append(
            {"row": i, "key": list(key), "text": text, "text_sha256": text_sha256(text)}
        )
    return rows


def write_export(
    out_dir: Path, rows: Sequence[dict[str, Any]], manifest: dict[str, Any]
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / EXPORT_UNITS_FILE
    # mtime=0 keeps the compressed file byte-identical across re-exports.
    with gzip.GzipFile(path, "wb", mtime=0) as raw:
        for r in rows:
            raw.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
    manifest = {**manifest, "units_file_sha256": file_sha256(path)}
    (out_dir / EXPORT_MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def read_export(export_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(
        (export_dir / EXPORT_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    if manifest.get("format_version") != EXPORT_FORMAT_VERSION:
        raise EmbeddingStoreError(
            f"unsupported export format {manifest.get('format_version')!r}"
        )
    with gzip.open(export_dir / EXPORT_UNITS_FILE, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if content_sha256(rows) != manifest["content_sha256"]:
        raise EmbeddingStoreError(
            "export rows do not match export_manifest content_sha256"
        )
    return manifest, rows


def export_manifest(
    rows: Sequence[dict[str, Any]],
    *,
    dataset: Path,
    splits: Sequence[str],
    n_cases: int,
    config: DenseConfig,
) -> dict[str, Any]:
    lengths = [len(r["text"]) for r in rows]
    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {"path": dataset.name, "sha256": file_sha256(dataset)},
        "splits": list(splits),
        "n_cases": n_cases,
        "rows": len(rows),
        "content_sha256": content_sha256(rows),
        "candidate_pool": POOL_RULE,
        "include_path": config.include_path,
        "unit_text_version": UNIT_TEXT_VERSION,
        "text_chars": {
            "total": sum(lengths),
            "max": max(lengths, default=0),
            "mean": (sum(lengths) / len(lengths)) if lengths else 0,
        },
        # Pins the notebook must use; it refuses to run with different values.
        "model": {
            "name": config.model_name,
            "revision": config.revision,
            "code_revision": config.code_revision,
            "weights_sha256": config.weights_sha256,
            "query_prefix": config.query_prefix,
            "trust_remote_code": config.trust_remote_code,
        },
    }


# ----------------------------------------------------------------------- verify
def verify_alignment(
    store: EmbeddingStore, export: tuple[dict[str, Any], list[dict[str, Any]]]
) -> list[str]:
    """Problems with row/key/text alignment between artifacts and their export."""
    manifest, rows = export
    problems: list[str] = []
    art_export = store.manifest.get("export", {})
    if art_export.get("content_sha256") != manifest["content_sha256"]:
        problems.append(
            "artifacts were produced from a different export (content_sha256 differs)"
        )
    partial = bool(store.manifest.get("partial"))
    if not partial and len(store) != len(rows):
        problems.append(f"artifacts have {len(store)} rows, export has {len(rows)}")
    if len(store) > len(rows):
        problems.append(
            f"artifacts have more rows ({len(store)}) than the export ({len(rows)})"
        )
    for record, row in zip(store.ids, rows, strict=False):
        if (
            list(record["key"]) != list(row["key"])
            or record["text_sha256"] != row["text_sha256"]
        ):
            problems.append(
                f"row {row['row']} is misaligned (key or text hash differs)"
            )
            break
    return problems


def coverage(
    store: EmbeddingStore, cases: Sequence[EvalCase], provider: UnitsProvider
) -> dict[str, Any]:
    keys = set(store.keys())
    total = missing = 0
    cases_missing: list[str] = []
    for case in cases:
        pool = provider(case)
        absent = sum(
            1
            for u in pool
            if (u.file_path, u.qualified_name, u.content_hash) not in keys
        )
        total += len(pool)
        missing += absent
        if absent:
            cases_missing.append(case.case_id)
    return {
        "pool_units": total,
        "missing": missing,
        "cases_with_missing": cases_missing,
    }


def spot_check(
    store: EmbeddingStore,
    rows: Sequence[dict[str, Any]],
    config: DenseConfig,
    n: int,
    *,
    seed: int = 0,
    encoder_factory: Callable[[DenseConfig], Any] = SentenceTransformerEncoder,
) -> dict[str, Any]:
    """Re-embed ``n`` rows on CPU with the pinned model; compare cosine similarity."""
    import numpy as np

    picks = sorted(random.Random(seed).sample(range(len(store)), min(n, len(store))))
    encoder = encoder_factory(config.model_copy(update={"device": "cpu"}))
    fresh = encoder.encode([rows[i]["text"] for i in picks], is_query=False)
    cosines = [
        float(np.dot(store.matrix[i], np.asarray(v, dtype=np.float32)))
        for i, v in zip(picks, fresh, strict=True)
    ]
    return {"rows": picks, "cosine": cosines, "min_cosine": min(cosines, default=1.0)}


def verify(
    artifacts: Path,
    *,
    export_dir: Path | None = None,
    config: DenseConfig | None = None,
    cases: Sequence[EvalCase] = (),
    provider: UnitsProvider | None = None,
    spot: int = 0,
    encoder_factory: Callable[[DenseConfig], Any] = SentenceTransformerEncoder,
) -> dict[str, Any]:
    report: dict[str, Any] = {"artifacts": str(artifacts), "checks": {}, "problems": []}
    store = EmbeddingStore.load(artifacts)  # format, shapes, ids, normalization
    report["checks"]["format"] = "ok"
    report["rows"], report["dim"] = len(store), store.dim
    report["partial"] = bool(store.manifest.get("partial"))
    report["runtime"] = store.manifest.get("runtime", {})
    rows: list[dict[str, Any]] = []
    if export_dir is not None:
        export = read_export(export_dir)
        rows = export[1]
        problems = verify_alignment(store, export)
        report["checks"]["alignment"] = "ok" if not problems else "failed"
        report["problems"] += problems
    if config is not None:
        try:
            check_compatible(store.manifest, config)
            report["checks"]["config"] = "ok"
        except EmbeddingStoreError as exc:
            report["checks"]["config"] = "failed"
            report["problems"].append(str(exc))
    if cases and provider is not None:
        cov = coverage(store, cases, provider)
        report["coverage"] = cov
        report["checks"]["coverage"] = (
            "ok"
            if cov["missing"] == 0
            else ("partial" if report["partial"] else "failed")
        )
        if cov["missing"] and not report["partial"]:
            report["problems"].append(f"{cov['missing']} pool units have no embedding")
    if spot:
        if config is None or not rows:
            report["problems"].append("--spot-check needs --export and --dense-config")
        else:
            result = spot_check(
                store, rows, config, spot, encoder_factory=encoder_factory
            )
            report["spot_check"] = result
            ok = result["min_cosine"] >= SPOT_CHECK_MIN_COSINE
            report["checks"]["spot_check"] = "ok" if ok else "failed"
            if not ok:
                report["problems"].append(
                    f"CPU re-embedding disagrees: min cosine "
                    f"{result['min_cosine']:.4f} "
                    f"< {SPOT_CHECK_MIN_COSINE}"
                )
    report["ok"] = not report["problems"]
    return report


# -------------------------------------------------------------------------- CLI
def _dense_config(spec: str) -> DenseConfig:
    params = load_params(spec)
    return DenseConfig(**{k: v for k, v in params.items() if k != "embeddings_path"})


def _load_cases(
    dataset: Path, splits: Sequence[str], limit: int | None
) -> list[EvalCase]:
    cases = [c for c in read_cases(dataset) if c.suitable and c.split in set(splits)]
    cases.sort(key=lambda c: (c.repo, c.base_commit, c.case_id))
    return cases[:limit] if limit else cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.gpu_embeddings", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("export", help="write units for GPU embedding")
    ex.add_argument("--dataset", type=Path, required=True)
    ex.add_argument(
        "--splits", nargs="+", default=["dev", "test"], choices=["dev", "test"]
    )
    ex.add_argument(
        "--dense-config",
        default=str(PROJECT_ROOT / "configs" / "dense_swerank_small.json"),
    )
    ex.add_argument(
        "--output", type=Path, required=True, help="Export directory to create"
    )
    ex.add_argument(
        "--limit-cases", type=int, help="Only the first N cases (smoke export)"
    )
    ex.add_argument(
        "--limit-units", type=int, help="Only the first N units (smoke export)"
    )
    ex.add_argument("--db", type=Path)

    ve = sub.add_parser("verify", help="check downloaded embedding artifacts")
    ve.add_argument("--artifacts", type=Path, required=True)
    ve.add_argument(
        "--export", type=Path, help="Export directory the artifacts came from"
    )
    ve.add_argument(
        "--dense-config", help="Dense config (JSON or path) to check compatibility"
    )
    ve.add_argument(
        "--dataset", type=Path, help="Check every pool of these cases is covered"
    )
    ve.add_argument(
        "--splits", nargs="+", default=["dev", "test"], choices=["dev", "test"]
    )
    ve.add_argument(
        "--spot-check", type=int, default=0, help="Re-embed N rows on CPU and compare"
    )
    ve.add_argument("--db", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    store_db = IndexStore(args.db or sqlite_path_from_url(settings.database_url))
    provider = IndexedUnitsProvider(settings, store_db)
    try:
        if args.command == "export":
            config = _dense_config(args.dense_config)
            cases = _load_cases(args.dataset, args.splits, args.limit_cases)
            if not cases:
                print(
                    "error: no suitable cases in the requested splits", file=sys.stderr
                )
                return 2
            rows = build_export(
                cases,
                provider,
                include_path=config.include_path,
                max_units=args.limit_units,
            )
            manifest = export_manifest(
                rows,
                dataset=args.dataset,
                splits=args.splits,
                n_cases=len(cases),
                config=config,
            )
            path = write_export(args.output, rows, manifest)
            size_mb = path.stat().st_size / 1e6
            print(
                json.dumps(
                    {
                        "export": str(args.output),
                        "cases": len(cases),
                        "rows": len(rows),
                        "content_sha256": manifest["content_sha256"],
                        "units_file_mb": round(size_mb, 2),
                        "text_chars": manifest["text_chars"],
                    },
                    indent=2,
                )
            )
            return 0
        check_config = _dense_config(args.dense_config) if args.dense_config else None
        cases = _load_cases(args.dataset, args.splits, None) if args.dataset else []
        report = verify(
            args.artifacts,
            export_dir=args.export,
            config=check_config,
            cases=cases,
            provider=provider if cases else None,
            spot=args.spot_check,
        )
    except (EmbeddingStoreError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1 if args.command == "verify" else 2
    except DenseUnavailableError as exc:
        print(f"error [dense_unavailable]: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
