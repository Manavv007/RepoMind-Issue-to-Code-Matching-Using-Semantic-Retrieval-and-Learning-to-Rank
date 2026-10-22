"""Command-line interface: ``python -m app.indexing <source> [options]``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from app.config import get_settings
from app.indexing.pipeline import index_repository
from app.ingestion.errors import IngestionError
from app.storage.index_store import IndexStore, sqlite_path_from_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.indexing",
        description="Index a Python repository (GitHub URL or local path) into SQLite.",
    )
    parser.add_argument(
        "source", help="https://github.com/<owner>/<repo> or a local path"
    )
    parser.add_argument("--commit", help="Pin an exact commit hash (7-40 hex chars)")
    parser.add_argument("--db", type=Path, help="SQLite path (default: from settings)")
    parser.add_argument(
        "--include-nested-functions",
        action="store_true",
        help="Also emit functions/classes defined inside functions",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    db_path = args.db or sqlite_path_from_url(settings.database_url)
    try:
        result = index_repository(
            args.source,
            commit=args.commit,
            settings=settings,
            store=IndexStore(db_path),
            include_nested_functions=args.include_nested_functions,
        )
    except IngestionError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 2

    repo = result.repository
    summary = {
        "repo_id": repo.repo_id,
        "source": repo.source,
        "commit": repo.commit,
        "dirty": repo.dirty,
        "files_indexed": repo.num_files,
        "units": repo.num_units,
        "units_by_kind": result.units_by_kind,
        "skipped": result.skipped,
        "files_with_syntax_errors": result.files_with_syntax_errors,
        "elapsed_seconds": result.elapsed_seconds,
        "database": str(db_path),
        "warnings": result.warnings,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
