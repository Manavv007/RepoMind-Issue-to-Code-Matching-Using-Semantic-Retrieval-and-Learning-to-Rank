"""SWE-bench Lite loader.

Rows come from the public Hugging Face datasets-server (no credentials needed) or
from a local JSONL cache. Only the fields RepoMind needs are kept.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DATASET_ID = "princeton-nlp/SWE-bench_Lite"
DATASET_LICENSE_NOTE = (
    "SWE-bench code is MIT-licensed; the Hugging Face dataset card for "
    f"{DATASET_ID} declares no license. Issue texts come from public GitHub issues "
    "of the listed repositories and stay subject to those projects' terms. Used "
    "here for non-commercial research evaluation."
)
_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_INFO_URL = f"https://huggingface.co/api/datasets/{DATASET_ID}"
_ALLOWED_HOSTS = frozenset({"datasets-server.huggingface.co", "huggingface.co"})
_PAGE_SIZE = 100  # datasets-server maximum
_USER_AGENT = "repomind-eval-builder/0.1 (research)"


@dataclass(frozen=True)
class SweInstance:
    """The subset of a SWE-bench row used to build an evaluation case."""

    instance_id: str
    repo: str
    base_commit: str
    patch: str
    problem_statement: str
    fail_to_pass: tuple[str, ...]
    created_at: str
    version: str
    hf_split: str


def parse_row(row: dict[str, Any], hf_split: str) -> SweInstance:
    """Convert a raw dataset row; tolerates FAIL_TO_PASS as JSON string or list."""
    f2p = row.get("FAIL_TO_PASS", [])
    if isinstance(f2p, str):
        try:
            f2p = json.loads(f2p)
        except json.JSONDecodeError:
            f2p = [f2p]
    return SweInstance(
        instance_id=str(row["instance_id"]),
        repo=str(row["repo"]),
        base_commit=str(row["base_commit"]),
        patch=str(row["patch"]),
        problem_statement=str(row["problem_statement"]),
        fail_to_pass=tuple(str(t) for t in f2p),
        created_at=str(row.get("created_at", "")),
        version=str(row.get("version", "")),
        hf_split=hf_split,
    )


def _get_json(url: str, *, timeout: int = 60, retries: int = 3) -> Any:
    host = urllib.parse.urlparse(url).hostname
    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"Refusing to fetch from unexpected host: {host}")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                request, timeout=timeout
            ) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def fetch_dataset_revision() -> str | None:
    """Pinned Hugging Face dataset revision (git sha), if available."""
    try:
        info = _get_json(_INFO_URL)
    except RuntimeError:
        return None
    sha = info.get("sha")
    return str(sha) if sha else None


def fetch_instances(splits: Iterable[str] = ("test", "dev")) -> list[SweInstance]:
    """Download all rows of the requested splits."""
    instances: list[SweInstance] = []
    for split in splits:
        offset = 0
        while True:
            query = urllib.parse.urlencode(
                {
                    "dataset": DATASET_ID,
                    "config": "default",
                    "split": split,
                    "offset": offset,
                    "length": _PAGE_SIZE,
                }
            )
            page = _get_json(f"{_ROWS_URL}?{query}")
            rows = page.get("rows", [])
            instances.extend(parse_row(r["row"], split) for r in rows)
            offset += len(rows)
            if not rows or offset >= int(page.get("num_rows_total", 0)):
                break
    return sorted(instances, key=lambda i: i.instance_id)


def save_raw(path: Path, instances: Iterable[SweInstance]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for inst in instances:
            handle.write(json.dumps(asdict(inst), sort_keys=True) + "\n")


def load_raw(path: Path) -> list[SweInstance]:
    instances: list[SweInstance] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data = json.loads(line)
                data["fail_to_pass"] = tuple(data["fail_to_pass"])
                instances.append(SweInstance(**data))
    return sorted(instances, key=lambda i: i.instance_id)
