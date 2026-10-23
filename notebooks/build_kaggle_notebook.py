"""Generate ``notebooks/kaggle_swerank_embeddings.ipynb`` (source of truth for its cells).

    python notebooks/build_kaggle_notebook.py          # (re)write the notebook
    python notebooks/build_kaggle_notebook.py --check  # fail if it is out of date

The notebook is self-contained: it needs only the export directory produced by
``python -m evaluation.gpu_embeddings export`` and the pinned packages below. The
``[engine]`` cell (batching, checkpointing, merging) imports no torch so the test
suite can execute it on CPU with a fake encoder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).with_name("kaggle_swerank_embeddings.ipynb")

INTRO = """\
# RepoMind: one-time GPU embedding with SweRankEmbed-Small

Embeds every function in a RepoMind export (`python -m evaluation.gpu_embeddings export`)
with the pinned **Salesforce/SweRankEmbed-Small** model and writes artifacts that RepoMind's
CPU retrieval pipeline loads directly (`DenseConfig.embeddings_path`).

**Run order**
1. Keep `SMOKE_ONLY = True` for the first run (embeds the first `SMOKE_ROWS` functions).
   Download that zip and verify it locally (`python -m evaluation.gpu_embeddings verify ...`).
2. Set `SMOKE_ONLY = False` and run all cells again for the full export.

Shards are checkpointed to `OUTPUT_DIR/shards`. Re-running skips finished shards; to resume
after a lost Kaggle session, attach the previous version's output as input data (see
`docs/gpu_embeddings.md`). The model runs custom code from its Hugging Face repo
(`trust_remote_code`), pinned to exact revisions and a weights hash below.
"""

CONFIG = """\
# [config] Defaults find the export on Kaggle/Colab automatically. RM_* environment
# variables exist only for running this notebook locally in tests; ignore them.
import glob
import json
import os
from pathlib import Path


def _env(name, default):
    return os.environ.get(name, default)


SMOKE_ONLY = _env("RM_SMOKE_ONLY", "1") == "1"  # True first; False for the full run
SMOKE_ROWS = int(_env("RM_SMOKE_ROWS", "256"))
MAX_SEQ_LENGTH = int(_env("RM_MAX_SEQ_LENGTH", "1024"))
BATCH_SIZE = int(_env("RM_BATCH_SIZE", "32"))  # halved automatically on CUDA OOM
SHARD_ROWS = int(_env("RM_SHARD_ROWS", "2048"))  # rows per checkpoint file
PRECISION = _env("RM_PRECISION", "fp16")  # "fp16" (GPU) or "fp32"
ALLOW_CPU = _env("RM_ALLOW_CPU", "0") == "1"  # refuse to run without a GPU unless set
SKIP_INSTALL = _env("RM_SKIP_INSTALL", "0") == "1"

# Exact pins (must match configs/dense_swerank_small.json in RepoMind).
MODEL_NAME = "Salesforce/SweRankEmbed-Small"
MODEL_REVISION = "745d2a06103a66d3cfa600aa52fc0d3523010daa"
CODE_REVISION = "92d97331f1f4b6a366c1f161354b9f3390cc219f"  # Snowflake/snowflake-arctic-embed-m-long
WEIGHTS_SHA256 = "e40687da7f148e684648691facbaa2222503829afb9d21b9b3913034bef14f41"
QUERY_PREFIX = "Represent this query for searching relevant code: "
PINNED_PACKAGES = [
    "sentence-transformers==3.3.1",
    "transformers==4.57.6",
    "huggingface_hub==0.36.2",
    "einops==0.8.0",
]
FORMAT_VERSION = "repomind-embeddings/v1"
EXPORT_FORMAT_VERSION = "repomind-embedding-export/v1"


def _find_export():
    if _env("RM_EXPORT_DIR", ""):
        return Path(os.environ["RM_EXPORT_DIR"])
    for pattern in ("/kaggle/input/**/export_manifest.json", "/content/**/export_manifest.json"):
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return Path(hits[0]).parent
    raise FileNotFoundError(
        "No RepoMind export found. Kaggle: 'Add Input' -> your uploaded export dataset. "
        "Colab: unzip the export under /content."
    )


EXPORT_DIR = _find_export()
_default_out = (
    "/kaggle/working/repomind_embeddings"
    if Path("/kaggle/working").exists()
    else "/content/repomind_embeddings"
)
OUTPUT_DIR = Path(_env("RM_OUTPUT_DIR", _default_out))
# Shards from a previous run attached as input (Kaggle) are imported automatically.
RESUME_SHARD_DIRS = [
    Path(p).parent
    for p in sorted(glob.glob("/kaggle/input/**/shards/progress.json", recursive=True))
]
print(f"export: {EXPORT_DIR}\\noutput: {OUTPUT_DIR}\\nresume from: {RESUME_SHARD_DIRS or 'none'}")
print(f"SMOKE_ONLY={SMOKE_ONLY} MAX_SEQ_LENGTH={MAX_SEQ_LENGTH} PRECISION={PRECISION}")
"""

INSTALL = """\
# [install] Pinned versions (torch is the platform's preinstalled CUDA build).
import subprocess
import sys

if not SKIP_INSTALL:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PINNED_PACKAGES])
print("packages ready")
"""

ENGINE = '''\
# [engine] Batching, checkpointing, resume, and merge. No torch here (unit-tested on CPU).
import gzip
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


def content_sha256(records):
    digest = hashlib.sha256()
    for r in records:
        line = json.dumps([r["row"], list(r["key"]), r["text_sha256"]], separators=(",", ":"))
        digest.update(line.encode("utf-8") + b"\\n")
    return digest.hexdigest()


def read_export(export_dir):
    export_dir = Path(export_dir)
    manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != EXPORT_FORMAT_VERSION:
        raise ValueError(f"unsupported export format {manifest.get('format_version')!r}")
    with gzip.open(export_dir / "units.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for i, r in enumerate(rows):
        if r["row"] != i:
            raise ValueError(f"export row {i} is labeled {r['row']}")
        if hashlib.sha256(r["text"].encode("utf-8")).hexdigest() != r["text_sha256"]:
            raise ValueError(f"export row {i}: text does not match text_sha256")
    if content_sha256(rows) != manifest["content_sha256"]:
        raise ValueError("export rows do not match content_sha256 (corrupt or edited export)")
    return manifest, rows


def plan_shards(n_rows, shard_rows):
    return [(s, min(s + shard_rows, n_rows)) for s in range(0, n_rows, shard_rows)]


def shard_path(shards_dir, start, end):
    return Path(shards_dir) / f"{start:08d}-{end:08d}.npy"


def prepare_shards_dir(shards_dir, settings, resume_dirs=()):
    """Create/validate the checkpoint dir; import matching shards from earlier runs.

    ``settings`` (export hash, model pins, precision, ...) must match exactly, so
    shards produced under different conditions are never mixed.
    """
    shards_dir = Path(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    progress = shards_dir / "progress.json"
    if progress.exists():
        existing = json.loads(progress.read_text(encoding="utf-8"))
        if existing != settings:
            raise ValueError(
                f"{shards_dir} holds shards made with different settings; use a new "
                f"OUTPUT_DIR or delete it.\\nexisting={existing}\\nnow={settings}"
            )
    else:
        progress.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    imported = 0
    for other in resume_dirs:
        other = Path(other)
        try:
            other_settings = json.loads((other / "progress.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if other_settings != settings or other.resolve() == shards_dir.resolve():
            continue
        for shard in sorted(other.glob("*-*.npy")):
            if shard.name.endswith(".partial.npy"):
                continue  # interrupted write; never import
            target = shards_dir / shard.name
            if not target.exists():
                shutil.copy2(shard, target)
                imported += 1
    return imported


def shard_is_done(path, n_rows, dim, np):
    if not Path(path).exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False  # truncated/corrupt file: recompute it
    return arr.shape[0] == n_rows and (dim is None or arr.shape[1] == dim)


def save_atomic(path, array, np):
    path = Path(path)
    tmp = path.with_name(path.stem + ".partial.npy")
    np.save(tmp, array)
    os.replace(tmp, path)  # a shard is either complete or absent


def is_oom(exc):
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def encode_with_backoff(encode_fn, texts, batch_size, on_oom=None, log=print):
    while True:
        try:
            return encode_fn(texts, batch_size), batch_size
        except Exception as exc:
            if not is_oom(exc) or batch_size <= 1:
                raise
            if on_oom is not None:
                on_oom()
            batch_size = max(1, batch_size // 2)
            log(f"  out of memory; retrying with batch_size={batch_size}")


def run_shards(rows, shards_dir, encode_fn, *, limit, shard_rows, batch_size, np,
               dim=None, on_oom=None, log=print):
    """Embed rows[:limit] shard by shard, skipping shards already on disk."""
    plan = plan_shards(limit, shard_rows)
    todo = [(s, e) for s, e in plan if not shard_is_done(shard_path(shards_dir, s, e), e - s, dim, np)]
    log(f"{len(plan)} shards planned, {len(plan) - len(todo)} already done, {len(todo)} to run")
    started, done_rows = time.time(), 0
    for i, (start, end) in enumerate(todo, start=1):
        texts = [r["text"] for r in rows[start:end]]
        vecs, batch_size = encode_with_backoff(encode_fn, texts, batch_size, on_oom, log)
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.shape[0] != end - start or (dim is not None and vecs.shape[1] != dim):
            raise ValueError(f"shard {start}-{end}: got shape {vecs.shape}")
        if not np.isfinite(vecs).all():
            raise ValueError(f"shard {start}-{end}: non-finite values")
        save_atomic(shard_path(shards_dir, start, end), vecs, np)
        done_rows += end - start
        rate = done_rows / max(time.time() - started, 1e-9)
        left = sum(e - s for s, e in todo[i:])
        log(f"  shard {i}/{len(todo)} rows {start}-{end}  {rate:.1f} rows/s  ETA {left / rate / 60:.1f} min")
    return {"shards": len(plan), "computed": len(todo), "rows": limit, "seconds": time.time() - started}


def merge_shards(rows, shards_dir, out_dir, *, limit, shard_rows, export_manifest, model_info,
                 runtime, np):
    """Concatenate shards for rows[:limit] and write manifest, embeddings, ids."""
    out_dir = Path(out_dir)
    parts = []
    for start, end in plan_shards(limit, shard_rows):
        path = shard_path(shards_dir, start, end)
        if not shard_is_done(path, end - start, None, np):
            raise FileNotFoundError(f"missing or incomplete shard {path.name}; re-run the embedding cell")
        parts.append(np.load(path))
    matrix = np.concatenate(parts).astype(np.float32) if parts else np.zeros((0, model_info["dim"]), np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    if matrix.shape != (limit, model_info["dim"]) or (limit and float(np.abs(norms - 1).max()) > 0.02):
        raise ValueError(f"merged matrix failed checks: shape {matrix.shape}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_atomic(out_dir / "embeddings.npy", matrix, np)
    with open(out_dir / "ids.jsonl", "w", encoding="utf-8", newline="\\n") as handle:
        for r in rows[:limit]:
            handle.write(json.dumps({"row": r["row"], "key": r["key"], "text_sha256": r["text_sha256"]}) + "\\n")
    manifest = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": limit,
        "dim": model_info["dim"],
        "dtype": "float32",
        "normalized": True,
        "partial": limit < len(rows),
        "model": model_info,
        "export": {
            "content_sha256": export_manifest["content_sha256"],
            "rows_total": len(rows),
            "include_path": export_manifest["include_path"],
            "unit_text_version": export_manifest["unit_text_version"],
            "dataset": export_manifest.get("dataset"),
            "splits": export_manifest.get("splits"),
        },
        "runtime": runtime,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


print("engine ready")
'''

ENVIRONMENT = """\
# [environment] Require a CUDA GPU (unless ALLOW_CPU) and record versions.
import sys

import numpy as np
import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
elif ALLOW_CPU:
    DEVICE, GPU_NAME = "cpu", None
    print("WARNING: no GPU; running on CPU (slow; intended only for local tests)")
else:
    raise SystemExit(
        "No CUDA GPU detected. Kaggle: Settings -> Accelerator -> 'GPU T4 x2' (or P100), "
        "then restart. Colab: Runtime -> Change runtime type -> T4 GPU."
    )
if DEVICE == "cpu" and PRECISION == "fp16":
    PRECISION = "fp32"  # fp16 is GPU-only
VERSIONS = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__}
for _pkg in ("sentence_transformers", "transformers", "huggingface_hub"):
    VERSIONS[_pkg] = __import__(_pkg).__version__
print(f"device={DEVICE} gpu={GPU_NAME} precision={PRECISION}")
print(VERSIONS)
"""

MODEL = """\
# [model] Load the export, check pins, download the pinned snapshot, verify weights.
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

EXPORT_MANIFEST, ROWS = read_export(EXPORT_DIR)
_expected = {"name": MODEL_NAME, "revision": MODEL_REVISION, "code_revision": CODE_REVISION,
             "weights_sha256": WEIGHTS_SHA256, "query_prefix": QUERY_PREFIX}
_mismatch = {k: (EXPORT_MANIFEST["model"].get(k), v) for k, v in _expected.items()
             if EXPORT_MANIFEST["model"].get(k) != v}
if _mismatch:
    raise ValueError(f"export was made for a different model config (export, notebook): {_mismatch}")
print(f"export rows: {len(ROWS)}  content_sha256: {EXPORT_MANIFEST['content_sha256'][:16]}")

# The model's remote code loads weights itself and ignores `revision`, so load the
# pinned snapshot from disk and check the weights hash.
_local = Path(snapshot_download(MODEL_NAME, revision=MODEL_REVISION))
_h = hashlib.sha256()
with open(_local / "model.safetensors", "rb") as _f:
    for _chunk in iter(lambda: _f.read(1 << 20), b""):
        _h.update(_chunk)
if _h.hexdigest() != WEIGHTS_SHA256:
    raise ValueError(f"weights hash mismatch: {_h.hexdigest()}")
model = SentenceTransformer(
    str(_local),
    device=DEVICE,
    trust_remote_code=True,
    model_kwargs={"code_revision": CODE_REVISION},
    config_kwargs={"code_revision": CODE_REVISION},
)
model.max_seq_length = MAX_SEQ_LENGTH
DIM = model.get_sentence_embedding_dimension()


def encode_fn(texts, batch_size):
    vecs = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    vecs = np.asarray(vecs, dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # renormalize in fp32


def free_cuda_memory():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


print(f"model loaded: dim={DIM} max_seq_length={model.max_seq_length}")
"""

SMOKE = """\
# [smoke test] Dimensions, normalization, batch-size stability, fp16 agreement, query path.
_sample = [r["text"] for r in ROWS[:16]]
_ref = encode_fn(_sample, 8)  # fp32 reference
assert _ref.shape == (len(_sample), DIM), _ref.shape
assert np.allclose(np.linalg.norm(_ref, axis=1), 1.0, atol=1e-3)
_again = encode_fn(_sample, 3)  # different batching/padding
SMOKE_RESULTS = {"rows": len(_sample), "dim": DIM,
                 "batch_stability_min_cos": float((_ref * _again).sum(1).min())}
assert SMOKE_RESULTS["batch_stability_min_cos"] > 0.999, SMOKE_RESULTS
if DEVICE == "cuda" and PRECISION == "fp16":
    model.half()
    _half = encode_fn(_sample, 8)
    SMOKE_RESULTS["fp16_vs_fp32_min_cos"] = float((_ref * _half).sum(1).min())
    assert SMOKE_RESULTS["fp16_vs_fp32_min_cos"] > 0.995, SMOKE_RESULTS
_q = model.encode([QUERY_PREFIX + "crash when parsing an empty config file"],
                  normalize_embeddings=True, convert_to_numpy=True)
assert _q.shape == (1, DIM)
print("SMOKE TEST PASSED", SMOKE_RESULTS)
"""

RUN = """\
# [embed] Checkpointed, resumable embedding of the export (or its first SMOKE_ROWS rows).
LIMIT = min(SMOKE_ROWS, len(ROWS)) if SMOKE_ONLY else len(ROWS)
SHARDS_DIR = OUTPUT_DIR / "shards"
SETTINGS = {"export_content_sha256": EXPORT_MANIFEST["content_sha256"], "model": _expected,
            "max_seq_length": MAX_SEQ_LENGTH, "precision": PRECISION, "shard_rows": SHARD_ROWS}
_imported = prepare_shards_dir(SHARDS_DIR, SETTINGS, RESUME_SHARD_DIRS)
print(f"imported {_imported} shards from previous runs")
_t0 = time.time()
RUN_STATS = run_shards(ROWS, SHARDS_DIR, encode_fn, limit=LIMIT, shard_rows=SHARD_ROWS,
                       batch_size=BATCH_SIZE, np=np, dim=DIM, on_oom=free_cuda_memory)
print("done:", RUN_STATS)
"""

FINALIZE = """\
# [finalize] Merge shards, write artifacts, self-check, and zip for download.
import zipfile

MODEL_INFO = {**_expected, "dim": DIM, "max_seq_length": MAX_SEQ_LENGTH}
RUNTIME = {"device": DEVICE, "gpu": GPU_NAME, "precision": PRECISION, "batch_size": BATCH_SIZE,
           "versions": VERSIONS, "smoke_test": SMOKE_RESULTS, "run": RUN_STATS}
FINAL = merge_shards(ROWS, SHARDS_DIR, OUTPUT_DIR, limit=LIMIT, shard_rows=SHARD_ROWS,
                     export_manifest=EXPORT_MANIFEST, model_info=MODEL_INFO, runtime=RUNTIME, np=np)
_emb = np.load(OUTPUT_DIR / "embeddings.npy", mmap_mode="r")
with open(OUTPUT_DIR / "ids.jsonl", encoding="utf-8") as _f:
    _ids = [json.loads(line) for line in _f]
assert _emb.shape == (LIMIT, DIM) and len(_ids) == LIMIT
assert all(i == r["row"] and r["key"] == ROWS[i]["key"] for i, r in enumerate(_ids))
_name = "swerank-small-v1-smoke" if FINAL["partial"] else "swerank-small-v1"  # matches configs/dense_swerank_small.json
ZIP_PATH = OUTPUT_DIR.parent / f"{_name}.zip"
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as _z:
    for _file in ("manifest.json", "embeddings.npy", "ids.jsonl"):
        _z.write(OUTPUT_DIR / _file, f"{_name}/{_file}")
print(f"ARTIFACTS READY rows={LIMIT}/{len(ROWS)} partial={FINAL['partial']} dim={DIM}")
print(f"download: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")
"""

OUTRO = """\
## Next steps

* **Kaggle:** open the *Output* panel (right sidebar, or the saved version's *Output* tab) and
  download `swerank-small-v1-smoke.zip` (smoke run) or `swerank-small-v1.zip` (full run).
* **Colab:** `from google.colab import files; files.download(str(ZIP_PATH))`.
* Locally: unzip into `data/embeddings/` and run
  `python -m evaluation.gpu_embeddings verify ...` (see `docs/gpu_embeddings.md`).
"""

CELLS = [
    ("markdown", INTRO),
    ("code", CONFIG),
    ("code", INSTALL),
    ("code", ENGINE),
    ("code", ENVIRONMENT),
    ("code", MODEL),
    ("code", SMOKE),
    ("code", RUN),
    ("code", FINALIZE),
    ("markdown", OUTRO),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        current = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
        if current != text:
            print(
                f"{NOTEBOOK.name} is out of date; run: python notebooks/build_kaggle_notebook.py"
            )
            return 1
        print(f"{NOTEBOOK.name} is up to date")
        return 0
    NOTEBOOK.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
