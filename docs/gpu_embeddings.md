# One-time GPU embedding with Kaggle

Embeds every candidate function of the evaluation set with the pinned
**SweRankEmbed-Small** model on a free Kaggle GPU. RepoMind then loads the vectors
on CPU and only encodes queries locally. This laptop encodes about 1-2 functions
per second on CPU; the full subset export has 16,350 functions.

**Status:** everything except the GPU run is verified locally. Exporting works,
all notebook cells run on CPU with the real pinned model (16 functions), and the
resulting artifacts pass `verify`, including a CPU re-embedding check (min cosine
0.9999999). The notebook has **not yet been run on a Kaggle GPU**; fp16 behaviour
and GPU speed are unverified until you run the smoke step below.

## What runs where

| Step | Where | Command / file |
|---|---|---|
| 1. Export functions (text + IDs) | local | `python -m evaluation.gpu_embeddings export ...` |
| 2. Embed (smoke, then full) | Kaggle GPU | `notebooks/kaggle_swerank_embeddings.ipynb` |
| 3. Verify artifacts | local | `python -m evaluation.gpu_embeddings verify ...` |
| 4. Use them | local | `--retriever dense --retriever-config configs/dense_swerank_small.json` |

The export reuses RepoMind's indexed pre-fix revisions, its candidate-pool rule, and
`app.retrieval.dense.unit_text`, so the notebook never parses code. Functions are
de-duplicated by `(file_path, qualified_name, content_hash)`, the same key the dense
retriever looks vectors up by.

## Pins

| | Value |
|---|---|
| Model | `Salesforce/SweRankEmbed-Small` @ `745d2a06103a66d3cfa600aa52fc0d3523010daa` |
| Remote code | `Snowflake/snowflake-arctic-embed-m-long` @ `92d97331f1f4b6a366c1f161354b9f3390cc219f` |
| Weights | `model.safetensors` SHA-256 `e40687da...bef14f41` (checked before loading) |
| Packages | `sentence-transformers==3.3.1`, `transformers==4.57.6`, `huggingface_hub==0.36.2`, `einops==0.8.0`; torch is the platform's preinstalled CUDA build |
| Text | `path-name-source/v1`: `"<file_path> <qualified_name>\n<source>"`, truncated to 1,024 tokens |
| Query prefix | `Represent this query for searching relevant code: ` |

The model's remote code loads weights itself and ignores `revision`. The notebook,
and RepoMind's CPU loader, therefore download the pinned snapshot, verify the
weights hash, and load from disk. `trust_remote_code` executes Python from the two
repositories above at the pinned revisions.

All pins live in `configs/dense_swerank_small.json`. The notebook repeats them and
refuses to run if the export was made for different values; a test checks the two
stay identical.

## 1. Export (local, about 10 seconds)

```powershell
python -m evaluation.gpu_embeddings export `
    --dataset evaluation/datasets/swebench_lite_subset.jsonl `
    --output data/gpu_export/swebench_lite_subset
```

This writes `units.jsonl.gz` (about 3 MB) and `export_manifest.json`. Verified
output: 41 cases, 16,350 functions, content hash `1f52036c...`. Re-running produces
byte-identical files.

## 2. Embed on Kaggle

One-time setup:

1. Sign in at kaggle.com. **Phone-verify the account** (Settings → Phone
   verification); Kaggle only allows GPU and internet access after this. The
   notebook needs internet to install packages and download the model.
2. **Upload the export as a dataset:** *Datasets → + New Dataset*. Drag in both
   files from `data/gpu_export/swebench_lite_subset/`, name it e.g.
   `repomind-export`, keep it **Private**, click *Create*.
3. **Import the notebook:** *Code → + New Notebook*, then *File → Import Notebook*
   and upload `notebooks/kaggle_swerank_embeddings.ipynb`.
4. In the notebook's right sidebar (*Session options*):
   * **Accelerator:** *GPU T4 x2* (or *GPU P100*). The notebook uses one GPU.
   * **Internet:** *On*.
   * **+ Add Input → Datasets → Your Datasets →** `repomind-export`.

**Smoke run (first):** leave `SMOKE_ONLY = True` in the config cell and click
*Run All*. It embeds the first 256 functions. Check that the output shows:

* `device=cuda gpu=Tesla T4` (or P100). Without a GPU it stops with instructions.
* `SMOKE TEST PASSED`, including `fp16_vs_fp32_min_cos` above 0.995. If the fp16
  check fails, set `PRECISION = "fp32"` and run again (slower but safe).
* `ARTIFACTS READY rows=256/16350 partial=True dim=768`, plus a `rows/s` figure.
  Use it to estimate the full run: 16,350 / rate.

Download `/kaggle/working/swerank-small-v1-smoke.zip` from the *Output* panel and
verify it locally (step 3) **before** the full run.

**Full run:** set `SMOKE_ONLY = False`, then *Save Version → Save & Run All
(Commit)*. The commit runs in the background, and its `/kaggle/working` files appear
in the version's *Output* tab when it finishes. Kaggle limits session length and
weekly GPU hours; check your remaining quota on the Kaggle settings page.

**Recovery after an interrupted run:**

* *Interactive session still open:* run the cells again. Finished shards in
  `/kaggle/working/repomind_embeddings/shards` are skipped. Each shard is written
  atomically, so a crash never leaves a half-written shard.
* *Session lost:* if the stopped version's *Output* tab still lists the `shards/`
  folder, attach it with *+ Add Input → Notebook Output* and run again. The config
  cell finds `shards/progress.json` under `/kaggle/input` and imports matching shards.
  Shards are imported only if the export hash, model pins, sequence length,
  precision, and shard size are identical. (Whether Kaggle keeps output from a
  failed version has not been verified here.)
* *CUDA out of memory:* the batch size halves automatically and the shard is retried.

**Colab instead:** *File → Upload notebook*, *Runtime → Change runtime type → T4
GPU*, and upload or unzip the export anywhere under `/content` (the config cell
finds it). Download with `from google.colab import files; files.download(str(ZIP_PATH))`.
Colab's `/content` is wiped when the runtime resets, so point `OUTPUT_DIR` at a
mounted Google Drive folder if you need recovery.

## 3. Verify locally

```powershell
Expand-Archive swerank-small-v1.zip -DestinationPath data/embeddings
python -m evaluation.gpu_embeddings verify `
    --artifacts data/embeddings/swerank-small-v1 `
    --export data/gpu_export/swebench_lite_subset `
    --dense-config configs/dense_swerank_small.json `
    --dataset evaluation/datasets/swebench_lite_subset.jsonl `
    --spot-check 8
```

(For the smoke zip, use `data/embeddings/swerank-small-v1-smoke` and leave out
`--dataset`. Coverage is expected to be partial.)

Checks, each listed in the JSON report: file format, shapes, finite and L2-normalized
vectors; row-by-row key and text-hash alignment with the export; model, text and
prefix compatibility with the config; every evaluation pool function covered; and
8 random rows re-embedded on CPU agreeing with the GPU vectors (cosine ≥ 0.99).
Exit code 0 means all passed, 1 means a check failed, 3 means the model can't be
loaded locally.

## 4. Use the embeddings

```powershell
python -m evaluation.run_retrieval_eval --dataset evaluation/datasets/swebench_lite_subset.jsonl `
    --retriever dense --retriever-config configs/dense_swerank_small.json --tag swerank --split dev
python -m evaluation.run_retrieval_eval --dataset evaluation/datasets/swebench_lite_subset.jsonl `
    --retriever hybrid --retriever-config configs/hybrid_swerank_small.json --tag swerank --split dev
```

With `on_missing: "error"` (the configs' setting), a function missing from the
artifacts fails that case loudly instead of being encoded slowly on CPU. Set
`"on_missing": "encode"` to fill gaps locally instead.

Artifact format (`repomind-embeddings/v1`):

* `embeddings.npy`: float32, shape [rows, 768], L2-normalized.
* `ids.jsonl`: `row`, `key`, and `text_sha256` for each row.
* `manifest.json`: pins, export hash, `partial` flag, and runtime (GPU,
  precision, package versions, smoke-test results, timings).

## Regenerating the notebook

Cell sources live in `notebooks/build_kaggle_notebook.py`. After editing it, run
`python notebooks/build_kaggle_notebook.py`. A test fails if the committed `.ipynb`
is out of date.
