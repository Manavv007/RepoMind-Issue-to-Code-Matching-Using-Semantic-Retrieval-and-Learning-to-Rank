# RepoMind

**An NLP-based framework for semantic code retrieval and bug localization.**

Given a natural-language bug report and a software repository, RepoMind ranks
the code files and functions most likely involved in the bug, explains the
evidence behind each candidate, and can optionally propose a patch that is
tested in isolation.

RepoMind is deliberately built as a **measurable information-retrieval
pipeline**, not "an LLM that reads code." Its core contribution is comparing
lexical retrieval (BM25) against dense, hybrid, reranked, and
dependency-aware methods on a labeled benchmark — and reporting which
components actually improve localization, for which issues, and at what cost.

See [`project.md`](project.md) for the full specification and
[`plan.md`](plan.md) for the phased build plan.

---

## Status

Early development. **Phase 0 (foundation)**, **Phase 1 (repository parsing &
indexing, Python only)**, **Phase 2 (evaluation harness & dataset)**, and
**Phase 3 (BM25 lexical baseline)** are complete. A repository at an exact commit
can be parsed with Tree-sitter into functions, methods, and classes and stored in
SQLite, and any retriever can be scored on a labeled SWE-bench Lite subset with
Recall/Hit@k, MRR, nDCG@10, file/function accuracy, latency, and bootstrap
confidence intervals. A code-aware BM25 retriever is the baseline to beat (held-out
function hit@5 = 0.53 vs 0.00 for random); query construction (Phase 4) and dense
retrieval (Phase 5) are next. See
[`plan.md`](plan.md) for what comes next and
[`docs/evaluation.md`](docs/evaluation.md) for the benchmark protocol and results.

## Requirements

- Python 3.11+ (developed on 3.13)

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install the base package with dev tooling
pip install -e ".[dev]"

# Copy and edit environment configuration
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Optional dependency groups are installed per phase, e.g.:

```bash
pip install -e ".[parsing]"     # Phase 1
pip install -e ".[retrieval]"   # Phase 3+
pip install -e ".[eval]"        # Phase 2
```

## Index a repository (Phase 1)

```bash
pip install -e ".[parsing]"

# Public GitHub repo (cloned into data/checkouts, never executed)
python -m app.indexing https://github.com/pallets/click

# Pin an exact commit (7-40 hex chars; branch/tag names are rejected)
python -m app.indexing https://github.com/pallets/click --commit 2247b35

# Local directory
python -m app.indexing path/to/repo
```

The command prints a JSON summary (resolved commit, file/unit counts, skipped
files, warnings) and writes the index to the SQLite database configured by
`REPOMIND_DATABASE_URL`. Errors are reported as `error [code]: message` with a
non-zero exit status. Notes:

- Only Python (`.py`) files are indexed. Dependency, build, VCS, and cache
  directories are skipped; symlinks are never followed.
- `--include-nested-functions` also emits functions defined inside functions.
- Class units contain the full class text; filter on `kind` to index only
  leaf functions/methods.
- Call references are *syntactic* (not resolved to definitions).

## Evaluate a retriever (Phase 2)

```bash
# Score a retriever on the held-out split
python -m evaluation.run_retrieval_eval \
    --dataset evaluation/datasets/swebench_lite_subset.jsonl \
    --retriever bm25 --split test --tag tuned

# Textbook BM25 for comparison, via a config file (or inline JSON)
python -m evaluation.run_retrieval_eval \
    --dataset evaluation/datasets/swebench_lite_subset.jsonl \
    --retriever bm25 --retriever-config configs/bm25_classic.json \
    --tag classic --split test

# Paired comparison of two runs (same dataset and split)
python -m evaluation.compare_runs --a <baseline>.json --b <candidate>.json

# Re-tune BM25 on the dev split only
python -m evaluation.tune_bm25 \
    --dataset evaluation/datasets/swebench_lite_subset.jsonl

# Rebuild the evaluation set from SWE-bench Lite (downloads rows, clones repos)
python -m evaluation.build_eval_set --repos psf/requests pallets/flask \
    --output evaluation/datasets/my_subset.jsonl
```

Reports (JSON + Markdown) are written to `evaluation/reports/`. Retrievers
receive only the issue text and the pre-fix code units; gold labels never reach
them. On Windows, set `REPOMIND_DATA_DIR` to a short path (e.g. `C:\rm`) if git
reports "Filename too long". See [`docs/evaluation.md`](docs/evaluation.md).

## Run the API (health check)

```bash
uvicorn app.main:app --reload
# then visit http://127.0.0.1:8000/health
```

## Development

```bash
pytest            # run tests
ruff check .      # lint
black .           # format
mypy              # type check
```

## Project layout

```text
app/                 Application packages (parsing, retrieval, graph, api, ...)
  config.py          Centralized settings (pydantic-settings)
  main.py            FastAPI entry point
  ingestion/         Repository loader (commit pinning) and file walker
  parsing/           Tree-sitter parser and symbol extractor
  indexing/          Indexing pipeline and CLI
  storage/           SQLite index store
  schemas/           CodeUnit / FileRecord / RepositoryRecord models
  retrieval/         Retriever interface, tokenizer, BM25, random baseline, registry
evaluation/          Eval schema, labeler, leakage controls, metrics, runner,
                     eval-set builder; datasets/ and reports/ are committed
tests/               Test suite
configs/             Run / preprocessing configuration files
docs/                Documentation
```

## Safety note

Repositories and issue text are treated as **untrusted input**. RepoMind does
not execute repository code during indexing, and any patch validation runs in
a disposable, resource-limited sandbox. See `project.md` §16.

## License

MIT
