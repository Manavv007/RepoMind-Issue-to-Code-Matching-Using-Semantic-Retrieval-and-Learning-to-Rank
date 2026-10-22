# RepoMind — End-to-End Build Plan

A phased, verifiable plan for building RepoMind: an NLP-based framework for
semantic code retrieval and bug localization. Each phase has a clear scope,
concrete deliverables, and acceptance criteria that can be checked before
moving on.

> **Guiding principle from the spec:** research evaluation must begin early.
> Do **not** spend most of the project on the UI before proving the retrieval
> pipeline works. BM25 is the mandatory baseline; every neural method must
> prove improvement over it on the *same* evaluation cases.

---

## Phase Map (at a glance)

| Phase | Theme | Core output | Depends on |
|-------|-------|-------------|------------|
| 0 | Project foundation | Repo scaffold, env, config, CI | — |
| 1 | Repository parsing & indexing | Metadata-rich code index | 0 |
| 2 | Evaluation harness & dataset | Benchmark subset + metrics scripts | 0 |
| 3 | Lexical retrieval (BM25 baseline) | Reproducible baseline + metrics | 1, 2 |
| 4 | Issue understanding | Query builder, technical-token extraction | 1 |
| 5 | Dense & hybrid retrieval | Embeddings, FAISS, RRF fusion | 3, 4 |
| 6 | Reranking & error analysis | Cross-encoder rerank + failure report | 5 |
| 7 | Dependency-aware context | Code graph + bounded expansion ablation | 1, 6 |
| 8 | Ranking & evidence assembly | Transparent score fusion + evidence | 6, 7 |
| 9 | Diagnosis & patch proposal (optional) | Grounded LLM diagnosis + diff | 8 |
| 10 | Patch validation (optional) | Isolated sandbox test runner | 9 |
| 11 | API layer (FastAPI) | REST endpoints over the pipeline | 8 |
| 12 | Frontend (React + Vite) | Interactive analysis UI | 11 |
| 13 | Learning-to-rank (optional/research) | Learned ranker + ablation | 8 |
| 14 | Hardening, docs, demo | Security, docs, reproducible demo | all |

The **first milestone** = Phases 0–3: given a bug report, return a ranked list
of relevant functions with valid source locations, plus a repeatable
evaluation of whether those functions are actually relevant.

---

## Phase 0 — Project Foundation & Scaffolding

**Goal:** Establish a modular Python project so individual methods can be
compared, with reproducible environments and testing from day one.

**Tasks**
- Initialize the repository layout from the spec (§15), creating only what the
  early phases need (`app/parsing`, `app/retrieval`, `evaluation/`, `tests/`,
  `configs/`, `docs/`).
- Set up `pyproject.toml` with pinned dependencies and a dependency manager
  (uv or poetry or pip-tools).
- Create `.env.example`, a `configs/` directory for run/preprocessing configs,
  and a config loader.
- Add `pytest`, linting (ruff), formatting (black), and type checking (mypy)
  with a basic CI workflow.
- Write a `README.md` stub and `docs/` placeholder.

**Deliverable:** A clean, installable project skeleton with passing CI on an
empty test suite.

**Acceptance criteria**
- `pip install -e .` (or equivalent) succeeds in a fresh environment.
- `pytest` runs green (even if only a smoke test).
- Linters/formatters run clean.

---

## Phase 1 — Repository Parsing & Indexing

**Goal:** Turn a repository at an exact commit into a metadata-rich,
searchable code index. (Spec §6.1, §6.2; Roadmap Phase 1)

**Scope:** Python only, to start.

**Tasks**
- Repository connector: accept a public GitHub URL or local path; check out a
  specific revision; **record the exact commit hash**.
- Repository walker: ignore generated files, binaries, dependency folders,
  build output, and configured exclusions. Enforce size/time limits.
- Tree-sitter parser: parse source into functions, methods, and classes.
- Symbol extractor: store file path, symbol name, line range, imports, and any
  available call relationships per code unit.
- Persist index metadata (SQLite to start) keyed by repo + commit.
- **Do not execute repository code during indexing.**

**Deliverable:** A searchable, metadata-rich code index for a chosen Python
repository at a recorded commit.

**Acceptance criteria**
- Parser + indexing unit tests pass on sample files (incl. nested classes,
  decorators, async defs).
- Every indexed code unit has a valid file path and line range that maps back
  to real source lines.
- Re-indexing the same commit is deterministic.

---

## Phase 2 — Evaluation Harness & Dataset

**Goal:** Make measurement possible before building retrieval, so every method
is judged on the same cases. (Spec §10, §11; begin early.)

**Tasks**
- Choose and verify a benchmark: **BugsInPy** (manageable Python start) and/or
  a **SWE-bench** subset. Verify current format, license, and function-level
  ground-truth availability before committing.
- Build an initial evaluation set schema recording per issue: issue id/text,
  repo URL + exact commit, buggy/relevant file paths, relevant functions or
  line ranges, associated tests, suitability flag, and provenance.
- Support **multiple relevant code units** per issue.
- Implement `evaluation/metrics.py`: Recall@5, Recall@10, MRR, nDCG@10,
  file-level and function-level localization accuracy, latency capture.
- Implement `evaluation/run_retrieval_eval.py`: a runner that takes any
  retriever and produces a metrics report with explicit denominators/cutoffs.
- Implement **data-leakage controls**: repo/project-aware splits, dedup of
  near-identical issues/patches, index the **pre-fix** revision, exclude fix
  patches from retrieval input.

**Deliverable:** A reproducible benchmark subset + metrics scripts that can
score any retriever.

**Acceptance criteria**
- Running the eval runner on a trivial/random retriever produces sensible,
  reproducible metric numbers with stated cutoffs and denominators.
- Ground-truth labels validated against the recorded pre-fix commit.
- Leakage controls documented and enforced in the split.

---

## Phase 3 — Lexical Retrieval (BM25 Baseline)

**Goal:** Establish the mandatory non-neural baseline. (Spec §6.4, §7.2;
Roadmap Phase 2)

**Tasks**
- Convert each code unit into a searchable document (code text, comments,
  identifiers, file paths, symbol names).
- Build an inverted index and implement BM25 search.
- Return top-k functions for a given issue query (initially raw issue text).
- Wire BM25 into the Phase 2 eval runner.
- Record BM25's raw score and rank per candidate (needed later for fusion).

**Deliverable:** A reproducible BM25 baseline with initial Recall@k / MRR /
nDCG@10 numbers on the benchmark subset.

**Acceptance criteria**
- `run_retrieval_eval.py` produces a committed baseline report.
- Numbers are reproducible across runs (fixed seeds / deterministic index).
- **This completes the first milestone.**

---

## Phase 4 — Issue Understanding

**Goal:** Extract useful signals from noisy bug reports without destroying
technical tokens. (Spec §6.3, §7.1)

**Tasks**
- Normalize issue text **without** damaging identifiers, exception names,
  paths, version strings, or stack-trace punctuation.
- Separate narrative text, code blocks, logs, stack traces, and reproduction
  steps.
- Extract technical entities: error types, file paths, function/API names,
  stack-trace locations.
- Build one or more search queries from the issue; retain the original report.

**Deliverable:** A query-preparation module that converts raw issues into
retrieval queries plus extracted entities.

**Acceptance criteria**
- Unit tests confirm technical tokens are preserved through normalization.
- Feeding prepared queries into BM25 does **not** regress (ideally improves)
  the Phase 3 baseline on the eval set.

---

## Phase 5 — Dense & Hybrid Retrieval

**Goal:** Retrieve conceptually relevant code even with few shared words, then
fuse with lexical. (Spec §6.4, §7.3; Roadmap Phase 3)

**Tasks**
- Select a pretrained embedding model (general sentence-embedding and/or
  code-aware; decide experimentally).
- Generate and store code-unit embeddings; build a **FAISS** vector index.
- Embed issue queries and run similarity search (cosine or appropriate).
- Implement hybrid fusion (reciprocal rank fusion and/or tuned score combo),
  retaining each method's original score and rank.
- Evaluate lexical vs dense vs hybrid on the **same** cases; report
  Recall@k, MRR, nDCG@10, and latency.

**Deliverable:** A measured comparison of lexical, dense, and hybrid retrieval.

**Acceptance criteria**
- Embeddings cached/persisted and reproducible.
- Hybrid results retain per-method provenance.
- Comparison report committed with latency alongside quality.

---

## Phase 6 — Reranking & Error Analysis

**Goal:** Improve the order of a smaller candidate set with a supervised
relevance model, and understand where methods fail. (Spec §6.5, §7.4;
Roadmap Phase 4)

**Tasks**
- Add a cross-encoder (e.g., a code-aware model) to rerank the hybrid top-k.
- Keep the two-stage design: broad recall first, expensive scoring second.
- Create the clear experimental comparison: retrieval alone vs retrieval +
  reranking.
- Perform error analysis: inspect false positives/negatives; categorize issue
  types where each method succeeds or fails.

**Deliverable:** A measured reranking experiment + a failure-analysis report.

**Acceptance criteria**
- Reranking evaluated against the identical baseline and eval cases.
- Any improvement (or regression) is backed by measured numbers.
- Failure report documents issue classes and trade-offs.

---

## Phase 7 — Dependency-Aware Context (Ablation)

**Goal:** Test whether repository relationships improve ranking, with
controlled expansion. (Spec §6.6, §8; Roadmap Phase 5)

**Tasks**
- Extract imports and feasible call relationships (Tree-sitter + static
  analysis where possible).
- Build a simple code-unit graph (NetworkX to start).
- Expand candidates with **bounded** traversal (limit depth to control noise).
- Label relationships by provenance: **known vs inferred**.
- Compare ranking quality and latency **with vs without** graph context.

**Deliverable:** An ablation study of dependency-aware ranking.

**Acceptance criteria**
- Expansion depth is bounded and configurable.
- Graph context's effect on metrics is measured, not assumed.
- Inferred links are never presented as certain.

---

## Phase 8 — Ranking & Evidence Assembly

**Goal:** Produce the final, evidence-backed localization report via
transparent score fusion. (Spec §6.7)

**Tasks**
- Combine signals: BM25 rank/score, dense similarity, cross-encoder score,
  exact stack-trace/identifier matches, relationship signals, test/config
  evidence.
- Start with **transparent score fusion**; establish this baseline before any
  learned ranker.
- For every candidate, preserve file path, line range, symbol name, and all
  supporting signals.
- Assemble a localization report: ranked files/functions, source locations,
  per-candidate evidence, and explicit uncertainty.
- Report limitations when evidence is insufficient rather than inventing a
  diagnosis.

**Deliverable:** An end-to-end localization report object produced from an
issue + indexed repo, with full evidence provenance.

**Acceptance criteria**
- Every displayed candidate carries verifiable location + signal breakdown.
- Score fusion is explainable (no opaque magic number).
- Insufficient-evidence cases are reported, not fabricated.

---

## Phase 9 — Diagnosis & Patch Proposal (Optional)

**Goal:** Use an LLM to explain leading candidates and optionally propose a
minimal patch — strictly grounded. (Spec §6.8, §7.6; Roadmap Phase 6)

**Tasks**
- Select/configure an LLM (local or authorized hosted API).
- Build the grounded prompt: original issue, top candidates, verified
  surrounding code, selected dependencies, relevant tests/errors.
- Instruct the model to distinguish observations from hypotheses and to
  abstain when evidence is weak.
- Produce: diagnosis hypothesis, supporting evidence, alternatives/missing
  evidence, a proposed patch, suggested regression tests.
- **Validate all references** (file paths, line ranges) against the indexed
  commit; reject fabricated references.
- Scan/exclude likely secret files before sending context to any external
  provider.

**Deliverable:** An end-to-end grounded diagnosis prototype producing an
optional diff.

**Acceptance criteria**
- No fabricated file paths/line refs survive validation.
- Model output clearly separates hypothesis from fact and can abstain.
- Context sent to any hosted model is disclosed and secret-scanned.

---

## Phase 10 — Patch Validation (Optional)

**Goal:** Safely test a proposed patch in isolation. (Spec §6.9; Roadmap
Phase 7)

**Tasks**
- Create an isolated, disposable worktree/checkout at the recorded commit.
- Apply the proposed patch without touching the original repo.
- Run configured checks/tests with **strict CPU, memory, time, filesystem,
  and network limits**.
- Capture stdout, stderr, exit codes, and test summaries.
- Compare changed files against the proposed scope.
- Report which checks passed, failed, or could not run.

**Deliverable:** An isolated patch-validation workflow.

**Acceptance criteria**
- Validation runs in a sandbox with enforced resource limits.
- Original repository is never mutated.
- Output clearly distinguishes test success from verified bug resolution.

---

## Phase 11 — API Layer (FastAPI)

**Goal:** Expose the pipeline through a stable API. (Spec §6.10, §14;
Roadmap Phase 7)

**Tasks**
- Define Pydantic models with input limits, authorization, and error
  contracts.
- Implement endpoints:
  - `POST /repositories/index`
  - `POST /issues/analyze`
  - `GET /analyses/{analysis_id}`
  - `POST /patches/propose` (optional)
  - `POST /patches/validate` (optional)
- Persist analysis records (SQLite/PostgreSQL).
- Add clear status reporting and error handling for unsupported languages or
  inaccessible repositories.

**Deliverable:** A working REST API over the retrieval/localization pipeline.

**Acceptance criteria**
- Endpoints validate input and return structured, documented responses.
- Failure modes (bad repo, unsupported language, incomplete issue) return
  clear errors, not crashes.
- API-level integration tests pass.

---

## Phase 12 — Frontend (React + Vite)

**Goal:** A simple UI to submit issues and inspect evidence-backed results.
(Spec §6.10; Roadmap Phase 7)

**Tasks**
- Build a React + Vite app that lets users submit an issue + repo.
- Display ranked functions, open source context, and show ranking evidence.
- Allow comparing alternative candidates.
- If patching is enabled, show the diff and test results with provenance.
- Surface status, errors, and uncertainty clearly.

**Deliverable:** A demonstrable interactive analysis interface.

**Acceptance criteria**
- A user can run the full submit → ranked results → evidence flow.
- Source locations in the UI match the indexed commit.
- Clear handling of loading, error, and low-confidence states.

---

## Phase 13 — Learning-to-Rank (Optional / Research)

**Goal:** Train a ranking model only once enough labels and a reliable split
exist. (Spec §6.7, §7.5; Roadmap Phase F)

**Tasks**
- Assemble issue–function relevance labels + retrieval features.
- Apply negative sampling / hard negatives without leakage across splits.
- Train pointwise/pairwise/listwise ranker; choose a ranking loss.
- Evaluate against the transparent-fusion baseline on held-out projects.

**Deliverable:** A learned ranker with an ablation vs transparent fusion.

**Acceptance criteria**
- Training/eval splits are leakage-free and repo-aware.
- Learned ranker is compared fairly to the fusion baseline.
- Any gain is measured, with cost reported.

---

## Phase 14 — Hardening, Docs & Demo

**Goal:** Security, reproducibility, documentation, and a convincing demo.
(Spec §12, §16, §18)

**Tasks**
- Enforce security safeguards end to end: no code execution at indexing,
  sandboxed tests, resource limits, secret scanning, path-traversal
  prevention, payload/size limits, token hygiene.
- Write documentation: installation, configuration, evaluation procedure,
  limitations, and safe usage.
- Produce the analysis report: metrics, ablations, failure cases, trade-offs.
- Build a reproducible demo on a public repo + known issue at a recorded
  commit.

**Deliverable:** A demonstrable prototype with reproducible evaluation and
complete documentation.

**Acceptance criteria (maps to spec §12 success criteria)**
- Submit repo + report → ranked functions with verified paths/line ranges.
- Each result retains retrieval method + evidence.
- Results evaluable against labeled issues via repeatable scripts.
- BM25 vs dense compared fairly; hybrid + reranking evaluated vs baseline.
- Component contributions examined via ablations; tested on held-out repos.
- Failure cases and trade-offs documented; everything reproducible.

---

## Cross-Cutting Concerns (apply to every phase)

- **Measurability:** same eval cases, revision, ground truth, and metric
  definitions across experiments. Always report latency with quality.
- **Provenance:** retain each retrieval method's raw score/rank; never collapse
  signals prematurely.
- **Grounding:** validate all file references/line ranges against the exact
  indexed commit.
- **Security:** treat repos and issue text as untrusted input; never trust
  repository text as model instructions.
- **Reproducibility:** record preprocessing/indexing configs, seeds, and model
  versions.
- **Honesty:** a ranking score is a prioritization signal, not a probability;
  an LLM explanation is not proof; passing tests is evidence, not a guarantee.

---

## Suggested Timeline (≈6–8 weeks, adjustable)

| Week | Phases |
|------|--------|
| 1 | Phase 0 + Phase 1 (scope, parsing, indexing) |
| 2 | Phase 2 + Phase 3 (eval harness + BM25 baseline) → **first milestone** |
| 3 | Phase 4 + Phase 5 (issue understanding + dense/hybrid) |
| 4 | Phase 6 (reranking + error analysis) |
| 5 | Phase 7 (dependency-aware context ablation) |
| 6 | Phase 8 + Phase 9 (evidence assembly + optional diagnosis) |
| 7–8 | Phase 10–12 + Phase 14 (validation, API, UI, docs, demo); Phase 13 if labels allow |

> Order can be adjusted, but **evaluation must start early**. Do not build the
> UI before the retrieval pipeline is proven.
