# Evaluation

How RepoMind measures bug localization: the benchmark, how labels are derived,
how leakage is controlled, the metrics, and how to reproduce every number.

All results are computed by `python -m evaluation.run_retrieval_eval` on the same
cases, pre-fix revisions, candidate pool, and metric definitions for every
retriever, so methods are directly comparable.

---

## 1. Benchmark decision

Checked on 2026-10-22 (formats, licenses, and labels were inspected directly, not
assumed).

| | BugsInPy | SWE-bench Lite (chosen) |
|---|---|---|
| Bug-report text | **None.** Per bug: `buggy_commit_id`, `fixed_commit_id`, `bug_patch.txt`, `test_file`. | `problem_statement` (the GitHub issue text) |
| Pre-fix revision | `buggy_commit_id` | `base_commit` |
| Location labels | None; derivable from the patch | None; derivable from the gold `patch` |
| Size | 17 projects | 300 `test` + 23 `dev` instances, 18 repositories |
| License | No license file detected by the GitHub API | Code: MIT. The Hugging Face dataset card declares no license |

**Decision: SWE-bench Lite.** RepoMind maps a *natural-language bug report* to code;
BugsInPy contains no reports, so it cannot evaluate that task without inventing
synthetic ones. Neither benchmark ships function-level labels, so RepoMind derives
them from the gold patch (section 3). BugsInPy remains a candidate for a later
patch-validation study (it provides runnable test scripts).

Source: Hugging Face `princeton-nlp/SWE-bench_Lite`, revision
`6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`. Issue text comes from public GitHub
issues and remains subject to those projects' terms; it is used here for
non-commercial research evaluation.

## 2. The evaluation set

`evaluation/datasets/swebench_lite_subset.jsonl` (+ `.manifest.json`): 41 cases
from six repositories, small enough to index quickly, varied in size.

| Split | Repositories | Cases |
|---|---|---|
| `dev` (tuning / training) | psf/requests (6), pylint-dev/pylint (6), pylint-dev/astroid (5), mwaskom/seaborn (4) | 21 |
| `test` (held out, final reporting) | pytest-dev/pytest (17), pallets/flask (3) | 20 |

* 39 of 41 cases have at least one function-level label (mean 1.05 gold functions
  per case); all 41 have file-level labels.
* Splits are assigned **per repository** by a seeded hash (`--seed`, default 0), not
  chosen by hand.
* The `test` split is dominated by pytest (17/20). Report per-repository results
  before drawing conclusions about it.
* Each `EvalCase` separates what a retriever sees (`issue`) from the scoring-only
  `ground_truth`. The gold patch is **never stored**, only its SHA-256.

### Rebuild

```bash
python -m evaluation.build_eval_set \
    --repos psf/requests pallets/flask mwaskom/seaborn pylint-dev/astroid \
            pylint-dev/pylint pytest-dev/pytest \
    --output evaluation/datasets/swebench_lite_subset.jsonl
```

Rebuilding from scratch reproduces the committed file byte for byte (verified by
SHA-256). Raw rows are cached in `evaluation/datasets/raw/` (git-ignored).

## 3. Ground-truth labels

Annotation method `patch-hunk-to-function/v1`. Labels are computed against the
index of the **pre-fix** revision (`base_commit`):

* **File level:** a Python file that exists at `base_commit` and is modified or
  deleted by the fix.
* **Function level:** a function or method whose line range contains a removed or
  rewritten line, or into whose body the fix inserts lines. An insertion right
  after a function's last line counts only if the added code is indented deeper
  than the `def` (it extends the body rather than starting a sibling).
* **Not labeled:** edits outside any function (imports, constants, class bodies)
  and brand-new functions. They are tallied as `unlocated_changes`. Class units
  are never labeled, since a class span would contain all of its methods.

Validation performed at build time and in tests:

* The gold patch must apply cleanly to the checkout (`git apply --check`); all 41
  cases pass.
* At run time every gold unit must exist in the freshly indexed pool, otherwise
  the case is excluded as `label_mismatch` instead of being scored.
* Independent audit: in 10 cases a function name in git's hunk header differs from
  our label. In every one, git's function does **not** contain an edited line (the
  header names the *preceding* `def`); our unit does in 9 of 10, and the tenth only
  adds a new method, so no unit label is correct.

Known limits: a bug can involve code the fix never touched, and a fix can be
imperfect or touch several places. Labels reflect what the *fix changed*, which is
the standard but imperfect proxy for "where the bug is".

## 4. Candidate pool

The pool is the set of units a retriever ranks. It is defined without reference
to any label: **leaf units (functions and methods) outside test files** (paths
under `tests/`, `test/`, `testing/`, or named `test_*.py`, `*_test.py`,
`conftest.py`). Mean pool size on the `test` split is about 1,340 units. Cases
whose gold files fall under test paths are marked unsuitable.

## 5. Leakage controls

| Risk | Control |
|---|---|
| Train/eval overlap | Repository-aware split: all cases of a repo share one split (`repos_spanning_splits` is checked and tested). |
| Duplicate bugs | Near-identical issues (3-gram Jaccard >= 0.8) and identical gold patches are collapsed; the later case is marked `duplicate_of:<id>` and excluded. |
| Fix information as input | Retrievers receive only `RetrievalQuery` (`query_id`, `issue_title`, `issue_body`); this is enforced by the type and tested. The patch is not stored. |
| Post-fix index | Repositories are indexed at `base_commit`; the gold patch must apply cleanly to that checkout. |
| Issues that quote the fix | Flagged, not dropped (`issue_contains_patch_lines`): 2 of 41 cases (astroid-1978, pytest-5227). Use `--exclude-patch-leaks` to drop them. |
| Issues naming the file | Informational only (`issue_mentions_gold_file`): 15 of 41. These cases are easier for lexical retrieval. |
| Unreproducible preprocessing | Every build writes a manifest with dataset revision, builder options, index configuration, and exclusions. |

## 6. Metrics

Computed at two levels. Each aggregate reports its **denominator** (`n`) and a
95% percentile bootstrap confidence interval over cases (1,000 resamples, seeded).

* **hit@k**: 1 if at least one relevant item is in the top *k*. This is the
  "Recall@k" of `project.md` §11.
* **recall@k**: fraction of the case's relevant items found in the top *k*.
* **MRR**: reciprocal rank of the first relevant item in the returned list
  (default depth 100; 0 if absent).
* **nDCG@10**: binary-relevance nDCG over the top 10.
* **File level:** ranked units are collapsed to distinct files (first occurrence
  kept) and scored against the relevant files. Denominator: all evaluated cases.
* **Function level:** scored against the relevant functions. Denominator: only
  cases with at least one function label.
* **Latency:** per-query retrieval seconds (mean, median, p95, max) and per-revision
  `prepare` time (index build) are reported separately.

File-level scores are much higher than function-level ones for the same ranking:
a file has many functions, so the chance of touching the right file is far larger.
Compare methods within a level, never across levels.

## 7. Running

```bash
python -m evaluation.run_retrieval_eval \
    --dataset evaluation/datasets/swebench_lite_subset.jsonl \
    --retriever random --split test --seed 0
```

Writes `evaluation/reports/<dataset>__<retriever>__<split>__seed<N>.{json,md}`.
The JSON includes the full config, environment, per-case results, and top-10
candidates for error analysis. New retrievers implement
`app.retrieval.base.Retriever` (`prepare(units)`, `retrieve(query, k)`) and are
registered in `app.retrieval.registry`. The runner rejects any result that is not
a real, unique pool member with consecutive ranks.

## 8. Baseline: random ranking

Held-out `test` split, seed 0, 20 cases (19 with function labels). This is the
floor every real retriever must beat.

| Level | Metric | Mean | 95% CI | n |
|---|---|---|---|---|
| file | hit@5 | 0.150 | [0.000, 0.300] | 20 |
| file | hit@10 | 0.300 | [0.100, 0.500] | 20 |
| file | MRR | 0.083 | [0.057, 0.113] | 20 |
| file | nDCG@10 | 0.108 | [0.038, 0.183] | 20 |
| function | hit@5 | 0.000 | [0.000, 0.000] | 19 |
| function | hit@10 | 0.000 | [0.000, 0.000] | 19 |
| function | MRR | 0.003 | [0.000, 0.008] | 19 |
| function | nDCG@10 | 0.000 | [0.000, 0.000] | 19 |

Full report: `evaluation/reports/swebench_lite_subset__random__test__seed0.md`.
The wide file-level intervals reflect the small sample (n = 20).

## 9. BM25 baseline (Phase 3)

Lexical retrieval is the mandatory non-neural baseline that every later method
must beat on the same cases (project.md ?7.2).

**Implementation** (`app/retrieval/bm25.py`, `app/retrieval/tokenizer.py`): an
in-house BM25 over an inverted index, not a library. It is verified against an
independent brute-force implementation on random corpora (agreement to 1e-12), and
the tests were checked to fail when the IDF, length normalization, or tie-breaking
is deliberately broken.

* **Tokenizer:** code-aware. Identifiers are kept whole *and* split into sub-words
  (`restoreSession` -> `restoresession`, `restore`, `session`), so a camelCase
  mention in a bug report matches a snake_case function. Stopwords (English and
  Python syntax words) and one-character tokens are dropped; numbers, error names
  and paths survive.
* **Documents:** one per function/method, with four fields: qualified symbol name,
  file path, docstring, and full source (which includes comments and string
  literals). Term frequencies are field-weighted (BM25F with shared length
  normalization).
* **Query:** the raw issue text (title + body), tokenized the same way. Better
  query construction is Phase 4.
* **Output:** top-*k* units with `rank`, `score`, and `signals={"bm25": score}`, so
  the raw score is kept for fusion. Only units sharing at least one term with the
  query are returned, so a list can be shorter than *k*. `BM25Retriever.explain()`
  lists the terms behind a unit's score.

**Tuning protocol.** Hyperparameters were chosen on the `dev` split only
(`python -m evaluation.tune_bm25`; the `test` split is never read) with the
selection rule fixed beforehand: function-level MRR, then file-level MRR. The grid
had 72 configurations (`k1`, `b`, name/path field weights, query term-frequency
mode). The chosen values are frozen as the `BM25Config` defaults, and
`configs/bm25_classic.json` keeps the textbook configuration for comparison.

| | k1 | b | w_name | w_path | w_doc | w_body | query tf |
|---|---|---|---|---|---|---|---|
| classic | 1.2 | 0.75 | 1 | 1 | 1 | 1 | raw |
| **tuned (frozen)** | 0.9 | 0.75 | 3 | 2 | 1 | 1 | log |

Tuning sensitivity (`evaluation/reports/bm25_tuning_dev.md`): only the query
term-frequency mode clearly mattered (`binary` 0.371 vs `log`/`raw` 0.478 mean
function MRR). `k1`, `b`, and the field weights each moved the mean by about 0.01
or less. The tuned configuration is therefore a modest, partly noise-driven
improvement, which the held-out split confirms only for MRR (below).

### Results

Mean [95% bootstrap CI]; "fn" = function level, n = cases scored (function/file).

| Method | Split | fn hit@5 | fn hit@10 | fn MRR | fn nDCG@10 | file hit@10 | file MRR |
|---|---|---|---|---|---|---|---|
| random | test (n=19/20) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.300 [0.100, 0.500] | 0.083 [0.057, 0.113] |
| BM25 classic | test (n=19/20) | 0.316 [0.105, 0.526] | 0.421 [0.211, 0.632] | 0.245 [0.101, 0.406] | 0.266 [0.104, 0.434] | 0.950 [0.850, 1.000] | 0.434 [0.290, 0.604] |
| **BM25 tuned** | test (n=19/20) | 0.526 [0.316, 0.737] | 0.526 [0.316, 0.737] | 0.335 [0.187, 0.496] | 0.374 [0.203, 0.551] | 0.950 [0.850, 1.000] | 0.554 [0.404, 0.701] |
| BM25 classic | dev (n=20/21) | 0.700 [0.500, 0.900] | 0.750 [0.550, 0.900] | 0.465 [0.284, 0.653] | 0.532 [0.352, 0.704] | 0.857 [0.667, 1.000] | 0.628 [0.458, 0.786] |
| BM25 tuned | dev (n=20/21) | 0.650 [0.450, 0.850] | 0.700 [0.500, 0.900] | 0.513 [0.327, 0.699] | 0.552 [0.365, 0.734] | 0.905 [0.762, 1.000] | 0.695 [0.530, 0.852] |

The `dev` rows are optimistic (the tuned row was selected on them); the `test` rows
are the honest numbers. Retrieval takes about 3.5 ms per query (p95 about 6.7 ms);
building a revision's index takes about 0.14 s.

Paired comparisons on `test` (`python -m evaluation.compare_runs`, difference B - A
with a paired bootstrap CI over cases; reports in `evaluation/reports/compare_*.md`):

* **BM25 classic vs random:** every function- and file-level metric improves and
  every CI excludes zero (function MRR +0.241 [+0.100, +0.406]; hit@10 +0.421).
* **Tuned vs classic:** function MRR +0.091 [+0.010, +0.171] and file MRR +0.120
  [+0.003, +0.252] exclude zero, but hit@5, hit@10, and nDCG@10 do not
  (e.g. hit@5 +0.211 [-0.053, +0.421]). With 19 cases and 72 candidate
  configurations tried, treat tuning as a small, unconfirmed gain; classic BM25 is
  the safer reference when a strictly untuned baseline is wanted.

### Where lexical retrieval fails (held-out, tuned; small n, descriptive only)

* Finding the right **file** is easy (hit@10 = 0.95); finding the right **function**
  is not (0.53). The residual is mostly ranking within files.
* When the issue names the gold file or module: hit@10 = 1.00 (n = 4). When it does
  not: 0.40 (n = 15). Lexical overlap carries most of the signal.
* Short issues (< 1,970 characters) do better than long ones (hit@10 0.67 vs 0.40).
  Long reports contain tracebacks and unrelated code whose tokens dilute the query.
  This motivates query construction in Phase 4.
* Of the nine misses, four have the gold function outside the top 100 (including
  pytest-5221, whose issue is only 239 characters). Vocabulary mismatch between the
  report and the code is the likely cause, but this has not been verified; it is the
  hypothesis Phase 5's dense retrieval will test.

These are observations on 19 cases, not established effects.

## 10. Dense and hybrid retrieval with SweRankEmbed-Small

Code-specific dense retrieval, as planned for Phase 5, run early because the
embeddings became available (`docs/gpu_embeddings.md`).

**Setup.** Every candidate function was embedded once on a Kaggle Tesla T4 (fp16,
16,350 unique functions in 124 s) with `Salesforce/SweRankEmbed-Small`, pinned to
an exact revision and weights hash. The artifacts passed `verify`: format,
row alignment with the export, config compatibility, full coverage of all 51,844
pool functions, and an 8-row CPU re-embedding check (minimum cosine 0.9999985).
At query time only the issue text is encoded, on CPU.

* **Dense:** cosine similarity between the issue and every function
  (`configs/dense_swerank_small.json`).
* **Hybrid:** reciprocal rank fusion of tuned BM25 and dense, with equal weights,
  `rrf_k` = 60 and 200 candidates from each (`configs/hybrid_swerank_small.json`).
  These are untuned defaults.

Nothing was tuned on `test`. Hybrid was named as the candidate from `dev` results
before `test` was run.

### Results

Mean [95% bootstrap CI]; "fn" = function level, n = cases scored (function/file).

| Method | Split | fn hit@5 | fn hit@10 | fn MRR | fn nDCG@10 | file hit@10 | file MRR |
|---|---|---|---|---|---|---|---|
| BM25 tuned | test (n=19/20) | 0.526 [0.316, 0.737] | 0.526 [0.316, 0.737] | 0.335 [0.187, 0.496] | 0.374 [0.203, 0.551] | 0.950 [0.850, 1.000] | 0.554 [0.404, 0.701] |
| Dense | test (n=19/20) | 0.632 [0.421, 0.842] | 0.737 [0.526, 0.895] | 0.410 [0.241, 0.597] | 0.461 [0.296, 0.623] | 1.000 [1.000, 1.000] | 0.668 [0.505, 0.808] |
| **Hybrid** | test (n=19/20) | 0.632 [0.421, 0.842] | 0.737 [0.526, 0.895] | 0.507 [0.326, 0.690] | 0.535 [0.351, 0.715] | 0.950 [0.850, 1.000] | 0.687 [0.520, 0.835] |
| BM25 tuned | dev (n=20/21) | 0.650 | 0.700 | 0.513 | 0.552 | 0.905 | 0.695 |
| Dense | dev (n=20/21) | 0.700 | 0.750 | 0.506 | 0.561 | 0.952 | 0.761 |
| Hybrid | dev (n=20/21) | 0.700 | 0.750 | 0.560 | 0.601 | 0.952 | 0.773 |

Paired differences on `test`, B - A against tuned BM25 (`evaluation/reports/compare_test_bm25-tuned_vs_*.md`):

| Metric | Dense | Hybrid |
|---|---|---|
| fn hit@5 | +0.105 [+0.000, +0.263], W/T/L 2/17/0 | +0.105 [+0.000, +0.263], 2/17/0 |
| fn hit@10 | **+0.211 [+0.053, +0.421]**, 4/15/0 | **+0.211 [+0.053, +0.421]**, 4/15/0 |
| fn MRR | +0.075 [-0.102, +0.253], 10/3/6 | **+0.172 [+0.077, +0.272]**, 13/6/0 |
| fn nDCG@10 | +0.087 [-0.065, +0.233], 7/7/5 | **+0.160 [+0.078, +0.248]**, 9/10/0 |
| file MRR | +0.113 [-0.013, +0.252], 8/8/4 | **+0.133 [+0.033, +0.239]**, 8/10/2 |

Bold: the interval excludes zero (no multiple-comparison correction).

* **Hybrid is the best method so far.** It never loses a case to BM25 at the
  function level. Its gains on hit@10, MRR and nDCG@10 are significant on `test`.
* **Dense alone finds more functions but orders them less consistently**
  (MRR wins/losses 10/6).
* **hit@5 goes from 0.53 to 0.63** for both, but the interval ends at zero, so
  that gain is not yet confirmed.
* **Cost:** about 2 s per query on CPU (encoding the issue), against about 4 ms for BM25.
  Function vectors are computed once on a GPU.

These are 19 held-out cases, 17 from pytest. Confirming them is why the
evaluation set is being scaled up (Phase 4, section 11).

## 11. Full SWE-bench Lite evaluation set (Phase 4)

The 41-case subset is too small to confirm gains under about 15 points, so the
evaluation set was rebuilt from **all of SWE-bench Lite** with the same labeler,
leakage controls, and candidate-pool rule.

```bash
python -m evaluation.build_eval_set \
    --heldout-repos astropy/astropy django/django marshmallow-code/marshmallow \
        pallets/flask pvlib/pvlib-python pydicom/pydicom pytest-dev/pytest sqlfluff/sqlfluff \
    --checkpoint data/build_full.ckpt.jsonl \
    --output evaluation/datasets/swebench_lite_full.jsonl
```

* 323 instances processed, **321 suitable** (2 excluded: gold file has syntax
  errors), **293 with function labels**. 19 issues quote lines of the fix
  (flagged, not dropped). The build took about 2 h 25 min on this laptop; it can be
  resumed (`--checkpoint`).
* **Split, fixed before any full-set result:** the old subset's repositories keep
  their split (pytest and flask stay held out). The 12 new repositories were split
  by the seeded hash (`assign_splits`, seed 0, fraction 0.5).
  * `dev` (164 cases): sympy 77, scikit-learn 23, matplotlib 21, sphinx 16,
    pylint 6, requests 6, astroid 5, xarray 5, seaborn 4, pyvista 1.
  * `test` (157 cases): django 114, pytest 17, astropy 6, pvlib 5, pydicom 5,
    sqlfluff 5, flask 3, marshmallow 2.
* Every report now also includes **acc@k** (all gold functions in the top k), the
  metric used by Agentless, LocAgent and SweRank.
* The full `test` split is dominated by django (73%), as SWE-bench Lite itself is.

### BM25 on the full set

| Split | n (fn/file) | fn hit@5 | fn acc@5 | fn hit@10 | fn acc@10 | fn MRR | file hit@10 | file MRR |
|---|---|---|---|---|---|---|---|---|
| dev | 150/164 | 0.520 [0.440, 0.600] | 0.460 [0.380, 0.533] | 0.587 | 0.540 | 0.355 [0.294, 0.419] | 0.805 | 0.619 |
| test | 143/157 | 0.594 [0.510, 0.672] | 0.566 [0.483, 0.643] | 0.671 | 0.657 | 0.468 [0.393, 0.538] | 0.854 | 0.642 |

Mean candidate pool: 11,628 functions per case on `dev`, 6,032 on `test`. Retrieval
takes 25-45 ms per query. BM25 settings are the ones tuned on the old 21-case
`dev`; they were not re-tuned.

Our full-set BM25 is far stronger than the BM25 in the SweRank and LocAgent papers
(function Acc@5 31.8 on SWE-bench Lite). The number is not directly comparable:
their set has 274 cases, and their tokenization and documents differ. But it
confirms the code-aware tokenizer and field weights matter.

### Issue-mention signals (`--retriever bm25-mentions`)

`app/issue_understanding/entities.py` extracts names, file paths and dotted modules
the issue mentions. `app/retrieval/mention_boost.py` adds them to normalized BM25,
with weights chosen on the old `dev` (`w_mention` 0.25, `w_path` 0.5). Paired
against BM25 (`evaluation/reports/compare_full_*_bm25-tuned_vs_bm25-mentions.md`):

| Split | fn hit@5 | fn acc@5 | fn MRR | file MRR |
|---|---|---|---|---|
| dev (n=150/164) | +0.013 [-0.033, +0.067] | +0.000 | **+0.030 [+0.005, +0.058]** | **+0.037 [+0.009, +0.064]** |
| test (n=143/157) | +0.000 [-0.028, +0.028] | +0.007 | +0.007 [-0.020, +0.036] | +0.015 [-0.010, +0.041] |

The small `dev` gain in ranking quality **does not carry over to `test`**
(function MRR wins/ties/losses 14/106/23). The mention boost is kept as an
optional retriever but is not part of the default pipeline.

### Not yet run on the full set

Dense and hybrid retrieval (section 10) need function embeddings for the new
repositories: a new export and Kaggle GPU run (`docs/gpu_embeddings.md`).
Their gains are therefore still measured only on the 19-case subset.

## 12. Limitations

* **Small sample (subset).** The 41-case subset (20 held out) detects only large
  effects; the full set (section 11) has 157 held-out cases but is 73% django.
* **Original subset:** 41 cases (20 held out) detect only large effects. Expect wide
  confidence intervals; prefer paired comparisons and report per-repository results.
* **Skewed held-out split.** pytest contributes 17 of 20 test cases.
* **SWE-bench Lite bias.** It favors issues whose fixes are small and well-specified.
* **Tuning on 21 dev cases.** BM25 hyperparameters are weakly identified at this
  sample size (section 9); prefer the classic configuration when an untuned
  reference is required.
* **Test split reuse.** `test` has now been scored for BM25, the issue-mention
  experiment, and dense/hybrid. Each method was fixed beforehand, but the split
  is no longer untouched; the scaled-up set adds fresh held-out repositories.
* **Pool definition matters.** Including test code or class units changes absolute
  numbers; compare only runs with the same pool rule (recorded in each report).
* **Windows path length.** Deep repositories (astroid, pytest) can exceed the
  260-character limit. Checkouts use short directory names and `core.longpaths`;
  if git still reports "Filename too long", set `REPOMIND_DATA_DIR` to a short path
  such as `C:\rm`.
