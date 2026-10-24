# Can we beat BM25's 53%? Findings (2026-10-22)

**Update (2026-10-23): yes.** With function embeddings computed once on a
Kaggle GPU, the hybrid of tuned BM25 and SweRankEmbed-Small raises held-out
function hit@5 from 0.526 to 0.632 and hit@10 from 0.526 to 0.737. It never ranked
the gold function lower than BM25 did on any of the 19 cases (MRR +0.172, 95% CI
[+0.077, +0.272]). The hit@5 gain alone is not yet significant (CI [+0.000,
+0.263]). Details: `docs/evaluation.md` section 10. The original analysis below
is kept as written; its conclusion "not on this laptop" was about doing the
embedding itself on CPU.

**Original short answer (2026-10-22):** not yet on this laptop. None of the methods
that run here in a reasonable time beat tuned BM25 on held-out function hit@5
(0.526, n = 19). The literature shows two methods that do: a code-specific
embedding model run over the whole repository, and LLM reranking. Both need
resources this machine doesn't have (a GPU, or an LLM API). The roadmap has been
revised accordingly (`plan.md`).

All numbers below were produced in this session. Every method was compared on the
`dev` split. The `test` split was used **once**, to check the single method chosen
on `dev` (section 3).

## 1. Is 53% actually low?

Published function-level Acc@5 on SWE-bench Lite (274 cases; Acc@k counts a hit
only if *all* gold functions are in the top k, which is nearly our hit@k since we
average 1.05 gold functions per case):

| Method | Function Acc@5 | Needs |
|---|---|---|
| BM25 (LocAgent / SweRank papers) | 31.8 | CPU |
| Agentless + GPT-4o | 55.5 | LLM API |
| CodeRankEmbed (137M) | 51.8 | GPU for repo-scale indexing |
| SweRankEmbed-Small (137M) | 63.1 | GPU for repo-scale indexing |
| LocAgent + Claude-3.5 | 73.4 | LLM API, agent |
| SweRankEmbed-Large + GPT-4.1 listwise rerank | 81.8 | GPU + LLM API |

Sources: SweRank (arXiv 2505.07849, Table 1); LocAgent (ACL 2025).

Our tuned BM25 scores 52.6 on our held-out subset. Same-sized numbers aren't
directly comparable (different subset, n = 19 vs 274), but our BM25 is not weak:
code-aware tokenization and field weights put it in the range of embedding
retrievers and Agentless with GPT-4o. Satisfying results (70%+) in the literature
come from **code-specific dense retrieval plus LLM reranking**.

## 2. Experiments on `dev` (21 cases, 20 with function labels)

| Method | fn hit@5 | fn hit@10 | fn MRR | file hit@10 |
|---|---|---|---|---|
| **BM25 tuned (baseline)** | **0.650** | 0.700 | 0.513 | 0.905 |
| Dense, all-MiniLM-L6-v2 (general English model, 256 tokens) | 0.350 | 0.350 | 0.334 | 0.667 |
| Hybrid BM25 + MiniLM (RRF, equal weights) | 0.500 | 0.600 | 0.352 | 0.810 |
| BM25 top-30 reranked by SweRankEmbed-Small (dense only) | 0.600 | 0.700 | 0.374 | 0.905 |
| BM25 top-30 + SweRankEmbed-Small, RRF | 0.650 | 0.750 | 0.513 | 0.952 |
| BM25 top-30 + SweRankEmbed-Small, RRF, dense weight 2 | 0.650 | 0.700 | 0.503 | 0.952 |
| BM25 + issue-mention signals (symbol and file names in the issue) | **0.700** | 0.700 | **0.588** | – |

Paired against BM25 on `dev` (bootstrap 95% CI): MiniLM dense hit@5 −0.300
[−0.550, −0.050] (significantly worse); MiniLM hybrid −0.150 [−0.350, +0.050].

What this shows:

1. **A general-purpose sentence model hurts.** MiniLM was trained on English
   sentences, not on bug reports and code. It is worse than BM25 alone, and fusing
   it pulls BM25 down. Model choice matters more than "dense vs lexical".
2. **SweRankEmbed-Small is too slow to run over the whole repository on this CPU.**
   It encodes about 0.5-1.5 functions per second, so indexing the 9,377 distinct
   `dev` functions would take about 1.7 hours (the run aborted itself after
   projecting this). The published 63% assumes the whole repository is indexed.
3. **Used only to re-sort BM25's top 30, it adds almost nothing** (hit@10 +0.05,
   hit@5 and MRR unchanged). Two compromises made it feasible here and probably
   weaken it: functions were truncated to 256 tokens, and it can only reorder what
   BM25 already found. This is not a fair test of the model, only of what is
   feasible on this machine.
4. **Cheap lexical signals help a little.** Boosting functions whose name or file
   the issue mentions (backticks, calls, tracebacks, paths) raises `dev` MRR by
   0.075. A "file prior" (boosting functions in high-scoring files) hurt and was
   dropped.

## 3. Held-out check of the one method chosen on `dev`

The mention/path signals (weights chosen on `dev`, grid of 48, selection rule fixed
beforehand), run once on `test` (n = 19):

| | fn hit@5 | fn hit@10 | fn MRR |
|---|---|---|---|
| BM25 tuned | 0.526 | 0.526 | 0.335 |
| + mention/path signals | 0.526 | 0.526 | 0.381 |

Paired MRR +0.045 [−0.032, +0.133], wins/ties/losses 2/15/2; hit@5 unchanged.
It reorders the top results slightly better but finds no new functions. **It does
not move the 53%.**

## 4. Why the 53% can't move without more resources

* The misses are mostly **recall**, not ordering: in 4 of the 9 held-out misses
  the gold function is outside BM25's top 100. Reordering BM25 output (signals,
  top-k reranking) can't recover those. Finding them needs a retriever that
  matches meaning, and on this benchmark only code-specific models do that.
* That model needs a **GPU to index**. Embeddings only need computing once per
  function version (the cache keys on content hash), so a one-off GPU job
  (e.g. a free Colab T4) followed by CPU-only query encoding is enough.
* The largest published gain, +10-20 points, comes from **LLM listwise
  reranking** of the top 20-30 candidates. A 7B model is impractical on this CPU,
  so that means a hosted LLM API, which sends issue text and code to a third party
  and costs money per issue. That needs your approval.
* **The test set is too small to see modest gains.** With n = 19, a 95% interval
  is about ±0.2, so improvements under about 15 points can't be distinguished from
  noise. Full SWE-bench Lite (274 function-labeled cases) would also make our
  numbers comparable to the papers.

## 5. Environment findings (this machine)

* `platform.platform()` hung the evaluation on Windows (it queries WMI). Reports
  now use a WMI-free call (`evaluation/env_info.py`).
* Loading torch is unreliable here: one stall inside torch's DLL loading, one
  OSError 1455 ("paging file too small"). Both happened before a reboot; after the
  reboot every load succeeded. Model loading now has a timeout and raises
  `DenseUnavailableError` instead of hanging.

## 6. Update: SweRankEmbed-Small over the whole repository (2026-10-23)

Section 2 could only re-sort BM25's top 30 with SweRankEmbed-Small, truncated to
256 tokens, because CPU encoding was too slow. With all 16,350 functions embedded
on a GPU (1,024 tokens), the model works as the papers suggest:

| Held-out `test` (n = 19) | fn hit@5 | fn hit@10 | fn MRR |
|---|---|---|---|
| BM25 tuned | 0.526 | 0.526 | 0.335 |
| Dense, SweRankEmbed-Small | 0.632 | 0.737 | 0.410 |
| Hybrid BM25 + SweRankEmbed | 0.632 | 0.737 | 0.507 |

So the earlier top-30 result reflected the compromises (truncation, BM25-only
candidates), not the model. Of the four held-out misses BM25 left outside its top
100 (section 4), dense retrieval finds three: pytest-5413 at rank 4, pytest-5221
at 16, and pytest-7220 at 68. pytest-5103 is still not found.

## 7. Reproduce

Scripts are in `experiments/`; reports are in `evaluation/reports/`.

```bash
python experiments/dense_minilm_dev.py        # dense + hybrid, MiniLM (~8 min)
python experiments/swerank_rerank_dev.py      # BM25 top-30 + SweRankEmbed (~7 min, needs trust_remote_code)
python experiments/bm25_signals.py            # mention/path/file signals, dev grid + one test check (~15 s)
```

`SweRankEmbed-Small` runs custom model code (`trust_remote_code=True`) from its
Hugging Face repository. Pin a revision before relying on it.
