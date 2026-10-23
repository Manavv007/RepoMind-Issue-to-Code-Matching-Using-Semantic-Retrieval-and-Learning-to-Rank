# Paired comparison on the `test` split

- A (baseline): `random` null
- B (candidate): `bm25` {"b": 0.75, "k1": 1.2, "query_tf": "raw", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 1.0, "w_path": 1.0}
- Difference = B - A per case; CI is a paired percentile bootstrap over cases.
  `excludes 0` means the interval does not contain zero (not a multiple-comparison-corrected claim).

| Level | Metric | n | Mean diff | 95% CI | W/T/L | Excludes 0 |
|---|---|---|---|---|---|---|
| function | hit@5 | 19 | +0.316 | [+0.105, +0.526] | 6/13/0 | yes |
| function | hit@10 | 19 | +0.421 | [+0.211, +0.632] | 8/11/0 | yes |
| function | mrr | 19 | +0.241 | [+0.100, +0.406] | 15/4/0 | yes |
| function | ndcg@10 | 19 | +0.266 | [+0.108, +0.442] | 8/11/0 | yes |
| file | hit@5 | 20 | +0.500 | [+0.150, +0.800] | 13/4/3 | yes |
| file | hit@10 | 20 | +0.650 | [+0.450, +0.850] | 13/7/0 | yes |
| file | mrr | 20 | +0.351 | [+0.193, +0.531] | 16/0/4 | yes |
| file | ndcg@10 | 20 | +0.445 | [+0.289, +0.612] | 16/1/3 | yes |

Function-level MRR difference by repository: pallets/flask +0.357, pytest-dev/pytest +0.220
