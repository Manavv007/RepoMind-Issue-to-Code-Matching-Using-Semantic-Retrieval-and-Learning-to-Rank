# Paired comparison on the `test` split

- A (baseline): `bm25` {"b": 0.75, "k1": 1.2, "query_tf": "raw", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 1.0, "w_path": 1.0}
- B (candidate): `bm25` {"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}
- Difference = B - A per case; CI is a paired percentile bootstrap over cases.
  `excludes 0` means the interval does not contain zero (not a multiple-comparison-corrected claim).

| Level | Metric | n | Mean diff | 95% CI | W/T/L | Excludes 0 |
|---|---|---|---|---|---|---|
| function | hit@5 | 19 | +0.211 | [-0.053, +0.421] | 5/13/1 | no |
| function | hit@10 | 19 | +0.105 | [-0.105, +0.316] | 3/15/1 | no |
| function | mrr | 19 | +0.091 | [+0.010, +0.171] | 9/8/2 | yes |
| function | ndcg@10 | 19 | +0.109 | [-0.007, +0.220] | 6/12/1 | no |
| file | hit@5 | 20 | +0.100 | [-0.150, +0.350] | 4/14/2 | no |
| file | hit@10 | 20 | +0.000 | [+0.000, +0.000] | 0/20/0 | no |
| file | mrr | 20 | +0.120 | [+0.003, +0.252] | 9/9/2 | yes |
| file | ndcg@10 | 20 | +0.094 | [-0.000, +0.198] | 8/10/2 | no |

Function-level MRR difference by repository: pallets/flask +0.006, pytest-dev/pytest +0.107
