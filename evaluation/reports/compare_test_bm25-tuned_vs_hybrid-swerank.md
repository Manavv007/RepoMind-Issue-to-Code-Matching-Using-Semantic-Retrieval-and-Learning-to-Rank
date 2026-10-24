# Paired comparison on the `test` split

- A (baseline): `bm25` {"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}
- B (candidate): `hybrid` {"bm25": {"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}, "dense": {"batch_size": 8, "code_revision": "92d97331f1f4b6a366c1f161354b9f3390cc219f", "device": "cpu", "embeddings_path": "data/embeddings/swerank-small-v1", "include_path": true, "load_timeout_seconds": 300.0, "max_seq_length": 1024, "model_name": "Salesforce/SweRankEmbed-Small", "on_missing": "error", "query_prefix": "Represent this query for searching relevant code: ", "revision": "745d2a06103a66d3cfa600aa52fc0d3523010daa", "trust_remote_code": true, "weights_sha256": "e40687da7f148e684648691facbaa2222503829afb9d21b9b3913034bef14f41"}, "fusion": {"depth": 200, "rrf_k": 60, "w_bm25": 1.0, "w_dense": 1.0}}
- Difference = B - A per case; CI is a paired percentile bootstrap over cases.
  `excludes 0` means the interval does not contain zero (not a multiple-comparison-corrected claim).

| Level | Metric | n | Mean diff | 95% CI | W/T/L | Excludes 0 |
|---|---|---|---|---|---|---|
| function | hit@5 | 19 | +0.105 | [+0.000, +0.263] | 2/17/0 | no |
| function | hit@10 | 19 | +0.211 | [+0.053, +0.421] | 4/15/0 | yes |
| function | mrr | 19 | +0.172 | [+0.077, +0.272] | 13/6/0 | yes |
| function | ndcg@10 | 19 | +0.160 | [+0.078, +0.248] | 9/10/0 | yes |
| file | hit@5 | 20 | +0.150 | [+0.000, +0.300] | 3/17/0 | no |
| file | hit@10 | 20 | +0.000 | [-0.150, +0.150] | 1/18/1 | no |
| file | mrr | 20 | +0.133 | [+0.033, +0.239] | 8/10/2 | yes |
| file | ndcg@10 | 20 | +0.102 | [+0.004, +0.207] | 8/10/2 | yes |

Function-level MRR difference by repository: pallets/flask +0.057, pytest-dev/pytest +0.193
