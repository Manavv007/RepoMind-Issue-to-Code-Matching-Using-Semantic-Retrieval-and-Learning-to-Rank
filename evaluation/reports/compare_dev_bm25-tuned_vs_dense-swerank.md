# Paired comparison on the `dev` split

- A (baseline): `bm25` {"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}
- B (candidate): `dense` {"batch_size": 8, "code_revision": "92d97331f1f4b6a366c1f161354b9f3390cc219f", "device": "cpu", "embeddings_path": "data/embeddings/swerank-small-v1", "include_path": true, "load_timeout_seconds": 300.0, "max_seq_length": 1024, "model_name": "Salesforce/SweRankEmbed-Small", "on_missing": "error", "query_prefix": "Represent this query for searching relevant code: ", "revision": "745d2a06103a66d3cfa600aa52fc0d3523010daa", "trust_remote_code": true, "weights_sha256": "e40687da7f148e684648691facbaa2222503829afb9d21b9b3913034bef14f41"}
- Difference = B - A per case; CI is a paired percentile bootstrap over cases.
  `excludes 0` means the interval does not contain zero (not a multiple-comparison-corrected claim).

| Level | Metric | n | Mean diff | 95% CI | W/T/L | Excludes 0 |
|---|---|---|---|---|---|---|
| function | hit@5 | 20 | +0.050 | [-0.100, +0.250] | 2/17/1 | no |
| function | hit@10 | 20 | +0.050 | [-0.100, +0.201] | 2/17/1 | no |
| function | mrr | 20 | -0.007 | [-0.127, +0.120] | 7/6/7 | no |
| function | ndcg@10 | 20 | +0.008 | [-0.110, +0.131] | 5/10/5 | no |
| file | hit@5 | 21 | +0.095 | [+0.000, +0.238] | 2/19/0 | no |
| file | hit@10 | 21 | +0.048 | [+0.000, +0.143] | 1/20/0 | no |
| file | mrr | 21 | +0.066 | [-0.089, +0.237] | 6/11/4 | no |
| file | ndcg@10 | 21 | +0.065 | [-0.063, +0.211] | 6/11/4 | no |

Function-level MRR difference by repository: mwaskom/seaborn +0.086, psf/requests +0.052, pylint-dev/astroid -0.114, pylint-dev/pylint -0.058
