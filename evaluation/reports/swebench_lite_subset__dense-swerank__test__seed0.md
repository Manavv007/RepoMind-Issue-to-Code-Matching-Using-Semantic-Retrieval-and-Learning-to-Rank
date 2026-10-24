# Retrieval evaluation: `dense`

- Dataset: `swebench_lite_subset.jsonl`
- Split: `test`  |  seed: 0  |  depth: 100
- Candidate pool: leaf units (function, method) excluding test files
- Retriever parameters: `{"batch_size": 8, "code_revision": "92d97331f1f4b6a366c1f161354b9f3390cc219f", "device": "cpu", "embeddings_path": "data/embeddings/swerank-small-v1", "include_path": true, "load_timeout_seconds": 300.0, "max_seq_length": 1024, "model_name": "Salesforce/SweRankEmbed-Small", "on_missing": "error", "query_prefix": "Represent this query for searching relevant code: ", "revision": "745d2a06103a66d3cfa600aa52fc0d3523010daa", "trust_remote_code": true, "weights_sha256": "e40687da7f148e684648691facbaa2222503829afb9d21b9b3913034bef14f41"}`
- Cases: 20 evaluated of 20 selected (41 in dataset); excluded: {'other_split': 21}
- Generated: 2026-10-23T15:18:39+00:00  |  RepoMind commit: `8479231e60`
- Pool size per case: mean 1343, median 1440 (chance level scales with this)

## Results (mean, 95% bootstrap CI over cases, n = cases scored)

| Level | Metric | Mean | 95% CI | n |
|---|---|---|---|---|
| file | hit@5 | 0.900 | [0.750, 1.000] | 20 |
| file | recall@5 | 0.900 | [0.750, 1.000] | 20 |
| file | hit@10 | 1.000 | [1.000, 1.000] | 20 |
| file | recall@10 | 1.000 | [1.000, 1.000] | 20 |
| file | mrr | 0.668 | [0.505, 0.808] | 20 |
| file | ndcg@10 | 0.748 | [0.623, 0.856] | 20 |
| function | hit@5 | 0.632 | [0.421, 0.842] | 19 |
| function | recall@5 | 0.632 | [0.421, 0.842] | 19 |
| function | hit@10 | 0.737 | [0.526, 0.895] | 19 |
| function | recall@10 | 0.711 | [0.500, 0.895] | 19 |
| function | mrr | 0.410 | [0.241, 0.597] | 19 |
| function | ndcg@10 | 0.461 | [0.296, 0.623] | 19 |

## Latency (retrieval only, seconds per query)

mean 2.0270, median 1.7666, p95 3.7522, max 3.8088 (n = 20)

Per-revision `prepare` (index build): mean 0.0318s (n = 20)

## Metric definitions

- **hit@k**: 1 if at least one relevant item is in the top k (project.md 'Recall@k').
- **recall@k**: Fraction of the case's relevant items found in the top k.
- **mrr**: Reciprocal rank of the first relevant item in the returned list (0 if absent), averaged over cases.
- **ndcg@k**: Binary-relevance nDCG over the top k.
- **file level**: The ranked units are collapsed to distinct files (first occurrence kept); relevant files are the Python files the fix modifies.
- **function level**: Relevant units are the functions/methods the fix modifies, evaluated only on cases that have at least one such unit.
