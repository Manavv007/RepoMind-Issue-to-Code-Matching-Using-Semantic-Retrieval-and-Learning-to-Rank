# Retrieval evaluation: `hybrid`

- Dataset: `swebench_lite_subset.jsonl`
- Split: `dev`  |  seed: 0  |  depth: 100
- Candidate pool: leaf units (function, method) excluding test files
- Retriever parameters: `{"bm25": {"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}, "dense": {"batch_size": 8, "code_revision": "92d97331f1f4b6a366c1f161354b9f3390cc219f", "device": "cpu", "embeddings_path": "data/embeddings/swerank-small-v1", "include_path": true, "load_timeout_seconds": 300.0, "max_seq_length": 1024, "model_name": "Salesforce/SweRankEmbed-Small", "on_missing": "error", "query_prefix": "Represent this query for searching relevant code: ", "revision": "745d2a06103a66d3cfa600aa52fc0d3523010daa", "trust_remote_code": true, "weights_sha256": "e40687da7f148e684648691facbaa2222503829afb9d21b9b3913034bef14f41"}, "fusion": {"depth": 200, "rrf_k": 60, "w_bm25": 1.0, "w_dense": 1.0}}`
- Cases: 21 evaluated of 21 selected (41 in dataset); excluded: {'other_split': 20}
- Generated: 2026-10-22T18:44:28+00:00  |  RepoMind commit: `f4e95dff58`
- Pool size per case: mean 1190, median 1302 (chance level scales with this)

## Results (mean, 95% bootstrap CI over cases, n = cases scored)

| Level | Metric | Mean | 95% CI | n |
|---|---|---|---|---|
| file | hit@5 | 0.905 | [0.762, 1.000] | 21 |
| file | recall@5 | 0.905 | [0.762, 1.000] | 21 |
| file | hit@10 | 0.952 | [0.857, 1.000] | 21 |
| file | recall@10 | 0.952 | [0.857, 1.000] | 21 |
| file | mrr | 0.773 | [0.618, 0.911] | 21 |
| file | ndcg@10 | 0.816 | [0.681, 0.932] | 21 |
| function | hit@5 | 0.700 | [0.500, 0.900] | 20 |
| function | recall@5 | 0.700 | [0.500, 0.900] | 20 |
| function | hit@10 | 0.750 | [0.550, 0.900] | 20 |
| function | recall@10 | 0.750 | [0.550, 0.900] | 20 |
| function | mrr | 0.560 | [0.358, 0.749] | 20 |
| function | ndcg@10 | 0.601 | [0.408, 0.773] | 20 |

## Latency (retrieval only, seconds per query)

mean 1.9631, median 1.6786, p95 4.3943, max 4.5800 (n = 21)

Per-revision `prepare` (index build): mean 0.3678s (n = 21)

## Metric definitions

- **hit@k**: 1 if at least one relevant item is in the top k (project.md 'Recall@k').
- **recall@k**: Fraction of the case's relevant items found in the top k.
- **mrr**: Reciprocal rank of the first relevant item in the returned list (0 if absent), averaged over cases.
- **ndcg@k**: Binary-relevance nDCG over the top k.
- **file level**: The ranked units are collapsed to distinct files (first occurrence kept); relevant files are the Python files the fix modifies.
- **function level**: Relevant units are the functions/methods the fix modifies, evaluated only on cases that have at least one such unit.
