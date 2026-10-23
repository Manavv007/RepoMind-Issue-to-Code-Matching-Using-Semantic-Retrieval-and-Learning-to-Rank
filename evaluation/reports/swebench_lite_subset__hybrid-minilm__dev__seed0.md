# Retrieval evaluation: `hybrid`

- Dataset: `swebench_lite_subset.jsonl`
- Split: `dev`  |  seed: 0  |  depth: 100
- Candidate pool: leaf units (function, method) excluding test files
- Retriever parameters: `{"bm25": {"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}, "dense": {"batch_size": 32, "device": "cpu", "include_path": true, "load_timeout_seconds": 180.0, "max_seq_length": 256, "model_name": "sentence-transformers/all-MiniLM-L6-v2", "query_prefix": "", "revision": null, "trust_remote_code": false}, "fusion": {"depth": 200, "rrf_k": 60, "w_bm25": 1.0, "w_dense": 1.0}}`
- Cases: 21 evaluated of 21 selected (41 in dataset); excluded: {'other_split': 20}
- Generated: 2026-10-22T16:52:30+00:00  |  RepoMind commit: `f4e95dff58`
- Pool size per case: mean 1190, median 1302 (chance level scales with this)

## Results (mean, 95% bootstrap CI over cases, n = cases scored)

| Level | Metric | Mean | 95% CI | n |
|---|---|---|---|---|
| file | hit@5 | 0.714 | [0.524, 0.905] | 21 |
| file | recall@5 | 0.714 | [0.524, 0.905] | 21 |
| file | hit@10 | 0.810 | [0.619, 0.952] | 21 |
| file | recall@10 | 0.810 | [0.619, 0.952] | 21 |
| file | mrr | 0.520 | [0.363, 0.698] | 21 |
| file | ndcg@10 | 0.580 | [0.425, 0.744] | 21 |
| function | hit@5 | 0.500 | [0.299, 0.700] | 20 |
| function | recall@5 | 0.500 | [0.299, 0.700] | 20 |
| function | hit@10 | 0.600 | [0.400, 0.800] | 20 |
| function | recall@10 | 0.575 | [0.350, 0.775] | 20 |
| function | mrr | 0.352 | [0.186, 0.532] | 20 |
| function | ndcg@10 | 0.395 | [0.211, 0.570] | 20 |

## Latency (retrieval only, seconds per query)

mean 0.2130, median 0.1692, p95 0.3730, max 0.3800 (n = 21)

Per-revision `prepare` (index build): mean 0.3229s (n = 21)

## Metric definitions

- **hit@k**: 1 if at least one relevant item is in the top k (project.md 'Recall@k').
- **recall@k**: Fraction of the case's relevant items found in the top k.
- **mrr**: Reciprocal rank of the first relevant item in the returned list (0 if absent), averaged over cases.
- **ndcg@k**: Binary-relevance nDCG over the top k.
- **file level**: The ranked units are collapsed to distinct files (first occurrence kept); relevant files are the Python files the fix modifies.
- **function level**: Relevant units are the functions/methods the fix modifies, evaluated only on cases that have at least one such unit.
