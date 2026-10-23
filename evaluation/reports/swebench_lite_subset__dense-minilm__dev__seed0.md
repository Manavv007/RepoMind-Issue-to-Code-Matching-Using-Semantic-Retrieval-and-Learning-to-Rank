# Retrieval evaluation: `dense`

- Dataset: `swebench_lite_subset.jsonl`
- Split: `dev`  |  seed: 0  |  depth: 100
- Candidate pool: leaf units (function, method) excluding test files
- Retriever parameters: `{"batch_size": 32, "device": "cpu", "include_path": true, "load_timeout_seconds": 180.0, "max_seq_length": 256, "model_name": "sentence-transformers/all-MiniLM-L6-v2", "query_prefix": "", "revision": null, "trust_remote_code": false}`
- Cases: 21 evaluated of 21 selected (41 in dataset); excluded: {'other_split': 20}
- Generated: 2026-10-22T16:52:12+00:00  |  RepoMind commit: `f4e95dff58`
- Pool size per case: mean 1190, median 1302 (chance level scales with this)

## Results (mean, 95% bootstrap CI over cases, n = cases scored)

| Level | Metric | Mean | 95% CI | n |
|---|---|---|---|---|
| file | hit@5 | 0.571 | [0.381, 0.810] | 21 |
| file | recall@5 | 0.571 | [0.381, 0.810] | 21 |
| file | hit@10 | 0.667 | [0.476, 0.857] | 21 |
| file | recall@10 | 0.667 | [0.476, 0.857] | 21 |
| file | mrr | 0.476 | [0.300, 0.670] | 21 |
| file | ndcg@10 | 0.511 | [0.338, 0.701] | 21 |
| function | hit@5 | 0.350 | [0.150, 0.550] | 20 |
| function | recall@5 | 0.350 | [0.150, 0.550] | 20 |
| function | hit@10 | 0.350 | [0.150, 0.550] | 20 |
| function | recall@10 | 0.350 | [0.150, 0.550] | 20 |
| function | mrr | 0.334 | [0.137, 0.532] | 20 |
| function | ndcg@10 | 0.322 | [0.122, 0.522] | 20 |

## Latency (retrieval only, seconds per query)

mean 0.1591, median 0.1206, p95 0.3100, max 0.3121 (n = 21)

Per-revision `prepare` (index build): mean 19.5209s (n = 21)

## Metric definitions

- **hit@k**: 1 if at least one relevant item is in the top k (project.md 'Recall@k').
- **recall@k**: Fraction of the case's relevant items found in the top k.
- **mrr**: Reciprocal rank of the first relevant item in the returned list (0 if absent), averaged over cases.
- **ndcg@k**: Binary-relevance nDCG over the top k.
- **file level**: The ranked units are collapsed to distinct files (first occurrence kept); relevant files are the Python files the fix modifies.
- **function level**: Relevant units are the functions/methods the fix modifies, evaluated only on cases that have at least one such unit.
