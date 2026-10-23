# Retrieval evaluation: `bm25`

- Dataset: `swebench_lite_subset.jsonl`
- Split: `test`  |  seed: 0  |  depth: 100
- Candidate pool: leaf units (function, method) excluding test files
- Retriever parameters: `{"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}`
- Cases: 20 evaluated of 20 selected (41 in dataset); excluded: {'other_split': 21}
- Generated: 2026-10-22T07:31:59+00:00  |  RepoMind commit: `f4e95dff58`
- Pool size per case: mean 1343, median 1440 (chance level scales with this)

## Results (mean, 95% bootstrap CI over cases, n = cases scored)

| Level | Metric | Mean | 95% CI | n |
|---|---|---|---|---|
| file | hit@5 | 0.750 | [0.550, 0.950] | 20 |
| file | recall@5 | 0.750 | [0.550, 0.950] | 20 |
| file | hit@10 | 0.950 | [0.850, 1.000] | 20 |
| file | recall@10 | 0.950 | [0.850, 1.000] | 20 |
| file | mrr | 0.554 | [0.404, 0.701] | 20 |
| file | ndcg@10 | 0.646 | [0.516, 0.771] | 20 |
| function | hit@5 | 0.526 | [0.316, 0.737] | 19 |
| function | recall@5 | 0.500 | [0.263, 0.711] | 19 |
| function | hit@10 | 0.526 | [0.316, 0.737] | 19 |
| function | recall@10 | 0.526 | [0.316, 0.737] | 19 |
| function | mrr | 0.335 | [0.187, 0.496] | 19 |
| function | ndcg@10 | 0.374 | [0.203, 0.551] | 19 |

## Latency (retrieval only, seconds per query)

mean 0.0035, median 0.0034, p95 0.0067, max 0.0068 (n = 20)

Per-revision `prepare` (index build): mean 0.1385s (n = 20)

## Metric definitions

- **hit@k**: 1 if at least one relevant item is in the top k (project.md 'Recall@k').
- **recall@k**: Fraction of the case's relevant items found in the top k.
- **mrr**: Reciprocal rank of the first relevant item in the returned list (0 if absent), averaged over cases.
- **ndcg@k**: Binary-relevance nDCG over the top k.
- **file level**: The ranked units are collapsed to distinct files (first occurrence kept); relevant files are the Python files the fix modifies.
- **function level**: Relevant units are the functions/methods the fix modifies, evaluated only on cases that have at least one such unit.
