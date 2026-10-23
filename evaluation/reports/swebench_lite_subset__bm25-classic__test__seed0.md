# Retrieval evaluation: `bm25`

- Dataset: `swebench_lite_subset.jsonl`
- Split: `test`  |  seed: 0  |  depth: 100
- Candidate pool: leaf units (function, method) excluding test files
- Retriever parameters: `{"b": 0.75, "k1": 1.2, "query_tf": "raw", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 1.0, "w_path": 1.0}`
- Cases: 20 evaluated of 20 selected (41 in dataset); excluded: {'other_split': 21}
- Generated: 2026-10-22T07:31:52+00:00  |  RepoMind commit: `f4e95dff58`
- Pool size per case: mean 1343, median 1440 (chance level scales with this)

## Results (mean, 95% bootstrap CI over cases, n = cases scored)

| Level | Metric | Mean | 95% CI | n |
|---|---|---|---|---|
| file | hit@5 | 0.650 | [0.450, 0.850] | 20 |
| file | recall@5 | 0.650 | [0.450, 0.850] | 20 |
| file | hit@10 | 0.950 | [0.850, 1.000] | 20 |
| file | recall@10 | 0.950 | [0.850, 1.000] | 20 |
| file | mrr | 0.434 | [0.290, 0.604] | 20 |
| file | ndcg@10 | 0.552 | [0.429, 0.686] | 20 |
| function | hit@5 | 0.316 | [0.105, 0.526] | 19 |
| function | recall@5 | 0.316 | [0.105, 0.526] | 19 |
| function | hit@10 | 0.421 | [0.211, 0.632] | 19 |
| function | recall@10 | 0.395 | [0.184, 0.605] | 19 |
| function | mrr | 0.245 | [0.101, 0.406] | 19 |
| function | ndcg@10 | 0.266 | [0.104, 0.434] | 19 |

## Latency (retrieval only, seconds per query)

mean 0.0036, median 0.0035, p95 0.0065, max 0.0069 (n = 20)

Per-revision `prepare` (index build): mean 0.1385s (n = 20)

## Metric definitions

- **hit@k**: 1 if at least one relevant item is in the top k (project.md 'Recall@k').
- **recall@k**: Fraction of the case's relevant items found in the top k.
- **mrr**: Reciprocal rank of the first relevant item in the returned list (0 if absent), averaged over cases.
- **ndcg@k**: Binary-relevance nDCG over the top k.
- **file level**: The ranked units are collapsed to distinct files (first occurrence kept); relevant files are the Python files the fix modifies.
- **function level**: Relevant units are the functions/methods the fix modifies, evaluated only on cases that have at least one such unit.
