# Retrieval evaluation: `bm25`

- Dataset: `swebench_lite_subset.jsonl`
- Split: `dev`  |  seed: 0  |  depth: 100
- Candidate pool: leaf units (function, method) excluding test files
- Retriever parameters: `{"b": 0.75, "k1": 1.2, "query_tf": "raw", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 1.0, "w_path": 1.0}`
- Cases: 21 evaluated of 21 selected (41 in dataset); excluded: {'other_split': 20}
- Generated: 2026-10-22T07:31:38+00:00  |  RepoMind commit: `f4e95dff58`
- Pool size per case: mean 1190, median 1302 (chance level scales with this)

## Results (mean, 95% bootstrap CI over cases, n = cases scored)

| Level | Metric | Mean | 95% CI | n |
|---|---|---|---|---|
| file | hit@5 | 0.857 | [0.667, 1.000] | 21 |
| file | recall@5 | 0.857 | [0.667, 1.000] | 21 |
| file | hit@10 | 0.857 | [0.667, 1.000] | 21 |
| file | recall@10 | 0.857 | [0.667, 1.000] | 21 |
| file | mrr | 0.628 | [0.458, 0.786] | 21 |
| file | ndcg@10 | 0.680 | [0.509, 0.825] | 21 |
| function | hit@5 | 0.700 | [0.500, 0.900] | 20 |
| function | recall@5 | 0.700 | [0.500, 0.900] | 20 |
| function | hit@10 | 0.750 | [0.550, 0.900] | 20 |
| function | recall@10 | 0.750 | [0.550, 0.900] | 20 |
| function | mrr | 0.465 | [0.284, 0.653] | 20 |
| function | ndcg@10 | 0.532 | [0.352, 0.704] | 20 |

## Latency (retrieval only, seconds per query)

mean 0.0036, median 0.0029, p95 0.0062, max 0.0143 (n = 21)

Per-revision `prepare` (index build): mean 0.1672s (n = 21)

## Metric definitions

- **hit@k**: 1 if at least one relevant item is in the top k (project.md 'Recall@k').
- **recall@k**: Fraction of the case's relevant items found in the top k.
- **mrr**: Reciprocal rank of the first relevant item in the returned list (0 if absent), averaged over cases.
- **ndcg@k**: Binary-relevance nDCG over the top k.
- **file level**: The ranked units are collapsed to distinct files (first occurrence kept); relevant files are the Python files the fix modifies.
- **function level**: Relevant units are the functions/methods the fix modifies, evaluated only on cases that have at least one such unit.
