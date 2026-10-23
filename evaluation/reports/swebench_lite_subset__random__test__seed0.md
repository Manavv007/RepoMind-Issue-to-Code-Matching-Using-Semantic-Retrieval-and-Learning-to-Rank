# Retrieval evaluation: `random`

- Dataset: `swebench_lite_subset.jsonl`
- Split: `test`  |  seed: 0  |  depth: 100
- Candidate pool: leaf units (function, method) excluding test files
- Cases: 20 evaluated of 20 selected (41 in dataset); excluded: {'other_split': 21}
- Generated: 2026-10-22T07:12:20+00:00  |  RepoMind commit: `f4e95dff58`
- Pool size per case: mean 1343, median 1440 (chance level scales with this)

## Results (mean, 95% bootstrap CI over cases, n = cases scored)

| Level | Metric | Mean | 95% CI | n |
|---|---|---|---|---|
| file | hit@5 | 0.150 | [0.000, 0.300] | 20 |
| file | recall@5 | 0.150 | [0.000, 0.300] | 20 |
| file | hit@10 | 0.300 | [0.100, 0.500] | 20 |
| file | recall@10 | 0.300 | [0.100, 0.500] | 20 |
| file | mrr | 0.083 | [0.057, 0.113] | 20 |
| file | ndcg@10 | 0.108 | [0.038, 0.183] | 20 |
| function | hit@5 | 0.000 | [0.000, 0.000] | 19 |
| function | recall@5 | 0.000 | [0.000, 0.000] | 19 |
| function | hit@10 | 0.000 | [0.000, 0.000] | 19 |
| function | recall@10 | 0.000 | [0.000, 0.000] | 19 |
| function | mrr | 0.003 | [0.000, 0.008] | 19 |
| function | ndcg@10 | 0.000 | [0.000, 0.000] | 19 |

## Latency (retrieval only, seconds per query)

mean 0.0006, median 0.0005, p95 0.0012, max 0.0013 (n = 20)

Per-revision `prepare` (index build): mean 0.0026s (n = 20)

## Metric definitions

- **hit@k**: 1 if at least one relevant item is in the top k (project.md 'Recall@k').
- **recall@k**: Fraction of the case's relevant items found in the top k.
- **mrr**: Reciprocal rank of the first relevant item in the returned list (0 if absent), averaged over cases.
- **ndcg@k**: Binary-relevance nDCG over the top k.
- **file level**: The ranked units are collapsed to distinct files (first occurrence kept); relevant files are the Python files the fix modifies.
- **function level**: Relevant units are the functions/methods the fix modifies, evaluated only on cases that have at least one such unit.
