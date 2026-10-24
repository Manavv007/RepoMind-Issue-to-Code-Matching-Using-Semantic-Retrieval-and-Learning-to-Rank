# Retrieval evaluation: `bm25-mentions`

- Dataset: `swebench_lite_full.jsonl`
- Split: `test`  |  seed: 0  |  depth: 100
- Candidate pool: leaf units (function, method) excluding test files
- Retriever parameters: `{"bm25": {"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}, "w_mention": 0.25, "w_path": 0.5}`
- Cases: 157 evaluated of 157 selected (323 in dataset); excluded: {'unsuitable': 2, 'other_split': 164}
- Generated: 2026-10-24T05:40:41+00:00  |  RepoMind commit: `8479231e60`
- Pool size per case: mean 6032, median 7456 (chance level scales with this)

## Results (mean, 95% bootstrap CI over cases, n = cases scored)

| Level | Metric | Mean | 95% CI | n |
|---|---|---|---|---|
| file | hit@5 | 0.764 | [0.701, 0.828] | 157 |
| file | recall@5 | 0.764 | [0.701, 0.828] | 157 |
| file | acc@5 | 0.764 | [0.701, 0.828] | 157 |
| file | hit@10 | 0.847 | [0.790, 0.904] | 157 |
| file | recall@10 | 0.847 | [0.790, 0.904] | 157 |
| file | acc@10 | 0.847 | [0.790, 0.904] | 157 |
| file | mrr | 0.657 | [0.589, 0.721] | 157 |
| file | ndcg@10 | 0.700 | [0.638, 0.759] | 157 |
| function | hit@5 | 0.594 | [0.510, 0.671] | 143 |
| function | recall@5 | 0.583 | [0.499, 0.661] | 143 |
| function | acc@5 | 0.573 | [0.490, 0.657] | 143 |
| function | hit@10 | 0.671 | [0.587, 0.748] | 143 |
| function | recall@10 | 0.668 | [0.584, 0.745] | 143 |
| function | acc@10 | 0.664 | [0.580, 0.741] | 143 |
| function | mrr | 0.475 | [0.402, 0.547] | 143 |
| function | ndcg@10 | 0.513 | [0.439, 0.580] | 143 |

## Latency (retrieval only, seconds per query)

mean 0.0336, median 0.0344, p95 0.0594, max 0.0751 (n = 157)

Per-revision `prepare` (index build): mean 0.5009s (n = 157)

1 case(s) have a gold file with no candidate units in the pool (e.g. module-level-only code); they stay in the file-level denominator.

## Metric definitions

- **hit@k**: 1 if at least one relevant item is in the top k (project.md 'Recall@k').
- **recall@k**: Fraction of the case's relevant items found in the top k.
- **acc@k**: 1 if all of the case's relevant items are in the top k (Acc@k in Agentless/LocAgent/SweRank).
- **mrr**: Reciprocal rank of the first relevant item in the returned list (0 if absent), averaged over cases.
- **ndcg@k**: Binary-relevance nDCG over the top k.
- **file level**: The ranked units are collapsed to distinct files (first occurrence kept); relevant files are the Python files the fix modifies.
- **function level**: Relevant units are the functions/methods the fix modifies, evaluated only on cases that have at least one such unit.
