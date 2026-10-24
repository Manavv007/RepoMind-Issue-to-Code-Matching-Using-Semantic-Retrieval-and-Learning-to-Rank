# Retrieval evaluation: `bm25`

- Dataset: `swebench_lite_full.jsonl`
- Split: `dev`  |  seed: 0  |  depth: 100
- Candidate pool: leaf units (function, method) excluding test files
- Retriever parameters: `{"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}`
- Cases: 164 evaluated of 164 selected (323 in dataset); excluded: {'other_split': 157, 'unsuitable': 2}
- Generated: 2026-10-24T04:47:57+00:00  |  RepoMind commit: `8479231e60`
- Pool size per case: mean 11628, median 6468 (chance level scales with this)

## Results (mean, 95% bootstrap CI over cases, n = cases scored)

| Level | Metric | Mean | 95% CI | n |
|---|---|---|---|---|
| file | hit@5 | 0.750 | [0.683, 0.811] | 164 |
| file | recall@5 | 0.750 | [0.683, 0.811] | 164 |
| file | acc@5 | 0.750 | [0.683, 0.811] | 164 |
| file | hit@10 | 0.805 | [0.738, 0.866] | 164 |
| file | recall@10 | 0.805 | [0.738, 0.866] | 164 |
| file | acc@10 | 0.805 | [0.738, 0.866] | 164 |
| file | mrr | 0.619 | [0.555, 0.678] | 164 |
| file | ndcg@10 | 0.660 | [0.597, 0.717] | 164 |
| function | hit@5 | 0.520 | [0.440, 0.600] | 150 |
| function | recall@5 | 0.489 | [0.409, 0.565] | 150 |
| function | acc@5 | 0.460 | [0.380, 0.533] | 150 |
| function | hit@10 | 0.587 | [0.500, 0.660] | 150 |
| function | recall@10 | 0.563 | [0.479, 0.638] | 150 |
| function | acc@10 | 0.540 | [0.453, 0.620] | 150 |
| function | mrr | 0.355 | [0.294, 0.419] | 150 |
| function | ndcg@10 | 0.393 | [0.329, 0.457] | 150 |

## Latency (retrieval only, seconds per query)

mean 0.0245, median 0.0177, p95 0.0636, max 0.0947 (n = 164)

Per-revision `prepare` (index build): mean 1.0862s (n = 161)

## Metric definitions

- **hit@k**: 1 if at least one relevant item is in the top k (project.md 'Recall@k').
- **recall@k**: Fraction of the case's relevant items found in the top k.
- **acc@k**: 1 if all of the case's relevant items are in the top k (Acc@k in Agentless/LocAgent/SweRank).
- **mrr**: Reciprocal rank of the first relevant item in the returned list (0 if absent), averaged over cases.
- **ndcg@k**: Binary-relevance nDCG over the top k.
- **file level**: The ranked units are collapsed to distinct files (first occurrence kept); relevant files are the Python files the fix modifies.
- **function level**: Relevant units are the functions/methods the fix modifies, evaluated only on cases that have at least one such unit.
