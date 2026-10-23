# Retrieval evaluation: `bm25top30-rrf`

- Dataset: `swebench_lite_subset.jsonl`
- Split: `dev`  |  seed: 0  |  depth: 100
- Candidate pool: leaf units (function, method) excluding test files
- Retriever parameters: `{"bm25": {"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}, "dense_model": "Salesforce/SweRankEmbed-Small", "max_seq_length": 256, "mode": "rrf", "n": 30, "w_dense": 1.0}`
- Cases: 21 evaluated of 21 selected (21 in dataset); excluded: none
- Generated: 2026-10-22T17:04:28+00:00  |  RepoMind commit: `f4e95dff58`
- Pool size per case: mean 1190, median 1302 (chance level scales with this)

## Results (mean, 95% bootstrap CI over cases, n = cases scored)

| Level | Metric | Mean | 95% CI | n |
|---|---|---|---|---|
| file | hit@5 | 0.905 | [0.762, 1.000] | 21 |
| file | recall@5 | 0.905 | [0.762, 1.000] | 21 |
| file | hit@10 | 0.952 | [0.857, 1.000] | 21 |
| file | recall@10 | 0.952 | [0.857, 1.000] | 21 |
| file | mrr | 0.704 | [0.542, 0.855] | 21 |
| file | ndcg@10 | 0.762 | [0.628, 0.890] | 21 |
| function | hit@5 | 0.650 | [0.450, 0.850] | 20 |
| function | recall@5 | 0.650 | [0.450, 0.850] | 20 |
| function | hit@10 | 0.750 | [0.550, 0.900] | 20 |
| function | recall@10 | 0.750 | [0.550, 0.900] | 20 |
| function | mrr | 0.513 | [0.326, 0.702] | 20 |
| function | ndcg@10 | 0.566 | [0.375, 0.734] | 20 |

## Latency (retrieval only, seconds per query)

mean 0.4508, median 0.4608, p95 0.5458, max 0.5886 (n = 21)

Per-revision `prepare` (index build): mean 0.0569s (n = 21)

## Metric definitions

- **hit@k**: 1 if at least one relevant item is in the top k (project.md 'Recall@k').
- **recall@k**: Fraction of the case's relevant items found in the top k.
- **mrr**: Reciprocal rank of the first relevant item in the returned list (0 if absent), averaged over cases.
- **ndcg@k**: Binary-relevance nDCG over the top k.
- **file level**: The ranked units are collapsed to distinct files (first occurrence kept); relevant files are the Python files the fix modifies.
- **function level**: Relevant units are the functions/methods the fix modifies, evaluated only on cases that have at least one such unit.
