# Paired comparison on the `test` split

- A (baseline): `bm25` {"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}
- B (candidate): `bm25-mentions` {"bm25": {"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}, "w_mention": 0.25, "w_path": 0.5}
- Difference = B - A per case; CI is a paired percentile bootstrap over cases.
  `excludes 0` means the interval does not contain zero (not a multiple-comparison-corrected claim).

| Level | Metric | n | Mean diff | 95% CI | W/T/L | Excludes 0 |
|---|---|---|---|---|---|---|
| function | hit@5 | 143 | +0.000 | [-0.028, +0.028] | 2/139/2 | no |
| function | hit@10 | 143 | +0.000 | [-0.028, +0.028] | 2/139/2 | no |
| function | acc@5 | 143 | +0.007 | [-0.021, +0.042] | 3/138/2 | no |
| function | acc@10 | 143 | +0.007 | [-0.021, +0.042] | 3/138/2 | no |
| function | mrr | 143 | +0.007 | [-0.020, +0.036] | 14/106/23 | no |
| function | ndcg@10 | 143 | +0.008 | [-0.014, +0.031] | 13/117/13 | no |
| file | hit@5 | 157 | -0.013 | [-0.032, +0.000] | 0/155/2 | no |
| file | hit@10 | 157 | -0.006 | [-0.019, +0.000] | 0/156/1 | no |
| file | acc@5 | 157 | -0.013 | [-0.032, +0.000] | 0/155/2 | no |
| file | acc@10 | 157 | -0.006 | [-0.019, +0.000] | 0/156/1 | no |
| file | mrr | 157 | +0.015 | [-0.010, +0.041] | 10/129/18 | no |
| file | ndcg@10 | 157 | +0.011 | [-0.008, +0.031] | 10/136/11 | no |

Function-level MRR difference by repository: astropy/astropy -0.073, django/django -0.005, marshmallow-code/marshmallow +0.000, pallets/flask -0.001, pvlib/pvlib-python +0.193, pydicom/pydicom +0.019, pytest-dev/pytest +0.054, sqlfluff/sqlfluff +0.000
