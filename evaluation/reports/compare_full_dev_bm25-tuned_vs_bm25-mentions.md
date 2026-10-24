# Paired comparison on the `dev` split

- A (baseline): `bm25` {"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}
- B (candidate): `bm25-mentions` {"bm25": {"b": 0.75, "k1": 0.9, "query_tf": "log", "use_stopwords": true, "w_body": 1.0, "w_doc": 1.0, "w_name": 3.0, "w_path": 2.0}, "w_mention": 0.25, "w_path": 0.5}
- Difference = B - A per case; CI is a paired percentile bootstrap over cases.
  `excludes 0` means the interval does not contain zero (not a multiple-comparison-corrected claim).

| Level | Metric | n | Mean diff | 95% CI | W/T/L | Excludes 0 |
|---|---|---|---|---|---|---|
| function | hit@5 | 150 | +0.013 | [-0.033, +0.067] | 8/136/6 | no |
| function | hit@10 | 150 | +0.007 | [-0.027, +0.040] | 4/143/3 | no |
| function | acc@5 | 150 | +0.000 | [-0.047, +0.053] | 7/136/7 | no |
| function | acc@10 | 150 | +0.000 | [-0.040, +0.040] | 5/140/5 | no |
| function | mrr | 150 | +0.030 | [+0.005, +0.058] | 26/102/22 | yes |
| function | ndcg@10 | 150 | +0.021 | [-0.003, +0.046] | 23/110/17 | no |
| file | hit@5 | 164 | +0.037 | [+0.006, +0.073] | 7/156/1 | yes |
| file | hit@10 | 164 | +0.024 | [-0.006, +0.061] | 6/156/2 | no |
| file | acc@5 | 164 | +0.037 | [+0.006, +0.073] | 7/156/1 | yes |
| file | acc@10 | 164 | +0.024 | [-0.006, +0.061] | 6/156/2 | no |
| file | mrr | 164 | +0.037 | [+0.009, +0.064] | 31/113/20 | yes |
| file | ndcg@10 | 164 | +0.036 | [+0.012, +0.061] | 28/126/10 | yes |

Function-level MRR difference by repository: matplotlib/matplotlib -0.009, mwaskom/seaborn +0.042, psf/requests +0.222, pydata/xarray +0.017, pylint-dev/astroid +0.125, pylint-dev/pylint -0.083, pyvista/pyvista +0.000, scikit-learn/scikit-learn +0.032, sphinx-doc/sphinx +0.006, sympy/sympy +0.035
