# BM25 hyperparameter tuning (dev split only)

- Dataset: `swebench_lite_subset.jsonl` (sha256 `e8a714baaf88...`)
- Cases: 21 `dev` cases; held-out `test` was **not** read
- Grid: 72 configurations: k1 in [0.9, 1.5], b in [0.5, 0.75, 0.9], w_name in [1.0, 3.0], w_path in [1.0, 2.0], query_tf in ['binary', 'log', 'raw']
- Selection (fixed in advance): function-level MRR, then file-level MRR, then grid order

## Chosen configuration

`{"b": 0.75, "k1": 0.9, "query_tf": "log", "w_name": 3.0, "w_path": 2.0}`

| Config | fn MRR | fn hit@5 | fn hit@10 | fn nDCG@10 | file MRR | file hit@10 |
|---|---|---|---|---|---|---|
| chosen | 0.513 | 0.650 | 0.700 | 0.552 | 0.695 | 0.905 |
| classic (k1=1.2, b=0.75, raw) | 0.465 | 0.700 | 0.750 | 0.532 | 0.628 | 0.857 |

Function-level MRR by repository (dev):

| Repo | chosen | classic |
|---|---|---|
| mwaskom/seaborn | 0.229 | 0.219 |
| psf/requests | 0.468 | 0.483 |
| pylint-dev/astroid | 0.635 | 0.550 |
| pylint-dev/pylint | 0.667 | 0.556 |

## Sensitivity (mean function MRR over configs sharing a value)

- **k1**: 0.9 -> 0.438, 1.5 -> 0.447
- **b**: 0.5 -> 0.428, 0.75 -> 0.452, 0.9 -> 0.448
- **w_name**: 1.0 -> 0.440, 3.0 -> 0.445
- **w_path**: 1.0 -> 0.441, 2.0 -> 0.444
- **query_tf**: binary -> 0.371, log -> 0.478, raw -> 0.478

Function MRR across the grid: best 0.513, median 0.470, worst 0.328.

## Top 10 configurations

| # | params | fn MRR | fn hit@10 | file MRR |
|---|---|---|---|---|
| 1 | `{"b": 0.75, "k1": 0.9, "query_tf": "log", "w_name": 3.0, "w_path": 2.0}` | 0.513 | 0.700 | 0.695 |
| 2 | `{"b": 0.9, "k1": 1.5, "query_tf": "log", "w_name": 3.0, "w_path": 2.0}` | 0.509 | 0.700 | 0.654 |
| 3 | `{"b": 0.9, "k1": 1.5, "query_tf": "raw", "w_name": 3.0, "w_path": 2.0}` | 0.503 | 0.750 | 0.646 |
| 4 | `{"b": 0.75, "k1": 1.5, "query_tf": "log", "w_name": 3.0, "w_path": 2.0}` | 0.503 | 0.700 | 0.656 |
| 5 | `{"b": 0.5, "k1": 0.9, "query_tf": "raw", "w_name": 1.0, "w_path": 1.0}` | 0.503 | 0.700 | 0.686 |
| 6 | `{"b": 0.9, "k1": 0.9, "query_tf": "log", "w_name": 1.0, "w_path": 1.0}` | 0.501 | 0.650 | 0.656 |
| 7 | `{"b": 0.9, "k1": 1.5, "query_tf": "log", "w_name": 3.0, "w_path": 1.0}` | 0.500 | 0.700 | 0.644 |
| 8 | `{"b": 0.75, "k1": 1.5, "query_tf": "log", "w_name": 1.0, "w_path": 2.0}` | 0.500 | 0.650 | 0.654 |
| 9 | `{"b": 0.9, "k1": 1.5, "query_tf": "raw", "w_name": 3.0, "w_path": 1.0}` | 0.497 | 0.750 | 0.645 |
| 10 | `{"b": 0.75, "k1": 0.9, "query_tf": "raw", "w_name": 3.0, "w_path": 2.0}` | 0.495 | 0.750 | 0.651 |
