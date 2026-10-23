"""Dev-selected, test-confirmed BM25 extensions (temp experiment).

Signals added to normalized BM25 (all derived from the issue text and the pool only):
  mention : issue names the unit's symbol (backticks, calls, tracebacks, identifiers)
  path    : issue names the unit's file (path suffix or dotted module)
  file    : file-level evidence = sum of the file's top-3 unit BM25 scores (hierarchical prior)
Selection rule (fixed before running): function MRR on dev, then hit@5, then grid order.
"""

import faulthandler
import itertools
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

faulthandler.dump_traceback_later(280, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
t0 = time.time()

from app.config import get_settings  # noqa: E402
from app.retrieval.base import RetrievalQuery  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.storage.index_store import IndexStore, sqlite_path_from_url  # noqa: E402
from evaluation import metrics as m  # noqa: E402
from evaluation.dataset import read_cases  # noqa: E402
from evaluation.run_retrieval_eval import IndexedUnitsProvider  # noqa: E402

IDENT = [
    re.compile(r"`([A-Za-z_][\w.]*)`"),
    re.compile(r"\b([A-Za-z_]\w*)\s*\("),
    re.compile(r'line \d+, in ([A-Za-z_]\w*)'),
    re.compile(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b"),
    re.compile(r"\b([a-z]+[A-Z][A-Za-z0-9]*)\b"),
    re.compile(r"\b([A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b"),
]
PATH_RE = re.compile(r"([\w./\\-]+\.py)\b")
MODULE_RE = re.compile(r"\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*){2,})\b")
COMMON = {"self", "print", "len", "str", "int", "list", "dict", "set", "type", "range", "open", "isinstance", "super"}


def mentions(text: str):
    names: set[str] = set()
    for rx in IDENT:
        for g in rx.findall(text):
            names.add(g.split(".")[-1])
    names -= COMMON
    paths = {p.replace("\\", "/").lstrip("./") for p in PATH_RE.findall(text)}
    modules = set(MODULE_RE.findall(text))
    return names, paths, modules


def file_mentioned(path: str, paths: set[str], modules: set[str]) -> bool:
    if any(p.endswith(path) or path.endswith(p) for p in paths if "/" in p or p == path.rsplit("/", 1)[-1]):
        return True
    dotted = path[:-3].replace("/", ".")
    return any(dotted.endswith(mod) or mod.startswith(dotted) for mod in modules)


ds = ROOT / "evaluation/datasets/swebench_lite_subset.jsonl"
cases = [c for c in read_cases(ds) if c.suitable and c.ground_truth.units]
settings = get_settings()
provider = IndexedUnitsProvider(settings, IndexStore(sqlite_path_from_url(settings.database_url)))
bm25 = BM25Retriever()

# Precompute per-case signals once (one full BM25 retrieval per case).
prepared = []
for case in sorted(cases, key=lambda c: (c.repo, c.base_commit, c.case_id)):
    pool = provider(case)
    bm25.prepare(pool)
    text = case.issue.body
    res = {c.unit_id: c.score for c in bm25.retrieve(RetrievalQuery(query_id=case.case_id, issue_body=text), len(pool))}
    names, paths, modules = mentions(text)
    top = max(res.values(), default=1.0) or 1.0
    per_file = defaultdict(list)
    for u in pool:
        per_file[u.file_path].append(res.get(u.unit_id, 0.0))
    fsum = {f: sum(sorted(v, reverse=True)[:3]) for f, v in per_file.items()}
    fmax = max(fsum.values(), default=1.0) or 1.0
    fmention = {f: file_mentioned(f, paths, modules) for f in per_file}
    rows = [
        (u.unit_id, res.get(u.unit_id, 0.0) / top, float(u.symbol_name in names), float(fmention[u.file_path]),
         fsum[u.file_path] / fmax)
        for u in pool
    ]
    prepared.append((case, rows))
print(f"prepared {len(prepared)} cases in {time.time() - t0:.0f}s", flush=True)


def score(split: str, bm: float, bp: float, lf: float):
    out = defaultdict(list)
    for case, rows in prepared:
        if case.split != split:
            continue
        ranked = sorted(rows, key=lambda r: (-(r[1] + bm * r[2] + bp * r[3] + lf * r[4]), r[0]))
        ids = [r[0] for r in ranked[:100] if (r[1] + bm * r[2] + bp * r[3] + lf * r[4]) > 0]
        gold = [g.unit_id for g in case.ground_truth.units]
        s = m.score_ranking(ids, gold)
        for k in ("hit@5", "hit@10", "mrr"):
            out[k].append(s[k])
        out["_case"].append((case.case_id, case.repo, s["mrr"], s["hit@5"]))
    return out


def mean(v):
    return sum(v) / len(v)


grid = list(itertools.product([0.0, 0.25, 0.5, 1.0], [0.0, 0.25, 0.5], [0.0, 0.25, 0.5, 1.0]))
results = []
for bm, bp, lf in grid:
    s = score("dev", bm, bp, lf)
    results.append(((bm, bp, lf), mean(s["mrr"]), mean(s["hit@5"]), mean(s["hit@10"])))
base = next(r for r in results if r[0] == (0.0, 0.0, 0.0))
best = sorted(results, key=lambda r: (-round(r[1], 12), -round(r[2], 12)))[0]
print(f"DEV n={len(score('dev', 0, 0, 0)['mrr'])}  baseline mrr={base[1]:.3f} hit@5={base[2]:.3f} hit@10={base[3]:.3f}")
for r in sorted(results, key=lambda r: (-r[1], -r[2]))[:6]:
    print(f"DEV  (mention,path,file)={r[0]}  mrr={r[1]:.3f} hit@5={r[2]:.3f} hit@10={r[3]:.3f}")
for name, idx in (("mention", 0), ("path", 1), ("file", 2)):
    only = [r for r in results if all(r[0][j] == 0 for j in range(3) if j != idx) and r[0][idx] > 0]
    print(f"DEV  single-signal {name}: " + ", ".join(f"{r[0][idx]}->mrr {r[1]:.3f}/h5 {r[2]:.3f}" for r in only))
print(f"CHOSEN on dev: {best[0]}")

# Held-out confirmation: baseline and the dev-chosen config only.
tb, tc = score("test", 0, 0, 0), score("test", *best[0])
print(f"TEST n={len(tb['mrr'])}  baseline mrr={mean(tb['mrr']):.3f} hit@5={mean(tb['hit@5']):.3f} hit@10={mean(tb['hit@10']):.3f}")
print(f"TEST chosen     mrr={mean(tc['mrr']):.3f} hit@5={mean(tc['hit@5']):.3f} hit@10={mean(tc['hit@10']):.3f}")
d_mrr = [b - a for a, b in zip(tb["mrr"], tc["mrr"], strict=True)]
d_h5 = [b - a for a, b in zip(tb["hit@5"], tc["hit@5"], strict=True)]
for label, d in (("mrr", d_mrr), ("hit@5", d_h5)):
    lo, hi = m.bootstrap_ci(d, n_resamples=2000, seed=0)
    w = sum(x > 1e-12 for x in d); lo_ = sum(x < -1e-12 for x in d)
    print(f"TEST paired {label}: {mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]  W/T/L {w}/{len(d) - w - lo_}/{lo_}")
json.dump({"grid": [[list(r[0]), r[1], r[2], r[3]] for r in results], "chosen": list(best[0]),
           "test": {"baseline": {k: mean(tb[k]) for k in ("mrr", "hit@5", "hit@10")},
                    "chosen": {k: mean(tc[k]) for k in ("mrr", "hit@5", "hit@10")}}},
          open(ROOT / "evaluation/reports/exp_bm25_signals.json", "w"), indent=2)
print(f"DONE {time.time() - t0:.0f}s")
