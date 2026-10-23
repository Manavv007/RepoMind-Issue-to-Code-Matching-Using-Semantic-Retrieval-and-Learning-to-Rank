"""Dev-split: BM25 top-N candidates re-scored by SweRankEmbed-Small (temp experiment).

Modes evaluated on the same candidates:
  rerank-dense : top-N reordered by dense similarity alone
  rerank-rrf   : top-N reordered by RRF(BM25 rank, dense rank)
Units beyond N keep their BM25 order, so recall@100 equals BM25's.
"""

import faulthandler
import json
import sys
import time
from pathlib import Path

CAP = 1800
ENCODE_BUDGET = 1200
faulthandler.dump_traceback_later(CAP - 30, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


import numpy as np  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.retrieval.base import RetrievalQuery, Retriever, make_candidate  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.dense import DenseConfig, SentenceTransformerEncoder, rrf_fuse, unit_text  # noqa: E402
from app.storage.index_store import IndexStore, sqlite_path_from_url  # noqa: E402
from evaluation.dataset import file_sha256, read_cases  # noqa: E402
from evaluation.run_retrieval_eval import IndexedUnitsProvider, render_markdown, run_evaluation  # noqa: E402

CACHE_DIR = ROOT / "data" / "emb_cache"
KEYS_F, VECS_F = CACHE_DIR / "swerank-small-256_keys.json", CACHE_DIR / "swerank-small-256_vecs.npy"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def key(u):
    return (u.file_path, u.qualified_name, u.content_hash)


cache: dict = {}
if KEYS_F.exists():
    cache = {tuple(k): v for k, v in zip(json.loads(KEYS_F.read_text()), np.load(VECS_F), strict=True)}
log(f"loaded {len(cache)} cached embeddings")


def save() -> None:
    ks = list(cache)
    KEYS_F.write_text(json.dumps([list(k) for k in ks]))
    np.save(VECS_F, np.asarray([cache[k] for k in ks], dtype=np.float32))


config = DenseConfig(
    model_name="Salesforce/SweRankEmbed-Small",
    trust_remote_code=True,
    query_prefix="Represent this query for searching relevant code: ",
    max_seq_length=256,
    batch_size=8,
    load_timeout_seconds=300,
)
encoder = SentenceTransformerEncoder(config)
log("model loaded")


class TopNRerank(Retriever):
    def __init__(self, n: int, mode: str, w_dense: float = 1.0) -> None:
        self.n, self.mode, self.w_dense = n, mode, w_dense
        self.name = f"bm25top{n}-{mode}"
        self.bm25 = BM25Retriever()
        self._by_id: dict = {}

    def describe(self):
        return {"n": self.n, "mode": self.mode, "w_dense": self.w_dense, "dense_model": config.model_name,
                "max_seq_length": config.max_seq_length, "bm25": self.bm25.describe()}

    def prepare(self, units):
        self.bm25.prepare(units)
        self._by_id = {u.unit_id: u for u in units}

    def retrieve(self, query: RetrievalQuery, k: int):
        cands = self.bm25.retrieve(query, max(self.n, k))
        top = cands[: self.n]
        missing = [self._by_id[c.unit_id] for c in top if key(self._by_id[c.unit_id]) not in cache]
        if missing:  # should not happen after pre-encoding
            for u, v in zip(missing, encoder.encode([unit_text(u) for u in missing], is_query=False), strict=True):
                cache[key(u)] = np.asarray(v, dtype=np.float32)
        q = np.asarray(encoder.encode([query.text], is_query=True)[0], dtype=np.float32)
        dense = {c.unit_id: float(np.dot(q, cache[key(self._by_id[c.unit_id])])) for c in top}
        dense_order = sorted(dense, key=lambda i: (-dense[i], i))
        if self.mode == "dense":
            order = [(i, dense[i]) for i in dense_order]
        else:
            order = rrf_fuse([(1.0, [c.unit_id for c in top]), (self.w_dense, dense_order)], 60)
        bm = {c.unit_id: c for c in cands}
        out = []
        for rank, (uid, score) in enumerate(order, start=1):
            out.append(make_candidate(self._by_id[uid], rank=rank, score=score, method=self.name,
                                      signals={"bm25": bm[uid].score, "bm25_rank": float(bm[uid].rank), "dense": dense[uid]}))
        for c in cands[self.n:]:
            out.append(make_candidate(self._by_id[c.unit_id], rank=len(out) + 1, score=-float(c.rank),
                                      method=self.name, signals={"bm25": c.score, "bm25_rank": float(c.rank)}))
        return out[:k]


ds = ROOT / "evaluation/datasets/swebench_lite_subset.jsonl"
cases = [c for c in read_cases(ds) if c.suitable and c.split == "dev"]
settings = get_settings()
provider = IndexedUnitsProvider(settings, IndexStore(sqlite_path_from_url(settings.database_url)))

# Candidate units per N, from BM25 (same retriever config as evaluation).
need: dict[int, dict] = {30: {}}
bm = BM25Retriever()
for case in cases:
    pool = provider(case)
    bm.prepare(pool)
    by_id = {u.unit_id: u for u in pool}
    res = bm.retrieve(RetrievalQuery(query_id=case.case_id, issue_body=case.issue.body), 30)
    for n in need:
        for c in res[:n]:
            u = by_id[c.unit_id]
            need[n].setdefault(key(u), u)

# Warm throughput measurement.
sample = [u for k, u in need[30].items() if k not in cache][:40]
encoder.encode([unit_text(u) for u in sample[:8]], is_query=False)
t = time.time()
vecs = encoder.encode([unit_text(u) for u in sample[8:]], is_query=False)
rate = max(len(sample) - 8, 1) / (time.time() - t)
for u, v in zip(sample[8:], vecs, strict=True):
    cache[key(u)] = np.asarray(v, dtype=np.float32)
chosen = None
for n in (30,):
    todo = [k for k in need[n] if k not in cache]
    projected = len(todo) / rate
    log(f"N={n}: {len(need[n])} candidate units, {len(todo)} to encode, warm rate {rate:.2f}/s, projected {projected:.0f}s")
    if projected <= ENCODE_BUDGET:
        chosen = n
        break
if chosen is None:
    save()
    log("ABORT: even N=50 exceeds the encode budget")
    sys.exit(4)

todo = [k for k in need[chosen] if k not in cache]
for i in range(0, len(todo), 64):
    chunk = todo[i : i + 64]
    for k, v in zip(chunk, encoder.encode([unit_text(need[chosen][k]) for k in chunk], is_query=False), strict=True):
        cache[k] = np.asarray(v, dtype=np.float32)
    save()
    log(f"encoded {min(i + 64, len(todo))}/{len(todo)}")
save()

out = ROOT / "evaluation/reports"
for retriever in (TopNRerank(chosen, "dense"), TopNRerank(chosen, "rrf"), TopNRerank(chosen, "rrf", 2.0)):
    start = time.time()
    report = run_evaluation(cases, retriever, provider, split="dev", n_resamples=1000)
    report["dataset"] = {"path": ds.name, "sha256": file_sha256(ds)}
    tag = retriever.name + ("" if retriever.w_dense == 1.0 else f"-wd{retriever.w_dense:g}")
    stem = f"{ds.stem}__{tag}-swerank__dev__seed0"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / f"{stem}.md").write_text(render_markdown(report, ds.name), encoding="utf-8")
    fn, fl = report["aggregate"]["function"]["metrics"], report["aggregate"]["file"]["metrics"]
    log(f"RESULT {tag}: evaluated={report['counts']['evaluated']} errors={report['counts']['excluded'].get('error', 0)} "
        f"fn hit@5={fn['hit@5']['mean']:.3f} hit@10={fn['hit@10']['mean']:.3f} mrr={fn['mrr']['mean']:.3f} "
        f"file hit@10={fl['hit@10']['mean']:.3f} ({time.time() - start:.0f}s)")
log("DONE")

