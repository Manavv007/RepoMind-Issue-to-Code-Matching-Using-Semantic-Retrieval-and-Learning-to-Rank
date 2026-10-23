"""Dev-split dense + hybrid evaluation sharing one embedding cache (temp script)."""

import faulthandler
import json
import sys
import time
from pathlib import Path

faulthandler.dump_traceback_later(840, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
t0 = time.time()

from app.config import get_settings  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.dense import DenseConfig, DenseRetriever, HybridRetriever  # noqa: E402
from app.storage.index_store import IndexStore, sqlite_path_from_url  # noqa: E402
from evaluation.dataset import file_sha256, read_cases  # noqa: E402
from evaluation.run_retrieval_eval import (  # noqa: E402
    IndexedUnitsProvider,
    render_markdown,
    run_evaluation,
)

ds = ROOT / "evaluation/datasets/swebench_lite_subset.jsonl"
cases = read_cases(ds)
settings = get_settings()
provider = IndexedUnitsProvider(settings, IndexStore(sqlite_path_from_url(settings.database_url)))
dense = DenseRetriever(DenseConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
print(f"STAGE model ready {time.time() - t0:.1f}s", flush=True)


def progress(case):
    pool = provider(case)
    print(f"  {case.case_id} pool={len(pool)} t={time.time() - t0:.0f}s", flush=True)
    return pool


out = ROOT / "evaluation/reports"
for name, retriever in (("dense-minilm", dense), ("hybrid-minilm", HybridRetriever(BM25Retriever(), dense))):
    start = time.time()
    report = run_evaluation(cases, retriever, progress, split="dev", n_resamples=1000)
    report["dataset"] = {"path": ds.name, "sha256": file_sha256(ds)}
    stem = f"{ds.stem}__{name}__dev__seed0"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / f"{stem}.md").write_text(render_markdown(report, ds.name), encoding="utf-8")
    fn = report["aggregate"]["function"]["metrics"]
    print(
        f"RESULT {name}: evaluated={report['counts']['evaluated']} excluded={report['counts']['excluded']} "
        f"fn hit@5={fn['hit@5']['mean']:.3f} hit@10={fn['hit@10']['mean']:.3f} mrr={fn['mrr']['mean']:.3f} "
        f"({time.time() - start:.0f}s)",
        flush=True,
    )
print(f"STAGE done {time.time() - t0:.1f}s", flush=True)
