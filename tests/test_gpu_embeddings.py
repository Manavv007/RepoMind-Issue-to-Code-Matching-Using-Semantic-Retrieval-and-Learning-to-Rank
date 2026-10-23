"""Tests for the GPU-embedding round trip, without a GPU or a real model.

export (evaluation.gpu_embeddings) -> notebook [engine] cell with a fake encoder
-> artifacts -> EmbeddingStore / DenseRetriever / verify on CPU.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from app.config import PROJECT_ROOT
from app.parsing.symbol_extractor import extract_file
from app.retrieval.base import RetrievalQuery
from app.retrieval.dense import (
    DenseConfig,
    DenseRetriever,
    MissingEmbeddingsError,
    unit_text,
)
from app.retrieval.embedding_store import EmbeddingStore, EmbeddingStoreError
from evaluation import gpu_embeddings as ge
from tests.test_eval_runner import make_case

DIM = 8
PINNED = json.loads((PROJECT_ROOT / "configs" / "dense_swerank_small.json").read_text())


def fake_vec(text: str) -> np.ndarray:
    raw = np.frombuffer(
        hashlib.sha256(text.encode()).digest()[: DIM * 4], dtype=np.uint32
    )
    vec = raw.astype(np.float64) / 2**32 - 0.5
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def fake_encode(texts, batch_size):
    return np.stack([fake_vec(t) for t in texts])


class FakeEncoder:
    """DenseRetriever encoder matching fake_encode (queries get no prefix)."""

    def __init__(self, *_):
        self.doc_calls = 0

    def encode(self, texts, *, is_query):
        if not is_query:
            self.doc_calls += len(texts)
        return [fake_vec(t).tolist() for t in texts]


def pinned_config(**overrides) -> DenseConfig:
    params = {k: v for k, v in PINNED.items() if k != "embeddings_path"}
    return DenseConfig(**{**params, **overrides})


def units_for(n_files: int = 2, per_file: int = 6):
    out = []
    for f in range(n_files):
        src = "\n\n".join(
            f"def fn_{f}_{i}(x):\n    return x + {i}\n" for i in range(per_file)
        )
        out += extract_file(f"pkg/m{f}.py", "python", src.encode()).units
    return sorted(out, key=lambda u: u.unit_id)


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    pool = units_for()
    cases = [
        make_case("a", pool, [0], commit="a" * 40),
        make_case("b", pool[:7], [1], commit="b" * 40),
    ]
    pools = {"a": pool, "b": pool[:7]}
    rows = ge.build_export(cases, lambda c: pools[c.case_id], include_path=True)
    dataset = tmp_path / "d.jsonl"
    dataset.write_text("x")
    manifest = ge.export_manifest(
        rows, dataset=dataset, splits=["dev", "test"], n_cases=2, config=pinned_config()
    )
    out = tmp_path / "export"
    ge.write_export(out, rows, manifest)
    return out


def notebook_namespace(export_dir: Path, out_dir: Path, monkeypatch) -> dict:
    """Execute the committed notebook's [config] and [engine] cells."""
    nb = json.loads(
        (PROJECT_ROOT / "notebooks" / "kaggle_swerank_embeddings.ipynb").read_text(
            encoding="utf-8"
        )
    )
    sources = {
        "".join(c["source"]).split("\n", 1)[0]: "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code"
    }
    monkeypatch.setenv("RM_EXPORT_DIR", str(export_dir))
    monkeypatch.setenv("RM_OUTPUT_DIR", str(out_dir))
    ns: dict = {}
    for tag in ("# [config]", "# [engine]"):
        source = next(src for first, src in sources.items() if first.startswith(tag))
        exec(compile(source, tag, "exec"), ns)  # noqa: S102 - our own notebook code
    return ns


def run_notebook_engine(ns: dict, *, encode=fake_encode, shard_rows: int = 4) -> dict:
    manifest, rows = ns["read_export"](ns["EXPORT_DIR"])
    shards = ns["OUTPUT_DIR"] / "shards"
    settings = {
        "export_content_sha256": manifest["content_sha256"],
        "shard_rows": shard_rows,
    }
    ns["prepare_shards_dir"](shards, settings)
    stats = ns["run_shards"](
        rows,
        shards,
        encode,
        limit=len(rows),
        shard_rows=shard_rows,
        batch_size=4,
        np=np,
        dim=DIM,
        log=lambda *_: None,
    )
    model_info = {
        "name": ns["MODEL_NAME"],
        "revision": ns["MODEL_REVISION"],
        "code_revision": ns["CODE_REVISION"],
        "weights_sha256": ns["WEIGHTS_SHA256"],
        "query_prefix": ns["QUERY_PREFIX"],
        "dim": DIM,
        "max_seq_length": 64,
    }
    ns["merge_shards"](
        rows,
        shards,
        ns["OUTPUT_DIR"],
        limit=len(rows),
        shard_rows=shard_rows,
        export_manifest=manifest,
        model_info=model_info,
        runtime={"device": "test"},
        np=np,
    )
    return stats


# ------------------------------------------------------------------ export
def test_export_dedups_units_across_revisions_and_is_deterministic(
    export_dir, tmp_path
):
    manifest, rows = ge.read_export(export_dir)
    assert (
        manifest["rows"] == len(rows) == 12
    )  # case b's 7 units are a subset of a's 12
    assert [r["row"] for r in rows] == list(range(12))
    assert rows == sorted(rows, key=lambda r: r["key"])
    assert manifest["model"]["revision"] == PINNED["revision"]
    assert manifest["unit_text_version"] == "path-name-source/v1"
    first = (export_dir / ge.EXPORT_UNITS_FILE).read_bytes()
    ge.write_export(tmp_path / "again", rows, manifest)
    assert (tmp_path / "again" / ge.EXPORT_UNITS_FILE).read_bytes() == first


def test_export_text_is_the_dense_retriever_text(export_dir):
    _, rows = ge.read_export(export_dir)
    by_key = {tuple(r["key"]): r["text"] for r in rows}
    for unit in units_for():
        assert by_key[
            (unit.file_path, unit.qualified_name, unit.content_hash)
        ] == unit_text(unit, True)


def test_tampered_export_is_rejected(export_dir):
    manifest = json.loads((export_dir / ge.EXPORT_MANIFEST_FILE).read_text())
    manifest["content_sha256"] = "0" * 64
    (export_dir / ge.EXPORT_MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(EmbeddingStoreError):
        ge.read_export(export_dir)


# ----------------------------------------------------- notebook engine round trip
def test_notebook_hash_function_matches_repomind(export_dir, tmp_path, monkeypatch):
    ns = notebook_namespace(export_dir, tmp_path / "out", monkeypatch)
    manifest, rows = ns["read_export"](export_dir)
    assert (
        ns["content_sha256"](rows)
        == ge.content_sha256(rows)
        == manifest["content_sha256"]
    )
    assert ns["FORMAT_VERSION"] == "repomind-embeddings/v1"
    for field, key in (
        ("MODEL_NAME", "model_name"),
        ("MODEL_REVISION", "revision"),
        ("CODE_REVISION", "code_revision"),
        ("WEIGHTS_SHA256", "weights_sha256"),
        ("QUERY_PREFIX", "query_prefix"),
    ):
        assert (
            ns[field] == PINNED[key]
        ), field  # notebook pins == configs/dense_swerank_small.json


def test_full_round_trip_loads_into_dense_retriever(export_dir, tmp_path, monkeypatch):
    out = tmp_path / "out"
    ns = notebook_namespace(export_dir, out, monkeypatch)
    stats = run_notebook_engine(ns)
    assert stats["computed"] == 3  # 12 rows / 4 per shard

    report = ge.verify(
        out, export_dir=export_dir, config=pinned_config(), encoder_factory=FakeEncoder
    )
    assert report["ok"], report["problems"]
    assert report["checks"] == {"format": "ok", "alignment": "ok", "config": "ok"}

    store = EmbeddingStore.load(out)
    assert store.matrix.shape == (12, DIM) and not store.manifest["partial"]

    encoder = FakeEncoder()
    retriever = DenseRetriever(
        pinned_config(embeddings_path=str(out), on_missing="error"), encoder=encoder
    )
    pool = units_for()
    retriever.prepare(pool)
    assert encoder.doc_calls == 0  # every unit came from the artifacts
    gold = pool[5]
    top = retriever.retrieve(
        RetrievalQuery(query_id="q", issue_body=unit_text(gold, True)), 3
    )
    assert top[0].unit_id == gold.unit_id and top[0].score == pytest.approx(
        1.0, abs=1e-5
    )


def test_interrupted_run_resumes_without_recomputing(export_dir, tmp_path, monkeypatch):
    ns = notebook_namespace(export_dir, tmp_path / "out", monkeypatch)
    calls = {"n": 0}

    def flaky(texts, batch_size):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("session lost")
        return fake_encode(texts, batch_size)

    with pytest.raises(KeyboardInterrupt):
        run_notebook_engine(ns, encode=flaky)
    shards = sorted(p.name for p in (tmp_path / "out" / "shards").glob("*.npy"))
    assert shards == ["00000000-00000004.npy"]  # completed shard kept, no partial file
    assert run_notebook_engine(ns)["computed"] == 2


def test_resume_imports_matching_shards_and_skips_corrupt_ones(
    export_dir, tmp_path, monkeypatch
):
    ns = notebook_namespace(export_dir, tmp_path / "first", monkeypatch)
    run_notebook_engine(ns)
    previous = tmp_path / "first" / "shards"
    (previous / "00000004-00000008.partial.npy").write_bytes(b"junk")
    manifest, rows = ns["read_export"](export_dir)
    settings = {"export_content_sha256": manifest["content_sha256"], "shard_rows": 4}
    fresh = tmp_path / "second" / "shards"
    assert ns["prepare_shards_dir"](fresh, settings, [previous]) == 3
    (fresh / "00000008-00000012.npy").write_bytes(b"truncated")
    stats = ns["run_shards"](
        rows,
        fresh,
        fake_encode,
        limit=12,
        shard_rows=4,
        batch_size=4,
        np=np,
        dim=DIM,
        log=lambda *_: None,
    )
    assert stats["computed"] == 1  # only the corrupt shard is redone
    with pytest.raises(ValueError, match="different settings"):
        ns["prepare_shards_dir"](fresh, {**settings, "shard_rows": 8})


def test_cuda_oom_halves_batch_size(export_dir, tmp_path, monkeypatch):
    ns = notebook_namespace(export_dir, tmp_path / "out", monkeypatch)

    class OutOfMemoryError(RuntimeError):
        pass

    seen = []

    def picky(texts, batch_size):
        seen.append(batch_size)
        if batch_size > 1:
            raise OutOfMemoryError("CUDA out of memory")
        return fake_encode(texts, batch_size)

    vecs, final = ns["encode_with_backoff"](picky, ["a", "b"], 4, log=lambda *_: None)
    assert seen == [4, 2, 1] and final == 1 and vecs.shape == (2, DIM)
    with pytest.raises(ValueError):
        ns["encode_with_backoff"](
            lambda t, b: (_ for _ in ()).throw(ValueError("bug")), ["a"], 4
        )


def test_smoke_subset_is_partial_and_flagged(export_dir, tmp_path, monkeypatch):
    out = tmp_path / "out"
    ns = notebook_namespace(export_dir, out, monkeypatch)
    manifest, rows = ns["read_export"](export_dir)
    shards = out / "shards"
    ns["prepare_shards_dir"](shards, {"x": 1})
    ns["run_shards"](
        rows,
        shards,
        fake_encode,
        limit=5,
        shard_rows=4,
        batch_size=4,
        np=np,
        dim=DIM,
        log=lambda *_: None,
    )
    info = {
        "name": ns["MODEL_NAME"],
        "revision": ns["MODEL_REVISION"],
        "code_revision": ns["CODE_REVISION"],
        "weights_sha256": ns["WEIGHTS_SHA256"],
        "query_prefix": ns["QUERY_PREFIX"],
        "dim": DIM,
        "max_seq_length": 64,
    }
    ns["merge_shards"](
        rows,
        shards,
        out,
        limit=5,
        shard_rows=4,
        export_manifest=manifest,
        model_info=info,
        runtime={},
        np=np,
    )
    report = ge.verify(out, export_dir=export_dir, config=pinned_config())
    assert report["ok"] and report["partial"] and report["rows"] == 5
    with pytest.raises(MissingEmbeddingsError, match="have no precomputed"):
        DenseRetriever(
            pinned_config(embeddings_path=str(out), on_missing="error"),
            encoder=FakeEncoder(),
        ).prepare(units_for())
    encoder = FakeEncoder()
    DenseRetriever(
        pinned_config(embeddings_path=str(out), on_missing="encode"), encoder=encoder
    ).prepare(units_for())
    assert encoder.doc_calls == 7  # on_missing="encode" fills only the gaps


# ------------------------------------------------------------- validation
def _artifacts(export_dir, tmp_path, monkeypatch) -> Path:
    out = tmp_path / "out"
    run_notebook_engine(notebook_namespace(export_dir, out, monkeypatch))
    return out


def test_incompatible_config_is_rejected(export_dir, tmp_path, monkeypatch):
    out = _artifacts(export_dir, tmp_path, monkeypatch)
    for change in (
        {"revision": "0" * 40},
        {"query_prefix": ""},
        {"include_path": False},
    ):
        with pytest.raises(EmbeddingStoreError, match="do not match"):
            DenseRetriever(
                pinned_config(embeddings_path=str(out), **change), encoder=FakeEncoder()
            )


@pytest.mark.parametrize(
    "damage", ["swap_ids", "drop_id", "denormalize", "manifest_rows", "nan"]
)
def test_damaged_artifacts_are_rejected(damage, export_dir, tmp_path, monkeypatch):
    out = _artifacts(export_dir, tmp_path, monkeypatch)
    ids = (out / "ids.jsonl").read_text().splitlines()
    emb = np.load(out / "embeddings.npy")
    if damage == "swap_ids":
        ids[0], ids[1] = ids[1], ids[0]
        (out / "ids.jsonl").write_text("\n".join(ids) + "\n")
    elif damage == "drop_id":
        (out / "ids.jsonl").write_text("\n".join(ids[:-1]) + "\n")
    elif damage == "denormalize":
        np.save(out / "embeddings.npy", emb * 2)
    elif damage == "nan":
        emb[3, 0] = np.nan
        np.save(out / "embeddings.npy", emb)
    else:
        manifest = json.loads((out / "manifest.json").read_text())
        manifest["rows"] = 99
        (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(EmbeddingStoreError):
        EmbeddingStore.load(out)


def test_artifacts_from_another_export_fail_alignment(
    export_dir, tmp_path, monkeypatch
):
    out = _artifacts(export_dir, tmp_path, monkeypatch)
    manifest, rows = ge.read_export(export_dir)
    rows[0]["text_sha256"] = "f" * 64
    other = {**manifest, "content_sha256": ge.content_sha256(rows)}
    problems = ge.verify_alignment(EmbeddingStore.load(out), (other, rows))
    assert any("different export" in p for p in problems) and any(
        "misaligned" in p for p in problems
    )


def test_spot_check_uses_cpu_encoder_and_detects_disagreement(
    export_dir, tmp_path, monkeypatch
):
    out = _artifacts(export_dir, tmp_path, monkeypatch)
    good = ge.verify(
        out,
        export_dir=export_dir,
        config=pinned_config(),
        spot=4,
        encoder_factory=FakeEncoder,
    )
    assert good["ok"] and good["spot_check"]["min_cosine"] == pytest.approx(
        1.0, abs=1e-5
    )

    class Wrong(FakeEncoder):
        def encode(self, texts, *, is_query):
            return [fake_vec("other" + t).tolist() for t in texts]

    bad = ge.verify(
        out,
        export_dir=export_dir,
        config=pinned_config(),
        spot=4,
        encoder_factory=Wrong,
    )
    assert not bad["ok"] and bad["checks"]["spot_check"] == "failed"


def test_verify_cli_exit_codes(export_dir, tmp_path, monkeypatch, capsys):
    out = _artifacts(export_dir, tmp_path, monkeypatch)
    cfg = str(PROJECT_ROOT / "configs" / "dense_swerank_small.json")
    capsys.readouterr()  # discard output printed by the notebook cells
    assert (
        ge.main(
            [
                "verify",
                "--artifacts",
                str(out),
                "--export",
                str(export_dir),
                "--dense-config",
                cfg,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["ok"] is True
    (out / "ids.jsonl").write_text("")
    assert ge.main(["verify", "--artifacts", str(out)]) == 1


def test_committed_notebook_matches_its_generator():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "nbgen", PROJECT_ROOT / "notebooks" / "build_kaggle_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    committed = (
        PROJECT_ROOT / "notebooks" / "kaggle_swerank_embeddings.ipynb"
    ).read_text(encoding="utf-8")
    assert (
        committed == module.render()
    ), "run: python notebooks/build_kaggle_notebook.py"
    for cell in module.build()["cells"]:
        if cell["cell_type"] == "code":
            compile(
                "".join(cell["source"]), "cell", "exec"
            )  # every code cell is valid Python
    assert (
        os.environ.get("RM_EXPORT_DIR") is None
    )  # tests must not leak notebook env vars
