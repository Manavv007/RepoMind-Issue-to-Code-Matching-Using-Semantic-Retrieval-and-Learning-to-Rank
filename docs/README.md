# RepoMind Documentation

This directory holds project documentation. It will grow with each phase.

Planned documents:

- `evaluation.md` — benchmark decision, labels, leakage controls, metrics, and how
  to reproduce results. **(written in Phase 2)**
- `improvement_findings.md` — what does and does not beat BM25, and why the
  roadmap changed. **(2026-10-22)**
- `gpu_embeddings.md` — one-time Kaggle GPU embedding: export, notebook, verify,
  and loading the artifacts on CPU.
- `architecture.md` — component overview and data flow.
- `api.md` — endpoint contracts (Pydantic schemas, limits, errors).
- `limitations.md` — known failure modes and trade-offs.
- `security.md` — sandboxing, untrusted-input handling, secret scanning.

For now, see the root [`project.md`](../project.md) (full specification) and
[`plan.md`](../plan.md) (phased build plan).
