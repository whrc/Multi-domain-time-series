# CLAUDE.md — Multi-Domain Time Series Prediction

## Project
Monthly time-series models for **Arctic**, **Amazon**, **Rangeland**: first one dedicated model per domain, then one shared **Multi-Domain** model testing whether pooling improves on them. Every model is a **causal same-step emulator** (inputs up to *t* → target at *t*, no forecasting), evaluated by **spatial generalization** (held-out units, full time range). Details: `README.MD`.

Current stage: `project_management/current_project_status.md` Section 1 (authoritative).

## Hard Rules
- Use `.venv` for everything (`.venv\Scripts\python.exe` on Windows). Jupyter kernel: `woodwell-ts`.
- Read the domain's `*_description.md` before implementing in it; ask if unclear.
- GCS data policy and compute placement (which VM, start/stop discipline): `project_management/environment_spec.md`. Rangeland's `RangeSTAR_data/` CSVs are local and untracked (not git, not GCS).
- No hardcoding: parameters, paths and hyperparameters go in config files / GCS.
- Notebooks are for EDA only.
- Scaffold structure only; model-architecture decisions need sign-off.
- Round saved numeric outputs to suitable precision (usually 3 decimals).
- `vm-sandeep` start fails with `ZONE_RESOURCE_POOL_EXHAUSTED`: retry every ~90s until it starts. Never move or recreate it outside `us-central1-f`, whatever GCP suggests — don't ask. Stop it as soon as its job finishes; never leave it idle.

## How to Work
1. **Think first.** State assumptions; present multiple interpretations instead of picking silently; if unclear, stop and say what's confusing.
2. **Simplicity.** Minimum code, nothing speculative (no unrequested features, abstractions, configurability, or handling of impossible cases). One function, one job; flat over nested (≤2 indentation levels in logic); don't abstract until it hurts twice. Before finishing: would a senior engineer simplify this? If so, do it.
3. **Surgical changes.** Touch only what's requested; match existing style; remove only what your edit orphaned; flag pre-existing dead code rather than delete it. Every changed line should trace to the request.
4. **Goal-driven.** Turn asks into verifiable goals (e.g. failing test → pass). For multi-step tasks, state a plan first: `1. [step] → verify: [check]`.
5. **Hygiene.** Readable names, no magic values, no dead or commented-out code. Type hints on all functions; docstrings only when name + types aren't enough. Fail loudly with specific exceptions; never swallow errors. `logging`, not `print`. Print array/tensor shapes early when building pipelines.
6. **Fit the real machine.** Fixes must be correct long-term, not just for the case tried. For anything scaling with data (caches, buffers, batch sizes, disk writes), check the worst case against the target machine's specs (`environment_spec.md`).

## Project Management
At the start of every conversation, before coding, read `project_management/proj_mgmt.md` (index to diary, SSOT, logging, environment, review/git protocols).
