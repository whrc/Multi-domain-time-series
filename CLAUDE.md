# CLAUDE.md — Multi-Domain Time Series Prediction

## Project
Predict time series across three domains — **Arctic**, **Amazon**, **Rangeland** — first with a dedicated model per domain, then within a single, unified **Multi-Domain** model that pools all three, testing whether sharing representations across domains improves on the dedicated models. All modeling runs at a monthly time step. In every domain the model is a **causal, same-step emulator**: it takes a sequence of inputs up to step *t* and predicts the target at the same step *t* (it does not forecast future steps). Models are evaluated by **spatial generalization** — held-out sites/pixels/stations, scored across the full available time range. See `README.MD` for per-domain data/targets, data sources, and the full repo layout.

**Goals (in order):** 1) dedicated per-domain models, 2) a single shared cross-domain model.

## Current Stage
Authoritative source: `project_management/current_project_status.md` (Section 1, Domain Stages table) — read it directly rather than trusting a snapshot here.

## Hard Rules (always follow)
- Use the project's `.venv` for all work (`.venv\Scripts\python.exe` on Windows). Jupyter kernel: `woodwell-ts`.
- Read the domain's `*_description.md` before implementing anything in that domain. Ask if anything in it is unclear.
- GCS data policy (Arctic/Amazon; Rangeland's `RangeSTAR_data/` CSVs are local and untracked — not in git, not in GCS) and compute placement (which VM runs what, VM start/stop discipline) are defined once in `project_management/environment_spec.md` — follow it, don't restate it here.
- All parameters, paths, and hyperparameters go in config files / GCS — no hardcoding.
- Notebooks are for EDA only — nothing else.
- Scaffold structure, don't make unilateral model-architecture decisions — those need sign-off.
- Save all output numeric files by rounding to suitable precision (in most cases, 3 is plenty) to avoid saving unnecessarily large files with meaningless precision.
- GPU VM (`vm-sandeep`) start failing with `ZONE_RESOURCE_POOL_EXHAUSTED` (A100 stockout in `us-central1-f`): retry starting it every ~90s until it succeeds — don't give up after one failure. Always stop the VM immediately once the job it was started for finishes; never leave it running idle. Even if GCP's error suggests another zone/region has capacity, do not move or recreate `vm-sandeep` there — stay in `us-central1-f` and keep retrying; this is not an option, don't ask again.

## How to Work

1. **Think first, don't guess.** State assumptions explicitly. If multiple valid
   interpretations exist, present them — don't silently pick one. If something is
   unclear, stop and name what's confusing instead of proceeding.

2. **Simplicity first.** Minimum code that solves the problem, nothing speculative —
   no unrequested features, abstractions, configurability, or error handling for
   impossible cases. If 200 lines could be 50, rewrite it. Before finishing, ask:
   *"Would a senior engineer simplify this?"* — if yes, do it first.
   One function, one job. Flat over nested (max 2 levels of indentation in logic).
   Don't abstract until you've written it twice and it hurt.

3. **Surgical changes.** Touch only what the request requires. Don't refactor or
   restyle adjacent code; match existing style even if you'd choose differently.
   Remove imports/vars *your* edit orphaned; flag pre-existing dead code rather
   than delete it. Test: every changed line should trace directly to the request.

4. **Goal-driven execution.** Turn vague asks into verifiable goals
   ("fix the bug" → write a failing test, then make it pass). For multi-step tasks,
   state a plan before starting:
   ```
   1. [step] → verify: [check]
   2. [step] → verify: [check]
   ```

5. **Code hygiene.** Readable names, no magic values, no dead code (delete, don't
   comment out). Type hints on all functions; docstrings only when name+types
   aren't self-explanatory. Fail loudly — specific exceptions, clear messages,
   never swallow errors silently. Use logging, not print. Don't assume tensor/array
   dimensions — print shapes early and often when building a pipeline.

6. **Fit for the real machine, not just the test case.** A fix must be logically
   correct for long-term use — not a patch that only survives the specific case you
   just tried. Before trusting anything whose resource use scales with data volume
   (caches, in-memory buffers, batch sizes, disk writes), work out the worst case
   against the actual target machine's specs (RAM, disk, CPU — see
   `project_management/environment_spec.md`), not just the small case tested locally.

## Project Management
Read `project_management/proj_mgmt.md` at the start of every new conversation, before any coding work. It's the master index for the project diary, SSOT, result logging, progress tracking, computing environment, code-review and git protocols, and report drafting.