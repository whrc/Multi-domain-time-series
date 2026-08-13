# CLAUDE.md — Multi-Domain Time Series Prediction

## Project
Predict time series across three domains — **Arctic**, **Amazon**, **Rangeland** — first with a dedicated model per domain, then within a single, unified **Multi-Domain** model that pools all three, testing whether sharing representations across domains improves on the dedicated models. Each domain has unique data, targets, and challenges, but they also have commonalities the shared model exploits. All modeling runs at a monthly time step. In every domain the model is a **causal, same-step emulator**: it takes a sequence of inputs up to step *t* and predicts the target at the same step *t* (it does not forecast future steps). Models are evaluated by **spatial generalization** — held-out sites/pixels/stations the model never saw in training, scored across the full available time range (historical and, where present, projected periods).

- **Arctic** — Emulates the Terrestrial Ecosystem Model (TEM) over the circumpolar region. Inputs are gridded environmental variables (climate, soil, vegetation, fire); targets are GPP and RECO.
- **Amazon** — Predicts river discharge and wildfire (active fire count, burned area) at the watershed level using climate and land-use variables as inputs.
- **Rangeland** — Emulates a process model (RangeSTAR) predicting carbon fluxes: GPP, RECO, maintenance respiration (Rm), and growth respiration (Rg).

**Goals (in order):**
1. Dedicated per-domain models
2. Single shared cross-domain model

Work strictly in order — don't start goal 2 while goal 1 is unfinished.

## Current Stage
> Quick-reference pointer — authoritative source is `project_management/current_project_status.md`.
- [Production run complete, incl. final 5-seed publication sweep] step 1: Dedicated model for Arctic domain, `domains/arctic_domain/`
- [Production run complete, incl. final 5-seed publication sweep] step 2: Dedicated model for Amazon domain, `domains/amazon_domain/`
- [Production run complete, incl. final 5-seed publication sweep] step 3: Dedicated model for Rangeland domain, `domains/rangeland_domain/`
- [Production run complete — flux-only variant has run the final 5-seed sweep; full-target variant has not] step 4: Shared model for all domains, `domains/multi_domain/`

## Layout

```
Multi-domain-time-series/
│
├── config/
│   ├── config.py              # Load configs
│   ├── arctic_domain.yaml     # Domain settings
│   ├── amazon_domain.yaml
│   ├── rangeland_domain.yaml
│   └── multi_domain.yaml
│
├── shared/
│   ├── transformer.py         # Causal transformer — shared across all domains
│   ├── metrics.py             # RMSE, NSE, KGE, PBIAS
│   ├── plots.py               # Loss curves, scatter, boxplot, CDF, timeseries, spatial map
│   ├── dataset.py             # WindowedDataset + records_to_segments
│   ├── training.py            # masked_mse_loss, run_lr_finder, train_model
│   ├── inference.py           # predict_last_position (dense stride-1 inference)
│   ├── evaluate.py            # predict_and_inverse, per_unit_metrics, stack_by_target
│   ├── io.py                  # GCS filesystem + NetCDF/CSV readers
│   ├── runner.py              # Subprocess pipeline orchestration
│   ├── tracking.py            # MLflow helpers (gated by mlflow.enabled in config)
│   └── seed_aggregation.py    # Mean/std-across-seeds rollup for multi-seed publication runs
│
├── domains/                   # Each domain is self-contained
│   ├── arctic_domain/
│   │   ├── arctic_description.md  # Full pipeline spec — read before implementing
│   │   ├── 00_eda.ipynb
│   │   ├── 01_preprocess.py
│   │   ├── 02_train.py
│   │   ├── 03_predict.py
│   │   ├── 04_evaluate.py
│   │   ├── 05_learning_curve.py   # Val performance vs train-set size saturation
│   │   └── run_preprocess_resilient.sh  # Auto-relaunches 01_preprocess.py until it succeeds
│   │
│   ├── amazon_domain/         # Same structure, own *_description.md
│   ├── rangeland_domain/      # Same structure, own *_description.md
│   └── multi_domain/          # Two-stage shared model
│       ├── model.py            # MultiDomainModel: per-domain projection → transformer → MLP heads
│       ├── multi_description.md
│       ├── 01_preprocess.py   # Pre-flight check
│       ├── 02_train.py        # Stage 1 pretrain + Stage 2 per-domain finetune
│       ├── 03_predict.py      # Inference per domain × checkpoint stage
│       ├── 04_evaluate.py     # Metrics + plots for both stages
│       └── flux_only.py       # Flux-only target-subset selection, shared by 02/03/04
│
├── outputs/
│   ├── arctic_domain/
│   │   ├── preprocessed/      # train_{size}.pkl (e.g. train_50K.pkl, train_500K.pkl), val.pkl,
│   │   │                      # test.pkl, each with a co-located {name}.meta.json sidecar, plus
│   │   │                      # .grid_pass1_summary_cache/, .grid_pass2_records_cache/, and
│   │   │                      # .grid_failed_cache/ (per-grid resumability caches, see
│   │   │                      # arctic_description.md)
│   │   ├── scaler.pkl         # {"mean", "std"} — fit on train
│   │   ├── models/            # best_model.pt
│   │   ├── predictions/       # predictions in designated format
│   │   └── evaluation/        # metrics, figures
│   │
│   ├── amazon_domain/         # Same structure
│   ├── rangeland_domain/      # Same structure
│   └── multi_domain/          # Same structure
│
├── project_management/        # proj_mgmt.md, current_project_status.md, key_findings_log.md, environment_spec.md, protocols/
│
├── figures/                   # Manuscript figures
│   ├── scripts/                # make_figureN_*.py generators
│   └── svg/                    # Vector-source outputs
│
├── ablation_test/             # Capacity-matched + pairwise ablation study — why multi-domain
│                               # helps (ablation_description.md)
│
├── hyperparameter_tuning/     # Per-domain architecture sweeps — hidden_dim/dropout/etc.
│                               # (hyperparameter_tuning_description.md)
│
├── metric_decomposition/      # KGE -> r/alpha/beta decomposition per target
│                               # (metric_decomposition_description.md)
│
├── tests/                     # e.g. tests/arctic_domain/test_grid_split.py
│
├── RangeSTAR_data/            # Local Rangeland CSVs — tracked in git (rounded to 3 dp)
│
├── run_arctic.py              # Entry point — arctic domain
├── run_amazon.py              # Entry point — amazon domain
├── run_rangeland.py           # Entry point — rangeland domain
├── run_multi_domain.py        # Entry point — multi-domain model
├── run_seed_sweep.py          # Orchestrates the final 5-seed publication run across all domains
│
├── requirements.txt
├── README.MD
└── CLAUDE.md
```

## Hard Rules (always follow)
- Use the project's `.venv` for all work (`.venv\Scripts\python.exe` on Windows). Jupyter kernel: `woodwell-ts`.
- Read the domain's `*_description.md` before implementing anything in that domain. Ask if anything in it is unclear.
- GCS data policy (Arctic/Amazon; Rangeland's `RangeSTAR_data/` CSVs are the tracked-in-git exception) and compute placement (which VM runs what, VM start/stop discipline) are defined once in `project_management/environment_spec.md` — follow it, don't restate it here.
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