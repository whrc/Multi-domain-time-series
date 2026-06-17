# CLAUDE.md — Multi-Domain Time Series Forecasting

## Project
Forecast time series across three domains — **Arctic**, **Amazon**, **Rangeland** — separately, and eventually within a unified **Multi-Domain** framework. Each domain has unique data, targets, and challenges, but they also have commonalities, so shared modeling approaches will also be explored. All modeling runs at a monthly time step; in every domain the model takes a sequence of past time steps as input and predicts the next time step's output (next-token prediction).

- **Arctic** — Emulates the Terrestrial Ecosystem Model (TEM) over the circumpolar region. Inputs are gridded environmental variables (climate, soil, vegetation, fire); targets are TEM outputs like GPP, RECO, ALD, VEGC.
- **Amazon** — Forecasts river discharge and wildfire at the watershed level using climate and land-use variables as inputs.
- **Rangeland** — Emulates a process model (RangeSTAR) predicting carbon fluxes and pools (NEE, GPP, etc.).

**Goals (in order):**
1. Dedicated per-domain models
2. Single shared cross-domain model
3. [Optional] Fine-tune a foundation model per domain

Work strictly in order — don't start goal 2 or 3 while goal 1 is unfinished.

## Current Stage
> Quick-reference pointer — authoritative source is `project_management/current_project_status.md`.
- [Current] step 1: Dedicated model for Arctic domain, `domains/arctic_domain/`
- [Current] step 2: Dedicated model for Amazon domain, `domains/amazon_domain/`
- [Current] step 3: Dedicated model for Rangeland domain, `domains/rangeland_domain/`
- [Not Started] step 4: Shared model for all domains, `domains/multi_domain/`
- [Not Started] step 5: Foundation model fine-tuning (TBD)

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
│   ├── metrics.py             # RMSE, NSE, KGE, PBIAS — shared across all domains
│   └── plots.py               # Loss curves, scatter, boxplot, CDF, spatial map — shared
│
├── domains/                   # Each domain is self-contained
│   ├── arctic_domain/
│   │   ├── arctic_description.md  # Full pipeline spec — read before implementing
│   │   ├── 00_eda.ipynb
│   │   ├── 01_preprocess.py
│   │   ├── 02_train.py
│   │   ├── 03_predict.py
│   │   └── 04_evaluate.py
│   │
│   ├── amazon_domain/         # Same structure, own *_description.md
│   ├── rangeland_domain/      # Same structure, own *_description.md
│   └── multi_domain/          # Same structure, own *_description.md
│
├── outputs/
│   ├── arctic_domain/
│   │   ├── preprocessed/      # train.pkl, val.pkl, test.pkl, scaler.pkl
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
├── RangeSTAR_data/            # Local Rangeland CSVs — gitignored
│
├── run_arctic.py              # Entry point — arctic domain
├── run_amazon.py              # Entry point — amazon domain
├── run_rangeland.py           # Entry point — rangeland domain
├── run_multi_domain.py        # Entry point — multi-domain model
│
├── requirements.txt
├── README.MD
└── CLAUDE.md
```

## Hard Rules (always follow)
- Use the project's `.venv` for all work (`.venv\Scripts\python.exe` on Windows). Jupyter kernel: `woodwell-ts`.
- Read the domain's `*_description.md` before implementing anything in that domain. Ask if anything in it is unclear.
- Arctic and Amazon data live in GCS — never download to local disk, never commit data files. Rangeland is the exception: local CSVs in `RangeSTAR_data/` (gitignored).
- All parameters, paths, and hyperparameters go in config files / GCS — no hardcoding.
- Notebooks are for EDA only — nothing else.
- Scaffold structure, don't make unilateral model-architecture decisions — those need sign-off.
- Save all output numeric files by rounding to suitable precision (in most cases, 3 is plenty) to avoid saving unnecessarily large files with meaningless precision.

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

## Project Management
Read `project_management/proj_mgmt.md` at the start of every new conversation, before any coding work. It's the master index for the project diary, SSOT, result logging, progress tracking, computing environment, code-review and git protocols, and report drafting.