# CLAUDE.md — Multi-Domain Time Series Forecasting

## Project
Forecast time series across multiple (now two) domains: **Arctic** and **Amazon**.
The Arctic domain focuses on emulating the Terrestrial Ecosystem Model (TEM) using deep learning over the circumpolar region where inputs are gridded environmental variables including climate, soil, vegetation, and fire data, and targets are TEM output variables like GPP, RECO, ALD, and VEGC. The Amazon domain will focus on forecasting river discharge and wildfire at watershed level using climate and land use variable as inputs.
All data lives in GCS — never download to local disk, never commit data files.

**Goals (in order):**
1. Dedicated per-domain models
2. Single shared cross-domain model
3. [Optional] Fine-tune a foundation model per domain

## Current Stage
> Update this as work progresses.
- [completed] step 1: Dedicated model for Arctic domain, `domains/arctic_domain/`
- [Current] step 2: Dedicated model for Amazon domain, `domains/amazon_domain/`
- [Not Started] step 3: Shared model for all domains, `domains/multi_domain/`
- [Not Started] step 4: Foundation model fine-tuning (TBD)

## Layout
Multi-domain-time-series/
│
├── config/
│   ├── config.py              # Load configs
│   ├── arctic_domain.yaml     # Domain settings
│   ├── amazon_domain.yaml
│   └── multi_domain.yaml
│
├── models/
│   ├── transformer.py
│   └── lstm.py
│
├── domains/                   # Each domain is self-contained
│   ├── arctic_domain/
│   │   ├── arctic_description.md
│   │   ├── 00_eda.ipynb
│   │   ├── 01_preprocess.py
│   │   ├── 02_train.py
│   │   ├── 03_predict.py
│   │   └── 04_evaluate.py
│   │
│   ├── amazon_domain/         # Same structure replicated
│   └── multi_domain/          # Same structure replicated
│
├── outputs/
│   ├── arctic_domain/
│   │   ├── models/
│   │   ├── predictions/
│   │   └── evaluation/
│   │
│   ├── amazon_domain/         # Same structure replicated
│   └── multi_domain/          # Same structure replicated
│
├──run_arctic.py # Entry point for arctic domain
├──run_amazon.py # Entry point for amazon domain
├──run_multi_domain.py # Entry point for multi-domain model
│
├── requirements.txt
├── README.md
└── CLAUDE.md

## Rules (always follow)
- Use the `.venv` in the project root for all work (`\.venv\Scripts\python.exe` on Windows). Jupyter kernel is registered as `woodwell-ts`.
- Read instruction provided in the md files included in each domain folder carefully before writing code. Ask for clarification if anything is unclear.

**Process**
- One step at a time. Confirm before moving on.
- Ask before assuming — schema, model choice, metrics, design decisions.
- Never refactor and add features in the same step.
- Model architecture decisions belong to the human. Scaffold structure, not internals.

**Code**
- Write the shortest correct code. If it can be 50 lines, don't write 200.
- Before finishing: ask *"Would a senior engineer simplify this?"* — if yes, do it first.
- One function, one job. Flat over nested. Max 2 levels of indentation in logic.
- Readable names. No magic values. No dead code (delete, don't comment out).
- Type hints on all functions. Docstrings only when name+types aren't self-explanatory.
- Fail loudly — specific exceptions, clear messages. Never silently swallow errors.
- Logging over print.
- Don't abstract until you've written it twice and it hurt.
- Write code you'd be comfortable reading at 2am during an incident.
- Don't assume dimensions, print shapes early and often when building the pipeline.
- Use config files for all parameters, paths, and hyperparameters, GCS. No hardcoding.

## Don't
- Build goal 2/3 while working on goal 1
- Add CLI parsers, heavy abstractions, or tests for every internal helper prematurely
- Use notebooks for anything beyond EDA

# Project Management

Read `proj_mgmt.md` for project management specifications, e.g. project diary, single source of truth (SSOT), result logging, progress tracking, computing environment, agentic code review and testing, artefatct storage, report drafting etc.