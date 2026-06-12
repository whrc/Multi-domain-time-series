# Multi-Domain Time Series Forecasting

Deep learning pipelines for forecasting and emulating process models across three environmental domains: **Arctic**, **Amazon**, and **Rangeland**. All data streams from GCS; no data is committed to this repo.

---

## Goals

| Stage | Description | Status |
|-------|-------------|--------|
| S1 | Dedicated model per domain — separate pipelines in `domains/` | In progress |
| S2 | Single shared model across all domains in `domains/multi_domain/` | Not started |
| S3 | Foundation model fine-tuned per domain | Optional / TBD |

## Domain Status

| Domain | Focus | EDA | Preprocess | Train | Predict | Evaluate |
|--------|-------|-----|------------|-------|---------|----------|
| Arctic | TEM emulation (GPP, RECO, ALD, VEGC) | Complete | Spec ready | — | — | — |
| Amazon | River discharge & wildfire forecasting | Complete | Complete | — | — | — |
| Rangeland | Carbon flux emulation (NEE, GPP) | — | — | — | — | — |

---

## Repo Structure

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
│   │   ├── arctic_description.md   # Full pipeline spec — read before implementing
│   │   ├── 00_eda.ipynb
│   │   ├── 01_preprocess.py
│   │   ├── 02_train.py
│   │   ├── 03_predict.py
│   │   └── 04_evaluate.py
│   │
│   ├── amazon_domain/         # Same structure
│   ├── rangeland_domain/      # Same structure
│   └── multi_domain/          # Same structure
│
├── outputs/
│   ├── arctic_domain/
│   │   ├── preprocessed/      # train.pkl, val.pkl, test.pkl, scaler.pkl
│   │   ├── models/            # best_model.pt
│   │   ├── predictions/       # NetCDF predictions
│   │   └── evaluation/        # metrics.csv, figures
│   │
│   ├── amazon_domain/         # Same structure
│   ├── rangeland_domain/      # Same structure
│   └── multi_domain/          # Same structure
│
├── run_arctic.py              # Entry point — arctic domain
├── run_amazon.py              # Entry point — amazon domain
├── run_rangeland.py           # Entry point — rangeland domain
├── run_multi_domain.py        # Entry point — multi-domain model
│
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## Key Design Decisions

- **Monthly resolution** — all domains operate at monthly time steps; yearly inputs (CO2, fire) are excluded where correlation is weak.
- **Causal transformer** — `shared/transformer.py`; each time step attends only to past positions.
- **Per-pixel sequences** — data is structured as `List[Dict]` with keys `{grid, ssp, y, x, data}` where `data` is `(T, nFeatures+nTargets)`.
- **NaN masking loss** — yearly targets (e.g. ALD, VEGC in Arctic) are kept as NaN at non-annual positions; the training loss masks these out so each target contributes at its native resolution.
- **Shared utilities** — `shared/metrics.py` and `shared/plots.py` are used across all domains for consistency.

---

## Setup

Install dependencies using the project virtual environment:

```bash
pip install -r requirements.txt
```

Python environment: `.venv` in project root (`\.venv\Scripts\python.exe` on Windows). Jupyter kernel: `woodwell-ts`.

All GCS paths and hyperparameters are in `config/<domain>.yaml` — no hardcoding.

---

## How to Run

Each domain has a description markdown (`<domain>_description.md`) that serves as the authoritative spec. **Read it before implementing or modifying any pipeline script.**

```bash
# Arctic domain (S1)
python run_arctic.py

# Amazon domain (S1)
python run_amazon.py
```

Each entry point runs: preprocess → train → predict → evaluate in sequence.
