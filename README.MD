# Multi-Domain Time Series Forecasting

This repository contains deep learning pipelines for time series forecasting across three environmental domains: Arctic, Amazon, and Rangeland. Each domain has distinct data, targets, and modeling challenges, so models are developed separately per domain, while shared modeling approaches are explored where commonalities allow predictions to eventually live within a unified framework. All modeling is carried out at a monthly time step to maintain consistency across domains, and in all cases the model takes a sequence of past time steps as input and predicts the next time step's output (next-token prediction).

- **Arctic**: Emulates the Terrestrial Ecosystem Model (TEM) over the circumpolar region using deep learning. Inputs are gridded environmental variables (climate, soil, vegetation, fire); targets are TEM output variables such as GPP, RECO, ALD, and VEGC.
- **Amazon**: Focuses on forecasting river discharge and wildfire at many watersheds in the Amazon basin using climate and land use variables as inputs.
- **Rangeland**: Emulates a process model called RangeSTAR that predicts carbon fluxes and pools (NEE, GPP, etc.) at many sites in the rangelands.

---

## Goals

| Stage | Description | Status |
|-------|-------------|--------|
| S1 | Dedicated model per domain — separate pipelines in `domains/` | In progress |
| S2 | Single shared model across all domains in `domains/multi_domain/` | Not started |
| S3 | Foundation model fine-tuned per domain | Optional / TBD |

---

## Data Sources

| Domain | Source | Type |
|--------|--------|------|
| Arctic | `gs://circumpolar-readonly/raw` | GCS — gridded NetCDF, ~4 km resolution |
| Amazon | `gs://fr_v1/am_hydro_fire_risk_V2/` | GCS — station-level CSVs |
| Rangeland | `RangeSTAR_data/` | Local CSVs — 4 files, one per PFT group (gitignored) |

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
│   └── multi_domain/          # Same structure (not yet in use)
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
│   └── multi_domain/          # Same structure (not yet in use)
│
├── project_management/        # Diary, SSOT, protocols, environment spec
│
├── RangeSTAR_data/            # Local Rangeland CSVs (gitignored)
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

---

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name woodwell-ts --display-name "woodwell-ts"
```

Domain-level pipeline specs live in `domains/<domain>/<domain>_description.md` — read these before implementing any pipeline step.
