# Multi-Domain Time Series Prediction

This repository contains deep learning pipelines for time series prediction across three environmental domains: Arctic, Amazon, and Rangeland. Each domain has distinct data, targets, and modeling challenges, so models are developed separately per domain, while shared modeling approaches are explored where commonalities allow predictions to eventually live within a unified framework. All modeling is carried out at a monthly time step to maintain consistency across domains. In all cases the model takes a sequence of inputs up to step *t* and predicts the target variables at the same step *t*. Models are evaluated by **spatial generalization** to held-out sites/pixels/stations across the full available time range.

- **Arctic**: Emulates the Terrestrial Ecosystem Model (TEM) over the circumpolar region using deep learning. Inputs are gridded environmental variables (climate, soil, vegetation, fire); targets are TEM output variables such as GPP, RECO, ALD, and VEGC.
- **Amazon**: Predicts river discharge and wildfire at many watersheds in the Amazon basin using climate and land use variables as inputs.
- **Rangeland**: Emulates a process model called RangeSTAR that predicts carbon fluxes and biomass pools (NEE, GPP, etc.) at many sites in the rangelands.

---

## Goals

| Stage | Description | Status |
|-------|-------------|--------|
| S1 | Dedicated model per domain — separate pipelines in `domains/` | Code-complete; dev-verified on CPU; awaiting production run on GCP A100 |
| S2 | Single shared model across all domains in `domains/multi_domain/` | Scaffolded (model + 01–04); begins after S1 production runs complete |
| S3 | Foundation model fine-tuned per domain | Optional / TBD |

---

## Data Sources

| Domain | Source | Type |
|--------|--------|------|
| Arctic | `gs://circumpolar-readonly/raw` | GCS — gridded NetCDF, ~4 km resolution |
| Amazon | `gs://fr_v1/am_hydro_fire_risk_V2/` | GCS — station-level CSVs |
| Rangeland | `RangeSTAR_data/` | Local CSVs — 4 files, one per PFT group (tracked at 3 dp) |

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
│   ├── metrics.py             # RMSE, NSE, KGE, PBIAS
│   ├── plots.py               # Loss curves, scatter, boxplot, CDF, timeseries, spatial map
│   ├── dataset.py             # WindowedDataset + records_to_segments
│   ├── training.py            # masked_mse_loss, run_lr_finder, train_model
│   ├── inference.py           # predict_last_position (dense stride-1 inference)
│   ├── evaluate.py            # predict_and_inverse, per_unit_metrics, stack_by_target
│   ├── io.py                  # GCS filesystem + NetCDF/CSV readers
│   ├── runner.py              # Subprocess pipeline orchestration
│   └── tracking.py            # MLflow helpers (gated by mlflow.enabled in config)
│
├── domains/                   # Each domain is self-contained
│   ├── arctic_domain/
│   │   ├── arctic_description.md   # Full pipeline spec — read before implementing
│   │   ├── 00_eda.ipynb
│   │   ├── 01_preprocess.py
│   │   ├── 02_train.py
│   │   ├── 03_predict.py
│   │   ├── 04_evaluate.py
│   │   └── 05_learning_curve.py    # Val performance vs train-set size
│   │
│   ├── amazon_domain/         # Same structure, own *_description.md
│   ├── rangeland_domain/      # Same structure, own *_description.md
│   └── multi_domain/          # Two-stage model (pretrain + per-domain finetune)
│       ├── model.py            # MultiDomainModel: per-domain projection → transformer → MLP heads
│       ├── 01_preprocess.py   # Pre-flight check — verifies individual domain pkl files
│       ├── 02_train.py        # Stage 1 (joint pretrain) + Stage 2 (per-domain finetune)
│       ├── 03_predict.py      # Inference per domain × checkpoint stage
│       └── 04_evaluate.py     # Metrics + plots for both stages across all domains
│
├── outputs/
│   ├── arctic_domain/
│   │   ├── preprocessed/      # train.pkl, val.pkl, test.pkl
│   │   ├── scaler.pkl         # {"mean", "std"} — fit on train
│   │   ├── models/            # best_model.pt
│   │   ├── predictions/       # predictions in designated format
│   │   └── evaluation/        # metrics, figures
│   │
│   ├── amazon_domain/         # Same structure
│   ├── rangeland_domain/      # Same structure
│   └── multi_domain/          # Same structure
│
├── project_management/        # Diary, SSOT, protocols, environment spec, audit reports
│
├── RangeSTAR_data/            # Local Rangeland CSVs — tracked in git (rounded to 3 dp)
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

The repository is organized so that settings, shared code, and per-domain work stay cleanly separated. All paths and hyperparameters live in `config/` as one YAML file per domain (plus `multi_domain.yaml`), loaded and validated by `config.py`, which keeps them out of the code and makes runs reproducible. Code that every domain reuses sits in `shared/` — the causal transformer, dataset utilities (windowed sliding-window dataset, segment flattening), training loop (masked MSE loss, LR finder, early stopping), dense inference helper, evaluation helpers (inverse-transform, per-unit metrics), GCS I/O, subprocess runner, MLflow tracking wrappers, and plotting functions. Each domain then has its own self-contained folder under `domains/`, containing a `*_description.md` spec that documents its pipeline, an EDA notebook, and the numbered pipeline stages run in order (Arctic also has a `05_learning_curve.py` for saturation analysis). The `multi_domain/` folder holds the two-stage shared model — `model.py` defines the architecture (per-domain projection → shared causal transformer → per-domain MLP heads), and `01`–`04` implement the pretrain/finetune/predict/evaluate pipeline. Everything those pipelines produce — preprocessed splits, the fitted scaler, the trained model, predictions, and evaluation metrics and figures — is written to a matching folder under `outputs/`; these artifacts are gitignored by file type, since raw data and large binaries are never committed. The remaining top-level items support the project around the code: `project_management/` keeps the project diary and single source of truth, key-findings log, environment spec, protocols, and the report generator; `RangeSTAR_data/` is the one local dataset used by the Rangeland domain — CSVs are tracked in git (rounded to 3 dp); the `run_*.py` scripts are the entry points for each domain; and `requirements.txt` and `CLAUDE.md` capture the dependencies and the contributor and agent instructions.

---

## Setup

Clone the repo, then create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .\.venv\Scripts\activate       # Windows
pip install -r requirements.txt
python -m ipykernel install --user --name woodwell-ts --display-name "woodwell-ts"
```

Arctic and Amazon domains read data from GCS, so you need Google Cloud credentials. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and run `gcloud auth application-default login` — sign in with your institutional Google account and credentials are picked up automatically. Rangeland uses local CSVs and runs fully offline. The `.venv/`, `outputs/`, and `mlruns/` directories are gitignored and must be recreated on each new machine; everything else is in the repo.

Domain-level pipeline specs live in `domains/<domain>/<domain>_description.md` — read these before implementing any pipeline step.
