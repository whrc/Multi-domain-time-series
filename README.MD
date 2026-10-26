# Multi-Domain Time Series Prediction

Deep-learning time-series models for three environmental domains. A dedicated model is trained per domain, then a single shared model is trained on all three to test whether pooling data helps, especially for data-scarce domains. All modeling is monthly. Each model takes inputs up to step *t* and predicts the targets at *t*, and is evaluated by **spatial generalization** to held-out sites/pixels/stations over the full time range.

- **Arctic** — emulates the Terrestrial Ecosystem Model (TEM) over the circumpolar region from gridded climate, soil, vegetation and fire inputs; targets GPP and RECO.
- **Amazon** — predicts river discharge, active fire count and burned area at Amazon-basin watersheds from climate and land-use inputs.
- **Rangeland** — emulates the RangeSTAR process model's carbon fluxes (GPP, RECO, maintenance respiration Rm, growth respiration Rg) at rangeland sites.

## Data

| Domain | Source | Type |
|--------|--------|------|
| Arctic | `gs://circumpolar-readonly/raw` | Gridded NetCDF, ~4 km |
| Amazon | `gs://fr_v1/am_hydro_fire_risk_V2/` | Station-level CSVs |
| Rangeland | `RangeSTAR_data/` | 4 local CSVs, one per PFT group (not in git) |

## Layout

```
config/                  # config.py loader + one YAML per domain (incl. multi_domain.yaml)
shared/                  # transformer, dataset, training, inference, evaluate, metrics, plots,
                         # io (GCS readers), runner, tracking (MLflow), seed_aggregation
domains/
  <domain>/              # arctic_domain, amazon_domain, rangeland_domain:
    *_description.md     #   pipeline spec — read before implementing
    00_eda.ipynb         #   EDA
    01_preprocess.py … 04_evaluate.py
  arctic_domain/05_learning_curve.py, run_preprocess_resilient.sh
  multi_domain/          # two-stage model (joint pretrain + per-domain finetune):
                         #   model.py, flux_only.py, 01–04 scripts, multi_description.md
outputs/<domain>/        # preprocessed/, scaler.pkl, models/, predictions/, evaluation/ (gitignored)
figures/                 # scripts/make_figureN*.py, svg/ — manuscript figures
ablation_test/           # capacity-matched + pairwise ablation: why multi-domain helps
hyperparameter_tuning/   # per-domain architecture sweeps
metric_decomposition/    # KGE → r/alpha/beta decomposition per target
tests/
project_management/      # status, findings, protocols, environment spec
run_<domain>.py          # entry points: arctic, amazon, rangeland, multi_domain
run_seed_sweep.py        # 5-seed runs across all domains
```

Each supporting study directory (`ablation_test/`, `hyperparameter_tuning/`, `metric_decomposition/`) has its own `*_description.md` and reuses existing checkpoints/predictions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name woodwell-ts --display-name "woodwell-ts"
```

- Arctic and Amazon read from GCS: install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and run `gcloud auth application-default login` with your institutional account. Rangeland runs offline.
- `.venv/`, `outputs/` and `mlruns/` are gitignored and regenerate. `RangeSTAR_data/` is gitignored raw data — obtain a copy directly.
