# Multi-Domain Time Series Forecasting

Deep learning pipelines for time series prediction across three environmental domains: **Arctic**, **Amazon**, and **Rangeland**. All data streams from GCS; no data is committed to this repo.

---

## Goals

| Stage | Description | Status |
|-------|-------------|--------|
| S1 | Dedicated model per domain — separate pipelines in `domains/` | In progress |
| S2 | Single shared model across all domains in `domains/multi_domain/` | Not started |
| S3 | Foundation model fine-tuned per domain | Optional / TBD |


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