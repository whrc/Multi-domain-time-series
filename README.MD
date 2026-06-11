# Multi-domain-time-series
Woodwell multidomain timeseries forecast AI FCS project.

### Overview < Preliminary, update as you go >

This project develops Time series prediction models across environmental domains (arctic, amazon, and others <add later>). The Arctic domain focuses on emulating the Terrestrial Ecosystem Model (TEM) using deep learning over the circumpolar region to predict variables including GPP, RECO, ALD, and VEGC. The Amazon domain will focus on predicting river discharge and wildfire at watershed level. All necessary datasets streams from GCP buckets.

The work will be carried out in three stages:
- S1: Dedicated model per domain (arctic, amazon, etc.) — separate pipelines inside domains/ folder.
- S2: Single shared model across all domains in multi_domain/ folder.
- S3: Foundation model finetuned on each domain (optional, TBD).
---

# Repo structure

```
Multi-domain-time-series/
│
├── config/
│   ├── config.py              # Load configs
│   ├── arctic_domain.yaml     # Domain settings
│   ├── amazon_domain.yaml
│   ├── ethiopia_domain.yaml
│   ├── rangeland_domain.yaml
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
│   ├── ethiopia_domain/       # Same structure replicated
│   ├── rangeland_domain/      # Same structure replicated
│   └── multi_domain/          # Same structure replicated
│
├── outputs/
│   ├── arctic_domain/
│   │   ├── models/
│   │   ├── predictions/
│   │   └── evaluation/
│   │
│   ├── amazon_domain/         # Same structure replicated
│   ├── ethiopia_domain/       # Same structure replicated
│   ├── rangeland_domain/      # Same structure replicated
│   └── multi_domain/          # Same structure replicated
│
├──run_arctic.py       # Entry point for arctic domain
├──run_amazon.py       # Entry point for amazon domain
├──run_ethiopia.py     # Entry point for ethiopia domain
├──run_rangeland.py    # Entry point for rangeland domain
├──run_multi_domain.py # Entry point for multi-domain model
│
├── requirements.txt
├── README.md
└── CLAUDE.md
```

### Project structure
- `config/` — all settings and hyperparameters (no hardcoding)
- `domains/<name>/` — per-domain pipelines (EDA → preprocess → train → predict → evaluate)
- `models/` — shared model architectures
- `outputs/` — all results (models, predictions, evaluations) organized by domain; data files are gitignored

### How to run

Install dependencies (use the project venv):
```
pip install -r requirements.txt
```

**Arctic domain** (S1 — current):
```
python run_arctic.py
```
Steps run in order: preprocess → train → predict → evaluate. Configure paths, GCS buckets, and hyperparameters in `config/arctic_domain.yaml`.

> Amazon domain and multi-domain model are planned but not yet implemented.
