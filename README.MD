# Multi-domain-time-series
Woodwell multidomain timeseries forecast AI FCS project.

### Overview

This project develops Time series prediction models across environmental domains (arctic, amazon, and few others which will be added).
Dataset streams from GCP buckets — and  nothing is downloaded locally.
The work will be carried out in three stages or with three goals in mind:
- S1: Dedicated model per domain (arctic, amazon, etc.) — separate pipelines inside domains/ folder.
- S2: Single shared model across all domains in multi_domain/ folder.
- S3: Foundation model will be finetuned on each domain — this is a placeholder for now, may or may not be pursued depending on time and resources.
---

# Repo structure

Multi-domain-time-series/
│
├── config/
│   ├── config.py              # Load configs
│   ├── arctic_domain.yaml     # Domain settings
│   ├── amazon_domain.yaml
│   └── multi_domain.yaml
│
├── models/
│   └── transformer.py
│
├── domains/
│   ├── arctic_domain/
│   │   ├── arctic_description.md
│   │   ├── 00_eda.ipynb
│   │   ├── 01_preprocess.py
│   │   ├── 02_train.py
│   │   ├── 03_predict.py
│   │   └── 04_evaluate.py
│   │
│   ├── amazon_domain/         # Same structure
│   └── multi_domain/          # Same structure
│
├── outputs/
│   ├── arctic_domain/
│   │   ├── models/
│   │   ├── predictions/
│   │   └── evaluation/
│   │
│   ├── amazon_domain/         # Same structure
│   └── multi_domain/          # Same structure
│
├──run_arctic.py # Entry point for arctic domain
├──run_amazon.py # Entry point for amazon domain
├──run_multi_domain.py # Entry point for multi-domain model
│
├── requirements.txt
├── README.md
└── CLAUDE.md
                 
### Project Structure
- Point to config/ for all settings
- Point to domains/<name>/ for per-domain pipelines
- Point to models/ for shared model architectures
- Point to outputs/ for all results (models, predictions, evaluations) — organized by domain


### Setup
```bash
pip install -r requirements.txt
# Authenticate with GCP
gcloud auth application-default login
```

### How to run this repo
- For Arctic domain: `python run_arctic.py`
    This will execute the entire pipeline for Arctic domain which are defined in `domains/arctic_domain/` folder. It will read necessary configs from `config/arctic_domain.yaml` and save outputs to `outputs/arctic_domain/`.
- For Amazon domain: `python run_amazon.py`
    This will execute the entire pipeline for Amazon domain which are defined in `domains/amazon_domain/` folder. It will read necessary configs from `config/amazon_domain.yaml` and save outputs to `outputs/amazon_domain/`.
- For multi-domain model: `python run_multi_domain.py`
    This will execute the entire pipeline for the multi-domain model which is defined in `domains/multi_domain/` folder. It will read necessary configs from `config/multi_domain.yaml` and save outputs to `outputs/multi_domain/`.

