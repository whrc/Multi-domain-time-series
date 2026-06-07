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
```
                 
### Project Structure
- Point to config/ for all settings
- Point to domains/<name>/ for per-domain pipelines
- Point to models/ for shared model architectures
- Point to outputs/ for all results (models, predictions, evaluations) — organized by domain

### How to run