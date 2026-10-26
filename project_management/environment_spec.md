# Environment Specification

<!-- Human and Claude may edit. Read before running any pipeline step: venv path, GCS auth, MLflow URI, hardware limits. -->

## Python

| Field | Value |
| --- | --- |
| venv (Windows) | `<repo_root>\.venv\Scripts\python.exe` |
| venv (both VMs) | `~/Multi-domain-time-series/.venv/bin/python` |
| Jupyter kernel | `woodwell-ts` |
| Python / PyTorch | `3.11.2` / `2.12.1+cu130` (CUDA 13.0, A100) |
| Key packages | `numpy==2.4.6`, `pandas==2.3.3`, `xarray==2026.4.0`, `gcsfs==2026.6.0`, `mlflow==3.14.0` |

Refresh versions with `pip freeze` after package updates.

## GCS

| Field | Value |
| --- | --- |
| Buckets | Arctic `gs://circumpolar-readonly/raw`; Amazon `gs://fr_v1/am_hydro_fire_risk_V2/` |
| Auth | VM service account reads Arctic only; Amazon needs personal ADC (`gcloud auth application-default login`) on VMs and laptop (verified 2026-07-01) |
| Data policy | Never download to local disk. Never commit data files. |

## MLflow

| Field | Value |
| --- | --- |
| Tracking URI | `mlruns/` at repo root (`mlflow_tracking_uri` in each domain YAML) |
| Status | Enabled for Arctic/Amazon/Rangeland; not wired for `multi_domain` (`mlflow.enabled: false`) |
| UI | `mlflow ui --backend-store-uri <repo_root>/mlruns` |

## VMs

Both in `us-central1-f`, project `spherical-berm-323321`, Debian 12, 100 GB `pd-balanced` boot disk, ephemeral external IP, Linux user `sp2596`, `default` VPC.

| | `vm-sandeep` (GPU) | `vm-cpu-sandeep` (CPU) |
| --- | --- | --- |
| Machine | `a2-highgpu-1g`, 12 vCPU, 85 GB RAM | `n2-standard-32`, 32 vCPU, 128 GB RAM |
| GPU | 1x A100-SXM4-40GB, driver 580.159.03, CUDA 13.0 | None |
| Cost | $3.67/hr | $1.55/hr |
| Access | VSCode Remote-SSH host `vm-sandeep` (custom `~/.ssh/config` `ProxyCommand` resolves the current IP) | `gcloud compute ssh vm-cpu-sandeep --zone=us-central1-f` |
| Notes | Daily boot-disk snapshot (`default-schedule-1`). Batch-size ceiling not measured (Arctic `batch_size=2048` fits). | Cloned 2026-07-08 from a `vm-sandeep` snapshot (same `.venv`, repo, auth). Same service account (`419943536854-compute@developer.gserviceaccount.com`, `devstorage.read_only` + logging/monitoring scopes). |

OS Login is off: a different client machine needs `User sp2596` in its SSH config to get the same home dir.

## Compute placement

- CPU-bound (preprocessing, fetching, diagnostics) → `vm-cpu-sandeep`.
- GPU-bound (training, evaluation, GPU inference) → `vm-sandeep`.
- Laptop: orchestration only (start/stop/monitor/SSH).
- Start a VM right before it's needed; stop it as soon as the work finishes.

## Reproducibility

- Seed `42` (`preprocessing.random_seed`); all hyperparameters in `config/<domain>.yaml`.
- Checkpoint `outputs/<domain>/models/best_model.pt` (gitignored) + `best_model.run_id` sidecar.
