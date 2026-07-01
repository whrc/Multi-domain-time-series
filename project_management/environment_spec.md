# Environment Specification

<!-- Claude Instructions ─────────────────────────────────────────────────────
Both Claude and the human may update this file.
Claude should READ this file before running any pipeline step to confirm:
  - Correct venv path for subprocess/terminal calls
  - GCS auth status before streaming from the bucket
  - MLflow tracking URI before calling mlflow.set_tracking_uri()
  - Hardware constraints that affect batch size or grid subset choices
──────────────────────────────────────────────────────────────────────────── -->

---

## Python Environment

| Field | Value |
| --- | --- |
| venv path (Windows) | `<repo_root>\.venv\Scripts\python.exe` |
| venv path (GCE VM `vm-sandeep`) | `~/Multi-domain-time-series/.venv/bin/python` |
| Jupyter kernel name | `woodwell-ts` |
| Python version | `3.11.2` |
| PyTorch version | `2.12.1+cu130` |
| CUDA available | `True` (A100, driver 580.159.03, CUDA 13.0) |

## Key Package Versions

<!-- Run: pip freeze | Select-String "torch|numpy|pandas|xarray|gcsfs|mlflow"
     and paste output here. Re-run whenever packages are updated. -->

```
torch==2.12.1+cu130
numpy==2.4.6
pandas==2.3.3
xarray==2026.4.0
gcsfs==2026.6.0
mlflow==3.14.0
```

---

## GCS Access

| Field | Value |
| --- | --- |
| Bucket | `gs://circumpolar-readonly/raw` (Arctic), `gs://fr_v1/am_hydro_fire_risk_V2/` (Amazon) |
| Auth status | Verified from `vm-sandeep` (2026-07-01) — VM's default service account can read the Arctic bucket only; Amazon bucket requires personal-account ADC (`gcloud auth application-default login`), same as local-machine setup |
| Data policy | Never download to local disk. Never commit data files. |

---

## MLflow Tracking

| Field | Value |
| --- | --- |
| Tracking URI | `mlruns/` at repo root (set via `mlflow_tracking_uri` in each domain YAML) |
| Status | Not yet configured — Stage 2 |
| UI command | `mlflow ui --backend-store-uri <repo_root>/mlruns` |

---

## VM specification

| Field | Value |
| --- | --- |
| Machine | GCE `vm-sandeep` (`us-central1-f`, project `spherical-berm-323321`) — Debian GNU/Linux 12 (bookworm), machine type `a2-highgpu-1g`, 12 vCPU, 85 GB RAM, 100 GB boot disk |
| GPU | 1x NVIDIA A100-SXM4-40GB, driver 580.159.03 (LTS branch), CUDA 13.0 |
| Practical batch-size ceiling | *(fill in based on observed OOM threshold once a production run is executed)* |
| External IP | Ephemeral — changes on every stop/start |
| Access | VSCode Remote-SSH via `~/.ssh/config` host `vm-sandeep` — custom entry (not `gcloud compute config-ssh`'s output, which lists every instance in the shared GCP project) using a `ProxyCommand` that resolves the current external IP via `gcloud compute instances describe` on each connection, so it survives IP changes with no manual update needed |
| Linux user on VM | `sp2596` (matches local OS username on the machine that first SSH'd in — OS Login is not enabled on this project, so a different client machine with a different local username would get a different Linux user/home dir unless the same `User sp2596` override is set in that machine's SSH config) |

---

## Reproducibility Notes

- Random seed: `42` (set in `config/arctic_domain.yaml` under `preprocessing.random_seed`)
- All hyperparameters in `config/<domain>.yaml` — no hardcoding in scripts
- Checkpoints saved to `outputs/<domain>/models/best_model.pt` (gitignored)
- `best_model.run_id` sidecar written at checkpoint-save time (Stage 2)
