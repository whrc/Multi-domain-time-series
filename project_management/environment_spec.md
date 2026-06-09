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
| Jupyter kernel name | `woodwell-ts` |
| Python version | *(fill in: `python --version`)* |
| PyTorch version | *(fill in: `python -c "import torch; print(torch.__version__)"`)* |
| CUDA available | *(fill in: `python -c "import torch; print(torch.cuda.is_available())"`)* |

## Key Package Versions

<!-- Run: pip freeze | Select-String "torch|numpy|pandas|xarray|gcsfs|mlflow"
     and paste output here. Re-run whenever packages are updated. -->

```
torch==
numpy==
pandas==
xarray==
gcsfs==
mlflow==       # not yet installed — Stage 2
```

---

## GCS Access

| Field | Value |
| --- | --- |
| Bucket | **fillin*`gs://` |
| Auth status | *(update after verifying access: Verified / Unverified)* |
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
| Machine | *(fill in: OS, CPU, RAM)* |
| GPU | *(fill in, or "CPU only")* |
| Practical batch-size ceiling | *(fill in based on observed OOM threshold)* |

---

## Reproducibility Notes

- Random seed: `42` (set in `config/arctic_domain.yaml` under `preprocessing.random_seed`)
- All hyperparameters in `config/<domain>.yaml` — no hardcoding in scripts
- Checkpoints saved to `outputs/<domain>/models/best_model.pt` (gitignored)
- `best_model.run_id` sidecar written at checkpoint-save time (Stage 2)
