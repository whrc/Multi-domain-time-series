# Experiment Logging Specification

<!-- Human-maintained. Generalized to all three domains during the 2026-06-17
     methodology-audit remediation (was previously Arctic-only). -->

## Purpose

Defines exactly what to record, where to record it, and what artifacts to
generate for every experiment, **across all domains** (Arctic, Amazon,
Rangeland). Covers three pipeline stages: training, prediction, and evaluation.
This file is the SSOT for all logging decisions — if a metric or artifact is not
listed here, it is not logged.

**Conventions shared by every domain:**
- One causal, same-step transformer per domain; checkpoint on the **lowest
  validation loss**.
- Loss = mean MSE over valid (non-NaN) target positions, computed on z-scored
  targets (all targets on one scale). Logged as an overall value and per target.
- All saved numeric outputs are rounded to **3 decimals**.

---

## Stage 1 — Training (`02_train.py`)  — identical across domains

### MLflow (wired for Arctic/Amazon/Rangeland since 2026-06-18; still not wired for multi_domain)

| What | MLflow call | Notes |
| --- | --- | --- |
| Config params | `mlflow.log_params(flatten_cfg(cfg))` | Flatten nested dict; exclude `paths.*` and `gcs.*` keys |
| Train loss (per epoch) | `mlflow.log_metric("train_loss", v, step=epoch)` | |
| Val loss (per epoch) | `mlflow.log_metric("val_loss", v, step=epoch)` | |
| Per-target loss (per epoch) | `mlflow.log_metric(f"val_loss_{target}", v, step=epoch)` | one per target, to see if all targets are learned |
| Best checkpoint | `mlflow.log_artifact(str(checkpoint_path))` | Log at checkpoint-save time, not at run start |
| Architecture tag | `mlflow.set_tag("arch", cfg["model"]["architecture"])` | only `transformer` is implemented |
| exp_id tag | `mlflow.set_tag("exp_id", f"<DD>-{run_id[:8]}")` | `<DD>` = 2-letter domain code: `AR` arctic, `AM` amazon, `RG` rangeland |

### Files generated

| Artifact | Path | Format | Notes |
| --- | --- | --- | --- |
| Best checkpoint | `outputs/<domain>/models/best_model.pt` | PyTorch `.pt` | Dict: `{epoch, model_state_dict, optimizer_state_dict, val_loss, num_features, num_targets, cfg}` |
| MLflow run_id sidecar | `outputs/<domain>/models/best_model.run_id` | Plain text | Written at checkpoint-save time; read by `03_predict.py` and `04_evaluate.py` |

### Console / log output to capture

The training loop emits one INFO line per epoch:

```text
Epoch  N | train=X.XXXX  val=X.XXXX  lr=X.XXe-XX
```

Record best val_loss and epoch in `key_findings_log.md` "What happened" section.

---

## Stage 2 — Prediction (`03_predict.py`)

Inference is identical in mechanism across domains: slide the window with
**stride = 1**, record the prediction at the **last position** of each window
(`window_start + seq_len − 1`); the first `seq_len − 1` steps have no prediction
(NaN). Inverse-transform with the scaler's target columns.

### MLflow

| What | MLflow call | Notes |
| --- | --- | --- |
| Completion flag | `mlflow.log_metric("prediction_complete", 1)` | Reopen training run via run_id sidecar |
| Timestamp tag | `mlflow.set_tag("predict_timestamp", utc_iso_string)` | |

### Prediction file formats (per domain)

| Domain | Path pattern | Format | Contents |
| --- | --- | --- | --- |
| Arctic | `outputs/arctic_domain/predictions/<grid>/<ssp>/<var>_<resolution>_pred_<split>.nc` | NetCDF4 | Gridded `(time, y, x)` per variable; `split` = `tr` (historical) or `sc` (projected) |
| Amazon | `outputs/amazon_domain/predictions/amazon_test_predictions.parquet` | Parquet | `station_id, year, month` + 3 `*_pred` target columns |
| Rangeland | `outputs/rangeland_domain/predictions/predictions.parquet` | Parquet | `site, date` + 11 predicted columns (10 targets + derived NEE) |

**Arctic SSP identifiers:** `ssp1_2_6_mri_esm2_0`, `ssp5_8_5_mri_esm2_0`.
**Arctic variable × resolution:** ALD (yearly), GPP (monthly), RECO (monthly), VEGC (yearly).

Prediction files are **not** logged to MLflow as artifacts (too large).

---

## Stage 3 — Evaluation (`04_evaluate.py`)

### Primary table: `metrics.csv` — unified schema (all domains)

Saved to `outputs/<domain>/evaluation/metrics.csv`. One row per (unit, target) — plus `period` for Arctic.

| Column group | Columns | Notes |
| --- | --- | --- |
| Metric columns (all domains) | `target`, `RMSE`, `NSE`, `KGE`, `PBIAS` | metric names uppercase |
| Arctic id columns | `grid`, `y`, `x`, `lat`, `lon`, `ssp`, `period` | per-pixel; `period` ∈ {`historical`,`projected`} (Arctic is the only domain with multiple periods) |
| Amazon id columns | `station_id` | per-station |
| Rangeland id columns | `site`, `pft` | per-site |

Metric definitions (see `shared/metrics.py`): NSE and KGE (−∞ to 1; ≥0.5 is good);
PBIAS in % (0 perfect; positive = overprediction). Degenerate cases (constant or
zero-sum observations, < 2 points) yield `NaN` and are dropped from aggregates.

### Summary stats (computed on demand, not saved)

Group by `target` (and `period`/`ssp` where present); take the **median** across units.
Computed by `generate_report.py` and logged to MLflow as summary metrics.

### Figures generated (all via `shared/plots.py`)

| Figure | Function | Applies to |
| --- | --- | --- |
| Loss curves (overall + per-target) | `plot_loss_curves` | all domains |
| Predicted-vs-true scatter (per target) | `plot_pred_vs_true` (use `log_scale=True` for skewed targets) | all domains |
| Metric boxplots (4 panels) | `plot_metric_boxplot` (`group_col`=`period` Arctic / `pft` Rangeland / none Amazon) | all domains |
| Spatial NSE maps | `plot_spatial_map` (per ssp × period × variable) | Arctic only (gridded) |
| Representative-unit time series | `plot_timeseries` | Amazon, Rangeland |

### MLflow summary metrics (logged from `04_evaluate.py`)

For each target: `mlflow.log_metric("<TARGET>_NSE_med", v)`, `_KGE_med`, etc.
Also log `metrics.csv` and all figures as MLflow artifacts.

---

## What Goes Into `key_findings_log.md`

After evaluation, add an entry **only if** at least one criterion is met:

1. First evaluation for this domain
2. A config change produced NSE delta > 0.05 vs. prior run on same domain
3. A failure or unexpected behaviour occurred
4. A design decision was made as a result of results

Each entry contains:

- Best val_loss and epoch reached during training
- Summary stats table (median NSE/KGE/PBIAS per target, per group where applicable)
- Notable patterns (e.g. a target or region with persistently low NSE)
- Interpretation and decisions (human fills; Claude drafts "What happened")

See `key_findings_log.md` for the entry format and ownership rules.

---

## Triggering the HTML Report

After completing the full pipeline (train → predict → evaluate):

```bash
python project_management/generate_report.py
```

The report reads MLflow for experiment data, `current_project_status.md` for
status, and `key_findings_log.md` for written-up findings. It renders the
per-domain summary stats and embeds evaluation figures.
