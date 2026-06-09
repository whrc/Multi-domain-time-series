# Experiment Logging Specification

<!-- Human-maintained. Claude reads only. -->

## Purpose

Defines exactly what to record, where to record it, and what artifacts to
generate for every experiment. Covers three pipeline stages: training,
prediction, and evaluation. This file is the SSOT for all logging decisions —
if a metric or artifact is not listed here, it is not logged.

---

## Stage 1 — Training (`02_train.py`)

### MLflow (Stage 2 — not yet wired)

| What | MLflow call | Notes |
| --- | --- | --- |
| Config params | `mlflow.log_params(flatten_cfg(cfg))` | Flatten nested dict; exclude `paths.*` and `gcs.*` keys |
| Train loss (per epoch) | `mlflow.log_metric("train_loss", v, step=epoch)` | |
| Val loss (per epoch) | `mlflow.log_metric("val_loss", v, step=epoch)` | |
| Best checkpoint | `mlflow.log_artifact(str(checkpoint_path))` | Log at checkpoint-save time, not at run start |
| Architecture tag | `mlflow.set_tag("arch", cfg["model"]["architecture"])` | `transformer` or `lstm` |
| exp_id tag | `mlflow.set_tag("exp_id", f"<DOMAIN_2L>-{run_id[:8]}")` | Set immediately after run is created |

### Files generated

| Artifact | Path | Format | Notes |
| --- | --- | --- | --- |
| Best checkpoint | `outputs/<domain>/models/best_model.pt` | PyTorch `.pt` | Dict: `{epoch, model_state_dict, optimizer_state_dict, val_loss, num_features, cfg}` |
| MLflow run_id sidecar | `outputs/<domain>/models/best_model.run_id` | Plain text | Written at checkpoint-save time; read by `03_predict.py` and `04_evaluate.py` |

### Console / log output to capture

The training loop emits one INFO line per epoch:

```text
Epoch  N | train=X.XXXX  val=X.XXXX  lr=X.XXe-XX
```

Record best val_loss and epoch in `key_findings_log.md` "What happened" section.

---

## Stage 2 — Prediction (`03_predict.py`)

### MLflow (Stage 2)

| What | MLflow call | Notes |
| --- | --- | --- |
| Completion flag | `mlflow.log_metric("prediction_complete", 1)` | Reopen training run via run_id sidecar |
| Timestamp tag | `mlflow.set_tag("predict_timestamp", utc_iso_string)` | |

### Prediction files generated

| Artifact | Path pattern | Format | Notes |
| --- | --- | --- | --- |
| Prediction NetCDF | `outputs/<domain>/predictions/<grid>/<ssp>/<var>_<resolution>_pred_<split>.nc` | NetCDF4 | One file per (variable, split_key); split_key = `tr` (historical) or `sc` (projected) |

**SSP identifiers:** `ssp1_2_6_mri_esm2_0`, `ssp5_8_5_mri_esm2_0`

**Variable × resolution:**

| Variable | Resolution | Notes |
| --- | --- | --- |
| ALD | yearly | Active layer depth |
| GPP | monthly | Gross primary production |
| RECO | monthly | Ecosystem respiration |
| VEGC | yearly | Vegetation carbon |

Prediction NetCDFs are **not** logged to MLflow as artifacts (too large).

---

## Stage 3 — Evaluation (`04_evaluate.py`)

### Primary table: `metrics.csv`

Saved to: `outputs/<domain>/evaluation/metrics.csv`

| Column | Type | Description |
| --- | --- | --- |
| `grid` | str | Grid tile name (e.g. `H1_V10`) |
| `y` | int | Pixel row index |
| `x` | int | Pixel column index |
| `lat` | float | Pixel latitude |
| `lon` | float | Pixel longitude |
| `ssp` | str | Full SSP string (e.g. `ssp1_2_6_mri_esm2_0`) |
| `period` | str | `historical` or `projected` |
| `variable` | str | `ALD`, `GPP`, `RECO`, or `VEGC` |
| `RMSE` | float | Root mean squared error (original units) |
| `NSE` | float | Nash-Sutcliffe efficiency (−∞ to 1; ≥0.5 is good) |
| `KGE` | float | Kling-Gupta efficiency (−∞ to 1; ≥0.5 is good) |
| `PBIAS` | float | Percent bias (%; 0 is perfect) |

One row per (grid, pixel, ssp, period, variable). This is the raw per-pixel table.

### Summary stats table (computed from `metrics.csv`)

Group by `(variable, ssp, period)`, take **median** across pixels:

| variable | ssp | period | RMSE_med | NSE_med | KGE_med | PBIAS_med |
| --- | --- | --- | --- | --- | --- | --- |

This table is not saved to disk — it is computed on demand by
`generate_report.py` and logged to MLflow as summary metrics (Stage 2).

### MLflow summary metrics (Stage 2, logged from `04_evaluate.py`)

For each variable: `mlflow.log_metric("<VAR>_NSE_med", v)`,
`mlflow.log_metric("<VAR>_KGE_med", v)`, etc.
Also log `metrics.csv` as an MLflow artifact.

### Figures generated

**Boxplot figures** (one per SSP):

| File | Path | Description |
| --- | --- | --- |
| `metrics_boxplot_ssp1.png` | `outputs/<domain>/evaluation/` | 2×2 subplots (RMSE, NSE, KGE, PBIAS); x-axis = variable; colour = period (historical=blue, projected=orange); whiskers = 5–95th percentile; no outliers shown |
| `metrics_boxplot_ssp5.png` | `outputs/<domain>/evaluation/` | Same layout for SSP5-8.5 |

**Spatial NSE maps** (one per ssp × period × variable):

| File pattern | Path | Description |
| --- | --- | --- |
| `NSE_<ssp_short>_<period>_<var>.png` | `outputs/<domain>/evaluation/spatial_metrics_maps/` | Scatter map coloured by NSE (RdYlGn, −1 to 1); one dot per test pixel; axes = lon/lat |

`ssp_short`: `ssp1` for SSP1-2.6, `ssp5` for SSP5-8.5.

**All figures are logged to MLflow as artifacts (Stage 2).**

---

## What Goes Into `key_findings_log.md`

After evaluation, add an entry **only if** at least one criterion is met:

1. First evaluation for this domain
2. A config change produced NSE delta > 0.05 vs. prior run on same domain
3. A failure or unexpected behaviour occurred
4. A design decision was made as a result of results

Each entry contains:

- Best val_loss and epoch reached during training
- Summary stats table (median NSE/KGE/PBIAS per variable per SSP+period)
- Notable spatial patterns (e.g. "NSE < 0 in permafrost zone")
- Interpretation and decisions (human fills; Claude drafts "What happened")

See `key_findings_log.md` for the entry format and ownership rules.

---

## Triggering the HTML Report

After completing the full pipeline (train → predict → evaluate):

```bash
python project_management/generate_report.py
```

The report reads MLflow for experiment data, `current_project_status.md` for
status, and `key_findings_log.md` for written-up findings. It embeds the
boxplot figures as base64 and renders the summary stats table.
