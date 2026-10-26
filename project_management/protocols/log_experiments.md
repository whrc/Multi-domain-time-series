# Experiment Logging Specification

<!-- Human-maintained. -->

SSOT for what every experiment logs, in all domains. Anything not listed here is not logged.

**Shared conventions:** one causal same-step transformer per domain; checkpoint on lowest val loss; loss = mean MSE over non-NaN positions of z-scored targets, logged overall and per target; saved numbers rounded to 3 decimals.

## Stage 1 — Training (`02_train.py`)

MLflow is wired for Arctic/Amazon/Rangeland, not yet for multi_domain.

| What | MLflow call | Notes |
| --- | --- | --- |
| Config params | `mlflow.log_params(flatten_cfg(cfg))` | Exclude `paths.*`, `gcs.*` |
| Train / val loss per epoch | `log_metric("train_loss" / "val_loss", v, step=epoch)` | |
| Per-target val loss | `log_metric(f"val_loss_{target}", v, step=epoch)` | |
| Best checkpoint | `log_artifact(checkpoint_path)` | At save time |
| Tags | `arch` = `cfg["model"]["architecture"]` (only `transformer`); `exp_id` = `<DD>-{run_id[:8]}`, `<DD>` ∈ `AR`/`AM`/`RG` | |

Files: `outputs/<domain>/models/best_model.pt` (dict: `epoch, model_state_dict, optimizer_state_dict, val_loss, num_features, num_targets, cfg`) and `best_model.run_id` sidecar (read by `03`/`04`).

Per-epoch log line: `Epoch  N | train=X.XXXX  val=X.XXXX  lr=X.XXe-XX`. Record best val loss + epoch in `key_findings_log.md`.

## Stage 2 — Prediction (`03_predict.py`)

Stride-1 sliding window; keep the last-position prediction (`window_start + seq_len − 1`); first `seq_len − 1` steps are NaN; inverse-transform with the scaler's target columns. MLflow (reopened via run_id sidecar): `log_metric("prediction_complete", 1)`, tag `predict_timestamp` (UTC ISO). Prediction files are not logged to MLflow.

| Domain | Path | Contents |
| --- | --- | --- |
| Arctic | `outputs/arctic_domain/predictions/<grid>/<ssp>/<var>_<resolution>_pred_<split>.nc` | NetCDF `(time, y, x)`; `split` = `tr` (historical) / `sc` (projected) |
| Amazon | `outputs/amazon_domain/predictions/amazon_test_predictions.parquet` | `station_id, year, month` + 3 `*_pred` |
| Rangeland | `outputs/rangeland_domain/predictions/predictions.parquet` | `site, date` + 10 targets + derived NEE |

Arctic SSPs: `ssp1_2_6_mri_esm2_0`, `ssp5_8_5_mri_esm2_0`. Variables: ALD, VEGC (yearly); GPP, RECO (monthly).

## Stage 3 — Evaluation (`04_evaluate.py`)

`outputs/<domain>/evaluation/metrics_test.csv`, one row per (unit, target[, period]):

| Columns | |
| --- | --- |
| All domains | `target`, `RMSE`, `NSE`, `KGE`, `PBIAS` |
| Arctic | `grid`, `y`, `x`, `lat`, `lon`, `ssp`, `period` (`historical`/`projected`) |
| Amazon | `station_id` |
| Rangeland | `site`, `pft` |

NSE/KGE: −∞ to 1 (≥0.5 good). PBIAS: % (0 perfect, positive = overprediction). Degenerate cases (constant/zero-sum obs, <2 points) → NaN, dropped from aggregates (`shared/metrics.py`).

Summary (not saved): median across units by `target` (and `period`/`ssp`), computed by `generate_report.py`. MLflow: `<TARGET>_NSE_med`, `_KGE_med`, etc., plus `metrics_test.csv` and all figures as artifacts.

Figures (`shared/plots.py`):

| Figure | Function | Domains |
| --- | --- | --- |
| Loss curves (overall + per target) | `plot_loss_curves` | all |
| Predicted vs true | `plot_pred_vs_true` (`log_scale=True` for skewed targets) | all |
| Metric boxplots (4 panels) | `plot_metric_boxplot` (`group_col`: `period` Arctic, `pft` Rangeland, none Amazon) | all |
| Spatial NSE maps | `plot_spatial_map` (ssp × period × variable) | Arctic |
| Representative time series | `plot_timeseries` | Amazon, Rangeland |

## `key_findings_log.md` entries

Add one only if: (1) first evaluation for the domain, (2) a config change moved NSE by >0.05 vs the prior run, (3) a failure or unexpected behaviour, or (4) a design decision followed from results.

Contents: best val loss + epoch; median NSE/KGE/PBIAS per target (per group where applicable); notable patterns; interpretation (human) / "What happened" (Claude draft). Format and ownership: see `key_findings_log.md`.

## HTML report

After train → predict → evaluate: `python project_management/generate_report.py` (reads MLflow, `current_project_status.md`, `key_findings_log.md`).
