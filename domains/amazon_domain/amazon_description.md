# Amazon Domain: Discharge and Wildfire Prediction

## Overview

- **Task:** causal, same-step monthly prediction of `discharge`, `active_fire_count`, `burned_area` per watershed (station).
- **Evaluation:** spatial generalization — held-out stations, full monthly record.
- **Bucket:** `gs://fr_v1/am_hydro_fire_risk_V2/` · **Config:** `config/amazon_domain.yaml` (`mode: dev | production`; all hyperparameters, paths, column names).
- **Scripts:** `00_eda.ipynb` → `01_preprocess.py` → `02_train.py` → `03_predict.py` → `04_evaluate.py` (all implemented; `run_amazon.py` runs 01→04).
- **Shared core:** `shared/dataset.py` (`WindowedDataset`, `records_to_segments`), `shared/training.py` (`masked_mse_loss`, `run_lr_finder`, `train_model`), `shared/inference.py` (`predict_last_position`), `shared/evaluate.py` (`predict_and_inverse`, `per_unit_metrics`). Numbered scripts are thin wrappers.
- **Production architecture:** `hidden_dim=64, num_layers=3, num_heads=4, feedforward_dim=256, dropout=0.10`; `batch_size=256, num_epochs=100, warmup_epochs=10, early_stopping_patience=12` (~98 stations, ~18K windows, A100 40GB). Promoted from a 4-dimension sweep for efficiency only — no accuracy difference (was `hidden_dim=128, feedforward_dim=512, dropout=0.2`). See `hyperparameter_tuning/hyperparameter_tuning_description.md`.

## Data

| File | Role |
|---|---|
| `FR_Monthly_NaturalVeg_2000_2024_v16.csv` | Monthly predictors + targets per station, 2000–2024 |
| `ANA_StationList_filteredAF_nov2025.csv` | Station allowlist (column `x`) |
| `main_drainage_Filtered_checkLU_v4.gpkg` | Station lat/lon for the map only (`EstacCd`, `Latitud`, `Longitd`) |

| Column | Raw | Type | Notes |
|---|---|---|---|
| `station_id` | `EstacaoCod` | ID | |
| `year`, `month` | same | time | Not predictors; month → `month_sin/cos = sin/cos(2π·month/12)` |
| `precip`, `tmax`, `tmin`, `et`, `vpd`, `drainage_area` | `Prec`, `tmax`, `tmin`, `ET`, `vpd`, `DrangAr` | dyn | |
| `mean_/sd_` of `precip`, `tmax`, `tmin` | — | static | Per-station, from the station's full record (all splits; predictors fully observed → no leakage) |
| `discharge` | `vazao` | target | ~6% NaN, kept and masked in loss |
| `active_fire_count` | `AF` | target | MODIS Active Fire, monthly sum per watershed (source `MODIS_AF_LU_2000_2024_WS_v4.csv`) |
| `burned_area` | `BA` | target | MODIS Burned Area, km²/month per watershed (source `MODIS_BA_NatVegFar_km2_month.csv`) |

**Column order (17):** `precip, tmax, tmin, et, vpd, drainage_area, month_sin, month_cos` (8) | `mean_precip, sd_precip, mean_tmax, sd_tmax, mean_tmin, sd_tmin` (6) | targets `discharge, active_fire_count, burned_area` (idx 14–16). `nFeatures=14`, `nTargets=3`.

## Step 0 — EDA

- Done: station counts/timelines, gaps, predictor–target correlation heatmap, summary tables.
- Only `discharge` has NaN; no imputation, no outlier drops.

## Step 1 — Preprocessing

1. Load CSV from GCS; keep allowlisted stations; select/rename via `columns.rename` in config.
2. Add `month_sin`/`month_cos`.
3. Gaps: sort by year/month ordinal, find breaks with `np.diff(ords) != 1`, log total missing station-months. No NaN rows inserted.
4. Station-level random split (`train_frac`/`val_frac`/`test_frac`, `preprocessing.random_seed`).
5. Per-station climatological mean/std (above), broadcast across time steps.
6. Reorder columns as above.
7. Target transforms (before scaler fit):
   - `discharge /= drainage_area` (raised test NSE 0.014 → 0.351). Not applied to fire targets — it made `burned_area` worse (median NSE 0.014 → −1.08).
   - `log1p` on all three targets (NaN-safe).
   - Identity output head (`model.nonneg_output: false`). Softplus was dropped: on z-scored log1p outputs it floored predictions at the train mean, not 0.
8. Scaler fit on train only, on transformed targets (NaN excluded); `std=1` where 0. Save `{"mean","std"}` shape `(17,)` to `paths.scaler`; apply `(x−mean)/std` to all splits.
9. Contiguous segments split at gaps; drop segments shorter than `preprocessing.seq_len`. Each `(T_seg, 17)`.
10. Save `train/val/test.pkl` (`HIGHEST_PROTOCOL`) to `paths.preprocessed_dir`: `List[Dict]` with `station_id`, `segments`, `segment_starts` (`(year, month)` per segment), `drainage_area` (raw, for inverse transform).

## Step 2 — Training

- `WindowedDataset`: windows of `preprocessing.seq_len`, step `preprocessing.stride`, per segment; input `[:, :14]`, target `[:, 14:]`.
- `TransformerModel(num_features=14, num_targets=3, cfg)` (`shared/transformer.py`: causal mask, sinusoidal PE, GELU).
- AdamW (`training.weight_decay`), LR = `training.optimized_lr` or, if null, LR finder from `training.initial_lr` (plot `{paths.evaluation}/lr_finder.png`). Linear warmup `training.warmup_epochs`, then cosine to 0.
- Loss: masked MSE over all valid positions/targets. Validate every `training.eval_every_n_epochs`; checkpoint best to `paths.best_model`; early stop after `training.early_stopping_patience` evaluations.
- Log train/val loss overall and per target; save loss curve, val scatter, metric boxplots (RMSE/NSE/KGE/PBIAS) via `shared/metrics.py`/`shared/plots.py`. MLflow if `mlflow.enabled`.
- `--seed N`: seeds weight init + shuffle only (split fixed); suffix `_seedN`; also on 03/04. Production = 5 seeds via `run_seed_sweep.py`, averaged with `shared/seed_aggregation.py`.
- Study-only flags (suffix outputs, never production): `--model-size {xxsmall,xsmall,small,medium,large,ffn_narrow,ffn_std,layers2,layers4,layers6,dropout10,dropout20,dropout30}` → `model_{size}` block (see hyperparameter-tuning doc); `--capacity-matched` → `model_capacity_matched` (retired ablation control; outputs kept on disk, not plotted/rerun).

## Step 3 — Prediction

1. Load `paths.best_model` (or `best_model_seedN.pt` with `--seed`) and `test.pkl`.
2. `WindowedDataset` stride 1; keep only last-position prediction; first `seq_len−1` steps per segment = NaN.
3. Inverse: z-score → `expm1` → `discharge × drainage_area`.
4. Save `outputs/amazon_domain/predictions/amazon_test_predictions.parquet`: `station_id, year, month, discharge_pred, active_fire_count_pred, burned_area_pred`, time-ordered per station.

## Step 4 — Evaluation

1. Predictions from `paths.predictions`; ground truth from `test.pkl`, inverted as in Step 3, aligned on `(station_id, year, month)`.
2. Per-station, per-target RMSE/NSE/KGE/PBIAS → `outputs/amazon_domain/evaluation/metrics_test.csv` (`station_id, target, RMSE, NSE, KGE, PBIAS`).
3. Plots to `outputs/amazon_domain/evaluation/`: metric boxplots per target; time series for 2–3 test stations; train/val/test station map on a regional basemap (lat/lon from the gpkg, read via `sqlite3.deserialize` — no geopandas, no local write).

## Outputs (`outputs/amazon_domain/`)

| Path | Contents |
|---|---|
| `preprocessed/{train,val,test}.pkl` | Normalised splits (Step 1 §10) |
| `scaler.pkl` | `{"mean","std"}` `(17,)`, train-fit |
| `models/best_model{_seedN}.pt` | Checkpoint |
| `predictions/amazon_test_predictions{_seedN}.parquet` | Physical-unit predictions |
| `evaluation/` | `metrics_test.csv`, figures, `history.csv` |
