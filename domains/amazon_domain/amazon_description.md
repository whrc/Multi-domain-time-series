# Amazon Domain: Discharge and Wildfire Forecasting

## Overview

Train a transformer model to predict **discharge**, **wildfire count**, and **wildfire burned area** in the Amazon basin at the watershed level using monthly climate and fire variables.

**Bucket:** `gs://fr_v1/am_hydro_fire_risk_V2/`  
**Config:** `config/amazon_domain.yaml` — all hyperparameters, paths, and column names.

**Pipeline steps:**
| Step | File | Status |
|------|------|--------|
| EDA | `00_eda.ipynb` | Complete |
| Preprocessing | `01_preprocess.py` | Complete |
| Training | `02_train.py` | Not started |
| Prediction | `03_predict.py` | Not started |
| Evaluation | `04_evaluate.py` | Not started |

---

## Data Layout

Only two files from the bucket are used:

| File | Role |
|------|------|
| `FR_Monthly_NaturalVeg_2000_2024_v16.csv` | Primary analysis file — monthly predictors + targets per station, 2000–2024 |
| `ANA_StationList_filteredAF_nov2025.csv` | Station allowlist — column `x` = valid station IDs |

---

## Column Reference

`dyn` = dynamic predictor, `static` = static predictor, `—` = computed in preprocessing.

| Column | Raw name | Type | Notes |
|--------|----------|------|-------|
| `station_id` | `EstacaoCod` | ID | |
| `year` | `year` | time | May drop in training |
| `month` | `month` | time | May drop in training |
| `precip` | `Prec` | dyn | |
| `tmax` | `tmax` | dyn | |
| `tmin` | `tmin` | dyn | |
| `et` | `ET` | dyn | |
| `vpd` | `vpd` | dyn | |
| `drainage_area` | `DrangAr` | dyn | |
| `month_sin` | — | dyn | `sin(2π×month/12)` — cyclical encoding, preserves Dec→Jan wrap |
| `month_cos` | — | dyn | `cos(2π×month/12)` — paired with month_sin |
| `mean_precip` | — | static | Per-station long-term climate stats; helps model distinguish sites |
| `sd_precip` | — | static | |
| `mean_tmax` | — | static | |
| `sd_tmax` | — | static | |
| `mean_tmin` | — | static | |
| `sd_tmin` | — | static | |
| `discharge` | `vazao` | target | 6.3% NaN — kept as-is, masked in loss |
| `active_fire_count` | `AF` | target | |
| `burned_area` | `BA` | target | |

**Column order:** `station_id, year, month` → dynamic predictors → static predictors → targets

---

## Step 0 — EDA (`00_eda.ipynb`)

Run on raw CSV from GCS. Document:

- Station count; time span (min/max year-month per station)
- Missing values: % per column; missingness heatmap (station × year)
- Distributions for all predictors and targets
- Time series for 3–5 sample stations (all three targets)
- Seasonal patterns: monthly mean across stations
- Correlation matrix: predictors → targets
- Summary table: min/max/mean/std/% missing; flag outliers or implausible ranges

### EDA Results & Decisions

- **Missing values:** Only `discharge` has NaN. All predictors and other two targets fully observed. No imputation needed.
- **NaN handling:** Keep discharge NaN as-is. Training loop masks NaN positions when computing loss (applies to all three targets for generality).
- **No outlier drops:** Distributions are physically plausible.

---

## Step 1 — Preprocessing (`01_preprocess.py`)

**Goal:** Clean parquet + time-based train/val/test split + fitted StandardScaler.

1. **Load** primary CSV from GCS.
2. **Filter stations** — keep rows where `EstacaoCod` is in `ANA_StationList_filteredAF_nov2025.csv` (column `x`).
3. **Select and rename** — keep the 12 raw columns; rename per the table above.
4. **Add cyclical month encoding** — `month_sin = sin(2π×month/12)`, `month_cos = cos(2π×month/12)`.
5. **Add static features** — per-station mean and std of `precip`, `tmax`, `tmin`; broadcast as constant columns.
6. **Apply EDA decisions** — only `discharge` has NaN; keep as-is.
7. **Ensure temporal completeness** — reindex each station to its full monthly range; insert NaN rows for any gaps; log a warning per station with count of inserted rows.
8. **Sort** by `station_id`, `year`, `month`.
9. **Reorder columns** per the table above.
10. **Save** full dataset → `outputs/amazon_domain/preprocessed/amazon_preprocessed.parquet`.
11. **Split by time** using `train_end_year` and `val_end_year` from config (all stations in every split):
    - Train: `year ≤ train_end_year` (2000–2014, 15 years)
    - Val: `train_end_year < year ≤ val_end_year` (2015–2019, 5 years)
    - Test: `year > val_end_year` (2020–2024, 5 years)
12. **Fit StandardScaler** on train set only (all predictor and target columns; NaN rows excluded from fit for `discharge`). Transform train, val, and test splits. Save scaler to `paths.scaler` from config. Overwrite the three split parquets with scaled values.

**Run results:** 98 stations, 25,836 rows, 20 columns. Discharge NaN: 6.3%. No temporal gaps. Train 15,756 rows (2000–2014) / Val 5,184 rows (2015–2019) / Test 4,896 rows (2020–2024). Scaler fit on train.

---

## Step 2 — Training (`02_train.py`)

**Goal:** Train the transformer (`models/transformer.py`) and checkpoint on validation loss. All hyperparameters from `config/amazon_domain.yaml`.

1. **Load** scaled train/val parquets; build sliding-window sequences of length `model.seq_len` per station.
2. **Create `Dataset` and `DataLoader`** for train and val (`training.batch_size` from config).
3. **Initialize** transformer model (`model.*`), Adam optimizer (`training.learning_rate`), and MSE loss.
4. **Training loop** for `training.num_epochs`:
   - Forward pass → compute MSE loss (mask NaN target positions across all three targets) → backward → optimizer step.
   - Every `training.eval_every_n_epochs` epochs: evaluate on val set; save checkpoint to `paths.best_model` if val loss improves.
   - Stop early if val loss does not improve for `training.early_stopping_patience` consecutive evaluations.
5. **Log** train and val loss per epoch.

---

## Step 3 — Prediction (`03_predict.py`)

**Goal:** Run inference on the test set and save predictions.

1. **Load** best checkpoint from `outputs/amazon_domain/models/`.
2. **Load** scaled test parquet → create `Dataset` and `DataLoader`.
3. **Run inference** on test inputs → collect scaled predictions.
4. **Inverse-transform** predictions using the saved scaler (`paths.scaler` from config).
5. **Save** to `outputs/amazon_domain/predictions/amazon_test_predictions.parquet` with columns: `station_id, year, month, discharge_pred, active_fire_count_pred, burned_area_pred`.

---

## Step 4 — Evaluation (`04_evaluate.py`)

**Goal:** Compute metrics and produce diagnostic figures on test set predictions.

1. **Load** test predictions and ground truth (inverse-transformed) from parquet.
2. **Compute metrics** per station for each target: RMSE, MAE, NSE, KGE.
3. **Produce diagnostic plots:**
   - Boxplots + CDF plots for NSE and KGE across all stations, one panel per target.
   - Time series plots for a sample of stations (predictions vs. ground truth).
4. **Save** metrics and plots to `outputs/amazon_domain/evaluation/`.

---

## Outputs

| Path | Contents |
|------|----------|
| `outputs/amazon_domain/preprocessed/amazon_preprocessed.parquet` | Full cleaned dataset (unscaled) |
| `outputs/amazon_domain/preprocessed/train.parquet` | Scaled train split (2000–2014) |
| `outputs/amazon_domain/preprocessed/val.parquet` | Scaled val split (2015–2019) |
| `outputs/amazon_domain/preprocessed/test.parquet` | Scaled test split (2020–2024) |
| `outputs/amazon_domain/scaler.pkl` | Fitted StandardScaler (fit on train) |
| `outputs/amazon_domain/models/best_model.pt` | Best model checkpoint |
| `outputs/amazon_domain/predictions/amazon_test_predictions.parquet` | Test set predictions (original units) |
| `outputs/amazon_domain/evaluation/metrics.csv` | Per-station metrics for all targets |
| `outputs/amazon_domain/evaluation/metrics_boxplot.png` | Boxplots comparing performance across stations |
| `outputs/amazon_domain/evaluation/sample_time_series.png` | Predictions vs. ground truth for sample stations |
