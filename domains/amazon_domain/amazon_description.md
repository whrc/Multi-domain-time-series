# Amazon Domain: Discharge and Wildfire Prediction

## Overview

Train a transformer model to predict **discharge**, **wildfire count**, and **wildfire burned area** in the Amazon basin at the watershed level using monthly climate and fire variables.

This is a **causal, same-step** model: it consumes a sequence of monthly inputs up to step *t* and predicts the targets at the same step *t* (it does not forecast future months). Evaluation is by **spatial generalization** — held-out stations the model never saw in training, scored across their full monthly record.

**Bucket:** `gs://fr_v1/am_hydro_fire_risk_V2/`  
**Config:** `config/amazon_domain.yaml` — all hyperparameters, paths, and column names.

**Pipeline steps:**
| Step | File | Status |
|------|------|--------|
| EDA | `00_eda.ipynb` | Completed |
| Preprocessing | `01_preprocess.py` | Implemented |
| Training | `02_train.py` | Implemented |
| Prediction | `03_predict.py` | Implemented |
| Evaluation | `04_evaluate.py` | Implemented |

**Implementation note (shared core):** the sliding-window dataset, training loop, and
inference are provided by the shared, multi-domain-ready core rather than per-domain
classes: `shared/dataset.py` (`WindowedDataset`, `records_to_segments`),
`shared/training.py` (`masked_mse_loss`, `run_lr_finder`, `train_model`),
`shared/inference.py` (`predict_last_position`), and `shared/evaluate.py`
(`predict_and_inverse`, `per_unit_metrics`). The numbered scripts are thin wrappers that
adapt this domain's records to that core. The LR finder runs automatically when
`training.optimized_lr` is null. `run_amazon.py` runs `01`→`04` in sequence.

---

## Config Modes

Set `mode: dev | production` in `config/amazon_domain.yaml`.  
Model and training hyperparameters are selected by mode. Production values (as of 2026-08-13):
`hidden_dim=64, num_layers=3, num_heads=4, feedforward_dim=256, dropout=0.10`,
`batch_size=256, num_epochs=100, warmup_epochs=10, early_stopping_patience=12`, sized for
~98 stations / ~18K production windows on an A100 40GB. These architecture values (originally
`hidden_dim=128, feedforward_dim=512, dropout=0.2`, hand-picked with no grid search) come from a
real hyperparameter-tuning sweep across four architecture dimensions — hidden_dim,
feedforward_dim, num_layers, and dropout — that found no size or setting with a measurable
accuracy advantage in any of them; the smallest/fastest combination found was promoted anyway,
purely for efficiency, not accuracy. See `hyperparameter_tuning/hyperparameter_tuning_description.md`
and `project_management/key_findings_log.md` `AZ-retune0813` for the full sweep and rationale.

---

## Data Layout

Only two files from the bucket are required for this domain:

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
| `year` | `year` | time | Not used as a direct predictor |
| `month` | `month` | time | Not used raw — encoded as sine/cosine |
| `precip` | `Prec` | dyn | |
| `tmax` | `tmax` | dyn | |
| `tmin` | `tmin` | dyn | |
| `et` | `ET` | dyn | |
| `vpd` | `vpd` | dyn | |
| `drainage_area` | `DrangAr` | dyn | |
| `month_sin` | — | dyn | `sin(2π×month/12)` — cyclical encoding, preserves Dec→Jan wrap |
| `month_cos` | — | dyn | `cos(2π×month/12)` — paired with month_sin |
| `mean_precip` | — | static | Per-station long-term mean; computed from each station's own records (all splits) |
| `sd_precip` | — | static | Per-station long-term std |
| `mean_tmax` | — | static | |
| `sd_tmax` | — | static | |
| `mean_tmin` | — | static | |
| `sd_tmin` | — | static | |
| `discharge` | `vazao` | target | ~6% NaN — kept as-is, masked in loss |
| `active_fire_count` | `AF` | target | MODIS Active Fire detections, summed monthly per watershed (bucket source: `MODIS_AF_LU_2000_2024_WS_v4.csv`) |
| `burned_area` | `BA` | target | MODIS Burned Area product (km²), integrated monthly per watershed (bucket source: `MODIS_BA_NatVegFar_km2_month.csv`) |

**Feature vector (exact column order in preprocessed arrays):**

| Block | Columns | Count |
|---|---|---|
| Dynamic predictors | precip, tmax, tmin, et, vpd, drainage_area, month_sin, month_cos | 8 |
| Station climatological means | mean_precip, sd_precip, mean_tmax, sd_tmax, mean_tmin, sd_tmin | 6 |
| **Total features (`nFeatures`)** | | **14** |
| Targets | discharge, active_fire_count, burned_area | 3 |
| **Total columns per row** | | **17** |

`nFeatures = 14`, `nTargets = 3`. Targets always occupy the last 3 columns (indices 14–16).

---

## Step 0 — EDA (`00_eda.ipynb`)

Run on raw CSV from GCS. Document:

- Total number of unique stations; temporal range of all variables per station; any temporal gaps; figure showing station timelines (station × year, colored by data availability)
- Correlation between all dynamic predictors and targets and show a heatmap
- Summary table of each variable, separately for predictors and targets.
- Brief description of each input and target variable, including any key properties that inform preprocessing.

### EDA Results & Decisions

- **Missing values:** Only `discharge` has NaN. All predictors and other two targets fully observed. No imputation needed.
- **NaN handling:** Keep discharge NaN as-is. Training loop masks NaN positions when computing loss (applies to all three targets for generality).
- **No outlier drops:** Distributions are physically plausible.

---

## Step 1 — Preprocessing (`01_preprocess.py`)

**Goal:** Load raw CSV → filter stations → feature engineering → station-level train/val/test split → fit scaler on train → build contiguous segments → save as pkl.

1. **Load** primary CSV from GCS.
2. **Filter stations** — keep rows where `EstacaoCod` is in `ANA_StationList_filteredAF_nov2025.csv` (column `x`).
3. **Select and rename** — keep the raw columns and rename them using the `columns.rename` map in `config/amazon_domain.yaml` (the raw→clean mapping shown in the table above is mirrored there as the authoritative, machine-readable source — no hardcoding in the script).
4. **Add cyclical month encoding** — `month_sin = sin(2π×month/12)`, `month_cos = cos(2π×month/12)`.
5. **Handle temporal gaps** — sort each station's rows by year/month ordinal. Identify gaps (non-consecutive months) by finding breaks in the ordinal sequence (`np.diff(ords) != 1`). Log total missing station-months globally. Do NOT insert NaN rows — gaps are handled naturally in step 10 by splitting at break points into separate contiguous segments.
6. **Station-level train/val/test split** — randomly split unique station IDs into train/val/test at `train_frac`/`val_frac`/`test_frac` from config. Use `preprocessing.random_seed` for reproducibility.
7. **Station climatological means** — compute per-station mean and std of `[precip, tmax, tmin]` from **each station's full record** (all time steps, regardless of split), the same way for train, val, and test stations (no global-mean substitution). This is intentional — the predictors are fully observed for all stations across all time periods, so computing statistics from the full record introduces no leakage with respect to the targets. Broadcast as constant columns across all time steps per station. *(This is separate from the scaler in step 9, which is fit on training stations only.)*
8. **Reorder columns** per the feature vector table above. Targets always last 3 columns (indices 14–16).
9. **Target transforms (production methodology)** — applied to all three targets before the scaler fit:
   - **Drainage-area normalization of `discharge`**: `discharge /= drainage_area` (specific discharge — `Q ~ precip × area × runoff coefficient`, so dividing by basin area removes most of the between-station scale variance, and generalizes to held-out test stations since `drainage_area` is a known static covariate for every station). Raised discharge test NSE from 0.014 to 0.351. The same normalization made `burned_area` markedly worse (median test NSE 0.014 → -1.08) and was **not** applied to `active_fire_count`/`burned_area` — reverted after testing (`key_findings_log.md` `AZ-5e809245`/`AZ-2ffbfcd3`).
   - **`log1p`** on all three targets (all non-negative, severely right-skewed) so the global z-score isn't dominated by a few large/volatile stations. NaN-safe, so discharge's ~6% missing rate is unaffected.
   - Model output head uses **softplus** (`model.nonneg_output: true`) since all three targets are non-negative post-inverse-transform.
   - Together these took all three targets from negative to positive test NSE (`key_findings_log.md` `AZ-71935d7c`).
10. **Fit scaler on train split only** — column-wise mean and std over all train rows, computed on the *transformed* targets from step 9 (NaN rows excluded from fit for `discharge`). Set `std = 1` where `std == 0`. Save to `paths.scaler` as `{"mean": np.ndarray(17,), "std": np.ndarray(17,)}` (shape `(17,)` = `nFeatures + nTargets` = `14 + 3`, fit column-wise over the full concatenated `[features | targets]` array). Normalise all three splits with `(data − mean) / std`.
11. **Build contiguous segments** — for each station, sort by year/month and identify runs of consecutive months (gaps found via `np.diff` on the ordinal month index; segments split at gap boundaries, no NaN rows inserted). Discard any segment shorter than `preprocessing.seq_len`. Each segment → `np.ndarray` of shape `(T_seg, 17)` with normalised values.
12. **Save** each split as pkl (`pickle.HIGHEST_PROTOCOL`) to `paths.preprocessed_dir`: `train.pkl`, `val.pkl`, `test.pkl`. Each file is `List[Dict]` with keys:
    - `station_id` (str): station identifier
    - `segments` (List[np.ndarray]): one array per contiguous segment, shape `(T_seg, 17)`
    - `segment_starts` (List[Tuple[int, int]]): `(year, month)` start for each segment (aligned with `segments`) so timestamps can be reconstructed
    - `drainage_area` (float): station's basin area, raw (pre-z-score) — needed by step 3 to undo the discharge normalization above

---

## Step 2 — Training (`02_train.py`)

**Goal:** Train `TransformerModel` from `shared/transformer.py` (causal encoder with sinusoidal positional encoding, shared across all domains) and checkpoint on validation loss. All hyperparameters from `config/amazon_domain.yaml`.

1. **Load** `train.pkl` and `val.pkl` from `paths.preprocessed_dir`. `nFeatures = 14`, `nTargets = 3`.

2. **`WindowedDataset`** (shared core, not a per-domain class — see Implementation note above) — sliding-window PyTorch `Dataset` over normalised per-station segments:
   - Window length: `preprocessing.seq_len`
   - Step: `preprocessing.stride`
   - For each station dict, iterate over its `segments` list. For each segment of length `T_seg ≥ seq_len`: generate windows at each valid start.
   - Each item: `input = segment[start:start+seq_len, :14]` → `(seq_len, 14)`; `target = segment[start:start+seq_len, 14:]` → `(seq_len, 3)`.
   - Build a flat index `[(station_idx, seg_idx, window_start)]` at init.

3. **`DataLoader`** for train and val with `training.batch_size`.

4. **Initialise** `TransformerModel(num_features=14, num_targets=3, cfg=cfg)` from `shared/transformer.py` (feedforward activation: GELU, softplus output head — see Step 1 §9); AdamW optimiser (`training.weight_decay`) with `training.optimized_lr` if set, otherwise `training.initial_lr`; linear warmup for `training.warmup_epochs` epochs then cosine decay to 0 (`training.lr_scheduler`). Device: `cuda` if available, else `cpu`.

**`--seed` / multi-seed runs:** optional training RNG seed (weight init + minibatch shuffle order only — the train/val/test station split is fixed regardless of seed). When given, seeds torch/numpy/random and appends `_seedN` to the checkpoint/eval-folder names. `03_predict.py`/`04_evaluate.py` accept `--seed` to load the matching checkpoint. **Current production methodology runs 5 seeds** (`run_seed_sweep.py` at the repo root) and reports seed-averaged metrics via `shared/seed_aggregation.py`.

**`--model-size` / `--capacity-matched` (hyperparameter-tuning and ablation studies only):**
`--model-size {xxsmall,xsmall,small,medium,large,ffn_narrow,ffn_std,layers2,layers4,layers6,dropout10,dropout20,dropout30}`
overrides the config's `production` block with the named `model_{size}` block (each an isolated
one-dimension architecture sweep — see `hyperparameter_tuning/hyperparameter_tuning_description.md`);
`--capacity-matched` overrides it with `model_capacity_matched` (the multi-domain shared trunk's
architecture) — used by `ablation_test/ablation_description.md`'s original capacity-confound
control, now superseded by the direct hyperparameter-tuning sweep above and retired from the
active methodology; its checkpoints/CSVs remain on disk as a historical record but are no
longer plotted or rerun. Both flags append a matching suffix to the checkpoint/eval-folder
names, same convention as `--seed`. Neither is used in production training.

**MLflow tracking:** all four steps log params, per-epoch/per-target losses, and artifacts to `mlruns/` (gated by `mlflow.enabled` in config, same mechanism as the other domains).

5. **LR finder** — `02_train.py` runs the LR finder **automatically** at startup when `training.optimized_lr` is null — it performs a range test starting from `training.initial_lr`, logs the suggested LR, and saves the loss-vs-LR curve to `{paths.evaluation}/lr_finder.png`. To skip the finder and use a fixed LR, set `training.optimized_lr` to the desired value in the config.

6. **Training loop** for `training.num_epochs`:
   - Forward pass: `pred = model(input)` → `(batch, seq_len, 3)` in normalised space. The `TransformerModel` from `shared/transformer.py` applies a causal mask, so each position attends only to itself and prior positions — enforcing the same-step emulator contract (prediction at position `t` uses context from positions `0` through `t` only).
   - **Loss:** `valid = ~torch.isnan(target)`; `loss = ((pred − target)[valid] ** 2).mean()` — single MSE scalar over all valid positions across all 3 targets. In practice only `discharge` contains NaN; the masked MSE is applied to all 3 targets for generality and future-proofing.
   - Backward + optimiser step.
   - Every `training.eval_every_n_epochs` epochs: compute val loss (same masked MSE, no gradients); if improved, save checkpoint to `paths.best_model`.
   - Stop early if no val improvement for `training.early_stopping_patience` consecutive evaluations.

7. **Log** train and val loss per epoch (mean across all targets, and also seperately for each target to see if all targets are being learned). At end of training: save the loss curve and validation scatter plot to `{paths.evaluation}/` (these are diagnostic plots, not part of the formal evaluation in Step 4), and also save plots for metrics such as RMSE, NSE, KGE, and PBIAS in form of box plots. Use `shared/metrics.py` for metric computation and `shared/plots.py` for all figure generation.

---

## Step 3 — Prediction (`03_predict.py`)

**Goal:** Run inference on the test set and save predictions.

`--seed N` loads the matching seeded checkpoint (`best_model_seedN.pt`) and suffixes the output parquet — see Step 2.

1. **Load** best checkpoint from `paths.best_model`; load `test.pkl`.
2. **Inference** — use `WindowedDataset` with **stride = 1** to densely cover the full time range of each segment. For each window, record the prediction only at the **last position** (`window_start + seq_len − 1`) — this position has seen maximum context. The first `seq_len − 1` time steps of each segment have no prediction; fill with NaN.
3. **Inverse-transform predictions** — undo the z-score (`pred * std[14:] + mean[14:]`), then undo `log1p` (`expm1`), then multiply `discharge` back by its station's `drainage_area` (carried on each record) to undo the area normalization — reversing Step 1 §9 in order.
4. **Save** to `outputs/amazon_domain/predictions/amazon_test_predictions.parquet` with columns: `station_id, year, month, discharge_pred, active_fire_count_pred, burned_area_pred` (derive `year`/`month` by expanding each segment from its recorded start date) in correct temporal order per station.

---

## Step 4 — Evaluation (`04_evaluate.py`)

**Goal:** Compute metrics and produce diagnostic figures on test set predictions.

1. **Load** test predictions from the prediction parquet (`paths.predictions`), and ground truth from `test.pkl` — inverse-transformed the same way as Step 3 §3 (z-score → `expm1` → `× drainage_area`) — aligned to `(station_id, year, month)` reconstructed from each segment's recorded start date. The prediction parquet holds predictions only; ground truth always comes from `test.pkl`.
2. **Compute metrics** per station for each target using `shared/metrics.py`: RMSE, NSE, KGE, PBIAS. Save to `outputs/amazon_domain/evaluation/metrics_test.csv` with columns: `station_id, target, RMSE, NSE, KGE, PBIAS`.
3. **Produce diagnostic plots:**
   - Boxplots of each metric (RMSE, NSE, KGE, PBIAS) across stations for each target.
   - Time series plots for 2–3 representative test stations (predictions vs. ground truth for all 3 targets).
   - Station map: train/val/test station sites on a regional basemap (coastlines/borders/rivers).
     Split membership comes from `train.pkl`/`val.pkl`/`test.pkl` (`station_id` per record); lat/lon
     comes from the attribute table of `main_drainage_Filtered_checkLU_v4.gpkg` (`EstacCd`,
     `Latitud`, `Longitd`) in the GCS bucket — read via `sqlite3.deserialize` (a GeoPackage is a
     SQLite database), no geopandas dependency, no local disk write. Neither the primary CSV nor
     the station allowlist carries coordinates.
4. **Save** plots to `outputs/amazon_domain/evaluation/` with descriptive filenames.

---

## Outputs

| Path | Contents |
|------|----------|
| `outputs/amazon_domain/preprocessed/{train,val,test}.pkl` | Normalised splits — `List[Dict{station_id, segments, segment_starts, drainage_area}]` (targets: area-normalized discharge, log1p, z-scored — see Step 1 §9) |
| `outputs/amazon_domain/scaler.pkl` | `{"mean": np.ndarray(17,), "std": np.ndarray(17,)}` — fit on train |
| `outputs/amazon_domain/models/best_model{_seedN}.pt` | Model checkpoint; `_seedN` suffix when `--seed` is used |
| `outputs/amazon_domain/predictions/amazon_test_predictions{_seedN}.parquet` | Predictions in physical units (post `expm1`/area de-normalization): station_id, year, month, 3 predicted target columns |
| `outputs/amazon_domain/evaluation/metrics_test.csv` | Per-station, per-target test-set metrics |
| `outputs/amazon_domain/evaluation/` | Figures, plots, `history.csv` |
