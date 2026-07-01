# Rangeland Domain: Process Model Emulation with Deep Learning

## Overview

Train a transformer model to emulate the RangeSTAR process model that predicts carbon fluxes and biomass pools across rangeland sites.

This is a **causal, same-step emulator**: it consumes a sequence of monthly inputs up to step *t* and predicts the RangeSTAR targets at the same step *t* (it does not forecast future months). Evaluation is by **spatial generalization** — the split is by site (PFT-stratified), so a test site is one the model never saw in training, scored across its full monthly record.

**Location:** Local CSV files in `RangeSTAR_data/` (not a GCS bucket). During preprocessing, extract required columns and convert to pkl. Preprocessed outputs go to `outputs/rangeland_domain/preprocessed/`.  
**Config:** `config/rangeland_domain.yaml` — all hyperparameters, paths, and file names.

**Pipeline steps:**
| Step | File | Status |
|------|------|--------|
| EDA | `00_eda.ipynb` | Complete |
| Preprocessing | `01_preprocess.py` | Not started |
| Training | `02_train.py` | Not started |
| Prediction | `03_predict.py` | Implemented |
| Evaluation | `04_evaluate.py` | Implemented |

**Implementation note (shared core):** the sliding-window dataset, training loop, and
inference are provided by the shared, multi-domain-ready core rather than per-domain
classes: `shared/dataset.py` (`WindowedDataset`, `records_to_segments`),
`shared/training.py` (`masked_mse_loss`, `run_lr_finder`, `train_model`),
`shared/inference.py` (`predict_last_position`), and `shared/evaluate.py`
(`predict_and_inverse`, `per_unit_metrics`). The numbered scripts are thin wrappers that
adapt this domain's records to that core. The LR finder runs automatically when
`training.optimized_lr` is null. `run_rangeland.py` runs `01`→`04` in sequence.

---

## Config Modes

Set `mode: dev | production` in `config/rangeland_domain.yaml`.  
Model and training hyperparameters are selected by mode. Production values are TBD — revisit after initial dev runs reveal data volume and training dynamics.

---

## Data Layout

Four CSV files in `RangeSTAR_data/`, one per Plant Functional Type (PFT) group:

| File | PFT Group | Notation |
|------|-----------|------|
| `NEE-desert-scrub_*.csv` | Desert scrub | US-Jo1 |
| `NEE-grass_*.csv` | Grass | US-A32 |
| `NEE-grass-tree_*.csv` | Grass-tree | US-Mpj |
| `NEE-sagebrush_*.csv` | Sagebrush | US-Hn1 |

Each file shares an identical 44-column schema combining satellite remote sensing, gridded meteorological drivers (NLDAS, Daymet), eddy covariance tower observations, and RangeSTAR process-model predictions. This project uses only the columns listed below.

### Monthly aggregation rules

Data is at approximately 5-day intervals (pentad sampling). The model works on monthly time steps; aggregate each variable as follows during preprocessing:

| Variable(s) | Aggregation |
|---|---|
| `prcp` | **Sum** — cumulative quantity (mm/month) |
| `tavg`, `tmax`, `tmin`, `tsoil`, `vpd`, `EVI2`, `sm1`, `sm2`, `SW_IN_NLDAS` | **Mean** — state/rate variables |
| `clay` | As-is (time-invariant static property) |
| All flux targets (g C m⁻² d⁻¹) | **Mean** — keeps units as monthly-mean daily rate |
| All pool targets (g C m⁻²) | **Mean** — standing stock snapshot |

---

## Input variables (predictors)

- `site`: Unique site identifier (AmeriFlux / NEON registry). Do not use as a predictor.
- `time`: Date of observation (YYYY-MM-DD, ~5-day intervals). Do not use raw — after monthly aggregation, encode as sine and cosine of month-of-year: `month_sin = sin(2π×month/12)`, `month_cos = cos(2π×month/12)`.
- `PFT`: Plant Functional Type. Categorical; one-hot encode and use as predictor. Groups: desert-scrub, grass, grass-tree, sagebrush. 4-dimensional one-hot (one binary column per PFT group: desert-scrub, grass, grass-tree, sagebrush). No baseline column is dropped; all 4 columns are retained in the feature vector (sum across the 4 columns always equals 1).
- `EVI2`: Enhanced Vegetation Index 2. Dimensionless (–0.10 to 0.38). Source: Landsat-MODIS STARFM.
- `tsoil`: Soil Temperature near surface. °C (–12.6 to 36.0). Source: NLDAS.
- `sm1`: Volumetric Soil Moisture Layer 1 (shallow). m³/m³ (0.08 to 0.44). Source: NLDAS.
- `sm2`: Volumetric Soil Moisture Layer 2 (deeper). m³/m³, but values can exceed 1.0 (0.37 to 1.32) — likely a moisture index rather than true volumetric content. Source: NLDAS.
- `clay`: Clay content of the soil profile. % (6.6 to 29.3). **Time-invariant static property.** Source: SOLUS.
- `vpd`: Vapor Pressure Deficit. hPa (0–40 range). Note: gridded `vpd` is in hPa; the tower-observed `VPD_obs` column (not used) is in kPa — do not mix units. Source: Daymet.
- `SW_IN_NLDAS`: Incoming Downward Shortwave Radiation. W/m² (scaled). Source: NLDAS.
- `tavg`: Average Air Temperature. °C. Source: Daymet.
- `tmax`: Maximum Air Temperature. °C. Source: Daymet.
- `tmin`: Minimum Air Temperature. °C. Source: Daymet.
- `prcp`: Total Precipitation. mm/day. Source: Daymet.

---

## Target variables (RangeSTAR process-model outputs)

`NEE_predicted` is excluded as a model output — it equals `RECO_predicted − GPP_predicted` exactly and is derived from predictions at inference.

### Fluxes — units: g C m⁻² d⁻¹ (model predicts monthly-mean daily rate)

- `GPP_predicted`: Predicted Gross Primary Productivity — total canopy photosynthetic carbon capture. Always ≥ 0.
- `RECO_predicted`: Predicted Ecosystem Respiration — total biotic carbon release (autotrophic + heterotrophic). Always ≥ 0.
- `Rm_predicted`: Predicted Maintenance Respiration — metabolic cost of maintaining existing plant tissues.
- `Rg_predicted`: Predicted Growth Respiration — carbon cost of synthesising new tissue.

**NEE convention (derived at inference):** `NEE = RECO − GPP`. Negative = net carbon sink; positive = net carbon source.

### Pools — units: g C m⁻²

- `AGB_predicted`: Predicted Aboveground Biomass — stems, leaves, structural tissue.
- `BGB_predicted`: Predicted Belowground Biomass — roots and structural belowground tissue.
- `AGL_predicted`: Predicted Aboveground Litter — fallen leaves and dead surface organic material.
- `BGL_predicted`: Predicted Belowground Litter — dead roots and fine organic matter below surface.
- `POC_predicted`: Predicted Particulate Organic Carbon — fast-cycling soil fraction.
- `HOC_predicted`: Predicted Humus Organic Carbon — slow-cycling passive/protected soil fraction.

---

## Step 0 — EDA (`00_eda.ipynb`)

**Goal:**
- How many unique sites per CSV and overall? What is the time range per site? Plot: date on x-axis, site on y-axis, one line per site showing available data coverage. Confirm 5-day time resolution and whether it is consistent across all sites and CSVs. Check for missing values in all predictors and targets listed above. Flag any issues that would complicate monthly aggregation (e.g., months with too few 5-day records).
- Correlation analysis between all dynamic predictors and all target variables. Predictors on x-axis, target variables on y-axis — show a correlation heatmap.
- Brief descriptive paragraph covering predictor and target variables, any data quality issues, and key properties that will inform preprocessing and modeling decisions.

### EDA Results & Decisions

**Dataset:** 29,723 records across 59 sites and 4 PFT groups (grass 39, desert-scrub 7, sagebrush 7, grass-tree 6), spanning 2002–2024. Zero missing values across all predictors and targets — no imputation needed.

**Time resolution & gaps:** Native cadence is 5-day (pentad). 1.7% of steps have gaps > 10 days; these are handled implicitly by the sparse-month filter during aggregation.

**Monthly aggregation:** 352 site-months (~1.2%) contain fewer than 4 records, mostly at site boundaries or around instrument gaps. These are dropped before aggregation to avoid biased monthly means.

**Predictors:** EVI2 is by far the strongest predictor for flux targets (r = 0.88–0.89 for GPP/RECO). Temperature variables (tmin, tsoil, tavg) contribute moderately (r ≈ 0.4–0.5). Soil moisture and precipitation are weak predictors (r < 0.2) at monthly scale. Pool targets (AGL, BGL, POC, HOC) have near-zero correlation with all dynamic predictors (r ≤ 0.27) — their dynamics are driven by long-term carbon accumulation that the transformer must learn from sequences.

**Target scales:** Fluxes are in g C m⁻² d⁻¹ (means 0.4–1.9); pools are in g C m⁻² (means 60–7,332). Per-target z-score normalisation is required before training.

**Sequence construction:** After monthly aggregation, each site's timeline is split into contiguous segments at missing months. Windows of `preprocessing.seq_len` months (from config) are slid over each segment; segments shorter than `seq_len` are discarded. No padding or variable-length sequences. *(Note: carbon pools accumulate over multiple years, so a short window may limit pool prediction — `seq_len` is configurable and may need to be increased; see Step 2.)*

---

## Step 1 — Preprocessing (`01_preprocess.py`)

**Goal:** Merge four raw CSV files → monthly aggregation → site-level train/val/test split stratified by PFT → fit scaler on train → build contiguous segments per site → save as pkl.

1. **Load & merge** — read all four CSVs keeping only `KEEP` columns (`site`, `time`, `PFT`, static, dynamic, targets). Concatenate into a single DataFrame.

2. **Monthly aggregation** — group by `(site, year_month)`. Apply aggregation rules from the table above. Drop any site-month with fewer than 4 records before aggregating.

3. **Site-level train/val/test split** — split at the site level (not time level) so held-out sites are fully unseen. Ensure each PFT group is represented in all three splits: within each PFT group, randomly assign sites to train/val/test at the configured `train_frac`/`val_frac`/`test_frac` ratios. **Guarantee at least one site in each split per PFT group**: the small PFT groups (desert-scrub 7, sagebrush 7, grass-tree 6 sites) can produce fewer than 1 val or test site by pure rounding at small fractions, so explicitly allocate ≥1 site to val and ≥1 to test before assigning the remainder to train. Use `preprocessing.random_seed` for reproducibility. **Algorithm:** for each PFT group independently, shuffle that group's sites with `preprocessing.random_seed`; assign the first shuffled site to val, the second to test, and all remaining to train. After processing all PFT groups, merge and shuffle within each split. This guarantees ≥1 site per PFT per split as long as each PFT group contains ≥3 sites (check during EDA). **Note:** with so few sites per small PFT, per-PFT test metrics rest on 1–2 sites and are high-variance — interpret per-PFT results cautiously.

4. **Site climatological means** — compute per-site means of `[prcp, tavg, vpd, tsoil, SW_IN_NLDAS]` from **each site's own records**, the same way for train, val, and test sites (no global-mean substitution). These features are derived purely from predictors, which are observed for every site, so computing them per-site is not leakage. These 5 values are static per site and tiled across all time steps. *(This is separate from the scaler in step 6, which is fit on training sites only.)*

5. **Feature engineering** — for each monthly aggregated row, build the full feature vector in this exact column order:

   | Block | Columns | Count |
   |---|---|---|
   | Dynamic predictors | EVI2, tsoil, sm1, sm2, vpd, SW_IN_NLDAS, tavg, tmax, tmin, prcp | 10 |
   | Static | clay | 1 |
   | PFT one-hot | desert-scrub, grass, grass-tree, sagebrush | 4 |
   | Cyclical time | month_sin, month_cos | 2 |
   | Site climatological means | mean_prcp, mean_tavg, mean_vpd, mean_tsoil, mean_SW_IN_NLDAS | 5 |
   | **Total features (`nFeatures`)** | | **22** |
   | Targets (always last) | GPP, RECO, Rm, Rg, AGB, BGB, AGL, BGL, POC, HOC | 10 |
   | **Total columns per row** | | **32** |

   `nFeatures = 22`, `nTargets = 10`. Targets always occupy the last 10 columns (indices 22–31).

6. **Fit scaler on train split only** — column-wise `mean` and `std` over all train rows. Set `std = 1` where `std == 0`. Save to `paths.scaler` as `{"mean": np.ndarray(32,), "std": np.ndarray(32,)}` — shape `(32,)` = `nFeatures + nTargets` = `22 + 10`, the scaler is fit column-wise over the full concatenated `[features | targets]` array. Normalise all three splits with `(data − mean) / std`.

7. **Build contiguous segments** — for each site, sort by `year_month` and identify runs of consecutive months (no gap). Discard any segment shorter than `preprocessing.seq_len`. Each segment becomes one `np.ndarray` of shape `(T_seg, 32)` with normalised values.

8. **Save** each split as pkl (`pickle.HIGHEST_PROTOCOL`) to `paths.preprocessed_dir`: `train.pkl`, `val.pkl`, `test.pkl`. Each file is `List[Dict]` with keys:
   - `site` (str): site identifier
   - `pft` (str): PFT group label
   - `segments` (List[np.ndarray]): one array per contiguous segment, shape `(T_seg, 32)`
   - `segment_starts` (List[Tuple[int, int]]): `(year, month)` start of each segment (aligned with `segments`) so dates can be reconstructed in `03_predict.py`

---

## Step 2 — Training (`02_train.py`)

**Goal:** Train `TransformerModel` from `shared/transformer.py` (causal encoder with sinusoidal positional encoding, shared across all domains) and checkpoint on validation loss. All hyperparameters from `config/rangeland_domain.yaml`.

1. **Load** `train.pkl` and `val.pkl` from `paths.preprocessed_dir`. `nFeatures = 22`, `nTargets = 10`.

2. **`RangelandDataset`** — sliding-window PyTorch `Dataset` over normalised per-site segments:
   - Window length: `preprocessing.seq_len`.
   - Step: `preprocessing.stride`
   - For each site dict, iterate over its `segments` list. For each segment of length `T_seg ≥ seq_len`: generate windows at each valid start.
   - Each item: `input = segment[start:start+seq_len, :22]` → `(seq_len, 22)`; `target = segment[start:start+seq_len, 22:]` → `(seq_len, 10)`.
   - Build a flat index `[(site_idx, seg_idx, window_start)]` at init.

3. **`DataLoader`** for train and val with `training.batch_size`.

4. **Initialise** `TransformerModel(num_features=22, num_targets=10, cfg=cfg)` from `shared/transformer.py` (feedforward activation: GELU); AdamW optimiser (`training.weight_decay`) with `training.optimized_lr` if set, otherwise `training.initial_lr`; linear warmup for `training.warmup_epochs` epochs then cosine decay to 0 (`training.lr_scheduler`). Device: `cuda` if available, else `cpu`. The `TransformerModel` from `shared/transformer.py` applies a causal mask, so each position attends only to itself and prior positions — enforcing the same-step emulator contract (prediction at position `t` uses context from positions `0` through `t` only).

5. **LR finder** — `02_train.py` runs the LR finder **automatically** at startup when `training.optimized_lr` is null — it performs a range test starting from `training.initial_lr`, logs the suggested LR, and saves the loss-vs-LR curve to `{paths.evaluation}/lr_finder.png`. To use a fixed LR instead, set `training.optimized_lr` to the desired value in the config.

6. **Training loop** for `training.num_epochs`:
   - Forward pass: `pred = model(input)` → `(batch, seq_len, 10)` in normalised space.
   - **Loss:** `valid = ~torch.isnan(target)`; `loss = ((pred − target)[valid] ** 2).mean()` — single MSE scalar over all valid positions across all 10 targets.
   - Backward + optimiser step.
   - Every `training.eval_every_n_epochs` epochs: compute val loss (same masked MSE, no gradients); if improved, save checkpoint to `paths.best_model`.
   - Stop early if no val improvement for `training.early_stopping_patience` consecutive evaluations.

7. **Log** train and val loss per epoch (mean across all targets, and also separately for each target to see if all targets are being learned) At end of training: plot loss curves and a scatter plot of predicted vs actual values for the validation set, and also show plot for metrics such as RMSE, NSE, KGE, and PBIAS in form of box plots. Use `shared/metrics.py` for metric computation and `shared/plots.py` for all figure generation.

---

## Step 3 — Prediction (`03_predict.py`)

**Goal:** Run inference on the test set and save predictions as a parquet dataframe with site ID, date, and predicted values. Only run when validation performance is satisfactory — the test set is used once, at the very end.

1. **Load** best checkpoint from `paths.best_model`; load `test.pkl`.

2. **Inference** — use `RangelandDataset` with **stride = 1** to densely cover the full time range of each segment. For each window, record the prediction only at the **last position** (`window_start + seq_len − 1`) — this position has seen maximum context. The first `seq_len − 1` time steps of each segment have no prediction; fill with NaN.

3. **Inverse-transform targets** — apply `pred * std[22:] + mean[22:]` using the last 10 entries of the scaler (target columns, indices 22–31).

4. **Derive NEE** — `NEE = RECO_predicted − GPP_predicted` (not a model output; computed from predictions).

5. **Save** as parquet to `outputs/rangeland_domain/predictions/predictions.parquet` with columns: `site, date` plus the 11 predicted columns (10 model targets + derived NEE) — `GPP_predicted, RECO_predicted, Rm_predicted, Rg_predicted, AGB_predicted, BGB_predicted, AGL_predicted, BGL_predicted, POC_predicted, HOC_predicted, NEE_predicted` — in correct temporal order per site. `NEE_predicted` is derived after inverse-transforming the 10 model targets (`NEE = RECO_predicted − GPP_predicted`) and appended as the 11th column. NEE is not produced by the model and does not correspond to any scaler column.

---

## Step 4 — Evaluation (`04_evaluate.py`)

**Goal:** Compute metrics and produce diagnostic figures on the test set predictions.

1. **Load** predictions from `paths.predictions`; load ground truth from `test.pkl` (inverse-transform targets using scaler).

2. **Compute metrics** per site, per target variable using `shared/metrics.py`: RMSE, NSE, KGE, PBIAS.

3. **Produce diagnostic plots** using `shared/plots.py`:
   - Boxplots of RMSE, NSE, KGE, PBIAS across all test sites — one panel per target variable (10 panels).
   - Time series plots for 1 representative test sites per PFT group: grass, desert-scrub, sagebrush or grass-tree, showing predicted vs ground truth for all 10 targets.

4. **Save** metrics to `outputs/rangeland_domain/evaluation/metrics.csv` with id columns `{site, pft}`, plus `target` and the four metric columns `RMSE, NSE, KGE, PBIAS`. Save figures to `outputs/rangeland_domain/evaluation/` with descriptive file names.

---

## Outputs

| Path | Contents |
|------|----------|
| `outputs/rangeland_domain/preprocessed/train.pkl` | Normalised train split — `List[Dict{site, pft, segments, segment_starts}]` |
| `outputs/rangeland_domain/preprocessed/val.pkl` | Normalised val split |
| `outputs/rangeland_domain/preprocessed/test.pkl` | Normalised test split |
| `outputs/rangeland_domain/scaler.pkl` | `{"mean": np.ndarray(32,), "std": np.ndarray(32,)}` — fit on train |
| `outputs/rangeland_domain/models/best_model.pt` | Best model checkpoint |
| `outputs/rangeland_domain/predictions/predictions.parquet` | Predictions: `site`, `date`, and 11 predicted columns (10 model targets + derived NEE) |
| `outputs/rangeland_domain/evaluation/metrics.csv` | Per-site, per-target metrics |
| `outputs/rangeland_domain/evaluation/` | Figures and plots |
