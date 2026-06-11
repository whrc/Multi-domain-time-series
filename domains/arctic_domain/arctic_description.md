# Arctic Domain: TEM Emulation with Deep Learning

## Overview

Train a transformer model to emulate the Terrestrial Ecosystem Model (TEM) for the circumpolar Arctic. The model maps gridded environmental inputs to TEM output variables across historical and projected SSP climate scenarios. Data are organised in grid folders (e.g., `H1_V10`, `H1_V7`) each covering a piece of the circumpolar region at ~4 km pixel resolution.

**Bucket:** `gs://circumpolar-readonly/raw`  
**Config:** `config/arctic_domain.yaml` — all hyperparameters, paths, and file names.

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

Each grid folder contains four subfolders:

| Subfolder | Role |
|-----------|------|
| `ssp1_2_6_mri_esm2_0/` | Inputs — SSP1-2.6 |
| `ssp5_8_5_mri_esm2_0/` | Inputs — SSP5-8.5 |
| `ssp1_2_6_mri_esm2_0_split/all_merged/` | Targets — SSP1-2.6 (historical + projected) |
| `ssp5_8_5_mri_esm2_0_split/all_merged/` | Targets — SSP5-8.5 (projected only) |

---

## Input Files

| File | Type | Notes |
|------|------|-------|
| `soil-texture.nc`, `drainage.nc`, `fri-fire.nc`, `topo.nc`, `vegetation.nc` | Static | Space only; ~44% NaN (ocean pixels) |
| `co2.nc`, `projected-co2.nc` | Dynamic (time only) | Dimension named `year`, not `time`; expanded to monthly |
| `historic-climate.nc`, `projected-climate.nc` | Dynamic (space + time) | Monthly; 365-day calendar (requires `cftime`) |
| `historic-explicit-fire.nc`, `projected-explicit-fire.nc` | Dynamic (space + time) | **Yearly** — forward-filled to monthly in preprocessing |

---

## Target Files

Located in `<grid>/<scenario>_split/all_merged/`. Suffix `_tr` = historical, `_sc` = projected.

| Variable | File | Resolution | Notes |
|----------|------|------------|-------|
| ALD | `ALD_yearly_tr.nc` / `ALD_yearly_sc.nc` | Yearly | Forward-filled to monthly |
| GPP | `GPP_monthly_tr.nc` / `GPP_monthly_sc.nc` | Monthly | |
| RECO | `RECO_monthly_tr.nc` / `RECO_monthly_sc.nc` | Monthly | |
| VEGC | `VEGC_yearly_tr.nc` / `VEGC_yearly_sc.nc` | Yearly | Forward-filled to monthly |

**Note:** Historical targets (`_tr`) only exist under the SSP1-2.6 split folder. SSP5-8.5 has projected period only.

---

## Step 0 — EDA (`00_eda.ipynb`)

Run on `H1_V10` and `H1_V7` only (`gcs.eda_grids` from config). Documents:

- Shape and coordinate dimensions of all input and target variables
- Static vs. dynamic classification; temporal ranges per scenario
- NaN patterns and spatial heatmaps
- Summary table: all variables with dimensions, temporal coverage, NaN %

### EDA Results & Decisions

- **Ocean pixels:** ~44% NaN in H1_V10, ~33% in H1_V7 — dropped before splitting.
- **Climate:** monthly, 365-day calendar — requires `cftime` for decoding.
- **Fire:** yearly — forward-filled to monthly in preprocessing.
- **CO2:** dimension named `year` — read by dim name, expanded to monthly by year-mapping.
- **Historical targets:** only under SSP1-2.6 split folder; SSP5-8.5 projected period only. Verify with data owner whether this is intentional.
- **Projected yearly target time labels** appear wrong (e.g., 1901–1976 instead of 2025–2100) — overridden to 2025–2100 in preprocessing; verify with data owner.
- **Coordinate naming:** target files use lowercase `y`/`x`; input files use uppercase `Y`/`X` — normalised in preprocessing.

---

## Step 1 — Preprocessing (`01_preprocess.py`)

**Goal:** Build per-pixel monthly sequences → pixel-based split → normalise → save as pkl files.

1. **Load** inputs and targets from GCS for each grid and each SSP scenario.
2. **Drop ocean pixels** — exclude any pixel where all target values are NaN.
3. **Build input sequences** per pixel per SSP:
   - Static: tile across all time steps → `(T, nStatic)`
   - CO2: expand yearly values to monthly → `(T, 1)`
   - Climate: align to monthly index using standard `DatetimeIndex` → `(T, nClimate)`
   - Fire: forward-fill yearly to monthly → `(T, nFire)`
   - Concatenate → `(T, nFeatures)`
4. **Build target sequences** — upsample yearly targets (ALD, VEGC) to monthly via forward-fill; align monthly targets (GPP, RECO) → `(T, 4)`.
5. **Concatenate features + targets** → `(T, nFeatures + 4)` per pixel per SSP. This keeps inputs and targets aligned in time, making splitting and windowing straightforward.
6. **Split by pixel** — randomly assign unique pixels (not time steps) to train/val/test at `train_frac`/`val_frac`/`test_frac` from config. Both SSP sequences for a pixel land in the same split. Seed from `preprocessing.random_seed`.
7. **Fit normaliser** on train set only — compute column-wise `nanmean` and `nanstd` across all concatenated train sequences (features + targets). Set std = 1 for constant columns. Save to `paths.scaler` as `{"mean": ..., "std": ...}`.
8. **Normalise** train, val, and test splits using saved mean/std.
9. **Save** splits as pickle to `paths.preprocessed_dir`: `train.pkl`, `val.pkl`, `test.pkl`.

**Run results:** needs re-run (GCS access required).

---

## Step 2 — Training (`02_train.py`)

**Goal:** Train the transformer (`models/transformer.py`) and checkpoint on validation loss. All hyperparameters from `config/arctic_domain.yaml`.

1. **Load** `train.pkl` and `val.pkl` from `paths.preprocessed_dir`.
2. **`ArcticDataset`** — sliding-window PyTorch `Dataset` over normalised per-pixel sequences:
   - Window length: `preprocessing.seq_len` months
   - Step between windows: `preprocessing.stride`
   - Each item returns `(input, target)` where input is `(seq_len, nFeatures)` and target is `(seq_len, 4)`
   - Build flat window index `(record_idx, window_start)` at init
3. **`DataLoader`** for train and val using `training.batch_size`.
4. **Initialise** transformer model (`model.*`), Adam optimizer (`training.learning_rate`), MSE loss.
5. **Training loop** for `training.num_epochs`:
   - Forward pass → MSE loss (mask NaN positions across all 4 targets) → backward → optimizer step
   - Every `training.eval_every_n_epochs` epochs: evaluate on val set; if val loss improves, save checkpoint to `paths.best_model`
   - Stop early if no improvement for `training.early_stopping_patience` consecutive evaluations
6. **Log** train and val loss per epoch.

---

## Step 3 — Prediction (`03_predict.py`)

**Goal:** Run inference on the test set and save predictions as NetCDF.

1. **Load** best checkpoint from `paths.best_model`.
2. **Load** `test.pkl`; create `ArcticDataset` and `DataLoader`.
3. **Run inference** → collect normalised predictions `(seq_len, 4)` per window.
4. **Inverse-transform** using `paths.scaler` (load `{"mean", "std"}`, apply `pred * std + mean` to target columns only).
5. **Reconstruct spatial arrays** — map predictions back to `(time, y, x)` per variable per grid per SSP.
6. **Save** to `paths.predictions` as NetCDF per variable, matching the naming convention of original TEM output files.

---

## Step 4 — Evaluation (`04_evaluate.py`)

**Goal:** Compute metrics and produce diagnostic figures on the test set.

1. **Load** test predictions and ground truth (inverse-transformed).
2. **Compute metrics** per pixel, disaggregated by SSP and by period (historical vs. projected):
   Metrics to compute: RMSE, NSE, KGE, PBIAS.

3. **Produce diagnostic plots:**
   - One boxplot figure per SSP scenario: one panel per metric, comparing historical vs. projected across all test pixels
   - Spatial NSE maps for both SSPs and both periods (historical / projected) for all 4 target variables
4. **Save** metrics CSV and all figures to `paths.evaluation`.

---

## Outputs

| Path | Contents |
|------|----------|
| `outputs/arctic_domain/preprocessed/train.pkl` | Normalised train split |
| `outputs/arctic_domain/preprocessed/val.pkl` | Normalised val split |
| `outputs/arctic_domain/preprocessed/test.pkl` | Normalised test split |
| `outputs/arctic_domain/scaler.pkl` | `{"mean": ..., "std": ...}` — fit on train |
| `outputs/arctic_domain/models/best_model.pt` | Best model checkpoint |
| `outputs/arctic_domain/predictions/` | Per-variable NetCDF predictions |
| `outputs/arctic_domain/evaluation/metrics.csv` | Per-pixel metrics for both SSPs and periods |
| `outputs/arctic_domain/evaluation/metrics_boxplot_ssp1.png` | Boxplot — SSP1-2.6 |
| `outputs/arctic_domain/evaluation/metrics_boxplot_ssp5.png` | Boxplot — SSP5-8.5 |
| `outputs/arctic_domain/evaluation/spatial_metrics_maps/` | NSE spatial maps |
