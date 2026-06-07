# Arctic Domain: TEM Emulation with Deep Learning

## Overview

Train a transformer model to emulate the Terrestrial Ecosystem Model (TEM) for the circumpolar Arctic. The model maps gridded environmental inputs to TEM output variables across historical and projected SSP climate scenarios. Data are organized in folders for each grids (or piece of circumpolar region) (e.g., `H1_V10`, `H1_V7`) each of which then contains pixels at ~4 km spatial resolution. The input and target variables are stored as NetCDF files in GCS.

**Bucket:** `gs://circumpolar-readonly/raw`  
**Config:** `config/arctic_domain.yaml` — all hyperparameters and paths must be read from here.

---

## Data Layout

Each grid folder (e.g., `H1_V10/`) contains four subfolders:

| Folder | Role |
|---|---|
| `ssp1_2_6_mri_esm2_0/` | Inputs — SSP1-2.6 |
| `ssp5_8_5_mri_esm2_0/` | Inputs — SSP5-8.5 |
| `ssp1_2_6_mri_esm2_0_split/` | Outputs/targets — SSP1-2.6 |
| `ssp5_8_5_mri_esm2_0_split/` | Outputs/targets — SSP5-8.5 |

### Input Files

| File | Nature |
|---|---|
| `soil-texture.nc`, `drainage.nc`, `fri-fire.nc`, `topo.nc`, `vegetation.nc` | Static (space only) |
| `co2.nc`, `projected-co2.nc` | Dynamic (time only) |
| `historic-climate.nc`, `projected-climate.nc` | Dynamic (space + time) |
| `historic-explicit-fire.nc`, `projected-explicit-fire.nc` | Dynamic (space + time) |

### Target Files

Located in `<split_folder>/all_merged/`. Suffix `_tr.nc` = historical; `_sc.nc` = projected.

| Variable | Resolution |
|---|---|
| `ALD_yearly_tr.nc` / `ALD_yearly_sc.nc` | Yearly |
| `GPP_monthly_tr.nc` / `GPP_monthly_sc.nc` | Monthly |
| `RECO_monthly_tr.nc` / `RECO_monthly_sc.nc` | Monthly |
| `VEGC_yearly_tr.nc` / `VEGC_yearly_sc.nc` | Yearly |

---

## Output Directory Structure

```
output/arctic_domain/
├── models/       # Best training checkpoint
├── predictions/  # Test set predictions (NetCDF)
└── evaluation/   # Metrics CSV and figures
```

---

## Step 0 — EDA (`00_eda.ipynb`)

Run on `H1_V10` and `H1_V7` only. Document:

- Shape and coordinate dimensions (time/space/both) of all input and target variables
- Check and confirm which inputs are static vs. dynamic
- Confirm temporal ranges for historical and projected periods per scenario
- NaN patterns, make map showing heatmap of missing data across space and time for each variable
- Verify that the input and target variables described to this point above are correct as per the EDA, if not need to update the description above and also the config file accordingly. Need to make sure before moving forward.
- Make a summary or table of all variables with their dimensions, temporal coverage, and NaN patterns to inform preprocessing decisions.

### EDA Results & Decisions

- Static: ~44% NaN in H1_V10 (ocean), ~33% in H1_V7. `lat`/`lon` are data_vars — skip as features.
- Climate: monthly, `365_day` calendar (requires `cftime`).
- Fire: **yearly** (not monthly) — forward-fill to monthly in preprocessing. *(Team: interpolation vs. forward-fill?)*
- CO2: dim named `year` not `time` — read by dim name, forward-fill to monthly.
- Historical `_tr` targets only exist under `ssp1_2_6_split/` and not under `ssp5_8_5_split/` — need to verify if this is intentional or not. If intentional, model will use both historical and projected periods for training under SSP1-2.6, but only projected period for SSP5-8.5. *(Team: verify with data owner.)*
- Projected ALD/VEGC time labels look wrong (1901–1976 instead of 2025–2100) — therefore use them as 2025–2100 but flag to verify. *(Team: verify with data owner.)*
- Target coords are lowercase `y`/`x`; input coords are uppercase `Y`/`X` — normalize in preprocessing.

---

## Step 1 — Preprocessing (`01_preprocess.py`)

**Goal:** Build normalized per-pixel monthly sequences and split into train/val/test sets. Note that each pixel is thought of as an independent site. This needs to be done for all pieces of the circumpolar region (all grids) looping through all folders, and for all sites/pixels within each grid (but ofcourse after dropping pixels such as ocean pixels which are only NaN), and for both ssp scenarios.

1. **Drop ocean pixels.** Exclude any pixel that is NaN across all target variables before splitting.

2. **Build input sequences.** For each pixel, concatenate all features into a monthly time series. ALso note that there are two SSP scenarios, so each site will have two input sequences and there corresponding taget sequences also.
   - Static variables: repeat across all time steps.
   - Dynamic variables: concatenate historical and projected periods in chronological order.
   - Verify that all dynamic input variables have same temporal resolution, if not raise a flag to me to make a decision on how to handle that.
   - Final input shape: `(total_months, num_features)` per pixel per SSP scenario.

3. **Build target sequences.** For each pixel, prepare targets to the monthly the same monthly time steps.
    - Output variables can have different temporal resolution, but that is fine as this can be handled by bringing all variables to the same monthly resolution.
   - Final shape: `(total_months, num_targets)` per pixel per SSP scenario.
   - Now concate the input with corresponding target so they align in time for each pixel and each SSP scenario. SO the final shape is `(total_months, num_features + num_targets)` per pixel per SSP scenario. This will make it easier to split into train/val/test and also to feed into the model later on.

4. **Split.** Randomly (but keep the random seed as per the config) assign pixels (not time steps) to train/val/test at a 70/15/15 ratio across all grids and both SSP scenarios.

5. **Normalize.** Compute mean and std from training pixels only. Apply to all splits. Save scaler parameters to disk for use in prediction and evaluation later.

6. **Dataset class.** Implement a PyTorch `Dataset` returning:
   - Input: `(batch_size, seq_len_input, num_features)`
   - Target: `(batch_size, seq_len_target, num_targets)`
   
   Sequence lengths and batch sizes come from the config.

---

## Step 2 — Training (`02_train.py`)

**Goal:** Train the transformer (`models/transformer.py`) and checkpoint on validation loss.

- Optimizer: Adam
- Loss: MSE
- LR scheduler: as specified in config
- Evaluate on validation set each epoch
- Early stopping: halt after `early_stopping_patience` epochs without improvement
- Save best checkpoint to `output/arctic_domain/models/`

All hyperparameters (`learning_rate`, `batch_size`, `num_epochs`, `early_stopping_patience`, `hidden_dim`, `num_layers`, `dropout`) from config.

---

## Step 3 — Prediction (`03_predict.py`)

**Goal:** Run inference on the test set and save predictions in NetCDF format similar to how original TEM output files are structured.

1. Load the best checkpoint from `output/arctic_domain/models/`.
2. Run inference on normalized test inputs.
3. Inverse-transform outputs using the saved training scaler.
4. Save to `output/arctic_domain/predictions/` as NetCDF, matching the variable naming and structure of the original TEM output files.

---

## Step 4 — Evaluation (`04_evaluate.py`)

**Goal:** Compute metrics and produce diagnostic figures for the test set.

**Metrics** (per pixel, disaggregated by historical vs. projected period and by SSP scenario):

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error |
| NSE | Nash-Sutcliffe Efficiency |
| KGE | Kling-Gupta Efficiency |
| PBIAS | Percentage Bias |

**Outputs:**

1. `output/arctic_domain/evaluation/metrics.csv` — per-pixel metrics for both time periods and SSP scenarios.
2. `output/arctic_domain/evaluation/metrics_boxplot.png` — Figure with one panel per metric; each panel shows two boxplots (historical vs. projected) across all test pixels. Make two of such figures, one for each SSP scenario.
3. `output/arctic_domain/evaluation/spatial_metrics_maps/` — For NSE metric, create and save spatial maps showing the geographic distribution of performance across the circumpolar region for both historical and projected periods, and for both SSP scenarios.