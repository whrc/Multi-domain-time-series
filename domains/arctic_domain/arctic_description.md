# Arctic Domain: TEM Emulation

## Overview

- **Task:** transformer emulator of the Terrestrial Ecosystem Model (TEM), circumpolar Arctic, ~4 km grid tiles (e.g. `H1_V10`), historical + projected SSP scenarios.
- **Model:** causal, same-step — monthly inputs up to *t* → TEM targets at *t* (no forecasting).
- **Evaluation:** spatial generalization. Whole grid tiles go to one split (latitude-stratified, see `arctic_description_data_handling.md` §3), so test pixels sit in regions never trained on; scored over the full historical + projected range. Not a temporal-extrapolation test.
- **Bucket:** `gs://circumpolar-readonly/raw`. **Config:** `config/arctic_domain.yaml` (`mode: dev | production` selects model/training hyperparameters).
- **Production choice:** `train_size=500000`, `train_capped_stride=400`; val/test at `capped_stride=24`. Stride 400 won the 50K stride sweep and held at 500K (best val loss ~halved vs 50K, GPP NSE 0.934). 2M declined for now (insufficient disk on `vm-cpu-sandeep`). Numbers: `key_findings_log.md`.

| Step | File |
|------|------|
| EDA | `00_eda.ipynb` |
| Preprocessing | `01_preprocess.py` |
| Training | `02_train.py` |
| Prediction (opt-in) | `03_predict.py` |
| Evaluation | `04_evaluate.py` |
| Learning curve | `05_learning_curve.py` |

`run_arctic.py` runs `01`→`04` (predict only with `--include-predict`). Scripts are thin wrappers over `shared/` (`dataset.py`: `WindowedDataset`, `records_to_segments`; `training.py`: `masked_mse_loss`, `run_lr_finder`, `train_model`; `inference.py`: `predict_last_position`; `evaluate.py`; `metrics.py`; `plots.py`).

---

## Data Layout

| Grid subfolder | Role |
|-----------|------|
| `ssp1_2_6_mri_esm2_0/` | Inputs, SSP1-2.6 |
| `ssp5_8_5_mri_esm2_0/` | Inputs, SSP5-8.5 |
| `ssp1_2_6_mri_esm2_0_split/all_merged/` | Targets, SSP1-2.6 (historical + projected) |
| `ssp5_8_5_mri_esm2_0_split/all_merged/` | Targets, SSP5-8.5 (projected only; historical = SSP1-2.6) |

**Inputs** (monthly model; CO2 and climate from MRI-ESM2-0, CMIP6 ScenarioMIP; static layers have no source attribution):

| File | Type | Notes |
|------|------|-------|
| `soil-texture.nc`, `drainage.nc`, `fri-fire.nc`, `topo.nc`, `vegetation.nc` | Static | ~44% NaN (ocean) |
| `co2.nc`, `projected-co2.nc` | Time only | Yearly, dim `year` (int 1901–2024 / 2025–2100); interpolated to monthly. Included (strong predictor, SSP driver) |
| `historic-climate.nc`, `projected-climate.nc` | Space × time | Monthly, `noleap`; `tair`, `precip`, `nirr`, `vapor_press` (`lat`/`lon` are metadata, not inputs) |
| `historic-explicit-fire.nc`, `projected-explicit-fire.nc` | Space × time | Yearly; **excluded** (near-constant spatial mean) |

**Targets** (TEM simulation outputs, not observations; `_tr` = historical, `_sc` = projected; `_tr` only under SSP1-2.6):

| Variable | Files | Resolution | Loss positions |
|----------|------|------------|-------|
| ALD | `ALD_yearly_{tr,sc}.nc` | Yearly | January only |
| GPP | `GPP_monthly_{tr,sc}.nc` | Monthly | Every month |
| RECO | `RECO_monthly_{tr,sc}.nc` | Monthly | Every month |
| VEGC | `VEGC_yearly_{tr,sc}.nc` | Yearly | January only |

No target resampling; the masked loss handles mixed resolutions.

---

## Step 0 — EDA (`00_eda.ipynb`)

Grids `H1_V10`, `H1_V7` (`gcs.eda_grids`). Covers shapes/dims, NaN patterns and heatmaps, input–target correlations, and summary tables (static / dynamic / targets). Decisions:
- CO2 included (interpolated); climate included (r = 0.67–0.96 vs GPP/RECO); fire excluded.
- ALD/VEGC projected files have wrong time labels (1901..1976) → override to 2025–2100. GPP/RECO labels are correct.
- Inputs use `Y`/`X`, targets `y`/`x` → normalise.
- Ocean: 44.6% NaN (H1_V10), 32.6% (H1_V7) → drop pixels whose targets are all NaN.

---

## Step 1 — Preprocessing (`01_preprocess.py`)

Per-pixel monthly sequences → grid-level split → normalise → pkl. Mechanics (passes, split, sizing, sidecars, caches, VMs): `arctic_description_data_handling.md`.

- **Time spans:** SSP1-2.6 T=2400 (1901-01→2100-12); SSP5-8.5 T=912 (2025-01→2100-12).
- **Grids:** dev = `preprocessing.dev.grids` (6 grids across latitudes; one grid can't fill all splits). Production = auto-discover all bucket grids minus `KNOWN_BROKEN_GRIDS`/`FLAKY_GRIDS_20260710` → 252 grids. `--grids a,b,...` overrides.
- **Sizing strategy:** train/val/test are always size-capped (`train_size`/`val_size`/`test_size`; no uncapped mode, null/0 fails). Coarse `capped_stride` + round-robin-across-grids subsampling keep small sets geographically spread (data_handling §5). Every grid is visited (no early stop).

1. **Static:** merge 5 static files; `Y`/`X` → `y`/`x`; keep 2D `(y, x)` vars except `lat`/`lon`.
2. **CO2:** SSP1 = `co2.nc` + `projected-co2.nc`; SSP5 = `projected-co2.nc`. Reindex `year` to Jan-1 dates, linearly interpolate to the monthly axis → `(T,)`.
3. **Climate:** concat historic + projected on `time`; keep the 4 vars; cftime → `DatetimeIndex` via `.strftime("%Y-%m-%d")`; reindex to monthly.
4. **Targets:** SSP1 = `_tr` + `_sc`, SSP5 = `_sc` only.
   - Always mask fill values: `targets[targets <= -9000] = np.nan` (some `_sc` files lack `_FillValue` and load literal -9999).
   - If ALD/VEGC `time[0].year < 2000`, relabel to `pd.date_range("2025-01-01", periods=N, freq="YS")`.
   - Yearly targets: reindex to monthly **without fill** (values only in January). Monthly: reindex directly.
5. **Drop ocean pixels:** all target values NaN over T.
6. **Records:** `data = [static (tiled) | co2 | climate(4) | ALD GPP RECO VEGC]` → `(T, nFeatures + 4)`; targets always last 4 columns. `nStatic` is inferred at runtime (`nFeatures = data.shape[1] - 4`). Record keys: `grid, ssp, y, x` (integer indices; targets have no coords, so alignment is positional), `ny, nx, lat, lon, data`.
7. **Split** (`assign_grid_splits()`, phase 1b after all grids are visited): whole grids, latitude-stratified, 60/20/20; a pixel's SSP1 and SSP5 series share its grid's split. Data_handling §3.
   - 7a. **Staggered windowing:** each pixel's series is front-trimmed by a deterministic offset so windows land on different calendar months. Always on. Data_handling §3a.
8. **Scaler:** column `nanmean`/`nanstd` over all train-pool pixels (before subsampling); `std=0` → 1. Saved to `paths.scaler` as `{"mean", "std"}`.
9. **Subsample:** per pixel, windows = `Σ_ssp floor((T_ssp − seq_len)/stride + 1)`. Shuffle each split's pixels (`random_seed`), round-robin one pixel per grid per pass until the size target. Train uses `--train-capped-stride`/`--sweep-strides` if given; val/test always `capped_stride`. CLI: `--train-size`, `--train-capped-stride`, `--sweep-strides N,N,...`, `--capped-stride` (all splits).
10. **Normalise:** `(data − mean)/std`, then set feature-column NaNs (not targets) to 0 (= z-score mean). Target NaNs are masked in the loss.
11. **Save:** `train_{label}.pkl` every run (e.g. `train_50K.pkl`; sizes coexist). `val.pkl`/`test.pkl` written only if absent; each pkl has a `.meta.json` sidecar — a mismatched val/test sidecar fails loudly unless `--force-recompute` (data_handling §6). Format: `List[Dict]` pickle (`HIGHEST_PROTOCOL`; variable-length arrays).

---

## Step 2 — Training (`02_train.py`)

**Size-labeled outputs:** `--train-size N` (default `preprocessing.train_size`, 50K) selects `train_{label}.pkl` and labels `models/best_model_{label}.pt`, its `.run_id`, `models/val_metrics_{label}.csv`, and `evaluation/{label}/`. Pass the same value to `03`/`04`.

1. Load train/val pkls; `nFeatures = data.shape[1] - 4`.
2. `WindowedDataset`: `seq_len`/`stride` come from each pkl's sidecar (missing sidecar fails loudly). Item: input `data[s:s+seq_len, :-4]`, target `data[s:s+seq_len, -4:]`. Both SSPs contribute windows.
3. `DataLoader` with `training.batch_size`.
4. `TransformerModel(num_features, num_targets=4, cfg)` (`shared/transformer.py`: causal encoder, sinusoidal PE, GELU). AdamW (`weight_decay`), linear warmup (`warmup_epochs`) then cosine to 0. `cuda` if available.
5. **LR:** if `training.optimized_lr` is null, the LR finder runs, saves `lr_finder.png`, and clamps its suggestion to a safe range (`shared/training.py`; added after two divergence incidents, see `key_findings_log.md`). Otherwise the fixed LR is used.
6. **Loop:** masked MSE over all valid (non-NaN) target positions; grad clip (`grad_clip_norm`, default 1.0); validate every `eval_every_n_epochs`, checkpoint on improvement; stop after `early_stopping_patience` evals without improvement.
7. **Outputs:** `loss_curves.png` (mean + per-target *val* loss), `history.csv`, `val_pred_vs_true.png`, `metrics_boxplot_val.png` (RMSE/NSE/KGE/PBIAS per target; boxes historical / projected-ssp126 / projected-ssp585 via `shared/evaluate.py:metrics_df_by_period`; `obs_degenerate` rows excluded), `metrics_boxplot_val_fluxes.png` (GPP, RECO only — pool skill is much weaker and would hide flux scale).

**Flags** (direct script invocation only; `run_arctic.py` does not forward `--flux-only`/`--seed`):
- `--flux-only`: GPP+RECO only, reusing the full pkls/scaler (`_naming.py:select_flux_target_columns`/`select_flux_scaler_stats`); outputs suffixed `_fluxonly`. Same GPP/RECO accuracy as full-target; used by multi-domain.
- `--seed N`: seeds init + shuffle only (split unchanged); suffix `_seedN`. Production runs 5 seeds for flux-only only (`run_seed_sweep.py`, aggregated by `shared/seed_aggregation.py`); full-target is single-seed.
- `--model-size {small,medium,large}`: tuning only; uses `model_{size}`, suffix `_{size}`. See `hyperparameter_tuning/hyperparameter_tuning_description.md`.

---

## Step 3 — Prediction (`03_predict.py`) — opt-in

**Warning:** writes dense `(time, y, x)` NetCDF per tile (mostly NaN padding) — can reach hundreds of GB (filled 99 GB after ~65 of ~257 grids). Not needed for evaluation. Run via `--include-predict` or `--stage predict` only if gridded files are needed.

1. Load `models/best_model_{label}.pt` and `test.pkl`.
2. `WindowedDataset` with stride 1; keep the last position of each window; first `seq_len − 1` steps are NaN.
3. Inverse-transform with scaler entries `[-4:]`.
4. Reconstruct `(time, y, x)` per `(grid, ssp)` and target.
5. Save per variable/grid/SSP to `predictions/{label}/` with TEM names (`ALD_yearly`, `GPP_monthly`, ...). ALD/VEGC set to NaN except January.

---

## Step 4 — Evaluation (`04_evaluate.py`)

Recomputes predictions from the labeled checkpoint; ground truth = inverse-transformed `test.pkl` (no GCS or step-3 dependency). Outputs in `evaluation/{label}/`.

- **Positions:** ALD/VEGC January only; GPP/RECO all months. Historical = `time < 2025`, projected = `≥ 2025`.
- **Metrics** (`shared/metrics.py`) per pixel × target × SSP × period: columns `grid, y, x, lat, lon, ssp, target, period, RMSE, NSE, KGE, PBIAS` → `metrics_test.csv`.
- **Plots:** `metrics_boxplot_test.png` and `metrics_boxplot_test_fluxes.png` (same design as step 2's val boxplots); `spatial_median_nse_{ssp}_{period}.png` (all test sites at real lat/lon, coloured by median NSE across targets).
- **`prediction_sample.parquet`:** raw monthly obs vs pred, all 4 targets, both SSPs, for 50 test pixels drawn with `random_seed` from sorted unique test pixels — deterministic, survives deleting `test.pkl`.

---

## Step 5 — Learning Curve (`05_learning_curve.py`)

Find the train size where val performance saturates; that size is used for both the Arctic and multi-domain models.

```
python run_arctic.py --stage preprocess --train-size 100000
python run_arctic.py --stage train --train-size 100000   # same --train-size so train loads train_100K.pkl
# repeat for other sizes, then:
python run_arctic.py --stage learning-curve
```

- Each training run writes `models/val_metrics_{label}.csv`: columns `train_windows, ssp, period, target, RMSE, NSE, KGE, PBIAS`; **median** over val pixels per `(train_windows, ssp, period, target)`.
- `05_learning_curve.py` reads all `val_metrics_*.csv`, plots val RMSE/NSE per target vs train windows → `evaluation/learning_curve/learning_curve.png`. No training.

---

## Where Preprocessing Runs

CPU-only `vm-cpu-sandeep` (not laptop or GPU VM), wrapped by `run_preprocess_resilient.sh` (relaunches until success, resuming from caches); results go VM-to-VM to `vm-sandeep`. Details: data_handling §9–10.

---

## Outputs (`outputs/arctic_domain/`)

| Path | Contents |
|------|----------|
| `preprocessed/train_{label}.pkl` | Normalised, capped train split; sizes coexist |
| `preprocessed/val.pkl`, `test.pkl` | Capped val/test (`val_size`/`test_size`), sidecar-validated, frozen |
| `preprocessed/{name}.meta.json` | Per-pkl sidecar (data_handling §6) |
| `scaler.pkl` | `{"mean", "std"}` from full train pool |
| `models/best_model_{label}.pt`, `.run_id` | Checkpoint + MLflow run id |
| `models/val_metrics_{label}.csv` | Learning-curve row |
| `predictions/{label}/` | Opt-in NetCDF (hundreds of GB) |
| `evaluation/{label}/` | Step 2 + 4 figures/CSVs (`lr_finder.png`, `loss_curves.png`, `history.csv`, `val_pred_vs_true.png`, `metrics_boxplot_{val,test}[_fluxes].png`, `metrics_test.csv`, `spatial_median_nse_{ssp}_{period}.png`, `prediction_sample.parquet`) |
| `evaluation/learning_curve/learning_curve.png` | Val metric vs train size |

`{label}` gains `_fluxonly` / `_seedN` / `_{size}` suffixes with those flags.
