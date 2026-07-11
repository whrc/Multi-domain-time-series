# Arctic Domain: TEM Emulation with Deep Learning

## Overview

Train a transformer model to emulate the Terrestrial Ecosystem Model (TEM) for the circumpolar Arctic. The model maps gridded environmental inputs to TEM output variables across historical and projected SSP climate scenarios. Data are organised in grid folders (e.g., `H1_V10`, `H1_V7`), each covering a patch of the circumpolar region at ~4 km resolution.

This is a **causal, same-step emulator**: it consumes a sequence of monthly inputs up to step *t* and predicts the TEM targets at the same step *t* (it does not forecast future steps). Evaluation is by **spatial generalization** — the train/val/test split assigns **whole grid tiles** to one split each, stratified by latitude (see step 1, item 7, and `arctic_description_data_handling.md` §3 for the full mechanism), so a test pixel is not just unseen itself but sits in a region the model never trained on any part of; its predictions are scored across the full time range, over both the historical and projected periods. This measures how well the emulator reproduces TEM at unseen *regions*, **not** temporal extrapolation skill.

**Bucket:** `gs://circumpolar-readonly/raw`  
**Config:** `config/arctic_domain.yaml` — all hyperparameters, paths, and file names.

**Pipeline steps:**
| Step | File | Status |
|------|------|--------|
| EDA | `00_eda.ipynb` | Complete |
| Preprocessing | `01_preprocess.py` | Implemented |
| Training | `02_train.py` | Implemented |
| Prediction | `03_predict.py` | Implemented |
| Evaluation | `04_evaluate.py` | Implemented |
| Learning Curve | `05_learning_curve.py` | Not started |

**Implementation notes (shared core + Arctic specifics):**
- The sliding-window dataset, training loop, and inference come from the shared,
  multi-domain-ready core: `shared/dataset.py` (`WindowedDataset`, `records_to_segments` —
  the single-array `data` record is treated as one segment), `shared/training.py`
  (`masked_mse_loss`, `run_lr_finder`, `train_model`), `shared/inference.py`
  (`predict_last_position`), and `shared/evaluate.py`. The numbered scripts are thin
  wrappers; the LR finder runs automatically when `training.optimized_lr` is null;
  `run_arctic.py` runs `01`→`04` in sequence.
- **Positional alignment:** target files carry no x/y coordinate values, so all variables
  (same per-grid shape) are aligned by integer index. `y`/`x` in records are integer grid
  indices; `lat`/`lon`/`ny`/`nx` are stored per record for evaluation and reconstruction.
- **Feature-NaN imputation:** sparse NaNs in feature columns on land pixels (e.g. fire
  fields) are set to 0 (the post-z-score mean) so model inputs are finite; target NaNs are
  preserved and masked by the loss.
- **Evaluation source:** `04_evaluate.py` recomputes predictions from the checkpoint and
  uses `test.pkl` (inverse-transformed) as ground truth instead of re-reading the saved
  NetCDF and GCS — results are identical, with no GCS dependency at evaluation time.

---

## Config Modes

Set `mode: dev | production` in `config/arctic_domain.yaml`.
Model and training hyperparameters are selected by mode.

**Production choice (as of 2026-07-10, `AR-500Kstride400-0710`):** `train_size=500000`,
`train_capped_stride=400` (val/test stay at the config default `capped_stride=24` — see
`arctic_description_data_handling.md` §5/§6). `stride=400` won a 9-point sweep at 50K
(`AR-gridsplitsweep0710`, `AR-gridsplit4005000710`) and confirmed the win scaled to 500K, where
it substantially outperformed the 50K/`stride=400` baseline on every metric (best val loss
nearly halved, GPP NSE reached 0.934). A further 2M scale-up was considered and explicitly
declined for now (disk headroom on `vm-cpu-sandeep` was insufficient without a resize) — 500K is
the current settled scale. See `key_findings_log.md` for the full numbers.

---

## Data Layout

Each grid folder contains four subfolders:

| Subfolder | Role |
|-----------|------|
| `ssp1_2_6_mri_esm2_0/` | Inputs — SSP1-2.6 |
| `ssp5_8_5_mri_esm2_0/` | Inputs — SSP5-8.5 |
| `ssp1_2_6_mri_esm2_0_split/all_merged/` | Targets — SSP1-2.6 (historical + projected) |
| `ssp5_8_5_mri_esm2_0_split/all_merged/` | Targets — SSP5-8.5 (projected only; historical is identical to SSP1-2.6) |

---

## Input Files

The model runs at monthly resolution. CO2 (yearly) is linearly interpolated to monthly. Fire (yearly) is excluded. All remaining predictors are natively monthly or static.

| File | Type | Notes |
|------|------|-------|
| `soil-texture.nc`, `drainage.nc`, `fri-fire.nc`, `topo.nc`, `vegetation.nc` | Static | Spatial only; ~44% NaN (ocean pixels) |
| `co2.nc`, `projected-co2.nc` | Dynamic (time only) | Yearly (dim `year`, integers 1901–2024 / 2025–2100); **included** — linearly interpolated to monthly; strong predictor and key SSP scenario driver (CO2 fertilisation effect) |
| `historic-climate.nc`, `projected-climate.nc` | Dynamic (space × time) | Monthly, `noleap` calendar; variables: `tair`, `precip`, `nirr`, `vapor_press` (`lat`/`lon` also in file — coordinate metadata, not model inputs) |
| `historic-explicit-fire.nc`, `projected-explicit-fire.nc` | Dynamic (space × time) | Yearly; **excluded** — near-constant spatial mean, all other predictors already show strong correlations |

---

## Target Files

Located in `<grid>/<scenario>_split/all_merged/`. Suffix `_tr` = historical, `_sc` = projected.

| Variable | File | Resolution | Notes |
|----------|------|------------|-------|
| ALD | `ALD_yearly_tr.nc` / `ALD_yearly_sc.nc` | Yearly | Model predicts monthly; loss computed at yearly positions only (no monthly data exists) |
| GPP | `GPP_monthly_tr.nc` / `GPP_monthly_sc.nc` | Monthly | Loss computed monthly |
| RECO | `RECO_monthly_tr.nc` / `RECO_monthly_sc.nc` | Monthly | Loss computed monthly |
| VEGC | `VEGC_yearly_tr.nc` / `VEGC_yearly_sc.nc` | Yearly | Model predicts monthly; loss computed at yearly positions only (no monthly data exists) |

**Notes:**
- Historical targets (`_tr`) exist only under the SSP1-2.6 split; SSP5-8.5 has projected period only.
- No resampling needed for targets — the multi-objective loss handles mixed temporal resolutions via masking.

---

## Step 0 — EDA (`00_eda.ipynb`)

Run on `H1_V10` and `H1_V7` only (`gcs.eda_grids` from config).

**Goals:**
- Shape, coordinate dimensions, and sample values for every input and target variable
- NaN patterns and spatial heatmaps
- Correlation of all dynamic inputs with each target variable
- Three summary tables (one each for static inputs, dynamic inputs, and targets): variable, dimensions, temporal coverage, NaN %
- Brief descriptive paragraph for each file: resolution, nature, and any key properties that inform preprocessing

**EDA Decisions:**
- **CO2:** Yearly (dim `year`, integers). Linearly interpolated to monthly. Strong predictor confirmed in correlation analysis; also a key differentiator between SSP scenarios. **Included.**
- **Climate:** Monthly, `noleap` calendar. Variables: `tair`, `precip`, `nirr`, `vapor_press` — strong correlations (r = 0.67–0.96 vs GPP/RECO). `lat`/`lon` fields in file are coordinate metadata — not model inputs. **Included.**
- **Fire:** Yearly; **Excluded.**
- **Projected yearly target labels:** ALD and VEGC projected have wrong labels (1901-01-01..1976-01-01 instead of 2025-01-01..2100-01-01). Override time axis to 2025–2100 in preprocessing. GPP/RECO projected labels are correct.
- **Coordinate naming:** Inputs use uppercase `Y`/`X`; targets use lowercase `y`/`x`. Normalise in preprocessing.
- **Ocean pixels:** 44.6% NaN in H1_V10, 32.6% in H1_V7. Drop pixels where all target values are NaN.

---

## Step 1 — Preprocessing (`01_preprocess.py`)

**Goal:** Build per-pixel monthly sequences of features and targets → pixel-based split → normalise → save as pkl.

**Time spans per scenario:**
- SSP1-2.6: T = 2400 months (1901-01 → 2100-12) — historical + projected
- SSP5-8.5: T = 912 months (2025-01 → 2100-12) — projected only (no historical targets exist)

**Grids:** Dev mode: `preprocessing.dev.grids` in config (currently 6 grids spanning multiple latitude bands — `H1_V10, H1_V7, H9_V9, H14_V6, H19_V10, H23_V13` — a single grid isn't enough under the whole-grid split, since it would land entirely in one split and leave the others empty). Production mode: `preprocessing.grids` is omitted, so all grid folders in the GCS bucket are auto-discovered, excluding `KNOWN_BROKEN_GRIDS` (3 permanently unfetchable) and `FLAKY_GRIDS_20260710` (8 more, currently excluded — single-day observation, see `arctic_description_data_handling.md` §1), leaving 252 grids. This ensures the full circumpolar range is represented. `--grids H1_V10,H1_V7,...` (CLI) overrides auto-discovery for local/small-scale validation runs.

**Sizing strategy (why a "50K" dataset is still geographically representative):** pixel time series are never truncated or non-contiguously resampled — that would break the causal, contiguous-window requirement the model depends on. Instead, representativeness for every split (train, val, test are all always size-capped — there is no uncapped/"full" train mode) is achieved by using a coarser accounting/materialization stride, `preprocessing.capped_stride`, so each pixel contributes far fewer windows than a stride=1 full scan would (stride=1 gives ~3,290 windows/pixel; `capped_stride=24` gives ~138). A fixed window budget then needs many more pixels — and therefore many more grids — to satisfy, via the same round-robin-across-grids subsampling described in step 9. `preprocessing.train_size` must be a positive int (config default: `50000`, the smallest capped size, chosen so a bare/accidental run stays cheap) — `01_preprocess.py` fails loudly if it's null/0, since an uncapped fetch would be hundreds of GB and hours long. Because reaching even a 50K-window target at `capped_stride` density needs more pixels than there are grids, preprocessing always visits every grid — there is no early-stop optimization here (an earlier version had one; it was removed because it structurally conflicts with representativeness once a coarse accounting stride is used, and because early-stopping also made the scaler's train pool vary by `train_size`, contradicting step 8 below). Each pkl's actual `stride`/`seed`/`size` is recorded in a co-located `.meta.json` sidecar (see step 11) so `02_train.py` always uses the stride that pkl was actually built with.

1. **Load static inputs** — merge all 5 static files for the grid/scenario; rename uppercase coords `Y`/`X` → `y`/`x`; keep all 2D `(y, x)` data vars, excluding `lat`/`lon` (coordinate metadata, not model inputs).

2. **Load CO2** — For SSP1-2.6: load `co2.nc` (years 1901–2024) and `projected-co2.nc` (years 2025–2100) from the scenario folder; concatenate along the year axis. For SSP5-8.5: load `projected-co2.nc` (years 2025–2100) only. Reindex the integer `year` dimension to January-1 `DatetimeIndex`, then linearly interpolate to the full monthly time axis (1901-01 → 2100-12 for SSP1-2.6; 2025-01 → 2100-12 for SSP5-8.5). CO2 data has one value per year (Jan 1 anchor). Linear interpolation fills intermediate months such that months between year Y (January) and year Y+1 (January) receive linearly spaced values between those two anchor values. Use `pandas` or `xarray` linear interpolation after reindexing to the monthly time axis. Result: a single `(T,)` CO2 series aligned with the monthly index.

3. **Load climate inputs** — concatenate `historic-climate.nc` and `projected-climate.nc` along `time`. Keep only `tair`, `precip`, `nirr`, `vapor_press` — exclude `lat`/`lon` data vars that also appear in the file. Convert `noleap` cftime index to standard `DatetimeIndex` via `.strftime("%Y-%m-%d")`; reindex to the scenario's monthly index.

4. **Load targets** — for each of ALD, GPP, RECO, VEGC:
   - SSP1: load `_tr` (historical) + `_sc` (projected), concatenate. SSP5: load `_sc` only.
   - **Mask the raw `~-9999` fill sentinel to NaN explicitly** (`targets[targets <= -9000] = np.nan`), regardless of whether the source NetCDF declares a `_FillValue` attribute. The historical (`_tr`) files do declare it (auto-masked by xarray on load), but some grids' projected (`_sc`) files don't — confirmed on a real production grid where `ssp5`'s projected file loaded literal `-9999.0` as if it were valid data while the equivalent `ssp1` file (which does declare `_FillValue`) loaded correctly. Never trust upstream fill-value metadata to be consistent across every grid/scenario; this explicit mask is what makes the ocean-pixel drop (below) and the scaler/training data trustworthy.
   - **Fix ALD/VEGC projected labels:** if `time[0].year < 2000`, the file has wrong time labels — override to `pd.date_range("2025-01-01", periods=N, freq="YS")`.
   - Yearly targets (ALD, VEGC): convert time index to Jan-1 `DatetimeIndex`; reindex to monthly index **without fill** — values appear only at January positions; all other months remain NaN.
   - Monthly targets (GPP, RECO): convert cftime index via `.strftime`; reindex to monthly index (no fill needed).

5. **Drop ocean pixels** — skip any pixel `(y, x)` where all target values across the full T time steps are NaN.

6. **Build per-pixel sequences** — for each land pixel:
   - Static: tile `(nStatic,)` to `(T, nStatic)`
   - CO2: broadcast `(T,)` to `(T, 1)`
   - Climate: slice `(T, 4)` from the aligned climate array
   - **Feature order: `[static | co2 | climate]` → `(T, nFeatures)`** where `nFeatures = nStatic + 1 + 4`. The exact value of `nStatic` is not fixed — it equals the total number of data variables across all 5 static NetCDF files (soil-texture, drainage, fri-fire, topo, vegetation). The preprocessing code infers it automatically from the data at runtime (`nFeatures = records[0]["data"].shape[1] - 4`). Confirm the actual count from EDA before production runs.
   - **Target order: `[ALD | GPP | RECO | VEGC]` → `(T, 4)`** — ALD/VEGC have NaN at all non-January months
   - Concatenate: `data = [features | targets]` → `(T, nFeatures + 4)`, targets always in the last 4 columns. After concatenating **and after** the z-scoring in point 10, fill NaN values in **feature columns only** (not target columns) with 0 in normalized space. Target NaNs are preserved for loss masking. Filling with 0 post-normalization sets imputed positions to exactly the z-score mean.
   - Store as `{"grid": str, "ssp": str, "y": int, "x": int, "ny": int, "nx": int, "lat": float, "lon": float, "data": np.ndarray(T, nFeatures+4)}` (`y`/`x` are integer grid indices; `ny`/`nx`/`lat`/`lon` support reconstruction and evaluation)

7. **Split by whole grid, latitude-stratified** (`assign_grid_splits()`) — this needs every grid's summary first, so it runs as phase 1b *after* every grid has been visited (phase 1a), not streamed grid-by-grid. Compute each grid's centroid from its land pixels' lat/lon; bin all grids into `preprocessing.split_lat_bins` (default 6) latitude quantile strata; within each stratum, shuffle grids with `preprocessing.random_seed` and cut at `train_frac`/`val_frac`/`test_frac`. Every pixel in a grid inherits that grid's split label — **a whole grid tile lands in exactly one split**, never split across train/val/test. This makes held-out grids spatially independent of training grids (not just deduplicated pixels), a genuinely stronger spatial-generalization test than the earlier per-grid-pixel split it replaced. For each `(grid, y, x)` pixel, both its SSP1-2.6 and SSP5-8.5 time series are assigned to the same split (inherited from the grid), so a pixel never appears in one split under one scenario and a different split under the other. Full mechanism, history, and the latitude-stratification rationale: `arctic_description_data_handling.md` §3.

7a. **Staggered windowing** — every pixel's raw time series is trimmed at the front by a small deterministic offset (`crc32(f"{seed}:{grid}:{y}:{x}") % stride`) before windowing, so different pixels' fixed-stride windows land on different calendar months instead of all pixels sampling the identical fixed set of month-offsets. Unconditional (not a flag) — confirmed beneficial at both 50K and 500K scale under the prior split. See `arctic_description_data_handling.md` §3a.

8. **Fit scaler on ALL available train pixels** — column-wise `nanmean` and `nanstd` over all train pixel arrays (before any train-size subsampling in the next step). Set `std = 1` where `std == 0` (constant columns). Save to `paths.scaler` as `{"mean": np.ndarray, "std": np.ndarray}`. Using the full train pool for the scaler ensures `val.pkl` and `test.pkl` are normalised consistently across all learning curve runs.

9. **Subsample pixels for every split** — train (target `preprocessing.train_size`/`--train-size`, at `--train-capped-stride`/`--sweep-strides` if given) and val/test (targets `preprocessing.val_size`/`test_size`, always at `preprocessing.capped_stride`) are all size-capped, so compute the window count per pixel as `sum over SSPs of floor((T_ssp − seq_len) / stride + 1)`. Shuffle each split's own assigned grids' pixels with `preprocessing.random_seed`, then round-robin across those grids (one pixel per grid per pass, repeating passes as needed) accumulating pixels until the cumulative window count reaches the size target — this, combined with the coarser stride, is what spreads a capped dataset across most/all of that split's assigned grids instead of a handful (see "Sizing strategy" above). Only the selected pixels are written out. This mechanism enables the learning curve experiment (see Step 5) without re-running the expensive scaler fit. CLI overrides: `--train-size N`, `--train-capped-stride N` (train only), `--sweep-strides N,N,...` (several train strides in one GCS pass), `--capped-stride N` (all splits) passed to `01_preprocess.py` override the config values at runtime.

10. **Normalise** — apply `(data − mean) / std` to all records.

11. **Save** — write the train split on every run, named by size: `train_{label}.pkl` (e.g. `train_50K.pkl`, `train_500K.pkl`, `train_2M.pkl` — label from window count via the same `50K`/`2M`-style formatting used for the split-coverage plot). Multiple train variants can coexist on disk, so different learning-curve sizes don't overwrite each other. Write `val.pkl` and `test.pkl` only if they do not already exist **and** their sidecar matches the current config (see below) — they are constant across all learning curve experiments since the pixel split and scaler are always identical for a fixed seed. Each file is `List[Dict]` with keys `{grid, ssp, y, x, ny, nx, lat, lon, data}`. Format: pickle (`HIGHEST_PROTOCOL`) — sequences are variable-length numpy arrays in nested dicts; parquet requires flat rectangular tables.

**Sidecar metadata:** every saved pkl gets a co-located `{name}.meta.json` recording `seed, stride, seq_len, grids_hash, size_target, size_label, actual_window_count, num_grids_covered, num_pixels, train_frac, val_frac, test_frac, split_unit, split_lat_bins`. This serves two purposes: (1) `02_train.py` reads a split's actual `stride`/`seq_len` from its sidecar rather than assuming the current config's stride, since different variants may have been built with different strides; (2) cache validity for `val.pkl`/`test.pkl` is checked by comparing the sidecar's `seed`/`stride`/`seq_len`/`size_target`/`grids_hash`/`train_frac`/`val_frac`/`test_frac`/`split_unit`/`split_lat_bins` against the current run, rather than trusting file existence alone. `grids_hash` (a CRC32 of the sorted grid list) exists specifically so a val/test built from a smaller or different grid set — e.g. a `--grids`-scoped debug run, or the bucket's grid list changing — is never mistaken for a match; the split-fraction and `split_unit`/`split_lat_bins` fields likewise catch a split-mechanism or stratification change that would otherwise leak pixels between a stale val/test and a freshly-regenerated train. **If any of these mismatch, `01_preprocess.py` fails loudly with a field-level diff and refuses to proceed** — `val.pkl`/`test.pkl` are never silently regenerated, since a different pixel population would silently break comparability with every prior result; pass `--force-recompute` to intentionally rebuild them. See `arctic_description_data_handling.md` §6 for why this is enforced this strictly (a real incident, not a hypothetical). `actual_window_count`/`num_pixels`/`num_grids_covered` are computed from what was actually written to the pkl (not the pre-selection target), so they can't overstate the real contents if a grid's pass-2 fetch failed.

**Per-grid resumability caches** (separate from the sidecar above, used internally by `01_preprocess.py`, not by `02_train.py`): `.grid_pass1_summary_cache/{grid}.pkl` holds pass 1a's small, unconditional derived summary per grid (pixel keys, lat/lon, scaler contribution — a few hundred KB even for a huge grid, unlike the multi-GB raw fetch), keyed on a bare schema-version constant — since the summary no longer depends on the split decision, changing `random_seed`, the split fractions, or `split_lat_bins` no longer forces a full re-fetch, only phase 1b (cheap, in-memory) re-runs. `.grid_pass2_records_cache/{grid}.pkl` holds pass 2's already-filtered, normalised records for that grid's wanted pixels, keyed on exactly which `(y, x)` pixels were wanted from that grid (**exact match only — no subset reuse**, so growing the wanted set re-fetches the whole grid) — so re-running with a different `--train-size`/stride (the documented learning-curve workflow, step 5) invalidates and recomputes rather than silently reusing a smaller/different pixel selection from an earlier size. `.grid_failed_cache/{grid}.failed` marks a grid that exhausted retries, expiring after an hour so a transient failure doesn't permanently exclude a grid across the many restarts a multi-hour resilient run involves. All three self-invalidate via their key/expiry — no manual cache-clearing is needed even when switching `--train-size`, `random_seed`, or the split fractions between runs. This cache can grow to tens of GB on a large run — see `arctic_description_data_handling.md` §9 for disk planning.

---

## Step 2 — Training (`02_train.py`)

**Goal:** Train the transformer defined in `shared/transformer.py` (causal encoder with sinusoidal positional encoding, shared across all domains) and checkpoint on validation loss. All hyperparameters from `config/arctic_domain.yaml`.

**Size-labeled outputs:** every run is labeled by its `--train-size` (the same label used for `train_{label}.pkl`, e.g. `50K`; omitting `--train-size` falls back to `preprocessing.train_size` from config, currently `50000`/`50K`). This label is threaded through `02_train.py`, `03_predict.py`, and `04_evaluate.py` so outputs from different sizes never collide or get silently overwritten: checkpoint `models/best_model_{label}.pt`, its `.run_id` sidecar, the learning-curve row `models/val_metrics_{label}.csv`, and everything under `evaluation/{label}/` (both this step's own figures and step 4's evaluation figures land in the same labeled folder). `--train-size` on `03_predict.py`/`04_evaluate.py` selects which labeled checkpoint to load — pass the same value used for training.

1. **Load** the train and val pkl variants from `paths.preprocessed_dir`. `--train-size N` (mirrors `01_preprocess.py`) selects which `train_{label}.pkl` variant to load; omit to fall back to `preprocessing.train_size` from config. Infer `nFeatures = records[0]["data"].shape[1] - 4` — last 4 columns are always targets.

2. **`ArcticDataset`** — sliding-window PyTorch `Dataset` over normalised per-pixel sequences:
   - Window length and step (`seq_len`, `stride`) are read from each pkl's own `.meta.json` sidecar (see step 11), **not** from the current config — every variant (train at any size, val, test) uses `preprocessing.capped_stride`, but reading it from the sidecar rather than assuming the current config avoids silently mismatching a pkl built under a different `capped_stride`. Loading fails loudly if a sidecar is missing, since silently falling back to config could train on the wrong window density.
   - Each item: `input = data[start:start+seq_len, :-4]` → `(seq_len, nFeatures)`; `target = data[start:start+seq_len, -4:]` → `(seq_len, 4)`
   - Build flat window index `[(record_idx, window_start)]` at init; both SSP1 (T=2400) and SSP5 (T=912) sequences contribute windows

3. **`DataLoader`** for train and val with `training.batch_size`.

4. **Initialise** `TransformerModel(num_features=nFeatures, num_targets=4, cfg=cfg)` from `shared/transformer.py` (feedforward activation: GELU); AdamW optimiser (`training.weight_decay`) with `training.optimized_lr` if set, otherwise `training.initial_lr`; linear warmup for `training.warmup_epochs` epochs then cosine decay to 0 (`training.lr_scheduler`). Device: `cuda` if available, else `cpu`.

5. **Learning rate:** `02_train.py` runs the LR finder automatically at startup when `training.optimized_lr` is null — it performs a range test, logs the suggested LR, and saves the loss-vs-LR curve to `{paths.evaluation}/lr_finder.png`. To skip the finder and use a fixed LR, set `training.optimized_lr` to the desired value in the config.

6. **Training loop** for `training.num_epochs`:
   - Forward pass: `pred = model(input)` → `(batch, seq_len, 4)` in normalised space
   - **Loss:** `valid = ~torch.isnan(target)`; `loss = ((pred - target)[valid] ** 2).mean()` — single MSE scalar over all valid positions across all 4 targets. ALD/VEGC contribute once per year (January only); GPP/RECO contribute every month.
   - Backward, then gradient clipping (`torch.nn.utils.clip_grad_norm_`, `training.grad_clip_norm`, default 1.0) before the optimiser step — guards against a sudden loss spike/divergence from a bad batch or an overly high post-warmup LR.
   - Every `training.eval_every_n_epochs` epochs: compute val loss (same masked MSE, no gradients); if improved, save checkpoint to `paths.best_model`
   - Stop early if no val improvement for `training.early_stopping_patience` consecutive evaluations

7. **Log** train and val loss per epoch (mean across all targets, and also seperately for each target's *validation* loss, to see if all targets are being learned — the per-target panel of `loss_curves.png` is validation loss, not train loss). At end of training: plot loss curves and a scatter plot of predicted vs actual values for the validation set, and save `metrics_boxplot_val.png` — one figure, RMSE/NSE/KGE/PBIAS per target, with 3 boxes each (historical / projected-ssp126 / projected-ssp585, via `shared/evaluate.py:metrics_df_by_period` + `scenario_period_label`, shared with step 4 so val and test use identical metric definitions), excluding `obs_degenerate` rows (constant-observed windows, where NSE/KGE are mathematically undefined). Also save `metrics_boxplot_val_fluxes.png` — the same boxplot restricted to the monthly flux targets (GPP, RECO); the yearly pool targets (ALD, VEGC) have much weaker per-pixel skill (see `project_management/key_findings_log.md`) and their wide axis scale otherwise hides how well the fluxes are doing. Use `shared/metrics.py` for metric computation and `shared/plots.py` for all figure generation.

---

## Step 3 — Prediction (`03_predict.py`) — OPT-IN, not part of the default pipeline

**⚠ Caution — large output:** this step reconstructs a full dense `(time, y, x)` grid per circumpolar tile, even though only a handful of that tile's pixels are actually in the test set (everything else is NaN-padded). At real scale this can reach **hundreds of GB** — it filled a 99GB VM disk after only ~65 of ~257 grids on one run. It is excluded from `run_arctic.py`'s default pipeline; pass `--include-predict` to add it to a full run, or `--stage predict` to run it standalone. **It is not required for evaluation metrics or figures** — step 4 (`04_evaluate.py`) recomputes predictions directly from the checkpoint and never reads this step's output. Only run this if you specifically need the gridded NetCDF files (e.g. for GIS/spatial tooling), and confirm the target disk has room for hundreds of GB first.

**Goal:** Run inference on the test set and save predictions as NetCDF. Only run when validation performance is satisfactory — the test set is used once, at the very end.

`--train-size N` selects which labeled checkpoint to load (`models/best_model_{label}.pt` — must match a size already trained via step 2); omit to fall back to `preprocessing.train_size` from config. Predictions are saved under `predictions/{label}/...`, keeping different sizes' NetCDFs separate.

1. **Load** best checkpoint from `models/best_model_{label}.pt`; load `test.pkl`.

2. **Inference** — use `ArcticDataset` with **stride = 1** to densely cover the full time range. For each window, record the prediction only at the **last position** (`window_start + seq_len − 1`) — this position has seen maximum context. The first `seq_len − 1` time steps of each sequence have no prediction; fill with NaN.

3. **Inverse-transform targets** — apply `pred * std[-4:] + mean[-4:]` using the last 4 entries of the scaler (target columns only, indices `−4:` of `{"mean", "std"}`).

4. **Reconstruct spatial arrays** — group test records by `(grid, ssp)`; for each group, map pixel predictions back to `(time, y, x)` for each of the 4 target variables.

5. **Save** as NetCDF per variable per grid per SSP to `paths.predictions`, matching original TEM naming convention (`ALD_yearly`, `GPP_monthly`, etc.) in correct temporal order. ALD/VEGC predictions are computed at every time step during inference but the model was never trained at non-January positions for these targets. **Set ALD/VEGC predicted values to NaN at all non-January positions before saving** — only January values are meaningful. Evaluation uses January only.

---

## Step 4 — Evaluation (`04_evaluate.py`)

**Goal:** Compute metrics and produce diagnostic figures on the test set predictions.

`--train-size N` selects which labeled checkpoint to load, same as step 3 (this step recomputes predictions from the checkpoint directly rather than reading step 3's saved NetCDFs — see the module docstring). Outputs are saved under `evaluation/{label}/`, alongside step 2's training figures for that same size.

1. **Load** ground truth from `test.pkl` (inverse-transformed to original units using the saved scaler); predictions are recomputed from the labeled checkpoint, not read from step 3's NetCDF output.

2. **Temporal position selection:**
   - ALD, VEGC: extract predictions and ground truth at **January positions only** (one value per year) — model was not trained on other months for these variables
   - GPP, RECO: use all monthly positions
   - Periods: historical = `time < 2025`; projected = `time ≥ 2025`

3. **Compute metrics** per pixel, per target variable, per SSP, per period using `shared/metrics.py`: RMSE, NSE, KGE, PBIAS. Store results in a DataFrame using the project-wide metrics schema — id columns `{grid, y, x, lat, lon, ssp}`, plus `target`, `period` (`historical`/`projected`), and the four metric columns `RMSE, NSE, KGE, PBIAS` (uppercase).

4. **Produce diagnostic plots** using `shared/plots.py`:
   - One combined boxplot (`metrics_boxplot_test.png`), all metrics, 3 boxes per target: historical, projected-ssp126, projected-ssp585 — same design and metric definitions as step 2's `metrics_boxplot_val.png`, so val and test are directly comparable and the two filenames make clear which split each is; plus `metrics_boxplot_test_fluxes.png` (GPP, RECO only — see step 2)
   - One circumpolar spatial overview map per (SSP, period) — every test site plotted at its real lat/lon, colored by its median NSE across all target variables (a single summary map per scenario/period, not one per grid — a per-grid dense-array version once generated ~2800 tiny files and 74GB of NetCDF-scale output for comparison, see step 3's caution)

5. **Save** metrics as CSV to `paths.evaluation/metrics_test.csv` and all figures to
   `paths.evaluation/`. Also saves `prediction_sample.parquet`: full monthly obs-vs-predicted
   time series (all 4 targets, both SSPs) for a small, deterministic sample of 50 test
   pixels — unlike `metrics_test.csv`'s aggregated per-pixel/target/period error metrics,
   this keeps raw values so a specific pixel's time series can still be plotted even after
   `test.pkl` is deleted to free disk space. The 50 pixels are a seeded draw
   (`preprocessing.random_seed`) over the sorted set of unique test pixels, so the sample
   is identical every time this runs against the same (frozen) `test.pkl` — the same sites
   stay directly comparable in a future multi-domain comparison.

---

---

## Step 5 — Learning Curve (`05_learning_curve.py`)

**Goal:** Determine at what training set size model performance saturates on the validation set. This is an interactive experiment — run it before committing to a train size for the final individual Arctic model (and for multi-domain). The optimal size found here is then used consistently in both pipelines.

**Workflow (user-driven, one run at a time):**
```
# Start small — pass the same --train-size to both stages so 02_train.py loads the variant
# 01_preprocess.py just generated (train_100K.pkl), not the config default (train_50K.pkl).
python run_arctic.py --stage preprocess --train-size 100000
python run_arctic.py --stage train --train-size 100000
# Inspect val metrics. If performance is already good, try smaller; if poor, go larger.
python run_arctic.py --stage preprocess --train-size 1000000
python run_arctic.py --stage train --train-size 1000000
# After all desired sizes:
python run_arctic.py --stage learning-curve  # reads saved summaries, plots curve
```

**What `02_train.py` saves per run:** after training, it computes `actual_windows = len(train_ds)` and saves `outputs/arctic_domain/models/val_metrics_{actual_windows}.csv` — a summary table with columns `train_windows, ssp, period, target, RMSE, NSE, KGE, PBIAS`. One row per `(train_windows, ssp, period, target)` combination, where `ssp` is e.g. `ssp126`/`ssp585` and `period` is `historical`/`projected`. Metrics are the mean across all val pixels for that combination. It also saves a size-keyed checkpoint copy `best_model_{actual_windows}.pt` alongside the primary `best_model.pt`.

**`05_learning_curve.py`:** reads all `val_metrics_*.csv` files from `outputs/arctic_domain/models/`; plots val RMSE and NSE per target (y) vs train window count (x); saves to `outputs/arctic_domain/evaluation/learning_curve/learning_curve.png`. Does not run training itself.

---

## Where Preprocessing Runs

Preprocessing (`01_preprocess.py`) never touches the GPU — it's network- and CPU-bound (fetching +
windowing), not compute-bound. Per the project's compute placement policy (`environment_spec.md`,
effective 2026-07-08), it runs on the CPU-only `vm-cpu-sandeep` (32 vCPU / 128GB RAM) via SSH —
**not** on the laptop, and not on the GPU `vm-sandeep`. The laptop only orchestrates (start/stop
the VMs, SSH in, monitor logs). Every variant (`train_50K.pkl`, `train_500K.pkl`, `train_2M.pkl`,
and `val.pkl`/`test.pkl` at their fixed 50K cap) is size-capped but still ranges from tens of MB
to tens of GB at production scale — see `arctic_description_data_handling.md` §9 for disk
planning before a big run.

**Resilience:** `domains/arctic_domain/run_preprocess_resilient.sh` wraps `01_preprocess.py`,
relaunching it until it exits successfully — useful even on the VM for transient per-attempt
failures (a grid exhausting its retry budget, a stockout-style GCS blip), not just the
random-SIGKILL failure mode this script was originally written for in local Claude-Code-tool
sessions (that specific issue, observed pre-2026-07-08, doesn't occur in a real terminal/tmux/SSH
session on the VM — see the script's own docstring). Every restart resumes almost instantly from
the pass-1/pass-2 caches instead of re-fetching from GCS. Example (run via SSH on
`vm-cpu-sandeep`, typically with `nohup ... &` so it survives the SSH session ending):
```
domains/arctic_domain/run_preprocess_resilient.sh --train-size 500000 --train-capped-stride 400 --max-workers 12
```

**Prerequisite:** the VM needs GCS read access to `gs://circumpolar-readonly/raw` via Application
Default Credentials (`gcloud auth application-default login` for project `spherical-berm-323321`,
once). `01_preprocess.py` checks this at startup and raises a clear error if it's missing.

**Transferring results to `vm-sandeep` for training** (VM-to-VM, never through the laptop —
manual, not automated by any script): resolve `vm-sandeep`'s internal IP from the laptop first
(`vm-cpu-sandeep` can't resolve it on its own), then `scp` directly between the VMs and verify
with `md5sum` on both sides:
```
# from the laptop:
gcloud compute instances describe vm-sandeep --zone=us-central1-f --format='value(networkInterfaces[0].networkIP)'
# then, via ssh on vm-cpu-sandeep, using that IP:
scp outputs/arctic_domain/preprocessed/train_500K_s400.pkl outputs/arctic_domain/preprocessed/train_500K_s400.meta.json \
    outputs/arctic_domain/preprocessed/val.pkl outputs/arctic_domain/preprocessed/val.meta.json \
    outputs/arctic_domain/preprocessed/test.pkl outputs/arctic_domain/preprocessed/test.meta.json \
    outputs/arctic_domain/scaler.pkl \
    sp2596@<vm-sandeep-internal-ip>:~/Multi-domain-time-series/outputs/arctic_domain/preprocessed/
```
(confirm the exact remote path and label against your checkout before running).

**Local runs are still possible** (e.g. quick dev-mode iteration on a small `--grids` subset) but
are no longer the default workflow — see `feedback-compute-placement` in memory for the
rationale. If run locally, `caffeinate -disu &` keeps a laptop awake through a multi-hour run
(macOS Power Nap can otherwise stop everything even on AC power with the lid open).

---

## Outputs

| Path | Contents |
|------|----------|
| `outputs/arctic_domain/preprocessed/train_{label}.pkl` | Normalised, size-capped train split (e.g. `train_50K.pkl`); multiple sizes coexist |
| `outputs/arctic_domain/preprocessed/{name}.meta.json` | Sidecar per pkl: seed, stride, seq_len, size target/actual, grids/pixels covered |
| `outputs/arctic_domain/preprocessed/val.pkl` | Normalised val split, capped at `val_size` (cached — sidecar-validated) |
| `outputs/arctic_domain/preprocessed/test.pkl` | Normalised test split, capped at `test_size` (cached — sidecar-validated) |
| `outputs/arctic_domain/scaler.pkl` | `{"mean": ..., "std": ...}` — always fit on full train pool |
| `outputs/arctic_domain/models/best_model_{label}.pt` | Best checkpoint for the run trained at this size (e.g. `best_model_50K.pt`); multiple sizes coexist |
| `outputs/arctic_domain/models/best_model_{label}.run_id` | MLflow run id sidecar for that checkpoint |
| `outputs/arctic_domain/models/val_metrics_{label}.csv` | Val metrics summary for the learning curve run at this size (`train_windows` column holds the real window count) |
| `outputs/arctic_domain/predictions/{label}/` | Per-variable NetCDF predictions for the run at this size — **opt-in only** (`--include-predict` or `--stage predict`), can reach hundreds of GB, not needed for evaluation |
| `outputs/arctic_domain/evaluation/{label}/` | All step 2 + step 4 figures/metrics for this size: `lr_finder.png`, `loss_curves.png`, `val_pred_vs_true.png`, `metrics_boxplot_val.png`, `metrics_boxplot_val_fluxes.png`, `metrics_test.csv`, `metrics_boxplot_test.png`, `metrics_boxplot_test_fluxes.png`, `spatial_median_nse_{ssp}_{period}.png` (one map per SSP × period, all test sites), `prediction_sample.parquet` (raw obs-vs-pred time series for 50 deterministic test pixels — see step 4) |
| `outputs/arctic_domain/evaluation/learning_curve/learning_curve.png` | Val metric vs train size saturation plot |
