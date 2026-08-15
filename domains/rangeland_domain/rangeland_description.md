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
`training.optimized_lr` is null. `run_rangeland.py` runs `01`→`04` in sequence.

---

## Config Modes

Set `mode: dev | production` in `config/rangeland_domain.yaml`.  
Model and training hyperparameters are selected by mode. Production values (as of 2026-08-12):
`hidden_dim=256, num_layers=3, num_heads=4, dropout=0.15, feedforward_dim=256`,
`batch_size=64, num_epochs=100, warmup_epochs=5, early_stopping_patience=12`, on an A100 40GB —
35 train / 11 val / 8 test sites, PFT-stratified (see Step 1 §3). These architecture values were
originally `hidden_dim=64, dropout=0.3` — a small, heavily-regularized model chosen by hand (no
grid search) to guard against overfitting on this small dataset. A real hyperparameter-tuning
sweep found that reasoning backwards: `hidden_dim=256` (now the *largest* size tested, tied with
the multi-domain shared trunk's own capacity) gives a genuine ~40% validation-loss improvement
over the original config, not a plateau — production was promoted to the exact tested
configuration. See `hyperparameter_tuning/hyperparameter_tuning_description.md` and
`project_management/key_findings_log.md` `RG-retune0812` for the full sweep and its
implication for the manuscript's Rangeland framing (flagged `NEEDS HUMAN REVIEW` — the retuned
individual model is now competitive with, and for some targets slightly better than, the
multi-domain fine-tuned model).

---

## Data Layout

Four CSV files in `RangeSTAR_data/`, one per Plant Functional Type (PFT) group:

| File | PFT Group | Notation |
|------|-----------|------|
| `NEE-desert-scrub_*.csv` | Desert scrub | US-Jo1 |
| `NEE-grass_*.csv` | Grass | US-A32 |
| `NEE-grass-tree_*.csv` | Grass-tree | US-Mpj |
| `NEE-sagebrush_*.csv` | Sagebrush | US-Hn1 |

Each file shares an identical 46-column schema combining satellite remote sensing, gridded meteorological drivers (NLDAS, Daymet), eddy covariance tower observations, and RangeSTAR process-model predictions. This project uses only the columns listed below.

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

## Input variables (predictors) and targets

Full column definitions, units, valid ranges, and sources: `RangeSTAR_data/README.md`
(Sections B/C for predictors, E/F for targets) — that file is the sole owner of the raw-CSV
data dictionary. This pipeline consumes a subset, with these pipeline-specific deltas:

- **Predictors used:** `EVI2, tsoil, sm1, sm2, clay, vpd, SW_IN_NLDAS, tavg, tmax, tmin, prcp`
  (README Sections B/C) — all gridded/remote-sensing driver columns, none of the `_obs`
  tower-observation columns (Section D) or model-uncertainty columns (Section E's
  `NEE_pred_*`/`NEE_original`).
- **`site`**: identifier only, not a predictor. **`time`**: not used raw — after monthly
  aggregation, encoded as `month_sin = sin(2π×month/12)` / `month_cos = cos(2π×month/12)`.
- **`PFT`**: one-hot encoded (4 columns, one per group, no baseline dropped).
- **`vpd`** is consumed in its native gridded unit, **hPa** (README Note 2) — the tower
  `VPD_obs` column, in kPa, is unused.
- **Targets used** (RangeSTAR process-model outputs, not observations — see Overview):
  `GPP_predicted, RECO_predicted, Rm_predicted, Rg_predicted` (fluxes, g C m⁻² d⁻¹) and
  `AGB_predicted, BGB_predicted, AGL_predicted, BGL_predicted, POC_predicted, HOC_predicted`
  (pools, g C m⁻²) — README Sections E/F. `NEE_predicted` is
  **not** a model output; it's derived at inference as `NEE = RECO_predicted − GPP_predicted`
  (README Note 1: negative = net carbon sink, positive = net carbon source).

---

## Step 0 — EDA (`00_eda.ipynb`)

Site coverage/gaps, predictor-target correlations, and per-variable data-quality checks — complete; results below.

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

3. **Site-level train/val/test split** — split at the site level (not time level) so held-out sites are fully unseen. **Algorithm:** for each PFT group independently, shuffle that group's sites with `preprocessing.random_seed`; take `n_val = max(1, round(val_frac * n))` sites for val and `n_test = max(1, round(test_frac * n))` for test (shrinking `n_test` then `n_val` if needed to leave ≥1 site for train), remainder to train. This guarantees ≥1 site per PFT per split as long as each PFT group has ≥3 sites. **Production split: 35 train / 11 val / 8 test** across the 4 PFT groups. **Note:** with so few sites per small PFT, per-PFT test metrics rest on 1–2 sites and are high-variance — interpret per-PFT results cautiously.

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

   A site with no segment reaching `preprocessing.seq_len` contributes nothing to its pkl — of the 59 EDA sites, 54 have at least one qualifying segment (production split: 35/11/8, above).

---

## Step 2 — Training (`02_train.py`)

**Goal:** Train `TransformerModel` from `shared/transformer.py` (causal encoder with sinusoidal positional encoding, shared across all domains) and checkpoint on validation loss. All hyperparameters from `config/rangeland_domain.yaml`.

1. **Load** `train.pkl` and `val.pkl` from `paths.preprocessed_dir`. `nFeatures = 22`, `nTargets = 10`.

2. **`WindowedDataset`** (shared core, not a per-domain class — see Implementation note above) — sliding-window PyTorch `Dataset` over normalised per-site segments:
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

**`--flux-only` mode:** train on GPP/RECO/Rm/Rg only, dropping the 6 pool targets (AGB, BGB, AGL, BGL, POC, HOC). Reuses the existing full-target train/val pkl and scaler, sliced to the flux columns. Output checkpoint/eval-folder get a `_fluxonly` suffix. `03_predict.py`/`04_evaluate.py` accept the same flag. No accuracy difference vs. the full-target model on the flux targets specifically (`RG-5f0c3603`); recommended checkpoint for flux-only downstream use (e.g. the multi-domain model).

**`--seed` / multi-seed runs:** optional training RNG seed (weight init + minibatch shuffle order only — the site split is fixed regardless of seed). When given, seeds torch/numpy/random and appends `_seedN` to output names. `03_predict.py`/`04_evaluate.py` accept `--seed` to load the matching checkpoint. **Current production methodology runs 5 seeds** (`run_seed_sweep.py` at the repo root) and reports seed-averaged metrics via `shared/seed_aggregation.py`.

**`--model-size` / `--capacity-matched` / `--amazon-sized` (hyperparameter-tuning and ablation
studies only, mutually exclusive):** `--model-size {small,medium,large,xlarge,xxlarge}` overrides
the config's `production` block with the named `model_{size}` block (an isolated `hidden_dim`
sweep — see `hyperparameter_tuning/hyperparameter_tuning_description.md`); `--capacity-matched`
overrides it with `model_capacity_matched` (the multi-domain shared trunk's architecture) and
`--amazon-sized` overrides it with Amazon's own production architecture as a borrowed proxy —
both were superseded by the direct hyperparameter-tuning sweep above (which found this domain's
real capacity-starved baseline and fixed it directly) and are retired from the active
methodology; their checkpoints/CSVs remain on disk as a historical record but are no longer
plotted or rerun, see `ablation_test/ablation_description.md`. Each appends a matching suffix to
output names, same convention as `--seed`. None are used in production training.

---

## Step 3 — Prediction (`03_predict.py`)

**Goal:** Run inference on the test set and save predictions as a parquet dataframe with site ID, date, and predicted values. Only run when validation performance is satisfactory — the test set is used once, at the very end.

1. **Load** best checkpoint from `paths.best_model`; load `test.pkl`.

2. **Inference** — use `WindowedDataset` with **stride = 1** to densely cover the full time range of each segment. For each window, record the prediction only at the **last position** (`window_start + seq_len − 1`) — this position has seen maximum context. The first `seq_len − 1` time steps of each segment have no prediction; fill with NaN.

3. **Inverse-transform targets** — apply `pred * std[22:] + mean[22:]` using the last 10 entries of the scaler (target columns, indices 22–31).

4. **Derive NEE** — `NEE = RECO_predicted − GPP_predicted` (not a model output; computed from predictions).

5. **Save** as parquet to `outputs/rangeland_domain/predictions/predictions.parquet` with columns: `site, date` plus the 11 predicted columns (10 model targets + derived NEE) — `GPP_predicted, RECO_predicted, Rm_predicted, Rg_predicted, AGB_predicted, BGB_predicted, AGL_predicted, BGL_predicted, POC_predicted, HOC_predicted, NEE_predicted` — in correct temporal order per site. `NEE_predicted` is derived after inverse-transforming the 10 model targets (`NEE = RECO_predicted − GPP_predicted`) and appended as the 11th column. NEE is not produced by the model and does not correspond to any scaler column.

---

## Step 4 — Evaluation (`04_evaluate.py`)

**Goal:** Compute metrics and produce diagnostic figures on the test set predictions.

1. **Load** predictions from `paths.predictions`; load ground truth from `test.pkl` (inverse-transform targets using scaler).

2. **Compute metrics** per site, per target variable using `shared/metrics.py`: RMSE, NSE, KGE, PBIAS.

3. **Produce diagnostic plots** using `shared/plots.py`:
   - Metric boxplots (RMSE, NSE, KGE, PBIAS), split into flux (GPP/RECO/Rm/Rg) and pool (AGB/BGB/AGL/BGL/POC/HOC) targets — each as both a by-PFT panel and an all-PFTs-pooled panel (4 files total: `metrics_boxplot_test_{fluxes,pools}_{by_pft,pooled}.png`). Kept separate because pool RMSE (hundreds-thousands) would otherwise squash the flux boxes (RMSE ~1) onto an unreadable axis.
   - Time series plots for 1 representative test site per PFT group: grass, desert-scrub, sagebrush or grass-tree, showing predicted vs ground truth for all 10 targets.
   - Site split map (`site_map.png`) — train/val/test sites plotted by lat/lon, from `RangeSTAR_data/ameriflux_sites.geojson`.

4. **Save** metrics to `outputs/rangeland_domain/evaluation/metrics_test.csv` with id columns `{site, pft}`, plus `target` and the four metric columns `RMSE, NSE, KGE, PBIAS`. Save figures to `outputs/rangeland_domain/evaluation/` with descriptive file names.

---

## Outputs

| Path | Contents |
|------|----------|
| `outputs/rangeland_domain/preprocessed/{train,val,test}.pkl` | Normalised splits — `List[Dict{site, pft, segments, segment_starts}]` |
| `outputs/rangeland_domain/scaler.pkl` | `{"mean": np.ndarray(32,), "std": np.ndarray(32,)}` — fit on train |
| `outputs/rangeland_domain/models/best_model{_fluxonly}{_seedN}.pt` | Model checkpoint; suffixed when `--flux-only`/`--seed` are used |
| `outputs/rangeland_domain/predictions/predictions{_fluxonly}{_seedN}.parquet` | Predictions: `site`, `date`, and predicted columns (10 model targets + derived NEE, or 4 flux targets + NEE if `--flux-only`) |
| `outputs/rangeland_domain/evaluation/metrics_test.csv` | Per-site, per-target test-set metrics |
| `outputs/rangeland_domain/evaluation/` | Figures and plots, incl. `site_map.png`, `history.csv` |
