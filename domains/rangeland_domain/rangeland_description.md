# Rangeland Domain: Process Model Emulation with Deep Learning

## Overview

- **Task:** causal, same-step monthly emulation of RangeSTAR process-model carbon fluxes and pools per site.
- **Evaluation:** spatial generalization — PFT-stratified site split, full monthly record per test site.
- **Data:** local CSVs in `RangeSTAR_data/` (not GCS) → pkl in `outputs/rangeland_domain/preprocessed/`. **Config:** `config/rangeland_domain.yaml` (`mode: dev | production`).
- **Scripts:** `00_eda.ipynb` → `01_preprocess.py` → `02_train.py` → `03_predict.py` → `04_evaluate.py` (all implemented; `run_rangeland.py` runs 01→04).
- **Shared core:** same as Amazon — see `domains/amazon_domain/amazon_description.md` Overview.
- **Production architecture:** `hidden_dim=256, num_layers=3, num_heads=4, dropout=0.15, feedforward_dim=256`; `batch_size=64, num_epochs=100, warmup_epochs=5, early_stopping_patience=12`; A100 40GB. Promoted from the sweep (~40% val-loss gain over original hand-picked `hidden_dim=64, dropout=0.3`). After retune, Individual ≈ multi-domain with a slight multi-domain edge (smaller margin than first reported, not reversed). See `hyperparameter_tuning/hyperparameter_tuning_description.md`.

## Data

Four CSVs (identical 46-column schema: satellite, NLDAS/Daymet met, tower obs, RangeSTAR outputs):

| File | PFT group | Notation |
|---|---|---|
| `NEE-desert-scrub_*.csv` | Desert scrub | US-Jo1 |
| `NEE-grass_*.csv` | Grass | US-A32 |
| `NEE-grass-tree_*.csv` | Grass-tree | US-Mpj |
| `NEE-sagebrush_*.csv` | Sagebrush | US-Hn1 |

Data dictionary (units, ranges, sources): `RangeSTAR_data/README.md` (sole owner). Pipeline deltas:

- **Predictors:** `EVI2, tsoil, sm1, sm2, clay, vpd, SW_IN_NLDAS, tavg, tmax, tmin, prcp` (README §B/C). No `_obs` tower columns (§D), no `NEE_pred_*`/`NEE_original`.
- `vpd` in native gridded **hPa** (tower `VPD_obs`, kPa, unused).
- `site`: ID only. `time` → `month_sin/cos = sin/cos(2π·month/12)` after aggregation. `PFT`: one-hot, 4 cols, no baseline dropped.
- **Targets** (model outputs, not observations): fluxes `GPP, RECO, Rm, Rg` (`*_predicted`, g C m⁻² d⁻¹); pools `AGB, BGB, AGL, BGL, POC, HOC` (g C m⁻²). `NEE = RECO − GPP` derived at inference (negative = sink).

**Monthly aggregation** (native ~5-day pentads): `prcp` sum (mm/month); `tavg, tmax, tmin, tsoil, vpd, EVI2, sm1, sm2, SW_IN_NLDAS`, all fluxes, all pools mean; `clay` static as-is. Drop site-months with < 4 records.

## Step 0 — EDA (complete)

- 29,723 records, 59 sites (grass 39, desert-scrub 7, sagebrush 7, grass-tree 6), 2002–2024; zero missing values.
- 1.7% of steps have gaps > 10 days; 352 site-months (~1.2%) have < 4 records → dropped.
- EVI2 strongest flux predictor (r = 0.88–0.89 GPP/RECO); tmin/tsoil/tavg r ≈ 0.4–0.5; soil moisture/prcp r < 0.2. Pools (AGL, BGL, POC, HOC) r ≤ 0.27 with all dynamic predictors.
- Fluxes mean 0.4–1.9; pools mean 60–7,332 → per-target z-score required.
- Pools accumulate over years; `seq_len` may need increasing for pool skill.

## Step 1 — Preprocessing

1. Load all four CSVs (`KEEP` columns), concatenate.
2. Aggregate by `(site, year_month)` per rules above.
3. **Split per PFT:** shuffle sites (`preprocessing.random_seed`); `n_val = max(1, round(val_frac·n))`, `n_test = max(1, round(test_frac·n))` (shrink `n_test` then `n_val` to keep ≥1 train); rest train. ≥1 site per PFT per split if PFT has ≥3 sites. **Production: 35/11/8.** Per-PFT test metrics rest on 1–2 sites — high variance.
4. Per-site means of `prcp, tavg, vpd, tsoil, SW_IN_NLDAS` from each site's own records (predictor-only → no leakage), tiled over time.
5. **Column order (32):**

   | Block | Columns | n |
   |---|---|---|
   | Dynamic | EVI2, tsoil, sm1, sm2, vpd, SW_IN_NLDAS, tavg, tmax, tmin, prcp | 10 |
   | Static | clay | 1 |
   | PFT one-hot | desert-scrub, grass, grass-tree, sagebrush | 4 |
   | Cyclical | month_sin, month_cos | 2 |
   | Site means | mean_prcp, mean_tavg, mean_vpd, mean_tsoil, mean_SW_IN_NLDAS | 5 |
   | Targets (idx 22–31) | GPP, RECO, Rm, Rg, AGB, BGB, AGL, BGL, POC, HOC | 10 |

   `nFeatures=22`, `nTargets=10`.
6. Scaler fit on train rows; `std=1` where 0; `{"mean","std"}` `(32,)` → `paths.scaler`; normalise all splits.
7. Contiguous monthly segments per site; drop those < `preprocessing.seq_len`. Each `(T_seg, 32)`.
8. Save `train/val/test.pkl` to `paths.preprocessed_dir`: `List[Dict]` with `site`, `pft`, `segments`, `segment_starts` (`(year, month)`). Sites with no qualifying segment are omitted — 54 of 59 remain.

## Step 2 — Training

- Same loop as Amazon Step 2 (`WindowedDataset` with `preprocessing.seq_len`/`stride`; AdamW + warmup/cosine; LR finder when `training.optimized_lr` is null; masked MSE; checkpoint/early stopping; per-target loss logging; loss/scatter/boxplot figures), with `TransformerModel(num_features=22, num_targets=10, cfg)`, input `[:, :22]`, target `[:, 22:]`.
- `--flux-only`: GPP/RECO/Rm/Rg only, slicing existing full-target pkl/scaler; suffix `_fluxonly`; also on 03/04. Same flux accuracy as full-target; recommended for flux-only downstream use (e.g. multi-domain).
- `--seed N`: as Amazon (split fixed; `_seedN`; production = 5 seeds via `run_seed_sweep.py` + `shared/seed_aggregation.py`).
- Study-only, mutually exclusive, never production (suffix outputs): `--model-size {small,medium,large,xlarge,xxlarge}` → `model_{size}` (`hidden_dim` sweep); `--capacity-matched` → `model_capacity_matched`; `--amazon-sized` → Amazon's production architecture. The latter two are retired (outputs kept, not plotted/rerun; see `ablation_test/ablation_description.md`).

## Step 3 — Prediction

Run only once validation is satisfactory (test set used once).

1. Load `paths.best_model`, `test.pkl`.
2. Stride-1 windows, last-position prediction only; first `seq_len−1` steps = NaN.
3. Inverse: `pred * std[22:] + mean[22:]`.
4. `NEE_predicted = RECO_predicted − GPP_predicted` (no scaler column).
5. Save `outputs/rangeland_domain/predictions/predictions.parquet`: `site, date`, the 10 `*_predicted` targets, `NEE_predicted`; time-ordered per site.

## Step 4 — Evaluation

1. Predictions from `paths.predictions`; ground truth from `test.pkl` (inverse-scaled).
2. Per-site, per-target RMSE/NSE/KGE/PBIAS → `outputs/rangeland_domain/evaluation/metrics_test.csv` (`site, pft, target, RMSE, NSE, KGE, PBIAS`).
3. Plots (`shared/plots.py`) to `outputs/rangeland_domain/evaluation/`:
   - `metrics_boxplot_test_{fluxes,pools}_{by_pft,pooled}.png` (fluxes/pools split so pool RMSE doesn't squash flux boxes).
   - Time series for 1 test site per PFT, all targets.
   - `site_map.png` from `RangeSTAR_data/ameriflux_sites.geojson`.

## Outputs (`outputs/rangeland_domain/`)

| Path | Contents |
|---|---|
| `preprocessed/{train,val,test}.pkl` | Normalised splits (Step 1 §8) |
| `scaler.pkl` | `{"mean","std"}` `(32,)`, train-fit |
| `models/best_model{_fluxonly}{_seedN}.pt` | Checkpoint |
| `predictions/predictions{_fluxonly}{_seedN}.parquet` | `site, date`, 10 targets + NEE (or 4 fluxes + NEE) |
| `evaluation/` | `metrics_test.csv`, figures, `site_map.png`, `history.csv` |
