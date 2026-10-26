# Key Findings Log

<!-- Claude instructions: add an entry only for a first domain eval, a surprising result
(NSE delta > 0.05 vs. prior run, same domain), an unexplained failure, or a design decision
driven by results. Append after the run completes; exp_id = <PREFIX>-<run_id[:8]>, also set as
MLflow tag "exp_id". MLflow is SSOT for metrics. Claude drafts "What happened"; humans fill
"Interpretation & Decisions" and remove the NEEDS HUMAN REVIEW marker — Claude never removes it. -->

---

## AR-21c64242 — arctic_domain — 2026-07-07
**run_id:** `21c642428f5e43e4b036f35624bccfa1`
**Config delta:** First 50K run after fixing `-9999` fill-value contamination in `01_preprocess.py` (some grids' projected/`_sc` NetCDFs lack `_FillValue`; ~534M entries now masked explicitly) + `grad_clip_norm: 1.0` in `shared/training.py`. Prior pre-fix 50K run invalid, deleted.

### What happened
- Loss converges smoothly (prior run spiked ~epoch 18-20); early stop at epoch 40.
- Pooled val/test NSE modest but real for all 4 targets, ~0.15-0.34 (`val_pred_vs_true.png`).
- Per-pixel NSE/KGE fine for GPP/RECO; ALD/VEGC very weak (most pixels < -1). Two causes:
  (a) ~1% of ALD/VEGC pixel-periods have constant obs (NSE/KGE undefined) — flagged `obs_degenerate` in `metrics_df_by_period`, excluded from boxplots/maps only (`metrics.csv` keeps all rows);
  (b) the other ~99% are genuinely weak — relative-variance exclusion up to 10% of global std still left median per-pixel NSE deeply negative.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
- Claude hypothesis (unconfirmed): inputs are static covariates (soil, drainage, fire-return interval, topo, veg type), CO2 and a 12-month climate window (`tair`, `precip`, `nirr`, `vapor_press`) — no lagged target. Fluxes (GPP/RECO) follow concurrent climate; pools (ALD/VEGC) depend on decades of history the model can't see, so it regresses to a static-covariate baseline → decent pooled but poor per-pixel NSE.
- Decision: accept as known limitation; goal 1 scope unchanged.

### Follow-up
- Try `seq_len=24` (or 36) before any design change. An autoregressive prior-target input needs its own design discussion (recursive state at inference, error accumulation over multi-decade projections).

---

## AR-d59b948a — arctic_domain — 2026-07-07
**run_id:** `d59b948aa6ab4939a8413169278d5c51` (H1 only; the other 5 diagnostics ran outside `tracking.training_run()`, no MLflow run)
**Config delta:** 6 disposable diagnostics (why 50K→500K and seq_len 12→24 didn't help held-out per-pixel NSE): `domains/arctic_domain/diag_h5_insample_eval.py`, `diag_h2_fluxonly.py`, `diag_h3_smallmodel.py`, `diag_h4_lastpos.py`, `diag_h3h4_combined.py` + H1 = `01_preprocess.py --capped-stride 150` (vs 24, ~50K windows). Production code untouched; all but H1 reuse the 50K pkls.

### What happened
- **H5** (500K checkpoint on its own train pixels): in-sample median NSE nearly as bad as held-out (ALD -367 vs -434, GPP -0.26 vs -0.30, RECO -0.10 vs -0.17, VEGC -263 vs -358) → not mainly a spatial-generalization problem; points to optimization/capacity/sampling.
- **H2** (flux-only loss): GPP/RECO got *worse* (GPP 0.007 vs baseline 0.127, RECO -0.159 vs -0.069) → refutes "pool targets corrupt flux gradients".
- **H3** (model 5-6M → 101K params, 356-pixel data): GPP 0.127→0.563, RECO -0.069→0.035; ALD/VEGC still catastrophic.
- **H4** (loss only at last position, matching how 03/04 score): ALD -577→-67, but GPP -0.708, RECO -0.335, VEGC -2338 — mixed, confounded with oversized model.
- **H1** (`capped_stride` 24→150, full model, 356→2148 train pixels, ~137→~23 windows/pixel): largest gain — ALD -577→-191, GPP 0.13→**0.72**, RECO -0.07→**0.30**, VEGC -1057→-374. Beats H3 and the 500K run (3631 pixels) → the lever is pixel diversity relative to per-pixel window depth, not raw pixel or window count.
- **H3+H4** on H1's 2148-pixel data: ALD -307, GPP 0.409, RECO 0.141, VEGC -1293 — worse than H1 alone on every target; those tweaks only helped on the sparse baseline.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Push `capped_stride` further (300-500) to see if GPP/RECO keep improving and whether ALD/VEGC respond to diversity or need the autoregressive input (AR-21c64242).
- The 6 `diag_*.py` scripts stay on disk pending review (delete, or promote one).

---

## AR-widesweep0708 — arctic_domain — 2026-07-08
**run_id:** N/A (MLflow not yet wired)
**Config delta:** `capped_stride` 300 at ~50K windows ("50K-wide"), per AR-d59b948a follow-up.

### What happened
- Median per-pixel val NSE catastrophic (ssp1_2_6: ALD ≈ -7.05M, GPP ≈ -432K, RECO ≈ -733, VEGC ≈ -3.05M) — worse than strides 24 and 150.
- **Root cause:** `01_preprocess.py`'s `effective_stride` used the same `capped_stride` for train/val/test, so each stride change also changed the held-out population (356 / ~2145 / ~4057 different pixels for 24/150/300). The 24→150→300 pattern conflated density with population changes.
- Archived (`train_50K_s300.pkl`, `val_metrics_50K_s300.csv`); numbers not comparable to any other entry.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Fixed (see AR-controlledsweep0708): `--train-capped-stride` decouples train stride from val/test; `--label` keeps density variants from overwriting checkpoints; `--sweep-strides` fetches each grid once and writes N stride-specific train pkls.

---

## AR-controlledsweep0708 — arctic_domain — 2026-07-08
**run_id:** N/A (MLflow not yet wired)
**Config delta:** Val/test built once at `capped_stride`=150 (2143/2145 pixels, same `grids_hash` for all points); train swept over stride ∈ {50,100,150,200,250} at ~50K windows. Scored on **val** only (test reserved for final model choice).

### What happened
- Median per-pixel val NSE by train stride:

  | stride | ALD | GPP | RECO | VEGC | best val loss |
  |---|---|---|---|---|---|
  | 50  | -65.5  | -1.26 | -10.23 | -3150.1 | *(lost)* |
  | 100 | -126.9 | 0.61  | 0.33   | -547.6  | 0.3995 |
  | 150 | -190.1 | 0.57  | 0.12   | -1516.6 | 0.3861 |
  | **200** | **-39.2** | **0.73** | **0.38** | **-395.3** | **0.3725** |
  | 250 | -394.1 | 0.51  | -0.03  | -2870.5 | 0.5092 |

  RMSE (50/100/150/200/250): ALD 1.26/1.29/1.26/**0.94**/1.52; GPP 111.0/40.5/40.2/**33.7**/51.7; RECO 105.6/27.8/29.8/**26.8**/36.9; VEGC 10499.7/5871.6/8705.8/**5574.4**/9163.0 (per-ssp: `outputs/arctic_domain/models/val_metrics_50K_s*.csv`).
- Stride 200 best on every target (NSE, RMSE, val loss); 50 worst; VEGC catastrophic throughout.
- Vs H1 (stride 150, but H1 was on test): ALD -190.1 vs -191.4, GPP 0.57 vs 0.724, RECO 0.12 vs 0.298, VEGC -1516.6 vs -373.6 — VEGC gap not fully explained.
- s50's best val loss lost to a log-file overwrite; not rerun.
- `shared/inference.py` dense eval (~140x more windows than training's val loop) sped up ~5 min → ~2:40-2:44 per point (batch_size 256→8192, num_workers=4). Vectorizing per-window assignment was numerically identical but gave no further speedup (remaining time is GPU/DataLoader).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Planned: 500K at stride 200 (val/test sizes are fixed in config, so the same population carries over).
- **Superseded by AR-sspfix0708** — stride 200 no longer wins once the SSP-collapse bug is fixed.

---

## AR-sspfix0708 — arctic_domain — 2026-07-08
**run_id:** N/A (MLflow not yet wired)
**Config delta:** Bug fix + rerun of AR-controlledsweep0708. Train records were keyed by `(grid, y, x)` without `ssp`, so each pixel's two scenarios (`ssp1_2_6`, `ssp5_8_5`) collapsed to one (records:pixels was 1:1 in `train_50K_s200.pkl`/`train_500K_s200.pkl`, should be ~2:1 like val/test). Fixed with `defaultdict(list)` per pixel; 4-grid regression test ratio 1.0→2.0. Buggy outputs archived to `outputs/arctic_domain/_archive_buggy_ssp_collapse_20260708/`.

### What happened
- Same 5-point sweep, same locked val/test, retrained on corrected data:

  | stride | ALD | GPP | RECO | VEGC | best val loss |
  |---|---|---|---|---|---|
  | 50  | -64.7  | 0.754 | 0.415 | -391.7 | 0.4263 |
  | 100 | -54.5  | 0.798 | 0.470 | -138.3 | 0.3858 |
  | 150 | -239.6 | 0.762 | 0.431 | -378.6 | 0.3567 |
  | 200 | -55.1  | 0.736 | 0.460 | -571.1 | 0.3824 |
  | **250** | **-43.5** | **0.822** | **0.528** | **-134.5** | **0.3393** |

  RMSE (50→250): ALD 0.99/0.88/1.20/0.98/**0.79**; GPP 33.6/30.3/31.5/36.9/**30.1**; RECO 27.2/23.4/24.2/27.4/**23.3**; VEGC 3867/2951/4902/4685/**3175** (`val_metrics_50K_s*.csv` on `vm-sandeep`).
- **Stride 250 wins every target**, reversing the buggy stride-200 result. Every point improved on GPP/RECO (e.g. s100 GPP 0.61→0.80, RECO 0.33→0.47).
- Non-monotonic trend (150 dips) — treated as single-run noise.
- 53 grid failures → all points ~39,200-39,400 windows / 200 grids (vs 50K/~220 target), uniformly, so the relative comparison holds.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Next: staggered windowing at stride 250 (per-pixel phase offset; window starts are currently identical across pixels — see `arctic_description_data_handling.md`), then scale to 500K/2M.
- Stride 300 open: the only prior s300 point (AR-widesweep0708) predates both fixes.

---

## AR-stagger0709 — arctic_domain — 2026-07-09
**run_id:** N/A (MLflow not yet wired)
**Config delta:** New `--stagger` flag (train-only, save-time): each pixel trims `crc32(seed:grid:y:x) % stride` leading rows, same phase for both SSPs. `50K_s250_staggered` vs AR-sspfix0708's vanilla s250, same val/test (`grids_hash` matches), same 260-grid list. Accepted confound: staggered run fetched more grids (233/260, 3298 pixels) than vanilla (200/260, 2816 pixels).

### What happened

  | metric | vanilla s250 | staggered s250 |
  |---|---|---|
  | ALD NSE | -43.5 | -54.8 |
  | GPP NSE | 0.822 | **0.842** |
  | RECO NSE | 0.528 | **0.555** |
  | VEGC NSE | -134.5 | **-131.1** |
  | best val loss | 0.3393 | **0.3223** |

  RMSE vanilla/staggered: ALD 0.79/0.82, GPP 30.1/**28.9**, RECO 23.3/**23.1**, VEGC 3175/**2831** (`val_metrics_50K_s250{,_staggered}.csv`).
- Staggering: ~5% lower val loss, better GPP/RECO/VEGC; ALD slightly worse (consistent with ALD's high variance, not clearly staggering-specific).
- **Staggered s250 = new winning recipe** (no extra preprocessing cost).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Scale staggered s250 to 500K and 2M (deferred to this point per user's sequencing).

---

## AR-500Kstagger0709 — arctic_domain — 2026-07-09
**run_id:** N/A (MLflow not yet wired)
**Config delta:** `--train-size 500000 --train-capped-stride 250 --stagger`, same val/test (`grids_hash` verified), `--grids` pinned to the 260-grid list (auto-discovery is unstable — an unpinned run found 263 grids and would have silently rebuilt val/test). Preprocessing on `vm-cpu-sandeep` 1h29m (pass 1 ~9 min; pass 2 re-fetched 253/260 grids, ~1h20m). `train_500K_s250_staggered.pkl`: 209/260 grids, 29,175 pixels, 57,783 records (ratio 1.98), 7.8GB.

### What happened

  | metric | 50K staggered | 500K staggered |
  |---|---|---|
  | best val loss | 0.3223 | **0.1858** (-42%) |
  | ALD NSE / RMSE | -54.8 / 0.82 | **-9.4 / 0.37** |
  | GPP NSE / RMSE | 0.842 / 28.9 | **0.934 / 17.0** |
  | RECO NSE / RMSE | 0.555 / 23.1 | **0.757 / 15.4** |
  | VEGC NSE / RMSE | -131.1 / 2831 | **-53.0 / 1347** |

  (`val_metrics_500K_s250_staggered.csv`, median over both SSPs.)
- Every target improved with no confound; ALD ~5x less negative but still the weakest. Confirms H1: pixel diversity is the dominant lever, no saturation at 500K.
- Training ~18 min on `vm-sandeep` (early stop epoch 54, best 44), GPU ~97-98%, 12-15GB/83GB RAM, ~5GB/40GB GPU.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Scale to 2M (same recipe/grid list) to check plateau, and whether ALD needs an autoregressive input.
- After 50K/500K/2M, formalize the saturation check with `05_learning_curve.py`.

---

## AZ-184e096d — amazon_domain — 2026-07-11
**run_id:** `184e096dc9e84376ab7418ce9c96957d`
**Config delta:** First production run (`outputs/amazon_domain/` was empty). Dev-mode ground test (all 4 stages, `stride=24`) passed first. Production: `hidden_dim=128, num_layers=3, num_heads=4, feedforward_dim=512, batch_size=256, num_epochs=100, stride=1`; 98 stations → 59/20/19 train/val/test.

### What happened
- Full 100 epochs, no early stop; val loss plateaued ~0.58, train ~0.47.
- Test median (19 stations):

  | target | NSE | RMSE |
  |---|---|---|
  | discharge | -0.931 | 687.9 |
  | active_fire_count | -0.118 | 195.4 |
  | burned_area | -1.908 | 119.8 |

- All negative (worse than station mean) — weak result, not a pipeline bug (outputs structurally sane, `outputs/amazon_domain/evaluation/`).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Untuned (no LR sweep beyond the finder, no architecture iteration). Check `loss_curves.png` for undertraining vs early plateau, and whether skewed/zero-inflated `active_fire_count`/`burned_area` need a different loss or transform than MSE.
- No stride/size sweep run (small dataset); consider one if results stay weak.

---

## AR-gridsplitsweep0710 — arctic_domain — 2026-07-10
**run_id:** N/A (MLflow not yet wired)
**Config delta:** First sweep under the grid-level stratified split (`assign_grid_splits()` replaces per-grid pixel split `_grid_split_labels()`): whole grids → train/val/test 60/20/20, stratified into 6 latitude bins. Staggering now unconditional (`--stagger` removed; see `arctic_description_data_handling.md`). `FLAKY_GRIDS_20260710` (separate from `KNOWN_BROKEN_GRIDS`) excludes 8 grids that failed ~5 retry cycles even with `fetch_timeout_seconds` 180→300: `H11_V16`, `H11_V19`, `H14_V15`, `H16_V7`, `H17_V3`, `H19_V13`, `H19_V18`, `H9_V19`. New val/test populations (~48 grids/~366 pixels val, similar test, at capped_stride=24; an earlier "1,920/2,013 pixels" figure was a transcription error — see AR-gridsplit4005000710). Not comparable to pixel-split-era entries. Swept stride ∈ {50..350} at ~50K windows in one `--sweep-strides` pass.

### What happened
**Superseded:** val.pkl was later silently regenerated, so these numbers don't match disk — use AR-gridsplit4005000710's 9-point table.

  | stride | best val loss | ALD | GPP | RECO | VEGC |
  |---|---|---|---|---|---|
  | 50  | 0.3935 | -54.1  | 0.836 | 0.420 | -170.5 |
  | 100 | 0.3710 | -84.5  | 0.749 | 0.385 | -800.0 |
  | 150 | 0.3548 | -120.3 | 0.774 | 0.399 | -868.8 |
  | 200 | 0.3400 | -70.0  | 0.791 | 0.478 | -952.7 |
  | 250 | 0.3025 | -49.4  | 0.808 | 0.504 | -249.1 |
  | 300 | 0.3393 | -56.5  | 0.792 | 0.442 | -707.3 |
  | **350** | **0.2827** | **-33.4** | **0.870** | **0.519** | **-77.3** |

  RMSE (50→350): ALD 0.77/0.93/0.99/0.86/0.69/0.84/**0.59**; GPP 28.9/35.5/34.8/31.3/**26.7**/32.0/27.0; RECO 23.5/28.9/30.4/28.2/24.1/28.1/**23.2**; VEGC 2674/5639/5135/4669/2684/4414/**2257** (`val_metrics_50K_s*.csv`).
- Stride 350 (widest) wins val loss and NSE+RMSE on 3/4 targets (GPP RMSE second to s250's 26.7); much more decisive than the pixel-split sweep. s300 dips below both neighbours. ALD/VEGC still deeply negative but least bad at 350.
- Training ~33 min for all 7 points (smaller val/test: ~42-44 grids each). Preprocessing ~50 min in one attempt after excluding the flaky grids (before, they blocked pass 1's `missing_grids` check twice, ~9-15 min each).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- 350 is the widest point and still winning — test 400+ before committing.
- 500K/2M scale-up deferred until stride is settled.
- `FLAKY_GRIDS_20260710` permanence unconfirmed (single day) — retest without exclusion later.

---

## RG-83fdf771 — rangeland_domain — 2026-07-11
**run_id:** `83fdf7715edc44d4b052c41b553e6d80`
**Config delta:** First production run (previously dev-verified only, 2026-06-30). Dev ground test (`stride=6`) passed first. Production: `hidden_dim=64, num_layers=3, num_heads=4, dropout=0.3, feedforward_dim=256, batch_size=64, num_epochs=100, stride=1`; 59 sites → 35/11/8, PFT-stratified (4 groups).

### What happened
- Early stop epoch 63 (best val 0.3865 @ 51).
- Test median (8 sites, 10 targets):

  | target | NSE | RMSE |
  |---|---|---|
  | GPP_predicted  | 0.853  | 0.514 |
  | RECO_predicted | 0.855  | 0.405 |
  | AGB_predicted  | 0.882  | 14.38 |
  | BGB_predicted  | -0.808 | 18.76 |

  By PFT (median NSE, all 10 targets): sagebrush 0.928 (n=10), grass-tree 0.711 (10), grass 0.200 (50), desert-scrub -6.212 (10).
- Fluxes and AGB strong (NSE 0.85+), like Arctic's flux-vs-pool pattern; BGB negative. desert-scrub = 1 test site, high variance by construction (see `methodology_audit_20260617.md`).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- BGB: check `timeseries_*.png` for right shape/wrong scale (PBIAS, fixable) vs wrong dynamics.
- Don't over-interpret desert-scrub's single site.
- No stride/size sweep (tiny data) — revisit only if BGB/desert-scrub points to data volume.

---

## AR-gridsplit4005000710 — arctic_domain — 2026-07-10
**run_id:** N/A
**Config delta:** Added strides {400, 500} via `--sweep-strides 400,500`, ~50K windows. Pass 1 cached; pass 2 re-fetched 246/252 grids (scattered one-off failures, not new flaky grids).

**Bug fixed (`3d19d6e`):** this run silently regenerated `val.pkl`/`test.pkl` (326/327 pixels) despite matching `grids_hash` (root cause unknown; old sidecar overwritten). `01_preprocess.py` now raises with a field-level sidecar diff; rebuild only with `--force-recompute`. Val/test frozen from here; strides 50-350 retrained (~2-3 min each, A100) against it.

### What happened
- Corrected 9-point comparison (median, same val.pkl):

  | stride | best val loss | ALD NSE | ALD RMSE | GPP NSE | GPP RMSE | RECO NSE | RECO RMSE | VEGC NSE | VEGC RMSE |
  |---|---|---|---|---|---|---|---|---|---|
  | 50  | 0.2893 | -89.2  | 0.819 | 0.840 | 29.5 | 0.512 | 24.0 | -561.4 | 3288.7 |
  | 100 | 0.3025 | -114.6 | 0.907 | 0.797 | 33.2 | 0.477 | 28.0 | -683.7 | 4378.3 |
  | 150 | 0.2801 | -148.5 | 0.912 | 0.783 | 34.3 | 0.449 | 26.8 | -585.1 | 4790.6 |
  | 200 | 0.2573 | -128.2 | 0.838 | 0.799 | 32.2 | 0.508 | 25.5 | -404.9 | 3963.3 |
  | 250 | 0.2274 | -89.3  | 0.690 | 0.858 | 27.8 | 0.556 | 23.4 | -269.2 | 2828.3 |
  | 300 | 0.2976 | -139.7 | 1.025 | 0.784 | 33.4 | 0.492 | 27.1 | -525.1 | 5039.8 |
  | 350 | 0.2330 | -68.6  | 0.677 | 0.809 | 28.0 | 0.460 | 24.0 | -196.7 | 2856.5 |
  | **400** | **0.2219** | **-63.0** | **0.600** | **0.868** | **25.5** | **0.583** | **21.4** | **-103.7** | **2625.6** |
  | 500 | 0.2528 | -125.2 | 0.746 | 0.789 | 29.1 | 0.535 | 25.4 | -478.4 | 2931.6 |

  (`val_metrics_50K_s*.csv`, 2026-07-10 17:14-17:40.)
- **Stride 400 wins outright** (val loss + NSE and RMSE on all 4 targets). 500 is worse on every metric → optimum ~350-400 plus run-to-run noise, not monotonic.
- s200's first retrain was an outlier (val loss 0.677, early stop epoch 8 — bad LR-finder/init draw; not fully seeded); retrain gave 0.2573.
- ALD/VEGC least bad at 400 by a wide margin (ALD -63.0 vs -68.6 at 350; VEGC -103.7 vs -196.7).
- **Recommendation:** stride 400 = candidate default for 500K/2M, pending user review.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Val/test regeneration root cause unknown; if the new guard fires unexpectedly, start from its field-level diff.
- s200's variance (0.677 vs 0.257, identical config) suggests the LR range test / init isn't deterministic.
- Per user decision, no further stride widening (e.g. 450) unless 500K/2M results suggest it.

---

## RG-5f0c3603 — rangeland_domain — 2026-07-11
**run_id:** `5f0c3603c6a34f258bc9e5976bb7d7e2`
**Config delta:** New `--flux-only` mode (02/03/04): trains on GPP/RECO/Rm/Rg only (drops AGB/BGB/AGL/BGL/POC/HOC), reusing existing pkls (fluxes are the first 4 of 10 target columns). Outputs suffixed `_fluxonly` (`models/best_model_fluxonly.pt`, `evaluation_fluxonly/`). Same hyperparameters. Also: `04_evaluate.py` boxplots split into flux/pool subsets (by-PFT and pooled), since pool metrics squashed flux boxes.

### What happened
- Early stop epoch 80 (best val 0.0629 @ 68).
- Test median NSE / RMSE vs full-target RG-83fdf771:

  | target | full-target | flux-only |
  |---|---|---|
  | GPP  | 0.8535 / 0.514  | 0.860 / 0.4275 |
  | RECO | 0.8545 / 0.405  | 0.862 / 0.3875 |
  | Rg   | 0.8190 / 0.1315 | 0.823 / 0.1315 |
  | Rm   | 0.8375 / 0.1810 | 0.846 / 0.1545 |

- All fluxes slightly better or equal — modest gain from dropping noisy pool targets.
- New `metrics_boxplot_test_fluxes_by_pft.png` / `..._pooled.png` make flux boxes legible.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- `best_model_fluxonly.pt` is the recommended model for GPP/RECO/Rm/Rg-only consumers.
- Pool targets (esp. BGB) need their own investigation.

---

## AR-500Kstride400-0710 — arctic_domain — 2026-07-10
**run_id:** N/A
**Config delta:** First grid-split scale-up: `--train-size 500000 --train-capped-stride 400`. Pass 1 cached; pass 2 re-fetched ~150 train grids, zero failures. Val/test reused from cache ("val.pkl already exists and matches config — skipping (cached)") — first real proof the `3d19d6e` guard works.

### What happened

  | run | best val loss | ALD NSE | ALD RMSE | GPP NSE | GPP RMSE | RECO NSE | RECO RMSE | VEGC NSE | VEGC RMSE |
  |---|---|---|---|---|---|---|---|---|---|
  | 50K_s400  | 0.2219 | -63.0 | 0.600 | 0.868 | 25.5 | 0.583 | 21.4 | -103.7 | 2625.6 |
  | **500K_s400** | **0.1145** | **-19.2** | **0.307** | **0.934** | **17.0** | **0.737** | **14.4** | **-25.4** | **1243.1** |

  (`val_metrics_{50K,500K}_s400.csv`. `train_500K_s400.pkl`: 452,517/500,000 windows, 136 grids, 50,476 pixels.)
- Every metric improved; val loss nearly halved. ALD (~3.3x less bad) and VEGC (~4.1x) improved most in relative terms, RMSE roughly halved — first sign they converge with scale (data-hungry rather than unmodelable), though still negative.
- Training ~18 min (58 epochs, ~16.4 s/epoch vs 50K's ~1.7 s and 86 epochs). Preprocessing ~30 min.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- No saturation at 500K → 2M is the next scale point, esp. for ALD/VEGC.
- Watch whether ALD/VEGC need a structural fix (autoregressive input, dedicated loss term) regardless of data volume.
- Docs rewrite (`arctic_description.md`, `arctic_description_data_handling.md`) for the grid-level split deferred until scale-up settles.

---

## AZ-71935d7c — amazon_domain — 2026-07-11
**run_id:** `71935d7cfb224caca3fc3909a6a99e7e`
**Config delta:** vs AZ-184e096d: (1) `model.nonneg_output: true` → config-gated softplus output head (`shared/transformer.py`), since all targets are ≥0; (2) all 3 targets log1p-transformed before the scaler fit (`01_preprocess.py`), `expm1` in 03/04 inverse. EDA: severe right skew (discharge mean 2443.5 vs median 435.6), `drainage_area` spans ~3 orders of magnitude. Re-ran `01_preprocess.py`.

### What happened
- Early stop epoch 35 (best val 0.5948 @ 23).

  | target | old NSE / RMSE | new NSE / RMSE |
  |---|---|---|
  | discharge | -0.931 / 687.9 | 0.014 / 460.3 |
  | active_fire_count | -0.118 / 195.4 | 0.521 / 93.5 |
  | burned_area | -1.908 / 119.8 | 0.260 / 57.1 |

- All NSE now positive, RMSE -33 to -52%; skewed fire/burn improved most (log1p is the main fix); discharge barely positive.
- Predictions all ≥0; log1p/z-score/expm1 round-trip exact (incl. NaN passthrough for discharge's ~6% missing).
- `active_fire_count_pred` rounded to integer in the parquet (cosmetic; no count-specific loss).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Discharge NSE 0.014 still weak — per-station normalization is the next lever (cross-station scale from `drainage_area` is separate from within-station skew).
- Still no LR sweep or architecture iteration.

---

## AR-500Ktesteval-0711 — arctic_domain — 2026-07-11
**run_id:** `364cd7351b5a427da9fc2ce56c0a82c9`
**Config delta:** No model/data change — first `04_evaluate.py` on frozen `test.pkl` for 500K_s400 (only val metrics existed; irreproducible once pkls are deleted). Commit `68fe014`: `metrics.csv` → **`metrics_test.csv`**; new `prediction_sample.parquet` = monthly obs/pred (4 targets, both SSPs) for a deterministic 50-pixel test sample (seed `preprocessing.random_seed`), reproducible across runs/pipelines.

### What happened
- `04_evaluate.py --train-size 500000 --label 500K_s400` on `vm-sandeep`: 3,868 metric rows, 327 test pixels; sample 164,688 rows (~4.6MB).
- Test median (excl. `obs_degenerate`):

  | target | NSE | RMSE | n |
  |---|---|---|---|
  | ALD  | -76.32 | 0.395 | 961 |
  | GPP  | 0.903  | 20.09 | 946 |
  | RECO | 0.610  | 16.34 | 946 |
  | VEGC | -18.13 | 2040.8 | 946 |

- Vs val (ALD -19.2, GPP 0.934, RECO 0.737, VEGC -25.4): GPP/RECO slightly lower, VEGC less bad, ALD worse — within expected val-vs-test population differences.
- `pyarrow` (undeclared transitive dep) added to `requirements.txt` (`68fe014`).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Saved-results gap closed: `metrics_test.csv`, test plots and `prediction_sample.parquet` in `outputs/arctic_domain/evaluation/500K_s400/`, safe after pkls are deleted.
- Deliberately only 500K_s400; the other sweep points keep only training-time val metrics.

---

## AZ-5e809245 — amazon_domain — 2026-07-11
**run_id:** `5e809245f290467b8e40b01f2d38dc36`
**Config delta:** AZ-71935d7c's discharge NSE 0.014 (vs ~0.5 typical): small stations catastrophic (NSE down to -38, PBIAS up to +366%), large ones 0.5-0.8 (Spearman corr(RMSE, NSE) = 0.70) — one global z-score across 59 stations spanning ~3 orders of magnitude in `drainage_area`. **Fix** (`01_preprocess.py`): discharge / `drainage_area` before log1p (specific discharge, cf. Kratzert et al. CAMELS LSTM; generalizes to unseen stations); raw `drainage_area` saved per record for 03/04 to multiply back. Fire/burn untouched.

### What happened
- Early stop epoch 24 (best val 0.5259 @ 12, vs 35 / 0.5948).

  | discharge | AZ-71935d7c | AZ-5e809245 |
  |---|---|---|
  | median NSE | 0.014 | **0.351** |
  | median RMSE | 460.3 | 276.1 |
  | stations NSE > 0 | 10/19 | **17/19** |
  | Spearman corr(RMSE, NSE) | 0.70 | **0.10** |

- Small-station bias resolved; only station 15200000 strongly negative (NSE -9.3; was 9 negative stations). Predictions ≥0.
- **Fire/burn declined** with unchanged preprocessing: active_fire_count 0.521→0.298, burned_area 0.260→0.014. Likely multi-task reallocation (shared backbone, joint MSE) plus no fixed seed in `shared/training.py`; aggregate val loss improved (0.526 vs 0.595).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Rerun with a different seed to separate interference from seed variance (no seeding currently).
- If it persists: per-target loss weighting or separate heads (architectural, out of scope).
- Check station 15200000 (drainage_area sanity, unusual regime).

---

## AZ-2ffbfcd3 — amazon_domain — 2026-07-11 (burned_area area-normalization: tried, reverted)
**run_id:** `2ffbfcd37d9144f28d8297b9591761d9`
**Config delta:** Applied AZ-5e809245's drainage-area normalization to `burned_area` (Spearman corr(RMSE, NSE) = 0.846; low-RMSE stations median NSE -1.73, PBIAS +501%; burned fraction of basin). `active_fire_count` untouched (Spearman -0.23, well-calibrated PBIAS).

### What happened
- **Hypothesis wrong:** burned_area median test NSE 0.014 → **-1.08**, PBIAS up to **+22735%** at small stations.
- Revised cause: unlike discharge (`Q ~ precip x area x runoff coefficient`), fire products are likely computed in a **fixed buffer** around each station (bucket `am_hydro_fire_risk`), not over the watershed — dividing by `drainage_area` added distortion. A matching symptom (RMSE~NSE) doesn't imply the same mechanism.
- **Reverted** (01/03/04; burned_area back to log1p only) and re-ran: burned_area NSE 0.077 (11/19 positive vs broken run's 8/19), discharge 0.381, active_fire_count 0.380 — within run-to-run variance of AZ-5e809245 (no fixed seed). Predictions ≥0.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- If burned_area (0.077 vs discharge 0.38) needs work: normalize by the actual buffer footprint (if available) or per-station train-period stats (generalizes less well) — not `drainage_area`.
- Lesson: verify the mechanism before trusting a correlation-matched symptom; always confirm empirically.

---

## AR-c3aaf88b — arctic_domain — 2026-07-11
**run_id:** `c3aaf88be64740939e6f9b77dfc073f9`
**Config delta:** New `--flux-only` (02/03/04), GPP+RECO only: `_naming.py`'s `select_flux_target_columns`/`select_flux_scaler_stats` reorder `[ALD | GPP | RECO | VEGC]` to `[features | GPP | RECO]` and slice the scaler; reuses 500K_s400 pkls. Output label `500K_s400_fluxonly`. Same hyperparameters. `04_evaluate.py` now plots 2 per-pixel timeseries from the 50-pixel sample.

### What happened
- Early stop epoch 70 (best @ 60), ~22 min (13GB pkl load + 70 epochs).
- Test median NSE / RMSE vs full-target AR-500Ktesteval-0711 (excl. `obs_degenerate`):

  | target | full-target | flux-only |
  |---|---|---|
  | GPP  | 0.903 / 20.09 | 0.906 / 18.948 |
  | RECO | 0.610 / 16.34 | 0.576 / 16.132 |

- Essentially a wash (unlike Rangeland); value is a clean dedicated flux-only output.
- `timeseries_H11_V9_9_8.png` tracks seasonal cycles 1901-2100 well; `timeseries_H11_V9_16_9.png` misses 2 RECO spikes (~750, ~2000-2002; predicts ~0-50) — real failure case.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- `best_model_500K_s400_fluxonly.pt` is the recommended GPP/RECO-only model.
- Investigate the RECO-spike miss (isolated extreme event vs broader pattern) if RECO accuracy matters.

---

## MD-devsmoke0711 — multi_domain — 2026-07-11
**run_id:** N/A (`mlflow.enabled: false`)
**Config delta:** First multi-domain pipeline run (01-04), `mode: dev` (`common_dim=64`, `num_layers=2`; Arctic still loads production `train_500K_s400.pkl`), on `vm-cpu-sandeep`.

### What happened
- Pre-flight: Arctic train windows = 413,748, matching the individual pipeline (verifies the sidecar-stride fix below).
- **Bug fixed before GPU use:** training windowed Arctic at global `cfg.preprocessing.stride` (1) instead of the pkl's sidecar stride (400) → ~200M windows/epoch. Now read via `domains/arctic_domain/_naming.py::load_stride_seq_len`. Caught by 8-angle code review.
- Pretrain: 1 dev epoch, train ~0.94-1.02, val ~0.68-1.32, saved to `models/pretrained/best.pt`; ~13.9GB RSS (10.5% of 128GB).
- **Second (pre-existing scaffold) bug:** Arctic finetune ran 10 epochs (train 0.9508→0.5167, val →0.3945, `best_val=0.3911`, `models/finetuned/arctic_best.pt`) then crashed in `post_train_plots` (`AttributeError: 'function' object has no attribute 'eval'`) — eval sites wrapped the model in a lambda but `predict_last_position` calls `.eval()`. Fixed with `DomainRoutedModel(nn.Module)` in `model.py` (same pattern as `_LRProbeModel`, now consolidated).
- Re-verified via `04_evaluate.py` on existing checkpoints (all domains × pretrained, arctic × finetuned); missing amazon/rangeland finetuned checkpoints skip cleanly.
- Arctic eval matches AR-500Ktesteval-0711 exactly (3,868 rows, 164,688-row sample) — `sample_test_pixels`/`save_prediction_sample` moved to `arctic_domain/_naming.py`, same seed → identical sites across pipelines (user request).
- `--flux-only`: pretrain 1 epoch (train ~0.93-0.95, val ~0.44-1.31, `models/pretrained_fluxonly/best.pt`), ~25GB RSS transient (known ~2x copy from `select_flux_target_columns`). Eval: Arctic 1,934 rows (half of 3,868), Amazon 57 (unchanged — ignores flag), Rangeland 32 (4 vs 10 targets).
- `03_predict.py --domain amazon --checkpoint pretrained` writes `predictions/pretrained/amazon/amazon_predictions.parquet`.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Amazon/Rangeland finetune not exercised (same code path as Arctic; low risk).
- Next: production run on `vm-sandeep` (separate go-ahead needed).

---

## MD-prod0712 — multi_domain — 2026-07-12
**run_id:** N/A (01-04 don't call `shared/tracking.py`; metrics kept here, CSVs only on `vm-sandeep`)
**Config delta:** First production run, full-target and `--flux-only`: `common_dim=256, num_layers=6, num_heads=8, feedforward_dim=1024` (= Arctic standalone), `batch_size=1024`, `pretrain_epochs=100`, `finetune_epochs=50`/domain, `early_stopping_patience=10` evals at `eval_every_n_epochs=2`. `vm-sandeep`: A100 96%, ~5/40GB GPU, ~25/85GB RAM peak. Single seed.

### What happened
- **Pretrain (full):** early stop epoch 26 (best mean-val 0.2862 @ 6), ~25 min, ~55 s/epoch — triggered by Rangeland overfitting (val 0.38→0.57) under the mean-across-domains criterion.
- **Finetune (full)** (own val loss, trunk frozen): Arctic all 50 epochs (best_val 0.1165 @ 34); Amazon 0.2527; Rangeland stop epoch 30 (0.4009).
- **Full-target test median NSE:**

  | Domain | Target | Finetuned | Pretrained |
  |---|---|---|---|
  | Arctic | GPP | 0.899 | 0.862 |
  | Arctic | RECO | 0.582 | 0.478 |
  | Arctic | ALD | -62.1 | -102.8 |
  | Arctic | VEGC | -27.9 | -33.2 |
  | Amazon | discharge | 0.793 | 0.789 |
  | Amazon | active_fire_count | 0.850 | 0.815 |
  | Amazon | burned_area | 0.697 | 0.694 |
  | Rangeland | GPP | 0.945 | 0.923 |
  | Rangeland | RECO | 0.892 | 0.861 |
  | Rangeland | Rm | 0.893 | 0.864 |
  | Rangeland | Rg | 0.880 | 0.857 |
  | Rangeland | AGB | 0.896 | 0.877 |
  | Rangeland | BGB | 0.562 | 0.507 |
  | Rangeland | AGL/BGL/POC/HOC | all deeply negative | — |

  Finetuning improved essentially every target.
- **Pretrain (flux-only):** early stop epoch 56 (best mean-val 0.1167 @ 36), same Rangeland pattern, ~25GB RSS as expected.
- **Finetune (flux-only):** Arctic epoch 34 (best_val 0.0757), Amazon 40 (0.2315), Rangeland 24 (0.0342).
- **Flux-only finetuned test median NSE** (vs full-target finetuned):

  | Domain | Target | Flux-only | Full-target |
  |---|---|---|---|
  | Arctic | GPP | 0.947 | 0.899 |
  | Arctic | RECO | 0.720 | 0.582 |
  | Amazon | discharge | 0.783 | 0.793 |
  | Amazon | active_fire_count | 0.885 | 0.850 |
  | Amazon | burned_area | 0.644 | 0.697 |
  | Rangeland | GPP | 0.979 | 0.945 |
  | Rangeland | RECO | 0.965 | 0.892 |
  | Rangeland | Rm | 0.867 | 0.893 |
  | Rangeland | Rg | 0.978 | 0.880 |

- Arctic/Rangeland gain far more than their individual flux-only runs (AR-c3aaf88b ~none, RG-5f0c3603 small) — plausibly less gradient noise in the shared trunk. Amazon (flag doesn't apply) mixed = interference noise.
- `outputs/multi_domain/` = 214MB (Arctic dense per-grid NetCDF predict skipped by design, see `multi_description.md`).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Wire MLflow for multi_domain (only record is this entry).
- Single seed — final comparison should average 3-5 seeds.
- Arctic NetCDF predictions: run `03_predict.py --domain arctic` if spatial output is needed.
- Individual vs joint vs fine-tuned comparison (`compare_models.py`) not implemented.

---

## MD-fluxrerun0713 — multi_domain — 2026-07-13
**run_id:** N/A (not wired)
**Config delta:** MD-prod0712 flux-only with `training.production.finetune_epochs` 50→100 (`warmup_epochs` 5). Also `02_train.py` now writes per-epoch `history.csv` to `pretrained[_fluxonly]/` and `finetuned[_fluxonly]/{domain}/` (commit `2bd3d9c`, for Figure 5). Old checkpoints backed up to `outputs/multi_domain/models_backup_MD-prod0712_20260713/`. Single seed.

### What happened
- **Pretrain:** full 100 epochs, no early stop (best mean-val 0.1779 @ 100) vs MD-prod0712's 0.1167 @ 36. Smooth convergence to a worse plateau (val_arctic ~0.227, val_amazon ~0.247, val_rangeland ~0.060 by epoch 98) — not a divergence.
- **Finetune:** Arctic epoch 62 (best_val 0.1949 vs 0.0757), Amazon 32 (0.2541 vs 0.2315), Rangeland 28 (0.0579 vs 0.0342) — all worse, inherited from weaker pretrain.
- **Flux-only finetuned test median NSE:**

  | Domain | Target | This run (ft=100) | MD-prod0712 (ft=50) |
  |---|---|---|---|
  | Arctic | GPP | 0.815 | 0.947 |
  | Arctic | RECO | 0.416 | 0.720 |
  | Amazon | discharge | 0.748 | 0.783 |
  | Amazon | active_fire_count | 0.810 | 0.885 |
  | Amazon | burned_area | 0.593 | 0.644 |
  | Rangeland | GPP | 0.937 | 0.979 |
  | Rangeland | RECO | 0.861 | 0.965 |
  | Rangeland | Rg | 0.905 | 0.978 |
  | Rangeland | Rm | 0.891 | 0.867 |

  All worse except Rangeland Rm; gap traces to pretrain (finetune_epochs only raises the cap).
- New Arctic individual 250K/stride=400/flux-only point (Figure 3b): GPP NSE 0.920 (ssp1)/0.927 (ssp5), RECO 0.677/0.692 — monotonic 50K → 250K → 500K.
- Figures (`run_figures_main.py`, flux-only, 300dpi, colorblind-safe): 3, 4, 5 (first version), 6a/b/c, in `./figures/`.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
- These numbers back Figures 4-6 but are worse than MD-prod0712's (cited in `current_project_status.md` Section 1 and PR #17). Likely pretrain seed variance, not the config change. Human call: (a) accept, (b) rerun pretrain and pick/average, or (c) fix a seed as the first multi-seed data point.

### Follow-up
- **LR-finder instability:** for the 250K Arctic point, the finder's LR (0.093; 100-300x above the documented 1e-5–3e-4 range) diverged (train 0.916→2.89, val →18.55; best val 0.605; test NSE -0.27 to -1.36). Fixed for that run with `training.optimized_lr: 2.0e-4` (reverted to `null` after; commits `580322f`/`99bce46`): best val 0.119 @ 52, NSE 0.68-0.93. Second Arctic LR-finder failure (after AR-gridsplit4005000710's s200) — cap the finder's LR or hardcode `optimized_lr` for Arctic.
- Still open: MLflow, multi-seed, `compare_models.py`; full-target not rerun at `finetune_epochs=100`.
- *(2026-07-14: superseded by MD-seedsweep0714; `finetune_epochs` reverted to 50 instead of rerunning full-target.)*

---

## AR-seedsweep0714 — arctic_domain — 2026-07-14
**run_id:** N/A — 5 runs (seeds 1-5), no `.run_id` sidecars.
**Config delta:** `--flux-only --seed {1..5}`, else AR-500Kstride400-0710's production config (500K, grid split, `stride=400`).

### What happened
- `run_seed_sweep.py`; seed-avg in `outputs/arctic_domain/evaluation/500K_s400_fluxonly_seedavg/metrics_test_seedavg.csv`.
- Median NSE: GPP 0.859 (historical) / 0.874 (projected), RECO 0.472 (both); `n_seeds=5`.
- Full-target variant not swept.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW -->

---

## AZ-seedsweep0714 — amazon_domain — 2026-07-14
**run_id:** N/A — 5 runs (seeds 1-5), no `.run_id` sidecars.
**Config delta:** `--seed {1..5}`, else AZ-2ffbfcd3's production config.

### What happened
- Seed-avg in `outputs/amazon_domain/evaluation_seedavg/metrics_test_seedavg.csv`.
- Median NSE: discharge 0.356, active_fire_count 0.368, burned_area 0.047.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW -->

---

## RG-seedsweep0714 — rangeland_domain — 2026-07-14
**run_id:** N/A — 5 runs (seeds 1-5), no `.run_id` sidecars.
**Config delta:** `--flux-only --seed {1..5}`, else RG-5f0c3603's config.

### What happened
- Seed-avg in `outputs/rangeland_domain/evaluation_fluxonly_seedavg/metrics_test_seedavg.csv`.
- Median NSE: GPP 0.904, RECO 0.880, Rg 0.887, Rm 0.873.
- Full-target variant not swept.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW -->

---

## MD-seedsweep0714 — multi_domain — 2026-07-14
**run_id:** N/A (not wired)
**Config delta:** `--flux-only --seed {1..5}` for pretrain and finetune; `finetune_epochs` reverted 100→50.

### What happened
- Seed-avg in `outputs/multi_domain/evaluation/{pretrained,finetuned}_fluxonly_seedavg/`.
- Finetuned median NSE: Arctic GPP 0.938 (hist)/0.935 (proj), RECO 0.745/0.653; Amazon discharge 0.764, active_fire_count 0.886, burned_area 0.723; Rangeland GPP 0.972, RECO 0.931, Rg 0.968, Rm 0.905. (Amazon values wrong — see below.)
- Vs individual 5-seed: big gains for data-scarce domains (Amazon discharge 0.356→0.764, active_fire_count 0.368→0.886); Arctic smaller (GPP 0.859/0.874 → 0.935/0.938).
- Pretrain ≈ finetune (Arctic GPP 0.925/0.926 vs 0.935/0.938) — finetuning adds little.
- Full-target variant not swept.

### Interpretation & Decisions
- Human-verified 2026-07-16: Amazon/Rangeland gain is genuine cross-domain transfer, not normalization/leakage; cleared as the paper's headline result.
- **Superseded 2026-07-16 (MD-unitsbugfix0716):** Amazon multi-domain numbers were in log1p/area-normalized space (individual in physical units); the verification above missed this. Corrected: discharge 0.356→0.760, active_fire_count 0.368→0.707, burned_area 0.047→0.521. Arctic/Rangeland unaffected.
<!-- Still open: pretrain vs. finetune being this close is itself worth a comment — does
finetuning still earn its complexity? -->

---

## MD-unitsbugfix0716 — multi_domain — 2026-07-16
**run_id:** N/A (not wired)
**Config delta:** Bug fix only; re-evaluated existing `finetuned_fluxonly`/`pretrained_fluxonly` checkpoints (seeds 1-5, MD-seedsweep0714), no retraining.

### What happened
- Found building Figure 8: multi-domain 03/04 undid only Amazon's z-score, not `log1p` (or discharge's drainage-area normalization) — since Amazon joined multi-domain. Discharge preds ~0.02 vs individual ~5000+ m^3/s at the same station.
- Affected all multi-domain Amazon metrics (`evaluation/{pretrained,finetuned}_fluxonly*/amazon/amazon_metrics*.csv`, 5 seeds + seedavg, both variants) and `predictions/*/amazon/amazon_predictions.parquet`; RMSE/PBIAS and NSE/KGE (log1p is nonlinear) all distorted.
- Fix: `inverse_amazon_log1p()` in `domains/multi_domain/flux_only.py` (mirrors individual `ground_truth_long()`), on preds+obs in 04 and preds in 03 (commit `364a13d`).
- Re-ran `04_evaluate.py --flux-only --seed {1..5}` (~1 min/seed, 10 files) and `03_predict.py --domain amazon --checkpoint finetuned --flux-only --seed 1`; re-aggregated (`run_seed_sweep.py` `aggregate()` now also covers `pretrained_fluxonly`).
- Corrected finetuned seed-avg NSE: discharge 0.356 (individual) → 0.760 (was 0.764), active_fire_count 0.368 → 0.707 (was 0.886), burned_area 0.047 → 0.521. Pretrain ≈ finetune still. Arctic/Rangeland identical to MD-seedsweep0714.
- Regenerated Figures 6 and 7 (all 3 metric files each); removed the stale log-scale RMSE special case (added for the buggy ~1000x Amazon RMSE gap) in `make_remaining_figures.py`'s Figure 6.

### Interpretation & Decisions
- Headline survives: multi-domain fine-tuning still substantially improves Amazon, by a smaller margin. Use these corrected numbers, not MD-seedsweep0714's.
- No other pipeline affected (Arctic/Rangeland have no log1p; individual pipelines correct).

### Follow-up
- Full-target multi-domain variant has the same bug, not yet re-evaluated (not in any figure; fix before use).

---

## AB-capacitypairwise0806 — ablation_test — 2026-08-06 (extended to 5-seed 2026-08-07)
**run_id:** N/A for pairwise runs (multi_domain not wired); capacity-matched runs: `.run_id` sidecars next to `outputs/{amazon,rangeland}_domain/models/best_model*_capmatched.pt` on `vm-sandeep`.
**Config delta:** New `--capacity-matched` (amazon/rangeland 02/03/04 + `model_capacity_matched:`) and `--domains` (multi_domain 02/03/04 + `flux_only.py` helpers); branch `ablation_tests`; design in `ablation_test/ablation_description.md`. 5 flux-only run types: Amazon and Rangeland capacity-matched; pairwise pretrains {Arctic,Amazon}, {Arctic,Rangeland}, {Amazon,Rangeland}. Seed 1 2026-08-06, seeds 2-5 2026-08-07 (`run_ablation.py --seeds 2 3 4 5`, ~6 h), aggregated by new `ablation_test/aggregate_ablation_seeds.py`. Full-3 anchor = MD-seedsweep0714 `pretrained_fluxonly_seed{1..5}` (not rerun).

### What happened
- All 25 runs succeeded. Pre-run code review caught: a naive anchor rerun would have overwritten MD-seedsweep0714's seed-1 checkpoint (now reused instead); `03_predict.py` lacked `--domains` (fixed).
- **Capacity-matched** (5-seed median NSE): Amazon barely moves (active_fire 0.368→0.330, discharge 0.356→0.372, burned_area 0.047→0.032) → capacity isn't Amazon's gain. Rangeland moves a lot (GPP 0.904→0.947, RECO 0.880→0.952, Rg 0.887→0.935, Rm 0.873→0.939) → capacity/dropout explains most of Rangeland's gain (its config used `dropout=0.3` against overfitting on ~2K windows).
- **Amazon pairwise:** fire/burn need Arctic specifically (capacity-matched 0.330/0.032 → +Rangeland 0.431/0.274 → +Arctic 0.708/0.639 → full-3 0.707/0.553); discharge gains from any pooling (0.372 → +Rangeland 0.750 → +Arctic 0.812 → full-3 0.758). Two mechanisms, confirmed at 5 seeds.
- **Arctic** (architecture already = trunk; 500K windows vs ~18K/~2K) gains esp. RECO: individual 0.472 → +Amazon 0.733 / +Rangeland 0.702 / full-3 0.674. Amazon (no respiration target) helps as much → regularization/training dynamics, not target transfer; individual RECO has the highest seed std (0.192 vs 0.01-0.06) → optimization-unstable, not data-starved. One partner beats three (GPP 0.945/0.934 vs 0.925; RECO 0.733/0.702 vs 0.674), plausibly gradient-budget dilution.
- Seed-1 vs 5-seed: most targets within ~0.01-0.03 NSE; Arctic RECO and Rangeland Rm swung ~0.07-0.11.
- Figures: `ablation_test/figures/ablation_comparison_{RMSE,NSE,PBIAS}.png` (seed 1) and `..._seedavg.png` (cite these), plus `ablation_summary_metrics[_seedavg].csv`, via `shared/plots.py`'s `draw_metric_boxplot_panel`.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
- Arctic RECO training-dynamics mechanism is still a hypothesis; compare epoch/early-stopping counts across conditions.
- multi_domain MLflow still not wired; full-target variant not covered.

---

## HP-sweep0812 — hyperparameter_tuning — 2026-08-12
**run_id:** N/A — cells log to each domain's MLflow (IDs not captured); multi_domain has none.
**Config delta:** New `--model-size {small,medium,large}` + `model_{size}:` config blocks for Arctic/Amazon/Rangeland 02/03/04 (additive); see `hyperparameter_tuning/hyperparameter_tuning_description.md`. Branch `feat/individual-tuning-and-nse-decomposition` (off `main` + `ablation_tests`); `feat/rangeland-obs-multidomain` is parked and intentionally excluded.

### What happened
- `hidden_dim` sweep, seed 1, select by min val loss: Arctic 128/256/512, Amazon 64/128/256, Rangeland 32/64/128 (+ extensions). Arctic/Amazon cells reused from the parked branch (same config/data); only Rangeland trained fresh.
- **Arctic:** medium (256) = current production; 0.102/0.085/0.085 (256 and 512 tied; won on tie order).
- **Amazon:** small (64) by argmin, but 0.517/0.518/0.517 (~0.2% spread = noise).
- **Rangeland:** 32/64/128 = 0.042/0.042/0.038; extended → 256 (xlarge) 0.0249 (~40% better than 128), 512 (xxlarge) 0.0356 → 256 is a real peak.
- Added `shared/metrics.py::kge_components()` (r/alpha/beta) to `compute_metrics()` and to `shared/seed_aggregation.py`'s default `metric_columns`; new `metric_decomposition/decompose_kge.py` (Individual/Pretrained/Fine-tuned per target).
- Figure 6 replaced by its per-target variant; Figures 6/7 gained KGE panels (fig6d/fig7d); `make_remaining_figures.py` split into `make_figureN.py` + `_common.py`.

### Interpretation & Decisions
- **Amazon: not promoted** (production 128 unchanged); a real answer would need more seeds.
- **Rangeland: promoted** — see RG-retune0812.
<!-- NEEDS HUMAN REVIEW: does the Amazon "no meaningfully better size" finding belong in the
     manuscript as a stated robustness check, or omitted as a null result? -->

### Follow-up
- AB-capacitypairwise0806 used Rangeland's original 64/dropout=0.3 baseline, not the new 256/0.15 (flagged in `ablation_description.md`); rerun is open.
- Amazon extended along a different axis instead of more seeds — see HP-amazonext0813.

---

## HP-amazonext0813 — hyperparameter_tuning — 2026-08-13
**run_id:** N/A (Amazon MLflow, IDs not captured)
**Config delta:** New `model_xsmall` (hidden_dim=32) and `model_xxsmall` (16) in `config/amazon_domain.yaml`; `--model-size` choices extended in `domains/amazon_domain/{02_train,03_predict,04_evaluate}.py`.

### What happened
- Extended down to 32, 16 (dropout=0.15, seed 1): best val 16→0.513, 32→0.517, 64→0.517, 128→0.518, 256→0.517 — <1% spread over 16x, no trend (`hyperparameter_tuning_results.png` panel b is flat).

### Interpretation & Decisions
- Confirms HP-sweep0812's "not promoted": Amazon val loss is insensitive to hidden_dim. Production (128) unchanged.
- `hyperparameter_tuning_winners.yaml`'s `amazon: xxsmall` is a mechanical argmin over a flat line, not a winner; `hyperparameter_tuning_description.md`'s Resolution table is authoritative.
- Same NEEDS HUMAN REVIEW question as HP-sweep0812 (robustness statement for 16-256, or omit?).

### Follow-up
- Next: `feedforward_dim` (held at 512, an 8:1 ratio at hidden=64) — see HP-amazonffn0813. Dropout/num_layers untested.

---

## HP-amazonffn0813 — hyperparameter_tuning — 2026-08-13
**run_id:** N/A (same as above)
**Config delta:** New `model_ffn_narrow` (feedforward_dim=128) and `model_ffn_std` (256), both hidden_dim=64; `--model-size` choices extended.

### What happened
- ffn sweep at hidden=64: 128 (2:1), 256 (4:1), and reused `model_small` for 512 (8:1; production is 128/512 = 4:1). Best val: 128→0.517, 256→0.513, 512→0.517 — flat.

### Interpretation & Decisions
- Two architecture axes flat (~0.513-0.518) → val loss floor likely set by irreducible noise in the targets, not capacity. Production (hidden=128, ffn=512) kept.
- Same NEEDS HUMAN REVIEW question, now with two null axes.

### Follow-up
- Dropout and num_layers untested; next: num_layers — see HP-amazonlayers0813.

---

## HP-amazonlayers0813 — hyperparameter_tuning — 2026-08-13
**run_id:** N/A (same as above)
**Config delta:** New `model_layers2`/`model_layers4`/`model_layers6` (hidden=64, ffn=256); `--model-size` choices extended.

### What happened
- `num_layers` 2, 4, 6 (6 = multi-domain trunk depth), anchor = `model_ffn_std`'s 3 layers (0.513). Best val: 2→0.516, 3→0.513, 4→0.520, 6→0.519 — ~1.4% spread, vs Rangeland's ~40% capacity gain.

### Interpretation & Decisions
- Three axes flat → reinforces the noise-floor reading. Production (128/512/3 layers) kept.
- Slight depth penalty (4/6) consistent with mild overfitting but within noise.

### Follow-up
- Dropout last untested axis — see HP-amazondropout0813.

---

## HP-amazondropout0813 — hyperparameter_tuning — 2026-08-13
**run_id:** N/A (same as above)
**Config delta:** New `model_dropout10`/`model_dropout20`/`model_dropout30` (hidden=64, layers=3, ffn=256); `--model-size` choices extended.

### What happened
- Dropout 0.10/0.20/0.30, anchor 0.15 (0.513). 0.20 = Amazon production; 0.30 = Rangeland's original. Best val: 0.10→0.511, 0.15→0.513, 0.20→0.518, 0.30→0.516 — same ~1.4% band.

### Interpretation & Decisions
- **Four for four** flat (all cells 0.511-0.520): best-supported explanation is irreducible variance in Amazon's targets, not architecture. No promotion candidate on skill grounds; production (128/512/3/0.2) kept at this point.
- Combined plot: `hyperparameter_tuning/figures/amazon_architecture_search.png` (`plot_amazon_architecture_search.py`; 4 panels, shared y-axis).
- NEEDS HUMAN REVIEW question from HP-sweep0812 now fully backed: state "no better architecture across 4 dimensions" as a robustness result in the manuscript?

### Follow-up
- All dimensions swept. Smallest/fastest point (hidden=64, ffn=256, layers=3, dropout=0.10) promoted anyway, purely for efficiency — see AZ-retune0813.

---

## AZ-retune0813 — amazon_domain — 2026-08-13
**run_id:** per-seed runs in Amazon MLflow — `.run_id` sidecars next to `outputs/amazon_domain/models/best_model_seed*.pt` on `vm-sandeep`.
**Config delta:** `config/amazon_domain.yaml` `production:` `hidden_dim` 128→64, `dropout` 0.2→0.10, `feedforward_dim` 512→256 (`num_layers` 3, `num_heads` 4 unchanged) — the exact tested config.

### What happened
- Promotion **purely for efficiency** (all 4 sweeps flat), unlike RG-retune0812's ~40% gain. 5-seed rerun (~30 s/seed), outputs overwritten in place.
- 5-seed median NSE old → new: discharge 0.356→0.371, active_fire_count 0.368→0.310, burned_area 0.047→0.008 (old per AZ-seedsweep0714). burned_area no-skill either way; active_fire -0.058 within known noise (cf. AB-capacitypairwise0806's 0.368→0.330).
- Headline unchanged: Individual still well below multi-domain fine-tuned (discharge 0.371 vs 0.76, active_fire_count 0.310 vs 0.71, burned_area 0.008 vs 0.52; `fig6b`/`fig6d`).
- Regenerated: Figure 4 (Amazon row), 5 (Amazon curves), 6 (Amazon, 4 metrics), 7 (Amazon maps; KGE color scale now ±791% due to near-zero burned_area baseline), 8 (Amazon station), ablation figures (`make_ablation_figures.py`, Individual arm), KGE decomposition (Amazon panel).
- Resynced stale local `outputs/amazon_domain/preprocessed/test.pkl` (lacked `drainage_area`, broke `make_figure8.py`). HP figures no longer highlight any point (user instruction).

### Interpretation & Decisions
- Deliberate efficiency choice; manuscript should say "Amazon doesn't need as large an architecture as assumed", not "we found a better architecture".
<!-- NEEDS HUMAN REVIEW: does this efficiency-only promotion need its own methods sentence,
     separate from Rangeland's genuine-improvement retune, so readers don't conflate them? -->

### Follow-up
- None; closes the Amazon tuning thread (HP-sweep0812 → AZ-retune0813).

---

## RG-retune0812 — rangeland_domain — 2026-08-12
**run_id:** per-seed runs in Rangeland MLflow — `.run_id` sidecars next to `outputs/rangeland_domain/models/best_model_seed*.pt` on `vm-sandeep`.
**Config delta:** `config/rangeland_domain.yaml` `production:` `hidden_dim` 64→256, `dropout` 0.3→0.15 (exact tested "xlarge" from HP-sweep0812); `num_layers`/`num_heads`/`feedforward_dim` unchanged (3/4/256).

### What happened
- 5-seed rerun (flux-only, ~1 min/seed), re-aggregated.
- 5-seed median NSE old → new: GPP 0.904→0.97, RECO 0.880→0.96 (now above multi-domain finetuned 0.93), Rg 0.887→0.95, Rm 0.873→0.96 (above finetuned 0.90). `kge_decomposition_rangeland.png`: the beta gap (Individual 0.85-0.95 vs 0.97-1.0) has closed; r/alpha/beta ~0.95-1.0 for all arms.
- Regenerated: Figure 4 (Rangeland row), 6 (Rangeland, 4 metrics), 7 (Rangeland maps; vmax NSE 25.5%→13.5%, KGE 53.4%→23.4%), 8c, `metric_decomposition/` Rangeland panel. Arctic/Amazon panels byte-identical (verified).

### Interpretation & Decisions
- Quantitatively confirms AB-capacitypairwise0806: with a tuned baseline, individual Rangeland is competitive with multi-domain and slightly better for RECO/Rm.
<!-- NEEDS HUMAN REVIEW: does the manuscript's Rangeland framing change from "multi-domain helps"
     to "multi-domain doesn't hurt; a fair individual baseline explains the earlier gain"?
     Substantive claim change, not just a number update. -->

### Follow-up
- Rangeland full-target variant not retuned (lower priority, see `current_project_status.md`).
- Multi-domain unaffected (individual architecture doesn't feed the trunk).

---

## AB-rangelandupdate0812 — ablation_test — 2026-08-12
**run_id:** N/A — post-hoc aggregation/plotting only.
**Config delta:** `make_ablation_figures.py`: Rangeland "Individual" = tuned production model (RG-retune0812) instead of the `--amazon-sized` stand-in; Rangeland "Capacity-matched" arm dropped (see `ablation_description.md`); added KGE panel. `aggregate_ablation_seeds.py` pinned to RMSE/NSE/KGE/PBIAS (runs predate r/alpha/beta).

### What happened
- Pre-existing gap: the 3 pairwise arms' 5-seed data had never been aggregated into `_seedavg` (only capacity-matched had); fixed by running `aggregate_ablation_seeds.py` (CPU, no retraining).
- 5-seed median NSE with tuned Individual: matches or beats every pairwise/full-3 arm — GPP 0.97 vs 0.93-0.98, RECO 0.96 vs 0.90-0.94, Rg 0.95 vs 0.93-0.98 (full-3/+Arctic narrowly ahead), Rm 0.96 vs 0.89-0.91. KGE similar but more mixed (Individual leads GPP/RECO/Rm, +Arctic/full-3 lead Rg). Multi-domain has no clear Rangeland edge once the baseline is fair.
- Amazon/Arctic panels unchanged.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: with a fair baseline, cross-domain pretraining shows no clear Rangeland
     benefit on any flux target (Rm looks worse). With RG-retune0812, does the manuscript story become
     "capacity/regularization was the entire Rangeland effect" -- hypothesis 1 (capacity confound)
     explains Rangeland, hypotheses 2/3 (anchor-domain / generic pooling transfer) are Amazon-only? -->

### Follow-up
- Rangeland `--amazon-sized`/`--capacity-matched` outputs kept on disk but not plotted.
- Check whether any other reported ablation numbers came from seed-1-only data (not verified beyond this fix).
- Same day: Amazon "Capacity-matched" arm also dropped (user request, for consistency) and the seed-1 figure/CSV variant removed — `ablation_test/figures/` now holds only 5-seed PNGs (RMSE/NSE/PBIAS/KGE) and one summary CSV.

---

## Entry Template (copy when logging a new run)

```
## AR-<run_id[:8]> — arctic_domain — YYYY-MM-DD
**run_id:** `<full-uuid>`
**Config delta:** <what changed vs. prior run, or "initial run">

### What happened
- <2-3 terse bullets: key numbers, derived from MLflow>

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->

### Follow-up
-
```
