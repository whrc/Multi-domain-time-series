# Key Findings Log

<!-- Claude Instructions ─────────────────────────────────────────────────────
PURPOSE: Human interpretation + Claude-drafted summaries of what a run revealed.
Metric numbers do NOT go here — MLflow is SSOT for all numeric data.

WHEN TO ADD AN ENTRY (only when criteria are met):
  1. A new domain evaluation completes for the first time
  2. A config change produces surprising or notably different results
     (NSE delta > 0.05 vs. prior run on same domain)
  3. A failure or unexpected behaviour needs explanation
  4. A design decision was made as a direct result of experiment outcomes

APPEND RULES:
  - One entry per qualifying run. Never edit past entries.
  - Create the entry AFTER the run completes (when MLflow run_id is known).
  - exp_id = AR-<run_id[:8]> (first 8 hex chars of MLflow run_id).
  - In MLflow: call mlflow.set_tag("exp_id", "AR-<run_id[:8]>") so the two
    are cross-referenceable.

OWNERSHIP:
  - "What happened" bullets → Claude drafts (derived from MLflow metrics,
    NOT copied verbatim).
  - "Interpretation & Decisions" → NEEDS HUMAN REVIEW marker stays until
    the human fills this section and removes the marker.
  - Claude never removes the NEEDS HUMAN REVIEW marker.
──────────────────────────────────────────────────────────────────────────── -->

---

## AR-21c64242 — arctic_domain — 2026-07-07
**MLflow run_id:** `21c642428f5e43e4b036f35624bccfa1`
**Config delta:** First 50K run after fixing a `-9999` fill-value contamination bug in
`01_preprocess.py` (some grids' projected/`_sc` NetCDFs don't declare `_FillValue`, so
xarray couldn't auto-mask it — ~534M entries masked explicitly instead) + added gradient
clipping (`grad_clip_norm: 1.0`) to `shared/training.py`. Prior 50K run (pre-fix) is
invalid and was deleted.

### What happened
- Training loss now converges smoothly with no divergence (prior run spiked sharply
  around epoch 18-20); early stopping triggered normally at epoch 40.
- Pooled validation/test NSE (all pixels combined) is modest but real for all 4 targets —
  roughly 0.15-0.34 depending on target, per `val_pred_vs_true.png`.
- Per-pixel NSE/KGE for ALD and VEGC (yearly, slowly-accumulating pool targets) is very
  weak — most individual pixels score below -1, in contrast to GPP/RECO (monthly fluxes)
  which look reasonable per-pixel too. Traced to two distinct causes: (a) ~1% of ALD/VEGC
  pixel-periods have a literally constant observed value, making NSE/KGE mathematically
  undefined (now flagged via `obs_degenerate` in `metrics_df_by_period`, excluded from
  boxplot/spatial-map aggregation only — `metrics.csv` keeps every raw row); (b) for the
  remaining ~99% of rows, per-pixel skill for ALD/VEGC is genuinely weak — confirmed by
  testing relative-variance exclusion thresholds up to 10% of global std, which still left
  median per-pixel NSE deeply negative for both targets while GPP/RECO stayed fine.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
- Working hypothesis (Claude draft, pending confirmation): the model's only inputs are
  static covariates (soil, drainage, fire-return-interval, topo, vegetation type), CO2,
  and a 12-month climate window (`tair`, `precip`, `nirr`, `vapor_press`) — there is no
  autoregressive/lagged-target feature, i.e. the model never sees a target's own prior
  value. GPP/RECO are fluxes largely set by concurrent climate, so a stateless window
  suffices. ALD and VEGC are accumulated pool quantities whose current level depends on
  decades of prior thermal/disturbance history that isn't visible to the model at all, so
  it converges toward each pixel's static-covariate-implied baseline rather than its true
  trajectory — consistent with decent pooled (cross-pixel) NSE but poor per-pixel
  (within-pixel) NSE.
- Decision: accept this as a known limitation for now rather than change the architecture
  mid-flight; goal 1 (dedicated per-domain models) stays in scope as originally defined.

### Follow-up
- Try `seq_len=24` (or 36) months for the Arctic model — a hyperparameter change, not an
  architecture change — to see whether a longer climate history window alone recovers
  some ALD/VEGC per-pixel skill, before considering a bigger design change (e.g. feeding
  the target's own prior value back in as an autoregressive input, which would need a
  dedicated design discussion — recursive state at inference, error accumulation over
  multi-decade projections).

---

## AR-d59b948a — arctic_domain — 2026-07-07
**MLflow run_id:** `d59b948aa6ab4939a8413169278d5c51` (H1 only — see below; the other 5 diagnostic
scripts in this batch trained outside `02_train.py`'s `tracking.training_run()` wrapper and
have no MLflow run of their own)
**Config delta:** Follow-up to AR-21c64242's "try seq_len=24" note. A staged batch of 6
diagnostic experiments (`domains/arctic_domain/diag_h5_insample_eval.py`,
`diag_h2_fluxonly.py`, `diag_h3_smallmodel.py`, `diag_h4_lastpos.py`, `diag_h3h4_combined.py`,
plus one `01_preprocess.py --capped-stride 150` run labeled H1) probing 4 hypotheses for why
50K→500K and seq_len=12→24 both failed to meaningfully improve held-out per-pixel test NSE.
None of these diagnostics touch `shared/*.py`, `run_arctic.py`, or `config/arctic_domain.yaml`
production defaults — all disposable, all reuse the existing 50K train/val/test pkls except H1
(which reprocessed with `capped_stride=150` instead of 24, at the same ~50K window budget).

### What happened
- **H5 (in-sample check, 500K checkpoint on its own train pixels):** median per-pixel NSE
  nearly as bad in-sample as held-out (ALD -367 vs -434, GPP -0.26 vs -0.30, RECO -0.10 vs
  -0.17, VEGC -263 vs -358) — ruled out pure spatial-generalization difficulty as the dominant
  explanation; pointed at an optimization/capacity/sampling problem instead.
- **H2 (drop ALD/VEGC from the loss, flux-only):** made GPP/RECO *worse*, not better (GPP
  0.007 vs 50K baseline's 0.127, RECO -0.159 vs baseline's -0.069) — refuted the "pool targets
  are corrupting flux-loss gradients" hypothesis.
- **H3 (shrink model 5-6M→101K params, same 356-pixel baseline data):** GPP 0.127→0.563, RECO
  -0.069→0.035; ALD/VEGC stayed catastrophic.
- **H4 (loss scored only at each window's last/max-context position, matching how
  03_predict.py/04_evaluate.py always score a checkpoint, same 356-pixel baseline data):** ALD
  -577→-67 (big improvement), but GPP/RECO/VEGC all got worse (GPP -0.708, RECO -0.335, VEGC
  -2338) — mixed result, confounded with the still-oversized model.
- **H1 (pixel-sampling density: `capped_stride` 24→150 at the same ~50K window budget, full
  production-size model, 356→2148 distinct train pixels, ~137→~23 windows/pixel):** by far the
  largest single improvement of the batch — ALD -577→-191, GPP 0.13→**0.72**, RECO
  -0.07→**0.30**, VEGC -1057→-374. Notably better than H3 alone on GPP/RECO, and better than
  the 500K run despite H1 having *fewer* total pixels (2148 vs 500K's 3631) — the lever appears
  to be pixel *diversity relative to per-pixel window depth* in the sampling budget, not raw
  pixel count or raw window count on their own.
- **H3+H4 combined (small model + last-position loss, run on H1's denser 2148-pixel data, not
  the original 356-pixel baseline):** ALD -307, GPP 0.409, RECO 0.141, VEGC -1293 — *worse*
  across every target than H1 alone (full-size model, same pixel data, ordinary loss). The
  model-size and last-position-loss changes that helped on the sparse 356-pixel baseline
  actively hurt once the underlying pixel-density problem was fixed.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- H1's result (pixel diversity >> model size / loss-position tweaks) suggests the next
  experiment is pushing `capped_stride` further (e.g. 300-500) at the same or a larger window
  budget to see whether GPP/RECO keep improving or plateau, and whether ALD/VEGC (still
  catastrophic in every variant so far) respond at all to pixel diversity alone or need the
  separately-hypothesized autoregressive-state input (AR-21c64242) regardless.
- All 6 `diag_*.py` scripts are still on disk, undeleted, pending the user's review of this
  entry — to be deleted (or one promoted into the production pipeline) once reviewed.

---

## AR-widesweep0708 — arctic_domain — 2026-07-08
**MLflow run_id:** N/A — this run predates the density-sweep code changes below; MLflow
tracking was not active for it (Stage 2, not yet wired — see `environment_spec.md`).
**Config delta:** Follow-up to H1's `capped_stride`=150 result — pushed `capped_stride` to 300
at the same ~50K window budget ("50K-wide"), per AR-d59b948a's own follow-up suggestion.

### What happened
- Median per-pixel val NSE was catastrophic across every target (ALD ≈ -7.05M, GPP ≈ -432K,
  RECO ≈ -733, VEGC ≈ -3.05M for the ssp1_2_6 scenario alone) — far worse than both the
  `capped_stride`=24 baseline and H1's `capped_stride`=150 result, breaking the "wider stride is
  better" trend H1 suggested.
- **Root cause (found while investigating this result, not before running it):**
  `01_preprocess.py`'s `effective_stride = {"train": capped_stride, "val": capped_stride, "test":
  capped_stride}` tied val/test's pixel subsampling to the *same* `capped_stride` as train. Every
  time `capped_stride` changed to test a new training density (24 → 150 → 300), **the held-out
  test/val population also changed** (356 / ~2145 / ~4057 pixels — different actual pixels, not
  just counts). So the 24→150→300 pattern (bad → good → bad again) conflated genuine
  training-density sensitivity with the held-out population's own composition changing every
  time — not a real "wider is worse past 150" finding.
- This run's own checkpoint/data were archived (`train_50K_s300.pkl`, `val_metrics_50K_s300.csv`)
  rather than discarded, but its numbers are **not directly comparable** to any other run in this
  log — they were scored against a val/test population unique to this run alone.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- Fixed in code (see AR-controlledsweep0708 below): decoupled train's stride from val/test's
  (`--train-capped-stride`), added a `--label` override so density variants stop overwriting
  each other's checkpoints, and added `--sweep-strides` to fetch each grid once and split it
  into N stride-specific train pkls instead of paying for N separate GCS passes.

---

## AR-controlledsweep0708 — arctic_domain — 2026-07-08
**MLflow run_id:** N/A — MLflow tracking not active this run (Stage 2, not yet wired).
**Config delta:** Properly controlled re-run of the density question above. Val/test built
**once**, locked at `capped_stride`=150 (2143/2145 pixels respectively, same `grids_hash` across
all 5 points below), then train swept over `capped_stride` ∈ {50, 100, 150, 200, 250} at the
same ~50K window budget, all scored against that one fixed held-out population — isolating
genuine training-density sensitivity from the population-confound in AR-widesweep0708.
Evaluated on **validation** only (test set intentionally not touched — reserved for the final
model choice, not repeated peeking during this comparison).

### What happened
- Median per-pixel val NSE / RMSE per target, by train `capped_stride`:

  | stride | ALD (NSE) | GPP (NSE) | RECO (NSE) | VEGC (NSE) | best val loss |
  |---|---|---|---|---|---|
  | 50  | -65.5  | -1.26 | -10.23 | -3150.1 | *(not captured — see note)* |
  | 100 | -126.9 | 0.61  | 0.33   | -547.6  | 0.3995 |
  | 150 | -190.1 | 0.57  | 0.12   | -1516.6 | 0.3861 |
  | **200** | **-39.2** | **0.73** | **0.38** | **-395.3** | **0.3725** |
  | 250 | -394.1 | 0.51  | -0.03  | -2870.5 | 0.5092 |

  (RMSE per target/stride, same ordering: ALD 1.26/1.29/1.26/**0.94**/1.52, GPP
  111.0/40.5/40.2/**33.7**/51.7, RECO 105.6/27.8/29.8/**26.8**/36.9, VEGC
  10499.7/5871.6/8705.8/**5574.4**/9163.0 — see `outputs/arctic_domain/models/val_metrics_50K_s*.csv`
  for full per-ssp breakdown.)
- **`capped_stride`=200 wins on every single target simultaneously** (highest NSE, lowest RMSE,
  lowest best-val-loss) — a clean, non-arbitrary signal now that the population confound is
  fixed, unlike the noisy 24→150→300 pattern that motivated this whole re-run.
- `stride`=50 is clearly the worst point (matches the already-known-bad 24 endpoint); 100/150/250
  are all mediocre-to-poor with VEGC catastrophic throughout; only 200 gets GPP/RECO solidly
  positive (0.73/0.38) with ALD/VEGC still very negative but far less so than elsewhere.
- Sanity check vs H1 (`stride`=150, but H1's numbers were on **test**, this run's are **val** —
  not a perfect apples-to-apples comparison): ALD matches closely (-190.1 vs H1's -191.4), GPP/
  RECO are same sign but lower (0.57 vs 0.724, 0.12 vs 0.298), VEGC deviates substantially
  (-1516.6 vs -373.6) — plausible given per-pixel NSE's known high variance and a different
  (val, not test) population, but worth keeping in mind rather than fully explained.
- `50K_s50`'s best-val-loss was lost to an operator error (same log filename reused across a
  relaunch, overwriting the line) — not rerun to avoid discarding its already-valid NSE/RMSE
  results for a non-seeded stochastic replicate; a real but minor gap in this record.
- Also fixed in the same work: `shared/inference.py`'s dense (stride=1) evaluation pass was
  taking ~5 min per point (batch_size=256, no DataLoader parallelism, ~140x more windows than
  training's own sparse validation loop touches) — batching (batch_size 256→8192,
  num_workers=4) cut this to ~2:40-2:44; a follow-up vectorization of the per-window result
  assignment (millions of Python-loop iterations → one vectorized slice per segment) was
  verified numerically identical but did not meaningfully reduce wall time further, meaning the
  remaining ~2:40 is genuine GPU/DataLoader cost, not that specific loop.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- Next: preprocess a 500K-window budget at the winning `capped_stride`=200 (train only — val/test
  are fixed-size constants in config, independent of `--train-size`, so the *same* locked
  val/test population carries over automatically), train+evaluate on `vm-sandeep`, and compare
  against this entry's 50K/`stride`=200 result to see whether more data further improves GPP/RECO
  or helps ALD/VEGC at all.
- **Superseded by AR-sspfix0708 below** — this entry's winning stride (200) does not hold once a
  separate SSP-scenario-collapse bug (found afterward) is fixed; kept here unedited per the
  append-only rule.

---

## AR-sspfix0708 — arctic_domain — 2026-07-08
**MLflow run_id:** N/A — MLflow tracking not active this run (Stage 2, not yet wired).
**Config delta:** Bug-fix re-run of AR-controlledsweep0708's density sweep. `01_preprocess.py`'s
train-save step keyed train records by `(grid, y, x)` only (excluding `ssp`), so a pixel's two
SSP-scenario records (`ssp1_2_6`, `ssp5_8_5`) silently collapsed to one via dict-overwrite —
every train pixel in AR-controlledsweep0708 (and the already-trained 500K/`stride`=200 run)
carried only one scenario instead of both. Verified directly: records:pixels ratio was exactly
1:1 for `train_50K_s200.pkl`/`train_500K_s200.pkl`, should be ~2:1 (matching val/test's already-
correct ~2:1 ratio, since they don't go through this code path). Fixed by regrouping into a
`defaultdict(list)` keyed by pixel (preserving every scenario record per pixel); verified via a
tiny 4-grid regression test (ratio 1.0 → 2.0) before redoing the real sweep. AR-controlledsweep0708's
buggy checkpoints/csvs archived to `outputs/arctic_domain/_archive_buggy_ssp_collapse_20260708/`
(not deleted).

### What happened
- Redid the same 5-point `capped_stride` ∈ {50, 100, 150, 200, 250} sweep at ~50K windows, same
  locked val/test population (only train regenerated), retrained all 5 points on corrected data:

  | stride | ALD (NSE) | GPP (NSE) | RECO (NSE) | VEGC (NSE) | best val loss |
  |---|---|---|---|---|---|
  | 50  | -64.7  | 0.754 | 0.415 | -391.7 | 0.4263 |
  | 100 | -54.5  | 0.798 | 0.470 | -138.3 | 0.3858 |
  | 150 | -239.6 | 0.762 | 0.431 | -378.6 | 0.3567 |
  | 200 | -55.1  | 0.736 | 0.460 | -571.1 | 0.3824 |
  | **250** | **-43.5** | **0.822** | **0.528** | **-134.5** | **0.3393** |

  (RMSE per target/stride, same ordering: ALD 0.99/0.88/1.20/0.98/**0.79**, GPP
  33.6/30.3/31.5/36.9/**30.1**, RECO 27.2/23.4/24.2/27.4/**23.3**, VEGC
  3867/2951/4902/4685/**3175** — source: `outputs/arctic_domain/models/val_metrics_50K_s*.csv` on
  `vm-sandeep`, not yet copied locally.)
- **`stride`=250 now wins on every target simultaneously** (highest NSE, lowest RMSE, lowest
  best-val-loss) — this **reverses** AR-controlledsweep0708's buggy conclusion that `stride`=200
  won. Restoring the missing SSP scenario changed which density is best, not just the absolute
  numbers.
- Every point improved substantially over its AR-controlledsweep0708 (buggy) counterpart on
  GPP/RECO (e.g. `stride`=100's GPP NSE 0.61→0.80, RECO 0.33→0.47) — consistent with the fix
  simply giving the model twice as much (correctly paired) training signal per pixel, not a
  fluke.
- Trend across 50→100→150→200→250 is not monotonic (150 dips, 200 partially recovers, 250 peaks)
  — read as still within run-to-run noise for a single non-seeded run per point, not a smooth
  "wider is strictly better" curve.
- 53 grid failures during the redo (more than typical) left all 5 points at ~39,200-39,400 actual
  windows / 200 covered grids (below the 50K/~220-grid target) — uniformly across all 5 points,
  so the *relative* comparison should still be valid even though the absolute dataset came in
  short.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- Next: build a staggered-windowing variant of the winning `stride`=250 (per-pixel deterministic
  phase offset so different pixels sample different calendar positions, since window starts are
  currently identical across all pixels — see `arctic_description_data_handling.md`), compare
  vanilla vs. staggered `stride`=250 to pick the final recipe before scaling to 500K/2M.
- Whether `stride`=300 is worth adding to this sweep is an open question — the only prior
  `stride`=300 data point (AR-widesweep0708) predates *both* the population-confound fix and this
  SSP-collapse fix, so it's not usable evidence either way.

---

## AR-stagger0709 — arctic_domain — 2026-07-09
**MLflow run_id:** N/A — MLflow tracking not active this run (Stage 2, not yet wired).
**Config delta:** New `--stagger` flag in `01_preprocess.py` (train-only, save-time transform):
each train pixel gets a deterministic phase offset `crc32(seed:grid:y:x) % stride`, trimming that
many rows off the front of its time series before saving, so different pixels' windows start at
different calendar positions — previously every pixel sampled the identical fixed set of window
starts (see AR-sspfix0708's follow-up and `arctic_description_data_handling.md`). Same phase for
both of a pixel's SSP scenario records. Compared `50K_s250_staggered` against AR-sspfix0708's
winning vanilla `stride`=250, same val/test population (`grids_hash` verified to match), same
260-grid list. **Caveat accepted by the user**: the staggered run had better per-grid pass-2 fetch
success this time (233/260 grids, 3298 pixels) than the archived vanilla run (200/260 grids, 2816
pixels) — pass 2 tolerates per-grid failures, so two separate runs of "the same" experiment can
realize slightly different populations. Not re-controlled for; accepted as a minor confound.

### What happened
- Median per-pixel val NSE / RMSE per target, vanilla vs. staggered `stride`=250:

  | metric | vanilla s250 | staggered s250 |
  |---|---|---|
  | ALD NSE | -43.5 | -54.8 |
  | GPP NSE | 0.822 | **0.842** |
  | RECO NSE | 0.528 | **0.555** |
  | VEGC NSE | -134.5 | **-131.1** |
  | best val loss | 0.3393 | **0.3223** |

  (RMSE, vanilla/staggered: ALD 0.79/0.82, GPP 30.1/**28.9**, RECO 23.3/**23.1**, VEGC
  3175/**2831** — source: `outputs/arctic_domain/models/val_metrics_50K_s250{,_staggered}.csv`.)
- Staggering improves best val loss (~5% lower) and 3/4 targets (GPP, RECO, VEGC — both NSE and
  RMSE), but ALD gets modestly worse (NSE -43.5→-54.8, RMSE 0.79→0.82). ALD has been the weakest,
  most volatile target throughout every variant tested since AR-21c64242 (accumulated-pool target
  with no autoregressive input), so a single-run regression there is consistent with that
  target's known high variance rather than a clear staggering-specific harm.
- **Staggered `stride`=250 is the new overall winning recipe** — net positive across the
  aggregate loss and most targets, at no extra preprocessing cost (reuses the same pixel
  selection, only a save-time trim).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- Next: scale the winning recipe (staggered `stride`=250) up to 500K and 2M train sizes, now that
  a final density + windowing recipe is settled — deferred until this point per the user's
  explicit sequencing request earlier in this investigation.

---

## AR-500Kstagger0709 — arctic_domain — 2026-07-09
**MLflow run_id:** N/A — MLflow tracking not active this run (Stage 2, not yet wired).
**Config delta:** Scaled the winning `50K_s250_staggered` recipe up to `--train-size 500000`
(same `--train-capped-stride 250 --stagger`, same locked val/test population — `grids_hash`
verified to match before launch), pinned to the same 260-grid list used throughout this
investigation (auto-discovery is not stable run-to-run, so `--grids` must be pinned explicitly
to keep val/test locked — learned the hard way when a bare, unpinned run this session found 263
grids instead of 260 and would have silently rebuilt val/test with a different population).
Preprocessing on `vm-cpu-sandeep` took 1h29m (pass 1: ~9 min, pass 2 refetch of 253/260 grids:
~1h20m — needed nearly a full re-fetch since 500K's wider pixel target invalidates most of the
pass-2 cache built for 50K). `train_500K_s250_staggered.pkl`: 209/260 grids covered, 29,175
pixels, 57,783 records (ratio 1.98, correctly paired SSP scenarios), 7.8GB.

### What happened
- Median per-pixel val NSE / RMSE per target, 50K vs. 500K staggered `stride`=250:

  | metric | 50K staggered | 500K staggered |
  |---|---|---|
  | best val loss | 0.3223 | **0.1858** (-42%) |
  | ALD NSE | -54.8 | **-9.4** |
  | ALD RMSE | 0.82 | **0.37** |
  | GPP NSE | 0.842 | **0.934** |
  | GPP RMSE | 28.9 | **17.0** |
  | RECO NSE | 0.555 | **0.757** |
  | RECO RMSE | 23.1 | **15.4** |
  | VEGC NSE | -131.1 | **-53.0** |
  | VEGC RMSE | 2831 | **1347** |

  (Source: `outputs/arctic_domain/models/val_metrics_500K_s250_staggered.csv`, both SSP
  scenarios' median.)
- **Every target improved, cleanly, with no confound this time** (same 260-grid list, same
  locked val population, only train size changed) — best val loss dropped 42%, GPP/RECO both
  crossed further into solidly-positive territory, and even ALD — the weakest, most stubborn
  target across this entire investigation since AR-21c64242 — improved by ~5x in NSE terms
  (-54.8 → -9.4), though it's still deeply negative and the weakest target by far.
- Confirms H1's original finding (AR-d59b948a): pixel diversity/density is the dominant lever,
  and it keeps paying off at 10x the data — no sign of saturation yet at 500K.
- Training itself took ~18 min on `vm-sandeep` (early stopping at epoch 54, best epoch 44),
  GPU utilization ~97-98% throughout, no memory pressure (12-15GB/83GB system RAM,
  ~5GB/40GB GPU memory).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- Next: scale to 2M train size (staggered `stride`=250, same pinned 260-grid list) to see
  whether GPP/RECO/VEGC continue improving or start to plateau, and whether ALD keeps closing
  the gap or has structurally hit its ceiling without an autoregressive input (still an open
  question from AR-21c64242).
- Once the learning-curve (50K/500K/2M) is complete, this becomes the final recipe for
  production Arctic training — worth revisiting `05_learning_curve.py` to formalize the
  saturation check rather than eyeballing three points.

---

## AR-gridsplitsweep0710 — arctic_domain — 2026-07-10
**MLflow run_id:** N/A — MLflow tracking not active this run (Stage 2, not yet wired).
**Config delta:** First sweep under the new grid-level stratified split (`assign_grid_splits()`,
replacing the per-grid pixel split `_grid_split_labels()`) — whole grids are now assigned to
train/val/test (60/20/20), stratified by latitude into 6 bins, so a held-out val/test pixel is
never geographically adjacent to a training pixel in the same tile. Staggered windowing is now
unconditional (the `--stagger` flag was removed — see `arctic_description_data_handling.md` for
the full design). 8 grids (`H11_V16`, `H11_V19`, `H14_V15`, `H16_V7`, `H17_V3`, `H19_V13`,
`H19_V18`, `H9_V19`) were excluded after failing to fetch across ~5 real retry cycles today,
including one with `fetch_timeout_seconds` raised 180→300 — tracked as `FLAKY_GRIDS_20260710`,
separate from the confirmed-permanent `KNOWN_BROKEN_GRIDS`, since this has only been observed on
one day so far. Val/test are entirely new populations under the new split (~48 grids/~366
pixels planned for val, similar for test, at capped_stride=24 — **correction, 2026-07-10 later
same day:** this entry originally said "42 grids/1,920 pixels for val, 44 grids/2,013 pixels for
test," which was a transcription error, not real data — 1,920 pixels at capped_stride=24 would
overshoot the 50K-window target by ~5x; the round-robin subsampler stops once it hits the
target, so ~366 pixels was always the mathematically-consistent figure. See `AR-gridsplit4005000710`
for the full story, including a real bug this mistake led to being caught.) — **not comparable to
any pixel-split-era entry** (`AR-sspfix0708`, `AR-stagger0709`, `AR-500Kstagger0709`), which are
kept unedited as the historical reference for that superseded regime. Swept `capped_stride` ∈
{50, 100, 150, 200, 250, 300, 350} at a ~50K window budget (7 points in one `--sweep-strides`
pass), staggering baked in for every point (no separate vanilla/staggered comparison this time,
per the decision to make staggering permanent).

### What happened
**Superseded, 2026-07-10 later same day:** the table below was evaluated against a val.pkl that
was later silently regenerated (see `AR-gridsplit4005000710`) — the numbers for strides
50-350 in this table no longer match what's on disk. Refer to `AR-gridsplit4005000710` for the
current, mutually-comparable 9-point table (50 through 500). Kept here unedited as the original
record.
- Median per-pixel val NSE / RMSE per target, by stride:

  | stride | best val loss | ALD NSE | GPP NSE | RECO NSE | VEGC NSE |
  |---|---|---|---|---|---|
  | 50  | 0.3935 | -54.1  | 0.836 | 0.420 | -170.5 |
  | 100 | 0.3710 | -84.5  | 0.749 | 0.385 | -800.0 |
  | 150 | 0.3548 | -120.3 | 0.774 | 0.399 | -868.8 |
  | 200 | 0.3400 | -70.0  | 0.791 | 0.478 | -952.7 |
  | 250 | 0.3025 | -49.4  | 0.808 | 0.504 | -249.1 |
  | 300 | 0.3393 | -56.5  | 0.792 | 0.442 | -707.3 |
  | **350** | **0.2827** | **-33.4** | **0.870** | **0.519** | **-77.3** |

  (RMSE, same stride order: ALD 0.77/0.93/0.99/0.86/0.69/0.84/**0.59**, GPP
  28.9/35.5/34.8/31.3/**26.7**/32.0/27.0, RECO 23.5/28.9/30.4/28.2/24.1/28.1/**23.2**, VEGC
  2674/5639/5135/4669/2684/4414/**2257** — source:
  `outputs/arctic_domain/models/val_metrics_50K_s*.csv`.)
- **`stride`=350 — the widest point tested — wins cleanly on best val loss and on 3 of 4
  targets' NSE and RMSE simultaneously** (ALD, RECO, VEGC; GPP RMSE is a close second to
  `stride`=250's 26.7). This is a much more decisive signal than the old pixel-split sweep ever
  produced (`AR-sspfix0708`'s winner, `stride`=250, beat its neighbors by a much smaller margin).
- The trend isn't perfectly monotonic — `stride`=300 dips noticeably below both its neighbors
  (250 and 350) on every target — but the overall shape strongly favors wider strides under this
  harder, more honest generalization test, more so than under the old split.
- ALD and VEGC remain deeply negative throughout (consistent with every prior finding since
  `AR-21c64242` — accumulated-pool targets with no autoregressive input), but both are
  substantially less catastrophic at `stride`=350 than at any other point in this sweep,
  suggesting wider pixel diversity helps them too, not just GPP/RECO.
- Training completed in ~33 minutes total for all 7 points (much faster than the pixel-split-era
  sweeps) — val/test are far smaller under the grid-level split (only ~42-44 eligible grids each
  vs. up to ~250 under the old per-grid split), so dense val evaluation is proportionally faster.
- Preprocessing (pass 1 + pass 2 + all 7 stride saves) took ~50 minutes in one clean attempt,
  after excluding `FLAKY_GRIDS_20260710` — before that fix, the same 8 grids exhausted their
  full retry budget twice in a row (~9 min each, later ~15 min each after raising the timeout),
  blocking pass 1's fail-loud `missing_grids` check entirely.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- `stride`=350 is the sweep's widest point and still winning — worth testing whether even wider
  (400+) continues to help or has started to plateau, before committing to 350 as final.
- Scaling to 500K/2M under the grid-level split is deferred until the stride question above is
  resolved, per the same "confirm before scaling" sequencing used for the pixel-split era.
- `FLAKY_GRIDS_20260710`'s permanence is unconfirmed (single-day observation) — worth re-testing
  without the exclusion on a future date to see if it was a transient bad stretch.

---

## AR-gridsplit4005000710 — arctic_domain — 2026-07-10
**MLflow run_id:** N/A — MLflow tracking not active this run.
**Config delta:** Extended `AR-gridsplitsweep0710`'s sweep with two wider points, `capped_stride`
∈ {400, 500}, launched via `--sweep-strides 400,500` on the same 50K window budget. Pass 1 was
fully cached (instant); pass 2 re-fetched 246/252 grids to cover the wider pixel selection, with
a scattered set of one-off grid failures (different grids than `FLAKY_GRIDS_20260710`, single
occurrences each — normal noise, not a new persistent-flaky pattern).

**Bug found and fixed:** this preprocessing run silently regenerated `val.pkl`/`test.pkl` with a
different pixel population than `AR-gridsplitsweep0710` used (326/327 pixels vs. that entry's
mistakenly-logged 1,920/2,013 — see the correction note on that entry; the true prior figure was
likely already close to 326-366, so the practical difference is probably small, but it couldn't
be verified since the original val.meta.json was already overwritten by the time this was
noticed). The regeneration happened despite `grids_hash` matching between runs — root cause not
fully pinned down (the original val.pkl's exact sidecar content was never captured before being
overwritten). Regardless of root cause, this broke the guarantee that different `--train-size`/
stride runs are evaluated against the same held-out set — **fixed in `3d19d6e`**:
`01_preprocess.py` now raises loudly with a field-level diff of the mismatched sidecar keys
instead of silently rebuilding val/test, and only rebuilds on explicit `--force-recompute`. Val/
test are now effectively frozen from this point forward for any 50K-budget run; a genuinely
intentional change (e.g. different split fractions) will still require `--force-recompute`
deliberately.

To get a valid apples-to-apples comparison across all 9 strides against the now-current (and
now-frozen) val/test, the 7 existing checkpoints (50-350) were retrained from scratch (cheap —
~2-3 min each on the A100) rather than written as a one-off eval-only script, since
`02_train.py` already computes `val_metrics_50K_s<stride>.csv` directly as part of training.

### What happened
- Full corrected 9-point comparison (median NSE/RMSE per target, all evaluated against the same
  val.pkl):

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

  (Source: `outputs/arctic_domain/models/val_metrics_50K_s*.csv`, all timestamped 2026-07-10
  17:14-17:40.)
- **`stride`=400 wins outright — best val loss, and best NSE + best RMSE on all 4 targets
  simultaneously.** This is a stronger, cleaner sweep than `AR-gridsplitsweep0710` produced (that
  one had 350 winning 3/4 targets, GPP RMSE a close second).
- **500 is worse than 400 on every single metric** (val loss 0.2528 vs. 0.2219, every target's
  NSE and RMSE both worse) — the trend peaks at 400, not "wider is always better." Combined with
  300 dipping below its neighbors in the original sweep, the stride-vs-performance relationship
  looks like a real optimum around 350-400 with some run-to-run noise superimposed, not a
  monotonic curve.
- `stride`=200's first retrain produced a clear outlier (best val loss 0.677, early-stopped at
  epoch 8 — a bad LR-finder/init draw). Retrained once more and got 0.2573, back in the normal
  range and consistent with its neighbors. Training-run variance of this magnitude is worth
  keeping in mind when reading any single point in these sweeps — the LR-range-test/init isn't
  fully seeded run-to-run. Not investigated further this session (out of scope), but a candidate
  for a future robustness pass if it recurs.
- ALD and VEGC are still deeply negative everywhere (same accumulated-pool-target pattern as
  every prior entry since `AR-21c64242`), but both are least-bad at `stride`=400 by a wide margin
  (ALD -63.0 vs. next-best -68.6 at 350; VEGC -103.7 vs. next-best -196.7 at 350) — the widest
  useful point so far is also where the hardest targets do best.
- **Recommendation:** `stride`=400 is the new candidate default for scaling to 500K/2M, pending
  the user's review — this is still deferred per the original plan's sequencing.

### Follow-ups
- Root cause of the val/test silent-regeneration bug is not fully understood (only the fix, not
  the "why," is confirmed) — if `01_preprocess.py`'s new loud-failure guard ever fires
  unexpectedly on what looks like a legitimate re-run, the field-level diff in the error message
  is the place to start.
- `stride`=200's training-run variance (0.677 vs. 0.257 on identical config/data) suggests the
  LR-range-test or an unseeded init source isn't fully deterministic — worth a look if a future
  sweep point looks like an unexplained outlier.
- 400 being the widest point tested and still winning leaves open whether 450 or other
  in-between values matter, but per the user's decision this sweep stops at {200,300,400,500}
  plus the retained 50/100 reference points — no further stride widening planned unless the
  500K/2M scale-up results suggest otherwise.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

---

## AR-500Kstride400-0710 — arctic_domain — 2026-07-10
**MLflow run_id:** N/A — MLflow tracking not active this run.
**Config delta:** First scale-up under the grid-level split, using `stride`=400 (the winner from
`AR-gridsplit4005000710`) at a 500K window budget instead of 50K — `--train-size 500000
--train-capped-stride 400`. Pass 1 fully cached (instant); pass 2 re-fetched ~150 train-pool
grids with **zero grid failures** this time. val/test were correctly reused unchanged from cache
(confirmed via the log: "val.pkl already exists and matches config — skipping (cached)") — the
first real-world proof that the `3d19d6e` guard works as intended, since this run legitimately
needed to leave val/test untouched while still touching train.

### What happened
- 500K vs 50K, both `stride`=400, evaluated against the identical frozen val.pkl:

  | run | best val loss | ALD NSE | ALD RMSE | GPP NSE | GPP RMSE | RECO NSE | RECO RMSE | VEGC NSE | VEGC RMSE |
  |---|---|---|---|---|---|---|---|---|---|
  | 50K_s400  | 0.2219 | -63.0 | 0.600 | 0.868 | 25.5 | 0.583 | 21.4 | -103.7 | 2625.6 |
  | **500K_s400** | **0.1145** | **-19.2** | **0.307** | **0.934** | **17.0** | **0.737** | **14.4** | **-25.4** | **1243.1** |

  (Source: `outputs/arctic_domain/models/val_metrics_{50K,500K}_s400.csv`. `train_500K_s400.pkl`:
  452,517/500,000 windows achieved, 136 grids, 50,476 pixels.)
- **More data helped substantially on every single metric, not just the easy ones.** Best val
  loss nearly halved (0.2219 → 0.1145). GPP NSE reaches 0.934 (up from an already-good 0.868) —
  genuinely strong skill. RECO NSE jumps from 0.583 to 0.737.
- **ALD and VEGC — the chronically weak accumulated-pool targets — improved the most in relative
  terms**, even though they're still net-negative: ALD NSE goes from -63.0 to -19.2 (roughly
  3.3x less bad), VEGC NSE from -103.7 to -25.4 (roughly 4.1x less bad). RMSE for both also
  roughly halved. This is the first result all session where ALD/VEGC's negative NSE looks like
  it's converging toward zero with scale rather than being a structural ceiling — consistent
  with the standing hypothesis (`AR-21c64242` onward) that these targets are data-hungry
  (yearly-accumulated, no autoregressive input) rather than fundamentally unmodelable, though
  they're not fixed yet.
- Training took ~18 minutes (58 epochs to early stopping, ~16.4s/epoch vs 50K's ~1.7s/epoch —
  roughly the expected 10x-data slowdown per epoch, offset by needing fewer epochs to converge:
  58 vs 50K_s400's 86).
- Preprocessing took ~30 minutes end-to-end (pass 1 instant from cache, pass 2 re-fetch of ~150
  grids with zero failures) — faster than the 50K sweep's pass 2 despite fetching more data per
  grid, likely because val/test's ~48 additional grids didn't need re-fetching this time.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- Every metric improved with 10x more data and no sign of saturating yet — 2M (the next
  planned scale point) is a natural next step to see whether the gains continue or start to
  plateau, especially for ALD/VEGC.
- ALD/VEGC are still net-negative despite the big relative improvement — worth watching whether
  2M closes the gap further or whether a structural fix (e.g. autoregressive input, a dedicated
  loss term) becomes necessary regardless of data volume.
- Docs rewrite (`arctic_description.md`, `arctic_description_data_handling.md`) for the
  grid-level split mechanism is still deferred, per the original plan's sequencing — due once
  the scale-up story is settled.

---

## AR-500Ktesteval-0711 — arctic_domain — 2026-07-11
**MLflow run_id:** `364cd7351b5a427da9fc2ce56c0a82c9`
**Config delta:** No model/data change — this run closes a gap left by `AR-500Kstride400-0710`:
the winning 500K/`stride=400` config only ever had its **training-time** val metrics saved
(`val_metrics_500K_s400.csv`, aggregate). Nobody had run `04_evaluate.py` against the frozen
`test.pkl` for it, so there was no per-pixel test-set `metrics_test.csv` and no test plots —
a real problem if the Arctic preprocessed pkls are deleted later to free disk space for
multi-domain work, since that evaluation would then be irreproducible.

Also ships two small additions to `04_evaluate.py` itself (commit `68fe014`, done ahead of this
run): renamed its output from `metrics.csv` to **`metrics_test.csv`** (unambiguous at a glance
vs. the training-time `val_metrics_*.csv`), and added `prediction_sample.parquet` — full monthly
observed-vs-predicted time series (all 4 targets, both SSPs) for a **50-pixel deterministic
sample** of the test set (seeded from `preprocessing.random_seed`, drawn from the sorted set of
unique test pixels). Unlike `metrics_test.csv`'s aggregated per-pixel/target/period error
metrics, this keeps raw values so a handful of specific pixels' time series can still be plotted
after `test.pkl` is deleted — and since `test.pkl` is now frozen (guard added in
`AR-gridsplit4005000710`), the same 50 pixels will reproduce identically in any future run,
including a comparison against a future multi-domain model.

### What happened
- Ran `04_evaluate.py --train-size 500000 --label 500K_s400` on `vm-sandeep` against the
  existing checkpoint + frozen `test.pkl` (no retraining needed). Completed cleanly:
  **3,868 metric rows across 327 test pixels**, plus the new 50-pixel prediction sample
  (164,688 rows, ~4.6MB as parquet — trivial size, well under the "few MB" estimate).
- Test-set median NSE / RMSE per target (excluding `obs_degenerate` rows):

  | target | median NSE | median RMSE | n |
  |---|---|---|---|
  | ALD  | -76.32 | 0.395 | 961 |
  | GPP  | 0.903  | 20.09 | 946 |
  | RECO | 0.610  | 16.34 | 946 |
  | VEGC | -18.13 | 2040.8 | 946 |

  Compared to `AR-500Kstride400-0710`'s val-time numbers (ALD -19.2, GPP 0.934, RECO 0.737,
  VEGC -25.4): GPP and RECO are close and slightly lower on test (expected — val and test are
  different held-out grid sets, both genuinely unseen); **VEGC is notably less bad on test
  (-18.1 vs -25.4 val)**; ALD is somewhat worse on test (-76.3 vs -19.2 val). All differences are
  within the range expected from val and test being different (if similarly-sized) held-out
  populations, not a sign of a val/test inconsistency — both sets are frozen, spatially
  independent, whole-grid samples under the same split mechanism.
- `pyarrow` was an undeclared transitive dependency (used by Rangeland's `predictions.parquet`
  since earlier, and now by this run's `prediction_sample.parquet`) — added explicitly to
  `requirements.txt` (commit `68fe014`) so it isn't silently missing on a fresh VM setup.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- Arctic's saved-results gap is now closed: `metrics_test.csv` + full test plots +
  `prediction_sample.parquet` all exist locally (`outputs/arctic_domain/evaluation/500K_s400/`)
  and are safe to keep even after the preprocessed pkls are eventually deleted.
- Only the 500K/`stride=400` config got this treatment — the other 8 points from the stride
  sweep (50K, strides 50-500) still only have their training-time val metrics saved. This was a
  deliberate scope decision (only the winning/production config needed the full test-set
  artifact), not an oversight — revisit only if a past sweep point needs re-inspection later.

---

## Entry Template (copy when logging a new run)

```
## AR-<run_id[:8]> — arctic_domain — YYYY-MM-DD
**MLflow run_id:** `<full-uuid>`
**Config delta:** <what changed vs. prior run, or "initial run">

### What happened
<!-- Claude drafts: 2-3 bullets of observed outcomes derived from MLflow metrics -->
-
-

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
-
```
