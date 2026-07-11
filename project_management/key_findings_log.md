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

## AZ-184e096d — amazon_domain — 2026-07-11
**MLflow run_id:** `184e096dc9e84376ab7418ce9c96957d`
**Config delta:** First production run, ever — `01_preprocess.py` had never been run past dev
mode (`outputs/amazon_domain/` was completely empty before this session). Ran a dev-mode
ground test first (all 4 stages, tiny model, sparse `stride=24`) to confirm the pipeline still
works end-to-end before committing to production — passed cleanly. Flipped `mode: production`
(`hidden_dim=128, num_layers=3, num_heads=4, feedforward_dim=512, batch_size=256,
num_epochs=100, stride=1`) and re-ran `01_preprocess.py` for the real data (98 stations ->
59/20/19 train/val/test split), then `02_train.py` -> `03_predict.py` -> `04_evaluate.py`.

### What happened
- Training ran the full 100 epochs (no early stop) — val loss plateaued around 0.58, train
  loss around 0.47, no divergence.
- Test-set median NSE / RMSE per target (19 test stations):

  | target | median NSE | median RMSE |
  |---|---|---|
  | discharge | -0.931 | 687.9 |
  | active_fire_count | -0.118 | 195.4 |
  | burned_area | -1.908 | 119.8 |

  All three targets have negative median test NSE — the model is currently doing worse than
  predicting each station's own mean. This is a genuinely weak first result, not a pipeline
  bug (the pipeline itself ran cleanly at every stage, dev mode included, and metrics/plots
  all look structurally sane — see `outputs/amazon_domain/evaluation/`).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- This is a first, untuned production run — no LR sweep beyond the automatic finder, no
  architecture iteration. Worth investigating before concluding the model can't learn this
  task: check the per-target loss curves (`outputs/amazon_domain/evaluation/loss_curves.png`)
  for whether val loss is still trending down at epoch 100 (i.e. undertrained) vs. plateaued
  early, and whether `active_fire_count`/`burned_area` (count/area targets, likely
  zero-inflated and skewed) need a different loss or transform than plain MSE.
- No stride/size sweep was run (per the plan for this session — the dataset is small enough
  that one wasn't judged necessary), but if results stay weak after investigating the above,
  a small sweep over `capped_stride`-equivalent settings or model size may still be worth
  trying, mirroring what worked for Arctic's `stride`=400 finding.

---

## RG-83fdf771 — rangeland_domain — 2026-07-11
**MLflow run_id:** `83fdf7715edc44d4b052c41b553e6d80`
**Config delta:** First production run, ever — Rangeland had only been dev-verified locally
(2026-06-30), never in production mode or on a VM. Ran a dev-mode ground test first (all 4
stages, tiny model, sparse `stride=6`) to confirm the pipeline still works end-to-end —
passed cleanly. Flipped `mode: production` (`hidden_dim=64, num_layers=3, num_heads=4,
dropout=0.3, feedforward_dim=256, batch_size=64, num_epochs=100, stride=1`) and re-ran
`01_preprocess.py` for the real data (59 sites -> 35/11/8 train/val/test split, PFT-stratified
across 4 groups), then `02_train.py` -> `03_predict.py` -> `04_evaluate.py`.

### What happened
- Training converged normally: early stopping fired at epoch 63 (best val=0.3865 @ epoch 51),
  no divergence.
- Test-set median NSE / RMSE, by target (8 test sites, 10 targets):

  | target | median NSE | median RMSE |
  |---|---|---|
  | GPP_predicted  | 0.853  | 0.514 |
  | RECO_predicted | 0.855  | 0.405 |
  | AGB_predicted  | 0.882  | 14.38 |
  | BGB_predicted  | -0.808 | 18.76 |

  and by PFT (median NSE across all 10 targets):

  | PFT | median NSE | n |
  |---|---|---|
  | sagebrush    | 0.928  | 10 |
  | grass-tree   | 0.711  | 10 |
  | grass        | 0.200  | 50 |
  | desert-scrub | -6.212 | 10 |

  **Fluxes (GPP, RECO) and most pools (AGB) score strongly** (NSE 0.85+), consistent with
  Arctic's own flux-vs-pool pattern (fluxes are easier, driven by concurrent climate). BGB
  (belowground biomass) is a clear exception — negative NSE despite AGB being strong.
  `desert-scrub`'s deeply negative median NSE lines up with the small-per-PFT-test-set caveat
  already flagged in `methodology_audit_20260617.md` — only 1 desert-scrub site's worth of
  data (10 rows = 1 site x 10 targets) drives that whole PFT's aggregate, so it's high-variance
  by construction, not necessarily a sign the model handles that PFT badly.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- BGB's negative NSE alongside AGB's strong NSE is worth a closer look — check
  `outputs/rangeland_domain/evaluation/timeseries_*.png` for whether BGB predictions track
  the right shape but wrong scale (a PBIAS-driven issue, fixable) vs. genuinely wrong dynamics.
- `desert-scrub`'s single-test-site result shouldn't be over-interpreted on its own; the
  existing methodology-audit caveat about small per-PFT test sets applies directly here.
- No stride/size sweep was run, matching the plan for this session (data is tiny; a sweep
  wasn't judged necessary) — revisit only if the BGB/desert-scrub investigation above points
  at a data-volume rather than a modeling issue.

---

## RG-5f0c3603 — rangeland_domain — 2026-07-11
**MLflow run_id:** `5f0c3603c6a34f258bc9e5976bb7d7e2`
**Config delta:** New `--flux-only` mode (`02_train.py`/`03_predict.py`/`04_evaluate.py`) —
trains on GPP/RECO/Rm/Rg only (drops the 6 pool targets AGB/BGB/AGL/BGL/POC/HOC), reusing the
existing preprocessed pkls (fluxes are already the first 4 of the 10 trailing target columns,
so no re-preprocessing needed). Output checkpoint/eval/predictions all get a `_fluxonly` suffix
(`outputs/rangeland_domain/models/best_model_fluxonly.pt`,
`outputs/rangeland_domain/evaluation_fluxonly/`) so they never collide with the full-target run.
Same production hyperparameters otherwise. Also fixed the full-target run's boxplots in this
same session (`04_evaluate.py`, no model change) — split into flux/pool subsets, each with a
by-PFT and an all-PFTs-pooled variant, since pool targets' orders-of-magnitude-larger RMSE/NSE
were squashing the flux boxes into invisible flat lines on the old single shared-axis plot.

### What happened
- Training converged normally: early stopping fired at epoch 80 (best val=0.0629 @ epoch 68).
- Test-set median NSE / RMSE, flux-only vs. the existing full-target run (`RG-83fdf771`,
  fluxes subset only):

  | target | full-target NSE / RMSE | flux-only NSE / RMSE |
  |---|---|---|
  | GPP  | 0.8535 / 0.514  | 0.860 / 0.4275 |
  | RECO | 0.8545 / 0.405  | 0.862 / 0.3875 |
  | Rg   | 0.8190 / 0.1315 | 0.823 / 0.1315 |
  | Rm   | 0.8375 / 0.1810 | 0.846 / 0.1545 |

  All four fluxes are slightly better (or equal) in the flux-only model — consistent with the
  hypothesis that dropping the noisy, near-zero-variance pool targets frees up model capacity
  and gradient signal for the fluxes, though the effect is modest here since the fluxes were
  already the strong targets in the full-target run.
- The new flux/pool-split boxplots (`outputs/rangeland_domain/evaluation/
  metrics_boxplot_test_fluxes_by_pft.png` and `..._pooled.png`) confirm the fix worked:
  GPP/RECO/Rm/Rg's per-PFT boxes are now clearly legible across all 4 metrics, where they were
  previously flattened to invisible lines by HOC/POC's exploded values.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- The dedicated flux-only checkpoint (`best_model_fluxonly.pt`) is now the recommended model
  to use for GPP/RECO/Rm/Rg-only downstream consumers, avoiding any confusion with the
  full-target model's 10-column output.
- Pool targets (especially BGB, flagged in `RG-83fdf771`) still need their own investigation —
  out of scope for this flux-focused round.

---

## AZ-71935d7c — amazon_domain — 2026-07-11
**MLflow run_id:** `71935d7cfb224caca3fc3909a6a99e7e`
**Config delta:** Two changes vs. the first production run (`AZ-184e096d`), both targeting the
across-the-board negative test NSE seen there: (1) `model.nonneg_output: true` in
`config/amazon_domain.yaml` adds a config-gated softplus activation on the output head
(`shared/transformer.py`) — discharge/active_fire_count/burned_area are all physically ≥0, but
the raw linear output could (and did) predict negative values; (2) all 3 targets are now
log1p-transformed before the scaler fit (`01_preprocess.py`), with `expm1` added everywhere the
inverse z-score is reconstructed (`03_predict.py`, `04_evaluate.py`) — EDA had shown severe
right-skew (discharge mean 2443.5 vs median 435.6) driven by `drainage_area` spanning ~3 orders
of magnitude across stations, letting a few large/volatile stations dominate the global z-score.
Required re-running `01_preprocess.py` (cheap — GCS CSV read) before retraining.

### What happened
- Training converged with early stopping at epoch 35 (best val=0.5948 @ epoch 23) — faster
  convergence than the prior run's full 100 epochs with no early stop.
- Test-set median NSE / RMSE per target, vs. `AZ-184e096d`:

  | target | old NSE / RMSE | new NSE / RMSE |
  |---|---|---|
  | discharge          | -0.931 / 687.9 | 0.014 / 460.3 |
  | active_fire_count   | -0.118 / 195.4 | 0.521 / 93.5  |
  | burned_area         | -1.908 / 119.8 | 0.260 / 57.1  |

  All three targets moved from negative median NSE (worse than predicting each station's own
  mean) to positive, with RMSE cut by 33-52%. `active_fire_count` and `burned_area` — the two
  most skewed/zero-inflated targets — improved the most, consistent with log1p being the
  dominant fix; `discharge`'s NSE is barely positive (0.014), a real improvement in direction
  but still weak in absolute terms.
- Spot-checked predictions: all values ≥0 across all 3 targets (softplus constraint holding),
  and a synthetic log1p/z-score/expm1 round-trip recovers the original value exactly (including
  NaN passthrough for discharge's ~6% missing rate) — no correctness issue in the transform.
- `active_fire_count_pred` is rounded to the nearest integer in the saved parquet for
  readability (cosmetic only, per plan decision not to change the loss/output to a count-specific
  distribution).

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- `discharge`'s NSE (0.014) is still weak despite the direction improving — per-station
  normalization (deferred in this round, see the plan) is the natural next lever if further
  improvement is wanted, since drainage_area's ~3-order-of-magnitude spread across stations is
  a separate issue from within-station skew that log1p alone doesn't fully address.
- No LR sweep or architecture iteration was done this round either — same caveat as
  `AZ-184e096d`.

---

## AZ-5e809245 — amazon_domain — 2026-07-11
**MLflow run_id:** `5e809245f290467b8e40b01f2d38dc36`
**Config delta:** Follow-up to `AZ-71935d7c`, prompted by discharge's median NSE (0.014)
still being far below the ~0.5 typically achievable for monthly discharge with precip/temp
inputs. Diagnosed via per-station breakdown: stations split cleanly into two groups by
RMSE — low-RMSE (small/low-flow) stations had catastrophic NSE (down to -38, PBIAS up to
+366%, i.e. massive over-prediction), while high-RMSE (large-discharge) stations already
scored NSE 0.5-0.8 (Spearman corr(RMSE, NSE) across stations = 0.70 — systematic, not noise).
Root cause: `log1p` fixed within-station skew but not cross-station scale heterogeneity —
`drainage_area` spans ~3 orders of magnitude, and discharge was still z-scored with one global
mean/std pooled across all 59 train stations, so the model had to infer each station's
absolute scale implicitly from a single static feature; small stations regressed toward the
global mean, reading as large over-prediction.

**Fix** (`01_preprocess.py`): divide discharge by `drainage_area` before log1p (specific
discharge / unit runoff — standard in DL rainfall-runoff literature, e.g. Kratzert et al.'s
CAMELS LSTM). Unlike a per-station learned mean/std, this generalizes to held-out test
stations since `drainage_area` is a known static covariate for every station.
`drainage_area` is saved raw on each preprocessed record so `03_predict.py`/`04_evaluate.py`
can multiply back to report discharge in physical units. `active_fire_count`/`burned_area`
preprocessing is untouched.

### What happened
- Re-ran `01_preprocess.py` → `02_train.py` → `03_predict.py` → `04_evaluate.py` on
  `vm-sandeep`, same production hyperparameters. Training converged faster this time (early
  stopped at epoch 24, best val=0.5259 @ epoch 12, vs. `AZ-71935d7c`'s epoch 35/val=0.5948) —
  the joint validation loss (all 3 targets combined) is lower overall, consistent with
  discharge now being a much easier target to fit.
- Discharge, before -> after:

  | metric | AZ-71935d7c | AZ-5e809245 |
  |---|---|---|
  | median NSE | 0.014 | **0.351** |
  | median RMSE | 460.3 | 276.1 |
  | stations with NSE > 0 | 10/19 | **17/19** |
  | Spearman corr(RMSE, NSE) | 0.70 | **0.10** |

  The small-station bias is essentially resolved — only one station (15200000) remains
  strongly negative (NSE -9.3), down from 9 negative-NSE stations before. Predictions
  remain non-negative everywhere (softplus constraint holding).
- **`active_fire_count`/`burned_area` both declined** despite their preprocessing being
  unchanged: active_fire_count median NSE 0.521 -> 0.298, burned_area 0.260 -> 0.014. Root
  cause is almost certainly multi-task interference, not a bug — all 3 targets share one
  transformer backbone and one joint MSE loss, `shared/training.py` has no fixed random seed
  (confirmed: fresh weight init + minibatch order every run), and discharge's now-much-easier
  loss landscape likely pulled shared-capacity/gradient allocation away from fire/burn during
  training. The *aggregate* val loss improved (0.526 vs 0.595), so this is a real reallocation
  of fit quality across targets within one shared model, not a regression from a broken run.

### Interpretation & Decisions
<!-- NEEDS HUMAN REVIEW: fill in WHY these results occurred and what to try next -->
-

### Follow-up
- The fire/burn regression is worth a controlled re-run (same discharge transform, different
  random seed) to separate "multi-task interference from the discharge fix" from ordinary
  run-to-run stochastic variance — `shared/training.py` has no seeding at all currently, so
  this can't be distinguished from the logs alone.
- If the fire/burn regression persists across seeds, consider per-target loss weighting or
  separate output heads (bigger architectural change, out of scope for this round) so
  discharge's improved loss dynamics don't come at fire/burn's expense.
- Station 15200000 is now the single worst discharge outlier (NSE -9.3) — worth a closer look
  (drainage_area value sanity check, or a genuinely unusual flow regime) if discharge accuracy
  needs to improve further.

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
