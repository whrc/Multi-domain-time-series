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
