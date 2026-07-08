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
