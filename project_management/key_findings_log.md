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
