# Current Project Status

<!-- Claude Instructions ─────────────────────────────────────────────────────
THIS FILE HAS THREE INDEPENDENT SECTIONS with different update cadences.
When updating, touch ONLY the section relevant to the update.

DOMAINS TABLE (Section 1) — low-churn, in-place update only:
  - Update the `stage` column in-place when a domain advances.
  - Never append rows. Never reformat the table.

DIARY (Section 2) — append-only:
  - At session end: copy the CURRENT block verbatim into PAST (with today's
    date as the heading), then overwrite the CURRENT block with new content.
  - Do NOT touch the DOMAINS table or Open Questions when updating the diary.

OPEN QUESTIONS (Section 3) — update status in-place; add rows as needed.
──────────────────────────────────────────────────────────────────────────── -->

---

## Section 1 — Domain Stages

Stage enum: `Not Started → EDA → Preprocessing → Training → Evaluation → Complete`

| domain | stage | active | notes |
| --- | --- | --- | --- |
| arctic_domain | EDA | Yes | EDA done; description spec + config + shared utilities ready; pipeline 01–04 not yet implemented |
| amazon_domain | EDA | Yes | EDA done; description spec + config (incl. rename map) ready; pipeline 01–04 not yet implemented |
| rangeland_domain | EDA | Yes | EDA done; description spec + config ready; pipeline 01–04 not yet implemented |
| multi_domain | Not Started | No | Begins after the three single-domain models are complete |

---

## Section 2 — Diary

### CURRENT

**Date:** 2026-06-17
**Working on:** Methodology-audit remediation — specs, configs, shared code, governance
**Status:** In Progress

- Adversarial methodology audit written to `methodology_audit_20260617.md`
- Corrected framing repo-wide: the model is a causal **same-step emulator** (inputs ≤ t → target at t), not forecasting / next-token; evaluation is **spatial generalization** to unseen units
- Climatology features now computed per-unit from each unit's own data for all splits; z-score scaler stays train-only
- Unified LR config (`initial_lr` + `optimized_lr`) and the `metrics.csv` schema across domains; set dev (smoke) + production (A100) hyperparameters
- Implemented `shared/metrics.py` (robust degenerate-case handling) and `shared/plots.py` (domain-agnostic + `plot_timeseries`); hardened `transformer.py` and `config.py`
- Generalized `protocols/log_experiments.md` and `generate_report.py` to all three domains

### NEXT

1. Human reviews the remediation diff on branch `review/methodology-audit-20260617`
2. Confirm the DOMAINS-table `active` flags and the proposed production hyperparameters
3. Begin Stage-1 pipeline implementation (`01_preprocess` → `04_evaluate`) against the corrected specs
4. Stage 2: instrument `02_train.py`, `03_predict.py`, `04_evaluate.py` with MLflow per `protocols/log_experiments.md`

### PAST

<!-- Append completed milestones here, newest first. Never delete entries. -->

#### 2026-06-09 — project_management infrastructure setup
- Created project management folder with all management MD files and generate_report.py
- MLflow integration into pipeline scripts deferred to Stage 2

#### 2026-06-09 — Arctic domain: EDA + pipeline spec complete
- EDA complete; full pipeline specification written in `arctic_description.md`
- Pipeline scripts `01–04` are scaffolded placeholders — NOT yet implemented
- *(Corrected 2026-06-17 during methodology-audit remediation: the original entry claimed the Arctic pipeline was "fully implemented" with a transformer/LSTM model, which was inaccurate — only EDA + the written spec existed; `01–04` are empty and no LSTM was ever implemented.)*

---

## Section 3 — Open Questions

| # | Question | Raised | Status |
| --- | --- | --- | --- |
| 1 | Historical `_tr` targets: SSP1-2.6 only — intentional? | 2026-06-09 | Open |
| 2 | Projected ALD/VEGC time labels (1901–1976 vs 2025–2100) — confirm correct period mapping | 2026-06-09 | Open |
| 3 | Amazon domain: confirm input variables, target variables, GCS bucket path, scenarios | 2026-06-09 | Resolved — defined in `amazon_description.md` + `config/amazon_domain.yaml` |
| 4 | Model framing: same-step emulation (inputs ≤ t → target at t), not forecasting? | 2026-06-17 | Resolved — confirmed same-step; docs/specs corrected |
| 5 | Climatology features for val/test units — own data or train-global mean? | 2026-06-17 | Resolved — per-unit from each unit's own data, all splits |
| 6 | DOMAINS-table `active` flags + proposed production hyperparameters | 2026-06-17 | Resolved — all three domains active (Goal-1 single-domain models per CLAUDE.md); production hyperparameters intentionally remain placeholders, to be tuned after EDA establishes data volume |
| 7 | Temporal evaluation — spatial holdout only, or also a future-period holdout? | 2026-06-17 | Resolved — spatial holdout only (held-out units scored over the full time range) |
| 8 | Rangeland window vs multi-year pool accumulation (`seq_len`) | 2026-06-17 | Resolved — `seq_len` is config-controlled; a sufficient value is set for production runs |
