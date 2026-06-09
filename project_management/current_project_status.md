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
| arctic_domain | Complete | No | Pipeline complete; H1_V10 evaluated under SSP1-2.6 and SSP5-8.5 |
| amazon_domain | Not Started | Yes | Next domain; config YAML and description not yet written |
| multi_domain | Not Started | No | Begins after both single-domain models are complete |

---

## Section 2 — Diary

### CURRENT

**Date:** 2026-06-09
**Working on:** project_management — infrastructure setup
**Status:** In Progress

- Created project management folder with all management MD files and generate_report.py
- MLflow integration into pipeline scripts deferred to Stage 2

### NEXT

1. Human reviews project_management PR and merges; Claude cleans up branch on confirmation
2. Stage 2: add `mlflow>=2.0` to requirements.txt, update `.gitignore` (`mlruns/`, `report.html`, `best_model.run_id`), add `mlflow_tracking_uri` to `config/arctic_domain.yaml`
3. Stage 2: instrument `02_train.py`, `03_predict.py`, `04_evaluate.py` with MLflow tracking per `protocols/log_experiments.md`

### PAST

<!-- Append completed milestones here, newest first. Never delete entries. -->

#### 2026-06-09 — Arctic domain pipeline complete
- Dedicated transformer/LSTM model for Arctic domain fully implemented
- Pipeline: 00_eda → 01_preprocess → 02_train → 03_predict → 04_evaluate
- Evaluation outputs: metrics.csv, boxplots, spatial NSE maps for H1_V10 grid
- Scenarios: SSP1-2.6 and SSP5-8.5

---

## Section 3 — Open Questions

| # | Question | Raised | Status |
| --- | --- | --- | --- |
| 1 | Historical `_tr` targets: SSP1-2.6 only — intentional? | 2026-06-09 | Open |
| 2 | Projected ALD/VEGC time labels (1901–1976 vs 2025–2100) — confirm correct period mapping | 2026-06-09 | Open |
| 3 | Amazon domain: confirm input variables, target variables, GCS bucket path, scenarios | 2026-06-09 | Open |
