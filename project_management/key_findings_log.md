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

<!-- No entries yet — MLflow integration is Stage 2.
     First entry will be created after 02_train.py and 04_evaluate.py
     are instrumented with MLflow tracking. -->

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
