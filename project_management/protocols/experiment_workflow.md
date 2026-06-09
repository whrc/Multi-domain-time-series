# Experiment Workflow Protocol

<!-- Human-maintained. Claude reads only. -->

## Purpose

Steps to follow after `04_evaluate.py` completes successfully. MLflow already
has metrics logged at this point (Stage 2 prerequisite). This protocol
coordinates the three outputs: MLflow (metrics), `key_findings_log.md`
(interpretation), and `generate_report.py` (HTML).

---

## Steps

### 1 — Verify outputs

Confirm `outputs/<domain>/evaluation/metrics.csv` is non-empty.
This is a sanity check only — MLflow is SSOT for all numeric data.

### 2 — Retrieve run_id

```
run_id = outputs/<domain>/models/best_model.run_id   (sidecar written by 02_train.py)
```

If the file is missing: raise `FileNotFoundError` with a clear message.
Do not proceed without a valid run_id.

### 3 — Confirm MLflow run status

```python
import mlflow
client = mlflow.tracking.MlflowClient()
status = client.get_run(run_id).info.status   # must be "FINISHED"
```

If status is not FINISHED, log a warning and note this in the findings entry.

### 4 — Check entry criteria

Log a `key_findings_log.md` entry **only if** at least one criterion is met:

1. First evaluation for this domain ever
2. Config change produced NSE delta > 0.05 vs. the prior run on the same domain
3. A failure or unexpected behaviour occurred
4. A design decision was made as a result of the outcomes

If no criterion is met, skip steps 5 and go straight to step 6.

### 5 — Append entry to `key_findings_log.md`

Query MLflow for per-variable median metrics:

```python
runs = mlflow.search_runs(
    experiment_names=["<domain>"],
    filter_string=f"tags.exp_id = 'AR-{run_id[:8]}'",
)
```

Draft "What happened" bullets from the retrieved metrics (observed outcomes,
not copied numbers verbatim). Write the entry using the template in
`key_findings_log.md`. Leave "Interpretation & Decisions" with the
`NEEDS HUMAN REVIEW` marker — never remove it.

### 6 — Update `current_project_status.md`

- Move CURRENT block verbatim to PAST (with today's date as heading)
- Write new CURRENT block reflecting that evaluation is done
- Update NEXT with the 2–3 most concrete next actions
- If domain stage changed, update the DOMAINS table in-place

### 7 — Regenerate HTML report

```
python project_management/generate_report.py
```

Confirm `project_management/report.html` was written without errors.

---

## SSOT reminder

MLflow owns metric numbers. `key_findings_log.md` owns interpretation.
`current_project_status.md` owns project narrative.
`generate_report.py` reads all three — nothing else duplicates data.
