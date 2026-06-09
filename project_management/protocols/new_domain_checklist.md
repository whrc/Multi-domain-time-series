# New Domain Checklist

<!-- Human-maintained. Claude reads only. -->

## Purpose

Project-management steps for adding a new domain (e.g., Amazon, multi-domain).
This file covers only management infrastructure — not science decisions
(data schema, model architecture, EDA findings). Those belong in the
domain's own `<domain>_description.md`.

---

## Checklist

### Step 1 — Register the domain

- [ ] Add a row to `current_project_status.md` DOMAINS table:
  `| <domain> | Not Started | Yes | <one-line note> |`
- [ ] Set the previously-active domain's `active` column to `No`

### Step 2 — Config YAML

- [ ] Confirm `config/<domain>.yaml` exists (human creates this)
- [ ] Confirm it contains a `mlflow_tracking_uri` key pointing to the
  repo-root `mlruns/` directory (Stage 2 requirement)

### Step 3 — MLflow experiment

- [ ] Create the MLflow experiment once:
  ```python
  import mlflow
  mlflow.set_tracking_uri("<repo_root>/mlruns")
  mlflow.create_experiment("<domain>")
  ```
  Run this from the repo root, not from inside the domain folder.

### Step 4 — Output directories

- [ ] Confirm `outputs/<domain>/preprocessed/`, `models/`, `predictions/`,
  `evaluation/` exist (or create them with `.gitkeep` placeholders)
- [ ] Confirm `.gitignore` covers `outputs/<domain>/` content

### Step 5 — CLAUDE.md

No update needed — `CLAUDE.md` already points to `current_project_status.md`
for domain stage. The DOMAINS table update in Step 1 is sufficient.

### Step 6 — Update project status

- [ ] Update `current_project_status.md` CURRENT block to reflect the new
  domain is now the active focus
- [ ] Update NEXT with the first concrete actions for the new domain

---

## Notes

- Science decisions (variables, metrics, data layout) belong in the domain
  description file — do not mix them into this checklist.
- Do not build the shared model (Goal 2/3 in CLAUDE.md) while a single-domain
  pipeline is still in progress.
