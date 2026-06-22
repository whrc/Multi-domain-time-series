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
| arctic_domain | Training | Yes | Pipeline 01–04 implemented on shared core; dev-verified end-to-end on CPU (1 grid, GCS NetCDF); production run pending on GCP A100 |
| amazon_domain | Training | Yes | Pipeline 01–04 implemented on shared core; dev-verified end-to-end on CPU (GCS CSV); production run pending on GCP A100 |
| rangeland_domain | Training | Yes | Pipeline 01–04 implemented on shared core; dev-verified end-to-end on CPU (local CSVs); production run pending on GCP A100 |
| multi_domain | Not Started | No | Begins after the three single-domain models are complete |

---

## Section 2 — Diary

### CURRENT

**Date:** 2026-06-18
**Working on:** Stage-1 pipeline implementation for all three domains (branch `feature/domain-pipelines`)
**Status:** In Progress — code complete + dev-verified; awaiting review and production run

- Built a multi-domain-ready shared core: `shared/io.py` (GCS NetCDF/CSV), `shared/dataset.py` (`WindowedDataset` + `records_to_segments`), `shared/training.py` (`masked_mse_loss`, `run_lr_finder`, `train_model`), `shared/inference.py` (`predict_last_position`), `shared/evaluate.py` (`predict_and_inverse`, `per_unit_metrics`), `shared/runner.py` (subprocess orchestration)
- Implemented `01`–`04` + `run_<domain>.py` for **rangeland**, **amazon**, **arctic**; each numbered script is a thin wrapper over the shared core (consolidates the specs' per-domain Dataset/loop so the future multi-domain model reuses it unchanged)
- LR finder wired into `02_train.py` (auto-runs when `optimized_lr` is null); `torch-lr-finder` installed in `.venv`
- **Dev-verified end-to-end on CPU** for all three: Rangeland (local), Amazon (GCS CSV, 98 stations), Arctic (GCS NetCDF, 1 grid H1_V10, 992 pixel-records). Outputs (pkl, scaler, checkpoint, predictions, `metrics.csv`, figures) all produced with correct schemas
- Set **justified production hyperparameters** (A100 40GB, no grid search): model size scaled to data volume, batch for throughput/generalisation, LR from finder — Rangeland 64/3/4/256 b64, Amazon 128/3/8/256 b256, Arctic 256/6/8/1024 b1024
- Minor spec-driven additions documented in the description files: Rangeland `segment_starts`; Arctic `lat/lon/ny/nx` + feature-NaN imputation + eval recomputes from checkpoint
- **MLflow wired now (brought forward from Stage 2):** new `shared/tracking.py`; `02`/`03`/`04` log params, per-epoch + per-target losses, prediction-complete flag, eval median metrics, and artifacts (checkpoint, `metrics.csv`, figures) to a local `mlruns/` store, gated by `mlflow.enabled` in each config. `mlruns/` gitignored. **Note for human:** `protocols/log_experiments.md` (human-owned) still labels MLflow "Stage 2 — not yet wired"; those labels are now stale and should be refreshed.
- Spatial maps for Amazon/Rangeland investigated and **skipped** — neither dataset carries coordinates (Rangeland CSVs have none; Amazon coords only in bucket GeoPackages). Arctic keeps its gridded NSE maps.

### NEXT

1. Human reviews the implementation diff on branch `feature/domain-pipelines`
2. Run production mode on GCP A100 per domain (`mode: production` → `python run_<domain>.py`); provide GCP project + bucket-access confirmation
3. Refine production hyperparameters if dev/prod window counts or training dynamics warrant
4. Human: refresh the "Stage 2 — not yet wired" MLflow labels in `protocols/log_experiments.md` (MLflow is now wired)

### PAST

<!-- Append completed milestones here, newest first. Never delete entries. -->

#### 2026-06-17 — Methodology-audit remediation — specs, configs, shared code, governance
- Adversarial methodology audit written to `methodology_audit_20260617.md`
- Corrected framing repo-wide: the model is a causal **same-step emulator** (inputs ≤ t → target at t), not forecasting / next-token; evaluation is **spatial generalization** to unseen units
- Climatology features now computed per-unit from each unit's own data for all splits; z-score scaler stays train-only
- Unified LR config (`initial_lr` + `optimized_lr`) and the `metrics.csv` schema across domains; set dev (smoke) + production (A100) hyperparameters
- Implemented `shared/metrics.py` (robust degenerate-case handling) and `shared/plots.py` (domain-agnostic + `plot_timeseries`); hardened `transformer.py` and `config.py`
- Generalized `protocols/log_experiments.md` and `generate_report.py` to all three domains

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
