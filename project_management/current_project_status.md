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

**Date:** 2026-07-02
**Working on:** Arctic preprocessing — size-labeled, geographically representative datasets for local-then-VM workflow (branch `fix/arctic-preprocess-oom-and-perf`)
**Status:** In Progress — code complete, pushed, dev-verified; the actual 50K production data generation is queued but not yet run

- Extended the branch's earlier OOM fix (streaming two-pass design, subprocess-isolated GCS fetch) with a full sizing redesign, commit `c8960ee` (pushed) + a follow-up cache-correctness fix (uncommitted as of this entry — commit before resuming):
  - **Fixed a real bug:** `targets_met()` early-stop treated `train_size=None` (production's actual setting) as trivially satisfied, so a real production run would have stopped scanning after ~1 grid instead of the full circumpolar set. Early-stop is now removed entirely — every grid is always visited, which is also required for representativeness (see next point).
  - **Representativeness fix:** at production stride (1), one pixel yields ~3,290 windows, so a 50K-window cap was previously satisfied by ~16 of ~263 grids. New `preprocessing.capped_stride` (24, reused from dev) makes each pixel contribute far fewer windows (~138) whenever a split is size-capped, forcing round-robin subsampling to draw from many more grids for the same window budget.
  - **Output naming + sidecars:** train pkls are now size-labeled (`train_50K.pkl`, `train_full.pkl`, ...) and every split gets a co-located `{name}.meta.json` sidecar (seed, stride, seq_len, grids_hash, size target/actual, grids/pixels covered). `02_train.py` reads stride/seq_len from the sidecar instead of assuming config, and now takes `--train-size` to pick which variant to load.
  - **Cache-validity fix (found live, during this session, before the real run):** the existing on-disk `val.pkl`/`test.pkl` (from an old 1-grid dev run) had a sidecar that would have matched a full 263-grid run's expected cache key (seed/stride/seq_len/size_target alone can't distinguish "50K from 1 grid" from "50K from 263 grids") — would have silently kept non-representative val/test forever. Fixed by adding `grids_hash` (CRC32 of the sorted grid list) to the cache-validity comparison and sidecar. Also made a `--grids`-scoped debug run never produce a trusted/cacheable val/test.
  - Added concurrent per-grid fetch (`--max-workers`, `ThreadPoolExecutor` over the existing isolated-subprocess-per-grid fetch) and a GCS-access preflight check.
  - **Verified locally against the live bucket:** dev-mode regression produced a byte-identical `train_full.pkl` vs. the pre-change run; reported train/val window counts matched sidecar `actual_window_count` exactly; `--train-size` CLI plumbing verified end-to-end (`01_preprocess.py` → `02_train.py`).
  - **Moderate-scale concurrency validation (this session):** `--max-workers 4` across 25 real grids (mix of small and very large — up to ~9,800 land pixels each) — no permanent hangs; one grid hit the existing 180s timeout under load and the existing retry logic recovered it on the next attempt automatically. `max_workers=4` is the validated choice for the real run; higher values were not tested.
  - Confirmed the real bucket has exactly **263 grids** (matches the user's own estimate).
  - `arctic_description.md` updated throughout (Sizing strategy note, step 9/11, Outputs table, new "Local vs VM Preprocessing" section with the ADC prerequisite and manual `scp` copy command).

### NEXT

1. **50K production preprocessing is RUNNING** — launched 2026-07-02 21:45 local time: `nohup caffeinate -i -s .venv/bin/python domains/arctic_domain/01_preprocess.py --train-size 50000 --max-workers 4 > outputs/arctic_domain/preprocess_50k_run.log 2>&1 &` (detached with `nohup`+`disown`, wrapped in `caffeinate -i -s` to block idle/AC sleep — laptop lid must stay open, since caffeinate cannot override lid-close sleep). Confirmed alive and fetching grid 1/263 at launch. **This process is detached from any Claude Code session, so no automatic completion notification will fire — check status manually** (or ask Claude to check at the start of the next session): `tail -f outputs/arctic_domain/preprocess_50k_run.log`, `ps aux | grep 01_preprocess`, or check whether `outputs/arctic_domain/preprocessed/train_50K.pkl` + `.meta.json` exist yet. Estimated 3-5 hours total (pass 1 scans all 263 grids, pass 2 re-fetches nearly the same set).
2. Once done, verify: `train_50K.pkl` + `.meta.json` (expect `num_grids_covered` close to 263, `actual_window_count` close to 50000), same for `val.pkl`/`test.pkl` (also 50K cap — the old non-representative 1-grid versions were deleted this session), and `scaler.pkl`.
3. Copy `train_50K.pkl`, `val.pkl`, `test.pkl` (+ their `.meta.json` sidecars) and `scaler.pkl` to the VM (see `arctic_description.md` "Local vs VM Preprocessing" for the `scp` command), then run the training pipeline there.
4. Human reviews the implementation diff on branch `fix/arctic-preprocess-oom-and-perf` (or open a PR) before merging to `main`.

### PAST

<!-- Append completed milestones here, newest first. Never delete entries. -->

#### 2026-06-18 — Stage-1 pipeline implementation for all three domains
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
