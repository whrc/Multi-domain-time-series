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
| arctic_domain | Evaluation | No | Production run complete: grid-level latitude-stratified split, staggered windowing, 500K windows @ `stride=400` settled as current config (see `key_findings_log.md` AR-500Kstride400-0710, AR-500Ktesteval-0711). Flux-only variant (GPP/RECO) also available (AR-c3aaf88b). Branch `feat/arctic-grid-level-split` merged to `main` via PR #14. |
| amazon_domain | Evaluation | No | First production run complete 2026-07-11 (98 stations, 59/20/19 split) — see `key_findings_log.md` AZ-184e096d. Non-negative output + log1p transform (AZ-71935d7c) and drainage-area normalization for discharge (AZ-5e809245) brought all 3 targets to positive test NSE (discharge 0.351); the same normalization made burned_area worse and was reverted (AZ-2ffbfcd3). Branch `feat/amazon-rangeland-production-run` merged to `main` via PR #15. |
| rangeland_domain | Evaluation | No | First production run complete 2026-07-11 (59 sites, 35/11/8 split, PFT-stratified) — see `key_findings_log.md` RG-83fdf771. Fluxes (GPP/RECO) and AGB strong (NSE 0.85+); BGB and desert-scrub PFT weak, likely small-test-set variance. Flux-only mode added (RG-5f0c3603) — recommended checkpoint for GPP/RECO/Rm/Rg-only downstream use. Branch `feat/amazon-rangeland-production-run` merged to `main` via PR #15. |
| multi_domain | Not Started | No | Begins now that all three dedicated models (Goal 1) are merged to `main` — spec alignment + flux-only support work starting |

---

## Section 2 — Diary

### CURRENT

**Date:** 2026-07-11
**Working on:** Multi-domain (Goal 2) spec alignment — reconciling `domains/multi_domain/multi_description.md` with what the now-merged Arctic/Amazon/Rangeland pipelines actually produced, plus adding a flux-only multi-domain variant (Arctic GPP/RECO + Rangeland GPP/RECO/Rm/Rg, mirroring each domain's own flux-only mode).
**Status:** In Progress — both Goal-1 branches (`feat/arctic-grid-level-split` PR #14, `feat/amazon-rangeland-production-run` PR #15) are now merged to `main`, unblocking Goal 2. Fixed a conflict-resolution defect from PR #15's merge: the Domains table below had picked up duplicate rows for all three domains (both branches' updates concatenated instead of merged in-place), and the diary CURRENT block had an orphaned duplicate paragraph — both cleaned up in this update.

### NEXT

1. Update `multi_description.md` for Arctic's real artifact naming (`train_500K_s400.pkl`, not a plain `train.pkl`) and add the flux-only variant design.
2. Add `--flux-only` support to `domains/multi_domain/02_train.py`/`03_predict.py`/`04_evaluate.py`.
3. Dev-mode smoke test on `vm-cpu-sandeep` for both the full-target and flux-only variants; production run on `vm-sandeep` only after that's confirmed clean.

### PAST

<!-- Append completed milestones here, newest first. Never delete entries. -->

#### 2026-07-10 — Arctic domain wrap-up — grid-level split redesign, stride/scale sweeps, docs rewrite
**Working on:** Arctic domain wrap-up — grid-level split redesign, stride/scale sweeps, docs rewrite (branch `feat/arctic-grid-level-split`)
**Status:** Complete and merged (PR #14, 2026-07-11) — user reviewed results and explicitly chose to stop scaling and wrap up. Docs (`arctic_description.md`, `arctic_description_data_handling.md`) rewritten to match.

**How we got here (brief — full detail in `key_findings_log.md`, tags in parens):**
- Fixed a `-9999` fill-value contamination bug in Arctic target loading + training instability (`AR-sspfix0708`).
- Found staggered windowing (a small deterministic per-pixel offset before windowing, so different pixels' fixed-stride windows land on different calendar months) improves results at both 50K and 500K under the old split (`AR-stagger0709`, `AR-500Kstagger0709`).
- **Redesigned the train/val/test split from per-grid-pixel to whole-grid, latitude-stratified** — the old design let a held-out pixel sit immediately next to a training pixel in the same tile (weak generalization test); the new design holds out entire grid tiles, spatially independent of train. Staggering made unconditional (no more flag/comparison). New `tests/arctic_domain/test_grid_split.py`.
- 8 grids (`FLAKY_GRIDS_20260710`) excluded after exhausting retries across ~5 real cycles in one day — separate from the 3 permanently-broken `KNOWN_BROKEN_GRIDS`, flagged as possibly-transient and worth re-testing later.
- **9-point stride sweep at 50K** (50 through 500) under the new split — `stride=400` won outright: best val loss and best NSE+RMSE on all 4 targets (ALD, GPP, RECO, VEGC) simultaneously (`AR-gridsplitsweep0710`, `AR-gridsplit4005000710`).
- **Found and fixed a real bug** while extending the sweep: `01_preprocess.py` was silently rebuilding `val.pkl`/`test.pkl` with a different pixel population whenever its sidecar didn't exact-match the current run — undetected until a stride-400/500 run's val/test ended up meaningfully different from the original sweep's, forcing a retrain of all 7 already-trained checkpoints just to get a valid comparison again. Fixed (commit `3d19d6e`): now fails loudly with a field-level diff instead of silently rebuilding; `--force-recompute` required to intentionally rebuild. Val/test are effectively frozen going forward.
- **Scaled the `stride=400` winner to 500K** — beat the 50K baseline on every single metric (best val loss nearly halved 0.222→0.114; GPP NSE reached 0.934; the chronically-weak ALD/VEGC targets improved 3-4x in relative terms, though still net-negative) (`AR-500Kstride400-0710`).
- **Considered 2M, declined for now** — `vm-cpu-sandeep`'s 44GB free disk wasn't enough for the ~52GB+ a 2M pkl plus cache growth would need without either deleting more or resizing the disk (one-way, small recurring cost). User chose to settle at 500K rather than spend on either.
- Cleaned up all pre-grid-split-era output files (local + both VMs) to avoid confusion/save storage, keeping the full 50-500 grid-split sweep results.
- Rewrote both description docs to match current reality: whole-grid split mechanism, staggering, the frozen-val/test guarantee and its enforcement, current grid-exclusion lists, chosen production config (500K/`stride=400`), and corrected disk/memory guidance from this session's real numbers.
- Added a real `--flux-only` training mode (GPP+RECO), evaluated on the frozen test set (`AR-c3aaf88b`) — no meaningful accuracy change vs. the full-target model, but gives a clean dedicated checkpoint for flux-only downstream use.

#### 2026-07-11 — Amazon + Rangeland production runs, flux-only mode, discharge normalization
**Working on:** Amazon + Rangeland first production runs, Rangeland flux-only mode, Amazon target-transform fixes (branch `feat/amazon-rangeland-production-run`)
**Status:** Complete and merged (PR #15, 2026-07-11).

- Amazon + Rangeland flipped to production mode; first production runs completed for both (`AZ-184e096d`, `RG-83fdf771`).
- Added Rangeland `--flux-only` training mode (GPP/RECO/Rm/Rg, dropping the 6 pool targets) — `RG-5f0c3603`.
- Non-negative output (softplus) + log1p transform for Amazon discharge/fire/burn targets — fixed all 3 targets from negative to positive test NSE (`AZ-71935d7c`).
- Normalized Amazon discharge by drainage_area (specific discharge, per Kratzert et al. CAMELS LSTM) — discharge NSE 0.014 → 0.351 (`AZ-5e809245`). Tried the same normalization for burned_area — made it much worse, reverted (`AZ-2ffbfcd3`).
- Renamed `metrics.csv` → `metrics_test.csv`, fixed stale pipeline-status docs.

#### 2026-07-03 — Arctic 50K production preprocessing, first attempt (superseded)
**Working on:** Arctic preprocessing — size-labeled, geographically representative datasets for local-then-VM workflow (branch `fix/arctic-preprocess-oom-and-perf`)
**Status:** Completed, then substantially superseded — this was the *first* Arctic production preprocessing effort, under the original per-grid-pixel split. That split mechanism was later replaced entirely (see 2026-07-10 CURRENT/above) by a whole-grid, latitude-stratified split for a stronger spatial-generalization guarantee. Kept here as the historical record of the original local-preprocessing workflow and its failure modes.

- **Attempt 1 crashed** ~19.5 min in / 46 grids into pass 1, zero output saved (nothing is checkpointed to disk until both passes finish — a real limitation to keep in mind for future long runs). Root cause confirmed by reproduction: grid `H13_V7` deterministically fails all 3 fetch attempts (three separate 180s timeouts, identical failure on both attempts) — genuinely a bad/slow grid, not a fluke — and the resulting `RuntimeError` propagated uncaught out of `fetch_grids_concurrent`, killing the entire multi-hour job over one grid. Fixed in commit `375b439`: `fetch_grids_concurrent` now catches that `RuntimeError` per-grid, logs it, and continues — confirmed working live in attempt 2 (`H13_V7` failed the same way, was skipped, and the run continued well past that point).
- Also fixed in the same commit: `scaler.pkl` now gets the same `--grids`-debug-run protection `val.pkl`/`test.pkl` already had (a debug-scoped run must never overwrite the canonical scaler).
- **Attempt 2 crashed too** — ~23 min in, again with no Python traceback for the actual death. `pmset -g log` showed `caffeinate ClientDied` at the same moment, both times, which redirected suspicion away from a code bug. **Tried disabling the sandbox on the launch (attempt 3, `dangerouslyDisableSandbox`)** — died anyway, at ~17 min, same signature. Three consistent data points (19.5/23/17 min) independent of sandbox setting means this repo's tool environment kills even `nohup`+`disown`ed background processes after roughly 15-25 minutes — not a code bug, not the specific grid content, and not fixable from inside a single Bash tool call.
- **Real fix (commit `db53ab5`):** since nothing was checkpointed until both passes fully finished (~90+ min each), a restart every ~20 min could never complete. Added a per-grid disk cache (`{preprocessed_dir}/.grid_cache/{grid}.pkl`, plus `{grid}.failed` markers for grids that exhaust retries, e.g. `H13_V7`/`H14_V15`) so every restart resumes near-instantly through already-fetched grids instead of starting over — verified locally (2nd run of a 2-grid test dropped from 38s to 4.6s). This also incidentally fixes the known pass-1/pass-2 double-fetch inefficiency.
- **Attempt 4** (with the cache) ran ~19 min, reached 51/263 grids cached (2 failed: `H13_V7`, `H14_V15`) before dying the same way — expected, now cheap to resume from.
- **Long idle gap overnight:** after attempt 4 died (~2026-07-02 23:18), the next check-in only happened the next morning (2026-07-03 ~09:30) — a ~10 hour gap with no progress. Most likely the laptop actually went to sleep for an extended stretch despite `caffeinate -i -s` (which cannot override lid-close sleep), or the session was otherwise dormant. Nothing was lost (cache is on disk), but real time was wasted.
- **Attempt 5 relaunched** ~2026-07-03 09:31, resuming from the 51 already-cached grids; eventually completed.
- Extended the branch's earlier OOM fix (streaming two-pass design, subprocess-isolated GCS fetch) with a full sizing redesign, commit `c8960ee`:
  - **Fixed a real bug:** `targets_met()` early-stop treated `train_size=None` (production's actual setting) as trivially satisfied, so a real production run would have stopped scanning after ~1 grid instead of the full circumpolar set. Early-stop removed entirely.
  - **Representativeness fix:** at production stride (1), one pixel yields ~3,290 windows, so a 50K-window cap was previously satisfied by ~16 of ~263 grids. New `preprocessing.capped_stride` (24) makes each pixel contribute far fewer windows (~138), forcing round-robin subsampling to draw from many more grids for the same window budget — this mechanism survived the later grid-level-split redesign essentially unchanged.
  - **Output naming + sidecars:** train pkls size-labeled (`train_50K.pkl`, ...); every split gets a co-located `{name}.meta.json` sidecar — this mechanism also survived into the current design, extended with `split_unit`/`split_lat_bins`.
  - **Cache-validity fix:** added `grids_hash` to the val/test cache-validity comparison, so a debug-scoped run's val/test could never be mistaken for the real thing — this concept is what later evolved (2026-07-10) into the full fail-loud sidecar-mismatch guard.
  - Added concurrent per-grid fetch (`--max-workers`) and a GCS-access preflight check.
  - Confirmed the real bucket has exactly **263 grids** (later reduced to 252 in active use after excluding `KNOWN_BROKEN_GRIDS` + `FLAKY_GRIDS_20260710`).

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
