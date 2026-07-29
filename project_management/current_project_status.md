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
| arctic_domain | Evaluation | No | Production run complete: grid-level latitude-stratified split, staggered windowing, 500K windows @ `stride=400` settled as current config (see `key_findings_log.md` AR-500Kstride400-0710, AR-500Ktesteval-0711). Flux-only variant (GPP/RECO) also available (AR-c3aaf88b) and has since completed the final 5-seed publication sweep (`AR-seedsweep0714`); full-target variant remains single-seed. Branch `feat/arctic-grid-level-split` merged to `main` via PR #14. |
| amazon_domain | Evaluation | No | First production run complete 2026-07-11 (98 stations, 59/20/19 split) — see `key_findings_log.md` AZ-184e096d. Non-negative output + log1p transform (AZ-71935d7c) and drainage-area normalization for discharge (AZ-5e809245) brought all 3 targets to positive test NSE (discharge 0.351); the same normalization made burned_area worse and was reverted (AZ-2ffbfcd3). Completed the final 5-seed publication sweep, its only variant (`AZ-seedsweep0714`). Branch `feat/amazon-rangeland-production-run` merged to `main` via PR #15. |
| rangeland_domain | Evaluation | No | First production run complete 2026-07-11 (59 sites, 35/11/8 split, PFT-stratified) — see `key_findings_log.md` RG-83fdf771. Fluxes (GPP/RECO) and AGB strong (NSE 0.85+); BGB and desert-scrub PFT weak, likely small-test-set variance. Flux-only mode added (RG-5f0c3603) and has since completed the final 5-seed publication sweep (`RG-seedsweep0714`); full-target variant remains single-seed. Branch `feat/amazon-rangeland-production-run` merged to `main` via PR #15. |
| multi_domain | Evaluation | No | First production run complete 2026-07-12 (`mode: production`, both full-target and flux-only variants) — see `key_findings_log.md` `MD-prod0712`. Fluxes strong (Arctic GPP 0.90/0.95, Rangeland GPP 0.95/0.98, Amazon 0.65-0.89, full-target/flux-only), pool/depth targets weak (same pattern as individual pipelines). PR #17 merged 2026-07-12. A single-seed flux-only rerun (`finetune_epochs` 50->100) regressed vs. `MD-prod0712` (`MD-fluxrerun0713`) and was superseded, not reconciled: `finetune_epochs` reverted to 50, and the question was resolved by building full seed control instead of chasing single-seed variance. **Final 5-seed flux-only publication sweep complete** (`MD-seedsweep0714`) — cross-domain pretraining benefits the data-scarce domains far more than either domain's own flux-only experiment suggested. Amazon's numbers were corrected 2026-07-16 after finding a units bug (multi-domain eval never undid Amazon's log1p/drainage-area transform — see `key_findings_log.md` `MD-unitsbugfix0716`); corrected finding still holds, smaller margin than first reported: Amazon discharge NSE individual 0.356 -> multi-domain finetuned 0.760, active_fire_count 0.368 -> 0.707, burned_area 0.047 -> 0.521. Arctic/Rangeland unaffected by the bug. Full-target variant not yet through the seed sweep (and still has the same unfixed bug, lower priority). `compare_models.py` still not implemented. |

---

## Section 2 — Diary

### CURRENT

**Date:** 2026-07-15
**Working on:** Rewiring Figures 4/6/7 to the seed-averaged (`seedavg`) metrics instead of single-seed results (branch `flux-only-multiple-seeds-run`), plus redesigning Figure 5 into two figures (5a training loss / 5b validation loss) with every seed plotted as its own line rather than an across-seed average, since seeds early-stop at different epochs.
**Status:** Complete (commit `a7f4293`). Not yet PR'd.

### NEXT

1. Open a PR for branch `flux-only-multiple-seeds-run` (seed control + seedavg rewiring).
2. Wire up `shared/tracking.py` (MLflow) for multi-domain — still the only domain without it.
3. LR-finder divergence (`AR-gridsplit4005000710`, 07-13 Arctic 250K rerun) now has a safety clamp (`ef0e315`) as a mitigation, but the root cause is still not identified.
4. `compare_models.py` (Individual vs. Unified-joint vs. Unified-fine-tuned) still not implemented.
5. Arctic/Rangeland full-target variants and multi-domain's full-target variant have not been through the 5-seed sweep — decide if that's needed for the paper.

### PAST

<!-- Append completed milestones here, newest first. Never delete entries. -->

#### 2026-07-14 — Seed control + final 5-seed publication run
**Working on:** `--seed` CLI plumbing across all four pipelines (torch/numpy/random seeding, reproducible DataLoader shuffling) and seed-suffixed output paths, an LR-finder safety clamp, per-epoch `history.csv` saves, `shared/seed_aggregation.py`, and `run_seed_sweep.py` to orchestrate the full sweep (commit `ef0e315`). Also reverted multi-domain's `finetune_epochs` 100->50.
**Status:** Complete — ran all 5 seeds for Arctic (flux-only), Rangeland (flux-only), Amazon (its only variant), and multi-domain (flux-only pretrain+finetune). Full detail: `key_findings_log.md` `AR-seedsweep0714`/`AZ-seedsweep0714`/`RG-seedsweep0714`/`MD-seedsweep0714`.

#### 2026-07-13 — Figure suite completion (Figures 1, 2, 4, 6, 7) + Amazon station map
**Working on:** Rounding out the manuscript figure set beyond Figures 3-6: Figure 1 (study-site map, all three domains), Figure 2a/2b (hand-built methodology schematics, individual vs. unified two-stage model), Figure 7 (per-site %-change maps, individual vs. fine-tuned multi-domain), an Amazon station split map, plus iterative polish on Figures 4/5/6 (NSE-floor calibration, whisker clipping, box sizing, legend/label fixes).
**Status:** Complete. `figures/*.py` moved to `figures/scripts/` for a consistent layout; `shared/plots.py` gained reusable regional-map/coastline helpers used across Figures 1, 2, 7.

#### 2026-07-13 — Publication figures (Figures 3-6) + multi-domain flux-only rerun
**Working on:** Building the manuscript's 4 main figures (`run_figures_main.py`, branch `feat/publication-figures` off `main`) — Arctic sampling/dataset-size sweep, per-domain RMSE/NSE/PBIAS results, multi-domain training curves, Individual/Pretrained/Fine-tuned comparison. All flux-only.
**Status:** Complete — 6 PNGs (`fig3`, `fig4`, `fig5`, `fig6a/b/c`) generated at 300dpi, colorblind-safe, visually verified. Full detail: `key_findings_log.md` `MD-fluxrerun0713`.

- Figure 5 had no source data anywhere (multi-domain's `02_train.py` only `logger.info`'d per-epoch loss, never persisted it, and MLflow was never wired) — fixed by adding `history.csv` writes per pretrain/finetune stage folder (commit `2bd3d9c`), then rerunning flux-only pretrain+finetune on `vm-sandeep` (prior `MD-prod0712` checkpoints backed up first).
- Along the way, bumped `finetune_epochs` 50->100 (to match `pretrain_epochs`) per user request, mid-pretrain-run (finetune hadn't started yet, so safe).
- The rerun's flux-only numbers came out visibly worse than `MD-prod0712`'s on every target but one (e.g. Arctic GPP NSE 0.815 vs 0.947), despite the larger finetune budget — traced to a weaker pretrain plateau on this seed (smooth convergence, not divergence). Single seed, no seed control yet; flagged for human review rather than silently accepted.
- Added a new Arctic 250K-window/stride=400/flux-only training-set-size data point for Figure 3 panel b (preprocessed on `vm-cpu-sandeep`, trained on `vm-sandeep`) — GPP NSE 0.92-0.93, RECO NSE 0.68-0.69, fits the 50K->500K trend monotonically.
- Hit a second LR-finder divergence during that Arctic run (auto-suggested LR 0.093, ~100-300x too high, caused catastrophic mid-training blowup) — fixed with a temporary `optimized_lr=2e-4` override, reverted after. Second occurrence of this failure mode (first: `AR-gridsplit4005000710`), now flagged as needing a real fix rather than one-off overrides.



#### 2026-07-12 — Multi-domain first production run (both target-set variants)
**Working on:** Production run of the multi-domain pipeline on `vm-sandeep` (GPU) — full-target and `--flux-only` variants, each through pretrain → finetune → predict → evaluate.
**Status:** Complete. See `key_findings_log.md` `MD-prod0712` for full metrics.

- Pretrain (full-target) early-stopped at epoch 26/100 (best mean-val at epoch 6); finetune ran per-domain independently (Arctic used its full 50-epoch budget, Amazon completed, Rangeland early-stopped at epoch 30).
- Fluxes strong across the board: Arctic GPP 0.899, RECO 0.582; Rangeland GPP 0.945, RECO 0.892; Amazon discharge/fire/burn 0.70-0.85 — pool/depth targets (ALD, VEGC, AGL, BGL, POC, HOC) deeply negative, same pattern as every individual-domain pipeline (accumulated targets with no autoregressive input).
- Flux-only variant's fluxes came out notably *stronger* than full-target's (Arctic GPP 0.947, RECO 0.720; Rangeland GPP 0.979, RECO 0.965) — a much bigger effect than either individual domain's own flux-only experiment showed, suggesting the shared-transformer setting benefits more from dropping noisy pool targets than a dedicated single-domain model does.
- Arctic's dense per-grid NetCDF predictions were deliberately skipped (not required for evaluation, disk-risk) — total multi-domain output footprint was 214MB.
- `vm-sandeep` (A100) ran at 96% GPU utilization throughout, well within memory budget (~5GB/40GB GPU, ~25GB/85GB system RAM peak during flux-only's known transient copy).

#### 2026-07-11 — Multi-domain spec alignment + flux-only support + dev-mode smoke test
**Working on:** Multi-domain (Goal 2) spec alignment — reconciling `domains/multi_domain/multi_description.md` with what the now-merged Arctic/Amazon/Rangeland pipelines actually produced, plus adding a flux-only multi-domain variant (Arctic GPP/RECO + Rangeland GPP/RECO/Rm/Rg, mirroring each domain's own flux-only mode) — then validating the whole pipeline end-to-end on `vm-cpu-sandeep`.
**Status:** Complete. Branch `docs/multi-domain-spec-update`, not yet merged to `main`.

- Both Goal-1 branches (`feat/arctic-grid-level-split` PR #14, `feat/amazon-rangeland-production-run` PR #15) merged to `main`, unblocking Goal 2. Fixed a conflict-resolution defect from PR #15's merge along the way: the Domains table had picked up duplicate rows for all three domains, and the diary CURRENT block had an orphaned duplicate paragraph.
- `multi_description.md` rewritten: Arctic's real artifact naming (`train_500K_s400.pkl`, not a plain `train.pkl`), a new "Flux-Only Variant" section, removed the stale production-TBD hedge, fixed the `batch_size` pseudocode/code mismatch, marked `compare_models.py` explicitly deferred/unbuilt. Added `--flux-only` to `02_train.py`/`03_predict.py`/`04_evaluate.py`/`run_multi_domain.py` (new shared `domains/multi_domain/flux_only.py` module). Restructured pretrain/finetune outputs into separate `pretrained[_fluxonly]/{domain}/` and `finetuned[_fluxonly]/{domain}/` subfolders (was flat `stage1_*`/`stage2_*`-prefixed filenames), per the user's request for easy stage × variant comparison later.
- **8-angle code review (high effort) before touching any VM found and fixed a severe bug**: the training loop was windowing Arctic's `train_500K_s400.pkl` at the global `cfg.preprocessing.stride` (1) instead of the pkl's own sidecar stride (400) — would have inflated one epoch to ~200M windows. Fixed by reading Arctic's stride per-pkl from its sidecar. Also fixed a doc/code drift, a predictions-folder collision risk, a hardcoded target list now derived from config, a non-fail-loud fallback for unknown domains, and consolidated duplicated path-construction logic into `flux_only.py`.
- Per the user's request, Arctic's multi-domain evaluation now exports a 50-pixel deterministic prediction sample using the *exact same seed and site selection* as the individual Arctic pipeline (`sample_test_pixels`/`save_prediction_sample` moved into the shared `arctic_domain/_naming.py`) — verified in the dev test to produce byte-identical row counts (3,868 metric rows, 164,688 sample rows) to the individual pipeline's own numbers.
- **Dev-mode smoke test on `vm-cpu-sandeep` found and fixed a second, pre-existing bug** (inherited from the original never-before-executed multi-domain scaffold): every evaluation call site wrapped `MultiDomainModel` in a bare `lambda x: model(x, domain=d)`, but `shared/inference.py::predict_last_position` calls `.eval()` on whatever it receives — a plain function has no such method. This had never been caught because the multi-domain pipeline had never been run end-to-end before this session (Arctic's finetune stage ran all 10 dev epochs successfully and saved its checkpoint before hitting this crash in the post-training plotting step). Fixed with a `DomainRoutedModel(nn.Module)` wrapper in `model.py`.
- Full pipeline (`01_preprocess.py` → `02_train.py` pretrain/finetune → `03_predict.py` → `04_evaluate.py`) verified working end-to-end for both the full-target and `--flux-only` variants on `vm-cpu-sandeep`. Full detail: `key_findings_log.md` `MD-devsmoke0711`.

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
