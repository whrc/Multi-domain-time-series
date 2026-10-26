# Current Project Status

<!-- Update rules:
- Section 1: edit `stage`/`active`/`notes` in place; never add rows or reformat.
- Section 2: at session end, move CURRENT into PAST (dated heading, newest first), then rewrite CURRENT/NEXT. Don't touch Sections 1/3 when doing so.
- Section 3: update status in place; add rows as needed.
Run IDs/tags refer to `key_findings_log.md`. -->

## Section 1 — Domain Stages

Stages: `Not Started → EDA → Preprocessing → Training → Evaluation → Complete`

| domain | stage | active | notes |
| --- | --- | --- | --- |
| arctic_domain | Evaluation | No | Whole-grid latitude-stratified split, staggered windows, 500K @ `stride=400` (AR-500Kstride400-0710, AR-500Ktesteval-0711). Flux-only (GPP/RECO, AR-c3aaf88b) has the 5-seed sweep (`AR-seedsweep0714`); full-target is single-seed. |
| amazon_domain | Evaluation | No | 98 stations, 59/20/19 split (AZ-184e096d). Softplus + log1p (AZ-71935d7c) and drainage-area-normalized discharge (AZ-5e809245) give positive NSE on all 3 targets; same normalization for burned_area reverted (AZ-2ffbfcd3). HP sweep flat on all 4 dims; smallest point (hidden=64, ffn=256, dropout=0.10) promoted for efficiency and 5-seed sweep rerun (`AZ-retune0813`). Insensitivity attributed to small data volume. |
| rangeland_domain | Evaluation | No | 59 sites, 35/11/8 PFT-stratified split (RG-83fdf771). Flux-only (RG-5f0c3603) retuned to `hidden_dim=256, dropout=0.15` after real HP gain (`HP-sweep0812`) and 5-seed sweep rerun (`RG-retune0812`). Individual vs multi-domain: very similar, slight edge to multi-domain (2026-09-21). Full-target still on original architecture (64, dropout 0.3). |
| multi_domain | Evaluation | No | Production run `MD-prod0712` (full-target + flux-only). Flux-only 5-seed sweep `MD-seedsweep0714`; Amazon corrected for units bug (`MD-unitsbugfix0716`): NSE individual→finetuned discharge 0.356→0.760, active_fire_count 0.368→0.707, burned_area 0.047→0.521. Full-target not seed-swept and still has the units bug. `finetune_epochs`=50. `compare_models.py` not implemented. |

## Section 2 — Diary

### CURRENT

**2026-09-21** — Repo-wide docstring/comment cleanup (commit `ca64d17`, no logic changes); removed `paper/` (manuscript now in Google Docs); untracked `RangeSTAR_data/`; deleted superseded audit files; resolved Open Questions #1/#2.
- Rangeland framing: "retuned individual beats multi-domain" (`RG-retune0812`) was a false alarm; multi-domain keeps a slight edge.
- Amazon framing: architecture insensitivity (`AZ-retune0813`) is a data-volume effect; a small architecture is appropriate.

### NEXT

1. Wire `shared/tracking.py` (MLflow) into multi-domain — the only pipeline without it.
2. LR-finder divergence (`AR-gridsplit4005000710`) has only a safety clamp, no root-cause fix.
3. Implement `compare_models.py` (Individual vs. joint vs. fine-tuned, paired significance tests); `metric_decomposition/` covers only the "why".
4. Decide whether Arctic/Rangeland/multi-domain full-target variants need the 5-seed sweep for the paper.

### PAST

<!-- Newest first. Gaps 2026-07-15–08-06 and 2026-08-14–09-20 are unrecorded; see `key_findings_log.md` and git history. -->

- **2026-08-13** — Amazon HP sweep extended (ffn/layers/dropout, all flat); smallest point promoted, 5-seed rerun (`AZ-retune0813`); Figures 4–8, ablation and KGE figures regenerated. Merged `feat/individual-tuning-and-nse-decomposition` to `main`; deleted it, `ablation_tests`, `feat/rangeland-obs-multidomain`, `docs/claim-validation-plan`. Docs audit.
- **2026-08-12** — Per-domain HP sweep (`HP-sweep0812`); KGE decomposition (r/alpha/beta) in `shared/metrics.py`; Rangeland retune + 5-seed rerun (`RG-retune0812`). Merged ablation study (`AB-capacitypairwise0806`). Split figure scripts into one `make_figureN.py` each + `_common.py`.
- **2026-07-15** — Figures 4/6/7 use seed-averaged metrics; Figure 5 split into 5a/5b with one line per seed (commit `a7f4293`, PRs #21/#22).
- **2026-07-14** — Seed control (`--seed`, seeded DataLoader, seed-suffixed outputs), LR-finder clamp, per-epoch `history.csv`, `shared/seed_aggregation.py`, `run_seed_sweep.py`. 5-seed sweeps: `AR-`/`AZ-`/`RG-`/`MD-seedsweep0714`. Multi-domain `finetune_epochs` 100→50.
- **2026-07-13** — Figures 1, 2a/2b, 3–7 and Amazon station map; scripts moved to `figures/scripts/`. Multi-domain flux-only single-seed rerun (`finetune_epochs` 100) regressed vs `MD-prod0712` (`MD-fluxrerun0713`) — superseded by seed sweep. Arctic 250K/stride-400 flux-only point added (GPP NSE 0.92–0.93). Second LR-finder divergence (LR 0.093) worked around with `optimized_lr=2e-4`.
- **2026-07-12** — Multi-domain production run (`MD-prod0712`, PR #17). Fluxes strong, pool/depth targets negative. Flux-only beat full-target (Arctic GPP 0.947 vs 0.899). Arctic dense NetCDF predictions skipped (disk). A100 ~5/40 GB GPU, ~25/85 GB RAM peak.
- **2026-07-11 (multi-domain)** — Spec aligned with Goal-1 outputs; `--flux-only` added (`flux_only.py`); outputs split into `pretrained[_fluxonly]/` and `finetuned[_fluxonly]/`. Fixed: Arctic windowed at global stride instead of pkl sidecar stride (400); `DomainRoutedModel` wrapper for `.eval()`. Arctic eval exports the same 50-pixel sample as the individual pipeline. Dev smoke test passed (`MD-devsmoke0711`).
- **2026-07-11 (Amazon/Rangeland)** — First production runs (`AZ-184e096d`, `RG-83fdf771`, PR #15); Rangeland flux-only; Amazon softplus+log1p and discharge normalization (NSE 0.014→0.351); `metrics.csv` → `metrics_test.csv`.
- **2026-07-10** — Arctic wrap-up (PR #14): fixed `-9999` fill contamination (`AR-sspfix0708`); staggered windowing (`AR-stagger0709`); whole-grid latitude-stratified split + `tests/arctic_domain/test_grid_split.py`; 8 `FLAKY_GRIDS_20260710` excluded (re-test later) besides 3 `KNOWN_BROKEN_GRIDS`; stride sweep at 50K → `stride=400` (`AR-gridsplitsweep0710`); val/test now frozen, sidecar mismatch fails loudly (`--force-recompute` to rebuild, commit `3d19d6e`); scaled to 500K (val loss 0.222→0.114, `AR-500Kstride400-0710`); 2M declined (disk); flux-only mode (`AR-c3aaf88b`).
- **2026-07-03** — First Arctic production preprocessing (per-pixel split, superseded). Per-grid failures caught and skipped (`375b439`); per-grid disk cache for resumable runs since the tool environment kills background jobs after ~15–25 min (`db53ab5`); removed broken early-stop, added `capped_stride` (24), size-labeled pkls with `.meta.json` sidecars, `--max-workers`, GCS preflight (`c8960ee`). Bucket has 263 grids.
- **2026-06-18** — Shared core (`io`, `dataset`, `training`, `inference`, `evaluate`, `runner`, `tracking`) and `01`–`04` + `run_<domain>.py` for all three domains, dev-verified on CPU. LR finder auto-runs when `optimized_lr` is null. MLflow wired (gated by `mlflow.enabled`). No spatial maps for Amazon/Rangeland (no coordinates).
- **2026-06-17** — Methodology audit: same-step emulator framing, spatial-generalization eval, per-unit climatology, train-only scaler, unified LR config and metrics schema, `shared/metrics.py`/`plots.py`.
- **2026-06-09** — Project-management setup; Arctic EDA and spec (`arctic_description.md`).

## Section 3 — Open Questions

| # | Question | Raised | Status |
| --- | --- | --- | --- |
| 1 | Historical `_tr` targets: SSP1-2.6 only — intentional? | 2026-06-09 | Resolved — intentional (2026-09-21) |
| 2 | Projected ALD/VEGC time labels (1901–1976 vs 2025–2100) — correct mapping? | 2026-06-09 | Resolved — correct (2026-09-21) |
| 3 | Amazon inputs, targets, bucket, scenarios | 2026-06-09 | Resolved — `amazon_description.md` + `config/amazon_domain.yaml` |
| 4 | Same-step emulation, not forecasting? | 2026-06-17 | Resolved — same-step |
| 5 | Climatology features for val/test — own data or train mean? | 2026-06-17 | Resolved — per-unit, all splits |
| 6 | `active` flags + production hyperparameters | 2026-06-17 | Resolved — hyperparameters tuned after EDA |
| 7 | Temporal holdout as well as spatial? | 2026-06-17 | Resolved — spatial only, full time range |
| 8 | Rangeland `seq_len` vs multi-year pool accumulation | 2026-06-17 | Resolved — config-controlled, set for production |
