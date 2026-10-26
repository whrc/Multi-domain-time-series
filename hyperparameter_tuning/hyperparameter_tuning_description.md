# Individual-Model Hyperparameter Tuning

## Motivation

Production architectures were originally hand-sized (no grid search). Each individual baseline should be best-effort before comparing against multi-domain.

## Design

- Sweep `hidden_dim` (half / current / double), single seed (seed 1); other dims = each domain's production values; dropout fixed at **0.15** for all cells.
- Target sets as production: Arctic/Rangeland `--flux-only`; Amazon full (no flux-only mode).
- Select by **minimum validation loss** (no test leakage). Promotion to a 5-seed production rerun is a separate decision.
- Driven via `--model-size` → `model_{size}` config blocks; `hyperparameter_tuning/run_hyperparameter_tuning.py`.

| Domain | layers | heads | ffn | hidden_dim candidates |
|---|---|---|---|---|
| Arctic | 6 | 8 | 1024 | 128 / 256 / 512 |
| Amazon | 3 | 4 | 512 | 64 / 128 / 256, extended to 32, 16 (flat) |
| Rangeland | 3 | 4 | 256 | 32 / 64 / 128, extended to 256, 512 (still improving) |

## Resolution

Authoritative over `hyperparameter_tuning_winners.yaml` (whose `amazon: xxsmall` is a mechanical argmin over a flat line).

| Domain | Production (hidden/layers/heads/dropout/ffn) | Decision |
|---|---|---|
| Arctic | 256 / 6 / 8 / 0.15 / 1024 | Winner (medium) = existing production; unchanged |
| Amazon | 64 / 3 / 4 / 0.10 / 256 | All 4 dims flat; smallest/fastest promoted for **efficiency only** (was 128/3/4/0.2/512) |
| Rangeland | 256 / 3 / 4 / 0.15 / 256 | xlarge (256) ~40% lower val loss than original 64/dropout 0.3; xxlarge (512) worse (0.0356 vs 0.0249) → genuine peak; promoted |

- Both promotions were rerun at 5 seeds; they change that domain's "Individual" numbers in Figures 4/6/7/8, ablation figures, and the KGE decomposition. Multi-domain trunk unaffected.
- Amazon 5-seed median NSE, old → new: discharge 0.356 → 0.371, active_fire_count 0.368 → 0.310, burned_area 0.047 → 0.008 (both near no-skill). Individual-vs-multi-domain gap unchanged.
- The retired capacity-matched ablation used Rangeland's original 64/dropout 0.3 config; its numbers are a historical record, not comparable to current production, not rerun.

### Amazon: hidden_dim extension

`model_xsmall`=32, `model_xxsmall`=16 (dropout 0.15, ffn 512). Best val loss: 16→0.513, 32→0.517, 64→0.517, 128→0.518, 256→0.517 — <1% over a 16× range, no trend.

### Amazon: feedforward_dim sweep

hidden 64; `model_ffn_narrow`=128, `model_ffn_std`=256, 512 reused from `model_small`. Val: 0.517 / 0.513 / 0.517 — flat. Suggests a data-noise floor, not capacity.

### Amazon: num_layers sweep

hidden 64, ffn 256; `model_layers2/4/6` (6 = trunk depth), 3 reused from `model_ffn_std`. Val: 2→0.516, 3→0.513, 4→0.520, 6→0.519 — ~1.4% spread, flat.

### Amazon: dropout sweep

hidden 64, layers 3, ffn 256; `model_dropout{10,20,30}`, 0.15 reused. Val: 0.10→0.511, 0.15→0.513, 0.20→0.518, 0.30→0.516 — flat. Winner (0.511) is the promoted config.

## Reuse note

Arctic and Amazon seed-1 base sweeps were copied from the parked `feat/rangeland-obs-multidomain` branch (same grid, unchanged config/data); only Rangeland was trained fresh. See `project_management/key_findings_log.md`.

## Output locations

- Per-cell runs: `outputs/{arctic,amazon,rangeland}_domain/.../..._{size}_seed1/` for `small/medium/large`; Amazon also `xsmall, xxsmall, ffn_narrow, ffn_std, layers2, layers4, layers6, dropout10, dropout20, dropout30`; Rangeland also `xlarge, xxlarge`.
- `hyperparameter_tuning/hyperparameter_tuning_winners.yaml`
- `hyperparameter_tuning/figures/hyperparameter_tuning_results.{png,csv}` (hidden_dim, all domains)
- `hyperparameter_tuning/figures/amazon_architecture_search.png` (all 4 Amazon dims; `plot_amazon_architecture_search.py`)
