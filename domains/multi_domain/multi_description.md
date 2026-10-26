# Multi-Domain: Unified Transformer Across Arctic, Amazon, and Rangeland

## Overview

- **Model:** `MultiDomainModel` (`domains/multi_domain/model.py`) — per-domain input projection → shared causal transformer → per-domain MLP heads. Causal same-step monthly predictions; spatial-generalization evaluation.
- **Goal:** test whether data-rich Arctic improves data-scarce Amazon/Rangeland via shared representations.
- **Stages:** **pretrain** (joint, mixed-step batching) → **finetune** (shared weights frozen, each head tuned alone). Compared variants: Individual (per-domain pipelines), Unified-joint (pretrain), Unified-fine-tuned (finetune). Cross-model comparison (`compare_models.py`) is **not implemented**.
- **Target-set variants:** full-target and flux-only (see "Flux-Only Variant"), each through the full pipeline.
- **Config:** `config/multi_domain.yaml` (`mode: dev | production`). Production trunk = Arctic's architecture (`common_dim=256, num_layers=6, num_heads=8, feedforward_dim=1024`) — settled.
- **Prerequisite:** run all three individual pipelines first; multi-domain reuses their pkl files and scalers (no own preprocessing). All use 60/20/20 splits → identical test sets across variants.

| Step | File | Role |
|---|---|---|
| Pre-flight | `01_preprocess.py` | Verify domain pkl/scaler files; log sizes |
| Training | `02_train.py` | `--stage {pretrain,finetune}`, `--flux-only` |
| Prediction | `03_predict.py` | Per domain × checkpoint × variant |
| Evaluation | `04_evaluate.py` | Both stages, per variant |

**Shared modules:** `shared/transformer.py`, `shared/dataset.py`, `shared/training.py` (`masked_mse_loss` only — `train_model`/`_evaluate` take no domain arg), `shared/inference.py`, `shared/evaluate.py` (`predict_and_inverse`, `per_unit_metrics`), `shared/metrics.py`, `shared/plots.py`. From `domains/arctic_domain/_naming.py`: `select_flux_target_columns`, `select_flux_scaler_stats`, `load_stride_seq_len`, `sample_test_pixels`, `save_prediction_sample`. `shared/tracking.py` (MLflow) is **not wired in** — `mlflow.enabled` is a no-op here.

## Config notes (`config/multi_domain.yaml`)

- `model.seq_len` = 12 — **read `cfg.model.seq_len`, not `cfg.preprocessing.seq_len`** (differs from domain configs).
- `model.common_dim` = shared embedding D; `model.hidden_dim = common_dim`; `model.feedforward_dim` (default 2×`common_dim`); `model.head_hidden_dim`, `model.head_num_layers` (1 or 2).
- `training.batch_size`: per-domain sub-batch (3× per step). Training only — inference uses `shared/inference.py` default (8192).
- `training.steps_per_epoch`: null → `len(arctic_train_loader)`; scarce domains cycle ~2–3×/epoch.
- `training.finetune_epochs`: 50 (100 regressed). `training.finetune_lr`: fallback when LR finder isn't reused. `training.early_stopping_patience`: both stages.
- `paths.arctic.train_label = "500K_s400"` → `train_500K_s400.pkl` (500K windows, grid-level split, stride 400). `paths.{domain}.{preprocessed_dir,scaler}` point at `outputs/{domain}_domain/`.

## Architecture

`MultiDomainModel(cfg, domain_specs)`, `domain_specs = {domain: {"nFeatures", "nTargets"}}` (variant-agnostic; caller passes `nTargets`).

- `projections` (ModuleDict): `nn.Linear(nFeatures_d, common_dim)` per domain.
- `transformer` (one shared `TransformerModel(num_features=common_dim, num_targets=common_dim, cfg)`): output `(batch, seq_len, common_dim)`; its internal `input_proj` is a learned common_dim→common_dim linear.
- `heads` (ModuleDict): `head_num_layers` × `[Linear(·, head_hidden_dim), GELU]` → `Linear(head_hidden_dim, nTargets_d)`.
- `forward(x, domain)`: projection → transformer → head → `(batch, seq_len, nTargets_d)`.
- `DomainRoutedModel(model, domain)`: `nn.Module` wrapper with `forward(x)` so shared inference/eval code (which calls `.eval()` and `model(x)`) works. Use wherever a domain model goes to `predict_last_position`/`predict_and_inverse`.
- Checkpoints: `state_dict`. Finetune checkpoints are complete (frozen base + that domain's head).

## Domain Data Reference (full-target)

| Domain | nFeatures | nTargets | Feature order | Target cols | Train pkl |
|---|---|---|---|---|---|
| Arctic | `data.shape[1] − 4` (nStatic + 5) | 4 | `[static \| CO2 \| climate]` | `-4:` | `train_500K_s400.pkl` |
| Amazon | 14 | 3 | `[dynamic \| clim. means]` | `14:17` | `train.pkl` |
| Rangeland | 22 | 10 | `[dynamic \| static \| PFT \| cyclical \| site means]` | `22:32` | `train.pkl` |

`val.pkl`/`test.pkl` unlabeled for all. Record schemas and splits: each domain's `*_description.md`. Amazon/Rangeland `nFeatures` are constants from their column lists — update if those change.

## Flux-Only Variant

`--flux-only` on 02/03/04 (and `run_multi_domain.py`). No re-preprocessing; columns reduced in memory on records and scalers. Outputs go to sibling `_fluxonly` folders. `DOMAIN_TARGET_NAMES` in `02_train.py` has a flux-only lookup.

| Domain | nTargets full → flux | Flux targets | Mechanism |
|---|---|---|---|
| Arctic | 4 → 2 | GPP, RECO | Reorder (mid-order) via `select_flux_target_columns`/`select_flux_scaler_stats`, `full_order` from `config/arctic_domain.yaml` |
| Amazon | 3 → 3 | unchanged | No effect |
| Rangeland | 10 → 4 | GPP, RECO, Rm, Rg | Truncate to first 4 target cols |

## Multi-Seed Publication Runs

- `--seed N` on 02/03/04: seeds init + shuffle (split fixed); `_seedN` suffix (`checkpoint_path`/`stage_output_dir` in `flux_only.py`). Use the same seed for pretrain and its finetune.
- `shared/seed_aggregation.py::aggregate_seed_metrics` → mean/std across seeds; fails if seeds have mismatched unit/target sets.
- Production: **flux-only × 5 seeds** via `run_seed_sweep.py`. Full-target not seed-swept.

## Step 1 — Pre-flight check

1. Raise `FileNotFoundError` (with path) if missing: `{split}.pkl` for val/test (all domains); `train_{train_label}.pkl` (Arctic) / `train.pkl` (others); each scaler.
2. Log unique Arctic `grid` values in train; warn (don't abort) if only one (likely dev).
3. Per train pkl: record count and approx windows `Σ ⌊(T_seg − seq_len)/stride + 1⌋` (Arctic: one segment per record; stride per-domain as in Step 2).
4. Log summary table (domain, split, records, windows). No output files.

## Step 2 — Training

### Pretrain stage

1. Load train/val pkls and each domain's own scaler (no cross-domain scaler). Apply flux-only reduction if set.
2. `domain_specs`: Arctic `nTargets` 4 (flux 2), Amazon 3, Rangeland 10 (flux 4); `nFeatures` per table above.
3. Per-domain `WindowedDataset` (train/val), `seq_len = cfg.model.seq_len`. **Stride:** Amazon/Rangeland `cfg.preprocessing.stride`; Arctic from each pkl's sidecar (`load_stride_seq_len`) — using stride 1 on `train_500K_s400.pkl` inflates windows ~400×. Arctic records = one `data` array; others = `segments` list.
4. `MultiDomainModel`; AdamW (`initial_lr`/LR finder → `optimized_lr`, `weight_decay`); warmup `warmup_epochs`, then cosine `T_max = pretrain_epochs − warmup_epochs`.
5. **Mixed-step loop:** each step draws `batch_size` from every domain (`itertools.cycle` iterators), `total_loss = mean(masked_mse_loss per domain)`, one backward/step. Arctic supplies ~10–15× more windows per epoch than the others; assess bias empirically.
6. Log per-domain and mean loss per epoch.
7. Validate every `eval_every_n_epochs` with an inline per-domain loop; early stopping on the **unweighted mean** val loss (patience in evaluations). Best → `models_dir/pretrained[_fluxonly]/best.pt`.
8. Post-training val plots (loss curves, scatter per target, metric boxplots) → `evaluation_dir/pretrained[_fluxonly]/`.

### Finetune stage

1. Rebuild model, load `pretrained[_fluxonly]/best.pt`.
2. Freeze `model.transformer` and `model.projections`; only heads train.
3. Per domain, in order Arctic → Amazon → Rangeland: full train split (stride as above, same flux reduction); AdamW over `model.heads[d]` only, `lr = finetune_lr` (or reused pretrain LR), warmup + cosine `T_max = finetune_epochs − warmup_epochs`; masked MSE; per-domain early stopping; best → `models_dir/finetuned[_fluxonly]/{d}_best.pt`.
4. Plots per domain → `evaluation_dir/finetuned[_fluxonly]/{d}/`.

## Step 3 — Prediction

CLI: `--domain {arctic,amazon,rangeland}`, `--checkpoint {pretrained,finetuned}` (default `finetuned`), `--flux-only`.

- **Arctic caution:** writes dense `(time, y, x)` grids per test tile — many GB. Not needed for evaluation (Step 4 recomputes). Run only if NetCDFs are needed; check disk first.

1. Load `pretrained[_fluxonly]/best.pt` or `finetuned[_fluxonly]/{domain}_best.pt`; `model.eval()`.
2. Load `test.pkl` + domain scaler; apply flux-only reduction.
3. Stride-1 `WindowedDataset`; `predict_last_position` via `DomainRoutedModel`; first `seq_len−1` steps NaN.
4. Inverse-transform: Arctic `std[-4:]/mean[-4:]`; Amazon `14:17` then `expm1` × `drainage_area` (`inverse_amazon_log1p` in `flux_only.py`); Rangeland `22:32`. Flux-only uses the reduced scaler stats.
5. Save to `predictions_dir/{stage}[_fluxonly]/{domain}/` (always nested by domain):
   - Arctic: NetCDF `{grid}_{ssp}_{variable}_{split}.nc` (`split` = `tr` historical / `sc` projected), TEM names (`GPP_monthly`, `ALD_yearly`, …), as `arctic_domain/03_predict.py`.
   - Amazon: `amazon_predictions.parquet` — `station_id, year, month, discharge_pred, active_fire_count_pred, burned_area_pred`.
   - Rangeland: `rangeland_predictions.parquet` — `site, date`, `GPP/RECO/Rm/Rg_predicted` (+ pools for full-target), `NEE_predicted = RECO − GPP`.

## Step 4 — Evaluation

CLI: `--flux-only`.

1. Recompute predictions for both stages from checkpoints (`predict_and_inverse` + `DomainRoutedModel` on `test.pkl`; also returns inverted observations). Arctic ALD/VEGC: January positions only (yearly); GPP/RECO and Amazon/Rangeland: all steps.
2. Per-unit RMSE/NSE/KGE/PBIAS: Arctic per pixel × target × SSP × period (`historical` < 2025 ≤ `projected`); Amazon per station; Rangeland per site. Schema `{id cols, target, stage, RMSE, NSE, KGE, PBIAS}`.
3. Save `evaluation_dir/{stage}[_fluxonly]/{domain}/{domain}_metrics.csv` plus scatter and boxplots in the same folder.
4. **Arctic 50-pixel sample:** `sample_test_pixels`/`save_prediction_sample` with seed from **`config/arctic_domain.yaml`** `preprocessing.random_seed` (same sites as the individual Arctic `prediction_sample.parquet`) → `{stage}[_fluxonly]/arctic/prediction_sample.parquet`. Only per-pixel Arctic output here.

## Entry Point (`run_multi_domain.py`)

`--stage {preprocess,pretrain,finetune,predict,evaluate}`, `--domain` (required for predict), `--checkpoint` (predict, default `finetuned`), `--flux-only`. Does **not** forward `--seed` — use `run_seed_sweep.py` for seeded runs.

```
python run_multi_domain.py --stage preprocess
python run_multi_domain.py --stage pretrain [--flux-only]   # inspect plots before finetune
python run_multi_domain.py --stage finetune [--flux-only]
python run_multi_domain.py --stage predict --domain {arctic,amazon,rangeland} --checkpoint {pretrained,finetuned} [--flux-only]
python run_multi_domain.py --stage evaluate [--flux-only]
```

## Outputs (`outputs/multi_domain/`)

Inputs reused from individual pipelines: `outputs/{domain}_domain/preprocessed/*.pkl`, `outputs/{domain}_domain/scaler.pkl`. Each `{stage}` folder below has a `_fluxonly` sibling (and `_seedN` suffix for seeded runs).

| Path | Contents |
|---|---|
| `models/pretrained/best.pt` | Pretrain checkpoint |
| `models/finetuned/{domain}_best.pt` | Finetune checkpoint per domain |
| `predictions/{pretrained,finetuned}/{domain}/` | Per-domain predictions (Step 3 formats) |
| `evaluation/{pretrained,finetuned}/{domain}/` | `{domain}_metrics.csv`, `{domain}_scatter.png`, `{domain}_boxplot.png`; Arctic also `prediction_sample.parquet` |
| `evaluation/pretrained/lr_finder.png` | Pretrain LR finder (Arctic-routed probe; not under a domain) |
