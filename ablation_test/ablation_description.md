# Ablation Study — Why Does Multi-Domain Beat Domain-Specific?

## Current status

- **Active:** three pairwise multi-domain runs — `{Arctic,Amazon}`, `{Arctic,Rangeland}`, `{Amazon,Rangeland}` — flux-only, pretrain + finetune, 5 seeds (matches `run_seed_sweep.py`).
- **Retired:** capacity-matched Amazon/Rangeland individual controls and Rangeland's `--amazon-sized` baseline. Superseded by `hyperparameter_tuning/` (direct sweep for every domain). Outputs kept on disk; not read by `make_ablation_figures.py`.
- `make_ablation_figures.py`'s "Individual" arm always loads current production (`outputs/{amazon,rangeland}_domain/evaluation_seedavg/`), so retunes propagate automatically.

## Motivation

Multi-domain beats individual models, especially for scarce domains — e.g. Amazon discharge NSE 0.356 → 0.760, active_fire_count 0.368 → 0.707 (finetuned); Rangeland GPP/RECO also improve (`project_management/key_findings_log.md`). This study asks *why*.

## Hypotheses under test

1. **Capacity confound** — gain is from the larger trunk (`hidden_dim=256, layers=6, ff=1024, dropout=0.1`) vs. smaller original individual models (Amazon `128/3/512`; Rangeland `64/3/256, dropout=0.3`). **Answered by `hyperparameter_tuning/`:** more capacity didn't help Amazon, did help Rangeland, but the multi-domain gap persists for both.
2. **Anchor-specific transfer** — Arctic (500K windows vs. Amazon ~18K, Rangeland ~2K) drives the gain if {Arctic, X} recovers most of the 3-domain gain.
3. **Generic pooling** — {Amazon, Rangeland} without Arctic still improving supports a general shared-representation mechanism.

Not mutually exclusive.

## Experiment design

| # | Run | Domains | Capacity | Stage | Hypothesis |
|---|---|---|---|---|---|
| 1 | Amazon capacity-matched | Amazon | trunk spec | — | 1 (retired) |
| 2 | Rangeland capacity-matched | Rangeland | trunk spec | — | 1 (retired) |
| 3 | Pairwise | Arctic + Amazon | production trunk | pretrain + finetune | 2 |
| 4 | Pairwise | Arctic + Rangeland | production trunk | pretrain + finetune | 2 |
| 5 | Pairwise | Amazon + Rangeland | production trunk | pretrain + finetune | 3 |

Finetune was added later for consistency with the production sweep (original design was pretrain-only).

### Matched-seed anchor — reused, not rerun

- Single-seed pretrain NSE varies ~0.10–0.13, so compare pairwise arms against the full 3-domain run **at the same seeds**.
- Anchor = existing `outputs/multi_domain/{models,evaluation}/{pretrained,finetuned}_fluxonly_seed{1..5}/` from `run_seed_sweep.py`.
- **Do not rerun** (`02_train.py --flux-only --seed N` without `--domains` writes the same paths and would overwrite published artifacts). `run_ablation.py` excludes it.

### Baselines reused (not rerun)

- Individual production: `outputs/{amazon,rangeland,arctic}_domain/` (Arctic already at trunk capacity).
- Full 3-domain pretrained+finetuned, seeds 1–5 (anchor) and 5-seed average (seed-noise sanity check).

## What's held constant

- Data splits: no `01_preprocess.py` rerun; Amazon/Rangeland splits fixed by `preprocessing.random_seed=42` and fractions; multi-domain uses the pkls pinned in `config/multi_domain.yaml`.
- Same held-out test units, `per_unit_metrics`/`predict_and_inverse`, metric definitions, `seq_len`, scalers.

## Known limitations

- Residual seed noise: treat deltas < 0.05 NSE as inconclusive, not null.
- Capacity-matched controls (retired) changed architecture only.

## Output locations

- Pairwise (`<subset>` ∈ `amazon-arctic`, `arctic-rangeland`, `amazon-rangeland`): `outputs/multi_domain/{models,evaluation}/{pretrained,finetuned}_fluxonly_dom-<subset>_seed{1..5}/` — never collides with the anchor paths.
- Retired capacity-matched: `outputs/{amazon,rangeland}_domain/models/best_model_capmatched.pt`, `outputs/{amazon,rangeland}_domain/evaluation_capmatched/`.
- Retired `--amazon-sized`: `outputs/rangeland_domain/models/best_model_fluxonly_amazonsized.pt`, `outputs/rangeland_domain/evaluation_fluxonly_seed{1,avg}_amazonsized/`.
