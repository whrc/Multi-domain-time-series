# Ablation Study — Why Does Multi-Domain Beat Domain-Specific?

> **Stale-config note (2026-08-12):** the "Individual production config" row for Rangeland
> below (`hidden_dim=64, dropout=0.3`) reflects `config/rangeland_domain.yaml` *as it was when
> this study ran* (2026-08-06/07) — accurate for the numbers actually produced, but Rangeland's
> production config has since changed to `hidden_dim=256, dropout=0.15` (see
> `hyperparameter_tuning/hyperparameter_tuning_description.md`'s "Resolution" section). This
> study's capacity-matched results were **not** rerun against the new config; treat the
> Rangeland row/numbers here as describing the pre-2026-08-12 architecture, not current
> production.

## Motivation

Production results show the multi-domain shared-transformer model beats domain-specific
baselines, especially for the data-scarce domains: Amazon discharge NSE 0.356 (individual) →
0.764 (multi-domain finetuned), active_fire_count 0.368 → 0.886; Rangeland GPP/RECO similarly
improved (`key_findings_log.md` `AZ-seedsweep0714`, `RG-seedsweep0714`, `MD-seedsweep0714`). We
can currently say multi-domain is better, but not *why*. This ablation study is designed to
isolate the actual cause, so the eventual paper can make a mechanistic claim instead of just
reporting a comparison.

## Hypotheses under test

1. **Capacity confound.** Amazon's and Rangeland's individual production models are meaningfully
   smaller and more regularized than the multi-domain shared trunk they're compared against:

   | Domain | Individual production config | Multi-domain shared trunk |
   |---|---|---|
   | Amazon | `hidden_dim=128, layers=3, ff=512` | `hidden_dim=256, layers=6, ff=1024, dropout=0.1` |
   | Rangeland | `hidden_dim=64, layers=3, ff=256, dropout=0.3` | `hidden_dim=256, layers=6, ff=1024, dropout=0.1` |

   If most of the observed gain disappears once the individual models are given the same
   capacity and dropout as the shared trunk, the "multi-domain" story is largely a capacity/
   regularization artifact rather than genuine cross-domain transfer.

2. **Anchor-domain-specific transfer.** Arctic is by far the largest domain (500K training
   windows vs. Amazon's ~18K and Rangeland's ~2K). If most of a small domain's 3-domain gain is
   already present with just {Arctic, that domain} pretrained together, Arctic's data volume is
   the primary driver — not "any additional domain."

3. **Generic cross-domain pooling.** If {Amazon, Rangeland} pretrained together (no Arctic) still
   shows a meaningful improvement over either domain alone, that argues for a more general
   "shared temporal/seasonal representation" mechanism that doesn't require a large anchor
   domain specifically — pooling any second domain's data helps, not just the biggest one.

These aren't mutually exclusive — the real answer is likely a mix, and part of the point of this
study is to see how the observed 3-domain gain decomposes across them.

## Experiment design

All new runs: **flux-only** target set (matches the flagship `MD-seedsweep0714` finding being
explained), **seed=1**, and evaluated at the **pretrain-stage checkpoint only** — not finetuned.

**Pretrain-only is a deliberate simplification, not an oversight.** `MD-prod0712`'s pretrained-
vs-finetuned NSE gap is small everywhere (Arctic GPP +0.04, RECO +0.10, Amazon ~+0.00–0.04,
Rangeland +0.02–0.03), and `MD-fluxrerun0713` showed finetune quality tracks pretrain quality
rather than doing independent causal work — a weak pretrain checkpoint produced a weak finetune
result across the board, on a larger epoch budget. So the pretrain-stage checkpoint already
captures the large majority of the effect this study is trying to explain.

**Single seed is a deliberate cost/rigor tradeoff.** This is exploratory/explanatory work, not
the publication headline — accepted given the cost of GPU time on `vm-sandeep`.

### The five new runs

| # | Run | Domains trained | Capacity | Tests hypothesis |
|---|---|---|---|---|
| 1 | Amazon capacity-matched | Amazon only | multi-domain trunk spec | 1 |
| 2 | Rangeland capacity-matched | Rangeland only | multi-domain trunk spec | 1 |
| 3 | Pairwise pretrain {Arctic, Amazon} | Arctic + Amazon | multi-domain trunk spec | 2 |
| 4 | Pairwise pretrain {Arctic, Rangeland} | Arctic + Rangeland | multi-domain trunk spec | 2 |
| 5 | Pairwise pretrain {Amazon, Rangeland} | Amazon + Rangeland | multi-domain trunk spec | 3 |

### Matched-seed anchor — reused, not rerun

Past runs show pretrain-stage NSE can swing ~0.10–0.13 between single seeds (`MD-fluxrerun0713`:
Arctic GPP 0.815 vs. 0.947 across two single-seed runs). Comparing the five new seed=1 runs above
against the existing 5-seed average, or an old unseeded run, risks mistaking ordinary seed
variance for a domain-subset or capacity effect — every comparison should instead be made against
the full 3-domain pretrain at the *same* seed=1.

That comparison point already exists: `02_train.py --stage pretrain --flux-only --seed 1` (no
`--domains` override) is exactly the command already run as seed 1 of the completed 5-seed
publication sweep (`run_seed_sweep.py`) — its checkpoint and metrics are already on disk at
`outputs/multi_domain/{models,evaluation}/pretrained_fluxonly_seed1/`. **Do not rerun this** —
`ablation_test/run_ablation.py` deliberately does not include it, since rerunning would silently
overwrite that published production artifact (`checkpoint_path`'s no-`--domains` case reproduces
the exact same path by design, so a naive rerun and the existing production run are
indistinguishable on disk). Use the existing `pretrained_fluxonly_seed1` outputs directly as the
matched-seed anchor when comparing against runs 3–5.

### Existing baselines reused (not rerun)

- Amazon individual production (`outputs/amazon_domain/`)
- Rangeland individual production (`outputs/rangeland_domain/`)
- Arctic individual production (`outputs/arctic_domain/`) — already at the multi-domain trunk's
  capacity (`hidden_dim=256, num_layers=6`), so it needs no capacity-matched control of its own
- Full 3-domain pretrained, seed=1 (`outputs/multi_domain/evaluation/pretrained_fluxonly_seed1/`)
  — the matched-seed anchor, see above
- Full 3-domain pretrained, 5-seed average (`MD-seedsweep0714`) — secondary comparison, for
  sanity-checking how much of any observed effect could be ordinary seed noise

## What's held constant

- **Data splits**: no domain's `01_preprocess.py` is rerun. Amazon/Rangeland train/val/test splits
  are a deterministic function of `preprocessing.random_seed=42` and fixed train/val/test
  fractions, independent of model architecture — reusing the existing `train.pkl`/`val.pkl`/
  `test.pkl` guarantees byte-identical data to the production runs. Multi-domain pairwise/anchor
  runs reuse the exact same per-domain preprocessed files pinned in `config/multi_domain.yaml`.
- **Held-out test set**: every comparison in this study evaluates against the same held-out
  sites/pixels/stations as the existing production baselines.
- **Evaluation methodology**: same `per_unit_metrics`/`predict_and_inverse` pipeline, same metric
  definitions (RMSE, NSE, KGE, PBIAS), no changes to `shared/evaluate.py` or `shared/metrics.py`.
- **seq_len, scaler**: unchanged from production.

## Known limitations

- **Residual seed noise.** Comparing against the single matched-seed anchor removes most, but not
  all, seed-to-seed variance risk — a genuinely small effect could still be within noise. Any
  comparison producing a small (<0.05 NSE) delta should be treated as inconclusive rather than a
  confirmed null result.
- **Finetune-stage residual not captured.** This study explains the pretrain-stage gain, which is
  the majority of the total effect, not 100% of it — up to ~0.10 NSE (RECO in particular) is
  attributable to the finetune stage and isn't addressed here.
- **Amazon/Rangeland capacity-matched controls change architecture only, not the flux-only target
  reduction or any other production setting** — isolates capacity/dropout specifically.

## Output locations

- Capacity-matched: `outputs/{amazon,rangeland}_domain/models/best_model_capmatched.pt`,
  `outputs/{amazon,rangeland}_domain/evaluation_capmatched/` — existing production checkpoints
  untouched.
- Pairwise pretrain (the 3 new runs): `outputs/multi_domain/models/pretrained_fluxonly_dom-<subset>_seed1/`,
  `outputs/multi_domain/evaluation/pretrained_fluxonly_dom-<subset>_seed1/` — distinct from the
  no-subset path by construction, so the existing full-3-domain production/publication checkpoint
  at `pretrained_fluxonly_seed1/` (the matched-seed anchor, reused not rerun — see above) is never
  touched.
