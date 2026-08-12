# Ablation Study — Why Does Multi-Domain Beat Domain-Specific?

> **Update (2026-08-12) — Rangeland's capacity confound is now resolved, not just tested.**
> This study originally tested Rangeland's capacity-confound hypothesis two ways: a
> `--capacity-matched` control (artificially resized to exactly match the multi-domain trunk:
> 6 layers/1024-ff/dropout=0.1) and, later, an `--amazon-sized` stand-in used as the
> "Individual" baseline (Rangeland had never been properly tuned, so a borrowed architecture
> was the fairest available proxy — see `key_findings_log.md` `AB-capacitypairwise0806`). Both
> were necessary only because no real tuned Rangeland baseline existed yet. A real
> hyperparameter-tuning sweep now does (`hyperparameter_tuning/hyperparameter_tuning_description.md`
> "Resolution"; `key_findings_log.md` `RG-retune0812`) — production is now `hidden_dim=256,
> dropout=0.15`, and Rangeland's individual model is competitive with (RECO/Rm: *better than*)
> the multi-domain fine-tuned model. Consequently:
> - **`make_ablation_figures.py`'s Rangeland "Individual" arm now loads the real tuned
>   production model**, not `--amazon-sized`.
> - **The "Capacity-matched" arm has been dropped from the plotted comparison for both Amazon
>   and Rangeland**, per explicit user request — a real tuned individual baseline is the more
>   direct and rigorous way to answer what capacity-matched was trying to isolate for either
>   domain. (Amazon's tuning sweep itself found no size with a real, non-noise advantage over
>   production — see `HP-sweep0812` — but capacity-matched is still not shown, even though it
>   remains a theoretically distinct question — "would a substantially bigger/deeper
>   architecture help" — from "did the hidden_dim sweep find a better size.")
> - The `--amazon-sized`/`--capacity-matched` checkpoints and CSVs (both domains) are untouched
>   on disk (historical record of what was actually run — see "Hypotheses under test" and
>   "Output locations" below, describing the *original* pre-2026-08-12 study as run), just no
>   longer plotted. `make_ablation_figures.py`'s Rangeland "Individual" arm now loads the real
>   tuned production model, not `--amazon-sized`.
> - The figures no longer have a seed=1-only variant either — only the 5-seed average is
>   produced now (see the script's own docstring).

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
  untouched. Neither domain's is plotted anymore (see the 2026-08-12 update note at the top of
  this file) but the files remain on disk.
- Rangeland's `--amazon-sized` stand-in (superseded, no longer plotted, same reasoning):
  `outputs/rangeland_domain/models/best_model_fluxonly_amazonsized.pt`,
  `outputs/rangeland_domain/evaluation_fluxonly_seed{1,avg}_amazonsized/`.
- Pairwise pretrain (the 3 new runs): `outputs/multi_domain/models/pretrained_fluxonly_dom-<subset>_seed1/`,
  `outputs/multi_domain/evaluation/pretrained_fluxonly_dom-<subset>_seed1/` — distinct from the
  no-subset path by construction, so the existing full-3-domain production/publication checkpoint
  at `pretrained_fluxonly_seed1/` (the matched-seed anchor, reused not rerun — see above) is never
  touched.
