# Ablation Study — Why Does Multi-Domain Beat Domain-Specific?

> **Update (2026-08-14) — Finetune stage added to the 3 pairwise arms; stray capacity/obs
> variants identified as stale and deleted.**
> The pairwise arms (`{Arctic,Amazon}`, `{Arctic,Rangeland}`, `{Amazon,Rangeland}`) were
> originally pretrain-stage only (see "Experiment design" below, which described the *original*
> design). Finetune was later added for consistency with the full 3-domain production sweep,
> which is also pretrain+finetune — the ablation should compare like with like. Along the way,
> `outputs/multi_domain/{models,evaluation}/` accumulated a set of `trunk-medium`/`trunk-small`/
> `obs_trunk` variants for the pairwise arms (both pretrain and finetune, 5 seeds) with no
> corresponding code in `run_ablation.py` or `02_train.py`, and no `key_findings_log.md` entry.
> Investigation traced these to two now-abandoned side experiments: a trunk-capacity sweep that
> was folded in around the same time the capacity-matched study below was being retired (see the
> 2026-08-12 update), and, for the Rangeland-containing subsets only (`obs_trunk`), a since-
> abandoned attempt to train Rangeland on its observed tower-flux data instead of the RangeSTAR
> process-model targets that are the actual production target (see `rangeland_description.md`
> § Overview — targets used are explicitly "not observations"). Both were confirmed stale and
> deleted from disk (2026-08-14). **The clean, currently-valid pairwise runs are the ones with no
> `trunk`/`obs` suffix** — pretrain already existed at 5 seeds for all 3 arms; finetune is being
> (re)run cleanly from those existing pretrain checkpoints via `run_ablation.py`'s new
> `pairwise_{arctic_amazon,arctic_rangeland,amazon_rangeland}_finetune` run types.

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
> - **The "Capacity-matched" arm is retired, not just unplotted, for both Amazon and Rangeland**
>   — a real hyperparameter-tuning sweep across all of hidden_dim, feedforward_dim, num_layers,
>   and dropout (`hyperparameter_tuning_description.md`) directly answers what capacity-matched
>   was only ever a proxy for ("is the individual baseline capacity-starved relative to the
>   shared trunk?"), for both domains, more rigorously than a single artificially-resized probe
>   could. It is no longer part of the active methodology.
> - The `--amazon-sized`/`--capacity-matched` checkpoints and CSVs (both domains) are untouched
>   on disk (historical record of what was actually run — see "Hypotheses under test" and
>   "Output locations" below, describing the *original* pre-2026-08-12 study as run), just no
>   longer plotted. `make_ablation_figures.py`'s Rangeland "Individual" arm now loads the real
>   tuned production model, not `--amazon-sized`.
> - The figures no longer have a seed=1-only variant either — only the 5-seed average is
>   produced now (see the script's own docstring).

## Current state (2026-08-13)

Both domains' individual baselines have now been retuned from what this study originally tested
against — the "Hypotheses under test" table below is a **historical snapshot of the pre-retune
architectures**, not the current configs:

| Domain | Individual production config (as tested here, pre-retune) | Individual production config (current) |
|---|---|---|
| Amazon | `hidden_dim=128, layers=3, ff=512, dropout=0.2` | `hidden_dim=64, layers=3, ff=256, dropout=0.10` — retuned for efficiency, not accuracy (`AZ-retune0813`) |
| Rangeland | `hidden_dim=64, layers=3, ff=256, dropout=0.3` | `hidden_dim=256, layers=3, ff=256, dropout=0.15` — retuned for a genuine ~40% validation-loss improvement (`RG-retune0812`) |

`make_ablation_figures.py`'s "Individual" arm for **both** domains now loads whichever config is
currently in production (it always reads live from `outputs/{amazon,rangeland}_domain/
evaluation_seedavg/`, so the ablation figures already reflect these retunes automatically — no
code change was needed there). The `--capacity-matched` control's own reasoning (isolating
architecture/dropout as the cause of the multi-domain gap) is unaffected by either retune, since
it was never plotted as "Individual" — see the 2026-08-12 update note above for why it's dropped
from the figures regardless.

## Motivation

Production results show the multi-domain shared-transformer model beats domain-specific
baselines, especially for the data-scarce domains: Amazon discharge NSE 0.356 (individual) →
0.760 (multi-domain finetuned), active_fire_count 0.368 → 0.707 (units-bug-corrected figures —
see `key_findings_log.md` `MD-unitsbugfix0716`); Rangeland GPP/RECO similarly improved
(`key_findings_log.md` `AZ-seedsweep0714`, `RG-seedsweep0714`, `MD-seedsweep0714`). We can
currently say multi-domain is better, but not *why*. This ablation study is designed to isolate
the actual cause, so the eventual paper can make a mechanistic claim instead of just reporting a
comparison. (Both individual baselines quoted here are Amazon's/Rangeland's *pre-retune*
numbers — see "Current state" below for what changed since.)

## Hypotheses under test

1. **Capacity confound.** Amazon's and Rangeland's individual production models were, *at the
   time this study was designed*, meaningfully smaller and more regularized than the
   multi-domain shared trunk they're compared against (table below is historical — see "Current
   state" above for what's actually in production now):

   | Domain | Individual production config (as originally tested) | Multi-domain shared trunk |
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
explained), **5 seeds** (extended from the original single-seed design, `AB-capacitypairwise0806`).

**Originally pretrain-stage only, later extended to finetune (2026-08-14 update above).** The
original reasoning for pretrain-only: `MD-prod0712`'s pretrained-vs-finetuned NSE gap is small
everywhere (Arctic GPP +0.04, RECO +0.10, Amazon ~+0.00–0.04, Rangeland +0.02–0.03), and
`MD-fluxrerun0713` showed finetune quality tracks pretrain quality rather than doing independent
causal work — a weak pretrain checkpoint produced a weak finetune result across the board, on a
larger epoch budget. So the pretrain-stage checkpoint already captures the large majority of the
effect this study is trying to explain, and finetune was deemed out of scope. It was later added
back anyway, for direct consistency with the full 3-domain production sweep (which is always
reported pretrain+finetune) — see the 2026-08-14 update above for what that finetune extension
actually is and isn't (the trunk/obs variants entangled with an earlier attempt at this are stale
and were deleted).

### The runs

| # | Run | Domains trained | Capacity | Stage | Tests hypothesis |
|---|---|---|---|---|---|
| 1 | Amazon capacity-matched | Amazon only | multi-domain trunk spec | — | 1 (retired, see 2026-08-12 update) |
| 2 | Rangeland capacity-matched | Rangeland only | multi-domain trunk spec | — | 1 (retired, see 2026-08-12 update) |
| 3 | Pairwise {Arctic, Amazon} | Arctic + Amazon | production trunk spec | pretrain + finetune | 2 |
| 4 | Pairwise {Arctic, Rangeland} | Arctic + Rangeland | production trunk spec | pretrain + finetune | 2 |
| 5 | Pairwise {Amazon, Rangeland} | Amazon + Rangeland | production trunk spec | pretrain + finetune | 3 |

### Matched-seed anchor — reused, not rerun

Past runs show pretrain-stage NSE can swing ~0.10–0.13 between single seeds (`MD-fluxrerun0713`:
Arctic GPP 0.815 vs. 0.947 across two single-seed runs). Comparing the pairwise runs above
against the existing 5-seed average, or an old unseeded run, risks mistaking ordinary seed
variance for a domain-subset or capacity effect — every comparison should instead be made against
the full 3-domain pretrain+finetune at the *same* seeds.

That comparison point already exists: `02_train.py --stage {pretrain,finetune} --flux-only --seed
N` (no `--domains` override) is exactly the set of commands already run as seeds 1-5 of the
completed 5-seed publication sweep (`run_seed_sweep.py`) — checkpoints and metrics are already on
disk at `outputs/multi_domain/{models,evaluation}/{pretrained,finetuned}_fluxonly_seed{1..5}/`.
**Do not rerun this** — `ablation_test/run_ablation.py` deliberately does not include it, since
rerunning would silently overwrite those published production artifacts (`checkpoint_path`'s
no-`--domains` case reproduces the exact same path by design, so a naive rerun and the existing
production run are indistinguishable on disk). Use the existing `{pretrained,finetuned}_fluxonly_
seed{1..5}` outputs directly as the matched-seed anchor when comparing against the pairwise arms.

### Existing baselines reused (not rerun)

- Amazon individual production (`outputs/amazon_domain/`)
- Rangeland individual production (`outputs/rangeland_domain/`)
- Arctic individual production (`outputs/arctic_domain/`) — already at the multi-domain trunk's
  capacity (`hidden_dim=256, num_layers=6`), so it needs no capacity-matched control of its own
- Full 3-domain pretrained+finetuned, seeds 1-5
  (`outputs/multi_domain/evaluation/{pretrained,finetuned}_fluxonly_seed{1..5}/`) — the
  matched-seed anchor, see above
- Full 3-domain pretrained+finetuned, 5-seed average (`MD-seedsweep0714`) — secondary comparison,
  for sanity-checking how much of any observed effect could be ordinary seed noise

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
- **Finetune-stage residual now captured (2026-08-14).** Originally out of scope (see "Experiment
  design" above) since the pretrain-stage gain was already the majority of the effect; finetune
  was added afterward for consistency with the full 3-domain production sweep, so both stages are
  now covered for the pairwise arms.
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
- Pairwise pretrain + finetune (the 3 arms, seeds 1-5):
  `outputs/multi_domain/models/{pretrained,finetuned}_fluxonly_dom-<subset>_seed{1..5}/`,
  `outputs/multi_domain/evaluation/{pretrained,finetuned}_fluxonly_dom-<subset>_seed{1..5}/` —
  `<subset>` is one of `amazon-arctic`, `arctic-rangeland`, `amazon-rangeland`. Distinct from the
  no-subset path by construction, so the existing full-3-domain production/publication checkpoints
  at `{pretrained,finetuned}_fluxonly_seed{1..5}/` (the matched-seed anchor, reused not rerun —
  see above) are never touched. No `trunk`/`obs` suffix — see the 2026-08-14 update at the top of
  this file for why any output with such a suffix is stale and should not exist anymore.
