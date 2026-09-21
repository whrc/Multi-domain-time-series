# Ablation Study — Why Does Multi-Domain Beat Domain-Specific?

## Current status

The active part of this study is the three pairwise multi-domain runs — `{Arctic,Amazon}`,
`{Arctic,Rangeland}`, `{Amazon,Rangeland}` — each trained pretrain+finetune at 5 seeds, for
direct consistency with the full 3-domain production sweep (`run_seed_sweep.py`), which is also
always pretrain+finetune.

The capacity-matched controls (Amazon and Rangeland individual models resized to the
multi-domain trunk's architecture) and Rangeland's `--amazon-sized` stand-in baseline are
retired. Both existed only because Rangeland had no properly-tuned individual baseline yet, so a
resized or borrowed architecture was the fairest available proxy for asking "is the individual
baseline capacity-starved relative to the shared trunk?" `hyperparameter_tuning/` now answers
that question directly and more rigorously, via a real sweep across hidden_dim,
feedforward_dim, num_layers, and dropout for every domain (see
`hyperparameter_tuning/hyperparameter_tuning_description.md`). The retired checkpoints and CSVs
remain on disk as a historical record but are no longer read by `make_ablation_figures.py`; its
"Individual" arm for both Amazon and Rangeland always loads whichever config is currently in
production (`outputs/{amazon,rangeland}_domain/evaluation_seedavg/`), so the figures track any
future retune automatically without a code change.

Both domains' individual baselines were retuned after this study was first designed: Rangeland
moved to `hidden_dim=256, dropout=0.15` for a genuine validation-loss improvement; Amazon moved
to `hidden_dim=64, dropout=0.10` for efficiency only (its architecture sweep found no accuracy
difference across the tested range). Neither retune touches this study's own arms — the pairwise
runs and the matched-seed anchor don't depend on either domain's individual-model architecture —
but it does mean the capacity-confound reasoning below is now moot (see "Hypotheses under test").

## Motivation

Production results show the multi-domain shared-transformer model beats domain-specific
baselines, especially for the data-scarce domains: Amazon discharge NSE 0.356 (individual) →
0.760 (multi-domain finetuned), active_fire_count 0.368 → 0.707; Rangeland GPP/RECO similarly
improved (see `project_management/key_findings_log.md` for full numbers). We can currently say
multi-domain is better, but not *why*. This ablation study isolates the actual cause, so the
paper can make a mechanistic claim instead of just reporting a comparison.

## Hypotheses under test

1. **Capacity confound.** At the time this study was designed, Amazon's and Rangeland's
   individual production models were meaningfully smaller and more regularized than the
   multi-domain shared trunk (`hidden_dim=256, layers=6, ff=1024, dropout=0.1`) they're compared
   against — e.g. Amazon at `hidden_dim=128, layers=3, ff=512`, Rangeland at `hidden_dim=64,
   layers=3, ff=256, dropout=0.3`. If most of the observed gain disappeared once the individual
   models were given the same capacity and dropout as the shared trunk, the "multi-domain" story
   would be largely a capacity/regularization artifact rather than genuine cross-domain
   transfer. This is now answered by `hyperparameter_tuning/` instead of the capacity-matched
   controls originally run here (see "Current status" above): retuning found no accuracy gain
   for Amazon from more capacity, and a real gain for Rangeland, but the multi-domain gap
   persists for both after retuning, so capacity alone doesn't explain it.

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

All new runs: **flux-only** target set (matches the flagship multi-domain finding this study is
explaining), **5 seeds**, **pretrain + finetune** for each of the three pairwise arms. Finetune
was added after the fact for direct consistency with the full 3-domain production sweep, which
is always reported pretrain+finetune; the original design was pretrain-only, reasoning that the
pretrain checkpoint already captures the large majority of the effect since finetune quality
tracks pretrain quality rather than doing independent causal work.

### The runs

| # | Run | Domains trained | Capacity | Stage | Tests hypothesis |
|---|---|---|---|---|---|
| 1 | Amazon capacity-matched | Amazon only | multi-domain trunk spec | — | 1 (retired — see "Current status") |
| 2 | Rangeland capacity-matched | Rangeland only | multi-domain trunk spec | — | 1 (retired — see "Current status") |
| 3 | Pairwise {Arctic, Amazon} | Arctic + Amazon | production trunk spec | pretrain + finetune | 2 |
| 4 | Pairwise {Arctic, Rangeland} | Arctic + Rangeland | production trunk spec | pretrain + finetune | 2 |
| 5 | Pairwise {Amazon, Rangeland} | Amazon + Rangeland | production trunk spec | pretrain + finetune | 3 |

### Matched-seed anchor — reused, not rerun

Pretrain-stage NSE can swing ~0.10–0.13 between single seeds. Comparing the pairwise runs above
against a 5-seed average or an old unseeded run risks mistaking ordinary seed variance for a
domain-subset or capacity effect — every comparison should instead be made against the full
3-domain pretrain+finetune at the *same* seeds.

That comparison point already exists: `02_train.py --stage {pretrain,finetune} --flux-only --seed
N` (no `--domains` override) is exactly the set of commands already run as seeds 1-5 of the
completed 5-seed publication sweep (`run_seed_sweep.py`) — checkpoints and metrics are already on
disk at `outputs/multi_domain/{models,evaluation}/{pretrained,finetuned}_fluxonly_seed{1..5}/`.
**Do not rerun this** — `ablation_test/run_ablation.py` deliberately does not include it, since
rerunning would silently overwrite those published production artifacts (the no-`--domains` case
reproduces the exact same path by design, so a naive rerun and the existing production run are
indistinguishable on disk). Use the existing `{pretrained,finetuned}_fluxonly_seed{1..5}` outputs
directly as the matched-seed anchor when comparing against the pairwise arms.

### Existing baselines reused (not rerun)

- Amazon individual production (`outputs/amazon_domain/`)
- Rangeland individual production (`outputs/rangeland_domain/`)
- Arctic individual production (`outputs/arctic_domain/`) — already at the multi-domain trunk's
  capacity (`hidden_dim=256, num_layers=6`), so it needs no capacity-matched control of its own
- Full 3-domain pretrained+finetuned, seeds 1-5
  (`outputs/multi_domain/evaluation/{pretrained,finetuned}_fluxonly_seed{1..5}/`) — the
  matched-seed anchor, see above
- Full 3-domain pretrained+finetuned, 5-seed average — secondary comparison, for
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
- **Amazon/Rangeland capacity-matched controls change architecture only, not the flux-only target
  reduction or any other production setting** — they isolate capacity/dropout specifically (and
  are retired, see "Current status").

## Output locations

- Capacity-matched (retired, not plotted): `outputs/{amazon,rangeland}_domain/models/best_model_capmatched.pt`,
  `outputs/{amazon,rangeland}_domain/evaluation_capmatched/`.
- Rangeland's `--amazon-sized` stand-in (retired, not plotted):
  `outputs/rangeland_domain/models/best_model_fluxonly_amazonsized.pt`,
  `outputs/rangeland_domain/evaluation_fluxonly_seed{1,avg}_amazonsized/`.
- Pairwise pretrain + finetune (the 3 arms, seeds 1-5):
  `outputs/multi_domain/models/{pretrained,finetuned}_fluxonly_dom-<subset>_seed{1..5}/`,
  `outputs/multi_domain/evaluation/{pretrained,finetuned}_fluxonly_dom-<subset>_seed{1..5}/` —
  `<subset>` is one of `amazon-arctic`, `arctic-rangeland`, `amazon-rangeland`. Distinct from the
  no-subset path by construction, so the existing full-3-domain production/publication checkpoints
  at `{pretrained,finetuned}_fluxonly_seed{1..5}/` (the matched-seed anchor, reused not rerun —
  see above) are never touched.
