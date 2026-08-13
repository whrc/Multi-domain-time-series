# Individual-Model Hyperparameter Tuning

## Motivation

Every domain's production config *originally* stated "no grid search" in its own comments —
architectures were sized by judgment (data volume, A100 memory headroom), not swept. Before
comparing individual per-domain models against the multi-domain model, each individual
baseline should be a genuine best-effort model, not an assumed-adequate one. (Rangeland's
config now reflects this sweep's result — see "Resolution" below; Arctic's and Amazon's
comments are still accurate as of this sweep.)

## Design

Single architecture dimension (`hidden_dim`), 3 candidates bracketing each domain's current
production value (half / current / double), single seed. Base architecture
(`num_layers`/`num_heads`/`feedforward_dim`) is each domain's own real production value,
unchanged. Dropout is fixed at **0.15** across every domain/size for consistency, differing
from each domain's own live production dropout (which is untouched).

| Domain | num_layers | num_heads | feedforward_dim | dropout | hidden_dim candidates |
|---|---|---|---|---|---|
| Arctic | 6 | 8 | 1024 | 0.15 | 128 / 256 / 512 |
| Amazon | 3 | 4 | 512 | 0.15 | 64 / 128 / 256, extended to 32 and 16 after 64/128/256 came back flat (noise-level, ~0.2% spread) to check where -- if anywhere -- performance actually drops off (see "Amazon extension" under "Resolution" below) |
| Rangeland | 3 | 4 | 256 | 0.15 | 32 / 64 / 128, extended to 256 and 512 after 32/64/128 showed a monotonic, non-plateauing improvement (see "Resolution" below) |

Target sets match each domain's own production convention: Arctic and Rangeland run
`--flux-only` (GPP/RECO and GPP/RECO/Rm/Rg respectively); Amazon has no flux-only mode.

Selection is by **minimum validation loss** (not test-set performance) — avoids test-set
leakage into model selection. Each domain's winning size becomes a candidate "final" individual
model; whether to promote it to a full 5-seed production rerun is a separate decision made
after the sweep, not automatic — see "Resolution" below for what was actually decided.

## Resolution (2026-08-12)

| Domain | Winner | vs. original production | Decision |
|---|---|---|---|
| Arctic | medium (256) | Matches current production exactly | No change. |
| Amazon | xxsmall (16) | Extended down from the original 3-point tie to check for a real drop-off (see "Amazon extension" below) — none found. Val loss across all 5 sizes (16/32/64/128/256, a 16x range) spans 0.513–0.518, under 1%, still noise | **Hidden_dim itself not promoted off this tie** — picking a winner here would mean choosing Amazon's baseline architecture arbitrarily. But after all 4 architecture dimensions independently came back flat (see "Amazon feedforward_dim sweep"/"Amazon num_layers sweep"/"Amazon dropout sweep" below), the smallest/fastest combination found across the whole walk (hidden=64, ffn=256, layers=3, dropout=0.10) **was promoted to production for efficiency** — no measurable accuracy cost, not a claim of improvement. See `key_findings_log.md` `AZ-retune0813`. |
| Rangeland | xlarge (256) | ~40% better validation loss than the original 64/dropout=0.3 config; confirmed a genuine peak, not just best-tested, by an xxlarge/512 probe that came back *worse* (0.0356 vs. 0.0249) | **Promoted to production** — `config/rangeland_domain.yaml`'s `production:` block now uses the exact tested config (`hidden_dim=256, dropout=0.15`), and Rangeland's individual pipeline was rerun at all 5 publication seeds. This changes Rangeland's "Individual" numbers in Figures 4/6/7/8 and the KGE decomposition; Arctic/Amazon and the multi-domain trunk are unaffected (Rangeland's own architecture doesn't feed the shared trunk). |

### Amazon extension (2026-08-13)

The original small/medium/large sweep (64/128/256) came back flat (0.517/0.518/0.517) —
different from Rangeland's case, where 32/64/128 was still monotonically *improving* and
therefore worth extending upward to find where it peaks. Amazon showed no such trend, but the
open question was whether the plateau was hiding a real floor just below the tested range (i.e.
was 64 an arbitrary lower bound, not a true minimum). Extended down two more steps
(`model_xsmall`=32, `model_xxsmall`=16 in `config/amazon_domain.yaml`; same base architecture,
dropout=0.15, single seed) to check.

**Result: the plateau simply continues.** Best val loss across all 5 sizes: 16→0.513,
32→0.517, 64→0.517, 128→0.518, 256→0.517 — a 16x range in hidden_dim moves val loss by under 1%,
with no monotonic trend in either direction. This is a genuine negative result, not an
unexplored edge: Amazon's validation loss is insensitive to hidden_dim across the entire
tested range, not just tied at the 3 original points. Reinforces (does not change) the
"not promoted" decision above. `hyperparameter_tuning_winners.yaml`'s `amazon: xxsmall` entry is
the script's mechanical argmin over a flat line, not a substantive winner — do not read it as
one; the Resolution table above is authoritative.

### Amazon feedforward_dim sweep (2026-08-13)

Both hidden_dim extensions above (`HP-sweep0812`'s original 3 points plus the `HP-amazonext0813`
extension) held `feedforward_dim=512` fixed throughout — an 8:1 ratio at hidden_dim=64, wider
than the standard 4:1 transformer convention (and than Amazon's own live production ratio:
hidden=128/ffn=512 is exactly 4:1). With hidden_dim settled at 64 (the smallest architecture
tied for best, per the extension above), swept `feedforward_dim` instead: `model_ffn_narrow`=128
(2:1 ratio), `model_ffn_std`=256 (4:1 ratio), reusing `model_small`'s already-computed 512
(8:1 ratio) as the third point.

**Result: also flat.** Best val loss: 128->0.517, 256->0.513, 512->0.517 (reused from `model_small`).
Combined with every hidden_dim point tested (16-256) landing in the same 0.513-0.518 band,
Amazon's validation loss appears to sit at a floor that's insensitive to both hidden_dim and
feedforward_dim across the ranges tested — consistent with the floor being set by irreducible
noise/label variance in the hydrological data itself (discharge/fire/burned-area targets are
inherently noisy), not by model capacity in either dimension. Production (hidden=128, ffn=512)
kept as-is.

### Amazon num_layers sweep (2026-08-13)

With hidden_dim=64 and feedforward_dim=256 settled as the marginal-best/standard-ratio point
(`model_ffn_std`), swept `num_layers` next: 2 and 4 (`model_layers2`/`model_layers4`), plus 6
(`model_layers6`, matching the multi-domain shared trunk's depth as a reference point), reusing
`model_ffn_std`'s already-computed 3-layer cell (val=0.513) as the anchor.

**Result: still flat, with a faint (likely still noise-level) shallower-is-slightly-better
tilt.** Best val loss: 2->0.516, 3->0.513 (reused), 4->0.520, 6->0.519 — a ~1.4% spread, larger
than the hidden_dim/feedforward_dim sweeps' spreads (~0.2-1%) but still small next to Rangeland's
genuine ~40% capacity-driven improvement. Three independent architecture dimensions (hidden_dim,
feedforward_dim, num_layers) have now all come back essentially flat for Amazon.

### Amazon dropout sweep (2026-08-13)

Last untested architecture dimension. Fixed at hidden_dim=64/num_layers=3/feedforward_dim=256
(`model_ffn_std`'s settings) and swept dropout: 0.10, 0.20, 0.30 (`model_dropout{10,20,30}`),
reusing `model_ffn_std`'s already-computed dropout=0.15 (val=0.513) as the anchor. 0.30 also
matches Rangeland's original (pre-retune) dropout for cross-domain reference; 0.20 matches
Amazon's live production dropout.

**Result: still flat.** Best val loss: 0.10->0.511, 0.15->0.513 (reused), 0.20->0.518,
0.30->0.516 — same ~1.4% band as the other three sweeps. Four independent architecture
dimensions (hidden_dim, feedforward_dim, num_layers, dropout) have now all come back
essentially flat for Amazon, converging on the data-noise-floor reading. See
`hyperparameter_tuning/figures/amazon_architecture_search.png` for all four swept together.

### Promoted to production (2026-08-13)

Despite the null results above, the smallest/fastest point found across the whole walk
(hidden=64, ffn=256, num_layers=3, dropout=0.10, best val=0.511) **was promoted to
production** — `config/amazon_domain.yaml`'s `production:` block now uses this exact tested
config (was hidden=128, ffn=512, num_layers=3, dropout=0.2), and Amazon's individual pipeline
was rerun at all 5 publication seeds. This is an **efficiency-only promotion**, not an accuracy
claim — every sweep above found no real skill difference between this config and the prior one.
5-seed median NSE, old -> new: discharge 0.356->0.371 (unchanged), active_fire_count
0.368->0.310, burned_area 0.047->0.008 (both near-zero/no-skill before and after). The
Individual-vs-multi-domain gap that motivates the paper's headline finding is unchanged. See
`key_findings_log.md` `AZ-retune0813` for the full before/after and its manuscript-framing
implication (flagged `NEEDS HUMAN REVIEW`). This changes Amazon's "Individual" numbers in
Figures 4/6/7/8, the ablation figures, and the KGE decomposition; Arctic/Rangeland and the
multi-domain trunk are unaffected (Amazon's own architecture doesn't feed the shared trunk).

Downstream implication: `ablation_test/ablation_description.md`'s capacity-matched study
(`AB-capacitypairwise0806`) was run against Rangeland's *original* individual config
(64/dropout=0.3) — that study's own numbers are unaffected (historical record of what was
actually run) but are no longer directly comparable to current production. Not rerun as part
of this change; flagged in that file.

## Reuse note

Arctic and Amazon's seed=1 sweeps were already run once (on the parked
`feat/rangeland-obs-multidomain` branch, as part of an unrelated observation-transfer study
that also swept these two domains at the exact same architecture grid). Since neither domain's
config or data changed since then, those outputs were copied forward rather than retrained —
see `project_management/key_findings_log.md`. Only Rangeland was actually trained fresh for
this sweep.

## Output locations

- Per-cell runs: `outputs/{arctic,amazon,rangeland}_domain/.../..._{small,medium,large}_seed1/`,
  plus `outputs/amazon_domain/.../..._{xsmall,xxsmall}_seed1/` (hidden_dim extension),
  `outputs/rangeland_domain/.../..._{xlarge,xxlarge}_seed1/` (Rangeland extension),
  `outputs/amazon_domain/.../..._{ffn_narrow,ffn_std}_seed1/` (feedforward_dim sweep),
  `outputs/amazon_domain/.../..._{layers2,layers4,layers6}_seed1/` (num_layers sweep), and
  `outputs/amazon_domain/.../..._{dropout10,dropout20,dropout30}_seed1/` (dropout sweep)
- Study artifacts: `hyperparameter_tuning/hyperparameter_tuning_winners.yaml`,
  `hyperparameter_tuning/figures/hyperparameter_tuning_results.{png,csv}` (domain comparison,
  hidden_dim only), `hyperparameter_tuning/figures/amazon_architecture_search.png` (all 4 of
  Amazon's swept dimensions combined, `hyperparameter_tuning/plot_amazon_architecture_search.py`)
