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
| Amazon | 3 | 4 | 512 | 0.15 | 64 / 128 / 256 |
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
| Amazon | small (64) | Val loss differs by ~0.2% across all 3 sizes (0.517/0.518/0.517) — noise, not signal | **Not promoted.** Picking a winner off a tie this close would mean choosing Amazon's baseline architecture arbitrarily; production (128) kept as-is. See `key_findings_log.md` for the full reasoning. |
| Rangeland | xlarge (256) | ~40% better validation loss than the original 64/dropout=0.3 config; confirmed a genuine peak, not just best-tested, by an xxlarge/512 probe that came back *worse* (0.0356 vs. 0.0249) | **Promoted to production** — `config/rangeland_domain.yaml`'s `production:` block now uses the exact tested config (`hidden_dim=256, dropout=0.15`), and Rangeland's individual pipeline was rerun at all 5 publication seeds. This changes Rangeland's "Individual" numbers in Figures 4/6/7/8 and the KGE decomposition; Arctic/Amazon and the multi-domain trunk are unaffected (Rangeland's own architecture doesn't feed the shared trunk). |

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

- Per-cell runs: `outputs/{arctic,amazon,rangeland}_domain/.../..._{small,medium,large}_seed1/`
- Study artifacts: `hyperparameter_tuning/hyperparameter_tuning_winners.yaml`,
  `hyperparameter_tuning/figures/hyperparameter_tuning_results.{png,csv}`
