# Individual-Model Hyperparameter Tuning

## Motivation

Every domain's production config states "no grid search" in its own comments (see
`config/{arctic,amazon,rangeland}_domain.yaml`) — architectures were sized by judgment (data
volume, A100 memory headroom), not swept. Before comparing individual per-domain models against
the multi-domain model, each individual baseline should be a genuine best-effort model, not an
assumed-adequate one.

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
| Rangeland | 3 | 4 | 256 | 0.15 | 32 / 64 / 128 |

Target sets match each domain's own production convention: Arctic and Rangeland run
`--flux-only` (GPP/RECO and GPP/RECO/Rm/Rg respectively); Amazon has no flux-only mode.

Selection is by **minimum validation loss** (not test-set performance) — avoids test-set
leakage into model selection. Each domain's winning size becomes a candidate "final" individual
model; whether to promote it to a full 5-seed production rerun is a separate decision made
after the sweep (see `project_management/current_project_status.md`), not automatic.

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
