# KGE Decomposition

## Motivation

Figures 4/6/7 show that multi-domain training changes KGE (and NSE/RMSE/PBIAS) for most
targets, but not *why* — a KGE improvement could come from better timing (correlation),
better matching the target's variability (variability ratio), better matching its mean level
(bias ratio), or some mix. `shared/metrics.py::kge()` already computes all three internally
to build the single KGE scalar; `kge_components()` exposes them directly, and
`shared/metrics.py::compute_metrics()` now includes them by default — so every existing
evaluation code path (`per_unit_metrics`, `metrics_df_by_period`, and each domain's own
`04_evaluate.py`) gets `r`/`alpha`/`beta` for free.

## Design

- **r** — Pearson correlation between prediction and observation (timing/pattern match).
- **alpha** — `std(pred) / std(obs)` (variability ratio; 1 = matches the target's spread).
- **beta** — `mean(pred) / mean(obs)` (bias ratio; 1 = matches the target's mean level).
- All three are 1 at a perfect prediction, by construction:
  `KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)`.

**Per target, not aggregated across targets** — GPP and RECO (or any two targets within a
domain) can improve via different mechanisms, so lumping them into one domain-level number
would hide that. Every target in every domain gets its own panel; nothing is pre-filtered for
"interesting" results — that judgment call comes after looking at all of them.

Compares all three arms already used elsewhere in the paper (Individual / Multi-domain
pretrained / Multi-domain fine-tuned), median across held-out units, 5-seed average — same
`*_metrics_seedavg.csv` files and `_domain_combined()` loader Figure 6 uses (see
`figures/scripts/_common.py`), just pulling `r`/`alpha`/`beta` instead of RMSE/NSE/KGE/PBIAS.

## Reuse note

Getting `r`/`alpha`/`beta` into the existing `*_metrics_seedavg.csv` files required
re-running each domain's `04_evaluate.py` (and multi-domain's) for the existing 5 seeds, then
re-running `run_seed_sweep.py --aggregate` — **re-evaluation only, no retraining**: every
`04_evaluate.py` reads already-saved checkpoints/predictions and recomputes metrics from them,
so this doesn't touch model weights or change any existing RMSE/NSE/KGE/PBIAS value, only adds
the three new columns.

## Output locations

- `metric_decomposition/figures/kge_decomposition_summary.csv` — one row per
  (domain, target, model, component).
- `metric_decomposition/figures/kge_decomposition_{arctic,amazon,rangeland}.png` — one figure
  per domain, one panel per target, grouped bars (r/alpha/beta x Individual/Pretrained/
  Fine-tuned).

This first pass is intentionally exploratory (every target, unfiltered) — a polished,
publication-styled version matching Figure 6/7's exact visual convention can follow once the
interesting targets are picked.
