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
- KGE itself is deliberately **not** plotted here — it's already shown in Figure 6d/7d, and this
  study is purely about which component (r/alpha/beta) explains a KGE change, not about
  re-showing KGE. `COMPONENTS` in `decompose_kge.py` is `["r", "alpha", "beta"]` only.

**Per target, not aggregated across targets** — GPP and RECO (or any two targets within a
domain) can improve via different mechanisms, so lumping them into one domain-level number
would hide that. Every target in every domain gets its own panel; nothing is pre-filtered for
"interesting" results — that judgment call comes after looking at all of them.

Compares two of the three arms already used elsewhere in the paper — Individual and Multi-domain
fine-tuned (`PLOT_MODELS` in `decompose_kge.py`) — median across held-out units, 5-seed average —
same `*_metrics_seedavg.csv` files and `_domain_combined()` loader Figure 6 uses (see
`figures/scripts/_common.py`), just pulling `r`/`alpha`/`beta` instead of RMSE/NSE/KGE/PBIAS.
Multi-domain pretrained is intentionally omitted from this comparison for simplicity — it tracks
fine-tuned closely everywhere in this project (see `decompose_kge.py`'s own docstring).

## Reuse note

Getting `r`/`alpha`/`beta` into the existing `*_metrics_seedavg.csv` files required
re-running each domain's `04_evaluate.py` (and multi-domain's) for the existing 5 seeds, then
re-running `run_seed_sweep.py --aggregate` — **re-evaluation only, no retraining**: every
`04_evaluate.py` reads already-saved checkpoints/predictions and recomputes metrics from them,
so this doesn't touch model weights or change any existing RMSE/NSE/KGE/PBIAS value, only adds
the three new columns.

## Output locations

- `metric_decomposition/figures/kge_decomposition_summary.csv` — one row per
  (domain, target, model, component), component in `{r, alpha, beta}`.
- `metric_decomposition/figures/kge_decomposition_all_domains.png` — the only figure (no
  separate per-domain figures): one row per domain, one panel per target, grouped bars
  (r/alpha/beta x Individual/Fine-tuned), IQR error bars, publication-styled (tight, minimal
  whitespace, Okabe-Ito colors, rectangular legend) to match Figure 6/7's visual convention.
  Same manual inch-based layout as `figures/scripts/make_figure6.py`. Rangeland's row shares
  one y-axis across its targets since they all sit in a tight band. Both Arctic's and
  Rangeland's panels are floored (0.5 and 0.7 respectively, not 0) since every r/alpha/beta
  value in the summary CSV sits comfortably above those — starting at 0 would waste most of
  each panel on empty space below the real data.
