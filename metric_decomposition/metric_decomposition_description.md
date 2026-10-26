# KGE Decomposition

## Motivation

Explain *which* KGE component drives the multi-domain change in Figures 4/6/7. `shared/metrics.py::kge_components()` exposes them; `compute_metrics()` includes them by default, so every evaluation path (`per_unit_metrics`, `metrics_df_by_period`, each `04_evaluate.py`) outputs `r`/`alpha`/`beta`.

## Design

- **r**: Pearson correlation (timing). **alpha**: `std(pred)/std(obs)` (variability). **beta**: `mean(pred)/mean(obs)` (bias). All = 1 when perfect; `KGE = 1 − sqrt((r−1)² + (alpha−1)² + (beta−1)²)`.
- KGE itself not plotted (already in Figure 6d/7d); `COMPONENTS = ["r", "alpha", "beta"]` in `decompose_kge.py`.
- Per target, never aggregated across targets; no pre-filtering.
- Arms (`PLOT_MODELS`): Individual vs. Multi-domain fine-tuned (pretrained omitted — tracks fine-tuned closely). Median across held-out units, 5-seed average, from `*_metrics_seedavg.csv` via `_domain_combined()` (`figures/scripts/_common.py`), as Figure 6.

## Reuse note

Columns were added by re-running each `04_evaluate.py` (incl. multi-domain) for the 5 seeds, then `run_seed_sweep.py --aggregate`. Re-evaluation only — no retraining; existing RMSE/NSE/KGE/PBIAS unchanged.

## Output locations

- `metric_decomposition/figures/kge_decomposition_summary.csv` — one row per (domain, target, model, component).
- `metric_decomposition/figures/kge_decomposition_all_domains.png` — only figure: row per domain, panel per target, grouped bars (r/alpha/beta × Individual/Fine-tuned), IQR error bars, Okabe-Ito, inch-based layout as `figures/scripts/make_figure6.py`. Rangeland row shares one y-axis; Arctic/Rangeland y-floors 0.5/0.7.
