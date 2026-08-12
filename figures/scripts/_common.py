"""
Shared constants and helpers used by 2+ of the make_figureN.py scripts (paths, styling,
seedavg loading). Each make_figureN.py is otherwise self-contained and dedicated to one
figure — see each script's own docstring. Not a figure-producing script itself; has no
main().
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# figures/scripts/_common.py -> repo root is two levels up
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.plots import PALETTE  # noqa: E402

FIGURES_DIR = REPO_ROOT / "figures"
DPI = 300

# Final figures use the 5-seed average (seedavg) results, not any single seed -- see
# project_management/current_project_status.md and shared/seed_aggregation.py. Per-seed
# variance (std) is deliberately not plotted yet -- _load_seedavg drops it.
ARCTIC_FLUXONLY_TEST = REPO_ROOT / "outputs/arctic_domain/evaluation/500K_s400_fluxonly_seedavg/metrics_test_seedavg.csv"
RANGELAND_FLUXONLY_TEST = REPO_ROOT / "outputs/rangeland_domain/evaluation_fluxonly_seedavg/metrics_test_seedavg.csv"
AMAZON_TEST = REPO_ROOT / "outputs/amazon_domain/evaluation_seedavg/metrics_test_seedavg.csv"
MD_EVAL_DIR = REPO_ROOT / "outputs/multi_domain/evaluation"
MD_PRETRAINED_SEEDAVG = MD_EVAL_DIR / "pretrained_fluxonly_seedavg"
MD_FINETUNED_SEEDAVG = MD_EVAL_DIR / "finetuned_fluxonly_seedavg"
SEEDS = [1, 2, 3, 4, 5]
DOMAINS = ["arctic", "amazon", "rangeland"]

SSP_LABELS = {"ssp1_2_6_mri_esm2_0": "SSP1-2.6", "ssp5_8_5_mri_esm2_0": "SSP5-8.5"}
AMAZON_TARGET_LABELS = {
    "active_fire_count": "Active fire\ncount",
    "burned_area": "Burned area",
    "discharge": "Discharge",
}
RANGELAND_DAY_TO_MONTH = 30  # display-only rescale of Rangeland's per-day RMSE to per-month,
                             # so it's magnitude-comparable to Arctic's native per-month
                             # fluxes -- see each figure's own comments

# One domain->color mapping shared across figures. PALETTE[0:3] are reserved for grouped
# categories (Arctic's historical/SSP1-2.6/SSP5-8.5 in Figure 4, Individual/Pretrained/
# Fine-tuned in Figure 6), so domain identity colors start at PALETTE[4].
DOMAIN_COLOR = {"arctic": PALETTE[6], "amazon": PALETTE[5], "rangeland": PALETTE[4]}


def _load_seedavg(path: Path) -> pd.DataFrame:
    """Load a seedavg metrics CSV, renaming '{metric}_mean' columns back to the plain metric
    name so downstream code (written for single-seed metrics_test.csv) works unchanged.
    Per-seed std/n_seeds columns are dropped -- not plotted yet (see module-level note)."""
    df = pd.read_csv(path)
    df = df.rename(columns={c: c[:-len("_mean")] for c in df.columns if c.endswith("_mean")})
    drop_cols = [c for c in df.columns if c.endswith("_std")] + ["n_seeds"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.linewidth": 0.7,
        "boxplot.boxprops.linewidth": 0.8,
        "boxplot.whiskerprops.linewidth": 0.8,
        "boxplot.capprops.linewidth": 0.8,
        "boxplot.medianprops.linewidth": 1.0,
    })


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _add_grid(ax: plt.Axes) -> None:
    """Sparse, low-alpha dotted gridlines on an axis's existing (major) ticks."""
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)


def _horizontal_xticks(ax: plt.Axes) -> None:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")


MODEL_ORDER = ["Individual", "Pretrained", "Fine-tuned"]


def _normalize_individual(domain: str, metric: str) -> pd.DataFrame:
    """Load an individual domain's flux-only test-set metric, normalized to a plain
    {target, metric} frame (drops Arctic's degenerate rows / period column, Rangeland's
    '_predicted' target-name suffix — neither matches multi-domain's own conventions)."""
    if domain == "arctic":
        df = _load_seedavg(ARCTIC_FLUXONLY_TEST)
        df = df[~df["obs_degenerate"]]
        return df[["target", metric]]
    if domain == "rangeland":
        df = _load_seedavg(RANGELAND_FLUXONLY_TEST)
        df = df.assign(target=df["target"].str.replace("_predicted", "", regex=False))
        return df[["target", metric]]
    df = _load_seedavg(AMAZON_TEST)  # amazon: no flux-only variant, no normalization needed
    return df[["target", metric]]


def _domain_combined(domain: str, metric: str) -> pd.DataFrame:
    """Individual/Pretrained/Fine-tuned rows for one domain, one metric, as a single
    {target, metric, model} frame -- used by figures that compare all three arms per target
    (Figure 6) and by metric_decomposition/ (KGE component comparison)."""
    individual = _normalize_individual(domain, metric).assign(model="Individual")
    pretrained = _load_seedavg(MD_PRETRAINED_SEEDAVG / domain / f"{domain}_metrics_seedavg.csv")[
        ["target", metric]].assign(model="Pretrained")
    finetuned = _load_seedavg(MD_FINETUNED_SEEDAVG / domain / f"{domain}_metrics_seedavg.csv")[
        ["target", metric]].assign(model="Fine-tuned")
    combined = pd.concat([individual, pretrained, finetuned], ignore_index=True)
    if domain == "rangeland" and metric == "RMSE":
        # Display-only per-day -> per-month rescale (see RANGELAND_DAY_TO_MONTH) -- applied
        # once here so it covers all three model sources (individual/pretrained/finetuned)
        # consistently. Only RMSE needs it; r/alpha/beta/NSE/KGE/PBIAS are scale-invariant or
        # already unitless.
        combined[metric] = combined[metric] * RANGELAND_DAY_TO_MONTH
    # draw_metric_boxplot_panel groups via `sorted(unique())`; an ordered Categorical makes
    # that read Individual -> Pretrained -> Fine-tuned instead of alphabetical.
    combined["model"] = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
    return combined
