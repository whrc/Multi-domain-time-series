"""Shared paths, styling and seedavg-loading helpers for the make_figureN.py scripts."""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.plots import PALETTE  # noqa: E402

FIGURES_DIR = REPO_ROOT / "figures"
DPI = 300

# Figures use 5-seed-average (seedavg) metrics; per-seed std is not plotted.
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
RANGELAND_DAY_TO_MONTH = 30  # display-only per-day -> per-month rescale, to match Arctic's units

# PALETTE[0:3] are reserved for grouped categories (SSP/period, model arm), so domains use 4-6.
DOMAIN_COLOR = {"arctic": PALETTE[6], "amazon": PALETTE[5], "rangeland": PALETTE[4]}

JITTER_DEG = 0.35  # degrees; separates co-located sites at regional map scale
CLUSTER_RADIUS_DEG = 0.05  # ~5 km; closer sites (e.g. US-LS1/US-LS2) count as coincident


def _jitter_coincident_sites(coords: pd.DataFrame) -> pd.DataFrame:
    """Spread sites within CLUSTER_RADIUS_DEG evenly in a circle around their shared location."""
    coords = coords.reset_index(drop=True).copy()
    lat, lon = coords["lat"].to_numpy(), coords["lon"].to_numpy()
    dist = np.hypot(lat[:, None] - lat[None, :], lon[:, None] - lon[None, :])
    cluster = np.arange(len(coords))
    for i, j in zip(*np.where(np.triu(dist < CLUSTER_RADIUS_DEG, k=1))):
        cluster[cluster == cluster[j]] = cluster[i]
    coords["_cluster"] = cluster
    group = coords.groupby("_cluster")
    rank = group.cumcount()
    size = group["lat"].transform("size")
    angle = 2 * np.pi * rank / size
    dupes = size > 1
    coords.loc[dupes, "lat"] += JITTER_DEG * np.sin(angle)[dupes]
    coords.loc[dupes, "lon"] += JITTER_DEG * np.cos(angle)[dupes]
    return coords.drop(columns="_cluster")


def _load_seedavg(path: Path) -> pd.DataFrame:
    """Load a seedavg metrics CSV with '{metric}_mean' renamed to '{metric}' and std/n_seeds dropped."""
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
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)


def _horizontal_xticks(ax: plt.Axes) -> None:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")


# Multiples of 0.1 only, so 1-decimal tick labels never misstate a tick's position.
_KGE_TICK_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]


def _apply_kge_ticks(ax: plt.Axes, bounded_above: bool = True) -> None:
    """Y ticks anchored at 1.0 with 1-decimal labels. bounded_above=True (KGE) allows no ticks
    above 1.0; pass False for r/alpha/beta, which can exceed 1.0. ylim is reapplied because
    set_yticks can silently expand the view."""
    ylim = ax.get_ylim()
    lo, hi = ylim[0] - 1.0, ylim[1] - 1.0
    # Always keep the 1.0 reference tick in view, with a small margin.
    if hi < 0.02:
        hi = 0.02
    if not bounded_above and lo > -0.02:
        lo = -0.02
    ylim = (lo + 1.0, hi + 1.0)

    def ticks_for(step: float) -> list[float]:
        start = math.floor(lo / step) * step
        n = int(math.ceil((hi - start) / step)) + 1
        candidates = [round(start + i * step, 10) for i in range(n)]
        # Keep only ticks inside the visible range (and <=1.0 when bounded_above).
        upper = 1e-9 if bounded_above else hi + 1e-9
        return sorted(round(t + 1.0, 10) for t in candidates if lo - 1e-9 <= t <= upper)

    step = next((s for s in _KGE_TICK_STEPS if s >= (hi - lo) / 4), _KGE_TICK_STEPS[-1])
    ticks = ticks_for(step)
    # Fall back to finer steps until at least 3 ticks are visible.
    for finer in reversed([s for s in _KGE_TICK_STEPS if s < step]):
        if len(ticks) >= 3:
            break
        ticks = ticks_for(finer)
    ax.set_yticks(ticks)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


MODEL_ORDER = ["Individual", "Pretrained", "Fine-tuned"]


def _normalize_individual(domain: str, metric: str) -> pd.DataFrame:
    """Individual model's flux-only test metric as a {target, metric} frame, matching
    multi-domain naming (Arctic degenerate rows dropped, Rangeland '_predicted' suffix stripped)."""
    if domain == "arctic":
        df = _load_seedavg(ARCTIC_FLUXONLY_TEST)
        df = df[~df["obs_degenerate"]]
        return df[["target", metric]]
    if domain == "rangeland":
        df = _load_seedavg(RANGELAND_FLUXONLY_TEST)
        df = df.assign(target=df["target"].str.replace("_predicted", "", regex=False))
        return df[["target", metric]]
    df = _load_seedavg(AMAZON_TEST)  # amazon: no flux-only variant
    return df[["target", metric]]


def _domain_combined(domain: str, metric: str) -> pd.DataFrame:
    """Individual/Pretrained/Fine-tuned rows for one domain and metric as one {target, metric, model} frame."""
    individual = _normalize_individual(domain, metric).assign(model="Individual")
    pretrained = _load_seedavg(MD_PRETRAINED_SEEDAVG / domain / f"{domain}_metrics_seedavg.csv")[
        ["target", metric]].assign(model="Pretrained")
    finetuned = _load_seedavg(MD_FINETUNED_SEEDAVG / domain / f"{domain}_metrics_seedavg.csv")[
        ["target", metric]].assign(model="Fine-tuned")
    combined = pd.concat([individual, pretrained, finetuned], ignore_index=True)
    if domain == "rangeland" and metric == "RMSE":
        combined[metric] = combined[metric] * RANGELAND_DAY_TO_MONTH  # other metrics are scale-invariant
    # Ordered Categorical so draw_metric_boxplot_panel's sorted() follows MODEL_ORDER, not alphabetical.
    combined["model"] = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
    return combined
