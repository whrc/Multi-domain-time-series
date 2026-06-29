"""
Shared plotting utilities — used across all domains (and future multi-domain work).

Standards applied throughout all functions:
- Color palette: Okabe-Ito 8-color color-blind-safe palette (see PALETTE below)
- Figure size: never exceeds 10 × 10 inches; default to the smallest sensible size per plot type
- All functions are domain-agnostic: they adapt to any number of target variables and
  make no assumptions about SSP/period/grid structure.
- All functions return a matplotlib Figure; the caller is responsible for saving (fig.savefig)
  or displaying (plt.show). Pass save_path to have the function save and close automatically.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from shared.metrics import nse, rmse

# Okabe-Ito 8-color color-blind-safe palette
PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

_METRICS = ["RMSE", "NSE", "KGE", "PBIAS"]


def _finalize(fig: Figure, save_path: Path | None) -> Figure:
    """Save+close if a path is given; always return the figure."""
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def _grid_shape(n: int) -> tuple[int, int]:
    """Near-square (nrows, ncols) for n panels."""
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    per_target: dict[str, list[float]] | None = None,
    eval_every: int = 1,
    save_path: Path | None = None,
) -> Figure:
    """Epoch-by-epoch training and validation loss.

    Train loss is recorded every epoch; validation (and per-target) loss every
    ``eval_every`` epochs, so val curves are placed on an x-axis spaced by ``eval_every``.
    If per_target is given (target name -> per-eval loss), a second panel shows each
    target's loss curve so you can see whether all targets are being learned.
    """
    n_panels = 2 if per_target else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)

    val_x = [eval_every * (i + 1) for i in range(len(val_losses))]
    ax = axes[0, 0]
    ax.plot(range(1, len(train_losses) + 1), train_losses, color=PALETTE[4], label="train")
    ax.plot(val_x, val_losses, color=PALETTE[5], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss curves")
    ax.legend()

    if per_target:
        ax2 = axes[0, 1]
        for i, (name, series) in enumerate(per_target.items()):
            ax2.plot([eval_every * (j + 1) for j in range(len(series))], series,
                     color=PALETTE[i % len(PALETTE)], label=name)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("loss")
        ax2.set_title("Per-target loss")
        ax2.legend(fontsize="small")

    fig.tight_layout()
    return _finalize(fig, save_path)


def plot_pred_vs_true(
    pred_dict: dict[str, np.ndarray],
    obs_dict: dict[str, np.ndarray],
    log_scale: bool = False,
    save_path: Path | None = None,
) -> Figure:
    """Scatter of predicted vs true, one panel per target variable.

    Adapts to any number of targets (keys of pred_dict). Each panel is annotated
    with RMSE and NSE and a 1:1 reference line. Set log_scale=True for heavy-tailed
    targets (e.g. discharge, fire, carbon pools); only positive values are kept.
    """
    targets = list(pred_dict.keys())
    nrows, ncols = _grid_shape(len(targets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(min(3.2 * ncols, 10), min(3.2 * nrows, 10)), squeeze=False)
    flat = axes.flatten()

    for i, name in enumerate(targets):
        ax = flat[i]
        p = np.asarray(pred_dict[name], dtype=float)
        o = np.asarray(obs_dict[name], dtype=float)
        m = ~(np.isnan(p) | np.isnan(o))
        if log_scale:
            m &= (p > 0) & (o > 0)
        p, o = p[m], o[m]
        ax.scatter(o, p, s=6, alpha=0.4, color=PALETTE[i % len(PALETTE)], edgecolors="none")
        if p.size:
            lo = float(min(o.min(), p.min()))
            hi = float(max(o.max(), p.max()))
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, linestyle="--")
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_title(f"{name}\nRMSE={rmse(p, o):.3g}  NSE={nse(p, o):.3g}", fontsize="small")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")

    for j in range(len(targets), len(flat)):
        flat[j].axis("off")

    fig.tight_layout()
    return _finalize(fig, save_path)


def plot_metric_boxplot(
    metrics_df: pd.DataFrame,
    group_col: str | None = None,
    title: str | None = None,
    save_path: Path | None = None,
) -> Figure:
    """Boxplots of the four metrics across units, one subplot per metric.

    metrics_df must contain a `target` column and the metric columns
    RMSE, NSE, KGE, PBIAS. Boxes are grouped on the x-axis by `target`. If
    group_col is given (e.g. `period` for Arctic, `pft` for Rangeland), boxes
    are further split by that column within each target, with a legend.
    """
    targets = sorted(metrics_df["target"].unique())
    groups = sorted(metrics_df[group_col].unique()) if group_col else [None]

    fig, axes = plt.subplots(2, 2, figsize=(min(2.8 * len(targets) + 2, 10), 8), squeeze=False)
    for mi, metric in enumerate(_METRICS):
        ax = axes[mi // 2, mi % 2]
        width = 0.8 / len(groups)
        for gi, g in enumerate(groups):
            data = []
            for t in targets:
                sel = metrics_df["target"] == t
                if group_col:
                    sel &= metrics_df[group_col] == g
                vals = metrics_df.loc[sel, metric].to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                data.append(vals)
            positions = [x + (gi - (len(groups) - 1) / 2) * width for x in range(len(targets))]
            bp = ax.boxplot(
                data, positions=positions, widths=width * 0.9, patch_artist=True,
                whis=(5, 95), showfliers=False, manage_ticks=False,
            )
            color = PALETTE[gi % len(PALETTE)]
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.6)
            if group_col:
                bp["boxes"][0].set_label(str(g))
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=45, ha="right", fontsize="small")
        ax.set_title(metric)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
        if group_col:
            ax.legend(fontsize="small", title=group_col)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finalize(fig, save_path)


def plot_cdf(
    series_dict: dict[str, np.ndarray],
    xlabel: str,
    save_path: Path | None = None,
) -> Figure:
    """Overlapping empirical CDF lines, one line per key in series_dict."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (name, series) in enumerate(series_dict.items()):
        v = np.asarray(series, dtype=float)
        v = np.sort(v[~np.isnan(v)])
        if v.size == 0:
            continue
        y = np.arange(1, v.size + 1) / v.size
        ax.plot(v, y, color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cumulative probability")
    ax.set_ylim(0, 1)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return _finalize(fig, save_path)


def plot_timeseries(
    time: np.ndarray,
    pred_dict: dict[str, np.ndarray],
    obs_dict: dict[str, np.ndarray],
    title: str | None = None,
    save_path: Path | None = None,
) -> Figure:
    """Predicted vs observed time series for one unit, one panel per target variable.

    time: 1-D array of timestamps/positions shared by all targets.
    Adapts to any number of targets (keys of pred_dict).
    """
    targets = list(pred_dict.keys())
    fig, axes = plt.subplots(len(targets), 1, figsize=(9, min(2.0 * len(targets), 10)), squeeze=False, sharex=True)
    for i, name in enumerate(targets):
        ax = axes[i, 0]
        ax.plot(time, np.asarray(obs_dict[name], dtype=float), color="black", linewidth=1.0, label="true")
        ax.plot(time, np.asarray(pred_dict[name], dtype=float), color=PALETTE[5], linewidth=1.0, label="pred")
        ax.set_ylabel(name, fontsize="small")
        if i == 0:
            ax.legend(fontsize="small", ncol=2)
    axes[-1, 0].set_xlabel("time")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finalize(fig, save_path)


def plot_spatial_map(
    data_2d: np.ndarray,
    title: str,
    save_path: Path | None = None,
) -> Figure:
    """2D heatmap of per-pixel metric values (e.g. NSE map over a grid).

    Used by the gridded Arctic domain; generic enough for any 2-D array.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.asarray(data_2d, dtype=float), origin="lower", aspect="auto", cmap="RdYlGn")
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return _finalize(fig, save_path)


def plot_data_split_map(
    records: list[dict],
    split: dict[tuple, str],
    train_subset: set[tuple] | None,
    title: str,
    save_path: Path | None = None,
) -> Figure:
    """Lat/lon scatter map showing train/val/test pixel assignments across all grids.

    Green = selected train pixels for this run; light grey = unselected train pool;
    orange = val; sky blue = test. Each pixel appears once even though two SSP records
    exist per pixel. Useful for verifying geographic coverage of each training size.
    """
    _colors = {
        "train_selected": "#009E73",  # Okabe-Ito bluish green
        "train_pool":     "#CCCCCC",  # light grey
        "val":            "#E69F00",  # Okabe-Ito orange
        "test":           "#56B4E9",  # Okabe-Ito sky blue
    }
    _labels = {
        "train_pool":     "train pool (not selected)",
        "val":            "val",
        "test":           "test",
        "train_selected": "train (selected)",
    }
    _sizes = {"train_pool": 2, "val": 4, "test": 4, "train_selected": 6}

    buckets: dict[str, tuple[list, list]] = {r: ([], []) for r in _colors}
    seen: set[tuple] = set()
    for rec in records:
        k = (rec["grid"], rec["y"], rec["x"])
        if k in seen:
            continue
        seen.add(k)
        s = split[k]
        if s == "train":
            role = "train_selected" if (train_subset is None or k in train_subset) else "train_pool"
        else:
            role = s
        buckets[role][0].append(rec["lon"])
        buckets[role][1].append(rec["lat"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for role in ("train_pool", "test", "val", "train_selected"):
        lons, lats = buckets[role]
        if not lons:
            continue
        ax.scatter(lons, lats, s=_sizes[role], color=_colors[role],
                   label=_labels[role], alpha=0.8, edgecolors="none")

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize="small", loc="lower right")
    fig.tight_layout()
    return _finalize(fig, save_path)
