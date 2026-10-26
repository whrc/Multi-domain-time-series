"""Domain-agnostic plotting helpers (Okabe-Ito palette, figures <= 10x10 in).

Each plot_* returns a Figure; pass save_path to save and close it.
"""

import math
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    """Train/val loss per epoch (val every ``eval_every``); optional second panel of per-target val loss."""
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
        ax2.set_title("Per-target validation loss")
        ax2.legend(fontsize="small")

    fig.tight_layout()
    return _finalize(fig, save_path)


def plot_pred_vs_true(
    pred_dict: dict[str, np.ndarray],
    obs_dict: dict[str, np.ndarray],
    log_scale: bool = False,
    save_path: Path | None = None,
) -> Figure:
    """Pred-vs-true scatter per target with 1:1 line and RMSE/NSE; log_scale keeps only positive values."""
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


def _format_median(value: float, metric: str) -> str:
    """Median label: NSE/KGE to 2 dp; other metrics with precision scaled to magnitude."""
    if metric in ("NSE", "KGE"):
        return f"{value:.2f}"
    magnitude = abs(value)
    if magnitude < 1:
        return f"{value:.2f}"
    if magnitude < 10:
        return f"{value:.1f}"
    return f"{value:.0f}"


def draw_metric_boxplot_panel(
    ax: plt.Axes,
    metrics_df: pd.DataFrame,
    metric: str,
    group_col: str | None = None,
    box_width_frac: float = 0.9,
    group_span: float = 0.8,
    zero_line: bool = True,
    show_median_labels: bool = True,
) -> None:
    """Draw one metric's boxplot onto ``ax``, boxes grouped by `target` and optionally split by group_col.

    An ordered Categorical group_col is drawn in category order, otherwise sorted. Whiskers
    use the 1.5x-IQR rule with outliers hidden. box_width_frac scales each box within its
    slot; group_span scales the width of each target's cluster. zero_line adds a dotted y=0
    line; show_median_labels prints each median above its whisker.
    """
    targets = sorted(metrics_df["target"].unique())
    if group_col:
        col = metrics_df[group_col]
        if isinstance(col.dtype, pd.CategoricalDtype) and col.dtype.ordered:
            present = set(col.unique())
            groups = [c for c in col.dtype.categories if c in present]
        else:
            groups = sorted(col.unique())
    else:
        groups = [None]
    width = group_span / len(groups)
    label_specs = []  # (x, whisker_top, median); drawn after all groups so offsets use the final y-range
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
            data, positions=positions, widths=width * box_width_frac, patch_artist=True,
            whis=1.5, showfliers=False, manage_ticks=False,
        )
        color = PALETTE[gi % len(PALETTE)]
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.6)
        if group_col:
            bp["boxes"][0].set_label(str(g))
        for i, vals in enumerate(data):
            if vals.size == 0:
                continue
            whisker_top = bp["caps"][2 * i + 1].get_ydata()[0]
            label_specs.append((positions[i], whisker_top, float(np.median(vals))))
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize="small")
    ax.set_title(metric)
    # Restore ylim after axhline so y=0 never widens the axis; add headroom for median labels.
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    if show_median_labels:
        for x, whisker_top, median_val in label_specs:
            ax.text(x, whisker_top + 0.02 * span, _format_median(median_val, metric),
                    ha="center", va="bottom", fontsize="x-small", color="sienna")
    if zero_line:
        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    headroom = 0.12 if show_median_labels else 0.0
    ax.set_ylim(ylim[0], ylim[1] + headroom * span)
    if group_col:
        ax.legend(fontsize="small", title=group_col)


def plot_metric_boxplot(
    metrics_df: pd.DataFrame,
    group_col: str | None = None,
    title: str | None = None,
    save_path: Path | None = None,
) -> Figure:
    """2x2 boxplots of RMSE/NSE/KGE/PBIAS across units (see draw_metric_boxplot_panel)."""
    targets = sorted(metrics_df["target"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(min(2.8 * len(targets) + 2, 10), 8), squeeze=False)
    for mi, metric in enumerate(_METRICS):
        draw_metric_boxplot_panel(axes[mi // 2, mi % 2], metrics_df, metric, group_col)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finalize(fig, save_path)


def plot_timeseries(
    time: np.ndarray,
    pred_dict: dict[str, np.ndarray],
    obs_dict: dict[str, np.ndarray],
    title: str | None = None,
    save_path: Path | None = None,
) -> Figure:
    """Predicted vs observed time series for one unit, one panel per target."""
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
    """2-D heatmap of per-pixel metric values."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.asarray(data_2d, dtype=float), origin="lower", aspect="auto", cmap="RdYlGn")
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return _finalize(fig, save_path)


_OCEAN_COLOR = "aliceblue"  # distinct from the test-split/river blue (#56B4E9)
_LABEL_BBOX = {"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1}


def _circumpolar_axes(
    figsize: tuple[float, float] = (10, 6), fig: Figure | None = None, subplot_spec=None,
    rect: list[float] | None = None, lat_labels: list[float] | None = None,
    lon_labels: list[float] | None = None, coastline_color: str = "black",
    coastline_linewidth: float = 0.6,
):
    """Figure + NorthPolarStereo GeoAxes down to 44N (data's southern edge is ~45.6N).

    Embed via `fig` + `subplot_spec`, or `rect` ([left, bottom, width, height] figure
    fraction). Gridliner `draw_labels=True` crashes on polar projections, so pass
    `lat_labels`/`lon_labels` (degrees) for hand-placed labels: latitudes along 180E,
    longitudes at 46N.
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if rect is not None:
        ax = fig.add_axes(rect, projection=ccrs.NorthPolarStereo())
    elif subplot_spec is None:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    else:
        ax = fig.add_subplot(subplot_spec, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 44, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor=_OCEAN_COLOR, zorder=0)
    ax.coastlines(resolution="110m", linewidth=coastline_linewidth, color=coastline_color)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", linestyle=":")
    for lat in lat_labels or []:
        ax.text(180, lat, f"{lat:g}°N", transform=ccrs.PlateCarree(), fontsize=5, ha="center",
               va="center", zorder=20, bbox=_LABEL_BBOX)
    for lon in lon_labels or []:
        label = f"{lon:g}°E" if lon >= 0 else f"{-lon:g}°W"
        ax.text(lon, 46, label, transform=ccrs.PlateCarree(), fontsize=5, ha="center", va="center",
               zorder=20, bbox=_LABEL_BBOX)
    return fig, ax


def plot_metric_scatter_map(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    title: str,
    save_path: Path | None = None,
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "NSE",
) -> Figure:
    """Circumpolar scatter of one metric value per site."""
    fig, ax = _circumpolar_axes()
    sc = ax.scatter(lons, lats, c=values, s=20, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors="none", transform=ccrs.PlateCarree())
    fig.colorbar(sc, ax=ax, shrink=0.85, label=cbar_label)
    ax.set_title(title)
    fig.tight_layout()
    return _finalize(fig, save_path)


def _regional_axes(
    extent: tuple[float, float, float, float],
    figsize: tuple[float, float] = (7, 7),
    fig: Figure | None = None,
    subplot_spec=None,
    draw_labels: bool | list[str] = True,
    rect: list[float] | None = None,
    gridline_padding: float | None = None,
    gridline_fontsize: float | None = None,
    coastline_color: str = "black",
    coastline_linewidth: float = 0.6,
    rivers: bool = True,
    gridline_step: float | None = None,
):
    """Figure + PlateCarree GeoAxes with coastlines/borders/rivers for a regional extent.

    Embed via `fig` + `subplot_spec`, or `rect` ([left, bottom, width, height] figure
    fraction). A negative `gridline_padding` (points) draws tick labels inside the frame;
    `gridline_step` (degrees) fixes gridline spacing.
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if rect is not None:
        ax = fig.add_axes(rect, projection=ccrs.PlateCarree())
    elif subplot_spec is None:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(subplot_spec, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor=_OCEAN_COLOR, zorder=0)
    ax.coastlines(resolution="50m", linewidth=coastline_linewidth, color=coastline_color)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", color="gray")
    if rivers:
        ax.add_feature(cfeature.RIVERS, linewidth=0.4, color="#56B4E9", alpha=0.6)
    gl = ax.gridlines(draw_labels=draw_labels, linewidth=0.3, color="gray", linestyle=":")
    if gridline_step is not None:
        gl.xlocator = mticker.MultipleLocator(gridline_step)
        gl.ylocator = mticker.MultipleLocator(gridline_step)
    if gridline_padding is not None:
        gl.xpadding = gridline_padding
        gl.ypadding = gridline_padding
        label_style = {"zorder": 20, "bbox": dict(_LABEL_BBOX)}
        if gridline_fontsize is not None:
            label_style["size"] = gridline_fontsize
        gl.xlabel_style = label_style
        gl.ylabel_style = label_style
    return fig, ax


def plot_site_split_map(
    lons: np.ndarray,
    lats: np.ndarray,
    split_labels: np.ndarray,
    title: str,
    save_path: Path | None = None,
    pad_deg: float = 2.0,
) -> Figure:
    """Regional map of sites colored by train/val/test split, extent padded by pad_deg around the data."""
    _colors = {"train": "#009E73", "val": "#E69F00", "test": "#56B4E9"}
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    labels = np.asarray(split_labels)
    extent = (lons.min() - pad_deg, lons.max() + pad_deg, lats.min() - pad_deg, lats.max() + pad_deg)

    fig, ax = _regional_axes(extent)
    for role in sorted(_colors, key=lambda r: (labels == r).sum(), reverse=True):
        mask = labels == role
        if not mask.any():
            continue
        ax.scatter(lons[mask], lats[mask], s=35, color=_colors[role], label=role,
                   edgecolors="black", linewidths=0.3, alpha=0.9, zorder=5, transform=ccrs.PlateCarree())
    ax.set_title(title)
    ax.legend(markerscale=1.2, fontsize="small", loc="best")
    fig.tight_layout()
    return _finalize(fig, save_path)


def plot_data_split_map(
    records: list[dict],
    split: dict[tuple, str],
    subsets: dict[str, set[tuple] | None],
    title: str,
    save_path: Path | None = None,
) -> Figure:
    """Circumpolar map of the train/val/test pixels used this run, one point per pixel.

    ``subsets[role]`` restricts a role to the selected pixels (None = whole split pool).
    """
    _colors = {
        "train": "#009E73",
        "val":   "#E69F00",
        "test":  "#56B4E9",
    }

    buckets: dict[str, tuple[list, list]] = {r: ([], []) for r in _colors}
    seen: set[tuple] = set()
    for rec in records:
        k = (rec["grid"], rec["y"], rec["x"])
        if k in seen:
            continue
        seen.add(k)
        s = split[k]
        subset = subsets.get(s)
        if subset is not None and k not in subset:
            continue
        buckets[s][0].append(rec["lon"])
        buckets[s][1].append(rec["lat"])

    fig, ax = _circumpolar_axes()
    for role in sorted(buckets, key=lambda r: len(buckets[r][0]), reverse=True):
        lons, lats = buckets[role]
        if not lons:
            continue
        ax.scatter(lons, lats, s=20, color=_colors[role],
                   label=role, alpha=0.9, edgecolors="none", transform=ccrs.PlateCarree())

    ax.set_title(title)
    ax.legend(markerscale=2, fontsize="small", loc="lower left")
    fig.tight_layout()
    return _finalize(fig, save_path)
