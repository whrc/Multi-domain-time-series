"""
Figure 7 — test-site change maps, individual vs. multi-domain fine-tuned (fig7a RMSE, fig7b KGE).
One row per domain, one map per flux-only target; each dot is a test site/pixel colored by its
seedavg change: %-change for RMSE, absolute delta for KGE (%-change explodes near KGE=0).
Arctic change is averaged over ssp x period per pixel. Each domain row has its own colorbar.
"""

import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from config.config import load_config  # noqa: E402
from shared.plots import _circumpolar_axes, _regional_axes  # noqa: E402
from make_figure1 import (  # noqa: E402
    _draw_basins, _extent, _load_module, REGIONAL_TARGET_ASPECT, amazon_basins,
)
from _common import (  # noqa: E402
    AMAZON_TEST, ARCTIC_FLUXONLY_TEST, MD_FINETUNED_SEEDAVG, RANGELAND_FLUXONLY_TEST,
    _jitter_coincident_sites, _load_seedavg, _save, _style,
)

ARCTIC_TARGETS = ["GPP", "RECO"]
AMAZON_TARGETS = ["discharge", "active_fire_count", "burned_area"]
RANGELAND_TARGETS = ["GPP", "RECO", "Rm", "Rg"]
DOMAIN_TARGETS = {"arctic": ARCTIC_TARGETS, "amazon": AMAZON_TARGETS, "rangeland": RANGELAND_TARGETS}
ROW_LETTER = {"arctic": "(a)", "amazon": "(b)", "rangeland": "(c)"}
# Single-line labels (_common.py's version wraps "Active fire\ncount").
AMAZON_TARGET_LABELS = {"active_fire_count": "Active fire count", "burned_area": "Burned area",
                        "discharge": "Discharge"}

CMAP = {"RMSE": "PRGn_r", "KGE": "PRGn"}  # colorblind-safe; green = fine-tuning helped
PCT_METRICS = {"RMSE"}
DELTA_METRICS = {"KGE"}
METRIC_FILE_SUFFIX = {"RMSE": "a", "KGE": "b"}
CLIP_PERCENTILE = 90  # per-domain colorbar range = this percentile of |change| (Arctic's spread ~10x the others)

# Layout (inches)
MARGIN = 0.1
GAP_COLS = 0.06
GAP_ROWS = 0.12
H_ROW = {"arctic": 1.3, "amazon": 1.0, "rangeland": 1.0}
ROW_LABEL_W = 0.22  # strip left of the content area for row labels
ROW_LABEL_GAP = 0.12  # row label to the row's first panel
CBAR_W = 0.08
# CBAR_GAP / CBAR_TITLE_PAD are sized for the colorbar title, which is wider than the bar.
CBAR_GAP = 0.25
CBAR_TITLE_PAD = 0.2
CBAR_H_FRAC = 0.7  # colorbar height / row height
CBAR_TITLE_HEADROOM = 0.22  # space above the colorbar for its 2-line title
ARCTIC_DOT_S = 6    # ~300+ overlapping test pixels
OTHER_DOT_S = 16    # <20 test sites per domain
DOT_EDGE = {"arctic": "none", "amazon": "dimgrey", "rangeland": "dimgrey"}  # edges would smear Arctic's dense dots
DOT_EDGE_LW = 0.35
REGIONAL_PAD = 1.0  # degrees
AMAZON_LABEL_HEADROOM_DEG = 2.0  # northernmost gauges sit right under the in-panel label
COASTLINE_COLOR = "grey"
COASTLINE_LINEWIDTH = 0.35


def _change(individual: pd.Series, multi: pd.Series, metric: str) -> pd.Series:
    """%-change for RMSE, absolute delta for KGE."""
    if metric in DELTA_METRICS:
        return multi - individual
    return (multi - individual) / individual.replace(0, np.nan) * 100


def arctic_change(metric: str) -> pd.DataFrame:
    individual = _load_seedavg(ARCTIC_FLUXONLY_TEST)
    individual = individual[~individual["obs_degenerate"] & individual["target"].isin(ARCTIC_TARGETS)]
    multi = _load_seedavg(MD_FINETUNED_SEEDAVG / "arctic" / "arctic_metrics_seedavg.csv")
    multi = multi[multi["target"].isin(ARCTIC_TARGETS)]
    keys = ["grid", "y", "x", "lat", "lon", "ssp", "period", "target"]
    merged = individual.merge(multi, on=keys, suffixes=("_individual", "_multi"))
    merged["change"] = _change(merged[f"{metric}_individual"], merged[f"{metric}_multi"], metric)
    merged = merged.dropna(subset=["change"])
    # Mean change over ssp x period per pixel/target.
    return merged.groupby(["grid", "y", "x", "lat", "lon", "target"], as_index=False)["change"].mean()


def amazon_change(metric: str) -> pd.DataFrame:
    individual = _load_seedavg(AMAZON_TEST)
    individual = individual[individual["target"].isin(AMAZON_TARGETS)]
    multi = _load_seedavg(MD_FINETUNED_SEEDAVG / "amazon" / "amazon_metrics_seedavg.csv")
    multi = multi[multi["target"].isin(AMAZON_TARGETS)]
    merged = individual.merge(multi, on=["station_id", "target"], suffixes=("_individual", "_multi"))
    merged["change"] = _change(merged[f"{metric}_individual"], merged[f"{metric}_multi"], metric)
    merged = merged.dropna(subset=["change"])

    ev04 = _load_module("amazon_domain", "04_evaluate.py", "_am04_fig7")
    coords = ev04.load_station_coords(load_config("amazon_domain"))
    merged["station_id"] = merged["station_id"].astype(str)
    merged = merged.merge(coords, on="station_id", how="left").dropna(subset=["lat", "lon"])
    return merged[["station_id", "lat", "lon", "target", "change"]]


def rangeland_change(metric: str) -> pd.DataFrame:
    individual = _load_seedavg(RANGELAND_FLUXONLY_TEST)
    individual["target"] = individual["target"].str.replace("_predicted", "", regex=False)
    individual = individual[individual["target"].isin(RANGELAND_TARGETS)]
    multi = _load_seedavg(MD_FINETUNED_SEEDAVG / "rangeland" / "rangeland_metrics_seedavg.csv")
    multi = multi[multi["target"].isin(RANGELAND_TARGETS)]
    merged = individual.merge(multi, on=["site", "target"], suffixes=("_individual", "_multi"))
    merged["change"] = _change(merged[f"{metric}_individual"], merged[f"{metric}_multi"], metric)
    merged = merged.dropna(subset=["change"])

    ev04 = _load_module("rangeland_domain", "04_evaluate.py", "_rl04_fig7")
    coords = ev04.load_site_coords(load_config("rangeland_domain"))
    coords = coords[coords["site"].isin(merged["site"])]  # jitter only among plotted sites
    coords = _jitter_coincident_sites(coords)
    merged = merged.merge(coords, on="site", how="left").dropna(subset=["lat", "lon"])
    return merged[["lat", "lon", "target", "change"]]


def _rect(fig_w: float, fig_h: float, left: float, top: float, width: float, height: float) -> list[float]:
    """Inches with top measured from the figure top -> fig.add_axes figure-fraction rect."""
    return [left / fig_w, (fig_h - top - height) / fig_h, width / fig_w, height / fig_h]


def _target_label(domain: str, target: str) -> str:
    return AMAZON_TARGET_LABELS[target] if domain == "amazon" else target


def make_figure(metric: str) -> None:
    domain_data = {
        "arctic": arctic_change(metric),
        "amazon": amazon_change(metric),
        "rangeland": rangeland_change(metric),
    }

    is_pct = metric in PCT_METRICS
    vmax = {}
    for domain, df in domain_data.items():
        vmax[domain] = np.percentile(np.abs(df["change"].dropna().to_numpy()), CLIP_PERCENTILE)
        df["change"] = df["change"].clip(-vmax[domain], vmax[domain])
        unit = "%" if is_pct else ""
        precision = ".1f" if is_pct else ".3f"
        print(f"{metric} {domain}: {len(df)} rows across {df['target'].nunique()} targets, "
              f"vmax=+-{vmax[domain]:{precision}}{unit}")
    cmap = CMAP[metric]

    basins = amazon_basins(domain_data["amazon"]["station_id"].drop_duplicates())
    extents = {"arctic": (-180, 180, 44, 90)}
    aspects = {"arctic": 1.0}
    for domain in ("amazon", "rangeland"):
        geoms = list(basins.values()) if domain == "amazon" else ()
        ext = _extent(domain_data[domain], pad=REGIONAL_PAD, target_aspect=REGIONAL_TARGET_ASPECT,
                      geoms=geoms)
        if domain == "amazon":
            ext = (ext[0], ext[1], ext[2], ext[3] + AMAZON_LABEL_HEADROOM_DEG)
        extents[domain] = ext
        aspects[domain] = (ext[1] - ext[0]) / (ext[3] - ext[2])

    panel_w = {d: H_ROW[d] * a for d, a in aspects.items()}
    row_w = {d: len(DOMAIN_TARGETS[d]) * panel_w[d] + (len(DOMAIN_TARGETS[d]) - 1) * GAP_COLS for d in aspects}
    row_total_w = {d: row_w[d] + CBAR_GAP + CBAR_W for d in aspects}
    content_w = max(row_total_w.values())

    fig_w = MARGIN + ROW_LABEL_W + content_w + CBAR_TITLE_PAD + MARGIN
    fig_h = MARGIN + sum(H_ROW.values()) + 2 * GAP_ROWS + MARGIN
    fig = plt.figure(figsize=(fig_w, fig_h))

    cursor = MARGIN
    for domain in ("arctic", "amazon", "rangeland"):
        h_row = H_ROW[domain]
        row_left = MARGIN + ROW_LABEL_W + (content_w - row_total_w[domain]) / 2
        left = row_left
        dot_s = ARCTIC_DOT_S if domain == "arctic" else OTHER_DOT_S
        mappable = None
        for target in DOMAIN_TARGETS[domain]:
            rect = _rect(fig_w, fig_h, left, cursor, panel_w[domain], h_row)
            if domain == "arctic":
                _, ax = _circumpolar_axes(fig=fig, rect=rect, coastline_color=COASTLINE_COLOR,
                                          coastline_linewidth=COASTLINE_LINEWIDTH)
            else:
                _, ax = _regional_axes(extents[domain], fig=fig, rect=rect, draw_labels=False,
                                       coastline_color=COASTLINE_COLOR, coastline_linewidth=COASTLINE_LINEWIDTH,
                                       rivers=False)
            sub = domain_data[domain][domain_data[domain]["target"] == target]
            if domain == "amazon":
                _draw_basins(ax, {sid: basins[sid] for sid in sub["station_id"]})
            mappable = ax.scatter(sub["lon"], sub["lat"], c=sub["change"], cmap=cmap,
                                  vmin=-vmax[domain], vmax=vmax[domain], s=dot_s,
                                  edgecolors=DOT_EDGE[domain], linewidths=DOT_EDGE_LW,
                                  transform=ccrs.PlateCarree(), zorder=5)
            ax.text(0.5, 0.96, _target_label(domain, target), transform=ax.transAxes, fontsize=7,
                    ha="center", va="top", zorder=20,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5})
            left += panel_w[domain] + GAP_COLS

        # fig.text, not ax.set_ylabel: GeoAxes y-labels get dropped by bbox_inches="tight".
        row_label_x = (row_left - ROW_LABEL_GAP) / fig_w
        row_label_y = (fig_h - cursor - h_row / 2) / fig_h
        fig.text(row_label_x, row_label_y, f"{ROW_LETTER[domain]} {domain.capitalize()}", fontsize=8,
                fontweight="bold", rotation=90, ha="center", va="center")

        # Top-anchored (not centered) so the title always has CBAR_TITLE_HEADROOM above it.
        cbar_h = CBAR_H_FRAC * h_row
        cbar_top = cursor + CBAR_TITLE_HEADROOM
        cbar_left = row_left + row_w[domain] + CBAR_GAP
        cax = fig.add_axes(_rect(fig_w, fig_h, cbar_left, cbar_top, CBAR_W, cbar_h))
        cbar = fig.colorbar(mappable, cax=cax)
        title = f"% change\nin {metric}" if is_pct else f"$\\Delta${metric}"
        cbar.ax.set_title(title, fontsize=6, pad=3)

        cursor += h_row + GAP_ROWS

    suffix = METRIC_FILE_SUFFIX[metric]
    _save(fig, f"fig7{suffix}_map_individual_vs_finetuned_{metric.lower()}.png")


def main() -> None:
    _style()
    for metric in ("RMSE", "KGE"):
        make_figure(metric)


if __name__ == "__main__":
    main()
