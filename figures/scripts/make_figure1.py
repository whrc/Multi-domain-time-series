"""
Figure 1 — study site overview across all three domains (Arctic, Amazon, Rangeland).

Panel (a): global locator (Robinson projection) showing where each domain sits,
domain-colored, no split detail. Panels (b)/(c)/(d): one zoomed panel per domain, each
on its own natural projection, showing the actual train/val/test site split.

Site coordinates + splits are recomputed from each domain's own existing pipeline code
(dynamically imported, same pattern as domains/arctic_domain/_fetch_grid_worker.py uses
for 01_preprocess.py) rather than duplicated here:
- Arctic: grid centroids from the local .grid_pass1_summary_cache + assign_grid_splits
  (deterministic given the same config seed/fractions as the production run).
- Amazon: load_station_coords (small GCS GeoPackage attribute read) + station_split_table.
- Rangeland: load_site_coords (RangeSTAR_data/ameriflux_sites.geojson) + site_split_table.

Layout is computed manually (not GridSpec) so every panel's box is sized to its own
true data aspect ratio -- geographic projections always preserve true aspect, so an
equal-width GridSpec column leaves large aspect-driven blank margins around whichever
panel doesn't match that column's shape. Matching each box to its content's aspect
keeps the figure tight with near-zero forced whitespace.
"""

import importlib.util
import pickle
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config.config import load_config  # noqa: E402
from shared.plots import PALETTE, _OCEAN_COLOR, _circumpolar_axes, _regional_axes  # noqa: E402

FIGURES_DIR = REPO_ROOT / "figures"
SVG_DIR = FIGURES_DIR / "svg"
DPI = 300

# PALETTE[0:3] are reserved for grouped sub-categories elsewhere (see
# figures/scripts/_common.py's DOMAIN_COLOR) -- reuse the same mapping here
# so domain colors are consistent across every figure in the paper.
DOMAIN_COLOR = {"arctic": PALETTE[6], "amazon": PALETTE[5], "rangeland": PALETTE[4]}
SPLIT_COLOR = {"train": "#009E73", "val": "#E69F00", "test": "#56B4E9"}

# ── Layout constants (inches) ──────────────────────────────────────────────────────────
FIG_W = 10.0
MARGIN_L = MARGIN_R = 0.08
MARGIN_TOP = 0.08
MARGIN_BOTTOM = 0.06
TITLE_H = 0.22
GAP_ROWS = 0.04
GAP_COLS = 0.12
H_A = 1.45  # global locator map height (excludes its title band)
H_ROW = 2.3  # detail-panel row height (chosen directly, not solved from full row width --
             # see REGIONAL_TARGET_ASPECT below for why panel widths no longer need to sum
             # to the full available width)
REGIONAL_TARGET_ASPECT = 1.15  # Amazon/Rangeland's raw data aspect (lon:lat span) is much
    # wider than Arctic's fixed 1:1 circumpolar cap (1.58/1.95 vs 1.0), which made those two
    # panels dominate the row's width. _extent() pads their *latitude* range (never crops
    # longitude, so no site/context is lost) until their aspect matches this target, so all
    # three detail panels sit closer in width. The row is then centered rather than forced to
    # span the full figure width, since the narrowed panels no longer need to.
GRIDLINE_PADDING = -3  # pulls (c)/(d)'s lat/lon tick labels inside the frame, right up against
                        # the border rather than deep inside (see below)
GRIDLINE_FONTSIZE = 5  # (c)/(d)'s lat/lon tick labels, smaller than the rest of the figure's text

LEGEND_EDGE_COLOR = "grey"


def _load_module(domain_dir: str, filename: str, alias: str):
    """Import a numbered domain script (e.g. 01_preprocess.py) by file path -- these
    filenames aren't valid module names, so a normal `import` can't reach them."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "domains" / domain_dir / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    })


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    SVG_DIR.mkdir(exist_ok=True)
    png_path = FIGURES_DIR / f"{name}.png"
    svg_path = SVG_DIR / f"{name}.svg"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def arctic_sites() -> pd.DataFrame:
    """Grid centroids + train/val/test split, recomputed offline from the local pass-1
    summary cache -- same seed/fractions as the production run, no GCS/re-fetch needed."""
    pp01 = _load_module("arctic_domain", "01_preprocess.py", "_pp01_fig1")
    cfg = load_config("arctic_domain")
    cache_dir = Path(cfg["paths"]["preprocessed_dir"]) / ".grid_pass1_summary_cache"
    centroids = {}
    for path in sorted(cache_dir.glob("*.pkl")):
        with path.open("rb") as f:
            data = pickle.load(f)
        lat_lon = data["value"]["lat_lon"]
        if not lat_lon:  # all-ocean grid, no land pixels to center on
            continue
        centroids[path.stem] = pp01._grid_centroid(lat_lon)
    split = pp01.assign_grid_splits(centroids, cfg["preprocessing"])
    rows = [{"lon": centroids[g][1], "lat": centroids[g][0], "split": split[g]} for g in centroids]
    return pd.DataFrame(rows)


def amazon_sites() -> pd.DataFrame:
    """Station coords + train/val/test split, recomputed from GCS in-memory (no local
    outputs/amazon_domain/preprocessed/*.pkl on this machine -- that pipeline runs on the
    VM). split_stations() needs the same filtered station_id order load_filtered()
    produces, so both are re-derived here rather than reusing a stale local split."""
    pp01 = _load_module("amazon_domain", "01_preprocess.py", "_am01_fig1")
    ev04 = _load_module("amazon_domain", "04_evaluate.py", "_am04_fig1")
    cfg = load_config("amazon_domain")
    split = pp01.split_stations(pp01.load_filtered(cfg), cfg)
    coords = ev04.load_station_coords(cfg)
    coords["split"] = coords["station_id"].map(split)
    coords = coords.dropna(subset=["lat", "lon", "split"])
    return coords[["lon", "lat", "split"]]


def rangeland_sites() -> pd.DataFrame:
    ev04 = _load_module("rangeland_domain", "04_evaluate.py", "_rl04_fig1")
    cfg = load_config("rangeland_domain")
    coords = ev04.load_site_coords(cfg)
    splits = ev04.site_split_table(Path(cfg["paths"]["preprocessed_dir"]))
    df = splits.merge(coords, on="site", how="left").dropna(subset=["lat", "lon"])
    return df[["lon", "lat", "split"]]


def _extent(
    df: pd.DataFrame, pad: float = 2.0, target_aspect: float | None = None,
) -> tuple[float, float, float, float]:
    """Bounding box around `df`'s sites, padded by `pad` degrees on every side. If
    `target_aspect` is given and the box is currently wider than that (lon-range /
    lat-range), the *latitude* range is expanded (never the longitude range, so no site
    or context is cropped) until the box's aspect matches it -- used to narrow Amazon/
    Rangeland's panels toward Arctic's without losing any displayed extent."""
    lon_min, lon_max = df["lon"].min() - pad, df["lon"].max() + pad
    lat_min, lat_max = df["lat"].min() - pad, df["lat"].max() + pad
    if target_aspect is not None:
        needed_lat_range = (lon_max - lon_min) / target_aspect
        extra = max(0.0, needed_lat_range - (lat_max - lat_min)) / 2
        lat_min, lat_max = lat_min - extra, lat_max + extra
    return (lon_min, lon_max, lat_min, lat_max)


def _place_labels(
    extent: tuple[float, float, float, float], max_labels: int, min_sep_deg: float, inset_frac: float = 0.15,
) -> list[tuple[str, float, float]]:
    """Up to `max_labels` well-known place names, picked by Natural Earth population rank
    and greedily spaced >= min_sep_deg apart so labels stay sparse and don't clump.
    Candidates are searched within `extent` shrunk by `inset_frac` on each side (not the
    full extent) so a picked point always has room for its label text to land fully
    inside the panel once _draw_labels offsets it toward the panel's center.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    lon_pad, lat_pad = inset_frac * (lon_max - lon_min), inset_frac * (lat_max - lat_min)
    lon_min, lon_max = lon_min + lon_pad, lon_max - lon_pad
    lat_min, lat_max = lat_min + lat_pad, lat_max - lat_pad

    path = shpreader.natural_earth(resolution="50m", category="cultural", name="populated_places")
    candidates = []
    for rec in shpreader.Reader(path).records():
        a = rec.attributes
        lon, lat = a["LONGITUDE"], a["LATITUDE"]
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            candidates.append((a["SCALERANK"], -a["POP_MAX"], a["NAME"], lon, lat))
    candidates.sort()

    picked: list[tuple[str, float, float]] = []
    for _, _, name, lon, lat in candidates:
        if all(((lon - plon) ** 2 + (lat - plat) ** 2) ** 0.5 >= min_sep_deg for _, plon, plat in picked):
            picked.append((name, lon, lat))
        if len(picked) >= max_labels:
            break
    return picked


def _draw_labels(
    ax: plt.Axes, labels: list[tuple[str, float, float]], extent: tuple[float, float, float, float],
    offset_deg: float,
) -> None:
    """Marker + label text, offset toward the panel's center (not a fixed direction) so
    the text always lands further inside the box rather than risking running past the
    edge for a point near the boundary."""
    lon_min, lon_max, lat_min, lat_max = extent
    lon_mid, lat_mid = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
    for name, lon, lat in labels:
        ha = "left" if lon < lon_mid else "right"
        va = "bottom" if lat < lat_mid else "top"
        dx = offset_deg if ha == "left" else -offset_deg
        dy = offset_deg if va == "bottom" else -offset_deg
        ax.plot(lon, lat, marker="+", color="black", markersize=4, transform=ccrs.PlateCarree(), zorder=6)
        ax.text(lon + dx, lat + dy, name, transform=ccrs.PlateCarree(), fontsize=6, ha=ha, va=va, zorder=6,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.5})


def _scatter_by_split(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot largest-bucket split first so smaller buckets (usually val/test) draw on top
    and aren't hidden -- same ordering convention as shared.plots.plot_site_split_map."""
    for role in sorted(SPLIT_COLOR, key=lambda r: (df["split"] == r).sum(), reverse=True):
        mask = df["split"] == role
        if not mask.any():
            continue
        ax.scatter(df.loc[mask, "lon"], df.loc[mask, "lat"], s=18, color=SPLIT_COLOR[role],
                  edgecolors="black", linewidths=0.3, alpha=0.9, zorder=5, transform=ccrs.PlateCarree())


def _rect(fig_h: float, left: float, top: float, width: float, height: float) -> list[float]:
    """[left, top, width, height] in inches (top measured from the figure's top edge) ->
    [left, bottom, width, height] in figure-fraction, as fig.add_axes expects."""
    return [left / FIG_W, (fig_h - top - height) / fig_h, width / FIG_W, height / fig_h]


def main() -> None:
    _style()
    arctic = arctic_sites()
    amazon = amazon_sites()
    rangeland = rangeland_sites()
    print(f"Arctic grids: {len(arctic)}, Amazon stations: {len(amazon)}, Rangeland sites: {len(rangeland)}")

    # Amazon/Rangeland's raw aspect (lon-range / lat-range) is much wider than Arctic's fixed
    # 1:1 circumpolar cap -- narrow them toward it (padding latitude only, so no site/context
    # is cropped) before computing panel widths.
    amazon_extent = _extent(amazon, pad=5.0, target_aspect=REGIONAL_TARGET_ASPECT)
    rangeland_extent = _extent(rangeland, pad=5.0, target_aspect=REGIONAL_TARGET_ASPECT)

    # Each panel's box is sized to its own aspect ratio (width/height in the plotted
    # projection) so the map fills its box edge-to-edge -- no aspect-driven blank margin.
    robinson = ccrs.Robinson()
    robinson_aspect = (robinson.x_limits[1] - robinson.x_limits[0]) / (robinson.y_limits[1] - robinson.y_limits[0])
    arctic_aspect = 1.0  # a polar-centered circumpolar cap is always a circle in a square box
    amazon_aspect = (amazon_extent[1] - amazon_extent[0]) / (amazon_extent[3] - amazon_extent[2])
    rangeland_aspect = (rangeland_extent[1] - rangeland_extent[0]) / (rangeland_extent[3] - rangeland_extent[2])

    w_a = H_A * robinson_aspect
    left_a = (FIG_W - w_a) / 2

    widths = [H_ROW * a for a in (arctic_aspect, amazon_aspect, rangeland_aspect)]
    row_w = sum(widths) + 2 * GAP_COLS
    left0 = (FIG_W - row_w) / 2  # center the row -- narrowed panels no longer need the full width

    fig_h = MARGIN_TOP + TITLE_H + H_A + GAP_ROWS + TITLE_H + H_ROW + MARGIN_BOTTOM
    fig = plt.figure(figsize=(FIG_W, fig_h))

    cursor = MARGIN_TOP + TITLE_H
    ax_a = fig.add_axes(_rect(fig_h, left_a, cursor, w_a, H_A), projection=ccrs.Robinson())
    ax_a.set_global()
    ax_a.add_feature(cfeature.OCEAN, facecolor=_OCEAN_COLOR, zorder=0)
    ax_a.coastlines(resolution="110m", linewidth=0.5, color="black")
    for name, df in (("Arctic", arctic), ("Amazon", amazon), ("Rangeland", rangeland)):
        ax_a.scatter(df["lon"], df["lat"], s=6, color=DOMAIN_COLOR[name.lower()], alpha=0.7,
                    edgecolors="none", transform=ccrs.PlateCarree(), label=name)
    # Boxed, horizontal domain legend centered on the globe (lon=0) and shifted south into
    # open ocean, below every domain's sites -- real lon/lat anchor via the PlateCarree data
    # transform, not a guessed axes-fraction position, so it can't drift off the map center.
    ax_a.legend(loc="center", bbox_to_anchor=(0, -52), bbox_transform=ccrs.PlateCarree()._as_mpl_transform(ax_a),
               ncol=3, frameon=True, fancybox=False, edgecolor=LEGEND_EDGE_COLOR, facecolor="white",
               framealpha=0.95, handletextpad=0.3, columnspacing=0.8, borderpad=0.4, markerscale=1.4)
    ax_a.set_title("(a) Global overview", pad=3)

    cursor += H_A + GAP_ROWS + TITLE_H
    left = left0
    axes_bcd = []
    for width, extent_fn, kind in zip(
        widths,
        [None, amazon_extent, rangeland_extent],
        ["circumpolar", "regional", "regional"],
    ):
        rect = _rect(fig_h, left, cursor, width, H_ROW)
        if kind == "circumpolar":
            _, ax = _circumpolar_axes(fig=fig, rect=rect, lat_labels=[50, 60, 70, 80],
                                      lon_labels=[0, 90, -90, 180])
        else:
            _, ax = _regional_axes(extent_fn, fig=fig, rect=rect, draw_labels=["left", "bottom"],
                                   gridline_padding=GRIDLINE_PADDING, gridline_fontsize=GRIDLINE_FONTSIZE)
        axes_bcd.append(ax)
        left += width + GAP_COLS
    ax_b, ax_c, ax_d = axes_bcd

    _scatter_by_split(ax_b, arctic)
    _scatter_by_split(ax_c, amazon)
    _scatter_by_split(ax_d, rangeland)

    _draw_labels(ax_b, _place_labels((-180, 180, 44, 90), max_labels=4, min_sep_deg=30), (-180, 180, 44, 90), 3.0)
    _draw_labels(ax_c, _place_labels(amazon_extent, max_labels=3, min_sep_deg=6), amazon_extent, 0.6)
    _draw_labels(ax_d, _place_labels(rangeland_extent, max_labels=3, min_sep_deg=8), rangeland_extent, 0.8)

    ax_b.set_title("(b) Arctic", pad=3)
    ax_c.set_title("(c) Amazon", pad=3)
    ax_d.set_title("(d) Rangeland", pad=3)

    split_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=c, markeredgecolor="black",
              markeredgewidth=0.4, label=role.capitalize())
        for role, c in SPLIT_COLOR.items()
    ]
    # Boxed, inside the Amazon panel (upper center) rather than a separate reserved strip
    # below all three panels -- one shared legend still covers all three domains' splits
    # since the same three colors are used everywhere.
    ax_c.legend(handles=split_handles, loc="upper center", ncol=3, frameon=True, fancybox=False,
               edgecolor=LEGEND_EDGE_COLOR, facecolor="white", framealpha=0.9, handletextpad=0.3,
               columnspacing=1.0, borderpad=0.4, fontsize=6)

    _save(fig, "fig1_study_sites")


if __name__ == "__main__":
    main()
