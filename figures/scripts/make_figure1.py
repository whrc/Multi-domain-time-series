"""
Figure 1 — study site overview across all three domains (Arctic, Amazon, Rangeland).

Panel (a): global locator (Robinson projection) showing where each domain sits,
domain-colored, no split detail. Panels (b)/(c)/(d): one zoomed panel per domain, each
on its own natural projection, showing the actual train/val/test site split; (c) also
outlines each gauge's drainage basin (polygons from the same GeoPackage as its coordinates).

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
import sqlite3
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import shapely.wkb
from matplotlib.lines import Line2D
from shapely.geometry.base import BaseGeometry

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config.config import load_config  # noqa: E402
from shared.io import gcs_filesystem  # noqa: E402
from shared.plots import PALETTE, _OCEAN_COLOR, _circumpolar_axes, _regional_axes  # noqa: E402
from _common import _jitter_coincident_sites  # noqa: E402  (figures/scripts/ is on sys.path as the script's dir)

FIGURES_DIR = REPO_ROOT / "figures"
SVG_DIR = FIGURES_DIR / "svg"
DPI = 300

# PALETTE[0:3] are reserved for grouped sub-categories elsewhere (see
# figures/scripts/_common.py's DOMAIN_COLOR) -- reuse the same mapping here
# so domain colors are consistent across every figure in the paper.
DOMAIN_COLOR = {"arctic": PALETTE[6], "amazon": PALETTE[5], "rangeland": PALETTE[4]}
SPLIT_COLOR = {"train": "#009E73", "val": "#E69F00", "test": "#56B4E9"}

# ── Layout constants (inches) ──────────────────────────────────────────────────────────
# 2x2 layout: row 1 = (a) global locator | (b) Arctic, row 2 = (c) Amazon | (d) Rangeland.
MARGIN_L = MARGIN_R = 0.08
MARGIN_TOP = 0.08
MARGIN_BOTTOM = 0.06
TITLE_H = 0.22  # (a)'s title band above its map -- (b)/(c)/(d) carry their titles inside
LEGEND_H = 0.25  # (a)'s domain-legend band below its map -- kept off the map so it covers no land
GAP_ROWS = 0.2
GAP_COLS = 0.22
H_A = 1.45  # global locator map height (excludes its title band)
H_ROW = 2.3  # detail-panel height; every panel's width follows from its own aspect
REGIONAL_TARGET_ASPECT = 1.15  # Amazon/Rangeland's raw data aspect (lon:lat span) is much
    # wider than Arctic's fixed 1:1 circumpolar cap (1.58/1.95 vs 1.0). _extent() pads their
    # *latitude* range (never crops longitude, so no site/context is lost) until their aspect
    # matches this target, keeping row 2 close to row 1's width.
REGIONAL_PAD_DEG = 2.0  # (c)/(d) margin around the data before the aspect padding -- enough to
                        # keep the nearest coastline in view without adding empty map
REGIONAL_LABEL_INSET = 0.08  # (c)/(d) city-label search inset (see _place_labels) -- the tight
                             # extents need less than the default to keep e.g. Manaus/Chicago
INSIDE_TITLE_INSET = 0.03  # (b)/(c)/(d) in-panel title offset from the frame, axes fraction
GRIDLINE_PADDING = -3  # pulls (c)/(d)'s lat/lon tick labels inside the frame, right up against
                        # the border rather than deep inside (see below)
GRIDLINE_FONTSIZE = 5  # (c)/(d)'s lat/lon tick labels, smaller than the rest of the figure's text
GRIDLINE_STEP_DEG = 10  # (c)/(d)'s gridline spacing -- a few labels for orientation, not a grid to read

LEGEND_EDGE_COLOR = "grey"

# Amazon drainage-basin outlines. Basins are heavily nested (downstream gauges contain upstream
# ones), so outlines only -- stacked fills would darken toward the outlet and bury the gauge dots.
BASIN_SIMPLIFY_DEG = 0.01  # ~1 km; drops ~777k raw vertices to a size the SVG can carry
BASIN_EDGE_COLOR = "#2F5D8A"  # dark steel blue -- distinct from the gray borders/gridlines and
                              # from every split color, and reads as hydrology
BASIN_EDGE_WIDTH = 0.3
BASIN_EDGE_ALPHA = 0.8
# GeoPackage geometry blob = 8-byte header + optional envelope + WKB; the envelope's size is
# set by bits 1-3 of the header's flags byte (OGC GeoPackage spec, section 2.1.3).
GPKG_HEADER_BYTES = 8
GPKG_FLAGS_BYTE = 3
GPKG_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


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
    return coords[["station_id", "lon", "lat", "split"]]


def _gpkg_geometry(blob: bytes) -> BaseGeometry:
    envelope_code = (blob[GPKG_FLAGS_BYTE] >> 1) & 0b111
    if envelope_code not in GPKG_ENVELOPE_BYTES:
        raise ValueError(f"Invalid GeoPackage envelope code {envelope_code}")
    return shapely.wkb.loads(bytes(blob[GPKG_HEADER_BYTES + GPKG_ENVELOPE_BYTES[envelope_code]:]))


def amazon_basins(station_ids: pd.Series) -> dict[str, BaseGeometry]:
    """station_id -> simplified drainage-basin polygon (EPSG:4326), from the same GeoPackage
    load_station_coords reads attributes from -- parsed in memory via sqlite3, as there."""
    cfg = load_config("amazon_domain")
    gcs = cfg["gcs"]
    con = sqlite3.connect(":memory:")
    con.deserialize(gcs_filesystem().cat(f"{gcs['bucket']}/{gcs['files']['station_coords']}"))
    table = con.execute("SELECT table_name FROM gpkg_contents LIMIT 1").fetchone()[0]
    rows = con.execute(f'SELECT "EstacCd", "geom" FROM "{table}"').fetchall()
    con.close()
    basins = {str(int(sid)): _gpkg_geometry(blob) for sid, blob in rows}
    missing = set(station_ids) - set(basins)
    if missing:
        raise ValueError(f"No drainage basin for Amazon stations: {sorted(missing)}")
    return {sid: basins[sid].simplify(BASIN_SIMPLIFY_DEG) for sid in station_ids}


def rangeland_sites() -> pd.DataFrame:
    ev04 = _load_module("rangeland_domain", "04_evaluate.py", "_rl04_fig1")
    cfg = load_config("rangeland_domain")
    coords = ev04.load_site_coords(cfg)
    splits = ev04.site_split_table(Path(cfg["paths"]["preprocessed_dir"]))
    df = splits.merge(coords, on="site", how="left").dropna(subset=["lat", "lon"])
    return df[["lon", "lat", "split"]]


def _extent(
    df: pd.DataFrame, pad: float = 2.0, target_aspect: float | None = None,
    geoms: list[BaseGeometry] = (),
) -> tuple[float, float, float, float]:
    """Bounding box around `df`'s sites (and any `geoms`, e.g. drainage basins), padded by
    `pad` degrees on every side. If `target_aspect` is given and the box is currently wider
    than that (lon-range / lat-range), the *latitude* range is expanded (never the longitude
    range, so no site or context is cropped) until the box's aspect matches it -- used to
    narrow Amazon/Rangeland's panels toward Arctic's without losing any displayed extent."""
    lon_min, lon_max, lat_min, lat_max = df["lon"].min(), df["lon"].max(), df["lat"].min(), df["lat"].max()
    if len(geoms):
        g_lon_min, g_lat_min, g_lon_max, g_lat_max = shapely.total_bounds(geoms)
        lon_min, lon_max = min(lon_min, g_lon_min), max(lon_max, g_lon_max)
        lat_min, lat_max = min(lat_min, g_lat_min), max(lat_max, g_lat_max)
    lon_min, lon_max = lon_min - pad, lon_max + pad
    lat_min, lat_max = lat_min - pad, lat_max + pad
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


def _rect(fig_w: float, fig_h: float, left: float, top: float, width: float, height: float) -> list[float]:
    """[left, top, width, height] in inches (top measured from the figure's top edge) ->
    [left, bottom, width, height] in figure-fraction, as fig.add_axes expects."""
    return [left / fig_w, (fig_h - top - height) / fig_h, width / fig_w, height / fig_h]


def _inside_title(ax: plt.Axes, text: str, x: float, ha: str) -> None:
    ax.text(x, 1 - INSIDE_TITLE_INSET, text, transform=ax.transAxes, ha=ha, va="top",
            fontsize=plt.rcParams["axes.titlesize"], fontweight="bold", zorder=20)


def _draw_basins(ax: plt.Axes, basins: dict[str, BaseGeometry]) -> None:
    """Neutral basin outlines under the gauge dots -- not split-colored, since nested basins
    share boundaries across splits and a shared edge can't carry one split's color."""
    ax.add_geometries(list(basins.values()), crs=ccrs.PlateCarree(), facecolor="none",
                      edgecolor=BASIN_EDGE_COLOR, linewidth=BASIN_EDGE_WIDTH,
                      alpha=BASIN_EDGE_ALPHA, zorder=4)


def main() -> None:
    _style()
    arctic = arctic_sites()
    amazon = amazon_sites()
    rangeland = _jitter_coincident_sites(rangeland_sites())  # co-located towers stay visible
    basins = amazon_basins(amazon["station_id"])
    print(f"Arctic grids: {len(arctic)}, Amazon stations: {len(amazon)}, Rangeland sites: {len(rangeland)}")

    # Amazon/Rangeland's raw aspect (lon-range / lat-range) is much wider than Arctic's fixed
    # 1:1 circumpolar cap -- narrow them toward it (padding latitude only, so no site/context
    # is cropped) before computing panel widths.
    amazon_extent = _extent(amazon, pad=REGIONAL_PAD_DEG, target_aspect=REGIONAL_TARGET_ASPECT,
                            geoms=list(basins.values()))
    rangeland_extent = _extent(rangeland, pad=REGIONAL_PAD_DEG, target_aspect=REGIONAL_TARGET_ASPECT)

    # Each panel's box is sized to its own aspect ratio (width/height in the plotted
    # projection) so the map fills its box edge-to-edge -- no aspect-driven blank margin.
    robinson = ccrs.Robinson()
    robinson_aspect = (robinson.x_limits[1] - robinson.x_limits[0]) / (robinson.y_limits[1] - robinson.y_limits[0])
    arctic_aspect = 1.0  # a polar-centered circumpolar cap is always a circle in a square box
    amazon_aspect = (amazon_extent[1] - amazon_extent[0]) / (amazon_extent[3] - amazon_extent[2])
    rangeland_aspect = (rangeland_extent[1] - rangeland_extent[0]) / (rangeland_extent[3] - rangeland_extent[2])

    w_a = H_A * robinson_aspect
    w_b = H_ROW * arctic_aspect
    w_c = H_ROW * amazon_aspect
    w_d = H_ROW * rangeland_aspect

    fig_w = MARGIN_L + w_c + GAP_COLS + w_d + MARGIN_R
    fig_h = MARGIN_TOP + H_ROW + GAP_ROWS + H_ROW + MARGIN_BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Row 1: (a) left-aligned, its map + title band vertically centered on (b); (b) right-aligned.
    top_a = MARGIN_TOP + (H_ROW - TITLE_H - H_A - LEGEND_H) / 2 + TITLE_H
    ax_a = fig.add_axes(_rect(fig_w, fig_h, MARGIN_L, top_a, w_a, H_A), projection=ccrs.Robinson())
    ax_a.set_global()
    ax_a.add_feature(cfeature.OCEAN, facecolor=_OCEAN_COLOR, zorder=0)
    ax_a.coastlines(resolution="110m", linewidth=0.5, color="black")
    for name, df in (("Arctic", arctic), ("Amazon", amazon), ("Rangeland", rangeland)):
        ax_a.scatter(df["lon"], df["lat"], s=6, color=DOMAIN_COLOR[name.lower()], alpha=0.7,
                    edgecolors="none", transform=ccrs.PlateCarree(), label=name)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3, frameon=True, fancybox=False, edgecolor=LEGEND_EDGE_COLOR, facecolor="white",
               framealpha=0.95, handletextpad=0.3, columnspacing=0.8, borderpad=0.4, markerscale=1.4)
    ax_a.set_title("(a) Global overview", pad=3, fontweight="bold")

    rect_b = _rect(fig_w, fig_h, fig_w - MARGIN_R - w_b, MARGIN_TOP, w_b, H_ROW)
    _, ax_b = _circumpolar_axes(fig=fig, rect=rect_b, lat_labels=[50, 60, 70, 80])

    # Row 2: (c) | (d), spanning the full content width.
    top_row2 = MARGIN_TOP + H_ROW + GAP_ROWS
    regional_kwargs = {"draw_labels": ["left", "bottom"], "gridline_padding": GRIDLINE_PADDING,
                       "gridline_fontsize": GRIDLINE_FONTSIZE, "rivers": False,
                       "gridline_step": GRIDLINE_STEP_DEG}
    _, ax_c = _regional_axes(amazon_extent, fig=fig, rect=_rect(fig_w, fig_h, MARGIN_L, top_row2, w_c, H_ROW),
                             **regional_kwargs)
    _, ax_d = _regional_axes(rangeland_extent, fig=fig,
                             rect=_rect(fig_w, fig_h, MARGIN_L + w_c + GAP_COLS, top_row2, w_d, H_ROW),
                             **regional_kwargs)

    _scatter_by_split(ax_b, arctic)
    _draw_basins(ax_c, basins)
    _scatter_by_split(ax_c, amazon)
    _scatter_by_split(ax_d, rangeland)

    _draw_labels(ax_b, _place_labels((-180, 180, 44, 90), max_labels=4, min_sep_deg=30), (-180, 180, 44, 90), 3.0)
    _draw_labels(ax_c, _place_labels(amazon_extent, max_labels=3, min_sep_deg=6, inset_frac=REGIONAL_LABEL_INSET),
                 amazon_extent, 0.6)
    _draw_labels(ax_d, _place_labels(rangeland_extent, max_labels=3, min_sep_deg=8, inset_frac=REGIONAL_LABEL_INSET),
                 rangeland_extent, 0.8)

    _inside_title(ax_b, "(b) Arctic", INSIDE_TITLE_INSET, "left")
    _inside_title(ax_c, "(c) Amazon", 0.5, "center")
    _inside_title(ax_d, "(d) Rangeland", 0.5, "center")

    split_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=c, markeredgecolor="black",
              markeredgewidth=0.4, label=role.capitalize())
        for role, c in SPLIT_COLOR.items()
    ]
    # One shared split legend (same three colors in every domain), boxed inside the Arctic
    # panel's lower edge, over the mostly site-free North Atlantic/Europe sector.
    ax_b.legend(handles=split_handles, loc="lower center", ncol=3, frameon=True, fancybox=False,
               edgecolor=LEGEND_EDGE_COLOR, facecolor="white", framealpha=0.9, handletextpad=0.3,
               columnspacing=1.0, borderpad=0.4, fontsize=6)

    _save(fig, "fig1_study_sites")


if __name__ == "__main__":
    main()
