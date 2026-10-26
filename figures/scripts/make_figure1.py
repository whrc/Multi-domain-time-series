"""
Figure 1 — study sites: (a) global locator, (b)-(d) per-domain train/val/test split maps
(Amazon also shows drainage basins). Sites and splits are recomputed via each domain's own
pipeline code; panels are placed manually so each box matches its map's aspect ratio.
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
from _common import _jitter_coincident_sites  # noqa: E402

FIGURES_DIR = REPO_ROOT / "figures"
SVG_DIR = FIGURES_DIR / "svg"
DPI = 300

DOMAIN_COLOR = {"arctic": PALETTE[6], "amazon": PALETTE[5], "rangeland": PALETTE[4]}  # same as _common.py
SPLIT_COLOR = {"train": "#009E73", "val": "#E69F00", "test": "#56B4E9"}

# Layout (inches). Row 1 = (a) locator | (b) Arctic; row 2 = (c) Amazon | (d) Rangeland.
MARGIN_L = MARGIN_R = 0.08
MARGIN_TOP = 0.08
MARGIN_BOTTOM = 0.06
TITLE_H = 0.22  # (a)'s title band above its map; (b)-(d) titles sit inside the map
LEGEND_H = 0.25  # (a)'s legend band below its map
GAP_ROWS = 0.2
GAP_COLS = 0.22
H_A = 1.45  # locator map height, excluding title band
H_ROW = 2.3  # detail-panel height; widths follow each panel's aspect
REGIONAL_TARGET_ASPECT = 1.15  # (c)/(d) lon:lat aspect, reached by padding latitude only
REGIONAL_PAD_DEG = 2.0  # (c)/(d) padding around the data before aspect padding
REGIONAL_LABEL_INSET = 0.08  # (c)/(d) city-label search inset (see _place_labels)
INSIDE_TITLE_INSET = 0.03  # in-panel title offset from the frame, axes fraction
GRIDLINE_PADDING = -3  # pulls (c)/(d) lat/lon tick labels just inside the frame
GRIDLINE_FONTSIZE = 5
GRIDLINE_STEP_DEG = 10

LEGEND_EDGE_COLOR = "grey"

# Amazon basins are nested, so outlines only (fills would stack and hide the gauges).
BASIN_SIMPLIFY_DEG = 0.01  # ~1 km; keeps the SVG small (~777k raw vertices)
BASIN_EDGE_COLOR = "#2F5D8A"  # steel blue, distinct from borders and split colors
BASIN_EDGE_WIDTH = 0.3
BASIN_EDGE_ALPHA = 0.8
GPKG_HEADER_BYTES = 8
GPKG_FLAGS_BYTE = 3
GPKG_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def _load_module(domain_dir: str, filename: str, alias: str):
    """Import a numbered domain script (e.g. 01_preprocess.py), which `import` can't name."""
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
    """Grid centroids + split, recomputed from the local pass-1 summary cache."""
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
    """Station coords + split, recomputed from GCS (preprocessed outputs live on the VM)."""
    pp01 = _load_module("amazon_domain", "01_preprocess.py", "_am01_fig1")
    ev04 = _load_module("amazon_domain", "04_evaluate.py", "_am04_fig1")
    cfg = load_config("amazon_domain")
    split = pp01.split_stations(pp01.load_filtered(cfg), cfg)
    coords = ev04.load_station_coords(cfg)
    coords["split"] = coords["station_id"].map(split)
    coords = coords.dropna(subset=["lat", "lon", "split"])
    return coords[["station_id", "lon", "lat", "split"]]


def _gpkg_geometry(blob: bytes) -> BaseGeometry:
    # GeoPackage blob = 8-byte header + envelope (size from flags bits 1-3, spec 2.1.3) + WKB.
    envelope_code = (blob[GPKG_FLAGS_BYTE] >> 1) & 0b111
    if envelope_code not in GPKG_ENVELOPE_BYTES:
        raise ValueError(f"Invalid GeoPackage envelope code {envelope_code}")
    return shapely.wkb.loads(bytes(blob[GPKG_HEADER_BYTES + GPKG_ENVELOPE_BYTES[envelope_code]:]))


def amazon_basins(station_ids: pd.Series) -> dict[str, BaseGeometry]:
    """station_id -> simplified drainage-basin polygon (EPSG:4326) from the station GeoPackage."""
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
    """(lon_min, lon_max, lat_min, lat_max) around sites and `geoms`, padded by `pad` degrees;
    if wider than `target_aspect` (lon/lat span), latitude is widened to match."""
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
    """Up to `max_labels` Natural Earth places by rank, >= min_sep_deg apart, searched within
    `extent` shrunk by `inset_frac` per side so label text stays inside the panel."""
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
    """Marker + label, with text offset toward the panel center so it stays inside."""
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
    """Largest split drawn first so smaller ones stay visible on top."""
    for role in sorted(SPLIT_COLOR, key=lambda r: (df["split"] == r).sum(), reverse=True):
        mask = df["split"] == role
        if not mask.any():
            continue
        ax.scatter(df.loc[mask, "lon"], df.loc[mask, "lat"], s=18, color=SPLIT_COLOR[role],
                  edgecolors="black", linewidths=0.3, alpha=0.9, zorder=5, transform=ccrs.PlateCarree())


def _rect(fig_w: float, fig_h: float, left: float, top: float, width: float, height: float) -> list[float]:
    """Inches with top measured from the figure top -> fig.add_axes figure-fraction rect."""
    return [left / fig_w, (fig_h - top - height) / fig_h, width / fig_w, height / fig_h]


def _inside_title(ax: plt.Axes, text: str, x: float, ha: str) -> None:
    ax.text(x, 1 - INSIDE_TITLE_INSET, text, transform=ax.transAxes, ha=ha, va="top",
            fontsize=plt.rcParams["axes.titlesize"], fontweight="bold", zorder=20)


def _draw_basins(ax: plt.Axes, basins: dict[str, BaseGeometry]) -> None:
    """Neutral basin outlines under the gauges (nested basins share edges across splits)."""
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

    amazon_extent = _extent(amazon, pad=REGIONAL_PAD_DEG, target_aspect=REGIONAL_TARGET_ASPECT,
                            geoms=list(basins.values()))
    rangeland_extent = _extent(rangeland, pad=REGIONAL_PAD_DEG, target_aspect=REGIONAL_TARGET_ASPECT)

    robinson = ccrs.Robinson()
    robinson_aspect = (robinson.x_limits[1] - robinson.x_limits[0]) / (robinson.y_limits[1] - robinson.y_limits[0])
    arctic_aspect = 1.0  # circumpolar cap
    amazon_aspect = (amazon_extent[1] - amazon_extent[0]) / (amazon_extent[3] - amazon_extent[2])
    rangeland_aspect = (rangeland_extent[1] - rangeland_extent[0]) / (rangeland_extent[3] - rangeland_extent[2])

    w_a = H_A * robinson_aspect
    w_b = H_ROW * arctic_aspect
    w_c = H_ROW * amazon_aspect
    w_d = H_ROW * rangeland_aspect

    fig_w = MARGIN_L + w_c + GAP_COLS + w_d + MARGIN_R
    fig_h = MARGIN_TOP + H_ROW + GAP_ROWS + H_ROW + MARGIN_BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Row 1: (a) left, vertically centered on (b); (b) right-aligned.
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
    # Shared split legend, inside (b) over the site-free North Atlantic sector.
    ax_b.legend(handles=split_handles, loc="lower center", ncol=3, frameon=True, fancybox=False,
               edgecolor=LEGEND_EDGE_COLOR, facecolor="white", framealpha=0.9, handletextpad=0.3,
               columnspacing=1.0, borderpad=0.4, fontsize=6)

    _save(fig, "fig1_study_sites")


if __name__ == "__main__":
    main()
