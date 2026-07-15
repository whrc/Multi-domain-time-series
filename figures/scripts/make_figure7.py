"""
Figure 7 — per-site/pixel %-change maps: individual per-domain model vs. the fine-tuned
multi-domain model, at held-out test sites only.

Three separate figures, one per metric: fig7a = RMSE, fig7b = NSE, fig7c = PBIAS. Each
figure is a ragged grid of small geographic maps -- one row per domain, one map per target
variable within that row (not one grouped panel per domain like Figure 6):
    row 1, Arctic     (2 maps): GPP, RECO                       -- circumpolar basemap
    row 2, Amazon     (3 maps): discharge, active_fire_count, burned_area -- regional basemap
    row 3, Rangeland  (4 maps): GPP, RECO, Rm, Rg                -- regional basemap
Each dot is one test site/pixel (Arctic is plotted at native pixel resolution, ~240 test
pixels -- not aggregated to grid-tile centroids like Figure 1's locator does, since the
whole point here is to show *where within a grid* fine-tuning helped or hurt), colored by
that site's %change value for that target. Reuses the same basemap style as Figure 1
(shared/plots.py's _circumpolar_axes / _regional_axes: circumpolar polar-stereographic for
Arctic, PlateCarree for Amazon/Rangeland, pale ocean fill, coastlines) and the same
PALETTE-adjacent styling, dpi=300, tight-layout conventions as every other figure in this
directory. Gridline/tick labels are turned off on all panels (unlike Figure 1) -- with 9
small panels total, per-panel lat/lon ticks would be unreadable clutter; the point of this
figure is the relative spatial color pattern, not precise coordinates.

Target sets match the flux-only convention already used in Figures 4/6 (ARCTIC_FLUXONLY_TEST,
RANGELAND_FLUXONLY_TEST, and the multi-domain finetuned_fluxonly checkpoint) -- Amazon has no
flux-only variant, so all 3 of its targets are used.

Data sources (per-site/pixel rows -- each row's metric value is itself already the 5-seed
average for that site/pixel, via _load_seedavg; per-seed std is dropped, not plotted here):
  individual:  ARCTIC_FLUXONLY_TEST / RANGELAND_FLUXONLY_TEST / AMAZON_TEST (see
               make_remaining_figures.py, now pointing at the *_seedavg metrics files) --
               Arctic rows are filtered to `~obs_degenerate` first, matching
               figure4_individual_domain_results(). Rangeland's individual file has
               "_predicted"-suffixed target names (GPP_predicted, ...) while every other
               source (including its own multi-domain counterpart) doesn't -- stripped
               before joining.
  multi-domain: MD_FINETUNED_SEEDAVG / {domain} / "{domain}_metrics_seedavg.csv" -- same
               per-site/pixel grain as the individual files (verified: matching row
               counts/join keys), NOT the pretrained stage.
  join keys:   Arctic (grid, y, x, lat, lon, ssp, period, target); Amazon (station_id,
               target); Rangeland (site, target) -- Rangeland's `pft` column is redundant
               with `site` (one pft per site, verified) so it's ignored, not used to join.
  site/pixel coordinates: Arctic's lat/lon already come with its metrics rows; Amazon/
               Rangeland don't carry coordinates in their metrics files, so station/site
               lat-lon is pulled the same way make_figure1.py does (load_station_coords /
               load_site_coords, dynamically imported from each domain's 04_evaluate.py).

%-change formula, per metric (each of the 3 figures is read independently -- no sign/color
convention is shared *across* fig7a/b/c, only within each one):
  RMSE:  (multi - individual) / individual * 100, raw formula. Improvement is a *negative*
         value here (RMSE went down) -- handled by giving this figure's colormap the
         opposite orientation from NSE/PBIAS's (RdYlGn_r) so green still means "fine-tuning
         helped" when you look at fig7a on its own, without altering the underlying number.
  NSE:   (multi - individual) / individual * 100, raw formula. Improvement is positive
         (NSE went up) -- normal RdYlGn orientation.
  PBIAS: (multi - individual) / abs(individual) * 100, using abs(individual) as the
         denominator so a sign flip between the two runs doesn't produce a nonsensical
         ratio. Clipped to +-200% before plotting so the rare near-zero-individual-PBIAS
         site doesn't blow out the color scale for every other site. Improvement (bias
         magnitude shrank) is positive here -- normal RdYlGn orientation, same as NSE.
  Arctic's extra ssp (2 scenarios) x period (historical/projected) dimensions are averaged
  (mean %change, not mean of the raw metrics) per pixel per target *before* mapping,
  collapsing to the 2-map-per-row layout above (not faceted further).

Color scale: one shared diverging colorbar per figure (not per panel), symmetric around 0,
range = the 95th-percentile absolute %change across all of that figure's panels combined,
so a single outlier site doesn't wash out the rest of that figure's contrast. Independent
panels would allow more per-panel contrast but make cross-domain comparison within a figure
impossible -- a shared scale is the point of putting them in one figure.

Layout: within a row all panels share the same basemap/extent (same domain), so they're
identically sized; only the row height varies... no -- row *height* is fixed the same
across all three domains (like Figure 1's H_ROW), and each row's total width varies with
its panel count x that domain's own aspect ratio (Amazon/Rangeland narrowed toward a
square via the same latitude-padding trick as Figure 1, REGIONAL_TARGET_ASPECT, imported
from make_figure1.py rather than re-derived). Each row is centered under the figure's
content width (the widest row) -- laid out with the same manual aspect-matched-axes
approach as make_figure1.py, not a rectangular GridSpec, since a GridSpec would force
every column to the same width across differently-sized rows.

Output: fig7a_individual_vs_finetuned_rmse.png, fig7b_..._nse.png, fig7c_..._pbias.png --
PNG only (no SVG), matching Figures 3-6's convention for data-driven (non-schematic)
figures in make_remaining_figures.py, not Figure 1's PNG+SVG.
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
from make_figure1 import _extent, _load_module, REGIONAL_TARGET_ASPECT  # noqa: E402
from make_remaining_figures import (  # noqa: E402
    AMAZON_TEST, ARCTIC_FLUXONLY_TEST, MD_FINETUNED_SEEDAVG, RANGELAND_FLUXONLY_TEST,
    _load_seedavg, _save, _style,
)

ARCTIC_TARGETS = ["GPP", "RECO"]
AMAZON_TARGETS = ["discharge", "active_fire_count", "burned_area"]
RANGELAND_TARGETS = ["GPP", "RECO", "Rm", "Rg"]
DOMAIN_TARGETS = {"arctic": ARCTIC_TARGETS, "amazon": AMAZON_TARGETS, "rangeland": RANGELAND_TARGETS}
ROW_LETTER = {"arctic": "(a)", "amazon": "(b)", "rangeland": "(c)"}
# Single-line variant of make_remaining_figures.py's AMAZON_TARGET_LABELS -- that one wraps
# "Active fire\ncount" for its narrower boxplot x-tick-label context; this figure's panels
# have more horizontal room, so keep it on one line here instead of importing/mutating the
# shared dict (which Figures 4/6 still rely on wrapped).
AMAZON_TARGET_LABELS = {"active_fire_count": "Active fire count", "burned_area": "Burned area",
                        "discharge": "Discharge"}

CMAP = {"RMSE": "RdYlGn_r", "NSE": "RdYlGn", "PBIAS": "RdYlGn_r"}
METRIC_FILE_SUFFIX = {"RMSE": "a", "NSE": "b", "PBIAS": "c"}
CLIP_PERCENTILE = 90  # colorbar range = this percentile of |%change|, clipped -- keeps a
                       # handful of extreme sites (e.g. near-zero-individual-NSE pixels)
                       # from washing out the color contrast for every other site. Computed
                       # PER DOMAIN (not pooled across all 3): Arctic's %change spread is
                       # roughly 10x wider than Amazon/Rangeland's on every metric, so a
                       # pooled percentile is dominated by whichever domain has the most
                       # test points/longest tail (always Arctic) and washes out the other
                       # two domains' real, smaller-magnitude variation -- hence also one
                       # separate colorbar per domain row, not one shared across the figure.

# ── Layout constants (inches) -- same manual aspect-matched approach as make_figure1.py ──
MARGIN = 0.1
GAP_COLS = 0.06
GAP_ROWS = 0.12
H_ROW = {"arctic": 1.3, "amazon": 1.0, "rangeland": 1.0}  # Amazon/Rangeland have far fewer
                                                          # test sites than Arctic, so their
                                                          # panels don't need as much room
ROW_LABEL_W = 0.22  # reserved strip (from the figure's left margin) that a row's own panels
                    # can be centered within -- the label itself sits just before wherever
                    # that row's panels actually start (see ROW_LABEL_GAP), not fixed in place
ROW_LABEL_GAP = 0.12  # gap between the row label and that row's own first panel
CBAR_W = 0.08          # narrow vertical colorbar, one per domain row
# The 2-line "% change / in {metric}" title is centered on the colorbar and is much wider
# than the bar itself (CBAR_W=0.08in), so it overhangs left/right well past the bar -- both
# CBAR_GAP (left overhang, so it clears the row's last panel) and CBAR_TITLE_PAD (right
# overhang, so it doesn't get cropped at the figure edge) have to be sized for the *title's*
# width, not the bar's.
CBAR_GAP = 0.25
CBAR_TITLE_PAD = 0.2
CBAR_H_FRAC = 0.7      # colorbar height as a fraction of that row's own panel height --
                       # deliberately shorter than the row, so it reads as a small accent
                       # next to the maps rather than another full-height panel
CBAR_TITLE_HEADROOM = 0.22  # fixed space above the colorbar reserved for its 2-line title
ARCTIC_DOT_S = 6    # Arctic has ~300+ test pixels that overlap heavily at a larger size
OTHER_DOT_S = 16    # Amazon/Rangeland have <20 test sites each -- can afford bigger, clearer dots
REGIONAL_PAD = 1.0  # degrees of padding around Amazon/Rangeland's site extent
COASTLINE_COLOR = "grey"    # de-emphasized vs. the default black -- the point of this figure
COASTLINE_LINEWIDTH = 0.35  # is the site colors, not the basemap


def _pct_change(individual: pd.Series, multi: pd.Series, metric: str) -> pd.Series:
    """%change relative to the individual model. PBIAS uses %change in *|PBIAS|* (bias
    magnitude), not raw signed PBIAS: (multi-individual)/individual would flip sign
    depending on individual's own sign even for the same underlying improvement (e.g.
    individual=+10->multi=+5 is a shrinking-bias improvement but gives -50%, while
    individual=-10->multi=-5 is the identical improvement but gives +50%) -- using
    |multi| and |individual| instead makes the sign consistently mean "bias grew/shrank"
    regardless of which direction the original bias pointed, including sign-flip cases
    (individual=+10->multi=-3 is still an improvement, |3|<|10|, correctly negative)."""
    if metric == "PBIAS":
        return (multi.abs() - individual.abs()) / individual.abs().replace(0, np.nan) * 100
    return (multi - individual) / individual.replace(0, np.nan) * 100


def arctic_pct_change(metric: str) -> pd.DataFrame:
    individual = _load_seedavg(ARCTIC_FLUXONLY_TEST)
    individual = individual[~individual["obs_degenerate"] & individual["target"].isin(ARCTIC_TARGETS)]
    multi = _load_seedavg(MD_FINETUNED_SEEDAVG / "arctic" / "arctic_metrics_seedavg.csv")
    multi = multi[multi["target"].isin(ARCTIC_TARGETS)]
    keys = ["grid", "y", "x", "lat", "lon", "ssp", "period", "target"]
    merged = individual.merge(multi, on=keys, suffixes=("_individual", "_multi"))
    merged["pct_change"] = _pct_change(merged[f"{metric}_individual"], merged[f"{metric}_multi"], metric)
    merged = merged.dropna(subset=["pct_change"])
    # Collapse ssp x period to one mean %change per pixel/target (2 maps/row, not faceted further).
    return merged.groupby(["grid", "y", "x", "lat", "lon", "target"], as_index=False)["pct_change"].mean()


def amazon_pct_change(metric: str) -> pd.DataFrame:
    individual = _load_seedavg(AMAZON_TEST)
    individual = individual[individual["target"].isin(AMAZON_TARGETS)]
    multi = _load_seedavg(MD_FINETUNED_SEEDAVG / "amazon" / "amazon_metrics_seedavg.csv")
    multi = multi[multi["target"].isin(AMAZON_TARGETS)]
    merged = individual.merge(multi, on=["station_id", "target"], suffixes=("_individual", "_multi"))
    merged["pct_change"] = _pct_change(merged[f"{metric}_individual"], merged[f"{metric}_multi"], metric)
    merged = merged.dropna(subset=["pct_change"])

    ev04 = _load_module("amazon_domain", "04_evaluate.py", "_am04_fig7")
    coords = ev04.load_station_coords(load_config("amazon_domain"))
    merged["station_id"] = merged["station_id"].astype(str)
    merged = merged.merge(coords, on="station_id", how="left").dropna(subset=["lat", "lon"])
    return merged[["lat", "lon", "target", "pct_change"]]


def rangeland_pct_change(metric: str) -> pd.DataFrame:
    individual = _load_seedavg(RANGELAND_FLUXONLY_TEST)
    individual["target"] = individual["target"].str.replace("_predicted", "", regex=False)
    individual = individual[individual["target"].isin(RANGELAND_TARGETS)]
    multi = _load_seedavg(MD_FINETUNED_SEEDAVG / "rangeland" / "rangeland_metrics_seedavg.csv")
    multi = multi[multi["target"].isin(RANGELAND_TARGETS)]
    merged = individual.merge(multi, on=["site", "target"], suffixes=("_individual", "_multi"))
    merged["pct_change"] = _pct_change(merged[f"{metric}_individual"], merged[f"{metric}_multi"], metric)
    merged = merged.dropna(subset=["pct_change"])

    ev04 = _load_module("rangeland_domain", "04_evaluate.py", "_rl04_fig7")
    coords = ev04.load_site_coords(load_config("rangeland_domain"))
    merged = merged.merge(coords, on="site", how="left").dropna(subset=["lat", "lon"])
    return merged[["lat", "lon", "target", "pct_change"]]


def _rect(fig_w: float, fig_h: float, left: float, top: float, width: float, height: float) -> list[float]:
    """[left, top, width, height] in inches (top measured from the figure's top edge) ->
    [left, bottom, width, height] in figure-fraction, as fig.add_axes expects."""
    return [left / fig_w, (fig_h - top - height) / fig_h, width / fig_w, height / fig_h]


def _target_label(domain: str, target: str) -> str:
    return AMAZON_TARGET_LABELS[target] if domain == "amazon" else target


def make_figure(metric: str) -> None:
    domain_data = {
        "arctic": arctic_pct_change(metric),
        "amazon": amazon_pct_change(metric),
        "rangeland": rangeland_pct_change(metric),
    }

    # Percentile-clip PER DOMAIN (not pooled -- see CLIP_PERCENTILE's comment) so each
    # domain's own colorbar reflects its own spread instead of being dragged by another
    # domain's much longer tail.
    vmax = {}
    for domain, df in domain_data.items():
        vmax[domain] = np.percentile(np.abs(df["pct_change"].dropna().to_numpy()), CLIP_PERCENTILE)
        df["pct_change"] = df["pct_change"].clip(-vmax[domain], vmax[domain])
        print(f"{metric} {domain}: {len(df)} rows across {df['target'].nunique()} targets, "
              f"vmax=+-{vmax[domain]:.1f}%")
    cmap = CMAP[metric]

    extents = {"arctic": (-180, 180, 44, 90)}
    aspects = {"arctic": 1.0}
    for domain in ("amazon", "rangeland"):
        ext = _extent(domain_data[domain], pad=REGIONAL_PAD, target_aspect=REGIONAL_TARGET_ASPECT)
        extents[domain] = ext
        aspects[domain] = (ext[1] - ext[0]) / (ext[3] - ext[2])

    panel_w = {d: H_ROW[d] * a for d, a in aspects.items()}
    row_w = {d: len(DOMAIN_TARGETS[d]) * panel_w[d] + (len(DOMAIN_TARGETS[d]) - 1) * GAP_COLS for d in aspects}
    # Each row's own small vertical colorbar sits just past its last panel, so the row's
    # total footprint (panels + colorbar) is what gets centered under the figure's content width.
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
                                       coastline_color=COASTLINE_COLOR, coastline_linewidth=COASTLINE_LINEWIDTH)
            sub = domain_data[domain][domain_data[domain]["target"] == target]
            mappable = ax.scatter(sub["lon"], sub["lat"], c=sub["pct_change"], cmap=cmap,
                                  vmin=-vmax[domain], vmax=vmax[domain], s=dot_s, edgecolors="none",
                                  transform=ccrs.PlateCarree(), zorder=5)
            # In-panel label (top center), not ax.set_title -- keeps the target name inside
            # the map's own box rather than adding an external title band per panel.
            ax.text(0.5, 0.96, _target_label(domain, target), transform=ax.transAxes, fontsize=7,
                    ha="center", va="top", zorder=20,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5})
            left += panel_w[domain] + GAP_COLS

        # Row label via fig.text, not ax.set_ylabel -- a GeoAxes' y-axis label doesn't
        # reliably survive bbox_inches="tight" (it was silently omitted from every panel
        # in the first pass of this figure), so it's placed directly in figure coordinates.
        # Positioned relative to *this row's own* left edge (row_left, which shifts right
        # for narrower centered rows), not a fixed global margin -- so the label always
        # sits just before that row's own panels start, not stranded at the far left when
        # a row is narrower than the widest one.
        row_label_x = (row_left - ROW_LABEL_GAP) / fig_w
        row_label_y = (fig_h - cursor - h_row / 2) / fig_h
        fig.text(row_label_x, row_label_y, f"{ROW_LETTER[domain]} {domain.capitalize()}", fontsize=8,
                fontweight="bold", rotation=90, ha="center", va="center")

        # This row's own small vertical colorbar, shorter than the row so it reads as an
        # accent next to the maps rather than another panel. Anchored from the top with a
        # fixed headroom (not vertically centered) so its 2-line title always has enough
        # clearance above it regardless of that row's own height -- centering left too
        # little headroom for the shorter Amazon/Rangeland rows, and the title collided
        # with the neighboring panel's top edge.
        cbar_h = CBAR_H_FRAC * h_row
        cbar_top = cursor + CBAR_TITLE_HEADROOM
        cbar_left = row_left + row_w[domain] + CBAR_GAP
        cax = fig.add_axes(_rect(fig_w, fig_h, cbar_left, cbar_top, CBAR_W, cbar_h))
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.ax.set_title(f"% change\nin {metric}", fontsize=6, pad=3)

        cursor += h_row + GAP_ROWS

    suffix = METRIC_FILE_SUFFIX[metric]
    _save(fig, f"fig7{suffix}_individual_vs_finetuned_{metric.lower()}.png")


def main() -> None:
    _style()
    for metric in ("RMSE", "NSE", "PBIAS"):
        make_figure(metric)


if __name__ == "__main__":
    main()
