"""
Figure 6 — Individual, pretrained, and fine-tuned model comparison: one file per metric
(fig6a=RMSE, fig6b=KGE -- the two metrics used for the paper going forward, NSE/PBIAS
dropped), one boxplot subplot per target variable instead of one grouped boxplot per domain,
so each variable gets its own y-axis scale (a shared per-domain axis squashes small-magnitude
targets, e.g. Amazon's Active-fire-count/Burned-area next to Discharge, flat). Same idea as
Figure 7's one-map-per-target ragged grid, but for boxplots.

Every panel is the same fixed size regardless of how many targets its domain has (Arctic=2,
Amazon=3, Rangeland=4) -- rows with fewer targets are centered under the widest row and
leave open space at the sides, rather than stretching their panels wider. Manual inch-based
axes placement (mirrors Figure 7's _rect approach), since a GridSpec would force each row's
panels to a different width to fill the same row.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.plots import draw_metric_boxplot_panel  # noqa: E402
from _common import (  # noqa: E402
    DOMAINS, _add_grid, _apply_kge_ticks, _domain_combined, _horizontal_xticks, _save, _style,
)
from make_figure7 import AMAZON_TARGET_LABELS, _rect  # noqa: E402  (single-line "Active fire count")

METRIC_FILE_SUFFIX = {"RMSE": "a", "KGE": "b"}
ROW_LETTER = {"arctic": "(a)", "amazon": "(b)", "rangeland": "(c)"}
LEGEND_LABELS = {
    "Individual": "Domain-specific",
    "Pretrained": "Multi-domain (pretrained)",
    "Fine-tuned": "Multi-domain (finetuned)",
}
# Both toggleable, off for now for a cleaner look -- flip to True to bring either back.
SHOW_GRID = False
SHOW_MEDIAN_LABELS = False

# ── Layout constants (inches) -- fixed panel size for every row; narrower rows (fewer
# targets) are centered under the widest row's content width instead of stretching. ──
MARGIN, MARGIN_R = 0.08, 0.06
ROWLABEL_COL_W = 0.18  # rotated "(a) Arctic"-style row label (fig7 style)
TICK_CLEAR = 0.26      # space left of each row's own axes for its y tick numbers -- kept
                       # separate from the row-label column so the two text elements can't
                       # collide (they used to, when the row label doubled as the y-axis label)
METRIC_GAP, METRIC_COL_W = 0.06, 0.16  # rotated metric label just past each row's own last panel
MARGIN_TOP, MARGIN_BOTTOM = 0.48, 0.30  # top: legend; bottom: xtick label
PANEL_W, PANEL_H = 1.05, 1.0
GAP_COLS = 0.34  # wide enough that a neighboring panel's own tick labels (e.g. "4.5", "1e2")
                 # never touch this panel's right spine
GAP_COLS_TIGHT = 0.12  # KGE-only, shared-y rows only: non-first panels have their y-tick
                       # labels hidden (see SHARED_Y_DOMAINS), so there's no protruding tick
                       # text left to clear -- the panels can sit much closer together.
GAP_ROWS = 0.38  # room for one row's xtick label, between panels
# KGE-only (RMSE's per-target scales vary too much within a domain for a shared axis to help):
# Arctic's GPP/RECO and Rangeland's 4 targets each sit in a similarly-bounded KGE range, so
# sharing one y-axis per row removes redundant near-identical tick columns at no cost in
# legibility -- same convention as ablation_test/make_ablation_figures.py's SHARED_Y_DOMAINS.
SHARED_Y_DOMAINS = {"arctic", "rangeland"}
# Rangeland's floor: every real (Tukey-fenced) whisker across all 4 targets x 3 models is
# >=0.619 (checked against the actual Individual/Pretrained/Fine-tuned data), so 0.5 has a
# real margin and doesn't clip anything. Arctic has no fixed floor -- its true range varies too
# much between GPP and RECO (RECO's Individual whisker reaches -0.315) for a hand-picked
# constant, so it's computed from the row's own drawn data instead (see the two-pass sharing
# logic below).
RANGELAND_YFLOOR = 0.5
# Top for both shared rows: real whisker tops never exceed ~0.98 in either domain (checked).
# No median-value text is drawn above boxes (SHOW_MEDIAN_LABELS=False), so this only needs a
# small margin above the true max, not extra headroom for that text.
SHARED_Y_TOP = 1.02
# Amazon's Burned area panel: the Individual model's true whisker bottom is -7.77 (Tukey
# 1.5xIQR-fenced, not a lone outlier -- see ablation_test/make_ablation_figures.py's identical
# finding for the same underlying data). Flooring the view just below it removes wasted empty
# space without hiding anything.
AMAZON_YFLOOR = {"Burned area": -8}


def _gap_for(domain: str, metric: str) -> float:
    return GAP_COLS_TIGHT if (metric == "KGE" and domain in SHARED_Y_DOMAINS) else GAP_COLS


def figure6_model_comparison(metric: str = "RMSE") -> None:
    domain_data = {d: _domain_combined(d, metric) for d in DOMAINS}
    # This figure's wider panels use single-line Amazon labels (make_figure7's
    # AMAZON_TARGET_LABELS) -- _domain_combined() deliberately leaves Amazon's target names
    # raw so each figure can apply its own.
    domain_data["amazon"]["target"] = domain_data["amazon"]["target"].map(AMAZON_TARGET_LABELS)
    # sorted(unique()) matches how draw_metric_boxplot_panel itself orders targets, so panel
    # order here is identical to that ordering.
    domain_targets = {d: sorted(df["target"].unique()) for d, df in domain_data.items()}
    # content_w = the widest row's own footprint, each row using its own (possibly tightened)
    # gap -- not just the row with the most targets, since a tight-gap row with more targets
    # could still end up narrower than a wide-gap row with fewer.
    content_w = max(len(ts) * PANEL_W + (len(ts) - 1) * _gap_for(d, metric)
                    for d, ts in domain_targets.items())
    left_stack = MARGIN + ROWLABEL_COL_W + TICK_CLEAR
    fig_w = left_stack + content_w + METRIC_GAP + METRIC_COL_W + MARGIN_R
    fig_h = MARGIN_TOP + 3 * PANEL_H + 2 * GAP_ROWS + MARGIN_BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))

    patch_handles: list = []
    cursor = MARGIN_TOP
    for domain in DOMAINS:
        targets = domain_targets[domain]
        gap = _gap_for(domain, metric)
        row_w = len(targets) * PANEL_W + (len(targets) - 1) * gap
        row_left = left_stack + (content_w - row_w) / 2
        left = row_left
        share_y = metric == "KGE" and domain in SHARED_Y_DOMAINS
        row_axes: list[plt.Axes] = []
        for ti, target in enumerate(targets):
            rect = _rect(fig_w, fig_h, left, cursor, PANEL_W, PANEL_H)
            ax = fig.add_axes(rect)
            sub = domain_data[domain][domain_data[domain]["target"] == target]
            # Thin, small-multiple-style boxes -- a full-width box adds no extra information
            # over a thin one here (each panel only ever holds 3 boxes). KGE spreads its 3
            # boxes wider (group_span 0.75 vs RMSE's 0.55): shared/plots.py's median-value
            # labels sit just above each box's own whisker top, and KGE's boxes often cluster
            # with close values (e.g. Pretrained/Fine-tuned within 0.01) -- at RMSE's tighter
            # spacing those labels collide; the extra spread gives them room.
            draw_metric_boxplot_panel(ax, sub, metric, group_col="model",
                                       box_width_frac=0.5,
                                       group_span=0.75 if metric == "KGE" else 0.55,
                                       zero_line=True,
                                       show_median_labels=SHOW_MEDIAN_LABELS)
            ax.set_title("")
            # Force compact offset notation (e.g. "5.0" + "x10^2" instead of "500") only for
            # RMSE, whose units genuinely span large magnitudes across targets. KGE is
            # conventionally read as a plain value (bounded near [-1, 1] and always shown to 2
            # decimals -- see shared/plots.py's _format_median) -- forcing an offset there made
            # already-compact numbers (e.g. "0.90") less legible, not more.
            if metric == "RMSE":
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.yaxis.get_offset_text().set_fontsize(6)
            if metric == "KGE" and not share_y:
                if target in AMAZON_YFLOOR:
                    ax.set_ylim(bottom=AMAZON_YFLOOR[target])
                _apply_kge_ticks(ax)
            elif metric != "KGE":
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            # share_y panels get their ticks applied after the row's shared range is set below.
            ax.tick_params(axis="y", labelsize=6, pad=1)
            if SHOW_GRID:
                _add_grid(ax)
            _horizontal_xticks(ax)
            # Units intentionally omitted here (space-constrained small multiples) -- reported
            # in the manuscript text instead.
            ax.set_xticklabels([])
            ax.text(0.5, -0.1, target, transform=ax.transAxes, ha="center", va="top", fontsize=8)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            if not patch_handles:
                patch_handles = [p for p in ax.patches if p.get_label() and not p.get_label().startswith("_")]
            row_axes.append(ax)
            left += PANEL_W + gap

        if share_y:
            # Each panel was drawn independently above (ax.sharey() before drawing is broken
            # here -- see ablation_test/make_ablation_figures.py's module docstring for why),
            # so each ax's own ylim right now correctly reflects only its own data -- union
            # them for the row's shared floor.
            floor = RANGELAND_YFLOOR if domain == "rangeland" else min(ax.get_ylim()[0] for ax in row_axes)
            for i, ax in enumerate(row_axes):
                ax.set_ylim(floor, SHARED_Y_TOP)
                _apply_kge_ticks(ax)
                if i > 0:
                    plt.setp(ax.get_yticklabels(), visible=False)

        # Rotated row label just left of this row's own tick-number clearance, and the
        # metric label just right of this row's own last panel -- same placement convention
        # as Figure 7's row labels/colorbars, positioned relative to this row's own
        # (possibly centered/narrower) edges rather than a fixed global margin.
        row_label_y = (fig_h - cursor - PANEL_H / 2) / fig_h
        row_label_x = (row_left - TICK_CLEAR - ROWLABEL_COL_W / 2) / fig_w
        fig.text(row_label_x, row_label_y, f"{ROW_LETTER[domain]} {domain.capitalize()}",
                 fontsize=8, fontweight="bold", rotation=90, ha="center", va="center")
        metric_x = (row_left + row_w + METRIC_GAP + METRIC_COL_W / 2) / fig_w
        fig.text(metric_x, row_label_y, metric, fontsize=7, rotation=90, ha="center", va="center")

        cursor += PANEL_H + GAP_ROWS

    fig.legend(patch_handles, [LEGEND_LABELS[p.get_label()] for p in patch_handles],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=True,
               fancybox=False, fontsize=7, handlelength=2.2, handleheight=1.1, borderpad=0.4)
    suffix = METRIC_FILE_SUFFIX[metric]
    _save(fig, f"fig6{suffix}_individual_pretrained_finetuned_comparison_{metric.lower()}.png")


def main() -> None:
    _style()
    for metric in ("RMSE", "KGE"):
        figure6_model_comparison(metric)


if __name__ == "__main__":
    main()
