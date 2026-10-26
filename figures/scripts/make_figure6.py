"""
Figure 6 — individual vs. pretrained vs. fine-tuned boxplots (fig6a RMSE, fig6b KGE): one row
per domain, one fixed-size panel per target with its own y scale; shorter rows are centered.
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
SHOW_GRID = False
SHOW_MEDIAN_LABELS = False

# Layout (inches)
MARGIN, MARGIN_R = 0.08, 0.06
ROWLABEL_COL_W = 0.18  # rotated "(a) Arctic" row label
TICK_CLEAR = 0.26      # y tick numbers, kept separate from the row-label column
METRIC_GAP, METRIC_COL_W = 0.06, 0.16  # rotated metric label right of each row
MARGIN_TOP, MARGIN_BOTTOM = 0.48, 0.30  # top: legend; bottom: xtick label
PANEL_W, PANEL_H = 1.05, 1.0
GAP_COLS = 0.34  # clears the neighboring panel's y tick labels
GAP_COLS_TIGHT = 0.12  # shared-y KGE rows, where non-first panels hide y tick labels
GAP_ROWS = 0.38  # room for one row's xtick label
SHARED_Y_DOMAINS = {"arctic", "rangeland"}  # KGE only: one y-axis per row
RANGELAND_YFLOOR = 0.5  # all whiskers >= 0.619; Arctic's floor comes from its data
SHARED_Y_TOP = 1.02  # whisker tops <= ~0.98
AMAZON_YFLOOR = {"Burned area": -8}  # Individual whisker bottom is -7.77


def _gap_for(domain: str, metric: str) -> float:
    return GAP_COLS_TIGHT if (metric == "KGE" and domain in SHARED_Y_DOMAINS) else GAP_COLS


def figure6_model_comparison(metric: str = "RMSE") -> None:
    domain_data = {d: _domain_combined(d, metric) for d in DOMAINS}
    domain_data["amazon"]["target"] = domain_data["amazon"]["target"].map(AMAZON_TARGET_LABELS)
    domain_targets = {d: sorted(df["target"].unique()) for d, df in domain_data.items()}
    # Widest row footprint, using each row's own gap.
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
            # KGE gets a wider group_span so close-valued median labels don't collide.
            draw_metric_boxplot_panel(ax, sub, metric, group_col="model",
                                       box_width_frac=0.5,
                                       group_span=0.75 if metric == "KGE" else 0.55,
                                       zero_line=True,
                                       show_median_labels=SHOW_MEDIAN_LABELS)
            ax.set_title("")
            # Offset (x10^n) notation for RMSE only; its magnitudes vary widely across targets.
            if metric == "RMSE":
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.yaxis.get_offset_text().set_fontsize(6)
            if metric == "KGE" and not share_y:
                if target in AMAZON_YFLOOR:
                    ax.set_ylim(bottom=AMAZON_YFLOOR[target])
                _apply_kge_ticks(ax)
            elif metric != "KGE":
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            # share_y panels get ticks after the row's shared range is set below.
            ax.tick_params(axis="y", labelsize=6, pad=1)
            if SHOW_GRID:
                _add_grid(ax)
            _horizontal_xticks(ax)
            ax.set_xticklabels([])  # units are given in the manuscript text
            ax.text(0.5, -0.1, target, transform=ax.transAxes, ha="center", va="top", fontsize=8)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            if not patch_handles:
                patch_handles = [p for p in ax.patches if p.get_label() and not p.get_label().startswith("_")]
            row_axes.append(ax)
            left += PANEL_W + gap

        if share_y:
            # Panels were drawn unshared (sharey before drawing doesn't work here); union their ylims.
            floor = RANGELAND_YFLOOR if domain == "rangeland" else min(ax.get_ylim()[0] for ax in row_axes)
            for i, ax in enumerate(row_axes):
                ax.set_ylim(floor, SHARED_Y_TOP)
                _apply_kge_ticks(ax)
                if i > 0:
                    plt.setp(ax.get_yticklabels(), visible=False)

        # Row and metric labels follow this row's own (centered) edges.
        row_label_y = (fig_h - cursor - PANEL_H / 2) / fig_h
        row_label_x = (row_left - TICK_CLEAR - ROWLABEL_COL_W / 2) / fig_w
        fig.text(row_label_x, row_label_y, f"{ROW_LETTER[domain]} {domain.capitalize()}",
                 fontsize=8, fontweight="bold", rotation=90, ha="center", va="center")
        metric_x = (row_left + row_w + METRIC_GAP + METRIC_COL_W / 2) / fig_w
        fig.text(metric_x, row_label_y, metric, fontsize=7, fontweight="bold", rotation=90, ha="center",
                 va="center")

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
