"""
Figure 6 — Individual, pretrained, and fine-tuned model comparison: one file per metric
(fig6a=RMSE, fig6b=NSE, fig6c=PBIAS, fig6d=KGE), one boxplot subplot per target variable
instead of one grouped boxplot per domain, so each variable gets its own y-axis scale (a
shared per-domain axis squashes small-magnitude targets, e.g. Amazon's Active-fire-count/
Burned-area next to Discharge, flat). Same idea as Figure 7's one-map-per-target ragged grid,
but for boxplots.

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
import pandas as pd
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.plots import draw_metric_boxplot_panel  # noqa: E402
from _common import (  # noqa: E402
    AMAZON_TEST, ARCTIC_FLUXONLY_TEST, DOMAINS, MD_FINETUNED_SEEDAVG, MD_PRETRAINED_SEEDAVG,
    RANGELAND_DAY_TO_MONTH, RANGELAND_FLUXONLY_TEST,
    _add_grid, _horizontal_xticks, _load_seedavg, _save, _style,
)
from make_figure7 import AMAZON_TARGET_LABELS, _rect  # noqa: E402  (single-line "Active fire count")
from make_figure8 import ARCTIC_UNITS, AMAZON_UNITS, RANGELAND_UNITS  # noqa: E402

MODEL_ORDER = ["Individual", "Pretrained", "Fine-tuned"]
METRIC_FILE_SUFFIX = {"RMSE": "a", "NSE": "b", "PBIAS": "c", "KGE": "d"}
ROW_LETTER = {"arctic": "(a)", "amazon": "(b)", "rangeland": "(c)"}
LEGEND_LABELS = {
    "Individual": "Domain-specific",
    "Pretrained": "Multi-domain (pretrained)",
    "Fine-tuned": "Multi-domain (finetuned)",
}
# Same unit strings Figure 8 already uses on its time-series y-axes -- keyed here by each
# domain's own *display* target label (post AMAZON_TARGET_LABELS remap for Amazon) since
# that's what each panel's x-tick actually shows. Rangeland is overridden to month^-1 (not
# Figure 8's own RANGELAND_UNITS dict, left untouched) to match _domain_combined's per-day
# -> per-month RMSE rescale for this figure family -- see RANGELAND_DAY_TO_MONTH.
UNITS_BY_DOMAIN = {
    "arctic": ARCTIC_UNITS,
    "amazon": {AMAZON_TARGET_LABELS[k]: v for k, v in AMAZON_UNITS.items()},
    "rangeland": {k: "g C m$^{-2}$ month$^{-1}$" for k in RANGELAND_UNITS},
}

# ── Layout constants (inches) -- fixed panel size for every row; narrower rows (fewer
# targets) are centered under the widest row's content width instead of stretching. ──
MARGIN, MARGIN_R = 0.08, 0.06
ROWLABEL_COL_W = 0.18  # rotated "(a) Arctic"-style row label (fig7 style)
TICK_CLEAR = 0.26      # space left of each row's own axes for its y tick numbers -- kept
                       # separate from the row-label column so the two text elements can't
                       # collide (they used to, when the row label doubled as the y-axis label)
METRIC_GAP, METRIC_COL_W = 0.06, 0.16  # rotated metric label just past each row's own last panel
MARGIN_TOP, MARGIN_BOTTOM = 0.48, 0.42  # top: legend; bottom: xtick label + its unit line
PANEL_W, PANEL_H = 1.05, 1.0
GAP_COLS = 0.34  # wide enough that a neighboring panel's own tick labels (e.g. "4.5", "1e2")
                 # never touch this panel's right spine
GAP_ROWS = 0.5   # room for one row's xtick label + unit line, between panels


def _normalize_individual(domain: str, metric: str) -> pd.DataFrame:
    """Load an individual domain's flux-only test-set metric, normalized to a plain
    {target, metric} frame (drops Arctic's degenerate rows / period column, Rangeland's
    '_predicted' target-name suffix — neither matches multi-domain's own conventions)."""
    if domain == "arctic":
        df = _load_seedavg(ARCTIC_FLUXONLY_TEST)
        df = df[~df["obs_degenerate"]]
        return df[["target", metric]]
    if domain == "rangeland":
        df = _load_seedavg(RANGELAND_FLUXONLY_TEST)
        df = df.assign(target=df["target"].str.replace("_predicted", "", regex=False))
        return df[["target", metric]]
    df = _load_seedavg(AMAZON_TEST)  # amazon: no flux-only variant, no normalization needed
    return df[["target", metric]]


def _domain_combined(domain: str, metric: str) -> pd.DataFrame:
    """Individual/Pretrained/Fine-tuned rows for one domain, one metric, as a single
    {target, metric, model} frame."""
    individual = _normalize_individual(domain, metric).assign(model="Individual")
    pretrained = _load_seedavg(MD_PRETRAINED_SEEDAVG / domain / f"{domain}_metrics_seedavg.csv")[
        ["target", metric]].assign(model="Pretrained")
    finetuned = _load_seedavg(MD_FINETUNED_SEEDAVG / domain / f"{domain}_metrics_seedavg.csv")[
        ["target", metric]].assign(model="Fine-tuned")
    combined = pd.concat([individual, pretrained, finetuned], ignore_index=True)
    if domain == "rangeland" and metric == "RMSE":
        # Same display-only per-day -> per-month rescale as Figure 4 (see
        # RANGELAND_DAY_TO_MONTH) -- applied once here so it covers all three model sources
        # (individual/pretrained/finetuned) consistently.
        combined[metric] = combined[metric] * RANGELAND_DAY_TO_MONTH
    # draw_metric_boxplot_panel groups via `sorted(unique())`; an ordered Categorical makes
    # that read Individual -> Pretrained -> Fine-tuned instead of alphabetical.
    combined["model"] = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
    return combined


def figure6_model_comparison(metric: str = "RMSE") -> None:
    domain_data = {d: _domain_combined(d, metric) for d in DOMAINS}
    # This figure's wider panels use single-line Amazon labels (make_figure7's
    # AMAZON_TARGET_LABELS) -- _domain_combined() deliberately leaves Amazon's target names
    # raw so each figure can apply its own.
    domain_data["amazon"]["target"] = domain_data["amazon"]["target"].map(AMAZON_TARGET_LABELS)
    # sorted(unique()) matches how draw_metric_boxplot_panel itself orders targets, so panel
    # order here is identical to that ordering.
    domain_targets = {d: sorted(df["target"].unique()) for d, df in domain_data.items()}
    max_targets = max(len(t) for t in domain_targets.values())

    content_w = max_targets * PANEL_W + (max_targets - 1) * GAP_COLS
    left_stack = MARGIN + ROWLABEL_COL_W + TICK_CLEAR
    fig_w = left_stack + content_w + METRIC_GAP + METRIC_COL_W + MARGIN_R
    fig_h = MARGIN_TOP + 3 * PANEL_H + 2 * GAP_ROWS + MARGIN_BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))

    patch_handles: list = []
    cursor = MARGIN_TOP
    for domain in DOMAINS:
        targets = domain_targets[domain]
        row_w = len(targets) * PANEL_W + (len(targets) - 1) * GAP_COLS
        row_left = left_stack + (content_w - row_w) / 2
        left = row_left
        for ti, target in enumerate(targets):
            rect = _rect(fig_w, fig_h, left, cursor, PANEL_W, PANEL_H)
            ax = fig.add_axes(rect)
            sub = domain_data[domain][domain_data[domain]["target"] == target]
            # Thin, small-multiple-style boxes -- a full-width box adds no extra information
            # over a thin one here (each panel only ever holds 3 boxes).
            draw_metric_boxplot_panel(ax, sub, metric, group_col="model",
                                       box_width_frac=0.5, group_span=0.55)
            ax.set_title("")
            # Force compact offset notation (e.g. "5.0" + "x10^2" instead of "500") only for
            # RMSE, whose units genuinely span large magnitudes across targets. NSE/KGE/PBIAS
            # are conventionally read as plain values (NSE/KGE bounded near [-1, 1] and always
            # shown to 2 decimals -- see shared/plots.py's _format_median; PBIAS as a %) --
            # forcing an offset there made already-compact numbers (e.g. "0.90", "-14") less
            # legible, not more.
            if metric == "RMSE":
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.yaxis.get_offset_text().set_fontsize(6)
            if metric in ("NSE", "KGE"):
                # NSE/KGE are upper-bounded at 1 (a perfect score). Rather than taking
                # whatever grid MaxNLocator lands on and bolting a "1.0" tick onto it (which
                # could sit right on top of the nearest auto tick, crowding that end of the
                # axis), generate the grid anchored AT 1 to begin with: shift the view so 1.0
                # becomes 0, let MaxNLocator pick its usual nice round step around that zero
                # (it always includes 0 when 0 is in range), then shift back and drop
                # anything past 1.0 (the auto-locator's own headroom above the top box can
                # overshoot it).
                ylo, yhi = ax.get_ylim()
                shifted = MaxNLocator(nbins=4).tick_values(ylo - 1.0, yhi - 1.0)
                ax.set_yticks(sorted(t + 1.0 for t in shifted if t <= 1e-9))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis="y", labelsize=6, pad=1)
            _add_grid(ax)
            _horizontal_xticks(ax)
            # Target name + unit as two separately-sized lines (a single tick label can't mix
            # font sizes) -- the unit is genuinely informative (e.g. distinguishing Amazon's
            # discharge in m^3/s from a flux in g C/m^2/d) but shouldn't compete visually with
            # the target name, so it's set smaller and in parentheses underneath.
            ax.set_xticklabels([])
            ax.text(0.5, -0.1, target, transform=ax.transAxes, ha="center", va="top", fontsize=8)
            unit = UNITS_BY_DOMAIN[domain].get(target)
            if unit:
                ax.text(0.5, -0.2, f"({unit})", transform=ax.transAxes, ha="center", va="top",
                        fontsize=6, color="dimgrey")
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            if not patch_handles:
                patch_handles = [p for p in ax.patches if p.get_label() and not p.get_label().startswith("_")]
            left += PANEL_W + GAP_COLS

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
    for metric in ("RMSE", "NSE", "PBIAS", "KGE"):
        figure6_model_comparison(metric)


if __name__ == "__main__":
    main()
