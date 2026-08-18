"""
KGE decomposition — for each domain's individual target variable (no aggregation across
targets), breaks KGE into its three components (r = correlation, alpha = variability ratio,
beta = bias ratio; see shared/metrics.py::kge_components) and compares Individual vs.
Multi-domain fine-tuned, so it's visible *which* component moved when multi-domain training
changed a target's skill, not just that KGE moved. (Pretrained is intentionally omitted from
the plot/summary for simplicity -- it tracks fine-tuned closely everywhere in this project, see
e.g. key_findings_log.md's repeated "pretrained ~= finetuned" observation.)

Only r/alpha/beta are plotted -- KGE itself is deliberately NOT a fourth bar group here, since
it's already shown in Figure 6d/7d; this figure is purely about which component explains a KGE
change, not about re-showing KGE.

One figure only: all 3 domains combined (one row each) -- no separate per-domain figures, since
the combined figure already contains every panel the per-domain ones did.

Zero retraining -- reads the same seedavg metrics_test_seedavg.csv files Figure 6/7 already
use (regenerated once via each domain's 04_evaluate.py + run_seed_sweep.py --aggregate after
shared/metrics.py::compute_metrics started including r/alpha/beta -- pure re-evaluation of
already-trained checkpoints/predictions, no GPU training involved).

Error bars are the IQR (25th-75th percentile) across held-out units for that target -- same
convention as the boxplot whiskers in Figure 4/6/7 and the ablation figures, just collapsed to
a single bar+error-bar instead of a full box.

Exploratory pass over every target -- not all of them need to end up in the manuscript; see
metric_decomposition_description.md.

Run standalone:
    .venv/bin/python metric_decomposition/decompose_kge.py
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_SCRIPTS = REPO_ROOT / "figures" / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FIGURES_SCRIPTS))

from shared.plots import PALETTE  # noqa: E402
from _common import (  # noqa: E402
    DOMAINS, _add_grid, _apply_kge_ticks, _domain_combined, _horizontal_xticks, _style,
)
from make_figure6 import LEGEND_LABELS  # noqa: E402  (same "Domain-specific"/"Multi-domain (finetuned)" wording as Figure 6/7's legend)
from make_figure7 import AMAZON_TARGET_LABELS, _rect  # noqa: E402  (single-line "Active fire count")

ROW_LETTER = {"arctic": "(a)", "amazon": "(b)", "rangeland": "(c)"}
# Toggleable, off for now for a cleaner look -- matches figures/scripts/make_figure4.py,
# make_figure6.py, and ablation_test/make_ablation_figures.py's identical toggle.
SHOW_GRID = False

STUDY_DIR = Path(__file__).resolve().parent
COMPONENTS = ["r", "alpha", "beta"]
COMPONENT_LABELS = ["r", r"$\alpha$", r"$\beta$"]
PLOT_MODELS = ["Individual", "Fine-tuned"]
# Same Individual=orange / Fine-tuned=green mapping as Figure 6/7's legend (PALETTE[0]/[2] --
# PALETTE[1], sky blue, is reserved for "Pretrained" elsewhere and unused here since that arm
# is dropped from this plot).
MODEL_COLOR = {"Individual": PALETTE[0], "Fine-tuned": PALETTE[2]}
DPI = 300


def build_table() -> pd.DataFrame:
    """One row per (domain, target, model, component) -- median across held-out units, plus
    q25/q75 (interquartile range across those same units) for error bars."""
    rows = []
    for domain in DOMAINS:
        for component in COMPONENTS:
            combined = _domain_combined(domain, component)
            combined = combined[combined["model"].isin(PLOT_MODELS)]
            if domain == "amazon":
                combined["target"] = combined["target"].map(AMAZON_TARGET_LABELS)
            grouped = combined.groupby(["target", "model"], observed=True)[component]
            stats = grouped.agg(median="median", q25=lambda s: s.quantile(0.25),
                                q75=lambda s: s.quantile(0.75))
            for (target, model), row in stats.iterrows():
                rows.append({"domain": domain, "target": target, "model": model,
                            "component": component, "value": round(row["median"], 3),
                            "q25": round(row["q25"], 3), "q75": round(row["q75"], 3)})
    return pd.DataFrame(rows)


def _target_panel(ax: plt.Axes, table: pd.DataFrame, domain: str, target: str) -> None:
    sub = table[(table["domain"] == domain) & (table["target"] == target)]
    x = range(len(COMPONENTS))
    width = 0.28
    for mi, model in enumerate(PLOT_MODELS):
        rows = [sub[(sub["component"] == c) & (sub["model"] == model)].iloc[0] for c in COMPONENTS]
        vals = [r["value"] for r in rows]
        # Asymmetric error bars from the IQR (q25/q75 aren't equidistant from the median in
        # general) -- matplotlib's yerr wants [lower_length, upper_length], not the raw
        # quantile values, and neither can be negative even if q25 > value due to rounding.
        lower = [max(r["value"] - r["q25"], 0.0) for r in rows]
        upper = [max(r["q75"] - r["value"], 0.0) for r in rows]
        offsets = [xi + (mi - 0.5) * width for xi in x]
        ax.bar(offsets, vals, width=width, color=MODEL_COLOR[model], label=model,
              yerr=[lower, upper], capsize=2.0,
              error_kw={"elinewidth": 0.7, "ecolor": "black", "alpha": 0.6})
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.7, zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(COMPONENT_LABELS)
    ax.set_title(target, fontsize=8, pad=2)  # tight pad -- default (6pt) reads as if the title
                                              # belongs to whatever's above it, not this panel
    ax.spines[["top", "right"]].set_visible(False)
    # Matches figures/scripts/make_figure6.py's own panel styling (tick size/pad, horizontal
    # x-tick labels) -- y-ticks are finalized by the caller instead (via _apply_kge_ticks),
    # once this panel's ylim is truly final (after any Y_FLOOR/shared-axis override below).
    ax.tick_params(axis="both", labelsize=6, pad=1)
    _horizontal_xticks(ax)


# ── Combined figure (all 3 domains, one row each) -- layout mirrors
# figures/scripts/make_figure6.py's manual inch-based placement: fixed panel size regardless
# of how many targets a domain has, narrower rows centered under the widest row. No per-row
# "metric" label column here (unlike Figure 6, which is one metric per whole figure) since
# each panel already shows all 3 components together. Sized noticeably smaller/tighter than a
# first pass at this figure -- these panels only ever hold 2x3 bars, they don't need Figure 6's
# footprint. ──
MARGIN, MARGIN_R = 0.08, 0.05
ROWLABEL_COL_W = 0.16
TICK_CLEAR = 0.20
MARGIN_TOP, MARGIN_BOTTOM = 0.50, 0.22  # top: legend + first row's own panel titles
PANEL_W, PANEL_H = 1.2, 1.15
GAP_COLS = 0.26  # wide enough that a panel's own y-tick numbers (independent per-panel scale,
                 # see the SHARED_Y_DOMAINS comment below) never touch its left neighbor's
                 # right spine
GAP_ROWS = 0.38  # room for one row's xtick labels + the next row's panel titles
# Rangeland's 4 targets all sit in a tight 0.8-1.05 band (unlike Amazon's wildly different
# per-target scales) -- sharing one y-axis across its row removes 3 redundant near-identical
# tick-label columns and the crowding they caused, at no cost in legibility. Every other row
# keeps its own independent per-panel scale.
SHARED_Y_DOMAINS = {"rangeland"}
# Arctic's and Rangeland's r/alpha/beta values are always comfortably high (checked against
# metric_decomposition/figures/kge_decomposition_summary.csv: Arctic's lowest median/IQR bound
# across every target/component/model is 0.721, Rangeland's is 0.854), so starting either
# domain's y-axis at 0 wastes most of the panel on empty space below the actual data. Floors
# chosen with a healthy margin below each domain's real lowest value -- checked, nothing clips.
Y_FLOOR = {"arctic": 0.5, "rangeland": 0.7}


def make_combined_figure(table: pd.DataFrame) -> None:
    domain_targets = {d: sorted(table[table["domain"] == d]["target"].unique()) for d in DOMAINS}
    max_targets = max(len(t) for t in domain_targets.values())

    content_w = max_targets * PANEL_W + (max_targets - 1) * GAP_COLS
    left_stack = MARGIN + ROWLABEL_COL_W + TICK_CLEAR
    fig_w = left_stack + content_w + MARGIN_R
    fig_h = MARGIN_TOP + 3 * PANEL_H + 2 * GAP_ROWS + MARGIN_BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))

    handles: list = []
    labels: list = []
    cursor = MARGIN_TOP
    for domain in DOMAINS:
        targets = domain_targets[domain]
        row_w = len(targets) * PANEL_W + (len(targets) - 1) * GAP_COLS
        row_left = left_stack + (content_w - row_w) / 2
        left = row_left
        first_ax = None
        for target in targets:
            rect = _rect(fig_w, fig_h, left, cursor, PANEL_W, PANEL_H)
            ax = fig.add_axes(rect)
            if domain in SHARED_Y_DOMAINS and first_ax is not None:
                ax.sharey(first_ax)
            _target_panel(ax, table, domain, target)
            if domain in Y_FLOOR:
                # Set on every panel, not just the row's first one -- correct regardless of
                # whether this domain also shares one y-axis across its row (SHARED_Y_DOMAINS)
                # or keeps independent per-panel scales (a shared axis propagates the same
                # limit to every panel linked to it either way).
                ax.set_ylim(bottom=Y_FLOOR[domain])
            # Ticks are finalized here, after any Y_FLOOR override above, not inside
            # _target_panel -- an explicit set_yticks() (unlike a plain locator) is a one-time
            # snapshot of the current ylim, so it has to run after the panel's ylim is truly
            # final. bounded_above=False: unlike KGE, r/alpha/beta have no hard ceiling at 1.0
            # (see e.g. Amazon's beta, which reaches ~5.6) -- 1.0 is only their reference value.
            _apply_kge_ticks(ax, bounded_above=False)
            if SHOW_GRID:
                _add_grid(ax)
            if domain in SHARED_Y_DOMAINS:
                if first_ax is None:
                    first_ax = ax
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)
            if not handles:
                handles, labels = ax.get_legend_handles_labels()
            left += PANEL_W + GAP_COLS

        row_label_y = (fig_h - cursor - PANEL_H / 2) / fig_h
        row_label_x = (row_left - TICK_CLEAR - ROWLABEL_COL_W / 2) / fig_w
        fig.text(row_label_x, row_label_y, f"{ROW_LETTER[domain]} {domain.capitalize()}",
                 fontsize=8, fontweight="bold", rotation=90, ha="center", va="center")
        cursor += PANEL_H + GAP_ROWS

    fig.legend(handles, [LEGEND_LABELS[l] for l in labels], loc="upper center",
              bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True, fancybox=False, fontsize=7,
              handlelength=1.2, columnspacing=1.0)

    figs_dir = STUDY_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    path = figs_dir / "kge_decomposition_all_domains.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    _style()
    plt.rcParams.update({"font.size": 7})
    table = build_table()
    figs_dir = STUDY_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figs_dir / "kge_decomposition_summary.csv"
    table.to_csv(csv_path, index=False)
    print(f"Saved summary: {csv_path}")
    make_combined_figure(table)


if __name__ == "__main__":
    main()
