"""
KGE decomposition — for each domain's individual target variable (no aggregation across
targets), breaks KGE into its three components (r = correlation, alpha = variability ratio,
beta = bias ratio; see shared/metrics.py::kge_components) and compares Individual vs.
Multi-domain fine-tuned, so it's visible *which* component moved when multi-domain training
changed a target's skill, not just that KGE moved. (Pretrained is intentionally omitted from
the plot/summary for simplicity -- it tracks fine-tuned closely everywhere in this project, see
e.g. key_findings_log.md's repeated "pretrained ~= finetuned" observation.)

A fourth bar group, KGE itself, sits alongside the three components in every panel (bold tick
label) so the composite skill change reads directly off the same panel as the components that
explain it -- no need to cross-reference Figure 6/7's separate KGE panel. KGE shares
r/alpha/beta's "1 = perfect" reference line by construction
(KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)), so it plots on the same axis without
rescaling.

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
from _common import DOMAINS, _domain_combined, _style  # noqa: E402
from make_figure7 import AMAZON_TARGET_LABELS, _rect  # noqa: E402  (single-line "Active fire count")

ROW_LETTER = {"arctic": "(a)", "amazon": "(b)", "rangeland": "(c)"}

STUDY_DIR = Path(__file__).resolve().parent
COMPONENTS = ["r", "alpha", "beta", "KGE"]
COMPONENT_LABELS = ["r", r"$\alpha$", r"$\beta$", "KGE"]
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
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_title(target, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)


def make_domain_figure(domain: str, table: pd.DataFrame) -> None:
    targets = sorted(table[table["domain"] == domain]["target"].unique())
    # Independent y-scale per panel (not shared) -- Amazon's burned_area beta IQR reaches ~6
    # while its other targets sit near 0-1.5; sharing an axis would flatten those into
    # illegibility. Every panel keeps its own tick labels for the same reason.
    fig, axes = plt.subplots(1, len(targets), figsize=(1.75 * len(targets), 1.9), squeeze=False)
    for ax, target in zip(axes[0], targets):
        _target_panel(ax, table, domain, target)
    axes[0][0].set_ylabel("Median (IQR)", fontsize=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
              ncol=2, frameon=True, fancybox=False, fontsize=7, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    figs_dir = STUDY_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    path = figs_dir / f"kge_decomposition_{domain}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Combined figure (all 3 domains, one row each) -- layout mirrors
# figures/scripts/make_figure6.py's manual inch-based placement: fixed panel size regardless
# of how many targets a domain has, narrower rows centered under the widest row. No per-row
# "metric" label column here (unlike Figure 6, which is one metric per whole figure) since
# each panel already shows all 3 components together. ──
MARGIN, MARGIN_R = 0.10, 0.06
ROWLABEL_COL_W = 0.18
TICK_CLEAR = 0.24
MARGIN_TOP, MARGIN_BOTTOM = 0.62, 0.28  # top: legend + first row's own panel titles
PANEL_W, PANEL_H = 1.55, 1.5
GAP_COLS = 0.34  # matches make_figure6.py's own value -- wide enough that a panel's own
                 # y-tick numbers (independent per-panel scale, see make_domain_figure's own
                 # comment) never touch its left neighbor's right spine
GAP_ROWS = 0.5   # room for one row's xtick labels + the next row's panel titles
# Rangeland's 4 targets all sit in a tight 0.8-1.05 band (unlike Amazon's wildly different
# per-target scales, see make_domain_figure) -- sharing one y-axis across its row removes 3
# redundant near-identical tick-label columns and the crowding they caused, at no cost in
# legibility. Every other row keeps its own independent per-panel scale.
SHARED_Y_DOMAINS = {"rangeland"}


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

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=True, fancybox=False, fontsize=7, handlelength=1.2, columnspacing=1.0)

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
    for domain in DOMAINS:
        make_domain_figure(domain, table)
    make_combined_figure(table)


if __name__ == "__main__":
    main()
