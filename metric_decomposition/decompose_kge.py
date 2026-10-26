"""KGE decomposition figure: median r/alpha/beta (IQR error bars) per target, Individual vs. Multi-domain finetuned.

Reads existing seedavg metrics CSVs (no retraining). Run: .venv/bin/python metric_decomposition/decompose_kge.py
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
from make_figure6 import LEGEND_LABELS  # noqa: E402
from make_figure7 import AMAZON_TARGET_LABELS, _rect  # noqa: E402  (single-line "Active fire count")

ROW_LETTER = {"arctic": "(a)", "amazon": "(b)", "rangeland": "(c)"}
SHOW_GRID = False

STUDY_DIR = Path(__file__).resolve().parent
COMPONENTS = ["r", "alpha", "beta"]
COMPONENT_LABELS = ["r", r"$\alpha$", r"$\beta$"]
PLOT_MODELS = ["Individual", "Fine-tuned"]
# Same colors as Figure 6/7 (PALETTE[1] is reserved for "Pretrained").
MODEL_COLOR = {"Individual": PALETTE[0], "Fine-tuned": PALETTE[2]}
DPI = 300


def build_table() -> pd.DataFrame:
    """One row per (domain, target, model, component): median and q25/q75 across held-out units."""
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
        # yerr takes lengths, not quantiles; clamp at 0 for rounding.
        lower = [max(r["value"] - r["q25"], 0.0) for r in rows]
        upper = [max(r["q75"] - r["value"], 0.0) for r in rows]
        offsets = [xi + (mi - 0.5) * width for xi in x]
        ax.bar(offsets, vals, width=width, color=MODEL_COLOR[model], label=model,
              yerr=[lower, upper], capsize=2.0,
              error_kw={"elinewidth": 0.7, "ecolor": "black", "alpha": 0.6})
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.7, zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(COMPONENT_LABELS)
    ax.set_title(target, fontsize=8, pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    # y-ticks are set by the caller once ylim is final.
    ax.tick_params(axis="both", labelsize=6, pad=1)
    _horizontal_xticks(ax)


# Layout constants (inches)
MARGIN, MARGIN_R = 0.08, 0.05
ROWLABEL_COL_W = 0.16
TICK_CLEAR = 0.20
MARGIN_TOP, MARGIN_BOTTOM = 0.50, 0.22  # top: legend + first row's own panel titles
PANEL_W, PANEL_H = 1.2, 1.15
GAP_COLS = 0.26  # room for per-panel y-tick labels
GAP_ROWS = 0.38  # room for one row's xtick labels + the next row's panel titles
# Rangeland targets all sit in 0.8-1.05, so its row shares one y-axis.
SHARED_Y_DOMAINS = {"rangeland"}
# Lowest median/IQR bound: Arctic 0.721, Rangeland 0.854.
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
                ax.set_ylim(bottom=Y_FLOOR[domain])
            # After Y_FLOOR (set_yticks snapshots ylim). r/alpha/beta can exceed 1 (e.g. Amazon beta ~5.6).
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
