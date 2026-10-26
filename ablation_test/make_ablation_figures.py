"""Ablation KGE figure: one row per domain, one panel per target, finetuned 5-seed averages.

Shared-y rows set ylim after drawing each panel independently: ax.sharey() before drawing clips
panels because draw_metric_boxplot_panel calls set_ylim internally.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "figures" / "scripts"))
from shared.plots import draw_metric_boxplot_panel  # noqa: E402
from _common import _add_grid, _apply_kge_ticks, _load_seedavg, _style  # noqa: E402
from make_figure7 import AMAZON_TARGET_LABELS, _rect  # noqa: E402  (single-line Amazon labels)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DPI = 300
METRIC = "KGE"
ROW_LETTER = {"arctic": "(a)", "amazon": "(b)", "rangeland": "(c)"}
SHOW_GRID = False
SHOW_MEDIAN_LABELS = False

MD_EVAL = REPO_ROOT / "outputs" / "multi_domain" / "evaluation"


def _amazon_individual() -> pd.DataFrame:
    return _load_seedavg(REPO_ROOT / "outputs/amazon_domain/evaluation_seedavg/metrics_test_seedavg.csv")


def _rangeland_individual() -> pd.DataFrame:
    df = _load_seedavg(REPO_ROOT / "outputs/rangeland_domain/evaluation_fluxonly_seedavg/metrics_test_seedavg.csv")
    return df.assign(target=df["target"].str.replace("_predicted", "", regex=False))


def _arctic_individual() -> pd.DataFrame:
    return _load_seedavg(REPO_ROOT / "outputs/arctic_domain/evaluation/500K_s400_fluxonly_seedavg/metrics_test_seedavg.csv")


def _md(dom_pair: str, domain: str) -> pd.DataFrame:
    return _load_seedavg(MD_EVAL / f"finetuned_fluxonly_dom-{dom_pair}_seedavg" / domain / f"{domain}_metrics_seedavg.csv")


def _anchor(domain: str) -> pd.DataFrame:
    """Full 3-domain finetuned production seedavg (reused, not rerun)."""
    return _load_seedavg(MD_EVAL / "finetuned_fluxonly_seedavg" / domain / f"{domain}_metrics_seedavg.csv")


ARMS = {
    "amazon": [
        ("Individual", _amazon_individual),
        ("+ Rangeland", lambda: _md("amazon-rangeland", "amazon")),
        ("+ Arctic", lambda: _md("amazon-arctic", "amazon")),
        ("Full 3-domain", lambda: _anchor("amazon")),
    ],
    "rangeland": [
        ("Individual", _rangeland_individual),
        ("+ Amazon", lambda: _md("amazon-rangeland", "rangeland")),
        ("+ Arctic", lambda: _md("arctic-rangeland", "rangeland")),
        ("Full 3-domain", lambda: _anchor("rangeland")),
    ],
    "arctic": [
        ("Individual", _arctic_individual),
        ("+ Amazon", lambda: _md("amazon-arctic", "arctic")),
        ("+ Rangeland", lambda: _md("arctic-rangeland", "arctic")),
        ("Full 3-domain", lambda: _anchor("arctic")),
    ],
}


DOMAINS = ["arctic", "amazon", "rangeland"]


def build_domain_frame(domain: str) -> pd.DataFrame:
    arm_order = [label for label, _ in ARMS[domain]]
    frames = []
    for label, loader in ARMS[domain]:
        df = loader()[["target", METRIC]].copy()
        df["arm"] = label
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    if domain == "amazon":
        combined["target"] = combined["target"].map(AMAZON_TARGET_LABELS)
    # Ordered Categorical sets the box order in draw_metric_boxplot_panel.
    combined["arm"] = pd.Categorical(combined["arm"], categories=arm_order, ordered=True)
    return combined


# Layout constants (inches)
MARGIN, MARGIN_R = 0.10, 0.06
ROWLABEL_COL_W = 0.20
TICK_CLEAR = 0.32
LEGEND_GAP, LEGEND_COL_W = 0.10, 1.05
MARGIN_TOP, MARGIN_BOTTOM = 0.15, 0.42
PANEL_W, PANEL_H = 1.05, 1.35
GAP_COLS = 0.30
GAP_ROWS = 0.48
# Rows whose targets share a similar KGE range get one y-axis; Amazon's ranges differ too much.
SHARED_Y_DOMAINS = {"arctic", "rangeland"}
# Rangeland whiskers all >= 0.548; Arctic's floor is computed from the drawn data.
RANGELAND_YFLOOR = 0.5
# Whisker tops <= ~0.99 in both shared rows.
SHARED_Y_TOP = 1.02
# Burned area: lowest whisker is -7.77 (Individual arm); outliers below are not drawn.
AMAZON_YFLOOR = {"Burned area": -8}


def make_figure(domain_frames: dict[str, pd.DataFrame]) -> None:
    domain_targets = {d: sorted(domain_frames[d]["target"].unique()) for d in DOMAINS}
    max_targets = max(len(t) for t in domain_targets.values())

    content_w = max_targets * PANEL_W + (max_targets - 1) * GAP_COLS
    left_stack = MARGIN + ROWLABEL_COL_W + TICK_CLEAR
    fig_w = left_stack + content_w + LEGEND_GAP + LEGEND_COL_W + MARGIN_R
    fig_h = MARGIN_TOP + 3 * PANEL_H + 2 * GAP_ROWS + MARGIN_BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))

    cursor = MARGIN_TOP
    for domain in DOMAINS:
        targets = domain_targets[domain]
        row_w = len(targets) * PANEL_W + (len(targets) - 1) * GAP_COLS
        row_left = left_stack + (content_w - row_w) / 2
        left = row_left
        patch_handles: list = []
        share_y = domain in SHARED_Y_DOMAINS
        row_axes: list[plt.Axes] = []
        for target in targets:
            rect = _rect(fig_w, fig_h, left, cursor, PANEL_W, PANEL_H)
            ax = fig.add_axes(rect)
            sub = domain_frames[domain][domain_frames[domain]["target"] == target]
            draw_metric_boxplot_panel(ax, sub, METRIC, group_col="arm",
                                      box_width_frac=0.6, group_span=0.75, zero_line=True,
                                      show_median_labels=SHOW_MEDIAN_LABELS)
            ax.set_title("")  # legend carries the metric title
            ax.set_xlabel(target, fontsize=8)
            if not share_y and target in AMAZON_YFLOOR:
                ax.set_ylim(bottom=AMAZON_YFLOOR[target])
                _apply_kge_ticks(ax)
            elif not share_y:
                _apply_kge_ticks(ax)
            ax.set_xticks([])  # arm identity comes from the legend, not per-box x labels
            ax.tick_params(axis="y", labelsize=6, pad=1)
            if SHOW_GRID:
                _add_grid(ax)
            ax.spines[["top", "right"]].set_visible(False)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            if not patch_handles:
                patch_handles = [p for p in ax.patches if p.get_label() and not p.get_label().startswith("_")]
            row_axes.append(ax)
            left += PANEL_W + GAP_COLS

        if share_y:
            # Panels were drawn unlinked, so each ylim reflects only its own data.
            floor = RANGELAND_YFLOOR if domain == "rangeland" else min(ax.get_ylim()[0] for ax in row_axes)
            for i, ax in enumerate(row_axes):
                ax.set_ylim(floor, SHARED_Y_TOP)
                _apply_kge_ticks(ax)
                if i > 0:
                    plt.setp(ax.get_yticklabels(), visible=False)

        row_label_y = (fig_h - cursor - PANEL_H / 2) / fig_h
        row_label_x = (row_left - TICK_CLEAR - ROWLABEL_COL_W / 2) / fig_w
        fig.text(row_label_x, row_label_y, f"{ROW_LETTER[domain]} {domain.capitalize()}",
                 fontsize=8, fontweight="bold", rotation=90, ha="center", va="center")

        legend_x = (row_left + row_w + LEGEND_GAP) / fig_w
        row_legend = fig.legend(patch_handles, [p.get_label() for p in patch_handles],
                  loc="center left", bbox_to_anchor=(legend_x, row_label_y),
                  frameon=True, fancybox=False, fontsize=6.5, handlelength=1.1,
                  handleheight=1.0, borderpad=0.4, labelspacing=0.35,
                  title=METRIC, title_fontsize=7)
        row_legend.get_title().set_fontweight("bold")

        cursor += PANEL_H + GAP_ROWS

    path = FIGURES_DIR / f"ablation_comparison_{METRIC}_finetuned_seedavg.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    _style()
    FIGURES_DIR.mkdir(exist_ok=True)
    domain_frames = {domain: build_domain_frame(domain) for domain in DOMAINS}

    summary_rows = []
    for domain, df in domain_frames.items():
        for (target, arm), g in df.groupby(["target", "arm"], observed=True):
            summary_rows.append({
                "domain": domain, "target": target, "arm": arm, "n": len(g),
                f"{METRIC}_median": g[METRIC].median(),
            })
    summary = pd.DataFrame(summary_rows).round(3)
    summary_path = FIGURES_DIR / "ablation_summary_metrics_finetuned_seedavg.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    make_figure(domain_frames)


if __name__ == "__main__":
    main()
