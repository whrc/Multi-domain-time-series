"""
Ablation study — RMSE/NSE/PBIAS/KGE comparison figures, one file per metric, one row per
domain (Arctic/Amazon/Rangeland), one small-multiple panel per target -- same per-target
layout (and the same reasoning) as figures/scripts/make_figure6.py and
metric_decomposition/decompose_kge.py's combined figure: a single domain-level grouped boxplot
squashes small-magnitude/tightly-bounded targets next to large-swing ones (e.g. Amazon's
Discharge NSE, always 0-1, next to Burned area's NSE, which swings to -4), and forces one
redundant "arm" legend per row that ate ~20% of the figure's width for the same 4 colors
repeated three times.

See ablation_test/ablation_description.md for the full hypotheses and experiment design.
Reuses shared/plots.py's draw_metric_boxplot_panel (the same primitive behind the paper's own
Figure 4/6) and figures/scripts/_common.py's style/seedavg helpers, so these figures visually
match the rest of the project's figure set.

Produces the 5-seed average only -- the more robust one to cite, per project convention.

Each domain's arm set differs in exactly which second/third domain is paired in (e.g. Amazon's
"+ Rangeland"/"+ Arctic" vs. Rangeland's "+ Amazon"/"+ Arctic"), so each row keeps its own
compact legend rather than one shared figure-level legend across mismatched text (color
position *is* consistent across rows -- Individual/+X/+Y/Full 3-domain always land on the same
4 PALETTE colors -- only the label text differs).
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "figures" / "scripts"))
from shared.plots import draw_metric_boxplot_panel  # noqa: E402
from _common import AMAZON_TARGET_LABELS, _add_grid, _load_seedavg, _style  # noqa: E402
from make_figure7 import _rect  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DPI = 300
METRICS = ["RMSE", "NSE", "PBIAS", "KGE"]
METRIC_FILE_SUFFIX = {"RMSE": "a", "NSE": "b", "PBIAS": "c", "KGE": "d"}
ROW_LETTER = {"arctic": "(a)", "amazon": "(b)", "rangeland": "(c)"}
# Same display-only per-day -> per-month RMSE rescale as figures/scripts/_common.py's
# RANGELAND_DAY_TO_MONTH, so these numbers stay consistent with the paper's own Figure 4/6
# rather than silently diverging from it.
RANGELAND_DAY_TO_MONTH = 30

MD_EVAL = REPO_ROOT / "outputs" / "multi_domain" / "evaluation"


def _amazon_individual() -> pd.DataFrame:
    return _load_seedavg(REPO_ROOT / "outputs/amazon_domain/evaluation_seedavg/metrics_test_seedavg.csv")


def _rangeland_individual() -> pd.DataFrame:
    """"Individual" for Rangeland is the real, hyperparameter-tuned production model
    (hidden_dim=256, dropout=0.15 — see hyperparameter_tuning/hyperparameter_tuning_description.md
    "Resolution" and key_findings_log.md RG-retune0812). Prior to 2026-08-12 this loaded a
    stand-in (--amazon-sized, borrowing amazon_domain's architecture) because Rangeland had
    never been properly tuned and its original config (152K params, dropout=0.3) was known to
    be capacity-starved — that stand-in and the --capacity-matched control (now dropped for
    both Amazon and Rangeland) are superseded by having a real tuned baseline; see
    ablation_description.md's update note."""
    df = _load_seedavg(REPO_ROOT / "outputs/rangeland_domain/evaluation_fluxonly_seedavg/metrics_test_seedavg.csv")
    return df.assign(target=df["target"].str.replace("_predicted", "", regex=False))


def _arctic_individual() -> pd.DataFrame:
    return _load_seedavg(REPO_ROOT / "outputs/arctic_domain/evaluation/500K_s400_fluxonly_seedavg/metrics_test_seedavg.csv")


def _md(dom_pair: str, domain: str) -> pd.DataFrame:
    return _load_seedavg(MD_EVAL / f"pretrained_fluxonly_dom-{dom_pair}_seedavg" / domain / f"{domain}_metrics_seedavg.csv")


def _anchor(domain: str) -> pd.DataFrame:
    """Matched-seed anchor (full 3-domain pretrain) — an existing production artifact from the
    5-seed publication sweep, reused here rather than rerun (see
    ablation_test/ablation_description.md § "Matched-seed anchor")."""
    return _load_seedavg(MD_EVAL / "pretrained_fluxonly_seedavg" / domain / f"{domain}_metrics_seedavg.csv")


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
        df = loader()[["target", "RMSE", "NSE", "KGE", "PBIAS"]].copy()
        df["arm"] = label
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    if domain == "rangeland":
        combined["RMSE"] = combined["RMSE"] * RANGELAND_DAY_TO_MONTH
    if domain == "amazon":
        combined["target"] = combined["target"].map(AMAZON_TARGET_LABELS)
    # draw_metric_boxplot_panel draws an ordered Categorical in category order (not
    # alphabetical) -- see shared/plots.py.
    combined["arm"] = pd.Categorical(combined["arm"], categories=arm_order, ordered=True)
    return combined


# ── Layout constants (inches) -- mirrors figures/scripts/make_figure6.py's manual inch-based
# placement (fixed panel size regardless of target count, narrower rows centered under the
# widest row), plus a per-row legend column since arm label text differs by domain. ──
MARGIN, MARGIN_R = 0.10, 0.06
ROWLABEL_COL_W = 0.20
TICK_CLEAR = 0.32
LEGEND_GAP, LEGEND_COL_W = 0.10, 1.05
MARGIN_TOP, MARGIN_BOTTOM = 0.30, 0.30
PANEL_W, PANEL_H = 1.05, 1.35
GAP_COLS = 0.30
GAP_ROWS = 0.45


def make_figure(metric: str, domain_frames: dict[str, pd.DataFrame]) -> None:
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
        for target in targets:
            rect = _rect(fig_w, fig_h, left, cursor, PANEL_W, PANEL_H)
            ax = fig.add_axes(rect)
            sub = domain_frames[domain][domain_frames[domain]["target"] == target]
            draw_metric_boxplot_panel(ax, sub, metric, group_col="arm",
                                      box_width_frac=0.6, group_span=0.75)
            ax.set_title(target, fontsize=8)
            if metric == "RMSE":
                # Only Discharge/Burned-area-scale targets need this, but applying it
                # unconditionally is harmless -- matplotlib no-ops when the range doesn't
                # warrant an offset.
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.yaxis.get_offset_text().set_fontsize(6)
            if metric in ("NSE", "KGE"):
                # NSE/KGE are upper-bounded at 1 (a perfect score) -- anchor the tick grid AT
                # 1 rather than letting the auto-locator bolt one on wherever it lands (see
                # figures/scripts/make_figure6.py's identical logic/comment).
                ylo, yhi = ax.get_ylim()
                shifted = MaxNLocator(nbins=4).tick_values(ylo - 1.0, yhi - 1.0)
                ax.set_yticks(sorted(t + 1.0 for t in shifted if t <= 1e-9))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xticks([])  # arm identity comes from the legend, not per-box x labels
            ax.tick_params(axis="y", labelsize=6, pad=1)
            _add_grid(ax)
            ax.spines[["top", "right"]].set_visible(False)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            if not patch_handles:
                patch_handles = [p for p in ax.patches if p.get_label() and not p.get_label().startswith("_")]
            left += PANEL_W + GAP_COLS

        row_label_y = (fig_h - cursor - PANEL_H / 2) / fig_h
        row_label_x = (row_left - TICK_CLEAR - ROWLABEL_COL_W / 2) / fig_w
        fig.text(row_label_x, row_label_y, f"{ROW_LETTER[domain]} {domain.capitalize()}",
                 fontsize=8, fontweight="bold", rotation=90, ha="center", va="center")

        legend_x = (row_left + row_w + LEGEND_GAP) / fig_w
        fig.legend(patch_handles, [p.get_label() for p in patch_handles],
                  loc="center left", bbox_to_anchor=(legend_x, row_label_y),
                  frameon=True, fancybox=False, fontsize=6.5, handlelength=1.1,
                  handleheight=1.0, borderpad=0.4, labelspacing=0.35)

        cursor += PANEL_H + GAP_ROWS

    path = FIGURES_DIR / f"ablation_comparison_{metric}_seedavg.png"
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
                **{f"{m}_median": g[m].median() for m in ["RMSE", "NSE", "KGE", "PBIAS"]},
            })
    summary = pd.DataFrame(summary_rows).round(3)
    summary_path = FIGURES_DIR / "ablation_summary_metrics_seedavg.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    for metric in METRICS:
        make_figure(metric, domain_frames)


if __name__ == "__main__":
    main()
