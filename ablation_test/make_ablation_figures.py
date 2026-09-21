"""
Ablation study — KGE comparison figure (1 file), one row per domain (Arctic/Amazon/Rangeland),
one small-multiple panel per target -- same per-target layout (and the same reasoning) as
figures/scripts/make_figure6.py and metric_decomposition/decompose_kge.py's combined figure: a
single domain-level grouped boxplot squashes small-magnitude/tightly-bounded targets next to
large-swing ones, and forces one redundant "arm" legend per row that ate ~20% of the figure's
width for the same 4 colors repeated three times.

See ablation_test/ablation_description.md for the full hypotheses and experiment design.
Reuses shared/plots.py's draw_metric_boxplot_panel (the same primitive behind the paper's own
Figure 4/6) and figures/scripts/_common.py's style/seedavg helpers, so this figure visually
matches the rest of the project's figure set.

Produces the 5-seed average only -- the more robust one to cite, per project convention.
Finetune-stage only (not pretrain-stage) -- directly comparable to the paper's headline
Individual-vs-multi-domain-finetuned production numbers. KGE only (not RMSE/NSE/PBIAS) -- the
metric used going forward for this study.

Each domain's arm set differs in exactly which second/third domain is paired in (e.g. Amazon's
"+ Rangeland"/"+ Arctic" vs. Rangeland's "+ Amazon"/"+ Arctic"), so each row keeps its own
compact legend rather than one shared figure-level legend across mismatched text (color
position *is* consistent across rows -- Individual/+X/+Y/Full 3-domain always land on the same
4 PALETTE colors -- only the label text differs).

Arctic's and Rangeland's rows each share one y-axis across their panels (their targets sit in a
similarly-bounded KGE range within the domain; Amazon's targets don't, so it keeps independent
per-panel scales). This does NOT use matplotlib's ax.sharey(): sharing an axis *before* drawing
a boxplot onto it is broken here, because shared/plots.py's draw_metric_boxplot_panel calls an
explicit ax.set_ylim(...) internally (natural autoscale + a fixed headroom fraction for its
median-value text labels) -- once a later panel is ax.sharey()'d to an earlier one that already
had set_ylim() called on it, that later panel's own internal "read current ylim, add headroom"
logic reads the EARLIER panel's already-frozen range instead of its own data, silently clipping
whichever panel happens to need a wider range (this is exactly what was cutting off Arctic
RECO's real whisker before this fix). Instead: draw every panel in the row independently
(unlinked, so each one's own internal autoscale is correct), capture each one's own resulting
ylim afterward, compute the row's shared range from that, then explicitly overwrite every
panel's ylim with the shared range -- no reliance on matplotlib's shared-axis autoscale timing.
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
from make_figure7 import AMAZON_TARGET_LABELS, _rect  # noqa: E402  (single-line "Active fire count", not _common.py's wrapped 2-line version)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DPI = 300
METRIC = "KGE"
ROW_LETTER = {"arctic": "(a)", "amazon": "(b)", "rangeland": "(c)"}
# Both toggleable, off for now for a cleaner look -- matches figures/scripts/make_figure4.py
# and make_figure6.py's identical toggles -- flip to True to bring either back.
SHOW_GRID = False
SHOW_MEDIAN_LABELS = False

MD_EVAL = REPO_ROOT / "outputs" / "multi_domain" / "evaluation"


def _amazon_individual() -> pd.DataFrame:
    return _load_seedavg(REPO_ROOT / "outputs/amazon_domain/evaluation_seedavg/metrics_test_seedavg.csv")


def _rangeland_individual() -> pd.DataFrame:
    """"Individual" for Rangeland is the real, hyperparameter-tuned production model
    (hidden_dim=256, dropout=0.15 — see hyperparameter_tuning/hyperparameter_tuning_description.md
    "Resolution"). This previously loaded a stand-in (--amazon-sized, borrowing amazon_domain's
    architecture) because Rangeland had never been properly tuned and its original config
    (152K params, dropout=0.3) was known to be capacity-starved — that stand-in and the
    --capacity-matched control (now dropped for both Amazon and Rangeland) are superseded by
    having a real tuned baseline; see ablation_description.md's "Current status" section."""
    df = _load_seedavg(REPO_ROOT / "outputs/rangeland_domain/evaluation_fluxonly_seedavg/metrics_test_seedavg.csv")
    return df.assign(target=df["target"].str.replace("_predicted", "", regex=False))


def _arctic_individual() -> pd.DataFrame:
    return _load_seedavg(REPO_ROOT / "outputs/arctic_domain/evaluation/500K_s400_fluxonly_seedavg/metrics_test_seedavg.csv")


def _md(dom_pair: str, domain: str) -> pd.DataFrame:
    return _load_seedavg(MD_EVAL / f"finetuned_fluxonly_dom-{dom_pair}_seedavg" / domain / f"{domain}_metrics_seedavg.csv")


def _anchor(domain: str) -> pd.DataFrame:
    """Matched-seed anchor (full 3-domain, finetuned) — an existing production artifact from
    the 5-seed publication sweep, reused here rather than rerun (see
    ablation_test/ablation_description.md § "Matched-seed anchor")."""
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
MARGIN_TOP, MARGIN_BOTTOM = 0.15, 0.42
PANEL_W, PANEL_H = 1.05, 1.35
GAP_COLS = 0.30
GAP_ROWS = 0.48
# Arctic's GPP/RECO and Rangeland's 4 targets each sit in a similarly-bounded KGE range within
# the domain (Amazon's don't -- Discharge/Active-fire-count/Burned-area swing wildly
# differently), so sharing one y-axis per row removes redundant near-identical tick-label
# columns at no cost in legibility, same convention as
# metric_decomposition/decompose_kge.py's SHARED_Y_DOMAINS.
SHARED_Y_DOMAINS = {"arctic", "rangeland"}
# Rangeland's floor: every real (Tukey-fenced) whisker across all 4 targets/arms sits at
# >=0.548 (checked against the actual per-station finetuned metrics), so 0.5 has a real, if
# thin, margin and doesn't clip anything.
# Arctic has no fixed floor -- unlike Rangeland its true range genuinely varies a lot between
# GPP and RECO (RECO's Individual arm has a real whisker down to -0.315), so the floor is
# computed from the row's own drawn data instead of a hand-picked constant (see
# make_figure()'s two-pass sharing logic).
RANGELAND_YFLOOR = 0.5
# Top for both shared rows: real whisker tops never exceed ~0.99 in either domain (checked).
# No median-value text is drawn above boxes (SHOW_MEDIAN_LABELS=False), so this only needs a
# small margin above the true max, not extra headroom for that text.
SHARED_Y_TOP = 1.02
# Amazon's Burned area panel: the Individual arm's true whisker bottom is -7.77 (Tukey
# 1.5xIQR-fenced, NOT a lone outlier -- checked against
# outputs/amazon_domain/evaluation_seedavg/metrics_test_seedavg.csv: 4 stations sit between
# -3.9 and -7.77; the one true extreme outlier, -18.6, is already correctly excluded by the
# whisker rule and isn't drawn). Flooring the view just below that real whisker tip removes
# wasted empty space below the actual data without hiding anything.
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
            ax.set_title("")  # clear draw_metric_boxplot_panel's own internal ax.set_title(metric)
                              # ("KGE") -- redundant now that the legend carries a "KGE" title
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
            # Each panel was drawn independently above (see module docstring for why NOT
            # ax.sharey() before drawing), so each ax's own ylim right now correctly reflects
            # only its own data -- union them for the row's shared floor.
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
