"""
Ablation study — RMSE/NSE/PBIAS comparison figures.

See ablation_test/ablation_description.md for the full hypotheses and experiment design. One
PNG per metric (RMSE/NSE/PBIAS, matching figures/scripts/make_remaining_figures.py's Figure 6
convention), each with 3 stacked rows (Arctic/Amazon/Rangeland), each row a grouped boxplot of
that domain's ablation arms. Reuses shared/plots.py's draw_metric_boxplot_panel (the same
primitive behind the paper's own Figure 4/6) and make_remaining_figures.py's style helpers, so
these figures visually match the rest of the project's figure set.

Unlike Figure 6, each domain's arm set differs (Arctic has no "Capacity-matched" arm — its
individual config already matches the shared trunk's capacity — and no domain can be "paired
with itself"), so this script relies on draw_metric_boxplot_panel's own per-axis legend rather
than building one shared figure-level legend across mismatched arm sets.
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
from make_remaining_figures import AMAZON_TARGET_LABELS, _add_grid, _horizontal_xticks, _style  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DPI = 300
METRICS = ["RMSE", "NSE", "PBIAS"]  # KGE computed for the summary CSV but not plotted, matching
                                    # Figure 6's own 3-metric scope
# Same display-only per-day -> per-month RMSE rescale as make_remaining_figures.py's
# RANGELAND_DAY_TO_MONTH, so these numbers stay consistent with the paper's own Figure 4/6
# rather than silently diverging from it.
RANGELAND_DAY_TO_MONTH = 30

MD_EVAL = REPO_ROOT / "outputs" / "multi_domain" / "evaluation"


def _amazon_individual() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "outputs/amazon_domain/evaluation_seed1/metrics_test.csv")


def _amazon_capmatched() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "outputs/amazon_domain/evaluation_seed1_capmatched/metrics_test.csv")


def _rangeland_individual() -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "outputs/rangeland_domain/evaluation_fluxonly_seed1/metrics_test.csv")
    return df.assign(target=df["target"].str.replace("_predicted", "", regex=False))


def _rangeland_capmatched() -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "outputs/rangeland_domain/evaluation_fluxonly_seed1_capmatched/metrics_test.csv")
    return df.assign(target=df["target"].str.replace("_predicted", "", regex=False))


def _arctic_individual() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "outputs/arctic_domain/evaluation/500K_s400_fluxonly_seed1/metrics_test.csv")


def _md(dom_pair: str, domain: str) -> pd.DataFrame:
    return pd.read_csv(MD_EVAL / f"pretrained_fluxonly_dom-{dom_pair}_seed1" / domain / f"{domain}_metrics.csv")


def _anchor(domain: str) -> pd.DataFrame:
    """Matched-seed anchor (full 3-domain pretrain, seed=1) — an existing production artifact
    from the 5-seed publication sweep, reused here rather than rerun (see
    ablation_test/ablation_description.md § "Matched-seed anchor")."""
    return pd.read_csv(MD_EVAL / "pretrained_fluxonly_seed1" / domain / f"{domain}_metrics.csv")


ARMS = {
    "amazon": [
        ("Individual", _amazon_individual),
        ("Capacity-matched", _amazon_capmatched),
        ("+ Rangeland", lambda: _md("amazon-rangeland", "amazon")),
        ("+ Arctic", lambda: _md("amazon-arctic", "amazon")),
        ("Full 3-domain", lambda: _anchor("amazon")),
    ],
    "rangeland": [
        ("Individual", _rangeland_individual),
        ("Capacity-matched", _rangeland_capmatched),
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

DOMAIN_ROWS = [("arctic", "Arctic"), ("amazon", "Amazon"), ("rangeland", "Rangeland")]


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


def main() -> None:
    _style()
    FIGURES_DIR.mkdir(exist_ok=True)
    domain_frames = {domain: build_domain_frame(domain) for domain, _ in DOMAIN_ROWS}

    summary_rows = []
    for domain, df in domain_frames.items():
        for (target, arm), g in df.groupby(["target", "arm"], observed=True):
            summary_rows.append({
                "domain": domain, "target": target, "arm": arm, "n": len(g),
                **{f"{m}_median": g[m].median() for m in ["RMSE", "NSE", "KGE", "PBIAS"]},
            })
    summary = pd.DataFrame(summary_rows).round(3)
    summary_path = FIGURES_DIR / "ablation_summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    for metric in METRICS:
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.5))
        for ax, (domain, title) in zip(axes, DOMAIN_ROWS):
            draw_metric_boxplot_panel(ax, domain_frames[domain], metric, group_col="arm",
                                      box_width_frac=0.7, group_span=0.85)
            ax.set_title(title)
            ax.set_ylabel(metric)
            _add_grid(ax)
            _horizontal_xticks(ax)
            # draw_metric_boxplot_panel's own legend uses matplotlib's "best"-location auto
            # placement, which can land on top of a box (seen on the Rangeland/PBIAS panel) --
            # pin it outside the axes instead so it never overlaps data in any panel.
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title="arm", fontsize="small", loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        fig.tight_layout()
        path = FIGURES_DIR / f"ablation_comparison_{metric}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
