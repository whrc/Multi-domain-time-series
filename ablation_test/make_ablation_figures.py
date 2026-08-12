"""
Ablation study — RMSE/NSE/PBIAS/KGE comparison figures.

See ablation_test/ablation_description.md for the full hypotheses and experiment design. One
PNG per metric, matching the paper's own former grouped-by-domain Figure 6 convention -- Figure
6 itself has since moved to a per-target layout, see figures/scripts/make_figure6.py, which also
gained a 4th KGE panel -- this script now matches that (was RMSE/NSE/PBIAS only). Each PNG has 3
stacked rows (Arctic/Amazon/Rangeland), each row a grouped boxplot of that domain's ablation
arms. Reuses shared/plots.py's draw_metric_boxplot_panel (the same primitive behind the paper's
own Figure 4/6) and figures/scripts/_common.py's style/seedavg helpers, so these figures
visually match the rest of the project's figure set.

Produces the 5-seed average only (files suffixed "_seedavg", built from
ablation_test/aggregate_ablation_seeds.py's output plus each domain's own existing production
seedavg artifacts for the Individual/Full-3-domain arms) -- the more robust one to cite, per
project convention (see e.g. figures/scripts/_common.py). A seed=1-only variant existed
alongside it through 2026-08-12 and was dropped as clutter now that all 5 seeds are available
for every arm.

Unlike Figure 6, each domain's arm set differs (no domain has a "Capacity-matched" arm as of
2026-08-12 — see ablation_description.md's update note — and no domain can be "paired with
itself"), so this script relies on draw_metric_boxplot_panel's own per-axis legend rather than
building one shared figure-level legend across mismatched arm sets.
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
from _common import (  # noqa: E402
    AMAZON_TARGET_LABELS, _add_grid, _horizontal_xticks, _load_seedavg, _style,
)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DPI = 300
METRICS = ["RMSE", "NSE", "PBIAS", "KGE"]
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
    summary_path = FIGURES_DIR / "ablation_summary_metrics_seedavg.csv"
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
        path = FIGURES_DIR / f"ablation_comparison_{metric}_seedavg.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
