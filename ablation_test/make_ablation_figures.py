"""
Ablation study — RMSE/NSE/PBIAS comparison figures.

See ablation_test/ablation_description.md for the full hypotheses and experiment design. One
PNG per metric (RMSE/NSE/PBIAS, matching the paper's own former grouped-by-domain Figure 6
convention -- Figure 6 itself has since moved to a per-target layout, see
figures/scripts/make_figure6.py), each with 3 stacked rows (Arctic/Amazon/Rangeland), each row
a grouped boxplot of that domain's ablation arms. Reuses shared/plots.py's
draw_metric_boxplot_panel (the same primitive behind the paper's own Figure 4/6) and
figures/scripts/_common.py's style/seedavg helpers, so these figures visually match the rest of
the project's figure set.

Produces TWO variants, side by side: seed=1 only (files with no suffix, unchanged from the
original single-seed run) and the 5-seed average (files suffixed "_seedavg", built from
ablation_test/aggregate_ablation_seeds.py's output plus each domain's own existing production
seedavg artifacts for the Individual/Full-3-domain arms) — see ablation_description.md for why
both are worth keeping rather than replacing one with the other.

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
from _common import (  # noqa: E402
    AMAZON_TARGET_LABELS, _add_grid, _horizontal_xticks, _load_seedavg, _style,
)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DPI = 300
METRICS = ["RMSE", "NSE", "PBIAS"]  # KGE computed for the summary CSV but not plotted here --
                                    # this study's own scope, unrelated to Figure 6/7's metric set
MODES = ["seed1", "seedavg"]
# Same display-only per-day -> per-month RMSE rescale as figures/scripts/_common.py's
# RANGELAND_DAY_TO_MONTH, so these numbers stay consistent with the paper's own Figure 4/6
# rather than silently diverging from it.
RANGELAND_DAY_TO_MONTH = 30

MD_EVAL = REPO_ROOT / "outputs" / "multi_domain" / "evaluation"


def _load(seed1_path: Path, seedavg_path: Path, mode: str) -> pd.DataFrame:
    return pd.read_csv(seed1_path) if mode == "seed1" else _load_seedavg(seedavg_path)


def _amazon_individual(mode: str) -> pd.DataFrame:
    return _load(REPO_ROOT / "outputs/amazon_domain/evaluation_seed1/metrics_test.csv",
                REPO_ROOT / "outputs/amazon_domain/evaluation_seedavg/metrics_test_seedavg.csv", mode)


def _amazon_capmatched(mode: str) -> pd.DataFrame:
    return _load(REPO_ROOT / "outputs/amazon_domain/evaluation_seed1_capmatched/metrics_test.csv",
                REPO_ROOT / "outputs/amazon_domain/evaluation_seedavg_capmatched/metrics_test_seedavg.csv", mode)


def _rangeland_individual(mode: str) -> pd.DataFrame:
    """"Individual" for Rangeland is the --amazon-sized architecture (597K params, matches
    amazon_domain's production config exactly, dropout included), NOT the original production
    config (152K params, "no grid search" per its own config comment). The original was found
    to be capacity-starved, not appropriately regularized — its train/val loss ratio was no
    better than the much bigger capacity-matched model's, it just couldn't fit the data well at
    all. amazon-sized recovers 53-84% of the capacity-matched gain with 8x fewer params and is
    an independently-validated size (Amazon's own production config), so it's the fairer
    "as-strong-as-reasonably-possible" individual baseline — see ablation_description.md and
    key_findings_log.md AB-capacitypairwise0806 for the full comparison."""
    df = _load(REPO_ROOT / "outputs/rangeland_domain/evaluation_fluxonly_seed1_amazonsized/metrics_test.csv",
              REPO_ROOT / "outputs/rangeland_domain/evaluation_fluxonly_seedavg_amazonsized/metrics_test_seedavg.csv", mode)
    return df.assign(target=df["target"].str.replace("_predicted", "", regex=False))


def _rangeland_capmatched(mode: str) -> pd.DataFrame:
    df = _load(REPO_ROOT / "outputs/rangeland_domain/evaluation_fluxonly_seed1_capmatched/metrics_test.csv",
              REPO_ROOT / "outputs/rangeland_domain/evaluation_fluxonly_seedavg_capmatched/metrics_test_seedavg.csv", mode)
    return df.assign(target=df["target"].str.replace("_predicted", "", regex=False))


def _arctic_individual(mode: str) -> pd.DataFrame:
    return _load(REPO_ROOT / "outputs/arctic_domain/evaluation/500K_s400_fluxonly_seed1/metrics_test.csv",
                REPO_ROOT / "outputs/arctic_domain/evaluation/500K_s400_fluxonly_seedavg/metrics_test_seedavg.csv", mode)


def _md(dom_pair: str, domain: str, mode: str) -> pd.DataFrame:
    return _load(MD_EVAL / f"pretrained_fluxonly_dom-{dom_pair}_seed1" / domain / f"{domain}_metrics.csv",
                MD_EVAL / f"pretrained_fluxonly_dom-{dom_pair}_seedavg" / domain / f"{domain}_metrics_seedavg.csv",
                mode)


def _anchor(domain: str, mode: str) -> pd.DataFrame:
    """Matched-seed anchor (full 3-domain pretrain) — an existing production artifact from the
    5-seed publication sweep, reused here rather than rerun (see
    ablation_test/ablation_description.md § "Matched-seed anchor")."""
    return _load(MD_EVAL / "pretrained_fluxonly_seed1" / domain / f"{domain}_metrics.csv",
                MD_EVAL / "pretrained_fluxonly_seedavg" / domain / f"{domain}_metrics_seedavg.csv", mode)


ARMS = {
    "amazon": [
        ("Individual", _amazon_individual),
        ("Capacity-matched", _amazon_capmatched),
        ("+ Rangeland", lambda mode: _md("amazon-rangeland", "amazon", mode)),
        ("+ Arctic", lambda mode: _md("amazon-arctic", "amazon", mode)),
        ("Full 3-domain", lambda mode: _anchor("amazon", mode)),
    ],
    "rangeland": [
        ("Individual", _rangeland_individual),
        ("Capacity-matched", _rangeland_capmatched),
        ("+ Amazon", lambda mode: _md("amazon-rangeland", "rangeland", mode)),
        ("+ Arctic", lambda mode: _md("arctic-rangeland", "rangeland", mode)),
        ("Full 3-domain", lambda mode: _anchor("rangeland", mode)),
    ],
    "arctic": [
        ("Individual", _arctic_individual),
        ("+ Amazon", lambda mode: _md("amazon-arctic", "arctic", mode)),
        ("+ Rangeland", lambda mode: _md("arctic-rangeland", "arctic", mode)),
        ("Full 3-domain", lambda mode: _anchor("arctic", mode)),
    ],
}

DOMAIN_ROWS = [("arctic", "Arctic"), ("amazon", "Amazon"), ("rangeland", "Rangeland")]


def build_domain_frame(domain: str, mode: str) -> pd.DataFrame:
    arm_order = [label for label, _ in ARMS[domain]]
    frames = []
    for label, loader in ARMS[domain]:
        df = loader(mode)[["target", "RMSE", "NSE", "KGE", "PBIAS"]].copy()
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


def run(mode: str) -> None:
    suffix = "" if mode == "seed1" else "_seedavg"
    domain_frames = {domain: build_domain_frame(domain, mode) for domain, _ in DOMAIN_ROWS}

    summary_rows = []
    for domain, df in domain_frames.items():
        for (target, arm), g in df.groupby(["target", "arm"], observed=True):
            summary_rows.append({
                "domain": domain, "target": target, "arm": arm, "n": len(g),
                **{f"{m}_median": g[m].median() for m in ["RMSE", "NSE", "KGE", "PBIAS"]},
            })
    summary = pd.DataFrame(summary_rows).round(3)
    summary_path = FIGURES_DIR / f"ablation_summary_metrics{suffix}.csv"
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
        path = FIGURES_DIR / f"ablation_comparison_{metric}{suffix}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def main() -> None:
    _style()
    FIGURES_DIR.mkdir(exist_ok=True)
    for mode in MODES:
        run(mode)


if __name__ == "__main__":
    main()
