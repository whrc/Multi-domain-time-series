"""
KGE decomposition — for each domain's individual target variable (no aggregation across
targets), breaks KGE into its three components (r = correlation, alpha = variability ratio,
beta = bias ratio; see shared/metrics.py::kge_components) and compares Individual vs.
Multi-domain fine-tuned, so it's visible *which* component moved when multi-domain training
changed a target's skill, not just that KGE moved. (Pretrained is intentionally omitted from
the plot/summary for simplicity -- it tracks fine-tuned closely everywhere in this project, see
e.g. key_findings_log.md's repeated "pretrained ~= finetuned" observation.)

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
from make_figure7 import AMAZON_TARGET_LABELS  # noqa: E402  (single-line "Active fire count")

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
    width = 0.32
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
    ax.set_title(target, fontsize=8)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)


def make_domain_figure(domain: str, table: pd.DataFrame) -> None:
    targets = sorted(table[table["domain"] == domain]["target"].unique())
    # Independent y-scale per panel (not shared) -- Amazon's burned_area beta IQR reaches ~6
    # while its other targets sit near 0-1.5; sharing an axis would flatten those into
    # illegibility. Every panel keeps its own tick labels for the same reason.
    fig, axes = plt.subplots(1, len(targets), figsize=(1.55 * len(targets), 1.9), squeeze=False)
    for ax, target in zip(axes[0], targets):
        _target_panel(ax, table, domain, target)
    axes[0][0].set_ylabel("Median (IQR)", fontsize=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
              ncol=2, frameon=False, fontsize=7, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    figs_dir = STUDY_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    path = figs_dir / f"kge_decomposition_{domain}.png"
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


if __name__ == "__main__":
    main()
