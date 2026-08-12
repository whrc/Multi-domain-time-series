"""
KGE decomposition — for each domain's individual target variable (no aggregation across
targets), breaks KGE into its three components (r = correlation, alpha = variability ratio,
beta = bias ratio; see shared/metrics.py::kge_components) and compares Individual /
Multi-domain-pretrained / Multi-domain-fine-tuned, so it's visible *which* component moved
when multi-domain training changed a target's skill, not just that KGE moved.

Zero retraining -- reads the same seedavg metrics_test_seedavg.csv files Figure 6/7 already
use (regenerated once via each domain's 04_evaluate.py + run_seed_sweep.py --aggregate after
shared/metrics.py::compute_metrics started including r/alpha/beta -- pure re-evaluation of
already-trained checkpoints/predictions, no GPU training involved).

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
from _common import DOMAINS, MODEL_ORDER, _domain_combined, _style  # noqa: E402
from make_figure7 import AMAZON_TARGET_LABELS  # noqa: E402  (single-line "Active fire count")

STUDY_DIR = Path(__file__).resolve().parent
COMPONENTS = ["r", "alpha", "beta"]
MODEL_COLOR = {"Individual": PALETTE[0], "Pretrained": PALETTE[1], "Fine-tuned": PALETTE[2]}


def build_table() -> pd.DataFrame:
    """One row per (domain, target, model, component) -- median across held-out units."""
    rows = []
    for domain in DOMAINS:
        for component in COMPONENTS:
            combined = _domain_combined(domain, component)
            if domain == "amazon":
                combined["target"] = combined["target"].map(AMAZON_TARGET_LABELS)
            medians = combined.groupby(["target", "model"], observed=True)[component].median()
            for (target, model), value in medians.items():
                rows.append({"domain": domain, "target": target, "model": model,
                            "component": component, "value": round(value, 3)})
    return pd.DataFrame(rows)


def _target_panel(ax: plt.Axes, table: pd.DataFrame, domain: str, target: str) -> None:
    sub = table[(table["domain"] == domain) & (table["target"] == target)]
    x = range(len(COMPONENTS))
    width = 0.25
    for mi, model in enumerate(MODEL_ORDER):
        vals = [sub[(sub["component"] == c) & (sub["model"] == model)]["value"].iloc[0] for c in COMPONENTS]
        offsets = [xi + (mi - 1) * width for xi in x]
        ax.bar(offsets, vals, width=width, color=MODEL_COLOR[model], label=model)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(COMPONENTS)
    ax.set_title(target, fontsize=9)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)


def make_domain_figure(domain: str, table: pd.DataFrame) -> None:
    targets = sorted(table[table["domain"] == domain]["target"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(3.2 * len(targets), 3.2), squeeze=False)
    for ax, target in zip(axes[0], targets):
        _target_panel(ax, table, domain, target)
    axes[0][0].set_ylabel("median value (1 = perfect)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
              ncol=3, frameon=True, fancybox=False, fontsize=8)
    fig.suptitle(f"{domain.capitalize()} — KGE decomposition (r, alpha, beta) by target", y=1.16, fontsize=10)
    fig.tight_layout()

    figs_dir = STUDY_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    path = figs_dir / f"kge_decomposition_{domain}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    _style()
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
