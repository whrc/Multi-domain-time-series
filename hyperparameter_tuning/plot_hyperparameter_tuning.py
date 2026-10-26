"""Plot best validation loss vs. hidden_dim per domain; rewrites hyperparameter_tuning_winners.yaml to match.

Run after run_hyperparameter_tuning.py (and Rangeland's xlarge/xxlarge probes).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDY_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))
from config.config import load_config  # noqa: E402
from hyperparameter_tuning.run_hyperparameter_tuning import (  # noqa: E402
    DOMAIN_CONFIG_NAMES, DOMAINS, SIZES, best_val_loss,
)

POINT_COLOR = "#56B4E9"     # Okabe-Ito sky blue

DOMAIN_LABELS = {"arctic": "(a) Arctic", "amazon": "(b) Amazon", "rangeland": "(c) Rangeland"}
# Rangeland extended upward, Amazon downward (hidden_dim 32/16) beyond the standard 3 sizes.
DOMAIN_SIZES = {
    "arctic": SIZES,
    "amazon": ["xxsmall", "xsmall", *SIZES],
    "rangeland": [*SIZES, "xlarge", "xxlarge"],
}
# Sizes kept in the CSV/winner selection but not plotted.
PLOT_EXCLUDE = {"rangeland": {"small"}, "amazon": {"xxsmall"}}


def _style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "axes.linewidth": 0.8,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="Which seed's tuning results to plot (matches run_hyperparameter_"
                             "tuning.py's winner-selection seed).")
    args = parser.parse_args()
    _style()

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(3.0 * len(DOMAINS), 4.2), squeeze=False)
    rows = []
    winners: dict[str, str] = {}
    for ax, domain in zip(axes[0], DOMAINS):
        sizes = DOMAIN_SIZES[domain]
        domain_cfg = load_config(DOMAIN_CONFIG_NAMES[domain])
        hidden_dims = [domain_cfg[f"model_{size}"]["hidden_dim"] for size in sizes]
        losses = [best_val_loss(domain, size, args.seed) for size in sizes]
        winner_idx = min(range(len(sizes)), key=lambda i: losses[i])
        winners[domain] = sizes[winner_idx]
        for size, hd, loss in zip(sizes, hidden_dims, losses):
            rows.append({"domain": domain, "size": size, "hidden_dim": hd,
                        "best_val_loss": round(loss, 4), "winner": size == winners[domain]})

        exclude = PLOT_EXCLUDE.get(domain, set())
        plot_sizes = [s for s in sizes if s not in exclude]
        plot_hidden_dims = [domain_cfg[f"model_{s}"]["hidden_dim"] for s in plot_sizes]
        plot_losses = [best_val_loss(domain, s, args.seed) for s in plot_sizes]

        ax.plot(plot_hidden_dims, plot_losses, color=POINT_COLOR, marker="o", markersize=9,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)

        ax.set_title(DOMAIN_LABELS[domain])
        ax.set_xlabel("Hidden dimension size")
        if ax is axes[0][0]:
            ax.set_ylabel("Best validation loss")
        ax.set_xscale("log", base=2)
        ax.minorticks_off()
        ax.set_xticks(plot_hidden_dims)
        ax.set_xticklabels([str(h) for h in plot_hidden_dims])
        # y from 0 so noise-level differences don't look large.
        ax.set_ylim(0, max(plot_losses) * 1.15)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    figs_dir = STUDY_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figs_dir / "hyperparameter_tuning_results.png"
    svg_path = fig_path.with_suffix(".svg")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")
    print(f"Saved figure: {svg_path}")

    csv_path = figs_dir / "hyperparameter_tuning_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved summary: {csv_path}")

    winners_path = STUDY_DIR / "hyperparameter_tuning_winners.yaml"
    with winners_path.open("w") as f:
        yaml.safe_dump(winners, f, sort_keys=False)
    print(f"Wrote winners: {winners_path}: {winners}")


if __name__ == "__main__":
    main()
