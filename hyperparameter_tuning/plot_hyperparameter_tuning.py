"""
Plots the hyperparameter-tuning sweep: best validation loss vs. actual hidden_dim value, one
panel per domain, highlighting the winner (lowest validation loss). Winner selection is
recomputed here (not just read from hyperparameter_tuning_winners.yaml) so the figure and
that file can never drift apart, then the file is rewritten to match.

See hyperparameter_tuning/hyperparameter_tuning_description.md.
Run after run_hyperparameter_tuning.py (and, for Rangeland's extra "xlarge" probe, after that
one has also been trained/evaluated — see hyperparameter_tuning_description.md's "Rangeland
extension" note).
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

# Okabe-Ito color-blind-safe palette (shared/plots.py convention)
WINNER_COLOR = "#009E73"    # bluish green
OTHER_COLOR = "#56B4E9"     # sky blue

DOMAIN_LABELS = {"arctic": "Arctic", "amazon": "Amazon", "rangeland": "Rangeland"}
# Rangeland's sweep was extended past the standard 3 sizes (see
# hyperparameter_tuning_description.md) after small/medium/large showed a monotonic,
# non-plateauing improvement -- every other domain uses the standard 3-size grid.
DOMAIN_SIZES = {"arctic": SIZES, "amazon": SIZES, "rangeland": [*SIZES, "xlarge"]}


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

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(4.3 * len(DOMAINS), 4.2), squeeze=False)
    rows = []
    winners: dict[str, str] = {}
    for ax, domain in zip(axes[0], DOMAINS):
        sizes = DOMAIN_SIZES[domain]
        domain_cfg = load_config(DOMAIN_CONFIG_NAMES[domain])
        hidden_dims = [domain_cfg[f"model_{size}"]["hidden_dim"] for size in sizes]
        losses = [best_val_loss(domain, size, args.seed) for size in sizes]
        winner_idx = min(range(len(sizes)), key=lambda i: losses[i])
        winners[domain] = sizes[winner_idx]

        ax.plot(hidden_dims, losses, color=OTHER_COLOR, marker="o", markersize=9,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3,
                label="Other candidate")
        ax.plot(hidden_dims[winner_idx], losses[winner_idx], color=WINNER_COLOR, marker="o",
                markersize=15, markeredgecolor="white", markeredgewidth=1.5, zorder=5,
                linestyle="None", label="Selected (lowest validation loss)")

        ax.set_title(DOMAIN_LABELS[domain])
        ax.set_xlabel("Hidden dimension size")
        if ax is axes[0][0]:
            ax.set_ylabel("Best validation loss (masked MSE)")
        ax.set_xticks(hidden_dims)
        # y-axis starts at 0 (not auto-scaled to the data range) so line height is
        # proportional to the real relative difference between sizes -- an auto-scaled axis
        # zoomed into a tiny data range makes noise-level differences look as dramatic as a
        # real, large drop between sizes.
        ax.set_ylim(0, max(losses) * 1.15)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for size, hd, loss in zip(sizes, hidden_dims, losses):
            rows.append({"domain": domain, "size": size, "hidden_dim": hd,
                        "best_val_loss": round(loss, 4), "winner": size == winners[domain]})

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 0.0))
    # Reserve fixed top/bottom margins first (title block + legend), then place both title
    # lines inside that reserved space -- doing this after tight_layout (rather than via its
    # own rect+suptitle interaction, which reliably overlaps here) keeps the two text
    # elements from colliding regardless of figure width.
    fig.tight_layout(rect=[0, 0.06, 1, 0.86])
    fig.suptitle("Individual-model hyperparameter sweep: validation loss vs. hidden dimension",
                fontsize=14, fontweight="bold", y=0.97)
    fig.text(0.5, 0.90, f"Minimum validation loss selects the winning architecture size (seed {args.seed})",
             ha="center", fontsize=10, color="dimgrey")

    figs_dir = STUDY_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figs_dir / "hyperparameter_tuning_results.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    csv_path = figs_dir / "hyperparameter_tuning_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved summary: {csv_path}")

    winners_path = STUDY_DIR / "hyperparameter_tuning_winners.yaml"
    with winners_path.open("w") as f:
        yaml.safe_dump(winners, f, sort_keys=False)
    print(f"Wrote winners: {winners_path}: {winners}")


if __name__ == "__main__":
    main()
