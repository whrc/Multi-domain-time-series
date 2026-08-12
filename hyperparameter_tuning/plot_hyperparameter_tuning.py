"""
Plots the hyperparameter-tuning sweep: best validation loss vs. actual hidden_dim value, one
panel per domain, highlighting the winner recorded in hyperparameter_tuning_winners.yaml.

See hyperparameter_tuning/hyperparameter_tuning_description.md.
Run after run_hyperparameter_tuning.py.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="Which seed's tuning results to plot (matches run_hyperparameter_"
                             "tuning.py's winner-selection seed).")
    args = parser.parse_args()

    winners_path = STUDY_DIR / "hyperparameter_tuning_winners.yaml"
    with winners_path.open() as f:
        winners = yaml.safe_load(f)

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(4 * len(DOMAINS), 4), squeeze=False)
    rows = []
    for ax, domain in zip(axes[0], DOMAINS):
        domain_cfg = load_config(DOMAIN_CONFIG_NAMES[domain])
        hidden_dims = [domain_cfg[f"model_{size}"]["hidden_dim"] for size in SIZES]
        losses = [best_val_loss(domain, size, args.seed) for size in SIZES]
        ax.plot(hidden_dims, losses, color=OTHER_COLOR, marker="o", markersize=8, linewidth=2)
        winner_idx = SIZES.index(winners[domain])
        ax.plot(hidden_dims[winner_idx], losses[winner_idx], color=WINNER_COLOR,
                marker="o", markersize=13, zorder=5)
        ax.set_title(DOMAIN_LABELS[domain])
        ax.set_xlabel("hidden_dim")
        ax.set_ylabel("best validation loss")
        ax.set_xticks(hidden_dims)
        # y-axis starts at 0 (not auto-scaled to the data range) so bar/line height is
        # proportional to the real relative difference between sizes -- an auto-scaled axis
        # zoomed into a tiny data range makes noise-level differences look as dramatic as a
        # real, large small-vs-medium drop.
        ax.set_ylim(0, max(losses) * 1.15)
        ax.grid(alpha=0.3)
        for size, hd, loss in zip(SIZES, hidden_dims, losses):
            rows.append({"domain": domain, "size": size, "hidden_dim": hd,
                        "best_val_loss": round(loss, 4), "winner": size == winners[domain]})

    fig.suptitle(f"Hyperparameter tuning — best validation loss by hidden_dim (seed={args.seed})")
    fig.tight_layout()

    figs_dir = STUDY_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figs_dir / "hyperparameter_tuning_results.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    csv_path = figs_dir / "hyperparameter_tuning_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved summary: {csv_path}")


if __name__ == "__main__":
    main()
