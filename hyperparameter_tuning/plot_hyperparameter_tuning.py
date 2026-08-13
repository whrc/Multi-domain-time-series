"""
Plots the hyperparameter-tuning sweep: best validation loss vs. actual hidden_dim value, one
panel per domain. All points share one color -- the line shape makes the best point (or lack of
one, for Amazon's flat response) obvious without needing a separate highlight. Winner selection
is still computed here (not just read from hyperparameter_tuning_winners.yaml) so that file and
the CSV summary can never drift apart, then the file is rewritten to match -- it just isn't
visually marked in the plot anymore.

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
POINT_COLOR = "#56B4E9"     # sky blue -- single color, no winner highlight (see module docstring)

DOMAIN_LABELS = {"arctic": "(a) Arctic", "amazon": "(b) Amazon", "rangeland": "(c) Rangeland"}
# Rangeland's sweep was extended past the standard 3 sizes (see hyperparameter_tuning_
# description.md) after small/medium/large showed a monotonic, non-plateauing improvement.
# Amazon's was extended downward (xsmall/xxsmall, hidden_dim=32/16) after small/medium/large
# came back flat (noise-level, ~0.2% spread) to check where -- if anywhere -- performance
# actually drops below 64. Arctic uses the standard 3-size grid, untouched.
DOMAIN_SIZES = {
    "arctic": SIZES,
    "amazon": ["xxsmall", "xsmall", *SIZES],
    "rangeland": [*SIZES, "xlarge", "xxlarge"],
}
# Sizes tested (and still recorded in the CSV summary / winner selection) but dropped from
# the plot itself -- Rangeland's "small" sits on top of "medium" (both 0.042) and adds
# nothing visually once "xlarge"/"xxlarge" are also on the same axis. Amazon's "xxsmall" (16)
# is excluded as too far below any plausible production size to be visually useful.
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

        # Plotted subset can drop sizes that add nothing visually (see PLOT_EXCLUDE) without
        # affecting the winner (computed above over the full sizes list) or the CSV summary
        # (built above, also over the full list).
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
        # Log-scale x-axis (same convention as figures/scripts/make_figure3.py's dataset-size
        # panel): hidden_dims are powers of two, so a linear axis crowds the small values
        # together once the range is wide (Rangeland now spans 64-512, 8x) -- log-spacing
        # keeps every tick legibly separated regardless of how wide the sweep gets.
        ax.set_xscale("log", base=2)
        ax.minorticks_off()
        ax.set_xticks(plot_hidden_dims)
        ax.set_xticklabels([str(h) for h in plot_hidden_dims])
        # y-axis starts at 0 (not auto-scaled to the data range) so line height is
        # proportional to the real relative difference between sizes -- an auto-scaled axis
        # zoomed into a tiny data range makes noise-level differences look as dramatic as a
        # real, large drop between sizes.
        ax.set_ylim(0, max(plot_losses) * 1.15)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

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
