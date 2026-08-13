"""
Combines Amazon's four sequential architecture-dimension sweeps (hidden_dim, feedforward_dim,
num_layers, dropout -- see hyperparameter_tuning_description.md's "Amazon extension"/"Amazon
feedforward_dim sweep"/"Amazon num_layers sweep"/"Amazon dropout sweep" sections and
key_findings_log.md's HP-amazonext0813/HP-amazonffn0813/HP-amazonlayers0813/
HP-amazondropout0813) into one minimal, publication-styled figure. Each sweep fixes every other
dimension at the previous sweep's marginal-best point, so this reads left-to-right as the actual
search path taken, not four independent grids.

All four came back flat (0.511-0.520 across every cell) -- a shared y-axis across all 4 panels
makes that flatness directly comparable at a glance, the same convention
plot_hyperparameter_tuning.py uses (y-axis anchored at 0, not auto-scaled to the data range).
All points share one color -- the flat line shape makes the lack of a real winner obvious
without needing a separate highlight.

Run standalone (after all four sweeps' cells are trained/evaluated):
    .venv/bin/python hyperparameter_tuning/plot_amazon_architecture_search.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from hyperparameter_tuning.run_hyperparameter_tuning import best_val_loss  # noqa: E402

POINT_COLOR = "#56B4E9"     # sky blue -- single color, no winner highlight (see module docstring)

# (panel label, x-axis label, x-log-scale, [(x-value, model-size key), ...])
PANELS = [
    # 16 (xxsmall) excluded from display -- too far below any plausible production size to be
    # visually useful here; still recorded in HP-amazonext0813/hyperparameter_tuning_winners.yaml.
    ("(a) Hidden dim", "Hidden dim", True,
     [(32, "xsmall"), (64, "small"), (128, "medium"), (256, "large")]),
    ("(b) Feedforward dim", "Feedforward dim", True,
     [(128, "ffn_narrow"), (256, "ffn_std"), (512, "small")]),
    ("(c) Num. layers", "Num. layers", False,
     [(2, "layers2"), (3, "ffn_std"), (4, "layers4"), (6, "layers6")]),
    ("(d) Dropout", "Dropout", False,
     [(0.10, "dropout10"), (0.15, "ffn_std"), (0.20, "dropout20"), (0.30, "dropout30")]),
]


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.3,
        "axes.linewidth": 0.7,
    })


def main() -> None:
    _style()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(2.0 * len(PANELS), 1.9), sharey=True)

    all_losses = []
    for ax, (label, xlabel, logscale, points) in zip(axes, PANELS):
        xs = [x for x, _ in points]
        losses = [best_val_loss("amazon", size, 1) for _, size in points]
        all_losses.extend(losses)

        ax.plot(xs, losses, color=POINT_COLOR, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)

        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.set_xlabel(xlabel)
        if logscale:
            ax.set_xscale("log", base=2)
            ax.minorticks_off()
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], rotation=0)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Best validation loss")
    # Shared y-axis, anchored at 0 (not auto-scaled) -- same convention as
    # plot_hyperparameter_tuning.py, and the point of this figure: it makes the flatness across
    # all four dimensions directly visible rather than each panel auto-zooming into its own
    # noise band and looking like 4 separate stories.
    axes[0].set_ylim(0, max(all_losses) * 1.15)

    fig.tight_layout(w_pad=1.0)

    figs_dir = STUDY_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    path = figs_dir / "amazon_architecture_search.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
