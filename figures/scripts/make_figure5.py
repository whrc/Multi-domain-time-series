"""
Figure 5: Per-seed training-curve figures, two files -- fig5a = training loss, fig5b =
validation loss (kept separate, not combined into one busy panel) -- each with two rows:
(a) individual domain models' own 5-seed curves, (b) multi-domain model's pretrain-then-
fine-tune 5-seed curves per domain, with a star marking each seed's own pretrain-to-
fine-tune transition (seeds early-stop pretraining at different epochs, so there's no
single shared transition point to draw as a vertical divider).
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from _common import DOMAIN_COLOR, DOMAINS, MD_EVAL_DIR, SEEDS, _add_grid, _save, _style  # noqa: E402

# Per-seed history.csv location for each standalone individual-domain model -- mirrors each
# domain's own naming convention (Arctic: labeled subfolder; Rangeland/Amazon: suffixed
# sibling dir).
INDIVIDUAL_HISTORY_PATH = {
    "arctic": lambda s: REPO_ROOT / f"outputs/arctic_domain/evaluation/500K_s400_fluxonly_seed{s}/history.csv",
    "rangeland": lambda s: REPO_ROOT / f"outputs/rangeland_domain/evaluation_fluxonly_seed{s}/history.csv",
    "amazon": lambda s: REPO_ROOT / f"outputs/amazon_domain/evaluation_seed{s}/history.csv",
}


def _plot_training_curves(loss_kind: str) -> plt.Figure:
    """Two panels, one line per seed per domain in each (15 lines/panel; loss_kind selects
    'train' or 'val' -- the two are kept in separate figures rather than one busy panel):
    (a) standalone individual-domain models, each seed's own single training run.
    (b) multi-domain model: Stage 1 (joint pretraining) directly followed by that same
        seed's Stage 2 (per-domain fine-tuning), each seed's own pretrain segment ending
        exactly where its own history.csv ends (early stopping fires at a different epoch
        per seed, so there's no single shared pretrain/fine-tune boundary -- a star marks
        each seed's own transition instead of a shared vertical divider, which would
        misrepresent every seed but the one it happened to line up with)."""
    loss_col = f"{loss_kind}_loss"  # individual + finetune history.csv column name

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [0.8, 1.2]})

    for d in DOMAINS:
        for i, s in enumerate(SEEDS):
            hist = pd.read_csv(INDIVIDUAL_HISTORY_PATH[d](s)).dropna(subset=[loss_col])
            # dropna is required, not cosmetic: eval_every_n_epochs=2 for Arctic means
            # val_loss is NaN on every other row, so with the NaN rows left in, no two
            # consecutive points are ever both valid and matplotlib silently draws nothing.
            ax1.plot(hist["epoch"], hist[loss_col], color=DOMAIN_COLOR[d], alpha=0.55,
                     linewidth=1.0, label=d.capitalize() if i == 0 else None)

    for d in DOMAINS:
        for s in SEEDS:
            pretrain = pd.read_csv(MD_EVAL_DIR / f"pretrained_fluxonly_seed{s}" / "history.csv")
            stage1_end = pretrain["epoch"].max()  # last logged epoch = this seed's own stop point
            ft = pd.read_csv(MD_EVAL_DIR / f"finetuned_fluxonly_seed{s}" / d / "history.csv")
            x = pd.concat([pretrain["epoch"], stage1_end + ft["epoch"]])
            y = pd.concat([pretrain[f"{loss_kind}_{d}"], ft[loss_col]])
            ax2.plot(x, y, color=DOMAIN_COLOR[d], alpha=0.55, linewidth=1.0)
            ax2.plot(stage1_end, pretrain[f"{loss_kind}_{d}"].iloc[-1], marker="*", markersize=7,
                     color=DOMAIN_COLOR[d], markeredgecolor="black", markeredgewidth=0.4, zorder=5)

    # Fixed relative positions (not tied to any seed's actual epoch, since seeds stop at
    # different epochs -- see docstring) -- purely descriptive labels, not a boundary line.
    ax2.text(0.25, 0.93, "Stage 1: Joint pretraining", transform=ax2.transAxes, ha="center",
             va="top", fontsize=6.5, fontweight="bold")
    ax2.text(0.75, 0.93, "Stage 2: Per-domain fine-tuning", transform=ax2.transAxes, ha="center",
             va="top", fontsize=6.5, fontweight="bold")

    loss_label = "Training" if loss_kind == "train" else "Validation"
    ax1.set_title("(a) Individual domain models", loc="left", fontsize=8, fontweight="bold")
    ax2.set_title("(b) Multi-domain model", loc="left", fontsize=8, fontweight="bold")
    ax1.set_ylabel(f"{loss_label} MSE Loss")
    ax2.set_ylabel(f"{loss_label} MSE Loss")
    ax2.set_xlabel("Epoch")
    _add_grid(ax1)
    _add_grid(ax2)

    handles, labels = ax1.get_legend_handles_labels()
    star_handle = plt.Line2D([], [], marker="*", markersize=7, color="grey",
                             markeredgecolor="black", markeredgewidth=0.4, linestyle="None")
    # In panel (a)'s own empty upper-right space, not a separate figure-level legend below --
    # panel (a)'s curves flatten out well before the right edge, leaving room.
    ax1.legend(handles + [star_handle], labels + ["Pretraining stopped"], loc="upper right",
              frameon=True, fancybox=False, fontsize=6)
    fig.tight_layout()
    return fig


def figure5_training_curves() -> None:
    _save(_plot_training_curves("train"), "fig5a_multidomain_training_curves_train.png")
    _save(_plot_training_curves("val"), "fig5b_multidomain_training_curves_val.png")


def main() -> None:
    _style()
    figure5_training_curves()


if __name__ == "__main__":
    main()
