"""
Figure 5 — per-seed loss curves (fig5a train, fig5b val): (a) individual models, (b) multi-domain
pretrain -> fine-tune, with a star at each seed's own pretraining stop epoch.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from _common import DOMAIN_COLOR, DOMAINS, MD_EVAL_DIR, SEEDS, _save, _style  # noqa: E402

INDIVIDUAL_HISTORY_PATH = {
    "arctic": lambda s: REPO_ROOT / f"outputs/arctic_domain/evaluation/500K_s400_fluxonly_seed{s}/history.csv",
    "rangeland": lambda s: REPO_ROOT / f"outputs/rangeland_domain/evaluation_fluxonly_seed{s}/history.csv",
    "amazon": lambda s: REPO_ROOT / f"outputs/amazon_domain/evaluation_seed{s}/history.csv",
}


def _plot_training_curves(loss_kind: str) -> plt.Figure:
    """loss_kind: 'train' or 'val'."""
    loss_col = f"{loss_kind}_loss"  # individual + finetune history.csv column name

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 4.5), sharex=True,
                                   gridspec_kw={"height_ratios": [0.8, 1.2]})

    for d in DOMAINS:
        for i, s in enumerate(SEEDS):
            hist = pd.read_csv(INDIVIDUAL_HISTORY_PATH[d](s)).dropna(subset=[loss_col])
            # dropna required: Arctic val_loss is NaN every other epoch, which breaks the line.
            ax1.plot(hist["epoch"], hist[loss_col], color=DOMAIN_COLOR[d], alpha=0.55,
                     linewidth=1.0, label=d.capitalize() if i == 0 else None)

    for d in DOMAINS:
        for s in SEEDS:
            pretrain = pd.read_csv(MD_EVAL_DIR / f"pretrained_fluxonly_seed{s}" / "history.csv")
            stage1_end = pretrain["epoch"].max()
            ft = pd.read_csv(MD_EVAL_DIR / f"finetuned_fluxonly_seed{s}" / d / "history.csv")
            x = pd.concat([pretrain["epoch"], stage1_end + ft["epoch"]])
            y = pd.concat([pretrain[f"{loss_kind}_{d}"], ft[loss_col]])
            ax2.plot(x, y, color=DOMAIN_COLOR[d], alpha=0.55, linewidth=1.0)
            ax2.plot(stage1_end, pretrain[f"{loss_kind}_{d}"].iloc[-1], marker="*", markersize=7,
                     color=DOMAIN_COLOR[d], markeredgecolor="black", markeredgewidth=0.4, zorder=5)

    # Fixed positions: seeds switch stages at different epochs.
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

    handles, labels = ax1.get_legend_handles_labels()
    star_handle = plt.Line2D([], [], marker="*", markersize=7, color="grey",
                             markeredgecolor="black", markeredgewidth=0.4, linestyle="None")
    ax1.legend(handles + [star_handle], labels + ["Pretraining stopped"], loc="upper right",
              frameon=True, fancybox=False, fontsize=6)
    fig.tight_layout(pad=0.4, h_pad=1.0)
    return fig


def figure5_training_curves() -> None:
    _save(_plot_training_curves("train"), "fig5a_multidomain_training_curves_train.png")
    _save(_plot_training_curves("val"), "fig5b_multidomain_training_curves_val.png")


def main() -> None:
    _style()
    figure5_training_curves()


if __name__ == "__main__":
    main()
