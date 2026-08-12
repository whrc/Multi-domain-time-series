"""
Figure 3: Arctic sampling density and dataset-size sweep, two panels, GPP/RECO averaged across
SSP scenarios. Left: validation RMSE across capped sampling stride 50-500. Right: validation
RMSE vs. training-set size at the best stride (400) and staggered windowing.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.plots import PALETTE  # noqa: E402
from _common import _add_grid, _save, _style  # noqa: E402

ARCTIC_MODELS_DIR = REPO_ROOT / "outputs/arctic_domain/models"
STRIDES = [100, 150, 200, 250, 300, 350, 400, 500]
FLUX_TARGETS = ["GPP", "RECO"]


def figure3_arctic_sweep() -> None:
    """(a) val RMSE for GPP/RECO across capped stride 50-500 (50K windows), averaged across
    SSP scenarios. (b) val RMSE vs. train-set size (50K vs. 500K) at the winning stride=400,
    staggered windowing, also averaged across SSP scenarios."""
    sweep_rows = []
    for s in STRIDES:
        df = pd.read_csv(ARCTIC_MODELS_DIR / f"val_metrics_50K_s{s}.csv")
        df = df[df["target"].isin(FLUX_TARGETS)].copy()
        df["stride"] = s
        sweep_rows.append(df)
    sweep = pd.concat(sweep_rows, ignore_index=True)
    sweep_avg = sweep.groupby(["target", "stride"], as_index=False)["RMSE"].mean()

    size = pd.concat([
        pd.read_csv(ARCTIC_MODELS_DIR / "val_metrics_10K_s400_fluxonly.csv").assign(x=10_000),
        pd.read_csv(ARCTIC_MODELS_DIR / "val_metrics_50K_s400.csv").assign(x=50_000),
        pd.read_csv(ARCTIC_MODELS_DIR / "val_metrics_250K_s400_fluxonly.csv").assign(x=250_000),
        pd.read_csv(ARCTIC_MODELS_DIR / "val_metrics_500K_s400.csv").assign(x=500_000),
    ], ignore_index=True)
    size = size[size["target"].isin(FLUX_TARGETS)]
    size_avg = size.groupby(["target", "x"], as_index=False)["RMSE"].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    for ti, target in enumerate(FLUX_TARGETS):
        sub = sweep_avg[sweep_avg["target"] == target].sort_values("stride")
        ax1.plot(sub["stride"], sub["RMSE"], marker="o", color=PALETTE[ti], label=target)
    ax1.set_xlabel("Sampling stride")
    ax1.set_ylabel("Validation RMSE")
    ax1.set_title("(a) Sampling-density sweep (50K windows)")
    ax1.legend(frameon=False)
    _add_grid(ax1)

    for ti, target in enumerate(FLUX_TARGETS):
        sub = size_avg[size_avg["target"] == target].sort_values("x")
        ax2.plot(sub["x"], sub["RMSE"], marker="o", color=PALETTE[ti], label=target)
    ax2.set_xscale("log")
    ax2.minorticks_off()
    ax2.set_xticks([10_000, 50_000, 250_000, 500_000])
    ax2.set_xticklabels(["10K", "50K", "250K", "500K"])
    ax2.set_xlabel("Training-set windows at stride=400")
    ax2.set_title("(b) Dataset-size scale-up")
    ax2.legend(frameon=False)
    _add_grid(ax2)

    fig.tight_layout()
    _save(fig, "fig3_arctic_sampling_sweep.png")


def main() -> None:
    _style()
    figure3_arctic_sweep()


if __name__ == "__main__":
    main()
