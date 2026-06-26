"""
Arctic domain — Step 5: learning curve analysis.

See domains/arctic_domain/arctic_description.md § "Step 5 — Learning Curve".

Reads all val_metrics_{N}.csv files written by 02_train.py (one per learning curve run)
and produces a saturation plot of validation performance vs training set size.

Run after completing at least two training runs with different --train-size values:
    python run_arctic.py --stage learning-curve
"""

import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["RMSE", "NSE", "KGE", "PBIAS"]


def main() -> None:
    cfg = load_config("arctic_domain")
    models_dir = Path(cfg["paths"]["best_model"]).parent
    eval_dir = Path(cfg["paths"]["evaluation"]) / "learning_curve"
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(models_dir.glob("val_metrics_*.csv"))
    if not csv_paths:
        logger.error("No val_metrics_*.csv files found in %s — run 02_train.py with different --train-size values first", models_dir)
        sys.exit(1)
    logger.info("Found %d learning curve CSV(s): %s", len(csv_paths), [p.name for p in csv_paths])

    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    df = df.sort_values("train_windows")
    logger.info("Learning curve data:\n%s", df.to_string(index=False))

    targets = sorted(df["target"].unique())
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3.5 * n_metrics), sharex=True)
    fig.suptitle("Arctic Learning Curve: Validation Performance vs Training Set Size", fontsize=13)

    for ax, metric in zip(axes, METRICS):
        for target in targets:
            sub = df[df["target"] == target].sort_values("train_windows")
            ax.plot(sub["train_windows"], sub[metric], marker="o", label=target)
        ax.set_ylabel(metric)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Training Windows")

    out_path = eval_dir / "learning_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved learning curve plot to %s", out_path)


if __name__ == "__main__":
    main()
