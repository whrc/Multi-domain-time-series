"""
Amazon domain — Step 4: evaluation.

See domains/amazon_domain/amazon_description.md § "Step 4 — Evaluation".

Align test predictions (parquet) with ground truth (test.pkl, inverse-transformed),
compute per-station/per-target metrics, and write metrics_test.csv plus boxplot and
representative-station time-series figures.
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.metrics import compute_metrics  # noqa: E402
from shared.plots import plot_metric_boxplot, plot_timeseries  # noqa: E402
from shared import tracking  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_TARGETS = 3
N_REPRESENTATIVE = 3


def ground_truth_long(test_records: list[dict], scaler: dict, target_names: list[str]) -> pd.DataFrame:
    """Inverse-transform test targets and reshape to long (station_id, year, month, target, obs)."""
    mean_t = scaler["mean"][-NUM_TARGETS:]
    std_t = scaler["std"][-NUM_TARGETS:]
    frames = []
    for r in test_records:
        for seg, (year, month) in zip(r["segments"], r["segment_starts"]):
            idx = pd.date_range(start=f"{year}-{month:02d}-01", periods=seg.shape[0], freq="MS")
            # Targets were log1p-transformed before the scaler fit (01_preprocess.py); undo
            # the z-score first, then the log1p, to get back to physical units.
            df = pd.DataFrame(np.expm1(seg[:, -NUM_TARGETS:] * std_t + mean_t), columns=target_names)
            df["station_id"], df["year"], df["month"] = r["station_id"], idx.year, idx.month
            frames.append(df)
    wide = pd.concat(frames, ignore_index=True)
    return wide.melt(id_vars=["station_id", "year", "month"], value_vars=target_names,
                     var_name="target", value_name="obs")


def main() -> None:
    cfg = load_config("amazon_domain")
    target_names = cfg["targets"]
    eval_dir = Path(cfg["paths"]["evaluation"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    with (Path(cfg["paths"]["preprocessed_dir"]) / "test.pkl").open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)
    preds = pd.read_parquet(Path(cfg["paths"]["predictions"]) / "amazon_test_predictions.parquet")

    obs_long = ground_truth_long(test_records, scaler, target_names)
    pred_cols = [f"{t}_pred" for t in target_names]
    pred_long = preds.melt(id_vars=["station_id", "year", "month"], value_vars=pred_cols,
                           var_name="target", value_name="pred")
    pred_long["target"] = pred_long["target"].str.removesuffix("_pred")
    merged = obs_long.merge(pred_long, on=["station_id", "year", "month", "target"])

    rows = []
    for (station, target), g in merged.groupby(["station_id", "target"]):
        rows.append({"station_id": station, "target": target,
                     **compute_metrics(g["pred"].to_numpy(), g["obs"].to_numpy())})
    metrics_df = pd.DataFrame(rows).round(3)
    metrics_df.to_csv(eval_dir / "metrics_test.csv", index=False)
    logger.info("Saved metrics for %d stations x %d targets", metrics_df["station_id"].nunique(), len(target_names))

    plot_metric_boxplot(metrics_df, group_col=None, title="Test metrics", save_path=eval_dir / "metrics_boxplot.png")

    # Representative test stations: predicted vs observed time series.
    for station in sorted(merged["station_id"].unique())[:N_REPRESENTATIVE]:
        g = merged[merged["station_id"] == station].sort_values(["year", "month"])
        time = pd.to_datetime(g[["year", "month"]].assign(day=1).drop_duplicates())
        pred_d = {t: g.loc[g["target"] == t, "pred"].to_numpy() for t in target_names}
        obs_d = {t: g.loc[g["target"] == t, "obs"].to_numpy() for t in target_names}
        plot_timeseries(time.to_numpy(), pred_d, obs_d, title=f"Station {station}",
                        save_path=eval_dir / f"timeseries_{station}.png")
    logger.info("Saved evaluation figures to %s", eval_dir)

    enabled = tracking.setup(cfg)
    run_id = tracking.read_run_id(Path(cfg["paths"]["best_model"]).with_suffix(".run_id")) if enabled else None
    with tracking.resume_run(run_id) as active:
        if active:
            tracking.log_median_metrics(metrics_df, target_names)
            tracking.log_artifacts([eval_dir / "metrics_test.csv", *sorted(eval_dir.rglob("*.png"))])
            logger.info("Logged evaluation metrics + artifacts to MLflow run %s", run_id)


if __name__ == "__main__":
    main()
