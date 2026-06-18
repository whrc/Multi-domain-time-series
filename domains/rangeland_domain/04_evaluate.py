"""
Rangeland domain — Step 4: evaluation.

See domains/rangeland_domain/rangeland_description.md § "Step 4 — Evaluation".

Align test predictions (parquet) with ground truth (test.pkl, inverse-transformed),
compute per-site/per-target metrics, and write metrics.csv plus boxplot and
representative-site time-series figures.
"""

import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.metrics import compute_metrics  # noqa: E402
from shared.plots import plot_metric_boxplot, plot_timeseries  # noqa: E402
from shared import tracking  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_TARGETS = 10


def ground_truth_long(test_records: list[dict], scaler: dict, target_names: list[str]) -> pd.DataFrame:
    """Inverse-transform test targets and reshape to long (site, pft, date, target, obs)."""
    mean_t = scaler["mean"][-NUM_TARGETS:]
    std_t = scaler["std"][-NUM_TARGETS:]
    frames = []
    for r in test_records:
        for seg, (year, month) in zip(r["segments"], r["segment_starts"]):
            dates = pd.date_range(start=f"{year}-{month:02d}-01", periods=seg.shape[0], freq="MS")
            df = pd.DataFrame(seg[:, -NUM_TARGETS:] * std_t + mean_t, columns=target_names)
            df["site"], df["pft"], df["date"] = r["site"], r["pft"], dates
            frames.append(df)
    wide = pd.concat(frames, ignore_index=True)
    return wide.melt(id_vars=["site", "pft", "date"], value_vars=target_names,
                     var_name="target", value_name="obs")


def main() -> None:
    cfg = load_config("rangeland_domain")
    target_names = cfg["targets"]["fluxes"] + cfg["targets"]["pools"]
    eval_dir = Path(cfg["paths"]["evaluation"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    with (Path(cfg["paths"]["preprocessed_dir"]) / "test.pkl").open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)
    preds = pd.read_parquet(Path(cfg["paths"]["predictions"]) / "predictions.parquet")

    obs_long = ground_truth_long(test_records, scaler, target_names)
    pred_long = preds.melt(id_vars=["site", "date"], value_vars=target_names,
                           var_name="target", value_name="pred")
    merged = obs_long.merge(pred_long, on=["site", "date", "target"])

    rows = []
    for (site, pft, target), g in merged.groupby(["site", "pft", "target"]):
        rows.append({"site": site, "pft": pft, "target": target,
                     **compute_metrics(g["pred"].to_numpy(), g["obs"].to_numpy())})
    metrics_df = pd.DataFrame(rows).round(3)
    metrics_df.to_csv(eval_dir / "metrics.csv", index=False)
    logger.info("Saved metrics for %d sites x %d targets", metrics_df["site"].nunique(), len(target_names))

    plot_metric_boxplot(metrics_df, group_col="pft", title="Test metrics by PFT",
                        save_path=eval_dir / "metrics_boxplot.png")

    # One representative test site per PFT: predicted vs observed time series.
    for pft, g_pft in merged.groupby("pft"):
        site = g_pft["site"].iloc[0]
        g = merged[merged["site"] == site].sort_values("date")
        time = g["date"].drop_duplicates().to_numpy()
        pred_d = {t: g.loc[g["target"] == t, "pred"].to_numpy() for t in target_names}
        obs_d = {t: g.loc[g["target"] == t, "obs"].to_numpy() for t in target_names}
        plot_timeseries(time, pred_d, obs_d, title=f"{pft} — {site}",
                        save_path=eval_dir / f"timeseries_{pft}.png")
    logger.info("Saved evaluation figures to %s", eval_dir)

    enabled = tracking.setup(cfg)
    run_id = tracking.read_run_id(Path(cfg["paths"]["best_model"]).with_suffix(".run_id")) if enabled else None
    with tracking.resume_run(run_id) as active:
        if active:
            tracking.log_median_metrics(metrics_df, target_names)
            tracking.log_artifacts([eval_dir / "metrics.csv", *sorted(eval_dir.rglob("*.png"))])
            logger.info("Logged evaluation metrics + artifacts to MLflow run %s", run_id)


if __name__ == "__main__":
    main()
