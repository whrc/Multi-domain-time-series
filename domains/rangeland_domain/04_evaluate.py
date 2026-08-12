"""
Rangeland domain — Step 4: evaluation.

See domains/rangeland_domain/rangeland_description.md § "Step 4 — Evaluation".

Align test predictions (parquet) with ground truth (test.pkl, inverse-transformed),
compute per-site/per-target metrics, and write metrics_test.csv plus boxplot and
representative-site time-series figures.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.metrics import compute_metrics  # noqa: E402
from shared.plots import plot_metric_boxplot, plot_site_split_map, plot_timeseries  # noqa: E402
from shared import tracking  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_FEATURES = 22


def ground_truth_long(
    test_records: list[dict], scaler: dict, target_names: list[str], num_targets: int,
) -> pd.DataFrame:
    """Inverse-transform test targets and reshape to long (site, pft, date, target, obs)."""
    mean_t = scaler["mean"][-num_targets:]
    std_t = scaler["std"][-num_targets:]
    frames = []
    for r in test_records:
        for seg, (year, month) in zip(r["segments"], r["segment_starts"]):
            dates = pd.date_range(start=f"{year}-{month:02d}-01", periods=seg.shape[0], freq="MS")
            df = pd.DataFrame(seg[:, -num_targets:] * std_t + mean_t, columns=target_names)
            df["site"], df["pft"], df["date"] = r["site"], r["pft"], dates
            frames.append(df)
    wide = pd.concat(frames, ignore_index=True)
    return wide.melt(id_vars=["site", "pft", "date"], value_vars=target_names,
                     var_name="target", value_name="obs")


def load_site_coords(cfg: dict) -> pd.DataFrame:
    """Site lat/lon from the local AmeriFlux sites GeoJSON (RangeSTAR_data/ameriflux_sites.geojson)."""
    geojson_path = Path(cfg["data"]["dir"]) / "ameriflux_sites.geojson"
    with geojson_path.open() as f:
        features = json.load(f)["features"]
    rows = [{"site": feat["properties"]["site"], "lat": feat["properties"]["lat"],
             "lon": feat["properties"]["lon"]} for feat in features]
    return pd.DataFrame(rows)


def site_split_table(preprocessed_dir: Path) -> pd.DataFrame:
    """site -> split ("train"/"val"/"test"), read from the three preprocessed pkls."""
    rows = []
    for split in ("train", "val", "test"):
        with (preprocessed_dir / f"{split}.pkl").open("rb") as f:
            records = pickle.load(f)
        rows.extend({"site": r["site"], "split": split} for r in records)
    return pd.DataFrame(rows).drop_duplicates(subset="site")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux-only", action="store_true",
                        help="Evaluate the flux-only checkpoint/predictions (matches "
                             "--flux-only in 02_train.py/03_predict.py) instead of the "
                             "full-target run.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Which seeded checkpoint/predictions to load (matches --seed in "
                             "02_train.py/03_predict.py).")
    parser.add_argument("--capacity-matched", action="store_true",
                        help="Ablation only — evaluate the capacity-matched checkpoint/"
                             "predictions (matches --capacity-matched in 02_train.py/"
                             "03_predict.py). See ablation_test/ablation_description.md.")
    parser.add_argument("--amazon-sized", action="store_true",
                        help="Ablation only — evaluate the amazon-sized checkpoint/predictions "
                             "(matches --amazon-sized in 02_train.py/03_predict.py). See "
                             "ablation_test/ablation_description.md.")
    parser.add_argument("--model-size", choices=("small", "medium", "large"), default=None,
                        help="Hyperparameter-tuning sweep only — evaluate the model_{size} "
                             "checkpoint/predictions (matches --model-size in 02_train.py/"
                             "03_predict.py) instead of the default 'production' one.")
    args = parser.parse_args()
    if sum([args.capacity_matched, args.amazon_sized, args.model_size is not None]) > 1:
        parser.error("--capacity-matched, --amazon-sized, and --model-size are mutually exclusive")

    cfg = load_config("rangeland_domain")
    flux_names = cfg["targets"]["fluxes"]
    pool_names = cfg["targets"]["pools"]
    target_names = flux_names if args.flux_only else flux_names + pool_names
    num_targets = len(target_names)
    suffix = "_fluxonly" if args.flux_only else ""
    if args.model_size is not None:
        suffix += f"_{args.model_size}"
        if args.seed is not None:
            suffix += f"_seed{args.seed}"
    else:
        if args.seed is not None:
            suffix += f"_seed{args.seed}"
        if args.capacity_matched:
            suffix += "_capmatched"
        elif args.amazon_sized:
            suffix += "_amazonsized"
    eval_dir = Path(cfg["paths"]["evaluation"])
    eval_dir = eval_dir.with_stem(eval_dir.stem + suffix)
    eval_dir.mkdir(parents=True, exist_ok=True)

    with (Path(cfg["paths"]["preprocessed_dir"]) / "test.pkl").open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)
    if args.flux_only:
        keep = NUM_FEATURES + len(flux_names)
        test_records = [{**r, "segments": [s[:, :keep] for s in r["segments"]]} for r in test_records]
        scaler = {"mean": scaler["mean"][:keep], "std": scaler["std"][:keep]}
    preds = pd.read_parquet(Path(cfg["paths"]["predictions"]) / f"predictions{suffix}.parquet")

    obs_long = ground_truth_long(test_records, scaler, target_names, num_targets)
    pred_long = preds.melt(id_vars=["site", "date"], value_vars=target_names,
                           var_name="target", value_name="pred")
    merged = obs_long.merge(pred_long, on=["site", "date", "target"])

    rows = []
    for (site, pft, target), g in merged.groupby(["site", "pft", "target"]):
        rows.append({"site": site, "pft": pft, "target": target,
                     **compute_metrics(g["pred"].to_numpy(), g["obs"].to_numpy())})
    metrics_df = pd.DataFrame(rows).round(3)
    metrics_df.to_csv(eval_dir / "metrics_test.csv", index=False)
    logger.info("Saved metrics for %d sites x %d targets", metrics_df["site"].nunique(), len(target_names))

    # Fluxes and pools are plotted separately, each as both a per-PFT-grouped and an
    # all-PFTs-pooled boxplot: mixing e.g. GPP's RMSE~1 with HOC's RMSE~thousands (and NSE in
    # the billions for some near-constant slow-cycling pools) on one shared axis makes the flux
    # boxes invisible flat lines — see key_findings_log.md for the full diagnosis. Skipped for
    # pools in a --flux-only run, where there are none.
    flux_metrics = metrics_df[metrics_df["target"].isin(flux_names)]
    plot_metric_boxplot(flux_metrics, group_col="pft", title="Test metrics — fluxes, by PFT",
                        save_path=eval_dir / "metrics_boxplot_test_fluxes_by_pft.png")
    plot_metric_boxplot(flux_metrics, group_col=None, title="Test metrics — fluxes, all PFTs pooled",
                        save_path=eval_dir / "metrics_boxplot_test_fluxes_pooled.png")
    if not args.flux_only:
        pool_metrics = metrics_df[metrics_df["target"].isin(pool_names)]
        plot_metric_boxplot(pool_metrics, group_col="pft", title="Test metrics — pools, by PFT",
                            save_path=eval_dir / "metrics_boxplot_test_pools_by_pft.png")
        plot_metric_boxplot(pool_metrics, group_col=None, title="Test metrics — pools, all PFTs pooled",
                            save_path=eval_dir / "metrics_boxplot_test_pools_pooled.png")

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

    # Site map: train/val/test sites on a regional basemap.
    splits_df = site_split_table(Path(cfg["paths"]["preprocessed_dir"]))
    coords_df = load_site_coords(cfg)
    site_df = splits_df.merge(coords_df, on="site", how="left")
    missing = site_df[site_df["lat"].isna()]["site"].tolist()
    if missing:
        logger.warning("No coordinates found for %d sites: %s", len(missing), missing)
    site_df = site_df.dropna(subset=["lat", "lon"])
    plot_site_split_map(
        site_df["lon"].to_numpy(), site_df["lat"].to_numpy(), site_df["split"].to_numpy(),
        title="Rangeland sites: train/val/test sites", save_path=eval_dir / "site_map.png",
    )
    logger.info("Saved site split map to %s", eval_dir / "site_map.png")

    best_model_path = Path(cfg["paths"]["best_model"])
    best_model_path = best_model_path.with_stem(best_model_path.stem + suffix)
    enabled = tracking.setup(cfg)
    run_id = tracking.read_run_id(best_model_path.with_suffix(".run_id")) if enabled else None
    with tracking.resume_run(run_id) as active:
        if active:
            tracking.log_median_metrics(metrics_df, target_names)
            tracking.log_artifacts([eval_dir / "metrics_test.csv", *sorted(eval_dir.rglob("*.png"))])
            logger.info("Logged evaluation metrics + artifacts to MLflow run %s", run_id)


if __name__ == "__main__":
    main()
