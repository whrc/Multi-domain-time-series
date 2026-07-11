"""
Arctic domain — Step 4: evaluation.

See domains/arctic_domain/arctic_description.md § "Step 4 — Evaluation".

Compute per-pixel metrics for each target x SSP x period and produce metric boxplots and a
circumpolar median-NSE overview map per (SSP, period). Predictions are recomputed from the
checkpoint and rounded to 3 dp to
match the NetCDF written by 03_predict; ground truth comes from test.pkl (the same
inverse-transformed target values used everywhere), avoiding a re-read of GCS and a
re-alignment of the saved NetCDF. Yearly targets (ALD, VEGC) are scored at January
positions only; periods split at the configured projected_start_year.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.evaluate import metrics_df_by_period, predict_and_inverse, scenario_period_label  # noqa: E402
from shared.plots import plot_metric_boxplot, plot_metric_scatter_map  # noqa: E402
from shared.transformer import TransformerModel  # noqa: E402
from shared import tracking  # noqa: E402
from domains.arctic_domain._naming import load_stride_seq_len, run_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_TARGETS = 4


def save_prediction_sample(
    seg_meta: list[dict],
    pred_list: list[np.ndarray],
    obs_list: list[np.ndarray],
    target_names: list[str],
    idx_map: dict[str, pd.DatetimeIndex],
    seed: int,
    n_pixels: int,
    save_path: Path,
) -> None:
    """Save the full monthly obs-vs-predicted time series for a small, deterministic sample
    of test pixels. Unlike metrics_test.csv (aggregated RMSE/NSE/KGE/PBIAS per pixel/target/
    period), this keeps raw values so a specific pixel's time series can still be plotted
    after test.pkl is deleted to free disk space. The sample is a seeded draw over the sorted
    set of unique test pixels, so it reproduces identically every time this runs against the
    same test.pkl — which is now frozen (see 01_preprocess.py's sidecar-mismatch guard) — so
    the same sites stay directly comparable in a future multi-domain comparison.
    """
    unique_pixels = sorted({(m["grid"], m["y"], m["x"]) for m in seg_meta})
    rng = np.random.default_rng(seed)
    n = min(n_pixels, len(unique_pixels))
    sampled_idx = rng.choice(len(unique_pixels), size=n, replace=False)
    sampled_pixels = {unique_pixels[i] for i in sampled_idx}

    blocks = []
    for meta, pred, obs in zip(seg_meta, pred_list, obs_list):
        if (meta["grid"], meta["y"], meta["x"]) not in sampled_pixels:
            continue
        time = idx_map["ssp1" if "ssp1" in meta["ssp"] else "ssp5"]
        block = {
            "grid": meta["grid"], "y": meta["y"], "x": meta["x"],
            "lat": meta["lat"], "lon": meta["lon"], "ssp": meta["ssp"], "time": time,
        }
        for i, name in enumerate(target_names):
            # pred is already rounded to 3dp by the caller (matches the NetCDF written by
            # 03_predict); obs isn't rounded upstream, so round it here to match.
            block[f"obs_{name.lower()}"] = np.round(obs[:, i], 3)
            block[f"pred_{name.lower()}"] = pred[:, i]
        blocks.append(pd.DataFrame(block))

    sample_df = pd.concat(blocks, ignore_index=True)
    sample_df.to_parquet(save_path, index=False)
    logger.info("Saved prediction sample: %d pixels, %d rows -> %s", len(sampled_pixels), len(sample_df), save_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=None,
                        help="Which labeled checkpoint to load (matches the --train-size used "
                             "in 02_train.py). Omit to fall back to preprocessing.train_size "
                             "from config.")
    parser.add_argument("--label", type=str, default=None,
                        help="Which labeled checkpoint to load (matches the --label used in "
                             "02_train.py, e.g. '50K_s150' for a density-sweep point). Omit to "
                             "fall back to the default train_size-derived label.")
    args = parser.parse_args()

    cfg = load_config("arctic_domain")
    train_size = args.train_size if args.train_size is not None else cfg["preprocessing"]["train_size"]
    label = run_label(train_size, args.label)
    target_names = [t["name"] for t in cfg["targets"]]
    yearly = {t["name"] for t in cfg["targets"] if t["resolution"] == "yearly"}
    idx_map = {k: pd.date_range(v["start"], v["end"], freq="MS") for k, v in cfg["time"]["scenarios"].items()}
    proj_start = cfg["time"]["projected_start_year"]
    eval_dir = Path(cfg["paths"]["evaluation"]) / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_path = Path(cfg["paths"]["preprocessed_dir"]) / "test.pkl"
    _, seq_len = load_stride_seq_len(test_path)
    with test_path.open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)
    num_features = test_records[0]["data"].shape[1] - NUM_TARGETS

    models_dir = Path(cfg["paths"]["best_model"]).parent
    best_model_path = models_dir / f"best_model_{label}.pt"
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model = TransformerModel(num_features, NUM_TARGETS, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    seg_meta, pred_list, obs_list = predict_and_inverse(model, test_records, NUM_TARGETS, seq_len, device, scaler)
    pred_list = [np.round(p, 3) for p in pred_list]  # match the 3-dp NetCDF written by 03_predict

    metrics_df = metrics_df_by_period(
        seg_meta, pred_list, obs_list, target_names, yearly, idx_map, proj_start,
        id_fields=["grid", "y", "x", "lat", "lon", "ssp"],
    ).round(3)
    metrics_df.to_csv(eval_dir / "metrics_test.csv", index=False)
    logger.info("Saved %d metric rows (%d test pixels)", len(metrics_df), len(seg_meta) // len(cfg["scenarios"]))

    save_prediction_sample(
        seg_meta, pred_list, obs_list, target_names, idx_map,
        seed=cfg["preprocessing"]["random_seed"], n_pixels=50,
        save_path=eval_dir / "prediction_sample.parquet",
    )

    # Boxplot/spatial-map aggregation excludes obs_degenerate rows (constant-observed windows
    # make NSE/KGE mathematically undefined - see metrics_df_by_period docstring); metrics_test.csv
    # above keeps every row, degenerate or not, so nothing is hidden from the raw record.
    n_degenerate = int(metrics_df["obs_degenerate"].sum())
    if n_degenerate:
        logger.info("Excluding %d/%d degenerate (constant-observed) rows from boxplot/spatial-map aggregation",
                    n_degenerate, len(metrics_df))
    plot_df = metrics_df[~metrics_df["obs_degenerate"]].copy()

    # Combined boxplot: one figure, 3 groups per target (historical / projected-ssp126 /
    # projected-ssp585) - easier to compare than one figure per SSP.
    plot_df["scenario_period"] = [scenario_period_label(s, p) for s, p in zip(plot_df["ssp"], plot_df["period"])]
    plot_metric_boxplot(plot_df, group_col="scenario_period", title="Test metrics",
                        save_path=eval_dir / "metrics_boxplot_test.png")

    # Flux-only boxplot (GPP, RECO - monthly, concurrent-climate-driven): ALD/VEGC (yearly
    # pool variables) have much weaker per-pixel skill and their wide axis scale otherwise
    # hides how well the fluxes are actually doing - see this run's key_findings_log.md entry.
    flux_targets = [t for t in target_names if t not in yearly]
    flux_df = plot_df[plot_df["target"].isin(flux_targets)]
    plot_metric_boxplot(flux_df, group_col="scenario_period", title="Test metrics — fluxes (GPP, RECO) only",
                        save_path=eval_dir / "metrics_boxplot_test_fluxes.png")

    # Spatial overview: one map per (ssp, period), every test site colored by its median NSE
    # across all targets - a single circumpolar summary instead of one dense array per grid.
    site_median_nse = (
        plot_df.groupby(["ssp", "period", "grid", "y", "x", "lat", "lon"])["NSE"]
        .median()
        .reset_index()
    )
    for (ssp, period), g in site_median_nse.groupby(["ssp", "period"]):
        short = "ssp1" if "ssp1" in ssp else "ssp5"
        plot_metric_scatter_map(
            g["lon"].to_numpy(), g["lat"].to_numpy(), g["NSE"].to_numpy(),
            title=f"Median NSE across targets — {ssp} {period}",
            save_path=eval_dir / f"spatial_median_nse_{short}_{period}.png",
            vmin=-1, vmax=1,
        )
    logger.info("Saved evaluation figures to %s", eval_dir)

    enabled = tracking.setup(cfg)
    run_id = tracking.read_run_id(best_model_path.with_suffix(".run_id")) if enabled else None
    with tracking.resume_run(run_id) as active:
        if active:
            tracking.log_median_metrics(metrics_df, target_names)
            tracking.log_artifacts([eval_dir / "metrics_test.csv", *sorted(eval_dir.rglob("*.png"))])
            logger.info("Logged evaluation metrics + artifacts to MLflow run %s", run_id)


if __name__ == "__main__":
    main()
