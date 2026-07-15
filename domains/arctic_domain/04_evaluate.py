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
from shared.plots import plot_metric_boxplot, plot_metric_scatter_map, plot_timeseries  # noqa: E402
from shared.transformer import TransformerModel  # noqa: E402
from shared import tracking  # noqa: E402
from domains.arctic_domain._naming import (  # noqa: E402
    FLUX_TARGET_NAMES, load_stride_seq_len, run_label, sample_test_pixels,
    save_prediction_sample, select_flux_scaler_stats, select_flux_target_columns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def save_pixel_timeseries(
    seg_meta: list[dict],
    pred_list: list[np.ndarray],
    obs_list: list[np.ndarray],
    target_names: list[str],
    idx_map: dict[str, pd.DatetimeIndex],
    pixels: list[tuple],
    eval_dir: Path,
) -> None:
    """Obs-vs-predicted time series plots for a few representative test pixels, one figure
    each — matches Amazon/Rangeland's timeseries_*.png (Arctic previously had no per-unit
    timeseries view, only aggregate boxplots and spatial maps). Uses each pixel's ssp1_2_6
    series (full 1901-2100 historical+projected range — the more informative of the two
    scenarios, since ssp5_8_5 only covers the projected period)."""
    for grid, y, x in pixels:
        match = next(
            (m, p, o) for m, p, o in zip(seg_meta, pred_list, obs_list)
            if (m["grid"], m["y"], m["x"]) == (grid, y, x) and "ssp1" in m["ssp"]
        )
        meta, pred, obs = match
        time = idx_map["ssp1"]
        pred_d = {name: pred[:, i] for i, name in enumerate(target_names)}
        obs_d = {name: obs[:, i] for i, name in enumerate(target_names)}
        save_path = eval_dir / f"timeseries_{grid}_{y}_{x}.png"
        plot_timeseries(time.to_numpy(), pred_d, obs_d, title=f"{grid} ({y},{x})", save_path=save_path)
    logger.info("Saved %d pixel timeseries plots to %s", len(pixels), eval_dir)


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
    parser.add_argument("--flux-only", action="store_true",
                        help="Evaluate the GPP+RECO-only checkpoint (matches --flux-only in "
                             "02_train.py) instead of the full-target one.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Which seeded checkpoint to load (matches --seed in 02_train.py).")
    args = parser.parse_args()

    cfg = load_config("arctic_domain")
    train_size = args.train_size if args.train_size is not None else cfg["preprocessing"]["train_size"]
    label = run_label(train_size, args.label)
    output_label = f"{label}_fluxonly" if args.flux_only else label
    if args.seed is not None:
        output_label += f"_seed{args.seed}"
    full_target_names = [t["name"] for t in cfg["targets"]]
    target_names = FLUX_TARGET_NAMES if args.flux_only else full_target_names
    num_targets = len(target_names)
    yearly = {t["name"] for t in cfg["targets"] if t["resolution"] == "yearly"} & set(target_names)
    idx_map = {k: pd.date_range(v["start"], v["end"], freq="MS") for k, v in cfg["time"]["scenarios"].items()}
    proj_start = cfg["time"]["projected_start_year"]
    eval_dir = Path(cfg["paths"]["evaluation"]) / output_label
    eval_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_path = Path(cfg["paths"]["preprocessed_dir"]) / "test.pkl"
    _, seq_len = load_stride_seq_len(test_path)
    with test_path.open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)
    if args.flux_only:
        test_records = select_flux_target_columns(test_records, full_target_names)
        scaler = select_flux_scaler_stats(scaler, full_target_names)
    num_features = test_records[0]["data"].shape[1] - num_targets

    models_dir = Path(cfg["paths"]["best_model"]).parent
    best_model_path = models_dir / f"best_model_{output_label}.pt"
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model = TransformerModel(num_features, num_targets, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    seg_meta, pred_list, obs_list = predict_and_inverse(model, test_records, num_targets, seq_len, device, scaler)
    pred_list = [np.round(p, 3) for p in pred_list]  # match the 3-dp NetCDF written by 03_predict

    metrics_df = metrics_df_by_period(
        seg_meta, pred_list, obs_list, target_names, yearly, idx_map, proj_start,
        id_fields=["grid", "y", "x", "lat", "lon", "ssp"],
    ).round(3)
    metrics_df.to_csv(eval_dir / "metrics_test.csv", index=False)
    logger.info("Saved %d metric rows (%d test pixels)", len(metrics_df), len(seg_meta) // len(cfg["scenarios"]))

    sampled_pixels = sample_test_pixels(seg_meta, seed=cfg["preprocessing"]["random_seed"], n_pixels=50)
    save_prediction_sample(
        seg_meta, pred_list, obs_list, target_names, idx_map, sampled_pixels,
        save_path=eval_dir / "prediction_sample.parquet",
    )
    save_pixel_timeseries(
        seg_meta, pred_list, obs_list, target_names, idx_map, sampled_pixels[:2], eval_dir,
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
    # Skipped in --flux-only runs, where target_names is already flux-only — this would just
    # be a redundant duplicate of metrics_boxplot_test.png above.
    if not args.flux_only:
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
