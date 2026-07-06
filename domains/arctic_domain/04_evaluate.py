"""
Arctic domain — Step 4: evaluation.

See domains/arctic_domain/arctic_description.md § "Step 4 — Evaluation".

Compute per-pixel metrics for each target x SSP x period and produce metric boxplots and
spatial NSE maps. Predictions are recomputed from the checkpoint and rounded to 3 dp to
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
from shared.evaluate import predict_and_inverse  # noqa: E402
from shared.metrics import compute_metrics  # noqa: E402
from shared.plots import plot_metric_boxplot, plot_spatial_map  # noqa: E402
from shared.transformer import TransformerModel  # noqa: E402
from shared import tracking  # noqa: E402
from domains.arctic_domain._naming import run_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_TARGETS = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=None,
                        help="Which labeled checkpoint to load (matches the --train-size used "
                             "in 02_train.py). Omit to load the checkpoint trained on train_full.pkl.")
    args = parser.parse_args()
    label = run_label(args.train_size)

    cfg = load_config("arctic_domain")
    pp = cfg["preprocessing"]
    target_names = [t["name"] for t in cfg["targets"]]
    yearly = {t["name"] for t in cfg["targets"] if t["resolution"] == "yearly"}
    idx_map = {k: pd.date_range(v["start"], v["end"], freq="MS") for k, v in cfg["time"]["scenarios"].items()}
    proj_start = cfg["time"]["projected_start_year"]
    eval_dir = Path(cfg["paths"]["evaluation"]) / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with (Path(cfg["paths"]["preprocessed_dir"]) / "test.pkl").open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)
    num_features = test_records[0]["data"].shape[1] - NUM_TARGETS

    models_dir = Path(cfg["paths"]["best_model"]).parent
    best_model_path = models_dir / f"best_model_{label}.pt"
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model = TransformerModel(num_features, NUM_TARGETS, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    seg_meta, pred_list, obs_list = predict_and_inverse(model, test_records, NUM_TARGETS, pp["seq_len"], device, scaler)
    pred_list = [np.round(p, 3) for p in pred_list]  # match the 3-dp NetCDF written by 03_predict

    grid_shape = {meta["grid"]: (meta["ny"], meta["nx"]) for meta in seg_meta}
    rows = []
    for meta, pred, obs in zip(seg_meta, pred_list, obs_list):
        time = idx_map["ssp1" if "ssp1" in meta["ssp"] else "ssp5"]
        periods = (("historical", time.year < proj_start), ("projected", time.year >= proj_start))
        for i, name in enumerate(target_names):
            pos = (time.month == 1) if name in yearly else np.ones(len(time), dtype=bool)
            for period, in_period in periods:
                sel = pos & in_period
                if not sel.any():
                    continue
                rows.append({
                    "grid": meta["grid"], "y": meta["y"], "x": meta["x"],
                    "lat": meta["lat"], "lon": meta["lon"], "ssp": meta["ssp"],
                    "target": name, "period": period,
                    **compute_metrics(pred[sel, i], obs[sel, i]),
                })
    metrics_df = pd.DataFrame(rows).round(3)
    metrics_df.to_csv(eval_dir / "metrics.csv", index=False)
    logger.info("Saved %d metric rows (%d test pixels)", len(metrics_df), len(seg_meta) // len(cfg["scenarios"]))

    # Boxplots: one figure per SSP, historical vs projected within each target.
    for ssp, sub in metrics_df.groupby("ssp"):
        short = "ssp1" if "ssp1" in ssp else "ssp5"
        plot_metric_boxplot(sub, group_col="period", title=ssp,
                            save_path=eval_dir / f"metrics_boxplot_{short}.png")

    # Spatial NSE maps: per ssp x period x target.
    maps_dir = eval_dir / "spatial_metrics_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for (grid, ssp, period, target), g in metrics_df.groupby(["grid", "ssp", "period", "target"]):
        ny, nx = grid_shape[grid]
        arr = np.full((ny, nx), np.nan)
        arr[g["y"].to_numpy(), g["x"].to_numpy()] = g["NSE"].to_numpy()
        short = "ssp1" if "ssp1" in ssp else "ssp5"
        plot_spatial_map(arr, title=f"NSE {target} {short} {period}",
                         save_path=maps_dir / f"NSE_{grid}_{short}_{period}_{target}.png")
    logger.info("Saved evaluation figures to %s", eval_dir)

    enabled = tracking.setup(cfg)
    run_id = tracking.read_run_id(best_model_path.with_suffix(".run_id")) if enabled else None
    with tracking.resume_run(run_id) as active:
        if active:
            tracking.log_median_metrics(metrics_df, target_names)
            tracking.log_artifacts([eval_dir / "metrics.csv", *sorted(eval_dir.rglob("*.png"))])
            logger.info("Logged evaluation metrics + artifacts to MLflow run %s", run_id)


if __name__ == "__main__":
    main()
