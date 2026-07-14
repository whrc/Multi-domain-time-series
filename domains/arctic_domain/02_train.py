"""
Arctic domain — Step 2: training.

See domains/arctic_domain/arctic_description.md § "Step 2 — Training".

Train the shared causal transformer on the per-pixel sequences, checkpoint on the best
validation loss, and write loss-curve / scatter / metric-boxplot figures. The masked loss
naturally ignores the NaN months of the yearly targets (ALD, VEGC).
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.dataset import WindowedDataset, records_to_segments  # noqa: E402
from shared.evaluate import metrics_df_by_period, predict_and_inverse, scenario_period_label, stack_by_target  # noqa: E402
from shared.plots import plot_loss_curves, plot_metric_boxplot, plot_pred_vs_true  # noqa: E402
from shared.training import history_to_dataframe, run_lr_finder, set_seed, train_model  # noqa: E402
from shared.transformer import TransformerModel  # noqa: E402
from shared import tracking  # noqa: E402
from domains.arctic_domain._naming import (  # noqa: E402
    FLUX_TARGET_NAMES, load_stride_seq_len, run_label, select_flux_scaler_stats,
    select_flux_target_columns, train_pkl_name,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_split(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # flush every line even when redirected to a
    # log file (nohup, subprocess, ...) instead of a terminal, so progress is visible live
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=None,
                        help="Which train pkl variant to load (matches the --train-size used "
                             "in 01_preprocess.py). Omit to fall back to preprocessing.train_size "
                             "from config.")
    parser.add_argument("--label", type=str, default=None,
                        help="Which labeled train pkl variant to load (matches the --label used "
                             "in 01_preprocess.py, e.g. '50K_s150' for a density-sweep point). "
                             "Omit to fall back to the default train_size-derived label.")
    parser.add_argument("--flux-only", action="store_true",
                        help="Train on GPP+RECO only, dropping ALD/VEGC. Reuses the existing "
                             "full-target train/val pkl and scaler as-is (reordered/sliced down "
                             "to the flux columns, no re-preprocessing) — see "
                             "domains/arctic_domain/_naming.py:select_flux_target_columns. "
                             "Output checkpoint/eval-folder/val_metrics all get a '_fluxonly' "
                             "suffix so this never collides with the full-target run's outputs; "
                             "the *input* pkl lookup (--label/--train-size) is unaffected.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Training RNG seed (weight init + minibatch shuffle order). Omit "
                             "for today's unseeded behavior. When given, seeds torch/numpy/"
                             "random and appends '_seedN' to the output checkpoint/eval-folder "
                             "names — does not affect the (fixed) train/val/test data split.")
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    cfg = load_config("arctic_domain")
    train_size = args.train_size if args.train_size is not None else cfg["preprocessing"]["train_size"]
    label = run_label(train_size, args.label)
    output_label = f"{label}_fluxonly" if args.flux_only else label
    if args.seed is not None:
        output_label += f"_seed{args.seed}"
    tcfg = cfg["training"]
    full_target_names = [t["name"] for t in cfg["targets"]]
    target_names = FLUX_TARGET_NAMES if args.flux_only else full_target_names
    num_targets = len(target_names)
    yearly = {t["name"] for t in cfg["targets"] if t["resolution"] == "yearly"} & set(target_names)
    idx_map = {k: pd.date_range(v["start"], v["end"], freq="MS") for k, v in cfg["time"]["scenarios"].items()}
    proj_start = cfg["time"]["projected_start_year"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pre_dir = Path(cfg["paths"]["preprocessed_dir"])
    train_path = pre_dir / train_pkl_name(train_size, args.label)
    val_path = pre_dir / "val.pkl"
    train_records = load_split(train_path)
    val_records = load_split(val_path)
    if args.flux_only:
        train_records = select_flux_target_columns(train_records, full_target_names)
        val_records = select_flux_target_columns(val_records, full_target_names)
    num_features = train_records[0]["data"].shape[1] - num_targets
    logger.info("Device: %s | features=%d targets=%d", device, num_features, num_targets)

    train_stride, train_seq_len = load_stride_seq_len(train_path)
    val_stride, val_seq_len = load_stride_seq_len(val_path)

    train_segs, train_meta = records_to_segments(train_records)
    val_segs, val_meta = records_to_segments(val_records)
    train_ds = WindowedDataset(train_segs, train_meta, num_targets, train_seq_len, train_stride)
    val_ds = WindowedDataset(val_segs, val_meta, num_targets, val_seq_len, val_stride)
    actual_train_windows = len(train_ds)
    logger.info("Train windows: %d | Val windows: %d", actual_train_windows, len(val_ds))

    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, generator=generator,
        num_workers=tcfg["num_workers"], pin_memory=(device.type == "cuda"),
        persistent_workers=tcfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=tcfg["num_workers"], pin_memory=(device.type == "cuda"),
        persistent_workers=tcfg["num_workers"] > 0,
    )

    model = TransformerModel(num_features, num_targets, cfg).to(device)

    models_dir = Path(cfg["paths"]["best_model"]).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / f"best_model_{output_label}.pt"
    eval_dir = Path(cfg["paths"]["evaluation"]) / output_label
    eval_dir.mkdir(parents=True, exist_ok=True)
    enabled = tracking.setup(cfg)
    with tracking.training_run(cfg, "arctic_domain", enabled) as run:
        if run is not None:  # write the run_id sidecar up front so a later crash still records it
            sidecar = best_model_path.with_suffix(".run_id")
            sidecar.write_text(run.info.run_id)
        if tcfg["optimized_lr"] is not None:
            lr = float(tcfg["optimized_lr"])
            logger.info("Using configured optimized_lr=%.3e", lr)
        else:
            lr = run_lr_finder(model, train_loader, float(tcfg["initial_lr"]), device, eval_dir / "lr_finder.png")

        history = train_model(
            model, train_loader, val_loader, cfg, target_names, lr, device,
            best_model_path, num_features,
        )
        logger.info("Best val loss %.4f at epoch %d", history["best_val_loss"], history["best_epoch"])

        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        with Path(cfg["paths"]["scaler"]).open("rb") as f:
            scaler = pickle.load(f)
        if args.flux_only:
            scaler = select_flux_scaler_stats(scaler, full_target_names)

        figs = [eval_dir / "loss_curves.png", eval_dir / "val_pred_vs_true.png", eval_dir / "metrics_boxplot_val.png"]
        plot_loss_curves(history["train_loss"], history["val_loss"], history["per_target_val"],
                         eval_every=tcfg["eval_every_n_epochs"], save_path=figs[0])
        history_to_dataframe(history, tcfg["eval_every_n_epochs"]).round(3).to_csv(
            eval_dir / "history.csv", index=False)
        seg_meta, pred_list, obs_list = predict_and_inverse(model, val_records, num_targets, val_seq_len, device, scaler)
        pred_d, obs_d = stack_by_target(pred_list, obs_list, target_names)
        plot_pred_vs_true(pred_d, obs_d, log_scale=False, save_path=figs[1])
        val_metrics = metrics_df_by_period(
            seg_meta, pred_list, obs_list, target_names, yearly, idx_map, proj_start,
            id_fields=["grid", "y", "x", "ssp"],
        )
        # Exclude obs_degenerate rows (constant-observed windows make NSE/KGE undefined - see
        # metrics_df_by_period docstring) from both the boxplot and the learning-curve summary
        # below; val_metrics has no raw per-row CSV output to preserve, unlike test's metrics_test.csv.
        n_degenerate = int(val_metrics["obs_degenerate"].sum())
        if n_degenerate:
            logger.info("Excluding %d/%d degenerate (constant-observed) rows from val aggregation",
                        n_degenerate, len(val_metrics))
        val_metrics = val_metrics[~val_metrics["obs_degenerate"]].copy()
        val_metrics["scenario_period"] = [scenario_period_label(s, p) for s, p in zip(val_metrics["ssp"], val_metrics["period"])]
        plot_metric_boxplot(val_metrics, group_col="scenario_period", title="Validation metrics", save_path=figs[2])

        # Flux-only boxplot (GPP, RECO - monthly, concurrent-climate-driven): ALD/VEGC (yearly
        # pool variables) have much weaker per-pixel skill and their wide axis scale otherwise
        # hides how well the fluxes are actually doing - see this run's key_findings_log.md entry.
        # Skipped in --flux-only runs, where target_names is already flux-only — this would just
        # be a redundant duplicate of metrics_boxplot_val.png above.
        if not args.flux_only:
            flux_targets = [t for t in target_names if t not in yearly]
            flux_metrics = val_metrics[val_metrics["target"].isin(flux_targets)]
            flux_fig = eval_dir / "metrics_boxplot_val_fluxes.png"
            plot_metric_boxplot(flux_metrics, group_col="scenario_period", title="Validation metrics — fluxes (GPP, RECO) only",
                                save_path=flux_fig)
            figs.append(flux_fig)
        logger.info("Saved training figures to %s", eval_dir)

        # Save learning-curve summary row (train_windows column holds the real count regardless
        # of the label, so 05_learning_curve.py's saturation plot is unaffected by this naming).
        # median, not mean: per-pixel NSE is unbounded below (a near-constant-but-not-quite
        # observed pixel can score in the millions-negative range even after excluding exactly-
        # constant obs_degenerate rows), so a handful of outlier pixels wreck the mean.
        summary = val_metrics.groupby(["ssp", "target"])[["RMSE", "NSE", "KGE", "PBIAS"]].median().reset_index()
        summary.insert(0, "train_windows", actual_train_windows)
        summary_path = models_dir / f"val_metrics_{output_label}.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Saved learning curve snapshot: %s", summary_path)

        if run is not None:
            tracking.log_history(history, target_names, tcfg["eval_every_n_epochs"])
            tracking.log_artifacts([best_model_path, eval_dir / "lr_finder.png", *figs])


if __name__ == "__main__":
    main()
