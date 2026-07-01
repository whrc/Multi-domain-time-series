"""
Arctic domain — Step 2: training.

See domains/arctic_domain/arctic_description.md § "Step 2 — Training".

Train the shared causal transformer on the per-pixel sequences, checkpoint on the best
validation loss, and write loss-curve / scatter / metric-boxplot figures. The masked loss
naturally ignores the NaN months of the yearly targets (ALD, VEGC).
"""

import logging
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.dataset import WindowedDataset, records_to_segments  # noqa: E402
from shared.evaluate import per_unit_metrics, predict_and_inverse, stack_by_target  # noqa: E402
from shared.plots import plot_loss_curves, plot_metric_boxplot, plot_pred_vs_true  # noqa: E402
from shared.training import run_lr_finder, train_model  # noqa: E402
from shared.transformer import TransformerModel  # noqa: E402
from shared import tracking  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_TARGETS = 4


def load_split(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    cfg = load_config("arctic_domain")
    pp = cfg["preprocessing"]
    tcfg = cfg["training"]
    target_names = [t["name"] for t in cfg["targets"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pre_dir = Path(cfg["paths"]["preprocessed_dir"])
    train_records = load_split(pre_dir / "train.pkl")
    val_records = load_split(pre_dir / "val.pkl")
    num_features = train_records[0]["data"].shape[1] - NUM_TARGETS
    logger.info("Device: %s | features=%d targets=%d", device, num_features, NUM_TARGETS)

    train_segs, train_meta = records_to_segments(train_records)
    val_segs, val_meta = records_to_segments(val_records)
    train_ds = WindowedDataset(train_segs, train_meta, NUM_TARGETS, pp["seq_len"], pp["stride"])
    val_ds = WindowedDataset(val_segs, val_meta, NUM_TARGETS, pp["seq_len"], pp["stride"])
    actual_train_windows = len(train_ds)
    logger.info("Train windows: %d | Val windows: %d", actual_train_windows, len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model = TransformerModel(num_features, NUM_TARGETS, cfg).to(device)

    eval_dir = Path(cfg["paths"]["evaluation"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    enabled = tracking.setup(cfg)
    with tracking.training_run(cfg, "arctic_domain", enabled) as run:
        if run is not None:  # write the run_id sidecar up front so a later crash still records it
            sidecar = Path(cfg["paths"]["best_model"]).with_suffix(".run_id")
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(run.info.run_id)
        if tcfg["optimized_lr"] is not None:
            lr = float(tcfg["optimized_lr"])
            logger.info("Using configured optimized_lr=%.3e", lr)
        else:
            lr = run_lr_finder(model, train_loader, float(tcfg["initial_lr"]), device, eval_dir / "lr_finder.png")

        history = train_model(
            model, train_loader, val_loader, cfg, target_names, lr, device,
            Path(cfg["paths"]["best_model"]), num_features,
        )
        logger.info("Best val loss %.4f at epoch %d", history["best_val_loss"], history["best_epoch"])

        ckpt = torch.load(Path(cfg["paths"]["best_model"]), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        with Path(cfg["paths"]["scaler"]).open("rb") as f:
            scaler = pickle.load(f)

        figs = [eval_dir / "loss_curves.png", eval_dir / "val_pred_vs_true.png", eval_dir / "val_metrics_boxplot.png"]
        plot_loss_curves(history["train_loss"], history["val_loss"], history["per_target_val"],
                         eval_every=tcfg["eval_every_n_epochs"], save_path=figs[0])
        seg_meta, pred_list, obs_list = predict_and_inverse(model, val_records, NUM_TARGETS, pp["seq_len"], device, scaler)
        pred_d, obs_d = stack_by_target(pred_list, obs_list, target_names)
        plot_pred_vs_true(pred_d, obs_d, log_scale=False, save_path=figs[1])
        val_metrics = per_unit_metrics(seg_meta, pred_list, obs_list, target_names, ["grid", "y", "x", "ssp"])
        plot_metric_boxplot(val_metrics, group_col="ssp", title="Validation metrics", save_path=figs[2])
        logger.info("Saved training figures to %s", eval_dir)

        # Save size-keyed snapshot for learning curve tracking
        models_dir = Path(cfg["paths"]["best_model"]).parent
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(cfg["paths"]["best_model"]), models_dir / f"best_model_{actual_train_windows}.pt")
        summary = val_metrics.groupby(["ssp", "target"])[["RMSE", "NSE", "KGE", "PBIAS"]].mean().reset_index()
        summary.insert(0, "train_windows", actual_train_windows)
        summary_path = models_dir / f"val_metrics_{actual_train_windows}.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Saved learning curve snapshot: %s", summary_path)

        if run is not None:
            tracking.log_history(history, target_names, tcfg["eval_every_n_epochs"])
            tracking.log_artifacts([Path(cfg["paths"]["best_model"]), eval_dir / "lr_finder.png", *figs])


if __name__ == "__main__":
    main()
