"""
Rangeland domain — Step 2: training.

See domains/rangeland_domain/rangeland_description.md § "Step 2 — Training".

Train the shared causal transformer on the per-site segments, checkpoint on the best
validation loss, and write loss-curve / scatter / metric-boxplot figures.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

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

NUM_FEATURES = 22
NUM_TARGETS = 10


def load_split(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux-only", action="store_true",
                        help="Train on the 4 flux targets only (GPP, RECO, Rm, Rg), dropping "
                             "the 6 pool targets (AGB, BGB, AGL, BGL, POC, HOC). Fluxes are "
                             "already the first 4 of the 10 trailing target columns in the "
                             "existing preprocessed pkl, so this just truncates rather than "
                             "reordering — no re-preprocessing needed. Output checkpoint/"
                             "evaluation/predictions all get a '_fluxonly' suffix so this never "
                             "collides with the full-target run's outputs.")
    args = parser.parse_args()

    cfg = load_config("rangeland_domain")
    pp = cfg["preprocessing"]
    tcfg = cfg["training"]
    flux_names = cfg["targets"]["fluxes"]
    target_names = flux_names if args.flux_only else flux_names + cfg["targets"]["pools"]
    num_targets = len(target_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | features=%d targets=%d", device, NUM_FEATURES, num_targets)

    pre_dir = Path(cfg["paths"]["preprocessed_dir"])
    train_records = load_split(pre_dir / "train.pkl")
    val_records = load_split(pre_dir / "val.pkl")
    _check_shape = records_to_segments(train_records)[0][0]
    assert _check_shape.shape[1] == NUM_FEATURES + NUM_TARGETS, (
        f"data has {_check_shape.shape[1]} columns, expected {NUM_FEATURES + NUM_TARGETS} "
        f"(NUM_FEATURES={NUM_FEATURES} + NUM_TARGETS={NUM_TARGETS}) — config columns changed?"
    )
    if args.flux_only:
        # Fluxes are already the first 4 of the 10 trailing target columns (config order:
        # fluxes then pools) — WindowedDataset expects num_targets trailing columns, so
        # truncating (not reordering) is enough to keep just [features | GPP RECO Rm Rg].
        # Applied to the raw records (not just the derived segments below) since val_records
        # is reused as-is later for predict_and_inverse, which re-derives segments internally.
        keep = NUM_FEATURES + len(flux_names)
        train_records = [{**r, "segments": [s[:, :keep] for s in r["segments"]]} for r in train_records]
        val_records = [{**r, "segments": [s[:, :keep] for s in r["segments"]]} for r in val_records]

    train_segs, train_meta = records_to_segments(train_records)
    val_segs, val_meta = records_to_segments(val_records)
    train_ds = WindowedDataset(train_segs, train_meta, num_targets, pp["seq_len"], pp["stride"])
    val_ds = WindowedDataset(val_segs, val_meta, num_targets, pp["seq_len"], pp["stride"])
    logger.info("Train windows: %d | Val windows: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model = TransformerModel(NUM_FEATURES, num_targets, cfg).to(device)

    suffix = "_fluxonly" if args.flux_only else ""
    best_model_path = Path(cfg["paths"]["best_model"])
    best_model_path = best_model_path.with_stem(best_model_path.stem + suffix)
    eval_dir = Path(cfg["paths"]["evaluation"])
    eval_dir = eval_dir.with_stem(eval_dir.stem + suffix)
    eval_dir.mkdir(parents=True, exist_ok=True)
    enabled = tracking.setup(cfg)
    with tracking.training_run(cfg, "rangeland_domain", enabled) as run:
        if run is not None:  # write the run_id sidecar up front so a later crash still records it
            sidecar = best_model_path.with_suffix(".run_id")
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(run.info.run_id)
        if tcfg["optimized_lr"] is not None:
            lr = float(tcfg["optimized_lr"])
            logger.info("Using configured optimized_lr=%.3e", lr)
        else:
            lr = run_lr_finder(model, train_loader, float(tcfg["initial_lr"]), device, eval_dir / "lr_finder.png")

        history = train_model(
            model, train_loader, val_loader, cfg, target_names, lr, device,
            best_model_path, NUM_FEATURES,
        )
        logger.info("Best val loss %.4f at epoch %d", history["best_val_loss"], history["best_epoch"])

        # Reload best checkpoint and produce validation figures.
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        with Path(cfg["paths"]["scaler"]).open("rb") as f:
            scaler = pickle.load(f)
        if args.flux_only:
            keep = NUM_FEATURES + len(flux_names)
            scaler = {"mean": scaler["mean"][:keep], "std": scaler["std"][:keep]}

        figs = [eval_dir / "loss_curves.png", eval_dir / "val_pred_vs_true.png", eval_dir / "val_metrics_boxplot.png"]
        plot_loss_curves(history["train_loss"], history["val_loss"], history["per_target_val"],
                         eval_every=tcfg["eval_every_n_epochs"], save_path=figs[0])
        seg_meta, pred_list, obs_list = predict_and_inverse(model, val_records, num_targets, pp["seq_len"], device, scaler)
        pred_d, obs_d = stack_by_target(pred_list, obs_list, target_names)
        plot_pred_vs_true(pred_d, obs_d, log_scale=False, save_path=figs[1])
        val_metrics = per_unit_metrics(seg_meta, pred_list, obs_list, target_names, ["site", "pft"])
        plot_metric_boxplot(val_metrics, group_col="pft", title="Validation metrics", save_path=figs[2])
        logger.info("Saved training figures to %s", eval_dir)

        if run is not None:
            tracking.log_history(history, target_names, tcfg["eval_every_n_epochs"])
            tracking.log_artifacts([best_model_path, eval_dir / "lr_finder.png", *figs])


if __name__ == "__main__":
    main()
