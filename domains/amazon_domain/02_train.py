"""
Amazon domain — Step 2: training.

See domains/amazon_domain/amazon_description.md § "Step 2 — Training".

Train the shared causal transformer on per-station segments, checkpoint on the best
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
from shared.training import history_to_dataframe, run_lr_finder, set_seed, train_model  # noqa: E402
from shared.transformer import TransformerModel  # noqa: E402
from shared import tracking  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_FEATURES = 14
NUM_TARGETS = 3


def load_split(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Training RNG seed (weight init + minibatch shuffle order). Omit "
                             "for today's unseeded behavior. When given, seeds torch/numpy/"
                             "random and appends '_seedN' to the output checkpoint/eval/"
                             "predictions names — does not affect the (fixed) data split.")
    parser.add_argument("--capacity-matched", action="store_true",
                        help="Ablation only (see ablation_test/ablation_description.md): train "
                             "with the multi-domain shared trunk's architecture "
                             "(model_capacity_matched in config) instead of this domain's own "
                             "production architecture, to control for the capacity/dropout gap "
                             "between the individual and multi-domain models. Reuses the "
                             "existing train/val pkl — does not affect the data split. Outputs "
                             "go to *_capmatched so the production checkpoint is never touched.")
    parser.add_argument("--model-size", choices=("xxsmall", "xsmall", "small", "medium", "large", "ffn_narrow", "ffn_std", "layers2", "layers4", "layers6", "dropout10", "dropout20", "dropout30"), default=None,
                        help="Hyperparameter-tuning sweep only (see "
                             "hyperparameter_tuning/hyperparameter_tuning_description.md): use "
                             "the model_{size} architecture block (hidden_dim sweep) instead of "
                             "the config's default 'production' block. Appends '_{size}' to the "
                             "output checkpoint/eval names.")
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    cfg = load_config("amazon_domain")
    pp = cfg["preprocessing"]
    tcfg = cfg["training"]
    target_names = cfg["targets"]
    if args.capacity_matched:
        cfg["model"] = {**cfg["model"], **cfg["model_capacity_matched"]}
    if args.model_size is not None:
        cfg["model"] = {**cfg["model"], **cfg[f"model_{args.model_size}"]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | features=%d targets=%d", device, NUM_FEATURES, NUM_TARGETS)
    if args.model_size is not None:
        # Matches Arctic's own "_{size}_seed{seed}" convention (size before seed) so this
        # sweep's output paths are consistent across domains -- kept independent of the
        # seed/capacity_matched ordering below, which is untouched to avoid renaming
        # ablation_tests' already-computed *_capmatched outputs.
        suffix = f"_{args.model_size}"
        suffix += f"_seed{args.seed}" if args.seed is not None else ""
    else:
        suffix = f"_seed{args.seed}" if args.seed is not None else ""
        suffix += "_capmatched" if args.capacity_matched else ""
    best_model_path = Path(cfg["paths"]["best_model"])
    best_model_path = best_model_path.with_stem(best_model_path.stem + suffix)

    pre_dir = Path(cfg["paths"]["preprocessed_dir"])
    train_records = load_split(pre_dir / "train.pkl")
    val_records = load_split(pre_dir / "val.pkl")

    train_segs, train_meta = records_to_segments(train_records)
    val_segs, val_meta = records_to_segments(val_records)
    assert train_segs[0].shape[1] == NUM_FEATURES + NUM_TARGETS, (
        f"data has {train_segs[0].shape[1]} columns, expected {NUM_FEATURES + NUM_TARGETS} "
        f"(NUM_FEATURES={NUM_FEATURES} + NUM_TARGETS={NUM_TARGETS}) — config columns changed?"
    )
    train_ds = WindowedDataset(train_segs, train_meta, NUM_TARGETS, pp["seq_len"], pp["stride"])
    val_ds = WindowedDataset(val_segs, val_meta, NUM_TARGETS, pp["seq_len"], pp["stride"])
    logger.info("Train windows: %d | Val windows: %d", len(train_ds), len(val_ds))

    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model = TransformerModel(NUM_FEATURES, NUM_TARGETS, cfg).to(device)

    eval_dir = Path(cfg["paths"]["evaluation"])
    eval_dir = eval_dir.with_stem(eval_dir.stem + suffix)
    eval_dir.mkdir(parents=True, exist_ok=True)
    enabled = tracking.setup(cfg)
    with tracking.training_run(cfg, "amazon_domain", enabled) as run:
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

        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        with Path(cfg["paths"]["scaler"]).open("rb") as f:
            scaler = pickle.load(f)

        figs = [eval_dir / "loss_curves.png", eval_dir / "val_pred_vs_true.png", eval_dir / "val_metrics_boxplot.png"]
        plot_loss_curves(history["train_loss"], history["val_loss"], history["per_target_val"],
                         eval_every=tcfg["eval_every_n_epochs"], save_path=figs[0])
        history_to_dataframe(history, tcfg["eval_every_n_epochs"]).round(3).to_csv(
            eval_dir / "history.csv", index=False)
        seg_meta, pred_list, obs_list = predict_and_inverse(model, val_records, NUM_TARGETS, pp["seq_len"], device, scaler)
        pred_d, obs_d = stack_by_target(pred_list, obs_list, target_names)
        plot_pred_vs_true(pred_d, obs_d, log_scale=False, save_path=figs[1])
        val_metrics = per_unit_metrics(seg_meta, pred_list, obs_list, target_names, ["station_id"])
        plot_metric_boxplot(val_metrics, group_col=None, title="Validation metrics", save_path=figs[2])
        logger.info("Saved training figures to %s", eval_dir)

        if run is not None:
            tracking.log_history(history, target_names, tcfg["eval_every_n_epochs"])
            tracking.log_artifacts([best_model_path, eval_dir / "lr_finder.png", *figs])


if __name__ == "__main__":
    main()
