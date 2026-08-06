"""
Amazon domain — Step 3: prediction.

See domains/amazon_domain/amazon_description.md § "Step 3 — Prediction".

Run dense inference on the test set, inverse-transform, and save predictions with
station_id, year, month, and the three predicted target columns.
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
from shared.transformer import TransformerModel  # noqa: E402
from shared import tracking  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_FEATURES = 14
NUM_TARGETS = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Which seeded checkpoint to load (matches --seed in 02_train.py).")
    parser.add_argument("--capacity-matched", action="store_true",
                        help="Ablation only — load the capacity-matched checkpoint (matches "
                             "--capacity-matched in 02_train.py). See "
                             "ablation_test/ablation_description.md.")
    args = parser.parse_args()

    cfg = load_config("amazon_domain")
    pp = cfg["preprocessing"]
    target_names = cfg["targets"]
    if args.capacity_matched:
        cfg["model"] = {**cfg["model"], **cfg["model_capacity_matched"]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = f"_seed{args.seed}" if args.seed is not None else ""
    suffix += "_capmatched" if args.capacity_matched else ""
    best_model_path = Path(cfg["paths"]["best_model"])
    best_model_path = best_model_path.with_stem(best_model_path.stem + suffix)

    with (Path(cfg["paths"]["preprocessed_dir"]) / "test.pkl").open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)

    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    assert (ckpt["num_features"], ckpt["num_targets"]) == (NUM_FEATURES, NUM_TARGETS), (
        f"checkpoint dims {(ckpt['num_features'], ckpt['num_targets'])} != "
        f"{(NUM_FEATURES, NUM_TARGETS)} — retrain or fix the constants"
    )
    model = TransformerModel(NUM_FEATURES, NUM_TARGETS, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    seg_meta, pred_list, _ = predict_and_inverse(model, test_records, NUM_TARGETS, pp["seq_len"], device, scaler)
    # Targets were log1p-transformed before the scaler fit (01_preprocess.py); predict_and_inverse
    # only undoes the z-score, so undo the log1p here to get back to physical units.
    pred_list = [np.expm1(p) for p in pred_list]
    # discharge was additionally area-normalized (specific discharge) before log1p — multiply
    # back by each station's drainage_area (carried on seg_meta via records_to_segments) to
    # report discharge in physical units.
    discharge_idx = target_names.index("discharge")
    for pred, meta in zip(pred_list, seg_meta):
        pred[:, discharge_idx] *= meta["drainage_area"]

    pred_cols = [f"{t}_pred" for t in target_names]
    rows = []
    for meta, pred in zip(seg_meta, pred_list):
        year, month = meta["segment_starts"][meta["seg_idx"]]
        idx = pd.date_range(start=f"{year}-{month:02d}-01", periods=pred.shape[0], freq="MS")
        frame = pd.DataFrame(pred, columns=pred_cols)
        frame.insert(0, "station_id", meta["station_id"])
        frame.insert(1, "year", idx.year)
        frame.insert(2, "month", idx.month)
        rows.append(frame)

    out = pd.concat(rows, ignore_index=True).sort_values(["station_id", "year", "month"]).reset_index(drop=True)
    out[pred_cols] = out[pred_cols].round(3)
    out["active_fire_count_pred"] = out["active_fire_count_pred"].round()  # count is discrete; cosmetic only

    out_path = Path(cfg["paths"]["predictions"]) / f"amazon_test_predictions{suffix}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info("Saved %d prediction rows for %d stations to %s", len(out), out["station_id"].nunique(), out_path)

    enabled = tracking.setup(cfg)
    run_id = tracking.read_run_id(best_model_path.with_suffix(".run_id")) if enabled else None
    with tracking.resume_run(run_id) as active:
        if active:
            tracking.log_prediction_complete()
            logger.info("Logged prediction completion to MLflow run %s", run_id)


if __name__ == "__main__":
    main()
