"""
Rangeland domain — Step 3: prediction.

See domains/rangeland_domain/rangeland_description.md § "Step 3 — Prediction".

Run dense inference on the test set, inverse-transform, derive NEE = RECO - GPP, and
save one parquet with site, date, and the 11 predicted columns.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.evaluate import predict_and_inverse  # noqa: E402
from shared.transformer import TransformerModel  # noqa: E402
from shared import tracking  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_FEATURES = 22
NUM_TARGETS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux-only", action="store_true",
                        help="Load the flux-only checkpoint (matches --flux-only in "
                             "02_train.py) instead of the full-target one. Output parquet "
                             "has only the 4 flux columns (no NEE, since that needs GPP+RECO "
                             "only — which are included either way — but no pool columns).")
    args = parser.parse_args()

    cfg = load_config("rangeland_domain")
    pp = cfg["preprocessing"]
    flux_names = cfg["targets"]["fluxes"]
    target_names = flux_names if args.flux_only else flux_names + cfg["targets"]["pools"]
    num_targets = len(target_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with (Path(cfg["paths"]["preprocessed_dir"]) / "test.pkl").open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)
    if args.flux_only:
        keep = NUM_FEATURES + len(flux_names)
        test_records = [{**r, "segments": [s[:, :keep] for s in r["segments"]]} for r in test_records]
        scaler = {"mean": scaler["mean"][:keep], "std": scaler["std"][:keep]}

    suffix = "_fluxonly" if args.flux_only else ""
    best_model_path = Path(cfg["paths"]["best_model"])
    best_model_path = best_model_path.with_stem(best_model_path.stem + suffix)
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    assert (ckpt["num_features"], ckpt["num_targets"]) == (NUM_FEATURES, num_targets), (
        f"checkpoint dims {(ckpt['num_features'], ckpt['num_targets'])} != "
        f"{(NUM_FEATURES, num_targets)} — retrain or fix the constants"
    )
    model = TransformerModel(NUM_FEATURES, num_targets, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    seg_meta, pred_list, _ = predict_and_inverse(model, test_records, num_targets, pp["seq_len"], device, scaler)

    rows = []
    for meta, pred in zip(seg_meta, pred_list):
        year, month = meta["segment_starts"][meta["seg_idx"]]
        dates = pd.date_range(start=f"{year}-{month:02d}-01", periods=pred.shape[0], freq="MS")
        frame = pd.DataFrame(pred, columns=target_names)
        frame.insert(0, "site", meta["site"])
        frame.insert(1, "date", dates)
        # NEE convention: RECO - GPP — both are always present, flux-only or not.
        frame["NEE_predicted"] = frame["RECO_predicted"] - frame["GPP_predicted"]
        rows.append(frame)

    out = pd.concat(rows, ignore_index=True).sort_values(["site", "date"]).reset_index(drop=True)
    ordered = ["site", "date", *target_names, "NEE_predicted"]
    pred_cols = [c for c in ordered if c not in ("site", "date")]
    out[pred_cols] = out[pred_cols].round(3)
    out = out[ordered]

    out_path = Path(cfg["paths"]["predictions"]) / f"predictions{suffix}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info("Saved %d prediction rows for %d sites to %s", len(out), out["site"].nunique(), out_path)

    enabled = tracking.setup(cfg)
    run_id = tracking.read_run_id(best_model_path.with_suffix(".run_id")) if enabled else None
    with tracking.resume_run(run_id) as active:
        if active:
            tracking.log_prediction_complete()
            logger.info("Logged prediction completion to MLflow run %s", run_id)


if __name__ == "__main__":
    main()
