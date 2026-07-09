"""
Arctic domain — Step 3: prediction.

See domains/arctic_domain/arctic_description.md § "Step 3 — Prediction".

Run dense inference on the test pixels, inverse-transform, reconstruct gridded
(time, y, x) arrays per (grid, ssp), and save NetCDF per variable, split into
historical (_tr, time < 2025) and projected (_sc, time >= 2025) files.

OPT-IN / NOT part of the default pipeline (run_arctic.py excludes it unless
--include-predict is passed): a full dense (time, y, x) grid is saved per circumpolar tile
even though only a handful of that tile's pixels are actually in the test set, so real-scale
output can reach hundreds of GB. Not needed for evaluation metrics or figures —
04_evaluate.py recomputes predictions from the checkpoint directly, without reading this
script's output at all.
"""

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.evaluate import predict_and_inverse  # noqa: E402
from shared.transformer import TransformerModel  # noqa: E402
from shared import tracking  # noqa: E402
from domains.arctic_domain._naming import load_stride_seq_len, run_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_TARGETS = 4


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

    logger.warning(
        "03_predict.py writes a full dense NetCDF grid per circumpolar tile — real-scale "
        "output can reach hundreds of GB. Not required for evaluation metrics/figures "
        "(04_evaluate.py recomputes predictions from the checkpoint directly)."
    )

    cfg = load_config("arctic_domain")
    train_size = args.train_size if args.train_size is not None else cfg["preprocessing"]["train_size"]
    label = run_label(train_size, args.label)
    idx_map = {k: pd.date_range(v["start"], v["end"], freq="MS") for k, v in cfg["time"]["scenarios"].items()}
    proj_start = cfg["time"]["projected_start_year"]
    target_names = [t["name"] for t in cfg["targets"]]
    resolution = {t["name"]: t["resolution"] for t in cfg["targets"]}
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
    assert (ckpt["num_features"], ckpt["num_targets"]) == (num_features, NUM_TARGETS), (
        f"checkpoint dims {(ckpt['num_features'], ckpt['num_targets'])} != "
        f"{(num_features, NUM_TARGETS)} — test data and checkpoint disagree"
    )
    model = TransformerModel(num_features, NUM_TARGETS, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    seg_meta, pred_list, _ = predict_and_inverse(model, test_records, NUM_TARGETS, seq_len, device, scaler)

    groups: dict[tuple, list] = defaultdict(list)
    for meta, pred in zip(seg_meta, pred_list):
        groups[(meta["grid"], meta["ssp"])].append((meta, pred))

    pred_root = Path(cfg["paths"]["predictions"]) / label
    n_files = 0
    for (grid, ssp), items in groups.items():
        is_ssp1 = "ssp1" in ssp
        time = idx_map["ssp1" if is_ssp1 else "ssp5"]
        ny, nx = items[0][0]["ny"], items[0][0]["nx"]
        out_dir = pred_root / grid / ssp
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(target_names):
            arr = np.full((len(time), ny, nx), np.nan, dtype=np.float32)
            for meta, pred in items:
                arr[:, meta["y"], meta["x"]] = pred[:, i]
            if resolution[name] == "yearly":
                arr[time.month != 1, :, :] = np.nan
            da = xr.DataArray(np.round(arr, 3), dims=("time", "y", "x"), coords={"time": time})
            for split, mask in (("tr", time.year < proj_start), ("sc", time.year >= proj_start)):
                if (not is_ssp1 and split == "tr") or not mask.any():
                    continue
                sub = da.isel(time=np.where(mask)[0]).to_dataset(name=name)
                sub.to_netcdf(out_dir / f"{name}_{resolution[name]}_pred_{split}.nc", engine="h5netcdf")
                n_files += 1
        logger.info("%s/%s: reconstructed %d pixels onto %dx%d grid", grid, ssp, len(items), ny, nx)
    logger.info("Saved %d NetCDF prediction files under %s", n_files, pred_root)

    enabled = tracking.setup(cfg)
    run_id = tracking.read_run_id(best_model_path.with_suffix(".run_id")) if enabled else None
    with tracking.resume_run(run_id) as active:
        if active:
            tracking.log_prediction_complete()
            logger.info("Logged prediction completion to MLflow run %s", run_id)


if __name__ == "__main__":
    main()
