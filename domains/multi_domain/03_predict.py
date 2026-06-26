"""
Multi-domain — Step 3: prediction.

See domains/multi_domain/multi_description.md § "Step 3 — Prediction".

Run inference on the test set for a specified domain and checkpoint stage.
CLI: --domain {arctic,amazon,rangeland}  --checkpoint {stage1,stage2}
"""

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from domains.multi_domain.model import MultiDomainModel  # noqa: E402
from shared.dataset import records_to_segments  # noqa: E402
from shared.evaluate import predict_and_inverse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOMAIN_NTARGETS = {"arctic": 4, "amazon": 3, "rangeland": 10}

# Rangeland column order matches individual domain 03_predict.py exactly
_RG_TARGET_COLS = [
    "GPP_predicted", "RECO_predicted", "Rm_predicted", "Rg_predicted",
    "AGB_predicted", "BGB_predicted", "AGL_predicted", "BGL_predicted",
    "POC_predicted", "HOC_predicted",
]
_RG_ORDERED = ["site", "date"] + ["GPP_predicted", "RECO_predicted", "NEE_predicted",
               "Rm_predicted", "Rg_predicted", "AGB_predicted", "BGB_predicted",
               "AGL_predicted", "BGL_predicted", "POC_predicted", "HOC_predicted"]


def _save_arctic(seg_meta: list[dict], pred_list: list[np.ndarray],
                 checkpoint_tag: str, pred_root: Path) -> None:
    """Save per-variable per-grid per-SSP NetCDF files."""
    import xarray as xr
    arctic_cfg = load_config("arctic_domain")
    idx_map    = {k: pd.date_range(v["start"], v["end"], freq="MS")
                  for k, v in arctic_cfg["time"]["scenarios"].items()}
    proj_start = arctic_cfg["time"]["projected_start_year"]

    groups: dict[tuple, list] = defaultdict(list)
    for meta, pred in zip(seg_meta, pred_list):
        groups[(meta["grid"], meta["ssp"])].append((meta, pred))

    target_names = ["ALD", "GPP", "RECO", "VEGC"]
    n_files = 0
    for (grid, ssp), items in groups.items():
        is_ssp1 = "ssp1" in ssp
        time    = idx_map["ssp1" if is_ssp1 else "ssp5"]
        ny, nx  = items[0][0]["ny"], items[0][0]["nx"]
        out_dir = pred_root
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(target_names):
            arr = np.full((len(time), ny, nx), np.nan, dtype=np.float32)
            for meta, pred in items:
                arr[:, meta["y"], meta["x"]] = pred[:, i]
            da = xr.DataArray(np.round(arr, 3), dims=("time", "y", "x"), coords={"time": time})
            for split, mask in (("tr", time.year < proj_start), ("sc", time.year >= proj_start)):
                if (not is_ssp1 and split == "tr") or not mask.any():
                    continue
                sub = da.isel(time=np.where(mask)[0]).to_dataset(name=name)
                fname = f"arctic_{checkpoint_tag}_{grid}_{ssp}_{name}_{split}.nc"
                sub.to_netcdf(out_dir / fname, engine="h5netcdf")
                n_files += 1
        logger.info("arctic/%s/%s: %d pixels → %dx%d grid", grid, ssp, len(items), ny, nx)
    logger.info("Saved %d Arctic NetCDF files to %s", n_files, pred_root)


def _save_amazon(seg_meta: list[dict], pred_list: list[np.ndarray],
                 checkpoint_tag: str, pred_root: Path) -> None:
    target_cols = ["discharge_pred", "active_fire_count_pred", "burned_area_pred"]
    rows = []
    for meta, pred in zip(seg_meta, pred_list):
        year, month = meta["segment_starts"][meta["seg_idx"]]
        idx   = pd.date_range(start=f"{year}-{month:02d}-01", periods=pred.shape[0], freq="MS")
        frame = pd.DataFrame(np.round(pred, 3), columns=target_cols)
        frame.insert(0, "station_id", meta["station_id"])
        frame.insert(1, "year",  idx.year)
        frame.insert(2, "month", idx.month)
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True).sort_values(["station_id", "year", "month"]).reset_index(drop=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    out_path = pred_root / f"amazon_{checkpoint_tag}_predictions.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Saved %d Amazon rows (%d stations) → %s", len(out), out["station_id"].nunique(), out_path)


def _save_rangeland(seg_meta: list[dict], pred_list: list[np.ndarray],
                    checkpoint_tag: str, pred_root: Path) -> None:
    rows = []
    for meta, pred in zip(seg_meta, pred_list):
        year, month = meta["segment_starts"][meta["seg_idx"]]
        dates = pd.date_range(start=f"{year}-{month:02d}-01", periods=pred.shape[0], freq="MS")
        frame = pd.DataFrame(np.round(pred, 3), columns=_RG_TARGET_COLS)
        frame.insert(0, "site", meta["site"])
        frame.insert(1, "date", dates)
        frame["NEE_predicted"] = frame["RECO_predicted"] - frame["GPP_predicted"]
        rows.append(frame)
    out = (pd.concat(rows, ignore_index=True)
             .sort_values(["site", "date"])
             .reset_index(drop=True)[_RG_ORDERED])
    pred_root.mkdir(parents=True, exist_ok=True)
    out_path = pred_root / f"rangeland_{checkpoint_tag}_predictions.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Saved %d Rangeland rows (%d sites) → %s", len(out), out["site"].nunique(), out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",     choices=["arctic", "amazon", "rangeland"], required=True)
    parser.add_argument("--checkpoint", choices=["stage1", "stage2"], default="stage2")
    args   = parser.parse_args()
    domain = args.domain
    ckpt_tag = args.checkpoint

    cfg        = load_config("multi_domain")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(cfg["paths"]["models_dir"])
    pred_root  = Path(cfg["paths"]["predictions_dir"])
    seq_len    = cfg["model"]["seq_len"]
    n_targets  = DOMAIN_NTARGETS[domain]

    # Load checkpoint
    ckpt_file = "stage1_best.pt" if ckpt_tag == "stage1" else f"stage2_{domain}_best.pt"
    ckpt_path = models_dir / ckpt_file
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    with (Path(cfg["paths"][domain]["preprocessed_dir"]) / "test.pkl").open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"][domain]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)

    # nF_arctic is needed to reconstruct the full model regardless of the target domain
    arctic_test_path = Path(cfg["paths"]["arctic"]["preprocessed_dir"]) / "test.pkl"
    if domain == "arctic":
        nF_arctic = test_records[0]["data"].shape[1] - 4
    elif arctic_test_path.exists():
        with arctic_test_path.open("rb") as f:
            nF_arctic = pickle.load(f)[0]["data"].shape[1] - 4
    else:
        raise FileNotFoundError(f"Cannot determine Arctic nFeatures: {arctic_test_path} missing")

    domain_specs = {
        "arctic":    {"nFeatures": nF_arctic, "nTargets": 4},
        "amazon":    {"nFeatures": 14,        "nTargets": 3},
        "rangeland": {"nFeatures": 22,        "nTargets": 10},
    }

    model = MultiDomainModel(cfg, domain_specs).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    logger.info("Loaded checkpoint %s | domain=%s | device=%s", ckpt_file, domain, device)

    domain_model = lambda x, _d=domain: model(x, domain=_d)
    seg_meta, pred_list, _ = predict_and_inverse(domain_model, test_records, n_targets, seq_len, device, scaler)

    if domain == "arctic":
        _save_arctic(seg_meta, pred_list, ckpt_tag, pred_root)
    elif domain == "amazon":
        _save_amazon(seg_meta, pred_list, ckpt_tag, pred_root)
    else:
        _save_rangeland(seg_meta, pred_list, ckpt_tag, pred_root)


if __name__ == "__main__":
    main()
