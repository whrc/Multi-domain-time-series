"""
Multi-domain — Step 3: prediction.

See domains/multi_domain/multi_description.md § "Step 3 — Prediction".

Run inference on the test set for a specified domain, checkpoint stage, and target-set
variant. CLI: --domain {arctic,amazon,rangeland}  --checkpoint {pretrained,finetuned}  --flux-only
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
from domains.multi_domain.flux_only import (  # noqa: E402
    DOMAIN_NTARGETS,
    apply_flux_only,
    checkpoint_path,
    stage_output_dir,
    variant_ntargets,
    variant_target_names,
)
from domains.multi_domain.model import DomainRoutedModel, MultiDomainModel  # noqa: E402
from shared.evaluate import predict_and_inverse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Rangeland column order matches individual domain 03_predict.py exactly (flux-only keeps the
# leading 4 flux columns only)
_RG_POOL_COLS = ["AGB_predicted", "BGB_predicted", "AGL_predicted", "BGL_predicted",
                 "POC_predicted", "HOC_predicted"]
_RG_FLUX_COLS = ["GPP_predicted", "RECO_predicted", "Rm_predicted", "Rg_predicted"]


def _save_arctic(seg_meta: list[dict], pred_list: list[np.ndarray],
                 flux_only: bool, pred_root: Path) -> None:
    """Save per-variable per-grid per-SSP NetCDF files."""
    import xarray as xr
    arctic_cfg = load_config("arctic_domain")
    idx_map    = {k: pd.date_range(v["start"], v["end"], freq="MS")
                  for k, v in arctic_cfg["time"]["scenarios"].items()}
    proj_start = arctic_cfg["time"]["projected_start_year"]

    groups: dict[tuple, list] = defaultdict(list)
    for meta, pred in zip(seg_meta, pred_list):
        groups[(meta["grid"], meta["ssp"])].append((meta, pred))

    target_names = variant_target_names(flux_only)["arctic"]
    n_files = 0
    for (grid, ssp), items in groups.items():
        is_ssp1 = "ssp1" in ssp
        time    = idx_map["ssp1" if is_ssp1 else "ssp5"]
        ny, nx  = items[0][0]["ny"], items[0][0]["nx"]
        pred_root.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(target_names):
            arr = np.full((len(time), ny, nx), np.nan, dtype=np.float32)
            for meta, pred in items:
                arr[:, meta["y"], meta["x"]] = pred[:, i]
            da = xr.DataArray(np.round(arr, 3), dims=("time", "y", "x"), coords={"time": time})
            for split, mask in (("tr", time.year < proj_start), ("sc", time.year >= proj_start)):
                if (not is_ssp1 and split == "tr") or not mask.any():
                    continue
                sub = da.isel(time=np.where(mask)[0]).to_dataset(name=name)
                fname = f"{grid}_{ssp}_{name}_{split}.nc"
                sub.to_netcdf(pred_root / fname, engine="h5netcdf")
                n_files += 1
        logger.info("arctic/%s/%s: %d pixels → %dx%d grid", grid, ssp, len(items), ny, nx)
    logger.info("Saved %d Arctic NetCDF files to %s", n_files, pred_root)


def _save_amazon(seg_meta: list[dict], pred_list: list[np.ndarray], pred_root: Path) -> None:
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
    out_path = pred_root / "amazon_predictions.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Saved %d Amazon rows (%d stations) → %s", len(out), out["station_id"].nunique(), out_path)


def _save_rangeland(seg_meta: list[dict], pred_list: list[np.ndarray],
                    flux_only: bool, pred_root: Path) -> None:
    target_cols = _RG_FLUX_COLS if flux_only else _RG_FLUX_COLS + _RG_POOL_COLS
    ordered = ["site", "date"] + _RG_FLUX_COLS[:2] + ["NEE_predicted"] + _RG_FLUX_COLS[2:]
    if not flux_only:
        ordered += _RG_POOL_COLS
    rows = []
    for meta, pred in zip(seg_meta, pred_list):
        year, month = meta["segment_starts"][meta["seg_idx"]]
        dates = pd.date_range(start=f"{year}-{month:02d}-01", periods=pred.shape[0], freq="MS")
        frame = pd.DataFrame(np.round(pred, 3), columns=target_cols)
        frame.insert(0, "site", meta["site"])
        frame.insert(1, "date", dates)
        frame["NEE_predicted"] = frame["RECO_predicted"] - frame["GPP_predicted"]
        rows.append(frame)
    out = (pd.concat(rows, ignore_index=True)
             .sort_values(["site", "date"])
             .reset_index(drop=True)[ordered])
    pred_root.mkdir(parents=True, exist_ok=True)
    out_path = pred_root / "rangeland_predictions.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Saved %d Rangeland rows (%d sites) → %s", len(out), out["site"].nunique(), out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",     choices=["arctic", "amazon", "rangeland"], required=True)
    parser.add_argument("--checkpoint", choices=["pretrained", "finetuned"], default="finetuned")
    parser.add_argument("--flux-only", action="store_true",
                        help="Load the flux-only checkpoint/target-set variant — see "
                             "02_train.py --flux-only.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Which seeded checkpoint to load (matches --seed in 02_train.py).")
    args   = parser.parse_args()
    domain = args.domain
    stage  = args.checkpoint
    flux_only = args.flux_only

    cfg        = load_config("multi_domain")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(cfg["paths"]["models_dir"])
    pred_root  = Path(cfg["paths"]["predictions_dir"])
    seq_len    = cfg["model"]["seq_len"]
    ntargets   = variant_ntargets(flux_only)
    n_targets  = ntargets[domain]

    ckpt_path = checkpoint_path(models_dir, stage, domain, flux_only, args.seed)
    pred_out  = stage_output_dir(pred_root, stage, domain, flux_only, args.seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    with (Path(cfg["paths"][domain]["preprocessed_dir"]) / "test.pkl").open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"][domain]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)

    # nF_arctic is needed to reconstruct the full model regardless of the target domain —
    # inferred from arctic's raw (full-target) test.pkl, since flux-only only reduces target
    # columns, not feature columns.
    arctic_test_path = Path(cfg["paths"]["arctic"]["preprocessed_dir"]) / "test.pkl"
    if domain == "arctic":
        nF_arctic = test_records[0]["data"].shape[1] - DOMAIN_NTARGETS["arctic"]
    elif arctic_test_path.exists():
        with arctic_test_path.open("rb") as f:
            nF_arctic = pickle.load(f)[0]["data"].shape[1] - DOMAIN_NTARGETS["arctic"]
    else:
        raise FileNotFoundError(f"Cannot determine Arctic nFeatures: {arctic_test_path} missing")

    if flux_only:
        test_records, scaler = apply_flux_only(domain, test_records, scaler)

    domain_specs = {
        "arctic":    {"nFeatures": nF_arctic, "nTargets": ntargets["arctic"]},
        "amazon":    {"nFeatures": 14,        "nTargets": ntargets["amazon"]},
        "rangeland": {"nFeatures": 22,        "nTargets": ntargets["rangeland"]},
    }

    model = MultiDomainModel(cfg, domain_specs).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    logger.info("Loaded checkpoint %s | domain=%s | flux_only=%s | device=%s",
               ckpt_path, domain, flux_only, device)

    domain_model = DomainRoutedModel(model, domain)
    seg_meta, pred_list, _ = predict_and_inverse(domain_model, test_records, n_targets, seq_len, device, scaler)

    if domain == "arctic":
        _save_arctic(seg_meta, pred_list, flux_only, pred_out)
    elif domain == "amazon":
        _save_amazon(seg_meta, pred_list, pred_out)
    else:
        _save_rangeland(seg_meta, pred_list, flux_only, pred_out)


if __name__ == "__main__":
    main()
