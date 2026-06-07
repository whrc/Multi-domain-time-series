# domains/arctic_domain/03_predict.py

import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config

DOMAIN = os.environ.get("ARCTIC_DOMAIN", "arctic_domain")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUM_TARGETS = 4  # ALD, GPP, RECO, VEGC (always the last NUM_TARGETS columns)
SSP1 = "ssp1_2_6_mri_esm2_0"
SSP5 = "ssp5_8_5_mri_esm2_0"


# ── GCS helper ────────────────────────────────────────────────────────────────

def _open_nc(fs: gcsfs.GCSFileSystem, path: str) -> xr.Dataset:
    with fs.open(path, "rb") as f:
        magic = f.read(4)
    engine = "scipy" if magic[:3] == b"CDF" else "h5netcdf"
    return xr.open_dataset(fs.open(path), engine=engine, mask_and_scale=True)


def _monthly_index(start_year: int, end_year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_checkpoint(cfg: dict) -> tuple[nn.Module, int]:
    path = Path(cfg["paths"]["best_model"])
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}. Run 02_train.py first.")
    ckpt = torch.load(path, weights_only=False)
    num_features: int = ckpt["num_features"]
    arch = ckpt["cfg"]["model"]["architecture"]
    if arch == "transformer":
        from models.transformer import TransformerModel
        model = TransformerModel(num_features, NUM_TARGETS, ckpt["cfg"])
    elif arch == "lstm":
        from models.lstm import LSTMModel
        model = LSTMModel(num_features, NUM_TARGETS, ckpt["cfg"])
    else:
        raise ValueError(f"Unknown architecture '{arch}'")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Loaded checkpoint from %s (epoch=%d  val_loss=%.4f)", path, ckpt["epoch"], ckpt["val_loss"])
    return model, num_features


def load_test_data(cfg: dict) -> tuple[list[dict], dict]:
    base = Path(cfg["paths"]["preprocessed_dir"])
    with open(base / "test.pkl", "rb") as f:
        test_records = pickle.load(f)
    with open(cfg["paths"]["scaler"], "rb") as f:
        scaler = pickle.load(f)
    log.info("Loaded %d test records", len(test_records))
    return test_records, scaler


def get_grid_coords(
    grid: str, ssp: str, cfg: dict, fs: gcsfs.GCSFileSystem, cache: dict
) -> dict:
    """Read spatial coordinates from one reference static file. Cached per grid."""
    if grid in cache:
        return cache[grid]
    path = f"{cfg['gcs']['bucket']}/{grid}/{ssp}/soil-texture.nc"
    ds = _open_nc(fs, path)
    rename = {c: c.lower() for c in ds.coords if c in ("Y", "X")}
    if rename:
        ds = ds.rename(rename)
    coords = {
        "y":   ds.coords["y"].values,
        "x":   ds.coords["x"].values,
        "lat": ds["lat"].values if "lat" in ds.data_vars else None,
        "lon": ds["lon"].values if "lon" in ds.data_vars else None,
    }
    cache[grid] = coords
    return coords


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_all_pixels(
    records: list[dict],
    model: nn.Module,
    scaler: dict,
    device: torch.device,
    num_features: int,
    seq_len: int,
) -> np.ndarray:
    """
    Batched inference across all pixels for one SSP.
    Each chunk iterates over time; all N pixels are processed in one forward pass.
    Returns: (N, T, NUM_TARGETS) in original (de-normalised) space.
    """
    N = len(records)
    T = records[0]["data"].shape[0]
    features = np.stack([r["data"][:, :num_features] for r in records])  # (N, T, F)

    preds = np.zeros((N, T, NUM_TARGETS), dtype=np.float32)
    for start in range(0, T, seq_len):
        chunk = features[:, start : start + seq_len, :]       # (N, chunk_len, F)
        chunk_len = chunk.shape[1]
        if chunk_len < seq_len:
            pad = np.zeros((N, seq_len - chunk_len, chunk.shape[2]), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad], axis=1)
        x = torch.from_numpy(chunk).float().to(device)        # (N, seq_len, F)
        with torch.no_grad():
            out = model(x)                                     # (N, seq_len, 4)
        preds[:, start : start + chunk_len, :] = out[:, :chunk_len, :].cpu().numpy()

    mean = scaler["mean"][num_features:]                       # (NUM_TARGETS,)
    std  = scaler["std"][num_features:]
    return preds * std + mean                                  # (N, T, 4)


# ── Spatial reconstruction ────────────────────────────────────────────────────

def build_spatial_predictions(
    records: list[dict],
    model: nn.Module,
    scaler: dict,
    coords: dict,
    cfg: dict,
    device: torch.device,
    num_features: int,
    seq_len: int,
) -> dict[str, xr.Dataset]:
    """
    Run inference for all test pixels in this grid. Returns a dict keyed by ssp,
    each value a monthly xr.Dataset with all target variables at (time, y, x).
    """
    target_names = [t["name"] for t in cfg["targets"]]
    ny, nx = len(coords["y"]), len(coords["x"])
    ssp1_idx = _monthly_index(1901, 2100)
    ssp5_idx = _monthly_index(2025, 2100)

    # Pre-allocate full spatial grids (NaN everywhere)
    grids: dict[str, dict[str, np.ndarray]] = {
        SSP1: {name: np.full((len(ssp1_idx), ny, nx), np.nan, dtype=np.float32) for name in target_names},
        SSP5: {name: np.full((len(ssp5_idx), ny, nx), np.nan, dtype=np.float32) for name in target_names},
    }

    for ssp, time_idx in [(SSP1, ssp1_idx), (SSP5, ssp5_idx)]:
        ssp_records = [r for r in records if r["ssp"] == ssp]
        if not ssp_records:
            continue
        pred_orig = predict_all_pixels(ssp_records, model, scaler, device, num_features, seq_len)
        for i, rec in enumerate(ssp_records):
            yi, xi = rec["y"], rec["x"]
            for ti, name in enumerate(target_names):
                grids[ssp][name][:, yi, xi] = pred_orig[i, :, ti]

    result: dict[str, xr.Dataset] = {}
    for ssp, time_idx in [(SSP1, ssp1_idx), (SSP5, ssp5_idx)]:
        ssp_records = [r for r in records if r["ssp"] == ssp]
        if not ssp_records:
            continue
        data_vars: dict[str, xr.DataArray] = {}
        for name in target_names:
            data_vars[name] = xr.DataArray(
                grids[ssp][name],
                dims=["time", "y", "x"],
                coords={"time": time_idx, "y": coords["y"], "x": coords["x"]},
            )
        ds = xr.Dataset(data_vars)
        if coords["lat"] is not None:
            ds["lat"] = xr.DataArray(coords["lat"], dims=["y", "x"])
            ds["lon"] = xr.DataArray(coords["lon"], dims=["y", "x"])
        result[ssp] = ds

    log.info("  Ran inference on %d test records", len(records))
    return result


# ── Split, aggregate, save ────────────────────────────────────────────────────

def split_time(ds_monthly: xr.Dataset, ssp: str) -> dict[str, xr.Dataset]:
    """
    Split monthly dataset into historical / projected slices (no aggregation).
    Returns {'tr': ..., 'sc': ...} for SSP1-2.6 or {'sc': ...} for SSP5-8.5.
    """
    splits: dict[str, xr.Dataset] = {}
    if ssp == SSP1:
        hist_end   = pd.Timestamp("2024-12-01")
        proj_start = pd.Timestamp("2025-01-01")
        splits["tr"] = ds_monthly.sel(time=ds_monthly.time <= hist_end)
        splits["sc"] = ds_monthly.sel(time=ds_monthly.time >= proj_start)
    else:
        splits["sc"] = ds_monthly
    return splits


def save_predictions(
    split_ds: dict[str, xr.Dataset], grid: str, ssp: str, cfg: dict
) -> None:
    """
    Save one NetCDF per variable per split. Yearly targets (ALD, VEGC) are
    aggregated to annual mean here, before writing.
    """
    out_root = Path(cfg["paths"]["predictions"]) / grid / ssp
    out_root.mkdir(parents=True, exist_ok=True)

    for tgt in cfg["targets"]:
        name       = tgt["name"]
        resolution = tgt["resolution"]
        for split_key, ds in split_ds.items():
            da = ds[name]
            if resolution == "yearly":
                da = da.resample(time="YS").mean()
            out_ds = da.to_dataset(name=name)
            if "lat" in ds:
                out_ds["lat"] = ds["lat"]
                out_ds["lon"] = ds["lon"]
            out_path = out_root / f"{name}_{resolution}_pred_{split_key}.nc"
            out_ds.to_netcdf(out_path)
            log.info("  Saved %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = load_config(DOMAIN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model, num_features = load_checkpoint(cfg)
    model = model.to(device)
    test_records, scaler = load_test_data(cfg)
    seq_len = cfg["preprocessing"]["seq_len"]

    fs = gcsfs.GCSFileSystem(access="read_only")
    coord_cache: dict[str, dict] = {}

    # Group test records by grid
    by_grid: dict[str, list[dict]] = defaultdict(list)
    for rec in test_records:
        by_grid[rec["grid"]].append(rec)

    total_files = 0
    for grid, records in by_grid.items():
        log.info("=== Grid: %s  (%d test records) ===", grid, len(records))
        ref_ssp = records[0]["ssp"]
        coords = get_grid_coords(grid, ref_ssp, cfg, fs, coord_cache)
        ssp_ds = build_spatial_predictions(records, model, scaler, coords, cfg, device, num_features, seq_len)
        for ssp, ds_monthly in ssp_ds.items():
            split_ds = split_time(ds_monthly, ssp)
            save_predictions(split_ds, grid, ssp, cfg)
            total_files += sum(
                len([n for n in ds.data_vars if n not in ("lat", "lon")])
                for ds in split_ds.values()
            )

    log.info("Prediction complete. %d NetCDF files written.", total_files)


if __name__ == "__main__":
    main()
