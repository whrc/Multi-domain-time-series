"""
Arctic domain — Step 1: preprocessing.

See domains/arctic_domain/arctic_description.md § "Step 1 — Preprocessing".

Per grid x scenario: assemble static + CO2 + climate features and the four TEM targets
onto a monthly axis, build per-pixel sequences, split by pixel, fit the scaler on train,
normalise, and save train/val/test pkl + scaler.

Coordinates have no stored values, so all variables (28x32 per grid) are aligned
positionally. lat/lon are kept per pixel for evaluation. Feature-column NaNs (sparse,
e.g. fire fields on some land pixels) are mean-imputed to 0 after z-scoring; target NaNs
are preserved and masked by the loss.
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.io import gcs_filesystem, read_netcdf  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_TARGETS = 4
CLIM_VARS = ["tair", "precip", "nirr", "vapor_press"]
EXCLUDE = {"lat", "lon", "lambert_azimuthal_equal_area"}


def monthly_index_map(cfg: dict) -> dict[str, pd.DatetimeIndex]:
    """Per-scenario monthly DatetimeIndex from config (ssp1 -> 2400 months, ssp5 -> 912)."""
    return {k: pd.date_range(v["start"], v["end"], freq="MS") for k, v in cfg["time"]["scenarios"].items()}


def _to_y_x(ds: xr.Dataset) -> xr.Dataset:
    """Rename uppercase input coords Y/X to lowercase y/x to match the targets."""
    return ds.rename({k: k.lower() for k in ("Y", "X") if k in ds.dims})


def _month_start(times) -> pd.DatetimeIndex:
    """Map any (cftime or datetime) time stamps to first-of-month, for clean reindexing."""
    return pd.to_datetime([f"{t.year}-{t.month:02d}-01" for t in times])


def load_static(fs, inp, cfg) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Stack all static data vars (excluding lat/lon/CRS) into (nStatic, ny, nx); return lat/lon too."""
    names, layers = [], []
    lat = lon = None
    for fname in cfg["inputs"]["static"]:
        ds = _to_y_x(read_netcdf(fs, inp(fname)))
        for v in ds.data_vars:
            if v == "lat":
                lat = ds[v].values
            elif v == "lon":
                lon = ds[v].values
            elif v not in EXCLUDE:
                names.append(v)
                layers.append(ds[v].values)
    return names, np.stack(layers, axis=0), lat, lon


def load_co2(fs, inp, is_ssp1: bool, monthly_index: pd.DatetimeIndex) -> np.ndarray:
    """Yearly CO2 -> monthly via linear interpolation; returns (T,)."""
    files = ["co2.nc", "projected-co2.nc"] if is_ssp1 else ["projected-co2.nc"]
    years, vals = [], []
    for f in files:
        ds = read_netcdf(fs, inp(f))
        years.append(ds["year"].values)
        vals.append(ds["co2"].values)
    s = pd.Series(np.concatenate(vals), index=pd.to_datetime([f"{int(y)}-01-01" for y in np.concatenate(years)]))
    s = s[~s.index.duplicated()]
    # interpolate('time') does not extrapolate, so months past the last yearly anchor
    # (e.g. Feb-Dec 2100) stay NaN; ffill/bfill carries the nearest annual value instead.
    s = s.reindex(s.index.union(monthly_index)).interpolate("time").reindex(monthly_index).ffill().bfill()
    return s.to_numpy()


def load_climate(fs, inp, is_ssp1: bool, monthly_index: pd.DatetimeIndex) -> np.ndarray:
    """Climate vars onto the monthly axis; returns (T, ny, nx, 4)."""
    files = ["historic-climate.nc", "projected-climate.nc"] if is_ssp1 else ["projected-climate.nc"]
    parts = []
    for f in files:
        ds = _to_y_x(read_netcdf(fs, inp(f)))
        parts.append(ds.assign_coords(time=_month_start(ds["time"].values)))
    full = xr.concat(parts, dim="time", data_vars="minimal") if len(parts) > 1 else parts[0]
    return np.stack([full[v].reindex(time=monthly_index).values for v in CLIM_VARS], axis=-1)


def load_targets(fs, tgt, is_ssp1: bool, targets_cfg: list[dict], monthly_index: pd.DatetimeIndex,
                 proj_start: int) -> np.ndarray:
    """The four targets onto the monthly axis; returns (T, ny, nx, 4) in config order.

    Historical (_tr) only exists for SSP1. Yearly targets (ALD, VEGC) appear at January
    positions only; their projected files carry wrong year labels (1901-) which are
    overridden to 2025-.
    """
    layers = []
    for t in targets_cfg:
        var = t["name"]
        yearly = t["resolution"] == "yearly"
        parts = []
        if is_ssp1:
            ds_tr = read_netcdf(fs, tgt(t["historical"]))
            parts.append(ds_tr[var].assign_coords(time=_month_start(ds_tr["time"].values)))
        ds_sc = read_netcdf(fs, tgt(t["projected"]))
        times = ds_sc["time"].values
        if yearly and times[0].year < 2000:  # wrong projected labels -> projected_start_year..
            fixed = pd.to_datetime([f"{proj_start + i}-01-01" for i in range(len(times))])
        else:
            fixed = _month_start(times)
        parts.append(ds_sc[var].assign_coords(time=fixed))
        da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
        layers.append(da.reindex(time=monthly_index).values)
    return np.stack(layers, axis=-1)


def build_records(cfg: dict, grids: list[str]) -> list[dict]:
    """Assemble per-pixel raw (un-normalised) records over all grids and scenarios."""
    fs = gcs_filesystem()
    bucket = cfg["gcs"]["bucket"].replace("gs://", "")
    sub = cfg["gcs"]["target_subfolder"]
    idx_map = monthly_index_map(cfg)
    proj_start = cfg["time"]["projected_start_year"]
    records = []
    for grid in grids:
        for scenario in cfg["scenarios"]:
            is_ssp1 = "ssp1" in scenario
            monthly_index = idx_map["ssp1" if is_ssp1 else "ssp5"]

            def inp(f: str, _g=grid, _s=scenario) -> str:
                return f"{bucket}/{_g}/{_s}/{f}"

            def tgt(f: str, _g=grid, _s=scenario) -> str:
                return f"{bucket}/{_g}/{_s}_split/{sub}/{f}"

            _, static, lat, lon = load_static(fs, inp, cfg)
            co2 = load_co2(fs, inp, is_ssp1, monthly_index)
            climate = load_climate(fs, inp, is_ssp1, monthly_index)
            targets = load_targets(fs, tgt, is_ssp1, cfg["targets"], monthly_index, proj_start)
            T = len(monthly_index)
            nStatic, ny, nx = static.shape
            logger.info("%s/%s: T=%d static=%d grid=%dx%d", grid, scenario, T, nStatic, ny, nx)

            co2_col = co2[:, None]
            kept = 0
            for iy in range(ny):
                for ix in range(nx):
                    tgt_px = targets[:, iy, ix, :]
                    if np.all(np.isnan(tgt_px)):  # ocean pixel
                        continue
                    feat = np.concatenate(
                        [np.tile(static[:, iy, ix], (T, 1)), co2_col, climate[:, iy, ix, :]], axis=1
                    )
                    data = np.concatenate([feat, tgt_px], axis=1).astype(np.float32)
                    records.append({
                        "grid": grid, "ssp": scenario, "y": iy, "x": ix, "ny": ny, "nx": nx,
                        "lat": float(lat[iy, ix]), "lon": float(lon[iy, ix]), "data": data,
                    })
                    kept += 1
            logger.info("%s/%s: kept %d land pixels", grid, scenario, kept)
    return records


def fit_scaler(records: list[dict], split: dict, ncol: int) -> dict[str, np.ndarray]:
    """Column-wise nanmean/nanstd over train pixels via streaming sums (memory-friendly)."""
    s = np.zeros(ncol)
    ss = np.zeros(ncol)
    c = np.zeros(ncol)
    for r in records:
        if split[(r["grid"], r["y"], r["x"])] != "train":
            continue
        d = r["data"].astype(float)
        valid = ~np.isnan(d)
        s += np.nansum(d, axis=0)
        ss += np.nansum(d * d, axis=0)
        c += valid.sum(axis=0)
    mean = s / c
    std = np.sqrt(np.clip(ss / c - mean ** 2, 0, None))
    mean[~np.isfinite(mean)] = 0.0
    std[(std == 0) | ~np.isfinite(std)] = 1.0
    return {"mean": mean, "std": std}


def main() -> None:
    cfg = load_config("arctic_domain")
    pp = cfg["preprocessing"]
    grids = pp.get("grids")
    if not grids:
        fs = gcs_filesystem()
        bucket = cfg["gcs"]["bucket"].replace("gs://", "")
        grids = sorted(p.split("/")[-1] for p in fs.ls(bucket) if fs.isdir(p))
    logger.info("Grids: %s", grids)

    records = build_records(cfg, grids)
    ncol = records[0]["data"].shape[1]
    n_features = ncol - NUM_TARGETS
    logger.info("Built %d pixel-records | nFeatures=%d nTargets=%d", len(records), n_features, NUM_TARGETS)

    keys = sorted({(r["grid"], r["y"], r["x"]) for r in records})
    rng = np.random.default_rng(pp["random_seed"])
    rng.shuffle(keys)
    n = len(keys)
    n_train = round(pp["train_frac"] * n)
    n_val = round(pp["val_frac"] * n)
    split = {k: ("train" if i < n_train else "val" if i < n_train + n_val else "test") for i, k in enumerate(keys)}
    logger.info("Pixel split train/val/test = %d/%d/%d", n_train, n_val, n - n_train - n_val)

    scaler = fit_scaler(records, split, ncol)

    out_dir = Path(cfg["paths"]["preprocessed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    n_imputed = 0
    for r in records:
        d = (r["data"] - scaler["mean"]) / scaler["std"]
        n_imputed += int(np.isnan(d[:, :n_features]).sum())
        d[:, :n_features] = np.nan_to_num(d[:, :n_features], nan=0.0)  # impute sparse feature NaNs
        rec = {k: r[k] for k in ("grid", "ssp", "y", "x", "ny", "nx", "lat", "lon")}
        rec["data"] = d.astype(np.float32)
        splits[split[(r["grid"], r["y"], r["x"])]].append(rec)
    logger.info("Imputed %d feature-NaN values to 0 (z-score mean) across all records", n_imputed)

    for name, recs in splits.items():
        logger.info("%s: %d pixel-records", name, len(recs))
        with (out_dir / f"{name}.pkl").open("wb") as f:
            pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)

    scaler_path = Path(cfg["paths"]["scaler"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved scaler (%d columns) to %s", ncol, scaler_path)


if __name__ == "__main__":
    main()
