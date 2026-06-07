# domains/arctic_domain/01_preprocess.py

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config

DOMAIN = os.environ.get("ARCTIC_DOMAIN", "arctic_domain")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUM_TARGETS = 4  # ALD, GPP, RECO, VEGC (always last 4 columns in data array)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _open_nc(fs: gcsfs.GCSFileSystem, path: str) -> xr.Dataset:
    # Detect NetCDF format from magic bytes; CDF\x01/\x02 = NetCDF3 (scipy),
    # \x89HDF = NetCDF4/HDF5 (h5netcdf).
    with fs.open(path, "rb") as f:
        magic = f.read(4)
    engine = "scipy" if magic[:3] == b"CDF" else "h5netcdf"
    return xr.open_dataset(fs.open(path), engine=engine, mask_and_scale=True)


def _monthly_index(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """Return a monthly DatetimeIndex from Jan start_year to Dec end_year (inclusive)."""
    return pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_static(grid_path: str, ssp_folder: str, cfg: dict, fs: gcsfs.GCSFileSystem) -> xr.Dataset:
    """Load static input files for one grid/scenario; rename Y→y, X→x."""
    datasets = []
    for fname in cfg["inputs"]["static"]:
        path = f"{grid_path}/{ssp_folder}/{fname}"
        ds = _open_nc(fs, path)
        # Normalise coordinate names: static files use uppercase Y/X
        rename = {c: c.lower() for c in ds.coords if c in ("Y", "X")}
        if rename:
            ds = ds.rename(rename)
        datasets.append(ds)
    return xr.merge(datasets, compat="override")


def load_co2(
    grid_path: str, ssp_folder: str, cfg: dict, fs: gcsfs.GCSFileSystem, monthly_index: pd.DatetimeIndex
) -> pd.Series:
    """Load historic + projected CO2, return a Series indexed by monthly_index."""
    hist_path = f"{grid_path}/{ssp_folder}/co2.nc"
    proj_path = f"{grid_path}/{ssp_folder}/projected-co2.nc"

    def _read_co2(path: str) -> pd.Series:
        ds = _open_nc(fs, path)
        # CO2 files use a 'year' dimension instead of 'time'
        dim = "year" if "year" in ds.dims else list(ds.dims)[0]
        var = [v for v in ds.data_vars][0]
        years = ds[dim].values.astype(int)
        values = ds[var].values.ravel()
        return pd.Series(values, index=years)

    hist = _read_co2(hist_path)
    proj = _read_co2(proj_path)
    co2_by_year = pd.concat([hist, proj]).sort_index()

    # Expand: repeat each year's value for 12 months, then reindex to monthly_index
    monthly_years = pd.Series(monthly_index.year, index=monthly_index)
    co2_monthly = monthly_years.map(co2_by_year)
    if co2_monthly.isna().any():
        missing = monthly_years[co2_monthly.isna()].unique()
        log.warning("CO2 values missing for years: %s", missing)
    return co2_monthly


def load_climate(
    grid_path: str, ssp_folder: str, cfg: dict, fs: gcsfs.GCSFileSystem
) -> xr.Dataset:
    """Concatenate historic and projected climate along time."""
    hist = _open_nc(fs, f"{grid_path}/{ssp_folder}/historic-climate.nc")
    proj = _open_nc(fs, f"{grid_path}/{ssp_folder}/projected-climate.nc")
    return xr.concat([hist, proj], dim="time", data_vars="all")


def load_fire(
    grid_path: str, ssp_folder: str, cfg: dict, fs: gcsfs.GCSFileSystem, monthly_index: pd.DatetimeIndex
) -> xr.Dataset:
    """Concatenate historic + projected fire (yearly) and forward-fill to monthly."""
    hist = _open_nc(fs, f"{grid_path}/{ssp_folder}/historic-explicit-fire.nc")
    proj = _open_nc(fs, f"{grid_path}/{ssp_folder}/projected-explicit-fire.nc")
    fire_yearly = xr.concat([hist, proj], dim="time", data_vars="all")

    # Use cftime-safe conversion: extract year from each time step
    times = fire_yearly.indexes["time"]
    try:
        years = pd.DatetimeIndex(times).year
    except Exception:
        # cftime objects
        years = np.array([t.year for t in times])

    # Build a DatetimeIndex at yearly frequency (Jan 1 of each year)
    yearly_index = pd.DatetimeIndex([f"{y}-01-01" for y in years])
    fire_yearly = fire_yearly.assign_coords(time=yearly_index)

    # Reindex to monthly and forward-fill
    fire_monthly = fire_yearly.reindex(time=monthly_index, method="ffill")
    return fire_monthly


def _to_yearly_jan_index(ds: xr.Dataset) -> xr.Dataset:
    """Replace time coord with Jan-1 DatetimeIndex derived from the existing years."""
    times = ds.indexes["time"]
    try:
        years = pd.DatetimeIndex(times).year
    except Exception:
        years = np.array([t.year for t in times])
    return ds.assign_coords(time=pd.DatetimeIndex([f"{y}-01-01" for y in years]))


def _fix_projected_yearly(ds: xr.Dataset, name: str) -> xr.Dataset:
    """Override wrong time labels on projected yearly targets (e.g. 1901–1976 → 2025–2100)."""
    times = ds.indexes["time"]
    try:
        first_year = pd.DatetimeIndex(times).year[0]
    except Exception:
        first_year = times[0].year
    if first_year < 2000:
        log.warning(
            "Target %s projected time labels look wrong (start=%d); overriding to 2025–2100.",
            name, first_year,
        )
        correct = pd.date_range("2025-01-01", periods=len(times), freq="YS")
        ds = ds.assign_coords(time=correct)
    return ds


def load_targets(
    grid_path: str,
    split_folder: str,
    cfg: dict,
    fs: gcsfs.GCSFileSystem,
    monthly_index: pd.DatetimeIndex,
    include_historical: bool,
) -> xr.Dataset:
    """
    Load target variables for one grid/scenario, aligned to monthly_index.
    For SSP5-8.5 include_historical=False (no _tr files exist).
    Yearly targets are forward-filled to monthly.
    """
    target_dir = f"{grid_path}/{split_folder}/all_merged"
    datasets = []

    for tgt in cfg["targets"]:
        name = tgt["name"]
        resolution = tgt["resolution"]

        # Load projected; always fix yearly time labels before any concat
        ds_proj = _open_nc(fs, f"{target_dir}/{tgt['projected']}")
        if resolution == "yearly":
            ds_proj = _fix_projected_yearly(ds_proj, name)
            ds_proj = _to_yearly_jan_index(ds_proj)

        if include_historical:
            ds_hist = _open_nc(fs, f"{target_dir}/{tgt['historical']}")
            if resolution == "yearly":
                ds_hist = _to_yearly_jan_index(ds_hist)
            ds_combined = xr.concat([ds_hist, ds_proj], dim="time", data_vars="minimal")
        else:
            ds_combined = ds_proj

        if resolution == "yearly":
            ds_combined = ds_combined.reindex(time=monthly_index, method="ffill")
        else:
            # Monthly targets: convert cftime → standard DatetimeIndex, then align
            times = ds_combined.indexes["time"]
            try:
                std_times = pd.DatetimeIndex(times)
            except Exception:
                std_times = pd.DatetimeIndex([t.strftime("%Y-%m-%d") for t in times])
            ds_combined = ds_combined.assign_coords(time=std_times)
            ds_combined = ds_combined.reindex(time=monthly_index)

        # Rename variable to target name if not already
        if name not in ds_combined.data_vars and len(ds_combined.data_vars) == 1:
            old_name = list(ds_combined.data_vars)[0]
            ds_combined = ds_combined.rename({old_name: name})

        datasets.append(ds_combined[[name]])

    return xr.merge(datasets, compat="override")


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_grid_sequences(grid: str, cfg: dict, fs: gcsfs.GCSFileSystem) -> list[dict[str, Any]]:
    """
    Build per-pixel sequences for one grid across both SSP scenarios.
    Returns a list of dicts: {grid, ssp, y, x, data: np.ndarray(T, F+NUM_TARGETS)}.
    """
    bucket = cfg["gcs"]["bucket"]
    grid_path = f"{bucket}/{grid}"
    records: list[dict[str, Any]] = []

    ssp1 = cfg["scenarios"][0]  # ssp1_2_6_mri_esm2_0
    ssp5 = cfg["scenarios"][1]  # ssp5_8_5_mri_esm2_0

    scenario_cfg = [
        (ssp1, True,  _monthly_index(1901, 2100)),
        (ssp5, False, _monthly_index(2025, 2100)),
    ]

    for ssp, include_historical, monthly_index in scenario_cfg:
        log.info("Grid=%s  SSP=%s  months=%d", grid, ssp, len(monthly_index))
        split_folder = f"{ssp}_split"

        # ── Load inputs ──────────────────────────────────────────────────────
        static_ds = load_static(grid_path, ssp, cfg, fs)
        co2_series = load_co2(grid_path, ssp, cfg, fs, monthly_index)
        climate_ds = load_climate(grid_path, ssp, cfg, fs)
        fire_ds = load_fire(grid_path, ssp, cfg, fs, monthly_index)

        # Align climate time to standard DatetimeIndex
        times = climate_ds.indexes["time"]
        try:
            std_times = pd.DatetimeIndex(times)
        except Exception:
            std_times = pd.DatetimeIndex([t.strftime("%Y-%m-%d") for t in times])
        climate_ds = climate_ds.assign_coords(time=std_times)
        climate_ds = climate_ds.reindex(time=monthly_index)

        # ── Load targets ─────────────────────────────────────────────────────
        if not include_historical:
            log.warning(
                "SSP5-8.5 has no historical targets — using projected period only for %s.", grid
            )
        target_ds = load_targets(
            grid_path, split_folder, cfg, fs, monthly_index, include_historical
        )

        # ── Derive pixel grid from static dataset ────────────────────────────
        y_coords = static_ds.coords["y"].values
        x_coords = static_ds.coords["x"].values
        T = len(monthly_index)

        # CO2 column: shape (T,)
        co2_arr = co2_series.values.reshape(T, 1)  # (T, 1)

        # Climate array: (T, y, x, num_climate_vars)
        climate_vars = [v for v in climate_ds.data_vars]
        climate_arr = np.stack([climate_ds[v].values for v in climate_vars], axis=-1)  # (T, y, x, nC)

        # Fire array: (T, y, x, num_fire_vars)
        fire_vars = [v for v in fire_ds.data_vars]
        fire_arr = np.stack([fire_ds[v].values for v in fire_vars], axis=-1)  # (T, y, x, nF)

        # Target array: (T, y, x, 4)
        target_names = [t["name"] for t in cfg["targets"]]
        target_arr = np.stack([target_ds[n].values for n in target_names], axis=-1)  # (T, y, x, 4)

        # Static array: (y, x, num_static_vars) — keep only 2-D spatial vars, skip lat/lon/CRS scalars
        static_vars = [
            v for v in static_ds.data_vars
            if tuple(static_ds[v].dims) == ("y", "x") and v not in ("lat", "lon")
        ]
        static_arr = np.stack([static_ds[v].values for v in static_vars], axis=-1)  # (y, x, nS)

        log.info(
            "  climate=%s fire=%s static=%s co2=1 targets=4",
            len(climate_vars), len(fire_vars), len(static_vars),
        )

        # ── Iterate over pixels ───────────────────────────────────────────────
        ny, nx = len(y_coords), len(x_coords)
        dropped = 0
        added = 0
        for yi in range(ny):
            for xi in range(nx):
                tgt_pixel = target_arr[:, yi, xi, :]  # (T, 4)
                # Drop ocean pixels: all NaN across all target variables
                if np.all(np.isnan(tgt_pixel)):
                    dropped += 1
                    continue

                # Static features repeated across time: (T, nS)
                static_pixel = np.tile(static_arr[yi, xi, :], (T, 1))

                # Climate: (T, nC)
                climate_pixel = climate_arr[:, yi, xi, :]

                # Fire: (T, nF)
                fire_pixel = fire_arr[:, yi, xi, :]

                # Full feature array: (T, nS + 1 + nC + nF)
                features = np.concatenate(
                    [static_pixel, co2_arr, climate_pixel, fire_pixel], axis=1
                )

                # Concatenate features + targets: (T, F + 4)
                data = np.concatenate([features, tgt_pixel], axis=1).astype(np.float32)

                records.append({
                    "grid": grid,
                    "ssp": ssp,
                    "y": int(yi),
                    "x": int(xi),
                    "data": data,
                })
                added += 1

        log.info("  valid pixels=%d  dropped (ocean)=%d", added, dropped)

    return records


# ── Split ─────────────────────────────────────────────────────────────────────

def split_pixels(
    records: list[dict], cfg: dict
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Group records by (grid, y, x); randomly split unique pixels 70/15/15.
    Both SSP sequences for a pixel land in the same split.
    """
    rng = np.random.default_rng(cfg["preprocessing"]["random_seed"])
    pixel_keys = sorted({(r["grid"], r["y"], r["x"]) for r in records})
    rng.shuffle(pixel_keys)  # in-place

    n = len(pixel_keys)
    n_train = int(n * cfg["preprocessing"]["train_frac"])
    n_val   = int(n * cfg["preprocessing"]["val_frac"])

    train_keys = set(pixel_keys[:n_train])
    val_keys   = set(pixel_keys[n_train : n_train + n_val])
    test_keys  = set(pixel_keys[n_train + n_val :])

    train = [r for r in records if (r["grid"], r["y"], r["x"]) in train_keys]
    val   = [r for r in records if (r["grid"], r["y"], r["x"]) in val_keys]
    test  = [r for r in records if (r["grid"], r["y"], r["x"]) in test_keys]

    log.info(
        "Pixels — total=%d  train=%d  val=%d  test=%d",
        n, len(train_keys), len(val_keys), len(test_keys),
    )
    return train, val, test


# ── Normalisation ─────────────────────────────────────────────────────────────

def fit_scaler(train_records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Compute column-wise mean and std from training sequences only."""
    all_data = np.concatenate([r["data"] for r in train_records], axis=0)
    mean = np.nanmean(all_data, axis=0)
    std  = np.nanstd(all_data, axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero for constant columns
    return mean, std


def apply_scaler(
    records: list[dict], mean: np.ndarray, std: np.ndarray
) -> list[dict]:
    """Return new records with data normalised."""
    return [{**r, "data": ((r["data"] - mean) / std).astype(np.float32)} for r in records]


# ── Persistence ───────────────────────────────────────────────────────────────

def save_split(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Saved %d records → %s", len(records), path)


def load_split(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ArcticDataset(Dataset):
    """
    Sliding-window dataset over normalised per-pixel sequences.

    Each item returns:
        input_tensor  — (seq_len, num_features)
        target_tensor — (seq_len, NUM_TARGETS)
    """

    def __init__(self, records: list[dict], cfg: dict, num_features: int) -> None:
        seq_len: int = cfg["preprocessing"]["seq_len"]
        stride:  int = cfg["preprocessing"]["stride"]

        self.records      = records
        self.seq_len      = seq_len
        self.num_features = num_features

        # Build flat index: (record_idx, window_start)
        self.windows: list[tuple[int, int]] = []
        for i, r in enumerate(records):
            T = r["data"].shape[0]
            starts = range(0, T - seq_len + 1, stride)
            self.windows.extend((i, s) for s in starts)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec_idx, start = self.windows[idx]
        window = self.records[rec_idx]["data"][start : start + self.seq_len]  # (seq_len, F+4)
        x = torch.from_numpy(window[:, : self.num_features])
        y = torch.from_numpy(window[:, self.num_features :])
        return x, y


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config(DOMAIN)
    fs  = gcsfs.GCSFileSystem(access="read_only")

    bucket = cfg["gcs"]["bucket"]

    # Discover grids: use config subset if present, else list bucket
    if "grids" in cfg.get("preprocessing", {}):
        grids = cfg["preprocessing"]["grids"]
        log.info("Using config-specified grids: %s", grids)
    else:
        grids = [p.split("/")[-1] for p in fs.ls(bucket) if fs.isdir(p)]
        log.info("Discovered %d grids in bucket", len(grids))

    all_records: list[dict] = []
    for grid in grids:
        log.info("=== Processing grid: %s ===", grid)
        records = build_grid_sequences(grid, cfg, fs)
        all_records.extend(records)

    log.info("Total records (pixel × SSP): %d", len(all_records))

    train_records, val_records, test_records = split_pixels(all_records, cfg)

    mean, std = fit_scaler(train_records)

    train_norm = apply_scaler(train_records, mean, std)
    val_norm   = apply_scaler(val_records,   mean, std)
    test_norm  = apply_scaler(test_records,  mean, std)

    out_dir = Path(cfg["paths"]["preprocessed_dir"])
    save_split(train_norm, out_dir / "train.pkl") # save normalised train split
    save_split(val_norm,   out_dir / "val.pkl") 
    save_split(test_norm,  out_dir / "test.pkl")

    scaler_path = Path(cfg["paths"]["scaler"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Saved scaler → %s", scaler_path)

    # Quick sanity: report window counts
    num_features = train_norm[0]["data"].shape[1] - NUM_TARGETS
    for split_name, split_recs in [("train", train_norm), ("val", val_norm), ("test", test_norm)]:
        ds = ArcticDataset(split_recs, cfg, num_features)
        log.info("ArcticDataset %s: %d windows from %d records", split_name, len(ds), len(split_recs))

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
