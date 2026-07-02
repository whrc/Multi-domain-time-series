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

import argparse
import logging
import pickle
import re
import subprocess
import sys
import tempfile
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.io import gcs_filesystem, read_netcdf  # noqa: E402
from shared.plots import plot_data_split_map  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_TARGETS = 4
CLIM_VARS = ["tair", "precip", "nirr", "vapor_press"]
EXCLUDE = {"lat", "lon", "lambert_azimuthal_equal_area"}
GRID_NAME_RE = re.compile(r"^H\d+_V\d+$")  # excludes non-grid entries (e.g. bucket-root markers)


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
        ds = _to_y_x(read_netcdf(fs, inp(fname), prefer_engine="scipy"))
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
        ds = read_netcdf(fs, inp(f), prefer_engine="scipy")
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


def fetch_grid_records(cfg: dict, fs, grid: str, idx_map: dict, proj_start: int) -> list[dict]:
    """Fetch and assemble one grid's per-pixel raw (un-normalised) records, both scenarios.

    Scoped to a single grid so callers can fetch grids concurrently (I/O-bound) and process
    each grid's result immediately without holding every grid's data in memory at once.
    """
    bucket = cfg["gcs"]["bucket"].replace("gs://", "")
    sub = cfg["gcs"]["target_subfolder"]
    records = []
    for scenario in cfg["scenarios"]:
        is_ssp1 = "ssp1" in scenario
        monthly_index = idx_map["ssp1" if is_ssp1 else "ssp5"]

        def inp(f: str, _s=scenario) -> str:
            return f"{bucket}/{grid}/{_s}/{f}"

        def tgt(f: str, _s=scenario) -> str:
            return f"{bucket}/{grid}/{_s}_split/{sub}/{f}"

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


_WORKER = Path(__file__).parent / "_fetch_grid_worker.py"


def fetch_grid_records_isolated(grid: str, timeout: int = 180, retries: int = 2) -> list[dict]:
    """Fetch one grid's records in a fresh subprocess, retrying on failure.

    A shared gcsfs filesystem — even a freshly-constructed instance — was found to accumulate
    bad internal state after repeated use within a single process, eventually hanging
    indefinitely on a later grid's fetch. A new OS process per grid sidesteps this (see
    _fetch_grid_worker.py). If the worker hangs on shutdown *after* writing its result (also
    observed), the output file still exists and is used rather than discarded as a failure.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 2):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        tmp_path.unlink()
        try:
            subprocess.run(
                [sys.executable, str(_WORKER), "--grid", grid, "--out", str(tmp_path)],
                timeout=timeout, check=True, capture_output=True, text=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as err:
            if not tmp_path.exists():
                last_err = err
                logger.warning(
                    "fetch_grid_records_isolated(%s) attempt %d/%d failed with no output: %s",
                    grid, attempt, retries + 1, err,
                )
                continue
            logger.warning(
                "fetch_grid_records_isolated(%s): worker didn't exit cleanly (%s) but wrote "
                "output — using it", grid, err,
            )
        try:
            with tmp_path.open("rb") as f:
                return pickle.load(f)
        finally:
            tmp_path.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to fetch grid {grid!r} after {retries + 1} attempts") from last_err


def _grid_split_labels(grid: str, keys: list[tuple[int, int]], pp: dict) -> dict[tuple, str]:
    """Deterministic train/val/test assignment for one grid's (y, x) pixel keys.

    Grid-local (depends only on this grid's own pixel count), so it's safe to compute for
    grids in any order — including concurrently — while still being fully reproducible: the
    per-grid seed is derived from the global seed + a stable hash of the grid name.
    """
    rng = np.random.default_rng([pp["random_seed"], zlib.crc32(grid.encode())])
    ordered_keys = sorted(keys)
    rng.shuffle(ordered_keys)
    n = len(ordered_keys)
    n_train = round(pp["train_frac"] * n)
    n_val = round(pp["val_frac"] * n)
    return {
        (grid, y, x): ("train" if i < n_train else "val" if i < n_train + n_val else "test")
        for i, (y, x) in enumerate(ordered_keys)
    }


def subsample_pixels_round_robin(
    pixel_windows: dict[tuple, int],
    split_name: str,
    size: int,
    seed: int,
) -> set[tuple]:
    """Select pixels via round-robin across grids until the window target is met.

    Cycling through grids alphabetically (1 pixel per grid per pass) ensures geographic
    spread even at small target sizes. Subsampling is by pixel (not individual windows)
    to preserve temporal autocorrelation within each pixel's time series. ``pixel_windows``
    must already be scoped to the desired split (grid, y, x) -> window count.
    """
    # Group by grid and shuffle within each grid (sorted for determinism regardless of
    # pixel_windows' insertion order, which may vary run-to-run under concurrent fetching)
    by_grid: dict[str, list[tuple]] = defaultdict(list)
    for k in pixel_windows:
        by_grid[k[0]].append(k)

    rng = np.random.default_rng(seed)
    for grid in sorted(by_grid):
        rng.shuffle(by_grid[grid])

    # Round-robin across grids
    grid_list = sorted(by_grid.keys())
    grid_indices = {g: 0 for g in grid_list}

    selected: set[tuple] = set()
    cumulative = 0

    while cumulative < size:
        any_added = False
        for grid in grid_list:
            if cumulative >= size:
                break
            idx = grid_indices[grid]
            if idx < len(by_grid[grid]):
                k = by_grid[grid][idx]
                grid_indices[grid] += 1
                selected.add(k)
                cumulative += pixel_windows[k]
                any_added = True
        if not any_added:
            break  # all grids exhausted before reaching target

    logger.info(
        "%s size=%d: selected %d/%d pixels (~%d windows) across %d grids",
        split_name, size, len(selected), len(pixel_windows), cumulative,
        len({k[0] for k in selected}),
    )
    return selected


def _window_label(n: int) -> str:
    """Convert window count to short label: 50000 → '50K', 2000000 → '2M'."""
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0:
        return f"{n // 1_000}K"
    return str(n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=None,
                        help="Override preprocessing.train_size from config")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Delete cached val.pkl and test.pkl before preprocessing "
                             "(required when switching from dev to production mode).")
    args = parser.parse_args()

    cfg = load_config("arctic_domain")
    pp = cfg["preprocessing"]
    train_size = args.train_size if args.train_size is not None else pp.get("train_size")

    out_dir = Path(cfg["paths"]["preprocessed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.force_recompute:
        for fname in ("val.pkl", "test.pkl"):
            p = out_dir / fname
            if p.exists():
                p.unlink()
                logger.info("force-recompute: deleted %s", p)

    fs = gcs_filesystem()
    grids = pp.get("grids")
    if not grids:
        bucket = cfg["gcs"]["bucket"].replace("gs://", "")
        grids = sorted(
            p.split("/")[-1] for p in fs.ls(bucket)
            if fs.isdir(p) and GRID_NAME_RE.match(p.split("/")[-1])
        )
    logger.info("Grids: %s", grids)

    idx_map = monthly_index_map(cfg)
    proj_start = cfg["time"]["projected_start_year"]

    val_size  = pp.get("val_size")
    test_size = pp.get("test_size")
    val_cached  = (out_dir / "val.pkl").exists()
    test_cached = (out_dir / "test.pkl").exists()
    if val_cached:
        logger.info("val.pkl already exists — skipping (cached)")
    if test_cached:
        logger.info("test.pkl already exists — skipping (cached)")

    # ---------- Pass 1: fetch each grid (isolated subprocess — see fetch_grid_records_isolated),
    # derive split labels + scaler stats + window counts, then discard the grid's heavy data.
    # Peak memory is bounded to one grid at a time, not the whole circumpolar set.
    #
    # Sequential grid order also makes early-stopping below deterministic: round-robin
    # subsampling always walks grids in the same sorted order, so once enough *visited* grids
    # already cover train_size/val_size/test_size windows, every later grid is guaranteed
    # unnecessary and we can stop without reading the rest of the circumpolar bucket. This only
    # kicks in when a size cap is set (learning-curve runs); an uncapped run (train_size None, or
    # val/test uncapped) still scans every grid, matching prior behaviour. ----------
    split: dict[tuple, str] = {}
    pixel_windows: dict[tuple, int] = {}
    pixel_meta: dict[tuple, dict] = {}
    ncol: int | None = None
    scaler_sum = scaler_sumsq = scaler_count = None
    cum_windows = {"train": 0, "val": 0, "test": 0}
    visited_grids: list[str] = []

    def accumulate_grid(grid: str, recs: list[dict]) -> None:
        nonlocal ncol, scaler_sum, scaler_sumsq, scaler_count
        if not recs:
            return
        if ncol is None:
            ncol = recs[0]["data"].shape[1]
            scaler_sum = np.zeros(ncol)
            scaler_sumsq = np.zeros(ncol)
            scaler_count = np.zeros(ncol)
        keys = list({(r["y"], r["x"]) for r in recs})
        split.update(_grid_split_labels(grid, keys, pp))
        for r in recs:
            k = (r["grid"], r["y"], r["x"])
            T = r["data"].shape[0]
            w = max(0, (T - pp["seq_len"]) // pp["stride"] + 1)
            pixel_windows[k] = pixel_windows.get(k, 0) + w
            if k not in pixel_meta:
                pixel_meta[k] = {"lat": r["lat"], "lon": r["lon"]}
            cum_windows[split[k]] += w  # exact delta, whichever scenario this record is
            if split[k] == "train":
                d = r["data"].astype(float)
                scaler_sum += np.nansum(d, axis=0)
                scaler_sumsq += np.nansum(d * d, axis=0)
                scaler_count += (~np.isnan(d)).sum(axis=0)

    def targets_met() -> bool:
        train_enough = (not train_size) or cum_windows["train"] >= train_size
        val_enough   = val_cached or (not val_size) or cum_windows["val"] >= val_size
        test_enough  = test_cached or (not test_size) or cum_windows["test"] >= test_size
        return train_enough and val_enough and test_enough

    for i, grid in enumerate(grids, start=1):
        recs = fetch_grid_records_isolated(grid)
        visited_grids.append(grid)
        accumulate_grid(grid, recs)
        logger.info("[pass 1/2] %d/%d grids fetched (%s: %d land pixels)", i, len(grids), grid, len(recs) // 2)
        if targets_met():
            logger.info(
                "[pass 1/2] Early stop after %d/%d grids — enough windows for all requested splits",
                i, len(grids),
            )
            break

    n_features = ncol - NUM_TARGETS
    mean = scaler_sum / scaler_count
    std = np.sqrt(np.clip(scaler_sumsq / scaler_count - mean ** 2, 0, None))
    mean[~np.isfinite(mean)] = 0.0
    std[(std == 0) | ~np.isfinite(std)] = 1.0
    scaler = {"mean": mean, "std": std}

    total_train = sum(s == "train" for s in split.values())
    total_val = sum(s == "val" for s in split.values())
    total_test = sum(s == "test" for s in split.values())
    logger.info(
        "Grid-stratified pixel split: train=%d val=%d test=%d across %d/%d grids visited",
        total_train, total_val, total_test, len(visited_grids), len(grids),
    )

    # Subsample train pixels (varies per learning curve run)
    train_subset: set[tuple] | None = None
    if train_size:
        train_windows = {k: w for k, w in pixel_windows.items() if split[k] == "train"}
        train_subset = subsample_pixels_round_robin(train_windows, "train", train_size, pp["random_seed"])

    val_subset: set[tuple] | None = None
    if not val_cached and val_size:
        val_windows = {k: w for k, w in pixel_windows.items() if split[k] == "val"}
        val_subset = subsample_pixels_round_robin(val_windows, "val", val_size, pp["random_seed"])

    test_subset: set[tuple] | None = None
    if not test_cached and test_size:
        test_windows = {k: w for k, w in pixel_windows.items() if split[k] == "test"}
        test_subset = subsample_pixels_round_robin(test_windows, "test", test_size, pp["random_seed"])

    # Which pixels does the final output actually need? (mirrors old per-record skip logic,
    # but decided up front so pass 2 only re-fetches grids that contain a wanted pixel)
    wanted: set[tuple] = set()
    for k, s in split.items():
        if s == "train" and (train_subset is None or k in train_subset):
            wanted.add(k)
        elif s == "val" and not val_cached and (val_subset is None or k in val_subset):
            wanted.add(k)
        elif s == "test" and not test_cached and (test_subset is None or k in test_subset):
            wanted.add(k)
    wanted_grids = sorted({k[0] for k in wanted})
    logger.info("[pass 2/2] Re-fetching %d/%d grids containing wanted pixels", len(wanted_grids), len(grids))

    # ---------- Pass 2: re-fetch only grids with wanted pixels, filter + normalise + save.
    # Data is gridded (not pixel-addressable), so grids containing any wanted pixel must be
    # re-read in full — but the output buffer here is bounded by train/val/test size, not the
    # full circumpolar dataset. ----------
    bucket_splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    n_imputed = 0
    for grid in wanted_grids:
        for r in fetch_grid_records_isolated(grid):
            k = (r["grid"], r["y"], r["x"])
            if k not in wanted:
                continue
            s = split[k]
            d = (r["data"] - scaler["mean"]) / scaler["std"]
            n_imputed += int(np.isnan(d[:, :n_features]).sum())
            d[:, :n_features] = np.nan_to_num(d[:, :n_features], nan=0.0)
            rec = {key: r[key] for key in ("grid", "ssp", "y", "x", "ny", "nx", "lat", "lon")}
            rec["data"] = d.astype(np.float32)
            bucket_splits[s].append(rec)
    logger.info("Imputed %d feature-NaN values to 0 (z-score mean) across processed records", n_imputed)

    # Save train (always regenerated), val/test (only if not cached)
    for name, recs in bucket_splits.items():
        if name == "val" and val_cached:
            continue
        if name == "test" and test_cached:
            continue
        logger.info("%s: %d pixel-records", name, len(recs))
        with (out_dir / f"{name}.pkl").open("wb") as f:
            pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)

    scaler_path = Path(cfg["paths"]["scaler"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved scaler (%d columns) to %s", ncol, scaler_path)

    # Save split map for this training size (shows geographic coverage of each experiment)
    if train_size:
        eval_dir = Path(cfg["paths"]["evaluation"])
        eval_dir.mkdir(parents=True, exist_ok=True)
        label = _window_label(train_size)
        light_records = [
            {"grid": k[0], "y": k[1], "x": k[2], "lat": m["lat"], "lon": m["lon"]}
            for k, m in pixel_meta.items()
        ]
        plot_data_split_map(
            light_records, split, train_subset,
            title=f"Arctic split — train {label}",
            save_path=eval_dir / f"arctic_data_map_train_{label}.png",
        )
        logger.info("Saved split map: arctic_data_map_train_%s.png", label)


if __name__ == "__main__":
    main()
