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
import concurrent.futures
import logging
import pickle
import re
import subprocess
import sys
import tempfile
import time
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
from domains.arctic_domain._naming import (  # noqa: E402
    load_sidecar, sidecar_matches, train_pkl_name, window_label, write_sidecar,
)

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
_FAILED_MARKER_TTL_SECONDS = 3600  # retry a previously-failed grid after an hour, in case
                                    # the earlier failure was a transient blip rather than a
                                    # genuinely broken grid — otherwise a single bad-timing
                                    # failure during pass 2 would permanently exclude a grid
                                    # from every future run, even across many hours of the
                                    # resilient wrapper script's retries.


def fetch_grid_records_isolated(grid: str, cache_dir: Path, timeout: int = 180, retries: int = 2) -> list[dict]:
    """Fetch one grid's records in a fresh subprocess, retrying on failure.

    A shared gcsfs filesystem — even a freshly-constructed instance — was found to accumulate
    bad internal state after repeated use within a single process, eventually hanging
    indefinitely on a later grid's fetch. A new OS process per grid sidesteps this (see
    _fetch_grid_worker.py). If the worker hangs on shutdown *after* writing its result (also
    observed), the output file still exists and is used rather than discarded as a failure.

    Does NOT cache the raw records themselves — some grids carry ~10,000 land pixels'
    worth of full time series (multi-GB each), so caching every fetched grid raw quickly
    exhausted local disk (a 187-grid run once produced a 258GB cache from a handful of
    hours). Only cache_dir/{grid}.failed is kept (a marker for a grid that exhausted
    retries, expiring after _FAILED_MARKER_TTL_SECONDS so a later run doesn't burn the same
    ~9 minutes of retries again on a still-genuinely-broken grid, without permanently
    excluding one that recovers) — see _cached_by_grid() for the actual resumability cache.
    """
    failed_marker = cache_dir / f"{grid}.failed"
    if failed_marker.exists() and time.time() - failed_marker.stat().st_mtime < _FAILED_MARKER_TTL_SECONDS:
        raise RuntimeError(f"Grid {grid!r} previously exhausted retries (cached failure at {failed_marker})")

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
                recs = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as err:
            # A kill mid-write (this repo's tool environment has been observed to kill
            # background processes unpredictably) can leave a truncated output file. Treat
            # it the same as any other failed attempt rather than letting it crash the run.
            last_err = err
            logger.warning(
                "fetch_grid_records_isolated(%s) attempt %d/%d produced a corrupt output "
                "file (likely killed mid-write): %s", grid, attempt, retries + 1, err,
            )
            continue
        finally:
            tmp_path.unlink(missing_ok=True)
        return recs

    cache_dir.mkdir(parents=True, exist_ok=True)
    failed_marker.touch()
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


def _log_giveup(grid: str) -> None:
    logger.error("Giving up on grid %s after exhausting retries — skipping it "
                 "(that grid's coverage is lost, the rest of the run continues)",
                 grid, exc_info=True)


def _run_concurrent(work_fn, grids: list[str], max_workers: int):
    """Yield (grid, work_fn(grid)) for each grid, running work_fn via isolated subprocess.

    max_workers<=1 fetches strictly sequentially (today's proven-safe behaviour — the
    zero-risk rollback). max_workers>1 runs several work_fn calls concurrently via a thread
    pool: each call already shells out to a fresh OS subprocess (see
    fetch_grid_records_isolated), so the orchestrator only needs threads to have several
    subprocess.run calls in flight at once — no gcsfs state is shared across them, since the
    confirmed hang bug is about *in-process* gcsfs reuse across multiple fetches, not about
    concurrent fetches from separate processes. Generic over work_fn so both pass 1
    (fetch + summarize + cache) and pass 2 (plain fetch) share this orchestration.
    """
    if max_workers <= 1:
        for g in grids:
            try:
                yield g, work_fn(g)
            except RuntimeError:
                _log_giveup(g)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(work_fn, g): g for g in grids}
        for fut in concurrent.futures.as_completed(futures):
            grid = futures[fut]
            try:
                yield grid, fut.result()
            except RuntimeError:
                _log_giveup(grid)


def _cached_by_grid(cache_dir: Path, grid: str, key: dict, compute_fn):
    """Generic per-grid pickle cache, invalidated by `key`.

    Returns the cached value if cache_dir/{grid}.pkl exists and its stored key matches, else
    calls compute_fn() and caches the result under the current key. Using a key (rather than
    trusting any existing file forever) means a run with different parameters — a different
    train_size, seed, or split fractions — naturally recomputes instead of silently reusing
    a stale, mismatched cache entry; nothing needs to remember to clear this cache by hand.

    Writes are atomic (temp file + rename) so a process killed mid-write can never leave a
    corrupt file at the final path — the reader only ever sees the previous good entry or
    nothing, never a truncated one. A corrupt/unreadable existing file is treated as a miss.
    """
    path = cache_dir / f"{grid}.pkl"
    if path.exists():
        try:
            with path.open("rb") as f:
                cached = pickle.load(f)
            if cached["key"] == key:
                return cached["value"]
        except (EOFError, pickle.UnpicklingError, KeyError):
            pass  # corrupt or malformed entry -- fall through and recompute
    value = compute_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump({"key": key, "value": value}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)
    return value


def _grid_summary_from_records(recs: list[dict], pp: dict, grid: str) -> dict:
    """Reduce one grid's raw fetched records to the small summary pass 1 actually needs.

    Raw records hold a full (T, ncol) time series per land pixel — multi-GB for grids with
    thousands of land pixels. Pass 1 only ever needs, per pixel: which scenarios it
    appeared in (to recompute window counts for any stride later), lat/lon, and this grid's
    contribution to the train-split scaler sums (a handful of ncol-length arrays) — all of
    which is a few hundred KB at most regardless of grid size, safe to cache indefinitely.
    """
    keys = list({(r["y"], r["x"]) for r in recs})
    split_for_grid = _grid_split_labels(grid, keys, pp)
    ncol = recs[0]["data"].shape[1]
    scaler_sum = np.zeros(ncol)
    scaler_sumsq = np.zeros(ncol)
    scaler_count = np.zeros(ncol)
    lat_lon: dict[tuple, tuple] = {}
    scenarios_present: dict[tuple, list[str]] = defaultdict(list)
    for r in recs:
        k = (r["y"], r["x"])
        lat_lon[k] = (r["lat"], r["lon"])
        scenarios_present[k].append(r["ssp"])
        if split_for_grid[(grid, r["y"], r["x"])] == "train":
            d = r["data"].astype(float)
            scaler_sum += np.nansum(d, axis=0)
            scaler_sumsq += np.nansum(d * d, axis=0)
            scaler_count += (~np.isnan(d)).sum(axis=0)
    return {
        "keys": keys,
        "lat_lon": lat_lon,
        "scenarios_present": dict(scenarios_present),
        "ncol": ncol,
        "scaler_sum": scaler_sum,
        "scaler_sumsq": scaler_sumsq,
        "scaler_count": scaler_count,
    }


def _verify_gcs_access(fs, bucket: str) -> None:
    """Fail fast with an actionable message if local GCS credentials aren't set up."""
    try:
        fs.ls(bucket)
    except Exception as err:
        raise RuntimeError(
            f"Could not list GCS bucket '{bucket}'. If running locally, set up Application "
            "Default Credentials first: `gcloud auth application-default login` "
            "(project spherical-berm-323321)."
        ) from err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=None,
                        help="Override preprocessing.train_size from config")
    parser.add_argument("--capped-stride", type=int, default=None,
                        help="Override preprocessing.capped_stride from config")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Override preprocessing.max_workers from config "
                             "(concurrent isolated-subprocess grid fetches)")
    parser.add_argument("--grids", type=str, default=None,
                        help="Comma-separated grid names, overriding auto-discovery "
                             "(for local/small-scale validation runs)")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Delete cached val.pkl and test.pkl before preprocessing "
                             "(required when switching from dev to production mode).")
    args = parser.parse_args()

    cfg = load_config("arctic_domain")
    pp = cfg["preprocessing"]
    train_size = args.train_size if args.train_size is not None else pp.get("train_size")
    capped_stride = args.capped_stride if args.capped_stride is not None else pp["capped_stride"]
    max_workers = args.max_workers if args.max_workers is not None else pp["max_workers"]

    out_dir = Path(cfg["paths"]["preprocessed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # cache_dir holds only tiny per-grid ".failed" markers (see fetch_grid_records_isolated).
    # summary_cache_dir holds the small per-grid pass-1 summary (see get_grid_summary below) —
    # this is what actually makes pass 1 resumable across restarts, at negligible disk cost
    # (a few hundred KB/grid, vs. multi-GB if the raw fetch itself were cached).
    cache_dir = out_dir / ".grid_cache"
    summary_cache_dir = out_dir / ".grid_summary_cache"

    fs = gcs_filesystem()
    bucket = cfg["gcs"]["bucket"].replace("gs://", "")
    _verify_gcs_access(fs, bucket)  # check credentials before doing anything destructive below

    if args.force_recompute:
        for fname in ("val.pkl", "test.pkl"):
            p = out_dir / fname
            if p.exists():
                p.unlink()
                logger.info("force-recompute: deleted %s", p)

    grids = args.grids.split(",") if args.grids else pp.get("grids")
    if not grids:
        grids = sorted(
            p.split("/")[-1] for p in fs.ls(bucket)
            if fs.isdir(p) and GRID_NAME_RE.match(p.split("/")[-1])
        )
    logger.info("Grids: %s", grids)
    grids_hash = zlib.crc32(",".join(sorted(grids)).encode())

    idx_map = monthly_index_map(cfg)
    proj_start = cfg["time"]["projected_start_year"]

    val_size  = pp.get("val_size")
    test_size = pp.get("test_size")
    # Any capped split (train/val/test) uses capped_stride so each pixel contributes far fewer
    # windows, forcing round-robin subsampling to draw from many more grids for the same window
    # budget — an uncapped split uses full mode-density stride (production: 1). See
    # arctic_description.md "Sizing strategy" for the representativeness arithmetic behind this.
    effective_stride = {
        "train": capped_stride if train_size else pp["stride"],
        "val":   capped_stride if val_size else pp["stride"],
        "test":  capped_stride if test_size else pp["stride"],
    }
    # grids_hash is part of the expected cache key so a val/test built from a smaller/different
    # grid set (e.g. a --grids debug run, or the bucket's grid list changing) is never mistaken
    # for a match — seed/stride/seq_len/size_target alone can't tell "50K from 263 grids" apart
    # from "50K from 1 grid", since both reach the same window target. train/val/test_frac are
    # included too since they change which pixels _grid_split_labels assigns to val vs test —
    # without them, changing the split fractions wouldn't invalidate a stale val/test pkl and
    # could leak pixels between splits.
    frac_key = {"train_frac": pp["train_frac"], "val_frac": pp["val_frac"], "test_frac": pp["test_frac"]}
    expected_val = {"seed": pp["random_seed"], "stride": effective_stride["val"], "seq_len": pp["seq_len"],
                     "size_target": val_size, "grids_hash": grids_hash, **frac_key}
    expected_test = {"seed": pp["random_seed"], "stride": effective_stride["test"], "seq_len": pp["seq_len"],
                      "size_target": test_size, "grids_hash": grids_hash, **frac_key}
    val_cached  = ((out_dir / "val.pkl").exists()
                   and sidecar_matches(load_sidecar(out_dir / "val.pkl"), expected_val))
    test_cached = ((out_dir / "test.pkl").exists()
                   and sidecar_matches(load_sidecar(out_dir / "test.pkl"), expected_test))
    if (out_dir / "val.pkl").exists() and not val_cached:
        logger.warning("val.pkl exists but its sidecar doesn't match the current config — regenerating")
    if (out_dir / "test.pkl").exists() and not test_cached:
        logger.warning("test.pkl exists but its sidecar doesn't match the current config — regenerating")
    if val_cached:
        logger.info("val.pkl already exists and matches config — skipping (cached)")
    if test_cached:
        logger.info("test.pkl already exists and matches config — skipping (cached)")

    # ---------- Pass 1: fetch each grid (isolated subprocess — see fetch_grid_records_isolated),
    # derive split labels + scaler stats + window counts, then discard the grid's heavy data.
    # Peak memory is bounded to one grid at a time, not the whole circumpolar set.
    #
    # Every grid is always visited (no early stop): round-robin subsampling needs pixels from
    # (nearly) every grid to be geographically representative at any of the locked-in sizes —
    # e.g. at capped_stride's density, a 50K-window target needs ~362 pixels, more than the
    # available grid count, so it structurally requires ~all grids regardless of target size.
    # An early-stop optimization here would silently sacrifice representativeness (and, before
    # this fix, also made the scaler's train pool vary by train_size — see git history). ----------
    split: dict[tuple, str] = {}
    pixel_windows: dict[tuple, int] = {}
    pixel_meta: dict[tuple, dict] = {}
    ncol: int | None = None
    scaler_sum = scaler_sumsq = scaler_count = None
    visited_grids: list[str] = []

    # Split assignment (and thus which pixels feed the cached scaler sums) depends only on
    # these three config values — keying the cache on them means changing random_seed or the
    # split fractions between runs naturally invalidates stale cached grids instead of
    # silently mixing two different splits into one run.
    summary_key = {"seed": pp["random_seed"], "train_frac": pp["train_frac"],
                   "val_frac": pp["val_frac"], "test_frac": pp["test_frac"]}

    def get_grid_summary(grid: str) -> dict:
        """Fetch a grid (or reuse its cached summary) and return the pass-1 summary.

        Cached at summary_cache_dir/{grid}.pkl via _cached_by_grid — small (a few hundred KB
        even for a huge grid) since it holds derived stats, not raw time series, so it's
        cheap to keep indefinitely and makes pass 1 resumable across restarts.
        """
        def compute():
            recs = fetch_grid_records_isolated(grid, cache_dir)
            return _grid_summary_from_records(recs, pp, grid) if recs else {
                "keys": [], "lat_lon": {}, "scenarios_present": {}, "ncol": None,
                "scaler_sum": None, "scaler_sumsq": None, "scaler_count": None,
            }
        return _cached_by_grid(summary_cache_dir, grid, summary_key, compute)

    def apply_grid_summary(grid: str, summary: dict) -> None:
        nonlocal ncol, scaler_sum, scaler_sumsq, scaler_count
        if not summary["keys"]:
            return
        if ncol is None:
            ncol = summary["ncol"]
            scaler_sum = np.zeros(ncol)
            scaler_sumsq = np.zeros(ncol)
            scaler_count = np.zeros(ncol)
        split.update(_grid_split_labels(grid, summary["keys"], pp))
        scaler_sum += summary["scaler_sum"]
        scaler_sumsq += summary["scaler_sumsq"]
        scaler_count += summary["scaler_count"]
        for y, x in summary["keys"]:
            k = (grid, y, x)
            stride = effective_stride[split[k]]
            w = sum(
                max(0, (len(idx_map["ssp1" if "ssp1" in ssp else "ssp5"]) - pp["seq_len"]) // stride + 1)
                for ssp in summary["scenarios_present"][(y, x)]
            )
            pixel_windows[k] = w
            if k not in pixel_meta:
                lat, lon = summary["lat_lon"][(y, x)]
                pixel_meta[k] = {"lat": lat, "lon": lon}

    for i, (grid, summary) in enumerate(_run_concurrent(get_grid_summary, grids, max_workers), start=1):
        visited_grids.append(grid)
        apply_grid_summary(grid, summary)
        logger.info("[pass 1/2] %d/%d grids processed (%s: %d land pixels)", i, len(grids), grid, len(summary["keys"]))

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

    # Precompute each grid's own wanted (y, x) keys, sorted for a stable cache key below —
    # this is exactly the quantity that changes when train_size/val_size/test_size or the
    # subsampling seed changes between runs, so keying the pass-2 cache on it (rather than on
    # the individual parameters that produce it) invalidates automatically for any such change.
    wanted_by_grid: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for k in wanted:
        wanted_by_grid[k[0]].append((k[1], k[2]))
    for g in wanted_by_grid:
        wanted_by_grid[g].sort()

    # ---------- Pass 2: re-fetch only grids with wanted pixels, filter + normalise + save.
    # Data is gridded (not pixel-addressable), so grids containing any wanted pixel must be
    # re-read in full — but the output buffer here is bounded by train/val/test size, not the
    # full circumpolar dataset.
    #
    # Cached per-grid at pass2_cache_dir/{grid}.pkl: only the already-filtered, normalised
    # records for THIS grid's wanted pixels (typically 1-3 pixels/grid at these target
    # sizes) — small (comparable to the final output itself), unlike pass 1's raw fetch,
    # so this is safe to cache and makes pass 2 resumable across restarts too: without it,
    # a crash partway through pass 2 (which has no other checkpointing) would force
    # re-fetching all wanted_grids from scratch every time. ----------
    pass2_cache_dir = out_dir / ".grid_pass2_cache"

    def get_grid_pass2_records(grid: str) -> tuple[list[dict], int]:
        def compute():
            processed = []
            n_imputed_here = 0
            for r in fetch_grid_records_isolated(grid, cache_dir):
                k = (r["grid"], r["y"], r["x"])
                if k not in wanted:
                    continue
                d = (r["data"] - scaler["mean"]) / scaler["std"]
                n_imputed_here += int(np.isnan(d[:, :n_features]).sum())
                d[:, :n_features] = np.nan_to_num(d[:, :n_features], nan=0.0)
                rec = {key: r[key] for key in ("grid", "ssp", "y", "x", "ny", "nx", "lat", "lon")}
                rec["data"] = d.astype(np.float32)
                processed.append(rec)
            return {"records": processed, "n_imputed": n_imputed_here}

        pass2_key = {"wanted_keys": tuple(wanted_by_grid[grid])}
        cached = _cached_by_grid(pass2_cache_dir, grid, pass2_key, compute)
        return cached["records"], cached["n_imputed"]

    bucket_splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    n_imputed = 0
    for grid, (processed, n_imputed_here) in _run_concurrent(get_grid_pass2_records, wanted_grids, max_workers):
        n_imputed += n_imputed_here
        for rec in processed:
            k = (rec["grid"], rec["y"], rec["x"])
            bucket_splits[split[k]].append(rec)
    logger.info("Imputed %d feature-NaN values to 0 (z-score mean) across processed records", n_imputed)

    # Save train (always regenerated), val/test (only if not cached) — each with a sidecar
    # recording seed/stride/seq_len/size so later runs (incl. training) can trust or reuse it.
    filenames = {
        "train": train_pkl_name(train_size),
        "val": "val.pkl",
        "test": "test.pkl",
    }
    size_targets = {"train": train_size, "val": val_size, "test": test_size}
    for name, recs in bucket_splits.items():
        if name == "val" and val_cached:
            continue
        if name == "test" and test_cached:
            continue
        logger.info("%s: %d pixel-records", name, len(recs))
        pkl_path = out_dir / filenames[name]
        with pkl_path.open("wb") as f:
            pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Derived from the records actually saved (not the pre-pass-2 `wanted` target) so the
        # sidecar can't overstate what's really in the pkl if any grid's pass-2 fetch failed.
        split_pixels = {(r["grid"], r["y"], r["x"]) for r in recs}
        actual_windows = sum(pixel_windows[k] for k in split_pixels)
        write_sidecar(pkl_path, {
            "seed": pp["random_seed"],
            "stride": effective_stride[name],
            "seq_len": pp["seq_len"],
            "grids_hash": grids_hash,
            "size_target": size_targets[name],
            "size_label": window_label(size_targets[name]) if size_targets[name] else "full",
            "actual_window_count": actual_windows,
            "num_grids_covered": len({k[0] for k in split_pixels}),
            "num_pixels": len(split_pixels),
            "train_frac": pp["train_frac"],
            "val_frac": pp["val_frac"],
            "test_frac": pp["test_frac"],
        })

    # A --grids override scopes this run to a debug/validation subset (see the val/test cache
    # note above) — its scaler would be fit on far fewer train pixels than the real bucket, so
    # it must never overwrite the canonical scaler.pkl other runs depend on.
    if args.grids:
        logger.warning("--grids override active — not overwriting scaler.pkl (would be "
                        "computed from a debug subset, not the full train pool)")
    else:
        scaler_path = Path(cfg["paths"]["scaler"])
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with scaler_path.open("wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved scaler (%d columns) to %s", ncol, scaler_path)

    # Save split map (shows geographic coverage of this run's train/val/test selection)
    eval_dir = Path(cfg["paths"]["evaluation"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    train_label = window_label(train_size) if train_size else "full"
    light_records = [
        {"grid": k[0], "y": k[1], "x": k[2], "lat": m["lat"], "lon": m["lon"]}
        for k, m in pixel_meta.items()
    ]
    plot_data_split_map(
        light_records, split, train_subset,
        title=f"Arctic split — train {train_label}",
        save_path=eval_dir / f"arctic_data_map_train_{train_label}.png",
    )
    logger.info("Saved split map: arctic_data_map_train_%s.png", train_label)


if __name__ == "__main__":
    main()
