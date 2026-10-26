"""Arctic Step 1: build per-pixel train/val/test sequences + scaler from GCS grids.

Whole grids are split train/val/test (latitude-stratified). See arctic_description.md "Step 1".
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
from domains.arctic_domain._naming import (  # noqa: E402
    load_sidecar, sidecar_matches, sidecar_path, window_label, write_sidecar,
)
# shared.plots is imported lazily in main(): _fetch_grid_worker.py execs this file per grid fetch.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_TARGETS = 4
CLIM_VARS = ["tair", "precip", "nirr", "vapor_press"]
EXCLUDE = {"lat", "lon", "lambert_azimuthal_equal_area"}
GRID_NAME_RE = re.compile(r"^H\d+_V\d+$")  # excludes non-grid bucket entries
# Permanently unfetchable; excluded from auto-discovery only (an explicit --grids is unaffected).
KNOWN_BROKEN_GRIDS = {"H15_V13", "H17_V18", "H19_V17"}
# Repeatedly failed to fetch; excluded the same way. Worth re-testing.
FLAKY_GRIDS_20260710 = {"H11_V16", "H11_V19", "H14_V15", "H16_V7", "H17_V3", "H19_V13", "H19_V18", "H9_V19"}

CONFIG_MISMATCH_EXIT_CODE = 2
# Shared by assign_grid_splits() and main()'s val/test cache key; both must agree.
DEFAULT_SPLIT_LAT_BINS = 6


class ConfigMismatchError(RuntimeError):
    """Config/precondition error that no retry can fix; run_preprocess_resilient.sh stops on its exit code."""


def monthly_index_map(cfg: dict) -> dict[str, pd.DatetimeIndex]:
    """Per-scenario monthly DatetimeIndex from config (ssp1 -> 2400 months, ssp5 -> 912)."""
    return {k: pd.date_range(v["start"], v["end"], freq="MS") for k, v in cfg["time"]["scenarios"].items()}


def _to_y_x(ds: xr.Dataset) -> xr.Dataset:
    return ds.rename({k: k.lower() for k in ("Y", "X") if k in ds.dims})


def _month_start(times) -> pd.DatetimeIndex:
    """Map cftime or datetime stamps to first-of-month."""
    return pd.to_datetime([f"{t.year}-{t.month:02d}-01" for t in times])


def load_static(fs, inp, cfg) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Stack static vars (excluding lat/lon/CRS) into (nStatic, ny, nx); also return lat/lon."""
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
    # interpolate doesn't extrapolate past the last yearly anchor; ffill/bfill covers the tails.
    s = s.reindex(s.index.union(monthly_index)).interpolate("time").reindex(monthly_index).ffill().bfill()
    return s.to_numpy()


def load_climate(fs, inp, is_ssp1: bool, monthly_index: pd.DatetimeIndex) -> np.ndarray:
    """Climate vars onto the monthly axis; returns (T, ny, nx, 4)."""
    files = ["historic-climate.nc", "projected-climate.nc"] if is_ssp1 else ["projected-climate.nc"]
    parts = []
    for f in files:
        ds = _to_y_x(read_netcdf(fs, inp(f), prefer_engine="scipy"))
        parts.append(ds.assign_coords(time=_month_start(ds["time"].values)))
    full = xr.concat(parts, dim="time", data_vars="minimal") if len(parts) > 1 else parts[0]
    return np.stack([full[v].reindex(time=monthly_index).values for v in CLIM_VARS], axis=-1)


def load_targets(fs, tgt, is_ssp1: bool, targets_cfg: list[dict], monthly_index: pd.DatetimeIndex,
                 proj_start: int) -> np.ndarray:
    """Targets onto the monthly axis; returns (T, ny, nx, 4) in config order.

    Historical (_tr) exists only for SSP1. Yearly targets sit at January positions; their
    projected files carry wrong year labels (1901-), overridden to proj_start-.
    """
    layers = []
    for t in targets_cfg:
        var = t["name"]
        yearly = t["resolution"] == "yearly"
        parts = []
        if is_ssp1:
            ds_tr = read_netcdf(fs, tgt(t["historical"]), prefer_engine="scipy")
            parts.append(ds_tr[var].assign_coords(time=_month_start(ds_tr["time"].values)))
        ds_sc = read_netcdf(fs, tgt(t["projected"]), prefer_engine="scipy")
        times = ds_sc["time"].values
        if yearly and times[0].year < 2000:
            fixed = pd.to_datetime([f"{proj_start + i}-01-01" for i in range(len(times))])
        else:
            fixed = _month_start(times)
        parts.append(ds_sc[var].assign_coords(time=fixed))
        da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
        layers.append(da.reindex(time=monthly_index).values)
    return np.stack(layers, axis=-1)


def fetch_grid_records(cfg: dict, fs, grid: str, idx_map: dict, proj_start: int) -> list[dict]:
    """One grid's raw (un-normalised) per-land-pixel records, both scenarios."""
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
        # TEM fill sentinel (~-9999); some projected files lack a _FillValue attr, so mask explicitly.
        fill_mask = targets <= -9000
        n_masked = int(fill_mask.sum())
        if n_masked:
            targets[fill_mask] = np.nan
            logger.warning("%s/%s: masked %d fill-value (~-9999) target entries", grid, scenario, n_masked)
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


def fetch_grid_records_isolated(
    grid: str, cache_dir: Path, timeout: int, retries: int, failed_marker_ttl_seconds: int,
) -> list[dict]:
    """Fetch one grid's records in a fresh subprocess (gcsfs hangs on in-process reuse), with retries.

    Raw records are not cached (multi-GB per grid). A grid that exhausts retries gets a
    cache_dir/{grid}.failed marker that short-circuits later attempts until it expires.
    """
    failed_marker = cache_dir / f"{grid}.failed"
    if failed_marker.exists() and time.time() - failed_marker.stat().st_mtime < failed_marker_ttl_seconds:
        raise RuntimeError(f"Grid {grid!r} previously exhausted retries (cached failure at {failed_marker})")

    last_err: Exception | None = None
    for attempt in range(1, retries + 2):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        tmp_path.unlink()
        try:
            result = subprocess.run(
                [sys.executable, str(_WORKER), "--grid", grid, "--out", str(tmp_path)],
                timeout=timeout, check=True, capture_output=True, text=True,
            )
            for line in result.stderr.splitlines():
                if " WARNING " in line:
                    logger.warning("%s: %s", grid, line.split(" WARNING ", 1)[1])
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as err:
            # On TimeoutExpired, err.stderr is bytes even with text=True.
            raw_stderr = err.stderr
            if isinstance(raw_stderr, bytes):
                raw_stderr = raw_stderr.decode("utf-8", errors="replace")
            stderr_tail = "\n".join((raw_stderr or "").splitlines()[-20:])
            if not tmp_path.exists():
                last_err = err
                logger.warning(
                    "fetch_grid_records_isolated(%s) attempt %d/%d failed with no output: %s%s",
                    grid, attempt, retries + 1, err,
                    f"\n{stderr_tail}" if stderr_tail else "",
                )
                continue
            # Worker sometimes hangs on shutdown after writing its output; use the output.
            logger.warning(
                "fetch_grid_records_isolated(%s): worker didn't exit cleanly (%s) but wrote "
                "output — using it%s", grid, err,
                f"\n{stderr_tail}" if stderr_tail else "",
            )
        try:
            with tmp_path.open("rb") as f:
                recs = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as err:
            # Truncated output from a worker killed mid-write.
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


def _grid_centroid(lat_lon: dict[tuple, tuple]) -> tuple[float, float]:
    """Mean (lat, lon) over a grid's land pixels."""
    lats = [v[0] for v in lat_lon.values()]
    lons = [v[1] for v in lat_lon.values()]
    return float(np.mean(lats)), float(np.mean(lons))


def assign_grid_splits(grid_centroids: dict[str, tuple[float, float]], pp: dict) -> dict[str, str]:
    """Deterministic latitude-stratified train/val/test assignment of whole grids.

    Grids are binned into quantile-based latitude strata; each stratum is shuffled and cut
    at train_frac/val_frac/test_frac.
    """
    # Clamp bins to ~5 grids/stratum so small --grids runs don't round val/test to 0.
    grids = sorted(grid_centroids)
    n_lat_bins = min(pp.get("split_lat_bins", DEFAULT_SPLIT_LAT_BINS), max(1, len(grids) // 5))
    lats = np.array([grid_centroids[g][0] for g in grids])
    # Inner quantile edges only (len = n_lat_bins - 1).
    lat_edges = np.quantile(lats, np.linspace(0, 1, n_lat_bins + 1)[1:-1]) if n_lat_bins > 1 else np.array([])

    strata: dict[int, list[str]] = defaultdict(list)
    for g in grids:
        lat, _ = grid_centroids[g]
        strata[int(np.digitize(lat, lat_edges))].append(g)

    result: dict[str, str] = {}
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        rng = np.random.default_rng([pp["random_seed"], zlib.crc32(f"lat_stratum_{stratum}".encode())])
        rng.shuffle(members)
        n = len(members)
        n_train = round(pp["train_frac"] * n)
        n_val = round(pp["val_frac"] * n)
        for i, g in enumerate(members):
            result[g] = "train" if i < n_train else "val" if i < n_train + n_val else "test"

    n_g = len(result)
    counts = {s: sum(v == s for v in result.values()) for s in ("train", "val", "test")}
    if n_g == 0:
        logger.info("Grid-level split: 0 grids had any land pixels — nothing to split")
    else:
        logger.info(
            "Grid-level split: %d latitude strata, train=%d (%.1f%%) val=%d (%.1f%%) test=%d (%.1f%%) of %d grids",
            len(strata), counts["train"], 100 * counts["train"] / n_g, counts["val"], 100 * counts["val"] / n_g,
            counts["test"], 100 * counts["test"] / n_g, n_g,
        )
        for split_name in ("train", "val", "test"):
            if counts[split_name] == 0:
                logger.warning(
                    "Grid-level split: 0 grids assigned to %s (%d total grids across %d strata) — "
                    "%s.pkl will be empty. This can happen with a small --grids override; use more "
                    "grids, fewer split_lat_bins, or adjust the split fractions.",
                    split_name, n_g, len(strata), split_name,
                )
    return result


def subsample_pixels_round_robin(
    pixel_windows: dict[tuple, int],
    split_name: str,
    size: int,
    seed: int,
) -> set[tuple]:
    """Pick whole pixels round-robin across grids (one per grid per pass) until `size` windows.

    pixel_windows: (grid, y, x) -> window count, already scoped to one split.
    """
    by_grid: dict[str, list[tuple]] = defaultdict(list)
    for k in pixel_windows:
        by_grid[k[0]].append(k)

    rng = np.random.default_rng(seed)
    for grid in sorted(by_grid):
        rng.shuffle(by_grid[grid])

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
    """Yield (grid, work_fn(grid)); max_workers<=1 runs sequentially, else via a thread pool."""
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
    """Per-grid pickle cache at cache_dir/{grid}.pkl, recomputed when `key` differs; atomic writes."""
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


def _grid_summary_from_records(recs: list[dict]) -> dict:
    """Reduce a grid's raw records to pass 1's small summary: pixel keys, lat/lon, scenarios, scaler sums.

    Independent of the split, so the caller sums scaler stats over train grids only after splitting.
    """
    keys = list({(r["y"], r["x"]) for r in recs})
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


def _verify_gcs_access(fs, bucket: str, project_id: str) -> None:
    try:
        fs.ls(bucket)
    except Exception as err:
        raise RuntimeError(
            f"Could not list GCS bucket '{bucket}'. If running locally, set up Application "
            "Default Credentials first: `gcloud auth application-default login` "
            f"(project {project_id})."
        ) from err


def main() -> None:
    """Pass 1: fetch + summarise every grid; split grids and fit scaler. Pass 2: re-fetch wanted pixels, normalise, save."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=None,
                        help="Override preprocessing.train_size from config")
    parser.add_argument("--capped-stride", type=int, default=None,
                        help="Override preprocessing.capped_stride from config. Always used for "
                             "val/test; used for train too unless --train-capped-stride or "
                             "--sweep-strides is given.")
    parser.add_argument("--train-capped-stride", type=int, default=None,
                        help="Override capped_stride for TRAIN only, decoupled from val/test's "
                             "(which always use --capped-stride/config). Lets val/test be built "
                             "once, at one fixed density, and reused across different training-"
                             "density experiments. Mutually exclusive with --sweep-strides.")
    parser.add_argument("--label", type=str, default=None,
                        help="Override the train pkl filename / sidecar size_label / split-map "
                             "filename instead of the default train_size-derived label (e.g. "
                             "'50K'). Lets multiple density variants at the same --train-size "
                             "coexist without overwriting each other. Mutually exclusive with "
                             "--sweep-strides (which auto-labels per stride).")
    parser.add_argument("--sweep-strides", type=str, default=None,
                        help="Comma-separated TRAIN capped_stride values to build in one pass "
                             "(e.g. '50,100,150,200,250'). Fetches each wanted grid once (the "
                             "union of every listed stride's wanted train pixels), then splits "
                             "the result locally into one train_{label}_s{stride}.pkl per value "
                             "— avoids paying for N separate GCS passes to compare N densities. "
                             "val/test are built once, at --capped-stride, unaffected by this. "
                             "Mutually exclusive with --train-capped-stride and --label.")
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
    if not train_size:
        raise ValueError(
            "preprocessing.train_size (or --train-size) must be a positive int — uncapped/full "
            "training runs are not supported (too expensive to trigger by accident)."
        )
    capped_stride = args.capped_stride if args.capped_stride is not None else pp["capped_stride"]
    max_workers = args.max_workers if args.max_workers is not None else pp["max_workers"]

    sweep_mode = args.sweep_strides is not None
    if sweep_mode and (args.train_capped_stride is not None or args.label is not None):
        raise ValueError(
            "--sweep-strides is mutually exclusive with --train-capped-stride and --label — "
            "sweep mode auto-derives both (each stride gets its own train_{label}_s{stride}.pkl)."
        )
    train_capped_stride = args.train_capped_stride if args.train_capped_stride is not None else capped_stride
    # Non-sweep mode is a sweep of one stride.
    sweep_strides_list = (
        [int(s) for s in args.sweep_strides.split(",")] if sweep_mode else [train_capped_stride]
    )
    label = args.label  # None => window_label(train_size)

    out_dir = Path(cfg["paths"]["preprocessed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".grid_failed_cache"  # .failed markers only
    summary_cache_dir = out_dir / ".grid_pass1_summary_cache"  # makes pass 1 resumable

    fs = gcs_filesystem()
    bucket = cfg["gcs"]["bucket"].replace("gs://", "")
    _verify_gcs_access(fs, bucket, cfg["gcs"]["project_id"])  # before anything destructive

    if args.force_recompute:
        for fname in ("val.pkl", "test.pkl"):
            p = out_dir / fname
            if p.exists():
                p.unlink()
                logger.info("force-recompute: deleted %s", p)

    grids = args.grids.split(",") if args.grids else pp.get("grids")
    if not grids:
        excluded = KNOWN_BROKEN_GRIDS | FLAKY_GRIDS_20260710
        grids = sorted(
            p.split("/")[-1] for p in fs.ls(bucket)
            if fs.isdir(p) and GRID_NAME_RE.match(p.split("/")[-1])
            and p.split("/")[-1] not in excluded
        )
    logger.info("Grids: %s", grids)
    grids_hash = zlib.crc32(",".join(sorted(grids)).encode())

    idx_map = monthly_index_map(cfg)
    proj_start = cfg["time"]["projected_start_year"]

    val_size  = pp.get("val_size")
    test_size = pp.get("test_size")
    # val/test always use capped_stride (independent of train's), so one fixed held-out set is
    # reused across runs. Their cache key covers everything that changes which pixels they hold.
    frac_key = {"train_frac": pp["train_frac"], "val_frac": pp["val_frac"], "test_frac": pp["test_frac"]}
    strat_key = {"split_unit": "grid", "split_lat_bins": pp.get("split_lat_bins", DEFAULT_SPLIT_LAT_BINS)}
    expected_val = {"seed": pp["random_seed"], "stride": capped_stride, "seq_len": pp["seq_len"],
                     "size_target": val_size, "grids_hash": grids_hash, **frac_key, **strat_key}
    expected_test = {"seed": pp["random_seed"], "stride": capped_stride, "seq_len": pp["seq_len"],
                      "size_target": test_size, "grids_hash": grids_hash, **frac_key, **strat_key}
    val_cached  = ((out_dir / "val.pkl").exists()
                   and sidecar_matches(load_sidecar(out_dir / "val.pkl"), expected_val))
    test_cached = ((out_dir / "test.pkl").exists()
                   and sidecar_matches(load_sidecar(out_dir / "test.pkl"), expected_test))
    # Never silently rebuild an existing val/test (that would break comparability with prior
    # results); a missing or mismatched sidecar requires --force-recompute.
    for name, existing_meta, expected in (("val", load_sidecar(out_dir / "val.pkl"), expected_val),
                                           ("test", load_sidecar(out_dir / "test.pkl"), expected_test)):
        pkl_path = out_dir / f"{name}.pkl"
        if not pkl_path.exists():
            continue
        if existing_meta is None:
            raise ConfigMismatchError(
                f"{name}.pkl exists but its sidecar ({name}.meta.json) is missing or unreadable — "
                f"refusing to silently regenerate it (that would change the held-out population and "
                f"break comparability with every prior result evaluated against the current "
                f"{name}.pkl, with no trace of why). Inspect outputs/arctic_domain/preprocessed/ "
                f"manually — this can happen if a prior run was killed between writing the pkl and "
                f"its sidecar — then either restore a matching sidecar or pass --force-recompute to "
                f"intentionally rebuild."
            )
        if not sidecar_matches(existing_meta, expected):
            diff = {k: (existing_meta.get(k), v) for k, v in expected.items() if existing_meta.get(k) != v}
            raise ConfigMismatchError(
                f"{name}.pkl exists but its sidecar doesn't match this run's config — refusing to "
                f"silently regenerate it (that would change the held-out population and break "
                f"comparability with every prior result evaluated against the current {name}.pkl). "
                f"Mismatched fields (existing -> expected): {diff}. If this mismatch is intentional "
                f"(e.g. a deliberate split-fraction or stratification change), pass --force-recompute "
                f"to rebuild val/test on purpose."
            )
    if val_cached:
        logger.info("val.pkl already exists and matches config — skipping (cached)")
    if test_cached:
        logger.info("test.pkl already exists and matches config — skipping (cached)")

    # ---------- Pass 1a: fetch every grid (one at a time in memory) and keep only its summary.
    grid_summaries: dict[str, dict] = {}
    visited_grids: list[str] = []

    # Bump when _grid_summary_from_records' output changes, to invalidate cached summaries.
    SUMMARY_SCHEMA_VERSION = 2
    summary_key = {"schema_version": SUMMARY_SCHEMA_VERSION}

    def get_grid_summary(grid: str) -> dict:
        def compute():
            recs = fetch_grid_records_isolated(
                grid, cache_dir, pp["fetch_timeout_seconds"], pp["fetch_retries"],
                pp["failed_marker_ttl_seconds"],
            )
            return _grid_summary_from_records(recs) if recs else {
                "keys": [], "lat_lon": {}, "scenarios_present": {}, "ncol": None,
                "scaler_sum": None, "scaler_sumsq": None, "scaler_count": None,
            }
        return _cached_by_grid(summary_cache_dir, grid, summary_key, compute)

    def windows_for_pixel(scenarios: set, stride: int) -> int:
        return sum(
            max(0, (len(idx_map["ssp1" if "ssp1" in ssp else "ssp5"]) - pp["seq_len"]) // stride + 1)
            for ssp in scenarios
        )

    for i, (grid, summary) in enumerate(_run_concurrent(get_grid_summary, grids, max_workers), start=1):
        visited_grids.append(grid)
        grid_summaries[grid] = summary
        logger.info("[pass 1a/2] %d/%d grids fetched (%s: %d land pixels)", i, len(grids), grid, len(summary["keys"]))

    missing_grids = sorted(set(grids) - set(visited_grids))
    if missing_grids:
        raise RuntimeError(
            f"Pass 1 permanently gave up on {len(missing_grids)}/{len(grids)} grid(s) after "
            f"exhausting retries: {missing_grids} — failing loudly instead of silently "
            "shipping a train/val/test split with missing geographic coverage. "
            "run_preprocess_resilient.sh will retry the whole run; a transient grid failure "
            "(e.g. a GCS blip) typically clears on the next attempt, resuming from the "
            "pass-1/pass-2 caches for every grid that already succeeded."
        )

    # ---------- Pass 1b (in memory): split grids, build pixel split map, fit scaler on train grids.
    grid_centroids = {g: _grid_centroid(s["lat_lon"]) for g, s in grid_summaries.items() if s["keys"]}
    grid_split = assign_grid_splits(grid_centroids, pp)

    split: dict[tuple, str] = {}
    pixel_scenarios: dict[tuple, set] = {}
    pixel_meta: dict[tuple, dict] = {}
    ncol: int | None = None
    scaler_sum = scaler_sumsq = scaler_count = None
    for g, summary in grid_summaries.items():
        if not summary["keys"]:
            continue
        grid_label = grid_split[g]  # not `label`: would clobber the outer --label value
        if grid_label == "train":
            if ncol is None:
                ncol = summary["ncol"]
                scaler_sum = np.zeros(ncol)
                scaler_sumsq = np.zeros(ncol)
                scaler_count = np.zeros(ncol)
            scaler_sum += summary["scaler_sum"]
            scaler_sumsq += summary["scaler_sumsq"]
            scaler_count += summary["scaler_count"]
        for y, x in summary["keys"]:
            k = (g, y, x)
            split[k] = grid_label
            pixel_scenarios[k] = set(summary["scenarios_present"][(y, x)])
            lat, lon = summary["lat_lon"][(y, x)]
            pixel_meta[k] = {"lat": lat, "lon": lon}

    if ncol is None:
        raise ConfigMismatchError(
            "No grid was assigned to train — cannot fit a scaler. This can happen with a very "
            "small --grids override; use more grids spanning multiple latitude bands."
        )
    empty_splits = [s for s in ("val", "test") if not any(v == s for v in split.values())]
    if empty_splits:
        raise ConfigMismatchError(
            f"No grid was assigned to {' or '.join(empty_splits)} — cannot build a non-empty split. "
            "This can happen with a very small --grids override; use more grids spanning multiple "
            "latitude bands, or fewer split_lat_bins."
        )

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
        "Grid-level split: train=%d val=%d test=%d pixels across %d/%d grids visited",
        total_train, total_val, total_test, len(visited_grids), len(grids),
    )

    pixel_windows_valtest = {k: windows_for_pixel(s, capped_stride) for k, s in pixel_scenarios.items()}

    val_subset: set[tuple] | None = None
    if not val_cached and val_size:
        val_windows = {k: w for k, w in pixel_windows_valtest.items() if split[k] == "val"}
        val_subset = subsample_pixels_round_robin(val_windows, "val", val_size, pp["random_seed"])

    test_subset: set[tuple] | None = None
    if not test_cached and test_size:
        test_windows = {k: w for k, w in pixel_windows_valtest.items() if split[k] == "test"}
        test_subset = subsample_pixels_round_robin(test_windows, "test", test_size, pp["random_seed"])

    pixel_windows_by_stride: dict[int, dict[tuple, int]] = {}
    train_subset_by_stride: dict[int, set[tuple]] = {}
    for stride in sweep_strides_list:
        pixel_windows_by_stride[stride] = {k: windows_for_pixel(s, stride) for k, s in pixel_scenarios.items()}
        train_windows = {k: w for k, w in pixel_windows_by_stride[stride].items() if split[k] == "train"}
        train_subset_by_stride[stride] = subsample_pixels_round_robin(
            train_windows, f"train(stride={stride})", train_size, pp["random_seed"]
        )
    # Union across strides so pass 2 fetches each grid once.
    wanted_train: set[tuple] = set().union(*train_subset_by_stride.values())

    wanted: set[tuple] = set()
    for k, s in split.items():
        if s == "train" and k in wanted_train:
            wanted.add(k)
        elif s == "val" and not val_cached and (val_subset is None or k in val_subset):
            wanted.add(k)
        elif s == "test" and not test_cached and (test_subset is None or k in test_subset):
            wanted.add(k)
    wanted_grids = sorted({k[0] for k in wanted})
    logger.info("[pass 2/2] Re-fetching %d/%d grids containing wanted pixels", len(wanted_grids), len(grids))

    wanted_by_grid: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for k in wanted:
        wanted_by_grid[k[0]].append((k[1], k[2]))
    for g in wanted_by_grid:
        wanted_by_grid[g].sort()

    # ---------- Pass 2: re-fetch grids with wanted pixels, filter + normalise. Cached per grid
    # (filtered records only) keyed on the wanted pixels and the scaler, so it's resumable.
    pass2_cache_dir = out_dir / ".grid_pass2_records_cache"
    scaler_fingerprint = zlib.crc32(np.round(np.concatenate([scaler["mean"], scaler["std"]]), 6).tobytes())

    def get_grid_pass2_records(grid: str) -> tuple[list[dict], int]:
        def compute():
            processed = []
            n_imputed_here = 0
            for r in fetch_grid_records_isolated(
                grid, cache_dir, pp["fetch_timeout_seconds"], pp["fetch_retries"],
                pp["failed_marker_ttl_seconds"],
            ):
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

        pass2_key = {"wanted_keys": tuple(wanted_by_grid[grid]), "scaler_fingerprint": scaler_fingerprint}
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

    for name in ("val", "test"):
        recs = bucket_splits[name]
        if name == "val" and val_cached:
            continue
        if name == "test" and test_cached:
            continue
        logger.info("%s: %d pixel-records", name, len(recs))
        pkl_path = out_dir / f"{name}.pkl"
        # Drop old sidecar first, write sidecar last: a present sidecar means the pkl is complete.
        sidecar_path(pkl_path).unlink(missing_ok=True)
        pkl_tmp = pkl_path.with_suffix(".tmp")
        with pkl_tmp.open("wb") as f:
            pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
        pkl_tmp.replace(pkl_path)
        split_pixels = {(r["grid"], r["y"], r["x"]) for r in recs}
        actual_windows = sum(pixel_windows_valtest[k] for k in split_pixels)
        size_target = val_size if name == "val" else test_size
        write_sidecar(pkl_path, {
            "seed": pp["random_seed"],
            "stride": capped_stride,
            "seq_len": pp["seq_len"],
            "grids_hash": grids_hash,
            "size_target": size_target,
            "size_label": window_label(size_target),
            "actual_window_count": actual_windows,
            "num_grids_covered": len({k[0] for k in split_pixels}),
            "num_pixels": len(split_pixels),
            "train_frac": pp["train_frac"],
            "val_frac": pp["val_frac"],
            "test_frac": pp["test_frac"],
            **strat_key,
        })

    # One train pkl per swept stride, filtered from the union-fetched train records.
    train_records_by_pixel: dict[tuple, list[dict]] = defaultdict(list)
    for r in bucket_splits["train"]:
        train_records_by_pixel[(r["grid"], r["y"], r["x"])].append(r)
    for stride in sweep_strides_list:
        recs = [r for k in train_subset_by_stride[stride] if k in train_records_by_pixel
                for r in train_records_by_pixel[k]]
        # Staggered windowing: deterministic per-pixel phase offset (same for both scenarios).
        recs = [
            {**r, "data": r["data"][
                zlib.crc32(f"{pp['random_seed']}:{r['grid']}:{r['y']}:{r['x']}".encode()) % stride:
            ]}
            for r in recs
        ]
        if sweep_mode:
            stride_label = f"{window_label(train_size)}_s{stride}"
        elif label is not None:
            stride_label = label
        else:
            # Suffix the stride so a --train-capped-stride run doesn't overwrite the default pkl.
            stride_label = window_label(train_size)
            if args.train_capped_stride is not None:
                stride_label += f"_s{stride}"
        pkl_path = out_dir / f"train_{stride_label}.pkl"
        logger.info("train (stride=%d): %d pixel-records -> %s", stride, len(recs), pkl_path.name)
        sidecar_path(pkl_path).unlink(missing_ok=True)
        pkl_tmp = pkl_path.with_suffix(".tmp")
        with pkl_tmp.open("wb") as f:
            pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
        pkl_tmp.replace(pkl_path)
        split_pixels = {(r["grid"], r["y"], r["x"]) for r in recs}
        actual_windows = sum(pixel_windows_by_stride[stride][k] for k in split_pixels)
        write_sidecar(pkl_path, {
            "seed": pp["random_seed"],
            "stride": stride,
            "seq_len": pp["seq_len"],
            "grids_hash": grids_hash,
            "size_target": train_size,
            "size_label": stride_label,
            "actual_window_count": actual_windows,
            "num_grids_covered": len({k[0] for k in split_pixels}),
            "num_pixels": len(split_pixels),
            "train_frac": pp["train_frac"],
            "val_frac": pp["val_frac"],
            "test_frac": pp["test_frac"],
            **strat_key,
        })

    if args.grids:
        logger.warning("--grids override active — not overwriting scaler.pkl (would be "
                        "computed from a debug subset, not the full train pool)")
    else:
        scaler_path = Path(cfg["paths"]["scaler"])
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with scaler_path.open("wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved scaler (%d columns) to %s", ncol, scaler_path)

    # Split coverage map; in sweep mode "train" is the union over strides.
    from shared.plots import plot_data_split_map
    eval_dir = Path(cfg["paths"]["evaluation"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    map_label = f"{window_label(train_size)}_sweep" if sweep_mode else (label or window_label(train_size))
    light_records = [
        {"grid": k[0], "y": k[1], "x": k[2], "lat": m["lat"], "lon": m["lon"]}
        for k, m in pixel_meta.items()
    ]
    plot_data_split_map(
        light_records, split, {"train": wanted_train, "val": val_subset, "test": test_subset},
        title=f"Arctic split — train {map_label}",
        save_path=eval_dir / f"arctic_data_map_train_{map_label}.png",
    )
    logger.info("Saved split map: arctic_data_map_train_%s.png", map_label)


if __name__ == "__main__":
    try:
        main()
    except ConfigMismatchError as err:
        logger.error(str(err))
        sys.exit(CONFIG_MISMATCH_EXIT_CODE)
