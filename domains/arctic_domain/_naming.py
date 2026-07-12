"""Size-labeled pkl filenames and sidecar metadata for the Arctic preprocessing outputs.

A sidecar (`{stem}.meta.json`) is saved next to each train/val/test pkl recording the
seed/stride/seq_len/size actually used to build it, so `02_train.py` can read the correct
stride for whichever variant it loads instead of assuming the current config's stride, and so
a cached val/test pkl can be validated against the current config instead of trusted blindly.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def window_label(n: int) -> str:
    """Convert window count to short label: 50000 -> '50K', 2000000 -> '2M'."""
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0:
        return f"{n // 1_000}K"
    return str(n)


def train_pkl_name(train_size: int, label: str | None = None) -> str:
    return f"train_{label or window_label(train_size)}.pkl"


def run_label(train_size: int, label: str | None = None) -> str:
    """Label for this run's output artifacts, matching train_pkl_name's label
    (e.g. train_size=50000 -> '50K', matching train_50K.pkl). Pass the same
    ``--label`` used for preprocessing (e.g. '50K_s150' for a density-sweep
    variant) to load/write that variant's outputs instead of the default."""
    return label or window_label(train_size)


def sidecar_path(pkl_path: Path) -> Path:
    return pkl_path.with_suffix(".meta.json")


def write_sidecar(pkl_path: Path, meta: dict) -> None:
    """Write atomically (temp file + rename) so a process killed mid-write can never leave a
    truncated sidecar at the final path — callers that pair this with an already-committed
    pkl file rely on the sidecar's presence as the "this pkl is fully written and trustworthy"
    signal (see 01_preprocess.py's save loop), which only holds if this write can't be seen
    half-done."""
    rounded = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items()}
    target = sidecar_path(pkl_path)
    tmp = target.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(rounded, f, indent=2)
    tmp.replace(target)


def load_sidecar(pkl_path: Path) -> dict | None:
    p = sidecar_path(pkl_path)
    if not p.exists():
        return None
    try:
        with p.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def sidecar_matches(meta: dict | None, expected: dict) -> bool:
    if meta is None:
        return False
    return all(meta.get(k) == v for k, v in expected.items())


def load_stride_seq_len(pkl_path: Path) -> tuple[int, int]:
    """Read (stride, seq_len) from a pkl's sidecar — fail loudly if missing.

    Different train/val/test variants may have been built with different strides/seq_lens
    (see preprocessing.capped_stride and preprocessing.seq_len in config/arctic_domain.yaml),
    so falling back to the current config's values would silently window with the wrong
    density/context length if config changed after this pkl was built. Shared by
    02_train.py, 03_predict.py, and 04_evaluate.py so all three always agree with what a
    given pkl was actually built with, not with whatever config says right now.
    """
    meta = load_sidecar(pkl_path)
    if meta is None:
        raise FileNotFoundError(
            f"No sidecar found for {pkl_path} (expected {pkl_path.with_suffix('.meta.json')}). "
            "Re-run 01_preprocess.py to regenerate this split with its sidecar."
        )
    return meta["stride"], meta["seq_len"]


FLUX_TARGET_NAMES = ["GPP", "RECO"]

# Both helpers below let a flux-only run reuse the existing full-target preprocessed pkl/scaler
# as-is — no re-preprocessing needed. Split into two functions (rather than one doing both)
# because records and the scaler are loaded at different points in each caller and reordering
# the scaler a second time would slice an already-narrowed array incorrectly.


def select_flux_target_columns(records: list[dict], all_target_names: list[str]) -> list[dict]:
    """Reorder each record's `data` array down to `[features | GPP | RECO]`, dropping the
    accumulated-pool targets (ALD, VEGC) and moving the kept ones to the trailing position
    WindowedDataset expects.

    all_target_names: the full target order the records were built with (matches
    config/arctic_domain.yaml's `targets:` list order, e.g. [ALD, GPP, RECO, VEGC]).
    """
    n_full_targets = len(all_target_names)
    flux_idx = [all_target_names.index(t) for t in FLUX_TARGET_NAMES]
    new_records = []
    for r in records:
        data = r["data"]
        n_features = data.shape[1] - n_full_targets
        cols = [data[:, :n_features]] + [data[:, n_features + i:n_features + i + 1] for i in flux_idx]
        new_records.append({**r, "data": np.concatenate(cols, axis=1)})
    return new_records


def select_flux_scaler_stats(scaler: dict, all_target_names: list[str]) -> dict:
    """Slice/reorder the scaler's mean/std to match select_flux_target_columns' output columns.
    The scaler's per-column stats were fit independently per column, so slicing/reordering them
    (not refitting) is statistically valid."""
    n_full_targets = len(all_target_names)
    flux_idx = [all_target_names.index(t) for t in FLUX_TARGET_NAMES]
    mean, std = scaler["mean"], scaler["std"]
    n_features = len(mean) - n_full_targets
    new_mean = np.concatenate([mean[:n_features]] + [mean[n_features + i:n_features + i + 1] for i in flux_idx])
    new_std = np.concatenate([std[:n_features]] + [std[n_features + i:n_features + i + 1] for i in flux_idx])
    return {"mean": new_mean, "std": new_std}


def sample_test_pixels(seg_meta: list[dict], seed: int, n_pixels: int) -> list[tuple]:
    """Deterministic seeded draw of n_pixels distinct (grid, y, x) pixels from the sorted set
    of unique test pixels — shared by save_prediction_sample and the timeseries plots below so
    both draw from the identical sample. Reproduces identically every time this runs against
    the same (now-frozen) test.pkl, so the same sites stay directly comparable across runs and
    a future multi-domain comparison. Lives here (not in 04_evaluate.py) so the multi-domain
    pipeline can reuse the exact same site selection for its own Arctic evaluation."""
    unique_pixels = sorted({(m["grid"], m["y"], m["x"]) for m in seg_meta})
    rng = np.random.default_rng(seed)
    n = min(n_pixels, len(unique_pixels))
    sampled_idx = rng.choice(len(unique_pixels), size=n, replace=False)
    return sorted(unique_pixels[i] for i in sampled_idx)


def save_prediction_sample(
    seg_meta: list[dict],
    pred_list: list[np.ndarray],
    obs_list: list[np.ndarray],
    target_names: list[str],
    idx_map: dict[str, pd.DatetimeIndex],
    sampled_pixels: list[tuple],
    save_path: Path,
) -> None:
    """Save the full monthly obs-vs-predicted time series for a small, deterministic sample
    of test pixels (see sample_test_pixels). Unlike metrics_test.csv (aggregated RMSE/NSE/KGE/
    PBIAS per pixel/target/period), this keeps raw values so a specific pixel's time series can
    still be plotted after test.pkl is deleted to free disk space.
    """
    sampled_set = set(sampled_pixels)
    blocks = []
    for meta, pred, obs in zip(seg_meta, pred_list, obs_list):
        if (meta["grid"], meta["y"], meta["x"]) not in sampled_set:
            continue
        time = idx_map["ssp1" if "ssp1" in meta["ssp"] else "ssp5"]
        block = {
            "grid": meta["grid"], "y": meta["y"], "x": meta["x"],
            "lat": meta["lat"], "lon": meta["lon"], "ssp": meta["ssp"], "time": time,
        }
        for i, name in enumerate(target_names):
            # pred is expected pre-rounded to 3dp by the caller (matches the NetCDF written by
            # 03_predict); obs isn't rounded upstream, so round it here to match.
            block[f"obs_{name.lower()}"] = np.round(obs[:, i], 3)
            block[f"pred_{name.lower()}"] = pred[:, i]
        blocks.append(pd.DataFrame(block))

    sample_df = pd.concat(blocks, ignore_index=True)
    sample_df.to_parquet(save_path, index=False)
    logger.info("Saved prediction sample: %d pixels, %d rows -> %s", len(sampled_set), len(sample_df), save_path)
