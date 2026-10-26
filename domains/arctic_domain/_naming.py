"""Arctic pkl naming, sidecar metadata (`{stem}.meta.json`: the seed/stride/seq_len/size a pkl was built with), and shared eval helpers."""

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
    """Output label matching train_pkl_name (50000 -> '50K'), or the explicit --label."""
    return label or window_label(train_size)


def sidecar_path(pkl_path: Path) -> Path:
    return pkl_path.with_suffix(".meta.json")


def write_sidecar(pkl_path: Path, meta: dict) -> None:
    """Atomic write: a sidecar's presence signals its pkl is complete."""
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
    """(stride, seq_len) the pkl was built with, from its sidecar (not current config)."""
    meta = load_sidecar(pkl_path)
    if meta is None:
        raise FileNotFoundError(
            f"No sidecar found for {pkl_path} (expected {pkl_path.with_suffix('.meta.json')}). "
            "Re-run 01_preprocess.py to regenerate this split with its sidecar."
        )
    return meta["stride"], meta["seq_len"]


FLUX_TARGET_NAMES = ["GPP", "RECO"]

# Let a flux-only run reuse the full-target pkl/scaler without re-preprocessing.


def select_flux_target_columns(records: list[dict], all_target_names: list[str]) -> list[dict]:
    """Slice each record's `data` to [features | GPP | RECO]; all_target_names is the config target order."""
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
    """Slice the scaler's mean/std to match select_flux_target_columns' columns."""
    n_full_targets = len(all_target_names)
    flux_idx = [all_target_names.index(t) for t in FLUX_TARGET_NAMES]
    mean, std = scaler["mean"], scaler["std"]
    n_features = len(mean) - n_full_targets
    new_mean = np.concatenate([mean[:n_features]] + [mean[n_features + i:n_features + i + 1] for i in flux_idx])
    new_std = np.concatenate([std[:n_features]] + [std[n_features + i:n_features + i + 1] for i in flux_idx])
    return {"mean": new_mean, "std": new_std}


def sample_test_pixels(seg_meta: list[dict], seed: int, n_pixels: int) -> list[tuple]:
    """Deterministic seeded draw of n_pixels distinct (grid, y, x) test pixels (reused by multi-domain)."""
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
    """Save full monthly obs/pred series for the sampled pixels to parquet."""
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
            # pred arrives pre-rounded to 3 dp; obs doesn't.
            block[f"obs_{name.lower()}"] = np.round(obs[:, i], 3)
            block[f"pred_{name.lower()}"] = pred[:, i]
        blocks.append(pd.DataFrame(block))

    sample_df = pd.concat(blocks, ignore_index=True)
    sample_df.to_parquet(save_path, index=False)
    logger.info("Saved prediction sample: %d pixels, %d rows -> %s", len(sampled_set), len(sample_df), save_path)
