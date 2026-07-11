"""Size-labeled pkl filenames and sidecar metadata for the Arctic preprocessing outputs.

A sidecar (`{stem}.meta.json`) is saved next to each train/val/test pkl recording the
seed/stride/seq_len/size actually used to build it, so `02_train.py` can read the correct
stride for whichever variant it loads instead of assuming the current config's stride, and so
a cached val/test pkl can be validated against the current config instead of trusted blindly.
"""

import json
from pathlib import Path

import numpy as np


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
