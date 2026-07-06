"""Size-labeled pkl filenames and sidecar metadata for the Arctic preprocessing outputs.

A sidecar (`{stem}.meta.json`) is saved next to each train/val/test pkl recording the
seed/stride/seq_len/size actually used to build it, so `02_train.py` can read the correct
stride for whichever variant it loads instead of assuming the current config's stride, and so
a cached val/test pkl can be validated against the current config instead of trusted blindly.
"""

import json
from pathlib import Path


def window_label(n: int) -> str:
    """Convert window count to short label: 50000 -> '50K', 2000000 -> '2M'."""
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0:
        return f"{n // 1_000}K"
    return str(n)


def train_pkl_name(train_size: int | None) -> str:
    return f"train_{window_label(train_size)}.pkl" if train_size else "train_full.pkl"


def run_label(train_size: int | None) -> str:
    """Label for this run's output artifacts, matching train_pkl_name's label
    (e.g. train_size=50000 -> '50K', matching train_50K.pkl; None -> 'full')."""
    return window_label(train_size) if train_size else "full"


def sidecar_path(pkl_path: Path) -> Path:
    return pkl_path.with_suffix(".meta.json")


def write_sidecar(pkl_path: Path, meta: dict) -> None:
    rounded = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items()}
    with sidecar_path(pkl_path).open("w") as f:
        json.dump(rounded, f, indent=2)


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
