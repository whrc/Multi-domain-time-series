"""
Amazon domain — Step 1: preprocessing.

See domains/amazon_domain/amazon_description.md § "Step 1 — Preprocessing".

Load the monthly CSV from GCS -> filter to the station allowlist -> cyclical month +
per-station climatological features -> station-level split -> scaler fit on train ->
contiguous monthly segments -> save train/val/test pkl + scaler.
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.io import gcs_filesystem, read_csv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_TARGETS = 3


def load_filtered(cfg: dict) -> pd.DataFrame:
    """Load the primary CSV, keep allowlisted stations, and apply the rename map."""
    gcs = cfg["gcs"]
    rename = cfg["columns"]["rename"]
    fs = gcs_filesystem()
    bucket = gcs["bucket"]
    df = read_csv(fs, f"{bucket}/{gcs['files']['fire_risk_monthly']}", usecols=list(rename.keys()))
    stations = read_csv(fs, f"{bucket}/{gcs['files']['station_list']}")
    allow = set(stations["x"].astype(str))
    df = df.rename(columns=rename)
    df["station_id"] = df["station_id"].astype(str)
    df = df[df["station_id"].isin(allow)].copy()
    logger.info("Loaded %d rows for %d allowlisted stations", len(df), df["station_id"].nunique())
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical month encoding and the per-station climatological mean/std features."""
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    grp = df.groupby("station_id")
    for col in ("precip", "tmax", "tmin"):
        df[f"mean_{col}"] = grp[col].transform("mean")
        df[f"sd_{col}"] = grp[col].transform("std")
    df["ym_ord"] = df["year"] * 12 + (df["month"] - 1)
    return df


def split_stations(df: pd.DataFrame, cfg: dict) -> dict[str, str]:
    """Randomly assign each station to train/val/test."""
    pp = cfg["preprocessing"]
    rng = np.random.default_rng(pp["random_seed"])
    stations = np.asarray(df["station_id"].unique(), dtype=object)
    rng.shuffle(stations)
    n = len(stations)
    n_train = round(pp["train_frac"] * n)
    n_val = round(pp["val_frac"] * n)
    assign = {}
    for i, s in enumerate(stations):
        assign[s] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    logger.info("Stations -> train/val/test = %d/%d/%d", n_train, n_val, n - n_train - n_val)
    return assign


def main() -> None:
    cfg = load_config("amazon_domain")
    cols = cfg["columns"]
    seq_len = cfg["preprocessing"]["seq_len"]
    clim_feats = [f"{stat}_{col}" for col in cols["site_clim_cols"] for stat in ("mean", "sd")]
    all_cols = cols["dynamic"] + clim_feats + cfg["targets"]
    logger.info("Feature columns (%d) + targets (%d) = %d", len(all_cols) - NUM_TARGETS, NUM_TARGETS, len(all_cols))

    df = load_filtered(cfg)
    df = add_features(df)
    assign = split_stations(df, cfg)
    df["split"] = df["station_id"].map(assign)

    # Predictors are fully observed (EDA); fail loud rather than silently feed NaN to the model.
    feature_cols = all_cols[:-NUM_TARGETS]
    bad = [c for c in feature_cols if df[c].isna().any()]
    if bad:
        raise ValueError(f"Unexpected NaN in Amazon feature columns {bad}; predictors must be fully observed.")

    # Scaler on train rows only; nan-aware so the ~6% NaN discharge is excluded.
    train_rows = df[df["split"] == "train"]
    mean = np.nanmean(train_rows[all_cols].to_numpy(dtype=float), axis=0)
    std = np.nanstd(train_rows[all_cols].to_numpy(dtype=float), axis=0)
    mean[~np.isfinite(mean)] = 0.0
    std[(std == 0) | ~np.isfinite(std)] = 1.0
    scaler = {"mean": mean, "std": std}
    logger.info("Scaler fit on %d train rows (%d columns)", len(train_rows), len(all_cols))

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    total_gap = 0
    for station, grp in df.groupby("station_id"):
        grp = grp.sort_values("ym_ord")
        ords = grp["ym_ord"].to_numpy()
        total_gap += int(ords[-1] - ords[0] + 1 - len(ords))
        data = ((grp[all_cols].to_numpy(dtype=float) - mean) / std).astype(np.float32)
        breaks = np.where(np.diff(ords) != 1)[0] + 1
        segments, starts = [], []
        for seg, ord_piece in zip(np.split(data, breaks), np.split(ords, breaks)):
            if seg.shape[0] >= seq_len:
                segments.append(seg)
                year, month0 = divmod(int(ord_piece[0]), 12)
                starts.append((year, month0 + 1))
        if segments:
            splits[grp["split"].iloc[0]].append(
                {"station_id": station, "segments": segments, "segment_starts": starts}
            )
    logger.info("Total missing station-months (gaps) across stations: %d", total_gap)

    out_dir = Path(cfg["paths"]["preprocessed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        n_seg = sum(len(r["segments"]) for r in records)
        logger.info("%s: %d stations, %d segments", name, len(records), n_seg)
        with (out_dir / f"{name}.pkl").open("wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    scaler_path = Path(cfg["paths"]["scaler"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved scaler to %s", scaler_path)


if __name__ == "__main__":
    main()
