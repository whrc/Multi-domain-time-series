"""
Amazon domain — Step 1: preprocessing.

Loads raw data from GCS, filters stations, renames columns, adds cyclical month
encoding and per-station static climate features, ensures temporal completeness,
then saves a full cleaned parquet, time-based train/val/test split parquets
(scaled), and a fitted StandardScaler.

NaN note: only `discharge` has missing values. All NaN targets are kept as-is;
the training loop must mask NaN positions when computing loss.

Outputs (relative to project root):
  outputs/amazon_domain/preprocessed/amazon_preprocessed.parquet  (unscaled)
  outputs/amazon_domain/preprocessed/train.parquet                 (scaled)
  outputs/amazon_domain/preprocessed/val.parquet                   (scaled)
  outputs/amazon_domain/preprocessed/test.parquet                  (scaled)
  outputs/amazon_domain/scaler.pkl
"""

import logging
import sys
from pathlib import Path

import pickle

import gcsfs
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from config.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Column definitions ────────────────────────────────────────────────────────

RAW_COLS = [
    "EstacaoCod", "year", "month",
    "Prec", "tmax", "tmin", "ET", "vpd", "DrangAr",
    "vazao", "AF", "BA",
]

COL_RENAME = {
    "EstacaoCod": "station_id",
    "Prec":       "precip",
    "ET":         "et",
    "DrangAr":    "drainage_area",
    "vazao":      "discharge",
    "AF":         "active_fire_count",
    "BA":         "burned_area",
}

DYNAMIC_PREDICTORS = ["precip", "tmax", "tmin", "et", "vpd", "drainage_area", "month_sin", "month_cos"]
STATIC_PREDICTORS  = ["mean_precip", "sd_precip", "mean_tmax", "sd_tmax", "mean_tmin", "sd_tmin"]
TARGETS            = ["discharge", "active_fire_count", "burned_area"]

FINAL_COLS = ["station_id", "year", "month"] + DYNAMIC_PREDICTORS + STATIC_PREDICTORS + TARGETS


# ── Steps ─────────────────────────────────────────────────────────────────────

def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    fs     = gcsfs.GCSFileSystem(token="google_default")
    bucket = cfg["gcs"]["bucket"]
    files  = cfg["gcs"]["files"]
    log.info("Loading %s", files["fire_risk_monthly"])
    with fs.open(f"{bucket}/{files['fire_risk_monthly']}") as f:
        df = pd.read_csv(f, usecols=RAW_COLS)
    log.info("Loading %s", files["station_list"])
    with fs.open(f"{bucket}/{files['station_list']}") as f:
        stations = pd.read_csv(f)
    log.info("Primary file shape: %s", df.shape)
    return df, stations


def filter_and_rename(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    valid = set(stations["x"].astype(str))
    df = df.rename(columns=COL_RENAME)
    df = df[df["station_id"].astype(str).isin(valid)].copy()
    log.info("After filtering: %d stations, %d rows", df["station_id"].nunique(), len(df))
    return df


def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_static_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-station long-term mean and std for precip, tmax, tmin.
    Returns the augmented df and a station→statics lookup for use after
    temporal completeness (to fill NaN statics in inserted rows).
    """
    agg = df.groupby("station_id")[["precip", "tmax", "tmin"]].agg(["mean", "std"])
    agg.columns = [
        f"{'mean' if fn == 'mean' else 'sd'}_{col}"
        for col, fn in agg.columns
    ]
    statics = agg.reset_index()
    df = df.merge(statics, on="station_id", how="left")
    return df, statics


def ensure_temporal_completeness(df: pd.DataFrame, statics: pd.DataFrame) -> pd.DataFrame:
    """
    For each station, insert NaN rows for any missing year-month between its
    first and last record. Static feature columns are re-filled from `statics`
    after reindexing (they are constant per station and would be NaN in inserted rows).
    Logs a warning for each station that had gaps.
    """
    parts: list[pd.DataFrame] = []
    gapped: list[tuple] = []

    for sid, grp in df.groupby("station_id"):
        grp = grp.sort_values(["year", "month"])
        r0, r1 = grp.iloc[0], grp.iloc[-1]
        full_dates = pd.date_range(
            start=pd.Timestamp(int(r0["year"]), int(r0["month"]), 1),
            end=pd.Timestamp(int(r1["year"]), int(r1["month"]), 1),
            freq="MS",
        )
        full_ym = pd.DataFrame({"year": full_dates.year, "month": full_dates.month})
        merged  = full_ym.merge(
            grp.drop(columns=STATIC_PREDICTORS, errors="ignore"),
            on=["year", "month"],
            how="left",
        )
        merged["station_id"] = sid
        n_inserted = len(merged) - len(grp)
        if n_inserted > 0:
            gapped.append((sid, n_inserted))
        parts.append(merged)

    result = pd.concat(parts, ignore_index=True)

    if gapped:
        log.warning("%d station(s) had temporal gaps filled with NaN:", len(gapped))
        for sid, n in gapped:
            log.warning("  station %s: %d row(s) inserted", sid, n)
    else:
        log.info("No temporal gaps found — all stations have continuous monthly records.")

    # Re-attach static features so inserted rows get the correct constant values
    result = result.drop(columns=[c for c in STATIC_PREDICTORS if c in result.columns])
    result = result.merge(statics, on="station_id", how="left")
    return result


def split_by_time(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    split_cfg     = cfg["preprocessing"]["split"]
    train_end     = split_cfg["train_end_year"]
    val_end       = split_cfg["val_end_year"]

    train = df[df["year"] <= train_end].reset_index(drop=True)
    val   = df[(df["year"] > train_end) & (df["year"] <= val_end)].reset_index(drop=True)
    test  = df[df["year"] > val_end].reset_index(drop=True)

    log.info(
        "Time split — train: %d rows (≤%d)  val: %d rows (%d–%d)  test: %d rows (>%d)",
        len(train), train_end, len(val), train_end + 1, val_end, len(test), val_end,
    )
    return {"train": train, "val": val, "test": test}


def fit_and_apply_scaler(
    splits: dict[str, pd.DataFrame],
    scaler_path: Path,
) -> dict[str, pd.DataFrame]:
    """
    Fit StandardScaler on the train split (all predictor + target columns;
    NaN-containing rows excluded from fit). Apply to all splits. Save scaler.
    """
    scale_cols = DYNAMIC_PREDICTORS + STATIC_PREDICTORS + TARGETS

    train      = splits["train"]
    fit_mask   = train[scale_cols].notna().all(axis=1)
    scaler     = StandardScaler()
    scaler.fit(train.loc[fit_mask, scale_cols])

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info("Scaler saved: %s", scaler_path)

    scaled: dict[str, pd.DataFrame] = {}
    for name, df in splits.items():
        out = df.copy()
        out[scale_cols] = out[scale_cols].astype(float)
        valid = out[scale_cols].notna().all(axis=1)
        out.loc[valid, scale_cols] = scaler.transform(out.loc[valid, scale_cols])
        scaled[name] = out
    return scaled


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg     = load_config("amazon_domain")
    out_dir = root / cfg["paths"]["preprocessed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw, stations = load_raw(cfg)
    df = filter_and_rename(df_raw, stations)
    df = add_cyclical_month(df)
    df, statics = add_static_features(df)
    df = ensure_temporal_completeness(df, statics)

    df = df.sort_values(["station_id", "year", "month"]).reset_index(drop=True)
    df = df[FINAL_COLS]

    log.info("Final dataset: %d rows, %d columns", len(df), len(df.columns))
    log.info(
        "discharge NaN: %.1f%% (kept as-is — model masks NaN targets during loss)",
        df["discharge"].isna().mean() * 100,
    )
    log.info("Shapes — %s", {c: str(df[c].dtype) for c in FINAL_COLS})

    full_path = out_dir / "amazon_preprocessed.parquet"
    df.to_parquet(full_path, index=False)
    log.info("Saved full dataset: %s", full_path)

    splits = split_by_time(df, cfg)
    scaler_path = root / cfg["paths"]["scaler"]
    scaled_splits = fit_and_apply_scaler(splits, scaler_path)

    for name, split_df in scaled_splits.items():
        path = out_dir / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        log.info(
            "Saved %s: %d stations, %d rows → %s",
            name, split_df["station_id"].nunique(), len(split_df), path,
        )


if __name__ == "__main__":
    main()
