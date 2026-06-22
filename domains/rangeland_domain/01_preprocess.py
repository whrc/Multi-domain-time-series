"""
Rangeland domain — Step 1: preprocessing.

See domains/rangeland_domain/rangeland_description.md § "Step 1 — Preprocessing".

Merge the four PFT CSVs -> monthly aggregation -> PFT-stratified site split ->
per-site climatological means -> 22-feature/10-target rows -> scaler fit on train ->
contiguous segments -> save train/val/test pkl + scaler.
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_and_merge(cfg: dict) -> pd.DataFrame:
    """Read the four local CSVs keeping only the columns this project uses."""
    cols = cfg["columns"]
    targets = cfg["targets"]["fluxes"] + cfg["targets"]["pools"]
    keep = cols["id"] + cols["static"] + cols["dynamic"] + targets
    data_dir = Path(cfg["data"]["dir"])
    frames = [pd.read_csv(data_dir / f, usecols=keep) for f in cfg["data"]["files"]]
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    logger.info("Loaded %d rows across %d sites", len(df), df["site"].nunique())
    return df


def monthly_aggregate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Aggregate ~5-day records to monthly: prcp summed, everything else mean-aggregated.

    Site-months with fewer than ``min_records_per_month`` records are dropped first.
    """
    cols = cfg["columns"]
    targets = cfg["targets"]["fluxes"] + cfg["targets"]["pools"]
    min_records = cfg["preprocessing"]["min_records_per_month"]

    df = df.copy()
    df["ym"] = df["time"].dt.to_period("M")
    sizes = df.groupby(["site", "ym"])["time"].transform("size")
    dropped = int((sizes < min_records).sum())
    df = df[sizes >= min_records]
    logger.info("Dropped %d sparse site-month records (< %d)", dropped, min_records)

    value_cols = cols["dynamic"] + cols["static"] + targets
    agg = {c: "mean" for c in value_cols}
    agg["prcp"] = "sum"
    agg["PFT"] = "first"
    m = df.groupby(["site", "ym"], as_index=False).agg(agg)
    m["year"] = m["ym"].dt.year
    m["month"] = m["ym"].dt.month
    m["ym_ord"] = m["year"] * 12 + (m["month"] - 1)
    logger.info("Aggregated to %d site-months", len(m))
    return m


def split_sites(m: pd.DataFrame, cfg: dict) -> dict[str, str]:
    """Assign each site to train/val/test within its PFT, guaranteeing >=1 site per split."""
    pp = cfg["preprocessing"]
    rng = np.random.default_rng(pp["random_seed"])
    site_split: dict[str, str] = {}
    for pft in cfg["columns"]["pft_order"]:
        sites = np.asarray(m.loc[m["PFT"] == pft, "site"].unique(), dtype=object)
        rng.shuffle(sites)
        n = len(sites)
        n_val = max(1, round(pp["val_frac"] * n))
        n_test = max(1, round(pp["test_frac"] * n))
        # Guarantee >=1 train site by shrinking test then val (a PFT needs >=3 sites to fill all splits).
        while n - n_val - n_test < 1 and n_test > 1:
            n_test -= 1
        while n - n_val - n_test < 1 and n_val > 1:
            n_val -= 1
        val, test, train = sites[:n_val], sites[n_val : n_val + n_test], sites[n_val + n_test :]
        for s in train:
            site_split[s] = "train"
        for s in val:
            site_split[s] = "val"
        for s in test:
            site_split[s] = "test"
        logger.info("PFT %-12s -> train=%d val=%d test=%d", pft, len(train), len(val), len(test))
    return site_split


def build_feature_frame(m: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Add cyclical time, PFT one-hot, and per-site climatological means; return ordered cols."""
    cols = cfg["columns"]
    targets = cfg["targets"]["fluxes"] + cfg["targets"]["pools"]

    m = m.copy()
    m["month_sin"] = np.sin(2 * np.pi * m["month"] / 12)
    m["month_cos"] = np.cos(2 * np.pi * m["month"] / 12)
    for pft in cols["pft_order"]:
        m[pft] = (m["PFT"] == pft).astype(float)
    clim_mean_cols = [f"mean_{c}" for c in cols["site_clim_cols"]]
    m[clim_mean_cols] = m.groupby("site")[cols["site_clim_cols"]].transform("mean")

    feature_cols = (
        cols["dynamic"] + cols["static"] + cols["pft_order"] + ["month_sin", "month_cos"] + clim_mean_cols
    )
    all_cols = feature_cols + targets
    return m, all_cols


def fit_scaler(m: pd.DataFrame, all_cols: list[str]) -> dict[str, np.ndarray]:
    """Column-wise mean/std over train rows only; std=1 where constant."""
    train_rows = m[m["split"] == "train"]
    mean = train_rows[all_cols].mean().to_numpy(dtype=float, copy=True)
    std = train_rows[all_cols].std(ddof=0).to_numpy(dtype=float, copy=True)
    mean[~np.isfinite(mean)] = 0.0
    std[(std == 0) | ~np.isfinite(std)] = 1.0
    logger.info("Scaler fit on %d train rows (%d columns)", len(train_rows), len(all_cols))
    return {"mean": mean, "std": std}


def build_segments(m: pd.DataFrame, all_cols: list[str], scaler: dict, seq_len: int) -> dict[str, list[dict]]:
    """Per site: normalise, split into contiguous monthly runs, keep runs >= seq_len."""
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    mean, std = scaler["mean"], scaler["std"]
    for (site, pft), grp in m.groupby(["site", "PFT"]):
        grp = grp.sort_values("ym_ord")
        data = ((grp[all_cols].to_numpy() - mean) / std).astype(np.float32)
        ords = grp["ym_ord"].to_numpy()
        breaks = np.where(np.diff(ords) != 1)[0] + 1
        segments, starts = [], []
        for seg, ord_piece in zip(np.split(data, breaks), np.split(ords, breaks)):
            if seg.shape[0] >= seq_len:
                segments.append(seg)
                year, month0 = divmod(int(ord_piece[0]), 12)
                starts.append((year, month0 + 1))
        if segments:
            splits[grp["split"].iloc[0]].append(
                {"site": site, "pft": pft, "segments": segments, "segment_starts": starts}
            )
    return splits


def main() -> None:
    cfg = load_config("rangeland_domain")
    seq_len = cfg["preprocessing"]["seq_len"]

    df = load_and_merge(cfg)
    m = monthly_aggregate(df, cfg)
    site_split = split_sites(m, cfg)
    m, all_cols = build_feature_frame(m, cfg)
    m["split"] = m["site"].map(site_split)

    # Predictors are fully observed (EDA); fail loud rather than silently feed NaN to the model.
    n_targets = len(cfg["targets"]["fluxes"]) + len(cfg["targets"]["pools"])
    bad = [c for c in all_cols[:-n_targets] if m[c].isna().any()]
    if bad:
        raise ValueError(f"Unexpected NaN in Rangeland feature columns {bad}; predictors must be fully observed.")

    scaler = fit_scaler(m, all_cols)
    splits = build_segments(m, all_cols, scaler, seq_len)

    out_dir = Path(cfg["paths"]["preprocessed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        n_seg = sum(len(r["segments"]) for r in records)
        logger.info("%s: %d sites, %d segments", name, len(records), n_seg)
        with (out_dir / f"{name}.pkl").open("wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    scaler_path = Path(cfg["paths"]["scaler"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved scaler (%d columns) to %s", len(all_cols), scaler_path)


if __name__ == "__main__":
    main()
