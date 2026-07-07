"""
Arctic domain — Temporal holdout diagnostic (OOS-time).

Complements the default spatial split (OOS-space: held-out PIXELS, entirely unseen
locations, produced by 01-04). Here the SAME pixels are used for both train and test,
split along TIME instead: train on each pixel's early historical years, evaluate on
that same pixel's later held-out years. Tests whether the model generalizes well when
it has implicit continuity for a location, vs the spatial-holdout case where a pixel's
own history is never seen at all — directly probes the "missing autoregressive state"
hypothesis from project_management/key_findings_log.md (AR-21c64242).

Historical-only (ssp1, pre-2025) to avoid conflating with SSP-scenario extrapolation, a
different generalization axis. Reuses pixels already present in train_50K.pkl + val.pkl
+ test.pkl — no new GCS fetch needed, since all three already hold each pixel's full,
un-windowed, scaled time series (spatial role doesn't matter here, only the data).

Time split of the 1901-2024 historical range: train 1901-1994, val 1995-2004,
test 2005-2024. Train/val windows use the same stride as spatial preprocessing
(capped_stride); test uses dense stride=1, matching the spatial pipeline's evaluation.
All outputs are labeled 50K_OOS-time_seq12 (split type + seq_len both explicit) to stay
clearly distinct from the default spatial-split ("OOS-space") outputs.
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.dataset import WindowedDataset  # noqa: E402
from shared.evaluate import per_unit_metrics, predict_and_inverse, stack_by_target  # noqa: E402
from shared.plots import plot_loss_curves, plot_metric_boxplot, plot_pred_vs_true  # noqa: E402
from shared.training import run_lr_finder, train_model  # noqa: E402
from shared.transformer import TransformerModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_TARGETS = 4
LABEL = "50K_OOS-time_seq12"  # matches the seq_len this experiment is run with — keep this
# suffix in sync if seq_len changes; naming always states both the split type (OOS-space vs
# OOS-time) and seq_len so results never need external context to identify what they are.
TRAIN_END_YEAR = 1995   # train: 1901-01 .. 1994-12
VAL_END_YEAR = 2005     # val:   1995-01 .. 2004-12
# test: 2005-01 .. 2024-12 (end of historical range)


def load_pixel_pool(pre_dir: Path) -> list[dict]:
    """Combine ssp1 (historical-bearing) records from train/val/test pkls — one entry
    per pixel, each already holding its full un-windowed, scaled time series. ssp5
    records are skipped: they only cover 2025-2100, no historical range to split on."""
    seen = set()
    records = []
    for name in ("train_50K.pkl", "val.pkl", "test.pkl"):
        with (pre_dir / name).open("rb") as f:
            for r in pickle.load(f):
                if "ssp1" not in r["ssp"]:
                    continue
                k = (r["grid"], r["y"], r["x"])
                if k in seen:
                    continue
                seen.add(k)
                records.append(r)
    return records


def split_segments(
    records: list[dict], monthly_index: pd.DatetimeIndex, seq_len: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict]]:
    """Slice each pixel's historical segment into train/val/test sub-segments by time.

    val/test sub-segments borrow seq_len-1 months of lookback from the preceding split
    so their first window still has a full causal input — those borrowed months are
    already-seen (by train) data, not new information, matching how the spatial
    pipeline's own dense stride-1 evaluation always has full-length input context.
    """
    train_end = monthly_index.get_indexer([f"{TRAIN_END_YEAR}-01-01"])[0]
    val_end = monthly_index.get_indexer([f"{VAL_END_YEAR}-01-01"])[0]
    hist_end = int((monthly_index.year < 2025).sum())
    lb = seq_len - 1

    train_segs, val_segs, test_segs, meta = [], [], [], []
    for r in records:
        d = r["data"][:hist_end]
        train_segs.append(d[:train_end])
        val_segs.append(d[train_end - lb: val_end])
        test_segs.append(d[val_end - lb: hist_end])
        meta.append({k: r[k] for k in ("grid", "y", "x", "lat", "lon")})
    return train_segs, val_segs, test_segs, meta


def flag_degenerate(seg_meta, obs_list, target_names, id_fields) -> pd.DataFrame:
    """Per (unit, target): True if the observed test-period values are ~constant.

    Mirrors metrics_df_by_period's obs_degenerate handling (see shared/evaluate.py) —
    NSE/KGE divide by observed variance, so a near-constant window (common for the
    yearly pool targets even within a single 20-year test slice) makes them blow up to
    arbitrarily large negative numbers rather than being genuinely wrong.
    """
    rows = []
    for meta, obs in zip(seg_meta, obs_list):
        row = {f: meta[f] for f in id_fields}
        for i, name in enumerate(target_names):
            o = obs[:, i]
            o = o[~np.isnan(o)]
            rows.append({**row, "target": name,
                        "obs_degenerate": bool(o.size >= 2 and np.allclose(o, o[0]))})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config("arctic_domain")
    pp = cfg["preprocessing"]
    tcfg = cfg["training"]
    seq_len = pp["seq_len"]
    stride = pp["capped_stride"]
    target_names = [t["name"] for t in cfg["targets"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pre_dir = Path(cfg["paths"]["preprocessed_dir"])
    records = load_pixel_pool(pre_dir)
    logger.info("Loaded %d unique ssp1 pixels for temporal-holdout split", len(records))

    monthly_index = pd.date_range(cfg["time"]["scenarios"]["ssp1"]["start"],
                                   cfg["time"]["scenarios"]["ssp1"]["end"], freq="MS")
    train_segs, val_segs, test_segs, meta = split_segments(records, monthly_index, seq_len)

    num_features = records[0]["data"].shape[1] - NUM_TARGETS
    logger.info("Device: %s | features=%d targets=%d | seq_len=%d", device, num_features, NUM_TARGETS, seq_len)

    train_ds = WindowedDataset(train_segs, meta, NUM_TARGETS, seq_len, stride)
    val_ds = WindowedDataset(val_segs, meta, NUM_TARGETS, seq_len, stride)
    logger.info("Train windows: %d | Val windows: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                               num_workers=tcfg["num_workers"], pin_memory=(device.type == "cuda"),
                               persistent_workers=tcfg["num_workers"] > 0)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                             num_workers=tcfg["num_workers"], pin_memory=(device.type == "cuda"),
                             persistent_workers=tcfg["num_workers"] > 0)

    model = TransformerModel(num_features, NUM_TARGETS, cfg).to(device)
    models_dir = Path(cfg["paths"]["best_model"]).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / f"best_model_{LABEL}.pt"
    eval_dir = Path(cfg["paths"]["evaluation"]) / LABEL
    eval_dir.mkdir(parents=True, exist_ok=True)

    if tcfg["optimized_lr"] is not None:
        lr = float(tcfg["optimized_lr"])
        logger.info("Using configured optimized_lr=%.3e", lr)
    else:
        lr = run_lr_finder(model, train_loader, float(tcfg["initial_lr"]), device, eval_dir / "lr_finder.png")

    history = train_model(model, train_loader, val_loader, cfg, target_names, lr, device,
                           best_model_path, num_features)
    logger.info("Best val loss %.4f at epoch %d", history["best_val_loss"], history["best_epoch"])

    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    with Path(cfg["paths"]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)

    plot_loss_curves(history["train_loss"], history["val_loss"], history["per_target_val"],
                     eval_every=tcfg["eval_every_n_epochs"], save_path=eval_dir / "loss_curves.png")

    # Dense test-time evaluation: wrap each test sub-segment as a "record" (`data` +
    # identifying metadata) so predict_and_inverse — built for the spatial pipeline's
    # record shape — runs unchanged here for stride=1 inference.
    test_records = [{**m, "data": seg} for m, seg in zip(meta, test_segs)]
    seg_meta, pred_list, obs_list = predict_and_inverse(model, test_records, NUM_TARGETS, seq_len, device, scaler)
    pred_d, obs_d = stack_by_target(pred_list, obs_list, target_names)
    plot_pred_vs_true(pred_d, obs_d, log_scale=False, save_path=eval_dir / "pred_vs_true.png")

    id_fields = ["grid", "y", "x", "lat", "lon"]
    # Merge on full-precision id_fields (lat/lon included) before rounding anything - rounding
    # first would make metrics_df's lat/lon no longer match degenerate's unrounded values,
    # silently dropping every row out of the merge (obs_degenerate all-NaN -> float dtype ->
    # `~` fails below).
    metrics_df = per_unit_metrics(seg_meta, pred_list, obs_list, target_names, id_fields=id_fields)
    degenerate = flag_degenerate(seg_meta, obs_list, target_names, id_fields)
    metrics_df = metrics_df.merge(degenerate, on=[*id_fields, "target"], how="left")
    metric_cols = ["RMSE", "NSE", "KGE", "PBIAS"]
    metrics_df[metric_cols] = metrics_df[metric_cols].round(3)
    metrics_df.to_csv(eval_dir / "metrics.csv", index=False)
    n_degenerate = int(metrics_df["obs_degenerate"].sum())
    logger.info("Saved %d metric rows (%d pixels); %d degenerate (constant-observed)",
                len(metrics_df), len(test_records), n_degenerate)

    plot_df = metrics_df[~metrics_df["obs_degenerate"]]
    plot_metric_boxplot(plot_df, group_col=None, title="OOS-time test metrics (2005-2024 held out)",
                        save_path=eval_dir / "metrics_boxplot.png")

    logger.info("Saved OOS-time evaluation to %s", eval_dir)


if __name__ == "__main__":
    main()
