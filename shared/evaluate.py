"""Evaluation helpers for 02_train.py and 04_evaluate.py: dense inference in physical units, per-unit metrics."""

from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from shared.dataset import WindowedDataset, records_to_segments
from shared.inference import predict_last_position
from shared.metrics import compute_metrics


def predict_and_inverse(
    model: torch.nn.Module,
    records: list[dict],
    num_targets: int,
    seq_len: int,
    device: torch.device,
    scaler: dict,
) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    """Return parallel ``(seg_meta, pred_list, obs_list)``; arrays are ``(T_seg, num_targets)`` in original units."""
    segments, seg_meta = records_to_segments(records)
    dataset = WindowedDataset(segments, seg_meta, num_targets, seq_len, stride=1)
    preds = predict_last_position(model, dataset, device)
    mean_t = scaler["mean"][-num_targets:]
    std_t = scaler["std"][-num_targets:]
    pred_list = [p * std_t + mean_t for p in preds]
    obs_list = [s[:, -num_targets:] * std_t + mean_t for s in segments]
    return seg_meta, pred_list, obs_list


def stack_by_target(
    pred_list: list[np.ndarray],
    obs_list: list[np.ndarray],
    target_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Concatenate all segments into ``{target: 1-D array}`` dicts, dropping unpredicted rows."""
    if not pred_list:
        empty = {n: np.array([]) for n in target_names}
        return empty, dict(empty)
    P = np.concatenate(pred_list, axis=0)
    O = np.concatenate(obs_list, axis=0)
    pred_d, obs_d = {}, {}
    for i, name in enumerate(target_names):
        m = ~np.isnan(P[:, i])
        pred_d[name] = P[m, i]
        obs_d[name] = O[m, i]
    return pred_d, obs_d


def per_unit_metrics(
    seg_meta: list[dict],
    pred_list: list[np.ndarray],
    obs_list: list[np.ndarray],
    target_names: list[str],
    id_fields: list[str],
) -> pd.DataFrame:
    """One row per (unit, target), a unit being the ``id_fields`` tuple; a unit's segments are pooled."""
    buckets: dict[tuple, dict[str, list]] = defaultdict(lambda: {"pred": [], "obs": []})
    for meta, p, o in zip(seg_meta, pred_list, obs_list):
        key = tuple(meta[f] for f in id_fields)
        buckets[key]["pred"].append(p)
        buckets[key]["obs"].append(o)
    rows = []
    for key, d in buckets.items():
        P = np.concatenate(d["pred"], axis=0)
        O = np.concatenate(d["obs"], axis=0)
        for i, name in enumerate(target_names):
            row = dict(zip(id_fields, key))
            row["target"] = name
            row.update(compute_metrics(P[:, i], O[:, i]))
            rows.append(row)
    return pd.DataFrame(rows)


def scenario_period_label(ssp: str, period: str) -> str:
    """'historical', 'projected_ssp126' or 'projected_ssp585'."""
    if period == "historical":
        return "historical"
    return f"projected_{'ssp126' if 'ssp1' in ssp else 'ssp585'}"


def metrics_df_by_period(
    seg_meta: list[dict],
    pred_list: list[np.ndarray],
    obs_list: list[np.ndarray],
    target_names: list[str],
    yearly_targets: set[str],
    idx_map: dict[str, "pd.DatetimeIndex"],
    proj_start: int,
    id_fields: list[str],
) -> pd.DataFrame:
    """Arctic: one row per (unit, target, period), split at proj_start; yearly targets use January only.

    ``obs_degenerate`` flags constant-obs windows, where NSE/KGE explode; callers should
    drop those rows before plotting.
    """
    rows = []
    for meta, pred, obs in zip(seg_meta, pred_list, obs_list):
        time = idx_map["ssp1" if "ssp1" in meta["ssp"] else "ssp5"]
        periods = (("historical", time.year < proj_start), ("projected", time.year >= proj_start))
        for i, name in enumerate(target_names):
            pos = (time.month == 1) if name in yearly_targets else np.ones(len(time), dtype=bool)
            for period, in_period in periods:
                sel = pos & in_period
                if not sel.any():
                    continue
                row = {f: meta[f] for f in id_fields}
                row["target"] = name
                row["period"] = period
                row.update(compute_metrics(pred[sel, i], obs[sel, i]))
                o_valid = obs[sel, i]
                o_valid = o_valid[~np.isnan(o_valid)]
                row["obs_degenerate"] = bool(o_valid.size >= 2 and np.allclose(o_valid, o_valid[0]))
                rows.append(row)
    return pd.DataFrame(rows)
