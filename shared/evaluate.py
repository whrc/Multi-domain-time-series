"""
Shared evaluation helpers — used by 02_train.py (validation figures) and 04_evaluate.py.

Given preprocessed records, a trained model, and the scaler, these run dense (stride-1)
inference, inverse-transform back to physical units, and assemble either flat
prediction/observation arrays (for scatter plots) or a per-unit metrics table. Grouping
keys are passed in, so the same code serves every domain (and the multi-domain model).
"""

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
    """Per-segment predictions and observations in original units.

    Returns parallel lists ``(seg_meta, pred_list, obs_list)``; each array is
    ``(T_seg, num_targets)`` with NaN at positions the causal model could not predict.
    """
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
    """One row per (unit, target). A unit is the tuple of ``id_fields`` from the metadata.

    Segments sharing a unit (e.g. a site's multiple contiguous runs) are pooled before
    metrics are computed.
    """
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
    """Combine ssp + period into one label for a single combined boxplot: 'historical'
    (ssp1 only has a historical period) or 'projected_ssp126'/'projected_ssp585'."""
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
    """One row per (unit, target, period). Arctic-specific: splits historical/projected at
    proj_start and restricts yearly targets (e.g. ALD, VEGC) to January positions, unlike
    per_unit_metrics which pools a whole segment's time series into a single row. Shared by
    val (02_train.py) and test (04_evaluate.py) so both report identical, comparable metrics.

    Also flags ``obs_degenerate``: True when the observed values in that (unit, target,
    period) window are constant (within float tolerance). NSE/KGE divide by observed
    variance, so a constant window makes them mathematically undefined - not wrong, but any
    tiny prediction error blows up to an arbitrarily large negative number. That's a real,
    physically-occurring case (e.g. a pixel where TEM's active layer depth genuinely doesn't
    change year to year), not a data or model bug, so the raw (huge-magnitude) metric values
    are still computed and kept here for transparency; callers should exclude
    obs_degenerate rows before aggregating for plots, where these outliers would otherwise
    swamp the axis scale and hide the real, well-defined values.
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
