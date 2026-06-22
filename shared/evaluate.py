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
