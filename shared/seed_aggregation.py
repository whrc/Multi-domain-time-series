"""Aggregate per-seed metrics CSVs into a mean/std summary (used by scripts/run_seed_sweep.py --aggregate)."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_seed_metrics(
    metrics_paths: dict[int, Path],
    id_columns: list[str],
    metric_columns: tuple[str, ...] = ("RMSE", "NSE", "KGE", "PBIAS", "r", "alpha", "beta"),
    passthrough_columns: tuple[str, ...] = (),
    round_dp: int = 3,
) -> pd.DataFrame:
    """One row per id_columns combination with {metric}_mean/_std columns and n_seeds.

    Every seed must cover the same id combinations; passthrough_columns must be identical
    across seeds within a group. Both are checked and raise ValueError.
    """
    frames = []
    for seed, path in metrics_paths.items():
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby(id_columns)

    expected_seeds = frozenset(metrics_paths)
    seed_sets = grouped["seed"].agg(lambda s: frozenset(s))
    missing = seed_sets[seed_sets != expected_seeds]
    if not missing.empty:
        raise ValueError(
            f"{len(missing)} id_columns combination(s) are missing from at least one seed's "
            f"metrics file (expected seeds {sorted(expected_seeds)}) — first few: "
            f"{missing.index[:5].tolist()}"
        )

    agg = grouped[list(metric_columns)].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg["n_seeds"] = grouped.size()
    agg = agg.reset_index()

    for col in passthrough_columns:
        if not (grouped[col].nunique() == 1).all():
            raise ValueError(f"passthrough column {col!r} is not identical across seeds within a group")
        first = grouped[col].first().reset_index()
        agg = agg.merge(first, on=id_columns)

    return agg.round(round_dp)
