"""
Aggregate per-seed metrics CSVs into a mean/std summary across seeds.

Used by scripts/run_seed_sweep.py --aggregate once all seeds' 02_train.py/03_predict.py/
04_evaluate.py runs are done. Domain-agnostic — grouping/metric columns are passed in by the
caller (see ARCTIC_ID_COLS/RANGELAND_ID_COLS/AMAZON_ID_COLS in scripts/run_seed_sweep.py,
matched to each pipeline's actual metrics_test.csv columns — these are NOT the same as
domains/multi_domain/flux_only.py's DOMAIN_ID_FIELDS, which serves a different purpose there
and lacks columns like lat/lon/period/target that the metrics CSVs on disk always carry).
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_seed_metrics(
    metrics_paths: dict[int, Path],
    id_columns: list[str],
    metric_columns: tuple[str, ...] = ("RMSE", "NSE", "KGE", "PBIAS"),
    passthrough_columns: tuple[str, ...] = (),
    round_dp: int = 3,
) -> pd.DataFrame:
    """One row per unique id_columns combination, with mean/std of metric_columns across seeds.

    metrics_paths: {seed: path to that seed's metrics CSV}. Every seed must cover the exact
    same set of id_columns combinations (true by construction — the train/val/test split is
    seed-independent) — fails loudly if any combination is missing from any seed's file,
    since a partial average would misrepresent that unit/target's 5-seed mean.

    Returns id_columns + passthrough_columns (must be identical across seeds for a given
    id_columns combination — asserted, then copied through once) + one f"{metric}_mean" and
    f"{metric}_std" column per metric_columns, plus n_seeds, all rounded to round_dp.
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
