"""
Aggregates the ablation study's 5 new run types across seeds 1-5 into *_seedavg CSVs, using
shared/seed_aggregation.py::aggregate_seed_metrics -- the same tool the production 5-seed sweep
uses (run_seed_sweep.py --aggregate). The "Individual" and "Full 3-domain" arms are NOT
aggregated here -- their 5-seed averages already exist as production artifacts
(outputs/{amazon,rangeland,arctic}_domain's own seedavg dirs;
outputs/multi_domain/evaluation/pretrained_fluxonly_seedavg/) -- see
ablation_test/ablation_description.md.

Usage:
  python ablation_test/aggregate_ablation_seeds.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from shared.seed_aggregation import aggregate_seed_metrics  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
ARCTIC_ID_COLS = ["grid", "y", "x", "lat", "lon", "ssp", "target", "period"]
RANGELAND_ID_COLS = ["site", "pft", "target"]
AMAZON_ID_COLS = ["station_id", "target"]

EVAL_ROOT = REPO_ROOT / "outputs" / "multi_domain" / "evaluation"

# (pair-suffix, [(domain, id_cols), ...]) for the three pairwise ablation runs
PAIRS = [
    ("amazon-arctic", [("amazon", AMAZON_ID_COLS), ("arctic", ARCTIC_ID_COLS)]),
    ("arctic-rangeland", [("arctic", ARCTIC_ID_COLS), ("rangeland", RANGELAND_ID_COLS)]),
    ("amazon-rangeland", [("amazon", AMAZON_ID_COLS), ("rangeland", RANGELAND_ID_COLS)]),
]


def write(paths: dict[int, Path], id_cols: list[str], out_path: Path) -> None:
    out = aggregate_seed_metrics(paths, id_cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows, n_seeds={len(SEEDS)})")


def main() -> None:
    # Capacity-matched controls
    paths = {s: REPO_ROOT / f"outputs/amazon_domain/evaluation_seed{s}_capmatched/metrics_test.csv" for s in SEEDS}
    write(paths, AMAZON_ID_COLS,
         REPO_ROOT / "outputs/amazon_domain/evaluation_seedavg_capmatched/metrics_test_seedavg.csv")

    paths = {s: REPO_ROOT / f"outputs/rangeland_domain/evaluation_fluxonly_seed{s}_capmatched/metrics_test.csv" for s in SEEDS}
    write(paths, RANGELAND_ID_COLS,
         REPO_ROOT / "outputs/rangeland_domain/evaluation_fluxonly_seedavg_capmatched/metrics_test_seedavg.csv")

    # Pairwise multi-domain pretrain runs
    for pair, domains in PAIRS:
        for domain, id_cols in domains:
            paths = {s: EVAL_ROOT / f"pretrained_fluxonly_dom-{pair}_seed{s}" / domain / f"{domain}_metrics.csv"
                     for s in SEEDS}
            out_path = EVAL_ROOT / f"pretrained_fluxonly_dom-{pair}_seedavg" / domain / f"{domain}_metrics_seedavg.csv"
            write(paths, id_cols, out_path)


if __name__ == "__main__":
    main()
