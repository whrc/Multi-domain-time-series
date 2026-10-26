"""Average the ablation runs' metrics across seeds 1-5 into *_seedavg CSVs.

Individual and full 3-domain arms already have production seedavg outputs and are not aggregated here.
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

# (pair-suffix, [(domain, id_cols), ...])
PAIRS = [
    ("amazon-arctic", [("amazon", AMAZON_ID_COLS), ("arctic", ARCTIC_ID_COLS)]),
    ("arctic-rangeland", [("arctic", ARCTIC_ID_COLS), ("rangeland", RANGELAND_ID_COLS)]),
    ("amazon-rangeland", [("amazon", AMAZON_ID_COLS), ("rangeland", RANGELAND_ID_COLS)]),
]


# These runs' CSVs lack r/alpha/beta, so pin the original 4 metrics.
METRIC_COLUMNS = ("RMSE", "NSE", "KGE", "PBIAS")


def write(paths: dict[int, Path], id_cols: list[str], out_path: Path) -> None:
    out = aggregate_seed_metrics(paths, id_cols, metric_columns=METRIC_COLUMNS)
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

    # Pairwise multi-domain runs
    for stage in ("pretrained", "finetuned"):
        for pair, domains in PAIRS:
            for domain, id_cols in domains:
                paths = {s: EVAL_ROOT / f"{stage}_fluxonly_dom-{pair}_seed{s}" / domain / f"{domain}_metrics.csv"
                         for s in SEEDS}
                out_path = EVAL_ROOT / f"{stage}_fluxonly_dom-{pair}_seedavg" / domain / f"{domain}_metrics_seedavg.csv"
                write(paths, id_cols, out_path)


if __name__ == "__main__":
    main()
