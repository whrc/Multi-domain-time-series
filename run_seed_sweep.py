"""
Multi-seed publication sweep: Arctic, Rangeland and multi-domain (all flux-only), plus Amazon.
Calls the numbered scripts directly (run_*.py don't forward --seed/--flux-only). Runs
sequentially, fail-fast. Run on vm-sandeep (does not start/stop the VM).

Usage:
  python run_seed_sweep.py --seeds 1                     # validation pass, all pipelines
  python run_seed_sweep.py --seeds 2 3 4 5                # remaining seeds
  python run_seed_sweep.py --seeds 1 2 3 4 5 --pipelines arctic rangeland
  python run_seed_sweep.py --aggregate --seeds 1 2 3 4 5  # aggregate existing outputs
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "outputs" / "_seed_sweep_logs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_PIPELINES = ["arctic", "rangeland", "amazon", "multi_domain"]

# Production Arctic label; must be passed explicitly (scripts otherwise use the config default).
ARCTIC_LABEL = "500K_s400"

# Id columns of each pipeline's metrics CSV (not flux_only.py's DOMAIN_ID_FIELDS).
ARCTIC_ID_COLS = ["grid", "y", "x", "lat", "lon", "ssp", "target", "period"]
RANGELAND_ID_COLS = ["site", "pft", "target"]
AMAZON_ID_COLS = ["station_id", "target"]


def run_stage(cmd_args: list[str], log_path: Path) -> None:
    cmd = [sys.executable, *cmd_args]
    logger.info("Running: %s", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    elapsed = time.time() - t0
    logger.info("Done in %.1fs: %s", elapsed, " ".join(cmd))
    with log_path.open("a") as f:
        f.write(f"{elapsed:.1f}s  {' '.join(cmd)}\n")


def run_arctic(seed: int, log_path: Path) -> None:
    d = REPO_ROOT / "domains" / "arctic_domain"
    run_stage([str(d / "02_train.py"), "--label", ARCTIC_LABEL, "--flux-only", "--seed", str(seed)], log_path)
    # 03_predict skipped (dense NetCDF, hundreds of GB); 04_evaluate writes a prediction sample.
    run_stage([str(d / "04_evaluate.py"), "--label", ARCTIC_LABEL, "--flux-only", "--seed", str(seed)], log_path)


def run_rangeland(seed: int, log_path: Path) -> None:
    d = REPO_ROOT / "domains" / "rangeland_domain"
    run_stage([str(d / "02_train.py"), "--flux-only", "--seed", str(seed)], log_path)
    run_stage([str(d / "03_predict.py"), "--flux-only", "--seed", str(seed)], log_path)
    run_stage([str(d / "04_evaluate.py"), "--flux-only", "--seed", str(seed)], log_path)


def run_amazon(seed: int, log_path: Path) -> None:
    d = REPO_ROOT / "domains" / "amazon_domain"
    run_stage([str(d / "02_train.py"), "--seed", str(seed)], log_path)
    run_stage([str(d / "03_predict.py"), "--seed", str(seed)], log_path)
    run_stage([str(d / "04_evaluate.py"), "--seed", str(seed)], log_path)


def run_multi_domain(seed: int, log_path: Path) -> None:
    d = REPO_ROOT / "domains" / "multi_domain"
    run_stage([str(d / "02_train.py"), "--stage", "pretrain", "--flux-only", "--seed", str(seed)], log_path)
    run_stage([str(d / "02_train.py"), "--stage", "finetune", "--flux-only", "--seed", str(seed)], log_path)
    # Arctic predict skipped (disk size).
    run_stage([str(d / "03_predict.py"), "--domain", "amazon", "--checkpoint", "finetuned",
              "--flux-only", "--seed", str(seed)], log_path)
    run_stage([str(d / "03_predict.py"), "--domain", "rangeland", "--checkpoint", "finetuned",
              "--flux-only", "--seed", str(seed)], log_path)
    # Recomputes predictions from checkpoints; does not read 03_predict outputs.
    run_stage([str(d / "04_evaluate.py"), "--flux-only", "--seed", str(seed)], log_path)


PIPELINE_RUNNERS = {
    "arctic": run_arctic,
    "rangeland": run_rangeland,
    "amazon": run_amazon,
    "multi_domain": run_multi_domain,
}


def aggregate(seeds: list[int], pipelines: list[str]) -> None:
    from shared.seed_aggregation import aggregate_seed_metrics

    def write(paths: dict[int, Path], id_cols: list[str], out_path: Path,
              passthrough: tuple[str, ...] = ()) -> None:
        out = aggregate_seed_metrics(paths, id_cols, passthrough_columns=passthrough)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        logger.info("Wrote %s (%d rows, n_seeds=%d)", out_path, len(out), len(seeds))

    if "arctic" in pipelines:
        eval_root = REPO_ROOT / "outputs" / "arctic_domain" / "evaluation"
        paths = {s: eval_root / f"{ARCTIC_LABEL}_fluxonly_seed{s}" / "metrics_test.csv" for s in seeds}
        write(paths, ARCTIC_ID_COLS, eval_root / f"{ARCTIC_LABEL}_fluxonly_seedavg" / "metrics_test_seedavg.csv",
              passthrough=("obs_degenerate",))

    if "rangeland" in pipelines:
        root = REPO_ROOT / "outputs" / "rangeland_domain"
        paths = {s: root / f"evaluation_fluxonly_seed{s}" / "metrics_test.csv" for s in seeds}
        write(paths, RANGELAND_ID_COLS, root / "evaluation_fluxonly_seedavg" / "metrics_test_seedavg.csv")

    if "amazon" in pipelines:
        root = REPO_ROOT / "outputs" / "amazon_domain"
        paths = {s: root / f"evaluation_seed{s}" / "metrics_test.csv" for s in seeds}
        write(paths, AMAZON_ID_COLS, root / "evaluation_seedavg" / "metrics_test_seedavg.csv")

    if "multi_domain" in pipelines:
        eval_root = REPO_ROOT / "outputs" / "multi_domain" / "evaluation"
        domain_id_cols = {"arctic": ARCTIC_ID_COLS, "amazon": AMAZON_ID_COLS, "rangeland": RANGELAND_ID_COLS}
        for stage in ("pretrained_fluxonly", "finetuned_fluxonly"):
            for domain, id_cols in domain_id_cols.items():
                paths = {s: eval_root / f"{stage}_seed{s}" / domain / f"{domain}_metrics.csv" for s in seeds}
                write(paths, id_cols, eval_root / f"{stage}_seedavg" / domain / f"{domain}_metrics_seedavg.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--pipelines", choices=ALL_PIPELINES, nargs="+", default=ALL_PIPELINES)
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate existing per-seed outputs into a seedavg summary — "
                             "no retraining.")
    args = parser.parse_args()

    if args.aggregate:
        sys.path.insert(0, str(REPO_ROOT))
        aggregate(args.seeds, args.pipelines)
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"sweep_{time.strftime('%Y%m%dT%H%M%S')}.log"
    logger.info("Sweep log: %s", log_path)

    for seed in args.seeds:
        for pipeline in args.pipelines:
            logger.info("=== seed=%d pipeline=%s ===", seed, pipeline)
            PIPELINE_RUNNERS[pipeline](seed, log_path)


if __name__ == "__main__":
    main()
