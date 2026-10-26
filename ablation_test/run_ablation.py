"""Runs the ablation study's training runs (see ablation_description.md), sequentially and fail-fast.

The full 3-domain anchor is not run here: it is the production seed sweep; rerunning would overwrite it.
Run on vm-sandeep (GPU); this script does not start/stop the VM.

Usage:
  python ablation_test/run_ablation.py --seeds 1                          # all runs, seed 1
  python ablation_test/run_ablation.py --seeds 2 3 4 5                    # remaining seeds
  python ablation_test/run_ablation.py --seeds 1 2 3 4 5 --runs pairwise_arctic_amazon_finetune pairwise_arctic_rangeland_finetune pairwise_amazon_rangeland_finetune
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "outputs" / "_ablation_logs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_stage(cmd_args: list[str], log_path: Path) -> None:
    cmd = [sys.executable, *cmd_args]
    logger.info("Running: %s", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    elapsed = time.time() - t0
    logger.info("Done in %.1fs: %s", elapsed, " ".join(cmd))
    with log_path.open("a") as f:
        f.write(f"{elapsed:.1f}s  {' '.join(cmd)}\n")


def run_capacity_amazon(seed: int, log_path: Path) -> None:
    # No --flux-only: Amazon has no flux/pool distinction, so its scripts lack the flag.
    d = REPO_ROOT / "domains" / "amazon_domain"
    run_stage([str(d / "02_train.py"), "--seed", str(seed), "--capacity-matched"], log_path)
    run_stage([str(d / "03_predict.py"), "--seed", str(seed), "--capacity-matched"], log_path)
    run_stage([str(d / "04_evaluate.py"), "--seed", str(seed), "--capacity-matched"], log_path)


def run_capacity_rangeland(seed: int, log_path: Path) -> None:
    d = REPO_ROOT / "domains" / "rangeland_domain"
    run_stage([str(d / "02_train.py"), "--flux-only", "--seed", str(seed), "--capacity-matched"], log_path)
    run_stage([str(d / "03_predict.py"), "--flux-only", "--seed", str(seed), "--capacity-matched"], log_path)
    run_stage([str(d / "04_evaluate.py"), "--flux-only", "--seed", str(seed), "--capacity-matched"], log_path)


def _run_multi_domain(stage: str, domains: str, seed: int, log_path: Path) -> None:
    """Train one stage on a comma-separated domain subset, then evaluate. Finetune needs the matching pretrain."""
    d = REPO_ROOT / "domains" / "multi_domain"
    run_stage([str(d / "02_train.py"), "--stage", stage, "--flux-only",
              "--seed", str(seed), "--domains", domains], log_path)
    run_stage([str(d / "04_evaluate.py"), "--flux-only", "--seed", str(seed),
              "--domains", domains], log_path)


def run_pairwise_arctic_amazon(seed: int, log_path: Path) -> None:
    _run_multi_domain("pretrain", "arctic,amazon", seed, log_path)


def run_pairwise_arctic_rangeland(seed: int, log_path: Path) -> None:
    _run_multi_domain("pretrain", "arctic,rangeland", seed, log_path)


def run_pairwise_amazon_rangeland(seed: int, log_path: Path) -> None:
    _run_multi_domain("pretrain", "amazon,rangeland", seed, log_path)


def run_pairwise_arctic_amazon_finetune(seed: int, log_path: Path) -> None:
    _run_multi_domain("finetune", "arctic,amazon", seed, log_path)


def run_pairwise_arctic_rangeland_finetune(seed: int, log_path: Path) -> None:
    _run_multi_domain("finetune", "arctic,rangeland", seed, log_path)


def run_pairwise_amazon_rangeland_finetune(seed: int, log_path: Path) -> None:
    _run_multi_domain("finetune", "amazon,rangeland", seed, log_path)


RUNS = {
    "capacity_amazon": run_capacity_amazon,
    "capacity_rangeland": run_capacity_rangeland,
    "pairwise_arctic_amazon": run_pairwise_arctic_amazon,
    "pairwise_arctic_rangeland": run_pairwise_arctic_rangeland,
    "pairwise_amazon_rangeland": run_pairwise_amazon_rangeland,
    "pairwise_arctic_amazon_finetune": run_pairwise_arctic_amazon_finetune,
    "pairwise_arctic_rangeland_finetune": run_pairwise_arctic_rangeland_finetune,
    "pairwise_amazon_rangeland_finetune": run_pairwise_amazon_rangeland_finetune,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="Which seeds to run (e.g. --seeds 2 3 4 5).")
    parser.add_argument("--runs", choices=list(RUNS), nargs="+", default=list(RUNS),
                        help=f"Which of the {len(RUNS)} ablation run types to execute (default: all).")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"ablation_{time.strftime('%Y%m%dT%H%M%S')}.log"
    logger.info("Ablation log: %s", log_path)

    for seed in args.seeds:
        for run_name in args.runs:
            logger.info("=== seed=%d run=%s ===", seed, run_name)
            RUNS[run_name](seed, log_path)


if __name__ == "__main__":
    main()
