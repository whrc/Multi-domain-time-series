"""
Orchestrates the ablation study's training-run types — see ablation_description.md for the
hypotheses and full experiment design: capacity-matched individual controls for Amazon and
Rangeland (retired, kept for historical reference), and three pairwise multi-domain runs, each
with a pretrain step and a finetune step (finetune added for consistency with the full 3-domain
production sweep, which is also pretrain+finetune). Supports multiple seeds (like
run_seed_sweep.py) for a 5-seed-average comparison against the project's existing 5-seed
publication results.

The matched-seed anchor (full 3-domain pretrain + finetune) is NOT a new run here — for each seed
it's exactly `02_train.py --stage {pretrain,finetune} --flux-only --seed N` with no --domains
override, which is byte-identical to commands already run as part of the completed 5-seed
publication sweep (run_seed_sweep.py) for seeds 1-5. Its checkpoints/metrics already exist at
outputs/multi_domain/{models,evaluation}/{pretrained,finetuned}_fluxonly_seed{1..5}[/_seedavg]/
— reuse those directly rather than rerunning (rerunning would silently overwrite those published
production artifacts).

Invokes each domain's numbered 0X_*.py scripts directly with the new --capacity-matched /
--domains flags, mirroring run_seed_sweep.py's run_stage() helper — run_amazon.py/
run_rangeland.py/run_multi_domain.py don't forward these flags. Sequential execution (one A100
on vm-sandeep), fail-fast on the first error. Flux-only target set throughout — see
ablation_description.md for why.

Must be run on vm-sandeep (GPU) — this script does not start/stop the VM itself; see
project_management/environment_spec.md § Compute placement policy.

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
    # No --flux-only: amazon_domain's scripts don't define that flag (no flux/pool distinction
    # for Amazon's targets — see multi_description.md § Flux-Only Variant).
    d = REPO_ROOT / "domains" / "amazon_domain"
    run_stage([str(d / "02_train.py"), "--seed", str(seed), "--capacity-matched"], log_path)
    run_stage([str(d / "03_predict.py"), "--seed", str(seed), "--capacity-matched"], log_path)
    run_stage([str(d / "04_evaluate.py"), "--seed", str(seed), "--capacity-matched"], log_path)


def run_capacity_rangeland(seed: int, log_path: Path) -> None:
    # --flux-only: matches the flux-only individual baseline (RG-seedsweep0714) this control is
    # meant to isolate capacity/dropout from — see ablation_description.md.
    d = REPO_ROOT / "domains" / "rangeland_domain"
    run_stage([str(d / "02_train.py"), "--flux-only", "--seed", str(seed), "--capacity-matched"], log_path)
    run_stage([str(d / "03_predict.py"), "--flux-only", "--seed", str(seed), "--capacity-matched"], log_path)
    run_stage([str(d / "04_evaluate.py"), "--flux-only", "--seed", str(seed), "--capacity-matched"], log_path)


def _run_multi_domain(stage: str, domains: str, seed: int, log_path: Path) -> None:
    """Trains (pretrain or finetune stage) and evaluates the shared trunk on the given
    comma-separated domain subset — one of the three pairwise ablation runs. See
    ablation_description.md for why the (no-subset) full 3-domain case is deliberately NOT run
    here (it already exists as production seeds 1-5), and for why finetune was added for
    consistency with that full 3-domain production sweep. A finetune call requires the matching
    pretrain call to have already run for this (domains, seed)."""
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
