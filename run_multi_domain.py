"""
Entry point for the multi-domain pipeline.

Usage:
    run_multi_domain.py --stage preprocess
    run_multi_domain.py --stage pretrain
    run_multi_domain.py --stage finetune
    run_multi_domain.py --stage predict --domain {arctic,amazon,rangeland} [--checkpoint {stage1,stage2}]
    run_multi_domain.py --stage evaluate

Intended workflow:
    1. preprocess     — verify individual domain pkl files exist; log dataset sizes
    2. pretrain       — Stage 1 joint mixed-step training
    3. finetune       — Stage 2 per-domain head fine-tuning (requires pretrain)
    4. predict        — inference on test set per domain per checkpoint (run for each domain × stage)
    5. evaluate       — per-unit metrics + diagnostic plots for all domains × stages
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOMAIN_DIR = Path(__file__).resolve().parent / "domains" / "multi_domain"


def run_script(script: Path, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(script)] + (extra_args or [])
    logger.info("=== Running %s ===", script.name)
    subprocess.run(cmd, check=True)
    logger.info("=== Finished %s ===", script.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-domain pipeline runner")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "pretrain", "finetune", "predict", "evaluate"],
        required=True,
    )
    parser.add_argument(
        "--domain",
        choices=["arctic", "amazon", "rangeland"],
        default=None,
        help="Required for --stage predict",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["stage1", "stage2"],
        default="stage2",
        help="Checkpoint to use for --stage predict (default: stage2)",
    )
    args = parser.parse_args()

    if args.stage == "preprocess":
        run_script(DOMAIN_DIR / "01_preprocess.py")

    elif args.stage == "pretrain":
        run_script(DOMAIN_DIR / "02_train.py", ["--stage", "pretrain"])

    elif args.stage == "finetune":
        run_script(DOMAIN_DIR / "02_train.py", ["--stage", "finetune"])

    elif args.stage == "predict":
        if args.domain is None:
            parser.error("--domain is required for --stage predict")
        run_script(
            DOMAIN_DIR / "03_predict.py",
            ["--domain", args.domain, "--checkpoint", args.checkpoint],
        )

    elif args.stage == "evaluate":
        run_script(DOMAIN_DIR / "04_evaluate.py")


if __name__ == "__main__":
    main()
