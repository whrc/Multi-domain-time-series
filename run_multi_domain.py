"""Multi-domain pipeline entry point: preprocess → pretrain → finetune → predict (per domain) → evaluate.

--flux-only selects the flux-only target variant (see multi_description.md § "Flux-Only Variant").
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
        choices=["pretrained", "finetuned"],
        default="finetuned",
        help="Checkpoint to use for --stage predict (default: finetuned)",
    )
    parser.add_argument(
        "--flux-only",
        action="store_true",
        help="Select the flux-only target-set variant (see module docstring)",
    )
    args = parser.parse_args()
    flux_flag = ["--flux-only"] if args.flux_only else []

    if args.stage == "preprocess":
        run_script(DOMAIN_DIR / "01_preprocess.py")

    elif args.stage == "pretrain":
        run_script(DOMAIN_DIR / "02_train.py", ["--stage", "pretrain"] + flux_flag)

    elif args.stage == "finetune":
        run_script(DOMAIN_DIR / "02_train.py", ["--stage", "finetune"] + flux_flag)

    elif args.stage == "predict":
        if args.domain is None:
            parser.error("--domain is required for --stage predict")
        run_script(
            DOMAIN_DIR / "03_predict.py",
            ["--domain", args.domain, "--checkpoint", args.checkpoint] + flux_flag,
        )

    elif args.stage == "evaluate":
        run_script(DOMAIN_DIR / "04_evaluate.py", flux_flag)


if __name__ == "__main__":
    main()
