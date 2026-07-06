"""
Entry point for the arctic domain pipeline (orchestrates 01_preprocess -> 04_evaluate,
plus the optional learning-curve step).

The dev/production hyperparameter profile is selected via `mode` in
config/arctic_domain.yaml. Run with the project venv: `.venv\\Scripts\\python.exe run_arctic.py`.

03_predict.py (the `predict` stage) is OPT-IN and excluded from the default pipeline: it
writes a full dense NetCDF grid per circumpolar tile, which can reach hundreds of GB for the
real test set (see arctic_description.md Step 3). It is not needed for evaluation metrics or
figures — 04_evaluate.py recomputes predictions from the checkpoint directly. Pass
`--include-predict` to add it to a default run, or `--stage predict` to run it standalone.

Usage:
    run_arctic.py                                     # run preprocess -> train -> evaluate
    run_arctic.py --include-predict                   # also run predict (large NetCDF output!)
    run_arctic.py --stage preprocess                  # run one stage
    run_arctic.py --stage preprocess --train-size N   # subsample train to ~N windows
    run_arctic.py --stage predict --train-size N      # run predict standalone (large output!)
    run_arctic.py --stage learning-curve              # plot val performance vs train size
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOMAIN_DIR = Path(__file__).resolve().parent / "domains" / "arctic_domain"

STAGE_SCRIPTS = {
    "preprocess":     "01_preprocess.py",
    "train":          "02_train.py",
    "predict":        "03_predict.py",
    "evaluate":       "04_evaluate.py",
    "learning-curve": "05_learning_curve.py",
}

FULL_PIPELINE = ["preprocess", "train", "evaluate"]  # predict is opt-in — see --include-predict


def run_script(script: Path, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(script)] + (extra_args or [])
    logger.info("=== Running %s ===", script.name)
    subprocess.run(cmd, check=True)
    logger.info("=== Finished %s ===", script.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Arctic domain pipeline runner")
    parser.add_argument(
        "--stage",
        choices=list(STAGE_SCRIPTS),
        default=None,
        help="Run a single stage. Omit to run 01 → 04 in order.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Target train window count. With --stage preprocess, overrides "
             "preprocessing.train_size in config. With --stage train/predict/evaluate, "
             "selects which labeled train pkl/checkpoint/output set to use "
             "(must match a size already generated).",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Delete cached val.pkl and test.pkl before preprocessing. "
             "Required when switching from dev to production mode. Only effective with --stage preprocess.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Concurrent isolated-subprocess grid fetches (overrides preprocessing.max_workers). "
             "Only effective with --stage preprocess.",
    )
    parser.add_argument(
        "--capped-stride",
        type=int,
        default=None,
        help="Override preprocessing.capped_stride. Only effective with --stage preprocess.",
    )
    parser.add_argument(
        "--grids",
        type=str,
        default=None,
        help="Comma-separated grid names, overriding auto-discovery. Only effective with --stage preprocess.",
    )
    parser.add_argument(
        "--include-predict",
        action="store_true",
        help="Include the predict stage in the default (--stage-less) full pipeline. "
             "WARNING: predict writes a full dense NetCDF grid per circumpolar tile, which "
             "can reach hundreds of GB for the real test set — off by default. Not needed for "
             "evaluation metrics/figures (04_evaluate.py recomputes predictions from the "
             "checkpoint directly). Use `--stage predict` to run it standalone anytime.",
    )
    args = parser.parse_args()

    if args.stage:
        script = DOMAIN_DIR / STAGE_SCRIPTS[args.stage]
        extra: list[str] = []
        if args.stage == "preprocess":
            if args.train_size is not None:
                extra += ["--train-size", str(args.train_size)]
            if args.force_recompute:
                extra += ["--force-recompute"]
            if args.max_workers is not None:
                extra += ["--max-workers", str(args.max_workers)]
            if args.capped_stride is not None:
                extra += ["--capped-stride", str(args.capped_stride)]
            if args.grids is not None:
                extra += ["--grids", args.grids]
        elif args.stage in ("train", "predict", "evaluate") and args.train_size is not None:
            extra += ["--train-size", str(args.train_size)]
        run_script(script, extra)
    else:
        stages = list(FULL_PIPELINE)
        if args.include_predict:
            stages.insert(stages.index("evaluate"), "predict")
        for stage in stages:
            run_script(DOMAIN_DIR / STAGE_SCRIPTS[stage])
        logger.info("Pipeline complete for arctic_domain")


if __name__ == "__main__":
    main()
