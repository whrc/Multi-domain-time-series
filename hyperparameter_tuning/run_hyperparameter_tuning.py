"""
Orchestrates the bounded hyperparameter-tuning sweep: 3 domains (arctic, amazon, rangeland)
x 3 hidden_dim sizes (small/medium/large).

See hyperparameter_tuning/hyperparameter_tuning_description.md for the full grid and selection
rule. Single seed by design (default --seeds 1) — the tuning sweep itself is intentionally not
multi-seed.

Each domain's target set/flags match this project's settled production convention:
  - Arctic: --label 500K_s400 --flux-only (matches run_seed_sweep.py's settled production
    config; --flux-only keeps the target set at GPP/RECO).
  - Amazon / Rangeland: no --flux-only flag for Amazon (no flux/pool split at that
    domain's level); Rangeland runs --flux-only too, matching its own production convention
    (GPP/RECO/Rm/Rg) and Arctic's target family, so the 3 domains stay comparable.

Selection: minimum validation loss from each run's history.csv (not test-set performance --
avoids test-set leakage into model selection). Writes
hyperparameter_tuning/hyperparameter_tuning_winners.yaml.

Must be run on vm-sandeep (GPU) for real (production-mode) runs -- see
project_management/environment_spec.md. This script does not start/stop the VM itself.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDY_DIR = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "outputs" / "_hyperparameter_tuning_logs"

sys.path.insert(0, str(REPO_ROOT))
from config.config import load_config  # noqa: E402
from domains.arctic_domain._naming import run_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIZES = ["small", "medium", "large"]
DOMAINS = ["arctic", "amazon", "rangeland"]

ARCTIC_LABEL = "500K_s400"  # settled production label (arctic_description.md) -- see run_seed_sweep.py
DOMAIN_DIRS = {
    "arctic": REPO_ROOT / "domains" / "arctic_domain",
    "amazon": REPO_ROOT / "domains" / "amazon_domain",
    "rangeland": REPO_ROOT / "domains" / "rangeland_domain",
}
DOMAIN_CONFIG_NAMES = {
    "arctic": "arctic_domain",
    "amazon": "amazon_domain",
    "rangeland": "rangeland_domain",
}
FIXED_CLI_ARGS = {
    "arctic": ["--label", ARCTIC_LABEL, "--flux-only"],
    "amazon": [],
    "rangeland": ["--flux-only"],
}


def run_stage(cmd_args: list[str], log_path: Path) -> None:
    logger.info("Running: %s", " ".join(cmd_args))
    with log_path.open("a") as log_file:
        subprocess.run([sys.executable, *cmd_args], check=True, stdout=log_file, stderr=subprocess.STDOUT)


def run_cell(domain: str, size: str, seed: int, log_path: Path) -> None:
    d = DOMAIN_DIRS[domain]
    tail = [*FIXED_CLI_ARGS[domain], "--model-size", size, "--seed", str(seed)]
    run_stage([str(d / "02_train.py"), *tail], log_path)
    if domain != "arctic":  # Arctic's dense NetCDF export is opt-in/heavy -- see run_seed_sweep.py
        run_stage([str(d / "03_predict.py"), *tail], log_path)
    run_stage([str(d / "04_evaluate.py"), *tail], log_path)


def history_csv_path(domain: str, size: str, seed: int) -> Path:
    cfg = load_config(DOMAIN_CONFIG_NAMES[domain])
    if domain == "arctic":
        label = run_label(None, ARCTIC_LABEL)
        output_label = f"{label}_fluxonly_{size}_seed{seed}"
        return Path(cfg["paths"]["evaluation"]) / output_label / "history.csv"
    # amazon/rangeland's 02_train.py builds suffix as "_seed{seed}" then "_{size}" (see each
    # domain's own 02_train.py) -- rangeland's leading "_fluxonly" is empty here since
    # --flux-only doesn't touch the *_domain.py suffix construction the same way Arctic's does
    # (rangeland's flux_only suffix piece is applied before seed/size, but flux-only + this
    # sweep's seed/size combination never collide in practice; verified against the actual
    # suffix logic in domains/rangeland_domain/02_train.py).
    eval_dir = Path(cfg["paths"]["evaluation"])
    suffix = "_fluxonly" if domain == "rangeland" else ""
    suffix += f"_seed{seed}_{size}"
    eval_dir = eval_dir.with_stem(eval_dir.stem + suffix)
    return eval_dir / "history.csv"


def best_val_loss(domain: str, size: str, seed: int) -> float:
    path = history_csv_path(domain, size, seed)
    df = pd.read_csv(path)
    return float(df["val_loss"].min())


def select_winners(seed: int, domains: list[str]) -> dict[str, str]:
    winners = {}
    for domain in domains:
        losses = {size: best_val_loss(domain, size, seed) for size in SIZES}
        winner = min(losses, key=losses.get)
        winners[domain] = winner
        logger.info("%s: val losses %s -> winner=%s", domain, losses, winner)
    return winners


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1],
                        help="Training seed(s) to run each of the 9 cells with. The tuning "
                             "sweep is single-seed by design (default: 1). Winner selection "
                             "always uses the first seed passed, with a warning if more than "
                             "one is given.")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=DOMAINS,
                        help="Restrict the sweep to a subset of domains (e.g. just 'rangeland' "
                             "when Arctic/Amazon's sweep outputs are being reused from a prior "
                             "run rather than retrained -- see "
                             "hyperparameter_tuning_description.md).")
    args = parser.parse_args()
    if len(args.seeds) > 1:
        logger.warning("Multiple --seeds passed (%s) -- the tuning sweep is single-seed by "
                       "design; winner selection will use seed=%d only.", args.seeds, args.seeds[0])

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        log_path = LOG_DIR / f"seed{seed}.log"
        for domain in args.domains:
            for size in SIZES:
                logger.info("=== %s / %s / seed=%d ===", domain, size, seed)
                run_cell(domain, size, seed, log_path)

    winners_path = STUDY_DIR / "hyperparameter_tuning_winners.yaml"
    existing = {}
    if winners_path.exists():
        with winners_path.open() as f:
            existing = yaml.safe_load(f) or {}
    winners = {**existing, **select_winners(args.seeds[0], args.domains)}
    with winners_path.open("w") as f:
        yaml.safe_dump(winners, f, sort_keys=False)
    logger.info("Wrote winners to %s: %s", winners_path, winners)


if __name__ == "__main__":
    main()
