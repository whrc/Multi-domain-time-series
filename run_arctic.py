# run_arctic.py — Run the full Arctic domain pipeline in order.
# To target a different domain config, set the ARCTIC_DOMAIN env var before running.

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

os.environ.setdefault("ARCTIC_DOMAIN", "arctic_domain")
log.info("Domain: %s", os.environ["ARCTIC_DOMAIN"])

SCRIPTS_DIR = Path(__file__).parent / "domains" / "arctic_domain"


def _load_main(script_name: str):
    path = SCRIPTS_DIR / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".", "_"), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


pipeline_start = time.time()

for script, label in [
    ("01_preprocess.py", "Preprocess"),
    ("02_train.py",      "Train"),
    ("03_predict.py",    "Predict"),
    ("04_evaluate.py",   "Evaluate"),
]:
    log.info("=" * 60)
    log.info("STEP: %s", label)
    log.info("=" * 60)
    step_start = time.time()
    _load_main(script)()
    elapsed = (time.time() - step_start) / 60
    log.info("STEP %s complete. (%.2f min)", label, elapsed)

total = (time.time() - pipeline_start) / 60
log.info("Arctic pipeline finished. Total: %.2f min", total)