"""
Shared pipeline runner — backs every run_<domain>.py entry point.

Runs the four numbered scripts in order as separate subprocesses using the same
Python interpreter (the project venv). Process isolation keeps each stage clean and
sidesteps the non-importable numbered filenames; the run aborts on the first failure.
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS = ["01_preprocess.py", "02_train.py", "03_predict.py", "04_evaluate.py"]


def run_domain(domain_dir: Path) -> None:
    """Execute 01_preprocess -> 04_evaluate for the given domain directory."""
    for script in SCRIPTS:
        path = domain_dir / script
        logger.info("=== Running %s ===", path.name)
        subprocess.run([sys.executable, str(path)], check=True)
        logger.info("=== Finished %s ===", path.name)
    logger.info("Pipeline complete for %s", domain_dir.name)
