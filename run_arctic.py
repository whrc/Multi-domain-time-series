"""
Entry point for the arctic domain pipeline (orchestrates 01_preprocess -> 04_evaluate).

The dev/production hyperparameter profile is selected via `mode` in
config/arctic_domain.yaml. Run with the project venv: `.venv\\Scripts\\python.exe run_arctic.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.runner import run_domain  # noqa: E402

if __name__ == "__main__":
    run_domain(Path(__file__).resolve().parent / "domains" / "arctic_domain")
