"""Run the Amazon pipeline (01_preprocess -> 04_evaluate). Profile set by `mode` in
config/amazon_domain.yaml."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.runner import run_domain  # noqa: E402

if __name__ == "__main__":
    run_domain(Path(__file__).resolve().parent / "domains" / "amazon_domain")
