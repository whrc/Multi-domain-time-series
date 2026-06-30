"""
Multi-domain — Step 1: pre-flight check.

See domains/multi_domain/multi_description.md § "Step 1 — Pre-flight check".

Verifies that all individual domain pkl files and scalers exist, logs Arctic grid
coverage, and prints a window-count summary. Produces no output files.
"""

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOMAINS = ["arctic", "amazon", "rangeland"]


def _count_windows(records: list[dict], seq_len: int, stride: int) -> int:
    """Approximate window count across all records, handling both Arctic and Amazon/Rangeland formats."""
    total = 0
    for r in records:
        segs = r.get("segments") or [r["data"]]
        for seg in segs:
            T = seg.shape[0]
            total += max(0, (T - seq_len) // stride + 1)
    return total


def main() -> None:
    cfg = load_config("multi_domain")
    seq_len = cfg["model"]["seq_len"]
    stride  = cfg["preprocessing"]["stride"]

    # Step 1: verify all pkl and scaler files exist
    for domain in DOMAINS:
        pre_dir = Path(cfg["paths"][domain]["preprocessed_dir"])
        for split in ("train", "val", "test"):
            p = pre_dir / f"{split}.pkl"
            if not p.exists():
                raise FileNotFoundError(p)
        scaler_p = Path(cfg["paths"][domain]["scaler"])
        if not scaler_p.exists():
            raise FileNotFoundError(scaler_p)
        logger.info("%s: all files verified", domain)

    # Step 2: log Arctic grid coverage from train.pkl
    arctic_train_path = Path(cfg["paths"]["arctic"]["preprocessed_dir"]) / "train.pkl"
    with arctic_train_path.open("rb") as f:
        arctic_train = pickle.load(f)
    grids = sorted({r["grid"] for r in arctic_train})
    logger.info("Arctic train grids (%d): %s", len(grids), grids)
    if len(grids) == 1:
        logger.warning("Arctic train.pkl contains only 1 grid — likely a dev-mode run; consider using all grids for production")

    # Step 3+4: window counts per domain × split
    rows = []
    for domain in DOMAINS:
        pre_dir = Path(cfg["paths"][domain]["preprocessed_dir"])
        for split in ("train", "val", "test"):
            with (pre_dir / f"{split}.pkl").open("rb") as f:
                records = pickle.load(f)
            n_windows = _count_windows(records, seq_len, stride)
            rows.append((domain, split, len(records), n_windows))

    logger.info("%-12s %-8s %8s %12s", "domain", "split", "records", "~windows")
    for domain, split, n_rec, n_win in rows:
        logger.info("%-12s %-8s %8d %12d", domain, split, n_rec, n_win)


if __name__ == "__main__":
    main()
