"""Multi-domain step 1: verify per-domain pkl/scaler files exist and log grid coverage and window counts."""

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from domains.multi_domain.flux_only import DOMAINS, arctic_stride_seq_len, arctic_train_pkl_path  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _train_pkl_path(domain: str, cfg: dict) -> Path:
    """Arctic train pkl is size/stride-labeled (e.g. train_500K_s400.pkl); others use train.pkl."""
    if domain == "arctic":
        return arctic_train_pkl_path(cfg)
    return Path(cfg["paths"][domain]["preprocessed_dir"]) / "train.pkl"


def _count_windows(records: list[dict], seq_len: int, stride: int) -> int:
    """Approximate window count; handles both Arctic (segments) and Amazon/Rangeland (data) records."""
    total = 0
    for r in records:
        segs = r.get("segments") or [r["data"]]
        for seg in segs:
            T = seg.shape[0]
            total += max(0, (T - seq_len) // stride + 1)
    return total


def _stride_for(domain: str, split: str, path: Path, cfg: dict) -> int:
    """Arctic stride comes from the pkl's sidecar; other domains use cfg.preprocessing.stride."""
    if domain == "arctic":
        stride, _ = arctic_stride_seq_len(path)
        return stride
    return cfg["preprocessing"]["stride"]


def main() -> None:
    cfg = load_config("multi_domain")
    seq_len = cfg["model"]["seq_len"]

    for domain in DOMAINS:
        pre_dir = Path(cfg["paths"][domain]["preprocessed_dir"])
        for split in ("train", "val", "test"):
            p = _train_pkl_path(domain, cfg) if split == "train" else pre_dir / f"{split}.pkl"
            if not p.exists():
                raise FileNotFoundError(p)
        scaler_p = Path(cfg["paths"][domain]["scaler"])
        if not scaler_p.exists():
            raise FileNotFoundError(scaler_p)
        logger.info("%s: all files verified", domain)

    arctic_train_path = _train_pkl_path("arctic", cfg)
    with arctic_train_path.open("rb") as f:
        arctic_train = pickle.load(f)
    grids = sorted({r["grid"] for r in arctic_train})
    logger.info("Arctic train grids (%d): %s", len(grids), grids)
    if len(grids) == 1:
        logger.warning("Arctic train pkl contains only 1 grid — likely a dev-mode run; consider using all grids for production")

    rows = []
    for domain in DOMAINS:
        pre_dir = Path(cfg["paths"][domain]["preprocessed_dir"])
        for split in ("train", "val", "test"):
            path = _train_pkl_path(domain, cfg) if split == "train" else pre_dir / f"{split}.pkl"
            with path.open("rb") as f:
                records = pickle.load(f)
            stride = _stride_for(domain, split, path, cfg)
            n_windows = _count_windows(records, seq_len, stride)
            rows.append((domain, split, len(records), n_windows))

    logger.info("%-12s %-8s %8s %12s", "domain", "split", "records", "~windows")
    for domain, split, n_rec, n_win in rows:
        logger.info("%-12s %-8s %8d %12d", domain, split, n_rec, n_win)


if __name__ == "__main__":
    main()
