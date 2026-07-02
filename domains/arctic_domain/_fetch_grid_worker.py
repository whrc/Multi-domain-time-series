"""Standalone worker: fetch one grid's records in an isolated process, pickle the result.

Invoked as a subprocess (see fetch_grid_records_isolated in 01_preprocess.py) because a shared
gcsfs filesystem/event-loop was found to accumulate bad internal state after repeated use within
one process, hanging indefinitely on a later grid fetch (reproduced with both a shared and a
freshly-constructed GCSFileSystem instance — the leak survives instance boundaries, so a fresh
OS process per grid is the reliable fix).
"""

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from shared.io import gcs_filesystem  # noqa: E402

_spec = importlib.util.spec_from_file_location("_pp01", Path(__file__).parent / "01_preprocess.py")
_pp01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pp01)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config("arctic_domain")
    fs = gcs_filesystem()
    idx_map = _pp01.monthly_index_map(cfg)
    proj_start = cfg["time"]["projected_start_year"]
    recs = _pp01.fetch_grid_records(cfg, fs, args.grid, idx_map, proj_start)
    with open(args.out, "wb") as f:
        pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
