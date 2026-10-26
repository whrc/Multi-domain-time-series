"""GCS readers (NetCDF, CSV) shared by Arctic and Amazon; data is read in-memory, never written locally."""

from typing import Any

import gcsfs
import pandas as pd
import xarray as xr


def gcs_filesystem() -> gcsfs.GCSFileSystem:
    return gcsfs.GCSFileSystem(token="google_default")


def read_netcdf(fs: gcsfs.GCSFileSystem, path: str, prefer_engine: str | None = None) -> xr.Dataset:
    """Load a NetCDF from GCS, trying h5netcdf then scipy (NetCDF3); times decoded with cftime."""
    engines = ("h5netcdf", "scipy")
    if prefer_engine in engines:
        engines = (prefer_engine, *(e for e in engines if e != prefer_engine))
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    last_err: Exception | None = None
    for engine in engines:
        try:
            with fs.open(path, "rb") as f:
                return xr.open_dataset(f, engine=engine, decode_times=coder).load()
        except Exception as err:  # try the next engine
            last_err = err
    raise RuntimeError(f"Failed to open NetCDF '{path}' with h5netcdf or scipy.") from last_err


def read_csv(fs: gcsfs.GCSFileSystem, path: str, **kwargs: Any) -> pd.DataFrame:
    with fs.open(path) as f:
        return pd.read_csv(f, **kwargs)
