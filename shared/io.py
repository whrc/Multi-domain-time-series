"""
Shared I/O helpers — GCS access reused across domains.

Arctic and Amazon stream their raw data from GCS; this module centralises the
filesystem handle and the NetCDF/CSV readers so the pipeline scripts never
duplicate the connection or engine-fallback logic. Data is read directly from
the bucket and never written to local disk.
"""

from typing import Any

import gcsfs
import pandas as pd
import xarray as xr


def gcs_filesystem() -> gcsfs.GCSFileSystem:
    """Return a GCS filesystem authenticated with the ambient google credentials."""
    return gcsfs.GCSFileSystem(token="google_default")


def read_netcdf(fs: gcsfs.GCSFileSystem, path: str, prefer_engine: str | None = None) -> xr.Dataset:
    """Open a NetCDF file from GCS into memory.

    Tries the HDF5 engine first, falling back to scipy for NetCDF3 files (pass
    ``prefer_engine="scipy"`` for known-NetCDF3 files to skip the failing probe).
    Times are decoded with cftime so non-standard calendars (e.g. ``noleap``) load
    without error. The dataset is fully loaded so the GCS handle can close.
    """
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
    """Read a CSV file from GCS into a DataFrame. ``kwargs`` pass through to pandas."""
    with fs.open(path) as f:
        return pd.read_csv(f, **kwargs)
