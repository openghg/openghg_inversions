"""Exact coordinate identity shared by spatial artifacts and caches."""

import hashlib

import numpy as np
import xarray as xr


def coordinate_sha256(*, lat: xr.DataArray, lon: xr.DataArray) -> str:
    """Return the identity digest of latitude and longitude labels."""
    digest = hashlib.sha256()
    for name, coordinate in (("lat", lat), ("lon", lon)):
        values = np.asarray(coordinate.values)
        digest.update(name.encode())
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()
