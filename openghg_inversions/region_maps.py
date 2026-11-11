"""Eager reading of region maps and country files.

A region map is a two-dimensional ``(lat, lon)`` array of class codes, read
from a NetCDF variable named ``country`` or ``region``, together with the
file's attributes (for example ``inner_region_label`` on fixed outer-region
maps). A country file is a region map plus a ``name`` lookup table, and
optionally ``country_code``, indexed by ``ncountries``: code ``i`` belongs to
``name[i]``.

Every reader here is an eager boundary: data is loaded and the file closed
before returning. Reads go through :func:`openghg_inversions.utils.open_ds`,
which tries xarray backends in turn. When those all fail, a direct HDF5 reader
for this known variable layout is the last resort; it is needed when xarray
cannot decode the file's HDF5 dimension scales.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from openghg_inversions.utils import format_open_failures, open_ds

REGION_MAP_VARIABLES = ("country", "region")
"""Class-code variable names, in inference order."""

_LOOKUP_VARIABLES = ("name", "country_code")
_HDF5_BOOKKEEPING_ATTRS = {"DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME"}


def region_map_variable(dataset: xr.Dataset, variable: str | None = None) -> str:
    """Choose the class-code variable of a region-map dataset.

    Args:
        dataset: Dataset read from a region-map or country file.
        variable: Explicit variable name, or None to use ``country`` when
            present and otherwise ``region``.

    Returns:
        Name of a variable present in ``dataset``.

    Raises:
        KeyError: If the chosen variable, or both inferred names, are absent.
    """
    candidates = (variable,) if variable is not None else REGION_MAP_VARIABLES
    for name in candidates:
        if name in dataset:
            return name
    raise KeyError(f"Region map has none of the variables {candidates!r}; found {list(dataset.data_vars)!r}.")


def open_region_map_dataset(path: str | Path) -> xr.Dataset:
    """Eagerly read a region-map or country file, with HDF5 fallbacks.

    Variable names are kept as stored, so a file may provide ``region`` rather
    than ``country``; see :func:`load_country_dataset` for the canonical
    reporting view.

    Args:
        path: Exact NetCDF file.

    Returns:
        In-memory dataset with its file closed.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        OSError: If every xarray backend and the direct HDF5 reader fail.
    """
    try:
        return open_ds(path, decode_coords="all")
    except FileNotFoundError:
        raise
    except OSError as xarray_error:
        try:
            dataset = _read_region_map_with_h5py(Path(path))
        except Exception as h5py_error:
            raise OSError(f"{xarray_error}; {format_open_failures([('h5py', h5py_error)])}") from h5py_error
        warnings.warn(
            f"Falling back to direct HDF5 reader 'h5py' for region map {path} after {xarray_error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return dataset


def load_region_map(path: str | Path, variable: str | None = None) -> xr.DataArray:
    """Eagerly read one region map and close its file.

    Args:
        path: Exact NetCDF file, without directory discovery.
        variable: Class-code variable to select; None selects ``country``
            when present, otherwise ``region``.

    Returns:
        In-memory array with coordinates and dataset attributes, overridden by
        variable attributes where both supply the same key. No grid alignment
        or class interpretation is performed.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        OSError: If no reader can open the file.
        KeyError: If the selected variable is absent.
    """
    dataset = open_region_map_dataset(path)
    result = dataset[region_map_variable(dataset, variable)]
    result.attrs = {**dataset.attrs, **result.attrs}
    return result


def load_country_dataset(path: str | Path) -> xr.Dataset:
    """Eagerly read a country file with its class codes named ``country``.

    A file that stores codes as ``region`` is renamed, so every reader yields
    the same variable names. Lookup variables such as ``name`` are kept as
    stored.

    Args:
        path: Exact NetCDF country file.

    Returns:
        In-memory dataset with a ``country`` variable on ``(lat, lon)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        OSError: If no reader can open the file.
        KeyError: If neither ``country`` nor ``region`` is present.
    """
    dataset = open_region_map_dataset(path)
    variable = region_map_variable(dataset)
    return dataset if variable == "country" else dataset.rename({variable: "country"})


def _read_h5py_values(data: Any) -> np.ndarray:
    """Read a h5py dataset, decoding string arrays when needed."""
    if data.dtype.kind in {"O", "S"}:
        try:
            return np.asarray(data.asstr()[()])
        except (AttributeError, TypeError):
            return np.asarray(data[()]).astype(str)
    return np.asarray(data[()])


def _h5py_attrs(attrs: Any) -> dict[str, Any]:
    """Copy user attributes, dropping HDF5 and netCDF bookkeeping entries."""
    result: dict[str, Any] = {}
    for key, value in attrs.items():
        if key in _HDF5_BOOKKEEPING_ATTRS or key.startswith("_"):
            continue
        if isinstance(value, bytes):
            value = value.decode()
        elif isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(()).item()
        result[key] = value
    return result


def _read_region_map_with_h5py(path: Path) -> xr.Dataset:
    """Read the known region-map variables without HDF5 dimension-scale decoding.

    Values are returned as stored, without CF decoding of fill values or
    scaling. Only ``lat``, ``lon``, the class-code variables and the lookup
    variables are read.
    """
    import h5py

    with h5py.File(path, "r") as file:
        for name in ("lat", "lon"):
            if name not in file:
                raise KeyError(f"Region map {path} is missing required variable {name!r}.")
        data_vars: dict[str, Any] = {}
        for name in REGION_MAP_VARIABLES:
            if name in file:
                data_vars[name] = (("lat", "lon"), _read_h5py_values(file[name]), _h5py_attrs(file[name].attrs))
        if not data_vars:
            raise KeyError(f"Region map {path} has none of the variables {REGION_MAP_VARIABLES!r}.")
        for name in _LOOKUP_VARIABLES:
            if name in file:
                data_vars[name] = (("ncountries",), _read_h5py_values(file[name]).astype(str))
        coords = {
            name: (name, _read_h5py_values(file[name]), _h5py_attrs(file[name].attrs)) for name in ("lat", "lon")
        }
        attrs = _h5py_attrs(file.attrs)

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
