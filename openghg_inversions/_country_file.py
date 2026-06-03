"""Country file loading helpers."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

CountryFileEngine = Literal["default", "h5netcdf", "h5py"]

COUNTRY_FILE_ENGINE_ORDER: tuple[CountryFileEngine, ...] = ("default", "h5netcdf", "h5py")
COUNTRY_FILE_SELECTED_ENGINE_ATTR = "_openghg_inversions_country_file_engine"


def _read_h5py_dataset(data: Any) -> np.ndarray:
    """Read a h5py dataset, decoding string arrays when needed."""
    if data.dtype.kind in {"O", "S"}:
        try:
            return np.asarray(data.asstr()[()])
        except (AttributeError, TypeError):
            return np.asarray(data[()]).astype(str)
    return np.asarray(data[()])


def _load_country_dataset_with_h5py(country_file_path: Path) -> xr.Dataset:
    """Load required country-file fields without xarray/HDF5 scale decoding."""
    import h5py

    with h5py.File(country_file_path, "r") as country_file:
        for name in ("lat", "lon", "name"):
            if name not in country_file:
                raise ValueError(f"Country file {country_file_path} is missing required variable {name!r}.")

        country_var = (
            "country" if "country" in country_file else "region" if "region" in country_file else None
        )
        if country_var is None:
            raise ValueError(
                f"Country file {country_file_path} must contain either a 'country' or 'region' variable."
            )

        lat = _read_h5py_dataset(country_file["lat"])
        lon = _read_h5py_dataset(country_file["lon"])
        country = _read_h5py_dataset(country_file[country_var])
        country_names = _read_h5py_dataset(country_file["name"]).astype(str)

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
            "country": (("lat", "lon"), country),
            "name": (("ncountries",), country_names),
        }
        if "country_code" in country_file:
            data_vars["country_code"] = (
                ("ncountries",),
                _read_h5py_dataset(country_file["country_code"]).astype(str),
            )

    return xr.Dataset(data_vars=data_vars, coords={"lat": lat, "lon": lon})


def _open_country_dataset(path: Path, engine: CountryFileEngine) -> xr.Dataset:
    """Open a country-file dataset using a named backend path."""
    if engine == "default":
        return xr.open_dataset(path)
    if engine == "h5py":
        return _load_country_dataset_with_h5py(path)

    return xr.open_dataset(path, engine=engine)


def _format_failures(failures: list[tuple[CountryFileEngine, BaseException]]) -> str:
    return "; ".join(f"{engine}: {type(error).__name__}: {error}" for engine, error in failures)


def _fallback_label(engine: CountryFileEngine) -> str:
    if engine == "h5py":
        return "direct HDF5 reader"
    return "xarray engine"


def load_country_dataset(
    country_file: str | Path,
    engines: Iterable[CountryFileEngine] = COUNTRY_FILE_ENGINE_ORDER,
) -> xr.Dataset:
    """Open and load a country file with fallbacks for HDF5 backend issues.

    Args:
        country_file: NetCDF country file path.
        engines: Ordered backend labels to try. ``"default"`` uses xarray's
            engine selection, ``"h5netcdf"`` passes ``engine="h5netcdf"`` to
            xarray, and ``"h5py"`` uses a direct minimal HDF5 reader.

    Returns:
        Loaded xarray Dataset with its file handle closed.

    Raises:
        OSError: if all configured backends fail to open the country file.
    """
    path = Path(country_file)
    failures: list[tuple[CountryFileEngine, BaseException]] = []

    for engine in engines:
        try:
            dataset = _open_country_dataset(path, engine)
            try:
                loaded = dataset.load()
            finally:
                dataset.close()
        except Exception as error:
            failures.append((engine, error))
            continue

        if failures:
            warnings.warn(
                "Falling back to "
                f"{_fallback_label(engine)} {engine!r} for country file {path} "
                f"after {_format_failures(failures)}",
                RuntimeWarning,
                stacklevel=2,
            )

        loaded.attrs[COUNTRY_FILE_SELECTED_ENGINE_ATTR] = engine
        return loaded

    raise OSError(f"Unable to open country file {path}. Tried {_format_failures(failures)}") from (
        failures[-1][1] if failures else None
    )
