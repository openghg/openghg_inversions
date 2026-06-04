"""Country file loading helpers."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import xarray as xr

CountryFileEngine = Literal["default", "h5netcdf"]

COUNTRY_FILE_ENGINE_ORDER: tuple[CountryFileEngine, ...] = ("default", "h5netcdf")
COUNTRY_FILE_SELECTED_ENGINE_ATTR = "_openghg_inversions_country_file_engine"


def _open_country_dataset(path: Path, engine: CountryFileEngine) -> xr.Dataset:
    """Open a country-file dataset using a named xarray engine."""
    if engine == "default":
        return xr.open_dataset(path)

    return xr.open_dataset(path, engine=engine)


def _format_failures(failures: list[tuple[CountryFileEngine, BaseException]]) -> str:
    return "; ".join(f"{engine}: {type(error).__name__}: {error}" for engine, error in failures)


def load_country_dataset(
    country_file: str | Path,
    engines: Iterable[CountryFileEngine] = COUNTRY_FILE_ENGINE_ORDER,
) -> xr.Dataset:
    """Open and load a country file with a fallback for HDF5 backend issues.

    Args:
        country_file: NetCDF country file path.
        engines: Ordered xarray backend labels to try. ``"default"`` uses xarray's
            engine selection; any other value is passed as ``engine=...``.

    Returns:
        Loaded xarray Dataset with its file handle closed.

    Raises:
        OSError: if all configured backends fail to open the country file.
    """
    path = Path(country_file)
    failures: list[tuple[CountryFileEngine, BaseException]] = []

    for engine in engines:
        try:
            with _open_country_dataset(path, engine) as dataset:
                loaded = dataset.load()
        except Exception as error:
            failures.append((engine, error))
            continue

        if failures:
            warnings.warn(
                "Falling back to xarray engine "
                f"{engine!r} for country file {path} after {_format_failures(failures)}",
                RuntimeWarning,
                stacklevel=2,
            )

        loaded.attrs[COUNTRY_FILE_SELECTED_ENGINE_ATTR] = engine
        return loaded

    raise OSError(f"Unable to open country file {path}. Tried {_format_failures(failures)}") from (
        failures[-1][1] if failures else None
    )
