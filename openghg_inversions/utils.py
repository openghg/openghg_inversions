"""Common Python functions used by RHIME and supporting inversion workflows.

The main functions are related to applying basis functions to the flux and boundary
conditions, and their sensitivities.

Many functions in this submodule originated in the ACRG code base (in `acrg.name`).

"""

import re
import warnings
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from openghg.analyse import combine_datasets as openghg_combine_datasets

from openghg_inversions.array_ops import to_dense
from openghg_inversions.config.paths import Paths

openghginv_path = Paths.openghginv


def combine_datasets(
    dataset_a: xr.Dataset,
    dataset_b: xr.Dataset,
    method: str | None = "nearest",
    tolerance: float | None = None,
) -> xr.Dataset:
    """Merges two datasets and re-indexes to the first dataset.

    If "fp" variable is found within the combined dataset,
    the "time" values where the "lat", "lon" dimensions didn't match are removed.

    NOTE: this is temporary solution while waiting for `.load()` to be added to openghg version of combine_datasets

    Args:
        dataset_a: First dataset to merge
        dataset_b: Second dataset to merge
        method: One of None, nearest, ffill, bfill.
                See xarray.DataArray.reindex_like for list of options and meaning.
                Defaults to ffill (forward fill)
        tolerance: Maximum allowed tolerance between matches.

    Returns:
        xarray.Dataset: Combined dataset indexed to dataset_a
    """
    return openghg_combine_datasets(dataset_a, dataset_b.load(), method=method, tolerance=tolerance)


OPEN_DS_ENGINES: tuple[str | None, ...] = (None, "h5netcdf")
"""Default xarray backends tried, in order, when eagerly opening one file."""

_OPEN_FAILURES = (OSError, RuntimeError, ValueError)


def _engine_label(engine: str | None) -> str:
    """Name an xarray backend choice for messages."""
    return "default" if engine is None else engine


def format_open_failures(failures: Sequence[tuple[str, BaseException]]) -> str:
    """Summarise failed open attempts as ``label: ErrorType: message`` entries.

    Args:
        failures: Pairs of backend label and the error it raised, in attempt order.

    Returns:
        Semicolon-separated summary for warnings and error messages.
    """
    return "; ".join(f"{label}: {type(error).__name__}: {error}" for label, error in failures)


def open_ds(
    path: str | Path,
    chunks: dict | None = None,
    combine: Literal["by_coords", "nested"] = "by_coords",
    *,
    engines: Sequence[str | None] = OPEN_DS_ENGINES,
    decode_coords: bool | Literal["coordinates", "all"] = True,
) -> xr.Dataset:
    """Open a NetCDF file eagerly, or files lazily when ``chunks`` is given.

    Without ``chunks`` this is the eager read boundary for one file: the data
    is loaded and the file closed before returning. Each backend in
    ``engines`` is tried in order; a later backend that succeeds emits a
    ``RuntimeWarning`` naming the earlier failures. This works around HDF5
    backend problems that affect only some installations.

    Args:
        path: Path to file to open (or glob, when ``chunks`` is given).
        chunks: Size of chunks for each dimension, e.g. {'lat': 50, 'lon': 50}.
            Opens dataset with dask, such that it is opened 'lazily' and all of the data
            is not loaded into memory. Defaults to None - dataset is opened without dask.
        combine: Way in which the data should be combined (if using chunks), either:
            'by_coords': order the datasets before concatenating (default)
            'nested': concatenate datasets in the order supplied.
        engines: xarray backends for the eager path, in order; ``None`` lets
            xarray choose. Ignored when ``chunks`` is given.
        decode_coords: Passed to ``xarray.open_dataset`` on the eager path.

    Returns:
        xr.Dataset: Opened xarray Dataset.

    Raises:
        FileNotFoundError: If ``path`` does not exist (eager path; no backend
            fallback is attempted).
        OSError: If every backend fails, chained from the last failure.
    """
    if chunks is not None:
        return xr.open_mfdataset(path, chunks=chunks, combine=combine)

    failures: list[tuple[str, BaseException]] = []
    for engine in engines:
        try:
            with xr.open_dataset(path, engine=engine, decode_coords=decode_coords) as ds:
                loaded = ds.load()
        except FileNotFoundError:
            raise
        except _OPEN_FAILURES as error:
            failures.append((_engine_label(engine), error))
            continue
        if failures:
            warnings.warn(
                f"Falling back to xarray engine {_engine_label(engine)!r} for {path} "
                f"after {format_open_failures(failures)}",
                RuntimeWarning,
                stacklevel=2,
            )
        return loaded

    raise OSError(f"Unable to open {path}. Tried {format_open_failures(failures)}") from (
        failures[-1][1] if failures else None
    )


def read_netcdfs(
    files: list[str] | list[Path],
    dim: str = "time",
    chunks: dict | None = None,
    verbose: bool = True,
) -> xr.Dataset:
    """Use xarray to open sequential netCDF files and concatenate them along the specified dimension.

    Note: this function makes sure that file is closed after open_dataset call.

    Args:
        files: List of netCDF filenames.
        dim: Dimension of netCDF to use for concatenating the files. Default = "time".
        chunks: Size of chunks for each dimension, e.g. {'lat': 50, 'lon': 50}.
            Opens dataset with dask, such that it is opened 'lazily' and all of the data
            is not loaded into memory. Defaults to None - dataset is opened without dask.
        verbose: If True, print progress information.

    Returns:
        xr.Dataset: All files open as one concatenated xarray.Dataset object.

    Note:
        This could be done more efficiently with xr.open_mfdataset (most likely).
    """
    if verbose:
        print("Reading and concatenating files ...")
        for fname in files:
            print(fname)

    datasets = [open_ds(p, chunks=chunks) for p in sorted(files)]

    # reindex all of the lat-lon values to a common one to prevent floating point error differences
    fields_ds = open_ds(files[0])
    fp_lat = fields_ds["lat"].values
    fp_lon = fields_ds["lon"].values

    datasets = [
        ds.reindex(indexers={"lat": fp_lat, "lon": fp_lon}, method="nearest", tolerance=1e-5)
        for ds in datasets
    ]

    combined = xr.concat(datasets, dim)

    return combined


def datatree_ncdf_encoding(dt: xr.DataTree) -> dict:
    encoding = {}
    for g in dt.groups:
        if dt[g].is_leaf:
            encoding[g] = ncdf_encoding(dt[g].to_dataset())
    return encoding


def ncdf_encoding(ds_in: xr.Dataset) -> dict:
    """Define encoding for netCDF4 files.

    Args:
        ds_in: Xarray dataset to define encoding for.

    Returns:
        dict: Dictionary with encoding parameters for netCDF4 files.
    """
    # variables with variable length data types shouldn't be compressed
    # e.g. object ("O") or unicode ("U") type
    do_not_compress = []
    dtype_pat = re.compile(r"[<>=]?[UO]")  # regex for Unicode and Object dtypes
    for dv in ds_in.data_vars:
        if dtype_pat.match(ds_in[dv].data.dtype.str):
            do_not_compress.append(dv)
    encoding = {
        var: {"zlib": True, "complevel": 5, "shuffle": True}
        for var in ds_in.data_vars
        if var not in do_not_compress
    }

    return encoding


def write_netcdf_preserving_bounds_attrs(
    ds: xr.Dataset,
    path: str | Path,
    *,
    unlimited_dims: list[str] | None = None,
) -> None:
    """Write a compressed NetCDF preserving coordinates and explicit bounds metadata.

    Xarray's CF encoder removes ``units`` and ``calendar`` from a bounds
    variable when those attributes match its coordinate. Some external schemas,
    including the latest PARIS CDL templates, require the attributes on both
    variables, so they are appended after the normal write. The initial write
    overwrites ``path``; backend and filesystem exceptions propagate.

    Args:
        ds: Dataset carrying the required bounds attributes.
        path: Destination NetCDF path.
        unlimited_dims: Optional dimensions to encode as unlimited.
    """
    ds = ds.assign({name: to_dense(value) for name, value in ds.data_vars.items()})
    ds.to_netcdf(
        path,
        unlimited_dims=unlimited_dims,
        mode="w",
        encoding=ncdf_encoding(ds),
    )

    bounds_attrs: dict[str, dict[str, Any]] = {}
    for variable in ds.variables.values():
        bounds_name = variable.attrs.get("bounds")
        if not isinstance(bounds_name, str) or bounds_name not in ds:
            continue
        attrs = {
            name: ds[bounds_name].attrs[name]
            for name in ("units", "calendar")
            if name in ds[bounds_name].attrs
        }
        if attrs:
            bounds_attrs[bounds_name] = attrs

    for bounds_name, attrs in bounds_attrs.items():
        bounds = ds[bounds_name]
        metadata_patch = xr.Dataset({bounds_name: (bounds.dims, bounds.data, attrs)})
        metadata_patch.to_netcdf(path, mode="a")


def get_country_file_path(country_file: str | Path | None = None, domain: str | None = None):
    if isinstance(country_file, str | Path):
        result = Path(country_file)

        if not result.exists():
            raise FileNotFoundError(f"No country file found at path {result}")

        return result

    if domain is None:
        raise ValueError("If `country_file` is None, then `domain` must be specified.")

    # try to find country file in default location
    country_directory = openghginv_path / "countries"

    if not country_directory.exists():
        country_directory.mkdir()

        raise FileNotFoundError(
            f"Country definition file not found. Please add to {openghginv_path}/countries/"
        )

    result = country_directory / f"country_{domain}.nc"

    if not result.exists():
        raise FileNotFoundError(
            f"Country definition file not found. Please add to {openghginv_path}/countries/"
        )

    return result


def get_country(domain: str, country_file: str | Path | None = None):
    """Open country file for given domain and return as a SimpleNamespace.

    NOTE: a SimpleNamespace is a like dict with class like attribute access

    Args:
        domain: domain of inversion
        country_file: optional string or Path to country file. If `None`, then the first file found in
            `openghg_inversions/countries/` is used.

    Returns:
        SimpleNamespace with attributes: lon, lat, lonmax, lonmin, latmax, latmin, country, and name

    Raises:
        KeyError: If the file has neither a ``country`` nor a ``region`` variable,
            or no ``name`` variable.
    """
    from openghg_inversions.region_maps import load_country_dataset  # region_maps imports utils

    f = load_country_dataset(get_country_file_path(country_file=country_file, domain=domain))
    lon = f["lon"].values
    lat = f["lat"].values
    country = f["country"]
    name = f["name"].values.astype(str)

    result = dict(
        lon=lon,
        lat=lat,
        lonmax=np.max(lon),
        lonmin=np.min(lon),
        latmax=np.max(lat),
        latmin=np.min(lat),
        country=np.asarray(country),
        name=name,
    )
    return SimpleNamespace(**result)


def areagrid(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Calculate grid of areas (m^2), given arrays of latitudes and longitudes.

    Args:
        lat: 1D array of latitudes.
        lon: 1D array of longitudes.

    Returns:
        np.ndarray: 2D array of areas of size lat x lon.

    Examples:
        >>> import utils.areagrid
        >>> lat = np.arange(50., 60., 1.)
        >>> lon = np.arange(0., 10., 1.)
        >>> area = utils.areagrid(lat, lon)
    """
    rad_earth = 6367500.0  # radius of Earth in m

    dlon = abs(np.mean(lon[1:] - lon[0:-1])) * np.pi / 180.0
    dlat = abs(np.mean(lat[1:] - lat[0:-1])) * np.pi / 180.0
    theta = np.pi * (90.0 - lat) / 180.0

    area = np.zeros((len(lat), len(lon)))

    for latI in range(len(lat)):
        if theta[latI] == 0.0 or np.isclose(theta[latI], np.pi):
            area[latI, :] = (rad_earth**2) * abs(np.cos(dlat / 2.0) - np.cos(0.0)) * dlon
        else:
            lat1 = theta[latI] - dlat / 2.0
            lat2 = theta[latI] + dlat / 2.0
            area[latI, :] = (rad_earth**2) * (np.cos(lat1) - np.cos(lat2)) * dlon

    return area


# ---------------------------------------------------------------------
# Flux period helpers used by legacy and PyMC post-processing code paths.
# These stay private because they are implementation details for mapping
# observation times onto available time-varying flux slices.
# ---------------------------------------------------------------------
def _normalize_flux_period(time_period: object) -> str | None:
    """Normalize a calendar alias or validate a positive fixed period.

    Args:
        time_period: Source period metadata to interpret.

    Returns:
        ``"yearly"`` or ``"monthly"`` for recognized calendar aliases, the
        stripped input for a positive fixed duration, or ``None`` when the
        value is missing or unsupported.
    """
    period = str(time_period).strip()
    normalized_period = period.casefold()
    if _flux_period_is_missing(time_period):
        return None
    if normalized_period in {"1 year", "year", "yearly", "annual", "annually"}:
        return "yearly"
    if normalized_period in {"1 month", "month", "monthly"}:
        return "monthly"

    try:
        fixed_period = pd.to_timedelta(period)
    except (TypeError, ValueError):
        return None
    if pd.isna(fixed_period) or fixed_period <= pd.Timedelta(0):
        return None
    return period


def _flux_period_is_missing(time_period: object) -> bool:
    """Return whether source period metadata contains no usable value."""
    if time_period is None or bool(pd.isna(time_period)):
        return True
    return str(time_period).strip().casefold() in {"", "nan", "nat", "none"}


def _infer_calendar_flux_period(times: xr.DataArray | np.ndarray) -> str | None:
    """Infer a regular yearly or monthly calendar period from timestamps.

    Args:
        times: Candidate flux-period start timestamps.

    Returns:
        ``"yearly"`` or ``"monthly"`` when every adjacent pair follows that
        calendar offset, otherwise ``None``.
    """
    values = times.values if isinstance(times, xr.DataArray) else times
    time_values = pd.DatetimeIndex(pd.to_datetime(values)).sort_values().unique()
    if len(time_values) <= 1:
        return None

    consecutive_times = pairwise(time_values)
    if all(end == start + pd.DateOffset(years=1) for start, end in consecutive_times):
        return "yearly"

    consecutive_times = pairwise(time_values)
    if all(end == start + pd.DateOffset(months=1) for start, end in consecutive_times):
        return "monthly"

    return None


def _infer_flux_period(times: xr.DataArray | np.ndarray, time_period: str | None = None) -> str:
    """Infer whether flux slices are yearly or monthly."""
    if time_period is not None:
        normalized_period = _normalize_flux_period(time_period)
        if normalized_period in ("yearly", "monthly"):
            return normalized_period

    calendar_period = _infer_calendar_flux_period(times)
    if calendar_period is not None:
        return calendar_period

    values = times.values if isinstance(times, xr.DataArray) else times
    time_values = pd.to_datetime(values)
    if len(time_values) <= 1:
        return "yearly"

    deltas = pd.Series(time_values).sort_values().diff().dropna()
    if deltas.empty:
        return "yearly"

    delta = deltas.mode().iloc[0]
    return "yearly" if delta >= pd.Timedelta(days=330) else "monthly"


def _map_times_to_available_period_positions(
    times: xr.DataArray | np.ndarray, available_times: xr.DataArray | np.ndarray, period: str
) -> np.ndarray:
    """Map timestamps onto contiguous period positions defined by available flux periods."""
    period_code = "Y" if period == "yearly" else "M"
    time_values = times.values if isinstance(times, xr.DataArray) else times
    available_time_values = (
        available_times.values if isinstance(available_times, xr.DataArray) else available_times
    )
    time_periods = pd.to_datetime(time_values).to_period(period_code)
    available_periods = pd.Index(pd.to_datetime(available_time_values).to_period(period_code).unique())

    if len(time_periods) == 0:
        return np.array([], dtype=int)

    missing = pd.Index(time_periods).difference(available_periods)
    if len(missing) > 0:
        period_label = "years" if period == "yearly" else "months"
        raise ValueError(
            f"Observation {period_label} {list(missing.astype(str))} are missing from available flux periods."
        )

    period_positions = {period_value: idx for idx, period_value in enumerate(available_periods)}
    return np.array([period_positions[period_value] for period_value in time_periods], dtype=int)
