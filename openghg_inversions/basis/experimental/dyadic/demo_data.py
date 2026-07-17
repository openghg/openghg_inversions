"""Load the repository's TAC/MHD fixtures for the experimental dyadic demo.

This adapter deliberately bypasses OpenGHG retrieval and standardisation.  It
reconstructs a fine-grid emissions sensitivity directly from committed test
data, while taking observation ordering and error information from the frozen
``make_inv_inputs`` regression fixture.  The result is intended only for a
reproducible local stochastic-search demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

from .multiscale import CoarsenedGrid, sum_coarsen_grid

_FROZEN_FILENAME = "frozen_mhd_tac_make_inv_inputs_hbmcmc.npz"
_MHD_FOOTPRINT_FILENAME = "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc"
_TAC_FOOTPRINT_FILENAME = "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc"
_FLUX_FILENAME = "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"
_FLUX_COORDINATE_ATOL = 1.0e-4
_PPB_SCALE = 1.0e9


@dataclass(frozen=True)
class DemoDesignData:
    """Fine-grid design and observations for the TAC/MHD dyadic SLS demo.

    Attributes:
        G: Emissions sensitivity with shape ``(observation, lat, lon)`` in ppb
            per fine-cell multiplier.
        y: Observed mole fractions in frozen observation order.
        error: Observation repeatability/error values used by the frozen
            inversion input.
        min_error: Frozen lower bounds for model-measurement mismatch error.
        sites: Upper-case site label for every observation.
        times: Observation timestamps in the same order as the first axis of
            ``G``.
        lat: Latitude coordinate adopted from the footprint grid.
        lon: Longitude coordinate adopted from the footprint grid.
    """

    G: npt.NDArray[np.floating]
    y: npt.NDArray[np.floating]
    error: npt.NDArray[np.floating]
    min_error: npt.NDArray[np.floating]
    sites: npt.NDArray[np.str_]
    times: npt.NDArray[np.datetime64]
    lat: npt.NDArray[np.floating]
    lon: npt.NDArray[np.floating]

    def coarsen(self, factor: int) -> CoarsenedGrid:
        """Return a sum-preserving spatial coarsening of ``G``.

        Args:
            factor: Positive block width along both spatial dimensions.

        Returns:
            Coarsened values and fine-cell support counts, including partial
            blocks at the northern and eastern boundaries.
        """
        return sum_coarsen_grid(self.G, factor)


def load_tac_mhd_demo_data(data_directory: Path) -> DemoDesignData:
    """Reconstruct the committed TAC/MHD fine-grid demo design.

    Args:
        data_directory: Directory containing the committed ``tests/data``
            frozen NPZ, MHD and TAC footprint NetCDF files, and annual flux
            NetCDF file.

    Returns:
        Validated design data in the exact 47-row frozen order: 23 MHD
        observations followed by 24 TAC observations.

    Raises:
        FileNotFoundError: If a required fixture does not exist.
        TypeError: If ``data_directory`` is not path-like.
        ValueError: If fixture keys, dimensions, coordinates, timestamps, row
            order, or numeric values violate the demo data contract.
    """
    try:
        directory = Path(data_directory)
    except TypeError as exc:
        raise TypeError("data_directory must be path-like.") from exc

    frozen_path = _required_fixture(directory, _FROZEN_FILENAME)
    mhd_path = _required_fixture(directory, _MHD_FOOTPRINT_FILENAME)
    tac_path = _required_fixture(directory, _TAC_FOOTPRINT_FILENAME)
    flux_path = _required_fixture(directory, _FLUX_FILENAME)

    y, error, min_error, site_indicator, times = _load_frozen_observations(frozen_path)
    mhd_mask = site_indicator == 0
    tac_mask = site_indicator == 1
    expected_indicator = np.concatenate((np.zeros(23, dtype=np.int64), np.ones(24, dtype=np.int64)))
    if not np.array_equal(site_indicator, expected_indicator):
        raise ValueError("frozen site indicators must contain 23 MHD rows followed by 24 TAC rows.")

    mhd_footprint, mhd_lat, mhd_lon = _load_footprint(
        mhd_path,
        variable="srr",
        times=times[mhd_mask],
    )
    tac_footprint, tac_lat, tac_lon = _load_footprint(
        tac_path,
        variable="fp",
        times=times[tac_mask],
    )
    _require_equal_grid(mhd_lat, mhd_lon, tac_lat, tac_lon, comparison="MHD and TAC footprints")

    flux, flux_lat, flux_lon = _load_annual_flux(flux_path)
    _require_close_grid(
        mhd_lat,
        mhd_lon,
        flux_lat,
        flux_lon,
        comparison="footprint and flux",
        atol=_FLUX_COORDINATE_ATOL,
    )

    footprints = np.empty((times.size, mhd_lat.size, mhd_lon.size), dtype=np.float32)
    footprints[mhd_mask] = mhd_footprint
    footprints[tac_mask] = tac_footprint
    G = footprints * flux[np.newaxis, :, :] * _PPB_SCALE

    _require_finite(G, name="G")
    sites = np.where(mhd_mask, "MHD", "TAC")
    return DemoDesignData(
        G=G,
        y=y,
        error=error,
        min_error=min_error,
        sites=sites,
        times=times,
        lat=mhd_lat,
        lon=mhd_lon,
    )


def _required_fixture(directory: Path, filename: str) -> Path:
    """Return a required fixture path after checking that it is a file.

    Args:
        directory: Parent fixture directory.
        filename: Required fixture filename.

    Returns:
        Path to the existing fixture.

    Raises:
        FileNotFoundError: If the fixture path is not a file.
    """
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required dyadic demo fixture does not exist: {path}")
    return path


def _load_frozen_observations(
    path: Path,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.int64],
    npt.NDArray[np.datetime64],
]:
    """Load and validate observation vectors from the frozen NPZ fixture.

    Args:
        path: Frozen ``make_inv_inputs`` NPZ path.

    Returns:
        Observation values, errors, minimum errors, site indicators, and
        timestamps as independent arrays.

    Raises:
        ValueError: If a required key is absent, vectors do not have 47 rows,
            values are non-finite, errors are not positive, timestamps repeat
            within a site, or site indicators contain unsupported values.
    """
    keys = (
        "mcmc__Y",
        "mcmc__error",
        "mcmc__min_error",
        "mcmc__siteindicator",
        "post__Ytime",
    )
    with np.load(path, allow_pickle=False) as frozen:
        missing = [key for key in keys if key not in frozen]
        if missing:
            raise ValueError(f"frozen observation fixture is missing keys: {missing!r}")
        y = np.asarray(frozen["mcmc__Y"], dtype=np.float64).copy()
        error = np.asarray(frozen["mcmc__error"], dtype=np.float64).copy()
        min_error = np.asarray(frozen["mcmc__min_error"], dtype=np.float64).copy()
        site_indicator = np.asarray(frozen["mcmc__siteindicator"], dtype=np.int64).copy()
        times = np.asarray(frozen["post__Ytime"], dtype="datetime64[ns]").copy()

    vectors = {
        "y": y,
        "error": error,
        "min_error": min_error,
        "site_indicator": site_indicator,
        "times": times,
    }
    invalid_shapes = {name: values.shape for name, values in vectors.items() if values.shape != (47,)}
    if invalid_shapes:
        raise ValueError(f"frozen observation vectors must each have shape (47,): {invalid_shapes!r}")
    _require_finite(y, name="y")
    _require_finite(error, name="error")
    _require_finite(min_error, name="min_error")
    if np.any(error <= 0.0):
        raise ValueError("error must contain only positive values.")
    if np.any(min_error <= 0.0):
        raise ValueError("min_error must contain only positive values.")
    if not np.isin(site_indicator, (0, 1)).all():
        raise ValueError("site indicators must contain only 0 (MHD) and 1 (TAC).")
    if np.isnat(times).any():
        raise ValueError("observation timestamps must be finite.")
    if any(
        np.unique(times[site_indicator == index]).size != np.count_nonzero(site_indicator == index)
        for index in (0, 1)
    ):
        raise ValueError("observation timestamps must be unique within each site.")
    return y, error, min_error, site_indicator, times


def _load_footprint(
    path: Path,
    *,
    variable: str,
    times: npt.NDArray[np.datetime64],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Load requested footprint rows and normalize them to ``time, lat, lon``.

    Args:
        path: Site footprint NetCDF path.
        variable: Footprint variable name in the source file.
        times: Exact site timestamps to select, in desired output order.

    Returns:
        Selected footprint values and their latitude and longitude coordinates.

    Raises:
        ValueError: If the variable, dimensions, timestamps, coordinates, or
            footprint values do not satisfy the demo contract.
    """
    with xr.open_dataset(path) as dataset:
        if variable not in dataset:
            raise ValueError(f"footprint fixture {path.name!r} does not contain variable {variable!r}.")
        footprint = dataset[variable]
        rename = {
            name: canonical
            for name, canonical in (("latitude", "lat"), ("longitude", "lon"))
            if name in footprint.dims
        }
        footprint = footprint.rename(rename)
        if set(footprint.dims) != {"time", "lat", "lon"} or footprint.ndim != 3:
            raise ValueError(
                f"footprint variable {variable!r} must normalize to dimensions time, lat, lon; "
                f"found {footprint.dims!r}."
            )

        source_times = np.asarray(footprint.coords["time"].values, dtype="datetime64[ns]")
        missing_times = times[~np.isin(times, source_times)]
        if missing_times.size:
            raise ValueError(f"footprint fixture {path.name!r} is missing requested times: {missing_times!r}")
        selected = footprint.sel(time=times).transpose("time", "lat", "lon").load()
        selected_times = np.asarray(selected.coords["time"].values, dtype="datetime64[ns]")
        if not np.array_equal(selected_times, times):
            raise ValueError(f"footprint fixture {path.name!r} did not preserve requested timestamp order.")
        values = np.asarray(selected.values, dtype=np.float32)
        lat = _coordinate_values(selected, "lat", path=path)
        lon = _coordinate_values(selected, "lon", path=path)

    if values.shape != (times.size, lat.size, lon.size):
        raise ValueError(f"footprint fixture {path.name!r} has inconsistent selected shape {values.shape!r}.")
    _require_finite(values, name=f"{path.name} footprint")
    return values, lat, lon


def _load_annual_flux(
    path: Path,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Load the single-time annual flux field as ``lat, lon``.

    Args:
        path: Annual flux NetCDF path.

    Returns:
        Flux values and their latitude and longitude coordinates.

    Raises:
        ValueError: If the flux variable is absent, dimensions are unexpected,
            the time dimension is not singular, or values are non-finite.
    """
    with xr.open_dataset(path) as dataset:
        if "flux" not in dataset:
            raise ValueError(f"flux fixture {path.name!r} does not contain variable 'flux'.")
        flux = dataset["flux"]
        rename = {
            name: canonical
            for name, canonical in (("latitude", "lat"), ("longitude", "lon"))
            if name in flux.dims
        }
        flux = flux.rename(rename)
        if set(flux.dims) != {"time", "lat", "lon"} or flux.ndim != 3:
            raise ValueError(f"flux must normalize to dimensions time, lat, lon; found {flux.dims!r}.")
        if flux.sizes["time"] != 1:
            raise ValueError("annual flux fixture must contain exactly one time value.")
        field = flux.isel(time=0).transpose("lat", "lon").load()
        values = np.asarray(field.values, dtype=np.float32)
        lat = _coordinate_values(field, "lat", path=path)
        lon = _coordinate_values(field, "lon", path=path)

    if values.shape != (lat.size, lon.size):
        raise ValueError(f"flux fixture {path.name!r} has inconsistent spatial shape {values.shape!r}.")
    _require_finite(values, name="flux")
    return values, lat, lon


def _coordinate_values(
    array: xr.DataArray,
    name: str,
    *,
    path: Path,
) -> npt.NDArray[np.floating]:
    """Return a finite, strictly increasing one-dimensional coordinate.

    Args:
        array: DataArray containing the coordinate.
        name: Coordinate and dimension name.
        path: Source path used in validation errors.

    Returns:
        Floating-point coordinate values.

    Raises:
        ValueError: If the coordinate is missing, not one-dimensional, empty,
            non-finite, or not strictly increasing.
    """
    if name not in array.coords or array.coords[name].dims != (name,):
        raise ValueError(f"fixture {path.name!r} must have a one-dimensional {name!r} coordinate.")
    values = np.asarray(array.coords[name].values)
    if values.size == 0 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"fixture {path.name!r} has an invalid {name!r} coordinate.")
    values = np.asarray(values, dtype=np.float64)
    _require_finite(values, name=f"{path.name} {name}")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError(f"fixture {path.name!r} {name!r} coordinate must be strictly increasing.")
    return values


def _require_equal_grid(
    reference_lat: npt.NDArray[np.floating],
    reference_lon: npt.NDArray[np.floating],
    candidate_lat: npt.NDArray[np.floating],
    candidate_lon: npt.NDArray[np.floating],
    *,
    comparison: str,
) -> None:
    """Require two latitude/longitude grids to be exactly equal.

    Args:
        reference_lat: Reference latitude coordinate.
        reference_lon: Reference longitude coordinate.
        candidate_lat: Candidate latitude coordinate.
        candidate_lon: Candidate longitude coordinate.
        comparison: Human-readable comparison name for errors.

    Raises:
        ValueError: If either coordinate differs exactly.
    """
    if not np.array_equal(reference_lat, candidate_lat) or not np.array_equal(reference_lon, candidate_lon):
        raise ValueError(f"{comparison} must use exactly equal latitude and longitude grids.")


def _require_close_grid(
    reference_lat: npt.NDArray[np.floating],
    reference_lon: npt.NDArray[np.floating],
    candidate_lat: npt.NDArray[np.floating],
    candidate_lon: npt.NDArray[np.floating],
    *,
    comparison: str,
    atol: float,
) -> None:
    """Require two grids to agree within a fixed absolute tolerance.

    Args:
        reference_lat: Reference latitude coordinate to retain in output.
        reference_lon: Reference longitude coordinate to retain in output.
        candidate_lat: Candidate latitude coordinate.
        candidate_lon: Candidate longitude coordinate.
        comparison: Human-readable comparison name for errors.
        atol: Absolute coordinate tolerance; relative tolerance is always zero.

    Raises:
        ValueError: If coordinate shapes differ or values exceed ``atol``.
    """
    lat_close = reference_lat.shape == candidate_lat.shape and np.allclose(
        reference_lat, candidate_lat, rtol=0.0, atol=atol
    )
    lon_close = reference_lon.shape == candidate_lon.shape and np.allclose(
        reference_lon, candidate_lon, rtol=0.0, atol=atol
    )
    if not lat_close or not lon_close:
        raise ValueError(f"{comparison} grids must agree within an absolute tolerance of {atol:g} degrees.")


def _require_finite(values: npt.ArrayLike, *, name: str) -> None:
    """Require every numeric value in an array to be finite.

    Args:
        values: Numeric values to validate.
        name: Human-readable field name for errors.

    Raises:
        ValueError: If any value is NaN or infinite.
    """
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")


__all__ = ["DemoDesignData", "load_tac_mhd_demo_data"]
