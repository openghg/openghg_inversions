"""Load the repository's TAC/MHD fixtures for the experimental dyadic demo.

These adapters deliberately bypass OpenGHG retrieval and standardisation. They
reconstruct fine-grid emissions sensitivities directly from committed test
data. The compact adapter takes observation ordering and error information from
the frozen ``make_inv_inputs`` regression fixture; the full-week adapter
reproduces the scientifically relevant hourly aggregation from the committed
raw observation variables. The results are intended only for reproducible local
stochastic-search demonstrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

from .multiscale import CoarsenedGrid, sum_coarsen_grid

_FROZEN_FILENAME = "frozen_mhd_tac_make_inv_inputs_hbmcmc.npz"
_MHD_OBSERVATION_FILENAME = "obs_mhd_ch4_10m_2019-01-01_2019-01-07_data.nc"
_TAC_OBSERVATION_FILENAME = "obs_tac_ch4_185m_2019-01-01_2019-02-01_data.nc"
_MHD_FOOTPRINT_FILENAME = "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc"
_TAC_FOOTPRINT_FILENAME = "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc"
_FLUX_FILENAME = "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"
_FLUX_COORDINATE_ATOL = 1.0e-4
_PPB_SCALE = 1.0e9
_ONE_HOUR = np.timedelta64(1, "h")
_FROZEN_ERROR_DESCRIPTION = (
    "Frozen one-day observation errors and percentile minimum-mismatch floors. "
    "Using max(error, min_error) as a fixed covariance is a demo benchmark, not "
    "the production model-error likelihood."
)
_WEEK_ERROR_DESCRIPTION = (
    "Hourly MHD repeatability-plus-variability and TAC pooled variability, with "
    "zero pooled TAC variability replaced by the full-week site median and a "
    "per-site median-minus-fifth-percentile minimum-mismatch floor. Using "
    "max(error, min_error) as a fixed covariance is a demo benchmark, not the "
    "production model-error likelihood."
)


@dataclass(frozen=True)
class DemoDesignData:
    """Fine-grid design and observations for the TAC/MHD dyadic SLS demo.

    Attributes:
        G: Emissions sensitivity with shape ``(observation, lat, lon)`` in ppb
            per fine-cell multiplier.
        y: Observed mole fractions in frozen observation order.
        error: Observation uncertainty used by the fixed demo benchmark.
        min_error: Lower bounds for model-measurement mismatch error.
        sites: Upper-case site label for every observation.
        times: Observation timestamps in the same order as the first axis of
            ``G``.
        lat: Latitude coordinate adopted from the footprint grid.
        lon: Longitude coordinate adopted from the footprint grid.
        benchmark_error_description: Human-readable definition and limitation
            of the fixed error quantities supplied for the demo objective.
    """

    G: npt.NDArray[np.floating]
    y: npt.NDArray[np.floating]
    error: npt.NDArray[np.floating]
    min_error: npt.NDArray[np.floating]
    sites: npt.NDArray[np.str_]
    times: npt.NDArray[np.datetime64]
    lat: npt.NDArray[np.floating]
    lon: npt.NDArray[np.floating]
    benchmark_error_description: str

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
        benchmark_error_description=_FROZEN_ERROR_DESCRIPTION,
    )


def load_tac_mhd_week_demo_data(data_directory: Path) -> DemoDesignData:
    """Reconstruct an aligned full-week TAC/MHD benchmark design.

    Raw observations are aggregated into the exact hourly footprint bins.
    MHD mole fractions are averaged without weights, its repeatability is
    propagated as the uncertainty of the mean, and within-hour variability is
    included in quadrature. Single-measurement MHD hours retain zero within-hour
    variability because repeatability still supplies a positive error. TAC
    stores a mean, variability, and
    contributing sample count for each raw row; its hourly mean and variability
    are therefore pooled using those counts. Missing hourly observations are
    omitted rather than imputed, and rows remain in site-major order (MHD
    followed by TAC).

    The returned ``min_error`` is the reproducible per-site percentile floor
    used by the existing demo: median mole fraction minus its fifth percentile.
    It is not an inferred production model error. In particular, using
    ``max(error, min_error)`` as a fixed diagonal covariance is only a benchmark
    approximation to the production likelihood.

    Args:
        data_directory: Directory containing the committed raw observation,
            hourly footprint, and annual flux NetCDF fixtures.

    Returns:
        Validated full-week design data for every exactly aligned observation
        and footprint hour, in site-major order.

    Raises:
        FileNotFoundError: If a required fixture does not exist.
        TypeError: If ``data_directory`` is not path-like.
        ValueError: If variables, grids, times, values, or aggregation metadata
            violate the full-week demo contract.
    """
    try:
        directory = Path(data_directory)
    except TypeError as exc:
        raise TypeError("data_directory must be path-like.") from exc

    mhd_observation_path = _required_fixture(directory, _MHD_OBSERVATION_FILENAME)
    tac_observation_path = _required_fixture(directory, _TAC_OBSERVATION_FILENAME)
    mhd_footprint_path = _required_fixture(directory, _MHD_FOOTPRINT_FILENAME)
    tac_footprint_path = _required_fixture(directory, _TAC_FOOTPRINT_FILENAME)
    flux_path = _required_fixture(directory, _FLUX_FILENAME)

    mhd_footprint_times = _load_footprint_times(mhd_footprint_path, variable="srr")
    tac_footprint_times = _load_footprint_times(tac_footprint_path, variable="fp")
    mhd_times, mhd_y, mhd_error = _load_hourly_mhd_observations(
        mhd_observation_path,
        mhd_footprint_times,
    )
    tac_times, tac_y, tac_error = _load_hourly_tac_observations(
        tac_observation_path,
        tac_footprint_times,
    )

    flux, flux_lat, flux_lon = _load_annual_flux(flux_path)
    row_count = mhd_times.size + tac_times.size
    G = np.empty((row_count, flux_lat.size, flux_lon.size), dtype=np.float32)

    mhd_footprint, mhd_lat, mhd_lon = _load_footprint(
        mhd_footprint_path,
        variable="srr",
        times=mhd_times,
    )
    _require_close_grid(
        mhd_lat,
        mhd_lon,
        flux_lat,
        flux_lon,
        comparison="MHD footprint and flux",
        atol=_FLUX_COORDINATE_ATOL,
    )
    _scale_footprints_in_place(mhd_footprint, flux)
    G[: mhd_times.size] = mhd_footprint

    tac_footprint, tac_lat, tac_lon = _load_footprint(
        tac_footprint_path,
        variable="fp",
        times=tac_times,
    )
    _require_equal_grid(mhd_lat, mhd_lon, tac_lat, tac_lon, comparison="MHD and TAC footprints")
    _require_close_grid(
        tac_lat,
        tac_lon,
        flux_lat,
        flux_lon,
        comparison="TAC footprint and flux",
        atol=_FLUX_COORDINATE_ATOL,
    )
    _scale_footprints_in_place(tac_footprint, flux)
    G[mhd_times.size :] = tac_footprint

    y = np.concatenate((mhd_y, tac_y))
    error = np.concatenate((mhd_error, tac_error))
    min_error = np.concatenate(
        (
            np.full(mhd_y.shape, _percentile_min_error(mhd_y)),
            np.full(tac_y.shape, _percentile_min_error(tac_y)),
        )
    )
    sites = np.concatenate(
        (
            np.full(mhd_times.shape, "MHD", dtype="<U3"),
            np.full(tac_times.shape, "TAC", dtype="<U3"),
        )
    )
    times = np.concatenate((mhd_times, tac_times))

    _require_finite(G, name="G")
    _require_finite(y, name="y")
    _require_positive(error, name="error")
    _require_positive(min_error, name="min_error")
    return DemoDesignData(
        G=G,
        y=y,
        error=error,
        min_error=min_error,
        sites=sites,
        times=times,
        lat=mhd_lat,
        lon=mhd_lon,
        benchmark_error_description=_WEEK_ERROR_DESCRIPTION,
    )


def _load_footprint_times(path: Path, *, variable: str) -> npt.NDArray[np.datetime64]:
    """Load and validate the hourly timestamps for one footprint fixture.

    Args:
        path: Site footprint NetCDF path.
        variable: Footprint variable whose time dimension is required.

    Returns:
        Strictly increasing, unique hourly timestamps.

    Raises:
        ValueError: If the variable or time coordinate is missing, timestamps
            are invalid, or the fixture is not exactly hourly.
    """
    with xr.open_dataset(path) as dataset:
        if variable not in dataset:
            raise ValueError(f"footprint fixture {path.name!r} does not contain variable {variable!r}.")
        footprint = dataset[variable]
        if "time" not in footprint.dims or "time" not in footprint.coords:
            raise ValueError(f"footprint variable {variable!r} must have a time dimension and coordinate.")
        times = np.asarray(footprint.coords["time"].values, dtype="datetime64[ns]").copy()

    _require_valid_times(times, name=f"{path.name} footprint")
    if times.size > 1 and not np.all(np.diff(times) == _ONE_HOUR):
        raise ValueError(f"footprint fixture {path.name!r} must use exact hourly timestamps.")
    return times


def _load_hourly_mhd_observations(
    path: Path,
    footprint_times: npt.NDArray[np.datetime64],
) -> tuple[
    npt.NDArray[np.datetime64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Aggregate MHD measurements and uncertainties into footprint hours.

    Args:
        path: Raw MHD observation NetCDF path.
        footprint_times: Exact footprint hours eligible for output.

    Returns:
        Aligned timestamps, hourly mean mole fractions, and observation errors.
        Missing observation hours are omitted rather than imputed.

    Raises:
        ValueError: If required variables or values are invalid, or an
            aggregated timestamp cannot be matched exactly to a footprint.
    """
    observations = _load_observation_window(
        path,
        variables=("ch4", "ch4_repeatability"),
        footprint_times=footprint_times,
    )
    mole_fraction = observations["ch4"]
    repeatability = observations["ch4_repeatability"]
    hourly_mean = mole_fraction.resample(time="1h").mean(skipna=False)
    hourly_repeatability = (repeatability**2).resample(time="1h").sum(skipna=False) ** 0.5 / (
        repeatability.resample(time="1h").count()
    )
    hourly_variability = mole_fraction.resample(time="1h").std(skipna=False)
    hourly_error = (hourly_repeatability**2 + hourly_variability**2) ** 0.5
    return _align_hourly_observations(
        hourly_mean,
        hourly_error,
        footprint_times,
        site="MHD",
    )


def _load_hourly_tac_observations(
    path: Path,
    footprint_times: npt.NDArray[np.datetime64],
) -> tuple[
    npt.NDArray[np.datetime64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Pool TAC summary measurements into footprint hours.

    Each committed TAC row is itself a mean and population variability for a
    recorded number of observations. Hourly means and variabilities therefore
    use the standard pooled first and second moments rather than treating rows
    as equally weighted raw measurements.

    Args:
        path: Raw TAC observation NetCDF path.
        footprint_times: Exact footprint hours eligible for output.

    Returns:
        Aligned timestamps, pooled hourly mole fractions, and pooled hourly
        variability used as observation error. Missing hours are omitted rather
        than imputed.

    Raises:
        ValueError: If required variables, counts, or values are invalid, or an
            aggregated timestamp cannot be matched exactly to a footprint.
    """
    observations = _load_observation_window(
        path,
        variables=("ch4", "ch4_variability", "ch4_number_of_observations"),
        footprint_times=footprint_times,
    )
    mole_fraction = observations["ch4"]
    variability = observations["ch4_variability"]
    counts = observations["ch4_number_of_observations"]
    if bool((counts <= 0.0).any()):
        raise ValueError("TAC number_of_observations must contain only positive values.")

    hourly_counts = counts.resample(time="1h").sum(skipna=False)
    hourly_mean = (mole_fraction * counts).resample(time="1h").sum(skipna=False) / hourly_counts
    hourly_second_moment = ((variability**2 + mole_fraction**2) * counts).resample(time="1h").sum(
        skipna=False
    ) / hourly_counts
    hourly_variance = xr.where(
        hourly_second_moment >= hourly_mean**2,
        hourly_second_moment - hourly_mean**2,
        0.0,
    )
    hourly_variability = hourly_variance**0.5
    hourly_variability = _replace_zero_variability(hourly_variability, site="TAC")
    return _align_hourly_observations(
        hourly_mean,
        hourly_variability,
        footprint_times,
        site="TAC",
    )


def _load_observation_window(
    path: Path,
    *,
    variables: tuple[str, ...],
    footprint_times: npt.NDArray[np.datetime64],
) -> xr.Dataset:
    """Load required raw variables over the footprint time window.

    Args:
        path: Raw observation NetCDF path.
        variables: Variables required for site-specific aggregation.
        footprint_times: Footprint timestamps defining the inclusive first hour
            and exclusive end of the observation window.

    Returns:
        In-memory dataset containing only required variables and raw rows inside
        the footprint window.

    Raises:
        ValueError: If times or variables are absent, non-finite, duplicated, or
            outside a usable footprint window.
    """
    _require_valid_times(footprint_times, name="footprint")
    window_start = footprint_times[0]
    window_end = footprint_times[-1] + _ONE_HOUR
    with xr.open_dataset(path) as dataset:
        missing = [variable for variable in variables if variable not in dataset]
        if missing:
            raise ValueError(f"observation fixture {path.name!r} is missing variables: {missing!r}")
        if "time" not in dataset.coords or dataset.coords["time"].dims != ("time",):
            raise ValueError(
                f"observation fixture {path.name!r} must have a one-dimensional time coordinate."
            )
        source_times = np.asarray(dataset.coords["time"].values, dtype="datetime64[ns]")
        _require_valid_times(source_times, name=f"{path.name} observation")
        selected_indices = np.flatnonzero((source_times >= window_start) & (source_times < window_end))
        if selected_indices.size == 0:
            raise ValueError(f"observation fixture {path.name!r} has no rows in the footprint time window.")
        observations = dataset[list(variables)].isel(time=selected_indices).load()

    for variable in variables:
        values = np.asarray(observations[variable].values)
        _require_finite(values, name=f"{path.name} {variable}")
    return observations


def _replace_zero_variability(variability: xr.DataArray, *, site: str) -> xr.DataArray:
    """Replace single-observation zero variability with the site median.

    Args:
        variability: Hourly variability, potentially containing NaN gaps and
            zeros for hours with one contributing observation.
        site: Site label used in validation errors.

    Returns:
        Variability with finite zeros replaced by the median positive hourly
        variability. NaN gaps remain NaN and are omitted during alignment.

    Raises:
        ValueError: If no positive finite hourly variability is available.
    """
    values = np.asarray(variability.values, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError(f"{site} observations do not provide positive hourly variability.")
    return variability.where(variability != 0.0, float(np.median(positive)))


def _align_hourly_observations(
    mole_fraction: xr.DataArray,
    error: xr.DataArray,
    footprint_times: npt.NDArray[np.datetime64],
    *,
    site: str,
) -> tuple[
    npt.NDArray[np.datetime64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Select finite hourly observations on exactly matching footprint times.

    Args:
        mole_fraction: Hourly observed mole fraction.
        error: Hourly observation error aligned to ``mole_fraction``.
        footprint_times: Eligible footprint timestamps in desired order.
        site: Site label used in validation errors.

    Returns:
        Exact shared timestamps and corresponding finite observation vectors.
        Footprint hours without valid observations are intentionally absent.

    Raises:
        ValueError: If dimensions differ, finite aggregated observations fall
            outside the footprint timestamps, or no rows remain.
    """
    if mole_fraction.dims != ("time",) or error.dims != ("time",):
        raise ValueError(f"{site} aggregated observations and errors must be one-dimensional over time.")
    if not np.array_equal(mole_fraction.coords["time"].values, error.coords["time"].values):
        raise ValueError(f"{site} aggregated observations and errors must use exactly equal timestamps.")

    values = np.asarray(mole_fraction.values, dtype=np.float64)
    errors = np.asarray(error.values, dtype=np.float64)
    aggregate_times = np.asarray(mole_fraction.coords["time"].values, dtype="datetime64[ns]")
    valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0.0)
    available_times = aggregate_times[valid]
    unexpected = available_times[~np.isin(available_times, footprint_times)]
    if unexpected.size:
        raise ValueError(f"{site} hourly observations lack exact footprint matches: {unexpected!r}")
    aligned_times = footprint_times[np.isin(footprint_times, available_times)]
    if aligned_times.size == 0:
        raise ValueError(f"{site} has no exactly aligned hourly observation and footprint rows.")

    valid_mole_fraction = mole_fraction.isel(time=np.flatnonzero(valid)).sel(time=aligned_times)
    valid_error = error.isel(time=np.flatnonzero(valid)).sel(time=aligned_times)
    return (
        np.asarray(aligned_times, dtype="datetime64[ns]"),
        np.asarray(valid_mole_fraction.values, dtype=np.float64),
        np.asarray(valid_error.values, dtype=np.float64),
    )


def _scale_footprints_in_place(
    footprints: npt.NDArray[np.float32],
    flux: npt.NDArray[np.float32],
) -> None:
    """Convert footprint rows to ppb emissions sensitivities in place.

    Args:
        footprints: Footprint rows with shape ``(observation, lat, lon)``.
        flux: Annual flux field with shape ``(lat, lon)``.

    Raises:
        ValueError: If the spatial shapes do not agree.
    """
    if footprints.shape[1:] != flux.shape:
        raise ValueError("footprint and flux arrays must use the same spatial shape.")
    np.multiply(footprints, flux[np.newaxis, :, :], out=footprints)
    footprints *= _PPB_SCALE


def _percentile_min_error(values: npt.NDArray[np.floating]) -> float:
    """Return the median-minus-fifth-percentile mismatch floor for one site.

    Args:
        values: Finite observed mole fractions for one site and selected time
            window.

    Returns:
        Positive percentile-based minimum mismatch error.

    Raises:
        ValueError: If values are empty, non-finite, or do not produce a
            positive floor.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("percentile min_error requires at least one observation.")
    _require_finite(values, name="percentile min_error observations")
    result = float(np.quantile(values, 0.5) - np.quantile(values, 0.05))
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("percentile min_error must be finite and positive.")
    return result


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


def _require_positive(values: npt.ArrayLike, *, name: str) -> None:
    """Require every numeric value in an array to be finite and positive.

    Args:
        values: Numeric values to validate.
        name: Human-readable field name for errors.

    Raises:
        ValueError: If any value is non-finite, zero, or negative.
    """
    array = np.asarray(values)
    _require_finite(array, name=name)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")


def _require_valid_times(values: npt.ArrayLike, *, name: str) -> None:
    """Require a non-empty, strictly increasing datetime vector.

    Args:
        values: Values expected to form a one-dimensional datetime vector.
        name: Human-readable field name for errors.

    Raises:
        ValueError: If values are empty, not one-dimensional, contain ``NaT``,
            repeat, or are not sorted strictly increasingly.
    """
    times = np.asarray(values, dtype="datetime64[ns]")
    if times.ndim != 1 or times.size == 0:
        raise ValueError(f"{name} timestamps must be a non-empty one-dimensional vector.")
    if np.isnat(times).any():
        raise ValueError(f"{name} timestamps must not contain NaT.")
    if times.size > 1 and np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError(f"{name} timestamps must be unique and strictly increasing.")


__all__ = ["DemoDesignData", "load_tac_mhd_demo_data", "load_tac_mhd_week_demo_data"]
