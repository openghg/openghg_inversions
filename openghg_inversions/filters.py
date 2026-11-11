"""Functions for filtering data.

All filters are accessed and applied to data via the `filtering` function.

New filters are registered using `@register_filter`.
A filter function should accept as arguments: an xr.Dataset, a bool called "keep_missing",
and any keyword options that callers may supply through ``filter_options``.

To see the available filters call `list_filters`.
"""

import inspect
import logging
import re
from typing import Any, cast
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.utils import combine_datasets


logger = logging.Logger(__name__)


# this dictionary will be populated by using the decorator `register_filter`
filtering_functions: dict[str, Callable] = {}


def register_filter(filt: Callable) -> Callable:
    """Decorator function to register filters.

    Args:
        filt: filter function to register

    Returns:
        filt, the input function (no modifications made)


    For instance, the following use of `register_filter` as a decorator adds `my_new_filter`
    to the `filtering_functions` dictionary, under the key "my_new_filter":

    >>> @register_filter
        def my_new_filter(data):
            return data
    >>> "my_new_filter" in filtering_functions
    True

    """
    filtering_functions[filt.__name__] = filt
    return filt


def list_filters() -> None:
    """Print a list of the available filters with a short description."""
    spacing = max([len(k) for k in filtering_functions]) + 4

    print("All available filters:")
    for k, v in filtering_functions.items():
        # print function name and first line of docstring
        try:
            docstring = v.__doc__
            first_line_of_docstring = (
                docstring.strip().split("\n")[0] if docstring is not None else "No docstring"
            )
        except AttributeError:
            first_line_of_docstring = "No docstring"

        print(f"\t{k:{spacing}}{first_line_of_docstring}")


def filtering(
    datasets_in: dict,
    filters: str | None | dict[str, str | None | list[str | None]] | list[str | None],
    keep_missing: bool = False,
    filter_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict:
    """Applies time filtering to all datasets in `datasets_in`.

    If `filters` is a list, the same filters are applied to all sites. If `filters` is a dict
    with site codes as keys, then the filters applied to each site depend on the list supplied
    for that site.

    Filters supplied in a list are applied in order. For example, select
    daytime observations with sufficiently high boundary-layer heights using
    ``filtering(datasets_dictionary, ["daytime", "pblh_min"])``.

    If a site is `datasets_in` is not in `filters`, then no filters are applied to that site.

    Args:
        datasets_in: dictionary of datasets containing output from ModelScenario.footprints_merge().
        filters: filters to apply to the datasets. Either a list of filters, which will be applied to every site,
            or a dictionary of lists of the form  {<site code>: [filter1, filter2, ...]}, with specific filters to
            be applied at each site. Use the `list_filters` function to list available filters.
        keep_missing: if True, drop missing data
        filter_options: keyword options for named filters, applied wherever that
            filter runs; for example ``{"local_influence": {"threshold": 0.15}}``.
            Options for filters that are not requested are validated but unused.

    Returns:
        dict in same format as datasets_in, with filters applied

    Raises:
        KeyError: If a requested filter, or a filter named in ``filter_options``,
            is not registered.
        TypeError: If ``filter_options`` names an option a filter does not accept.

    """
    options = validate_filter_options(filter_options)
    if filters is None or filters == [] or filters == {}:
        return datasets_in

    # Get list of sites
    sites = [key for key in list(datasets_in.keys()) if key[0] != "."]

    # Put the filters in a dict of list
    if not isinstance(filters, dict):
        if not isinstance(filters, list):
            filters = [filters]  # type: ignore
        for filt in filters:
            if filt is None or filt not in filtering_functions:
                raise KeyError(filt)
        filters = dict.fromkeys(sites, filters)  # type: ignore
    else:
        for site, filt in filters.items():
            if filt is not None and not isinstance(filt, list):
                filters[site] = [filt]

    filter_map = cast(dict[str, list[str | None] | None], filters)

    # Validate active requests before a preceding filter can execute data.
    for site in sites:
        for filt in filter_map.get(site) or []:
            if filt is None or filt not in filtering_functions:
                raise KeyError(filt)

    # Check that filters are defined for all sites
    # TODO: just set filters for missing sites to None?
    tmp = [(site in filter_map) for site in sites]
    if not all(tmp):
        msg = f"Missing entry for sites {np.array(sites)[~np.array(tmp)]} in filters."
        logger.warning(msg)

    datasets = datasets_in.copy()

    # Apply filtering
    # NOTE: we only loop over sites that are in the filters dict
    # so not all sites must be specified
    for site, site_filters in filter_map.items():
        if site_filters is not None and site in sites:
            for filt in site_filters:
                n_nofilter = datasets[site].time.values.shape[0]

                datasets[site] = filtering_functions[filt](
                    datasets[site], keep_missing=keep_missing, **options.get(filt, {})
                )

                n_filter = datasets[site].time.values.shape[0]
                n_dropped = n_nofilter - n_filter
                perc_dropped = np.round(n_dropped / n_nofilter * 100, 2)
                print(f"{filt} filter removed {n_dropped} ({perc_dropped} %) obs at site {site}")
                if n_filter == 0:
                    break  # no values left, so we won't apply remaining filters

    return datasets


def validate_filter_options(
    filter_options: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Return owned filter options after checking them against filter signatures.

    Args:
        filter_options: Mapping from registered filter names to keyword options,
            or ``None``.

    Returns:
        New mapping with owned option dictionaries; empty for ``None``.

    Raises:
        KeyError: If a filter name is not registered.
        TypeError: If an option is reserved or not accepted by the filter.
    """
    if filter_options is None:
        return {}
    options: dict[str, dict[str, Any]] = {}
    for name, values in filter_options.items():
        if name not in filtering_functions:
            raise KeyError(name)
        options[name] = dict(values)
        reserved = {"dataset", "keep_missing"}.intersection(options[name])
        if reserved:
            raise TypeError(f"Filter option(s) {sorted(reserved)!r} cannot be set for {name!r}.")
        try:
            inspect.signature(filtering_functions[name]).bind(None, **options[name])
        except TypeError as exc:
            raise TypeError(f"Invalid options for filter {name!r}: {exc}") from exc
    return options


def _local_solar_time(dataset: xr.Dataset) -> list[int]:
    """Return local solar hours without modifying observation timestamps.

    Args:
        dataset: borrowed observations with timestamps and release longitude

    Returns:
        list of hours of the day for each time value in dataset.time
    """
    sitelon = dataset.release_lon.values[0]
    # convert lon to [-180,180], so time offset is negative west of 0 degrees
    if sitelon > 180:
        sitelon -= 360.0
    local_time = dataset.time + pd.Timedelta(minutes=float(24 * 60 * sitelon / 360.0))
    hours = local_time.dt.hour.values
    return list(hours)


@register_filter
def daytime(dataset: xr.Dataset, keep_missing: bool = False) -> xr.Dataset:
    """Subset local solar hours 11 through 15, preserving observation timestamps.

    Args:
        dataset: dataset to filter
        keep_missing: if True, drop time points removed by filter

    Returns:
        filtered dataset
    """
    hours = _local_solar_time(dataset)
    ti = [i for i, h in enumerate(hours) if h >= 11 and h <= 15]

    if keep_missing:
        dataset_temp = dataset[dict(time=ti)]
        dataset_out = dataset_temp.reindex_like(dataset)
        return dataset_out
    else:
        return dataset[dict(time=ti)]


@register_filter
def daytime9to5(dataset: xr.Dataset, keep_missing: bool = False) -> xr.Dataset:
    """Subset local solar hours 9 through 17, preserving observation timestamps.

    Args:
        dataset: dataset to filter
        keep_missing: if True, drop time points removed by filter

    Returns:
        filtered dataset
    """
    hours = _local_solar_time(dataset)
    ti = [i for i, h in enumerate(hours) if h >= 9 and h <= 17]

    if keep_missing:
        dataset_temp = dataset[dict(time=ti)]
        dataset_out = dataset_temp.reindex_like(dataset)
        return dataset_out
    else:
        return dataset[dict(time=ti)]


@register_filter
def nighttime(dataset: xr.Dataset, keep_missing: bool = False) -> xr.Dataset:
    """Subset local solar hours 23 through 3, preserving observation timestamps.

    Args:
        dataset: dataset to filter
        keep_missing: if True, drop time points removed by filter

    Returns:
        filtered dataset
    """
    hours = _local_solar_time(dataset)
    ti = [i for i, h in enumerate(hours) if h >= 23 or h <= 3]

    if keep_missing:
        dataset_temp = dataset[dict(time=ti)]
        dataset_out = dataset_temp.reindex_like(dataset)
        return dataset_out
    else:
        return dataset[dict(time=ti)]


@register_filter
def noon(dataset: xr.Dataset, keep_missing: bool = False) -> xr.Dataset:
    """Select local solar hour 12, preserving observation timestamps.

    Args:
        dataset: dataset to filter
        keep_missing: if True, drop time points removed by filter

    Returns:
        filtered dataset
    """
    hours = _local_solar_time(dataset)
    ti = [i for i, h in enumerate(hours) if h == 12]

    if keep_missing:
        dataset_temp = dataset[dict(time=ti)]
        dataset_out = dataset_temp.reindex_like(dataset)
        return dataset_out
    else:
        return dataset[dict(time=ti)]


LOCAL_RATIO_STANDARD_NAME = "local_ratio"
"""Legacy ``standard_name`` of a cached 5×5 local-ratio variable."""


def local_ratio_name(window_size: int) -> str:
    """Return the cached variable name for one square local-ratio window.

    Args:
        window_size: Positive odd number of grid cells along each window side.

    Returns:
        Name of the form ``local_ratio_{n}x{n}``.

    Raises:
        ValueError: If ``window_size`` is not a positive odd integer.
    """
    _validate_window_size(window_size)
    return f"local_ratio_{window_size}x{window_size}"


def _validate_window_size(window_size: int) -> None:
    """Reject window sizes that do not centre on the release cell."""
    if isinstance(window_size, bool) or not isinstance(window_size, int | np.integer):
        raise ValueError(f"Local-ratio window size must be an integer; got {window_size!r}.")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"Local-ratio window size must be a positive odd integer; got {window_size!r}.")


def local_ratios(dataset: xr.Dataset, window_sizes: Iterable[int] = (5,)) -> xr.Dataset:
    """Return lazy local-to-total footprint ratios for square release-point windows.

    For each observation, the ratio is the integrated footprint summed over the
    ``n``×``n`` grid cells centred on the release cell, divided by the
    footprint summed over the whole domain. The release cell is the grid cell
    whose centre is nearest to ``release_lat``/``release_lon``; windows are
    clipped at the domain edge. The ratio is zero when the release point lies
    outside the grid or the whole-domain footprint sum is zero.

    Because the denominator is the whole-domain footprint, compute the ratios
    before any nested-domain masking or cropping of ``fp``. The result can be
    merged into observation datasets so :func:`local_influence` runs without
    ``fp``.

    Args:
        dataset: Borrowed dataset with an integrated footprint ``fp`` on
            ``lat``/``lon``/``time`` and ``release_lat``/``release_lon``
            aligned to ``time``. It is not modified; Dask-backed inputs stay lazy.
        window_sizes: Positive odd window widths in grid cells.

    Returns:
        Dataset with one ``local_ratio_{n}x{n}`` variable on ``time`` per
        window size, sharing the input's Dask graph when the input is lazy.

    Raises:
        ValueError: If a window size is invalid or the footprint grid has fewer
            than two cells along ``lat`` or ``lon``.
    """
    sizes = tuple(dict.fromkeys(window_sizes))
    for size in sizes:
        _validate_window_size(size)
    if dataset.sizes.get("lat", 0) < 2 or dataset.sizes.get("lon", 0) < 2:
        raise ValueError("Local ratios need at least two footprint grid cells along lat and lon.")

    fp = dataset["fp"]
    lat = dataset["lat"]
    lon = dataset["lon"]
    dlat = abs(float(lat[1] - lat[0]))
    dlon = abs(float(lon[1] - lon[0]))
    # Missing release coordinates never match a cell, so their ratio is zero.
    lat_distance = abs(lat - dataset["release_lat"]).fillna(np.inf)
    lon_distance = abs(lon - dataset["release_lon"]).fillna(np.inf)
    matched = (lat_distance.min("lat") < dlat / 2) & (lon_distance.min("lon") < dlon / 2)
    lat_index = cast(xr.DataArray, lat_distance.argmin("lat"))
    lon_index = cast(xr.DataArray, lon_distance.argmin("lon"))
    lat_offset = abs(xr.DataArray(np.arange(lat.size), dims="lat") - lat_index)
    lon_offset = abs(xr.DataArray(np.arange(lon.size), dims="lon") - lon_index)

    total = fp.sum(("lat", "lon"))
    nonzero = matched & (total != 0)
    ratios = {}
    for size in sizes:
        half_width = size // 2
        window = (lat_offset <= half_width) & (lon_offset <= half_width)
        local = fp.where(window, 0.0).sum(("lat", "lon"))
        ratio = xr.where(nonzero, local / total.where(total != 0), 0.0)
        ratio = ratio.drop_vars([name for name in ratio.coords if name != "time"])
        ratio.attrs = {
            "long_name": f"{size}x{size} local-to-total footprint ratio",
            "units": "1",
            "window_size": size,
        }
        ratios[local_ratio_name(size)] = ratio
    return xr.Dataset(ratios)


def _local_ratio_series(dataset: xr.Dataset, window_size: int) -> xr.DataArray:
    """Return a cached local-ratio series, or compute it from ``fp``."""
    name = local_ratio_name(window_size)
    if name in dataset.data_vars:
        return dataset[name]
    if window_size == 5:
        legacy = dataset.filter_by_attrs(standard_name=LOCAL_RATIO_STANDARD_NAME)
        if legacy.data_vars:
            return legacy[next(iter(legacy.data_vars))]
    if "fp" not in dataset.data_vars:
        raise ValueError(
            f"The local_influence filter needs a cached {name!r} variable or a raw footprint 'fp'."
        )
    return local_ratios(dataset, (window_size,))[name]


@register_filter
def local_influence(
    dataset: xr.Dataset, keep_missing: bool = False, window_size: int = 5, threshold: float = 0.1,
) -> xr.Dataset:
    """Subset for times when "local influence" is at or below threshold.

    Local influence is the footprint in a ``window_size``×``window_size`` grid
    window around the release point, as a fraction of the whole-domain
    footprint. The filter reads a cached ``local_ratio_{n}x{n}`` series when
    present (or, for 5×5, a variable with ``standard_name="local_ratio"``), and
    otherwise computes it with :func:`local_ratios`. Only the ratio series is
    computed; other variables keep their storage.

    Args:
        dataset: dataset to filter
        keep_missing: if True, keep removed times with missing ``mf``
        window_size: positive odd window width in grid cells
        threshold: largest local ratio retained, between zero and one

    Returns:
        filtered dataset

    Raises:
        ValueError: If the window or threshold is invalid, or the ratio is
            neither cached nor computable from ``fp``.
    """
    _validate_window_size(window_size)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Local-influence threshold must lie between zero and one; got {threshold!r}.")
    keep = np.asarray(_local_ratio_series(dataset, window_size) <= threshold)
    ti = np.flatnonzero(keep)
    if keep_missing is True:
        mf_data_array = dataset.mf
        dataset_temp = dataset.drop_vars("mf")

        dataarray_temp = mf_data_array[dict(time=ti)]

        mf_ds = xr.Dataset(
            {"mf": (["time"], dataarray_temp.data)}, coords={"time": (dataarray_temp.coords["time"])}
        )

        dataset_out = combine_datasets(dataset_temp, mf_ds, method=None)
        return dataset_out
    else:
        return dataset[dict(time=ti)]


@register_filter
def pblh_min(dataset: xr.Dataset, pblh_threshold: float = 200.0, keep_missing: bool = False) -> xr.Dataset:
    """Subset for times when the PBLH is greater than 200m.

    Args:
        dataset: dataset to filter
        pblh_threshold: filter will discard times where PBLH/atmosphere boundary layer thickness is below pblh_threshold
        keep_missing: if True, drop time points removed by filter

    Returns:
        filtered dataset
    """
    pblh_da = dataset.PBLH if "PBLH" in dataset.data_vars else dataset.atmosphere_boundary_layer_thickness

    ti = [i for i, pblh in enumerate(pblh_da) if pblh > pblh_threshold]

    if keep_missing is True:
        mf_data_array = dataset.mf
        dataset_temp = dataset.drop("mf")

        dataarray_temp = mf_data_array[dict(time=ti)]

        mf_ds = xr.Dataset(
            {"mf": (["time"], dataarray_temp)}, coords={"time": (dataarray_temp.coords["time"])}
        )

        dataset_out = combine_datasets(dataset_temp, mf_ds, method=None)
        return dataset_out
    else:
        return dataset[dict(time=ti)]


@register_filter
def pblh_inlet_diff(
    dataset: xr.Dataset, diff_threshold: float = 50.0, keep_missing: bool = False
) -> xr.Dataset:
    """Subset for times when observations are taken at a height of less than 50 m below the PBLH.

    Args:
        dataset: dataset to filter
        diff_threshold: filter will discard times where obs. are taken at a height of less than diff_threshold below PBLH
        keep_missing: if True, drop time points removed by filter

    Returns:
        filtered dataset
    """
    if "inlet_height_magl" in dataset.attrs:
        inlet_height = dataset.inlet_height_magl
    elif "inlet" in dataset.attrs:
        m = re.search(r"\d+", dataset.attrs["inlet"])
        if m is not None:
            inlet_height = m.group(0)
    else:
        raise ValueError(
            "Could not find inlet height from `inlet_height_magl` or `inlet` dataset attributes."
        )

    inlet_height = float(inlet_height) if inlet_height != "multiple" else dataset.inlet

    pblh_da = dataset.PBLH if "PBLH" in dataset.data_vars else dataset.atmosphere_boundary_layer_thickness

    filt = pblh_da > inlet_height + diff_threshold
    drop = not keep_missing

    return dataset.where(filt.compute(), drop=drop)


@register_filter
def pblh(dataset: xr.Dataset, keep_missing: bool = False) -> xr.Dataset:
    """Deprecated: pblh is now called pblh_inlet_diff."""
    raise NotImplementedError("pblh is now called pblh_inlet_diff")
