"""
Functions for filtering data.

All filters are accessed and applied to data via the `filtering` function.

New filters are registered using `@register_filter`.
A filter function should accept as arguments: an xr.Dataset, a bool called "keep_missing"

To see the available filters call `list_filters`.
"""

import logging
import re
from typing import Callable, cast, Union

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.utils import combine_datasets


logger = logging.Logger(__name__)


# this dictionary will be populated by using the decorator `register_filter`
filtering_functions = {}


def register_filter(filt: Callable) -> Callable:
    """Decorator function to register filters

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
            first_line_of_docstring = v.__doc__.strip().split("\n")[0]
        except AttributeError:
            first_line_of_docstring = "No docstring"

        print(f"\t{k:{spacing}}{first_line_of_docstring}")


def filtering(
    datasets_in: dict, filters: Union[str, None, dict[str, list[str | None]], list[str | None]], keep_missing: bool = False
) -> dict:
    """
    Applies time filtering to all datasets in `datasets_in`.

    If `filters` is a list, the same filters are applied to all sites. If `filters` is a dict
    with site codes as keys, then the filters applied to each site depend on the list supplied
    for that site.

    In any case, filters supplied in a list are applied in order.
    For example, if you wanted a daily, daytime average, you could do this:

        datasets_dictionary = filtering(datasets_dictionary,
                                    ["daytime", "daily_median"])

    The order of the filters reflects the order they are applied, so for
    instance when applying the "daily_median" filter if you only wanted
    to look at daytime values the filters list should be
    ["daytime","daily_median"]

    If a site is `datasets_in` is not in `filters`, then no filters are applied to that site.

    Args:
        datasets_in: dictionary of datasets containing output from ModelScenario.footprints_merge().
        filters: filters to apply to the datasets. Either a list of filters, which will be applied to every site,
            or a dictionary of lists of the form  {<site code>: [filter1, filter2, ...]}, with specific filters to
            be applied at each site. Use the `list_filters` function to list available filters.
        keep_missing: if True, drop missing data

    Returns:
        dict in same format as datasets_in, with filters applied

    """
    if not filters:
        return datasets_in

    # Get list of sites
    sites = [key for key in list(datasets_in.keys()) if key[0] != "."]

    # Put the filters in a dict of list
    if not isinstance(filters, dict):
        if not isinstance(filters, list):
            filters = [filters]  # type: ignore
        filters = {site: filters for site in sites}  # type: ignore
    else:
        for site, filt in filters.items():
            if filt is not None and not isinstance(filt, list):
                filters[site] = [filt]

    filters = cast(dict[str, list[str | None]], filters)

    # Check that filters are defined for all sites
    # TODO: just set filters for missing sites to None?
    tmp = [(site in filters) for site in sites]
    if not all(tmp):
        msg = f"Missing entry for sites {np.array(sites)[~np.array(tmp)]} in filters."
        logger.warning(msg)

    datasets = datasets_in.copy()

    # Apply filtering
    # NOTE: we only loop over sites that are in the filters dict
    # so not all sites must be specified
    for site in filters:
        if filters[site] is not None and site in sites:
            for filt in filters[site]:
                n_nofilter = datasets[site].time.values.shape[0]

                datasets[site] = filtering_functions[filt](datasets[site], keep_missing=keep_missing)

                n_filter = datasets[site].time.values.shape[0]
                n_dropped = n_nofilter - n_filter
                perc_dropped = np.round(n_dropped / n_nofilter * 100, 2)
                print(f"{filt} filter removed {n_dropped} ({perc_dropped} %) obs at site {site}")

    return datasets


def _local_solar_time(dataset: xr.Dataset) -> list[int]:
    """
    Returns hour of day as a function of local solar time relative to the Greenwich Meridian.

    This function also modifies `dataset` by changing the time coordinates.

    NOTE: This is not a filter; it is used by other filters.
    TODO: do we want this to modify `dataset`? currently it changes the time coordinate
    TODO: return np.ndarray and use vectorised filtering?

    Args:
        dataset: dataset to extract hours of the day from; this dataset is modified in place

    Returns:
        list of hours of the day for each time value in dataset.time
    """
    sitelon = dataset.release_lon.values[0]
    # convert lon to [-180,180], so time offset is negative west of 0 degrees
    if sitelon > 180:
        sitelon = sitelon - 360.0
    dataset["time"] = dataset.time + pd.Timedelta(minutes=float(24 * 60 * sitelon / 360.0))
    hours = dataset.time.to_pandas().index.hour
    return list(hours)


@register_filter
def daily_median(dataset: xr.Dataset, keep_missing: bool = False) -> xr.Dataset:
    """Resample data to daily frequency and use daily median values.

    Args:
        dataset: dataset to filter
        keep_missing: if True, drop time points removed by filter

    Returns:
        filtered dataset
    """
    if keep_missing:
        return dataset.load().resample(indexer={"time": "1D"}).median()
    else:
        return dataset.load().resample(indexer={"time": "1D"}).median().dropna(dim="time")


@register_filter
def six_hr_mean(dataset: xr.Dataset, keep_missing: bool = False) -> xr.Dataset:
    """Resample data to 6h frequency and use 6h mean values.

    Args:
        dataset: dataset to filter
        keep_missing: if True, drop time points removed by filter

    Returns:
        filtered dataset

    """
    if keep_missing:
        return dataset.resample(indexer={"time": "6H"}).mean()
    else:
        return dataset.resample(indexer={"time": "6H"}).mean().dropna(dim="time")


@register_filter
def daytime(dataset: xr.Dataset, keep_missing: bool = False) -> xr.Dataset:
    """Subset during daytime hours (11:00-15:00)

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
    """Subset during daytime hours (9:00-17:00)

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
    """Subset during nighttime hours (23:00 - 03:00)

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
    """Select only 12pm data

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


def _local_ratio(dataset: xr.Dataset) -> np.ndarray:
    """
    Calculates the local ratio in the surrounding grid cells.

    NOTE: This is not a filter; it is used by the `local_influence` filter.
    """
    dlon = dataset.lon[1].values - dataset.lon[0].values
    dlat = dataset.lat[1].values - dataset.lat[0].values
    local_sum = np.zeros((len(dataset.mf)))

    for ti in range(len(dataset.mf)):
        release_lon = dataset.release_lon[ti].values
        release_lat = dataset.release_lat[ti].values
        wh_rlon = np.where(abs(dataset.lon.values - release_lon) < dlon / 2.0)
        wh_rlat = np.where(abs(dataset.lat.values - release_lat) < dlat / 2.0)
        if np.any(wh_rlon[0]) and np.any(wh_rlat[0]):
            local_sum[ti] = np.sum(
                dataset.fp[
                    wh_rlat[0][0] - 2 : wh_rlat[0][0] + 3, wh_rlon[0][0] - 2 : wh_rlon[0][0] + 3, ti
                ].values
            ) / np.sum(dataset.fp[:, :, ti].values)
        else:
            local_sum[ti] = 0.0

    return local_sum


@register_filter
def local_influence(dataset: xr.Dataset, keep_missing: bool = False) -> xr.Dataset:
    """
    Subset for times when "local influence" is below threshold.

    Local influence expressed as a fraction of the sum of entire footprint domain.

    Args:
        dataset: dataset to filter
        keep_missing: if True, drop time points removed by filter

    Returns:
        filtered dataset
    """
    if not dataset.filter_by_attrs(standard_name="local_ratio"):
        lr = _local_ratio(dataset)
    else:
        lr = dataset.local_ratio

    pc = 0.1
    ti = [i for i, local_ratio in enumerate(lr) if local_ratio <= pc]
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
def pblh_min(dataset: xr.Dataset, pblh_threshold: float = 200.0, keep_missing: bool = False) -> xr.Dataset:
    """
    Subset for times when the PBLH is greater than 200m.

    Args:
        dataset: dataset to filter
        pblh_threshold: filter will discard times where PBLH/atmosphere boundary layer thickness is below pblh_threshold
        keep_missing: if True, drop time points removed by filter

    Returns:
        filtered dataset

    TODO: need way to pass pblh_threshold to filter
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
    """
    Subset for times when observations are taken at a height of less than 50 m below the PBLH.

    Args:
        dataset: dataset to filter
        diff_threshold: filter will discard times where obs. are taken at a height of less than diff_threshold below PBLH
        keep_missing: if True, drop time points removed by filter

    Returns:
        filtered dataset

    TODO: need way to pass diff_threshold to filter
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

    if inlet_height != "multiple":
        inlet_height = float(inlet_height)
    else:
        inlet_height = dataset.inlet

    pblh_da = dataset.PBLH if "PBLH" in dataset.data_vars else dataset.atmosphere_boundary_layer_thickness

    filt = pblh_da > inlet_height + diff_threshold
    drop = not keep_missing

    return dataset.where(filt, drop=drop)

@register_filter
def pblh(dataset: xr.Dataset, keep_missing: bool = False) -> xr.Dataset:
    """Deprecated: pblh is now called pblh_inlet_diff"""
    raise NotImplementedError("pblh is now called pblh_inlet_diff")
