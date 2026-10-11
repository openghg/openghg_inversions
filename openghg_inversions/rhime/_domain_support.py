"""Native-grid support operations shared by scientific preparation recipes.

These functions preserve the target grid and borrowed array payloads. They do
not select sources, create states, or impose prior independence between domains.
"""

from __future__ import annotations

import xarray as xr


def rectangular_extent_mask(
    inner_grid: xr.Dataset | xr.DataArray,
    *,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
) -> xr.DataArray:
    """Mark target coordinates within an inner grid's inclusive bounding box.

    Args:
        inner_grid: Labelled native grid with nonempty indexed ``lat`` and
            ``lon`` coordinates. Only coordinates are inspected, not payloads.
        target_lat: Target latitude coordinates, in the inner grid's units.
        target_lon: Target longitude coordinates, in the inner grid's units.

    Returns:
        Boolean mask on the target coordinates, preserving their order. Bounds
        are coordinate minima and maxima, not cell edges or an irregular land
        mask. No interpolation or area weighting is performed.

    Raises:
        ValueError: Inner latitude or longitude coordinates are missing or empty.
    """
    if "lat" not in inner_grid.coords or "lon" not in inner_grid.coords:
        raise ValueError("Inner-domain datasets must contain indexed `lat` and `lon` coordinates.")
    inner_lat = inner_grid.get_index("lat")
    inner_lon = inner_grid.get_index("lon")
    if inner_lat.empty or inner_lon.empty:
        raise ValueError("Inner-domain latitude and longitude coordinates must not be empty.")
    lat_mask = (target_lat >= inner_lat.min()) & (target_lat <= inner_lat.max())
    lon_mask = (target_lon >= inner_lon.min()) & (target_lon <= inner_lon.max())
    return lat_mask & lon_mask


def remove_domain_overlap(array: xr.DataArray, overlap: xr.DataArray) -> xr.DataArray:
    """Zero overlapping native cells while retaining every other dimension.

    Args:
        array: Borrowed response or flux on the target native grid. Additional
            source, channel, time, or other dimensions are retained.
        overlap: Boolean mask whose dimensions are a subset of ``array`` and
            whose indexed coordinates agree exactly. True cells are removed.

    Returns:
        A labelled array with overlapping cells replaced by zero. Neither input
        is mutated; Dask payloads remain lazy and retain their shared graphs.

    Raises:
        ValueError: Mask dimensions or indexed coordinates do not match the array.
        TypeError: The mask is not Boolean.
    """
    if overlap.dtype.kind != "b":
        raise TypeError("Domain overlap mask must be Boolean.")
    if not set(overlap.dims).issubset(array.dims):
        raise ValueError("Domain overlap mask dimensions must belong to the native array.")
    array, overlap = xr.align(array, overlap, join="exact", copy=False)
    return array.where(~overlap, other=0.0)
