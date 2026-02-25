"""Helper functions for tests.

Mainly for creating fake data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def lat_lon_data(nlat: int, nlon: int, values: np.ndarray | list | None = None) -> xr.DataArray:
    lat = np.arange(nlat)
    lon = np.arange(nlon)

    if values is None:
        values = np.ones((nlat, nlon))

    return xr.DataArray(values, coords=[lat, lon], dims=["lat", "lon"])


rng = np.random.default_rng(seed=123)


def basis_function(nlat: int, nlon: int, nbasis: int) -> xr.DataArray:
    values = rng.integers(1, nbasis, size=nlat * nlon, endpoint=True).reshape((nlat, nlon))
    return lat_lon_data(nlat, nlon, values)


def lat_lon_time_data(
    nlat: int, nlon: int, start: str, end: str, ntime: int, values: np.ndarray | list | None = None
) -> xr.DataArray:
    lat = np.arange(nlat)
    lon = np.arange(nlon)
    time = pd.date_range(start, end, ntime)

    if values is None:
        values = np.ones((nlat, nlon, ntime))

    return xr.DataArray(values, coords=[lat, lon, time], dims=["lat", "lon", "time"])


def footprint(nlat: int, nlon: int, start: str, end: str, ntime: int) -> xr.DataArray:
    values = np.broadcast_to(np.arange(1, ntime + 1), (nlat, nlon, ntime))

    return lat_lon_time_data(nlat, nlon, start, end, ntime, values)


def make_time_coord(ntime: int, start: str = "2019-01-01T00:00:00") -> xr.DataArray:
    """Simple hourly time coordinate."""
    t = pd.date_range(start=start, periods=ntime, freq="1h")
    return xr.DataArray(t, dims="time", name="time")


def make_lat_lon_coords(nlat: int, nlon: int) -> tuple[xr.DataArray, xr.DataArray]:
    lat = xr.DataArray(np.arange(nlat, dtype=float), dims="lat", name="lat")
    lon = xr.DataArray(np.arange(nlon, dtype=float), dims="lon", name="lon")
    return lat, lon


# ----------------------------------------
# BASIS FUNCTIONS HELPERS
# ----------------------------------------
def make_fp_x_flux(
    nlat: int = 2,
    nlon: int = 2,
    ntime: int = 3,
    values: np.ndarray | None = None,
    start: str = "2019-01-01T00:00:00",
    name: str = "fp_x_flux",
) -> xr.DataArray:
    """Create synthetic fp_x_flux(lat, lon, time)."""
    lat, lon = make_lat_lon_coords(nlat, nlon)
    time = make_time_coord(ntime, start=start)

    if values is None:
        # deterministic: increasing with time, constant over lat/lon
        base = np.arange(1, ntime + 1, dtype=float)
        values = np.broadcast_to(base, (nlat, nlon, ntime))

    return xr.DataArray(
        values, coords={"lat": lat, "lon": lon, "time": time}, dims=("lat", "lon", "time"), name=name
    )


def make_fp_x_flux_sectoral(
    sources: list[str],
    nlat: int = 2,
    nlon: int = 2,
    ntime: int = 3,
    values_by_source: dict[str, np.ndarray] | None = None,
    start: str = "2019-01-01T00:00:00",
    name: str = "fp_x_flux_sectoral",
) -> xr.DataArray:
    """Create synthetic fp_x_flux_sectoral(source, lat, lon, time)."""
    pieces = []
    for i, s in enumerate(sources):
        if values_by_source is None:
            # deterministic distinct sources: (i+1) * fp_x_flux
            fp = make_fp_x_flux(nlat, nlon, ntime, start=start, name=name)
            fp = fp * float(i + 1)
        else:
            fp = make_fp_x_flux(nlat, nlon, ntime, values=values_by_source[s], start=start, name=name)
        pieces.append(fp.expand_dims(source=[s]))

    return xr.concat(pieces, dim="source")


def make_basis_flat_from_blocks(
    blocks: list[list[int]],
    region_start: int = 1,
    name: str = "basis",
) -> xr.DataArray:
    """Create a small deterministic basis_flat(lat, lon) from integer labels.

    blocks is a nested python list shaped (nlat, nlon) with labels starting at 1 by default.
    """
    arr = np.asarray(blocks, dtype=int)
    if arr.min() < region_start:
        raise ValueError("Basis labels must start from region_start (default 1).")
    nlat, nlon = arr.shape
    lat, lon = make_lat_lon_coords(nlat, nlon)
    return xr.DataArray(arr, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=name)


def expected_H_from_basis_sum(
    fp_x_flux: xr.DataArray, basis_flat: xr.DataArray, region_dim: str = "region"
) -> xr.DataArray:
    """Compute expected H(region, time) by explicit summation (slow, tiny arrays only).

    This is used for deterministic synthetic tests.
    """
    # unique region labels, sorted
    labels = np.unique(basis_flat.values.astype(int))
    labels = labels[labels > 0]

    out = []
    for i, lab in enumerate(labels):
        mask = xr.where(basis_flat == lab, 1.0, 0.0)
        # sum over lat/lon
        h_lab = (fp_x_flux * mask).sum(("lat", "lon"))

        # use 0..N-1, consistent with apply_fp_basis_functions
        out.append(h_lab.expand_dims({region_dim: [i]}))

    H = xr.concat(out, dim=region_dim).transpose(region_dim, "time")
    return H


def old_apply_fp_basis_functions_like(
    fp_x_flux: xr.DataArray, basis_flat: xr.DataArray, region_dim: str = "region"
) -> xr.DataArray:
    """Mimic legacy apply_fp_basis_functions using xarray only (no sparse).

    This is useful in tests to avoid importing private sparse helpers.
    """
    # NOTE: legacy aligns basis to fp_x_flux.isel(time=0), join="override"
    _, basis_aligned = xr.align(fp_x_flux.isel(time=0), basis_flat, join="override")

    labels = np.unique(basis_aligned.values.astype(int))
    labels = labels[labels > 0]  # basis uses labels 1..N by convention

    pieces = []
    for i, lab in enumerate(labels):
        mask = xr.where(basis_aligned == lab, 1.0, 0.0)
        h_lab = (fp_x_flux.fillna(0.0) * mask).sum(("lat", "lon"))

        # use 0..N-1, consistent with apply_fp_basis_functions
        pieces.append(h_lab.expand_dims({region_dim: [i]}))

    H = xr.concat(pieces, dim=region_dim).transpose(region_dim, "time")
    return H


def convert_old_multisector_H_to_gathered(
    H_old: xr.DataArray,
    *,
    source_dim: str = "source",
    region_dim: str = "region",
    gathered_dim: str = "region",
    sector_region_dim: str = "sector_region",
    drop_zero_rows: bool = True,
) -> xr.DataArray:
    """Convert legacy padded multisector sensitivity to gathered MultiIndex region.

    Legacy H is typically: (region=max_regions, time, source).
    Output is: (region=(source, sector_region), time) where region is a MultiIndex.

    We drop (source, sector_region) rows that are all-zero across time (and any other dims).
    """
    if source_dim not in H_old.dims:
        raise ValueError(f"Expected source dim {source_dim!r} in H_old.dims={H_old.dims}")
    if region_dim not in H_old.dims:
        raise ValueError(f"Expected region dim {region_dim!r} in H_old.dims={H_old.dims}")

    H = H_old.rename({region_dim: sector_region_dim})

    # Make sure sector_region is an integer coordinate; legacy region sometimes starts at 0
    # but for your padded H it looks like region starts at 0 in some places; keep as-is.
    # If you want to enforce 0/1-based, do it upstream.
    H = H.stack({gathered_dim: (source_dim, sector_region_dim)})

    if drop_zero_rows:
        # drop rows where the entire "row" across non-region dims is 0
        other_dims = [d for d in H.dims if d != gathered_dim]
        is_nonzero = (H != 0).any(dim=other_dims)
        H = H.isel({gathered_dim: is_nonzero.values})

    # transpose to (region, time, ...) canonical-ish
    if "time" in H.dims:
        H = H.transpose(gathered_dim, "time", ...)
    else:
        H = H.transpose(gathered_dim, ...)

    return H
