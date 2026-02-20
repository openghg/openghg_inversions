"""Functions for creating the inputs needed by PyMC."""

from collections.abc import Hashable, Mapping
import datetime as dt
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies
from openghg_inversions.model_error import percentile_error_method, residual_error_method, xr_setup_min_error

DatetimeLike = str | dt.datetime | np.datetime64 | pd.Timestamp


# HELPERS
def concat_gather_data_arrays(
    da_dict: Mapping[Hashable, xr.DataArray],
    key_dim: str,
    ragged_dim: str,
    stack_dim: str | None = None,
    **concat_kwargs,
) -> xr.DataArray:
    stack_dim = stack_dim or (key_dim + "_" + ragged_dim)

    pieces: list[xr.DataArray] = []
    key_vals: list[np.ndarray] = []
    ragged_vals: list[np.ndarray] = []

    for k, v in da_dict.items():
        piece = v.rename({ragged_dim: stack_dim})
        pieces.append(piece)

        n = piece.sizes[stack_dim]

        # make site indicator
        key_val = np.full(n, k)
        key_vals.append(key_val)

        # record times
        ragged_vals.append(v[ragged_dim])

    # concat pieces
    da = xr.concat(pieces, dim=stack_dim, **concat_kwargs)

    # now create and assign multi-index
    key_indicator = np.concat(key_vals)
    concat_ragged = np.concat(ragged_vals)
    multiindex = pd.MultiIndex.from_arrays([key_indicator, concat_ragged], names=[key_dim, ragged_dim])
    xr_multiindex = xr.Coordinates.from_pandas_multiindex(multiindex, stack_dim)

    da = da.assign_coords(xr_multiindex)

    return da


def concat_gather_datasets(
    ds_dict: Mapping[Hashable, xr.Dataset],
    key_dim: str,
    ragged_dim: str,
    stack_dim: str | None = None,
    **concat_kwargs,
) -> xr.Dataset:
    """Concatenate dictionary of xr.Datasets by gathering ragged coordinates.

    This assumes that all datasets have the same data variables.

    TODO: need to handle missing data variables.
    """
    dvs = next(iter(ds_dict.values())).data_vars

    gathered_dvs = {}

    for dv in dvs:
        da_dict = {k: v[dv] for k, v in ds_dict.items()}
        gathered_dvs[dv] = concat_gather_data_arrays(da_dict, key_dim, ragged_dim, stack_dim, **concat_kwargs)

    return xr.Dataset(gathered_dvs)


def concat_gather_datatree(
    dt: xr.DataTree, key_dim: str, ragged_dim: str, stack_dim: str | None = None, **concat_kwargs
) -> xr.Dataset:
    """Concatenate xr.DataTree children by gathering ragged coordinates.

    This assumes that all children have the same data variables.
    """
    ds_dict = {str(k): v.to_dataset() for k, v in dt.items()}
    dvs = next(iter(ds_dict.values())).data_vars

    gathered_dvs = {}

    for dv in dvs:
        da_dict = {k: v[dv] for k, v in ds_dict.items()}
        gathered_dvs[dv] = concat_gather_data_arrays(da_dict, key_dim, ragged_dim, stack_dim, **concat_kwargs)

    return xr.Dataset(gathered_dvs)


def xr_unique_inv(da: xr.DataArray) -> xr.DataArray:
    def np_unique_inv(arr: np.ndarray) -> np.ndarray:
        _, inv = np.unique(arr, return_inverse=True)
        return inv

    return xr.apply_ufunc(np_unique_inv, da)


# MAKE FUNCTIONS
def make_site_indicator(site_coord: xr.DataArray) -> xr.DataArray:
    return xr_unique_inv(site_coord).rename("site_indicator")


def make_freq_indicator(
    time: xr.DataArray,
    freq: Literal["monthly"] | str,
    *,
    anchor_time: DatetimeLike | None = None,
) -> xr.DataArray:
    if freq == "monthly":
        return time.dt.month - time.min().dt.month + 12 * (time.dt.year - time.min().dt.year)

    # fixed-duration freq strings (e.g. "8d", "12h", "3h")

    # NOTE: you could also use `return xr_unique_inv((time - anchor).dt.floor(freq))`
    # here, but this way is more explicit.
    anchor = np.datetime64(anchor_time) if anchor_time is not None else time.min().values
    dt = np.timedelta64(pd.to_timedelta(freq).value, "ns")  # robust to xarray dtype
    idx = ((time.values.astype("datetime64[ns]") - anchor) // dt).astype(int)
    idx = idx - idx.min()
    return xr.DataArray(idx, coords=time.coords, dims=time.dims)


def make_sigma_freq(
    time: xr.DataArray,
    freq: Literal["monthly"] | str | None = None,
    anchor_time: DatetimeLike | None = None,
) -> xr.DataArray:
    res = (
        xr.zeros_like(time).astype(int)
        if freq is None
        else make_freq_indicator(time, freq, anchor_time=anchor_time)
    )
    return res.rename("sigma_freq_index")


# ADD FUNCTIONS
def add_min_error(
    ds: xr.Dataset,
    fp_data: dict[str, Any],
    min_error: str | dict[str, float] | float = 0.0,
    min_error_per_site: bool = True,
) -> xr.Dataset:
    """Add min_error to combined Dataset."""
    if isinstance(min_error, float) or (isinstance(min_error, np.ndarray) and min_error.ndim == 0):
        ds["min_error"] = min_error * xr.ones_like(ds.mf)
    elif isinstance(min_error, dict):
        sites = [k for k in fp_data if not k.startswith(".")]
        err_per_site = np.array([min_error[site] for site in sites])
        ds["min_error"] = xr_setup_min_error(err_per_site, ds.site_indicator)
    elif min_error == "residual":
        res_err = residual_error_method(fp_data)
        if min_error_per_site:
            ds["min_error"] = xr_setup_min_error(res_err, ds.site_indicator)
        else:
            ds["min_error"] = res_err
    elif min_error == "percentile":
        perc_err = percentile_error_method(fp_data)
        ds["min_error"] = xr_setup_min_error(perc_err, ds.site_indicator)
    else:
        raise ValueError(f"Option '{min_error}' is not valid.")

    return ds


# TRANSFORM FUNCTIONS
def _transform_bc_freq(
    H_bc: xr.DataArray, freq=None, anchor_time: DatetimeLike | None = None
) -> xr.DataArray:
    freq_arr = (
        make_freq_indicator(H_bc.time, freq, anchor_time=anchor_time)
        if freq is not None
        else xr.zeros_like(H_bc.time)
    )
    dums = get_xr_dummies(freq_arr, return_sparse=False, cat_dim="bc_period")
    return (H_bc.rename(bc_region="bc_curtain") * dums).stack(bc_region=("bc_curtain", "bc_period"))


def transform_bc(ds: xr.Dataset, freq=None, anchor_time: DatetimeLike | None = None) -> xr.Dataset:
    """Convert ds so that ds.H_bc is converted to (curtain, period) coordinates."""
    if "H_bc" not in ds:
        raise ValueError("Cannot setup boundary conditions sensitivity; H_bc not in dataset.")

    # save temp version so we can drop "bc_region"; we need to reset this coordinate because
    # it has been modified.
    temp = _transform_bc_freq(ds.H_bc, freq=freq, anchor_time=anchor_time).transpose("bc_region", ...)
    ds = ds.drop_dims("bc_region")

    # IMPORTANT: strip the RHS of the nmeasure MultiIndex bundle to avoid merge logic
    # This is a hack to avoid a deprecation warning due to how xarray uses pandas' multi-index.
    # Just assigning ds["H_bc"] = temp is fine, but emits a warning.
    temp_values_only = xr.DataArray(
        temp.data,
        dims=("bc_region", "nmeasure"),
        coords={"bc_region": temp["bc_region"]},  # keep only the new dim coord(s)
        name="H_bc",
    )

    ds["H_bc"] = temp_values_only
    return ds


# INVERSION INPUTS PIPELINE
def make_inv_inputs(
    fp_data: dict[str, Any],
    sites: list[str] | None = None,
    bc_freq: str | None = None,
    sigma_freq: str | None = None,
    min_error: str | dict[str, float] | float = 0.0,
    min_error_per_site: bool = True,
    start_date: DatetimeLike | None = None,
) -> xr.Dataset:
    sites = sites or [k for k in fp_data if not k.startswith(".")]

    ds = concat_gather_datasets(
        {k: v for k, v in fp_data.items() if k in sites},
        key_dim="site",
        ragged_dim="time",
        stack_dim="nmeasure",
    )

    if "H_bc" in ds:
        ds = transform_bc(ds, freq=bc_freq, anchor_time=start_date)

    ds["site_indicator"] = make_site_indicator(ds.site)
    ds["sigma_freq_index"] = make_sigma_freq(ds.time, freq=sigma_freq, anchor_time=start_date)

    ds = add_min_error(ds, fp_data=fp_data, min_error=min_error, min_error_per_site=min_error_per_site)

    return ds
