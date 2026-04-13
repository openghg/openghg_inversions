"""Functions for creating the inputs needed by PyMC."""

import datetime as dt
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies, concat_gather_datasets
from openghg_inversions.model_error import percentile_error_method, residual_error_method, xr_setup_min_error

DatetimeLike = str | dt.datetime | np.datetime64 | pd.Timestamp


def xr_unique_inv(da: xr.DataArray, sort: bool = True) -> xr.DataArray:
    if sort:

        def unique_inv(arr: np.ndarray) -> np.ndarray:
            _, inv = np.unique(arr, return_inverse=True)
            return inv

    else:

        def unique_inv(arr: np.ndarray) -> np.ndarray:
            inv, _ = pd.factorize(arr, sort=False)
            return inv

    return xr.apply_ufunc(unique_inv, da)


def xr_factorize(
    da: xr.DataArray, indicator_name: str, label_name: str, label_dim: str, sort: bool = False
) -> xr.Dataset:
    """Create Dataset with integer indicators and labels for DataArray.

    Args:
        da: DataArray to find indicator for.
        indicator_name: name for indicator data variable
        label_name: name for label data variable
        label_dim: dimension for labels
        sort: if True, the labels will be sorted and the indicator shuffled
        accordingly

    Returns:
        Dataset with indicator and label data variables.

    """

    indicator_arr, label_arr = pd.factorize(da.values, sort=sort)
    indicator = xr.DataArray(indicator_arr, coords=da.coords, dims=da.dims)
    labels = xr.DataArray(label_arr, dims=(label_dim,))
    return xr.Dataset({indicator_name: indicator, label_name: labels})


# MAKE FUNCTIONS
def make_site_indicator(site_coord: xr.DataArray) -> xr.DataArray:
    """Make site_indicator from DataArray of site names.

    For instance, the values ["TAC", "TAC", "MHD"] would be converted to
    [0, 0, 1].
    """
    return xr_unique_inv(site_coord, sort=False).rename("site_indicator")


def make_site_names(site_coord: xr.DataArray) -> xr.DataArray:
    """Make site names DataArray corresponding to site indicator."""
    _, site_names = pd.factorize(site_coord.values, sort=False)
    return xr.DataArray(site_names, dims=("nsite",), name="site_names")


def make_freq_indicator(
    time: xr.DataArray,
    freq: Literal["monthly"] | str,
    *,
    anchor_time: DatetimeLike | None = None,
) -> xr.DataArray:
    if freq == "monthly":
        return time.dt.month - time.min().dt.month + 12 * (time.dt.year - time.min().dt.year)

    # fixed-duration freq strings (e.g. "8d", "12h", "3h")
    if isinstance(freq, str) and freq.isalpha():
        freq = f"1{freq}"
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


def add_site_indicator(ds: xr.Dataset, sort: bool = False) -> xr.Dataset:
    """Adds site_indicator and site_names data variables."""
    to_add = xr_factorize(
        ds.site, indicator_name="site_indicator", label_name="site_names", label_dim="nsite", sort=sort
    )
    return xr.merge([ds, to_add])


# TRANSFORM FUNCTIONS
def _transform_bc_freq(
    H_bc: xr.DataArray, freq: Literal["monthly"] | str | None = None, anchor_time: DatetimeLike | None = None
) -> xr.DataArray:
    freq_arr = (
        make_freq_indicator(H_bc.time, freq, anchor_time=anchor_time)
        if freq is not None
        else xr.zeros_like(H_bc.time)
    )
    dums = get_xr_dummies(freq_arr, return_sparse=False, cat_dim="bc_period")
    return (H_bc.rename(bc_region="bc_curtain") * dums).stack(bc_region=("bc_curtain", "bc_period"))


def transform_bc(
    ds: xr.Dataset, freq: Literal["monthly"] | str | None = None, anchor_time: DatetimeLike | None = None
) -> xr.Dataset:
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
def _drop_nan_and_compute(ds: xr.Dataset, drop_nan_from: Iterable[str] = ("H", "H_bc", "mf", "mf_error")) -> xr.Dataset:
    """Drop NaNs in required inversion variables and materialize core variables.

    This centralizes the dataset cleanup that was previously duplicated in
    hbmcmc.make_inv_inputs. It:
      - drops nmeasure rows with NaNs in required variables (H, H_bc, mf, mf_error)
      - triggers computation for a selected set of variables so returned dataset
        is ready for immediate consumption (avoids repeated dask computations)

    Args:
        ds: Input xarray.Dataset produced by make_inv_inputs logic.
        drop_nan_from: data variables to drop NaNs from; only data variables present in `ds`
            will be used.

    Returns:
        xarray.Dataset with NaNs dropped along `nmeasure` based on selected variables,
            and with certain variables computed.
    """
    # Variables that must not contain NaNs along the nmeasure dim
    drop_subset: list[str] = [v for v in drop_nan_from if v in ds]
    if drop_subset:
        ds = ds.dropna(dim="nmeasure", how="any", subset=drop_subset)

    # Variables we want to ensure are materialized (compute() only these)
    to_compute: list[str] = [
        "H",
        "H_bc",
        "mf",
        "mf_error",
        "mf_repeatability",
        "mf_variability",
        "mf_prior_factor",
        "mf_prior_upper_level_factor",
        "bc_mod",
        "mf_mod",
    ]
    to_compute = [v for v in to_compute if v in ds]
    if to_compute:
        ds[to_compute] = ds[to_compute].compute()

    return ds


def make_inv_inputs(
    fp_data: dict[str, Any],
    sites: list[str] | None = None,
    bc_freq: Literal["monthly"] | str | None = None,
    sigma_freq: Literal["monthly"] | str | None = None,
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

    ds = add_site_indicator(ds)
    ds["sigma_freq_index"] = make_sigma_freq(ds.time, freq=sigma_freq, anchor_time=start_date)

    ds = add_min_error(ds, fp_data=fp_data, min_error=min_error, min_error_per_site=min_error_per_site)

    ds = _drop_nan_and_compute(ds)

    return ds
