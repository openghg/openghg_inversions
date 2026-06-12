"""Functions for creating the inputs needed by PyMC."""

import datetime as dt
import numbers
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies, concat_gather_datasets
from openghg_inversions.model_error import percentile_error_method, residual_error_method, xr_setup_min_error

DatetimeLike = str | dt.datetime | np.datetime64 | pd.Timestamp


def _compact_integer_index(values: np.ndarray) -> np.ndarray:
    """Remap integer indicator values to contiguous 0..N-1 positions."""
    values = np.asarray(values, dtype=int)
    if values.size == 0:
        return values

    unique_values = np.unique(values)
    return np.searchsorted(unique_values, values).astype(int)


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
    if isinstance(freq, str):
        freq = freq.replace("H", "h")
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
    result = (
        xr.zeros_like(time).astype(int)
        if freq is None
        else make_freq_indicator(time, freq, anchor_time=anchor_time)
    )
    result = xr.apply_ufunc(_compact_integer_index, result.astype(int))
    return result.rename("sigma_freq_index")


# ADD FUNCTIONS
def add_min_error(
    ds: xr.Dataset,
    fp_data: dict[str, Any],
    min_error: str | dict[str, float] | int | float = 0.0,
    min_error_per_site: bool = True,
) -> xr.Dataset:
    """Add min_error to combined Dataset."""
    min_error_data: xr.DataArray | float | np.ndarray

    def site_names_for_min_error() -> list[str]:
        if "site_names" in ds:
            return [str(site) for site in ds.site_names.values]
        if "site" in ds:
            return [str(site) for site in make_site_names(ds.site).values]
        return [site for site in fp_data if not site.startswith(".")]

    def site_indicator_for_min_error() -> xr.DataArray:
        if "site_indicator" in ds:
            return ds.site_indicator
        if "site" in ds:
            return xr_unique_inv(ds.site, sort=False).rename("site_indicator")
        raise ValueError("Per-site min_error values require site_indicator or site data.")

    def fp_data_for_ds_sites() -> dict[str, Any]:
        return {site: fp_data[site] for site in site_names_for_min_error()}

    if isinstance(min_error, numbers.Real) and not isinstance(min_error, bool):
        min_error_data = float(min_error) * xr.ones_like(ds.mf)
    elif isinstance(min_error, np.ndarray) and min_error.ndim == 0:
        min_error_data = min_error * xr.ones_like(ds.mf)
    elif isinstance(min_error, dict):
        fp_data_for_sites = fp_data_for_ds_sites()
        sites = list(fp_data_for_sites)
        missing_sites = [site for site in sites if site not in min_error]
        if missing_sites:
            raise ValueError(f"min_error mapping is missing values for site(s): {missing_sites}")
        err_per_site = np.array([min_error[site] for site in sites])
        min_error_data = xr_setup_min_error(err_per_site, site_indicator_for_min_error())
    elif min_error == "residual":
        fp_data_for_sites = fp_data_for_ds_sites()
        res_err = residual_error_method(fp_data_for_sites, by_site=min_error_per_site)
        if min_error_per_site:
            min_error_data = xr_setup_min_error(res_err, site_indicator_for_min_error())
        else:
            min_error_data = res_err
    elif min_error == "percentile":
        fp_data_for_sites = fp_data_for_ds_sites()
        perc_err = percentile_error_method(fp_data_for_sites)
        min_error_data = xr_setup_min_error(perc_err, site_indicator_for_min_error())
    else:
        raise ValueError(f"Option '{min_error}' is not valid.")

    if not isinstance(min_error_data, xr.DataArray):
        min_error_data = xr.full_like(ds.mf, min_error_data).rename("min_error")
    elif "nmeasure" not in min_error_data.dims:
        min_error_data = xr.full_like(ds.mf, min_error_data.values).rename("min_error")
    else:
        min_error_data = min_error_data.rename("min_error")

    ds["min_error"] = xr.DataArray(min_error_data.data, dims=("nmeasure",), name="min_error")
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
def _drop_nan_and_compute(
    ds: xr.Dataset, drop_nan_from: Iterable[str] = ("H", "H_bc", "mf", "mf_error")
) -> xr.Dataset:
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
    for var_name in to_compute:
        computed = ds[var_name].compute()
        ds[var_name] = (computed.dims, computed.data, computed.attrs)

    return ds


def _check_required_inv_input_vars(
    ds: xr.Dataset, fp_data: dict[str, Any], sites: list[str], required_vars: Iterable[str] = ()
) -> None:
    """Raise if concat-drop mode removed variables required by the inversion pipeline.

    Args:
        ds: Gathered inversion-input dataset after concatenation.
        fp_data: Original per-site input datasets.
        sites: Site names included in the inversion input assembly.
        required_vars: Variables that must always be present after concatenation.

    Raises:
        ValueError: If a required variable is missing from the gathered dataset.
    """
    missing_required = [var for var in required_vars if var not in ds]

    if any("H_bc" in fp_data[site] for site in sites) and "H_bc" not in ds:
        missing_required.append("H_bc")

    if missing_required:
        raise ValueError(
            "Required inversion data variables were dropped during dataset gathering: "
            f"{sorted(set(missing_required))}"
        )


def make_inv_inputs(
    fp_data: dict[str, Any],
    sites: list[str] | None = None,
    bc_freq: Literal["monthly"] | str | None = None,
    sigma_freq: Literal["monthly"] | str | None = None,
    min_error: str | dict[str, float] | int | float = 0.0,
    min_error_per_site: bool = True,
    start_date: DatetimeLike | None = None,
) -> xr.Dataset:
    sites = sites or [k for k in fp_data if not k.startswith(".")]

    ds = concat_gather_datasets(
        {k: v for k, v in fp_data.items() if k in sites},
        key_dim="site",
        ragged_dim="time",
        stack_dim="nmeasure",
        missing_data_vars="drop",
    )

    # Check that we have variables for standard RHIME inversion (`inferpymc`).
    # Note that mf_prior_factor and mf_prior_upper_level_factor are only needed
    # for post-processing (and only if column data is used).
    _check_required_inv_input_vars(
        ds,
        fp_data=fp_data,
        sites=sites,
        required_vars=("H", "mf", "mf_error", "mf_repeatability", "mf_variability"),
    )

    if "H_bc" in ds:
        ds = transform_bc(ds, freq=bc_freq, anchor_time=start_date)

    ds = add_site_indicator(ds)
    sigma_freq_index = make_sigma_freq(ds.time, freq=sigma_freq, anchor_time=start_date)
    ds["sigma_freq_index"] = xr.DataArray(
        sigma_freq_index.data,
        dims=("nmeasure",),
        name="sigma_freq_index",
    )

    ds = add_min_error(ds, fp_data=fp_data, min_error=min_error, min_error_per_site=min_error_per_site)

    ds = _drop_nan_and_compute(ds)

    return ds
