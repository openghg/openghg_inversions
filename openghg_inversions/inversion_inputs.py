"""Create backend-neutral, observation-aligned inversion inputs.

``make_inv_inputs`` gathers selected per-site datasets into one ragged
``nmeasure`` dataset, validates shared state layouts, adds site/minimum-error
metadata, transforms boundary-condition periods, drops unusable rows, and
materializes core arrays. ``sites=None`` infers non-metadata entries; an
explicit empty selection is an error.
"""

import datetime as dt
import numbers
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import concat_gather_datasets, get_xr_dummies
from openghg_inversions.model_error import (
    normalise_min_error_options as normalise_min_error_options,  # noqa: PLC0414
)
from openghg_inversions.model_error import (
    percentile_error_method,
    residual_error_method,
    xr_setup_min_error,
)

DatetimeLike = str | dt.datetime | np.datetime64 | pd.Timestamp


def _validate_per_site_dimension_names(
    site_data: dict[str, xr.Dataset],
    *,
    ragged_dim: str,
) -> None:
    """Reject shared variables whose structural dimension names differ by site."""
    if len(site_data) < 2:
        return

    shared_vars = set.intersection(*(set(dataset.data_vars) for dataset in site_data.values()))
    for name in sorted(shared_vars):
        layouts = {
            site: frozenset(str(dim) for dim in dataset[name].dims if dim != ragged_dim)
            for site, dataset in site_data.items()
        }
        if len(set(layouts.values())) > 1:
            raise ValueError(
                f"Per-site variable {name!r} must use the same non-{ragged_dim} dimensions "
                f"before gathering; found {layouts!r}."
            )


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

    assert isinstance(min_error_data, xr.DataArray)
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


def _fill_missing_optional_observation_factors(
    site_data: dict[str, xr.Dataset],
) -> dict[str, xr.Dataset]:
    """Validate column-factor pairs and zero-fill them on surface-only sites.

    Raises:
        ValueError: If a site defines only one of the two column prior-factor
            variables.
    """
    factor_names = ("mf_prior_factor", "mf_prior_upper_level_factor")
    partial_sites = {
        site: [name for name in factor_names if name in dataset]
        for site, dataset in site_data.items()
        if sum(name in dataset for name in factor_names) == 1
    }
    if partial_sites:
        raise ValueError(
            "Column observation datasets must define both `mf_prior_factor` and "
            f"`mf_prior_upper_level_factor`; partial definitions: {partial_sites!r}."
        )

    result = dict(site_data)
    for name in factor_names:
        template = next((dataset[name] for dataset in site_data.values() if name in dataset), None)
        if template is None:
            continue

        for site, dataset in result.items():
            if name in dataset:
                continue
            if "mf" not in dataset:
                continue
            updated = dataset.copy()
            factor = xr.zeros_like(updated["mf"]).astype(template.dtype).rename(name)
            factor.attrs = template.attrs.copy()
            updated[name] = factor
            result[site] = updated
    return result


def make_inv_inputs(
    fp_data: dict[str, Any],
    sites: list[str] | None = None,
    bc_freq: Literal["monthly"] | str | None = None,
    min_error: str | dict[str, float] | int | float = 0.0,
    min_error_per_site: bool = True,
    start_date: DatetimeLike | None = None,
    missing_data_vars: Literal["error", "drop"] = "drop",
) -> xr.Dataset:
    """Create backend-neutral observation-aligned inversion inputs.

    The returned dataset contains shared observations, sensitivities, error
    terms, and site alignment metadata. Model-component-specific arrays are
    constructed by their owning components.

    Args:
        fp_data: Per-site merged observations and sensitivity data.
        sites: Sites to retain. ``None`` infers all non-metadata ``fp_data``
            keys in insertion order. An explicit empty list is invalid, and
            every named site must exist in ``fp_data``.
        bc_freq: Optional frequency used to transform boundary-condition
            sensitivities.
        min_error: Minimum-error value or calculation configuration.
        min_error_per_site: Whether a calculated minimum error varies by site.
        start_date: Optional anchor for fixed-duration boundary-condition
            frequencies.
        missing_data_vars: Policy for observation-aligned variables that are
            not present at every site. ``"drop"`` preserves the established
            OpenGHG/legacy behavior; ``"error"`` prevents extension fields
            from being discarded.

    Returns:
        Canonical inversion inputs aligned along ``nmeasure``.

    Raises:
        ValueError: If no sites can be inferred, the explicit selection is
            empty, a requested site is missing, required input variables are
            missing, the selected missing-variable policy is violated, or
            minimum-error configuration is invalid.
    """
    if sites is None:
        sites = [key for key in fp_data if not key.startswith(".")]
        if not sites:
            raise ValueError("`fp_data` does not contain any non-metadata site entries.")
    elif not sites:
        raise ValueError(
            "`sites` must contain at least one site. Pass `sites=None` to infer all "
            "non-metadata sites from `fp_data`."
        )

    missing_sites = [site for site in sites if site not in fp_data]
    if missing_sites:
        raise ValueError(f"`fp_data` is missing requested site(s): {missing_sites!r}.")

    site_data = {site: fp_data[site] for site in sites}
    site_data = _fill_missing_optional_observation_factors(site_data)
    _validate_per_site_dimension_names(site_data, ragged_dim="time")

    try:
        ds = concat_gather_datasets(
            site_data,
            key_dim="site",
            ragged_dim="time",
            stack_dim="nmeasure",
            missing_data_vars=missing_data_vars,
            join="exact",
        )
    except xr.AlignmentError as exc:
        raise ValueError(
            "Per-site inversion inputs must have identical indexes on every non-time dimension "
            "before gathering into nmeasure."
        ) from exc

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
    ds = add_min_error(ds, fp_data=fp_data, min_error=min_error, min_error_per_site=min_error_per_site)

    ds = _drop_nan_and_compute(ds)

    return ds
