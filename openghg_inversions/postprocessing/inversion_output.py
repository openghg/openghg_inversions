import warnings
from dataclasses import dataclass
from typing import TypeVar

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies, align_sparse_lat_lon


def filter_data_vars_by_prefix(
    ds: xr.Dataset, var_name_prefixes: str | list[str], sep: str = "_"
) -> xr.Dataset:
    """Select data variables that match the specified filters."""
    if isinstance(var_name_prefixes, str):
        var_name_prefixes = [var_name_prefixes]

    var_name_prefixes = [f"{name}{sep}" for name in var_name_prefixes]

    data_vars = []
    for dv in ds.data_vars:
        for name in var_name_prefixes:
            if str(dv).startswith(name):
                data_vars.append(dv)

    return ds[data_vars]


def convert_idata_to_dataset(
    idata: az.InferenceData, group_filters=["prior", "posterior"], add_suffix=True
) -> xr.Dataset:
    """Merge prior, prior predictive, posterior, and posterior predictive samples into a single
    xr.Dataset.
    """
    traces = []
    for group in idata.groups():
        if any(filt in group for filt in group_filters):
            trace = idata[group]
            if add_suffix:
                rename_dict = {dv: f"{dv}_{group}" for dv in trace.data_vars}
                trace = trace.rename_vars(rename_dict)
            if "chain" in trace.dims:
                trace = trace.isel(chain=0, drop=True)
            traces.append(trace)
    return xr.merge(traces)


def _add_attributes_to_trace_dataset(trace_ds: xr.Dataset, obs_units: str, obs_longname: str) -> None:
    """Add attributes to trace dataset.

    Args:
        trace_ds: trace dataset (probably created by `convert_idata_to_dataset`)
        obs_units: units for observation data used in inversion
    Returns:
        None: updates Dataset in-place
    """
    for dv in trace_ds.data_vars:
        if str(dv).endswith("prior_predictive"):
            trace_ds[dv].attrs["units"] = obs_units
            trace_ds[dv].attrs["long_name"] = "prior_predictive_" + obs_longname
        elif str(dv).endswith("posterior_predictive"):
            trace_ds[dv].attrs["units"] = obs_units
            trace_ds[dv].attrs["long_name"] = "posterior_predictive_" + obs_longname
        elif str(dv).startswith("mu_bc"):
            suffix = str(dv).removeprefix("mu_bc_")
            trace_ds[dv].attrs["units"] = obs_units
            trace_ds[dv].attrs["long_name"] = suffix + "_modelled_baseline"
        elif str(dv).endswith("prior"):
            prefix = str(dv).removesuffix("_prior")
            if prefix == "x":
                name = "flux_scaling_factor"
            elif "sig" in prefix:
                name = "pollution_event_scaling_factor"
            elif prefix == "bc":
                name = "boundary_conditions_scaling_factor"
            else:
                name = str(dv)
            trace_ds[dv].attrs["long_name"] = f"prior_trace_of_{name}"
        elif str(dv).endswith("posterior"):
            prefix = str(dv).removesuffix("_posterior")
            if prefix == "x":
                name = "flux_scaling_factor"
            elif "sig" in prefix:
                name = "pollution_event_scaling_factor"
            elif prefix == "bc":
                name = "boundary_conditions_scaling_factor"
            else:
                name = str(dv)
            trace_ds[dv].attrs["long_name"] = f"posterior_trace_of_{name}"


XrDataArrayOrSet = TypeVar("XrDataArrayOrSet", xr.DataArray, xr.Dataset)


def _nmeasure_to_site_time(
    data: XrDataArrayOrSet,
    site_indicators: xr.DataArray,
    times: xr.DataArray,
    site_names: xr.DataArray | dict | None = None,
) -> XrDataArrayOrSet:
    """Convert `nmeasure` dimension to multi-index over `site` and `time.`"""
    if len(site_indicators) != len(times):
        raise ValueError(
            "Site indicators and times must be same length, got:"
            f"\nsite indicators:\n{site_indicators}\ntimes:\n{times}"
        )
    if site_names is None:
        site_codes = site_indicators.values
    else:
        if isinstance(site_names, xr.DataArray):
            site_names = dict(site_names.to_series())

        site_codes = [site_names.get(x) for x in site_indicators.values]

    nmeasure_multiindex = pd.MultiIndex.from_arrays([site_codes, times.values], names=["site", "time"])

    result = data.assign_coords(nmeasure=nmeasure_multiindex)
    result.time.attrs = times.attrs

    return result


@dataclass
class InversionOutput:
    """dataclass to hold the quantities we need to calculate outputs."""

    obs: xr.DataArray
    obs_err: xr.DataArray
    obs_repeatability: xr.DataArray
    obs_variability: xr.DataArray
    flux: xr.DataArray
    basis: xr.DataArray
    trace: az.InferenceData
    site_indicators: xr.DataArray
    site_names: xr.DataArray
    times: xr.DataArray
    start_date: str
    end_date: str
    species: str
    domain: str
    model: pm.Model | None = None

    def __post_init__(self) -> None:
        """Check that trace has posterior traces, and fix flux time values."""
        if not hasattr(self.trace, "posterior"):
            raise ValueError("`trace` InferenceData must have `posterior` traces.")

        if self.model is not None:
            self.sample_predictive_distributions()

        # check if flux has time coordinate, and add one if necessary
        if "time" not in self.flux.dims:
            self.flux = self.flux.expand_dims(time=[self.start_time])

        # change "time" dim of flux to "flux_time" to avoid NaNs when merging with obs times
        self.flux = self.flux.rename(time="flux_time")

        # align basis with flux; this is necessary due to an issue with sparse matrices
        self.basis = align_sparse_lat_lon(self.basis, self.flux)

        # if basis has time, make sure it is aligned to flux
        if "time" in self.basis.dims:
            self.basis = self.basis.rename(time="flux_time")
            # self.basis = self.basis.assign_coords(flux_time=self.flux.flux_time)
        elif "time" in self.basis.coords:
            # time not in dims, so just delete the coord
            self.basis = self.basis.drop_vars("time")

        # create trace dataset
        trace_ds = convert_idata_to_dataset(self.trace)

        if "longname" in self.obs.attrs:
            obs_long_name = self.obs.attrs["longname"]
        else:
            obs_long_name = self.obs.attrs.get("long_name", "observed_mole_fraction")

        _add_attributes_to_trace_dataset(trace_ds, self.obs.attrs["units"], obs_long_name)
        self.trace_ds = self.nmeasure_to_site_time(trace_ds)

        # format obs data and errors
        self.obs = self.nmeasure_to_site_time(self.obs.rename("y_obs"))
        self.obs_err = self.nmeasure_to_site_time(self.obs_err.rename("y_obs_error"))
        self.obs_repeatability = self.nmeasure_to_site_time(
            self.obs_repeatability.rename("y_obs_repeatability")
        )
        self.obs_variability = self.nmeasure_to_site_time(self.obs_variability.rename("y_obs_variability"))

    def sample_predictive_distributions(self, ndraw: int | None = None) -> None:
        """Sample prior and posterior predictive distributions.

        This creates prior samples as a side-effect.
        """
        if self.model is None:
            warnings.warn("Cannot sample predictive distributions without PyMC model.")
            return None

        if ndraw is None:
            ndraw = self.trace.posterior.sizes["draw"]

        self.trace.extend(pm.sample_prior_predictive(ndraw, self.model))
        self.trace.extend(pm.sample_posterior_predictive(self.trace, model=self.model, var_names=["y"]))

    def nmeasure_to_site_time(self, data: XrDataArrayOrSet) -> XrDataArrayOrSet:
        return _nmeasure_to_site_time(data, self.site_indicators, self.times, self.site_names)

    def get_trace_dataset(self, var_names: str | list[str] | None = None) -> xr.Dataset:
        """Return an xarray Dataset containing a prior/posterior parameter/predictive samples.

        Args:
            convert_nmeasure: if True, convert `nmeasure` coordinate to multi-index comprising `time` and `site`.
            var_names: (list of) variables to select. For instance, "x" will return "x_prior" and "x_posterior".

        Returns:
            xarray Dataset containing a prior/posterior parameter/predictive samples.
        """
        result = self.trace_ds

        if var_names is not None:
            result = filter_data_vars_by_prefix(result, var_names)

        return result

    def get_model_data(self, var_names: str | list[str] | None = None) -> xr.Dataset:
        """Return an xarray Dataset containing the data input to the model.

        This data is captured using `pm.Data`, or when data is observed.

        Args:
            convert_nmeasure: if True, convert `nmeasure` coordinate to multi-index comprising `time` and `site`.
            var_names: (list of) variables to select. For instance, "hx" or "min_error"

        Returns:
            xarray Dataset containing model data
        """
        result = convert_idata_to_dataset(self.trace, group_filters=["data"], add_suffix=False)
        result = self.nmeasure_to_site_time(result)

        if var_names is not None:
            result = filter_data_vars_by_prefix(result, var_names, sep="")

        return result

    @property
    def start_time(self) -> pd.Timestamp:
        """Return start date of inversion."""
        if self.start_date is not None:
            return pd.to_datetime(self.start_date)
        return pd.to_datetime(self.times.min().values[0])

    @property
    def end_time(self) -> pd.Timestamp:
        """Return end date of inversion."""
        if self.end_date is not None:
            return pd.to_datetime(self.end_date)
        return pd.to_datetime(self.times.max().values[0])

    @property
    def period_midpoint(self) -> pd.Timestamp:
        """Return midpoint of inversion period."""
        return self.start_time + (self.start_time - self.end_time) / 2

    def get_total_err(self, take_mean: bool = True) -> xr.DataArray:
        """Return sqrt(repeatability**2 + variability**2 + model_error**2)

        Args:
            take_mean: if True, take mean over trace of error term

        """
        result = self.get_trace_dataset(var_names="epsilon").epsilon_posterior

        if take_mean:
            result = result.mean("draw")

        result.attrs["units"] = self.obs.attrs["units"]
        result.attrs["long_name"] = "total model-data mismatch error"

        return result.rename("total_error")

    def get_model_err(self) -> xr.DataArray:
        """Return model_error

        By default, `nmeasure` is converted to `site` and `time`.
        """
        total_err = self.get_total_err(take_mean=False)
        total_obs_err = self.obs_err

        result = np.sqrt(np.maximum(total_err**2 - total_obs_err**2, 0)).mean("draw")  # type: ignore
        result.attrs["units"] = self.obs.attrs["units"]
        result.attrs["long_name"] = "inferred model error"
        return result.rename("model_error")

    def get_obs_and_errors(self) -> xr.Dataset:
        # TODO: some of these variables could just be stored in a dataset in InversionOutput,
        # rather than in separate data arrays
        to_merge = [
            self.obs,
            self.obs_err,
            self.obs_repeatability,
            self.obs_variability,
            self.get_model_err(),
            self.get_total_err(),
        ]
        result = xr.merge(to_merge)
        result.attrs = {}

        return result

    def get_flat_basis(self) -> xr.DataArray:
        """Return 2D DataArray encoding basis regions."""
        if len(self.basis.dims) == 2:
            return self.basis

        region_dim = next(
            str(dim) for dim in self.basis.dims if dim not in ["lat", "lon", "latitude", "longitude"]
        )

        return (self.basis * self.basis[region_dim]).sum(region_dim).as_numpy().rename("basis")


def make_inv_out_for_fixed_basis_mcmc(
    fp_data: dict,
    Y: np.ndarray,
    Ytime: np.ndarray,
    error: np.ndarray,
    obs_repeatability: np.ndarray,
    obs_variability: np.ndarray,
    site_indicator: np.ndarray,
    site_names: np.ndarray | list[str],  # could be a list?
    mcmc_results: dict,
    start_date: str,
    end_date: str,
    species: str,
    domain: str,
) -> InversionOutput:
    nmeasure = np.arange(len(Y))
    y_obs = xr.DataArray(Y, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yobs")
    times = xr.DataArray(Ytime, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="times")
    y_error = xr.DataArray(error, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yerror")
    y_error_repeatability = xr.DataArray(
        obs_repeatability, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yerror_repeatability"
    )
    y_error_variability = xr.DataArray(
        obs_variability, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yerror_variability"
    )
    site_indicator_da = xr.DataArray(
        site_indicator, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="site_indicator"
    )
    site_names_da = xr.DataArray(
        site_names, dims=["nsite"], coords={"nsite": np.arange(len(site_names))}, name="site_names"
    )

    _, nx = mcmc_results["xouts"].shape
    nx = np.arange(nx)

    basis = get_xr_dummies(fp_data[".basis"], cat_dim="nx", categories=nx)

    scenarios = [v for k, v in fp_data.items() if not k.startswith(".")]

    try:
        flux = scenarios[0].flux_stacked
    except AttributeError:
        flux = next(iter(fp_data[".flux"].values())).data

    # TODO: this only works if there is one flux used (or if multiple, but ModelScenario stacks them)
    if isinstance(flux, xr.Dataset):
        if "flux" in flux:
            flux = flux.flux
        else:
            flux = flux[flux.data_vars[0]]

    if not isinstance(flux, xr.DataArray):
        raise ValueError("Flux from `fp_data` could not be converted to a xr.DataArray.")

    # add attributes
    scenario = scenarios[0]
    y_obs.attrs = scenario.mf.attrs
    times.attrs = scenario.time.attrs
    y_error.attrs = scenario.mf_error.attrs
    y_error_variability.attrs = scenario.mf_variability.attrs
    y_error_repeatability.attrs = scenario.mf_repeatability.attrs

    return InversionOutput(
        obs=y_obs,
        obs_err=y_error,
        obs_repeatability=y_error_repeatability,
        obs_variability=y_error_variability,
        site_indicators=site_indicator_da,
        flux=flux,
        basis=basis,
        model=mcmc_results["model"],
        trace=mcmc_results["trace"],
        site_names=site_names_da,
        times=times,
        start_date=start_date,
        end_date=end_date,
        species=species,
        domain=domain,
    )


# Functions to re-run post-processing on standard RHIME outputs
#
# This is a temporary fix. A more general fix will be possible when we can recreate the
# RHIME model using the modular model set-up.
#
# In particular, there are not prior or predictive traces, so only flux and country totals
# can be recomputed.
def _clean_rhime_output(ds: xr.Dataset) -> xr.Dataset:
    """Take raw RHIME output and rename/drop/create variables to get dataset ready for further processing."""
    use_bc = "bctrace" in ds.data_vars

    rename_vars_dict = dict(stepnum="draw", paramnum="nlatent", measurenum="nmeasure")

    rename_dict = {
        "nsites": "nsite",
        "nparam": "nx",
        "xtrace": "x",
        "sigtrace": "sigma",
    }

    if use_bc:
        rename_vars_dict["numBC"] = "nBC"
        rename_dict.update({"bctrace": "bc", "nBC": "nbc"})

    ds = (
        ds.rename_vars(rename_vars_dict)
        .drop_dims(["nUI", "nlatent"])
        .swap_dims(nsite="nsites", steps="draw")
        .rename(rename_dict)
    )
    ds["x"] = ds.x.assign_coords(nx=("nx", ds.basisfunctions.to_series().sort_values().unique()))

    data_vars = [
        "Yobs",
        "Yerror",
        "Yerror_repeatability",
        "Yerror_variability",
        "Ytime",
        "x",
        "sigma",
        "siteindicator",
        "sigmafreqindex",
        "sitenames",
        "fluxapriori",
        "basisfunctions",
        "xsensitivity",
    ]

    if use_bc:
        data_vars.extend(["bc", "bcsensitivity"])

    ds = ds[data_vars]

    return ds


def _make_idata_from_rhime_outs(rhime_out_ds: xr.Dataset) -> az.InferenceData:
    """Create arviz InferenceData with posterior group created from RHIME output."""
    trace_dvs = [dv for dv in rhime_out_ds.data_vars if "draw" in list(rhime_out_ds[dv].coords)]
    traces = rhime_out_ds[trace_dvs].expand_dims({"chain": [0]})

    return az.InferenceData(posterior=traces)


def make_inv_out_from_rhime_outputs(
    ds: xr.Dataset, species: str, domain: str, start_date: str | None = None, end_date: str | None = None
) -> InversionOutput:
    flux = ds.fluxapriori

    ds_clean = _clean_rhime_output(ds)
    site_indicators = ds_clean.siteindicator
    basis = get_xr_dummies(ds_clean.basisfunctions, cat_dim="nx", categories=ds_clean.nx.values)

    start_date = start_date or ds_clean.Ytime.min().values
    end_date = end_date or ds_clean.Ytime.max().values

    trace = _make_idata_from_rhime_outs(ds_clean)
    trace.add_groups(prior={"x": xr.ones_like(trace.posterior["x"])}, dims={"x": ["nx"]})

    return InversionOutput(
        obs=ds_clean.Yobs,
        obs_err=ds_clean.Yerror,
        obs_repeatability=ds_clean.Yerror_repeatability,
        obs_variability=ds_clean.Yerror_variability,
        flux=flux,
        basis=basis,
        trace=trace,
        site_indicators=site_indicators,
        site_names=ds_clean.sitenames,
        times=ds_clean.Ytime,
        species=species,
        domain=domain,
        start_date=start_date,
        end_date=end_date,
    )
