from dataclasses import dataclass
from typing import Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies, align_sparse_lat_lon


def convert_idata_to_dataset(idata: az.InferenceData, group_filters = ["prior", "posterior"], add_suffix=True) -> xr.Dataset:
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


def nmeasure_to_site_time_data_array(
    da: xr.DataArray, site_indicators: xr.DataArray, site_names: xr.DataArray, times: xr.DataArray
) -> xr.DataArray:
    """Convert `nmeasure` dimension to multi-index over `site` and `time.`"""
    site_dict = dict(site_names.to_series())
    da = (
        xr.concat(
            [
                da.where(site_indicators == site_num, drop=True)
                .assign_coords(nmeasure=times.where(site_indicators == site_num, drop=True))
                .rename(nmeasure="time")
                .expand_dims({"site": [site_code]})
                for site_num, site_code in site_dict.items()
            ],
            dim="site",
        )
        .stack(nmeasure=["site", "time"])
        .dropna("nmeasure")
        .transpose("nmeasure", ...)
    )

    return da


def nmeasure_to_site_time(
    ds: xr.Dataset, site_indicators: xr.DataArray, site_names: xr.DataArray, times: xr.DataArray
) -> xr.Dataset:
    """Convert `nmeasure` dimension to multi-index over `site` and `time.`"""
    time_vars = [dv for dv in ds.data_vars if "nmeasure" in list(ds[dv].coords)]
    ds_tv = ds[time_vars]
    ds_no_tv = ds.drop_dims("nmeasure")

    site_dict = dict(site_names.to_series())
    ds_tv = (
        xr.concat(
            [
                ds_tv.where(site_indicators == site_num, drop=True)
                .expand_dims({"site": [site_code]})
                .assign_coords(nmeasure=times.where(site_indicators == site_num, drop=True))
                .rename_vars(nmeasure="time")
                for site_num, site_code in site_dict.items()
            ],
            dim="site",
        )
        .swap_dims(nmeasure="time")
        .stack(nmeasure=["site", "time"])
        .dropna("nmeasure")
        .transpose("nmeasure", ...)
    )

    return xr.merge([ds_no_tv, ds_tv])


@dataclass
class InversionOutput:
    """dataclass to hold the quantities we need to calculate outputs."""

    obs: xr.DataArray
    obs_err: xr.DataArray
    obs_repeatability: xr.DataArray
    obs_variability: xr.DataArray
    flux: xr.DataArray
    basis: xr.DataArray
    model: pm.Model
    trace: az.InferenceData
    site_indicators: xr.DataArray
    site_names: xr.DataArray
    times: xr.DataArray
    start_date: str | None = None
    end_date: str | None = None

    def __post_init__(self) -> None:
        """Check that trace has posterior traces, and fix flux time values."""
        if not hasattr(self.trace, "posterior"):
            raise ValueError("`trace` InferenceData must have `posterior` traces.")
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

    def sample_predictive_distributions(self, ndraw: int | None = None) -> None:
        """Sample prior and posterior predictive distributions.

        This creates prior samples as a side-effect.
        """
        if ndraw is None:
            ndraw = self.trace.posterior.sizes["draw"]
        self.trace.extend(pm.sample_prior_predictive(ndraw, self.model))
        self.trace.extend(pm.sample_posterior_predictive(self.trace, model=self.model, var_names=["y"]))

    def get_trace_dataset(
        self, unstack_nmeasure: bool = True, var_names: Optional[Union[str, list[str]]] = None
    ) -> xr.Dataset:
        """Return an xarray Dataset containing a prior/posterior parameter/predictive samples.

        Args:
            convert_nmeasure: if True, convert `nmeasure` coordinate to multi-index comprising `time` and `site`.
            var_names: (list of) variables to select. For instance, "x" will return "x_prior" and "x_posterior".

        Returns:
            xarray Dataset containing a prior/posterior parameter/predictive samples.
        """
        trace_ds = convert_idata_to_dataset(self.trace)

        if unstack_nmeasure:
            trace_ds = nmeasure_to_site_time(trace_ds, self.site_indicators, self.site_names, self.times).unstack("nmeasure")

        if var_names is not None:
            if isinstance(var_names, str):
                var_names = [var_names]

            data_vars = []
            for dv in trace_ds.data_vars:
                for name in var_names:
                    if str(dv).startswith(name):
                        data_vars.append(dv)

            trace_ds = trace_ds[data_vars]

        # add attributes for predictive traces (usually these are obs traces)
        for dv in trace_ds.data_vars:
            if str(dv).endswith("prior_predictive"):
                trace_ds[dv].attrs["units"] = self.obs.attrs["units"]
                trace_ds[dv].attrs["long_name"] = "prior_predictive_" + self.obs.attrs["long_name"]
            elif str(dv).endswith("posterior_predictive"):
                trace_ds[dv].attrs["units"] = self.obs.attrs["units"]
                trace_ds[dv].attrs["long_name"] = "posterior_predictive_" + self.obs.attrs["long_name"]
            elif str(dv).startswith("mu_bc"):
                suffix = str(dv).removeprefix("mu_bc_")
                trace_ds[dv].attrs["units"] = self.obs.attrs["units"]
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

        return trace_ds

    def get_model_data(self, unstack_nmeasure: bool = True, var_names: Optional[Union[str, list[str]]] = None
    ) -> xr.Dataset:
        """Return an xarray Dataset containing the data input to the model.

        This data is captured using `pm.Data`, or when data is observed.

        Args:
            convert_nmeasure: if True, convert `nmeasure` coordinate to multi-index comprising `time` and `site`.
            var_names: (list of) variables to select. For instance, "hx" or "min_error"

        Returns:
            xarray Dataset containing model data
        """
        trace_ds = convert_idata_to_dataset(self.trace, group_filters=["data"], add_suffix=False)

        if unstack_nmeasure:
            trace_ds = nmeasure_to_site_time(trace_ds, self.site_indicators, self.site_names, self.times).unstack("nmeasure")

        if var_names is not None:
            if isinstance(var_names, str):
                var_names = [var_names]

            data_vars = []
            for dv in trace_ds.data_vars:
                for name in var_names:
                    if str(dv).startswith(name):
                        data_vars.append(dv)

            trace_ds = trace_ds[data_vars]

        return trace_ds

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

    def get_obs(self, unstack_nmeasure: bool = True) -> xr.DataArray:
        """Return y observations.

        By default, `nmeasure` is converted to `site` and `time`.
        """
        result = nmeasure_to_site_time_data_array(self.obs, self.site_indicators, self.site_names, self.times)

        if unstack_nmeasure:
            result = result.unstack("nmeasure")

        return result.rename("y_obs")

    def get_obs_err(self, unstack_nmeasure: bool = True) -> xr.DataArray:
        """Return y observations errors.

        By default, `nmeasure` is converted to `site` and `time`.
        """
        result = nmeasure_to_site_time_data_array(
            self.obs_err, self.site_indicators, self.site_names, self.times
        )

        if unstack_nmeasure:
            result = result.unstack("nmeasure")

        return result.rename("y_obs_error")

    def get_obs_repeatability(self, unstack_nmeasure: bool = True) -> xr.DataArray:
        """Return "repeatbility" uncertainty term for y observations.

        By default, `nmeasure` is converted to `site` and `time`.

        TODO: this needs to be fixed when we have separate repeatability and variability outputs
        from RHIME
        """
        result = nmeasure_to_site_time_data_array(
            self.obs_repeatability, self.site_indicators, self.site_names, self.times
        )

        if unstack_nmeasure:
            result = result.unstack("nmeasure")

        return result.rename("y_obs_repeatability")

    def get_obs_variability(self, unstack_nmeasure: bool = True) -> xr.DataArray:
        """Return "variability" uncertainty term for y observations.

        By default, `nmeasure` is converted to `site` and `time`.
        """
        result = nmeasure_to_site_time_data_array(
            self.obs_variability, self.site_indicators, self.site_names, self.times
        )

        if unstack_nmeasure:
            result = result.unstack("nmeasure")

        return result.rename("y_obs_variability")

    def get_total_err(self, unstack_nmeasure: bool = True, take_mean: bool = True) -> xr.DataArray:
        """Return sqrt(repeatability**2 + variability**2 + model_error**2)

        Args:
            unstack_nmeasure: if True, convert `nmeasure` `site` and `time`. (Default: True)
            take_mean: if True, take mean over trace of error term

        """
        result = self.get_trace_dataset(var_names="epsilon", unstack_nmeasure=unstack_nmeasure).epsilon_posterior

        if take_mean:
            result = result.mean("draw")

        result.attrs["units"] = self.obs.attrs["units"]
        result.attrs["long_name"] = "total model-data mismatch error"

        return result.rename("total_error")

    def get_model_err(self, unstack_nmeasure: bool = True) -> xr.DataArray:
        """Return model_error

        By default, `nmeasure` is converted to `site` and `time`.
        """
        total_err = self.get_total_err(unstack_nmeasure=unstack_nmeasure, take_mean=False)
        total_obs_err = self.get_obs_err(unstack_nmeasure=unstack_nmeasure)

        result = np.sqrt(np.maximum(total_err**2 - total_obs_err**2, 0)).mean("draw")  # type: ignore
        result.attrs["units"] = self.obs.attrs["units"]
        result.attrs["long_name"] = "inferred model error"
        return result.rename("model_error")

    def get_flat_basis(self) -> xr.DataArray:
        """Return 2D DataArray encoding basis regions."""
        if len(self.basis.dims) == 2:
            return self.basis

        region_dim = next(str(dim) for dim in self.basis.dims if dim not in ["lat", "lon", "latitude", "longitude"])

        return (self.basis * self.basis[region_dim]).sum(region_dim).as_numpy().rename("basis")


def make_inv_out(
    fp_data: dict,
    Y: np.ndarray,
    Ytime: np.ndarray,
    error: np.ndarray,
    obs_repeatability: np.ndarray,
    obs_variability: np.ndarray,
    site_indicator: np.ndarray,
    site_names: np.ndarray | list[str],  # could be a list?
    mcmc_results: dict,
    start_date: str | None = None,
    end_date: str | None = None,
) -> InversionOutput:
    nmeasure = np.arange(len(Y))
    y_obs = xr.DataArray(Y, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yobs")
    times = xr.DataArray(Ytime, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="times")
    y_error = xr.DataArray(error, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yerror")
    y_error_repeatability = xr.DataArray(obs_repeatability, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yerror_repeatability")
    y_error_variability = xr.DataArray(obs_variability, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yerror_variability")
    site_indicator_da = xr.DataArray(site_indicator, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="site_indicator")
    site_names_da = xr.DataArray(site_names, dims=["nsite"], coords={"nsite": np.arange(len(site_names))}, name="site_names")

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
    )
