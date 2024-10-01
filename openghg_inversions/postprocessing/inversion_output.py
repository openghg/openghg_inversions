from dataclasses import dataclass
from typing import Optional, Union

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.postprocessing.utils import add_suffix

from openghg_inversions.array_ops import get_xr_dummies


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
):
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

    if isinstance(flux, xr.Dataset):
        if "flux" in flux:
            flux = flux.flux
        else:
            flux = flux[flux.data_vars[0]]

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
    )


def convert_idata_to_dataset(idata: az.InferenceData) -> xr.Dataset:
    """Merge prior, prior predictive, posterior, and posterior predictive samples into a single
    xr.Dataset.
    """
    traces = []
    for group in idata.groups():
        if "prior" in group or "posterior" in group:
            trace = idata[group]
            rename_dict = {dv: f"{dv}_{group}" for dv in trace.data_vars}
            traces.append(trace.rename_vars(rename_dict).isel(chain=0, drop=True))
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

    def __post_init__(self) -> None:
        """Check that trace has posterior traces, and keep only chain 0"""
        if not hasattr(self.trace, "posterior"):
            raise ValueError("`trace` InferenceData must have `posterior` traces.")

    def sample_predictive_distributions(self, ndraw: int | None = None) -> None:
        """Sample prior and posterior predictive distributions.

        This creates prior samples as a side-effect.
        """
        if ndraw is None:
            ndraw = self.trace.posterior.sizes["draw"]
        self.trace.extend(pm.sample_prior_predictive(ndraw, self.model))
        self.trace.extend(pm.sample_posterior_predictive(self.trace, model=self.model, var_names=["y"]))

    def get_trace_dataset(
        self, convert_nmeasure: bool = True, var_names: Optional[Union[str, list[str]]] = None
    ) -> xr.Dataset:
        """Return an xarray Dataset containing a prior/posterior parameter/predictive samples.

        Args:
            convert_nmeasure: if True, convert `nmeasure` coordinate to multi-index comprising `time` and `site`.
            var_names: (list of) variables to select. For instance, "x" will return "x_prior" and "x_posterior".

        Returns:
            xarray Dataset containing a prior/posterior parameter/predictive samples.
        """
        trace_ds = convert_idata_to_dataset(self.trace)

        if convert_nmeasure:
            trace_ds = nmeasure_to_site_time(trace_ds, self.site_indicators, self.site_names, self.times)

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

    def start_time(self) -> np.datetime64:
        """Return start date of inversion."""
        return self.times.min().values  # type: ignore

    def period_midpoint(self) -> np.datetime64:
        """Return midpoint of inversion period."""
        half_of_period = (self.times.max().values - self.times.min().values) / 2
        return self.times.min().values + half_of_period  # type: ignore

    def get_obs(self, unstack_nmeasure: bool = True) -> xr.DataArray:
        """Return y observations.

        By default, `nmeasure` is converted to `site` and `time`.
        """
        result = nmeasure_to_site_time_data_array(self.obs, self.site_indicators, self.site_names, self.times)

        if unstack_nmeasure:
            return result.unstack("nmeasure")

        return result

    def get_obs_err(self, unstack_nmeasure: bool = True) -> xr.DataArray:
        """Return y observations errors.

        By default, `nmeasure` is converted to `site` and `time`.
        """
        result = nmeasure_to_site_time_data_array(
            self.obs_err, self.site_indicators, self.site_names, self.times
        )

        if unstack_nmeasure:
            return result.unstack("nmeasure")

        return result

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
            return result.unstack("nmeasure")

        return result.rename("obs_repeatability")

    def get_obs_variability(self, unstack_nmeasure: bool = True) -> xr.DataArray:
        """Return "variability" uncertainty term for y observations.

        By default, `nmeasure` is converted to `site` and `time`.
        """
        result = nmeasure_to_site_time_data_array(
            self.obs_variability, self.site_indicators, self.site_names, self.times
        )

        if unstack_nmeasure:
            result = result.unstack("nmeasure")

        return xr.zeros_like(result).rename("obs_variability")

    def get_total_err(self, unstack_nmeasure: bool = True, take_mean: bool = True) -> xr.DataArray:
        """Return sqrt(repeatability**2 + variability**2 + model_error**2)

        Args:
            unstack_nmeasure: if True, convert `nmeasure` `site` and `time`. (Default: True)
            take_mean: if True, take mean over trace of error term

        """
        result = self.get_trace_dataset(var_names="epsilon").epsilon_posterior

        if unstack_nmeasure:
            result = result.unstack("nmeasure")

        if take_mean:
            result = result.mean("draw")

        return result.rename("total_error")

    def get_model_err(self, unstack_nmeasure: bool = True) -> xr.DataArray:
        """Return model_error

        By default, `nmeasure` is converted to `site` and `time`.
        """
        total_err = self.get_total_err(unstack_nmeasure=unstack_nmeasure, take_mean=False)
        total_obs_err = self.get_obs_err(unstack_nmeasure=unstack_nmeasure)

        result = np.sqrt(np.maximum(total_err**2 - total_obs_err**2, 0)).mean("draw")  # type: ignore

        return result.rename("model_error")

    @add_suffix("diagnostics")
    def get_diagnostics(self) -> xr.Dataset:
        """Return diagnostics computed by arviz.

        Returns:
            xr.Dataset
        """
        return az.summary(self.trace, kind="diagnostics", fmt="xarray")  # type; ignore
