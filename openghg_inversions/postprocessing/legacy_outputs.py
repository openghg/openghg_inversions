"""Legacy-format output adapters built on top of postprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions import utils
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import (
    make_concentration_outputs,
    make_country_outputs,
    make_flux_outputs,
)


def _compute_apriori_flux(
    flux: xr.DataArray, start_date: str, end_date: str, times: xr.DataArray | np.ndarray | None = None
) -> xr.DataArray:
    """Compute the legacy-style prior flux map from a flux timeseries.

    Args:
        flux: Prior flux with dimensions ``lat``, ``lon``, and optionally ``flux_time``.
        start_date: Inversion start date (YYYY-mm-dd).
        end_date: Inversion end date (YYYY-mm-dd).
        times: Optional measurement times used to weight monthly periods.

    Returns:
        Two-dimensional prior flux map over ``lat`` and ``lon``.

    """
    if "flux_time" not in flux.dims or flux.sizes.get("flux_time", 1) == 1:
        return flux.isel(flux_time=0, drop=True) if "flux_time" in flux.dims else flux

    if times is None:
        month_source = flux.flux_time.values
    else:
        month_source = times.values if isinstance(times, xr.DataArray) else times

    allmonths = _contiguous_month_index(pd.to_datetime(month_source).month.values - 1)
    if len(allmonths) == 0:
        return flux.isel(flux_time=0, drop=True)

    apriori_flux = xr.zeros_like(flux.isel(flux_time=0, drop=True))
    for month_idx in np.unique(allmonths):
        apriori_flux = apriori_flux + flux.isel(flux_time=int(month_idx), drop=True) * (
            np.sum(allmonths == month_idx) / len(allmonths)
        )

    return apriori_flux


def _contiguous_month_index(month_index: np.ndarray) -> np.ndarray:
    """Remap month indices to contiguous 0..N-1 values for positional indexing."""
    month_index = np.asarray(month_index, dtype=int)
    if month_index.size == 0:
        return month_index

    uniq = np.unique(month_index)
    return np.searchsorted(uniq, month_index).astype(int)


def _set_legacy_var_attrs(ds: xr.Dataset, obs_units: str, country_units: str, use_bc: bool) -> None:
    """Set variable attributes to match legacy hbmcmc output style.

    Args:
        ds: Dataset to mutate.
        obs_units: Observation scaling prefix used in legacy units (e.g. ``1e-09``).
        country_units: Units used for country totals.
        use_bc: Whether boundary-condition variables are present.

    Returns:
        None. Attributes are set in-place.

    """
    units = {
        "fluxmode": "mol/m2/s",
        "fluxapriori": "mol/m2/s",
        "Yobs": f"{obs_units} mol/mol",
        "Yerror": f"{obs_units} mol/mol",
        "Yerror_repeatability": f"{obs_units} mol/mol",
        "Yerror_variability": f"{obs_units} mol/mol",
        "Yapriori": f"{obs_units} mol/mol",
        "Ymodmean": f"{obs_units} mol/mol",
        "Ymodmedian": f"{obs_units} mol/mol",
        "Ymodmode": f"{obs_units} mol/mol",
        "Ymod95": f"{obs_units} mol/mol",
        "Ymod68": f"{obs_units} mol/mol",
        "Yoffmean": f"{obs_units} mol/mol",
        "Yoffmedian": f"{obs_units} mol/mol",
        "Yoffmode": f"{obs_units} mol/mol",
        "Yoff95": f"{obs_units} mol/mol",
        "Yoff68": f"{obs_units} mol/mol",
        "countrymean": country_units,
        "countrymedian": country_units,
        "countrymode": country_units,
        "country68": country_units,
        "country95": country_units,
        "countrysd": country_units,
        "countryapriori": country_units,
        "xsensitivity": f"{obs_units} mol/mol",
        "sigtrace": f"{obs_units} mol/mol",
    }

    longname = {
        "Yobs": "observations",
        "Yerror": "measurement error",
        "Ytime": "time of measurements",
        "Yapriori": "a priori simulated measurements",
        "Ymodmean": "mean of posterior simulated measurements",
        "Ymodmedian": "median of posterior simulated measurements",
        "Ymodmode": "mode of posterior simulated measurements",
        "Ymod68": " 0.68 Bayesian credible interval of posterior simulated measurements",
        "Ymod95": " 0.95 Bayesian credible interval of posterior simulated measurements",
        "Yoffmean": "mean of posterior simulated offset between measurements",
        "Yoffmedian": "median of posterior simulated offset between measurements",
        "Yoffmode": "mode of posterior simulated offset between measurements",
        "Yoff68": " 0.68 Bayesian credible interval of posterior simulated offset between measurements",
        "Yoff95": " 0.95 Bayesian credible interval of posterior simulated offset between measurements",
        "xtrace": "trace of unitless scaling factors for emissions parameters",
        "sigtrace": "trace of model error parameters",
        "siteindicator": "index of site of measurement corresponding to sitenames",
        "sigmafreqindex": "perdiod over which the model error is estimated",
        "sitenames": "site names",
        "fluxapriori": "mean a priori flux over period",
        "fluxmode": "mode posterior flux over period",
        "scalingmean": "mean scaling factor field over period",
        "scalingmode": "mode scaling factor field over period",
        "basisfunctions": "basis function field",
        "countrymean": "mean of ocean and country totals",
        "countrymedian": "median of ocean and country totals",
        "countrymode": "mode of ocean and country totals",
        "country68": "0.68 Bayesian credible interval of ocean and country totals",
        "country95": "0.95 Bayesian credible interval of ocean and country totals",
        "countrysd": "standard deviation of ocean and country totals",
        "countryapriori": "prior mean of ocean and country totals",
        "countrydefinition": "grid definition of countries",
        "xsensitivity": "emissions sensitivity timeseries",
    }

    if use_bc:
        units.update(
            {
                "YmodmeanBC": f"{obs_units} mol/mol",
                "YmodmedianBC": f"{obs_units} mol/mol",
                "YmodmodeBC": f"{obs_units} mol/mol",
                "Ymod95BC": f"{obs_units} mol/mol",
                "Ymod68BC": f"{obs_units} mol/mol",
                "YaprioriBC": f"{obs_units} mol/mol",
                "bcsensitivity": f"{obs_units} mol/mol",
            }
        )
        longname.update(
            {
                "YaprioriBC": "a priori simulated boundary conditions",
                "YmodmeanBC": "mean of posterior simulated boundary conditions",
                "YmodmedianBC": "median of posterior simulated boundary conditions",
                "YmodmodeBC": "mode of posterior simulated boundary conditions",
                "Ymod68BC": " 0.68 Bayesian credible interval of posterior simulated boundary conditions",
                "Ymod95BC": " 0.95 Bayesian credible interval of posterior simulated boundary conditions",
                "bctrace": "trace of unitless scaling factors for boundary condition parameters",
                "bcsensitivity": "boundary conditions sensitivity timeseries",
            }
        )

    for dv, unit in units.items():
        if dv in ds:
            ds[dv].attrs["units"] = unit

    for dv, lname in longname.items():
        if dv in ds:
            ds[dv].attrs["longname"] = lname

    for dv in ds.data_vars:
        if "longname" not in ds[dv].attrs:
            ds[dv].attrs["longname"] = str(dv).replace("_", " ")


def _flatten_nmeasure_for_legacy(data: xr.DataArray) -> xr.DataArray:
    """Convert any stacked `nmeasure` coordinate back to the legacy flat form."""
    if "nmeasure" not in data.dims:
        return data

    result = data
    if "nmeasure" in result.indexes:
        result = result.reset_index("nmeasure", drop=True)

    drop_coords = [coord for coord in ("site", "time") if coord in result.coords and "nmeasure" in result[coord].dims]
    if drop_coords:
        result = result.drop_vars(drop_coords)

    return result.assign_coords(nmeasure=np.arange(result.sizes["nmeasure"]))


def make_legacy_hbmcmc_output(
    inv_out: InversionOutput,
    mcmc_results: dict,
    sigma_freq_index: np.ndarray | xr.DataArray,
    Hx: np.ndarray | xr.DataArray,
    Hbc: np.ndarray | xr.DataArray | None = None,
    country_file: str | Path | None = None,
    use_bc: bool = False,
) -> xr.Dataset:
    """Create a legacy-format hbmcmc output dataset from postprocessing products.

    Args:
        inv_out: Inversion outputs container.
        mcmc_results: Raw dictionary returned by ``inferpymc``.
        sigma_freq_index: Sigma frequency index per measurement.
        Hx: Emissions sensitivity matrix with shape ``(nparam, nmeasure)``.
        Hbc: Boundary-condition sensitivity matrix with shape ``(nBC, nmeasure)``.
        country_file: Optional path to country definition file.
        use_bc: Whether BC variables should be included.

    Returns:
        Legacy-style ``xr.Dataset`` matching key variable names/attrs from
        ``inferpymc_postprocessouts``.

    """
    Hx_arr = np.asarray(Hx)
    Hbc_arr = np.asarray(Hbc) if Hbc is not None else None
    sigma_freq = np.asarray(sigma_freq_index)

    conc = make_concentration_outputs(
        inv_out,
        stats=["mean", "median", "mode_kde", "hdi"],
        stats_args={
            "hdi__hdi_prob": [0.68, 0.95],
            "mode_kde__chunk_dim": "nmeasure",
            "mode_kde__chunk_size": 1,
        },
    )
    flux = make_flux_outputs(
        inv_out,
        stats=["mean", "mode_kde"],
        stats_args={"mode_kde__chunk_dim": "nx", "mode_kde__chunk_size": 1},
    )
    country = make_country_outputs(
        inv_out,
        country_file=country_file,
        stats=["mean", "median", "mode_kde", "stdev", "hdi"],
        stats_args={"hdi__hdi_prob": [0.68, 0.95], "mode_kde__chunk_dim": "country", "mode_kde__chunk_size": 1},
    )

    country_obj = utils.get_country(inv_out.domain, country_file=country_file)
    country_idx = country_obj.country
    apriori_flux = _compute_apriori_flux(inv_out.flux, str(inv_out.start_date), str(inv_out.end_date), inv_out.times)

    yapriori = Hx_arr.sum(axis=0)
    if use_bc and Hbc_arr is not None:
        yapriori = yapriori + Hbc_arr.sum(axis=0)

    data_vars: dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray]] = {
        "Yobs": inv_out.obs,
        "Yerror": inv_out.obs_err,
        "Yerror_repeatability": inv_out.obs_repeatability,
        "Yerror_variability": inv_out.obs_variability,
        "Ytime": inv_out.times,
        "Yapriori": ("nmeasure", yapriori),
        "Ymodmean": conc["y_posterior_predictive_mean"],
        "Ymodmedian": conc["y_posterior_predictive_median"],
        "Ymodmode": conc["y_posterior_predictive_mode"],
        "Ymod68": conc["y_posterior_predictive_hdi_68"],
        "Ymod95": conc["y_posterior_predictive_hdi_95"],
        "Yoffmean": conc.get("offset_posterior_mean", xr.zeros_like(inv_out.obs)),
        "Yoffmedian": conc.get("offset_posterior_median", xr.zeros_like(inv_out.obs)),
        "Yoffmode": conc.get("offset_posterior_mode", xr.zeros_like(inv_out.obs)),
        "Yoff68": conc.get("offset_posterior_hdi_68", xr.zeros_like(conc["y_posterior_predictive_hdi_68"])),
        "Yoff95": conc.get("offset_posterior_hdi_95", xr.zeros_like(conc["y_posterior_predictive_hdi_95"])),
        "xtrace": (("steps", "nparam"), mcmc_results["xouts"].values),
        "sigtrace": (("steps", "nsigma_site", "nsigma_time"), mcmc_results["sigouts"].values),
        "siteindicator": inv_out.site_indicators,
        "sigmafreqindex": ("nmeasure", sigma_freq),
        "sitenames": inv_out.site_names,
        "fluxapriori": apriori_flux,
        "fluxmode": flux["flux_posterior_mode"],
        "scalingmean": flux["scaling_posterior_mean"],
        "scalingmode": flux["scaling_posterior_mode"],
        "basisfunctions": inv_out.get_flat_basis(),
        "countrymean": country["country_posterior_mean"],
        "countrymedian": country["country_posterior_median"],
        "countrymode": country["country_posterior_mode"],
        "countrysd": country["country_posterior_stdev"],
        "country68": country["country_posterior_hdi_68"],
        "country95": country["country_posterior_hdi_95"],
        "countryapriori": country["country_prior_mean"],
        "countrydefinition": (("lat", "lon"), country_idx),
        "xsensitivity": (("nmeasure", "nparam"), Hx_arr.T),
    }

    if use_bc and Hbc_arr is not None:
        data_vars.update(
            {
                "YaprioriBC": ("nmeasure", Hbc_arr.sum(axis=0)),
                "YmodmeanBC": conc["mu_bc_posterior_mean"],
                "YmodmedianBC": conc["mu_bc_posterior_median"],
                "YmodmodeBC": conc["mu_bc_posterior_mode"],
                "Ymod95BC": conc["mu_bc_posterior_hdi_95"],
                "Ymod68BC": conc["mu_bc_posterior_hdi_68"],
                "bctrace": (("steps", "nBC"), mcmc_results["bcouts"].values),
                "bcsensitivity": (("nmeasure", "nBC"), Hbc_arr.T),
            }
        )

    for name, value in list(data_vars.items()):
        if isinstance(value, xr.DataArray):
            data_vars[name] = _flatten_nmeasure_for_legacy(value)

    out = xr.Dataset(data_vars)

    obs_units = inv_out.obs.attrs.get("units", "")
    if obs_units.endswith("mol/mol"):
        obs_units = obs_units.split(" ")[0]
    try:
        obs_units = f"{float(obs_units):.0e}"
    except (TypeError, ValueError):
        pass

    country_units = country["country_posterior_mean"].attrs.get("units", "g")
    _set_legacy_var_attrs(out, obs_units=obs_units, country_units=country_units, use_bc=use_bc)

    out.attrs["Start date"] = str(inv_out.start_date)
    out.attrs["End date"] = str(inv_out.end_date)
    if "convergence" in mcmc_results:
        out.attrs["Convergence"] = mcmc_results["convergence"]

    return out
