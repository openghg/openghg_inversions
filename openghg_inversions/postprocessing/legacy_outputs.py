"""Legacy-format output adapters built on top of postprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from openghg_inversions import utils
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import (
    make_concentration_outputs,
    make_country_outputs,
    make_flux_outputs,
)


def _set_legacy_var_attrs(ds: xr.Dataset, obs_units: str, country_units: str, use_bc: bool) -> None:
    """Set variable attrs to match legacy hbmcmc output style."""
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


def make_legacy_hbmcmc_output(
    inv_out: InversionOutput,
    mcmc_results: dict,
    sigma_freq_index,
    Hx,
    Hbc=None,
    country_file: str | Path | None = None,
    use_bc: bool = False,
) -> xr.Dataset:
    """Create a legacy-style hbmcmc dataset using postprocessing helpers."""
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

    yapriori = Hx.sum(axis=0)
    if use_bc and Hbc is not None:
        yapriori = yapriori + Hbc.sum(axis=0)

    data_vars = {
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
        "sigmafreqindex": ("nmeasure", sigma_freq_index),
        "sitenames": inv_out.site_names,
        "fluxapriori": flux["flux_prior_mode"],
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
        "xsensitivity": (("nmeasure", "nparam"), Hx.T),
    }

    if use_bc and Hbc is not None:
        data_vars.update(
            {
                "YaprioriBC": ("nmeasure", Hbc.sum(axis=0)),
                "YmodmeanBC": conc["mu_bc_posterior_mean"],
                "YmodmedianBC": conc["mu_bc_posterior_median"],
                "YmodmodeBC": conc["mu_bc_posterior_mode"],
                "Ymod95BC": conc["mu_bc_posterior_hdi_95"],
                "Ymod68BC": conc["mu_bc_posterior_hdi_68"],
                "bctrace": (("steps", "nBC"), mcmc_results["bcouts"].values),
                "bcsensitivity": (("nmeasure", "nBC"), Hbc.T),
            }
        )

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


# backward-compatible alias
make_legacy_hbmcmc_output_from_postprocessing = make_legacy_hbmcmc_output
