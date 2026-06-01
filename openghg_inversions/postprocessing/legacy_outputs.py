"""Legacy-format output adapters built on top of postprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions import utils
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import (
    flat_basis_for_output,
    make_concentration_outputs,
    make_country_outputs,
    make_flux_outputs,
    observation_inputs_for_outputs,
)


def _require_legacy_domain(inv_out: InversionOutput) -> str:
    """Return domain metadata required by the legacy-format product."""
    if inv_out.is_multisector:
        raise ValueError("Legacy HBMCMC output formatting supports only single-sector RHIME outputs.")

    domain = inv_out.domain
    if domain is None:
        raise ValueError("Legacy HBMCMC output formatting requires InversionOutput metadata field 'domain'.")
    return domain


def _compute_apriori_flux(
    flux: xr.DataArray, start_date: str, end_date: str, times: xr.DataArray | np.ndarray | None = None
) -> xr.DataArray:
    """Compute the legacy-style prior flux map from a flux timeseries.

    Args:
        flux: Prior flux with dimensions ``lat``, ``lon``, and optionally ``flux_time``.
        start_date: Inversion start date (YYYY-mm-dd). Retained for call-site compatibility.
        end_date: Inversion end date (YYYY-mm-dd). Retained for call-site compatibility.
        times: Optional measurement times used to weight available flux periods.

    Returns:
        Two-dimensional prior flux map over ``lat`` and ``lon``.

    """
    if "flux_time" not in flux.dims or flux.sizes.get("flux_time", 1) == 1:
        return flux.isel(flux_time=0, drop=True) if "flux_time" in flux.dims else flux

    if times is None:
        time_source = flux.flux_time.values
    else:
        time_source = times.values if isinstance(times, xr.DataArray) else times

    flux_period = utils._infer_flux_period(flux.flux_time.values, flux.attrs.get("time_period"))
    allmonths = utils._map_times_to_available_period_positions(
        time_source, flux.flux_time.values, flux_period
    )
    if len(allmonths) == 0:
        return flux.isel(flux_time=0, drop=True)

    apriori_flux = xr.zeros_like(flux.isel(flux_time=0, drop=True))
    for month_idx in np.unique(allmonths):
        apriori_flux = apriori_flux + flux.isel(flux_time=int(month_idx), drop=True) * (
            np.sum(allmonths == month_idx) / len(allmonths)
        )

    return apriori_flux


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
        "sigmafreqindex": "period over which the model error is estimated",
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


def _rename_hdi_for_legacy(ds: xr.Dataset) -> xr.Dataset:
    """Rename postprocessing HDI metadata to legacy names."""
    if "hdi" not in ds.dims and "hdi" not in ds.coords:
        return ds

    result = ds.rename(hdi="nUI")

    # Legacy files carry a separate `UInum` coordinate rather than string labels on `nUI`.
    if "nUI" in result.dims:
        if "nUI" in result.coords:
            result = result.drop_vars("nUI")
        result = result.assign_coords(UInum=("nUI", np.arange(result.sizes["nUI"])))

    return result


def _rename_country_for_legacy(ds: xr.Dataset) -> xr.Dataset:
    """Rename country metadata to legacy names."""
    if "country" not in ds.dims and "country" not in ds.coords:
        return ds

    result = ds.rename(country="countrynames")
    return result


def _collapse_flux_time_for_legacy(
    ds: xr.Dataset,
    times: xr.DataArray | np.ndarray,
    flux_time: xr.DataArray | np.ndarray,
    time_period: str | None = None,
) -> xr.Dataset:
    """Collapse `flux_time` to a single legacy period using observation-weighted averaging."""
    if "flux_time" not in ds.dims:
        return ds

    flux_period = utils._infer_flux_period(flux_time, time_period)
    period_index = utils._map_times_to_available_period_positions(times, flux_time, flux_period)

    if period_index.size == 0:
        return ds.isel(flux_time=0, drop=True)

    weights = np.zeros(len(flux_time), dtype=float)
    for period_pos in np.unique(period_index):
        weights[int(period_pos)] = np.sum(period_index == period_pos) / len(period_index)

    weights_da = xr.DataArray(weights, dims=("flux_time",), coords={"flux_time": ds["flux_time"]})
    return (ds * weights_da).sum("flux_time")


def _flatten_nmeasure_for_legacy(data: xr.DataArray) -> xr.DataArray:
    """Convert any stacked `nmeasure` coordinate back to the legacy flat form."""
    if "nmeasure" not in data.dims:
        return data

    result = data
    if "nmeasure" in result.indexes:
        result = result.reset_index("nmeasure", drop=True)

    drop_coords = [
        coord for coord in ("site", "time") if coord in result.coords and "nmeasure" in result[coord].dims
    ]
    if drop_coords:
        result = result.drop_vars(drop_coords)

    return result.assign_coords(nmeasure=np.arange(result.sizes["nmeasure"]))


def _legacy_measurement_times(inv_out: InversionOutput) -> xr.DataArray:
    """Return measurement times as a flat legacy ``nmeasure`` array."""
    obs_inputs = observation_inputs_for_outputs(inv_out)
    nmeasure_index = obs_inputs.indexes.get("nmeasure")
    if isinstance(nmeasure_index, pd.MultiIndex) and "time" in nmeasure_index.names:
        times = nmeasure_index.get_level_values("time").to_numpy(dtype="datetime64[ns]")
        return xr.DataArray(
            times, dims=("nmeasure",), coords={"nmeasure": np.arange(len(times))}, name="Ytime"
        )

    if "time" in obs_inputs.coords and obs_inputs["time"].dims == ("nmeasure",):
        return _flatten_nmeasure_for_legacy(obs_inputs["time"]).rename("Ytime")

    if "time" not in inv_out.inv_inputs:
        raise ValueError("Could not recover measurement times for legacy output formatting.")

    values = inv_out.inv_inputs["time"].values
    return xr.DataArray(
        np.asarray(values),
        dims=("nmeasure",),
        coords={"nmeasure": np.arange(len(values))},
        name="Ytime",
    )


def _legacy_site_fields(
    inv_out: InversionOutput,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return site indicators and site names in the legacy flat format."""
    obs_inputs = observation_inputs_for_outputs(inv_out)
    nmeasure_index = obs_inputs.indexes.get("nmeasure")
    if isinstance(nmeasure_index, pd.MultiIndex) and "site" in nmeasure_index.names:
        site_values = nmeasure_index.get_level_values("site")
        names = list(pd.unique(site_values))
        name_to_index = {name: idx for idx, name in enumerate(names)}
        indicators = np.array([name_to_index[value] for value in site_values], dtype=int)
        return (
            xr.DataArray(
                indicators,
                dims=("nmeasure",),
                coords={"nmeasure": np.arange(len(indicators))},
                name="siteindicator",
            ),
            xr.DataArray(
                np.asarray(names),
                dims=("nsite",),
                coords={"nsite": np.arange(len(names))},
                name="sitenames",
            ),
        )

    if "site_indicator" not in inv_out.inv_inputs:
        raise ValueError("Could not recover site indicators for legacy output formatting.")

    indicator_values = inv_out.inv_inputs["site_indicator"].values
    indicators = np.asarray(indicator_values)
    if "site_names" in inv_out.inv_inputs:
        site_names = inv_out.inv_inputs["site_names"]
    else:
        site_names = inv_out.run_metadata.get("sites")
    if site_names is None:
        names = np.unique(indicators)
    else:
        names = site_names.values if isinstance(site_names, xr.DataArray) else site_names

    return (
        xr.DataArray(
            indicators.astype(int, copy=False),
            dims=("nmeasure",),
            coords={"nmeasure": np.arange(len(indicators))},
            name="siteindicator",
        ),
        xr.DataArray(
            np.asarray(names),
            dims=("nsite",),
            coords={"nsite": np.arange(len(names))},
            name="sitenames",
        ),
    )


def _obs_input(inv_out: InversionOutput, name: str, legacy_name: str) -> xr.DataArray:
    """Return an observation-input field renamed and flattened for legacy output."""
    return _flatten_nmeasure_for_legacy(observation_inputs_for_outputs(inv_out)[name]).rename(legacy_name)


def _flux_scale_trace(inv_out: InversionOutput) -> xr.Dataset:
    """Return flux-scale traces using legacy-compatible ``x`` names."""
    trace = inv_out.trace_dataset(var_roles="flux_scale")
    flux_scale_name = inv_out.variable_name("flux_scale")
    return trace.rename(
        {
            data_var: str(data_var).replace(f"{flux_scale_name}_", "x_", 1)
            for data_var in trace.data_vars
            if str(data_var).startswith(f"{flux_scale_name}_")
        }
    )


def _as_array(data: Any) -> np.ndarray:
    """Convert xarray or array-like values to a NumPy array without relying on ``.values``."""
    return np.asarray(data.values if isinstance(data, xr.DataArray) else data)


def make_legacy_hbmcmc_output(
    inv_out: InversionOutput,
    mcmc_results: dict,
    sigma_freq_index: np.ndarray | xr.DataArray,
    Hx: np.ndarray | xr.DataArray,
    Hbc: np.ndarray | xr.DataArray | None = None,
    country_file: str | Path | None = None,
    use_bc: bool = False,
) -> xr.Dataset:
    """Create a legacy-format hbmcmc output dataset from compatibility inputs.

    TODO: needs to handle offsets

    Args:
        inv_out: Inversion outputs container.
        mcmc_results: Compatibility sampling outputs returned by ``inferpymc``.
        sigma_freq_index: Sigma frequency index per measurement.
        Hx: Emissions sensitivity matrix with shape ``(nparam, nmeasure)``.
        Hbc: Boundary-condition sensitivity matrix with shape ``(nBC, nmeasure)``.
        country_file: Optional path to country definition file.
        use_bc: Whether BC variables should be included.

    Returns:
        Legacy-style ``xr.Dataset`` matching key variable names/attrs from
        ``inferpymc_postprocessouts``.

    """
    Hx_arr = _as_array(Hx)
    Hbc_arr = _as_array(Hbc) if Hbc is not None else None
    sigma_freq = _as_array(sigma_freq_index)
    domain = _require_legacy_domain(inv_out)
    times = _legacy_measurement_times(inv_out)
    site_indicators, site_names = _legacy_site_fields(inv_out)

    conc = make_concentration_outputs(
        inv_out,
        stats=["mean", "median", "mode_kde", "hdi"],
        stats_args={
            "hdi__hdi_prob": [0.68, 0.95],
            "mode_kde__chunk_dim": "nmeasure",
            "mode_kde__chunk_size": 1,
        },
    )
    x_trace = _flux_scale_trace(inv_out)
    state_dim = inv_out.basis_functions.operator.meta.state_dim
    flux_chunk_dim = state_dim if state_dim in x_trace.dims else "nx" if "nx" in x_trace.dims else "region"
    flux = make_flux_outputs(
        inv_out,
        stats=["mean", "mode_kde"],
        stats_args={
            "mode_kde__chunk_dim": flux_chunk_dim,
            "mode_kde__chunk_size": 1,
        },
    )
    country = make_country_outputs(
        inv_out,
        country_file=country_file,
        stats=["mean", "median", "mode_kde", "stdev", "hdi"],
        stats_args={
            "hdi__hdi_prob": [0.68, 0.95],
            "mode_kde__chunk_dim": "country",
            "mode_kde__chunk_size": 1,
        },
    )
    conc = _rename_hdi_for_legacy(conc)
    country = _collapse_flux_time_for_legacy(
        country,
        times,
        inv_out.flux["flux_time"],
        inv_out.flux.attrs.get("time_period"),
    )
    country = _rename_hdi_for_legacy(_rename_country_for_legacy(country))

    country_obj = utils.get_country(domain, country_file=country_file)
    country_idx = country_obj.country
    apriori_flux = _compute_apriori_flux(
        inv_out.flux,
        str(inv_out.start_time.date()),
        str(inv_out.end_time.date()),
        times,
    )

    yapriori = Hx_arr.sum(axis=0)
    if use_bc and Hbc_arr is not None:
        yapriori = yapriori + Hbc_arr.sum(axis=0)

    obs = _obs_input(inv_out, "y_obs", "Yobs")
    zero_obs = xr.zeros_like(obs)
    zero_hdi = xr.zeros_like(conc["y_posterior_predictive_hdi_68"])

    data_vars: dict[str, Any] = {
        "Yobs": obs,
        "Yerror": _obs_input(inv_out, "y_obs_error", "Yerror"),
        "Yerror_repeatability": _obs_input(inv_out, "y_obs_repeatability", "Yerror_repeatability"),
        "Yerror_variability": _obs_input(inv_out, "y_obs_variability", "Yerror_variability"),
        "Ytime": times,
        "Yapriori": ("nmeasure", yapriori),
        "Ymodmean": conc["y_posterior_predictive_mean"],
        "Ymodmedian": conc["y_posterior_predictive_median"],
        "Ymodmode": conc["y_posterior_predictive_mode"],
        "Ymod68": conc["y_posterior_predictive_hdi_68"],
        "Ymod95": conc["y_posterior_predictive_hdi_95"],
        "Yoffmean": conc.get("offset_posterior_mean", zero_obs),
        "Yoffmedian": conc.get("offset_posterior_median", zero_obs),
        "Yoffmode": conc.get("offset_posterior_mode", zero_obs),
        "Yoff68": conc.get("offset_posterior_hdi_68", zero_hdi),
        "Yoff95": conc.get("offset_posterior_hdi_95", zero_hdi),
        "xtrace": (("steps", "nparam"), _as_array(mcmc_results["xouts"])),
        "sigtrace": (("steps", "nsigma_site", "nsigma_time"), _as_array(mcmc_results["sigouts"])),
        "siteindicator": site_indicators,
        "sigmafreqindex": ("nmeasure", sigma_freq),
        "sitenames": site_names,
        "fluxapriori": apriori_flux,
        "fluxmode": flux["flux_posterior_mode"],
        "scalingmean": flux["scaling_posterior_mean"],
        "scalingmode": flux["scaling_posterior_mode"],
        "basisfunctions": flat_basis_for_output(inv_out),
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
                "bctrace": (("steps", "nBC"), _as_array(mcmc_results["bcouts"])),
                "bcsensitivity": (("nmeasure", "nBC"), Hbc_arr.T),
            }
        )

    for name, value in list(data_vars.items()):
        if isinstance(value, xr.DataArray):
            data_vars[name] = _flatten_nmeasure_for_legacy(value)

    xouts_arr = _as_array(mcmc_results["xouts"])
    sigouts_arr = _as_array(mcmc_results["sigouts"])
    coords: dict[str, Any] = {
        "stepnum": ("steps", np.arange(xouts_arr.shape[0])),
        "paramnum": ("nparam", np.arange(Hx_arr.shape[0])),
        "measurenum": ("nmeasure", np.arange(Hx_arr.shape[1])),
        "nsigma_site": ("nsigma_site", np.arange(sigouts_arr.shape[1])),
        "nsigma_time": ("nsigma_time", np.arange(sigouts_arr.shape[2])),
        "countrynames": country["countrynames"],
    }
    if "nUI" in conc.dims or "nUI" in country.dims:
        n_ui = conc.sizes.get("nUI", country.sizes.get("nUI"))
        if n_ui is not None:
            coords["UInum"] = ("nUI", np.arange(n_ui))
    if use_bc and Hbc_arr is not None:
        coords["numBC"] = ("nBC", np.arange(Hbc_arr.shape[0]))

    out = xr.Dataset(data_vars, coords=coords)

    obs_units = observation_inputs_for_outputs(inv_out)["y_obs"].attrs.get("units", "")
    if obs_units.endswith("mol/mol"):
        obs_units = obs_units.split(" ")[0]
    try:
        obs_units = f"{float(obs_units):.0e}"
    except (TypeError, ValueError):
        pass

    country_units = country["country_posterior_mean"].attrs.get("units", "g")
    _set_legacy_var_attrs(out, obs_units=obs_units, country_units=country_units, use_bc=use_bc)

    out.attrs["Start date"] = str(inv_out.start_time.date())
    out.attrs["End date"] = str(inv_out.end_time.date())
    if "convergence" in mcmc_results:
        out.attrs["Convergence"] = mcmc_results["convergence"]

    return out
