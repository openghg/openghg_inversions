"""Legacy-format output adapters built on top of postprocessing helpers."""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path
from typing import Any, cast

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions import utils
from openghg_inversions._country_file import load_country_dataset
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
        "min_model_error": f"{obs_units} mol/mol",
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
        "sitelons": "site longitudes corresponding to site names",
        "sitelats": "site latitudes corresponding to site names",
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


def _cast_legacy_float_data_vars(ds: xr.Dataset) -> xr.Dataset:
    """Cast floating legacy data variables to float32 at the product boundary."""
    updates: dict[Hashable, xr.DataArray] = {}
    for name in ds.data_vars:
        if name in ds and np.issubdtype(ds[name].dtype, np.floating):
            updates[name] = ds[name].astype("float32")

    return ds.assign(updates) if updates else ds


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


def _legacy_min_model_error(inv_out: InversionOutput) -> xr.DataArray:
    """Return minimum model error in the legacy flat observation shape."""
    try:
        min_error = _model_or_input_var(
            inv_out,
            model_name=inv_out.variable_name("minimum_error"),
            input_name="min_error",
            description="minimum model error",
        )
    except ValueError:
        min_error = xr.zeros_like(_obs_input(inv_out, "y_obs", "Yobs"))

    if "nmeasure" not in min_error.dims:
        obs = _obs_input(inv_out, "y_obs", "Yobs")
        values = _as_array(min_error)
        if values.size != 1:
            raise ValueError(
                f"Legacy HBMCMC minimum model error must be scalar or have nmeasure; got {min_error.dims!r}."
            )
        min_error = xr.full_like(obs, float(values.reshape(-1)[0]))

    return _flatten_nmeasure_for_legacy(min_error).rename("min_model_error")


def _site_metadata_values(inv_out: InversionOutput, names: tuple[str, ...], nsite: int) -> np.ndarray:
    """Return site-aligned metadata values, or NaNs when unavailable."""
    for metadata in (inv_out.run_metadata, inv_out.output_metadata):
        for name in names:
            values = metadata.get(name)
            if values is None:
                continue
            array = np.asarray(values, dtype=float)
            if array.size != nsite:
                raise ValueError(
                    f"Legacy HBMCMC site metadata {name!r} has {array.size} values for {nsite} sites."
                )
            return array
    return np.full(nsite, np.nan, dtype=float)


def _legacy_site_locations(inv_out: InversionOutput, nsite: int) -> tuple[xr.DataArray, xr.DataArray]:
    """Return site longitudes and latitudes in the historical variables."""
    lons = _site_metadata_values(inv_out, ("sitelons", "site_lons", "site_longitudes"), nsite)
    lats = _site_metadata_values(inv_out, ("sitelats", "site_lats", "site_latitudes"), nsite)
    coords = {"nsite": np.arange(nsite)}
    return (
        xr.DataArray(lons, dims=("nsite",), coords=coords, name="sitelons"),
        xr.DataArray(lats, dims=("nsite",), coords=coords, name="sitelats"),
    )


def _as_array(data: Any) -> np.ndarray:
    """Convert xarray or array-like values to a NumPy array without relying on ``.values``."""
    return np.asarray(data.values if isinstance(data, xr.DataArray) else data)


def _legacy_country_index(domain: str, country_file: str | Path | None = None) -> np.ndarray:
    """Return the raw country index grid used by legacy-format outputs."""
    country_dataset = load_country_dataset(
        utils.get_country_file_path(country_file=country_file, domain=domain)
    )
    if "country" in country_dataset:
        return _as_array(country_dataset["country"])
    if "region" in country_dataset:
        return _as_array(country_dataset["region"])
    raise ValueError("Variables 'country' or 'region' not found in country file.")


def _model_or_input_var(
    inv_out: InversionOutput,
    *,
    model_name: str,
    input_name: str,
    description: str,
) -> xr.DataArray:
    """Return a variable from model constant data, falling back to inversion inputs."""
    try:
        model_data = inv_out.model_data()
    except ValueError:
        model_data = xr.Dataset()

    if model_name in model_data:
        return model_data[model_name]
    if input_name in inv_out.inv_inputs:
        return inv_out.inv_inputs[input_name]
    raise ValueError(
        f"Legacy HBMCMC output formatting requires {description} as model data {model_name!r} "
        f"or InversionOutput.inv_inputs variable {input_name!r}."
    )


def _as_legacy_sensitivity(data: xr.DataArray, description: str) -> np.ndarray:
    """Return a sensitivity matrix with legacy ``(parameter, nmeasure)`` shape."""
    if "nmeasure" not in data.dims:
        if data.ndim != 2:
            raise ValueError(f"{description} must be two-dimensional; got dims {data.dims!r}.")
        return _as_array(data)

    parameter_dims = [dim for dim in data.dims if dim != "nmeasure"]
    if len(parameter_dims) != 1:
        raise ValueError(f"{description} must have one parameter dimension and nmeasure; got {data.dims!r}.")

    return _as_array(data.transpose(parameter_dims[0], "nmeasure"))


def _posterior_first_chain(inv_out: InversionOutput) -> xr.Dataset:
    """Return posterior samples for the first chain, matching the old adapter."""
    posterior = getattr(inv_out.trace, "posterior", None)
    if posterior is None:
        raise ValueError("Legacy HBMCMC output formatting requires posterior trace samples.")
    if "chain" in posterior.dims:
        return posterior.isel(chain=0, drop=True)
    return posterior


def _posterior_var(posterior: xr.Dataset, names: tuple[str, ...], description: str) -> xr.DataArray:
    """Return a posterior variable by first available name."""
    for name in names:
        if name in posterior:
            return posterior[name]

    raise ValueError(f"Legacy HBMCMC output formatting requires posterior samples for {description}.")


def _legacy_trace_matrix(trace: xr.DataArray, description: str) -> np.ndarray:
    """Return first-chain trace samples with legacy ``(steps, nparam)`` shape."""
    if "draw" in trace.dims:
        trace = trace.transpose("draw", ...)
    values = _as_array(trace)
    if values.ndim == 1:
        return values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError(f"{description} trace must be one- or two-dimensional; got {trace.dims!r}.")
    return values


def _legacy_sigma_trace(trace: xr.DataArray) -> np.ndarray:
    """Return first-chain sigma samples with legacy ``(steps, nsigma_site, nsigma_time)`` shape."""
    if "draw" in trace.dims:
        trace = trace.transpose("draw", ...)
    values = _as_array(trace)
    if values.ndim == 1:
        return values[:, np.newaxis, np.newaxis]
    if values.ndim == 2:
        return values[:, np.newaxis, :]
    if values.ndim != 3:
        raise ValueError(f"Sigma trace must have at most three legacy dimensions; got {trace.dims!r}.")
    return values


def _legacy_convergence(inv_out: InversionOutput) -> str:
    """Return legacy convergence status from ArviZ R-hat when enough chains exist."""
    posterior = getattr(inv_out.trace, "posterior", None)
    if posterior is None or posterior.sizes.get("chain", 0) < 2 or posterior.sizes.get("draw", 0) < 2:
        return "Unavailable"

    x_name = inv_out.variable_name("flux_scale")
    if x_name not in posterior and "x" in posterior:
        x_name = "x"
    if x_name not in posterior:
        return "Unavailable"

    try:
        rhat_dataset = cast(xr.Dataset, az.rhat(inv_out.trace, var_names=[x_name]))
        rhat = rhat_dataset[x_name]
        max_rhat = float(rhat.max(skipna=True).item())
    except (AttributeError, KeyError, TypeError, ValueError):
        return "Unavailable"

    if not np.isfinite(max_rhat):
        return "Unavailable"
    return "Failed" if max_rhat > 1.05 else "Passed"


def _legacy_inferpymc_fields(inv_out: InversionOutput, *, use_bc: bool) -> dict[str, np.ndarray]:
    """Derive the old inferpymc result fields needed by the legacy formatter."""
    posterior = _posterior_first_chain(inv_out)
    xouts = _legacy_trace_matrix(
        _posterior_var(posterior, (inv_out.variable_name("flux_scale"), "x"), "emissions scaling"),
        "Emissions scaling",
    )
    sigouts = _legacy_sigma_trace(
        _posterior_var(posterior, ("sigma", inv_out.variable_name("model_error")), "model-error sigma")
    )
    fields = {"xouts": xouts, "sigouts": sigouts}

    if use_bc:
        fields["bcouts"] = _legacy_trace_matrix(
            _posterior_var(posterior, ("bc",), "boundary-condition scaling"),
            "Boundary-condition scaling",
        )

    return fields


def _legacy_postprocess_fields(inv_out: InversionOutput, *, use_bc: bool) -> dict[str, np.ndarray]:
    """Derive old postprocess/inferpymc field names from modern inversion output."""
    Hx = _as_legacy_sensitivity(
        _model_or_input_var(
            inv_out,
            model_name=inv_out.variable_name("emissions_sensitivity"),
            input_name="H",
            description="emissions sensitivity",
        ),
        "Emissions sensitivity",
    )
    Hbc = (
        _as_legacy_sensitivity(
            _model_or_input_var(
                inv_out,
                model_name=inv_out.variable_name("baseline_sensitivity"),
                input_name="H_bc",
                description="boundary-condition sensitivity",
            ),
            "Boundary-condition sensitivity",
        )
        if use_bc
        else None
    )
    sigma_freq_index = _model_or_input_var(
        inv_out,
        model_name="sigma_freq_index",
        input_name="sigma_freq_index",
        description="sigma frequency index",
    )
    sigma_freq = _as_array(
        sigma_freq_index.transpose("nmeasure") if "nmeasure" in sigma_freq_index.dims else sigma_freq_index
    ).astype(int, copy=False)

    fields = {
        "Hx": Hx,
        "sigma_freq_index": sigma_freq,
        **_legacy_inferpymc_fields(inv_out, use_bc=use_bc),
    }
    if Hbc is not None:
        fields["Hbc"] = Hbc
    return fields


def _legacy_flat_basis(inv_out: InversionOutput) -> xr.DataArray:
    """Return the legacy zero-based basis-function map."""
    basis = flat_basis_for_output(inv_out)
    values = _as_array(basis)
    if values.size and np.nanmin(values) >= 1:
        values = values - 1
    return xr.DataArray(
        values, dims=basis.dims, coords=basis.coords, attrs=basis.attrs, name="basisfunctions"
    )


def _format_legacy_attr_prior(prior: object) -> str | None:
    """Format prior metadata using the historical comma-separated attr shape."""
    if not isinstance(prior, dict):
        return None
    return ",".join(f"{key},{value}" for key, value in prior.items())


def _legacy_single_sector_x_prior(model_metadata: dict[str, Any]) -> object | None:
    """Return the single-sector emissions prior stored by modern RHIME metadata."""
    sectors = model_metadata.get("sectors")
    if isinstance(sectors, (list, tuple)) and sectors:
        sector = sectors[0]
        if isinstance(sector, dict):
            return sector.get("x_prior")
    return None


def _legacy_hbmcmc_attrs(inv_out: InversionOutput) -> dict[str, str]:
    """Return legacy dataset attrs from explicit compatibility metadata or modern metadata."""
    attrs: dict[str, str] = {}
    explicit_attrs = inv_out.output_metadata.get("legacy_hbmcmc_attrs")
    if isinstance(explicit_attrs, dict):
        attrs.update({str(key): str(value) for key, value in explicit_attrs.items()})

    sampler = inv_out.output_metadata.get("sampler")
    if isinstance(sampler, dict):
        if "burn" in sampler:
            attrs.setdefault("Burn in", str(int(sampler["burn"])))
        if "tune" in sampler:
            attrs.setdefault("Tuning steps", str(int(sampler["tune"])))
        if "chains" in sampler:
            attrs.setdefault("Number of chains", str(int(sampler["chains"])))

    posterior = getattr(inv_out.trace, "posterior", None)
    if posterior is not None and "chain" in posterior.sizes:
        attrs.setdefault("Number of chains", str(int(posterior.sizes["chain"])))

    model_metadata = inv_out.model_metadata
    attrs.setdefault("Error for each site", str(model_metadata.get("sigma_per_site", "Unavailable")))

    prior_attrs = {
        "Emissions Prior": _format_legacy_attr_prior(_legacy_single_sector_x_prior(model_metadata)),
        "Model error Prior": _format_legacy_attr_prior(model_metadata.get("sigma_prior")),
        "BCs Prior": _format_legacy_attr_prior(model_metadata.get("bc_prior")),
        "Offset Prior": _format_legacy_attr_prior(model_metadata.get("offset_prior")),
    }
    attrs.update({name: value for name, value in prior_attrs.items() if name not in attrs and value})

    return attrs


def make_legacy_hbmcmc_output(
    inv_out: InversionOutput,
    country_file: str | Path | None = None,
    use_bc: bool = False,
) -> xr.Dataset:
    """Create a legacy-format hbmcmc output dataset from modern inversion output.

    TODO: needs to handle offsets

    Args:
        inv_out: Inversion outputs container.
        country_file: Optional path to country definition file.
        use_bc: Whether BC variables should be included.

    Returns:
        Legacy-style ``xr.Dataset`` matching key variable names/attrs from
        ``inferpymc_postprocessouts``.

    """
    domain = _require_legacy_domain(inv_out)
    legacy_fields = _legacy_postprocess_fields(inv_out, use_bc=use_bc)
    Hx = legacy_fields["Hx"]
    Hbc = legacy_fields.get("Hbc")
    xouts = legacy_fields["xouts"]
    sigouts = legacy_fields["sigouts"]
    bcouts = legacy_fields.get("bcouts")
    sigma_freq_index = legacy_fields["sigma_freq_index"]
    times = _legacy_measurement_times(inv_out)
    site_indicators, site_names = _legacy_site_fields(inv_out)
    site_lons, site_lats = _legacy_site_locations(inv_out, site_names.sizes["nsite"])

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
        stats_args={
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

    country_idx = _legacy_country_index(domain, country_file=country_file)
    apriori_flux = _compute_apriori_flux(
        inv_out.flux,
        str(inv_out.start_time.date()),
        str(inv_out.end_time.date()),
        times,
    )

    yapriori = Hx.sum(axis=0)
    if use_bc and Hbc is not None and bcouts is not None:
        yapriori = yapriori + Hbc.sum(axis=0)

    obs = _obs_input(inv_out, "y_obs", "Yobs")
    zero_obs = xr.zeros_like(obs)
    zero_hdi = xr.zeros_like(conc["y_posterior_predictive_hdi_68"])

    data_vars: dict[str, Any] = {
        "Yobs": obs,
        "Yerror": _obs_input(inv_out, "y_obs_error", "Yerror"),
        "Yerror_repeatability": _obs_input(inv_out, "y_obs_repeatability", "Yerror_repeatability"),
        "Yerror_variability": _obs_input(inv_out, "y_obs_variability", "Yerror_variability"),
        "min_model_error": _legacy_min_model_error(inv_out),
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
        "xtrace": (("steps", "nparam"), xouts),
        "sigtrace": (("steps", "nsigma_site", "nsigma_time"), sigouts),
        "siteindicator": site_indicators,
        "sigmafreqindex": ("nmeasure", sigma_freq_index),
        "sitenames": site_names,
        "sitelons": site_lons,
        "sitelats": site_lats,
        "fluxapriori": apriori_flux,
        "fluxmode": flux["flux_posterior_mode"],
        "scalingmean": flux["scaling_posterior_mean"],
        "scalingmode": flux["scaling_posterior_mode"],
        "basisfunctions": _legacy_flat_basis(inv_out),
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

    if use_bc and Hbc is not None and bcouts is not None:
        data_vars.update(
            {
                "YaprioriBC": ("nmeasure", Hbc.sum(axis=0)),
                "YmodmeanBC": conc["mu_bc_posterior_mean"],
                "YmodmedianBC": conc["mu_bc_posterior_median"],
                "YmodmodeBC": conc["mu_bc_posterior_mode"],
                "Ymod95BC": conc["mu_bc_posterior_hdi_95"],
                "Ymod68BC": conc["mu_bc_posterior_hdi_68"],
                "bctrace": (("steps", "nBC"), bcouts),
                "bcsensitivity": (("nmeasure", "nBC"), Hbc.T),
            }
        )

    for name, value in list(data_vars.items()):
        if isinstance(value, xr.DataArray):
            data_vars[name] = _flatten_nmeasure_for_legacy(value)

    coords: dict[str, Any] = {
        "stepnum": ("steps", np.arange(xouts.shape[0])),
        "paramnum": ("nparam", np.arange(Hx.shape[0])),
        "measurenum": ("nmeasure", np.arange(Hx.shape[1])),
        "nsigma_site": ("nsigma_site", np.arange(sigouts.shape[1])),
        "nsigma_time": ("nsigma_time", np.arange(sigouts.shape[2])),
        "countrynames": country["countrynames"],
    }
    if "nUI" in conc.dims or "nUI" in country.dims:
        n_ui = conc.sizes.get("nUI", country.sizes.get("nUI"))
        if n_ui is not None:
            coords["UInum"] = ("nUI", np.arange(n_ui))
    if use_bc and Hbc is not None:
        coords["numBC"] = ("nBC", np.arange(Hbc.shape[0]))

    out = _cast_legacy_float_data_vars(xr.Dataset(data_vars, coords=coords))

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
    out.attrs["Convergence"] = _legacy_convergence(inv_out)
    out.attrs.update(_legacy_hbmcmc_attrs(inv_out))

    return out
