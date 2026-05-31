"""Modern public RHIME run functions and lightweight run specifications.

This module is the public RHIME runner boundary. It accepts Python keyword
arguments or RHIME ``.ini`` files, normalizes legacy spelling into the modern
spec vocabulary, prepares inversion inputs, builds a PyMC model, samples it,
and writes requested outputs.

Terminology used by the RHIME API:

- ``species`` is the primary gas or tracer name used for object-store lookup
  and output naming.
- ``source`` is the OpenGHG metadata key used to retrieve flux data. In
  sector-resolved inputs it is also the xarray coordinate on flux and
  sensitivity data.
- ``flux_sources`` is the RHIME config/API field containing requested OpenGHG
  flux ``source`` values.
- ``sector_sources`` optionally maps model sector names to OpenGHG flux
  ``source`` values when those labels differ.
- ``sector`` is a model component optimized separately in a multi-sector RHIME
  run, usually backed by one flux ``source``.
- ``tracer`` is an additional species used to constrain the primary species,
  normally with linked forward models. The current RHIME preparation path does
  not support tracer inversions.
- ``emissions_name`` is accepted only as a legacy compatibility spelling when
  ``flux_sources`` is absent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import inspect
from pathlib import Path
from typing import Any, cast
import time

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from openghg.util import split_function_inputs  # pyright: ignore[reportPrivateImportUsage]

from openghg_inversions.array_ops import sparse_xr_dot
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions import _rhime_params as rhime_params
from openghg_inversions._rhime_params import params_from_config, resolve_flux_sources
from openghg_inversions._rhime_specs import (
    OutputFormat,
    RhimeModelSpec,
    RhimeOutputSpec,
    RhimeRunSpec,
    RhimeSamplingSpec,
    SectorSpec,
)
from openghg_inversions.inversion_data import (
    RhimePreparedInputs as _RhimePreparedInputs,
    prepare_rhime_inputs as _prepare_rhime_inputs,
)
from openghg_inversions.models import (
    DEFAULT_X_PRIOR,
    build_rhime_model_from_spec,
    build_rhime_multisector_model_from_spec,
    safe_pymc_name,
)
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.utils import ncdf_encoding

__all__ = [
    "SectorSpec",
    "RhimeModelSpec",
    "RhimeOutputSpec",
    "RhimeSamplingSpec",
    "RhimeRunSpec",
    "RhimeResult",
    "params_from_config",
    "resolve_flux_sources",
    "run_rhime",
    "run_rhime_multisector",
]


@dataclass(frozen=True)
class _RhimeRunnerSetup:
    """Normalized RHIME setup derived from config or direct API parameters."""

    run_spec: RhimeRunSpec
    data_args: dict[str, Any]


@dataclass
class RhimeResult:
    """Modern RHIME run result.

    Args:
        run_spec: Top-level run metadata.
        model_spec: Model specification used to build the PyMC model.
        output_spec: Output settings used by the run.
        inv_inputs: Canonical xarray inversion inputs consumed by the model.
        idata: ArviZ ``InferenceData`` returned by sampling.
        output_metadata: Paths and notes for generated outputs.
        outputs: In-memory derived outputs keyed by output kind.
        model: Built PyMC model.
        inv_out: Modern inversion output object when created.
        basis_functions: Retained flux basis functions from preparation.
        sampling_spec: Sampling settings used by the run.
    """

    run_spec: RhimeRunSpec
    model_spec: RhimeModelSpec
    output_spec: RhimeOutputSpec
    inv_inputs: xr.Dataset
    idata: az.InferenceData
    output_metadata: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    basis_functions: BasisFunctions | None = None
    model: pm.Model | None = None
    inv_out: InversionOutput | None = None
    sampling_spec: RhimeSamplingSpec = field(default_factory=RhimeSamplingSpec)


def _validate_output_format(output_format: str) -> None:
    """Raise if a RHIME output format is not supported by the modern runners."""
    valid_formats = {"none", "inv_out", "basic", "paris"}
    if output_format not in valid_formats:
        raise ValueError(
            f"Unsupported RHIME output_format {output_format!r}; expected one of {sorted(valid_formats)!r}."
        )


def _validate_output_path_settings(
    *,
    output_format: str,
    output_path: str | None,
    save_trace: str | Path | bool,
    save_inversion_output: str | Path | bool,
    multisector: bool,
) -> None:
    """Raise if output settings imply a default save path but none is supplied."""
    if output_format == "none":
        return
    if output_path is not None:
        return
    if save_trace is True:
        raise ValueError("`output_path` is required when `save_trace=True`.")
    if not multisector and save_inversion_output is True:
        raise ValueError("`output_path` is required when saving the standard RHIME InversionOutput.")


def _resolve_output_path(
    save_setting: str | Path | bool, output_path: str | None, filename: str
) -> Path | None:
    """Resolve an optional output path from a bool/path save setting."""
    if not save_setting:
        return None
    if isinstance(save_setting, str | Path):
        return Path(save_setting)
    if output_path is None:
        raise ValueError("An output path is required when saving RHIME artifacts.")
    return Path(output_path) / filename


def _define_output_filename(
    output_path: str | Path,
    species: str,
    domain: str,
    output_name: str,
    start_date: str,
    *,
    ext: str = ".nc",
) -> Path:
    """Create the RHIME output filename used for derived NetCDF products."""
    return Path(output_path) / f"{output_name}_{species}_{domain}_{start_date}{ext}"


def _save_inferencedata(idata: az.InferenceData, path: str | Path) -> None:
    """Save InferenceData, preferring the h5netcdf backend with fallbacks."""
    failures = []
    for engine in ("h5netcdf", None, "netcdf4"):
        try:
            if engine is None:
                idata.to_netcdf(str(path), compress=True)
            else:
                idata.to_netcdf(str(path), engine=engine, compress=True)
        except Exception as exc:
            engine_name = "arviz-default" if engine is None else engine
            failures.append(f"{engine_name}: {exc}")
        else:
            return

    joined_failures = "\n".join(failures)
    raise RuntimeError(
        f"Could not save RHIME trace to {path}. Tried h5netcdf, ArviZ default, and netcdf4:\n{joined_failures}"
    )


def _make_model_spec(
    *,
    species: str,
    domain: str,
    flux_sources: list[str],
    x_prior: dict[str, Any] | None,
    sector_priors: Mapping[str, dict[str, Any]] | None,
    sector_sources: Mapping[str, str] | None,
    bc_prior: dict[str, Any] | None,
    sigma_prior: dict[str, Any] | None,
    offset_prior: dict[str, Any] | None,
    use_bc: bool,
    sigma_per_site: bool,
    add_offset: bool,
    pollution_events_from_obs: bool,
    no_model_error: bool,
    power: dict[str, Any] | float,
    offset_args: dict[str, Any] | None,
) -> RhimeModelSpec:
    """Create a lightweight model spec from normalized run parameters."""
    default_x_prior = DEFAULT_X_PRIOR.copy() if x_prior is None else x_prior.copy()
    sectors = []
    used_suffixes: set[str] = set()
    if sector_sources is not None:
        mapped_sources = list(dict.fromkeys(sector_sources.values()))
        if set(mapped_sources) != set(flux_sources):
            raise ValueError(
                "`sector_sources` values must match `flux_sources` so RHIME can retrieve the "
                "OpenGHG data used by each sector."
            )
        sector_items = list(sector_sources.items())
    else:
        sector_items = [(source, source) for source in flux_sources]

    for name, source in sector_items:
        suffix = safe_pymc_name(name)
        if suffix in used_suffixes:
            raise ValueError(
                "Sector names must be unique after PyMC name sanitisation; "
                f"duplicate sanitized name {suffix!r}."
            )
        used_suffixes.add(suffix)
        prior = (
            sector_priors[name] if sector_priors is not None and name in sector_priors else default_x_prior
        )
        sectors.append(
            SectorSpec(
                name=name,
                flux_source=source,
                x_prior=dict(prior),
                variable_suffix=suffix,
            )
        )
    return RhimeModelSpec(
        species=species,
        domain=domain,
        sectors=tuple(sectors),
        use_bc=use_bc,
        sigma_per_site=sigma_per_site,
        add_offset=add_offset,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        power=power,
        bc_prior=bc_prior,
        sigma_prior=sigma_prior,
        offset_prior=offset_prior,
        offset_args=offset_args,
    )


def _sample_model(
    model: pm.Model,
    sampling_spec: RhimeSamplingSpec,
) -> az.InferenceData:
    """Sample a built RHIME model using normalized sampling settings."""
    sampler_kwargs = dict(sampling_spec.sampler_kwargs or {})
    sampler_kwargs.setdefault("progressbar", sampling_spec.verbose)
    sampler_kwargs.setdefault("cores", sampling_spec.nchain)
    return _sample(
        model,
        draws=int(sampling_spec.nit),
        burn=int(sampling_spec.burn),
        tune=int(sampling_spec.tune),
        chains=int(sampling_spec.nchain),
        sample_prior_predictive=True,
        sample_posterior_predictive=["y"],
        nuts_sampler=sampling_spec.nuts_sampler,
        **sampler_kwargs,
    )


def _extend_inferencedata_predictive(
    trace: az.InferenceData,
    *,
    model: pm.Model,
    sample_prior_predictive: bool | int = False,
    sample_posterior_predictive: bool | list[str] = False,
) -> az.InferenceData:
    """Extend an InferenceData object with requested predictive groups."""
    if sample_prior_predictive:
        prior_draws = (
            cast(Any, trace).posterior.sizes["draw"]
            if sample_prior_predictive is True
            else int(sample_prior_predictive)
        )
        with model:
            trace.extend(pm.sample_prior_predictive(prior_draws, model))

    if sample_posterior_predictive:
        posterior_var_names = (
            None if sample_posterior_predictive is True else list(sample_posterior_predictive)
        )
        with model:
            trace.extend(pm.sample_posterior_predictive(trace, model=model, var_names=posterior_var_names))

    return trace


def _sample(
    model: pm.Model,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    burn: int = 0,
    sample_prior_predictive: bool | int = False,
    sample_posterior_predictive: bool | list[str] = False,
    **kwargs: Any,
) -> az.InferenceData:
    """Sample from a built RHIME model and apply burn slicing/predictive requests."""
    sample_kwargs = dict(kwargs)
    sample_kwargs.pop("return_inferencedata", None)
    idata_kwargs = dict(sample_kwargs.pop("idata_kwargs", {}))
    idata_kwargs["log_likelihood"] = True

    with model:
        raw_trace = cast(
            az.InferenceData,
            pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                return_inferencedata=True,
                idata_kwargs=idata_kwargs,
                **sample_kwargs,
            ),
        )

    burned_trace = cast(az.InferenceData, raw_trace.isel(draw=slice(burn, None)))
    return _extend_inferencedata_predictive(
        burned_trace,
        model=model,
        sample_prior_predictive=sample_prior_predictive,
        sample_posterior_predictive=sample_posterior_predictive,
    )


def _materialise_basis_and_flux_for_output(
    prepared: _RhimePreparedInputs,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return materialised arrays for transitional output adapters.

    TODO(#383/#429): postprocessing should consume ``BasisFunctions`` directly
    rather than requiring the runner to materialise a flat basis and flux.
    """
    basis = prepared.basis_functions.operator.basis_matrix
    current_state_dim = prepared.basis_functions.operator.meta.state_dim
    if current_state_dim != "region":
        basis = basis.rename({current_state_dim: "region"})
    region_coord = prepared.inv_inputs.region
    if region_coord.name is None:
        raise ValueError("prepared.inv_inputs.region must be named so it can be used for reindexing.")
    basis = basis.reindex({region_coord.name: region_coord})
    return basis, prepared.basis_functions.flux


def _make_inversion_output(
    *,
    prepared: _RhimePreparedInputs,
    idata: az.InferenceData,
    start_date: str,
    end_date: str,
    species: str,
    domain: str,
) -> InversionOutput:
    """Create an InversionOutput directly from RHIME inputs and InferenceData.

    This is a transitional direct constructor for the modern RHIME path. It is
    deliberately not routed through the fixed-basis/inferpymc legacy adapter.
    This should be refactored when issue #401 defines the modern
    ``InversionOutput`` contract.
    """
    inv_inputs = prepared.inv_inputs
    basis, flux = _materialise_basis_and_flux_for_output(prepared)
    nmeasure = np.arange(inv_inputs.sizes["nmeasure"])
    site_names = (
        inv_inputs["site_names"] if "site_names" in inv_inputs else xr.DataArray(prepared.sites, dims="nsite")
    )

    obs_prior_factor = inv_inputs["mf_prior_factor"] if "mf_prior_factor" in inv_inputs else None
    obs_prior_upper_level_factor = (
        inv_inputs["mf_prior_upper_level_factor"] if "mf_prior_upper_level_factor" in inv_inputs else None
    )

    def nmeasure_array(name: str, source: xr.DataArray) -> xr.DataArray:
        """Create a clean nmeasure DataArray without inherited MultiIndex coords."""
        result = xr.DataArray(
            source.values,
            dims=["nmeasure"],
            coords={"nmeasure": nmeasure},
            name=name,
        )
        result.attrs = source.attrs
        return result

    return InversionOutput(
        obs=nmeasure_array("Yobs", inv_inputs["mf"]),
        obs_err=nmeasure_array("Yerror", inv_inputs["mf_error"]),
        obs_repeatability=nmeasure_array("Yerror_repeatability", inv_inputs["mf_repeatability"]),
        obs_variability=nmeasure_array("Yerror_variability", inv_inputs["mf_variability"]),
        obs_prior_factor=(
            nmeasure_array("Yobs_prior_factor", obs_prior_factor) if obs_prior_factor is not None else None
        ),
        obs_prior_upper_level_factor=(
            nmeasure_array("Yobs_prior_upper_level_factor", obs_prior_upper_level_factor)
            if obs_prior_upper_level_factor is not None
            else None
        ),
        site_indicators=nmeasure_array("site_indicator", inv_inputs["site_indicator"]),
        flux=flux,
        basis=basis,
        trace=idata,
        site_names=site_names,
        times=nmeasure_array("times", inv_inputs["time"]),
        start_date=start_date,
        end_date=end_date,
        species=species,
        domain=domain,
    )


def _write_standard_outputs(
    *,
    result: RhimeResult,
    prepared: _RhimePreparedInputs,
    country_file: str | None,
) -> None:
    """Create and optionally save standard RHIME outputs."""
    output_spec = result.output_spec
    if output_spec.output_format == "none":
        return

    inv_out = _make_inversion_output(
        prepared=prepared,
        idata=result.idata,
        start_date=result.run_spec.start_date,
        end_date=result.run_spec.end_date,
        species=result.model_spec.species,
        domain=result.model_spec.domain,
    )
    result.inv_out = inv_out
    result.outputs["inversion_output"] = inv_out

    trace_path = _resolve_output_path(
        output_spec.save_trace,
        output_spec.output_path,
        f"{output_spec.output_name}{result.run_spec.start_date}_trace.nc",
    )
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        _save_inferencedata(result.idata, trace_path)
        result.output_metadata["trace_path"] = str(trace_path)

    inv_out_path = _resolve_output_path(
        output_spec.save_inversion_output,
        output_spec.output_path,
        f"{output_spec.output_name}{result.run_spec.start_date}_inversion_output.nc",
    )
    if inv_out_path is not None:
        inv_out_path.parent.mkdir(parents=True, exist_ok=True)
        inv_out.save(inv_out_path)
        result.output_metadata["inversion_output_path"] = str(inv_out_path)

    if output_spec.output_format == "basic":
        from openghg_inversions.postprocessing.make_outputs import basic_output

        result.outputs["basic"] = basic_output(inv_out, country_file=country_file)
    elif output_spec.output_format == "paris":
        from openghg_inversions.postprocessing.make_paris_outputs import make_paris_outputs

        obs_avg_period = prepared.averaging_period[0] or "0h"
        kwargs = output_spec.paris_postprocessing_kwargs or {}
        flux_outs, conc_outs = make_paris_outputs(
            inv_out,
            country_file=country_file,
            domain=result.model_spec.domain,
            obs_avg_period=obs_avg_period,
            **kwargs,
        )
        result.outputs["paris_flux"] = flux_outs
        result.outputs["paris_concentration"] = conc_outs

        if output_spec.output_path is not None:
            Path(output_spec.output_path).mkdir(parents=True, exist_ok=True)
            conc_file = _define_output_filename(
                output_spec.output_path,
                result.model_spec.species,
                result.model_spec.domain,
                output_spec.output_name + "_conc",
                result.run_spec.start_date,
                ext=".nc",
            )
            flux_file = _define_output_filename(
                output_spec.output_path,
                result.model_spec.species,
                result.model_spec.domain,
                output_spec.output_name + "_flux",
                result.run_spec.start_date,
                ext=".nc",
            )
            conc_outs.to_netcdf(
                conc_file, unlimited_dims=["time"], mode="w", encoding=ncdf_encoding(conc_outs)
            )
            flux_outs.to_netcdf(
                flux_file, unlimited_dims=["time"], mode="w", encoding=ncdf_encoding(flux_outs)
            )
            result.output_metadata["paris_concentration_path"] = str(conc_file)
            result.output_metadata["paris_flux_path"] = str(flux_file)


def _make_multisector_flux_diagnostics(
    *,
    idata: az.InferenceData,
    prepared: _RhimePreparedInputs,
    model_spec: RhimeModelSpec,
) -> xr.Dataset:
    """Create provisional sector-aware posterior flux diagnostics.

    TODO(#398/#429): replace this runner-local special case with
    sector-aware postprocessing once the modern output layer supports multiple
    sectors and ``BasisFunctions`` reconstruction directly.

    Args:
        idata: InferenceData returned by RHIME sampling.
        prepared: Prepared RHIME input object containing retained basis functions.
        model_spec: Model spec containing sector names and variable suffixes.

    Returns:
        Dataset containing posterior mean scaling factors, sector posterior flux
        means, and total posterior flux mean.
    """
    basis, flux = _materialise_basis_and_flux_for_output(prepared)
    posterior_flux = []
    posterior_scaling = []

    for sector in model_spec.sectors:
        x_name = f"x_{sector.variable_suffix}"
        x_mean = cast(Any, idata).posterior[x_name].mean(("chain", "draw"))
        scale_grid = sparse_xr_dot(basis, x_mean)
        sector_flux = flux.sel(source=sector.flux_source) if "source" in flux.dims else flux
        posterior_scaling.append(scale_grid.expand_dims(sector=[sector.name]))
        posterior_flux.append((scale_grid * sector_flux).expand_dims(sector=[sector.name]))

    scaling = xr.concat(posterior_scaling, dim="sector").rename("posterior_scaling_mean")
    flux_by_sector = xr.concat(posterior_flux, dim="sector").rename("posterior_flux_mean")
    total_flux = flux_by_sector.sum("sector").rename("posterior_flux_total_mean")
    return xr.merge([scaling, flux_by_sector, total_flux])


def _write_multisector_outputs(
    *,
    result: RhimeResult,
    prepared: _RhimePreparedInputs,
) -> None:
    """Create and optionally save transitional multi-sector RHIME outputs."""
    diagnostics = _make_multisector_flux_diagnostics(
        idata=result.idata,
        prepared=prepared,
        model_spec=result.model_spec,
    )
    result.outputs["sector_flux_diagnostics"] = diagnostics

    output_spec = result.output_spec
    if output_spec.output_format == "paris":
        result.output_metadata["paris_note"] = (
            "Multi-sector PARIS schema support is not implemented in issue #398; "
            "sector-aware modern diagnostics were generated instead."
        )
    if output_spec.output_path is not None and output_spec.output_format != "none":
        Path(output_spec.output_path).mkdir(parents=True, exist_ok=True)
        diagnostics_path = (
            Path(output_spec.output_path)
            / f"{output_spec.output_name}{result.run_spec.start_date}_sector_flux_diagnostics.nc"
        )
        diagnostics.to_netcdf(diagnostics_path, mode="w", encoding=ncdf_encoding(diagnostics))
        result.output_metadata["sector_flux_diagnostics_path"] = str(diagnostics_path)


def _tuple_from_optional_sequence(value: Any) -> tuple[str | None, ...]:
    """Convert optional scalar or sequence values into tuple metadata."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return tuple(cast(str | None, item) for item in value)
    return (str(value),)


def _make_output_spec(
    *,
    output_format: str,
    output_path: str | None,
    output_name: str,
    save_trace: str | Path | bool,
    save_inversion_output: str | Path | bool,
    country_file: str | None,
    paris_postprocessing_kwargs: dict[str, Any] | None,
    multisector: bool,
) -> RhimeOutputSpec:
    """Create validated output settings from normalized RHIME parameters."""
    _validate_output_format(output_format)
    _validate_output_path_settings(
        output_format=output_format,
        output_path=output_path,
        save_trace=save_trace,
        save_inversion_output=save_inversion_output,
        multisector=multisector,
    )
    return RhimeOutputSpec(
        output_format=cast(OutputFormat, output_format),
        output_path=output_path,
        output_name=output_name,
        save_trace=save_trace,
        save_inversion_output=save_inversion_output,
        country_file=country_file,
        paris_postprocessing_kwargs=paris_postprocessing_kwargs,
    )


def _make_rhime_runner_setup(
    *,
    params: Mapping[str, Any],
    multisector: bool,
) -> _RhimeRunnerSetup:
    """Normalize raw RHIME parameters into specs and preparation arguments."""
    normalized = rhime_params.normalise_rhime_params(params)
    rhime_params.validate_required_params(normalized)
    rhime_params.validate_supported_params(
        normalized,
        data_params=set(inspect.signature(_prepare_rhime_inputs).parameters),
    )

    remaining = dict(normalized)
    flux_sources = resolve_flux_sources(flux_sources=remaining.pop("flux_sources", None))
    sector_sources = rhime_params.normalise_sector_sources(remaining.pop("sector_sources", None))
    if not multisector and sector_sources is not None:
        raise ValueError("`sector_sources` is only supported by `run_rhime_multisector`.")
    data_flux_sources = (
        list(dict.fromkeys(sector_sources.values())) if sector_sources is not None else flux_sources
    )
    if sector_sources is not None and set(data_flux_sources) != set(flux_sources):
        raise ValueError(
            "`sector_sources` values must match `flux_sources` so RHIME can retrieve the "
            "OpenGHG data used by each sector."
        )
    if multisector and len(data_flux_sources) < 2:
        raise ValueError("`run_rhime_multisector` requires at least two flux sources.")
    if not multisector and len(flux_sources) != 1:
        raise ValueError("`run_rhime` requires exactly one flux source.")

    species = remaining.pop("species")
    sites = rhime_params.as_list(remaining.pop("sites")) or []
    domain = remaining.pop("domain")
    averaging_period = remaining.pop("averaging_period")
    start_date = remaining.pop("start_date")
    end_date = remaining.pop("end_date")
    output_path = remaining.pop("output_path", None)
    output_name = remaining.pop("output_name")

    x_prior = rhime_params.normalise_optional_mapping(remaining.pop("x_prior", None))
    bc_prior = rhime_params.normalise_optional_mapping(remaining.pop("bc_prior", None))
    sigma_prior = rhime_params.normalise_optional_mapping(remaining.pop("sigma_prior", None))
    offset_prior = rhime_params.normalise_optional_mapping(remaining.pop("offset_prior", None))
    sector_priors = rhime_params.normalise_sector_priors(remaining.pop("sector_priors", None))
    offset_args = rhime_params.normalise_optional_mapping(remaining.get("offset_args"))

    use_bc = remaining.get("use_bc", True)
    sigma_per_site = remaining.get("sigma_per_site", True)
    add_offset = remaining.get("add_offset", False)
    pollution_events_from_obs = remaining.pop("pollution_events_from_obs", False)
    no_model_error = remaining.pop("no_model_error", False)
    power = remaining.pop("power", 1.99)

    sampling_spec = RhimeSamplingSpec(
        nit=remaining.pop("nit", 1000),
        burn=remaining.pop("burn", 0),
        tune=remaining.pop("tune", 1000),
        nchain=remaining.pop("nchain", 4),
        nuts_sampler=remaining.pop("nuts_sampler", "pymc"),
        verbose=remaining.pop("verbose", False),
        sampler_kwargs=rhime_params.normalise_optional_mapping(remaining.pop("sampler_kwargs", None)),
    )
    output_spec = _make_output_spec(
        output_format=remaining.pop("output_format", "inv_out"),
        output_path=output_path,
        output_name=output_name,
        save_trace=remaining.pop("save_trace", False),
        save_inversion_output=remaining.pop("save_inversion_output", True),
        country_file=remaining.get("country_file"),
        paris_postprocessing_kwargs=rhime_params.normalise_optional_mapping(
            remaining.pop("paris_postprocessing_kwargs", None)
        ),
        multisector=multisector,
    )
    model_spec = _make_model_spec(
        species=species,
        domain=domain,
        flux_sources=flux_sources,
        x_prior=x_prior,
        sector_priors=sector_priors,
        sector_sources=sector_sources,
        bc_prior=bc_prior,
        sigma_prior=sigma_prior,
        offset_prior=offset_prior,
        use_bc=use_bc,
        sigma_per_site=sigma_per_site,
        add_offset=add_offset,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        power=power,
        offset_args=offset_args,
    )
    run_spec = RhimeRunSpec(
        start_date=start_date,
        end_date=end_date,
        sites=tuple(sites),
        averaging_period=_tuple_from_optional_sequence(averaging_period),
        model=model_spec,
        output=output_spec,
        sampling=sampling_spec,
        split_by_sectors=multisector,
    )

    data_args, _ = split_function_inputs(
        {
            **remaining,
            "species": species,
            "sites": sites,
            "domain": domain,
            "averaging_period": averaging_period,
            "start_date": start_date,
            "end_date": end_date,
            "output_name": output_name,
            "flux_sources": data_flux_sources,
            "split_by_sectors": multisector,
        },
        _prepare_rhime_inputs,
    )
    return _RhimeRunnerSetup(run_spec=run_spec, data_args=data_args)


def _run_spec_with_prepared_inputs(
    run_spec: RhimeRunSpec,
    prepared: _RhimePreparedInputs,
) -> RhimeRunSpec:
    """Update run metadata with retained sites from prepared RHIME inputs."""
    return replace(
        run_spec,
        sites=tuple(prepared.sites),
        averaging_period=tuple(prepared.averaging_period),
    )


def _run_common(
    *,
    multisector: bool,
    params: dict[str, Any],
) -> RhimeResult:
    """Run the shared RHIME pipeline after public wrapper/config normalization."""
    setup = _make_rhime_runner_setup(params=params, multisector=multisector)
    prepared = _prepare_rhime_inputs(**setup.data_args)
    run_spec = _run_spec_with_prepared_inputs(setup.run_spec, prepared)

    start_build = time.time()
    if multisector:
        model = build_rhime_multisector_model_from_spec(prepared.inv_inputs, run_spec.model)
    else:
        model = build_rhime_model_from_spec(prepared.inv_inputs, run_spec.model)

    idata = _sample_model(model, run_spec.sampling)
    result = RhimeResult(
        run_spec=run_spec,
        model_spec=run_spec.model,
        output_spec=run_spec.output,
        inv_inputs=prepared.inv_inputs,
        idata=idata,
        sampling_spec=run_spec.sampling,
        model=model,
        basis_functions=prepared.basis_functions,
        output_metadata={"build_and_sample_seconds": time.time() - start_build},
    )

    if multisector:
        _write_multisector_outputs(result=result, prepared=prepared)
    else:
        _write_standard_outputs(
            result=result,
            prepared=prepared,
            country_file=run_spec.output.country_file,
        )

    return result


def run_rhime(
    *,
    config_file: str | Path | None = None,
    **kwargs: Any,
) -> RhimeResult:
    """Run a standard single-sector RHIME inversion.

    Args:
        config_file: Optional INI configuration file. Values in ``kwargs``
            override values read from this file.
        **kwargs: RHIME run parameters using snake-case names, such as
            ``output_path``, ``output_name``, ``flux_sources``, and
            ``x_prior``. ``species`` names the primary gas or tracer used for
            object-store lookup and output naming. ``flux_sources`` contains
            OpenGHG flux ``source`` values. Legacy ``emissions_name`` is
            accepted only as a compatibility alias when ``flux_sources`` is
            absent.

    Returns:
        Modern RHIME result containing canonical inputs, InferenceData, specs,
        output metadata, and generated outputs.

    Raises:
        ValueError: If required parameters are missing, unsupported parameters
            are supplied, or the flux-source count is invalid.
    """
    params = params_from_config(config_file, extra_kwargs=kwargs) if config_file is not None else dict(kwargs)
    return _run_common(multisector=False, params=params)


def run_rhime_multisector(
    *,
    config_file: str | Path | None = None,
    **kwargs: Any,
) -> RhimeResult:
    """Run a shared-basis multi-sector RHIME inversion.

    Args:
        config_file: Optional INI configuration file. Values in ``kwargs``
            override values read from this file.
        **kwargs: RHIME run parameters using snake-case names. Multi-sector
            runs require at least two ``flux_sources`` and may include
            ``sector_priors`` keyed by sector name. When model sector labels
            differ from OpenGHG source values, pass ``sector_sources`` as a
            mapping from sector name to one value in ``flux_sources``. Legacy
            ``emissions_name`` is accepted only as a compatibility alias when
            ``flux_sources`` is absent.

    Returns:
        Modern RHIME result containing canonical inputs, InferenceData, specs,
        output metadata, and sector diagnostics.

    Raises:
        ValueError: If required parameters are missing, unsupported parameters
            are supplied, or fewer than two flux sources are provided.
    """
    params = params_from_config(config_file, extra_kwargs=kwargs) if config_file is not None else dict(kwargs)
    return _run_common(multisector=True, params=params)
