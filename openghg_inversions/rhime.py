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
from typing import Any, Literal, cast
import time
import warnings

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from openghg.util import split_function_inputs  # pyright: ignore[reportPrivateImportUsage]

from openghg_inversions.array_ops import sparse_xr_dot
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.config import config
from openghg_inversions.inversion_data import RhimePreparedInputs, prepare_rhime_inputs
from openghg_inversions.models import (
    DEFAULT_X_PRIOR,
    build_rhime_model,
    build_rhime_multisector_model,
    safe_pymc_name,
)
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.utils import ncdf_encoding

OutputFormat = Literal["none", "inv_out", "basic", "paris"]
MinErrorConfig = Literal["percentile", "residual"] | dict[str, float] | None | int | float


@dataclass(frozen=True)
class SectorSpec:
    """Configuration for one separately optimised flux sector.

    Args:
        name: User-facing sector name.
        flux_source: OpenGHG flux ``source`` used to retrieve this sector.
        x_prior: Prior specification for this sector's flux scaling factors.
        variable_suffix: PyMC-safe suffix used in model variable names.
    """

    name: str
    flux_source: str
    x_prior: dict[str, Any]
    variable_suffix: str


@dataclass(frozen=True)
class RhimeModelSpec:
    """Model options used to build a RHIME PyMC model.

    Args:
        species: Primary gas or tracer name used for object-store lookup and
            output naming.
        domain: Model domain name.
        sectors: Flux sectors included in the model. Each sector is optimized
            separately and is normally backed by one OpenGHG flux ``source``.
        use_bc: Whether boundary-condition scaling is included.
        sigma_per_site: Whether model-error terms vary by site.
        add_offset: Whether model-data offsets are included.
        pollution_events_from_obs: Whether model error scales with observed
            enhancements instead of modelled enhancements.
        no_model_error: Whether explicit model-error terms are disabled.
        power: Exponent or prior specification used in likelihood error scaling.
        bc_prior: Prior specification for boundary-condition scaling factors.
        sigma_prior: Prior specification for model-error terms.
        offset_prior: Prior specification for optional offsets.
        offset_args: Extra keyword arguments forwarded to the offset component.
    """

    species: str
    domain: str
    sectors: tuple[SectorSpec, ...]
    use_bc: bool = True
    sigma_per_site: bool = True
    add_offset: bool = False
    pollution_events_from_obs: bool = False
    no_model_error: bool = False
    power: dict[str, Any] | float = 1.99
    bc_prior: dict[str, Any] | None = None
    sigma_prior: dict[str, Any] | None = None
    offset_prior: dict[str, Any] | None = None
    offset_args: dict[str, Any] | None = None


@dataclass(frozen=True)
class RhimeOutputSpec:
    """Output settings for a RHIME run.

    Args:
        output_format: Output mode. ``"inv_out"`` saves/returns the modern
            inversion output, ``"basic"`` and ``"paris"`` additionally create
            derived outputs, and ``"none"`` skips output products.
        output_path: Directory for saved outputs.
        output_name: Base output name.
        save_trace: Trace save setting. If true, save to ``output_path`` using
            the default trace file name; if a path, save there.
        save_inversion_output: Inversion-output save setting. Defaults to true
            for CLI-friendly behaviour.
        country_file: Optional country mask file used by derived outputs.
        paris_postprocessing_kwargs: Extra keyword arguments for PARIS output
            creation.
    """

    output_format: OutputFormat = "inv_out"
    output_path: str | None = None
    output_name: str = "rhime"
    save_trace: str | Path | bool = False
    save_inversion_output: str | Path | bool = True
    country_file: str | None = None
    paris_postprocessing_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class RhimeSamplingSpec:
    """Sampling settings for a RHIME run.

    Args:
        nit: Number of post-tuning draws requested from PyMC.
        burn: Number of draws to discard from each chain after sampling.
        tune: Number of PyMC tuning draws.
        nchain: Number of MCMC chains.
        nuts_sampler: PyMC NUTS backend name.
        verbose: Whether PyMC progress output should be shown.
        sampler_kwargs: Extra keyword arguments forwarded to ``pm.sample``.
    """

    nit: int = 1000
    burn: int = 0
    tune: int = 1000
    nchain: int = 4
    nuts_sampler: str = "pymc"
    verbose: bool = False
    sampler_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class RhimeRunSpec:
    """Top-level run metadata for a RHIME run.

    Args:
        start_date: Inclusive inversion start date.
        end_date: Exclusive inversion end date.
        sites: Sites included after data preparation and filtering.
        averaging_period: Observation averaging period per retained site.
        model: Mathematical model specification.
        output: Output settings.
        sampling: Sampling settings.
        split_by_sectors: Whether flux data were prepared in sector-resolved
            mode.
    """

    start_date: str
    end_date: str
    sites: tuple[str, ...]
    averaging_period: tuple[str | None, ...]
    model: RhimeModelSpec
    output: RhimeOutputSpec
    sampling: RhimeSamplingSpec = field(default_factory=RhimeSamplingSpec)
    split_by_sectors: bool = False


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
        sampling_spec: Sampling settings used by the run.
        inv_inputs: Canonical xarray inversion inputs consumed by the model.
        idata: ArviZ ``InferenceData`` returned by sampling.
        output_metadata: Paths and notes for generated outputs.
        outputs: In-memory derived outputs keyed by output kind.
        model: Built PyMC model.
        inv_out: Modern inversion output object when created.
        basis_functions: Retained flux basis functions from preparation.
    """

    run_spec: RhimeRunSpec
    model_spec: RhimeModelSpec
    output_spec: RhimeOutputSpec
    inv_inputs: xr.Dataset
    idata: az.InferenceData
    sampling_spec: RhimeSamplingSpec = field(default_factory=RhimeSamplingSpec)
    output_metadata: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    basis_functions: BasisFunctions | None = None
    model: pm.Model | None = None
    inv_out: InversionOutput | None = None


def _as_list(value: str | Sequence[str] | None) -> list[str] | None:
    """Convert a scalar/list-like value to a list of strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def resolve_flux_sources(
    *,
    flux_sources: str | Sequence[str] | None = None,
    emissions_name: str | Sequence[str] | None = None,
) -> list[str]:
    """Resolve new ``flux_sources`` and legacy ``emissions_name`` arguments.

    Args:
        flux_sources: Preferred RHIME field containing OpenGHG flux
            ``source`` metadata values.
        emissions_name: Legacy compatibility spelling accepted only when
            ``flux_sources`` is absent.

    Returns:
        Resolved flux source names.

    Raises:
        ValueError: If no usable flux source is supplied.
    """
    resolved = _as_list(flux_sources)
    if resolved is None:
        resolved = _as_list(emissions_name)
    if not resolved or any(source in {"", "None", "none"} for source in resolved):
        raise ValueError("At least one flux source must be supplied via `flux_sources`.")
    return resolved


def params_from_config(
    config_file: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: str | None = None,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load RHIME run parameters from an INI config file.

    Args:
        config_file: Path to an INI configuration file.
        start_date: Optional command-line start-date override.
        end_date: Optional command-line end-date override.
        output_path: Optional command-line output-path override.
        extra_kwargs: Optional keyword overrides, normally parsed from CLI JSON.

    Returns:
        Normalized RHIME run parameters using snake-case public names.

    Raises:
        ValueError: If deprecated unsupported parameters are present or a
            structured RHIME option has an invalid type.
    """
    params = dict(config.all_param(str(config_file), exclude_not_found=True, allow_new=True))
    if start_date is not None:
        params["start_date"] = start_date
    if end_date is not None:
        params["end_date"] = end_date
    if output_path is not None:
        params["output_path"] = output_path
    if extra_kwargs:
        params.update(extra_kwargs)
    normalized = _normalise_params(params)
    _validate_rhime_param_types(normalized)
    return normalized


def _normalise_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy config spellings to modern snake-case names."""
    normalized = dict(params)
    aliases = {
        "outputpath": "output_path",
        "outputname": "output_name",
        "xprior": "x_prior",
        "bcprior": "bc_prior",
        "sigprior": "sigma_prior",
        "offsetprior": "offset_prior",
        "emissions_name": "flux_sources",
    }
    for old, new in aliases.items():
        if old not in normalized:
            continue
        if new in normalized:
            warnings.warn(
                f"Ignoring deprecated RHIME parameter {old!r} because {new!r} was also supplied.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"RHIME parameter {old!r} is deprecated; use {new!r} instead.",
                UserWarning,
                stacklevel=2,
            )
            normalized[new] = normalized[old]
        del normalized[old]

    if "calculate_min_error" in normalized:
        raise ValueError("`calculate_min_error` is not supported by RHIME runners; use `min_error`.")
    if "reparameterise_log_normal" in normalized:
        raise ValueError(
            "`reparameterise_log_normal` is not supported by RHIME runners; "
            "set `reparameterise` in the relevant prior dictionary if needed."
        )
    if "mcmc_type" in normalized:
        raise ValueError("`mcmc_type` is not supported by RHIME runners; use `nuts_sampler` if needed.")

    return normalized


def _invalid_config_type_message(name: str, expected: str, value: Any) -> str:
    """Build an actionable RHIME config type error."""
    return (
        f"Invalid RHIME config value for `{name}`: expected {expected}, "
        f"but got {type(value).__name__}. Check braces/quotes in the .ini file."
    )


def _validate_mapping_option(params: Mapping[str, Any], name: str) -> None:
    """Raise if a RHIME option is present but is not a mapping or None."""
    if name not in params or params[name] is None:
        return
    if not isinstance(params[name], Mapping):
        raise ValueError(_invalid_config_type_message(name, "a mapping/dict", params[name]))


def _validate_rhime_param_types(params: Mapping[str, Any]) -> None:
    """Validate structured RHIME parameter types before preparation begins."""
    for prior_name in ("x_prior", "bc_prior", "sigma_prior", "offset_prior"):
        _validate_mapping_option(params, prior_name)

    if "sector_priors" in params and params["sector_priors"] is not None:
        sector_priors = params["sector_priors"]
        if not isinstance(sector_priors, Mapping):
            raise ValueError(_invalid_config_type_message("sector_priors", "a mapping/dict", sector_priors))
        for sector, prior in sector_priors.items():
            if not isinstance(prior, Mapping):
                raise ValueError(
                    _invalid_config_type_message(
                        f"sector_priors[{sector!r}]",
                        "a mapping/dict",
                        prior,
                    )
                )

    for mapping_name in ("sampler_kwargs", "paris_postprocessing_kwargs", "offset_args"):
        _validate_mapping_option(params, mapping_name)


def _normalise_optional_mapping(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Copy an optional mapping so specs do not retain caller-owned dicts."""
    return None if value is None else dict(value)


def _normalise_sector_priors(
    sector_priors: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    """Copy optional sector-prior mappings with string sector keys."""
    if sector_priors is None:
        return None
    return {str(sector): dict(prior) for sector, prior in sector_priors.items()}


def _required_run_params() -> set[str]:
    return {
        "species",
        "sites",
        "averaging_period",
        "domain",
        "start_date",
        "end_date",
        "output_name",
    }


def _is_missing_required_value(value: Any) -> bool:
    """Return true when a required RHIME parameter has no usable value."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) == 0:
        return True
    return False


def _validate_required_params(params: Mapping[str, Any]) -> None:
    """Raise if normalized run parameters are missing required values."""
    missing = [
        name
        for name in sorted(_required_run_params())
        if name not in params or _is_missing_required_value(params[name])
    ]
    if missing:
        raise ValueError(f"Required RHIME parameter(s) missing: {missing!r}")


def _validate_supported_params(params: Mapping[str, Any]) -> None:
    """Raise if normalized run parameters contain unsupported keys."""
    data_params = set(inspect.signature(prepare_rhime_inputs).parameters)
    runner_params = {
        "x_prior",
        "bc_prior",
        "sigma_prior",
        "offset_prior",
        "sector_priors",
        "pollution_events_from_obs",
        "no_model_error",
        "power",
        "nit",
        "burn",
        "tune",
        "nchain",
        "nuts_sampler",
        "verbose",
        "sampler_kwargs",
        "output_format",
        "output_path",
        "save_trace",
        "save_inversion_output",
        "paris_postprocessing_kwargs",
        "offset_args",
        "country_file",
        "add_offset",
        "sigma_per_site",
    }
    required = _required_run_params()
    supported = data_params | runner_params | required
    unsupported = sorted(set(params) - supported)
    if unsupported:
        raise ValueError(f"Unsupported RHIME parameter(s): {unsupported!r}")


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
    for source in flux_sources:
        suffix = safe_pymc_name(source)
        if suffix in used_suffixes:
            raise ValueError(
                "Flux source names must be unique after PyMC name sanitisation; "
                f"duplicate sanitized name {suffix!r}."
            )
        used_suffixes.add(suffix)
        prior = (
            sector_priors[source]
            if sector_priors is not None and source in sector_priors
            else default_x_prior
        )
        sectors.append(
            SectorSpec(
                name=source,
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
    *,
    nit: int,
    burn: int,
    tune: int,
    nchain: int,
    nuts_sampler: str,
    verbose: bool,
    sampler_kwargs: dict | None,
) -> az.InferenceData:
    """Sample a built RHIME model and return InferenceData."""
    sampler_kwargs = dict(sampler_kwargs or {})
    sampler_kwargs.setdefault("progressbar", verbose)
    sampler_kwargs.setdefault("cores", nchain)
    return _sample(
        model,
        draws=int(nit),
        burn=int(burn),
        tune=int(tune),
        chains=int(nchain),
        sample_prior_predictive=True,
        sample_posterior_predictive=["y"],
        nuts_sampler=nuts_sampler,
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


def _basis_matrix_for_output(
    basis_functions: BasisFunctions,
    *,
    state_dim: str = "region",
    state_coord: xr.DataArray | None = None,
) -> xr.DataArray:
    """Materialise a basis matrix at the RHIME output boundary."""
    basis = basis_functions.operator.basis_matrix
    current_state_dim = basis_functions.operator.meta.state_dim
    if state_dim != current_state_dim:
        basis = basis.rename({current_state_dim: state_dim})
    if state_coord is not None:
        if state_coord.name is None:
            raise ValueError("state_coord must be named so it can be used for reindexing.")
        basis = basis.reindex({state_coord.name: state_coord})
    return basis


def _materialise_basis_and_flux_for_output(
    prepared: RhimePreparedInputs,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return materialised basis/flux arrays for current output adapters."""
    basis = _basis_matrix_for_output(
        prepared.basis_functions,
        state_dim="region",
        state_coord=prepared.inv_inputs.region,
    )
    return basis, prepared.basis_functions.flux


def _make_inversion_output(
    *,
    prepared: RhimePreparedInputs,
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
    prepared: RhimePreparedInputs,
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


def make_multisector_flux_diagnostics(
    *,
    idata: az.InferenceData,
    prepared: RhimePreparedInputs,
    model_spec: RhimeModelSpec,
) -> xr.Dataset:
    """Create sector-aware posterior flux diagnostics for shared-basis RHIME.

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
    prepared: RhimePreparedInputs,
) -> None:
    """Create and optionally save shared-basis multi-sector RHIME outputs."""
    diagnostics = make_multisector_flux_diagnostics(
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


def _make_sampling_spec(
    *,
    nit: Any,
    burn: Any,
    tune: Any,
    nchain: Any,
    nuts_sampler: str,
    verbose: bool,
    sampler_kwargs: dict[str, Any] | None,
) -> RhimeSamplingSpec:
    """Create sampling settings from normalized RHIME parameters."""
    return RhimeSamplingSpec(
        nit=int(nit),
        burn=int(burn),
        tune=int(tune),
        nchain=int(nchain),
        nuts_sampler=nuts_sampler,
        verbose=verbose,
        sampler_kwargs=sampler_kwargs,
    )


def _make_rhime_runner_setup(
    *,
    params: Mapping[str, Any],
    multisector: bool,
) -> _RhimeRunnerSetup:
    """Normalize raw RHIME parameters into specs and preparation arguments."""
    normalized = _normalise_params(params)
    _validate_rhime_param_types(normalized)
    _validate_required_params(normalized)
    _validate_supported_params(normalized)

    remaining = dict(normalized)
    flux_sources = resolve_flux_sources(flux_sources=remaining.pop("flux_sources", None))
    if multisector and len(flux_sources) < 2:
        raise ValueError("`run_rhime_multisector` requires at least two flux sources.")
    if not multisector and len(flux_sources) != 1:
        raise ValueError("`run_rhime` requires exactly one flux source.")

    species = remaining.pop("species")
    sites = _as_list(remaining.pop("sites")) or []
    domain = remaining.pop("domain")
    averaging_period = remaining.pop("averaging_period")
    start_date = remaining.pop("start_date")
    end_date = remaining.pop("end_date")
    output_path = remaining.pop("output_path", None)
    output_name = remaining.pop("output_name")

    x_prior = _normalise_optional_mapping(remaining.pop("x_prior", None))
    bc_prior = _normalise_optional_mapping(remaining.pop("bc_prior", None))
    sigma_prior = _normalise_optional_mapping(remaining.pop("sigma_prior", None))
    offset_prior = _normalise_optional_mapping(remaining.pop("offset_prior", None))
    sector_priors = _normalise_sector_priors(remaining.pop("sector_priors", None))
    offset_args = _normalise_optional_mapping(remaining.get("offset_args"))

    use_bc = remaining.get("use_bc", True)
    sigma_per_site = remaining.get("sigma_per_site", True)
    add_offset = remaining.get("add_offset", False)
    pollution_events_from_obs = remaining.pop("pollution_events_from_obs", False)
    no_model_error = remaining.pop("no_model_error", False)
    power = remaining.pop("power", 1.99)

    sampling_spec = _make_sampling_spec(
        nit=remaining.pop("nit", 1000),
        burn=remaining.pop("burn", 0),
        tune=remaining.pop("tune", 1000),
        nchain=remaining.pop("nchain", 4),
        nuts_sampler=remaining.pop("nuts_sampler", "pymc"),
        verbose=remaining.pop("verbose", False),
        sampler_kwargs=_normalise_optional_mapping(remaining.pop("sampler_kwargs", None)),
    )
    output_spec = _make_output_spec(
        output_format=remaining.pop("output_format", "inv_out"),
        output_path=output_path,
        output_name=output_name,
        save_trace=remaining.pop("save_trace", False),
        save_inversion_output=remaining.pop("save_inversion_output", True),
        country_file=remaining.get("country_file"),
        paris_postprocessing_kwargs=_normalise_optional_mapping(
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
            "flux_sources": flux_sources,
            "split_by_sectors": multisector,
        },
        prepare_rhime_inputs,
    )
    return _RhimeRunnerSetup(run_spec=run_spec, data_args=data_args)


def _run_spec_with_prepared_inputs(
    run_spec: RhimeRunSpec,
    prepared: RhimePreparedInputs,
) -> RhimeRunSpec:
    """Update run metadata with retained sites from prepared RHIME inputs."""
    return replace(
        run_spec,
        sites=tuple(prepared.sites),
        averaging_period=tuple(prepared.averaging_period),
    )


def _model_kwargs_from_spec(
    model_spec: RhimeModelSpec,
    *,
    multisector: bool,
) -> dict[str, Any]:
    """Build PyMC model-builder kwargs from a normalized model spec."""
    kwargs: dict[str, Any] = {
        "bc_prior": model_spec.bc_prior,
        "sigma_prior": model_spec.sigma_prior,
        "sigma_per_site": model_spec.sigma_per_site,
        "offset_prior": model_spec.offset_prior,
        "add_offset": model_spec.add_offset,
        "use_bc": model_spec.use_bc,
        "pollution_events_from_obs": model_spec.pollution_events_from_obs,
        "no_model_error": model_spec.no_model_error,
        "offset_args": model_spec.offset_args,
        "power": model_spec.power,
    }
    if multisector:
        kwargs["sectors"] = [sector.flux_source for sector in model_spec.sectors]
        kwargs["sector_priors"] = {sector.flux_source: dict(sector.x_prior) for sector in model_spec.sectors}
    else:
        kwargs["x_prior"] = dict(model_spec.sectors[0].x_prior)
    return kwargs


def _sample_model_from_spec(model: pm.Model, sampling_spec: RhimeSamplingSpec) -> az.InferenceData:
    """Sample a RHIME PyMC model using normalized sampling settings."""
    return _sample_model(
        model,
        nit=sampling_spec.nit,
        burn=sampling_spec.burn,
        tune=sampling_spec.tune,
        nchain=sampling_spec.nchain,
        nuts_sampler=sampling_spec.nuts_sampler,
        verbose=sampling_spec.verbose,
        sampler_kwargs=sampling_spec.sampler_kwargs,
    )


def _run_common(
    *,
    multisector: bool,
    params: dict[str, Any],
) -> RhimeResult:
    """Run the shared RHIME pipeline after public wrapper/config normalization."""
    setup = _make_rhime_runner_setup(params=params, multisector=multisector)
    prepared = prepare_rhime_inputs(**setup.data_args)
    run_spec = _run_spec_with_prepared_inputs(setup.run_spec, prepared)

    start_build = time.time()
    model_kwargs = _model_kwargs_from_spec(run_spec.model, multisector=multisector)
    if multisector:
        model = build_rhime_multisector_model(prepared.inv_inputs, **model_kwargs)
    else:
        model = build_rhime_model(prepared.inv_inputs, **model_kwargs)

    idata = _sample_model_from_spec(model, run_spec.sampling)
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
            ``sector_priors`` keyed by sector name. In the current shared-basis
            implementation, each sector is backed by one OpenGHG flux
            ``source`` from ``flux_sources``. Legacy ``emissions_name`` is
            accepted only as a compatibility alias when ``flux_sources`` is
            absent.

    Returns:
        Modern RHIME result containing canonical inputs, InferenceData, specs,
        output metadata, and sector diagnostics.

    Raises:
        ValueError: If required parameters are missing, unsupported parameters
            are supplied, or fewer than two flux sources are provided.
    """
    params = params_from_config(config_file, extra_kwargs=kwargs) if config_file is not None else dict(kwargs)
    return _run_common(multisector=True, params=params)
