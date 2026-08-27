"""RHIME parameter loading, normalisation, and validation helpers.

This internal package module keeps raw config/API parameter handling out of the
public RHIME runner. It is intentionally limited to INI compatibility, legacy
alias handling, simple scalar coercion, and validation of raw dictionaries.
Future YAML/schema frontends should target the same normalized parameter model
before constructing RHIME specs.
Preparation-option ownership is fixed by
``RHIME_PREPARATION_OPTION_NAMES`` rather than inferred from a callable
signature.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.config import config
from openghg_inversions.model_error import normalise_min_error_options
from openghg_inversions.models._flux import safe_pymc_name
from openghg_inversions.observation_error import AggregationErrorMode
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import (
    DEFAULT_X_PRIOR,
    AdditiveSigmaSettings,
    LikelihoodSettings,
    MismatchModel,
    PollutionEventSettings,
    RhimeModelSpec,
    RhimeRunSpec,
    SectorSpec,
    make_output_spec,
)

_COMMON_LIKELIHOOD_OPTIONS = {
    "sigma_prior",
    "sigma_freq",
    "sigma_per_site",
    "sigma_freq_anchor",
    "no_model_error",
}
_POLLUTION_EVENT_OPTIONS = {"pollution_events_from_obs", "power"}
_ADDITIVE_SIGMA_OPTIONS = {"use_minimum_error_floor"}
_LIKELIHOOD_OPTIONS = (
    _COMMON_LIKELIHOOD_OPTIONS | _POLLUTION_EVENT_OPTIONS | _ADDITIVE_SIGMA_OPTIONS
)

_ALIASES = {
    "outputpath": "output_path",
    "outputname": "output_name",
    "xprior": "x_prior",
    "bcprior": "bc_prior",
    "sigprior": "sigma_prior",
    "offsetprior": "offset_prior",
    "emissions_name": "flux_sources",
}
_OUTPUT_FORMAT_ALIASES = {
    "hbmcmc": "legacy",
    "hbmcmc_postprocessing": "legacy",
}

_INT_OPTIONS = ("draws", "burn", "tune", "chains")
_MAPPING_OPTIONS = (
    "sample_kwargs",
    "posterior_predictive_kwargs",
    "paris_postprocessing_kwargs",
    "offset_args",
    "min_error_options",
    "sector_sources",
)
_PRIOR_OPTIONS = ("x_prior", "bc_prior", "sigma_prior", "offset_prior")

#: This is the authoritative routing schema between raw RHIME options and
# ``prepare_rhime_inputs``.  Keep it explicit: accepted configuration must not
# change merely because an implementation helper gains a parameter.
RHIME_PREPARATION_OPTION_NAMES = frozenset(
    {
        "species",
        "sites",
        "domain",
        "averaging_period",
        "start_date",
        "end_date",
        "output_name",
        "flux_sources",
        "split_by_sectors",
        "bc_store",
        "obs_store",
        "footprint_store",
        "emissions_store",
        "met_model",
        "fp_model",
        "fp_height",
        "fp_species",
        "inlet",
        "instrument",
        "max_level",
        "calibration_scale",
        "obs_data_level",
        "platform",
        "use_tracer",
        "use_bc",
        "fp_basis_case",
        "basis_directory",
        "bc_basis_case",
        "bc_basis_directory",
        "country_directory",
        "bc_input",
        "basis_algorithm",
        "nbasis",
        "filters",
        "fix_basis_outer_regions",
        "averaging_error",
        "bc_freq",
        "reload_merged_data",
        "save_merged_data",
        "merged_data_dir",
        "merged_data_name",
        "basis_output_path",
        "min_error",
        "min_error_options",
        "flux_non_finite_check",
    }
)


def resolve_rhime_options(
    *,
    params: Mapping[str, Any],
    multisector: bool,
) -> RhimeRunnerSetup:
    """Normalize raw options into preparation, model, sampling, and output settings."""
    timing_start = timer_start()
    setup = make_rhime_runner_setup(params=params, multisector=multisector)
    log_timing("rhime.runner_setup", timer_seconds(timing_start), multisector=multisector)
    return setup

# Resolve stage defaults once, before the scientific recipe starts.  Keeping
# this mapping beside the explicit routing schema makes ``data_args`` a
# complete, inspectable preparation contract rather than asking individual
# stages to infer omitted values independently.
RHIME_PREPARATION_DEFAULTS: dict[str, Any] = {
    "split_by_sectors": False,
    "bc_store": "user",
    "obs_store": "user",
    "footprint_store": "user",
    "emissions_store": "user",
    "met_model": None,
    "fp_model": None,
    "fp_height": None,
    "fp_species": None,
    "inlet": None,
    "instrument": None,
    "max_level": None,
    "calibration_scale": None,
    "obs_data_level": None,
    "platform": None,
    "use_tracer": False,
    "use_bc": True,
    "fp_basis_case": None,
    "basis_directory": None,
    "bc_basis_case": "NESW",
    "bc_basis_directory": None,
    "country_directory": None,
    "bc_input": None,
    "basis_algorithm": "weighted",
    "nbasis": 100,
    "filters": None,
    "fix_basis_outer_regions": False,
    "averaging_error": True,
    "bc_freq": None,
    "reload_merged_data": False,
    "save_merged_data": False,
    "merged_data_dir": None,
    "merged_data_name": None,
    "basis_output_path": None,
    "min_error": 0.0,
    "min_error_options": None,
    "flux_non_finite_check": "lazy",
}


@dataclass(frozen=True)
class RhimeRunnerSetup:
    """Normalized RHIME setup derived from config or direct API parameters."""

    run_spec: RhimeRunSpec
    sampler: RhimeSampler
    data_args: dict[str, Any]


def as_list(value: str | Sequence[str] | None) -> list[str] | None:
    """Convert a scalar/list-like value to a list of strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _duplicate_names(values: Sequence[str]) -> list[str]:
    """Return duplicate names once each, preserving their first repeated order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


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
    resolved = as_list(flux_sources)
    if resolved is None:
        resolved = as_list(emissions_name)
    if not resolved or any(source in {"", "None", "none"} for source in resolved):
        raise ValueError("At least one flux source must be supplied via `flux_sources`.")
    duplicates = _duplicate_names(resolved)
    if duplicates:
        raise ValueError(
            f"`flux_sources` must contain unique OpenGHG source values; duplicate source(s): {duplicates!r}."
        )
    return resolved


def params_from_config(
    config_file: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: str | None = None,
    extra_kwargs: Mapping[str, Any] | None = None,
    normalise: bool = True,
) -> dict[str, Any]:
    """Load RHIME run parameters from an INI config file.

    Args:
        config_file: Path to an INI configuration file.
        start_date: Optional command-line start-date override.
        end_date: Optional command-line end-date override.
        output_path: Optional command-line output-path override.
        extra_kwargs: Optional keyword overrides, normally parsed from CLI JSON.
        normalise: Whether to normalize and validate the merged parameters.
            Complete runners defer this to their public resolution stage.

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
    return normalise_rhime_params(params) if normalise else params


def normalise_rhime_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize aliases, coerce simple scalars, and validate structured values."""
    normalized = normalise_param_aliases(params)
    normalise_output_format_alias(normalized)
    coerce_simple_param_types(normalized)
    validate_rhime_param_types(normalized)
    return normalized


def normalise_param_aliases(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy config spellings to modern snake-case names."""
    normalized = dict(params)
    for old, new in _ALIASES.items():
        if old not in normalized:
            continue
        if new in normalized:
            warnings.warn(
                f"Ignoring deprecated RHIME parameter {old!r} because {new!r} was also supplied.",
                UserWarning,
                stacklevel=3,
            )
        else:
            warnings.warn(
                f"RHIME parameter {old!r} is deprecated; use {new!r} instead.",
                UserWarning,
                stacklevel=3,
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


def normalise_output_format_alias(params: dict[str, Any]) -> None:
    """Normalize deprecated HBMCMC output format names in-place."""
    output_format = params.get("output_format")
    if output_format is None:
        return
    output_format = str(output_format).lower()
    alias = _OUTPUT_FORMAT_ALIASES.get(output_format)
    if alias is not None:
        warnings.warn(
            f"RHIME output_format {output_format!r} is deprecated; use {alias!r} instead.",
            UserWarning,
            stacklevel=3,
        )
        output_format = alias
    params["output_format"] = output_format


def coerce_simple_param_types(params: dict[str, Any]) -> None:
    """Coerce simple scalar options in-place before spec construction."""
    for name in _INT_OPTIONS:
        if name not in params or params[name] is None:
            continue
        params[name] = _coerce_int_option(name, params[name])


def _coerce_int_option(name: str, value: Any) -> int:
    """Coerce a RHIME integer option while rejecting ambiguous values."""
    if isinstance(value, bool):
        raise ValueError(_invalid_config_type_message(name, "an integer", value))
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(_invalid_config_type_message(name, "an integer", value))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_invalid_config_type_message(name, "an integer", value)) from exc


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


def validate_rhime_param_types(params: Mapping[str, Any]) -> None:
    """Validate structured RHIME parameter types before preparation begins."""
    for prior_name in _PRIOR_OPTIONS:
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

    for mapping_name in _MAPPING_OPTIONS:
        _validate_mapping_option(params, mapping_name)

    if "min_error_options" in params:
        normalise_min_error_options(params["min_error_options"])

    if "power" in params and params["power"] is not None:
        power = params["power"]
        if not isinstance(power, Mapping | int | float):
            raise ValueError(_invalid_config_type_message("power", "a mapping/dict or number", power))


def normalise_optional_mapping(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Copy an optional mapping so specs do not retain caller-owned dicts."""
    return None if value is None else dict(value)


def normalise_sector_priors(
    sector_priors: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    """Copy optional sector-prior mappings with string sector keys."""
    if sector_priors is None:
        return None
    return {str(sector): dict(prior) for sector, prior in sector_priors.items()}


def validate_multisector_x_prior(x_prior: Mapping[str, Any] | None) -> None:
    """Raise if multi-sector ``x_prior`` is not a shared prior spec."""
    if x_prior is None or "pdf" in x_prior:
        return
    raise ValueError(
        "Invalid RHIME config value for `x_prior`: multi-sector source-keyed priors are not "
        "supported via `xprior`/`x_prior`; use `sector_priors` keyed by sector name, or provide "
        "a single shared prior dict with top-level `pdf`."
    )


def normalise_sector_sources(
    sector_sources: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Copy optional sector-to-source mappings with string names."""
    if sector_sources is None:
        return None
    normalized = {str(sector): str(source) for sector, source in sector_sources.items()}
    invalid = [
        (sector, source)
        for sector, source in normalized.items()
        if not sector.strip() or not source.strip() or source in {"None", "none"}
    ]
    if invalid:
        raise ValueError(
            "`sector_sources` must map non-empty sector names to non-empty OpenGHG source values; "
            f"invalid mapping(s): {invalid!r}."
        )
    return normalized


def required_run_params() -> set[str]:
    """Return RHIME parameters required before data preparation."""
    return {
        "species",
        "sites",
        "averaging_period",
        "domain",
        "start_date",
        "end_date",
        "output_name",
    }


def is_missing_required_value(value: Any) -> bool:
    """Return true when a required RHIME parameter has no usable value."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) == 0:
        return True
    return False


def validate_required_params(params: Mapping[str, Any]) -> None:
    """Raise if normalized run parameters are missing required values."""
    missing = [
        name
        for name in sorted(required_run_params())
        if name not in params or is_missing_required_value(params[name])
    ]
    if missing:
        raise ValueError(f"Required RHIME parameter(s) missing: {missing!r}")


def validate_supported_params(params: Mapping[str, Any]) -> None:
    """Raise if normalized run parameters contain unsupported keys.

    Preparation-option ownership is explicit in
    :data:`RHIME_PREPARATION_OPTION_NAMES`; this validator never reflects over
    a callable signature.

    Args:
        params: Normalized RHIME parameters to validate.

    Raises:
        ValueError: If ``params`` contains one or more unsupported names.
    """
    runner_params = {
        "x_prior",
        "bc_prior",
        "sigma_prior",
        "offset_prior",
        "sector_priors",
        "sector_sources",
        "pollution_events_from_obs",
        "no_model_error",
        "power",
        "draws",
        "burn",
        "tune",
        "chains",
        "nuts_sampler",
        "progressbar",
        "sample_kwargs",
        "posterior_predictive_kwargs",
        "output_format",
        "output_path",
        "save_trace",
        "save_inversion_output",
        "paris_postprocessing_kwargs",
        "output_filename_convention",
        "offset_args",
        "country_file",
        "add_offset",
        "sigma_per_site",
        "sigma_freq",
        "sigma_freq_anchor",
        "mismatch_model",
        "use_minimum_error_floor",
        "aggregation_error_mode",
    }
    supported = RHIME_PREPARATION_OPTION_NAMES | runner_params | required_run_params()
    unsupported = sorted(set(params) - supported)
    if unsupported:
        raise ValueError(f"Unsupported RHIME parameter(s) for `resolve_rhime_options`: {unsupported!r}")


def _tuple_from_optional_sequence(value: Any) -> tuple[str | None, ...]:
    """Convert optional scalar or sequence values into tuple metadata."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return tuple(cast(str | None, item) for item in value)
    return (str(value),)


def _validate_sector_source_mapping(
    flux_sources: Sequence[str],
    sector_sources: Mapping[str, str],
) -> list[str]:
    """Validate the current one-to-one sector/source routing contract."""
    source_sectors: dict[str, list[str]] = {}
    for sector, source in sector_sources.items():
        source_sectors.setdefault(source, []).append(sector)
    duplicate_sources = {
        source: sector_names for source, sector_names in source_sectors.items() if len(sector_names) > 1
    }
    if duplicate_sources:
        details = ", ".join(
            f"source {source!r} is mapped by sectors {sector_names!r}"
            for source, sector_names in duplicate_sources.items()
        )
        raise ValueError(
            "`sector_sources` must map each current sector to a distinct OpenGHG source; " + details + "."
        )

    mapped_sources = list(sector_sources.values())
    missing_sources = [source for source in flux_sources if source not in mapped_sources]
    unrequested_sources = [source for source in mapped_sources if source not in flux_sources]
    if missing_sources or unrequested_sources:
        raise ValueError(
            "`sector_sources` values must match `flux_sources`; "
            f"missing source mapping(s): {missing_sources!r}; "
            f"unrequested source value(s): {unrequested_sources!r}."
        )
    return mapped_sources


def _make_model_spec(
    *,
    species: str,
    domain: str,
    flux_sources: list[str],
    x_prior: dict[str, Any] | None,
    sector_priors: Mapping[str, dict[str, Any]] | None,
    sector_sources: Mapping[str, str] | None,
    bc_prior: dict[str, Any] | None,
    offset_prior: dict[str, Any] | None,
    use_bc: bool,
    likelihood: LikelihoodSettings | None,
    add_offset: bool,
    offset_args: dict[str, Any] | None,
    aggregation_error_mode: AggregationErrorMode,
) -> RhimeModelSpec:
    """Create a lightweight model spec from normalized run parameters."""
    default_x_prior = DEFAULT_X_PRIOR.copy() if x_prior is None else x_prior.copy()
    sectors = []
    used_suffixes: set[str] = set()
    if sector_sources is not None:
        _validate_sector_source_mapping(flux_sources, sector_sources)
        sector_items = list(sector_sources.items())
    else:
        sector_items = [(source, source) for source in flux_sources]

    sector_names = [name for name, _ in sector_items]
    if sector_priors is not None:
        missing_priors = [name for name in sector_names if name not in sector_priors]
        unused_priors = [name for name in sector_priors if name not in sector_names]
        if missing_priors or unused_priors:
            raise ValueError(
                "`sector_priors` must define exactly one prior for every sector when supplied; "
                f"missing sector prior(s): {missing_priors!r}; "
                f"unused sector prior key(s): {unused_priors!r}."
            )

    for name, source in sector_items:
        suffix = safe_pymc_name(name)
        if suffix in used_suffixes:
            raise ValueError(
                "Sector names must be unique after PyMC name sanitisation; "
                f"duplicate sanitized name {suffix!r}."
            )
        used_suffixes.add(suffix)
        prior = sector_priors[name] if sector_priors is not None else default_x_prior
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
        likelihood=likelihood,
        add_offset=add_offset,
        bc_prior=bc_prior,
        offset_prior=offset_prior,
        offset_args=offset_args,
        aggregation_error_mode=aggregation_error_mode,
    )


def _make_likelihood_settings(
    remaining: dict[str, Any],
    *,
    mismatch_model: MismatchModel | None,
    start_date: str,
) -> LikelihoodSettings | None:
    """Consume only options owned by the selected built-in likelihood."""
    if mismatch_model not in (None, "pollution_event", "additive_sigma"):
        raise ValueError(
            "`mismatch_model` must be None, 'pollution_event', or 'additive_sigma'; "
            f"got {mismatch_model!r}."
        )
    if mismatch_model is None:
        unused = sorted(_LIKELIHOOD_OPTIONS & remaining.keys())
        if unused:
            raise ValueError(
                "Built-in likelihood option(s) cannot be used with `mismatch_model=None`: "
                f"{unused!r}. Pass custom options through `likelihood_kwargs`."
            )
        return None

    invalid = (
        _ADDITIVE_SIGMA_OPTIONS
        if mismatch_model == "pollution_event"
        else _POLLUTION_EVENT_OPTIONS
    ) & remaining.keys()
    if invalid:
        raise ValueError(
            f"`mismatch_model={mismatch_model!r}` does not accept option(s) {sorted(invalid)!r}."
        )

    common = {
        "sigma_prior": normalise_optional_mapping(remaining.pop("sigma_prior", None)),
        "sigma_freq": remaining.pop("sigma_freq", None),
        "sigma_per_site": remaining.pop("sigma_per_site", True),
        "sigma_freq_anchor": remaining.pop("sigma_freq_anchor", start_date),
        "no_model_error": remaining.pop("no_model_error", False),
    }
    if mismatch_model == "pollution_event":
        return PollutionEventSettings(
            **common,
            pollution_events_from_obs=remaining.pop("pollution_events_from_obs", False),
            power=remaining.pop("power", 1.99),
        )
    return AdditiveSigmaSettings(
        **common,
        use_minimum_error_floor=remaining.pop("use_minimum_error_floor", False),
    )


def make_rhime_runner_setup(
    *,
    params: Mapping[str, Any],
    multisector: bool,
) -> RhimeRunnerSetup:
    """Normalize raw RHIME parameters into specs and preparation arguments.

    Args:
        params: Raw direct-Python or configuration-derived RHIME options.
        multisector: Whether to construct the multi-sector runner setup.

    Returns:
        Resolved run specification, sampler, and preparation arguments.

    Raises:
        ValueError: If options are missing, unsupported, malformed, or
            incompatible with the selected runner mode.
    """
    normalized = normalise_rhime_params(params)
    validate_required_params(normalized)
    validate_supported_params(normalized)

    remaining = dict(normalized)
    flux_sources = resolve_flux_sources(flux_sources=remaining.pop("flux_sources", None))
    sector_sources = normalise_sector_sources(remaining.pop("sector_sources", None))
    if not multisector and sector_sources is not None:
        raise ValueError("`sector_sources` is only supported by `run_rhime_multisector`.")
    data_flux_sources = (
        _validate_sector_source_mapping(flux_sources, sector_sources)
        if sector_sources is not None
        else flux_sources
    )
    if multisector and len(data_flux_sources) < 2:
        raise ValueError("`run_rhime_multisector` requires at least two flux sources.")
    if not multisector and len(flux_sources) != 1:
        raise ValueError("`run_rhime` requires exactly one flux source.")

    species = remaining.pop("species")
    sites = as_list(remaining.pop("sites")) or []
    domain = remaining.pop("domain")
    averaging_period = remaining.pop("averaging_period")
    start_date = remaining.pop("start_date")
    end_date = remaining.pop("end_date")
    output_path = remaining.pop("output_path", None)
    output_name = remaining.pop("output_name")

    x_prior = normalise_optional_mapping(remaining.pop("x_prior", None))
    bc_prior = normalise_optional_mapping(remaining.pop("bc_prior", None))
    offset_prior = normalise_optional_mapping(remaining.pop("offset_prior", None))
    sector_priors = normalise_sector_priors(remaining.pop("sector_priors", None))
    if multisector:
        validate_multisector_x_prior(x_prior)
    offset_args = normalise_optional_mapping(remaining.get("offset_args"))

    use_bc = remaining.get("use_bc", True)
    mismatch_model = cast(
        MismatchModel | None,
        remaining.pop("mismatch_model", "pollution_event"),
    )
    likelihood = _make_likelihood_settings(
        remaining,
        mismatch_model=mismatch_model,
        start_date=start_date,
    )
    add_offset = remaining.get("add_offset", False)
    aggregation_error_mode = cast(
        AggregationErrorMode,
        remaining.pop("aggregation_error_mode", "none"),
    )

    sampler = RhimeSampler(
        draws=remaining.pop("draws", 1000),
        burn=remaining.pop("burn", 0),
        tune=remaining.pop("tune", 1000),
        chains=remaining.pop("chains", 4),
        nuts_sampler=remaining.pop("nuts_sampler", "pymc"),
        progressbar=remaining.pop("progressbar", False),
        sample_kwargs=normalise_optional_mapping(remaining.pop("sample_kwargs", None)),
        posterior_predictive_kwargs=normalise_optional_mapping(
            remaining.pop("posterior_predictive_kwargs", None)
        ),
    )
    output_format = remaining.pop("output_format", "inv_out")
    save_inversion_output = remaining.pop("save_inversion_output", output_format == "inv_out")
    output_spec = make_output_spec(
        output_format=output_format,
        output_path=output_path,
        output_name=output_name,
        save_trace=remaining.pop("save_trace", False),
        save_inversion_output=save_inversion_output,
        country_file=remaining.get("country_file"),
        paris_postprocessing_kwargs=normalise_optional_mapping(
            remaining.pop("paris_postprocessing_kwargs", None)
        ),
        output_filename_convention=remaining.pop("output_filename_convention", "rhime"),
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
        offset_prior=offset_prior,
        use_bc=use_bc,
        likelihood=likelihood,
        add_offset=add_offset,
        offset_args=offset_args,
        aggregation_error_mode=aggregation_error_mode,
    )
    run_spec = RhimeRunSpec(
        start_date=start_date,
        end_date=end_date,
        sites=tuple(sites),
        averaging_period=_tuple_from_optional_sequence(averaging_period),
        model=model_spec,
        output=output_spec,
        split_by_sectors=multisector,
    )

    data_candidate_args = {
        **RHIME_PREPARATION_DEFAULTS,
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
    }
    data_args = {
        name: value for name, value in data_candidate_args.items() if name in RHIME_PREPARATION_OPTION_NAMES
    }
    data_args["min_error_options"] = normalise_min_error_options(data_args["min_error_options"])
    return RhimeRunnerSetup(run_spec=run_spec, sampler=sampler, data_args=data_args)
