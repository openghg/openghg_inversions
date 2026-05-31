"""RHIME parameter loading, normalisation, and validation helpers.

This private module keeps raw config/API parameter handling out of the public
RHIME runner module. It is intentionally limited to INI compatibility, legacy
alias handling, simple scalar coercion, and validation of raw dictionaries.
Future YAML/schema frontends should target the same normalized parameter model
before constructing RHIME specs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import warnings

from openghg_inversions.config import config

_ALIASES = {
    "outputpath": "output_path",
    "outputname": "output_name",
    "xprior": "x_prior",
    "bcprior": "bc_prior",
    "sigprior": "sigma_prior",
    "offsetprior": "offset_prior",
    "emissions_name": "flux_sources",
}

_INT_OPTIONS = ("nit", "burn", "tune", "nchain")
_MAPPING_OPTIONS = (
    "sampler_kwargs",
    "paris_postprocessing_kwargs",
    "offset_args",
    "min_error_options",
    "sector_sources",
)
_PRIOR_OPTIONS = ("x_prior", "bc_prior", "sigma_prior", "offset_prior")


def as_list(value: str | Sequence[str] | None) -> list[str] | None:
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
    resolved = as_list(flux_sources)
    if resolved is None:
        resolved = as_list(emissions_name)
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
    return normalise_rhime_params(params)


def normalise_rhime_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize aliases, coerce simple scalars, and validate structured values."""
    normalized = normalise_param_aliases(params)
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


def normalise_sector_sources(
    sector_sources: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Copy optional sector-to-source mappings with string names."""
    if sector_sources is None:
        return None
    return {str(sector): str(source) for sector, source in sector_sources.items()}


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


def validate_supported_params(params: Mapping[str, Any], *, data_params: set[str]) -> None:
    """Raise if normalized run parameters contain unsupported keys."""
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
    supported = data_params | runner_params | required_run_params()
    unsupported = sorted(set(params) - supported)
    if unsupported:
        raise ValueError(f"Unsupported RHIME parameter(s): {unsupported!r}")
