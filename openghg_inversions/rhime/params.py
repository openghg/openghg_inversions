"""RHIME parameter loading, normalisation, and validation helpers.

This internal package module keeps raw config/API parameter handling out of the
public RHIME runner. It is intentionally limited to INI compatibility, legacy
alias handling, simple scalar coercion, and validation of raw dictionaries.
Future YAML/schema frontends should target the same normalized parameter model
before constructing RHIME specs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import warnings

from openghg_inversions.config import config
from openghg_inversions.models import DEFAULT_X_PRIOR, RhimeModelSpec, SectorSpec, safe_pymc_name
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import RhimeRunSpec, make_output_spec

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
    }
    supported = data_params | runner_params | required_run_params()
    unsupported = sorted(set(params) - supported)
    if unsupported:
        raise ValueError(f"Unsupported RHIME parameter(s): {unsupported!r}")


def _tuple_from_optional_sequence(value: Any) -> tuple[str | None, ...]:
    """Convert optional scalar or sequence values into tuple metadata."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return tuple(cast(str | None, item) for item in value)
    return (str(value),)


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
    sigma_freq: str | None,
    sigma_freq_anchor: str | None,
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
        sigma_freq=sigma_freq,
        sigma_freq_anchor=sigma_freq_anchor,
        add_offset=add_offset,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        power=power,
        bc_prior=bc_prior,
        sigma_prior=sigma_prior,
        offset_prior=offset_prior,
        offset_args=offset_args,
    )


def make_rhime_runner_setup(
    *,
    params: Mapping[str, Any],
    multisector: bool,
    data_param_names: set[str],
) -> RhimeRunnerSetup:
    """Normalize raw RHIME parameters into specs and preparation arguments."""
    normalized = normalise_rhime_params(params)
    validate_required_params(normalized)
    validate_supported_params(normalized, data_params=data_param_names)

    remaining = dict(normalized)
    flux_sources = resolve_flux_sources(flux_sources=remaining.pop("flux_sources", None))
    sector_sources = normalise_sector_sources(remaining.pop("sector_sources", None))
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
    sites = as_list(remaining.pop("sites")) or []
    domain = remaining.pop("domain")
    averaging_period = remaining.pop("averaging_period")
    start_date = remaining.pop("start_date")
    end_date = remaining.pop("end_date")
    output_path = remaining.pop("output_path", None)
    output_name = remaining.pop("output_name")

    x_prior = normalise_optional_mapping(remaining.pop("x_prior", None))
    bc_prior = normalise_optional_mapping(remaining.pop("bc_prior", None))
    sigma_prior = normalise_optional_mapping(remaining.pop("sigma_prior", None))
    offset_prior = normalise_optional_mapping(remaining.pop("offset_prior", None))
    sector_priors = normalise_sector_priors(remaining.pop("sector_priors", None))
    if multisector:
        validate_multisector_x_prior(x_prior)
    offset_args = normalise_optional_mapping(remaining.get("offset_args"))

    use_bc = remaining.get("use_bc", True)
    sigma_per_site = remaining.get("sigma_per_site", True)
    sigma_freq = remaining.pop("sigma_freq", None)
    add_offset = remaining.get("add_offset", False)
    pollution_events_from_obs = remaining.pop("pollution_events_from_obs", False)
    no_model_error = remaining.pop("no_model_error", False)
    power = remaining.pop("power", 1.99)

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
        sigma_prior=sigma_prior,
        offset_prior=offset_prior,
        use_bc=use_bc,
        sigma_per_site=sigma_per_site,
        sigma_freq=sigma_freq,
        sigma_freq_anchor=start_date,
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
        split_by_sectors=multisector,
    )

    data_candidate_args = {
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
    data_args = {name: value for name, value in data_candidate_args.items() if name in data_param_names}
    return RhimeRunnerSetup(run_spec=run_spec, sampler=sampler, data_args=data_args)
