"""RHIME option resolution and complete requested configuration.

The INI frontend decodes file options; callers apply overrides and consume
recipe-specific options before canonical resolution. Resolution constructs the
complete requested configuration before acquisition; retained run metadata is
derived only after preparation.
The concrete configuration and its composed values own supported names and
defaults; scientific callable signatures do not define the external schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, get_args, get_type_hints

import pandas as pd

from openghg_inversions.rhime._resources import resolve_resource_paths
from openghg_inversions.flux_sanitization import FluxNonFiniteCheck
from openghg_inversions.inversion_data._site_options import SiteOptions
from openghg_inversions.model_error import MinErrorConfig
from openghg_inversions.inversion_data.prepared_inputs import RhimePreparedInputs
from openghg_inversions.inversion_inputs import normalise_frequency
from openghg_inversions.model_error import normalise_min_error, normalise_min_error_options
from openghg_inversions.models._flux import safe_pymc_name
from openghg_inversions.observation_error import AggregationErrorMode
from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.rhime.specs import (
    DEFAULT_BC_PRIOR,
    DEFAULT_OFFSET_PRIOR,
    DEFAULT_X_PRIOR,
    LikelihoodSettings,
    RhimeModelSpec,
    RhimeOutputSpec,
    RhimeRunSpec,
    SectorSpec,
    LIKELIHOOD_OPTION_NAMES,
    LIKELIHOOD_PRIOR_OPTION_NAMES,
    _copy_option_containers,
    make_likelihood_settings as _make_likelihood_settings,
    normalise_optional_mapping,
)


@dataclass(frozen=True, kw_only=True)
class DataConfig:
    """Resolved acquisition and preparation choices, independent of inference.

    External options enter through ``from_params``. Ordinary option containers
    are owned; numerical payloads remain borrowed. Direct construction expects
    resolved values and does not copy or execute scientific work.
    """

    site_options: SiteOptions
    species: str
    domain: str
    start_date: str
    end_date: str
    flux_sources: tuple[str, ...]
    split_by_sectors: bool = False
    bc_store: str = "user"
    obs_store: str = "user"
    footprint_store: str = "user"
    flux_store: str = "user"
    flux_domain: str | None = None
    fp_model: str | None = None
    fp_species: str | None = None
    calibration_scale: str | None = None
    use_bc: bool = True
    basis_path: Path | None = None
    basis_partition_path: Path | None = None
    landsea_path: Path | None = None
    basis_partition_paths: tuple[Path, ...] | None = None
    bc_basis_paths: tuple[Path, ...] | None = None
    outer_regions_path: Path | None = None
    bc_input: str | None = None
    basis_algorithm: str | None = "weighted"
    nbasis: int = 100
    filters: str | list[str | None] | dict[str, str | None | list[str | None]] | None = None
    fix_basis_outer_regions: bool = False
    averaging_error: bool = True
    bc_freq: str | None = None
    basis_output_path: Path | None = None
    min_error: MinErrorConfig = 0.0
    min_error_options: dict[str, bool] = field(default_factory=lambda: normalise_min_error_options(None))
    flux_non_finite_check: FluxNonFiniteCheck = "lazy"


    @classmethod
    def option_names(cls) -> frozenset[str]:
        """External names declared by data and site-selector owners."""
        return frozenset(f.name for f in fields(cls) if f.name != "site_options") | frozenset(
            f.name for f in fields(SiteOptions)
        )

    @classmethod
    def required_option_names(cls) -> frozenset[str]:
        """Required data choices before site shorthand resolution."""
        return frozenset(
            f.name for f in fields(cls)
            if f.default is MISSING and f.default_factory is MISSING
            and f.name not in {"site_options", "flux_sources"}
        ) | frozenset(SiteOptions.REQUIRED_INPUT_NAMES)

    @classmethod
    def from_params(cls, params: Mapping[str, Any], *, multisector: bool = False) -> DataConfig:
        """Resolve acquisition and preparation options without constructing other groups.

        Unknown or wrong-phase names fail. An explicit ``split_by_sectors``
        must agree with the selected ``multisector`` recipe. Select the data mapping from
        ``group_params`` first when consuming a shared configuration file.

        Args:
            params: Canonical acquisition/preparation options, including site selectors.
            multisector: Explicit recipe selection; omitted layout values inherit it.

        Returns:
            Resolved data choices with owned option containers and borrowed arrays.

        Raises:
            ValueError: If required options are missing, values are invalid, or an
                explicit layout conflicts with the selected recipe.
        """
        remaining = normalise_rhime_params(params)
        remaining = resolve_resource_paths(remaining, annotations=get_type_hints(cls), base_dir=Path.cwd())
        unsupported = sorted(set(remaining) - cls.option_names())
        if unsupported:
            raise ValueError(f"Unsupported data parameter(s): {unsupported!r}")
        if "split_by_sectors" in remaining and remaining["split_by_sectors"] != multisector:
            raise ValueError(
                f"`split_by_sectors={remaining['split_by_sectors']!r}` conflicts with "
                f"the selected recipe (multisector={multisector!r})."
            )
        missing = sorted(name for name in cls.required_option_names()
                         if name not in remaining or is_missing_required_value(remaining[name]))
        if missing:
            raise ValueError(f"Required RHIME parameter(s) missing: {missing!r}")
        flux_sources = resolve_flux_sources(flux_sources=remaining.pop("flux_sources", None))
        if multisector and len(flux_sources) < 2:
            raise ValueError("`run_rhime_multisector` requires at least two flux sources.")
        if not multisector and len(flux_sources) != 1:
            raise ValueError("`run_standard` requires exactly one flux source.")
        check = remaining.get("flux_non_finite_check", cls.flux_non_finite_check)
        if check not in get_args(FluxNonFiniteCheck):
            raise ValueError(f"Invalid `flux_non_finite_check`: {check!r}.")
        basis_algorithm = remaining.get("basis_algorithm", cls.basis_algorithm)
        allowed = ("quadtree", "weighted", "region_constrained")
        if (remaining.get("basis_path") is None and remaining.get("basis_partition_path") is None
                and remaining.get("basis_partition_paths") is None and basis_algorithm not in allowed):
            raise ValueError(f"`basis_algorithm` must be one of {allowed!r}; got {basis_algorithm!r}.")
        _bc_frequency_policy(remaining.get("bc_freq", cls.bc_freq))
        remaining["sites"] = as_list(remaining["sites"])
        site_options = SiteOptions.from_inputs(**{
            f.name: remaining.pop(f.name) for f in fields(SiteOptions) if f.name in remaining
        })
        remaining["filters"] = _copy_option_containers(remaining.get("filters", cls.filters))
        remaining["min_error"] = normalise_min_error(remaining.get("min_error", cls.min_error))
        remaining["min_error_options"] = normalise_min_error_options(remaining.get("min_error_options"))
        remaining["split_by_sectors"] = multisector
        return cls(site_options=site_options, flux_sources=tuple(flux_sources), **remaining)

    def select(self, *names: str) -> dict[str, Any]:
        """Return named direct attributes in a new dictionary, borrowing values."""
        return {name: getattr(self, name) for name in names}

    def to_params(self) -> dict[str, Any]:
        """Return these choices as external options that :meth:`from_params` reads back.

        Site selectors are flattened to their option names; values are borrowed.
        """
        site_names = {f.name for f in fields(SiteOptions)}
        params = {name: getattr(self, name) for name in self.option_names() - site_names}
        params.update({name: getattr(self.site_options, name) for name in site_names})
        return params


def retained_run_spec(
    data: DataConfig, prepared: RhimePreparedInputs, *, model: RhimeModelSpec, output: RhimeOutputSpec,
) -> RhimeRunSpec:
    """Compose the sites retained by preparation with the requested dates and choices.

    Multisector and nested execution retain this transitional run envelope.
    Standard execution passes the separate choices and prepared inputs directly.

    Args:
        data: Data choices that produced ``prepared``.
        prepared: Prepared inputs whose retained sites and averaging periods are recorded.
        model: Resolved model choices.
        output: Requested output choices.

    Returns:
        The run specification used by model construction, results and products.
    """
    return RhimeRunSpec(
        start_date=data.start_date, end_date=data.end_date,
        sites=tuple(prepared.sites), averaging_period=tuple(prepared.averaging_period),
        model=model, output=output, split_by_sectors=data.split_by_sectors,
    )


@dataclass(frozen=True, kw_only=True)
class RhimeConfig:
    """Complete requested configuration with one authoritative owner per group."""

    data: DataConfig
    model: RhimeModelSpec
    output: RhimeOutputSpec
    sampler: RhimeSampler

    @classmethod
    def supported_option_names(cls) -> frozenset[str]:
        """Combine external choices declared by concrete owners."""
        return (DataConfig.option_names() | frozenset(RhimeModelSpec.CONFIG_OPTION_NAMES)
                | LIKELIHOOD_OPTION_NAMES | RhimeOutputSpec.option_names()
                | frozenset(RhimeSampler.CONFIG_OPTION_NAMES))

    @classmethod
    def from_params(cls, params: Mapping[str, Any], *, multisector: bool) -> RhimeConfig:
        """Resolve flat or grouped requests once before acquisition.

        Group mappings permit a model to omit BC terms while preparation keeps
        BC inputs available. Flat options retain their established precedence.

        Args:
            params: Canonical flat options or mappings grouped by their owners.
            multisector: Whether to resolve the multisector recipe.

        Returns:
            Acquisition, model, sampler and output configuration without scientific I/O.

        Raises:
            ValueError: If required options are missing, unsupported, inconsistent or invalid.
        """
        sections = group_params(params)
        if is_missing_required_value(sections["output"].get("output_name")):
            raise ValueError("Required RHIME parameter(s) missing: ['output_name']")
        # Preserve routing errors before the recipe's source-count check.
        sector_sources = sections["model"].get("sector_sources")
        if multisector and isinstance(sector_sources, Mapping):
            source_names = as_list(sections["data"].get("flux_sources")) or []
            if len(source_names) < 2:
                _validate_sector_source_mapping(source_names, normalise_sector_sources(sector_sources) or {})
        data = DataConfig.from_params(sections["data"], multisector=multisector)
        model = resolve_model_options(sections["model"], data_config=data)
        # Multisector source routing owns the retained source order.
        if multisector and sections["model"].get("sector_sources") is not None:
            data = replace(data, flux_sources=tuple(s.flux_source for s in model.sectors))
        return cls(
            data=data, model=model,
            sampler=resolve_sampler_options(sections["sampler"]),
            output=RhimeOutputSpec.from_params(sections["output"], multisector=multisector),
        )

    def retained_run_spec(self, prepared: RhimePreparedInputs) -> RhimeRunSpec:
        """Compose retained sites with this request; see :func:`retained_run_spec`."""
        return retained_run_spec(self.data, prepared, model=self.model, output=self.output)


# Data facts a model request may repeat; each must agree with the data configuration.
REQUEST_CLAIM_NAMES = (
    "species", "domain", "flux_sources", "start_date", "end_date", "split_by_sectors", "bc_freq", "use_bc",
)


def option_groups() -> dict[str, frozenset[str]]:
    """Expose canonical option names from their existing concrete owners."""
    return {
        "data": DataConfig.option_names(),
        "model": frozenset(RhimeModelSpec.CONFIG_OPTION_NAMES) | LIKELIHOOD_OPTION_NAMES | set(REQUEST_CLAIM_NAMES),
        "sampler": frozenset(RhimeSampler.CONFIG_OPTION_NAMES),
        "output": RhimeOutputSpec.option_names(),
    }


def group_params(params: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Group canonical flat options without validating inactive values or defaults.

    Already grouped mappings are accepted as well. Names are checked even for
    inactive groups; each selected resolver owns its value validation.

    Args:
        params: Canonical flat options or data/model/sampler/output group mappings.

    Returns:
        New dictionaries grouped by their explicit owners, borrowing option values.

    Raises:
        ValueError: If a group is not a mapping or an option name is unsupported.
    """
    owners = option_groups()
    if set(params) & owners.keys():
        unknown = list(set(params) - owners.keys())
        result = {}
        for group, names in owners.items():
            values = params.get(group, {})
            if not isinstance(values, Mapping):
                raise ValueError(f"`{group}` must be a mapping.")
            values = dict(values)
            unknown.extend(f"{group}.{name}" for name in set(values) - names)
            result[group] = values
        if unknown:
            raise ValueError(f"Unsupported grouped RHIME parameter(s): {sorted(unknown)!r}")
        return result
    normalized = dict(params)
    validate_supported_params(normalized)
    result = {group: {} for group in owners}
    for name, value in normalized.items():
        for group, names in owners.items():
            if name in names:
                result[group][name] = value
                break
    return result


def resolve_sampler_options(params: Mapping[str, Any]) -> RhimeSampler:
    """Resolve only sampler choices using the existing sampler's defaults.

    Args:
        params: Canonical sampler options; omitted options retain sampler defaults.

    Returns:
        A fresh sampler configuration without executing inference.

    Raises:
        ValueError: If an option is unsupported or has an invalid value.
    """
    remaining = normalise_rhime_params(params)
    unknown = sorted(set(remaining) - set(RhimeSampler.CONFIG_OPTION_NAMES))
    if unknown:
        raise ValueError(f"Unsupported sampler parameter(s): {unknown!r}")
    for name in RhimeSampler.MAPPING_OPTION_NAMES:
        if name in remaining:
            remaining[name] = normalise_optional_mapping(remaining[name])
    return RhimeSampler(**remaining)


def _bc_frequency_policy(value: str | None) -> str | pd.Timedelta | None:
    """Report the configuration key when the shared frequency policy is invalid."""
    try:
        return normalise_frequency(value)
    except ValueError as exc:
        raise ValueError(f"Invalid `bc_freq`: {value!r}.") from exc


def resolve_model_options(params: Mapping[str, Any], *, data_config: DataConfig) -> RhimeModelSpec:
    """Bind raw model choices to authoritative acquisition and preparation choices.

    Omitted BC use inherits the data configuration. Explicit BC use requires prepared BC
    inputs; repeated temporal or source facts must agree with the data configuration.

    Args:
        params: External model options and any repeated data facts.
        data_config: Resolved acquisition and preparation choices that own those facts.

    Returns:
        Model choices bound to the data configuration.

    Raises:
        ValueError: If options are unsupported or invalid, repeated data facts
            conflict, or the model requires unavailable BC inputs.
    """
    remaining = normalise_rhime_params(params)
    facts = tuple(name for name in REQUEST_CLAIM_NAMES if name != "use_bc")
    unknown = sorted(set(remaining) - set(RhimeModelSpec.CONFIG_OPTION_NAMES)
                     - LIKELIHOOD_OPTION_NAMES - set(facts) - {"use_bc"})
    if unknown:
        raise ValueError(f"Unsupported model parameter(s): {unknown!r}")
    for name in facts:
        if name in remaining:
            claimed = remaining.pop(name)
            claim = claimed
            if name == "flux_sources":
                claim = tuple(as_list(claim) or ())
            expected = getattr(data_config, name)
            matches = (
                _bc_frequency_policy(claim) == _bc_frequency_policy(expected)
                if name == "bc_freq" else claim == expected
            )
            if not matches:
                raise ValueError(f"`{name}` conflicts with data: claimed {claimed!r}, expected {expected!r}.")
    use_bc = remaining.pop("use_bc", data_config.use_bc)
    if use_bc and not data_config.use_bc:
        raise ValueError("Model `use_bc=True` requires data configured with BC inputs.")
    flux_sources = list(data_config.flux_sources)
    sector_sources = normalise_sector_sources(remaining.pop("sector_sources", None))
    if sector_sources is not None:
        if not data_config.split_by_sectors:
            raise ValueError("`sector_sources` is only supported by `run_rhime_multisector`.")
        _validate_sector_source_mapping(flux_sources, sector_sources)
    x_prior = remaining.pop("x_prior", None)
    if data_config.split_by_sectors:
        validate_multisector_x_prior(x_prior)
    sector_priors = remaining.pop("sector_priors", None)
    if sector_priors is not None:
        sector_priors = {str(name): prior for name, prior in sector_priors.items()}
    bc_prior = normalise_optional_mapping(remaining.pop("bc_prior", None))
    if use_bc and bc_prior is None:
        bc_prior = dict(DEFAULT_BC_PRIOR)
    add_offset = remaining.pop("add_offset", RhimeModelSpec.add_offset)
    offset_prior = normalise_optional_mapping(remaining.pop("offset_prior", None))
    if add_offset and offset_prior is None:
        offset_prior = dict(DEFAULT_OFFSET_PRIOR)
    offset_args = normalise_optional_mapping(remaining.pop("offset_args", None))
    aggregation = remaining.pop("aggregation_error_mode", RhimeModelSpec.aggregation_error_mode)
    if aggregation not in get_args(AggregationErrorMode):
        raise ValueError(f"Invalid `aggregation_error_mode`: {aggregation!r}.")
    likelihood = _make_likelihood_settings(
        remaining, mismatch_model=remaining.pop("mismatch_model", None), start_date=data_config.start_date,
    )
    return _make_model_spec(
        species=data_config.species, domain=data_config.domain, flux_sources=flux_sources,
        x_prior=x_prior, sector_priors=sector_priors, sector_sources=sector_sources,
        bc_prior=bc_prior, offset_prior=offset_prior, use_bc=use_bc, likelihood=likelihood,
        add_offset=add_offset, offset_args=offset_args, aggregation_error_mode=aggregation,
    )


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
) -> list[str]:
    """Validate canonical OpenGHG flux source names.

    Args:
        flux_sources: Preferred RHIME field containing OpenGHG flux
            ``source`` metadata values.

    Returns:
        Resolved flux source names.

    Raises:
        ValueError: If no usable flux source is supplied.
    """
    resolved = as_list(flux_sources)
    if not resolved or any(source in {"", "None", "none"} for source in resolved):
        raise ValueError("At least one flux source must be supplied via `flux_sources`.")
    duplicates = _duplicate_names(resolved)
    if duplicates:
        raise ValueError(
            f"`flux_sources` must contain unique OpenGHG source values; duplicate source(s): {duplicates!r}."
        )
    return resolved


def normalise_rhime_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Copy canonical options, coerce scalars and validate values."""
    normalized = dict(params)
    coerce_simple_param_types(normalized)
    validate_rhime_param_types(normalized)
    return normalized


def coerce_simple_param_types(params: dict[str, Any]) -> None:
    """Coerce simple scalar options in-place before spec construction."""
    if params.get("output_format") is not None:
        params["output_format"] = str(params["output_format"]).lower()
    for name in RhimeSampler.INTEGER_OPTION_NAMES:
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
        f"but got {type(value).__name__}."
    )


def _validate_mapping_option(params: Mapping[str, Any], name: str) -> None:
    """Raise if a RHIME option is present but is not a mapping or None."""
    if name not in params or params[name] is None:
        return
    if not isinstance(params[name], Mapping):
        raise ValueError(_invalid_config_type_message(name, "a mapping/dict", params[name]))


def validate_rhime_param_types(params: Mapping[str, Any]) -> None:
    """Validate structured RHIME parameter types before preparation begins."""
    for prior_name in RhimeModelSpec.PRIOR_OPTION_NAMES + LIKELIHOOD_PRIOR_OPTION_NAMES:
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

    mapping_names = (
        RhimeSampler.MAPPING_OPTION_NAMES + RhimeOutputSpec.MAPPING_OPTION_NAMES
        + RhimeModelSpec.MAPPING_OPTION_NAMES + ("min_error_options",)
    )
    for mapping_name in mapping_names:
        _validate_mapping_option(params, mapping_name)

    if "min_error_options" in params:
        normalise_min_error_options(params["min_error_options"])

    if "power" in params and params["power"] is not None:
        power = params["power"]
        if not isinstance(power, Mapping | int | float):
            raise ValueError(_invalid_config_type_message("power", "a mapping/dict or number", power))


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


def is_missing_required_value(value: Any) -> bool:
    """Return true when a required RHIME parameter has no usable value."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) == 0:
        return True
    return False


def validate_supported_params(params: Mapping[str, Any]) -> None:
    """Reject names absent from the configuration's consumer-owned choices."""
    unsupported = sorted(set(params) - RhimeConfig.supported_option_names())
    if unsupported:
        raise ValueError(f"Unsupported RHIME parameter(s): {unsupported!r}")


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
    default_x_prior = DEFAULT_X_PRIOR if x_prior is None else x_prior
    sectors = []
    used_suffixes: set[str] = set()
    if sector_sources is not None:
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
                x_prior=_copy_option_containers(dict(prior)),
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
