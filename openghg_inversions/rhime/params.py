"""RHIME parameter loading, normalisation, and validation helpers.

The INI frontend interprets file options and applies overrides before shared
semantic resolution. Resolution constructs the complete requested configuration
before acquisition; retained run metadata is derived only after preparation.
The concrete configuration and its composed values own supported names and
defaults; scientific callable signatures do not define the external schema.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, cast, get_args

from openghg_inversions.basis._functions import basis_functions
from openghg_inversions.flux_sanitization import FluxNonFiniteCheck
from openghg_inversions.inversion_data._site_options import SiteOptions
from openghg_inversions.inversion_data.preparation import MinErrorConfig
from openghg_inversions.inversion_data.prepared_inputs import RhimePreparedInputs
from openghg_inversions.model_error import normalise_min_error_options
from openghg_inversions.models._flux import safe_pymc_name
from openghg_inversions.observation_error import AggregationErrorMode
from openghg_inversions.inference.sampling import RhimeSampler
from .ini import params_from_config as params_from_config, read_rhime_ini as read_rhime_ini
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

_ALIASES = {
    "outputpath": "output_path",
    "outputname": "output_name",
    "xprior": "x_prior",
    "bcprior": "bc_prior",
    "sigprior": "sigma_prior",
    "offsetprior": "offset_prior",
    "emissions_name": "flux_sources",
    "outer_region_definition_file": "outer_regions_path",
}
_OUTPUT_FORMAT_ALIASES = {
    "hbmcmc": "legacy",
    "hbmcmc_postprocessing": "legacy",
}

@dataclass(frozen=True, kw_only=True)
class RhimeConfig:
    """Complete resolved requested configuration before scientific data access.

    Acquisition and preparation choices are direct fields. ``site_options``
    holds complete requested selectors; ``model`` holds scientific recipe
    choices, ``output`` final-product policy and ``sampler`` sampling settings.
    Inference executes only when the sampler receives a constructed model.

    Use :meth:`from_params` for external options; it owns ordinary configuration
    containers and borrows opaque numerical values. Direct construction and
    dataclass replacement expect already-resolved values and do not copy them.
    A frozen record does not make contained mappings or the sampler immutable.
    Acquisition and filtering leave the requested choices intact. The record
    contains no scientific inputs, raw options or retained execution description.
    """

    site_options: SiteOptions
    species: str
    domain: str
    start_date: str
    end_date: str
    output_name: str
    flux_sources: tuple[str, ...]
    split_by_sectors: bool = False
    bc_store: str = "user"
    obs_store: str = "user"
    footprint_store: str = "user"
    emissions_store: str = "user"
    emissions_domain: str | None = None
    fp_model: str | None = None
    fp_species: str | None = None
    calibration_scale: str | None = None
    use_bc: bool = True
    fp_basis_case: str | None = None
    basis_directory: str | Path | None = None
    bc_basis_case: str = "NESW"
    bc_basis_directory: str | Path | None = None
    country_directory: str | Path | None = None
    outer_regions_path: str | Path | None = None
    bc_input: str | None = None
    basis_algorithm: str | None = "weighted"
    nbasis: int = 100
    filters: str | list[str | None] | dict[str, str | None | list[str | None]] | None = None
    fix_basis_outer_regions: bool = False
    averaging_error: bool = True
    bc_freq: str | None = None
    reload_merged_data: bool = False
    save_merged_data: bool = False
    merged_data_dir: str | Path | None = None
    merged_data_name: str | None = None
    basis_output_path: str | Path | None = None
    min_error: MinErrorConfig = 0.0
    min_error_options: dict[str, bool] = field(default_factory=lambda: normalise_min_error_options(None))
    flux_non_finite_check: FluxNonFiniteCheck = "lazy"

    model: RhimeModelSpec
    output: RhimeOutputSpec
    sampler: RhimeSampler

    _COMPOSED_FIELDS: ClassVar[frozenset[str]] = frozenset({"site_options", "model", "output", "sampler"})

    @classmethod
    def preparation_option_names(cls) -> frozenset[str]:
        """Declare external preparation choices through their concrete owners."""
        return frozenset(setting.name for setting in fields(cls)) - cls._COMPOSED_FIELDS | frozenset(
            setting.name for setting in fields(SiteOptions)
        )

    @classmethod
    def required_option_names(cls) -> frozenset[str]:
        """Declare required raw values before translating composed settings.

        Sources keep their own validation and layout comes from the recipe;
        neither replaces the required site shorthand or ordinary request values.
        """
        return frozenset(
            setting.name for setting in fields(cls)
            if setting.default is MISSING and setting.default_factory is MISSING
            and setting.name not in cls._COMPOSED_FIELDS | {"flux_sources"}
        ) | frozenset(SiteOptions.REQUIRED_INPUT_NAMES)

    @classmethod
    def supported_option_names(cls) -> frozenset[str]:
        """Combine advertised external choices without exposing internal fields."""
        return (
            cls.preparation_option_names()
            | frozenset(RhimeModelSpec.CONFIG_OPTION_NAMES)
            | LIKELIHOOD_OPTION_NAMES
            | RhimeOutputSpec.option_names()
            | frozenset(RhimeSampler.CONFIG_OPTION_NAMES)
        )

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, Any],
        *,
        multisector: bool,
    ) -> RhimeConfig:
        """Resolve effective external options into the complete requested run.

        Args:
            params: Raw Python or file-derived options after supported overrides.
            multisector: Whether to resolve the multisector recipe.

        Returns:
            Direct acquisition/preparation fields, scientific model, output and
            existing sampler choices.
            Site shorthand is completely expanded before this returns. No data
            access, model construction or sampling is performed.

        Raises:
            ValueError: If options are missing, unsupported, malformed, or
                incompatible with the selected runner mode.
        """
        normalized = normalise_rhime_params(params)
        validate_required_params(normalized)
        validate_supported_params(normalized)

        for name, choices in (
            ("flux_non_finite_check", get_args(FluxNonFiniteCheck)),
            ("aggregation_error_mode", get_args(AggregationErrorMode)),
        ):
            if name in normalized and normalized[name] not in choices:
                raise ValueError(f"`{name}` must be one of {choices!r}; got {normalized[name]!r}.")

        remaining = normalized
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
        output_name = remaining.pop("output_name")

        x_prior = remaining.pop("x_prior", None)
        bc_prior = normalise_optional_mapping(remaining.pop("bc_prior", None))
        offset_prior = normalise_optional_mapping(remaining.pop("offset_prior", None))
        raw_sector_priors = remaining.pop("sector_priors", None)
        sector_priors = None if raw_sector_priors is None else {
            str(sector): prior for sector, prior in raw_sector_priors.items()
        }
        if multisector:
            validate_multisector_x_prior(x_prior)
        offset_args = normalise_optional_mapping(remaining.pop("offset_args", None))

        use_bc = remaining.get("use_bc", cls.use_bc)
        if use_bc and bc_prior is None:
            bc_prior = dict(DEFAULT_BC_PRIOR)
        mismatch_model = remaining.pop("mismatch_model", None)
        likelihood = _make_likelihood_settings(
            remaining,
            mismatch_model=mismatch_model,
            start_date=start_date,
        )
        add_offset = remaining.pop("add_offset", RhimeModelSpec.add_offset)
        if add_offset and offset_prior is None:
            offset_prior = dict(DEFAULT_OFFSET_PRIOR)
        aggregation_error_mode = cast(
            AggregationErrorMode,
            remaining.pop("aggregation_error_mode", RhimeModelSpec.aggregation_error_mode),
        )

        sampler_options = {
            name: remaining.pop(name) for name in RhimeSampler.CONFIG_OPTION_NAMES if name in remaining
        }
        for name in RhimeSampler.MAPPING_OPTION_NAMES:
            if name in sampler_options:
                sampler_options[name] = normalise_optional_mapping(sampler_options[name])
        sampler = RhimeSampler(**sampler_options)
        output_options = {
            name: remaining.pop(name) for name in RhimeOutputSpec.option_names() if name in remaining
        }
        output_spec = RhimeOutputSpec.from_params(
            {**output_options, "output_name": output_name}, multisector=multisector,
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
        basis_algorithm = remaining.get("basis_algorithm", cls.basis_algorithm)
        if remaining.get("fp_basis_case") is None and basis_algorithm not in basis_functions:
            raise ValueError(
                f"`basis_algorithm` must be one of {tuple(basis_functions)!r} when no `fp_basis_case` "
                f"is supplied; got {basis_algorithm!r}."
            )
        min_error = remaining.pop("min_error", cls.min_error)
        if isinstance(min_error, str) and min_error not in ("residual", "percentile"):
            raise ValueError(f"Named `min_error` methods must be 'residual' or 'percentile'; got {min_error!r}.")
        if min_error is None:
            min_error = cls.min_error
        elif isinstance(min_error, int) and not isinstance(min_error, bool):
            min_error = float(min_error)
        elif isinstance(min_error, dict):
            min_error = dict(min_error)
        min_error_options = normalise_min_error_options(remaining.pop("min_error_options", None))
        site_options = SiteOptions.from_inputs(
            sites=sites,
            averaging_period=averaging_period,
            **{
                setting.name: remaining.pop(setting.name)
                for setting in fields(SiteOptions)
                if setting.name in remaining
            },
        )
        filters = _copy_option_containers(remaining.pop("filters", cls.filters))
        # Recipe mode owns the resolved layout, preserving its precedence over
        # the legacy duplicate raw flag. Everything left is a direct field.
        remaining.pop("split_by_sectors", None)
        return cls(
            site_options=site_options,
            species=species,
            domain=domain,
            start_date=start_date,
            end_date=end_date,
            output_name=output_name,
            flux_sources=tuple(data_flux_sources),
            split_by_sectors=multisector,
            model=model_spec,
            output=output_spec,
            sampler=sampler,
            filters=filters,
            min_error=min_error,
            min_error_options=min_error_options,
            **remaining,
        )

    def select(self, *names: str) -> dict[str, Any]:
        """Select named resolved attributes for an explicit scientific call.

        Args:
            *names: Attribute names to forward, written at the call site.

        Returns:
            A new dictionary containing the selected values by reference.
            Selection does not copy containers or numerical arrays, resolve
            defaults, or apply overrides.

        Raises:
            AttributeError: If a requested attribute does not exist.
        """
        return {name: getattr(self, name) for name in names}

    def retained_run_spec(self, prepared: RhimePreparedInputs) -> RhimeRunSpec:
        """Describe execution using prepared sites and requested date bounds.

        Args:
            prepared: Canonical model inputs with authoritative retained sites
                and averaging periods. Its numerical arrays remain borrowed.

        Returns:
            Run metadata composed with the resolved model and output choices.
        """
        return RhimeRunSpec(
            start_date=self.start_date,
            end_date=self.end_date,
            sites=tuple(prepared.sites),
            averaging_period=tuple(prepared.averaging_period),
            model=self.model,
            output=self.output,
            split_by_sectors=self.split_by_sectors,
        )


RHIME_PREPARATION_OPTION_NAMES = RhimeConfig.preparation_option_names()


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


def normalise_rhime_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize aliases, coerce simple scalars, and validate structured values."""
    normalized = normalise_param_aliases(params)
    if normalized.pop("use_tracer", False):
        raise ValueError("`use_tracer=True` is not supported; tracer inversions are not implemented.")
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


def required_run_params() -> set[str]:
    """Return external requirements declared by the requested configuration."""
    return set(RhimeConfig.required_option_names())


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


def resolve_rhime_config(params: Mapping[str, Any], *, multisector: bool) -> RhimeConfig:
    """Resolve external options through :meth:`RhimeConfig.from_params`.

    Supported aliases, defaults and site shorthand are resolved before data
    access; caller containers and opaque numerical values remain unchanged.
    """
    return RhimeConfig.from_params(params, multisector=multisector)
