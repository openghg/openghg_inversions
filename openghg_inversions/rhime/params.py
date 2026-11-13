"""RHIME option resolution and complete requested configuration.

The INI frontend decodes file options; callers apply overrides and consume
recipe-specific options before canonical resolution. Resolution constructs the
complete requested configuration before acquisition; retained run metadata is
derived only after preparation.
The concrete configuration and its composed values own supported names and
defaults; scientific callable signatures do not define the external schema.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, fields
from math import isfinite
from pathlib import Path
from typing import Any, get_args, get_type_hints

import numpy as np
from pydantic import ConfigDict, TypeAdapter, ValidationInfo, field_serializer, field_validator
from pydantic.dataclasses import dataclass as validated_dataclass

from openghg_inversions.basis.boundary import BOUNDARY_BASIS_ALGORITHMS, boundary_basis
from openghg_inversions.rhime._resources import resolve_resource_paths
from openghg_inversions.filters import validate_filter_options
from openghg_inversions.flux_sanitization import FluxNonFiniteCheck
from openghg_inversions.inversion_data._site_options import SiteOptions
from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models._flux import safe_pymc_name
from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.rhime.specs import (
    DEFAULT_BC_PRIOR,
    DEFAULT_OFFSET_PRIOR,
    DEFAULT_X_PRIOR,
    LikelihoodSettings,
    RhimeModelSpec,
    RhimeOutputSpec,
    SectorSpec,
    LIKELIHOOD_OPTION_NAMES,
    LIKELIHOOD_PRIOR_OPTION_NAMES,
    _copy_option_containers,
    make_likelihood_settings as _make_likelihood_settings,
    normalise_optional_mapping,
)


@validated_dataclass(frozen=True, kw_only=True, config=ConfigDict(strict=True, extra="forbid", validate_default=True))
class BcBasisOptions:
    """Boundary-condition basis selection, used only when BCs are enabled.

    The external form is the nested ``bc_basis`` mapping keyed by field name.
    INI files spell the same options flat, for example ``bc_basis_algorithm``;
    see :func:`nest_flat_options`. Construction
    validates field types without reading files or executing numerical values.
    :meth:`from_params` checks algorithm applicability when BCs are active.
    JSON export requires finite numbers, including inactive split heights.

    Attributes:
        paths: Exact labelled BC basis files. When set, they take precedence
            and ``algorithm`` is inactive.
        algorithm: Built-in basis used without files: ``"NESW"`` (one region
            per whole curtain) or ``"twelve-region"`` (two lower horizontal
            halves and one upper region per curtain).
        split_height: Finite twelve-region split in the curtains' native
            height-coordinate units; no unit conversion or vertical datum is
            assumed. Ignored by NESW and by exact files.

    Raises:
        pydantic.ValidationError: If a field has an invalid type or an unknown
            constructor keyword is supplied.
    """

    paths: tuple[Path, ...] | None = None
    algorithm: str | None = "NESW"
    split_height: float = 3000.0

    @field_validator("split_height", mode="before")
    @classmethod
    def _eager_height(cls, value: Any, info: ValidationInfo) -> Any:
        """Accept eager scalar numbers without invoking arbitrary conversion hooks."""
        if isinstance(value, np.integer | np.floating):
            value = value.item()
        if type(value) not in (int, float):
            raise ValueError("split_height requires an eager numeric scalar")
        if info.mode == "json" and not isfinite(value):
            raise ValueError("JSON split_height must be finite")
        return value

    @field_serializer("split_height", when_used="json")
    @staticmethod
    def _json_height(value: float) -> float:
        """Reject nonfinite values before the JSON serializer can replace them with null."""
        if not isfinite(value):
            raise ValueError("JSON split_height must be finite")
        return value

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | BcBasisOptions, *, active: bool) -> BcBasisOptions:
        """Resolve the nested ``bc_basis`` mapping.

        The built-in algorithm and split height are validated only when the
        selection is active and no exact files are selected, so disabled or
        file-backed selections may carry inactive values.

        Args:
            params: Field values keyed by field name, with resolved filesystem
                paths, or an already resolved selection.
            active: Whether the data configuration uses BC inputs.

        Returns:
            The selection, with defaults for omitted fields.

        Raises:
            ValueError: If field types or names are invalid, or an active
                built-in selection has an unknown algorithm or invalid height.
        """
        if isinstance(params, BcBasisOptions):
            options = params
        else:
            if not isinstance(params, Mapping):
                raise ValueError(f"`bc_basis` must be a mapping; got {type(params).__name__}.")
            options = cls(**params)
        if active and options.paths is None:
            if options.algorithm not in BOUNDARY_BASIS_ALGORITHMS:
                raise ValueError(
                    f"`bc_basis.algorithm` must be one of {BOUNDARY_BASIS_ALGORITHMS!r} without selected files; "
                    f"got {options.algorithm!r}."
                )
            try:
                boundary_basis(options.algorithm, split_height=options.split_height)
            except ValueError as exc:
                raise ValueError(f"`bc_basis.split_height`: {exc}") from exc
        return options

    def to_params(self) -> dict[str, Any]:
        """Return nested Python options, retaining Path objects and path tuples.

        Returns:
            A new field mapping accepted by :meth:`from_params`.
        """
        return _BC_BASIS_ADAPTER.dump_python(self)

    def to_json(self) -> str:
        """Encode portable options without reading selected resources.

        Returns:
            JSON with paths encoded as strings and tuples as arrays.

        Raises:
            pydantic_core.PydanticSerializationError: If the split height is nonfinite.
        """
        return _BC_BASIS_ADAPTER.dump_json(self).decode()

    @classmethod
    def from_json(cls, encoded: str | bytes, *, active: bool) -> BcBasisOptions:
        """Read portable options and check applicability for the consuming phase.

        Args:
            encoded: JSON options with optional fields defaulted as in construction.
            active: Whether the data configuration uses BC inputs.

        Returns:
            Validated options. Relative paths remain relative; the frontend owns
            resolution against its invocation directory.

        Raises:
            ValueError: If JSON, field values, or the active selection are invalid.
        """
        return cls.from_params(_BC_BASIS_ADAPTER.validate_json(encoded), active=active)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Describe the portable option fields from the same dataclass declaration.

        Returns:
            JSON Schema for the nested ``bc_basis`` value. Conditional algorithm
            applicability remains owned by :meth:`from_params`.
        """
        return _BC_BASIS_ADAPTER.json_schema()


_BC_BASIS_ADAPTER = TypeAdapter(BcBasisOptions)


@dataclass(frozen=True)
class OptionGroup:
    """External spelling of one nested option group.

    The nested form is ``{key: {field: value}}``; the flat INI form of the
    same option is ``<prefix><key>_<field>``. Flat names are generated from
    the dataclass fields, never split at underscores.

    Attributes:
        key: Nested mapping key and the owning configuration field name.
        options: Dataclass whose fields are the group's option names.
    """

    key: str
    options: type

    def flat_names(self, prefix: str = "") -> dict[str, str]:
        """Map each flat option name to its field name.

        Args:
            prefix: Outer prefix stacked before ``key``, such as ``"inner_"``.

        Returns:
            A new mapping from ``<prefix><key>_<field>`` to ``field``.
        """
        stem = f"{prefix}{self.key}_"
        return {stem + f.name: f.name for f in fields(self.options)}


# Option groups nested inside ``DataConfig``; register new groups here.
DATA_OPTION_GROUPS: tuple[OptionGroup, ...] = (
    OptionGroup("bc_basis", BcBasisOptions),
)


def nest_flat_options(
    params: Mapping[str, Any], groups: Sequence[OptionGroup], *, prefix: str = "", deprecated: bool = False,
) -> dict[str, Any]:
    """Move flat group options into nested group mappings.

    For example, ``{"bc_basis_algorithm": "NESW"}`` becomes
    ``{"bc_basis": {"algorithm": "NESW"}}``. Only exact flat names derived
    from each group's fields are moved; other names are kept as they are.

    Args:
        params: Options for one configuration owner.
        groups: Option groups that the owner nests.
        prefix: Outer prefix stacked before each group key, such as ``"inner_"``;
            the nested key is then ``prefix + key``.
        deprecated: Whether flat spellings emit a ``DeprecationWarning``.

    Returns:
        A new dictionary, borrowing option values.

    Raises:
        ValueError: If a group is given in both flat and nested form.
    """
    result = dict(params)
    for group in groups:
        key = prefix + group.key
        flat = {name: field for name, field in group.flat_names(prefix).items() if name in result}
        if not flat:
            continue
        if key in result:
            raise ValueError(f"`{key}` is given both as a mapping and as flat option(s) {sorted(flat)!r}; use one form.")
        if deprecated:
            warnings.warn(
                f"Flat option(s) {sorted(flat)!r} are deprecated; pass a nested `{key}` mapping with "
                f"field(s) {sorted(flat.values())!r} instead.",
                DeprecationWarning, stacklevel=3,
            )
        result[key] = {field: result.pop(name) for name, field in flat.items()}
    return result


@dataclass(frozen=True, kw_only=True)
class DataConfig:
    """Resolved acquisition and preparation choices, independent of inference.

    External options enter through ``from_params``. Ordinary option containers
    are owned; numerical payloads remain borrowed. Direct construction expects
    resolved values and does not copy or execute scientific work. Option
    groups registered in :data:`DATA_OPTION_GROUPS`, such as ``bc_basis``,
    are nested mappings in external options; their flat INI spellings are
    accepted with a ``DeprecationWarning``.

    Attributes:
        site_options: Per-site observation and footprint selectors.
        species: Species name used for OpenGHG searches.
        domain: Footprint domain.
        start_date: Inclusive start of the inversion period.
        end_date: Exclusive end of the inversion period.
        flux_sources: OpenGHG flux ``source`` values, in model order.
        split_by_sectors: Whether flux sources are kept as separate sectors
            (multisector recipe) rather than combined.
        bc_store: OpenGHG store searched for boundary conditions.
        obs_store: OpenGHG store searched for observations.
        footprint_store: OpenGHG store searched for footprints.
        flux_store: OpenGHG store searched for fluxes.
        flux_domain: Domain of the source flux dataset; ``None`` uses ``domain``.
        fp_model: Footprint model name, when selecting among models.
        fp_species: Footprint species, when it differs from ``species``.
        calibration_scale: Observation calibration scale to convert to.
        use_bc: Whether boundary-condition inputs are acquired and prepared.
        basis_path: Exact retained ``BasisFunctions`` artifact.
        basis_partition_path: Exact labelled flux basis partition file.
        landsea_path: Land/sea class map used by basis preparation.
        basis_partition_paths: Exact labelled partition files, sorted before
            loading; mutually exclusive with the two singular basis paths.
        bc_basis: Boundary-condition basis selection, used when ``use_bc`` is true.
        outer_regions_path: Fixed outer-region map used by basis preparation.
        bc_input: OpenGHG boundary-condition input name, when selecting among several.
        basis_algorithm: Flux basis algorithm used when no basis file is
            selected: ``"quadtree"``, ``"weighted"`` or ``"region_constrained"``.
        nbasis: Requested number of flux basis regions.
        filters: Filter names, or per-site filter names, applied before
            observation errors and basis construction.
        filter_options: Keyword options for named filters, such as
            ``{"local_influence": {"window_size": 5, "threshold": 0.15}}``.
            Omitted options keep each filter's defaults.
        fix_basis_outer_regions: Whether outer regions are fixed from
            ``outer_regions_path`` during basis preparation.
        averaging_error: Whether observation averaging variability is added
            to the observation uncertainty.
        basis_output_path: Optional file at which the prepared basis is saved.
        flux_non_finite_check: Non-finite flux handling: ``"lazy"`` keeps the
            zero fill lazy; ``"count"`` computes exact non-finite counts.
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
    bc_basis: BcBasisOptions = BcBasisOptions()
    outer_regions_path: Path | None = None
    bc_input: str | None = None
    basis_algorithm: str | None = "weighted"
    nbasis: int = 100
    filters: str | list[str | None] | dict[str, str | None | list[str | None]] | None = None
    filter_options: dict[str, dict[str, Any]] | None = None
    fix_basis_outer_regions: bool = False
    averaging_error: bool = True
    basis_output_path: Path | None = None
    flux_non_finite_check: FluxNonFiniteCheck = "lazy"

    @classmethod
    def option_names(cls) -> frozenset[str]:
        """External names declared by data and site-selector owners, including group keys."""
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

        Option groups are nested mappings, such as ``bc_basis={"algorithm": "NESW"}``.
        Their flat spellings, such as ``bc_basis_algorithm``, are deprecated but
        still accepted. Unknown or wrong-phase names fail. An explicit ``split_by_sectors``
        must agree with the selected ``multisector`` recipe. Select the data mapping from
        ``group_params`` first when consuming a shared configuration file.

        Args:
            params: Canonical acquisition/preparation options, including site selectors.
            multisector: Explicit recipe selection; omitted layout values inherit it.

        Returns:
            Resolved data choices with owned option containers and borrowed arrays.

        Raises:
            ValueError: If required options are missing, values are invalid, an
                option group is given in both flat and nested form, or an
                explicit layout conflicts with the selected recipe.
        """
        remaining = nest_flat_options(normalise_rhime_params(params), DATA_OPTION_GROUPS, deprecated=True)
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
            raise ValueError("Multisector configuration requires at least two flux sources.")
        if not multisector and len(flux_sources) != 1:
            raise ValueError("Standard configuration requires exactly one flux source.")
        check = remaining.get("flux_non_finite_check", cls.flux_non_finite_check)
        if check not in get_args(FluxNonFiniteCheck):
            raise ValueError(f"Invalid `flux_non_finite_check`: {check!r}.")
        basis_algorithm = remaining.get("basis_algorithm", cls.basis_algorithm)
        allowed = ("quadtree", "weighted", "region_constrained")
        if (remaining.get("basis_path") is None and remaining.get("basis_partition_path") is None
                and remaining.get("basis_partition_paths") is None and basis_algorithm not in allowed):
            raise ValueError(f"`basis_algorithm` must be one of {allowed!r}; got {basis_algorithm!r}.")
        remaining["bc_basis"] = BcBasisOptions.from_params(
            remaining.get("bc_basis", {}), active=remaining.get("use_bc", cls.use_bc))
        remaining["sites"] = as_list(remaining["sites"])
        site_options = SiteOptions.from_inputs(**{
            f.name: remaining.pop(f.name) for f in fields(SiteOptions) if f.name in remaining
        })
        remaining["filters"] = _copy_option_containers(remaining.get("filters", cls.filters))
        if remaining.get("filter_options") is not None:
            remaining["filter_options"] = validate_filter_options(remaining["filter_options"])
        remaining["split_by_sectors"] = multisector
        return cls(site_options=site_options, flux_sources=tuple(flux_sources), **remaining)

    def select(self, *names: str) -> dict[str, Any]:
        """Return named direct attributes in a new dictionary, borrowing values."""
        return {name: getattr(self, name) for name in names}

    def to_params(self) -> dict[str, Any]:
        """Return these choices as external options that :meth:`from_params` reads back.

        Site selectors are flattened to their option names; option groups are
        nested mappings. Values are borrowed.
        """
        params = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "site_options"}
        params.update({f.name: getattr(self.site_options, f.name) for f in fields(SiteOptions)})
        params.update({group.key: getattr(self, group.key).to_params() for group in DATA_OPTION_GROUPS})
        return params


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
        return cls(
            data=data, model=model,
            sampler=resolve_sampler_options(sections["sampler"]),
            output=RhimeOutputSpec.from_params(sections["output"], multisector=multisector),
        )


# Data facts a model request may repeat; each must agree with the data configuration.
REQUEST_CLAIM_NAMES = (
    "species", "domain", "flux_sources", "start_date", "end_date", "split_by_sectors", "use_bc",
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
    inactive groups; each selected resolver owns its value validation. Flat
    spellings of data option groups, such as ``bc_basis_algorithm``, are
    nested with a ``DeprecationWarning``.

    Args:
        params: Canonical flat options or data/model/sampler/output group mappings.

    Returns:
        New dictionaries grouped by their explicit owners, borrowing option values.

    Raises:
        ValueError: If a group is not a mapping, an option name is unsupported,
            or a data option group is given in both flat and nested form.
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
            if group == "data":
                values = nest_flat_options(values, DATA_OPTION_GROUPS, deprecated=True)
            unknown.extend(f"{group}.{name}" for name in set(values) - names)
            result[group] = values
        if unknown:
            raise ValueError(f"Unsupported grouped RHIME parameter(s): {sorted(unknown)!r}")
        return result
    normalized = nest_flat_options(params, DATA_OPTION_GROUPS, deprecated=True)
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


def resolve_model_options(params: Mapping[str, Any], *, data_config: DataConfig) -> RhimeModelSpec:
    """Bind raw model choices to authoritative acquisition and preparation choices.

    Omitted BC use inherits the data configuration. Explicit BC use requires prepared BC
    inputs; repeated source facts must agree with the data configuration. BC
    periods and likelihood floors remain independently selectable model choices.

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
            if claim != expected:
                raise ValueError(f"`{name}` conflicts with data: claimed {claimed!r}, expected {expected!r}.")
    use_bc = remaining.pop("use_bc", data_config.use_bc)
    if use_bc and not data_config.use_bc:
        raise ValueError("Model `use_bc=True` requires data configured with BC inputs.")
    bc_freq = remaining.pop("bc_freq", None)
    bc_freq_anchor = remaining.pop("bc_freq_anchor", data_config.start_date)
    flux_sources = list(data_config.flux_sources)
    sector_sources = normalise_sector_sources(remaining.pop("sector_sources", None))
    if sector_sources is not None:
        if not data_config.split_by_sectors:
            raise ValueError("`sector_sources` is only supported by multisector configuration.")
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
    likelihood = _make_likelihood_settings(
        remaining, mismatch_model=remaining.pop("mismatch_model", None), start_date=data_config.start_date,
    )
    return _make_model_spec(
        species=data_config.species, domain=data_config.domain, flux_sources=flux_sources,
        x_prior=x_prior, sector_priors=sector_priors, sector_sources=sector_sources,
        bc_prior=bc_prior, offset_prior=offset_prior, use_bc=use_bc, likelihood=likelihood,
        add_offset=add_offset, offset_args=offset_args,
        bc_freq=bc_freq, bc_freq_anchor=bc_freq_anchor,
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
        "supported via `x_prior`; use `sector_priors` keyed by sector name, or provide "
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
    bc_freq: str | None,
    bc_freq_anchor: DatetimeLike | None,
    likelihood: LikelihoodSettings | None,
    add_offset: bool,
    offset_args: dict[str, Any] | None,
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
        bc_freq=bc_freq,
        bc_freq_anchor=bc_freq_anchor,
        likelihood=likelihood,
        add_offset=add_offset,
        bc_prior=bc_prior,
        offset_prior=offset_prior,
        offset_args=offset_args,
    )
