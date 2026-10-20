"""Lightweight immutable specifications for RHIME models and runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field, fields
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, TYPE_CHECKING, cast, get_args, get_type_hints

from pydantic import BeforeValidator, Field, PlainSerializer, WithJsonSchema, field_validator
from pydantic.dataclasses import dataclass

from openghg_inversions._record_schema import RuntimeConfig, RuntimeFloat

from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.rhime._resources import resolve_resource_paths
from openghg_inversions.models.priors import PriorArgs, trace_prior_records
from openghg_inversions.models.additive_sigma import DEFAULT_ADDITIVE_SIGMA_PRIOR
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import AggregationErrorMode

if TYPE_CHECKING:
    from openghg_inversions.rhime.params import DataConfig

OutputFormat = Literal["none", "inv_out", "basic", "paris", "legacy"]
OutputFilenameConvention = Literal["rhime", "legacy"]
MismatchModel = Literal["pollution_event", "additive_sigma", "fixed_error"]

DEFAULT_X_PRIOR: PriorArgs = {
    "pdf": "lognormal",
    "mean": 1.0,
    "stdev": 1.0,
    "reparameterise": True,
}
DEFAULT_BC_PRIOR: PriorArgs = {
    "pdf": "truncatednormal",
    "mu": 1.0,
    "sigma": 0.05,
    "lower": 0.0,
}
DEFAULT_POLLUTION_EVENT_SIGMA_PRIOR: PriorArgs = {
    "pdf": "uniform",
    "lower": 0.0,
    "upper": 0.1,
}
DEFAULT_OFFSET_PRIOR: PriorArgs = {"pdf": "normal", "mu": 0, "sigma": 1}


def _copy_option_containers(value: Any) -> Any:
    """Own ordinary option containers while borrowing opaque numerical values."""
    if isinstance(value, dict):
        return {key: _copy_option_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_option_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_option_containers(item) for item in value)
    return value


def normalise_optional_mapping(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Own an optional mapping and its ordinary configuration containers."""
    return None if value is None else _copy_option_containers(dict(value))


Name = Annotated[str, Field(pattern=r"\S")]
def _owned_mapping(value: Any) -> Any:
    """Own Python containers, letting the selected union validate their type."""
    return normalise_optional_mapping(value) if isinstance(value, Mapping) else value


BorrowedPrior = Annotated[PriorArgs, BeforeValidator(_owned_mapping)]
BorrowedOptions = Annotated[dict[str, Any], BeforeValidator(_owned_mapping)]
TemporalAnchor = Annotated[
    DatetimeLike,
    WithJsonSchema({"type": "string", "description": "Date or date-time string."}),
    PlainSerializer(str, return_type=str, when_used="json"),
]


@dataclass(frozen=True, config=RuntimeConfig)
class PollutionEventSettings:
    """Serializable settings for pollution-event-scaled model error.

    Args:
        sigma_prior: Prior for the observation-aligned fractional model error.
        sigma_freq: Frequency of the latent model-error periods. ``None`` uses
            one period.
        sigma_per_site: Whether model error varies by observation site.
        sigma_freq_anchor: Optional anchor for fixed-duration periods.
        pollution_events_from_obs: Derive pollution events from observations
            after removing the baseline instead of from modelled pollution.
        power: Exponent or prior used in pollution-event error scaling.
    """

    sigma_prior: BorrowedPrior | None = None
    sigma_freq: str | None = None
    sigma_per_site: bool = True
    sigma_freq_anchor: TemporalAnchor | None = None
    pollution_events_from_obs: bool = False
    power: BorrowedPrior | RuntimeFloat = 1.99

    @property
    def required_prepared_inputs(self) -> tuple[str, ...]:
        """Return prepared arrays owned by this likelihood."""
        return ("min_error",)


@dataclass(frozen=True, config=RuntimeConfig)
class AdditiveSigmaSettings:
    """Serializable settings for additive model-data-mismatch error.

    Args:
        sigma_prior: Prior for the additive model-error standard deviation.
        sigma_freq: Frequency of the latent model-error periods. ``None`` uses
            one period.
        sigma_per_site: Whether model error varies by observation site.
        sigma_freq_anchor: Optional anchor for fixed-duration periods.
        use_minimum_error_floor: Apply the prepared historical minimum total-
            error floor.
    """

    sigma_prior: BorrowedPrior | None = None
    sigma_freq: str | None = None
    sigma_per_site: bool = True
    sigma_freq_anchor: TemporalAnchor | None = None
    use_minimum_error_floor: bool = False

    @property
    def required_prepared_inputs(self) -> tuple[str, ...]:
        """Return prepared arrays owned by this likelihood."""
        return ("min_error",) if self.use_minimum_error_floor else ()


@dataclass(frozen=True, config=RuntimeConfig)
class FixedErrorSettings:
    """Serializable selection of reported observation error only."""

    @property
    def required_prepared_inputs(self) -> tuple[str, ...]:
        """Return prepared arrays owned by this likelihood."""
        return ()


LikelihoodSettings = PollutionEventSettings | AdditiveSigmaSettings | FixedErrorSettings

# These records declare external likelihood choices, not runtime array inputs.
LIKELIHOOD_OPTION_NAMES = frozenset(
    setting.name for settings in get_args(LikelihoodSettings) for setting in fields(settings)
)
LIKELIHOOD_PRIOR_OPTION_NAMES = ("sigma_prior",)


def make_likelihood_settings(
    remaining: dict[str, Any],
    *,
    mismatch_model: MismatchModel | None,
    start_date: str,
) -> LikelihoodSettings | None:
    """Consume the selected likelihood's external choices and resolve defaults.

    The mapping is a resolver-owned working copy. Priors and power mappings
    are copied without copying opaque numerical payloads. Date-dependent
    alignment defaults use the winning requested start date.

    Args:
        remaining: Owned working options after alias and type normalization;
            selected likelihood names are removed from this mapping.
        mismatch_model: Selected built-in mismatch equation, or ``None`` for a
            custom likelihood with no built-in options.
        start_date: Effective request date used for an omitted period anchor.

    Returns:
        Complete settings for the selected likelihood, or ``None`` for custom
        construction. No model, numerical input or inference is constructed.

    Raises:
        ValueError: If the selection is unknown or options belong to another
            built-in likelihood or a custom likelihood.
    """
    if mismatch_model is None:
        unused = sorted(LIKELIHOOD_OPTION_NAMES & remaining.keys())
        if unused:
            raise ValueError(
                "Built-in likelihood option(s) cannot be used with `mismatch_model=None`: "
                f"{unused!r}. Pass custom options through `likelihood_kwargs`."
            )
        return None
    settings_type = _LIKELIHOOD_SETTINGS.get(mismatch_model) if isinstance(mismatch_model, str) else None
    if settings_type is None:
        raise ValueError(
            "`mismatch_model` must be None, 'pollution_event', 'additive_sigma', or "
            "'fixed_error'; "
            f"got {mismatch_model!r}."
        )
    names = {setting.name for setting in fields(settings_type)}
    invalid = (LIKELIHOOD_OPTION_NAMES - names) & remaining.keys()
    if invalid:
        raise ValueError(
            f"`mismatch_model={mismatch_model!r}` does not accept option(s) {sorted(invalid)!r}."
        )
    options = {name: remaining.pop(name) for name in names if name in remaining}
    if settings_type is FixedErrorSettings:
        return FixedErrorSettings()
    if options.get("sigma_prior") is None:
        options["sigma_prior"] = (
            DEFAULT_POLLUTION_EVENT_SIGMA_PRIOR
            if settings_type is PollutionEventSettings
            else DEFAULT_ADDITIVE_SIGMA_PRIOR
        )
    options.setdefault("sigma_freq_anchor", start_date)
    return settings_type(**options)


@dataclass(frozen=True, config=RuntimeConfig)
class SectorSpec:
    """Configuration for one separately optimised flux sector.

    Args:
        name: User-facing sector name.
        flux_source: OpenGHG flux ``source`` used to retrieve this sector.
        x_prior: Prior specification for this sector's flux scaling factors.
        variable_suffix: Stable suffix for sector prior records, private latent
            variables and sector reporting names. The physical flux state and
            its total contribution use shared ``x`` and ``mu`` names.
        state_activity: Optional labelled active/fixed policy for this sector's
            flux-scaling states. ``None`` still applies the default flux policy,
            fixing exactly-zero sensitivity columns to one.
    """

    name: Name
    flux_source: Name
    x_prior: BorrowedPrior
    variable_suffix: str
    state_activity: StateActivity | None = field(default=None, kw_only=True)


@dataclass(frozen=True, config=RuntimeConfig)
class RhimeModelSpec:
    """Scientific options used by the concrete RHIME model recipes.

    Args:
        species: Primary gas or tracer name used for object-store lookup and
            output naming.
        domain: Model domain name.
        sectors: Flux sectors included in the model. Each sector is optimized
            separately and is normally backed by one OpenGHG flux ``source``.
        use_bc: Whether boundary-condition scaling is included.
        likelihood: Resolved built-in likelihood settings, or ``None`` when a
            Python-only custom likelihood owns that step.
        add_offset: Whether model-data offsets are included.
        aggregation_error_mode: Fixed aggregation-error covariance
            representation. The default ``"none"`` preserves the ordinary
            model; other modes are an explicit opt-in.
        bc_prior: Prior specification for boundary-condition scaling factors.
        offset_prior: Prior specification for optional offsets.
        offset_args: Extra keyword arguments forwarded to the offset component.
        bc_state_activity: Optional active/fixed policy for the boundary-
            condition scaling vector. ``None`` preserves the ordinary fully
            sampled BC graph without zero pruning. Supplying a policy opts into
            active/fixed BC construction.
        state_activity: Optional labelled active/fixed state policy shared by
            flux sectors. The default retains exact-zero pruning.

    """

    # External names include sector translation; internal state policies are
    # available to independent builders rather than this configured frontend.
    CONFIG_OPTION_NAMES: ClassVar[tuple[str, ...]] = (
        "x_prior", "bc_prior", "offset_prior", "sector_priors", "sector_sources",
        "offset_args", "add_offset", "mismatch_model", "aggregation_error_mode",
    )
    PRIOR_OPTION_NAMES: ClassVar[tuple[str, ...]] = ("x_prior", "bc_prior", "offset_prior")
    MAPPING_OPTION_NAMES: ClassVar[tuple[str, ...]] = ("sector_sources", "offset_args")

    species: Name
    domain: Name
    sectors: Annotated[tuple[SectorSpec, ...], Field(min_length=1)]
    use_bc: bool = True
    likelihood: LikelihoodSettings | None = field(default=None, kw_only=True)
    add_offset: bool = False
    aggregation_error_mode: AggregationErrorMode = field(default="none", kw_only=True)
    bc_prior: BorrowedPrior | None = None
    offset_prior: BorrowedPrior | None = None
    offset_args: BorrowedOptions | None = None
    bc_state_activity: StateActivity | None = field(default=None, kw_only=True)
    state_activity: StateActivity | None = field(default=None, kw_only=True)

    def describe(self, *, priors: Mapping[str, PriorArgs] | None = None) -> dict[str, Any]:
        """Describe structural choices and construction-owned prior parameters.

        ``priors`` contains inline numerical scalars and registered-array
        references captured while constructing the model. Unused configured
        prior mappings are never traversed or materialized here. Omitting
        ``priors`` omits prior parameter values from the record.

        Args:
            priors: Construction-owned descriptions of used prior parameters.

        Returns:
            Scientific choices and prior descriptions, independent of output policy.
        """
        priors = priors or {}
        likelihood = None
        if self.likelihood is not None:
            kind = next(name for name, cls in _LIKELIHOOD_SETTINGS.items() if isinstance(self.likelihood, cls))
            option_names = tuple(setting.name for setting in fields(self.likelihood))
            options = {name: getattr(self.likelihood, name) for name in option_names
                       if name != "sigma_prior"}
            if "sigma_prior" in option_names:
                options["sigma_prior"] = priors.get("sigma")
            if isinstance(options.get("power"), Mapping):
                options["power"] = priors.get("power")
            likelihood = {"kind": kind, "options": options}
        return {
            **{name: getattr(self, name) for name in _MODEL_RECORD_FIELDS
               if name not in {"bc_prior", "offset_prior"}},
            "bc_prior": priors.get("bc"),
            "offset_prior": priors.get("offset_latent"),
            "sectors": [{"name": sector.name, "flux_source": sector.flux_source,
                         "x_prior": priors.get("x" if len(self.sectors) == 1 and "x" in priors else f"x_{sector.variable_suffix}"),
                         "variable_suffix": sector.variable_suffix}
                        for sector in self.sectors],
            "likelihood": likelihood,
        }

    def validate_recordable(self) -> None:
        """Reject activity policies unsupported by the saved model format."""
        if (self.state_activity is not None or self.bc_state_activity is not None
                or any(sector.state_activity is not None for sector in self.sectors)):
            raise ValueError(
                "Saved sample records cannot represent `state_activity`, `bc_state_activity` "
                "or `SectorSpec.state_activity`; run these models in memory."
            )

    def to_record(self, *, trace: Any, data_config: DataConfig) -> dict[str, Any]:
        """Describe saved model choices using construction-owned trace priors.

        The result is a descriptive ``ModelDescription``, which can contain
        unused or symbolic priors. It cannot reconstruct executable settings.

        Args:
            trace: Sampled trace with explicit constructed-prior metadata.
            data_config: Preparation configuration owning species, domain and sources.

        Returns:
            Model choices without preparation-owned facts or numerical arrays.

        Raises:
            ValueError: If activity policies cannot be recorded, trace prior metadata
                is missing, or the model contradicts the preparation.
        """
        self.validate_recordable()
        if (self.species != data_config.species or self.domain != data_config.domain
                or set(sector.flux_source for sector in self.sectors) != set(data_config.flux_sources)
                or len(self.sectors) != len(data_config.flux_sources)
                or (self.use_bc and not data_config.use_bc)):
            raise ValueError("Sample model conflicts with its preparation.")
        record = self.describe(priors=trace_prior_records(trace))
        record.pop("species")
        record.pop("domain")
        if not data_config.split_by_sectors:
            for sector in record["sectors"]:
                sector.pop("flux_source")
        from openghg_inversions._record_schema import validate_json_keys
        from openghg_inversions._serialization_codecs import _json_value
        from ._model_records import ModelDescription

        validate_json_keys(record)
        return ModelDescription.model_validate(_json_value(record)).model_dump(mode="json", exclude_unset=True)

_MODEL_RECORD_FIELDS = (
    "species", "domain", "use_bc", "add_offset", "aggregation_error_mode",
    "bc_prior", "offset_prior", "offset_args",
)
_LIKELIHOOD_SETTINGS = {
    "fixed_error": FixedErrorSettings,
    "additive_sigma": AdditiveSigmaSettings,
    "pollution_event": PollutionEventSettings,
}


@dataclass(frozen=True, config=RuntimeConfig)
class RhimeOutputSpec:
    """Output settings for a RHIME run.

    Args:
        output_format: Output mode. ``"inv_out"`` saves/returns the modern
            inversion output, ``"basic"`` and ``"paris"`` additionally create
            derived outputs, ``"legacy"`` creates the old HBMCMC-compatible
            NetCDF product from modern RHIME output, and ``"none"`` skips
            output products.
        output_path: Directory for saved outputs.
        output_name: Base output name.
        save_trace: Trace save setting. If true, save to ``output_path`` using
            the default trace file name; if a path, save there.
        save_inversion_output: Inversion-output save setting. Runner parameter
            normalization defaults this to true for ``output_format="inv_out"``
            and false for derived product formats.
        country_file: Optional country mask file used by derived outputs.
        paris_postprocessing_kwargs: Extra keyword arguments for PARIS output
            creation.
        output_filename_convention: Filename convention for derived products.
            Direct RHIME runs use ``"rhime"``. The ``run_hbmcmc.py``
            compatibility shim uses ``"legacy"`` for old SLURM/config
            workflows.
    """

    MAPPING_OPTION_NAMES: ClassVar[tuple[str, ...]] = ("paris_postprocessing_kwargs",)

    @classmethod
    def option_names(cls) -> frozenset[str]:
        """Return the configured output choices declared by this record."""
        return frozenset(setting.name for setting in fields(cls))

    @classmethod
    def from_params(cls, params: Mapping[str, Any], *, multisector: bool) -> RhimeOutputSpec:
        """Resolve output choices without writing products or mutating inputs.

        The active format determines the omitted inversion-output save setting.
        Remaining omitted values use the record's declared defaults. PARIS
        option containers are owned; opaque numerical payloads remain borrowed.

        Relative filesystem paths resolve from the invocation working directory;
        already resolved absolute paths pass through unchanged. File frontends
        must supply paths resolved against their own documented file base.

        Args:
            params: Canonical output option names after aliases and overrides.
                Format and filename-convention values are case-insensitive.
            multisector: Whether to enforce the multisector output restrictions.

        Returns:
            Validated output choices ready for inspection before execution.

        Raises:
            ValueError: If format, path or recipe choices are inconsistent.
        """
        options = dict(params)
        active_format = options.get("output_format", cls.output_format).lower()
        options.setdefault("save_inversion_output", active_format == "inv_out")
        return make_output_spec(multisector=multisector, **options)

    @field_validator("output_path", "country_file", "save_trace", "save_inversion_output", mode="before")
    @classmethod
    def resolve_paths(cls, value: Any) -> Any:
        """Accept explicit path strings at this configuration boundary."""
        return Path(value).expanduser().resolve() if isinstance(value, str) else value

    output_format: OutputFormat = "inv_out"
    output_path: Path | None = None
    output_name: str = "rhime"
    save_trace: Path | bool = False
    save_inversion_output: Path | bool = True
    country_file: Path | None = None
    paris_postprocessing_kwargs: BorrowedOptions | None = None
    output_filename_convention: OutputFilenameConvention = "rhime"


def validate_output_format(output_format: str) -> None:
    """Validate a RHIME output-format name.

    Args:
        output_format: Requested output-format name.

    Raises:
        ValueError: If the format is unsupported.
    """
    valid_formats = {"none", "inv_out", "basic", "paris", "legacy"}
    if output_format not in valid_formats:
        raise ValueError(
            f"Unsupported RHIME output_format {output_format!r}; expected one of {sorted(valid_formats)!r}."
        )


def validate_output_path_settings(
    *,
    output_format: str,
    output_path: str | Path | None,
    save_trace: str | Path | bool,
    save_inversion_output: str | Path | bool,
    multisector: bool,
) -> None:
    """Validate output paths and single- versus multisector restrictions.

    Args:
        output_format: Requested output format.
        output_path: Optional default output directory.
        save_trace: Trace-save setting or explicit path.
        save_inversion_output: Inversion-output save setting or explicit path.
        multisector: Whether the run is multisector.

    Raises:
        ValueError: If the format is incompatible with the model or a required
            default output directory is absent.
    """
    if multisector and output_format in ("basic", "legacy"):
        raise ValueError(
            f"RHIME output_format {output_format!r} supports only single-sector runs."
        )
    if output_format == "none":
        return
    if output_path is not None:
        return
    if save_trace is True:
        raise ValueError("`output_path` is required when `save_trace=True`.")
    if save_inversion_output is True:
        raise ValueError("`output_path` is required when saving the RHIME InversionOutput.")


def validate_output_filename_convention(output_filename_convention: str) -> None:
    """Validate an output filename convention.

    Args:
        output_filename_convention: Requested filename convention.

    Raises:
        ValueError: If the convention is unsupported.
    """
    valid_conventions = {"rhime", "legacy"}
    if output_filename_convention not in valid_conventions:
        raise ValueError(
            "Unsupported RHIME output_filename_convention "
            f"{output_filename_convention!r}; expected one of {sorted(valid_conventions)!r}."
        )


def make_output_spec(
    *,
    output_format: str = RhimeOutputSpec.output_format,
    output_path: str | Path | None = RhimeOutputSpec.output_path,
    output_name: str = RhimeOutputSpec.output_name,
    save_trace: str | Path | bool = RhimeOutputSpec.save_trace,
    save_inversion_output: str | Path | bool = RhimeOutputSpec.save_inversion_output,
    country_file: str | Path | None = RhimeOutputSpec.country_file,
    paris_postprocessing_kwargs: dict[str, Any] | None = RhimeOutputSpec.paris_postprocessing_kwargs,
    output_filename_convention: str = RhimeOutputSpec.output_filename_convention,
    multisector: bool,
) -> RhimeOutputSpec:
    """Create validated output settings from normalized RHIME parameters.

    Args:
        output_format: Requested product format.
        output_path: Optional default output directory.
        output_name: Base name for generated artifacts.
        save_trace: Trace-save setting or explicit path.
        save_inversion_output: Inversion-output save setting or explicit path.
        country_file: Optional country mask for derived products.
        paris_postprocessing_kwargs: Optional PARIS product settings.
        output_filename_convention: Naming convention for derived files.
        multisector: Whether the run is multisector.

    Returns:
        Validated immutable output specification.

    Raises:
        ValueError: If formats, paths, or model restrictions are inconsistent.
    """
    output_format = output_format.lower()
    output_filename_convention = output_filename_convention.lower()
    validate_output_format(output_format)
    validate_output_filename_convention(output_filename_convention)
    validate_output_path_settings(
        output_format=output_format,
        output_path=output_path,
        save_trace=save_trace,
        save_inversion_output=save_inversion_output,
        multisector=multisector,
    )
    options = dict(
        output_format=cast(OutputFormat, output_format),
        output_path=output_path,
        output_name=output_name,
        save_trace=save_trace,
        save_inversion_output=save_inversion_output,
        country_file=country_file,
        paris_postprocessing_kwargs=paris_postprocessing_kwargs,
        output_filename_convention=cast(OutputFilenameConvention, output_filename_convention),
    )
    return RhimeOutputSpec(**resolve_resource_paths(options, annotations=get_type_hints(RhimeOutputSpec), base_dir=Path.cwd()))
