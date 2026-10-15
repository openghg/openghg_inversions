"""Lightweight immutable specifications for RHIME models and runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Literal, cast, get_type_hints

from openghg_inversions.rhime._model_records import ModelRecord
from openghg_inversions.rhime._resources import resolve_resource_paths
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models._likelihood_settings import LikelihoodSettings as _LikelihoodSettings
from openghg_inversions.models.additive_sigma import AdditiveSigmaSettings
from openghg_inversions.models.fixed_error import FixedErrorSettings
from openghg_inversions.models.pollution_event import (
    DEFAULT_POLLUTION_EVENT_SIGMA_PRIOR as DEFAULT_POLLUTION_EVENT_SIGMA_PRIOR,
    PollutionEventSettings,
)
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import AggregationErrorMode

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
DEFAULT_OFFSET_PRIOR: PriorArgs = {"pdf": "normal", "mu": 0, "sigma": 1}


_copy_option_containers = _LikelihoodSettings.own_containers


def normalise_optional_mapping(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Own an optional mapping and its ordinary configuration containers."""
    return None if value is None else _LikelihoodSettings.own_containers(dict(value))


LikelihoodSettings = PollutionEventSettings | AdditiveSigmaSettings | FixedErrorSettings
LIKELIHOOD_OPTION_NAMES = _LikelihoodSettings.option_names()
LIKELIHOOD_PRIOR_OPTION_NAMES = ("sigma_prior",)
make_likelihood_settings = _LikelihoodSettings.from_params


@dataclass(frozen=True)
class SectorSpec:
    """Configuration for one separately optimised flux sector.

    Args:
        name: User-facing sector name.
        flux_source: OpenGHG flux ``source`` used to retrieve this sector.
        x_prior: Prior specification for this sector's flux scaling factors.
        variable_suffix: PyMC-safe suffix used in multi-sector model variable
            names. Standard single-sector RHIME uses plain ``x``/``mu`` names.
        state_activity: Optional labelled active/fixed policy for this sector's
            flux-scaling states. ``None`` still applies the default flux policy,
            fixing exactly-zero sensitivity columns to one.
    """

    name: str
    flux_source: str
    x_prior: PriorArgs
    variable_suffix: str
    state_activity: StateActivity | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
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

    species: str
    domain: str
    sectors: tuple[SectorSpec, ...]
    use_bc: bool = True
    likelihood: LikelihoodSettings | None = field(default=None, kw_only=True)
    add_offset: bool = False
    aggregation_error_mode: AggregationErrorMode = field(default="none", kw_only=True)
    bc_prior: PriorArgs | None = None
    offset_prior: PriorArgs | None = None
    offset_args: dict[str, Any] | None = None
    bc_state_activity: StateActivity | None = field(default=None, kw_only=True)
    state_activity: StateActivity | None = field(default=None, kw_only=True)

    def to_record(self, *, priors: Mapping[str, PriorArgs] | None = None) -> dict[str, Any]:
        """Describe structural choices and construction-owned prior parameters.

        ``priors`` contains nullable scalar values, registered-array references
        and named or opaque symbolic descriptions captured during construction.
        Symbolic descriptions are provenance; recording them does not evaluate
        or reconstruct their dependencies. Unused configured prior mappings
        are never traversed or materialized here. Omitting ``priors`` omits
        prior parameter values from the record.
        """
        return ModelRecord.from_runtime(self, priors=dict(priors or {})).model_dump(mode="json")

    @classmethod
    def from_record(cls, record: Any, *, trace: Any = None) -> RhimeModelSpec:
        """Restore scalar and registered-array priors without executing saved callables.

        Named and unnamed symbolic descriptions need the original model recipe
        for executable restoration. Saved posterior products instead use the
        metadata view from :meth:`ModelRecord.for_postprocessing`.

        Args:
            record: Structural model choices and construction-owned prior records.
            trace: Saved inference data supplying any referenced ``constant_data``
                arrays. Scalar-only prior records need no trace lookup.

        Raises:
            ValueError: If the record is incomplete, names an unknown likelihood,
                references a missing registered prior array, or contains a
                symbolic parameter that cannot be restored as executable input.
        """
        return ModelRecord.model_validate(record).to_runtime(trace=trace)


# Kept for the field-coverage check; the wire schema is the single field list.
_MODEL_RECORD_FIELDS = tuple(name for name in ModelRecord.model_fields if name not in {"sectors", "likelihood"})


@dataclass(frozen=True)
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
        if "paris_postprocessing_kwargs" in options:
            options["paris_postprocessing_kwargs"] = normalise_optional_mapping(options["paris_postprocessing_kwargs"])
        return make_output_spec(multisector=multisector, **options)

    output_format: OutputFormat = "inv_out"
    output_path: Path | None = None
    output_name: str = "rhime"
    save_trace: Path | bool = False
    save_inversion_output: Path | bool = True
    country_file: Path | None = None
    paris_postprocessing_kwargs: dict[str, Any] | None = None
    output_filename_convention: OutputFilenameConvention = "rhime"


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
        split_by_sectors: Whether flux data were prepared in sector-resolved
            mode. Single-sector and multi-sector RHIME are runner/model modes;
            this flag records the prepared data layout.
    """

    start_date: str
    end_date: str
    sites: tuple[str, ...]
    averaging_period: tuple[str | None, ...]
    model: RhimeModelSpec
    output: RhimeOutputSpec
    split_by_sectors: bool = False


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
