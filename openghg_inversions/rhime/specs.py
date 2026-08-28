"""Lightweight immutable specifications for RHIME models and runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.priors import PriorArgs
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
DEFAULT_POLLUTION_EVENT_SIGMA_PRIOR: PriorArgs = {
    "pdf": "uniform",
    "lower": 0.0,
    "upper": 0.1,
}
DEFAULT_OFFSET_PRIOR: PriorArgs = {"pdf": "normal", "mu": 0, "sigma": 1}


@dataclass(frozen=True)
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

    sigma_prior: PriorArgs | None = None
    sigma_freq: str | None = None
    sigma_per_site: bool = True
    sigma_freq_anchor: DatetimeLike | None = None
    pollution_events_from_obs: bool = False
    power: PriorArgs | float = 1.99

    @property
    def required_prepared_inputs(self) -> tuple[str, ...]:
        """Return prepared arrays owned by this likelihood."""
        return ("min_error",)


@dataclass(frozen=True)
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

    sigma_prior: PriorArgs | None = None
    sigma_freq: str | None = None
    sigma_per_site: bool = True
    sigma_freq_anchor: DatetimeLike | None = None
    use_minimum_error_floor: bool = False

    @property
    def required_prepared_inputs(self) -> tuple[str, ...]:
        """Return prepared arrays owned by this likelihood."""
        return ("min_error",) if self.use_minimum_error_floor else ()


@dataclass(frozen=True)
class FixedErrorSettings:
    """Serializable selection of reported observation error only."""

    @property
    def required_prepared_inputs(self) -> tuple[str, ...]:
        """Return prepared arrays owned by this likelihood."""
        return ()


LikelihoodSettings = PollutionEventSettings | AdditiveSigmaSettings | FixedErrorSettings


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

    output_format: OutputFormat = "inv_out"
    output_path: str | None = None
    output_name: str = "rhime"
    save_trace: str | Path | bool = False
    save_inversion_output: str | Path | bool = True
    country_file: str | None = None
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
    output_path: str | None,
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
    output_format: str,
    output_path: str | None,
    output_name: str,
    save_trace: str | Path | bool,
    save_inversion_output: str | Path | bool,
    country_file: str | None,
    paris_postprocessing_kwargs: dict[str, Any] | None,
    output_filename_convention: str,
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
    return RhimeOutputSpec(
        output_format=cast(OutputFormat, output_format),
        output_path=output_path,
        output_name=output_name,
        save_trace=save_trace,
        save_inversion_output=save_inversion_output,
        country_file=country_file,
        paris_postprocessing_kwargs=paris_postprocessing_kwargs,
        output_filename_convention=cast(OutputFilenameConvention, output_filename_convention),
    )
