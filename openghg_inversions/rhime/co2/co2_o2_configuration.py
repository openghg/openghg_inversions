"""Independent configuration owners for the staged linked CO2/O2 recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from openghg_inversions.models import StateActivity
from openghg_inversions.boundary_sensitivity import BoundaryRepresentation
from openghg_inversions.models.boundary import Choice, MultiplicativeChoice, boundary_preparation
from openghg_inversions.models.boundary.preparation import check_representation
from openghg_inversions.models.outer_flux import OuterFluxChoice
from openghg_inversions.inversion_inputs import normalise_frequency
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR, normalise_optional_mapping

from .co2_o2_preparation import Co2O2PreparedInputs
from .configuration import (
    _validate_section_option_names, _SAMPLING_OPTION_NAMES,
    _ACQUISITION_OPTION_NAMES, _BASIS_OPTION_NAMES, _NATIVE_CALIBRATION_OPTION_NAMES,
    _NATIVE_PRIOR_OPTION_NAMES, _MULTISOURCE_NATIVE_PRIOR_OPTION_NAMES,
    _PRIOR_OPTION_NAMES, _MODEL_SECTION_OPTION_NAMES,
    _reject_unknown,
    _bool, _number, _path, _take, _model, _co2_likelihood, _frozen,
    _acquisition, _boundary_choice, _native_prior,
    Co2BasisSelection, Co2MultiSourceNativePrior,
    _sampling,
    _string,
    _table, _outer_flux_choice,
    load_co2_family_config,
    resolve_co2_prior_seed,
)


def load_co2_o2_params(
    *, config_file: str | Path | None = None, params_file: str | Path | None = None,
    params: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one TOML, JSON or Python request without resolving phase choices.

    Headers select format version 1 and recipe ``co2_o2`` when omitted. There
    is no inference-implementation selector. Overrides update owned sections,
    including individual channel choices, before their phase resolves them.
    Paths belong to their consuming phase and use the invocation directory.

    Raises:
        ValueError: If the source is ambiguous, JSON is not an object, the
            headers are unsupported or a top-level option is unknown.
    """
    if sum(value is not None for value in (config_file, params_file, params)) != 1:
        raise ValueError("Pass exactly one of config_file, params_file or params.")
    if config_file is not None:
        options = dict(load_co2_family_config(config_file))
    elif params_file is not None:
        options = json.loads(Path(params_file).read_text(encoding="utf-8"))
        if not isinstance(options, dict):
            raise ValueError("Linked CO2/O2 params file must contain one JSON object.")
    else:
        options = _table(params, "config")
    _validate_co2_o2_option_names(options, overrides or {})
    for name, value in (overrides or {}).items():
        if name in {"prepared_inputs", "channels", "likelihood", "sampling", "outputs"} and isinstance(value, Mapping):
            previous = _table(options.get(name, {}), name)
            if name == "channels":
                for channel, choices in value.items():
                    if isinstance(choices, Mapping):
                        previous[channel] = {**_table(previous.get(channel, {}), f"channels.{channel}"), **choices}
                    else:
                        previous[channel] = choices
                options[name] = previous
            else:
                options[name] = {**previous, **value}
        else:
            options[name] = value
    if type(options.get("format_version", 1)) is not int or options.get("format_version", 1) != 1:
        raise ValueError("config.format_version must be the integer 1.")
    if options.get("recipe", "co2_o2") != "co2_o2":
        raise ValueError("config.recipe must be 'co2_o2' for linked execution.")
    if "variant" in options:
        raise ValueError("Linked CO2/O2 configuration has no variant selector; select the scientific likelihood.")
    return options


def _validate_co2_o2_option_names(*requests: Mapping[str, object]) -> None:
    """Check request names while deferring scientific and inactive values."""
    allowed = {
        "format_version", "recipe", "variant", "prepared_inputs", "channels", "outer_flux", "likelihood",
        "sampling", "outputs",
    }
    from .co2_o2_outputs import Co2O2OutputOptions

    channel_sections = {
        "": frozenset({"units", "independent_error_sd", "boundary", "offset"}),
        **{name: options for name, options in _MODEL_SECTION_OPTION_NAMES.items() if name != "outer_flux"},
        "boundary": _MODEL_SECTION_OPTION_NAMES["boundary"] | {"activity"},
        "boundary.activity": frozenset({"active", "fixed_value"}),
    }
    _validate_section_option_names(requests, {
        "": frozenset(allowed),
        "prepared_inputs": _LINKED_PREPARATION_OPTION_NAMES,
        "prepared_inputs.acquisition": frozenset(LINKED_CHANNELS),
        **{f"prepared_inputs.acquisition.{channel}": _ACQUISITION_OPTION_NAMES for channel in LINKED_CHANNELS},
        "prepared_inputs.observations": frozenset(LINKED_CHANNELS),
        **{f"prepared_inputs.observations.{channel}": frozenset({"filters", "averaging_error"})
           for channel in LINKED_CHANNELS},
        "prepared_inputs.basis": _BASIS_OPTION_NAMES,
        "prepared_inputs.native_prior": _NATIVE_PRIOR_OPTION_NAMES | _MULTISOURCE_NATIVE_PRIOR_OPTION_NAMES,
        "prepared_inputs.native_prior.calibration": _NATIVE_CALIBRATION_OPTION_NAMES,
        "outer_flux": _MODEL_SECTION_OPTION_NAMES["outer_flux"],
        "channels": frozenset({"co2", "o2"}),
        **{f"channels.{channel}" + (f".{path}" if path else ""): names
           for channel in ("co2", "o2") for path, names in channel_sections.items()},
        "likelihood": frozenset({"kind", "tau_hours", "fixed_site_amplitudes", "site_amplitude_prior",
                                 "site_amplitude_prior_scale"}),
        "likelihood.site_amplitude_prior": _PRIOR_OPTION_NAMES,
        "sampling": _SAMPLING_OPTION_NAMES,
        "outputs": frozenset(Co2O2OutputOptions.__dataclass_fields__),
        "outputs.paris_postprocessing_kwargs": frozenset({
            "template_version", "obs_avg_period", "flux_frequency", "country_selections",
        }),
        "outputs.native_flux_bases": frozenset({"co2", "o2"}),
    })


LINKED_CHANNELS = ("co2", "o2")
"""Tracer channels of the linked recipe, in observation order."""

# A saved linked preparation is imported with ``path``; a new one is built from these tables.
_LINKED_NATIVE_OPTION_NAMES = frozenset({
    "acquisition", "observations", "basis", "native_prior", "source_states", "bc_centering_weights_path",
})
_LINKED_PREPARATION_OPTION_NAMES = frozenset({
    "path", "start_date", "end_date", "domain", "boundary_representation", "reconstruction_path",
    "aggregation_error_rank",
}) | _LINKED_NATIVE_OPTION_NAMES
_SOURCE_STATE_OPTION_NAMES = frozenset({
    "source", "tracer_scope", "co2_flux_source", "o2_flux_source", "flux_tracer",
})
_TRACER_SCOPES = ("shared", *LINKED_CHANNELS)


@dataclass(frozen=True)
class Co2O2SourceState:
    """How one native source enters the linked preparation.

    Attributes:
        source: Retained sector label (``"gpp"``, ``"ter"``, ``"ff"`` or
            ``"ocean"``, any case).
        tracer_scope: ``"shared"`` (one scaling field for both tracers'
            fluxes) or the one tracer, ``"co2"`` or ``"o2"``.
        co2_flux_source: Acquired CO2 flux source whose response this native
            source scales; required unless the scope is ``"o2"``.
        o2_flux_source: Acquired O2 flux source (signed O2 flux) whose
            response it scales; required unless the scope is ``"co2"``.
        flux_tracer: Tracer whose prior flux weights a shared source in the
            absolute-flux-weighted restriction and in amplitude calibration.
            Required for a shared source exactly when one of those uses it.
    """

    source: str
    tracer_scope: str
    co2_flux_source: str | None = None
    o2_flux_source: str | None = None
    flux_tracer: str | None = None

    @classmethod
    def from_params(cls, value: object, path: str) -> Co2O2SourceState:
        """Parse one ``[prepared_inputs.source_states.<source>]`` table."""
        options = _table(value, path)
        result: dict[str, Any] = {
            "source": _string(_take(options, "source", path), f"{path}.source"),
            "tracer_scope": _string(_take(options, "tracer_scope", path), f"{path}.tracer_scope"),
        }
        if result["tracer_scope"] not in _TRACER_SCOPES:
            raise ValueError(f"{path}.tracer_scope must be one of {_TRACER_SCOPES!r}.")
        for name in ("co2_flux_source", "o2_flux_source", "flux_tracer"):
            if name in options:
                result[name] = _string(options.pop(name), f"{path}.{name}")
        _reject_unknown(options, path)
        state = cls(**result)
        for channel in LINKED_CHANNELS:
            given = getattr(state, f"{channel}_flux_source") is not None
            needed = state.tracer_scope in ("shared", channel)
            if given != needed:
                raise ValueError(
                    f"{path}.{channel}_flux_source is {'required' if needed else 'not allowed'} for "
                    f"tracer_scope = {state.tracer_scope!r}."
                )
        if state.flux_tracer is not None:
            if state.tracer_scope != "shared":
                raise ValueError(f"{path}.flux_tracer applies only to shared sources; a {state.tracer_scope} "
                                 "source is weighted by its own tracer's flux.")
            if state.flux_tracer not in LINKED_CHANNELS:
                raise ValueError(f"{path}.flux_tracer must be one of {LINKED_CHANNELS!r}.")
        return state

    def to_params(self) -> dict[str, Any]:
        """Return the table :meth:`from_params` reads back."""
        return {item.name: getattr(self, item.name) for item in fields(self) if getattr(self, item.name) is not None}


@dataclass(frozen=True)
class Co2O2NativePreparationConfig:
    """Acquisition and native choices for a new linked CO2/O2 preparation.

    Each tracer has its own complete acquisition and observation choices;
    nothing is inherited between them. The per-source native priors are those
    of the configured CO2 route, keyed by native source, and
    ``source_states`` ties every native source to the acquired flux sources it
    scales. See
    :func:`~openghg_inversions.rhime.co2.co2_o2_native_preparation.prepare_native_co2_o2`.

    Attributes:
        acquisition: Per-channel keywords for
            :meth:`~openghg_inversions.inversion_data.MergedData.from_options`,
            other than species, domain and window.
        filters: Per-channel observation filters (``None`` for none).
        averaging_error: Per-channel choice to include variability in the
            derived observation errors.
        basis: Basis selection; ``partition_path`` holds one integer partition
            per native source on a ``source`` dimension.
        native_prior: Per-source native priors keyed by native source.
        source_states: How each native source maps to flux sources, tracers
            and retained labels.
        bc_centering_weights_path: Optional netCDF holding one
            ``bc_centering_weights`` array for weighted additive boundaries.
    """

    acquisition: Mapping[str, Mapping[str, Any]]
    basis: Co2BasisSelection
    native_prior: Co2MultiSourceNativePrior
    source_states: Mapping[str, Co2O2SourceState]
    filters: Mapping[str, Any] = field(default_factory=dict)
    averaging_error: Mapping[str, bool] = field(default_factory=dict)
    bc_centering_weights_path: Path | None = None

    @classmethod
    def from_params(cls, prepared: dict[str, object], *, domain: str) -> Co2O2NativePreparationConfig:
        """Pop and parse the build-route tables of a linked ``[prepared_inputs]``."""
        path = "prepared_inputs"
        if "aggregation_error_rank" in prepared:
            raise ValueError(
                f"{path}.aggregation_error_rank is not available for linked preparations, which keep the "
                "dense joint aggregation error (low-rank linked A is #1079)."
            )
        tables = _table(_take(prepared, "acquisition", path), f"{path}.acquisition")
        acquisition = {
            channel: _acquisition(
                _take(tables, channel, f"{path}.acquisition"), f"{path}.acquisition.{channel}", species=channel,
            )
            for channel in LINKED_CHANNELS
        }
        _reject_unknown(tables, f"{path}.acquisition")
        prior = _native_prior(_take(prepared, "native_prior", path))
        if not isinstance(prior, Co2MultiSourceNativePrior):
            raise ValueError(f"Linked preparation requires per-source priors in {path}.native_prior.sources.")
        states_table = _table(_take(prepared, "source_states", path), f"{path}.source_states")
        states = {
            _string(name, f"{path}.source_states key"): Co2O2SourceState.from_params(
                value, f"{path}.source_states.{name}",
            )
            for name, value in states_table.items()
        }
        if set(states) != set(prior.sources):
            raise ValueError(
                f"{path}.source_states must name exactly the native sources of {path}.native_prior.sources "
                f"{sorted(prior.sources)!r}; got {sorted(states)!r}."
            )
        for channel in LINKED_CHANNELS:
            mapped = [state.co2_flux_source if channel == "co2" else state.o2_flux_source for state in states.values()]
            mapped = [name for name in mapped if name is not None]
            acquired = list(acquisition[channel]["flux_sources"])
            if sorted(mapped) != sorted(acquired) or len(set(mapped)) != len(mapped):
                raise ValueError(
                    f"{path}.source_states must map each {channel} flux source in {path}.acquisition.{channel}."
                    f"flux_sources {sorted(acquired)!r} exactly once; got {sorted(mapped)!r}."
                )
        weighted = prior.restriction == "absolute_flux_weighted_mean"
        for name, state in states.items():
            calibrated = prior.calibration is not None and prior.sources[name].sigma is None
            if state.tracer_scope == "shared" and (weighted or calibrated) and state.flux_tracer is None:
                raise ValueError(
                    f"{path}.source_states.{name}.flux_tracer is required: the "
                    f"{'absolute-flux-weighted restriction' if weighted else 'amplitude calibration'} weights "
                    "this shared source by one tracer's prior flux; set 'co2' or 'o2'."
                )
        use_bc = any(table.get("use_bc", True) for table in acquisition.values())
        basis = Co2BasisSelection.from_params(prepared.pop("basis", {}), domain=domain, use_bc=use_bc)
        if basis.basis_partition_path is None or basis.basis_path is not None or basis.fix_basis_outer_regions:
            raise ValueError(
                f"Linked preparation requires {path}.basis.partition_path, with one partition per native source "
                "on a source dimension; basis paths, generated bases and fix_outer_regions are not supported."
            )
        result: dict[str, Any] = {
            "acquisition": _frozen({channel: _frozen(value) for channel, value in acquisition.items()}),
            "basis": basis, "native_prior": prior, "source_states": _frozen(states),
        }
        observations = _table(prepared.pop("observations", {}), f"{path}.observations")
        filters: dict[str, Any] = {}
        averaging: dict[str, bool] = {}
        for channel in LINKED_CHANNELS:
            table = _table(observations.pop(channel, {}), f"{path}.observations.{channel}")
            if "filters" in table:
                value = table.pop("filters")
                if isinstance(value, Mapping):
                    value = {str(site): item for site, item in value.items()}
                elif not isinstance(value, str | list):
                    raise TypeError(f"{path}.observations.{channel}.filters must be a name, list or table.")
                filters[channel] = value
            if "averaging_error" in table:
                averaging[channel] = _bool(table.pop("averaging_error"), f"{path}.observations.{channel}.averaging_error")
            _reject_unknown(table, f"{path}.observations.{channel}")
        _reject_unknown(observations, f"{path}.observations")
        result["filters"] = _frozen(filters)
        result["averaging_error"] = _frozen(averaging)
        if "bc_centering_weights_path" in prepared:
            result["bc_centering_weights_path"] = _path(
                prepared.pop("bc_centering_weights_path"), f"{path}.bc_centering_weights_path",
            )
        return cls(**result)

    def flux_sources(self, channel: str) -> dict[str, str]:
        """Map each of ``channel``'s acquired flux sources to its native source."""
        name = f"{channel}_flux_source"
        return {getattr(state, name): native for native, state in self.source_states.items()
                if getattr(state, name) is not None}

    def resource_paths(self) -> tuple[Path, ...]:
        """Identify the configured files native preparation reads."""
        basis, prior = self.basis, self.native_prior
        calibration = prior.calibration.targets_path if prior.calibration is not None else None
        paths = (basis.basis_partition_path, basis.landsea_path, basis.outer_regions_path,
                 *(basis.bc_basis_paths or ()), self.bc_centering_weights_path,
                 prior.class_labels_path, prior.retained_regions_path, calibration)
        return tuple(Path(item).resolve() for item in paths if item is not None)


@dataclass(frozen=True)
class Co2O2PreparationConfig:
    """Import a saved linked artifact, or build one from configured native choices.

    ``[prepared_inputs]`` either names a saved artifact with ``path`` or, with
    ``acquisition`` tables, describes a new preparation (``native``). Each
    channel's prepared boundary representation is resolved from its
    ``[channels.<channel>.boundary]`` choice, or the explicit
    ``boundary_representation``, as on the CO2 route. Only the channel
    boundary tables are read from ``[channels]``.
    """

    prepared_path: Path | None = None
    start_date: str | None = None
    end_date: str | None = None
    domain: str | None = None
    boundary_representation: BoundaryRepresentation | None = None
    boundary: Mapping[str, Choice | None] = field(default_factory=dict)
    native: Co2O2NativePreparationConfig | None = None

    @classmethod
    def from_params(cls, config: Mapping[str, object], *, require_source: bool = False) -> Co2O2PreparationConfig:
        """Parse linked prepared-input options and the channel boundary choices they serve.

        Args:
            config: Complete linked request.
            require_source: Require ``path`` or ``acquisition`` tables.

        Raises:
            ValueError: If options are missing, unknown or conflicting.
        """
        options = _table(config, "config")
        prepared = _table(options.get("prepared_inputs", {}), "prepared_inputs")
        if "reconstruction_path" in prepared:
            raise ValueError("Linked CO2/O2 preparation does not support prepared_inputs.reconstruction_path.")
        native = bool(_LINKED_NATIVE_OPTION_NAMES & set(prepared)) or "aggregation_error_rank" in prepared
        if native and "path" in prepared:
            raise ValueError("prepared_inputs.path imports a saved preparation; remove the build options.")
        if require_source and not native and "path" not in prepared:
            raise ValueError(
                "Missing required option prepared_inputs.path, or prepared_inputs.acquisition to build one."
            )
        result: dict[str, Any] = {}
        if "path" in prepared:
            result["prepared_path"] = _path(prepared.pop("path"), "prepared_inputs.path")
        for name in ("start_date", "end_date", "domain"):
            if name in prepared:
                result[name] = _string(prepared.pop(name), f"prepared_inputs.{name}")
        if "boundary_representation" in prepared:
            result["boundary_representation"] = check_representation(prepared.pop("boundary_representation"))
        channels = options.get("channels", {})
        boundary: dict[str, Choice | None] = {}
        if isinstance(channels, Mapping):
            for channel in LINKED_CHANNELS:
                table = channels.get(channel)
                if isinstance(table, Mapping) and "boundary" in table:
                    choice = {key: value for key, value in _table(table["boundary"], f"channels.{channel}.boundary").items()
                              if key != "activity"}
                    boundary[channel] = _boundary_choice(choice, f"channels.{channel}.boundary")
        result["boundary"] = _frozen(boundary)
        if native:
            for name in ("start_date", "end_date", "domain"):
                if name not in result:
                    raise ValueError(f"Missing required option prepared_inputs.{name} for a new preparation.")
            result["native"] = Co2O2NativePreparationConfig.from_params(prepared, domain=result["domain"])
            representation = result.get("boundary_representation")
            needs = {channel: boundary_preparation(boundary.get(channel), representation) for channel in LINKED_CHANNELS}
            weights = result["native"].bc_centering_weights_path
            if any(need.centering_weights for need in needs.values()) and weights is None:
                raise ValueError("The weighted additive boundary requires prepared_inputs.bc_centering_weights_path.")
            if weights is not None and not any(need.representation == "centered" for need in needs.values()):
                raise ValueError("prepared_inputs.bc_centering_weights_path requires centered boundaries.")
            for channel in LINKED_CHANNELS:
                if not result["native"].acquisition[channel].get("use_bc", True) and boundary.get(channel) is not None:
                    raise ValueError(
                        f"prepared_inputs.acquisition.{channel}.use_bc = false cannot serve "
                        f"channels.{channel}.boundary; enable use_bc or disable that boundary."
                    )
        _reject_unknown(prepared, "prepared_inputs")
        return cls(**result)

    def preparation_arguments(self) -> dict[str, Any]:
        """Return keywords for the linked prepare boundary."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


def validate_co2_o2_units(prepared: Co2O2PreparedInputs) -> str:
    """Require both prepared channels to use one supported mole-fraction unit.

    Reads the observation-aligned species and unit coordinates, leaving the
    borrowed artifact unchanged. This staged contract excludes heterogeneous
    ppm/per-meg operation until its scientific transform is defined.

    Returns:
        The common channel unit label.

    Raises:
        ValueError: If either channel or coordinate is missing, unit labels
            differ, or the common unit cannot be converted to mol/mol.
    """
    observations = prepared.observations
    for name in ("species", "observation_units"):
        if name not in observations.coords or observations[name].dims != ("observation",):
            raise ValueError(f"Linked prepared observations require observation-aligned {name}.")
    if set(observations.species.values.tolist()) != {"co2", "o2"}:
        raise ValueError("Linked prepared observations must contain exactly CO2 and O2 channels.")
    units = np.unique([str(unit) for unit in observations.observation_units.values])
    if units.size != 1:
        raise ValueError("Linked staged inputs require identical CO2 and O2 units (OPE-86).")
    unit = _string(str(units[0]), "prepared observation_units")
    if "meg" in unit.casefold() or "delta" in unit.casefold():
        raise ValueError("Linked staged inputs require mole-fraction units; per-meg/delta O2 requires OPE-86.")
    mole_fraction_unit_scale(unit, context="linked prepared observation_units")
    return unit


def _likelihood(config: Mapping[str, object]) -> dict[str, object] | None:
    options = _table(config.get("likelihood", {}), "likelihood")
    kind = options.get("kind", "fixed_independent")
    if kind == "fixed_independent":
        options.pop("kind", None)
        _reject_unknown(options, "likelihood")
        return None
    if kind != "fixed_ou":
        raise ValueError("Linked likelihood.kind must be 'fixed_independent' or 'fixed_ou'.")
    if "site_amplitude_prior" in options or "site_amplitude_prior_scale" in options:
        raise ValueError("Configured linked inferred OU amplitudes are not supported yet; use fixed_site_amplitudes.")
    if "fixed_site_amplitudes" not in options:
        raise ValueError("Configured linked fixed_ou requires fixed_site_amplitudes; inferred OU is not supported yet.")
    return options


@dataclass(frozen=True)
class Co2O2ModelConfig:
    """Linked channel science independent of prepared arrays and sampling.

    Both channels require identical mole-fraction units. Error overrides must
    be supplied for both channels or omitted for both to use saved errors.
    Fixed independent error and fixed-amplitude OU are supported; inferred OU
    and per-meg transformations remain separate work.
    ``boundary`` maps configured channels to a choice, or to ``None`` when
    that channel's boundary is explicitly disabled. Boundary frequency and
    anchor choices apply to every enabled channel; choices supplied under both
    channels must agree. :meth:`bind` gives channels without a boundary table
    the default multiplicative prior when the preparation has their transport.
    """

    co2_units: str
    o2_units: str
    independent_error_sd: Mapping[str, float]
    boundary: Mapping[str, Choice | None] | None = None
    offset_prior: Mapping[str, Any] | None = None
    offset_args: Mapping[str, Any] | None = None
    tau_hours: float | Mapping[str, float] | None = None
    fixed_site_amplitudes: float | Mapping[str, float] | None = None
    outer_flux: OuterFluxChoice | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Mapping):
                object.__setattr__(self, item.name, normalise_optional_mapping(value))
        if self.boundary is not None:
            # The single owner of the one-calendar rule for parsed and direct construction.
            object.__setattr__(self, "boundary", normalise_optional_mapping(_shared_boundary_periods(self.boundary)))

    @classmethod
    def from_params(cls, config: Mapping[str, object]) -> Co2O2ModelConfig:
        """Parse active channel/likelihood choices without touching arrays."""
        options = _table(config, "config")
        channels = _table(options.get("channels", {}), "channels")
        units: dict[str, str] = {}
        errors: dict[str, float] = {}
        model: dict[str, Any] = {}
        if "outer_flux" in options:
            model["outer_flux"] = _outer_flux_choice(options["outer_flux"], "outer_flux")
        for name in ("co2", "o2"):
            channel = _table(_take(channels, name, "channels"), f"channels.{name}")
            units[name] = _string(_take(channel, "units", f"channels.{name}"), f"channels.{name}.units")
            mole_fraction_unit_scale(units[name], context=f"channels.{name}.units")
            if "independent_error_sd" in channel:
                errors[name] = _number(channel.pop("independent_error_sd"), f"channels.{name}.independent_error_sd", positive=True)
            selected = {key: channel.pop(key) for key in ("boundary", "offset") if key in channel}
            activity = None
            if "boundary" in selected:
                boundary = _table(selected["boundary"], f"channels.{name}.boundary")
                if "activity" in boundary:
                    table = _table(boundary.pop("activity"), f"channels.{name}.boundary.activity")
                    active = _bool(table.pop("active", True), f"channels.{name}.boundary.activity.active")
                    fixed = _number(table.pop("fixed_value", 1.0), f"channels.{name}.boundary.activity.fixed_value")
                    _reject_unknown(table, f"channels.{name}.boundary.activity")
                    activity = StateActivity(active=active, fixed_value=fixed)
                selected["boundary"] = boundary
            for key, value in _model(selected, f"channels.{name}").items():
                if key == "boundary" and activity is not None:
                    if not isinstance(value, MultiplicativeChoice):
                        raise ValueError("Boundary activity requires an enabled multiplicative boundary.")
                    value = replace(value, activity=activity)
                model.setdefault(key, {})[name] = value
            _reject_unknown(channel, f"channels.{name}")
        _reject_unknown(channels, "channels")
        if errors and len(errors) != 2:
            raise ValueError("Configure independent_error_sd for both linked channels or omit it for both.")
        if units["co2"] != units["o2"]:
            raise ValueError("The linked configuration currently requires identical CO2 and O2 channel units.")
        likelihood = _likelihood(options)
        kwargs = {} if likelihood is None else _co2_likelihood(likelihood)["likelihood_kwargs"]
        return cls(units["co2"], units["o2"], _frozen(errors), **_frozen(model), **kwargs)

    def bind(self, prepared: Co2O2PreparedInputs) -> dict[str, Any]:
        """Bind channel errors and check units at the scientific array boundary.

        The artifact and its saved error vector remain borrowed. Explicit errors
        expand over labelled observations; agreement with saved errors is checked
        by model construction at its materialization boundary.
        """
        unit = validate_co2_o2_units(prepared)
        if unit != self.co2_units or unit != self.o2_units:
            raise ValueError("Configured channels.co2/o2.units do not match prepared observation_units.")
        observations = prepared.observations
        if self.independent_error_sd:
            species = observations.species.values
            errors = np.empty(observations.size, dtype=np.float64)
            for name in ("co2", "o2"):
                errors[species == name] = self.independent_error_sd[name]
            independent = observations.copy(data=errors).rename("independent_error_sd")
            independent.attrs = {"units": "mixed; see observation_units coordinate"}
        else:
            independent = prepared.independent_error_sd
            if independent is None:
                raise ValueError("Omitted channel independent_error_sd requires errors in the prepared artifact.")
        result = {item.name: getattr(self, item.name) for item in fields(self)
                  if item.name not in {"co2_units", "o2_units", "independent_error_sd", "boundary"}
                  and getattr(self, item.name) is not None}
        return {**result, "independent_error_sd": independent, "boundary": self._channel_boundaries(prepared)}

    def _channel_boundaries(self, prepared: Co2O2PreparedInputs) -> dict[str, Choice | None]:
        """Resolve configured choices and default prepared channels to choices.

        Explicitly disabled channels stay in the result as ``None`` so the
        runner can tell a deliberate omission from an unconfigured channel.
        """
        configured = dict(self.boundary or {})
        enabled = [choice for choice in configured.values() if choice is not None]
        default = MultiplicativeChoice(
            dict(DEFAULT_BC_PRIOR),
            frequency=enabled[0].frequency if enabled else None,
            anchor_time=enabled[0].anchor_time if enabled else None,
        )
        available = set(prepared.boundary_operands) | set(prepared.boundary_sensitivity)
        result: dict[str, Choice | None] = {}
        for channel in ("co2", "o2"):
            if channel in configured:
                result[channel] = configured[channel]
            elif channel in available:
                result[channel] = default
        return result


def _shared_boundary_periods(
    choices: Mapping[str, Choice | None],
) -> dict[str, Choice | None]:
    """Apply one boundary calendar to every enabled linked channel."""
    enabled = [choice for choice in choices.values() if choice is not None]
    frequencies = {normalise_frequency(choice.frequency) for choice in enabled if choice.frequency is not None}
    # Anchors are compared as instants, so equivalent spellings agree.
    anchors = {pd.Timestamp(choice.anchor_time) for choice in enabled if choice.anchor_time is not None}
    if len(frequencies) > 1 or len(anchors) > 1:
        raise ValueError("Linked boundary frequency and anchor_time must agree across channels.")
    frequency = next((choice.frequency for choice in enabled if choice.frequency is not None), None)
    anchor_time = next((choice.anchor_time for choice in enabled if choice.anchor_time is not None), None)
    return {
        channel: None if choice is None else replace(choice, frequency=frequency, anchor_time=anchor_time)
        for channel, choice in choices.items()
    }


def resolve_co2_o2_sampler(config: Mapping[str, object], *, model: Co2O2ModelConfig) -> RhimeSampler:
    """Resolve live sampling controls and the backend required by the likelihood.

    Fixed OU requires PyMC; fixed independent error defaults to NumPyro.
    Channel, prepared-input and output choices remain unused.
    """
    options = _table(config, "config")
    fixed_ou = model.tau_hours is not None
    sampling = _table(options.get("sampling", {}), "sampling")
    sampling.setdefault("nuts_sampler", "pymc" if fixed_ou else "numpyro")
    sampling.setdefault("target_accept", 0.95)
    sampler = _sampling(sampling)
    if fixed_ou and sampler.nuts_sampler != "pymc":
        raise ValueError("Linked fixed-OU requires sampling.nuts_sampler='pymc'.")
    return sampler


def resolve_co2_o2_prior_seed(config: Mapping[str, object]) -> int | None:
    """Resolve the prior-predictive seed without validating unused MCMC controls."""
    return resolve_co2_prior_seed(config)


__all__ = [
    "load_co2_o2_params", "Co2O2PreparationConfig", "Co2O2ModelConfig",
    "Co2O2NativePreparationConfig", "Co2O2SourceState", "LINKED_CHANNELS",
    "resolve_co2_o2_sampler", "resolve_co2_o2_prior_seed", "validate_co2_o2_units",
]
