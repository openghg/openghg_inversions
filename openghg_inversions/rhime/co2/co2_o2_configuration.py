"""Independent configuration owners for the staged linked CO2/O2 recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from openghg_inversions.models import StateActivity
from openghg_inversions.models.boundary import Choice, MultiplicativeChoice
from openghg_inversions.inversion_inputs import normalise_frequency
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR, normalise_optional_mapping

from .co2_o2_preparation import Co2O2PreparedInputs
from .configuration import (
    _validate_section_option_names, _PREPARATION_OPTION_NAMES, _SAMPLING_OPTION_NAMES,
    _PRIOR_OPTION_NAMES, _MODEL_SECTION_OPTION_NAMES,
    _reject_unknown,
    _bool, _number, _take, _model, _co2_likelihood, _frozen,
    Co2PreparationConfig,
    _sampling,
    _string,
    _table, _artifact_path,
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
    if options.get("recipe", "co2_o2") not in {"co2_o2", "co2_o2_shared_baseline"}:
        raise ValueError("config.recipe must be co2_o2 or co2_o2_shared_baseline.")
    if "variant" in options:
        raise ValueError("Linked CO2/O2 configuration has no variant selector; select the scientific likelihood.")
    return options


def _validate_co2_o2_option_names(*requests: Mapping[str, object]) -> None:
    """Check request names while deferring scientific and inactive values."""
    allowed = {"format_version", "recipe", "variant", "prepared_inputs", "channels", "biosphere", "likelihood", "sampling", "outputs"}
    from .co2_o2_outputs import Co2O2OutputOptions

    channel_sections = {
        "": frozenset({"units", "independent_error_sd", "boundary", "offset"}),
        **{name: options for name, options in _MODEL_SECTION_OPTION_NAMES.items() if name != "biosphere"},
        "boundary": _MODEL_SECTION_OPTION_NAMES["boundary"] | {"activity"},
        "boundary.activity": frozenset({"active", "fixed_value"}),
    }
    _validate_section_option_names(requests, {
        "": frozenset(allowed),
        "prepared_inputs": _PREPARATION_OPTION_NAMES,
        "biosphere": frozenset({"path"}),
        "channels": frozenset({"co2", "o2"}),
        **{f"channels.{channel}" + (f".{path}" if path else ""): names
           for channel in ("co2", "o2") for path, names in channel_sections.items()},
        "likelihood": frozenset({"kind", "tau_hours", "fixed_site_amplitudes", "site_amplitude_prior",
                                 "site_amplitude_prior_scale", "initial_site_amplitudes"}),
        "likelihood.site_amplitude_prior": _PRIOR_OPTION_NAMES,
        "sampling": _SAMPLING_OPTION_NAMES,
        "outputs": frozenset(Co2O2OutputOptions.__dataclass_fields__),
        "outputs.paris_postprocessing_kwargs": frozenset({
            "template_version", "obs_avg_period", "flux_frequency", "country_selections",
        }),
        "outputs.native_flux_bases": frozenset({"co2", "o2"}),
    })


@dataclass(frozen=True)
class Co2O2PreparationConfig:
    """Linked artifact selection and original requested context, without I/O."""

    prepared_path: Path | None = None
    start_date: str | None = None
    end_date: str | None = None
    domain: str | None = None

    @classmethod
    def from_params(cls, config: Mapping[str, object], *, require_path: bool = False) -> Co2O2PreparationConfig:
        """Parse only linked artifact options; reject affine reconstruction."""
        prepared = _table(_table(config, "config").get("prepared_inputs", {}), "prepared_inputs")
        if "reconstruction_path" in prepared:
            raise ValueError("Linked CO2/O2 preparation does not support prepared_inputs.reconstruction_path.")
        ordinary = Co2PreparationConfig.from_params({"prepared_inputs": prepared}, require_path=require_path)
        return cls(ordinary.prepared_path, ordinary.start_date, ordinary.end_date, ordinary.domain)

    def preparation_arguments(self) -> dict[str, Any]:
        """Return keywords for the linked artifact-import boundary."""
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


def _likelihood(config: Mapping[str, object], *, inferred_amplitudes: bool = False) -> dict[str, object] | None:
    options = _table(config.get("likelihood", {}), "likelihood")
    kind = options.get("kind", "fixed_independent")
    if "site_amplitude_prior_scale" in options:
        raise ValueError("likelihood.site_amplitude_prior_scale is unsupported; use site_amplitude_prior.")
    if "initial_site_amplitudes" in options and not (
        inferred_amplitudes and kind == "fixed_ou" and "site_amplitude_prior" in options
    ):
        raise ValueError("likelihood.initial_site_amplitudes is unused without shared inferred fixed_ou amplitudes.")
    if kind == "fixed_independent":
        options.pop("kind", None)
        _reject_unknown(options, "likelihood")
        return None
    if kind != "fixed_ou":
        raise ValueError("Linked likelihood.kind must be 'fixed_independent' or 'fixed_ou'.")
    if not inferred_amplitudes and ("site_amplitude_prior" in options or "site_amplitude_prior_scale" in options):
        raise ValueError("Configured linked inferred OU amplitudes are not supported yet; use fixed_site_amplitudes.")
    if not inferred_amplitudes and "fixed_site_amplitudes" not in options:
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
    biosphere_prior_path: Path | None = None
    recipe: str = "co2_o2"
    site_amplitude_prior: Mapping[str, Any] | None = None
    initial_site_amplitudes: float | Mapping[str, float] | None = None

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
        recipe = options.get("recipe", "co2_o2")
        if recipe not in {"co2_o2", "co2_o2_shared_baseline"}:
            raise ValueError("Unsupported linked recipe.")
        shared = recipe == "co2_o2_shared_baseline"
        channels = _table(options.get("channels", {}), "channels")
        units: dict[str, str] = {}
        errors: dict[str, float] = {}
        model: dict[str, Any] = {}
        if shared and "biosphere" in options:
            raise ValueError("Shared baseline calibration belongs to the frozen preparation.")
        if "biosphere" in options:
            biosphere = _table(options["biosphere"], "biosphere")
            model["biosphere_prior_path"] = _artifact_path(_take(biosphere, "path", "biosphere"), "biosphere.path")
            _reject_unknown(biosphere, "biosphere")
        for name in ("co2", "o2"):
            channel = _table(_take(channels, name, "channels"), f"channels.{name}")
            units[name] = _string(_take(channel, "units", f"channels.{name}"), f"channels.{name}.units")
            mole_fraction_unit_scale(units[name], context=f"channels.{name}.units")
            if "independent_error_sd" in channel:
                errors[name] = _number(channel.pop("independent_error_sd"), f"channels.{name}.independent_error_sd", positive=True)
            selected = {key: channel.pop(key) for key in ("boundary", "offset") if key in channel}
            if shared and selected:
                raise ValueError("Shared baseline boundary and offset choices belong to the frozen preparation.")
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
        likelihood = _likelihood(options, inferred_amplitudes=shared)
        kwargs = {} if likelihood is None else _co2_likelihood(likelihood, matched_ou=shared)["likelihood_kwargs"]
        return cls(units["co2"], units["o2"], _frozen(errors), recipe=recipe, **_frozen(model), **kwargs)

    def bind(self, prepared: Co2O2PreparedInputs) -> dict[str, Any]:
        """Bind channel errors and check units at the scientific array boundary.

        The artifact and its saved error vector remain borrowed. Explicit errors
        expand over labelled observations; agreement with saved errors is checked
        by model construction at its materialization boundary.
        """
        shared = bool(prepared.shared_baseline)
        if shared != (self.recipe == "co2_o2_shared_baseline"):
            raise ValueError("Configured linked recipe does not match the frozen preparation.")
        unit = prepared.shared_baseline["co2"].data.attrs["units"] if shared else validate_co2_o2_units(prepared)
        if shared and prepared.shared_baseline["o2"].data.attrs["units"] != unit:
            raise ValueError("Linked shared baselines require common channel units.")
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
                  if item.name not in {"co2_units", "o2_units", "independent_error_sd", "boundary", "biosphere_prior_path", "recipe"}
                  and getattr(self, item.name) is not None}
        if self.biosphere_prior_path is not None:
            from .prior_configuration import load_biosphere_prior

            result["biosphere_prior"] = load_biosphere_prior(self.biosphere_prior_path)
        if shared:
            return {**result, "independent_error_sd": independent}
        return {**result, "independent_error_sd": independent, "boundary": self._channel_boundaries(prepared)}

    def artifact_paths(self) -> tuple[Path, ...]:
        """Identify the calibrated prior file for write protection."""
        return () if self.biosphere_prior_path is None else (Path(self.biosphere_prior_path).resolve(),)

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
    "resolve_co2_o2_sampler", "resolve_co2_o2_prior_seed", "validate_co2_o2_units",
]
