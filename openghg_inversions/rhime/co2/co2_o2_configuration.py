"""Independent configuration owners for the staged linked CO2/O2 recipe."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np

from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.rhime.sampling import RhimeSampler

from .co2_o2_preparation import Co2O2PreparedInputs
from .configuration import (
    _reject_unknown,
    _resolve_linked,
    _sampling,
    _string,
    _table,
    load_co2_family_config,
    resolve_co2_prior_seed,
    resolve_co2_preparation,
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
    allowed = {"format_version", "recipe", "prepared_inputs", "channels", "likelihood", "sampling", "outputs"}
    _reject_unknown({name: value for name, value in options.items() if name not in allowed}, "config")
    return options


def resolve_co2_o2_preparation(
    config: Mapping[str, object], *, require_path: bool = False,
) -> dict[str, Any]:
    """Resolve a prepared artifact path and original requested dates/domain only.

    ``require_path`` selects importing callers; in-memory callers may omit it.
    Other phases' sections remain unused. Paths resolve from the invocation
    directory and no artifact is opened here.
    """
    prepared = _table(_table(config, "config").get("prepared_inputs", {}), "prepared_inputs")
    if "reconstruction_path" in prepared:
        raise ValueError("Linked CO2/O2 preparation does not support prepared_inputs.reconstruction_path.")
    result = resolve_co2_preparation({"prepared_inputs": prepared}, require_path=require_path)
    result.pop("affine_path")
    return result


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
    if set(observations.species.values.astype(str)) != {"co2", "o2"}:
        raise ValueError("Linked prepared observations must contain exactly CO2 and O2 channels.")
    units = np.unique(observations.observation_units.values.astype(str))
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


def resolve_co2_o2_model(
    config: Mapping[str, object], prepared: Co2O2PreparedInputs,
) -> dict[str, Any]:
    """Bind configured linked science to one unchanged prepared artifact.

    Required ``channels.co2/o2.units`` must match the artifact's common unit.
    Configure positive independent-error standard deviations for both channels
    or retain the saved vector. Existing replay validation owns agreement with
    saved errors. Channel boundary/offset options keep their ordinary meanings.
    Fixed independent error and fixed-amplitude fixed OU are supported here;
    inference and reporting settings are not consumed.

    Returns:
        Keywords for ``build_rhime_co2_o2``, excluding prepared inputs and sampler.

    Raises:
        ValueError: If channel labels/units or scientific options are invalid,
            errors are unavailable, or inferred OU amplitudes are selected.
    """
    validate_co2_o2_units(prepared)
    options = _table(config, "config")
    selected: dict[str, object] = {"channels": options.get("channels", {})}
    if likelihood := _likelihood(options):
        selected["likelihood"] = likelihood
    setup = _resolve_linked(selected, "linked")
    arguments = dict(setup.runner_arguments(prepared))
    arguments.pop("prepared_inputs")
    arguments.pop("sampler")
    return arguments


def resolve_co2_o2_sampler(config: Mapping[str, object]) -> RhimeSampler:
    """Resolve live sampling controls and the backend required by the likelihood.

    Fixed OU requires PyMC; fixed independent error defaults to NumPyro.
    Channel, prepared-input and output choices remain unused.
    """
    options = _table(config, "config")
    fixed_ou = _table(options.get("likelihood", {}), "likelihood").get("kind") == "fixed_ou"
    sampling = _table(options.get("sampling", {}), "sampling")
    sampling.setdefault("nuts_sampler", "pymc" if fixed_ou else "numpyro")
    sampling.setdefault("target_accept", 0.95)
    sampler = _sampling(sampling, cached=False)
    if fixed_ou and sampler.nuts_sampler != "pymc":
        raise ValueError("Linked fixed-OU requires sampling.nuts_sampler='pymc'.")
    return sampler


def resolve_co2_o2_prior_seed(config: Mapping[str, object]) -> int | None:
    """Resolve the prior-predictive seed without validating unused MCMC controls."""
    return resolve_co2_prior_seed(config)


__all__ = [
    "load_co2_o2_params", "resolve_co2_o2_preparation", "resolve_co2_o2_model",
    "resolve_co2_o2_sampler", "resolve_co2_o2_prior_seed", "validate_co2_o2_units",
]
