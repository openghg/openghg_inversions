"""Concrete CO2 construction and inference selected by the scientific likelihood."""

from collections.abc import Mapping
from typing import Any
import json

import xarray as xr

from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model
from openghg_inversions.inference.sampling import describe_recorded_sampler

from .co2_cached_sigma_model import Co2CachedSigmaModel
from .co2_cached_sigma_runner import (
    _annotate_cached_co2_trace,
    _sampler_for_cached_graph,
    build_rhime_co2_cached_sigma,
    co2_cached_sigma_build_result,
    validate_cached_co2_sampler,
)
from .co2_preparation import Co2PreparedInputs
from .co2_runner import annotate_co2_trace, build_co2_from_materialized, materialize_co2_inputs


def uses_cached_ou(model: Mapping[str, Any]) -> bool:
    """Identify the inferred independent-amplitude fixed-OU target."""
    return (
        model.get("likelihood_builder") is add_fixed_ou_gaussian_likelihood
        and "site_amplitude_prior" in model.get("likelihood_kwargs", {})
    )


def validate_co2_inference(model: Mapping[str, Any], sampler: RhimeSampler) -> None:
    """Validate graph-independent sampler policy before loading scientific inputs."""
    if uses_cached_ou(model):
        validate_cached_co2_sampler(sampler)


def build_co2_inference(
    prepared: Co2PreparedInputs, model: Mapping[str, Any],
) -> tuple[RhimeModelBuildResult, Co2CachedSigmaModel | None]:
    """Materialize and build the graph matched to the resolved CO2 likelihood."""
    if uses_cached_ou(model):
        options = dict(model)
        options.pop("likelihood_builder")
        likelihood = dict(options.pop("likelihood_kwargs"))
        cached = build_rhime_co2_cached_sigma(prepared_inputs=prepared, **options, **likelihood)
        return co2_cached_sigma_build_result(
            cached, prepared, bc_freq=options.get("bc_freq"), bc_anchor_time=options.get("bc_anchor_time"),
        ), cached
    inputs, priors = materialize_co2_inputs(prepared_inputs=prepared, **model)
    return build_co2_from_materialized(inputs, prepared_inputs=prepared, priors=priors, **model), None


def annotate_co2_inference(
    trace: xr.DataTree, built: RhimeModelBuildResult, prepared: Co2PreparedInputs,
    model: Mapping[str, Any], cached: Co2CachedSigmaModel | None,
) -> xr.DataTree:
    """Keep exact joint prediction and likelihood semantics on inferred OU output."""
    units = prepared.inv_inputs["mf"].attrs.get("units")
    if cached is not None:
        return _annotate_cached_co2_trace(trace, built, concentration_units=units)
    return annotate_co2_trace(
        trace, built, concentration_units=units,
        likelihood_builder=model.get("likelihood_builder"), likelihood_kwargs=model.get("likelihood_kwargs"),
    )


def describe_co2_inference(recorded: object) -> dict[str, Any]:
    """Read available CO2 algorithm provenance without constructing a sampler."""
    record: dict[str, Any] = dict(describe_recorded_sampler(recorded))
    if not isinstance(recorded, Mapping):
        return record
    if isinstance(recorded.get("algorithm"), str):
        record["algorithm"] = recorded["algorithm"]
    if recorded.get("target_accept_source") == "pm.sample_default":
        record["target_accept_source"] = "pm.sample_default"
    for name in ("target_accept", "sigma_target_accept", "state_target_accept"):
        value = recorded.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool) and 0 < value < 1:
            record[name] = value
    return record


def sample_co2_inference(
    prepared: Co2PreparedInputs, model: Mapping[str, Any], sampler: RhimeSampler, *,
    sigma_target_accept: float = 0.8, state_target_accept: float = 0.9,
) -> tuple[xr.DataTree, dict[str, Any]]:
    """Build the selected graph, sample its matched steps and record their settings."""
    validate_co2_inference(model, sampler)
    built, cached = build_co2_inference(prepared, model)
    if cached is not None:
        active = _sampler_for_cached_graph(
            sampler, cached_model=cached, sigma_target_accept=sigma_target_accept,
            state_target_accept=state_target_accept,
        )
        actual: dict[str, Any] = {**active.describe(), "algorithm": "cached_ou_sigma_nuts",
                  "sigma_target_accept": sigma_target_accept}
        if cached.states:
            actual.update(algorithm="cached_ou_sigma_then_state_nuts", state_target_accept=state_target_accept)
    else:
        active = sampler
        actual = {**active.describe(), "algorithm": "nuts"}
        sample_kwargs = sampler.sample_kwargs or {}
        if "target_accept" in sample_kwargs:
            actual["target_accept"] = sample_kwargs["target_accept"]
        else:
            actual["target_accept_source"] = "pm.sample_default"
    trace = sample_rhime_model(built, active)
    trace = annotate_co2_inference(trace, built, prepared, model, cached)
    trace.attrs["rhime_actual_inference"] = json.dumps(actual)
    return trace, actual
