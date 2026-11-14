"""Concrete CO2 construction and inference selected by the scientific likelihood."""

import json
from collections.abc import Mapping
from typing import Any, Literal

import xarray as xr

from openghg_inversions.inference.sampling import describe_recorded_sampler
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model

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
        return co2_cached_sigma_build_result(cached, prepared, biosphere_prior=model.get("biosphere_prior")), cached
    inputs, priors, boundary = materialize_co2_inputs(
        prepared_inputs=prepared,
        sigma_alignment=model.get("sigma_alignment"),
        sigma_prior=model.get("sigma_prior"),
        fixed_model_mismatch=model.get("fixed_model_mismatch"),
        likelihood_builder=model.get("likelihood_builder"),
        likelihood_kwargs=model.get("likelihood_kwargs"),
        no_model_error=model.get("no_model_error", False),
        boundary=model.get("boundary"),
        offset_prior=model.get("offset_prior"),
        offset_args=model.get("offset_args"),
    )
    return build_co2_from_materialized(
        inputs, prepared_inputs=prepared, priors=priors, boundary=boundary,
        biosphere_prior=model.get("biosphere_prior"),
        sigma_alignment=model.get("sigma_alignment"),
        sigma_prior=model.get("sigma_prior"),
        fixed_model_mismatch=model.get("fixed_model_mismatch"),
        likelihood_builder=model.get("likelihood_builder"),
        likelihood_kwargs=model.get("likelihood_kwargs"),
        no_model_error=model.get("no_model_error", False),
        offset_prior=model.get("offset_prior"),
        offset_args=model.get("offset_args"),
    ), None


def annotate_co2_inference(
    trace: xr.DataTree, built: RhimeModelBuildResult, prepared: Co2PreparedInputs,
    model: Mapping[str, Any], cached: Co2CachedSigmaModel | None,
) -> xr.DataTree:
    """Keep exact joint prediction and likelihood semantics on inferred OU output."""
    units = prepared.inv_inputs["mf"].attrs.get("units")
    if cached is not None:
        return _annotate_cached_co2_trace(
            trace, built, concentration_units=units, biosphere_prior=model.get("biosphere_prior"),
        )
    return annotate_co2_trace(
        trace, built, concentration_units=units, biosphere_prior=model.get("biosphere_prior"),
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
    seed = recorded.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0:
        record["seed"] = seed
    for name in ("target_accept", "sigma_target_accept", "state_target_accept"):
        value = recorded.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool) and 0 < value < 1:
            record[name] = value
    return record


def sample_co2_inference(
    prepared: Co2PreparedInputs, model: Mapping[str, Any], sampler: RhimeSampler, *,
    sigma_target_accept: float = 0.8, state_target_accept: float = 0.9,
    method: Literal["nuts", "gaussian"] = "nuts",
) -> tuple[xr.DataTree, dict[str, Any]]:
    """Build the selected graph, sample its matched steps and record their settings.

    ``method="gaussian"`` instead solves the exact fixed-noise Gaussian
    posterior with :func:`~openghg_inversions.rhime.co2.co2_gaussian.sample_co2_gaussian`
    and returns independent draws in the same trace layout.
    """
    if method == "gaussian":
        from .co2_gaussian import sample_co2_gaussian

        return sample_co2_gaussian(prepared, model, sampler)
    if method != "nuts":
        raise ValueError(f"Unknown CO2 inference method {method!r}; use 'nuts' or 'gaussian'.")
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
