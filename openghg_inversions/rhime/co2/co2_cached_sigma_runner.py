"""Named CO2 runner for accepted-state cached fixed-OU site sigma."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import json
from typing import Any, cast

import arviz as az
import numpy as np
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.models.site_sigma import SITE_SIGMA
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import (
    AggregationErrorMode,
    aggregation_error_input_names,
    resolve_aggregation_error,
)
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.cached_sigma import (
    CACHED_SIGMA_SAMPLER_METADATA,
    make_cached_sigma_compound_step,
)
from openghg_inversions.rhime.materialization import materialize_pymc_inputs
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model

from .co2_cached_sigma_model import Co2CachedSigmaModel, build_co2_cached_sigma_model
from .co2_runner import _annotate_co2_trace, _state_activity_from_inputs


_CO2_CACHED_SIGMA_INPUT_NAMES = (
    "H",
    "alpha_prior_mean",
    "alpha_prior_covariance",
    "fixed_prior_contribution",
    "mf",
    "mf_error",
)


def co2_cached_sigma_input_names(
    prepared_inputs: RhimePreparedInputs,
    *,
    aggregation_error_mode: AggregationErrorMode,
) -> tuple[str, ...]:
    """Declare arrays consumed by the named cached fixed-OU recipe."""
    inputs = prepared_inputs.inv_inputs
    names = list(_CO2_CACHED_SIGMA_INPUT_NAMES)
    names.extend(aggregation_error_input_names(inputs, aggregation_error_mode))
    if "state_is_active" in inputs:
        names.append("state_is_active")
        if "state_fixed_value" in inputs:
            names.append("state_fixed_value")
    missing = [name for name in names if name not in inputs]
    if missing:
        raise ValueError(
            "Cached-sigma CO2 prepared inputs are missing required variable(s): "
            f"{missing!r}."
        )
    return tuple(names)


def _sampler_for_cached_graph(
    sampler: RhimeSampler,
    *,
    cached_model: Co2CachedSigmaModel,
) -> RhimeSampler:
    if sampler.nuts_sampler != "pymc":
        raise ValueError("The cached-sigma CO2 recipe requires nuts_sampler='pymc'.")
    sample_kwargs = dict(sampler.sample_kwargs or {})
    if sample_kwargs.get("step") is not None:
        raise ValueError(
            "The cached-sigma CO2 recipe constructs its required sigma-then-state "
            "CompoundStep; do not pass sample_kwargs['step']."
        )
    with cached_model.model:
        sample_kwargs["step"] = make_cached_sigma_compound_step(
            model=cached_model.model,
            sigma=cached_model.sigma,
            state=cached_model.state,
            target=cached_model.target,
            shared_cache=cached_model.shared_cache,
            state_value_name=cached_model.state_value_name,
            state_location=cached_model.state_location,
            state_cholesky=cached_model.state_cholesky,
            prior_scale=cached_model.sigma_prior_scale,
            initial_point=cached_model.model.initial_point(),
        )
    sample_kwargs.setdefault("mp_ctx", "spawn")
    idata_kwargs = dict(sample_kwargs.get("idata_kwargs", {}))
    idata_kwargs["log_likelihood"] = False
    sample_kwargs["idata_kwargs"] = idata_kwargs
    configured = copy.copy(sampler)
    configured.sample_kwargs = sample_kwargs
    # This graph contains a normalized joint Potential, not an observed PyMC
    # distribution. Exact joint output is attached below from the same target.
    configured.sample_posterior_predictive = False
    return configured


def _posterior_predictive_requested(sampler: RhimeSampler) -> bool:
    requested = sampler.sample_posterior_predictive
    if isinstance(requested, bool):
        return requested
    return "y" in requested or "concentration" in requested


def _predictive_seed(sampler: RhimeSampler) -> Any:
    predictive_kwargs = dict(sampler.posterior_predictive_kwargs or {})
    unsupported = set(predictive_kwargs) - {"random_seed"}
    if unsupported:
        raise ValueError(
            "The cached-sigma CO2 joint predictive supports only "
            f"`random_seed`; got {sorted(unsupported)!r}."
        )
    if "random_seed" in predictive_kwargs:
        return predictive_kwargs["random_seed"]
    return dict(sampler.sample_kwargs or {}).get("random_seed")


def _seed_metadata(value: Any) -> int | list[int] | str | None:
    """Return JSON-safe seed provenance without consuming an RNG."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        array = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError):
        return f"runtime {type(value).__module__}.{type(value).__qualname__}"
    return [int(item) for item in array.reshape(-1)]


def _observation_coords(observations: xr.DataArray) -> dict[str, Any]:
    output_dim = str(observations.dims[0])
    coords: dict[str, Any] = {output_dim: observations.coords[output_dim]}
    for name, coordinate in observations.coords.items():
        if name != output_dim and coordinate.dims == (output_dim,):
            coords[str(name)] = coordinate
    return coords


def _append_joint_outputs(
    trace: az.InferenceData,
    *,
    cached_model: Co2CachedSigmaModel,
    observations: xr.DataArray,
    posterior_predictive: bool,
    random_seed: Any,
) -> az.InferenceData:
    """Attach exact joint log likelihood and optional joint replicates."""
    posterior = cast(xr.Dataset, trace.posterior)
    state = posterior[cached_model.state_output_name]
    sigma = posterior[SITE_SIGMA]
    state_dim = next(dim for dim in state.dims if dim not in {"chain", "draw"})
    sigma_dim = next(dim for dim in sigma.dims if dim not in {"chain", "draw"})
    state_values = np.asarray(
        state.transpose("chain", "draw", state_dim).values,
        dtype=np.float64,
    )
    sigma_values = np.asarray(
        sigma.transpose("chain", "draw", sigma_dim).values,
        dtype=np.float64,
    )
    chain_count, draw_count = state_values.shape[:2]
    log_likelihood = np.empty((chain_count, draw_count), dtype=np.float64)
    predictive = (
        np.empty(
            (chain_count, draw_count, cached_model.target.n_obs),
            dtype=np.float64,
        )
        if posterior_predictive
        else None
    )
    rng = np.random.default_rng(random_seed)
    for chain in range(chain_count):
        for draw in range(draw_count):
            draw_state = state_values[chain, draw]
            draw_sigma = sigma_values[chain, draw]
            log_likelihood[chain, draw] = cached_model.target.log_likelihood(
                draw_state,
                draw_sigma,
            )
            if predictive is not None:
                predictive[chain, draw] = cached_model.target.random(
                    draw_state,
                    draw_sigma,
                    rng=rng,
                )

    sample_coords = {
        "chain": posterior.coords["chain"],
        "draw": posterior.coords["draw"],
    }
    likelihood_data = xr.DataArray(
        log_likelihood,
        dims=("chain", "draw"),
        coords=sample_coords,
        name="y",
        attrs={
            "rhime_likelihood_scope": "joint_observation_vector",
            "rhime_normalized_log_likelihood": True,
        },
    )
    groups: dict[str, xr.Dataset] = {
        "log_likelihood": likelihood_data.to_dataset(),
    }
    if predictive is not None:
        output_dim = str(observations.dims[0])
        predictive_coords = {**sample_coords, **_observation_coords(observations)}
        predictive_data = xr.DataArray(
            predictive,
            dims=("chain", "draw", output_dim),
            coords=predictive_coords,
            name="y",
            attrs={"rhime_predictive_scope": "joint_observation_vector"},
        )
        groups["posterior_predictive"] = predictive_data.to_dataset()
    if "observed_data" not in trace.groups():
        groups["observed_data"] = observations.rename("y").to_dataset()
    trace.add_groups(groups)
    burn = trace.attrs.get("burn")
    for group in groups:
        dataset = cast(xr.Dataset, getattr(trace, group))
        if burn is not None and "draw" in dataset.dims:
            dataset.attrs["burn"] = burn
    return trace


def run_rhime_co2_cached_sigma(
    *,
    prepared_inputs: RhimePreparedInputs,
    tau_hours: float | Mapping[str, float],
    sigma_prior_scale: float,
    initial_site_sigma: float | Mapping[str, float] | None = None,
    sampler: RhimeSampler | None = None,
    aggregation_error_mode: AggregationErrorMode = "low_rank",
) -> az.InferenceData:
    """Run the production CO2 fixed-OU cached site-sigma recipe.

    The graph and sampler are a matched pair: sigma is updated first by the
    exact conditional bridge, the accepted cache is refreshed once, and stock
    PyMC NUTS then updates the correlated flux state against that cache.
    """
    prepared = prepared_inputs.validated()
    names = co2_cached_sigma_input_names(
        prepared,
        aggregation_error_mode=aggregation_error_mode,
    )
    model_inputs = materialize_pymc_inputs(prepared, variable_names=names)
    aggregation_error = resolve_aggregation_error(
        model_inputs,
        aggregation_error_mode,
    )
    prior_covariance = model_inputs["alpha_prior_covariance"]
    retained_prior = CorrelatedLognormalPrior(
        model_inputs["alpha_prior_mean"],
        prior_covariance,
        covariance_dim=str(prior_covariance.dims[-1]),
    )
    cached_model = build_co2_cached_sigma_model(
        model_inputs["H"],
        retained_prior=retained_prior,
        fixed_prior_contribution=model_inputs["fixed_prior_contribution"],
        observations=model_inputs["mf"],
        observation_error=model_inputs["mf_error"],
        aggregation_error=aggregation_error,
        tau_hours=tau_hours,
        sigma_prior_scale=sigma_prior_scale,
        initial_site_sigma=initial_site_sigma,
        state_activity=cast(StateActivity | None, _state_activity_from_inputs(model_inputs)),
    )
    requested_sampler = RhimeSampler() if sampler is None else sampler
    metadata = {
        "recipe": "co2_cached_sigma_fixed_ou",
        "kind": "builtin",
        "prior": "correlated arithmetic-moment lognormal",
        "mismatch_component": "fixed_within_site_ou",
        "sampler": {
            **dict(CACHED_SIGMA_SAMPLER_METADATA),
            "random_seed": _seed_metadata(
                dict(requested_sampler.sample_kwargs or {}).get("random_seed")
            ),
            "posterior_predictive_random_seed": _seed_metadata(
                _predictive_seed(requested_sampler)
            ),
        },
        "numerical_preparation": {
            "target": (
                "openghg_inversions.models.cached_sigma."
                "FixedOuCachedSigmaTarget"
            ),
            "fixed_ou_eigenbasis": "derived_at_model_build_from_prepared_inputs",
            "accepted_quadratic": "chain_local_runtime_state_not_an_external_artifact",
            "site_labels": list(cached_model.covariance.site_labels),
            "tau_hours_by_site": [
                float(value) for value in cached_model.covariance.tau_hours_by_site
            ],
            "aggregation_error_mode": aggregation_error.mode,
            "fixed_covariance_rank": cached_model.covariance.rank,
            "initial_site_sigma": {
                site: float(value)
                for site, value in zip(
                    cached_model.covariance.site_labels,
                    cached_model.initial_site_sigma,
                )
            },
        },
        "basis_artifact_source": getattr(
            prepared,
            "basis_artifact_source",
            "unknown",
        ),
        "basis_artifact_path": getattr(prepared, "basis_artifact_path", None),
    }
    built = RhimeModelBuildResult(
        model=cached_model.model,
        variable_roles={
            "observation": "Y",
            "observation_error": "error",
            "concentration": "y",
            "model_error": "epsilon",
            "model_mean": "modelled_concentration",
            "pollution_concentration": "co2_flux_contribution",
            "flux_scale": "flux_scaling",
            "site_model_error": SITE_SIGMA,
            "emissions_sensitivity": "co2_sensitivity",
            "coherent_prior_contribution": "fixed_prior_contribution",
        },
        metadata=metadata,
    )
    sampling_sampler = _sampler_for_cached_graph(
        requested_sampler,
        cached_model=cached_model,
    )
    trace = sample_rhime_model(built, sampling_sampler)
    trace = _append_joint_outputs(
        trace,
        cached_model=cached_model,
        observations=model_inputs["mf"],
        posterior_predictive=_posterior_predictive_requested(requested_sampler),
        random_seed=_predictive_seed(requested_sampler),
    )
    trace.attrs["rhime_sampler_provenance"] = json.dumps(
        metadata["sampler"],
        sort_keys=True,
    )
    return _annotate_co2_trace(
        trace,
        built,
        concentration_units=model_inputs["mf"].attrs.get("units"),
    )


__all__ = ["co2_cached_sigma_input_names", "run_rhime_co2_cached_sigma"]
