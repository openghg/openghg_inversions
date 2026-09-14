"""Named CO2 runner for accepted-state cached fixed-OU amplitudes."""

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
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import (
    AggregationErrorMode,
    aggregation_error_input_names,
    resolve_aggregation_error,
)
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.cached_sigma import make_cached_sigma_compound_step
from openghg_inversions.rhime.materialization import materialize_pymc_inputs
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model

from .co2_cached_sigma_model import (
    OU_SITE_AMPLITUDE,
    OU_SITE_INDEX,
    Co2CachedSigmaModel,
    build_co2_cached_sigma_model,
)
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
    sigma_target_accept: float,
    state_target_accept: float,
) -> RhimeSampler:
    if sampler.nuts_sampler != "pymc":
        raise ValueError("The cached-sigma CO2 recipe requires nuts_sampler='pymc'.")
    sample_kwargs = dict(sampler.sample_kwargs or {})
    if "target_accept" in sample_kwargs:
        raise ValueError(
            "The cached-sigma CO2 recipe owns two NUTS tuning controls; pass "
            "`sigma_target_accept` and `state_target_accept` to "
            "`run_rhime_co2_cached_sigma` instead of "
            "sampler.sample_kwargs['target_accept']."
        )
    if sample_kwargs.get("step") is not None:
        raise ValueError(
            "The cached-sigma CO2 recipe constructs its required sigma-then-state "
            "CompoundStep; do not pass sample_kwargs['step']."
        )
    with cached_model.model:
        sample_kwargs["step"] = make_cached_sigma_compound_step(
            model=cached_model.model,
            sigma=cached_model.amplitude,
            state=cached_model.state,
            target=cached_model.target,
            shared_cache=cached_model.shared_cache,
            initial_cache=cached_model.initial_cache,
            state_value_name=cached_model.state_value_name,
            state_location=np.asarray(
                cached_model.active_state_prior.latent_mean.values,
                dtype=np.float64,
            ),
            state_cholesky=np.asarray(
                cached_model.active_state_prior.latent_cholesky.values,
                dtype=np.float64,
            ),
            prior_scale=cached_model.site_amplitude_prior_scale,
            initial_point=cached_model.model.initial_point(),
            sigma_target_accept=sigma_target_accept,
            state_target_accept=state_target_accept,
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
    sigma = posterior[OU_SITE_AMPLITUDE]
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


def _annotate_cached_co2_trace(
    trace: az.InferenceData,
    built: RhimeModelBuildResult,
    *,
    concentration_units: str | None,
) -> az.InferenceData:
    """Add output semantics that belong only to the cached fixed-OU recipe."""
    trace = _annotate_co2_trace(
        trace,
        built,
        concentration_units=concentration_units,
    )
    for group_name in trace.groups():
        group = getattr(trace, group_name)
        if isinstance(group, xr.Dataset):
            group.attrs["rhime_recipe"] = "co2_cached_sigma_fixed_ou"

    if hasattr(trace, "log_likelihood") and "y" in trace.log_likelihood:
        likelihood = trace.log_likelihood["y"]
        likelihood.attrs["rhime_scientific_roles"] = json.dumps(["joint_log_likelihood"])
        likelihood.attrs.pop("units", None)
    if hasattr(trace, "posterior_predictive") and "y" in trace.posterior_predictive:
        predictive = trace.posterior_predictive["y"]
        predictive.attrs["rhime_scientific_roles"] = json.dumps(["concentration"])
        if concentration_units is not None:
            predictive.attrs["units"] = concentration_units
    if hasattr(trace, "observed_data") and "y" in trace.observed_data:
        observed = trace.observed_data["y"]
        observed.attrs["rhime_scientific_roles"] = json.dumps(["observation"])
        if concentration_units is not None:
            observed.attrs["units"] = concentration_units
    if hasattr(trace, "posterior") and OU_SITE_AMPLITUDE in trace.posterior:
        amplitude = trace.posterior[OU_SITE_AMPLITUDE]
        if concentration_units is not None:
            amplitude.attrs["units"] = concentration_units
    if hasattr(trace, "constant_data"):
        if "ou_tau_hours" in trace.constant_data:
            trace.constant_data["ou_tau_hours"].attrs["units"] = "hours"
        for name in ("Y", "error"):
            if concentration_units is not None and name in trace.constant_data:
                trace.constant_data[name].attrs["units"] = concentration_units
    return trace


def run_rhime_co2_cached_sigma(
    *,
    prepared_inputs: RhimePreparedInputs,
    tau_hours: float | Mapping[str, float],
    site_amplitude_prior_scale: float,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    sampler: RhimeSampler | None = None,
    sigma_target_accept: float = 0.8,
    state_target_accept: float = 0.9,
    aggregation_error_mode: AggregationErrorMode = "low_rank",
) -> az.InferenceData:
    """Run the production CO2 fixed-OU cached-amplitude recipe.

    The graph and sampler are a matched pair: site amplitudes are updated first by the
    exact conditional bridge, the accepted cache is refreshed once, and stock
    PyMC NUTS then updates the correlated flux state against that cache.
    ``sigma_target_accept`` and ``state_target_accept`` tune those two steps
    independently.

    Args:
        prepared_inputs: Coherent-reduction inputs containing ``H``,
            ``alpha_prior_mean``, ``alpha_prior_covariance``,
            ``fixed_prior_contribution``, ``mf``, and ``mf_error``. Observations
            must have aligned ``site`` and ``time`` coordinates. The selected
            aggregation-error representation must also be present.
        tau_hours: Fixed OU decorrelation time in hours. A scalar applies to
            every site; a mapping must cover every observed site label.
        site_amplitude_prior_scale: Scale of the independent HalfNormal site-
            amplitude priors, in the observations' concentration units.
        initial_site_amplitudes: Optional positive initial amplitude in the
            same units, supplied as one scalar or a mapping covering every
            observed site. Defaults to ``site_amplitude_prior_scale``.
        sampler: Optional sampling configuration. This recipe requires PyMC,
            constructs its own step, and supports only ``random_seed`` in
            ``posterior_predictive_kwargs``.
        sigma_target_accept: NUTS target acceptance probability for the site-
            amplitude transition.
        state_target_accept: NUTS target acceptance probability for the flux-
            state transition.
        aggregation_error_mode: Prepared aggregation-error representation.
            The default ``"low_rank"`` requires ``low_rank_factor`` and
            ``diagonal_residual_variance``.

    Returns:
        Sampled inference data with the normalized joint log likelihood as one
        value per complete observation vector and, when requested, correlated
        joint posterior-predictive vectors.

    Raises:
        ValueError: If prepared arrays, labels, numerical inputs, or model
            construction are invalid; if a site mapping is incomplete; or if
            the sampler is not PyMC, supplies ``step`` or generic
            ``target_accept``, or has unsupported predictive keywords.
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
        site_amplitude_prior_scale=site_amplitude_prior_scale,
        initial_site_amplitudes=initial_site_amplitudes,
        state_activity=cast(StateActivity | None, _state_activity_from_inputs(model_inputs)),
    )
    requested_sampler = RhimeSampler() if sampler is None else sampler
    metadata = {
        "recipe": "co2_cached_sigma_fixed_ou",
        "kind": "builtin",
        "prior": "correlated arithmetic-moment lognormal",
        "mismatch_component": "fixed_within_site_ou",
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
            "fixed_ou_site_amplitude": OU_SITE_AMPLITUDE,
            "observation_to_fixed_ou_site_index": OU_SITE_INDEX,
            "fixed_ou_timescale": "ou_tau_hours",
            "emissions_sensitivity": "co2_sensitivity",
            "coherent_prior_contribution": "fixed_prior_contribution",
        },
        metadata=metadata,
    )
    sampling_sampler = _sampler_for_cached_graph(
        requested_sampler,
        cached_model=cached_model,
        sigma_target_accept=sigma_target_accept,
        state_target_accept=state_target_accept,
    )
    trace = sample_rhime_model(built, sampling_sampler)
    trace = _append_joint_outputs(
        trace,
        cached_model=cached_model,
        observations=model_inputs["mf"],
        posterior_predictive=_posterior_predictive_requested(requested_sampler),
        random_seed=_predictive_seed(requested_sampler),
    )
    return _annotate_cached_co2_trace(
        trace,
        built,
        concentration_units=model_inputs["mf"].attrs.get("units"),
    )


__all__ = ["co2_cached_sigma_input_names", "run_rhime_co2_cached_sigma"]
