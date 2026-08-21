"""Explicit PyMC graph for the linked CO2/O2 recipe."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models import (
    ResolvedStateActivity,
    StateActivity,
    add_correlated_lognormal_state_with_activity,
    add_model_data,
    detect_zero_sensitivity,
    registered_model,
    resolve_state_activity,
)
from openghg_inversions.models.likelihoods import (
    add_aggregation_error_data,
    add_gaussian_observation_likelihood,
)
from openghg_inversions.observation_error import AggregationError


def _validate_linked_arrays(
    *,
    observations: xr.DataArray,
    prior_forward_mean: xr.DataArray,
    effective_observation_operator: xr.DataArray,
    independent_error_sd: xr.DataArray,
    aggregation_error: AggregationError,
    retained_prior: CorrelatedLognormalPrior,
    output_dim: str,
    covariance_dim: str,
) -> xr.DataArray:
    """Validate explicit scientific arrays at the linked-model boundary."""
    owner = "Linked CO2/O2 model"
    if observations.dims != (output_dim,):
        raise ValueError(f"{owner} observations must have dims ({output_dim!r},).")
    if output_dim not in observations.indexes or not observations.indexes[output_dim].is_unique:
        raise ValueError(f"{owner} observations require unique {output_dim!r} labels.")
    observation_values = np.asarray(observations.values)
    if not np.issubdtype(observation_values.dtype, np.number) or not np.isfinite(observation_values).all():
        raise ValueError(f"{owner} observations must be finite and numeric.")

    if prior_forward_mean.dims != (output_dim,):
        raise ValueError(f"{owner} prior forward mean must have dims ({output_dim!r},).")
    prior_forward_values = np.asarray(prior_forward_mean.values)
    if (
        not np.issubdtype(prior_forward_values.dtype, np.number)
        or not np.isfinite(prior_forward_values).all()
    ):
        raise ValueError(f"{owner} prior forward mean must be finite and numeric.")
    state_dim = retained_prior.state_dim
    if effective_observation_operator.dims != (output_dim, state_dim):
        raise ValueError(f"{owner} effective operator must have dims {(output_dim, state_dim)!r}.")
    if not effective_observation_operator.get_index(state_dim).equals(
        retained_prior.mean.get_index(state_dim)
    ):
        raise ValueError(f"{owner} operator state labels must match the retained prior.")
    operator_values = np.asarray(effective_observation_operator.values)
    if not np.issubdtype(operator_values.dtype, np.number) or not np.isfinite(operator_values).all():
        raise ValueError(f"{owner} effective operator must be finite and numeric.")
    mean = retained_prior.mean
    if any(name not in mean.coords for name in ("source", "tracer_scope")):
        raise ValueError(f"{owner} retained states require source and tracer_scope coordinates.")
    state_roles = {
        (str(source).lower(), str(scope).lower())
        for source, scope in zip(mean["source"].values, mean["tracer_scope"].values, strict=True)
    }
    if state_roles != {
        ("gpp", "shared"),
        ("ter", "shared"),
        ("ff", "shared"),
        ("ocean", "co2"),
        ("ocean", "o2"),
    }:
        raise ValueError(f"{owner} states must be shared GPP/TER/FF and tracer-specific CO2/O2 ocean states.")

    if independent_error_sd.dims != (output_dim,):
        raise ValueError(f"{owner} independent error must have dims ({output_dim!r},).")
    if "observation_units" not in observations.coords or "observation_units" not in (
        independent_error_sd.coords
    ):
        raise ValueError(f"{owner} observations and independent error require observation_units.")
    if not np.array_equal(
        observations["observation_units"].values,
        independent_error_sd["observation_units"].values,
    ):
        raise ValueError(f"{owner} independent-error units must match the observations.")
    error_values = np.asarray(independent_error_sd.values)
    if (
        not np.issubdtype(error_values.dtype, np.number)
        or not np.isfinite(error_values).all()
        or (error_values <= 0).any()
    ):
        raise ValueError(f"{owner} independent error must be finite and positive.")

    return independent_error_sd.rename("fixed_independent_error_sd")


def _resolve_activity(
    effective_observation_operator: xr.DataArray,
    state_activity: StateActivity | None,
    *,
    output_dim: str,
) -> ResolvedStateActivity:
    """Align an optional active/fixed policy with the joint operator."""
    return resolve_state_activity(
        detect_zero_sensitivity(
            effective_observation_operator,
            output_dim=output_dim,
        ),
        state_activity,
    )


def linked_co2_o2_prior_forward_mean(
    *,
    prior_forward_mean: xr.DataArray,
    effective_observation_operator: xr.DataArray,
    retained_prior: CorrelatedLognormalPrior,
    state_activity: StateActivity | None = None,
    output_dim: str = "observation",
) -> xr.DataArray:
    """Return the deterministic prior concentration after exact state fixing."""
    activity = _resolve_activity(
        effective_observation_operator,
        state_activity,
        output_dim=output_dim,
    )
    prior_state = xr.where(
        activity.active,
        retained_prior.mean,
        activity.fixed_value,
    )
    adjustment = xr.dot(
        effective_observation_operator,
        prior_state - retained_prior.mean,
        dim=retained_prior.state_dim,
    )
    return (prior_forward_mean + adjustment).rename("prior_forward_concentration")


def build_linked_co2_o2_model(
    *,
    observations: xr.DataArray,
    prior_forward_mean: xr.DataArray,
    effective_observation_operator: xr.DataArray,
    aggregation_error: AggregationError,
    retained_prior: CorrelatedLognormalPrior,
    independent_error_sd: xr.DataArray,
    state_activity: StateActivity | None = None,
    output_dim: str = "observation",
    covariance_dim: str = "observation_cov",
) -> pm.Model:
    """Build one affine linked signal and fixed-error joint likelihood.

    All scientific inputs are explicit labelled values. The UOB prototype
    supplies one ppm as ``independent_error_sd`` for each channel; this builder
    does not encode that run policy and does not infer a mismatch amplitude.
    """
    independent_error = _validate_linked_arrays(
        observations=observations,
        prior_forward_mean=prior_forward_mean,
        effective_observation_operator=effective_observation_operator,
        independent_error_sd=independent_error_sd,
        aggregation_error=aggregation_error,
        retained_prior=retained_prior,
        output_dim=output_dim,
        covariance_dim=covariance_dim,
    )
    activity = _resolve_activity(
        effective_observation_operator,
        state_activity,
        output_dim=output_dim,
    )
    prior_state = xr.where(
        activity.active,
        retained_prior.mean,
        activity.fixed_value,
    ).rename("prior_flux_scaling")
    activity_prior_forward = linked_co2_o2_prior_forward_mean(
        prior_forward_mean=prior_forward_mean,
        effective_observation_operator=effective_observation_operator,
        retained_prior=retained_prior,
        state_activity=state_activity,
        output_dim=output_dim,
    )

    with registered_model() as model:
        state = add_correlated_lognormal_state_with_activity(
            activity,
            retained_prior,
            var_name="flux_scaling",
        ).state
        operator = add_model_data(
            effective_observation_operator,
            "effective_observation_operator",
        )
        prior_state_data = add_model_data(prior_state)
        prior_forward_data = add_model_data(activity_prior_forward)
        modelled = pm.Deterministic(
            "modelled_concentration",
            prior_forward_data + pt.dot(operator, state - prior_state_data),
            dims=output_dim,
        )
        observed = add_model_data(observations, "observed_concentration")
        fixed_error = add_model_data(independent_error)
        registered_aggregation_error = add_aggregation_error_data(
            aggregation_error,
            observations,
            output_dim=output_dim,
        )
        add_gaussian_observation_likelihood(
            observed=observed,
            mean=modelled,
            independent_variance=fixed_error**2,
            aggregation_error=registered_aggregation_error,
            output_dim=output_dim,
        )
    return model
