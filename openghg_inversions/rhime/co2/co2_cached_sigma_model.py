"""CO2 graph for accepted-state cached fixed-OU site sigma sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.cached_sigma import FixedOuCachedSigmaTarget
from openghg_inversions.models.components import (
    add_coherent_affine_component,
    add_correlated_lognormal_state_with_activity,
    apply_linear_sensitivity,
    add_model_data,
)
from openghg_inversions.models.coords import add_coords, registered_model
from openghg_inversions.models.fixed_ou import FixedOuLowRank, prepare_fixed_ou_low_rank
from openghg_inversions.models.state_activity import (
    StateActivity,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.models.site_sigma import (
    SIGMA_OBSERVATION,
    SITE_SIGMA,
    SITE_SIGMA_DIM,
    SITE_SIGMA_INDEX,
    site_sigma_structure,
)
from openghg_inversions.observation_error import (
    AggregationError,
    aggregation_error_as_low_rank,
    validate_observation_error_arrays,
)
from openghg_inversions.rhime.cached_sigma import PytensorMarginalQuadraticCache


@dataclass(frozen=True)
class Co2CachedSigmaModel:
    """Concrete graph plus the numerical objects required by its sampler."""

    model: pm.Model
    target: FixedOuCachedSigmaTarget
    shared_cache: PytensorMarginalQuadraticCache
    covariance: FixedOuLowRank
    sigma: TensorVariable
    state: TensorVariable
    state_value_name: str
    state_location: np.ndarray
    state_cholesky: np.ndarray
    state_output_name: str
    sigma_prior_scale: float
    initial_site_sigma: np.ndarray


def _fixed_ou_site_structure(
    observations: xr.DataArray,
    *,
    output_dim: str,
) -> tuple[tuple[str, ...], xr.DataArray]:
    time = observations.coords.get("time")
    if time is None or time.dims != (output_dim,):
        raise ValueError(
            "The cached fixed-OU CO2 recipe requires observation-aligned "
            "'time' coordinates."
        )
    labels, codes = site_sigma_structure(observations, output_dim=output_dim)
    site_index = xr.DataArray(
        codes,
        dims=(output_dim,),
        coords={output_dim: observations.coords[output_dim]},
        name=SITE_SIGMA_INDEX,
    )
    return labels, site_index


def _site_values(
    value: float | Mapping[str, float] | None,
    *,
    labels: tuple[str, ...],
    default: float,
) -> np.ndarray:
    if value is None:
        values = np.full(len(labels), default, dtype=np.float64)
    elif isinstance(value, Mapping):
        if set(value) != set(labels):
            raise ValueError(
                "`initial_site_sigma` mapping keys must exactly match the observation sites."
            )
        values = np.asarray([value[label] for label in labels], dtype=np.float64)
    else:
        values = np.full(len(labels), value, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Initial site sigma values must be finite and strictly positive.")
    return values


def _active_latent_parameters(
    prior: CorrelatedLognormalPrior,
    active_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if active_indices.size == prior.mean.sizes[prior.state_dim]:
        active_prior = prior
    else:
        active_mean = prior.mean.isel({prior.state_dim: active_indices})
        active_covariance = prior.arithmetic_covariance.isel(
            {
                prior.state_dim: active_indices,
                prior.covariance_dim: active_indices,
            }
        )
        active_prior = CorrelatedLognormalPrior(
            active_mean,
            np.asarray(active_covariance.values, dtype=np.float64),
        )
    return (
        np.asarray(active_prior.latent_mean.values, dtype=np.float64),
        np.asarray(active_prior.latent_cholesky.values, dtype=np.float64),
    )


def build_co2_cached_sigma_model(
    flux_sensitivity: xr.DataArray,
    *,
    retained_prior: CorrelatedLognormalPrior,
    fixed_prior_contribution: xr.DataArray,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    tau_hours: float | Mapping[str, float],
    sigma_prior_scale: float,
    initial_site_sigma: float | Mapping[str, float] | None = None,
    state_activity: StateActivity | None = None,
    output_dim: str = "nmeasure",
) -> Co2CachedSigmaModel:
    """Build the named CO2 cached-sigma graph in scientific order.

    The observation likelihood is a normalized cached ``Potential`` over the
    active physical flux state. The returned target owns exact joint
    likelihood and predictive evaluation after sampling; no independent
    pointwise likelihood is fabricated in the graph.
    """
    if not np.isfinite(sigma_prior_scale) or sigma_prior_scale <= 0.0:
        raise ValueError("`sigma_prior_scale` must be finite and strictly positive.")
    validate_observation_error_arrays(
        observations,
        observation_error,
        None,
        owner="Cached fixed-OU CO2 likelihood",
        output_dim=output_dim,
    )
    labels, site_index = _fixed_ou_site_structure(
        observations,
        output_dim=output_dim,
    )
    initial_sigma = _site_values(
        initial_site_sigma,
        labels=labels,
        default=float(sigma_prior_scale),
    )
    prepared_flux = prepare_linear_sensitivity(
        flux_sensitivity,
        output_dim=output_dim,
    )
    activity = resolve_state_activity(prepared_flux.removed, state_activity)
    if activity.n_active == 0:
        raise ValueError("The cached-sigma recipe requires at least one active flux state.")

    full_design = np.asarray(
        flux_sensitivity.transpose(output_dim, activity.state_dim).compute().values,
        dtype=np.float64,
    )
    fixed = np.asarray(
        fixed_prior_contribution.transpose(output_dim).compute().values,
        dtype=np.float64,
    ).copy()
    if activity.fixed_indices.size:
        fixed_values = np.asarray(activity.fixed_value.compute().values, dtype=np.float64)
        fixed += full_design[:, activity.fixed_indices] @ fixed_values[activity.fixed_indices]
    active_design = full_design[:, activity.active_indices]

    factor, aggregation_diagonal = aggregation_error_as_low_rank(aggregation_error)
    covariance = prepare_fixed_ou_low_rank(
        factor,
        np.square(
            np.asarray(
                observation_error.transpose(output_dim).compute().values,
                dtype=np.float64,
            )
        )
        + aggregation_diagonal,
        np.asarray(observations.coords["time"].compute().values),
        np.asarray(site_index.values, dtype=np.int64),
        tau_hours,
        site_labels=labels,
    )
    target = FixedOuCachedSigmaTarget(
        prepared=covariance,
        observations=np.asarray(
            observations.transpose(output_dim).compute().values,
            dtype=np.float64,
        ),
        fixed_contribution=fixed,
        design=active_design,
    )
    shared_cache = PytensorMarginalQuadraticCache(target.refresh(initial_sigma))
    state_location, state_cholesky = _active_latent_parameters(
        retained_prior,
        activity.active_indices,
    )

    with registered_model() as model:
        state_result = add_correlated_lognormal_state_with_activity(
            activity,
            retained_prior,
            var_name="flux_scaling",
        )
        if state_result.latent is None:  # guarded above; keeps the type honest
            raise AssertionError("An active cached flux state must have a latent variable.")
        contribution = apply_linear_sensitivity(
            prepared_flux,
            state_result.state,
            data_name="co2_sensitivity",
            output_name="co2_flux_contribution",
        )
        add_coherent_affine_component(
            fixed_prior_contribution,
            contribution,
            output_name="modelled_concentration",
        )
        add_coords({SITE_SIGMA_DIM: np.asarray(labels, dtype=object)})
        add_model_data(observations.transpose(output_dim), "Y")
        add_model_data(observation_error.transpose(output_dim), "error")
        sigma_index = add_model_data(site_index, SITE_SIGMA_INDEX)
        add_model_data(
            xr.DataArray(
                covariance.tau_hours_by_site,
                dims=(SITE_SIGMA_DIM,),
                coords={SITE_SIGMA_DIM: np.asarray(labels, dtype=object)},
                name="ou_tau_hours",
            )
        )
        sigma = pm.HalfNormal(
            SITE_SIGMA,
            sigma=float(sigma_prior_scale),
            initval=pm.floatX(initial_sigma),
            dims=SITE_SIGMA_DIM,
        )
        pm.Deterministic(SIGMA_OBSERVATION, sigma[sigma_index], dims=output_dim)
        pm.Deterministic(
            "epsilon",
            pt.sqrt(covariance.marginal_variance(sigma)),
            dims=output_dim,
        )
        active_state_name = (
            "flux_scaling"
            if activity.n_active == activity.n_state
            else "flux_scaling_active"
        )
        active_state = cast(TensorVariable, model[active_state_name])
        pm.Potential(
            "cached_fixed_ou_likelihood",
            shared_cache.log_likelihood(active_state),
        )

    state_value_name = cast(
        str,
        model.rvs_to_values[state_result.latent].name,
    )
    return Co2CachedSigmaModel(
        model=model,
        target=target,
        shared_cache=shared_cache,
        covariance=covariance,
        sigma=cast(TensorVariable, model[SITE_SIGMA]),
        state=state_result.latent,
        state_value_name=state_value_name,
        state_location=state_location,
        state_cholesky=state_cholesky,
        state_output_name=active_state_name,
        sigma_prior_scale=float(sigma_prior_scale),
        initial_site_sigma=initial_sigma.copy(),
    )


__all__ = ["Co2CachedSigmaModel", "build_co2_cached_sigma_model"]
