"""CO2 graph for accepted-state cached fixed-OU amplitude sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.array_ops import expand_mapping
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.cached_sigma import (
    FixedOuCachedSigmaTarget,
    MarginalQuadraticCache,
)
from openghg_inversions.models.components import (
    _add_prepared_correlated_lognormal_state_with_activity,
    add_coherent_affine_component,
    add_model_data,
    apply_linear_sensitivity,
    prepare_active_correlated_lognormal_prior,
)
from openghg_inversions.models.coords import add_coords, registered_model
from openghg_inversions.models.fixed_ou import prepare_fixed_ou_low_rank
from openghg_inversions.models.state_activity import (
    StateActivity,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.observation_error import (
    AggregationError,
    aggregation_error_as_low_rank,
    validate_observation_error_arrays,
)
from openghg_inversions.rhime.cached_sigma import PytensorMarginalQuadraticCache
from openghg_inversions.sigma import SigmaAlignment


OU_SITE_DIM = "ou_site"
OU_SITE_INDEX = "ou_site_index"
OU_SITE_AMPLITUDE = "ou_site_amplitude"


@dataclass(frozen=True)
class Co2CachedSigmaModel:
    """Concrete graph plus the numerical objects required by its sampler."""

    model: pm.Model
    target: FixedOuCachedSigmaTarget
    shared_cache: PytensorMarginalQuadraticCache
    initial_cache: MarginalQuadraticCache
    amplitude: TensorVariable
    state: TensorVariable
    state_value_name: str
    active_state_prior: CorrelatedLognormalPrior
    state_output_name: str
    site_amplitude_prior_scale: float


def _fixed_ou_site_alignment(
    observations: xr.DataArray,
    *,
    output_dim: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    time = observations.coords.get("time")
    if time is None or time.dims != (output_dim,):
        raise ValueError(
            "The cached fixed-OU CO2 recipe requires observation-aligned "
            "'time' coordinates."
        )
    alignment = SigmaAlignment.from_observations(observations)
    site_index = alignment.site_index.rename(OU_SITE_INDEX)
    site_coord = alignment.site_labels.rename({"nsigma_site": OU_SITE_DIM}).rename(
        OU_SITE_DIM
    )
    return site_coord, site_index


def _site_values(
    value: float | Mapping[str, float] | None,
    *,
    site_coord: xr.DataArray,
    default: float,
) -> np.ndarray:
    if value is None:
        values = np.full(site_coord.size, default, dtype=np.float64)
    elif isinstance(value, Mapping):
        values = np.asarray(
            expand_mapping(value, site_coord, name=OU_SITE_AMPLITUDE).values,
            dtype=np.float64,
        )
    else:
        values = np.full(site_coord.size, value, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Initial site amplitudes must be finite and strictly positive.")
    return values


def build_co2_cached_sigma_model(
    flux_sensitivity: xr.DataArray,
    *,
    retained_prior: CorrelatedLognormalPrior,
    fixed_prior_contribution: xr.DataArray,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    tau_hours: float | Mapping[str, float],
    site_amplitude_prior_scale: float,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    state_activity: StateActivity | None = None,
    output_dim: str = "nmeasure",
) -> Co2CachedSigmaModel:
    """Build the named CO2 cached-sigma graph in scientific order.

    The observation likelihood is a normalized cached ``Potential`` over the
    active physical flux state. The returned target owns exact joint
    likelihood and predictive evaluation after sampling; no independent
    pointwise likelihood is fabricated in the graph.
    """
    if not np.isfinite(site_amplitude_prior_scale) or site_amplitude_prior_scale <= 0.0:
        raise ValueError("`site_amplitude_prior_scale` must be finite and strictly positive.")
    validate_observation_error_arrays(
        observations,
        observation_error,
        None,
        owner="Cached fixed-OU CO2 likelihood",
        output_dim=output_dim,
    )
    site_coord, site_index = _fixed_ou_site_alignment(
        observations,
        output_dim=output_dim,
    )
    site_labels = tuple(str(label) for label in site_coord.values)
    initial_amplitudes = _site_values(
        initial_site_amplitudes,
        site_coord=site_coord,
        default=float(site_amplitude_prior_scale),
    )
    prepared_flux = prepare_linear_sensitivity(
        flux_sensitivity,
        output_dim=output_dim,
    )
    activity = resolve_state_activity(prepared_flux.removed, state_activity)
    if activity.n_active == 0:
        raise ValueError("The cached-sigma recipe requires at least one active flux state.")
    active_state_prior = prepare_active_correlated_lognormal_prior(
        activity,
        retained_prior,
        var_name="flux_scaling",
    )
    assert active_state_prior is not None

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
        site_labels=site_labels,
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
    initial_cache = target.refresh(initial_amplitudes)
    shared_cache = PytensorMarginalQuadraticCache(initial_cache)

    with registered_model() as model:
        state_result = _add_prepared_correlated_lognormal_state_with_activity(
            activity,
            active_state_prior,
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
        add_coords({OU_SITE_DIM: site_coord})
        add_model_data(observations.transpose(output_dim), "Y")
        add_model_data(observation_error.transpose(output_dim), "error")
        add_model_data(site_index)
        add_model_data(
            xr.DataArray(
                covariance.tau_hours_by_site,
                dims=(OU_SITE_DIM,),
                coords={OU_SITE_DIM: site_coord},
                name="ou_tau_hours",
            )
        )
        amplitude = pm.HalfNormal(
            OU_SITE_AMPLITUDE,
            sigma=float(site_amplitude_prior_scale),
            initval=pm.floatX(initial_amplitudes),
            dims=OU_SITE_DIM,
        )
        pm.Deterministic(
            "epsilon",
            pt.sqrt(covariance.marginal_variance(amplitude)),
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
        initial_cache=initial_cache,
        amplitude=cast(TensorVariable, model[OU_SITE_AMPLITUDE]),
        state=state_result.latent,
        state_value_name=state_value_name,
        active_state_prior=active_state_prior,
        state_output_name=active_state_name,
        site_amplitude_prior_scale=float(site_amplitude_prior_scale),
    )


__all__ = [
    "OU_SITE_AMPLITUDE",
    "OU_SITE_DIM",
    "OU_SITE_INDEX",
    "Co2CachedSigmaModel",
    "build_co2_cached_sigma_model",
]
