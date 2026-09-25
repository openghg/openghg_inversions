"""Linked CO2/O2 affine graph using the CO2 fixed-OU quadratic cache."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models import (
    StateActivity,
    add_coherent_affine_component,
    apply_linear_sensitivity,
    prepare_linear_sensitivity,
    registered_model,
    resolve_state_activity,
)
from openghg_inversions.models.components import (
    LinearComponentResult,
    _add_prepared_correlated_lognormal_state_with_activity,
    prepare_active_correlated_lognormal_prior,
)
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.fixed_ou import prepare_fixed_ou_low_rank
from openghg_inversions.observation_error import (
    AggregationError,
    aggregation_error_as_low_rank,
    validate_observation_error_arrays,
)

from .co2_cached_sigma_model import (
    Co2CachedSigmaModel,
    _CachedAffineTerm,
    _add_cached_likelihood,
    _materialize_cached_linear_projection,
    _site_values,
)
from .co2_o2_fixed_ou import linked_fixed_ou_alignment
from .co2_o2_model import _gather_co2_o2_sensitivity, _add_co2_o2_baseline_components


def build_co2_o2_cached_sigma_model(
    *,
    observations: xr.DataArray,
    fixed_prior_contribution: xr.DataArray,
    co2_sensitivity: xr.DataArray,
    o2_sensitivity: xr.DataArray,
    aggregation_error: AggregationError,
    retained_prior: CorrelatedLognormalPrior,
    independent_error_sd: xr.DataArray,
    tau_hours: float | Mapping[str, float],
    site_amplitude_prior_scale: float,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    state_activity: StateActivity | None = None,
    boundary_sensitivity: Mapping[str, xr.DataArray] | None = None,
    bc_prior: Mapping[str, PriorArgs] | None = None,
    bc_state_activity: Mapping[str, StateActivity] | None = None,
    offset_prior: Mapping[str, PriorArgs] | None = None,
    offset_args: Mapping[str, Mapping[str, Any]] | None = None,
    output_dim: str = "observation",
) -> Co2CachedSigmaModel:
    """Build one joint fixed-OU target and its matched cached affine graph.

    Inputs have the same coherent-reduction, optional channel-baseline, and
    embedded O2-ratio contract as :func:`build_co2_o2_model`. Both channels must
    use the same concentration unit. OU blocks are independent by
    ``(species, site)`` while the prepared aggregation covariance retains all
    cross-channel entries. ``tau_hours`` is fixed, in hours; mappings use
    ``co2:SITE``/``o2:SITE`` keys. Amplitudes have independent HalfNormal priors
    with ``site_amplitude_prior_scale`` in concentration units. Optional positive
    initial amplitudes use the same scalar or species/site mapping convention.
    The returned graph must be sampled with the CO2 sigma-then-state
    CompoundStep, never stock NUTS alone.
    """
    if not np.isfinite(site_amplitude_prior_scale) or site_amplitude_prior_scale <= 0:
        raise ValueError("site_amplitude_prior_scale must be finite and strictly positive.")
    validate_observation_error_arrays(
        observations, independent_error_sd, None,
        owner="Linked cached fixed-OU", output_dim=output_dim,
    )
    alignment = linked_fixed_ou_alignment(observations)
    site_coord = alignment.site_labels.rename({"nsigma_site": "ou_site"}).rename("ou_site")
    site_index = alignment.site_index.rename("ou_site_index")
    initial_amplitudes = _site_values(
        initial_site_amplitudes,
        site_coord=site_coord,
        default=site_amplitude_prior_scale,
    )
    joint = _gather_co2_o2_sensitivity(co2_sensitivity, o2_sensitivity, output_dim=output_dim)
    joint = joint.sel({output_dim: observations[output_dim]})
    prepared_flux = prepare_linear_sensitivity(joint, output_dim=output_dim)
    activity = resolve_state_activity(prepared_flux.removed, state_activity)
    prepared_flux, active_design, fixed_flux = _materialize_cached_linear_projection(
        prepared_flux, activity, output_dim=output_dim,
    )
    active_prior = prepare_active_correlated_lognormal_prior(
        activity, retained_prior, var_name="flux_scaling",
    )
    factor, diagonal = aggregation_error_as_low_rank(aggregation_error)
    covariance = prepare_fixed_ou_low_rank(
        factor,
        diagonal + np.square(independent_error_sd.values),
        observations.time.values,
        site_index.values,
        tau_hours,
        site_labels=tuple(str(label) for label in site_coord.values),
    )
    with registered_model() as model:
        state = _add_prepared_correlated_lognormal_state_with_activity(
            activity, active_prior, var_name="flux_scaling",
        )
        flux = apply_linear_sensitivity(
            prepared_flux, state.state,
            data_name="co2_o2_sensitivity", output_name="co2_o2_flux_contribution",
        )
        affine = add_coherent_affine_component(
            fixed_prior_contribution, flux, output_name="co2_o2_affine_contribution",
        )
        terms = [_CachedAffineTerm(
            fixed_contribution=np.asarray(fixed_prior_contribution.values) + fixed_flux,
            active_design=active_design,
            coefficients=state.state[activity.active_indices] if activity.n_active else None,
            sampled_rvs=(state.latent,) if state.latent is not None else (),
            output=affine,
        )]
        baselines = _add_co2_o2_baseline_components(
            observations=observations, co2_sensitivity=co2_sensitivity,
            o2_sensitivity=o2_sensitivity, boundary_sensitivity=boundary_sensitivity,
            bc_prior=bc_prior, bc_state_activity=bc_state_activity,
            offset_prior=offset_prior, offset_args=offset_args,
        )
        for _, result, design in baselines:
            matrix = np.asarray(design.values, dtype=np.float64)
            if isinstance(result, LinearComponentResult):
                active = np.asarray(result.activity.active.values, dtype=bool)
                fixed = matrix[:, ~active] @ result.activity.fixed_value.values[~active]
                coefficients = result.state[result.activity.active_indices] if active.any() else None
                active_matrix = matrix[:, active]
            else:
                fixed = np.zeros(observations.size)
                coefficients = result.coefficients
                active_matrix = matrix
            terms.append(_CachedAffineTerm(
                fixed_contribution=fixed, active_design=active_matrix,
                coefficients=coefficients,
                sampled_rvs=(result.latent,) if result.latent is not None else (),
                output=result.output,
            ))
        mean = pm.Deterministic(
            "modelled_concentration", sum(term.output for term in terms), dims=output_dim,
        )
        cached = _add_cached_likelihood(
            tuple(terms), covariance=covariance, observations=observations,
            observation_error=independent_error_sd, site_coord=site_coord,
            site_index=site_index, initial_amplitudes=initial_amplitudes,
            site_amplitude_prior_scale=site_amplitude_prior_scale, output_dim=output_dim,
        )
    return Co2CachedSigmaModel(
        model=model, target=cached.target, shared_cache=cached.shared_cache,
        initial_cache=cached.initial_cache, amplitude=cached.amplitude,
        states=cached.states, modelled_mean=mean,
        site_amplitude_prior_scale=site_amplitude_prior_scale,
    )
