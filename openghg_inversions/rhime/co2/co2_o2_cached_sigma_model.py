"""Linked CO2/O2 affine graph using the CO2 fixed-OU quadratic cache."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from dask import compute as dask_compute
import pymc as pm
import xarray as xr

from openghg_inversions.models.biosphere import (
    RetainedFluxPrior,
    add_prepared_retained_flux_state,
    prepare_retained_flux_prior,
)
from openghg_inversions.array_ops import to_dense_tree
from openghg_inversions.models import (
    StateActivity,
    add_coherent_affine_component,
    apply_linear_sensitivity,
    prepare_linear_sensitivity,
    registered_model,
    resolve_state_activity,
)
from openghg_inversions.models.boundary import Bound
from openghg_inversions.models.components import (
    AffineBlock,
    AffineConcentrationTerm,
    likelihood_representation,
)
from openghg_inversions.models.fixed_ou import prepare_fixed_ou_low_rank
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.observation_error import (
    AggregationError,
    aggregation_error_as_low_rank,
    validate_observation_error_arrays,
)

from .co2_cached_sigma_model import (
    Co2CachedSigmaModel,
    _add_cached_likelihood,
    _materialize_cached_linear_projection,
    _site_values,
)
from .co2_o2_fixed_ou import linked_fixed_ou_alignment
from .co2_o2_model import (
    _add_co2_o2_baseline_components,
    _gather_co2_o2_sensitivity,
)


def build_co2_o2_cached_sigma_model(
    *,
    observations: xr.DataArray,
    fixed_prior_contribution: xr.DataArray,
    co2_sensitivity: xr.DataArray,
    o2_sensitivity: xr.DataArray,
    aggregation_error: AggregationError,
    retained_prior: RetainedFluxPrior,
    independent_error_sd: xr.DataArray,
    tau_hours: float | Mapping[str, float],
    site_amplitude_prior: PriorArgs,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    state_activity: StateActivity | None = None,
    boundary: Mapping[str, Bound] | None = None,
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
    ``co2:SITE``/``o2:SITE`` keys. Amplitudes have independent positive priors
    specified by ``site_amplitude_prior`` in concentration units. Optional positive
    initial amplitudes use the same scalar or species/site mapping convention.
    The returned graph must be sampled with the CO2 sigma-then-state
    CompoundStep and its explicit cached state function. The observed joint
    Gaussian remains exact independently of that cache. Scientific initial
    values are returned separately for sampling; observations, designs,
    activity masks, and covariance stay fixed after construction.

    Each channel consumes its previously bound boundary, including resolved
    periods, priors and activity. Centered likelihoods retain absolute named
    concentration outputs; raw predictive ``y`` needs absolute restoration.

    Args:
        observations: Joint CO2-then-O2 observation vector on ``output_dim``,
            with row-wise species and unit coordinates.
        fixed_prior_contribution: Joint coherent affine intercept on the same
            observation axis and in the corresponding row units.
        co2_sensitivity: CO2 sensitivity with its channel observation axis first and
            the retained-prior state axis second.
        o2_sensitivity: O2 sensitivity with its distinct channel observation axis
            first and the same state axis; its shared-state columns already
            embed the fixed oxidation ratios.
        aggregation_error: Joint fixed aggregation covariance, including its
            cross-channel block.
        retained_prior: Correlated arithmetic-moment prior for the one state
            vector shared by both channels, or a biosphere calibration of it
            already checked by
            :func:`~openghg_inversions.models.biosphere.resolve_retained_flux_prior`.
        independent_error_sd: Joint per-row independent standard deviations in
            each observation row's native units.
        state_activity: Optional labelled policy fixing or activating retained
            states.
        boundary: Resolved boundary component, keyed by channel.
        offset_prior: Channel-keyed offset priors. Omitted channels have no offset.
        offset_args: Per-channel offset_freq, drop_first, and per_site options,
            with the same meanings as the CO2 offset component. Baseline terms
            require identical channel units and are zero on the other channel.
        output_dim: Joint observation dimension used by the likelihood.
        tau_hours: Fixed OU timescale in hours, scalar or mapping keyed
            by ``co2:SITE``/``o2:SITE``. Requires common channel units.
        site_amplitude_prior: Prior for inferred species/site OU amplitudes in
            concentration units.

        initial_site_amplitudes: Positive initial amplitudes, as a scalar or
            species/site mapping, returned separately for sampling.

    Returns:
        A registered joint PyMC model, fixed-OU target, matched cache, affine
        coefficient variables and sampling initial values. Use the paired
        sigma-then-state CompoundStep to keep the state cache current.

    Raises:
        ValueError: If channel options, labels, units, activity, covariance,
            positive amplitudes or explicit double precision are invalid.
    """
    (observations, fixed_prior_contribution, co2_sensitivity, o2_sensitivity, independent_error_sd), boundary, priors = (
        dask_compute(*to_dense_tree((
            (observations, fixed_prior_contribution, co2_sensitivity, o2_sensitivity, independent_error_sd),
            dict(boundary or {}), {"amplitude": site_amplitude_prior, "offset": offset_prior},
        )))
    )
    site_amplitude_prior, offset_prior = priors["amplitude"], priors["offset"]
    return _build_co2_o2_cached_sigma_model(
        observations=observations,
        fixed_prior_contribution=fixed_prior_contribution,
        co2_sensitivity=co2_sensitivity,
        o2_sensitivity=o2_sensitivity,
        aggregation_error=aggregation_error,
        retained_prior=retained_prior,
        independent_error_sd=independent_error_sd,
        tau_hours=tau_hours,
        site_amplitude_prior=site_amplitude_prior,
        initial_site_amplitudes=initial_site_amplitudes,
        state_activity=state_activity,
        boundary=boundary,
        offset_prior=offset_prior,
        offset_args=offset_args,
        output_dim=output_dim,
    )


def _build_co2_o2_cached_sigma_model(
    *,
    observations: xr.DataArray,
    fixed_prior_contribution: xr.DataArray,
    co2_sensitivity: xr.DataArray,
    o2_sensitivity: xr.DataArray,
    aggregation_error: AggregationError,
    retained_prior: RetainedFluxPrior,
    independent_error_sd: xr.DataArray,
    tau_hours: float | Mapping[str, float],
    site_amplitude_prior: PriorArgs,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    state_activity: StateActivity | None = None,
    boundary: Mapping[str, Bound] | None = None,
    offset_prior: Mapping[str, PriorArgs] | None = None,
    offset_args: Mapping[str, Mapping[str, Any]] | None = None,
    output_dim: str = "observation",
) -> Co2CachedSigmaModel:
    """Construct the linked cached fixed-OU graph from eager inputs.

    Materialization belongs to the caller: :func:`build_co2_o2_cached_sigma_model` computes its
    arguments together, and the runner calls this after its own joint
    materialization so already-eager arrays are not handled twice. Arguments,
    returns and errors are those of :func:`build_co2_o2_cached_sigma_model`.
    """
    validate_observation_error_arrays(
        observations,
        independent_error_sd,
        None,
        owner="Linked cached fixed-OU",
        output_dim=output_dim,
    )
    alignment = linked_fixed_ou_alignment(observations)
    site_coord = xr.DataArray(alignment.site_labels).rename({"nsigma_site": "ou_site"}).rename("ou_site")
    site_index = alignment.site_index.rename("ou_site_index")
    initial_amplitudes = _site_values(
        initial_site_amplitudes,
        site_coord=site_coord,
    )
    joint = _gather_co2_o2_sensitivity(co2_sensitivity, o2_sensitivity, output_dim=output_dim)
    joint = joint.sel({output_dim: observations[output_dim]})
    prepared_flux = prepare_linear_sensitivity(joint, output_dim=output_dim)
    activity = resolve_state_activity(prepared_flux.removed, state_activity)
    prepared_flux, active_design, fixed_flux = _materialize_cached_linear_projection(
        prepared_flux,
        activity,
        output_dim=output_dim,
    )
    active_prior = prepare_retained_flux_prior(activity, retained_prior, var_name="flux_scaling")
    factor, diagonal = aggregation_error_as_low_rank(aggregation_error)
    covariance = prepare_fixed_ou_low_rank(
        factor,
        diagonal + np.square(independent_error_sd.values),
        observations.time.values,
        site_index.values,
        tau_hours,
        site_labels=tuple(str(label) for label in site_coord.values),
    )
    with registered_model():
        state = add_prepared_retained_flux_state(activity, active_prior, var_name="flux_scaling")
        flux = apply_linear_sensitivity(
            prepared_flux,
            state.state,
            data_name="co2_o2_sensitivity",
            output_name="co2_o2_flux_contribution",
        )
        affine = add_coherent_affine_component(
            fixed_prior_contribution,
            flux,
            output_name="co2_o2_affine_contribution",
        )
        terms = [
            AffineConcentrationTerm.absolute(
                affine,
                fixed_contribution=fixed_prior_contribution + fixed_flux,
                blocks=(AffineBlock(active_design,
                                    state.state[activity.active_indices]),) if activity.n_active else (),
                sampled_rvs=(state.latent,) if state.latent is not None else (),
            )
        ]
        baselines = _add_co2_o2_baseline_components(
            observations=observations,
            co2_sensitivity=co2_sensitivity,
            o2_sensitivity=o2_sensitivity,
            boundary=boundary,
            offset_prior=offset_prior,
            offset_args=offset_args,
        )
        terms.extend(baselines)
        mean = pm.Deterministic(
            "modelled_concentration",
            sum(term.output for term in terms),
            dims=output_dim,
        )
        likelihood_mean, likelihood_observations = likelihood_representation(
            terms,
            observations,
            mean,
        )
        cached = _add_cached_likelihood(
            tuple(terms),
            covariance=covariance,
            modelled_mean=likelihood_mean,
            observations=likelihood_observations,
            observation_error=independent_error_sd,
            site_coord=site_coord,
            site_index=site_index,
            initial_amplitudes=initial_amplitudes,
            site_amplitude_prior=site_amplitude_prior,
            output_dim=output_dim,
            flux_prior=retained_prior,
        )
        return cached
