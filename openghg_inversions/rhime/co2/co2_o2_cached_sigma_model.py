"""Linked CO2/O2 affine graph using the CO2 fixed-OU quadratic cache."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dask import compute as dask_compute
import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.boundary_sensitivity import bind_boundary_sensitivity
from openghg_inversions.array_ops import select_gathered_data_array, to_dense
from openghg_inversions.inversion_inputs import DatetimeLike
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
    AffineTerm,
    LinearComponentResult,
    OffsetComponentResult,
    add_model_data,
    likelihood_representation,
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
    _centered_cached_term,
    _site_values,
)
from .centered_boundary import bind_boundary_operands, require_centered_precision
from .co2_o2_fixed_ou import linked_fixed_ou_alignment
from .co2_o2_model import (
    _gather_co2_o2_sensitivity,
    _add_co2_o2_baseline_components,
    _bind_channel_corrections,
    _validate_channel_baseline_options,
)


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
    site_amplitude_prior: PriorArgs,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    state_activity: StateActivity | None = None,
    boundary_sensitivity: Mapping[str, xr.DataArray] | None = None,
    boundary_operands: Mapping[str, xr.Dataset] | None = None,
    bc_freq: str | None = None,
    bc_anchor_time: DatetimeLike | None = None,
    bc_mean_shift_prior_co2: PriorArgs | None = None,
    bc_mean_shift_prior_o2: PriorArgs | None = None,
    bc_anomaly_scale_co2: float | None = None,
    bc_anomaly_scale_o2: float | None = None,
    boundary_correction_sensitivity_co2: xr.DataArray | None = None,
    boundary_correction_sensitivity_o2: xr.DataArray | None = None,
    bc_centering_weights_co2: xr.DataArray | None = None,
    bc_centering_weights_o2: xr.DataArray | None = None,
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
    ``co2:SITE``/``o2:SITE`` keys. Amplitudes have independent positive priors
    specified by ``site_amplitude_prior`` in concentration units. Optional positive
    initial amplitudes use the same scalar or species/site mapping convention.
    The returned graph must be sampled with the CO2 sigma-then-state
    CompoundStep and its explicit cached state function. The observed joint
    Gaussian remains exact independently of that cache. Scientific initial
    values are returned separately for sampling; observations, designs,
    activity masks, and covariance stay fixed after construction.

    ``bc_mean_shift_prior_co2/o2`` and ``bc_anomaly_scale_co2/o2`` select
    independent channel additive corrections in concentration units, replacing
    that channel's multiplicative prior/activity. ``boundary_correction_sensitivity_co2/o2``
    provide dimensionless unit responses (centered operands default to their
    explicit mass); ``bc_centering_weights_co2/o2`` supply positive dimensionless
    weights on the complete resolved curtain/period labels. Their weighted
    anomaly means are zero; the numerical reference never scales corrections.
    Each anomaly scale must be finite and nonnegative; zero retains only that
    channel's common mean without sampled contrast variables.

    Raw centered ``boundary_operands`` require explicit PyTensor float64 and
    are aligned and expanded by ``bc_freq``/``bc_anchor_time`` here. Channel
    boundary states retain their independent labels and physical priors. The
    likelihood and sampler use directly centered means; named concentration
    outputs remain absolute. Raw predictive ``y`` requires absolute restoration.
    """
    _validate_channel_baseline_options(
        boundary_sensitivity,
        bc_prior,
        bc_state_activity,
        offset_prior,
        offset_args,
        boundary_operands,
        bc_mean_shift_prior_co2=bc_mean_shift_prior_co2,
        bc_mean_shift_prior_o2=bc_mean_shift_prior_o2,
        bc_anomaly_scale_co2=bc_anomaly_scale_co2,
        bc_anomaly_scale_o2=bc_anomaly_scale_o2,
    )
    if boundary_operands:
        require_centered_precision()
        aligned = {}
        for channel, operands in boundary_operands.items():
            sensitivity = co2_sensitivity if channel == "co2" else o2_sensitivity
            native = select_gathered_data_array(
                observations,
                key=channel,
                key_dim="species",
                ragged_dim="channel_observation",
                stack_dim=output_dim,
            ).rename({output_dim: str(sensitivity.dims[0])})
            aligned[channel] = bind_boundary_operands(
                operands,
                native,
                frequency=bc_freq,
                anchor_time=bc_anchor_time,
            )
        # Direct builders own this eager boundary too; runners may already supply eager arrays.
        (
            observations,
            fixed_prior_contribution,
            co2_sensitivity,
            o2_sensitivity,
            independent_error_sd,
            boundary_operands,
            boundary_sensitivity,
        ) = dask_compute(
            to_dense(observations),
            to_dense(fixed_prior_contribution),
            to_dense(co2_sensitivity),
            to_dense(o2_sensitivity),
            to_dense(independent_error_sd),
            {
                channel: xr.Dataset(
                    {name: to_dense(value).astype("float64") for name, value in operands.data_vars.items()}
                )
                for channel, operands in aligned.items()
            },
            {channel: to_dense(value) for channel, value in (boundary_sensitivity or {}).items()},
        )
    if boundary_sensitivity:
        aligned_boundaries = {}
        for channel, sensitivity in boundary_sensitivity.items():
            channel_sensitivity = co2_sensitivity if channel == "co2" else o2_sensitivity
            native = select_gathered_data_array(
                observations,
                key=channel,
                key_dim="species",
                ragged_dim="channel_observation",
                stack_dim=output_dim,
            ).rename({output_dim: str(channel_sensitivity.dims[0])})
            aligned_boundaries[channel] = bind_boundary_sensitivity(
                sensitivity,
                native,
                frequency=bc_freq,
                anchor_time=bc_anchor_time,
            )
        boundary_sensitivity = aligned_boundaries
    if (
        not boundary_sensitivity
        and not boundary_operands
        and (bc_freq is not None or bc_anchor_time is not None)
    ):
        raise ValueError("Boundary period choices require boundary inputs.")
    boundary_correction_sensitivity_co2, boundary_correction_sensitivity_o2 = _bind_channel_corrections(
        observations,
        co2_sensitivity,
        o2_sensitivity,
        boundary_correction_sensitivity_co2 if bc_mean_shift_prior_co2 is not None else None,
        boundary_correction_sensitivity_o2 if bc_mean_shift_prior_o2 is not None else None,
        bc_freq=bc_freq,
        bc_anchor_time=bc_anchor_time,
        output_dim=output_dim,
        boundary_operands=boundary_operands,
    )
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
    active_prior = prepare_active_correlated_lognormal_prior(
        activity,
        retained_prior,
        var_name="flux_scaling",
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
    with registered_model():
        state = _add_prepared_correlated_lognormal_state_with_activity(
            activity,
            active_prior,
            var_name="flux_scaling",
        )
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
            _CachedAffineTerm(
                term=AffineTerm.absolute(affine),
                fixed_contribution=np.asarray(fixed_prior_contribution.values) + fixed_flux,
                active_design=active_design,
                coefficients=state.state[activity.active_indices] if activity.n_active else None,
                sampled_rvs=(state.latent,) if state.latent is not None else (),
            )
        ]
        baselines = _add_co2_o2_baseline_components(
            observations=observations,
            co2_sensitivity=co2_sensitivity,
            o2_sensitivity=o2_sensitivity,
            boundary_sensitivity=boundary_sensitivity,
            boundary_operands=boundary_operands,
            bc_mean_shift_prior_co2=bc_mean_shift_prior_co2,
            bc_mean_shift_prior_o2=bc_mean_shift_prior_o2,
            bc_anomaly_scale_co2=bc_anomaly_scale_co2,
            bc_anomaly_scale_o2=bc_anomaly_scale_o2,
            boundary_correction_sensitivity_co2=boundary_correction_sensitivity_co2,
            boundary_correction_sensitivity_o2=boundary_correction_sensitivity_o2,
            bc_centering_weights_co2=bc_centering_weights_co2,
            bc_centering_weights_o2=bc_centering_weights_o2,
            bc_prior=bc_prior,
            bc_state_activity=bc_state_activity,
            offset_prior=offset_prior,
            offset_args=offset_args,
        )
        for baseline in baselines:
            result, design = baseline.component, baseline.design
            if baseline.additive is not None:
                assert baseline.additive_fixed is not None and baseline.additive_design is not None
                terms.append(
                    _CachedAffineTerm(
                        term=baseline.term,
                        fixed_contribution=baseline.additive_fixed,
                        active_design=baseline.additive_design,
                        coefficients=baseline.additive.coefficients,
                        sampled_rvs=baseline.additive.latents,
                    )
                )
                continue
            if baseline.centered is not None:
                terms.append(_centered_cached_term(baseline.centered, output_dim=output_dim))
                continue
            if isinstance(result, LinearComponentResult):
                assert design is not None
                matrix = np.asarray(design.values, dtype=np.float64)
                active = np.asarray(result.activity.active.values, dtype=bool)
                fixed = matrix[:, ~active] @ result.activity.fixed_value.values[~active]
                coefficients = result.state[result.activity.active_indices] if active.any() else None
                active_matrix = matrix[:, active]
            else:
                assert isinstance(result, OffsetComponentResult) and design is not None
                matrix = np.asarray(design.values, dtype=np.float64)
                fixed = np.zeros(observations.size)
                coefficients = result.coefficients
                active_matrix = matrix
            terms.append(
                _CachedAffineTerm(
                    term=baseline.term,
                    fixed_contribution=fixed,
                    active_design=active_matrix,
                    coefficients=coefficients,
                    sampled_rvs=(result.latent,) if result.latent is not None else (),
                )
            )
        affine_terms = [term.term for term in terms]
        mean = pm.Deterministic(
            "modelled_concentration",
            sum(term.output for term in affine_terms),
            dims=output_dim,
        )
        likelihood_mean, likelihood_observations, shift = likelihood_representation(
            affine_terms,
            observations,
            mean,
        )
        if shift is not None:
            add_model_data(shift.rename("boundary_reference_shift"))
        return _add_cached_likelihood(
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
        )
