"""Explicit PyMC graph for the CO2/O2 recipe."""

from __future__ import annotations

import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models import (
    PreparedLinearSensitivity,
    ResolvedStateActivity,
    StateActivity,
    add_correlated_lognormal_state_with_activity,
    add_linked_linear_component,
    add_model_data,
    prepare_linear_sensitivity,
    registered_model,
    resolve_state_activity,
)
from openghg_inversions.models.likelihoods import (
    add_aggregation_error_data,
    add_gaussian_observation_likelihood,
)
from openghg_inversions.observation_error import AggregationError


def _prepare_channel_sensitivities(
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
) -> tuple[PreparedLinearSensitivity, PreparedLinearSensitivity]:
    """Inspect both channels once and retain their independent output axes."""
    co2_dim = str(co2_operator.dims[0])
    o2_dim = str(o2_operator.dims[0])
    joint_dim = "co2_o2_observation"
    nco2 = co2_operator.sizes[co2_dim]
    no2 = o2_operator.sizes[o2_dim]
    joint_operator = xr.concat(
        (
            co2_operator.rename({co2_dim: joint_dim}).assign_coords(
                {joint_dim: range(nco2)}
            ),
            o2_operator.rename({o2_dim: joint_dim}).assign_coords(
                {joint_dim: range(nco2, nco2 + no2)}
            ),
        ),
        dim=joint_dim,
    )
    joint = prepare_linear_sensitivity(joint_operator, output_dim=joint_dim)
    co2_sensitivity = joint.sensitivity.isel({joint_dim: slice(0, nco2)}).rename(
        {joint_dim: co2_dim}
    )
    co2_sensitivity = co2_sensitivity.assign_coords({co2_dim: co2_operator[co2_dim]})
    o2_sensitivity = joint.sensitivity.isel({joint_dim: slice(nco2, None)}).rename(
        {joint_dim: o2_dim}
    )
    o2_sensitivity = o2_sensitivity.assign_coords({o2_dim: o2_operator[o2_dim]})
    return (
        PreparedLinearSensitivity(co2_sensitivity, joint.removed, co2_dim),
        PreparedLinearSensitivity(o2_sensitivity, joint.removed, o2_dim),
    )


def _resolve_activity(
    sensitivity: PreparedLinearSensitivity,
    state_activity: StateActivity | None,
) -> ResolvedStateActivity:
    """Resolve activity once from the shared two-channel sensitivity contract."""
    return resolve_state_activity(sensitivity.removed, state_activity)


def _co2_o2_prior_forward_mean(
    *,
    prior_forward_mean: xr.DataArray,
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
    retained_prior: CorrelatedLognormalPrior,
    activity: ResolvedStateActivity,
    output_dim: str,
) -> xr.DataArray:
    """Apply one resolved activity policy to the channel prior signals."""
    prior_state = xr.where(
        activity.active,
        retained_prior.mean,
        activity.fixed_value,
    )
    state_adjustment = prior_state - retained_prior.mean
    nco2 = co2_operator.sizes[co2_operator.dims[0]]
    adjustments = []
    for operator, selection in (
        (co2_operator, slice(0, nco2)),
        (o2_operator, slice(nco2, None)),
    ):
        channel_dim = str(operator.dims[0])
        adjustment = xr.dot(
            operator,
            state_adjustment,
            dim=retained_prior.state_dim,
        ).rename({channel_dim: output_dim})
        adjustment = adjustment.assign_coords(
            {output_dim: prior_forward_mean[output_dim].isel({output_dim: selection})}
        )
        adjustments.append(adjustment)
    joint_adjustment = xr.concat(adjustments, dim=output_dim)
    return (prior_forward_mean + joint_adjustment).rename("prior_forward_concentration")


def co2_o2_prior_forward_mean(
    *,
    prior_forward_mean: xr.DataArray,
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
    retained_prior: CorrelatedLognormalPrior,
    state_activity: StateActivity | None = None,
    output_dim: str = "observation",
) -> xr.DataArray:
    """Return the deterministic prior concentration after exact state fixing."""
    co2_sensitivity, _ = _prepare_channel_sensitivities(co2_operator, o2_operator)
    activity = _resolve_activity(co2_sensitivity, state_activity)
    return _co2_o2_prior_forward_mean(
        prior_forward_mean=prior_forward_mean,
        co2_operator=co2_operator,
        o2_operator=o2_operator,
        retained_prior=retained_prior,
        activity=activity,
        output_dim=output_dim,
    )


def build_co2_o2_model(
    *,
    observations: xr.DataArray,
    prior_forward_mean: xr.DataArray,
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
    aggregation_error: AggregationError,
    retained_prior: CorrelatedLognormalPrior,
    independent_error_sd: xr.DataArray,
    state_activity: StateActivity | None = None,
    output_dim: str = "observation",
) -> pm.Model:
    """Build separate CO2/O2 signals and one fixed-error joint likelihood.

    All scientific inputs are explicit labelled values. The UOB prototype
    supplies one ppm as ``independent_error_sd`` for each channel; this builder
    does not encode that run policy and does not infer a mismatch amplitude.
    """
    co2_dim = str(co2_operator.dims[0])
    co2_sensitivity, o2_sensitivity = _prepare_channel_sensitivities(
        co2_operator,
        o2_operator,
    )
    activity = _resolve_activity(co2_sensitivity, state_activity)
    prior_state = xr.where(
        activity.active,
        retained_prior.mean,
        activity.fixed_value,
    ).rename("prior_flux_scaling")
    activity_prior_forward = _co2_o2_prior_forward_mean(
        prior_forward_mean=prior_forward_mean,
        co2_operator=co2_operator,
        o2_operator=o2_operator,
        retained_prior=retained_prior,
        activity=activity,
        output_dim=output_dim,
    )

    with registered_model() as model:
        state = add_correlated_lognormal_state_with_activity(
            activity,
            retained_prior,
            var_name="flux_scaling",
        ).state
        prior_state_data = add_model_data(prior_state)
        prior_forward_data = add_model_data(activity_prior_forward)
        state_anomaly = state - prior_state_data
        co2_adjustment = add_linked_linear_component(
            co2_sensitivity,
            state_anomaly,
            data_name="co2_operator",
            output_name="co2_flux_adjustment",
        )
        o2_adjustment = add_linked_linear_component(
            o2_sensitivity,
            state_anomaly,
            data_name="o2_operator",
            output_name="o2_flux_adjustment",
        )
        nco2 = co2_operator.sizes[co2_dim]
        modelled = pm.Deterministic(
            "modelled_concentration",
            pt.concatenate(
                (
                    prior_forward_data[:nco2] + co2_adjustment,
                    prior_forward_data[nco2:] + o2_adjustment,
                )
            ),
            dims=output_dim,
        )
        observed = add_model_data(observations, "observed_concentration")
        fixed_error = add_model_data(independent_error_sd, "fixed_independent_error_sd")
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
