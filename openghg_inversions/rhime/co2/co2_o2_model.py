"""Explicit PyMC graph for the CO2/O2 recipe."""

from __future__ import annotations

import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models import (
    PreparedLinearSensitivity,
    ResolvedStateActivity,
    StateActivity,
    add_coherent_affine_component,
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


def _prepare_shared_state_channel_sensitivities(
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
) -> tuple[
    PreparedLinearSensitivity,
    PreparedLinearSensitivity,
    PreparedLinearSensitivity,
]:
    """Prepare two unequal observation axes against one shared state mask.

    The channel operators are temporarily row-stacked so zero-column removal
    sees their joint state support. A state is removed only when its column is
    zero in both channels. The two returned channel sensitivities therefore
    share one removal map and one retained-state coordinate, while their
    original, potentially unequal CO2 and O2 observation axes are restored.

    Args:
        co2_operator: CO2 sensitivity with its channel observation dimension
            first and the shared retained-state dimension second.
        o2_operator: O2 sensitivity with its distinct channel observation
            dimension first and the same retained-state dimension second.

    Returns:
        The prepared joint sensitivity followed by prepared CO2 and O2
        sensitivities. The joint value owns the shared zero-column mask; the
        channel values retain their original row dimensions and labels.
    """
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
        joint,
        PreparedLinearSensitivity(co2_sensitivity, joint.removed, co2_dim),
        PreparedLinearSensitivity(o2_sensitivity, joint.removed, o2_dim),
    )


def _resolve_activity(
    sensitivity: PreparedLinearSensitivity,
    state_activity: StateActivity | None,
) -> ResolvedStateActivity:
    """Resolve activity once from the shared two-channel sensitivity contract."""
    return resolve_state_activity(sensitivity.removed, state_activity)


def co2_o2_prior_forward_mean(
    *,
    fixed_prior_contribution: xr.DataArray,
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
    retained_prior: CorrelatedLognormalPrior,
    state_activity: StateActivity | None = None,
    output_dim: str = "observation",
) -> xr.DataArray:
    """Evaluate the coherent affine CO2/O2 model at its resolved prior state.

    Args:
        fixed_prior_contribution: Joint coherent-reduction intercept on
            ``output_dim``.
        co2_operator: CO2 retained-state operator with its own observation
            dimension and the retained-prior state dimension.
        o2_operator: O2 retained-state operator with a distinct observation
            dimension and the same retained-prior state dimension.
        retained_prior: Correlated retained-state arithmetic moments.
        state_activity: Optional labelled policy fixing or activating retained
            states. Structural zero columns are inactive in both channels.
        output_dim: Dimension of the returned joint observation vector.

    Returns:
        The labelled joint prior-forward concentration
        ``fixed_prior_contribution + H_joint @ prior_state``.

    Raises:
        ValueError: If the channel operators cannot form one shared-state
            sensitivity or ``state_activity`` does not align with that state.
    """
    joint_sensitivity, _, _ = _prepare_shared_state_channel_sensitivities(
        co2_operator,
        o2_operator,
    )
    activity = _resolve_activity(joint_sensitivity, state_activity)
    prior_state = xr.where(
        activity.active,
        retained_prior.mean,
        activity.fixed_value,
    )
    state_dim = retained_prior.state_dim
    sensitivity_state_dim = str(joint_sensitivity.sensitivity.dims[1])
    retained_state = prior_state.isel({state_dim: joint_sensitivity.retained_indices})
    if sensitivity_state_dim != state_dim:
        retained_state = retained_state.rename({state_dim: sensitivity_state_dim})
    joint_contribution = xr.dot(
        joint_sensitivity.sensitivity,
        retained_state,
        dim=sensitivity_state_dim,
    ).rename({joint_sensitivity.output_dim: output_dim})
    joint_contribution = joint_contribution.assign_coords(
        {output_dim: fixed_prior_contribution[output_dim]}
    )
    return (fixed_prior_contribution + joint_contribution).rename(
        "prior_forward_concentration"
    )


def _add_co2_o2_affine_signal(
    co2_sensitivity: PreparedLinearSensitivity,
    o2_sensitivity: PreparedLinearSensitivity,
    state: TensorVariable,
    /,
    *,
    fixed_prior_contribution: xr.DataArray,
) -> TensorVariable:
    """Add the affine CO2/O2 signal from one shared flux-scaling state."""
    co2_signal = add_linked_linear_component(
        co2_sensitivity,
        state,
        data_name="co2_operator",
        output_name="co2_flux_contribution",
    )
    # Preparation declares that this operator already contains the fixed,
    # signed O2:CO2 oxidation ratios; do not multiply them again.
    o2_signal = add_linked_linear_component(
        o2_sensitivity,
        state,
        data_name="o2_operator",
        output_name="o2_flux_contribution",
    )
    joint_signal = pt.concatenate((co2_signal, o2_signal))
    return add_coherent_affine_component(
        fixed_prior_contribution,
        joint_signal,
        output_name="modelled_concentration",
    )


def _add_co2_o2_fixed_error_likelihood(
    observations: xr.DataArray,
    modelled_concentration: TensorVariable,
    /,
    *,
    independent_error_sd: xr.DataArray,
    aggregation_error: AggregationError,
    output_dim: str,
) -> None:
    """Add the fixed-error joint likelihood for both observation channels."""
    observed = add_model_data(observations, "observed_concentration")
    fixed_error = add_model_data(independent_error_sd, "fixed_independent_error_sd")
    registered_aggregation_error = add_aggregation_error_data(
        aggregation_error,
        observations,
        output_dim=output_dim,
    )
    add_gaussian_observation_likelihood(
        observed=observed,
        mean=modelled_concentration,
        independent_variance=fixed_error**2,
        aggregation_error=registered_aggregation_error,
        output_dim=output_dim,
    )


def build_co2_o2_model(
    *,
    observations: xr.DataArray,
    fixed_prior_contribution: xr.DataArray,
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
    aggregation_error: AggregationError,
    retained_prior: CorrelatedLognormalPrior,
    independent_error_sd: xr.DataArray,
    state_activity: StateActivity | None = None,
    output_dim: str = "observation",
) -> pm.Model:
    """Build the shared-state CO2/O2 affine model and fixed-error likelihood.

    ``fixed_prior_contribution``, both channel operators, the retained prior,
    and every block of ``aggregation_error`` must be products of the same
    coherent reduction. CO2 and O2 retain separate operator row axes because
    their numeric units are declared independently for every row. The
    Verification Games replay uses ppm for both channels; other scientific O2
    products may, for example, use per meg. Operator rows must already be
    expressed in their declared channel units per dimensionless flux scaling;
    covariance blocks use the corresponding row-by-column unit products,
    including the CO2/O2 cross-covariance.

    The O2 operator unconditionally embeds the fixed, signed O2-per-CO2
    oxidation ratios declared by CO2/O2 preparation. This builder therefore
    passes the raw shared state to the linked-sensitivity component and must
    not multiply those ratios again. A recipe using a ratio-free O2 operator
    would instead visibly construct ``o2_state = oxidation_ratio * co2_state``
    immediately before passing ``o2_state`` to that component.

    The prepared-input runner validates the external ``independent_error_sd``
    as finite and positive. Direct custom callers own that check, exact label
    alignment, row-unit consistency, coherent covariance provenance, and the
    embedded-ratio contract before calling this lower-level builder.

    Args:
        observations: Joint CO2-then-O2 observation vector on ``output_dim``,
            with row-wise tracer and unit coordinates.
        fixed_prior_contribution: Joint coherent affine intercept on the same
            observation axis and in the corresponding row units.
        co2_operator: CO2 operator with its channel observation axis first and
            the retained-prior state axis second.
        o2_operator: O2 operator with its distinct channel observation axis
            first and the same state axis; its shared-state columns already
            embed the fixed oxidation ratios.
        aggregation_error: Joint fixed aggregation covariance, including its
            cross-channel block.
        retained_prior: Correlated arithmetic-moment prior for the one state
            vector shared by both channels.
        independent_error_sd: Joint per-row independent standard deviations in
            each observation row's native units.
        state_activity: Optional labelled policy fixing or activating retained
            states.
        output_dim: Joint observation dimension used by the likelihood.

    Returns:
        A registered PyMC model containing the shared state, separate channel
        linear signals, coherent affine intercept, and joint Gaussian
        likelihood.

    Raises:
        ValueError: If shared-state sensitivity preparation, activity
            resolution, or registered coordinate alignment fails.
    """
    joint_sensitivity, co2_sensitivity, o2_sensitivity = (
        _prepare_shared_state_channel_sensitivities(co2_operator, o2_operator)
    )
    activity = _resolve_activity(joint_sensitivity, state_activity)

    with registered_model() as model:
        state = add_correlated_lognormal_state_with_activity(
            activity,
            retained_prior,
            var_name="flux_scaling",
        ).state
        modelled = _add_co2_o2_affine_signal(
            co2_sensitivity,
            o2_sensitivity,
            state,
            fixed_prior_contribution=fixed_prior_contribution,
        )
        _add_co2_o2_fixed_error_likelihood(
            observations,
            modelled,
            independent_error_sd=independent_error_sd,
            aggregation_error=aggregation_error,
            output_dim=output_dim,
        )
    return model
