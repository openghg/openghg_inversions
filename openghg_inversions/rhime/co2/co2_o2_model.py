"""Explicit PyMC graph for the CO2/O2 recipe."""

from __future__ import annotations

import pymc as pm
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.array_ops import concat_gather_data_arrays
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models import (
    PreparedLinearSensitivity,
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


def _gather_co2_o2_operator(
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
    *,
    output_dim: str,
) -> xr.DataArray:
    """Gather unequal CO2 and O2 rows onto one labelled observation axis."""
    channel_dim = "channel_observation"
    return concat_gather_data_arrays(
        {
            "co2": co2_operator.rename({co2_operator.dims[0]: channel_dim}),
            "o2": o2_operator.rename({o2_operator.dims[0]: channel_dim}),
        },
        key_dim="species",
        ragged_dim=channel_dim,
        stack_dim=output_dim,
        join="exact",
    )


def _prepare_shared_state_sensitivity(
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
    *,
    output_dim: str = "observation",
) -> PreparedLinearSensitivity:
    """Prepare unequal CO2/O2 rows against one shared state mask.

    The gathered ``(species, channel_observation)`` index retains each native
    channel axis. A state is removed only when its column is zero in both
    channels.

    Args:
        co2_operator: CO2 sensitivity with its channel observation dimension
            first and the shared retained-state dimension second.
        o2_operator: O2 sensitivity with its distinct channel observation
            dimension first and the same retained-state dimension second.
        output_dim: Name for the gathered observation dimension.

    Returns:
        One gathered sensitivity and its shared zero-column mask.
    """
    return prepare_linear_sensitivity(
        _gather_co2_o2_operator(
            co2_operator,
            o2_operator,
            output_dim=output_dim,
        ),
        output_dim=output_dim,
    )


def evaluate_co2_o2_prior_forward_mean(
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
    joint_operator = _gather_co2_o2_operator(
        co2_operator,
        o2_operator,
        output_dim=output_dim,
    )
    joint_sensitivity = prepare_linear_sensitivity(
        joint_operator,
        output_dim=output_dim,
    )
    activity = resolve_state_activity(joint_sensitivity.removed, state_activity)
    # Active states are evaluated at their arithmetic prior means; inactive
    # states retain the exact values declared by the activity policy.
    resolved_prior_state = retained_prior.mean.where(
        activity.active,
        activity.fixed_value,
    )
    joint_contribution = xr.dot(
        joint_operator,
        resolved_prior_state,
        dim=retained_prior.state_dim,
    )
    return (fixed_prior_contribution + joint_contribution).rename(
        "prior_forward_concentration"
    )


def _add_co2_o2_affine_signal(
    joint_sensitivity: PreparedLinearSensitivity,
    state: TensorVariable,
    /,
    *,
    fixed_prior_contribution: xr.DataArray,
) -> TensorVariable:
    """Add the affine CO2/O2 signal from one shared flux-scaling state."""
    # Preparation declares that this operator already contains the fixed,
    # signed O2:CO2 oxidation ratios; do not multiply them again.
    joint_signal = add_linked_linear_component(
        joint_sensitivity,
        state,
        data_name="co2_o2_operator",
        output_name="co2_o2_flux_contribution",
    )
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
    oxidation ratios declared by CO2/O2 preparation. The two channel operators
    are therefore gathered and applied to the raw shared state once. A recipe
    starting from a ratio-free O2 operator would need to apply its labelled
    species/state ratio factor before constructing that joint sensitivity.

    The prepared-input runner validates the external ``independent_error_sd``
    as finite and positive. Direct custom callers own that check, exact label
    alignment, row-unit consistency, coherent covariance provenance, and the
    embedded-ratio contract before calling this lower-level builder.

    Args:
        observations: Joint CO2-then-O2 observation vector on ``output_dim``,
            with row-wise species and unit coordinates.
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
        A registered PyMC model containing the shared state, gathered joint
        linear signal, coherent affine intercept, and joint Gaussian likelihood.

    Raises:
        ValueError: If shared-state sensitivity preparation, activity
            resolution, or registered coordinate alignment fails.
    """
    joint_sensitivity = _prepare_shared_state_sensitivity(
        co2_operator,
        o2_operator,
        output_dim=output_dim,
    )
    activity = resolve_state_activity(joint_sensitivity.removed, state_activity)

    with registered_model() as model:
        state = add_correlated_lognormal_state_with_activity(
            activity,
            retained_prior,
            var_name="flux_scaling",
        ).state
        modelled = _add_co2_o2_affine_signal(
            joint_sensitivity,
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
