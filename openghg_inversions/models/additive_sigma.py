r"""Observation error with an additive model-data-mismatch scale.

For reported observation-error variance :math:`s_y^2`, mismatch scale
:math:`\sigma`, and fixed aggregation covariance :math:`C_{agg}`, this
component constructs the independent variance

.. math::

   v = s_y^2 + \sigma^2

and applies ``min_error`` as a floor on the total marginal standard
deviation. ``build_additive_sigma_error`` exposes the reusable error state;
``add_additive_sigma_gaussian_likelihood`` combines that state with an
explicitly supplied modelled concentration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.components import add_model_data, add_sigma_component
from openghg_inversions.models.likelihoods import add_gaussian_observation_likelihood
from openghg_inversions.observation_error import (
    AggregationError,
    validate_aggregation_error_alignment,
    validate_observation_alignment,
    validate_observation_error_arrays,
)
from openghg_inversions.sigma import SigmaAlignment


@dataclass(frozen=True)
class AdditiveSigmaErrorState:
    """Terms required to construct an additive-sigma observation distribution."""

    observed: TensorVariable
    independent_variance: TensorVariable
    aggregation_error: AggregationError
    error_scale: TensorVariable


def build_additive_sigma_error(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray,
    aggregation_error: AggregationError,
    sigma_alignment: SigmaAlignment,
    sigma_prior: Mapping[str, Any],
    no_model_error: bool,
    output_dim: str = "nmeasure",
) -> AdditiveSigmaErrorState:
    """Build observation-error terms with an additive mismatch scale.

    ``sigma`` is observation-aligned through ``sigma_alignment`` and enters as
    an independent variance term. When ``no_model_error`` is true, no ``sigma``
    variable is constructed. Fixed diagonal, dense, or low-rank aggregation
    error is included only when explicitly selected by
    ``aggregation_error_mode``.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        minimum_error: Minimum total-error standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        sigma_alignment: Mapping from observations to the mismatch-scale
            parameters.
        sigma_prior: Prior arguments used to construct ``sigma`` when model
            error is enabled.
        no_model_error: If true, omit ``sigma`` and use only reported and
            selected aggregation errors.
        output_dim: Observation dimension used for named PyMC variables.

    Returns:
        Labelled observed data, independent variance, fixed aggregation error,
        and total marginal error scale.

    Raises:
        ValueError: If the observation or aggregation-error inputs are
            inconsistent with ``output_dim``.
    """
    validate_observation_error_arrays(
        observations,
        observation_error,
        minimum_error,
        owner="Additive-sigma likelihood",
        output_dim=output_dim,
    )
    validate_observation_alignment(
        observations,
        sigma_alignment.site_index,
        input_name="sigma_alignment.site_index",
        owner="Additive-sigma likelihood",
        output_dim=output_dim,
    )
    validate_observation_alignment(
        observations,
        sigma_alignment.period_index,
        input_name="sigma_alignment.period_index",
        owner="Additive-sigma likelihood",
        output_dim=output_dim,
    )
    validate_aggregation_error_alignment(
        observations,
        aggregation_error,
        owner="Additive-sigma likelihood",
        output_dim=output_dim,
    )
    observed = add_model_data(observations.transpose(output_dim), "Y")
    reported_error = add_model_data(observation_error.transpose(output_dim), "error")
    minimum_error_data = add_model_data(minimum_error.transpose(output_dim), "min_error")

    independent_variance = reported_error**2
    if not no_model_error:
        sigma = add_sigma_component(sigma_alignment, prior_args=dict(sigma_prior))
        independent_variance = independent_variance + sigma**2

    aggregation_marginal_variance = pt.as_tensor_variable(
        pm.floatX(aggregation_error.marginal_variance)
    )
    floor_variance = cast(Any, pt.maximum)(
        minimum_error_data**2 - independent_variance - aggregation_marginal_variance,
        0.0,
    )
    independent_variance = independent_variance + floor_variance
    error_scale = pm.Deterministic(
        "epsilon",
        pt.sqrt(independent_variance + aggregation_marginal_variance),
        dims=output_dim,
    )
    return AdditiveSigmaErrorState(
        observed=observed,
        independent_variance=independent_variance,
        aggregation_error=aggregation_error,
        error_scale=error_scale,
    )


def add_additive_sigma_gaussian_likelihood(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray,
    aggregation_error: AggregationError,
    mean: TensorVariable,
    sigma_alignment: SigmaAlignment,
    sigma_prior: Mapping[str, Any],
    no_model_error: bool,
    output_dim: str = "nmeasure",
) -> TensorVariable:
    """Add a Gaussian likelihood with additive mismatch variance.

    This is the complete, opt-in observation model associated with
    :func:`build_additive_sigma_error`. The supplied ``mean`` is the completed
    forward-model concentration, including every pollution and baseline term
    selected by the calling model recipe. Model-data mismatch contributes
    ``sigma**2`` to the reported observation-error variance; it is not scaled
    by the pollution enhancement.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        minimum_error: Minimum total-error standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        mean: Completed forward-model concentration aligned with
            ``output_dim``.
        sigma_alignment: Mapping from observations to the mismatch-scale
            parameters.
        sigma_prior: Prior arguments used to construct ``sigma`` when model
            error is enabled.
        no_model_error: If true, omit ``sigma`` and use only reported and
            selected aggregation errors.
        output_dim: Observation dimension used for named PyMC variables.

    Returns:
        The observed Gaussian variable, named ``y``. The total marginal error
        scale is also recorded in the active model as ``epsilon``.

    Raises:
        ValueError: If the observation or aggregation-error inputs are
            inconsistent with ``output_dim``.
    """
    state = build_additive_sigma_error(
        observations=observations,
        observation_error=observation_error,
        minimum_error=minimum_error,
        aggregation_error=aggregation_error,
        sigma_alignment=sigma_alignment,
        sigma_prior=sigma_prior,
        no_model_error=no_model_error,
        output_dim=output_dim,
    )
    return add_gaussian_observation_likelihood(
        observed=state.observed,
        mean=mean,
        independent_variance=state.independent_variance,
        aggregation_error=state.aggregation_error,
        output_dim=output_dim,
    )


__all__ = [
    "AdditiveSigmaErrorState",
    "add_additive_sigma_gaussian_likelihood",
    "build_additive_sigma_error",
]
