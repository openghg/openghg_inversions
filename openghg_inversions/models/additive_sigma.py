r"""Observation error with an additive model-data-mismatch scale.

For reported observation-error variance :math:`s_y^2`, mismatch scale
:math:`\sigma`, and fixed aggregation covariance :math:`C_{agg}`, this
component constructs the independent variance

.. math::

   v = s_y^2 + \sigma^2

and applies ``min_error`` as a floor on the total marginal standard
deviation.  The component owns error construction only; a model recipe or
likelihood builder supplies the modelled concentration and distribution.
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
    AggregationErrorMode,
    resolve_aggregation_error,
    validate_observation_error_inputs,
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
    data: xr.Dataset,
    /,
    *,
    sigma_alignment: SigmaAlignment,
    sigma_prior: Mapping[str, Any],
    no_model_error: bool,
    aggregation_error_mode: AggregationErrorMode,
    output_dim: str = "nmeasure",
) -> AdditiveSigmaErrorState:
    """Build observation-error terms with an additive mismatch scale.

    ``sigma`` is observation-aligned through ``sigma_alignment`` and enters as
    an independent variance term. When ``no_model_error`` is true, no ``sigma``
    variable is constructed. Fixed diagonal, dense, or low-rank aggregation
    error is included only when explicitly selected by
    ``aggregation_error_mode``.

    Args:
        data: Observation dataset containing ``mf``, ``mf_error``, and
            ``min_error`` along ``output_dim``. It may also contain the fields
            required by the selected aggregation-error representation.
        sigma_alignment: Mapping from observations to the mismatch-scale
            parameters.
        sigma_prior: Prior arguments used to construct ``sigma`` when model
            error is enabled.
        no_model_error: If true, omit ``sigma`` and use only reported and
            selected aggregation errors.
        aggregation_error_mode: Fixed aggregation covariance representation
            prepared for the observation distribution.
        output_dim: Observation dimension used for named PyMC variables.

    Returns:
        Labelled observed data, independent variance, fixed aggregation error,
        and total marginal error scale.

    Raises:
        ValueError: If the observation or aggregation-error inputs are
            inconsistent with ``output_dim``.
    """
    validate_observation_error_inputs(data, output_dim=output_dim)
    aggregation_error = resolve_aggregation_error(
        data,
        aggregation_error_mode,
        output_dim=output_dim,
    )
    observed = add_model_data(data["mf"].transpose(output_dim), "Y")
    observation_error = add_model_data(data["mf_error"].transpose(output_dim), "error")
    minimum_error = add_model_data(data["min_error"].transpose(output_dim), "min_error")

    independent_variance = observation_error**2
    if not no_model_error:
        sigma = add_sigma_component(sigma_alignment, prior_args=dict(sigma_prior))
        independent_variance = independent_variance + sigma**2

    aggregation_marginal_variance = pt.as_tensor_variable(
        pm.floatX(aggregation_error.marginal_variance)
    )
    floor_variance = cast(Any, pt.maximum)(
        minimum_error**2 - independent_variance - aggregation_marginal_variance,
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
    data: xr.Dataset,
    /,
    *,
    mean: TensorVariable,
    pollution_mean: TensorVariable,
    pollution_event_baseline: TensorVariable | None,
    sigma_alignment: SigmaAlignment,
    sigma_prior: Mapping[str, Any],
    power: Mapping[str, Any] | float,
    pollution_events_from_obs: bool,
    no_model_error: bool,
    aggregation_error_mode: AggregationErrorMode,
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
        data: Observation dataset containing ``mf``, ``mf_error``, and
            ``min_error`` along ``output_dim``. It may also contain the fields
            required by the selected aggregation-error representation.
        mean: Completed forward-model concentration aligned with
            ``output_dim``.
        pollution_mean: Pollution contribution accepted for compatibility
            with the RHIME likelihood-builder protocol. Additive mismatch
            variance does not depend on the pollution enhancement.
        pollution_event_baseline: Baseline supplied by the RHIME recipe for
            pollution-event scaling. This likelihood does not derive pollution
            events, so the value is intentionally unused.
        sigma_alignment: Mapping from observations to the mismatch-scale
            parameters.
        sigma_prior: Prior arguments used to construct ``sigma`` when model
            error is enabled.
        power: Pollution-event exponent accepted for the RHIME
            likelihood-builder protocol and intentionally unused.
        pollution_events_from_obs: Pollution-event source policy accepted for
            the RHIME likelihood-builder protocol and intentionally unused.
        no_model_error: If true, omit ``sigma`` and use only reported and
            selected aggregation errors.
        aggregation_error_mode: Fixed aggregation covariance representation
            consumed by the Gaussian distribution.
        output_dim: Observation dimension used for named PyMC variables.

    Returns:
        The observed Gaussian variable, named ``y``. The total marginal error
        scale is also recorded in the active model as ``epsilon``.

    Raises:
        ValueError: If the observation or aggregation-error inputs are
            inconsistent with ``output_dim``.
    """
    del pollution_mean, pollution_event_baseline, power, pollution_events_from_obs
    state = build_additive_sigma_error(
        data,
        sigma_alignment=sigma_alignment,
        sigma_prior=sigma_prior,
        no_model_error=no_model_error,
        aggregation_error_mode=aggregation_error_mode,
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
