"""Pollution-event-scaled observation-error construction.

For an observation ``Y`` with reported standard deviation ``error``, the
historical model uses

``epsilon = max(sqrt(error**2 + (pollution_event * sigma)**power), min_error)``.

The pollution event is either the modelled pollution contribution or the
observation after removing an explicitly supplied modelled baseline. Fixed
aggregation covariance is an optional, explicitly selected addition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.components import add_model_data, add_sigma_component
from openghg_inversions.models.likelihoods import add_gaussian_observation_likelihood
from openghg_inversions.models.priors import parse_prior
from openghg_inversions.observation_error import (
    AggregationError,
    AggregationErrorMode,
    resolve_aggregation_error,
    validate_observation_error_inputs,
)
from openghg_inversions.sigma import SigmaAlignment


@dataclass(frozen=True)
class PollutionEventErrorState:
    """Terms required to construct an observation distribution."""

    observed: TensorVariable
    independent_variance: TensorVariable
    aggregation_error: AggregationError
    error_scale: TensorVariable


def build_pollution_event_error(
    data: xr.Dataset,
    /,
    *,
    pollution_mean: TensorVariable,
    pollution_event_baseline: TensorVariable | None,
    sigma_alignment: SigmaAlignment,
    sigma_prior: Mapping[str, Any],
    power: Mapping[str, Any] | float,
    pollution_events_from_obs: bool,
    no_model_error: bool,
    retain_unused_sigma: bool = False,
    aggregation_error_mode: AggregationErrorMode,
    output_dim: str = "nmeasure",
) -> PollutionEventErrorState:
    """Build the historical pollution-event error terms.

    ``pollution_mean`` is the modelled pollution contribution.
    ``pollution_event_baseline`` is the complete modelled baseline removed when
    deriving pollution events from observations. Compatibility adapters may
    explicitly supply a narrower historical term.

    ``retain_unused_sigma`` exists only for compatibility with graphs that
    stored ``sigma`` even when ``no_model_error`` made it irrelevant.

    Args:
        data: Canonical observations and reported error components.
        pollution_mean: Modelled pollution contribution used when mismatch
            scaling follows the forward model.
        pollution_event_baseline: Baseline removed when mismatch scaling uses
            observed enhancements.
        sigma_alignment: Observation alignment for mismatch parameters.
        sigma_prior: Prior specification for mismatch parameters.
        power: Exponent or prior specification for mismatch scaling.
        pollution_events_from_obs: Whether observed enhancements control
            mismatch scaling.
        no_model_error: Whether to omit inferred mismatch error.
        retain_unused_sigma: Whether to preserve the historical disconnected
            ``sigma`` variable when model error is disabled.
        aggregation_error_mode: Aggregation-error representation to include.
        output_dim: Observation dimension name.

    Returns:
        Observed data, independent variance, aggregation error, and canonical
        total-error scale.

    Raises:
        ValueError: If observation or aggregation-error inputs are invalid.
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
    sigma = None
    if not no_model_error or retain_unused_sigma:
        sigma = add_sigma_component(sigma_alignment, prior_args=dict(sigma_prior))

    if no_model_error:
        mean_observation = np.nanmean(data["mf"].values)
        small_amount = pm.floatX(1e-12 * mean_observation)
        # Preserve the exact historical expression before converting it to a
        # variance for covariance-aware observation distributions.
        independent_scale = cast(Any, pt.maximum)(pt.abs(observation_error), small_amount)
        independent_variance = independent_scale**2
    else:
        assert sigma is not None
        if pollution_events_from_obs:
            pollution_event = (
                pt.abs(observed - pollution_event_baseline)
                if pollution_event_baseline is not None
                else pt.abs(observed) + 1e-6 * pt.mean(observed)
            )
        else:
            pollution_event = pt.abs(pollution_mean)

        exponent = (
            parse_prior("power", dict(power)) if isinstance(power, Mapping) else power
        )
        raw_independent_variance = observation_error**2 + pt.pow(
            pollution_event * sigma,
            exponent,
        )
        aggregation_marginal_variance = pt.as_tensor_variable(
            pm.floatX(aggregation_error.marginal_variance)
        )
        floor_variance = cast(Any, pt.maximum)(
            minimum_error**2
            - raw_independent_variance
            - aggregation_marginal_variance,
            0.0,
        )
        independent_variance = raw_independent_variance + floor_variance

    total_marginal_variance = independent_variance + pt.as_tensor_variable(
        pm.floatX(aggregation_error.marginal_variance)
    )
    error_scale = pm.Deterministic(
        "epsilon",
        pt.sqrt(total_marginal_variance),
        dims=output_dim,
    )
    return PollutionEventErrorState(
        observed=observed,
        independent_variance=independent_variance,
        aggregation_error=aggregation_error,
        error_scale=error_scale,
    )


def build_pollution_event_gaussian_likelihood(
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
    retain_unused_sigma: bool = False,
    aggregation_error_mode: AggregationErrorMode,
    output_dim: str = "nmeasure",
) -> TensorVariable:
    """Build RHIME's pollution-event-scaled Gaussian likelihood.

    Args:
        data: Canonical inversion inputs containing observations and errors.
        mean: Completed modelled concentration, including the full baseline.
        pollution_mean: Modelled pollution contribution used for mismatch
            scaling when ``pollution_events_from_obs`` is false.
        pollution_event_baseline: Modelled baseline removed from observations
            when ``pollution_events_from_obs`` is true.
        sigma_alignment: Observation alignment for mismatch parameters.
        sigma_prior: Prior specification for mismatch parameters.
        power: Exponent or prior specification used in mismatch scaling.
        pollution_events_from_obs: Whether observed rather than modelled
            pollution enhancements control mismatch scaling.
        no_model_error: Whether to omit inferred mismatch error.
        retain_unused_sigma: Whether to retain the historical disconnected
            ``sigma`` variable when model error is disabled.
        aggregation_error_mode: Aggregation-error representation to include.
        output_dim: Observation dimension name.

    Returns:
        Observed Gaussian variable named ``y``. The component also creates the
        canonical total-error variable ``epsilon``.
    """
    state = build_pollution_event_error(
        data,
        pollution_mean=pollution_mean,
        pollution_event_baseline=pollution_event_baseline,
        sigma_alignment=sigma_alignment,
        sigma_prior=sigma_prior,
        power=power,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        retain_unused_sigma=retain_unused_sigma,
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
    "PollutionEventErrorState",
    "build_pollution_event_error",
    "build_pollution_event_gaussian_likelihood",
]
