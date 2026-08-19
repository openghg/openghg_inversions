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
