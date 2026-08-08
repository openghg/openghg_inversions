"""Modern RHIME likelihood assembly with fixed aggregation covariance."""

from __future__ import annotations

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
    AggregationErrorMode,
    resolve_aggregation_error,
    validate_observation_error_inputs,
)
from openghg_inversions.sigma import SigmaAlignment


def add_rhime_likelihood_component(
    data: xr.Dataset,
    /,
    mu: TensorVariable,
    mu_bc: TensorVariable | None,
    sigprior: dict,
    sigma_alignment: SigmaAlignment,
    offset: TensorVariable | None = None,
    power: dict | float = 1.99,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
    aggregation_error_mode: AggregationErrorMode = "auto",
    output_dim: str = "nmeasure",
) -> TensorVariable:
    """Add the modern RHIME observation model.

    Aggregation covariance is fixed prepared-input data. ``mf_error`` remains
    the raw observation-error standard deviation, and inferred model variance
    remains an independent diagonal contribution.
    """
    validate_observation_error_inputs(data, output_dim=output_dim)
    aggregation_error = resolve_aggregation_error(
        data,
        aggregation_error_mode,
        output_dim=output_dim,
    )
    y_data = add_model_data(data["mf"].transpose(output_dim), "Y")
    error_data = add_model_data(data["mf_error"].transpose(output_dim), "error")
    min_error_data = add_model_data(data["min_error"].transpose(output_dim), "min_error")

    sigma = add_sigma_component(sigma_alignment, prior_args=sigprior)
    if pollution_events_from_obs:
        pollution_event = pt.abs(y_data - mu_bc) if mu_bc is not None else pt.abs(y_data) + 1e-6 * pt.mean(y_data)
    else:
        pollution_event = pt.abs(mu)

    if no_model_error:
        mean_obs = np.nanmean(data["mf"].values)
        small_amount = pm.floatX(1e-12 * mean_obs)
        independent_variance = cast(Any, pt.maximum)(error_data**2, small_amount**2)
    else:
        power0 = parse_prior("power", power) if isinstance(power, dict) else power
        model_error_variance = pt.pow(pollution_event * sigma, power0)
        raw_independent_variance = error_data**2 + model_error_variance
        aggregation_marginal_variance = pt.as_tensor_variable(
            pm.floatX(aggregation_error.marginal_variance)
        )
        floor_extra = cast(Any, pt.maximum)(
            min_error_data**2 - raw_independent_variance - aggregation_marginal_variance,
            0.0,
        )
        independent_variance = raw_independent_variance + floor_extra

    total_mu = mu
    if mu_bc is not None:
        total_mu = total_mu + mu_bc
    if offset is not None:
        total_mu = total_mu + offset

    total_marginal_variance = independent_variance + pt.as_tensor_variable(
        pm.floatX(aggregation_error.marginal_variance)
    )
    epsilon = pm.Deterministic("epsilon", pt.sqrt(total_marginal_variance), dims=output_dim)
    add_gaussian_observation_likelihood(
        observed=y_data,
        mean=total_mu,
        independent_variance=independent_variance,
        aggregation_error=aggregation_error,
        output_dim=output_dim,
    )
    return epsilon
