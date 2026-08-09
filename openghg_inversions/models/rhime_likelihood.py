"""Modern RHIME likelihood assembly with fixed aggregation covariance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

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
class RhimeLikelihoodContext:
    """Complete labelled context supplied to a RHIME likelihood builder.

    A custom builder owns the complete observation component: error-model
    construction and the observed distribution. Use
    :func:`build_rhime_observation_state` to reuse RHIME's current error scale
    while replacing only the distribution.
    """

    data: xr.Dataset
    flux_mean: TensorVariable
    boundary_mean: TensorVariable | None
    offset: TensorVariable | None
    sigma_alignment: SigmaAlignment
    sigma_prior: Mapping[str, Any]
    power: Mapping[str, Any] | float
    pollution_events_from_obs: bool
    no_model_error: bool
    aggregation_error_mode: AggregationErrorMode
    output_dim: str = "nmeasure"


@dataclass(frozen=True)
class RhimeObservationState:
    """Current RHIME mean and error state before choosing a distribution."""

    observed: TensorVariable
    mean: TensorVariable
    independent_variance: TensorVariable
    aggregation_error: AggregationError
    error_scale: TensorVariable


@dataclass(frozen=True)
class RhimeLikelihoodResult:
    """Observed variable, optional error scale, and explicit semantic roles."""

    likelihood: TensorVariable
    variable_roles: Mapping[str, str]
    error_scale: TensorVariable | None = None
    supported_output_formats: tuple[str, ...] = ("none",)

    def __post_init__(self) -> None:
        """Require the observed-variable role without inferring its name."""
        roles = {str(role): str(name) for role, name in self.variable_roles.items()}
        if roles.get("concentration") != self.likelihood.name:
            raise ValueError(
                "`RhimeLikelihoodResult.variable_roles['concentration']` must equal the returned "
                f"likelihood name {self.likelihood.name!r}."
            )
        if self.error_scale is not None and "model_error" in roles:
            if roles["model_error"] != self.error_scale.name:
                raise ValueError(
                    "`RhimeLikelihoodResult.variable_roles['model_error']` must equal the returned "
                    f"error-scale name {self.error_scale.name!r}."
                )
        output_formats = tuple(dict.fromkeys(self.supported_output_formats))
        invalid_formats = sorted(
            set(output_formats) - {"none", "inv_out", "basic", "paris", "legacy"}
        )
        if invalid_formats:
            raise ValueError(
                "`RhimeLikelihoodResult.supported_output_formats` contains unsupported values: "
                f"{invalid_formats!r}."
            )
        if "none" not in output_formats:
            raise ValueError("`RhimeLikelihoodResult.supported_output_formats` must include 'none'.")
        object.__setattr__(self, "variable_roles", roles)
        object.__setattr__(self, "supported_output_formats", output_formats)


class RhimeLikelihoodBuilder(Protocol):
    """Callable contract for a complete RHIME observation component."""

    def __call__(self, context: RhimeLikelihoodContext, /) -> RhimeLikelihoodResult:
        """Add error and observed-distribution variables to the active model."""
        ...


def build_rhime_observation_state(context: RhimeLikelihoodContext) -> RhimeObservationState:
    """Build RHIME's current mean and error scale without choosing a likelihood."""
    data = context.data
    output_dim = context.output_dim
    validate_observation_error_inputs(data, output_dim=output_dim)
    aggregation_error = resolve_aggregation_error(
        data,
        context.aggregation_error_mode,
        output_dim=output_dim,
    )
    y_data = add_model_data(data["mf"].transpose(output_dim), "Y")
    error_data = add_model_data(data["mf_error"].transpose(output_dim), "error")
    min_error_data = add_model_data(data["min_error"].transpose(output_dim), "min_error")

    sigma = add_sigma_component(context.sigma_alignment, prior_args=dict(context.sigma_prior))
    if context.pollution_events_from_obs:
        pollution_event = (
            pt.abs(y_data - context.boundary_mean)
            if context.boundary_mean is not None
            else pt.abs(y_data) + 1e-6 * pt.mean(y_data)
        )
    else:
        pollution_event = pt.abs(context.flux_mean)

    if context.no_model_error:
        mean_obs = np.nanmean(data["mf"].values)
        small_amount = pm.floatX(1e-12 * mean_obs)
        independent_variance = cast(Any, pt.maximum)(error_data**2, small_amount**2)
    else:
        power0 = parse_prior("power", dict(context.power)) if isinstance(context.power, Mapping) else context.power
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

    total_mu = context.flux_mean
    if context.boundary_mean is not None:
        total_mu = total_mu + context.boundary_mean
    if context.offset is not None:
        total_mu = total_mu + context.offset

    total_marginal_variance = independent_variance + pt.as_tensor_variable(
        pm.floatX(aggregation_error.marginal_variance)
    )
    epsilon = pm.Deterministic("epsilon", pt.sqrt(total_marginal_variance), dims=output_dim)
    return RhimeObservationState(
        observed=y_data,
        mean=total_mu,
        independent_variance=independent_variance,
        aggregation_error=aggregation_error,
        error_scale=epsilon,
    )


def build_gaussian_rhime_likelihood(context: RhimeLikelihoodContext) -> RhimeLikelihoodResult:
    """Add the built-in Gaussian RHIME likelihood through the public contract."""
    state = build_rhime_observation_state(context)
    likelihood = add_gaussian_observation_likelihood(
        observed=state.observed,
        mean=state.mean,
        independent_variance=state.independent_variance,
        aggregation_error=state.aggregation_error,
        output_dim=context.output_dim,
    )
    return RhimeLikelihoodResult(
        likelihood=likelihood,
        error_scale=state.error_scale,
        variable_roles={"concentration": "y", "model_error": "epsilon"},
        supported_output_formats=("none", "inv_out", "basic", "paris", "legacy"),
    )


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
    result = build_gaussian_rhime_likelihood(
        RhimeLikelihoodContext(
            data=data,
            flux_mean=mu,
            boundary_mean=mu_bc,
            offset=offset,
            sigma_alignment=sigma_alignment,
            sigma_prior=sigprior,
            power=power,
            pollution_events_from_obs=pollution_events_from_obs,
            no_model_error=no_model_error,
            aggregation_error_mode=aggregation_error_mode,
            output_dim=output_dim,
        )
    )
    assert result.error_scale is not None
    return result.error_scale
