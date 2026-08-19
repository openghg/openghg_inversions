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


__all__ = ["AdditiveSigmaErrorState", "build_additive_sigma_error"]
