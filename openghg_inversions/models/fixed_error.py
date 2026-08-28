r"""Gaussian observations without inferred model-data mismatch.

For reported observation-error variance :math:`s_y^2` and optional fixed
aggregation covariance :math:`C_{agg}`, this component constructs the
independent variance

.. math::

   v = \max\left(|s_y|, 10^{-12}\bar{y}\right)^2.

The tiny scale guard preserves the historical no-model-error behaviour for
zero reported errors. No inferred ``sigma`` or prepared minimum-error floor is
part of this model.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models._gaussian_observation import (
    add_aggregation_error_data,
    add_gaussian_observation_likelihood,
)
from openghg_inversions.models.components import add_model_data
from openghg_inversions.observation_error import (
    AggregationError,
    validate_observation_error_arrays,
)


def add_fixed_error_likelihood(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    mean: TensorVariable,
    output_dim: str = "nmeasure",
    observation_error_name: str = "error",
) -> TensorVariable:
    """Add a Gaussian likelihood without inferred model-data mismatch.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        mean: Completed forward-model concentration aligned with
            ``output_dim``.
        output_dim: Observation dimension used for named PyMC variables.
        observation_error_name: PyMC data name for the reported error.

    Returns:
        The observed Gaussian variable, named ``y``. The total marginal error
        scale is also recorded in the active model as ``epsilon``.

    Raises:
        ValueError: If the observation or aggregation-error inputs are
            inconsistent with ``output_dim``.
    """
    validate_observation_error_arrays(
        observations,
        observation_error,
        None,
        owner="Fixed-error likelihood",
        output_dim=output_dim,
    )
    reported_error = add_model_data(
        observation_error.transpose(output_dim),
        observation_error_name,
    )
    registered_aggregation_error = add_aggregation_error_data(
        aggregation_error,
        observations,
        output_dim=output_dim,
    )

    observed = observations.transpose(output_dim).compute()
    small_amount = pm.floatX(1e-12 * np.nanmean(observed.values))
    independent_scale = cast(Any, pt.maximum)(pt.abs(reported_error), small_amount)
    independent_variance = independent_scale**2
    pm.Deterministic(
        "epsilon",
        pt.sqrt(independent_variance + registered_aggregation_error.marginal_variance),
        dims=output_dim,
    )
    return add_gaussian_observation_likelihood(
        observed=pm.floatX(observed.values),
        mean=mean,
        independent_variance=independent_variance,
        aggregation_error=registered_aggregation_error,
        output_dim=output_dim,
    )


__all__ = ["add_fixed_error_likelihood"]
