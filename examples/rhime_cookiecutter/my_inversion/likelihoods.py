"""Project-owned scientific variation for a cookiecutter-generated package."""

import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.observation_error import (
    validate_observation_error_arrays,
)


def likelihood_builder(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    mean: TensorVariable,
    output_dim: str,
    degrees_of_freedom: float = 4.0,
) -> TensorVariable:
    """Build an independent Student-t observation likelihood for RHIME.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        mean: Completed forward-model concentration.
        output_dim: Observation dimension used by named PyMC variables.
        degrees_of_freedom: Positive Student-t degrees of freedom supplied as
            a custom likelihood option.

    Returns:
        The observed Student-t variable, named ``y``.

    Raises:
        ValueError: If degrees of freedom are not positive.
    """
    if degrees_of_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be positive.")

    validate_observation_error_arrays(
        observations,
        observation_error,
        None,
        owner="Custom Student-t likelihood",
        output_dim=output_dim,
    )
    reported_error = pm.Data(
        "error",
        pm.floatX(observation_error.transpose(output_dim).compute().values),
        dims=output_dim,
    )
    epsilon = pm.Deterministic("epsilon", pt.abs(reported_error), dims=output_dim)
    observed = pm.StudentT(
        "y",
        nu=degrees_of_freedom,
        mu=mean,
        sigma=epsilon,
        observed=pm.floatX(observations.transpose(output_dim).compute().values),
        dims=output_dim,
    )
    return observed


__all__ = ["likelihood_builder"]
