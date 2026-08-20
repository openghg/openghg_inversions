"""Project-owned scientific variation for a cookiecutter-generated package."""

import pymc as pm
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.additive_sigma import build_additive_sigma_error
from openghg_inversions.observation_error import (
    AggregationError,
)


def likelihood_builder(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray,
    aggregation_error: AggregationError,
    mean: TensorVariable,
    pollution_mean: TensorVariable,
    pollution_event_baseline: TensorVariable | None,
    output_dim: str,
    degrees_of_freedom: float = 4.0,
) -> TensorVariable:
    """Build an independent Student-t observation likelihood for RHIME.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        minimum_error: Minimum total-error standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        mean: Completed forward-model concentration.
        pollution_mean: Modelled pollution contribution, unused by this
            fixed-error likelihood.
        pollution_event_baseline: Modelled baseline, unused by this
            fixed-error likelihood.
        output_dim: Observation dimension used by named PyMC variables.
        degrees_of_freedom: Positive Student-t degrees of freedom supplied as
            a custom likelihood option.

    Returns:
        The observed Student-t variable, named ``y``.

    Raises:
        ValueError: If aggregation error requires a multivariate likelihood.
    """
    if aggregation_error.mode not in {"none", "diagonal"}:
        raise ValueError("This Student-t model assumes independent observations.")
    if degrees_of_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be positive.")

    del pollution_mean, pollution_event_baseline
    state = build_additive_sigma_error(
        observations=observations,
        observation_error=observation_error,
        minimum_error=minimum_error,
        aggregation_error=aggregation_error,
        sigma_alignment=None,
        sigma_prior={},
        no_model_error=True,
        output_dim=output_dim,
    )
    observed = pm.StudentT(
        "y",
        nu=degrees_of_freedom,
        mu=mean,
        sigma=state.error_scale,
        observed=state.observed,
        dims=output_dim,
    )
    return observed


__all__ = ["likelihood_builder"]
