"""Editable likelihoods using RHIME's explicit scientific inputs.

This project-owned function changes RHIME's observation distribution while
reusing its current mean and fixed-error construction.

``likelihood_builder`` is the exported runner seam. Importing this module does
not build a model, retrieve data, sample, or write outputs.
"""

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
    """Build a fixed-error Student-t model from RHIME's common inputs.

    Degrees of freedom default to 4.0. RHIME's ``epsilon`` is passed as the
    Student-t scale, not its marginal standard deviation; at the default the
    marginal standard deviation is ``epsilon * sqrt(2)``.

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
        ValueError: If dense or low-rank aggregation error would require a
            multivariate Student-t likelihood.

    Notes:
        On success, this adds RHIME's canonical observation-state nodes and
        ``y`` to the active PyMC model. It performs no sampling or output
        writes. Unsupported aggregation modes are rejected before any nodes
        are added.
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
