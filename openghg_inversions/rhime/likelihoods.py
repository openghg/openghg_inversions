"""RHIME adapters for reusable observation likelihoods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.additive_sigma import (
    add_additive_sigma_gaussian_likelihood,
)
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.sigma import SigmaAlignment


def additive_sigma_likelihood_builder(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray,
    aggregation_error: AggregationError,
    mean: TensorVariable,
    pollution_mean: TensorVariable,
    pollution_event_baseline: TensorVariable | None,
    sigma_alignment: SigmaAlignment | None,
    sigma_prior: Mapping[str, Any],
    power: Mapping[str, Any] | float,
    pollution_events_from_obs: bool,
    no_model_error: bool,
    output_dim: str,
) -> TensorVariable:
    """Adapt the additive-sigma Gaussian to the RHIME likelihood seam.

    Additive mismatch variance is independent of the pollution enhancement.
    The RHIME seam's pollution contribution, baseline, exponent, and
    pollution-event source policy are therefore intentionally unused. They
    remain explicit here so the reusable likelihood does not need to accept
    scientifically irrelevant arguments.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        minimum_error: Minimum total-error standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        mean: Completed forward-model concentration.
        pollution_mean: Modelled pollution contribution, unused by additive
            mismatch variance.
        pollution_event_baseline: Baseline used by pollution-event-scaled
            likelihoods, unused by additive mismatch variance.
        sigma_alignment: Mapping from observations to mismatch parameters when
            model error is enabled.
        sigma_prior: Prior arguments used to construct ``sigma``.
        power: Pollution-event exponent, unused by additive mismatch variance.
        pollution_events_from_obs: Pollution-event source policy, unused by
            additive mismatch variance.
        no_model_error: Whether to omit inferred mismatch error.
        output_dim: Observation dimension used for named PyMC variables.

    Returns:
        The observed Gaussian variable, named ``y``. The reusable component
        also adds the canonical marginal error scale ``epsilon``.

    Raises:
        ValueError: If observation or aggregation-error inputs are invalid.
    """
    del pollution_mean, pollution_event_baseline, power, pollution_events_from_obs
    return add_additive_sigma_gaussian_likelihood(
        observations=observations,
        observation_error=observation_error,
        minimum_error=minimum_error,
        aggregation_error=aggregation_error,
        mean=mean,
        sigma_alignment=sigma_alignment,
        sigma_prior=sigma_prior,
        no_model_error=no_model_error,
        output_dim=output_dim,
    )


__all__ = ["additive_sigma_likelihood_builder"]
