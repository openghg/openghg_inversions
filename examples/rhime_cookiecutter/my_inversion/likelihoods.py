"""Project-owned scientific variation for a cookiecutter-generated package."""

from collections.abc import Mapping
from typing import Any

import pymc as pm
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.pollution_event import build_pollution_event_error
from openghg_inversions.observation_error import (
    AggregationError,
)
from openghg_inversions.sigma import SigmaAlignment


def likelihood_builder(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray,
    aggregation_error: AggregationError,
    mean: TensorVariable,
    pollution_mean: TensorVariable,
    pollution_event_baseline: TensorVariable | None,
    sigma_alignment: SigmaAlignment,
    sigma_prior: Mapping[str, Any],
    power: Mapping[str, Any] | float,
    pollution_events_from_obs: bool,
    no_model_error: bool,
    output_dim: str,
) -> TensorVariable:
    """Build an independent Student-t observation likelihood for RHIME.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        minimum_error: Minimum total-error standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        mean: Completed forward-model concentration.
        pollution_mean: Modelled pollution contribution used to scale the
            pollution-event error.
        pollution_event_baseline: Baseline removed when deriving pollution
            events from observations. Compatibility runs may deliberately
            supply the historical boundary-only baseline.
        sigma_alignment: Mapping from observations to sigma parameters.
        sigma_prior: Prior arguments used to construct sigma.
        power: Exponent applied to the pollution-event mismatch term.
        pollution_events_from_obs: Whether to derive pollution events from
            observations instead of the modelled pollution contribution.
        no_model_error: Whether to omit pollution-event mismatch error.
        output_dim: Observation dimension used by named PyMC variables.

    Returns:
        The observed Student-t variable, named ``y``.

    Raises:
        ValueError: If aggregation error requires a multivariate likelihood.
    """
    if aggregation_error.mode not in {"none", "diagonal"}:
        raise ValueError("This Student-t model assumes independent observations.")

    state = build_pollution_event_error(
        observations=observations,
        observation_error=observation_error,
        minimum_error=minimum_error,
        aggregation_error=aggregation_error,
        pollution_mean=pollution_mean,
        pollution_event_baseline=pollution_event_baseline,
        sigma_alignment=sigma_alignment,
        sigma_prior=sigma_prior,
        power=power,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        output_dim=output_dim,
    )
    observed = pm.StudentT(
        "y",
        nu=4.0,
        mu=mean,
        sigma=state.error_scale,
        observed=state.observed,
        dims=output_dim,
    )
    return observed


__all__ = ["likelihood_builder"]
