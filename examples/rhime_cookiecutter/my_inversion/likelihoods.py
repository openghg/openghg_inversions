"""Project-owned scientific variation for a cookiecutter-generated package."""

import pymc as pm

from openghg_inversions.models.pollution_event import build_pollution_event_error
from openghg_inversions.observation_error import select_aggregation_error_mode
from openghg_inversions.rhime import RhimeLikelihoodContext, RhimeLikelihoodResult


def likelihood_builder(context: RhimeLikelihoodContext) -> RhimeLikelihoodResult:
    """Build an independent Student-t observation likelihood for RHIME.

    Args:
        context: Labelled RHIME means, observations, and error settings for
            the active PyMC model.

    Returns:
        Student-t observation variable and the public roles, output support,
        and provenance metadata needed by the library-owned runner.

    Raises:
        ValueError: If aggregation error requires a multivariate likelihood.
    """
    aggregation_error_mode = select_aggregation_error_mode(
        context.data,
        context.aggregation_error_mode,
    )
    if aggregation_error_mode not in {"none", "diagonal"}:
        raise ValueError("This Student-t model assumes independent observations.")

    state = build_pollution_event_error(
        context.data,
        pollution_mean=context.pollution_mean,
        pollution_event_baseline=context.pollution_event_baseline,
        sigma_alignment=context.sigma_alignment,
        sigma_prior=context.sigma_prior,
        power=context.power,
        pollution_events_from_obs=context.pollution_events_from_obs,
        no_model_error=context.no_model_error,
        aggregation_error_mode=context.aggregation_error_mode,
        output_dim=context.output_dim,
    )
    observed = pm.StudentT(
        "student_y",
        nu=4.0,
        mu=context.mean,
        sigma=state.error_scale,
        observed=state.observed,
        dims=context.output_dim,
    )
    return RhimeLikelihoodResult(
        likelihood=observed,
        error_scale=state.error_scale,
        variable_roles={
            "concentration": "student_y",
            "model_error": "epsilon",
        },
        supported_output_formats=("none", "inv_out"),
        metadata={"family": "student_t", "degrees_of_freedom": 4.0},
    )


__all__ = ["likelihood_builder"]
