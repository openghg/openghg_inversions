"""Project-owned scientific variation for a cookiecutter-generated package."""

import pymc as pm

from openghg_inversions.models.rhime_likelihood import (
    RhimeLikelihoodContext,
    RhimeLikelihoodResult,
    build_rhime_observation_state,
)
from openghg_inversions.observation_error import select_aggregation_error_mode


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

    state = build_rhime_observation_state(context)
    observed = pm.StudentT(
        "student_y",
        nu=4.0,
        mu=state.mean,
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
