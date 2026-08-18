"""Editable Student-t likelihood shared by both RHIME examples.

This project-owned function changes RHIME's observation distribution while
reusing its current mean and error construction. It demonstrates the result
roles, compatible outputs, and JSON-compatible metadata that a likelihood
builder owns.

``likelihood_builder`` is the exported runner seam. Importing this module does
not build a model, retrieve data, sample, or write outputs.
"""

import pymc as pm

from openghg_inversions.rhime import (
    RhimeLikelihoodContext,
    RhimeLikelihoodResult,
    build_rhime_observation_state,
    select_aggregation_error_mode,
)


def likelihood_builder(context: RhimeLikelihoodContext) -> RhimeLikelihoodResult:
    """Build a Student-t observation model with RHIME's current error scale.

    Degrees of freedom are fixed at 4.0. RHIME's ``epsilon`` is passed as the
    Student-t scale, not its marginal standard deviation; for four degrees of
    freedom the marginal standard deviation is ``epsilon * sqrt(2)``.

    Args:
        context: Labelled RHIME means, observations, and error settings for
            the active PyMC model.

    Returns:
        Observed Student-t variable with semantic roles, conservative output
        support, and serializable provenance metadata.

    Raises:
        ValueError: If dense or low-rank aggregation error would require a
            multivariate Student-t likelihood.

    Notes:
        On success, this adds RHIME's named observation-state nodes (``Y``,
        ``error``, ``min_error``, and ``sigma`` or ``epsilon`` as applicable)
        plus ``student_y`` to the active PyMC model. It performs no sampling or
        output writes. Unsupported aggregation modes are rejected before any
        nodes are added.
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
