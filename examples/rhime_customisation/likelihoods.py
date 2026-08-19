"""Editable Student-t likelihood shared by both RHIME examples.

This project-owned function changes RHIME's observation distribution while
reusing its current mean and error construction. It demonstrates the result
roles, compatible outputs, and JSON-compatible metadata that a likelihood
builder owns.

``likelihood_builder`` is the exported runner seam. Importing this module does
not build a model, retrieve data, sample, or write outputs.
"""

import pymc as pm
import pytensor.tensor as pt

from openghg_inversions.models.components import add_model_data, add_sigma_component
from openghg_inversions.models.pollution_event import build_pollution_event_error
from openghg_inversions.observation_error import select_aggregation_error_mode
from openghg_inversions.rhime import RhimeLikelihoodContext, RhimeLikelihoodResult


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


def absolute_sigma_likelihood_builder(
    context: RhimeLikelihoodContext,
) -> RhimeLikelihoodResult:
    """Build an independent Gaussian with absolute model-data mismatch.

    This project-owned example deliberately supports no aggregation-error
    covariance. Its observation-scale mismatch is ``sigma`` itself, rather
    than RHIME's pollution-event-scaled ``pollution_event * sigma``.
    """
    aggregation_error_mode = select_aggregation_error_mode(
        context.data,
        context.aggregation_error_mode,
    )
    if aggregation_error_mode != "none":
        raise ValueError("This absolute-sigma example assumes independent observations.")

    observed = add_model_data(context.data["mf"].transpose(context.output_dim), "Y")
    measurement_error = add_model_data(
        context.data["mf_error"].transpose(context.output_dim),
        "error",
    )
    minimum_error = add_model_data(
        context.data["min_error"].transpose(context.output_dim),
        "min_error",
    )
    variance = measurement_error**2
    if not context.no_model_error:
        sigma = add_sigma_component(
            context.sigma_alignment,
            prior_args=dict(context.sigma_prior),
        )
        variance = variance + sigma**2
    epsilon = pm.Deterministic(
        "epsilon",
        pt.maximum(pt.sqrt(variance), minimum_error),
        dims=context.output_dim,
    )
    likelihood = pm.Normal(
        "y",
        mu=context.mean,
        sigma=epsilon,
        observed=observed,
        dims=context.output_dim,
    )
    return RhimeLikelihoodResult(
        likelihood=likelihood,
        error_scale=epsilon,
        variable_roles={"concentration": "y", "model_error": "epsilon"},
        supported_output_formats=("none", "inv_out", "basic", "paris", "legacy"),
        metadata={
            "family": "absolute_sigma_gaussian",
            "sigma_interpretation": "absolute",
        },
    )


__all__ = ["absolute_sigma_likelihood_builder", "likelihood_builder"]
