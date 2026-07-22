"""Integration tests for inferred OU errors and shared coefficient hierarchy state."""

from dataclasses import fields, replace
from math import log

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.core import (
    InferredOUErrorModel,
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
    gaussian_log_likelihood_numpy,
    lognormal_coefficient_log_prior_numpy,
    uniform_log_k_prior,
    update_dynamic_coefficient_state,
    update_error_model_state,
    update_shared_hierarchy_state,
    update_structural_state,
)
from openghg_inversions.experimental.rjmcmc.hierarchy import (
    SharedLognormalHierarchy,
    shared_coefficient_log_prior_numpy,
    shared_hyperprior_log_density_numpy,
)
from openghg_inversions.experimental.rjmcmc.likelihood import (
    IndependentSiteOUData,
    ou_log_likelihood_numpy,
)


def _problem(
    *,
    inferred_error: bool = False,
    hierarchy: bool = False,
) -> TransDimensionalProblem:
    """Build a deterministic problem with optional hierarchical target pieces."""
    observation_sd = np.array([0.4, 0.5, 0.6, 0.7])
    error_model = None
    if inferred_error:
        data = IndependentSiteOUData(
            observation_sd=observation_sd,
            observation_time=np.array([0.0, 2.0, 1.0, 5.0]),
            site_index=np.array([0, 0, 1, 1], dtype=np.int64),
            mismatch_group_index=np.array([0, 1, 0, 1], dtype=np.int64),
            site_tau_index=np.array([0, 1], dtype=np.int64),
        )
        error_model = InferredOUErrorModel(
            data=data,
            mismatch_sd_prior_lower=np.array([0.1, 0.2]),
            mismatch_sd_prior_upper=np.array([1.1, 2.2]),
            correlation_timescale_prior_lower=np.array([0.5, 1.0]),
            correlation_timescale_prior_upper=np.array([2.5, 5.0]),
        )
    coefficient_hierarchy = None
    if hierarchy:
        coefficient_hierarchy = SharedLognormalHierarchy(
            mean_hyperprior_median=1.0,
            mean_hyperprior_log_sd=0.6,
            sd_hyperprior_median=0.8,
            sd_hyperprior_log_sd=0.4,
        )
    return TransDimensionalProblem(
        observations=np.array([0.2, 0.9, -0.3, 1.1]),
        observation_sd=observation_sd,
        sensitivities=np.array(
            [
                [0.2, 0.4, 0.1, 0.3, 0.5],
                [0.8, 0.1, 0.5, 0.2, 0.3],
                [0.3, 0.6, 0.2, 0.7, 0.1],
                [0.5, 0.2, 0.9, 0.1, 0.4],
            ]
        ),
        grid_coordinates=np.arange(5, dtype=np.float64)[:, None],
        k_min=1,
        k_max=4,
        log_k_prior=uniform_log_k_prior(1, 4),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.7,
        error_model=error_model,
        coefficient_hierarchy=coefficient_hierarchy,
    )


def _assert_state_equal(actual: TransDimensionalState, expected: TransDimensionalState) -> None:
    """Compare all cached arrays exactly and scalar target terms tightly."""
    for state_field in fields(TransDimensionalState):
        actual_value = getattr(actual, state_field.name)
        expected_value = getattr(expected, state_field.name)
        if isinstance(actual_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == pytest.approx(expected_value, rel=0.0, abs=2e-13)


def test_unconfigured_problem_preserves_independent_fixed_prior_target() -> None:
    """Absent optional configurations must retain the original exact target."""
    problem = _problem()
    state = build_state(problem, [0, 4], [0.8, 1.3])

    assert state.mismatch_sd.shape == (0,)
    assert state.correlation_timescale.shape == (0,)
    assert state.eta == state.zeta == 0.0
    assert state.log_error_model_prior == 0.0
    assert state.log_coefficient_hyperprior == 0.0
    assert state.log_likelihood == gaussian_log_likelihood_numpy(
        state.residual,
        problem.observation_sd,
    )
    assert state.log_coefficient_prior == lognormal_coefficient_log_prior_numpy(
        state.coefficients,
        state.k,
        problem.coefficient_prior_mean,
        problem.coefficient_prior_sd,
    )


def test_correlated_target_matches_pure_likelihood_and_normalized_prior() -> None:
    """Configured error arrays must contribute the pure OU and uniform terms."""
    problem = _problem(inferred_error=True)
    mismatch_sd = np.array([0.4, 1.2])
    timescale = np.array([1.5, 3.0])
    state = build_state(
        problem,
        [0, 4],
        [0.8, 1.3],
        mismatch_sd=mismatch_sd,
        correlation_timescale=timescale,
    )
    assert problem.error_model is not None
    expected_likelihood = ou_log_likelihood_numpy(
        state.residual,
        problem.error_model.data,
        mismatch_sd,
        timescale,
    )
    expected_prior = -log(1.0) - log(2.0) - log(2.0) - log(4.0)

    assert state.log_likelihood == pytest.approx(expected_likelihood)
    assert state.log_error_model_prior == pytest.approx(expected_prior)
    assert state.log_target == pytest.approx(
        expected_likelihood
        + expected_prior
        + state.log_coefficient_prior
        + state.log_k_prior
        + state.log_nucleus_prior
    )


def test_error_model_requires_aligned_nugget_and_positive_prior_bounds() -> None:
    """OU data alignment and strictly positive parameter support are enforced."""
    problem = _problem(inferred_error=True)
    with pytest.raises(ValueError, match="exactly match"):
        replace(problem, observation_sd=problem.observation_sd + 0.01)
    assert problem.error_model is not None
    with pytest.raises(ValueError, match="strictly positive"):
        replace(
            problem.error_model,
            correlation_timescale_prior_lower=np.array([0.0, 1.0]),
        )


def test_shared_hyperprior_is_counted_once_for_all_active_coefficients() -> None:
    """A shared hierarchy must add one hyperprior, not one copy per region."""
    problem = _problem(hierarchy=True)
    state = build_state(
        problem,
        [0, 4],
        [0.8, 1.3],
        coefficient_prior_mean=1.2,
        coefficient_prior_sd=0.9,
    )
    assert problem.coefficient_hierarchy is not None
    expected_conditional = shared_coefficient_log_prior_numpy(
        state.coefficients,
        state.k,
        state.eta,
        state.zeta,
    )
    expected_hyperprior = shared_hyperprior_log_density_numpy(
        state.eta,
        state.zeta,
        problem.coefficient_hierarchy,
    )

    assert state.log_coefficient_prior == pytest.approx(expected_conditional)
    assert state.log_coefficient_hyperprior == pytest.approx(expected_hyperprior)
    target_without_hyperprior = state.log_target - state.log_coefficient_hyperprior
    assert state.log_target == pytest.approx(target_without_hyperprior + expected_hyperprior)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_full_fast_and_structural_builders_preserve_complete_target(backend: str) -> None:
    """Fast coefficient and structural paths must equal complete rebuilds."""
    problem = _problem(inferred_error=True, hierarchy=True)
    state = build_state(
        problem,
        [0, 4],
        [0.8, 1.3],
        mismatch_sd=[0.4, 1.2],
        correlation_timescale=[1.5, 3.0],
        coefficient_prior_mean=1.2,
        coefficient_prior_sd=0.9,
        backend=backend,  # type: ignore[arg-type]
    )
    fast = update_dynamic_coefficient_state(
        problem,
        state,
        coefficient_position=1,
        proposed_coefficient=1.6,
        backend=backend,  # type: ignore[arg-type]
    )
    full_fast = build_state(
        problem,
        fast.active_nuclei,
        fast.active_coefficients,
        mismatch_sd=fast.mismatch_sd,
        correlation_timescale=fast.correlation_timescale,
        coefficient_prior_mean=float(np.exp(fast.eta)),
        coefficient_prior_sd=float(np.exp(fast.zeta)),
        backend=backend,  # type: ignore[arg-type]
    )
    _assert_state_equal(fast, full_fast)

    structural = update_structural_state(
        problem,
        fast,
        [0, 2, 4],
        [0.8, 1.1, 1.6],
        backend=backend,  # type: ignore[arg-type]
    )
    full_structural = build_state(
        problem,
        [0, 2, 4],
        [0.8, 1.1, 1.6],
        mismatch_sd=fast.mismatch_sd,
        correlation_timescale=fast.correlation_timescale,
        coefficient_prior_mean=float(np.exp(fast.eta)),
        coefficient_prior_sd=float(np.exp(fast.zeta)),
        backend=backend,  # type: ignore[arg-type]
    )
    _assert_state_equal(structural, full_structural)


def test_error_and_hierarchy_updates_preserve_unrelated_caches() -> None:
    """Parameter-only updates must share geometry, predictions, and residuals."""
    problem = _problem(inferred_error=True, hierarchy=True)
    state = build_state(
        problem,
        [0, 4],
        [0.8, 1.3],
        mismatch_sd=[0.4, 1.2],
        correlation_timescale=[1.5, 3.0],
        coefficient_prior_mean=1.2,
        coefficient_prior_sd=0.9,
    )
    error_candidate = update_error_model_state(
        problem,
        state,
        mismatch_sd_position=1,
        proposed_mismatch_sd=1.5,
    )
    for name in ("nuclei", "coefficients", "labels", "design", "prediction", "residual"):
        assert getattr(error_candidate, name) is getattr(state, name)
    assert error_candidate.correlation_timescale is state.correlation_timescale
    assert error_candidate.mismatch_sd is not state.mismatch_sd

    hierarchy_candidate = update_shared_hierarchy_state(
        problem,
        error_candidate,
        proposed_eta=log(1.4),
        proposed_zeta=log(0.6),
    )
    for name in (
        "nuclei",
        "coefficients",
        "labels",
        "design",
        "prediction",
        "residual",
        "mismatch_sd",
        "correlation_timescale",
    ):
        assert getattr(hierarchy_candidate, name) is getattr(error_candidate, name)
    assert hierarchy_candidate.log_likelihood == error_candidate.log_likelihood
    assert hierarchy_candidate.log_error_model_prior == error_candidate.log_error_model_prior
