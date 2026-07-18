"""Parity tests for NumPy and Numba trans-dimensional MCMC kernels."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.tdmcmc.core import (
    TransDimensionalProblem,
    TransDimensionalState,
    aggregate_design_numba,
    aggregate_design_numpy,
    assign_cells_numba,
    assign_cells_numpy,
    build_state,
    gaussian_log_likelihood_numba,
    gaussian_log_likelihood_numpy,
    lognormal_coefficient_log_prior_numba,
    lognormal_coefficient_log_prior_numpy,
    uniform_log_k_prior,
)


def _deterministic_problem() -> TransDimensionalProblem:
    """Return the hand-calculated one-by-four parity problem."""
    return TransDimensionalProblem(
        observations=np.array([16.0, 2.0, 3.0]),
        observation_sd=np.array([1.0, 2.0, 0.5]),
        sensitivities=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.5, 0.0, 1.0, 0.0],
                [-1.0, 2.0, 0.0, 1.0],
            ]
        ),
        grid_coordinates=np.column_stack((np.arange(4, dtype=float), np.zeros(4))),
        k_min=1,
        k_max=3,
        log_k_prior=np.log(np.array([0.2, 0.5, 0.3])),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.5,
    )


def _random_case(seed: int) -> tuple[TransDimensionalProblem, np.ndarray, np.ndarray]:
    """Create one seeded valid problem and active state for backend comparison."""
    rng = np.random.default_rng(seed)
    n_observations = 7
    n_grid_cells = 8
    k_max = 6
    k = seed % k_max + 1
    problem = TransDimensionalProblem(
        observations=rng.normal(size=n_observations),
        observation_sd=rng.uniform(0.1, 2.0, size=n_observations),
        sensitivities=rng.normal(size=(n_observations, n_grid_cells)),
        grid_coordinates=rng.normal(size=(n_grid_cells, 2)),
        k_min=1,
        k_max=k_max,
        log_k_prior=uniform_log_k_prior(1, k_max),
        coefficient_prior_mean=1.2,
        coefficient_prior_sd=0.7,
    )
    nuclei = rng.choice(n_grid_cells, size=k, replace=False)
    coefficients = rng.lognormal(mean=-0.1, sigma=0.4, size=k)
    return problem, nuclei, coefficients


def _assert_state_parity(numpy_state: TransDimensionalState, numba_state: TransDimensionalState) -> None:
    """Assert exact discrete and tight floating-point state agreement."""
    assert numpy_state.k == numba_state.k
    np.testing.assert_array_equal(numpy_state.nuclei, numba_state.nuclei)
    np.testing.assert_array_equal(numpy_state.labels, numba_state.labels)
    np.testing.assert_array_equal(numpy_state.coefficients, numba_state.coefficients)
    for numpy_value, numba_value in (
        (numpy_state.design, numba_state.design),
        (numpy_state.prediction, numba_state.prediction),
        (numpy_state.residual, numba_state.residual),
    ):
        np.testing.assert_allclose(numpy_value, numba_value, rtol=0.0, atol=1e-12)
    assert numba_state.log_likelihood == pytest.approx(numpy_state.log_likelihood, rel=0.0, abs=1e-11)
    assert numba_state.log_coefficient_prior == pytest.approx(
        numpy_state.log_coefficient_prior, rel=0.0, abs=1e-12
    )
    assert numba_state.log_k_prior == numpy_state.log_k_prior
    assert numba_state.log_nucleus_prior == numpy_state.log_nucleus_prior
    assert numba_state.log_target == pytest.approx(numpy_state.log_target, rel=0.0, abs=1e-11)


def test_deterministic_geometry_and_aggregation_have_backend_parity() -> None:
    """Hand-calculated labels and padded aggregation should match across backends."""
    problem = _deterministic_problem()
    nuclei = np.array([0, 2, -1], dtype=np.int64)
    numpy_labels = assign_cells_numpy(problem.grid_coordinates, nuclei, k=2)
    numba_labels = assign_cells_numba(problem.grid_coordinates, nuclei, k=2)

    np.testing.assert_array_equal(numpy_labels, [0, 0, 1, 1])
    np.testing.assert_array_equal(numba_labels, numpy_labels)
    numpy_design = aggregate_design_numpy(problem.sensitivities, numpy_labels, k=2, k_max=3)
    numba_design = aggregate_design_numba(problem.sensitivities, numba_labels, k=2, k_max=3)
    np.testing.assert_allclose(numba_design, numpy_design, rtol=0.0, atol=0.0)


def test_deterministic_complete_state_has_backend_parity() -> None:
    """Canonical state construction should agree for the hand-calculated fixture."""
    problem = _deterministic_problem()

    numpy_state = build_state(problem, [3, 0], [2.0, 0.5], backend="numpy")
    numba_state = build_state(problem, [3, 0], [2.0, 0.5], backend="numba")

    _assert_state_parity(numpy_state, numba_state)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 11, 29])
def test_seeded_random_complete_states_have_backend_parity(seed: int) -> None:
    """Seeded varied dimensions and active counts should retain backend parity."""
    problem, nuclei, coefficients = _random_case(seed)

    numpy_state = build_state(problem, nuclei, coefficients, backend="numpy")
    numba_state = build_state(problem, nuclei, coefficients, backend="numba")

    _assert_state_parity(numpy_state, numba_state)
    assert numpy_state.k == seed % problem.k_max + 1


@pytest.mark.parametrize("seed", [3, 13, 23])
def test_seeded_random_scalar_kernels_have_backend_parity(seed: int) -> None:
    """Likelihood and active lognormal prior kernels should agree independently."""
    rng = np.random.default_rng(seed)
    residual = rng.normal(size=17)
    observation_sd = rng.uniform(0.05, 3.0, size=17)
    coefficients = rng.lognormal(mean=0.0, sigma=0.8, size=7)

    numpy_likelihood = gaussian_log_likelihood_numpy(residual, observation_sd)
    numba_likelihood = gaussian_log_likelihood_numba(residual, observation_sd)
    numpy_prior = lognormal_coefficient_log_prior_numpy(coefficients, 5, 1.3, 0.6)
    numba_prior = lognormal_coefficient_log_prior_numba(coefficients, 5, 1.3, 0.6)

    assert numba_likelihood == pytest.approx(numpy_likelihood, rel=0.0, abs=1e-12)
    assert numba_prior == pytest.approx(numpy_prior, rel=0.0, abs=1e-12)


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_out_of_support_lognormal_prior_has_backend_parity(value: float) -> None:
    """Both backends should assign zero density to invalid active coefficients."""
    coefficients = np.array([1.0, value, 99.0])

    numpy_prior = lognormal_coefficient_log_prior_numpy(coefficients, 2, 1.0, 0.5)
    numba_prior = lognormal_coefficient_log_prior_numba(coefficients, 2, 1.0, 0.5)

    assert numpy_prior == numba_prior == -np.inf
