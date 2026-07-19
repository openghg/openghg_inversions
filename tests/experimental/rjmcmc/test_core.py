"""Focused tests for the trans-dimensional MCMC numerical core."""

from __future__ import annotations

from math import comb, exp, log

import numpy as np
import pytest
from scipy import stats

from openghg_inversions.experimental.rjmcmc.core import (
    TransDimensionalProblem,
    aggregate_design_numpy,
    assign_cells_numpy,
    build_state,
    gaussian_log_likelihood_numpy,
    lognormal_coefficient_log_prior_numpy,
    lognormal_mu_sigma,
    uniform_log_k_prior,
    uniform_nucleus_set_log_prior,
)


def _problem_kwargs() -> dict[str, object]:
    """Return independent constructor inputs for the hand-calculated problem."""
    return {
        "observations": np.array([16.0, 2.0, 3.0]),
        "observation_sd": np.array([1.0, 2.0, 0.5]),
        "sensitivities": np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.5, 0.0, 1.0, 0.0],
                [-1.0, 2.0, 0.0, 1.0],
            ]
        ),
        "grid_coordinates": np.column_stack((np.arange(4, dtype=float), np.zeros(4))),
        "k_min": 1,
        "k_max": 3,
        "log_k_prior": np.log(np.array([0.2, 0.5, 0.3])),
        "coefficient_prior_mean": 1.0,
        "coefficient_prior_sd": 0.5,
    }


@pytest.fixture
def problem() -> TransDimensionalProblem:
    """Build the hand-calculated one-dimensional four-cell problem."""
    return TransDimensionalProblem(**_problem_kwargs())  # type: ignore[arg-type]


def test_hand_calculated_state_closes_and_is_canonical(problem: TransDimensionalProblem) -> None:
    """Canonical state caches should match the independent one-by-four calculation."""
    supplied_nuclei = np.array([3, 0])
    supplied_coefficients = np.array([2.0, 0.5])

    state = build_state(problem, supplied_nuclei, supplied_coefficients)

    np.testing.assert_array_equal(state.nuclei, [0, 3, -1])
    np.testing.assert_array_equal(state.coefficients, [0.5, 2.0, 0.0])
    np.testing.assert_array_equal(state.labels, [0, 0, 1, 1])
    np.testing.assert_allclose(
        state.design,
        np.array(
            [
                [3.0, 7.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(state.prediction, [15.5, 2.25, 2.5], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(state.residual, [-0.5, 0.25, -0.5], rtol=0.0, atol=0.0)
    assert state.k == 2
    np.testing.assert_array_equal(state.active_nuclei, [0, 3])
    np.testing.assert_array_equal(state.active_coefficients, [0.5, 2.0])

    supplied_nuclei[:] = -1
    supplied_coefficients[:] = -1.0
    np.testing.assert_array_equal(state.active_nuclei, [0, 3])
    np.testing.assert_array_equal(state.active_coefficients, [0.5, 2.0])


def test_voronoi_ties_choose_first_canonical_nucleus(problem: TransDimensionalProblem) -> None:
    """An equidistant cell should use the first active canonical nucleus."""
    nuclei = np.array([0, 2, -1], dtype=np.int64)

    labels = assign_cells_numpy(problem.grid_coordinates, nuclei, k=2)

    np.testing.assert_array_equal(labels, [0, 0, 1, 1])


def test_aggregation_matches_direct_cell_masks_and_zero_pads(problem: TransDimensionalProblem) -> None:
    """Padded design columns should equal explicit sums over labelled fine cells."""
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    design = aggregate_design_numpy(problem.sensitivities, labels, k=2, k_max=3)
    expected = np.zeros((problem.n_observations, problem.k_max))
    for region in range(2):
        expected[:, region] = problem.sensitivities[:, labels == region].sum(axis=1)

    np.testing.assert_allclose(design, expected, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(design[:, 2], 0.0)


def test_state_arrays_are_owned_and_read_only(problem: TransDimensionalProblem) -> None:
    """Problem and state arrays should not alias callers or permit cache mutation."""
    kwargs = _problem_kwargs()
    original_observations = kwargs["observations"]
    owned_problem = TransDimensionalProblem(**kwargs)  # type: ignore[arg-type]
    assert isinstance(original_observations, np.ndarray)
    original_observations[:] = -999.0
    np.testing.assert_array_equal(owned_problem.observations, [16.0, 2.0, 3.0])

    state = build_state(problem, [0, 3], [0.5, 2.0])
    for array in (
        problem.observations,
        problem.observation_sd,
        problem.sensitivities,
        problem.grid_coordinates,
        problem.log_k_prior,
        state.nuclei,
        state.coefficients,
        state.labels,
        state.design,
        state.prediction,
        state.residual,
    ):
        assert not array.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        state.coefficients[0] = 9.0


def test_k_max_state_is_reachable_without_padding(problem: TransDimensionalProblem) -> None:
    """The declared maximum active-region count should form a valid complete state."""
    state = build_state(problem, [3, 0, 1], [2.0, 0.5, 1.25])

    assert state.k == problem.k_max == 3
    np.testing.assert_array_equal(state.nuclei, [0, 1, 3])
    np.testing.assert_array_equal(state.coefficients, [0.5, 1.25, 2.0])
    assert np.all(state.nuclei >= 0)
    assert state.design.shape == (problem.n_observations, problem.k_max)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("observations", np.zeros((1, 3)), "one-dimensional"),
        ("observation_sd", np.ones(2), "same shape"),
        ("observation_sd", np.array([1.0, 0.0, 1.0]), "strictly positive"),
        ("sensitivities", np.zeros((2, 4)), "n_observations"),
        ("sensitivities", np.array([[1.0, np.nan, 2.0, 3.0]] * 3), "finite"),
        ("grid_coordinates", np.zeros((3, 2)), "n_grid_cells"),
        ("grid_coordinates", np.empty((4, 0)), "at least one"),
        ("k_min", 0, "1 <= k_min"),
        ("k_max", 5, "n_grid_cells"),
        ("log_k_prior", np.zeros(2), "one value"),
        ("log_k_prior", np.zeros(3), "normalized"),
        ("log_k_prior", np.array([0.0, -np.inf, np.nan]), "-inf only"),
        ("coefficient_prior_mean", 0.0, "finite and positive"),
        ("coefficient_prior_sd", np.inf, "finite and positive"),
    ],
    ids=(
        "observation-rank",
        "sd-shape",
        "sd-support",
        "sensitivity-shape",
        "sensitivity-finite",
        "coordinate-shape",
        "coordinate-dimension",
        "k-min",
        "k-max",
        "k-prior-shape",
        "k-prior-normalization",
        "k-prior-nan",
        "coefficient-mean",
        "coefficient-sd",
    ),
)
def test_problem_rejects_invalid_inputs(keyword: str, value: object, message: str) -> None:
    """Malformed numerical problems should fail with a targeted validation error."""
    kwargs = _problem_kwargs()
    kwargs[keyword] = value

    with pytest.raises(ValueError, match=message):
        TransDimensionalProblem(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("nuclei", "coefficients", "message"),
    [
        ([[0]], [1.0], "one-dimensional"),
        ([0], [[1.0]], "one-dimensional"),
        ([0], [1.0, 2.0], "equal length"),
        ([], [], "supported range"),
        ([0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0], "supported range"),
        ([-1], [1.0], "valid flattened"),
        ([4], [1.0], "valid flattened"),
        ([0, 0], [1.0, 2.0], "unique"),
        ([0], [np.inf], "finite"),
    ],
    ids=(
        "nuclei-rank",
        "coefficient-rank",
        "unequal-length",
        "below-k-min",
        "above-k-max",
        "negative-nucleus",
        "large-nucleus",
        "duplicate-nucleus",
        "nonfinite-coefficient",
    ),
)
def test_state_builder_rejects_invalid_active_values(
    problem: TransDimensionalProblem,
    nuclei: object,
    coefficients: object,
    message: str,
) -> None:
    """Malformed active state values should fail before numerical kernels run."""
    with pytest.raises(ValueError, match=message):
        build_state(problem, nuclei, coefficients)  # type: ignore[arg-type]


def test_state_builder_rejects_unknown_backend(problem: TransDimensionalProblem) -> None:
    """An unknown numerical backend should fail explicitly."""
    with pytest.raises(ValueError, match="numpy.*numba"):
        build_state(problem, [0], [1.0], backend="jax")  # type: ignore[arg-type]


def test_gaussian_log_likelihood_matches_scipy_normal_density() -> None:
    """The normalized Gaussian likelihood should match an independent SciPy oracle."""
    residual = np.array([-0.5, 0.25, -0.5])
    observation_sd = np.array([1.0, 2.0, 0.5])

    actual = gaussian_log_likelihood_numpy(residual, observation_sd)
    expected = float(stats.norm.logpdf(residual, loc=0.0, scale=observation_sd).sum())

    assert actual == pytest.approx(expected, rel=0.0, abs=1e-14)


def test_lognormal_prior_matches_scipy_and_ignores_padding() -> None:
    """The active normalized lognormal density should match SciPy exactly."""
    coefficients = np.array([0.5, 2.0, -100.0])
    mean = 1.0
    standard_deviation = 0.5
    mu, sigma = lognormal_mu_sigma(mean, standard_deviation)

    actual = lognormal_coefficient_log_prior_numpy(coefficients, 2, mean, standard_deviation)
    expected = float(stats.lognorm.logpdf(coefficients[:2], s=sigma, scale=exp(mu)).sum())

    assert actual == pytest.approx(expected, rel=0.0, abs=1e-14)
    assert exp(mu + 0.5 * sigma**2) == pytest.approx(mean)
    assert (exp(sigma**2) - 1.0) * exp(2.0 * mu + sigma**2) == pytest.approx(standard_deviation**2)


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_lognormal_prior_returns_negative_infinity_outside_support(value: float) -> None:
    """An invalid active lognormal coefficient should have zero prior density."""
    coefficients = np.array([1.0, value])

    assert lognormal_coefficient_log_prior_numpy(coefficients, 2, 1.0, 0.5) == -np.inf


def test_uniform_k_prior_is_normalized_and_read_only() -> None:
    """The discrete-uniform K prior should assign normalized equal mass."""
    log_prior = uniform_log_k_prior(2, 5)

    np.testing.assert_allclose(log_prior, -log(4.0), rtol=0.0, atol=0.0)
    assert np.exp(log_prior).sum() == pytest.approx(1.0)
    assert not log_prior.flags.writeable


@pytest.mark.parametrize(("k_min", "k_max"), [(0, 2), (3, 2)])
def test_uniform_k_prior_rejects_invalid_bounds(k_min: int, k_max: int) -> None:
    """Invalid discrete-uniform K bounds should fail explicitly."""
    with pytest.raises(ValueError, match="k_min"):
        uniform_log_k_prior(k_min, k_max)


@pytest.mark.parametrize("k", [0, 1, 2, 4])
def test_uniform_nucleus_set_prior_is_conditionally_normalized(k: int) -> None:
    """Every unordered nucleus set should receive reciprocal-combination mass."""
    n_grid_cells = 4
    log_prior = uniform_nucleus_set_log_prior(n_grid_cells, k)

    assert log_prior == pytest.approx(-log(comb(n_grid_cells, k)), abs=1e-14)
    assert comb(n_grid_cells, k) * exp(log_prior) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [-1, 5])
def test_uniform_nucleus_set_prior_has_zero_mass_outside_support(k: int) -> None:
    """Invalid nucleus counts should have zero conditional prior mass."""
    assert uniform_nucleus_set_log_prior(4, k) == -np.inf


def test_state_target_contains_all_normalized_factors(problem: TransDimensionalProblem) -> None:
    """The cached log target should include likelihood and every declared prior."""
    state = build_state(problem, [0, 3], [0.5, 2.0])
    expected_likelihood = float(stats.norm.logpdf(state.residual, scale=problem.observation_sd).sum())
    mu, sigma = lognormal_mu_sigma(problem.coefficient_prior_mean, problem.coefficient_prior_sd)
    expected_coefficient_prior = float(
        stats.lognorm.logpdf(state.active_coefficients, s=sigma, scale=exp(mu)).sum()
    )
    expected = (
        expected_likelihood + expected_coefficient_prior + log(0.5) - log(comb(problem.n_grid_cells, state.k))
    )

    assert state.log_target == pytest.approx(expected, rel=0.0, abs=1e-13)
