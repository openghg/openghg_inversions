"""Independent continuous-coefficient checks for RJ structural transitions."""

from __future__ import annotations

from math import erf, exp, log, pi, sqrt

import numpy as np
import pytest
from scipy import integrate

from openghg_inversions.experimental.rjmcmc.core import (
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
)
from openghg_inversions.experimental.rjmcmc.proposals import propose_birth, propose_death


def _informative_problem() -> TransDimensionalProblem:
    """Return an irregular problem whose target varies with locations and coefficients."""
    return TransDimensionalProblem(
        observations=np.array([1.3, -0.8, 0.2]),
        observation_sd=np.array([0.35, 0.9, 1.4]),
        sensitivities=np.array(
            [
                [0.5, -0.1, 1.2, 0.0, 0.8, -0.4],
                [0.2, 0.7, -0.3, 1.1, 0.4, 0.6],
                [-0.5, 0.1, 0.3, 0.9, -0.2, 0.7],
            ]
        ),
        grid_coordinates=np.array([[0.0], [0.7], [2.1], [5.0], [6.2], [10.0]]),
        k_min=1,
        k_max=4,
        log_k_prior=np.log(np.array([0.05, 0.2, 0.35, 0.4])),
        coefficient_prior_mean=1.1,
        coefficient_prior_sd=0.65,
    )


def _likelihood_free_problem() -> TransDimensionalProblem:
    """Return a two-cell target whose coefficient and count marginals are analytic."""
    return TransDimensionalProblem(
        observations=np.array([0.0]),
        observation_sd=np.array([1.0]),
        sensitivities=np.zeros((1, 2)),
        grid_coordinates=np.array([[0.0], [1.0]]),
        k_min=1,
        k_max=2,
        log_k_prior=np.log(np.array([0.5, 0.5])),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.5,
    )


def _assert_same_state(actual: TransDimensionalState, expected: TransDimensionalState) -> None:
    """Assert that a reciprocal transition reconstructs the complete source state."""
    assert actual.k == expected.k
    for actual_array, expected_array in (
        (actual.nuclei, expected.nuclei),
        (actual.coefficients, expected.coefficients),
        (actual.labels, expected.labels),
        (actual.design, expected.design),
        (actual.prediction, expected.prediction),
        (actual.residual, expected.residual),
    ):
        np.testing.assert_array_equal(actual_array, expected_array)
    assert actual.log_target == pytest.approx(expected.log_target, rel=0.0, abs=2e-13)


def _independent_parent_coefficient(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    destination: int,
) -> float:
    """Calculate the nearest-region coefficient without proposal-module helpers."""
    offsets = problem.grid_coordinates[state.active_nuclei] - problem.grid_coordinates[destination]
    parent_position = int(np.argmin(np.einsum("ij,ij->i", offsets, offsets)))
    return float(state.active_coefficients[parent_position])


def _lognormal_parameters(mean: float, standard_deviation: float) -> tuple[float, float]:
    """Convert arithmetic moments to SciPy's lognormal shape and scale."""
    sigma_squared = log(1.0 + (standard_deviation / mean) ** 2)
    sigma = np.sqrt(sigma_squared)
    mu = log(mean) - 0.5 * sigma_squared
    return float(sigma), exp(mu)


def _lognormal_density(value: float, *, shape: float, scale: float) -> float:
    """Evaluate a lognormal density independently of SciPy's frozen stubs."""
    if value <= 0.0:
        return 0.0
    standardized = (log(value) - log(scale)) / shape
    return exp(-0.5 * standardized**2) / (value * shape * sqrt(2.0 * pi))


def _normal_log_density(value: float, *, mean: float, standard_deviation: float) -> float:
    """Evaluate a normalized Gaussian log density independently."""
    standardized = (value - mean) / standard_deviation
    return -0.5 * standardized**2 - log(standard_deviation) - 0.5 * log(2.0 * pi)


def _normal_density(value: float, *, mean: float, standard_deviation: float) -> float:
    """Evaluate a normalized Gaussian density independently."""
    return exp(_normal_log_density(value, mean=mean, standard_deviation=standard_deviation))


def _normal_probability_below_zero(*, mean: float, standard_deviation: float) -> float:
    """Return the probability that one Gaussian draw is non-positive."""
    return 0.5 * (1.0 + erf(-mean / (standard_deviation * sqrt(2.0))))


def test_varied_continuous_birth_death_pairs_have_reciprocal_flux() -> None:
    """Varied coefficients, dimensions, locations, and scales should balance pointwise."""
    problem = _informative_problem()
    rng = np.random.default_rng(814237)

    for _ in range(64):
        k = int(rng.integers(problem.k_min, problem.k_max))
        nuclei = np.sort(rng.choice(problem.ncell, size=k, replace=False))
        coefficients = np.exp(rng.normal(loc=-0.15, scale=0.9, size=k))
        source = build_state(problem, nuclei, coefficients)
        unoccupied = np.setdiff1d(np.arange(problem.ncell), source.active_nuclei)
        destination = int(rng.choice(unoccupied))
        proposed_coefficient = float(np.exp(rng.normal(loc=-0.2, scale=1.15)))
        proposal_stdev = float(np.exp(rng.uniform(log(0.025), log(2.5))))

        birth = propose_birth(
            problem,
            source,
            new_nucleus=destination,
            proposed_coefficient=proposed_coefficient,
            proposal_stdev=proposal_stdev,
        )
        added_position = int(np.flatnonzero(birth.candidate.active_nuclei == destination)[0])
        death = propose_death(
            problem,
            birth.candidate,
            remove_position=added_position,
            proposal_stdev=proposal_stdev,
        )
        parent_coefficient = _independent_parent_coefficient(problem, source, destination)
        expected_birth_log_q = -log(problem.ncell - source.k) + _normal_log_density(
            proposed_coefficient,
            mean=parent_coefficient,
            standard_deviation=proposal_stdev,
        )

        assert birth.valid and death.valid
        assert birth.log_q_forward == pytest.approx(expected_birth_log_q, abs=2e-12)
        assert birth.log_q_reverse == pytest.approx(-log(source.k + 1), abs=2e-12)
        assert death.log_q_forward == pytest.approx(birth.log_q_reverse, abs=2e-12)
        assert death.log_q_reverse == pytest.approx(birth.log_q_forward, abs=2e-12)
        assert death.log_acceptance_ratio == pytest.approx(
            -birth.log_acceptance_ratio,
            rel=2e-13,
            abs=2e-11,
        )
        _assert_same_state(death.candidate, source)

        forward_log_flux = source.log_target + birth.log_q_forward + min(0.0, birth.log_acceptance_ratio)
        reverse_log_flux = (
            birth.candidate.log_target + death.log_q_forward + min(0.0, death.log_acceptance_ratio)
        )
        assert forward_log_flux == pytest.approx(reverse_log_flux, rel=0.0, abs=3e-11)


@pytest.mark.parametrize(
    ("source_coefficient", "proposal_stdev"),
    [(0.08, 0.7), (0.9, 0.15), (2.0, 1.1)],
)
def test_quadrature_matches_analytic_continuous_overlap_and_self_mass(
    source_coefficient: float,
    proposal_stdev: float,
) -> None:
    """Quadrature should recover analytic flux and all rejected Gaussian mass."""
    problem = _likelihood_free_problem()
    source = build_state(problem, [0], [source_coefficient])
    prior_shape, prior_scale = _lognormal_parameters(
        problem.coefficient_prior_mean,
        problem.coefficient_prior_sd,
    )

    def accepted_up_density(value: float) -> float:
        transition = propose_birth(
            problem,
            source,
            new_nucleus=1,
            proposed_coefficient=value,
            proposal_stdev=proposal_stdev,
        )
        proposal_density = _normal_density(
            value,
            mean=source_coefficient,
            standard_deviation=proposal_stdev,
        )
        return proposal_density * exp(min(0.0, transition.log_acceptance_ratio))

    def analytic_overlap_density(value: float) -> float:
        prior_density = _lognormal_density(value, shape=prior_shape, scale=prior_scale)
        proposal_density = _normal_density(
            value,
            mean=source_coefficient,
            standard_deviation=proposal_stdev,
        )
        return min(proposal_density, prior_density)

    def reverse_target_flux_density(value: float) -> float:
        birth = propose_birth(
            problem,
            source,
            new_nucleus=1,
            proposed_coefficient=value,
            proposal_stdev=proposal_stdev,
        )
        added_position = int(np.flatnonzero(birth.candidate.active_nuclei == 1)[0])
        death = propose_death(
            problem,
            birth.candidate,
            remove_position=added_position,
            proposal_stdev=proposal_stdev,
        )
        candidate_target_density = (
            0.5
            * _lognormal_density(source_coefficient, shape=prior_shape, scale=prior_scale)
            * _lognormal_density(value, shape=prior_shape, scale=prior_scale)
        )
        return float(candidate_target_density * 0.5 * exp(min(0.0, death.log_acceptance_ratio)))

    accepted_mass, accepted_error = integrate.quad(
        accepted_up_density,
        0.0,
        np.inf,
        epsabs=2e-11,
        epsrel=2e-10,
        limit=150,
    )
    overlap_mass, overlap_error = integrate.quad(
        analytic_overlap_density,
        0.0,
        np.inf,
        epsabs=2e-11,
        epsrel=2e-10,
        limit=150,
    )
    reverse_flux, reverse_error = integrate.quad(
        reverse_target_flux_density,
        0.0,
        np.inf,
        epsabs=2e-11,
        epsrel=2e-10,
        limit=150,
    )

    source_target_density = (
        0.5
        * 0.5
        * _lognormal_density(
            source_coefficient,
            shape=prior_shape,
            scale=prior_scale,
        )
    )
    forward_flux = source_target_density * accepted_mass
    negative_invalid_mass = _normal_probability_below_zero(
        mean=source_coefficient,
        standard_deviation=proposal_stdev,
    )
    positive_proposal_mass = 1.0 - negative_invalid_mass
    positive_rejection_mass = positive_proposal_mass - accepted_mass
    mixed_kernel_row_mass = 0.5 + 0.5 * (accepted_mass + positive_rejection_mass + negative_invalid_mass)

    assert accepted_error < 2e-9
    assert overlap_error < 2e-9
    assert reverse_error < 2e-9
    assert accepted_mass == pytest.approx(overlap_mass, rel=3e-9, abs=3e-11)
    assert forward_flux == pytest.approx(reverse_flux, rel=4e-9, abs=3e-12)
    assert negative_invalid_mass > 0.0
    assert positive_rejection_mass >= -2e-11
    assert mixed_kernel_row_mass == pytest.approx(1.0, rel=0.0, abs=2e-15)

    for invalid_coefficient in (0.0, -0.25, -4.0):
        invalid = propose_birth(
            problem,
            source,
            new_nucleus=1,
            proposed_coefficient=invalid_coefficient,
            proposal_stdev=proposal_stdev,
        )
        assert not invalid.valid
        assert invalid.candidate is source
        assert invalid.log_acceptance_ratio == -np.inf
