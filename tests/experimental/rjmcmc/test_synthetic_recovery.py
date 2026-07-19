"""Synthetic lognormal recovery tests for the first spatial TDMCMC sampler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.core import (
    TransDimensionalProblem,
    build_state,
    uniform_log_k_prior,
)
from openghg_inversions.experimental.rjmcmc.sampling import SamplerConfig, sample


@dataclass(frozen=True, slots=True)
class SyntheticRecoveryCase:
    """Complete two-cell enhancement-only inversion with seeded noise."""

    problem: TransDimensionalProblem
    sensitivities: np.ndarray
    truth: np.ndarray
    noise: np.ndarray
    noiseless_observations: np.ndarray


def _build_synthetic_case(*, seed: int = 20260718) -> SyntheticRecoveryCase:
    """Build a deterministic two-cell case with distinguishable sensitivity blocks."""
    observation_count = 40
    observation_sd = 0.05
    block_size = observation_count // 2
    sensitivities = np.zeros((observation_count, 2), dtype=np.float64)
    sensitivities[:block_size, 0] = np.linspace(0.8, 1.2, block_size)
    sensitivities[block_size:, 1] = np.linspace(1.2, 0.8, block_size)
    truth = np.array([0.5, 2.0], dtype=np.float64)
    noiseless_observations = sensitivities @ truth
    noise = np.random.default_rng(seed).normal(scale=observation_sd, size=observation_count)
    observations = noiseless_observations + noise
    problem = TransDimensionalProblem(
        observations=observations,
        observation_sd=np.full(observation_count, observation_sd),
        sensitivities=sensitivities,
        grid_coordinates=np.array([[0.0], [1.0]]),
        k_min=1,
        k_max=2,
        log_k_prior=uniform_log_k_prior(1, 2),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=1.0,
    )
    return SyntheticRecoveryCase(
        problem=problem,
        sensitivities=sensitivities,
        truth=truth,
        noise=noise,
        noiseless_observations=noiseless_observations,
    )


def test_synthetic_observations_and_two_region_representation_close() -> None:
    """Seeded observations and the true K=2 state should follow the declared equations."""
    case = _build_synthetic_case()
    expected_observations = case.sensitivities @ case.truth + case.noise
    true_state = build_state(case.problem, [0, 1], case.truth)

    np.testing.assert_allclose(case.problem.observations, expected_observations, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(true_state.labels, [0, 1])
    np.testing.assert_allclose(true_state.design, case.sensitivities, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        true_state.prediction,
        case.noiseless_observations,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(true_state.residual, -case.noise, rtol=0.0, atol=1e-15)


@pytest.mark.slow
def test_seeded_numba_sampler_recovers_two_regions_and_coefficients() -> None:
    """A 12k-transition chain should favor K=2 and recover both positive scalings."""
    case = _build_synthetic_case()
    initial_state = build_state(case.problem, [0], [1.0], backend="numba")
    result = sample(
        case.problem,
        initial_state,
        SamplerConfig(
            iterations=12_000,
            coefficient_proposal_sd=0.05,
            birth_proposal_sd=0.5,
            seed=1701,
            backend="numba",
        ),
    )

    retained_k = result.trace.k[3_000:]
    retained_coefficients = result.trace.coefficients[3_000:]
    k_two = retained_k == 2
    k_two_mass = float(np.mean(k_two))
    coefficient_medians = np.median(retained_coefficients[k_two, :2], axis=0)

    assert k_two_mass > 0.9
    np.testing.assert_allclose(coefficient_medians, case.truth, rtol=0.0, atol=0.1)
