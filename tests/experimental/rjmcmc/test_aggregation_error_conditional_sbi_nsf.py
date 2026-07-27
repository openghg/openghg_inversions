"""Tests for the normalized sbi NSF likelihood and simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mdn import (
    ResidualImageContext,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_sbi_nsf import (
    ConditionalResidualImageSbiNsf,
    conditional_residual_unit_covariances,
    make_conditional_residual_nsf,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)


def _aggregation() -> AdditiveDirichletAggregation:
    """Return a heterogeneous four-cell model with three observations."""
    return AdditiveDirichletAggregation(
        np.asarray([0.7, 1.1, 1.6, 0.9], dtype=np.float64),
        np.asarray(
            [
                [1.8, -0.5, 0.3, 0.9],
                [0.2, 1.4, -0.7, 0.1],
                [0.5, -0.2, 1.1, 0.8],
            ],
            dtype=np.float64,
        ),
        np.asarray([0.35, 0.8, 0.6], dtype=np.float64),
        np.empty((3, 0), dtype=np.float64),
    )


def _context(
    aggregation: AdditiveDirichletAggregation,
    *,
    labels: np.ndarray | None = None,
) -> ResidualImageContext:
    """Return one authenticated context."""
    return ResidualImageContext.from_aggregation(
        aggregation,
        (
            np.zeros(4, dtype=np.int64)
            if labels is None
            else np.asarray(labels, dtype=np.int64)
        ),
        np.asarray([101, 102, 201, 202], dtype=np.int64),
        source_provenance="sbi-NSF unit-test context",
    )


def _artifact(
    *,
    labels: np.ndarray | None = None,
) -> ConditionalResidualImageSbiNsf:
    """Return one initialized NSF with exact covariance whitening."""
    aggregation = _aggregation()
    context = _context(aggregation, labels=labels)
    return ConditionalResidualImageSbiNsf(
        context,
        conditional_residual_unit_covariances(aggregation, context),
        np.zeros(context.region_count, dtype=np.float64),
        np.ones(context.region_count, dtype=np.float64),
        make_conditional_residual_nsf(
            context.residual_rank,
            context.region_count,
            source_seed=17,
        ),
        initialization_seed=17,
        source_provenance="sbi-NSF unit-test artifact",
    )


def test_nsf_likelihood_matches_independent_change_of_variables() -> None:
    """The observation density must include every declared Jacobian term."""
    artifact = _artifact()
    masses = np.asarray([2.4])
    offset = np.asarray([0.1, -0.2, 0.05])
    observation = np.asarray([1.2, -0.4, 0.8])
    residual = (
        observation
        - offset
        - artifact.context.observation_mean_design @ masses
    ) / artifact.context.noise_sd
    coordinates = artifact.context.residual_basis.T @ residual
    orthogonal = residual - artifact.context.residual_basis @ coordinates
    cholesky = artifact.projected_cholesky(masses)
    standardized = np.linalg.solve(cholesky, coordinates)
    condition = artifact.conditioner(masses)
    with torch.no_grad():
        flow_log_prob = artifact.model.log_prob(
            torch.as_tensor(standardized[None, None, :], dtype=torch.float64),
            condition=torch.as_tensor(condition[None, :], dtype=torch.float64),
        )[0, 0]
    expected = (
        -float(np.log(artifact.context.noise_sd).sum())
        - 0.5
        * (
            (artifact.context.observation_count - artifact.residual_rank)
            * math.log(2.0 * math.pi)
            + float(orthogonal @ orthogonal)
        )
        + float(flow_log_prob)
        - float(np.log(np.diag(cholesky)).sum())
    )
    assert artifact.log_likelihood(
        observation,
        masses,
        offset=offset,
    ) == pytest.approx(expected, abs=2.0e-12)


def test_nsf_sample_log_prob_and_serialization_replay_exactly() -> None:
    """The NSF sampler, density, and authenticated bytes must replay."""
    artifact = _artifact()
    condition = torch.as_tensor(
        artifact.conditioner([1.3])[None, :],
        dtype=torch.float64,
    )
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        samples, sampled_log_prob = artifact.model.sample_and_log_prob(
            torch.Size([64]),
            condition=condition,
        )
    with torch.no_grad():
        separate_log_prob = artifact.model.log_prob(samples, condition)
    np.testing.assert_allclose(
        np.asarray(sampled_log_prob),
        np.asarray(separate_log_prob),
        rtol=0.0,
        atol=1.0e-6,
    )

    serialized = artifact.to_bytes()
    replay = ConditionalResidualImageSbiNsf.from_bytes(
        serialized,
        expected_sha256=artifact.artifact_sha256,
    )
    assert replay.to_bytes() == serialized
    assert replay.artifact_sha256 == artifact.artifact_sha256
    observation = np.asarray([0.2, -0.4, 1.1])
    assert replay.log_likelihood(observation, [1.3]) == artifact.log_likelihood(
        observation,
        [1.3],
    )
    np.testing.assert_array_equal(
        replay.sample_observation([1.3], sample_count=16, source_seed=43),
        artifact.sample_observation([1.3], sample_count=16, source_seed=43),
    )

    corrupted = bytearray(serialized)
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="fingerprint"):
        ConditionalResidualImageSbiNsf.from_bytes(
            bytes(corrupted),
            expected_sha256=artifact.artifact_sha256,
        )


def test_autograd_mass_gradient_matches_central_difference() -> None:
    """The published analytic gradient must differentiate the full density."""
    artifact = _artifact()
    observation = np.asarray([0.2, -0.4, 1.1])
    masses = np.asarray([1.3])
    value, gradient = artifact.log_likelihood_and_mass_gradient(
        observation,
        masses,
    )
    step = 1.0e-5
    finite_difference = (
        artifact.log_likelihood(observation, masses + step)
        - artifact.log_likelihood(observation, masses - step)
    ) / (2.0 * step)
    assert value == artifact.log_likelihood(observation, masses)
    scaled_error = abs(float(gradient[0]) - finite_difference) / (
        1.0 + abs(finite_difference)
    )
    assert scaled_error <= 1.0e-6


def test_batch_likelihood_and_preprocessing_match_scalar_methods() -> None:
    """Vectorized evaluation must replay scalar likelihoods."""
    artifact = _artifact(labels=np.asarray([0, 0, 1, 1], dtype=np.int64))
    observation = np.asarray([0.2, -0.4, 1.1])
    masses = np.asarray(
        [
            [0.4, 0.8],
            [1.2, 3.5],
            [4.0, 0.6],
        ]
    )
    batch = artifact.log_likelihood_batch(observation, masses, batch_size=2)
    scalar = np.asarray(
        [artifact.log_likelihood(observation, state) for state in masses]
    )
    np.testing.assert_allclose(batch, scalar, rtol=0.0, atol=3.0e-12)
    np.testing.assert_allclose(
        artifact.conditioners(masses),
        np.stack([artifact.conditioner(state) for state in masses]),
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        artifact.projected_choleskies(masses),
        np.stack([artifact.projected_cholesky(state) for state in masses]),
        rtol=0.0,
        atol=2.0e-15,
    )


@pytest.mark.parametrize(
    "masses",
    [
        [1.0, 2.0],
        [0.0],
        [np.nan],
    ],
)
def test_malformed_masses_fail_closed(masses: list[float]) -> None:
    """Mass dimension, support, and finiteness are strict."""
    artifact = _artifact()
    with pytest.raises(ValueError, match="one finite strictly positive"):
        artifact.log_likelihood(np.zeros(3), masses)
    with pytest.raises(ValueError, match="one finite strictly positive"):
        artifact.sample_observation(
            masses,
            sample_count=2,
            source_seed=1,
        )


def test_artifact_event_dimension_mismatch_fails_closed() -> None:
    """A model for another event rank cannot enter the artifact."""
    aggregation = _aggregation()
    context = _context(aggregation)
    with pytest.raises(ValueError, match="input shape"):
        ConditionalResidualImageSbiNsf(
            context,
            conditional_residual_unit_covariances(aggregation, context),
            np.zeros(1),
            np.ones(1),
            make_conditional_residual_nsf(1, 1, source_seed=7),
            initialization_seed=7,
            source_provenance="wrong event shape",
        )
