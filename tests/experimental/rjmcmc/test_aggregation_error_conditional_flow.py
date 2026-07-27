"""Tests for the normalized residual-image conditional flow and simulator."""

from __future__ import annotations

import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_flow import (
    ConditionalResidualImageFlow,
    conditional_residual_unit_covariances,
    make_conditional_residual_flow,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mdn import (
    ResidualImageContext,
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
        source_provenance="conditional-flow unit-test context",
    )


def _artifact(
    *,
    labels: np.ndarray | None = None,
) -> ConditionalResidualImageFlow:
    """Return an initialized identity flow with exact covariance whitening."""
    aggregation = _aggregation()
    context = _context(aggregation, labels=labels)
    return ConditionalResidualImageFlow(
        context,
        conditional_residual_unit_covariances(aggregation, context),
        np.zeros(context.region_count, dtype=np.float64),
        np.ones(context.region_count, dtype=np.float64),
        make_conditional_residual_flow(
            context.residual_rank,
            context.region_count,
            source_seed=17,
        ),
        initialization_seed=17,
        source_provenance="conditional-flow unit-test artifact",
    )


def test_exact_unit_covariance_matches_independent_dirichlet_formula() -> None:
    """The stored covariance must be the exact conditional Dirichlet result."""
    aggregation = _aggregation()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    context = _context(aggregation, labels=labels)
    actual = conditional_residual_unit_covariances(aggregation, context)

    expected = np.empty_like(actual)
    whitened_design = aggregation.design / aggregation.noise_sd[:, None]
    for region in range(context.region_count):
        selected = context.labels.reshape(-1) == region
        alpha = aggregation.cell_alphas.reshape(-1)[selected]
        mean = alpha / alpha.sum()
        covariance = (np.diag(mean) - np.outer(mean, mean)) / (alpha.sum() + 1.0)
        projected = context.residual_basis.T @ whitened_design[:, selected]
        expected[region] = projected @ covariance @ projected.T

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3.0e-15)
    assert not actual.flags.writeable


def test_flow_likelihood_matches_independent_change_of_variables() -> None:
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
    orthogonal = (
        residual
        - artifact.context.residual_basis @ coordinates
    )
    cholesky = artifact.projected_cholesky(masses)
    standardized = np.linalg.solve(cholesky, coordinates)
    expected = (
        -float(np.log(artifact.context.noise_sd).sum())
        - 0.5
        * (
            (artifact.context.observation_count - artifact.residual_rank)
            * math.log(2.0 * math.pi)
            + float(orthogonal @ orthogonal)
        )
        + float(
            artifact.flow.log_prob(
                jnp.asarray(standardized),
                jnp.asarray(artifact.conditioner(masses)),
            )
        )
        - float(np.log(np.diag(cholesky)).sum())
    )
    assert artifact.log_likelihood(
        observation,
        masses,
        offset=offset,
    ) == pytest.approx(expected, abs=2.0e-13)


def test_simulator_exactly_reconstructs_flow_and_orthogonal_draws() -> None:
    """Forward samples must use the fitted flow and exact Gaussian complement."""
    artifact = _artifact()
    masses = np.asarray([1.7])
    offset = np.asarray([0.1, -0.2, 0.05])
    sample_count = 4_096
    source_seed = 29
    samples = artifact.sample_observation(
        masses,
        sample_count=sample_count,
        source_seed=source_seed,
        offset=offset,
    )
    flow_key, orthogonal_key = jr.split(jr.key(source_seed))
    standardized = np.asarray(
        artifact.flow.sample(
            flow_key,
            (sample_count,),
            condition=jnp.asarray(artifact.conditioner(masses)),
        )
    )
    coordinates = standardized @ artifact.projected_cholesky(masses).T
    gaussian = np.asarray(
        jr.normal(
            orthogonal_key,
            (sample_count, artifact.context.observation_count),
            dtype=jnp.float64,
        )
    )
    basis = artifact.context.residual_basis
    orthogonal = gaussian - (gaussian @ basis) @ basis.T
    expected = (
        offset
        + artifact.context.observation_mean_design @ masses
        + (coordinates @ basis.T + orthogonal) * artifact.context.noise_sd
    )

    assert samples.shape == (sample_count, artifact.context.observation_count)
    np.testing.assert_array_equal(samples, expected)


def test_conditioner_uses_log_total_then_canonical_alr_shares() -> None:
    """The flow conditioner must preserve total and labelled share information."""
    artifact = _artifact(labels=np.asarray([0, 0, 1, 1], dtype=np.int64))
    masses = np.asarray([1.2, 3.5])
    expected = np.asarray(
        [
            math.log(float(masses.sum())),
            math.log(float(masses[0] / masses[1])),
        ]
    )
    np.testing.assert_allclose(
        artifact.conditioner(masses),
        expected,
        rtol=0.0,
        atol=2.0e-16,
    )


def test_flow_sampling_log_prob_and_serialization_replay_exactly() -> None:
    """The flow sampler, density, and authenticated bytes must replay."""
    artifact = _artifact()
    condition = jnp.asarray(artifact.conditioner([1.3]), dtype=jnp.float64)
    samples, sampled_log_prob = artifact.flow.sample_and_log_prob(
        jr.key(41),
        (64,),
        condition=condition,
    )
    separate_log_prob = artifact.flow.log_prob(samples, condition)
    np.testing.assert_allclose(
        sampled_log_prob,
        separate_log_prob,
        rtol=0.0,
        atol=1.0e-10,
    )

    serialized = artifact.to_bytes()
    replay = ConditionalResidualImageFlow.from_bytes(
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
        ConditionalResidualImageFlow.from_bytes(
            bytes(corrupted),
            expected_sha256=artifact.artifact_sha256,
        )


def test_batch_likelihood_and_preprocessing_match_scalar_methods() -> None:
    """Vectorized grid evaluation must replay the scalar likelihood exactly."""
    artifact = _artifact(labels=np.asarray([0, 0, 1, 1], dtype=np.int64))
    observation = np.asarray([0.2, -0.4, 1.1])
    masses = np.asarray(
        [
            [0.4, 0.8],
            [1.2, 3.5],
            [4.0, 0.6],
        ]
    )
    batch = artifact.log_likelihood_batch(observation, masses)
    scalar = np.asarray(
        [artifact.log_likelihood(observation, state) for state in masses]
    )
    np.testing.assert_allclose(batch, scalar, rtol=0.0, atol=3.0e-13)
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
    ("masses", "message"),
    [
        ([1.0, 2.0], "one finite strictly positive"),
        ([0.0], "one finite strictly positive"),
        ([np.nan], "one finite strictly positive"),
    ],
)
def test_malformed_masses_fail_closed(
    masses: list[float],
    message: str,
) -> None:
    """Mass dimension, support, and finiteness are strict."""
    artifact = _artifact()
    with pytest.raises(ValueError, match=message):
        artifact.log_likelihood(np.zeros(3), masses)
    with pytest.raises(ValueError, match=message):
        artifact.sample_observation(
            masses,
            sample_count=2,
            source_seed=1,
        )


def test_covariance_and_artifact_context_mismatches_fail_closed() -> None:
    """Native source arrays and flow event dimensions are authenticated."""
    aggregation = _aggregation()
    context = _context(aggregation)
    changed = AdditiveDirichletAggregation(
        aggregation.cell_alphas,
        aggregation.design + 0.01,
        aggregation.noise_sd,
        np.empty((3, 0), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="design"):
        conditional_residual_unit_covariances(changed, context)

    with pytest.raises(ValueError, match="event shape"):
        ConditionalResidualImageFlow(
            context,
            conditional_residual_unit_covariances(aggregation, context),
            np.zeros(1),
            np.ones(1),
            make_conditional_residual_flow(1, 1, source_seed=7),
            initialization_seed=7,
            source_provenance="wrong event shape",
        )
