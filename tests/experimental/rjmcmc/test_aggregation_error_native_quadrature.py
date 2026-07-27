"""Focused tests for the support-exact native quadrature marginal."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_native_quadrature import (
    ConditionalNativeQuadrature,
)

_GIT_SHA = "0" * 40
_DRIVER_SHA = "1" * 64
_PROTOCOL_SHA = "2" * 64


def _artifact(
    alphas: np.ndarray,
    design: np.ndarray,
    noise: np.ndarray,
    *,
    order: int,
    chart: str,
    cell_ids: np.ndarray | None = None,
) -> ConditionalNativeQuadrature:
    aggregation = AdditiveDirichletAggregation(
        alphas,
        design,
        noise,
        np.eye(design.shape[0], dtype=np.float64),
    )
    ids = np.arange(alphas.size, dtype=np.int64) if cell_ids is None else cell_ids
    return ConditionalNativeQuadrature.from_aggregation(
        aggregation,
        np.zeros(alphas.shape, dtype=np.int64),
        ids,
        quadrature_order=order,
        chart=chart,  # type: ignore[arg-type]
        source_git_revision=_GIT_SHA,
        driver_sha256=_DRIVER_SHA,
        protocol_sha256=_PROTOCOL_SHA,
        source_provenance="focused-test",
    )


def _four_cell_artifact(
    *,
    order: int = 8,
    chart: str = "column-first",
) -> ConditionalNativeQuadrature:
    return _artifact(
        np.asarray([0.35, 4.0, 1.2, 8.0], dtype=np.float64),
        np.asarray(
            [
                [1.80, 0.10, 0.50, -0.20],
                [-0.40, 1.20, 0.20, 0.85],
                [0.80, -0.30, 1.45, 0.10],
                [0.20, 0.35, -0.15, 1.60],
            ],
            dtype=np.float64,
        ),
        np.asarray([0.22, 0.30, 0.26, 0.34], dtype=np.float64),
        order=order,
        chart=chart,
    )


def test_two_cell_quadrature_is_normalized_and_support_exact() -> None:
    artifact = _artifact(
        np.asarray([0.12, 0.18], dtype=np.float64),
        np.asarray([[2.0, 0.0], [0.0, 1.7], [1.0, -1.0]]),
        np.asarray([0.12, 0.14, 0.13]),
        order=12,
        chart="single",
    )

    assert artifact.component_count == 12
    assert artifact.residual_rank == 1
    assert np.all(artifact.normalized_weights > 0.0)
    assert np.sum(artifact.normalized_weights) == pytest.approx(1.0, abs=1e-15)
    weighted_mean = artifact.normalized_weights @ artifact.projected_unit_mass_residual_factors[:, :, 0]
    assert weighted_mean == pytest.approx(np.zeros(1), abs=2e-12)


def test_four_cell_quadrature_matches_exact_dirichlet_covariance() -> None:
    artifact = _four_cell_artifact(order=8)
    weights = artifact.normalized_weights
    factors = artifact.projected_unit_mass_residual_factors[:, :, 0]
    mean = weights @ factors
    covariance = np.einsum(
        "s,si,sj->ij",
        weights,
        factors - mean,
        factors - mean,
        optimize=False,
    )
    aggregation = AdditiveDirichletAggregation(
        artifact.cell_alphas,
        np.asarray(
            [
                [1.80, 0.10, 0.50, -0.20],
                [-0.40, 1.20, 0.20, 0.85],
                [0.80, -0.30, 1.45, 0.10],
                [0.20, 0.35, -0.15, 1.60],
            ],
            dtype=np.float64,
        ),
        artifact.context.noise_sd,
        artifact.context.residual_basis,
    )
    exact = aggregation.partition_factors(np.zeros(4, dtype=np.int64)).summary_covariance_factors[0]

    assert np.max(np.abs(mean)) < 2e-14
    assert covariance == pytest.approx(exact, abs=3e-13)


def test_scalar_observation_density_integrates_to_one() -> None:
    artifact = _artifact(
        np.asarray([0.4, 1.7], dtype=np.float64),
        np.asarray([[1.2, -0.3]], dtype=np.float64),
        np.asarray([0.35], dtype=np.float64),
        order=20,
        chart="single",
    )

    integral, error = integrate.quad(
        lambda value: math.exp(artifact.log_likelihood([value], [1.3])),
        -math.inf,
        math.inf,
        epsabs=2e-10,
        epsrel=2e-10,
        limit=200,
    )

    assert integral == pytest.approx(1.0, abs=2e-9)
    assert error < 2e-9


def test_analytic_mass_gradient_matches_centered_difference() -> None:
    artifact = _four_cell_artifact(order=12)
    observation = np.asarray([0.23, 0.83, 0.36, 1.12])
    mass = np.asarray([1.0])
    value, gradient = artifact.log_likelihood_and_mass_gradient(
        observation,
        mass,
    )
    step = 1e-6
    finite_difference = (
        artifact.log_likelihood(observation, mass + step) - artifact.log_likelihood(observation, mass - step)
    ) / (2.0 * step)

    assert math.isfinite(value)
    assert gradient == pytest.approx([finite_difference], rel=2e-8, abs=2e-8)


def test_sampler_reproduces_artifact_moments() -> None:
    artifact = _four_cell_artifact(order=8)
    mass = np.asarray([0.9])
    expected_mean, expected_covariance = artifact.analytic_mean_and_covariance(mass)
    samples, indices = artifact.sample_with_component_indices(
        mass,
        sample_count=100_000,
        rng=np.random.default_rng(771),
    )

    assert samples.shape == (100_000, 4)
    assert indices.shape == (100_000,)
    mean_mcse = np.sqrt(np.diag(expected_covariance) / samples.shape[0])
    assert np.max(np.abs(samples.mean(axis=0) - expected_mean) / mean_mcse) < 4.5
    assert np.all(np.isfinite(samples))


def test_row_and_column_charts_converge_to_same_density_and_gradient() -> None:
    row = _four_cell_artifact(order=16, chart="row-first")
    column = _four_cell_artifact(order=16, chart="column-first")
    observation = np.asarray([0.23, 0.83, 0.36, 1.12])

    row_value, row_gradient = row.log_likelihood_and_mass_gradient(
        observation,
        [1.0],
    )
    column_value, column_gradient = column.log_likelihood_and_mass_gradient(
        observation,
        [1.0],
    )

    assert row_value == pytest.approx(column_value, abs=2e-10)
    assert row_gradient == pytest.approx(column_gradient, abs=2e-9)


def test_native_permutation_leaves_likelihood_unchanged() -> None:
    alphas = np.asarray([0.35, 4.0, 1.2, 8.0], dtype=np.float64)
    design = np.asarray(
        [
            [1.80, 0.10, 0.50, -0.20],
            [-0.40, 1.20, 0.20, 0.85],
            [0.80, -0.30, 1.45, 0.10],
            [0.20, 0.35, -0.15, 1.60],
        ],
        dtype=np.float64,
    )
    noise = np.asarray([0.22, 0.30, 0.26, 0.34])
    original = _artifact(
        alphas,
        design,
        noise,
        order=10,
        chart="column-first",
    )
    permutation = np.asarray([2, 0, 3, 1])
    permuted = _artifact(
        alphas[permutation],
        design[:, permutation],
        noise,
        order=10,
        chart="column-first",
        cell_ids=np.arange(4, dtype=np.int64)[permutation],
    )
    observation = np.asarray([0.23, 0.83, 0.36, 1.12])

    assert original.log_likelihood(observation, [1.1]) == pytest.approx(
        permuted.log_likelihood(observation, [1.1]),
        abs=2e-13,
    )
    assert original.log_likelihood_and_mass_gradient(
        observation,
        [1.1],
    )[1] == pytest.approx(
        permuted.log_likelihood_and_mass_gradient(
            observation,
            [1.1],
        )[1],
        abs=2e-12,
    )


def test_canonical_serialization_replays_and_rejects_tampering() -> None:
    artifact = _four_cell_artifact(order=6)
    serialized = artifact.to_bytes()
    replayed = ConditionalNativeQuadrature.from_bytes(
        serialized,
        expected_sha256=artifact.artifact_sha256,
    )
    observation = np.asarray([0.23, 0.83, 0.36, 1.12])

    assert replayed.to_bytes() == serialized
    assert replayed.artifact_sha256 == artifact.artifact_sha256
    assert replayed.log_likelihood(observation, [1.0]) == artifact.log_likelihood(
        observation,
        [1.0],
    )
    tampered = serialized[:-1] + bytes([serialized[-1] ^ 1])
    with pytest.raises(ValueError, match="SHA-256"):
        ConditionalNativeQuadrature.from_bytes(
            tampered,
            expected_sha256=artifact.artifact_sha256,
        )


@pytest.mark.parametrize(
    ("alphas", "chart"),
    [
        (np.asarray([0.3, 0.7]), "column-first"),
        (np.asarray([0.3, 0.7, 1.1, 2.0]), "single"),
    ],
)
def test_invalid_chart_is_rejected(
    alphas: np.ndarray,
    chart: str,
) -> None:
    with pytest.raises(ValueError, match="require"):
        _artifact(
            alphas,
            np.eye(alphas.size),
            np.ones(alphas.size),
            order=4,
            chart=chart,
        )


def test_malformed_evaluation_inputs_are_rejected() -> None:
    artifact = _four_cell_artifact(order=4)

    with pytest.raises(ValueError, match="observation"):
        artifact.log_likelihood([1.0], [1.0])
    with pytest.raises(ValueError, match="masses"):
        artifact.log_likelihood(np.zeros(4), [-1.0])
    with pytest.raises(TypeError, match="Generator"):
        artifact.sample([1.0], sample_count=2, rng=object())  # type: ignore[arg-type]
