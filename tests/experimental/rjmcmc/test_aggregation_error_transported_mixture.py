"""Tests for the normalized transported aggregation-error mixture."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    low_rank_gaussian_log_likelihood,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_transported_mixture import (
    TransportedGaussianMixture,
    TransportedMixtureFitterPolicy,
    postcentre_whiten_gaussian_mixture,
    principal_symmetric_psd_sqrt,
    transported_mixture_log_likelihood,
    transported_summary_moments,
)


def _raw_mixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray([0.3, 0.7], dtype=np.float64)
    means = np.asarray(
        [
            [-1.2, 0.45],
            [0.8, -0.1],
        ],
        dtype=np.float64,
    )
    covariances = np.asarray(
        [
            [[0.35, 0.08], [0.08, 0.75]],
            [[1.4, -0.18], [-0.18, 0.5]],
        ],
        dtype=np.float64,
    )
    return weights, means, covariances


def _mixture() -> TransportedGaussianMixture:
    return TransportedGaussianMixture.from_raw(*_raw_mixture())


def _direct_moments(
    mixture: TransportedGaussianMixture,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.einsum("c,cq->q", mixture.weights, mixture.means)
    covariance = np.zeros((mixture.dimension, mixture.dimension), dtype=np.float64)
    for component, weight in enumerate(mixture.weights):
        offset = mixture.means[component] - mean
        covariance += float(weight) * (mixture.covariances[component] + np.outer(offset, offset))
    return mean, covariance


def test_postcentre_whiten_is_moment_exact_and_component_order_canonical() -> None:
    raw = _raw_mixture()
    mixture = TransportedGaussianMixture.from_raw(*raw)
    reversed_mixture = TransportedGaussianMixture.from_raw(
        raw[0][::-1],
        raw[1][::-1],
        raw[2][::-1],
    )
    mean, covariance = _direct_moments(mixture)

    assert np.allclose(mean, np.zeros(2), rtol=0.0, atol=2.0e-15)
    assert np.allclose(covariance, np.eye(2), rtol=0.0, atol=4.0e-15)
    assert mixture.to_json() == reversed_mixture.to_json()
    assert mixture.sha256 == reversed_mixture.sha256
    assert not mixture.weights.flags.writeable
    assert not mixture.means.flags.writeable
    assert not mixture.covariances.flags.writeable


def test_postcentre_whiten_uses_symmetric_total_inverse_sqrt() -> None:
    raw = _raw_mixture()
    weights, means, covariances = postcentre_whiten_gaussian_mixture(*raw)
    normalized = raw[0] / math.fsum(float(value) for value in raw[0])
    raw_mean = np.einsum("c,cq->q", normalized, raw[1])
    raw_covariance = np.zeros((2, 2), dtype=np.float64)
    for component, weight in enumerate(normalized):
        offset = raw[1][component] - raw_mean
        raw_covariance += float(weight) * (raw[2][component] + np.outer(offset, offset))
    eigenvalues, eigenvectors = np.linalg.eigh(raw_covariance)
    inverse_sqrt = (eigenvectors * np.reciprocal(np.sqrt(eigenvalues))[None, :]) @ eigenvectors.T
    expected_means = (inverse_sqrt @ (raw[1] - raw_mean).T).T
    expected_covariances = np.asarray([inverse_sqrt @ value @ inverse_sqrt for value in raw[2]])

    # Canonical catalogue order is weight-descending for these unequal weights.
    assert np.array_equal(weights, normalized[::-1])
    assert np.allclose(means, expected_means[::-1], rtol=0.0, atol=2.0e-15)
    assert np.allclose(
        covariances,
        expected_covariances[::-1],
        rtol=0.0,
        atol=2.0e-15,
    )


def test_artifact_serialization_and_hash_replay_strictly() -> None:
    mixture = _mixture()
    serialized = mixture.to_json()
    restored = TransportedGaussianMixture.from_json(
        serialized,
        expected_sha256=mixture.sha256,
    )

    assert restored.to_json() == serialized
    assert restored.sha256 == mixture.sha256
    with pytest.raises(ValueError, match="SHA-256"):
        TransportedGaussianMixture.from_json(
            serialized,
            expected_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="canonical JSON"):
        TransportedGaussianMixture.from_json(json.dumps(json.loads(serialized)))

    payload = json.loads(serialized)
    payload["schema"] = "future-schema"
    with pytest.raises(ValueError, match="unsupported"):
        TransportedGaussianMixture.from_json(
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )


def test_canonical_weight_closure_is_bit_idempotent_for_reviewer_reproducer() -> None:
    raw_weights = np.asarray(
        [
            float.fromhex("0x1.9806cb5df637bp-2"),
            float.fromhex("0x1.00a5e699cf95cp-3"),
        ]
    )
    raw_means = np.asarray(
        [
            [float.fromhex("-0x1.7b1a8e3930641p-2")],
            [float.fromhex("-0x1.7f6de918b43c3p+0")],
        ]
    )
    raw_covariances = np.ones((2, 1, 1), dtype=np.float64)
    mixture = TransportedGaussianMixture.from_raw(
        raw_weights,
        raw_means,
        raw_covariances,
    )
    serialized = mixture.to_json()
    replay = TransportedGaussianMixture.from_json(
        serialized,
        expected_sha256=mixture.sha256,
    )

    assert math.fsum(float(value) for value in mixture.weights) == 1.0
    assert np.array_equal(replay.weights, mixture.weights)
    assert np.array_equal(replay.means, mixture.means)
    assert np.array_equal(replay.covariances, mixture.covariances)
    assert replay.to_json() == serialized
    assert replay.sha256 == mixture.sha256


def test_rank_zero_standard_normal_has_exact_dedicated_path() -> None:
    mixture = TransportedGaussianMixture.standard_normal(0)
    observation = np.asarray([0.3, -0.2])
    mean = np.asarray([0.1, 0.4])
    scale = np.asarray([0.7, 1.3])
    basis = np.empty((2, 0), dtype=np.float64)
    covariance = np.empty((0, 0), dtype=np.float64)

    expected = low_rank_gaussian_log_likelihood(
        observation,
        mean,
        scale,
        basis,
        covariance,
    )
    observed = transported_mixture_log_likelihood(
        observation,
        mean,
        scale,
        basis,
        covariance,
        mixture,
    )

    assert observed == expected
    summary_mean, summary_covariance = transported_summary_moments(
        covariance,
        mixture,
    )
    assert summary_mean.shape == (0,)
    assert summary_covariance.shape == (0, 0)


def test_standard_normal_control_is_exact_existing_gaussian_likelihood() -> None:
    rng = np.random.default_rng(7251)
    observation = rng.normal(size=5)
    mean = rng.normal(size=5)
    scale = np.exp(rng.normal(size=5))
    basis, _ = np.linalg.qr(rng.normal(size=(5, 3)))
    raw = rng.normal(size=(3, 3))
    covariance = raw @ raw.T + 0.2 * np.eye(3)
    mixture = TransportedGaussianMixture.standard_normal(3)

    expected = low_rank_gaussian_log_likelihood(
        observation,
        mean,
        scale,
        basis,
        covariance,
    )
    observed = transported_mixture_log_likelihood(
        observation,
        mean,
        scale,
        basis,
        covariance,
        mixture,
    )

    assert observed == expected


def test_exact_zero_covariance_ignores_non_gaussian_latent_shape() -> None:
    mixture = _mixture()
    observation = np.asarray([0.5, -1.2, 0.8])
    mean = np.asarray([-0.1, 0.2, 0.3])
    scale = np.asarray([0.4, 0.7, 1.1])
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    covariance = np.zeros((2, 2), dtype=np.float64)

    expected = low_rank_gaussian_log_likelihood(
        observation,
        mean,
        scale,
        basis,
        covariance,
    )
    observed = transported_mixture_log_likelihood(
        observation,
        mean,
        scale,
        basis,
        covariance,
        mixture,
    )

    assert observed == expected


def test_direct_likelihood_matches_dense_observation_space_mixture() -> None:
    rng = np.random.default_rng(8173)
    mixture = _mixture()
    observation = rng.normal(size=4)
    mean = rng.normal(size=4)
    scale = np.asarray([0.4, 0.8, 1.2, 1.7])
    basis, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    covariance = np.asarray([[1.3, 0.35], [0.35, 0.8]])
    covariance_sqrt = principal_symmetric_psd_sqrt(covariance)
    lifted = scale[:, None] * basis
    diagonal = np.diag(np.square(scale))
    component_logps = []
    for component, weight in enumerate(mixture.weights):
        dense_mean = mean + lifted @ covariance_sqrt @ mixture.means[component]
        dense_covariance = (
            diagonal + lifted @ covariance_sqrt @ mixture.covariances[component] @ covariance_sqrt @ lifted.T
        )
        component_logps.append(
            math.log(float(weight))
            + float(
                multivariate_normal.logpdf(
                    observation,
                    mean=dense_mean,
                    cov=dense_covariance,
                )
            )
        )
    expected = float(logsumexp(component_logps))

    observed = transported_mixture_log_likelihood(
        observation,
        mean,
        scale,
        basis,
        covariance,
        mixture,
    )

    assert observed == pytest.approx(expected, rel=0.0, abs=2.0e-13)


def test_scalar_density_is_numerically_normalized() -> None:
    raw_weights = np.asarray([0.4, 0.6])
    raw_means = np.asarray([[-1.4], [0.5]])
    raw_covariances = np.asarray([[[0.25]], [[1.1]]])
    mixture = TransportedGaussianMixture.from_raw(
        raw_weights,
        raw_means,
        raw_covariances,
    )

    def density(value: float) -> float:
        return math.exp(
            transported_mixture_log_likelihood(
                np.asarray([value]),
                np.asarray([0.0]),
                np.asarray([1.7]),
                np.asarray([[1.0]]),
                np.asarray([[2.3]]),
                mixture,
            )
        )

    integral, error = quad(density, -np.inf, np.inf, epsabs=2.0e-11)
    assert error < 1.0e-8
    assert integral == pytest.approx(1.0, rel=0.0, abs=2.0e-10)


def test_convolved_summary_moments_match_components_and_monte_carlo() -> None:
    rng = np.random.default_rng(9917)
    mixture = _mixture()
    covariance = np.asarray([[1.1, -0.28], [-0.28, 0.65]])
    covariance_sqrt = principal_symmetric_psd_sqrt(covariance)
    expected_mean, expected_covariance = transported_summary_moments(
        covariance,
        mixture,
    )

    direct_mean = np.zeros(2, dtype=np.float64)
    component_means = []
    component_covariances = []
    for component, weight in enumerate(mixture.weights):
        component_mean = covariance_sqrt @ mixture.means[component]
        component_covariance = np.eye(2) + covariance_sqrt @ mixture.covariances[component] @ covariance_sqrt
        component_means.append(component_mean)
        component_covariances.append(component_covariance)
        direct_mean += float(weight) * component_mean
    direct_covariance = np.zeros((2, 2), dtype=np.float64)
    for component, weight in enumerate(mixture.weights):
        offset = component_means[component] - direct_mean
        direct_covariance += float(weight) * (component_covariances[component] + np.outer(offset, offset))

    assert np.allclose(expected_mean, direct_mean, rtol=0.0, atol=2.0e-15)
    assert np.allclose(
        expected_covariance,
        direct_covariance,
        rtol=0.0,
        atol=3.0e-15,
    )
    assert np.allclose(expected_mean, np.zeros(2), rtol=0.0, atol=2.0e-15)
    assert np.allclose(
        expected_covariance,
        np.eye(2) + covariance,
        rtol=0.0,
        atol=4.0e-15,
    )

    sample_count = 120_000
    component = rng.choice(
        mixture.component_count,
        size=sample_count,
        p=mixture.weights,
    )
    samples = np.empty((sample_count, 2), dtype=np.float64)
    for index in range(mixture.component_count):
        selected = component == index
        samples[selected] = rng.multivariate_normal(
            component_means[index],
            component_covariances[index],
            size=int(np.sum(selected)),
        )
    assert np.allclose(samples.mean(axis=0), expected_mean, rtol=0.0, atol=0.015)
    assert np.allclose(
        np.cov(samples, rowvar=False),
        expected_covariance,
        rtol=0.0,
        atol=0.025,
    )


@pytest.mark.parametrize(
    "rotation",
    [
        np.asarray([[0.0, -1.0], [1.0, 0.0]]),
        np.asarray(
            [
                [math.cos(0.37), -math.sin(0.37)],
                [math.sin(0.37), math.cos(0.37)],
            ]
        ),
    ],
)
def test_summary_coordinate_orthogonal_equivariance(
    rotation: np.ndarray,
) -> None:
    rng = np.random.default_rng(10321)
    mixture = _mixture()
    observation = rng.normal(size=5)
    mean = rng.normal(size=5)
    scale = np.exp(rng.normal(size=5))
    basis, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    covariance = np.asarray([[0.9, 0.22], [0.22, 1.4]])
    rotated_basis = basis @ rotation.T
    rotated_covariance = rotation @ covariance @ rotation.T
    rotated_mixture = TransportedGaussianMixture(
        mixture.weights,
        mixture.means @ rotation.T,
        np.asarray([rotation @ component @ rotation.T for component in mixture.covariances]),
    )

    expected = transported_mixture_log_likelihood(
        observation,
        mean,
        scale,
        basis,
        covariance,
        mixture,
    )
    observed = transported_mixture_log_likelihood(
        observation,
        mean,
        scale,
        rotated_basis,
        rotated_covariance,
        rotated_mixture,
    )

    assert observed == pytest.approx(expected, rel=0.0, abs=5.0e-13)
    square_root = principal_symmetric_psd_sqrt(covariance)
    rotated_square_root = principal_symmetric_psd_sqrt(rotated_covariance)
    assert np.allclose(
        rotated_square_root,
        rotation @ square_root @ rotation.T,
        rtol=0.0,
        atol=3.0e-15,
    )


def test_nonzero_singular_summary_covariance_is_a_hard_stop() -> None:
    mixture = TransportedGaussianMixture.standard_normal(2)
    singular = np.diag([1.0, 0.0])
    with pytest.raises(ValueError, match="nonzero singular"):
        principal_symmetric_psd_sqrt(singular)
    with pytest.raises(ValueError, match="nonzero singular"):
        transported_mixture_log_likelihood(
            np.asarray([0.2, 0.4]),
            np.zeros(2),
            np.ones(2),
            np.eye(2),
            singular,
            mixture,
        )


@pytest.mark.parametrize(
    ("weights", "means", "covariances", "message"),
    [
        (
            np.asarray([0.0, 1.0]),
            np.zeros((2, 1)),
            np.ones((2, 1, 1)),
            "strictly positive",
        ),
        (
            np.asarray([0.5, 0.5]),
            np.zeros((2, 1)),
            np.asarray([[[1.0]], [[0.0]]]),
            "collapsed",
        ),
        (
            np.asarray([0.5, 0.5]),
            np.zeros((2, 2)),
            np.asarray(
                [
                    [[1.0, 0.0], [0.0, 5.0e-13]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ]
            ),
            "condition number",
        ),
    ],
)
def test_malformed_or_collapsed_mixtures_fail_closed(
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TransportedGaussianMixture.from_raw(weights, means, covariances)


def test_constructor_rejects_nonstandardized_and_nonliteral_single_component() -> None:
    with pytest.raises(ValueError, match="literal zero mean"):
        TransportedGaussianMixture(
            np.asarray([1.0]),
            np.asarray([[0.1]]),
            np.asarray([[[1.0]]]),
        )
    with pytest.raises(ValueError, match="identity covariance"):
        TransportedGaussianMixture(
            np.asarray([1.0]),
            np.asarray([[0.0]]),
            np.asarray([[[1.1]]]),
        )
    with pytest.raises(ValueError, match="latent mixture mean must be zero"):
        TransportedGaussianMixture(
            np.asarray([0.5, 0.5]),
            np.asarray([[-0.2], [0.3]]),
            np.asarray([[[0.8]], [[1.1]]]),
        )


def test_invalid_scientific_inputs_fail_closed() -> None:
    mixture = _mixture()
    with pytest.raises(ValueError, match="orthonormal"):
        transported_mixture_log_likelihood(
            np.ones(3),
            np.zeros(3),
            np.ones(3),
            np.ones((3, 2)),
            np.eye(2),
            mixture,
        )
    with pytest.raises(ValueError, match="dimension"):
        transported_mixture_log_likelihood(
            np.ones(3),
            np.zeros(3),
            np.ones(3),
            np.eye(3),
            np.eye(3),
            mixture,
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        principal_symmetric_psd_sqrt(np.diag([1.0, -0.1]))


def test_fitter_policy_is_bounded_deterministic_and_hashed() -> None:
    policy = TransportedMixtureFitterPolicy()
    replay = TransportedMixtureFitterPolicy(
        component_counts=policy.component_counts,
        restart_seeds=policy.restart_seeds,
        maximum_iterations=policy.maximum_iterations,
        convergence_tolerance=policy.convergence_tolerance,
        covariance_regularization=policy.covariance_regularization,
        minimum_component_weight=policy.minimum_component_weight,
        maximum_component_condition=policy.maximum_component_condition,
    )

    assert policy.sha256 == replay.sha256
    assert policy.component_counts == (1, 2, 4, 8)
    assert len(policy.restart_seeds) == 5
    with pytest.raises(ValueError, match="unique, increasing"):
        TransportedMixtureFitterPolicy(component_counts=(1, 4, 2))
    with pytest.raises(ValueError, match="artifact hard limit"):
        TransportedMixtureFitterPolicy(maximum_component_condition=1.0e13)


def test_fitter_policy_normalizes_real_fields_and_rejects_booleans() -> None:
    integer_values = TransportedMixtureFitterPolicy(
        convergence_tolerance=1,
        covariance_regularization=2,
        minimum_component_weight=0.5,
        maximum_component_condition=100,
    )
    float_values = TransportedMixtureFitterPolicy(
        convergence_tolerance=1.0,
        covariance_regularization=2.0,
        minimum_component_weight=0.5,
        maximum_component_condition=100.0,
    )

    assert integer_values.payload == float_values.payload
    assert integer_values.sha256 == float_values.sha256
    assert isinstance(integer_values.convergence_tolerance, float)
    assert isinstance(integer_values.covariance_regularization, float)
    assert isinstance(integer_values.minimum_component_weight, float)
    assert isinstance(integer_values.maximum_component_condition, float)

    with pytest.raises(TypeError, match="non-Boolean real"):
        TransportedMixtureFitterPolicy(convergence_tolerance=True)
    with pytest.raises(TypeError, match="non-Boolean real"):
        TransportedMixtureFitterPolicy(covariance_regularization=True)
    with pytest.raises(TypeError, match="non-Boolean real"):
        TransportedMixtureFitterPolicy(minimum_component_weight=True)
    with pytest.raises(TypeError, match="non-Boolean real"):
        TransportedMixtureFitterPolicy(maximum_component_condition=True)
