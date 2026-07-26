"""Tests for the portable residual-image conditional MDN likelihood."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

from openghg_inversions.experimental.rjmcmc import (
    aggregation_error_conditional_mdn as conditional_mdn_module,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mdn import (
    ConditionalResidualImageMDN,
    ResidualImageContext,
    conditional_residual_image_mdn_log_likelihood,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)


_LOG_TWO_PI = math.log(2.0 * math.pi)


def _aggregation(
    *,
    design: np.ndarray | None = None,
    noise_sd: np.ndarray | None = None,
) -> AdditiveDirichletAggregation:
    """Return a heterogeneous four-cell native model."""
    matrix = (
        np.asarray(
            [
                [1.8, -0.5, 0.3, 0.9],
                [0.2, 1.4, -0.7, 0.1],
                [0.5, -0.2, 1.1, 0.8],
            ],
            dtype=np.float64,
        )
        if design is None
        else np.asarray(design, dtype=np.float64)
    )
    scale = (
        np.asarray([0.35, 0.8, 0.6], dtype=np.float64)
        if noise_sd is None
        else np.asarray(noise_sd, dtype=np.float64)
    )
    return AdditiveDirichletAggregation(
        np.asarray([0.7, 1.1, 1.6, 0.9], dtype=np.float64),
        matrix,
        scale,
        np.empty((matrix.shape[0], 0), dtype=np.float64),
    )


def _context(
    labels: np.ndarray | None = None,
    *,
    aggregation: AdditiveDirichletAggregation | None = None,
    cell_ids: np.ndarray | None = None,
) -> ResidualImageContext:
    """Build one authenticated residual-image context."""
    return ResidualImageContext.from_aggregation(
        _aggregation() if aggregation is None else aggregation,
        np.asarray([0, 0, 1, 1], dtype=np.int64) if labels is None else labels,
        np.asarray([101, 102, 201, 202], dtype=np.int64) if cell_ids is None else cell_ids,
        source_provenance="unit-test residual-image context",
    )


def _inverse_softplus(value: float) -> float:
    """Return the stable inverse of ``log(1 + exp(x))``."""
    return value + math.log(-math.expm1(-value))


def _packed_raw_cholesky(
    cholesky: np.ndarray,
    *,
    diagonal_floor: float,
) -> np.ndarray:
    """Pack a lower triangle with inverse-softplus diagonal entries."""
    matrix = np.asarray(cholesky, dtype=np.float64)
    packed: list[float] = []
    for row in range(matrix.shape[0]):
        for column in range(row + 1):
            value = float(matrix[row, column])
            if row == column:
                value = _inverse_softplus(value - diagonal_floor)
            packed.append(value)
    return np.asarray(packed, dtype=np.float64)


def _constant_mdn(
    context: ResidualImageContext,
    *,
    logits: np.ndarray,
    means: np.ndarray,
    cholesky_factors: np.ndarray,
    diagonal_floor: float = 1.0e-4,
) -> ConditionalResidualImageMDN:
    """Return an MDN whose zero-weight network emits fixed components."""
    component_logits = np.asarray(logits, dtype=np.float64)
    component_means = np.asarray(means, dtype=np.float64)
    factors = np.asarray(cholesky_factors, dtype=np.float64)
    packed = np.concatenate(
        [_packed_raw_cholesky(factor, diagonal_floor=diagonal_floor) for factor in factors]
    )
    output_bias = np.concatenate(
        [component_logits, component_means.reshape(-1), packed],
    )
    return ConditionalResidualImageMDN(
        context,
        np.zeros((2, context.conditioner_dimension), dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        np.zeros((3, 2), dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros((output_bias.size, 3), dtype=np.float64),
        output_bias,
        component_count=component_logits.size,
        cholesky_diagonal_floor=diagonal_floor,
        source_provenance="unit-test constant residual GMM",
    )


def _logsumexp(values: np.ndarray) -> float:
    """Evaluate log-sum-exp independently for a short vector."""
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _normal_logpdf(
    value: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Evaluate a dense multivariate normal log density independently."""
    delta = np.asarray(value, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    sign, log_determinant = np.linalg.slogdet(covariance)
    assert sign == 1.0
    return -0.5 * (
        delta.size * _LOG_TWO_PI + float(log_determinant) + float(delta @ np.linalg.solve(covariance, delta))
    )


def _manual_mixture_log_likelihood(
    context: ResidualImageContext,
    observation: np.ndarray,
    masses: np.ndarray,
    *,
    offset: np.ndarray,
    logits: np.ndarray,
    means: np.ndarray,
    cholesky_factors: np.ndarray,
) -> float:
    """Evaluate the convolved residual-image mixture independently."""
    total = float(np.sum(masses))
    residual = (observation - offset - context.observation_mean_design @ masses) / context.noise_sd
    coordinates = context.residual_basis.T @ residual
    orthogonal = residual - context.residual_basis @ coordinates
    log_weights = logits - _logsumexp(logits)
    terms = np.empty(logits.size, dtype=np.float64)
    identity = np.eye(context.residual_rank, dtype=np.float64)
    for component in range(logits.size):
        covariance = identity + total * total * (cholesky_factors[component] @ cholesky_factors[component].T)
        terms[component] = log_weights[component] + _normal_logpdf(
            coordinates,
            total * means[component],
            covariance,
        )
    orthogonal_logp = -0.5 * (
        (context.observation_count - context.residual_rank) * _LOG_TWO_PI + float(orthogonal @ orthogonal)
    )
    return -float(np.sum(np.log(context.noise_sd))) + orthogonal_logp + _logsumexp(terms)


def test_context_recovers_exact_residual_image_projector_and_means() -> None:
    """The stored basis must span the exact whitened within-region image."""
    aggregation = _aggregation()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    context = _context(labels, aggregation=aggregation)

    expected_means = np.empty((3, 2), dtype=np.float64)
    whitened_centered: list[np.ndarray] = []
    flat_alphas = aggregation.cell_alphas.reshape(-1)
    for region in range(2):
        selected = labels == region
        proportions = flat_alphas[selected] / float(np.sum(flat_alphas[selected]))
        expected_mean = aggregation.design[:, selected] @ proportions
        expected_means[:, region] = expected_mean
        whitened_centered.append(
            (aggregation.design[:, selected] - expected_mean[:, np.newaxis])
            / aggregation.noise_sd[:, np.newaxis]
        )
    image = np.concatenate(whitened_centered, axis=1)
    expected_basis, singular_values, _ = np.linalg.svd(image, full_matrices=False)
    tolerance = max(image.shape) * np.finfo(np.float64).eps * singular_values[0]
    expected_rank = int(np.count_nonzero(singular_values > tolerance))
    expected_projector = expected_basis[:, :expected_rank] @ expected_basis[:, :expected_rank].T

    np.testing.assert_allclose(
        context.observation_mean_design,
        expected_means,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert context.residual_rank == expected_rank
    np.testing.assert_allclose(
        context.residual_basis.T @ context.residual_basis,
        np.eye(expected_rank),
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        context.residual_basis @ context.residual_basis.T,
        expected_projector,
        rtol=0.0,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(context.alpha_totals, [1.8, 2.5])
    assert context.conditioner_dimension == 1
    for value in (
        context.observation_mean_design,
        context.noise_sd,
        context.residual_basis,
        context.labels,
        context.cell_ids,
        context.alpha_totals,
    ):
        assert not value.flags.writeable


def test_root_two_region_and_rank_zero_contexts_have_declared_dimensions() -> None:
    """Root, two-region, and zero-image cases must preserve their dimensions."""
    root = _context(np.zeros(4, dtype=np.int64))
    assert root.region_count == 1
    assert root.conditioner_dimension == 0
    assert root.residual_rank == 3

    two_region = _context()
    assert two_region.region_count == 2
    assert two_region.conditioner_dimension == 1

    repeated_design = np.asarray(
        [
            [1.0, 1.0, -0.2, -0.2],
            [0.5, 0.5, 1.3, 1.3],
            [-0.7, -0.7, 0.1, 0.1],
        ]
    )
    rank_zero = _context(aggregation=_aggregation(design=repeated_design))
    assert rank_zero.residual_rank == 0
    assert rank_zero.residual_basis.shape == (3, 0)


def test_singleton_region_contributes_no_spurious_residual_direction() -> None:
    """A one-cell Dirichlet block must add no aggregation-error dimension."""
    aggregation = _aggregation()
    labels = np.asarray([0, 1, 1, 1], dtype=np.int64)
    context = _context(labels, aggregation=aggregation)
    selected = labels == 1
    alphas = aggregation.cell_alphas[selected]
    mean = aggregation.design[:, selected] @ (alphas / float(alphas.sum()))
    image = (aggregation.design[:, selected] - mean[:, np.newaxis]) / aggregation.noise_sd[:, np.newaxis]

    assert context.residual_rank == np.linalg.matrix_rank(image)
    np.testing.assert_allclose(
        context.residual_basis @ (context.residual_basis.T @ image),
        image,
        rtol=0.0,
        atol=3.0e-15,
    )


@pytest.mark.parametrize("tolerance_multiplier", [0.5, 2.0])
def test_near_threshold_residual_rank_fails_closed(
    tolerance_multiplier: float,
) -> None:
    """Singular values near either side of the rank cutoff are ambiguous."""
    epsilon = np.finfo(np.float64).eps
    ambiguous_value = 256.0 * epsilon * 2.0 * tolerance_multiplier
    image = np.diag(np.asarray([1.0, ambiguous_value]))

    with pytest.raises(ValueError, match="rank is numerically ambiguous"):
        conditional_mdn_module._canonical_residual_basis(image)


def test_cell_permutation_and_region_relabel_preserve_scientific_density() -> None:
    """Canonical regions must remove cell order and label-name effects."""
    aggregation = _aggregation()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    ids = np.asarray([101, 102, 201, 202], dtype=np.int64)
    original = _context(labels, aggregation=aggregation, cell_ids=ids)

    order = np.asarray([2, 0, 3, 1])
    permuted_aggregation = AdditiveDirichletAggregation(
        aggregation.cell_alphas[order],
        aggregation.design[:, order],
        aggregation.noise_sd,
        np.empty((3, 0), dtype=np.float64),
    )
    permuted = _context(
        labels[order],
        aggregation=permuted_aggregation,
        cell_ids=ids[order],
    )
    relabelled = _context(1 - labels, aggregation=aggregation, cell_ids=ids)

    original_projector = original.residual_basis @ original.residual_basis.T
    np.testing.assert_allclose(
        permuted.residual_basis @ permuted.residual_basis.T,
        original_projector,
        rtol=0.0,
        atol=4.0e-15,
    )
    np.testing.assert_allclose(
        relabelled.residual_basis @ relabelled.residual_basis.T,
        original_projector,
        rtol=0.0,
        atol=4.0e-15,
    )
    np.testing.assert_allclose(
        permuted.observation_mean_design,
        original.observation_mean_design,
        rtol=0.0,
        atol=3.0e-15,
    )
    np.testing.assert_array_equal(
        relabelled.observation_mean_design,
        original.observation_mean_design,
    )
    np.testing.assert_array_equal(relabelled.alpha_totals, original.alpha_totals)
    np.testing.assert_array_equal(relabelled.labels, original.labels)
    assert relabelled.artifact_sha256 == original.artifact_sha256
    np.testing.assert_array_equal(
        relabelled.canonicalize_masses(
            np.asarray([2.4, 1.8]),
            1 - labels,
        ),
        np.asarray([1.8, 2.4]),
    )

    def conditional(
        context: ResidualImageContext,
    ) -> ConditionalResidualImageMDN:
        component_count = 2
        rank = context.residual_rank
        triangle_size = rank * (rank + 1) // 2
        output_size = component_count * (1 + rank + triangle_size)
        output_weight = np.zeros((output_size, 2), dtype=np.float64)
        output_weight[0] = [0.4, -0.2]
        output_weight[1] = [-0.1, 0.3]
        output_weight[2] = [0.5, 0.2]
        output_weight[3] = [-0.25, 0.4]
        return ConditionalResidualImageMDN(
            context,
            np.asarray([[0.6], [-0.35]]),
            np.asarray([0.1, -0.2]),
            np.asarray([[0.5, -0.4], [0.2, 0.7]]),
            np.asarray([-0.1, 0.25]),
            output_weight,
            np.zeros(output_size, dtype=np.float64),
            component_count=component_count,
            cholesky_diagonal_floor=1.0e-4,
            source_provenance="conditional relabel-invariance test",
        )

    observation = np.asarray([0.4, -0.7, 1.2])
    masses = np.asarray([1.8, 2.4])
    original_logp = conditional(original).log_likelihood(observation, masses)
    assert conditional(permuted).log_likelihood(
        observation,
        masses,
    ) == pytest.approx(original_logp, abs=2.0e-13)
    assert conditional(relabelled).log_likelihood(
        observation,
        masses,
    ) == pytest.approx(original_logp, abs=2.0e-13)


def test_one_component_full_cholesky_matches_dense_observation_gaussian() -> None:
    """Noise convolution must equal the corresponding dense Gaussian."""
    context = _context()
    q = context.residual_rank
    assert q == 2
    factor = np.asarray([[0.7, 0.0], [-0.25, 0.45]])
    mean = np.asarray([[0.3, -0.4]])
    mdn = _constant_mdn(
        context,
        logits=np.asarray([0.0]),
        means=mean,
        cholesky_factors=factor[np.newaxis, :, :],
    )
    masses = np.asarray([1.3, 2.1])
    observation = np.asarray([0.7, -0.2, 1.4])
    offset = np.asarray([0.12, -0.15, 0.08])
    total = float(np.sum(masses))
    scaled_basis = context.noise_sd[:, np.newaxis] * context.residual_basis
    expected_mean = offset + context.observation_mean_design @ masses + scaled_basis @ (total * mean[0])
    expected_covariance = (
        np.diag(context.noise_sd**2) + (total * scaled_basis @ factor) @ (total * scaled_basis @ factor).T
    )
    expected = _normal_logpdf(observation, expected_mean, expected_covariance)

    assert mdn.log_likelihood(
        observation,
        masses,
        offset=offset,
    ) == pytest.approx(expected, abs=3.0e-13)
    assert conditional_residual_image_mdn_log_likelihood(
        observation,
        masses,
        mdn,
        offset=offset,
    ) == pytest.approx(expected, abs=3.0e-13)


def test_constant_mixture_retains_weights_full_covariance_and_total_scaling() -> None:
    """Mixture weights and the ``T`` and ``T**2`` scaling must be exact."""
    context = _context()
    logits = np.asarray([-0.7, 0.4])
    means = np.asarray([[0.3, -0.2], [-0.6, 0.5]])
    factors = np.asarray(
        [
            [[0.4, 0.0], [0.15, 0.6]],
            [[0.8, 0.0], [-0.2, 0.35]],
        ]
    )
    mdn = _constant_mdn(
        context,
        logits=logits,
        means=means,
        cholesky_factors=factors,
    )
    observation = np.asarray([0.7, -0.2, 1.4])
    offset = np.asarray([0.12, -0.15, 0.08])
    for masses in (
        np.asarray([0.15, 0.35]),
        np.asarray([1.5, 3.5]),
        np.asarray([15.0, 35.0]),
    ):
        expected = _manual_mixture_log_likelihood(
            context,
            observation,
            masses,
            offset=offset,
            logits=logits,
            means=means,
            cholesky_factors=factors,
        )
        assert mdn.log_likelihood(
            observation,
            masses,
            offset=offset,
        ) == pytest.approx(expected, abs=5.0e-13)


def test_two_region_network_uses_unclipped_log_mass_ratio_conditioner() -> None:
    """A nonconstant head must use the authoritative ALR conditioner."""
    context = _context()
    q = context.residual_rank
    component_count = 1
    output_size = component_count * (1 + q + q * (q + 1) // 2)
    weight_1 = np.asarray([[0.6], [-0.35]])
    bias_1 = np.asarray([0.1, -0.2])
    weight_2 = np.asarray([[0.5, -0.4], [0.2, 0.7]])
    bias_2 = np.asarray([-0.1, 0.25])
    weight_out = np.zeros((output_size, 2), dtype=np.float64)
    weight_out[1] = np.asarray([0.4, -0.1])
    weight_out[2] = np.asarray([-0.3, 0.25])
    bias_out = np.asarray(
        [
            0.0,
            0.05,
            -0.08,
            _inverse_softplus(0.5 - 1.0e-4),
            0.17,
            _inverse_softplus(0.4 - 1.0e-4),
        ]
    )
    input_center = np.asarray([0.2])
    input_scale = np.asarray([1.7])
    mdn = ConditionalResidualImageMDN(
        context,
        weight_1,
        bias_1,
        weight_2,
        bias_2,
        weight_out,
        bias_out,
        component_count=1,
        cholesky_diagonal_floor=1.0e-4,
        input_center=input_center,
        input_scale=input_scale,
        source_provenance="unit-test conditional head",
    )
    observation = np.asarray([0.7, -0.2, 1.4])
    offset = np.asarray([0.12, -0.15, 0.08])
    masses = np.asarray([1.2e-5, 2.8])
    conditioner = (np.asarray([math.log(masses[0] / masses[1])]) - input_center) / input_scale
    hidden_1 = np.tanh(weight_1 @ conditioner + bias_1)
    hidden_2 = np.tanh(weight_2 @ hidden_1 + bias_2)
    raw_output = weight_out @ hidden_2 + bias_out
    means = raw_output[1:3].reshape(1, 2)
    factor = np.asarray(
        [
            [
                math.log1p(math.exp(raw_output[3])) + 1.0e-4,
                0.0,
            ],
            [
                raw_output[4],
                math.log1p(math.exp(raw_output[5])) + 1.0e-4,
            ],
        ]
    )
    expected = _manual_mixture_log_likelihood(
        context,
        observation,
        masses,
        offset=offset,
        logits=np.asarray([raw_output[0]]),
        means=means,
        cholesky_factors=factor[np.newaxis, :, :],
    )

    assert mdn.log_likelihood(
        observation,
        masses,
        offset=offset,
    ) == pytest.approx(expected, abs=4.0e-13)


def test_rank_zero_is_the_exact_diagonal_gaussian() -> None:
    """An empty residual image must reduce to the base diagonal likelihood."""
    repeated_design = np.asarray(
        [
            [1.0, 1.0, -0.2, -0.2],
            [0.5, 0.5, 1.3, 1.3],
            [-0.7, -0.7, 0.1, 0.1],
        ]
    )
    context = _context(aggregation=_aggregation(design=repeated_design))
    mdn = _constant_mdn(
        context,
        logits=np.asarray([-0.5, 0.8]),
        means=np.empty((2, 0)),
        cholesky_factors=np.empty((2, 0, 0)),
    )
    observation = np.asarray([0.7, -0.2, 1.4])
    offset = np.asarray([0.12, -0.15, 0.08])
    masses = np.asarray([1.3, 2.1])
    residual = (observation - offset - context.observation_mean_design @ masses) / context.noise_sd
    expected = -float(np.sum(np.log(context.noise_sd))) - 0.5 * (
        observation.size * _LOG_TWO_PI + float(residual @ residual)
    )
    assert mdn.log_likelihood(
        observation,
        masses,
        offset=offset,
    ) == pytest.approx(expected, abs=3.0e-14)


def test_unconditional_root_gmm_is_normalized() -> None:
    """A zero-input root mixture must integrate to one in observation space."""
    aggregation = AdditiveDirichletAggregation(
        np.asarray([0.7, 1.1]),
        np.asarray([[1.8, -0.5]]),
        np.asarray([0.35]),
        np.empty((1, 0)),
    )
    context = ResidualImageContext.from_aggregation(
        aggregation,
        np.zeros(2, dtype=np.int64),
        np.asarray([11, 12], dtype=np.int64),
        source_provenance="one-dimensional normalization context",
    )
    mdn = _constant_mdn(
        context,
        logits=np.asarray([-0.8, 0.3]),
        means=np.asarray([[-0.4], [0.7]]),
        cholesky_factors=np.asarray([[[0.3]], [[0.8]]]),
    )
    assert context.conditioner_dimension == 0
    masses = np.asarray([2.4])
    integral, error = quad(
        lambda value: math.exp(mdn.log_likelihood(np.asarray([value]), masses)),
        -np.inf,
        np.inf,
        epsabs=2.0e-10,
        epsrel=2.0e-10,
    )
    assert error < 2.0e-9
    assert integral == pytest.approx(1.0, abs=2.0e-9)


def test_context_and_mdn_json_roundtrip_are_exact_and_fail_closed() -> None:
    """Artifacts must replay exactly and reject tampering or malformed shapes."""
    context = _context()
    context_replay = ResidualImageContext.from_json(
        context.to_json(),
        expected_sha256=context.artifact_sha256,
    )
    np.testing.assert_array_equal(
        context_replay.observation_mean_design,
        context.observation_mean_design,
    )
    np.testing.assert_array_equal(
        context_replay.residual_basis,
        context.residual_basis,
    )
    assert context_replay.artifact_sha256 == context.artifact_sha256
    assert context_replay.sha256 == context.artifact_sha256

    with pytest.raises(TypeError):
        ResidualImageContext.from_json(context.to_json())  # type: ignore[call-arg]
    noncanonical_context = json.dumps(
        json.loads(context.to_json()),
        sort_keys=True,
    )
    with pytest.raises(ValueError, match="canonical"):
        ResidualImageContext.from_json(
            noncanonical_context,
            expected_sha256=hashlib.sha256(
                noncanonical_context.encode(),
            ).hexdigest(),
        )

    mdn = _constant_mdn(
        context,
        logits=np.asarray([-0.7, 0.4]),
        means=np.asarray([[0.3, -0.2], [-0.6, 0.5]]),
        cholesky_factors=np.asarray(
            [
                [[0.4, 0.0], [0.15, 0.6]],
                [[0.8, 0.0], [-0.2, 0.35]],
            ]
        ),
    )
    serialized = mdn.to_json()
    assert mdn.sha256 == mdn.artifact_sha256
    replay = ConditionalResidualImageMDN.from_json(
        serialized,
        expected_sha256=mdn.artifact_sha256,
    )
    observation = np.asarray([0.7, -0.2, 1.4])
    masses = np.asarray([1.3, 2.1])
    assert replay.artifact_sha256 == mdn.artifact_sha256
    assert replay.log_likelihood(observation, masses) == mdn.log_likelihood(
        observation,
        masses,
    )
    for value in (
        replay.hidden_weight_1,
        replay.hidden_bias_1,
        replay.hidden_weight_2,
        replay.hidden_bias_2,
        replay.output_weight,
        replay.output_bias,
        replay.input_center,
        replay.input_scale,
    ):
        assert not value.flags.writeable

    tampered_payload = json.loads(serialized)
    tampered_payload["output_bias"][0] += 0.125
    with pytest.raises(ValueError, match="SHA-256|fingerprint|digest"):
        ConditionalResidualImageMDN.from_json(
            json.dumps(tampered_payload),
            expected_sha256=mdn.artifact_sha256,
        )

    with pytest.raises(TypeError):
        ConditionalResidualImageMDN.from_json(serialized)  # type: ignore[call-arg]

    noncanonical = json.dumps(json.loads(serialized), sort_keys=True)
    with pytest.raises(ValueError, match="canonical"):
        ConditionalResidualImageMDN.from_json(
            noncanonical,
            expected_sha256=hashlib.sha256(noncanonical.encode()).hexdigest(),
        )

    nested_tamper = json.loads(serialized)
    nested_tamper["context"]["alpha_totals"][0] += 0.25
    nested_serialized = json.dumps(
        nested_tamper,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(ValueError, match="SHA-256|fingerprint|digest"):
        ConditionalResidualImageMDN.from_json(
            nested_serialized,
            expected_sha256=hashlib.sha256(
                nested_serialized.encode(),
            ).hexdigest(),
        )

    malformed_payload = json.loads(serialized)
    malformed_payload["hidden_weight_1"] = [[0.0, 0.0]]
    malformed_serialized = json.dumps(
        malformed_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(ValueError):
        ConditionalResidualImageMDN.from_json(
            malformed_serialized,
            expected_sha256=hashlib.sha256(
                malformed_serialized.encode(),
            ).hexdigest(),
        )
