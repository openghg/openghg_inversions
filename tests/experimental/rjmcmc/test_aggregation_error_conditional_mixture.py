"""Tests for the frozen conditional-allocation aggregation mixture."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

from openghg_inversions.experimental.rjmcmc.aggregation_error import (
    TwoCellAggregationOracle,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (
    ConditionalAllocationMixture,
    conditional_allocation_mixture_log_likelihood,
    conditional_allocation_mixture_log_likelihood_and_gradient,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)


def _model(*, rank: int = 2) -> AdditiveDirichletAggregation:
    """Return one small heterogeneous fixed native model."""
    alphas = np.asarray([0.7, 1.1, 1.6, 0.9])
    design = np.asarray(
        [
            [1.8, -0.5, 0.3, 0.9],
            [0.2, 1.4, -0.7, 0.1],
            [0.5, -0.2, 1.1, 0.8],
        ]
    )
    noise_sd = np.asarray([0.35, 0.8, 0.6])
    if rank == 0:
        basis = np.empty((3, 0))
    else:
        raw_basis = np.asarray([[1.0, 0.3], [0.4, 1.0], [-0.2, 0.5]])
        basis, _ = np.linalg.qr(raw_basis)
        basis = basis[:, :rank]
    return AdditiveDirichletAggregation(alphas, design, noise_sd, basis)


def _artifact(
    *,
    sample_count: int = 2_000,
    rank: int = 2,
) -> ConditionalAllocationMixture:
    """Return a replayable two-region conditional mixture."""
    return ConditionalAllocationMixture.from_aggregation(
        _model(rank=rank),
        np.asarray([0, 0, 1, 1]),
        sample_count=sample_count,
        source_seed=731,
        source_provenance="unit-test fixed allocation bank",
        cell_ids=np.asarray([101, 102, 201, 202]),
    )


def test_frozen_bank_replays_dirichlet_moments_and_cached_factors() -> None:
    """Frozen projected draws should reproduce exact block-Dirichlet moments."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    cached = model.partition_factors(labels)
    artifact = ConditionalAllocationMixture.from_aggregation(
        model,
        cached,
        sample_count=80_000,
        source_seed=9281,
        source_provenance="moment regression",
    )
    masses = np.asarray([2.3, 1.7])
    residual_draws = np.einsum(
        "sqr,r->sq",
        artifact.projected_unit_mass_residual_factors,
        masses,
        optimize=False,
    )
    empirical_mean = residual_draws.mean(axis=0)
    empirical_covariance = np.cov(residual_draws, rowvar=False, bias=True)
    exact_covariance = cached.summary_residual_covariance(masses)

    np.testing.assert_allclose(empirical_mean, np.zeros(2), atol=4.0e-3)
    np.testing.assert_allclose(
        empirical_covariance,
        exact_covariance,
        rtol=2.5e-2,
        atol=2.0e-3,
    )
    np.testing.assert_array_equal(
        artifact.observation_mean_design,
        cached.observation_mean_design,
    )
    assert artifact.sample_count == 80_000
    assert artifact.summary_rank == 2
    assert artifact.region_count == 2
    assert artifact.storage_nbytes == sum(
        array.nbytes
        for array in (
            artifact.projected_unit_mass_residual_factors,
            artifact.observation_mean_design,
            artifact.noise_sd,
            artifact.summary_basis,
            artifact.labels,
            artifact.cell_ids,
            artifact.alpha_totals,
        )
    )


def test_one_dimensional_density_is_normalized() -> None:
    """The equal conditional Gaussian mixture must integrate to one."""
    model = AdditiveDirichletAggregation(
        np.asarray([0.8, 1.3]),
        np.asarray([[1.7, -0.4]]),
        0.65,
        np.ones((1, 1)),
    )
    artifact = ConditionalAllocationMixture.from_aggregation(
        model,
        np.zeros(2, dtype=np.int64),
        sample_count=257,
        source_seed=902,
        source_provenance="one-dimensional normalization",
    )

    integral, error = quad(
        lambda value: math.exp(artifact.log_likelihood([value], [2.1])),
        -np.inf,
        np.inf,
        epsabs=2.0e-10,
        epsrel=2.0e-10,
        limit=160,
    )

    assert integral == pytest.approx(1.0, abs=3.0e-10)
    assert error < 3.0e-9


def test_large_bank_converges_to_exact_two_cell_conditional_oracle() -> None:
    """A large frozen bank should approach the exact Beta-mixture likelihood."""
    alphas = np.asarray([0.7, 1.3])
    design = np.asarray([[1.5, -0.4]])
    noise_sd = np.asarray([0.8])
    model = AdditiveDirichletAggregation(
        alphas,
        design,
        noise_sd,
        np.ones((1, 1)),
    )
    artifact = ConditionalAllocationMixture.from_aggregation(
        model,
        np.zeros(2, dtype=np.int64),
        sample_count=100_000,
        source_seed=7741,
        source_provenance="two-cell exact-oracle convergence",
    )
    total = 2.2
    observation = np.asarray([0.7])
    oracle = TwoCellAggregationOracle(
        gamma_shape=2.0,
        gamma_rate=1.0,
        beta_first_shape=alphas[0],
        beta_second_shape=alphas[1],
        fraction_order=128,
    )

    expected = oracle.coarse_conditional_log_likelihood(
        total,
        observation,
        design,
        noise_sd,
    )
    observed = artifact.log_likelihood(observation, [total])

    assert observed == pytest.approx(expected, abs=4.0e-3)


def test_analytic_mass_gradient_matches_centered_finite_difference() -> None:
    """The gradient should include both retained mean and component transport."""
    artifact = _artifact(sample_count=1_500)
    observation = np.asarray([0.4, -0.7, 1.2])
    masses = np.asarray([1.8, 2.4])
    offset = np.asarray([0.1, -0.2, 0.05])
    logp, gradient = conditional_allocation_mixture_log_likelihood_and_gradient(
        observation,
        masses,
        artifact,
        mean_offset=offset,
    )
    step = 2.0e-6
    finite_difference = np.empty(2)
    for index in range(2):
        high = masses.copy()
        low = masses.copy()
        high[index] += step
        low[index] -= step
        finite_difference[index] = (
            conditional_allocation_mixture_log_likelihood(
                observation,
                high,
                artifact,
                mean_offset=offset,
            )
            - conditional_allocation_mixture_log_likelihood(
                observation,
                low,
                artifact,
                mean_offset=offset,
            )
        ) / (2.0 * step)

    assert logp == artifact.log_likelihood(
        observation,
        masses,
        mean_offset=offset,
    )
    np.testing.assert_allclose(
        gradient,
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-9,
    )
    assert not gradient.flags.writeable


def test_region_relabeling_and_cell_permutation_preserve_the_density() -> None:
    """Stable cell IDs should make replay independent of incidental ordering."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    cell_ids = np.asarray([101, 102, 201, 202])
    original = ConditionalAllocationMixture.from_aggregation(
        model,
        labels,
        sample_count=512,
        source_seed=831,
        source_provenance="permutation baseline",
        cell_ids=cell_ids,
    )
    relabelled = ConditionalAllocationMixture.from_aggregation(
        model,
        1 - labels,
        sample_count=512,
        source_seed=831,
        source_provenance="permutation baseline",
        cell_ids=cell_ids,
    )
    np.testing.assert_array_equal(
        original.projected_unit_mass_residual_factors,
        relabelled.projected_unit_mass_residual_factors[:, :, ::-1],
    )

    cell_order = np.asarray([2, 0, 3, 1])
    permuted_model = AdditiveDirichletAggregation(
        model.cell_alphas[cell_order],
        model.design[:, cell_order],
        model.noise_sd,
        model.summary_basis,
    )
    cell_permuted = ConditionalAllocationMixture.from_aggregation(
        permuted_model,
        labels[cell_order],
        sample_count=512,
        source_seed=831,
        source_provenance="permutation baseline",
        cell_ids=cell_ids[cell_order],
    )
    np.testing.assert_allclose(
        original.projected_unit_mass_residual_factors,
        cell_permuted.projected_unit_mass_residual_factors,
        rtol=0.0,
        atol=3.0e-16,
    )

    observation = np.asarray([0.4, -0.7, 1.2])
    masses = np.asarray([1.8, 2.4])
    assert original.log_likelihood(observation, masses) == pytest.approx(
        relabelled.log_likelihood(observation, masses[::-1]),
        abs=2.0e-15,
    )
    assert original.log_likelihood(observation, masses) == pytest.approx(
        cell_permuted.log_likelihood(observation, masses),
        abs=2.0e-15,
    )


def test_seeded_construction_replays_without_using_global_numpy_rng() -> None:
    """Bank construction should use only its private keyed PCG64 streams."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    np.random.seed(1928)
    expected_global_draws = np.random.random(8)
    np.random.seed(1928)

    first = ConditionalAllocationMixture.from_aggregation(
        model,
        labels,
        sample_count=128,
        source_seed=99,
        source_provenance="private RNG replay",
    )
    replay = ConditionalAllocationMixture.from_aggregation(
        model,
        labels,
        sample_count=128,
        source_seed=99,
        source_provenance="private RNG replay",
    )
    different = ConditionalAllocationMixture.from_aggregation(
        model,
        labels,
        sample_count=128,
        source_seed=100,
        source_provenance="private RNG replay",
    )
    observed_global_draws = np.random.random(8)

    np.testing.assert_array_equal(observed_global_draws, expected_global_draws)
    np.testing.assert_array_equal(
        first.projected_unit_mass_residual_factors,
        replay.projected_unit_mass_residual_factors,
    )
    assert not np.array_equal(
        first.projected_unit_mass_residual_factors,
        different.projected_unit_mass_residual_factors,
    )
    assert first.sha256 == replay.sha256
    assert first.sha256 != different.sha256


def test_observation_permutation_preserves_logp_and_gradient() -> None:
    """A simultaneous row permutation is only a change of observation order."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    original = ConditionalAllocationMixture.from_aggregation(
        model,
        labels,
        sample_count=600,
        source_seed=991,
        source_provenance="observation permutation",
    )
    order = np.asarray([2, 0, 1])
    permuted_model = AdditiveDirichletAggregation(
        model.cell_alphas,
        model.design[order],
        model.noise_sd[order],
        model.summary_basis[order],
    )
    permuted = ConditionalAllocationMixture.from_aggregation(
        permuted_model,
        labels,
        sample_count=600,
        source_seed=991,
        source_provenance="observation permutation",
    )
    observation = np.asarray([0.4, -0.7, 1.2])
    masses = np.asarray([1.8, 2.4])

    original_result = original.log_likelihood_and_mass_gradient(
        observation,
        masses,
    )
    permuted_result = permuted.log_likelihood_and_mass_gradient(
        observation[order],
        masses,
    )

    assert original_result[0] == pytest.approx(permuted_result[0], abs=3.0e-14)
    np.testing.assert_allclose(
        original_result[1],
        permuted_result[1],
        rtol=0.0,
        atol=3.0e-14,
    )


def test_signed_summary_coordinate_permutation_preserves_evaluation() -> None:
    """Signed orthogonal summary coordinates should not change the density."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    original = ConditionalAllocationMixture.from_aggregation(
        model,
        labels,
        sample_count=600,
        source_seed=225,
        source_provenance="summary coordinate baseline",
    )
    coordinate_map = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    transformed_model = AdditiveDirichletAggregation(
        model.cell_alphas,
        model.design,
        model.noise_sd,
        model.summary_basis @ coordinate_map,
    )
    transformed = ConditionalAllocationMixture.from_aggregation(
        transformed_model,
        labels,
        sample_count=600,
        source_seed=225,
        source_provenance="summary coordinate baseline",
    )
    observation = np.asarray([0.4, -0.7, 1.2])
    masses = np.asarray([1.8, 2.4])

    original_result = original.log_likelihood_and_mass_gradient(
        observation,
        masses,
    )
    transformed_result = transformed.log_likelihood_and_mass_gradient(
        observation,
        masses,
    )

    assert original_result[0] == pytest.approx(
        transformed_result[0],
        abs=3.0e-14,
    )
    np.testing.assert_allclose(
        original_result[1],
        transformed_result[1],
        rtol=0.0,
        atol=3.0e-14,
    )


def test_rank_zero_and_fine_partition_reduce_to_diagonal_gaussian() -> None:
    """Rank zero is the declared diagonal closure; a fine partition is exact."""
    observation = np.asarray([0.4, -0.7, 1.2])
    offset = np.asarray([0.1, -0.2, 0.05])

    rank_zero_model = _model(rank=0)
    rank_zero = ConditionalAllocationMixture.from_aggregation(
        rank_zero_model,
        np.asarray([0, 0, 1, 1]),
        sample_count=19,
        source_seed=52,
        source_provenance="rank-zero limit",
    )
    masses = np.asarray([1.8, 2.4])
    mean = offset + rank_zero.observation_mean_design @ masses
    residual = (observation - mean) / rank_zero.noise_sd
    expected = -float(np.sum(np.log(rank_zero.noise_sd))) - 0.5 * (
        3 * math.log(2.0 * math.pi) + float(residual @ residual)
    )
    assert rank_zero.summary_rank == 0
    assert rank_zero.projected_unit_mass_residual_factors.shape == (19, 0, 2)
    assert rank_zero.log_likelihood(
        observation,
        masses,
        mean_offset=offset,
    ) == pytest.approx(expected, abs=2.0e-15)

    fine_model = _model()
    fine = ConditionalAllocationMixture.from_aggregation(
        fine_model,
        np.arange(4, dtype=np.int64),
        sample_count=23,
        source_seed=52,
        source_provenance="fine-cell limit",
    )
    fine_masses = np.asarray([0.5, 1.2, 0.8, 1.5])
    np.testing.assert_array_equal(
        fine.projected_unit_mass_residual_factors,
        np.zeros((23, 2, 4)),
    )
    fine_mean = offset + fine.observation_mean_design @ fine_masses
    fine_residual = (observation - fine_mean) / fine.noise_sd
    fine_expected = -float(np.sum(np.log(fine.noise_sd))) - 0.5 * (
        3 * math.log(2.0 * math.pi) + float(fine_residual @ fine_residual)
    )
    assert fine.log_likelihood(
        observation,
        fine_masses,
        mean_offset=offset,
    ) == pytest.approx(fine_expected, abs=8.0e-15)


def test_artifact_json_roundtrip_fingerprint_and_immutability() -> None:
    """Canonical serialization should replay rank-zero shapes and identities."""
    artifact = _artifact(sample_count=17, rank=0)
    serialized = artifact.to_json()
    replay = ConditionalAllocationMixture.from_json(
        serialized,
        expected_sha256=artifact.sha256,
    )

    assert replay.to_json() == serialized
    assert replay.sha256 == artifact.sha256
    assert replay.partition_sha256 == artifact.partition_sha256
    assert replay.source_operator_sha256 == artifact.source_operator_sha256
    assert replay.projected_unit_mass_residual_factors.shape == (17, 0, 2)
    assert all(
        not values.flags.writeable
        for values in (
            replay.projected_unit_mass_residual_factors,
            replay.observation_mean_design,
            replay.noise_sd,
            replay.summary_basis,
            replay.labels,
            replay.cell_ids,
            replay.alpha_totals,
        )
    )
    for values in (
        replay.projected_unit_mass_residual_factors,
        replay.observation_mean_design,
        replay.noise_sd,
        replay.summary_basis,
        replay.labels,
        replay.cell_ids,
        replay.alpha_totals,
    ):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    with pytest.raises(ValueError, match="SHA-256"):
        ConditionalAllocationMixture.from_json(
            serialized,
            expected_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="canonical JSON"):
        noncanonical = json.dumps(json.loads(serialized))
        ConditionalAllocationMixture.from_json(
            noncanonical,
            expected_sha256=hashlib.sha256(noncanonical.encode()).hexdigest(),
        )
    axes_payload = json.loads(serialized)
    axes_payload["factor_axes"] = ["region", "summary", "sample"]
    wrong_axes = json.dumps(
        axes_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(ValueError, match="factor axes"):
        ConditionalAllocationMixture.from_json(
            wrong_axes,
            expected_sha256=hashlib.sha256(wrong_axes.encode()).hexdigest(),
        )


def test_replay_rejects_tampered_stored_arrays_and_external_identity() -> None:
    """Internal hashes and the required whole-artifact pin should fail closed."""
    artifact = _artifact(sample_count=17)
    serialized = artifact.to_json()

    noise_payload = json.loads(serialized)
    noise_payload["noise_sd"]["values"][0] *= 2.0
    tampered_noise = json.dumps(
        noise_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(ValueError, match="noise_sd_sha256"):
        ConditionalAllocationMixture.from_json(
            tampered_noise,
            expected_sha256=hashlib.sha256(tampered_noise.encode()).hexdigest(),
        )

    basis_payload = json.loads(serialized)
    basis_values = basis_payload["summary_basis"]["values"]
    for index in range(0, len(basis_values), artifact.summary_rank):
        basis_values[index] *= -1.0
    tampered_basis = json.dumps(
        basis_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(ValueError, match="summary_basis_sha256"):
        ConditionalAllocationMixture.from_json(
            tampered_basis,
            expected_sha256=hashlib.sha256(tampered_basis.encode()).hexdigest(),
        )

    external_payload = json.loads(serialized)
    external_payload["design_sha256"] = "0" * 64
    tampered_external_identity = json.dumps(
        external_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        ConditionalAllocationMixture.from_json(
            tampered_external_identity,
            expected_sha256=artifact.sha256,
        )


def test_likelihood_only_path_survives_unused_gradient_overflow() -> None:
    """Finite logp must not depend on representability of an unused gradient."""
    baseline = _artifact(sample_count=3, rank=0)
    extreme = ConditionalAllocationMixture(
        baseline.projected_unit_mass_residual_factors,
        np.full_like(baseline.observation_mean_design, 1.0e308),
        baseline.noise_sd,
        baseline.summary_basis,
        baseline.labels,
        baseline.cell_ids,
        baseline.alpha_totals,
        baseline.source_seed,
        "finite likelihood with unrepresentable derivative",
        baseline.cell_alphas_sha256,
        baseline.design_sha256,
        baseline.noise_sd_sha256,
        baseline.summary_basis_sha256,
    )
    observation = np.full(extreme.observation_count, 10.0)
    masses = np.zeros(extreme.region_count)

    assert math.isfinite(extreme.log_likelihood(observation, masses))
    with pytest.raises(ValueError, match="gradient"):
        extreme.log_likelihood_and_mass_gradient(observation, masses)


def test_construction_and_evaluation_reject_invalid_inputs() -> None:
    """Malformed scientific arrays and seeds should fail before evaluation."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    with pytest.raises(TypeError, match="integer array"):
        ConditionalAllocationMixture.from_aggregation(
            model,
            labels.astype(float),
            sample_count=10,
            source_seed=1,
            source_provenance="invalid labels",
        )
    with pytest.raises(TypeError, match="sample_count"):
        ConditionalAllocationMixture.from_aggregation(
            model,
            labels,
            sample_count=True,
            source_seed=1,
            source_provenance="invalid count",
        )
    with pytest.raises(ValueError, match="source_seed"):
        ConditionalAllocationMixture.from_aggregation(
            model,
            labels,
            sample_count=10,
            source_seed=-1,
            source_provenance="invalid seed",
        )
    with pytest.raises(ValueError, match="unique"):
        ConditionalAllocationMixture.from_aggregation(
            model,
            labels,
            sample_count=10,
            source_seed=1,
            source_provenance="invalid cell ids",
            cell_ids=np.zeros(4, dtype=np.int64),
        )
    with pytest.raises(ValueError, match="signed int64"):
        ConditionalAllocationMixture.from_aggregation(
            model,
            labels,
            sample_count=10,
            source_seed=1,
            source_provenance="out-of-range cell ids",
            cell_ids=np.asarray([0, 1, 2, 2**63], dtype=np.uint64),
        )

    artifact = _artifact(sample_count=20)
    with pytest.raises(ValueError, match="observation"):
        artifact.log_likelihood([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="masses"):
        artifact.log_likelihood(np.ones(3), [1.0, -2.0])
    with pytest.raises(ValueError, match="mean_offset"):
        artifact.log_likelihood(
            np.ones(3),
            [1.0, 2.0],
            mean_offset=np.ones(2),
        )
