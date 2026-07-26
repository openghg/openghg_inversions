"""Tests for the conditional low-rank aggregation-error Gaussian closure."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import log, pi

import numpy as np
import pytest
from scipy.integrate import quad

from openghg_inversions.experimental.rjmcmc.aggregation_error import FourCellAggregationOracle
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
    PartitionMassState,
    PartitionSummaryFactors,
    aggregation_from_full_tiling_problem,
    low_rank_gaussian_log_likelihood,
)
from openghg_inversions.experimental.rjmcmc.dyadic_tree import CanonicalDyadicTree
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    FullTilingProblem,
    initialize_full_tiling_posterior_state,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_tree import (
    GammaBetaTreePrior,
    GammaBetaTreeProblem,
    TreePartitionPrior,
)


def _model() -> AdditiveDirichletAggregation:
    """Return one small heterogeneous aggregation model."""
    alphas = np.array([[0.7, 1.1], [1.6, 0.9]])
    design = np.array(
        [
            [1.8, -0.5, 0.3, 0.9],
            [0.2, 1.4, -0.7, 0.1],
            [0.5, -0.2, 1.1, 0.8],
        ]
    )
    noise_sd = np.array([0.35, 0.8, 0.6])
    raw_basis = np.array([[1.0, 0.3], [0.4, 1.0], [-0.2, 0.5]])
    basis, _ = np.linalg.qr(raw_basis)
    return AdditiveDirichletAggregation(alphas, design, noise_sd, basis)


def _direct_covariance(
    columns: np.ndarray,
    alphas: np.ndarray,
    labels: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Return a direct block-Dirichlet covariance calculation."""
    result = np.zeros((columns.shape[0], columns.shape[0]))
    flat_labels = labels.reshape(-1)
    flat_alphas = alphas.reshape(-1)
    for region, mass in enumerate(masses):
        selected = flat_labels == region
        region_alphas = flat_alphas[selected]
        concentration = region_alphas.sum()
        probabilities = region_alphas / concentration
        dirichlet_covariance = (np.diag(probabilities) - np.outer(probabilities, probabilities)) / (
            concentration + 1.0
        )
        region_columns = columns[:, selected]
        result += mass**2 * region_columns @ dirichlet_covariance @ region_columns.T
    return result


def test_exact_moments_match_direct_dirichlet_formula() -> None:
    """Region loops should reproduce direct block-Dirichlet moment algebra."""
    model = _model()
    state = PartitionMassState(np.array([[0, 0], [1, 1]]), np.array([2.3, 1.7]))
    expected_observation = _direct_covariance(
        model.design,
        model.cell_alphas,
        state.labels,
        state.masses,
    )
    expected_summary = _direct_covariance(
        model.summary_design,
        model.cell_alphas,
        state.labels,
        state.masses,
    )

    np.testing.assert_allclose(
        model.observation_residual_covariance(state),
        expected_observation,
        rtol=2.0e-15,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        model.summary_residual_covariance(state),
        expected_summary,
        rtol=2.0e-15,
        atol=2.0e-16,
    )
    projected = (
        model.summary_basis.T
        @ (
            model.observation_residual_covariance(state)
            / model.noise_sd[:, np.newaxis]
            / model.noise_sd[np.newaxis, :]
        )
        @ model.summary_basis
    )
    np.testing.assert_allclose(model.summary_residual_covariance(state), projected, atol=5.0e-15)


def test_cached_partition_factors_exactly_replay_transparent_region_loop() -> None:
    """Cached unit-mass factors should replay means and covariance without a cell scan."""
    model = _model()
    labels = np.array([[0, 0], [1, 1]])
    masses = np.array([2.3, 1.7])
    state = PartitionMassState(labels, masses)
    factors = model.partition_factors(labels)

    np.testing.assert_allclose(
        factors.conditional_observation_mean(masses),
        model.conditional_observation_mean(state),
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_allclose(
        factors.summary_residual_covariance(masses),
        model.summary_residual_covariance(state),
        rtol=3.0e-16,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        factors.conditional_summary_mean(masses),
        model.summary_basis.T @ (model.conditional_observation_mean(state) / model.noise_sd),
        rtol=0.0,
        atol=2.0e-15,
    )
    assert factors.storage_nbytes == sum(
        array.nbytes
        for array in (
            factors.labels,
            factors.alpha_totals,
            factors.observation_mean_design,
            factors.summary_mean_design,
            factors.summary_covariance_factors,
        )
    )
    assert all(
        not array.flags.writeable
        for array in (
            factors.labels,
            factors.alpha_totals,
            factors.observation_mean_design,
            factors.summary_mean_design,
            factors.summary_covariance_factors,
        )
    )


def test_cached_factors_cover_singleton_singular_rank_zero_and_label_permutation() -> None:
    """Degenerate factors and consistent region relabeling should remain exact."""
    model = _model()
    fine_labels = np.arange(4, dtype=np.int64).reshape(2, 2)
    fine_masses = np.array([0.5, 1.2, 0.8, 1.5])
    fine = model.partition_factors(fine_labels)
    np.testing.assert_array_equal(fine.summary_covariance_factors, np.zeros((4, 2, 2)))
    np.testing.assert_array_equal(
        fine.summary_residual_covariance(fine_masses),
        np.zeros((2, 2)),
    )

    repeated_design = np.tile(np.array([[1.0], [-0.4], [0.8]]), (1, 4))
    singular_model = AdditiveDirichletAggregation(
        model.cell_alphas,
        repeated_design,
        model.noise_sd,
        model.summary_basis,
    )
    singular = singular_model.partition_factors(np.zeros((2, 2), dtype=np.int64))
    singular_state = PartitionMassState(np.zeros((2, 2), dtype=np.int64), [1.4])
    np.testing.assert_allclose(
        singular.summary_residual_covariance([1.4]),
        singular_model.summary_residual_covariance(singular_state),
        rtol=3.0e-16,
        atol=0.0,
    )
    assert np.linalg.matrix_rank(singular.summary_covariance_factors[0]) <= 1

    rank_zero = AdditiveDirichletAggregation(
        model.cell_alphas,
        model.design,
        model.noise_sd,
        np.empty((3, 0)),
    ).partition_factors(np.array([[0, 0], [1, 1]]))
    assert rank_zero.summary_covariance_factors.shape == (2, 0, 0)
    assert rank_zero.summary_residual_covariance([2.3, 1.7]).shape == (0, 0)

    original = model.partition_factors(np.array([[0, 0], [1, 1]]))
    permuted = model.partition_factors(np.array([[1, 1], [0, 0]]))
    np.testing.assert_array_equal(
        original.conditional_observation_mean([2.3, 1.7]),
        permuted.conditional_observation_mean([1.7, 2.3]),
    )
    np.testing.assert_array_equal(
        original.summary_residual_covariance([2.3, 1.7]),
        permuted.summary_residual_covariance([1.7, 2.3]),
    )


def test_cached_factors_use_stable_centered_covariance_for_large_offsets() -> None:
    """Large common design offsets must not corrupt a tiny aggregation covariance."""
    perturbation = np.array([-1.5, -0.5, 0.5, 1.5]) * 1.0e-4
    design = (1.0e8 + perturbation)[np.newaxis, :]
    model = AdditiveDirichletAggregation(
        np.ones((2, 2)),
        design,
        1.0,
        np.ones((1, 1)),
    )
    factors = model.partition_factors(np.zeros((2, 2), dtype=np.int64))
    expected = float(np.mean(perturbation**2) / 5.0)

    assert factors.summary_covariance_factors[0, 0, 0] > 0.0
    assert factors.summary_covariance_factors[0, 0, 0] == pytest.approx(
        expected,
        rel=2.0e-4,
    )


def test_partition_factor_constructor_rejects_invalid_covariance_factors() -> None:
    """Public cached factors must reject asymmetric or materially indefinite inputs."""
    common = {
        "labels": np.array([0]),
        "alpha_totals": np.array([1.0]),
        "observation_mean_design": np.ones((1, 1)),
        "summary_mean_design": np.ones((2, 1)),
    }
    with pytest.raises(ValueError, match="symmetric"):
        PartitionSummaryFactors(
            **common,
            summary_covariance_factors=np.array([[[1.0, 0.5], [0.0, 1.0]]]),
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        PartitionSummaryFactors(
            **common,
            summary_covariance_factors=np.array([[[1.0, 0.0], [0.0, -1.0]]]),
        )


def test_exact_moments_match_monte_carlo_hidden_allocations() -> None:
    """Analytic means and covariance should match sampled hidden Dirichlet shares."""
    model = _model()
    state = PartitionMassState(np.array([[0, 0], [1, 1]]), np.array([2.3, 1.7]))
    rng = np.random.default_rng(731)
    sample_count = 180_000
    native = np.empty((sample_count, 4))
    for region, mass in enumerate(state.masses):
        selected = np.flatnonzero(state.labels.reshape(-1) == region)
        shares = rng.dirichlet(model.cell_alphas.reshape(-1)[selected], size=sample_count)
        native[:, selected] = float(mass) * shares
    observation_samples = native @ model.design.T

    np.testing.assert_allclose(
        observation_samples.mean(axis=0),
        model.conditional_observation_mean(state),
        rtol=0.0,
        atol=4.5e-3,
    )
    np.testing.assert_allclose(
        np.cov(observation_samples, rowvar=False, ddof=0),
        model.observation_residual_covariance(state),
        rtol=1.2e-2,
        atol=3.0e-3,
    )


def test_hybrid_log_density_matches_dense_normalized_gaussian() -> None:
    """Small-space Woodbury evaluation should equal a dense Gaussian density."""
    model = _model()
    state = PartitionMassState(np.array([[0, 0], [1, 1]]), np.array([2.3, 1.7]))
    offset = np.array([0.4, -0.2, 0.1])
    mean = model.conditional_observation_mean(state) + offset
    observation = np.array([2.1, 0.8, -0.4])

    actual = model.hybrid_log_likelihood(observation, state, mean_offset=offset)
    covariance = model.dense_hybrid_covariance(state)
    sign, log_determinant = np.linalg.slogdet(covariance)
    residual = observation - mean
    expected = -0.5 * (
        observation.size * log(2.0 * pi) + log_determinant + residual @ np.linalg.solve(covariance, residual)
    )

    assert sign == 1.0
    assert actual == pytest.approx(expected, abs=2.0e-14)


def test_scalar_hybrid_density_integrates_to_one() -> None:
    """The determinant correction should retain exact Gaussian normalization."""
    integral, _ = quad(
        lambda value: np.exp(
            low_rank_gaussian_log_likelihood(
                [value],
                [0.4],
                [0.7],
                [[1.0]],
                [[1.3]],
            )
        ),
        -np.inf,
        np.inf,
        epsabs=2.0e-11,
    )

    assert integral == pytest.approx(1.0, abs=2.0e-10)


def test_zero_summary_covariance_is_independent_gaussian_baseline() -> None:
    """The hybrid density should reduce exactly to the diagonal Gaussian at S=0."""
    observation = np.array([0.7, -0.2, 0.5])
    mean = np.array([0.1, -0.1, 0.8])
    noise_sd = np.array([0.35, 0.8, 0.6])
    basis = np.eye(3)[:, :2]
    residual = (observation - mean) / noise_sd
    expected = (
        -0.5 * float(residual @ residual)
        - float(np.sum(np.log(noise_sd)))
        - 0.5 * observation.size * log(2.0 * pi)
    )

    actual = low_rank_gaussian_log_likelihood(
        observation,
        mean,
        noise_sd,
        basis,
        np.zeros((2, 2)),
    )

    assert actual == pytest.approx(expected, abs=2.0e-15)


def test_rank_zero_is_independent_gaussian_baseline() -> None:
    """An empty fixed summary should implement the predeclared rank-zero model."""
    observation = np.array([0.7, -0.2, 0.5])
    mean = np.array([0.1, -0.1, 0.8])
    noise_sd = np.array([0.35, 0.8, 0.6])
    residual = (observation - mean) / noise_sd
    expected = (
        -0.5 * float(residual @ residual)
        - float(np.sum(np.log(noise_sd)))
        - 0.5 * observation.size * log(2.0 * pi)
    )

    actual = low_rank_gaussian_log_likelihood(
        observation,
        mean,
        noise_sd,
        np.empty((observation.size, 0)),
        np.empty((0, 0)),
    )

    assert actual == pytest.approx(expected, abs=2.0e-15)


def test_large_summary_variance_avoids_quadratic_cancellation() -> None:
    """Large residuals and covariance must retain the small solved quadratic."""
    observation = np.array([1.0e10])
    variance = np.array([[1.0e16]])
    expected = -0.5 * (
        log(2.0 * pi) + log(1.0 + variance.item()) + observation.item() ** 2 / (1.0 + variance.item())
    )

    actual = low_rank_gaussian_log_likelihood(
        observation,
        [0.0],
        [1.0],
        [[1.0]],
        variance,
    )

    assert actual == pytest.approx(expected, rel=0.0, abs=2.0e-12)


def test_roundoff_indefinite_summary_covariance_is_repaired_to_psd_boundary() -> None:
    """A tolerated negative eigenmode must not become a smaller total variance."""
    observation = np.array([1.0, 2.0])
    basis = np.eye(2)
    roundoff_indefinite = np.diag([1.0e16, -0.5])
    clipped = np.diag([1.0e16, 0.0])

    actual = low_rank_gaussian_log_likelihood(
        observation,
        np.zeros(2),
        np.ones(2),
        basis,
        roundoff_indefinite,
    )
    expected = low_rank_gaussian_log_likelihood(
        observation,
        np.zeros(2),
        np.ones(2),
        basis,
        clipped,
    )

    assert actual == pytest.approx(expected, rel=0.0, abs=2.0e-15)


def test_fine_partition_has_zero_aggregation_error() -> None:
    """One cell per region should leave no hidden within-region allocation."""
    model = _model()
    state = PartitionMassState(np.array([[0, 1], [2, 3]]), np.array([0.5, 1.2, 0.8, 1.5]))

    np.testing.assert_array_equal(model.observation_residual_covariance(state), np.zeros((3, 3)))
    np.testing.assert_array_equal(model.summary_residual_covariance(state), np.zeros((2, 2)))
    np.testing.assert_allclose(
        model.conditional_native_mean(state),
        state.masses.reshape(model.cell_shape),
    )


def test_partition_label_permutation_does_not_change_moments_or_density() -> None:
    """Consistent relabeling must not attach scientific meaning to label order."""
    model = _model()
    original = PartitionMassState(np.array([[0, 0], [1, 1]]), np.array([2.3, 1.7]))
    permuted = PartitionMassState(np.array([[1, 1], [0, 0]]), np.array([1.7, 2.3]))
    observation = np.array([2.1, 0.8, -0.4])

    np.testing.assert_array_equal(
        model.conditional_native_mean(original),
        model.conditional_native_mean(permuted),
    )
    np.testing.assert_allclose(
        model.summary_residual_covariance(original),
        model.summary_residual_covariance(permuted),
        rtol=0.0,
        atol=2.0e-16,
    )
    assert model.hybrid_log_likelihood(observation, original) == pytest.approx(
        model.hybrid_log_likelihood(observation, permuted),
        abs=2.0e-15,
    )


def test_cell_and_observation_permutations_preserve_scientific_results() -> None:
    """Aligned storage permutations must not change moments or log density."""
    model = _model()
    state = PartitionMassState(np.array([[0, 0], [1, 1]]), np.array([2.3, 1.7]))
    observation = np.array([2.1, 0.8, -0.4])
    cell_order = np.array([2, 0, 3, 1])
    observation_order = np.array([2, 0, 1])
    permuted_labels = state.labels.reshape(-1)[cell_order]
    permuted_alphas = model.cell_alphas.reshape(-1)[cell_order]
    permuted_design = model.design[observation_order][:, cell_order]
    permuted_noise = model.noise_sd[observation_order]
    permuted_basis = model.summary_basis[observation_order]
    permuted_model = AdditiveDirichletAggregation(
        permuted_alphas,
        permuted_design,
        permuted_noise,
        permuted_basis,
    )
    permuted_state = PartitionMassState(permuted_labels, state.masses)

    np.testing.assert_allclose(
        permuted_model.conditional_observation_mean(permuted_state),
        model.conditional_observation_mean(state)[observation_order],
        rtol=0.0,
        atol=5.0e-16,
    )
    np.testing.assert_allclose(
        permuted_model.summary_residual_covariance(permuted_state),
        model.summary_residual_covariance(state),
        rtol=0.0,
        atol=5.0e-15,
    )
    assert permuted_model.hybrid_log_likelihood(
        observation[observation_order],
        permuted_state,
    ) == pytest.approx(
        model.hybrid_log_likelihood(observation, state),
        abs=2.0e-14,
    )


def test_orthogonal_summary_coordinate_rotation_preserves_density() -> None:
    """Only the retained summary subspace, not its coordinates, should matter."""
    model = _model()
    state = PartitionMassState(np.array([[0, 0], [1, 1]]), np.array([2.3, 1.7]))
    rotation = np.array([[0.8, -0.6], [0.6, 0.8]])
    rotated_model = AdditiveDirichletAggregation(
        model.cell_alphas,
        model.design,
        model.noise_sd,
        model.summary_basis @ rotation,
    )
    observation = np.array([2.1, 0.8, -0.4])

    np.testing.assert_allclose(
        rotated_model.summary_residual_covariance(state),
        rotation.T @ model.summary_residual_covariance(state) @ rotation,
        rtol=0.0,
        atol=5.0e-15,
    )
    assert rotated_model.hybrid_log_likelihood(
        observation,
        state,
    ) == pytest.approx(
        model.hybrid_log_likelihood(observation, state),
        abs=2.0e-14,
    )


def test_nonzero_singular_summary_covariance_is_supported() -> None:
    """A rank-deficient nonzero aggregation covariance needs no jitter."""
    covariance = np.array([[2.0, 2.0], [2.0, 2.0]])
    observation = np.array([1.3, -0.7])

    actual = low_rank_gaussian_log_likelihood(
        observation,
        np.zeros(2),
        np.ones(2),
        np.eye(2),
        covariance,
    )
    dense = np.eye(2) + covariance
    _, log_determinant = np.linalg.slogdet(dense)
    expected = -0.5 * (
        2.0 * log(2.0 * pi) + log_determinant + observation @ np.linalg.solve(dense, observation)
    )

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, abs=2.0e-15)


def test_four_cell_oracle_shapes_give_the_same_conditional_root_moments() -> None:
    """The closure should use the exact native model underlying the four-cell oracle."""
    shapes = np.array([0.7, 1.1, 1.6, 0.9])
    oracle = FourCellAggregationOracle(shapes, gamma_rate=1.4)
    model = _model()
    total = 3.4
    state = PartitionMassState(np.zeros((2, 2), dtype=np.int64), np.array([total]))
    direct = _direct_covariance(
        model.design,
        shapes.reshape(2, 2),
        state.labels,
        state.masses,
    )

    np.testing.assert_allclose(
        model.conditional_native_mean(state).reshape(-1),
        total * oracle.nominal_fractions,
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_allclose(model.observation_residual_covariance(state), direct, atol=5.0e-16)


def test_full_tiling_bridge_uses_physical_mass_design_and_additive_alpha() -> None:
    """The bridge mean must match FullTilingProblem's physical-mass convention."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    nominal_mass = np.array([1.0, 2.0, 3.0, 4.0])
    sensitivity = np.array(
        [
            [1.3, -0.4, 0.7, 2.1],
            [0.2, 1.6, -0.8, 0.5],
        ]
    )
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        nominal_mass,
        concentration=4.0,
        root_mean=1.0,
        root_variance=0.25,
    )
    base = GammaBetaTreeProblem(
        observations=np.zeros(2),
        observation_sd=np.array([0.5, 1.5]),
        sensitivity=sensitivity,
        prior=prior,
        partition_prior=TreePartitionPrior.uniform_k(tree),
    )
    problem = FullTilingProblem(base, concentration=7.0)
    posterior_state = initialize_full_tiling_posterior_state(problem, k=2)
    labels = np.empty(problem.shape, dtype=np.int64)
    for label, leaf in enumerate(posterior_state.tiling_state.tiling.leaves):
        labels[
            leaf.row_start : leaf.row_stop,
            leaf.col_start : leaf.col_stop,
        ] = label
    aggregation = aggregation_from_full_tiling_problem(
        problem,
        np.eye(problem.observations.size),
    )
    state = PartitionMassState(labels, posterior_state.leaf_masses)

    assert aggregation.design is problem.base.sensitivity
    assert aggregation.noise_sd is problem.observation_sd
    np.testing.assert_allclose(
        aggregation.conditional_observation_mean(state),
        posterior_state.dynamic_prediction,
        rtol=0.0,
        atol=3.0e-16,
    )
    for label, leaf in enumerate(posterior_state.tiling_state.tiling.leaves):
        assert aggregation.cell_alphas[labels == label].sum() == pytest.approx(
            problem.allocation_prior.alpha(leaf),
            abs=2.0e-15,
        )


def test_value_objects_own_read_only_arrays() -> None:
    """Public arrays should be copied, immutable, and frozen after validation."""
    labels = np.array([[0, 0], [1, 1]])
    masses = np.array([2.3, 1.7])
    state = PartitionMassState(labels, masses)
    labels[...] = 1
    masses[...] = 9.0

    np.testing.assert_array_equal(state.labels, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(state.masses, [2.3, 1.7])
    assert not state.labels.flags.writeable
    assert not state.masses.flags.writeable
    with pytest.raises(ValueError):
        state.masses[0] = 4.0
    with pytest.raises(FrozenInstanceError):
        state.masses = np.array([1.0, 1.0])  # type: ignore[misc]

    alphas = np.array([[0.7, 1.1], [1.6, 0.9]])
    design = np.arange(12.0).reshape(3, 4)
    basis = np.eye(3)[:, :2]
    model = AdditiveDirichletAggregation(alphas, design, 1.0, basis)
    alphas[...] = 9.0
    design[...] = 9.0
    basis[...] = 0.0
    np.testing.assert_array_equal(model.cell_alphas, [[0.7, 1.1], [1.6, 0.9]])
    np.testing.assert_array_equal(model.design, np.arange(12.0).reshape(3, 4))
    np.testing.assert_array_equal(model.summary_basis, np.eye(3)[:, :2])
    assert not model.cell_alphas.flags.writeable
    assert not model.design.flags.writeable
    assert not model.summary_basis.flags.writeable


@pytest.mark.parametrize(
    ("labels", "masses", "error", "message"),
    [
        ([0.0, 0.0], [1.0], TypeError, "integer"),
        ([0, 2], [1.0, 2.0], ValueError, "contiguous"),
        ([-1, 0], [1.0, 2.0], ValueError, "non-negative"),
        ([0, 1], [1.0], ValueError, "one entry"),
        ([0, 1], [1.0, 0.0], ValueError, "strictly positive"),
    ],
)
def test_partition_state_rejects_invalid_inputs(
    labels: list[float] | list[int],
    masses: list[float],
    error: type[Exception],
    message: str,
) -> None:
    """Malformed labels and masses should fail before scientific evaluation."""
    with pytest.raises(error, match=message):
        PartitionMassState(np.asarray(labels), np.asarray(masses))


def test_model_and_likelihood_reject_invalid_inputs() -> None:
    """Shape, basis, covariance, and state mismatches should fail closed."""
    design = np.ones((3, 4))
    with pytest.raises(ValueError, match="strictly positive"):
        AdditiveDirichletAggregation([1.0, 0.0, 2.0, 3.0], design, 1.0, np.eye(3))
    with pytest.raises(ValueError, match="finite additive total"):
        AdditiveDirichletAggregation(np.full(4, 1.0e308), design, 1.0, np.eye(3))
    with pytest.raises(ValueError, match="one column"):
        AdditiveDirichletAggregation(np.ones(4), np.ones((3, 3)), 1.0, np.eye(3))
    with pytest.raises(ValueError, match="orthonormal"):
        AdditiveDirichletAggregation(np.ones(4), design, 1.0, np.ones((3, 2)))
    with pytest.raises(ValueError, match="strictly positive"):
        AdditiveDirichletAggregation(np.ones(4), design, [1.0, 0.0, 1.0], np.eye(3))
    with pytest.raises(ValueError, match="positive semidefinite"):
        low_rank_gaussian_log_likelihood([0.0], [0.0], 1.0, [[1.0]], [[-2.0]])
    with pytest.raises(ValueError, match="symmetric"):
        low_rank_gaussian_log_likelihood(
            [0.0, 0.0],
            [0.0, 0.0],
            1.0,
            np.eye(2),
            [[1.0, 0.2], [0.0, 1.0]],
        )

    model = _model()
    wrong_shape = PartitionMassState(np.array([0, 0, 1, 1]), np.array([2.0, 2.0]))
    with pytest.raises(ValueError, match="same shape"):
        model.summary_residual_covariance(wrong_shape)
    with pytest.raises(ValueError, match="mean_offset"):
        model.hybrid_log_likelihood(
            [0.0, 0.0, 0.0], PartitionMassState([[0, 0], [0, 0]], [1.0]), mean_offset=[0.0]
        )
