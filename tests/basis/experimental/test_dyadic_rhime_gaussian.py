"""Tests for the Bocquet-consistent RHIME Gaussian multiscale model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from openghg_inversions.basis.experimental.dyadic.rhime_gaussian import RHIMEGaussianMultiscale
from openghg_inversions.basis.experimental.dyadic.state import PartitionState


def _native_example() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return a small fully supported RHIME design with nontrivial flux signs."""
    footprint = np.array(
        [
            [[1.0, 0.5], [0.2, -0.3]],
            [[-0.4, 0.8], [1.1, 0.6]],
            [[0.7, -0.2], [0.3, 1.2]],
        ]
    )
    prior_flux = np.array([[2.0, -1.0], [3.0, 4.0]])
    G = footprint * prior_flux[np.newaxis, ...]
    return G, prior_flux, np.array([0.7, 1.1, 0.9]), 0.6


def _model() -> RHIMEGaussianMultiscale:
    """Build the standard fully supported test model."""
    G, prior_flux, r_diag, relative_sd = _native_example()
    return RHIMEGaussianMultiscale.from_native_grid(
        G,
        prior_flux,
        r_diag,
        coarsen_factor=1,
        relative_prior_sd=relative_sd,
    )


def test_dense_weighted_inverse_and_covariances_match_model() -> None:
    """A dense fine-state oracle should reproduce prolongation and covariance formulas."""
    G, prior_flux, _, relative_sd = _native_example()
    model = _model()
    tree = model.design.tree
    state = PartitionState.root(tree).split(tree, tree.root_id)

    flux_vector = prior_flux.reshape(-1)
    fine_covariance = relative_sd**2 * np.diag(np.square(flux_vector))
    membership = np.zeros((2, flux_vector.size))
    membership[0, :2] = 1.0
    membership[1, 2:] = 1.0
    prolongation = np.diag(flux_vector) @ membership.T
    fine_precision = np.diag(1.0 / np.diag(fine_covariance))
    gamma = np.linalg.solve(
        prolongation.T @ fine_precision @ prolongation,
        prolongation.T @ fine_precision,
    )
    residual_operator = np.eye(flux_vector.size) - prolongation @ gamma

    np.testing.assert_allclose(gamma @ prolongation, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(
        gamma @ fine_covariance @ residual_operator.T,
        np.zeros((2, flux_vector.size)),
        atol=1e-14,
    )

    reduced_design, variance_diag = model.reduced_design_and_variance(state)
    dense_region_covariance = gamma @ fine_covariance @ gamma.T
    np.testing.assert_allclose(reduced_design, G.reshape(G.shape[0], -1) @ membership.T)
    np.testing.assert_allclose(np.diag(variance_diag), dense_region_covariance, atol=1e-14)

    footprint = G.reshape(G.shape[0], -1) @ np.diag(1.0 / flux_vector)
    dense_aggregation_covariance = (
        footprint @ residual_operator @ fine_covariance @ residual_operator.T @ footprint.T
    )
    np.testing.assert_allclose(
        model.aggregation_error_covariance(state),
        dense_aggregation_covariance,
        atol=1e-14,
    )


def test_effective_covariance_preserves_innovation_for_every_partition() -> None:
    """Adding reduced signal to effective error should recover one invariant innovation."""
    model = _model()
    tree = model.design.tree
    root = PartitionState.root(tree)
    intermediate = root.split(tree, tree.root_id)
    leaves = PartitionState(frozenset(tree.leaf_ids))

    for state in (root, intermediate, leaves):
        np.testing.assert_allclose(
            model.effective_observation_covariance(state) + model.reduced_signal_covariance(state),
            model.innovation_covariance,
            atol=1e-14,
        )


def test_additive_score_equals_direct_dfs_with_full_effective_covariance() -> None:
    """Precomputed tile contributions should equal direct observation-space DFS."""
    model = _model()
    tree = model.design.tree
    state = PartitionState.root(tree).split(tree, tree.root_id)
    first_child = tree.children(tree.root_id)[0]
    state = state.split(tree, first_child)

    reduced_signal = model.reduced_signal_covariance(state)
    effective_error = model.effective_observation_covariance(state)
    direct_dfs = float(np.trace(np.linalg.solve(effective_error + reduced_signal, reduced_signal)))

    np.testing.assert_allclose(model.score(state), direct_dfs, atol=1e-14)
    np.testing.assert_allclose(
        model.split_gain(tree.root_id),
        model.score(PartitionState.root(tree).split(tree, tree.root_id))
        - model.score(PartitionState.root(tree)),
        atol=1e-14,
    )


def test_fisher_scores_match_direct_base_error_formula() -> None:
    """Additive Fisher scores should use the declared base observation error."""
    model = _model()
    tree = model.design.tree
    state = PartitionState.root(tree).split(tree, tree.root_id)
    design, variances = model.reduced_design_and_variance(state)

    direct = float(np.sum(variances * np.sum(np.square(design) / model.r_diag[:, None], axis=0)))
    native_design = model.native_design.reshape(model.r_diag.size, -1)
    direct_full = float(
        model.relative_prior_sd**2
        * np.sum(np.square(native_design) / model.r_diag[:, None])
    )

    np.testing.assert_allclose(model.fisher_score(state), direct, atol=1e-14)
    np.testing.assert_allclose(model.full_grid_fisher, direct_full, atol=1e-14)
    assert model.fisher_score(state) <= model.full_grid_fisher + 1e-12


def test_fisher_whitening_avoids_finite_large_value_overflow() -> None:
    """Fisher construction should divide by error scale before squaring."""
    model = RHIMEGaussianMultiscale.from_native_grid(
        np.array([[[1.0e155]]]),
        np.ones((1, 1)),
        np.array([1.0e308]),
        coarsen_factor=1,
        relative_prior_sd=1.0e-2,
    )
    state = PartitionState.root(model.design.tree)

    np.testing.assert_allclose(model.fisher_score(state), 1.0e-2, rtol=1e-14)
    np.testing.assert_allclose(model.full_grid_fisher, 1.0e-2, rtol=1e-14)


def test_data_dependent_scores_match_equation_45_and_native_bound() -> None:
    """Equation 45 scores should retain the projected posterior-mean update."""
    model = _model()
    tree = model.design.tree
    state = PartitionState.root(tree).split(tree, tree.root_id)
    innovations = np.array([0.8, -0.3, 1.1])
    solved = np.linalg.solve(model.innovation_covariance, innovations)

    expected_tiles = model.prior_variance_by_node * np.square(model.design.values.T @ solved)
    native_design = model.native_design.reshape(innovations.size, -1)
    expected_full = model.relative_prior_sd**2 * np.sum(np.square(native_design.T @ solved))

    tile_scores = model.data_dependent_tile_scores(innovations)
    np.testing.assert_allclose(tile_scores, expected_tiles, atol=1e-14)
    np.testing.assert_allclose(
        model.data_dependent_score(state, innovations),
        np.sum(expected_tiles[list(state.ordered_active())]),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        model.full_grid_data_dependent_score(innovations),
        expected_full,
        atol=1e-14,
    )
    assert model.data_dependent_score(state, innovations) <= expected_full + 1e-12
    assert not tile_scores.flags.writeable


def test_native_posterior_marginals_match_dense_conditioning() -> None:
    """Chunked native posterior marginals should match a dense Gaussian oracle."""
    model = _model()
    innovations = np.array([0.8, -0.3, 1.1])
    native_design = model.native_design.reshape(innovations.size, -1)
    prior_covariance = model.relative_prior_sd**2 * np.eye(native_design.shape[1])
    solved_design = np.linalg.solve(model.innovation_covariance, native_design)
    expected_mean = prior_covariance @ native_design.T @ np.linalg.solve(
        model.innovation_covariance,
        innovations,
    )
    expected_covariance = prior_covariance - (
        prior_covariance @ native_design.T @ solved_design @ prior_covariance
    )

    posterior = model.native_posterior_marginals(innovations, chunk_size=2)

    np.testing.assert_allclose(posterior.mean_increment.ravel(), expected_mean, atol=1e-14)
    np.testing.assert_allclose(
        posterior.marginal_variance.ravel(),
        np.diag(expected_covariance),
        atol=1e-14,
    )
    np.testing.assert_array_equal(posterior.support, model.native_support)
    assert not posterior.mean_increment.flags.writeable
    assert not posterior.marginal_variance.flags.writeable
    assert not posterior.support.flags.writeable


def test_native_posterior_retains_prior_at_unsupported_locations() -> None:
    """A zero-design unsupported location should keep its prior marginal."""
    model = RHIMEGaussianMultiscale.from_native_grid(
        np.array([[[1.0, 0.0]], [[-0.4, 0.0]]]),
        np.array([[2.0, 0.0]]),
        np.array([0.7, 1.1]),
        coarsen_factor=1,
        relative_prior_sd=0.6,
    )

    posterior = model.native_posterior_marginals([0.5, -0.2], chunk_size=1)

    assert posterior.mean_increment[0, 1] == 0.0
    np.testing.assert_allclose(posterior.marginal_variance[0, 1], 0.6**2, atol=1e-14)
    assert not posterior.support[0, 1]


def test_native_posterior_variance_remains_positive_under_strong_information() -> None:
    """The SVD fallback should resolve posterior variance below subtraction precision."""
    model = RHIMEGaussianMultiscale.from_native_grid(
        np.ones((1, 1, 1)),
        np.ones((1, 1)),
        np.array([1.0e-16]),
        coarsen_factor=1,
        relative_prior_sd=1.0,
    )

    posterior = model.native_posterior_marginals([0.2])

    expected = 1.0 / (1.0 + 1.0e16)
    assert posterior.marginal_variance[0, 0] > 0.0
    np.testing.assert_allclose(posterior.marginal_variance[0, 0], expected, rtol=1e-14)


def test_aggregation_covariance_avoids_large_matrix_subtraction_cancellation() -> None:
    """Centered native scatter should remain valid for nearly equal huge columns."""
    G = np.array([[[1.0e8, 1.0e8 + 1.0]]])
    model = RHIMEGaussianMultiscale.from_native_grid(
        G,
        np.ones((1, 2)),
        [1.0],
        coarsen_factor=1,
    )
    state = PartitionState.root(model.design.tree)

    np.testing.assert_allclose(model.aggregation_error_covariance(state), [[0.5]], atol=1e-14)
    np.testing.assert_allclose(model.effective_observation_covariance(state), [[1.5]], atol=1e-14)
    np.testing.assert_allclose(
        model.effective_observation_covariance(state) + model.reduced_signal_covariance(state),
        model.innovation_covariance,
        rtol=1e-15,
        atol=8.0,
    )


def test_full_grid_dfs_bounds_root_intermediate_and_leaf_scores() -> None:
    """No dyadic reduction should contain more DFS than the native fine state."""
    model = _model()
    tree = model.design.tree
    root = PartitionState.root(tree)
    intermediate = root.split(tree, tree.root_id)
    leaves = PartitionState(frozenset(tree.leaf_ids))

    for state in (root, intermediate, leaves):
        assert model.score(state) <= model.full_grid_dfs + 1e-12
    np.testing.assert_allclose(model.score(leaves), model.full_grid_dfs, atol=1e-14)


def test_coarsening_counts_native_support_and_prunes_zero_support_regions() -> None:
    """Coarse leaves should retain native support counts and omit empty coefficients."""
    prior_flux = np.array(
        [
            [1.0, 1.0e-4, 0.0, 0.0, 2.0],
            [1.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 4.0],
        ]
    )
    support = np.abs(prior_flux) > 1.0e-3
    footprint = np.arange(1.0, 31.0).reshape(2, 3, 5) / 10.0
    G = footprint * prior_flux[np.newaxis, ...]
    G[:, ~support] = 0.0
    G[:, 0, 1] = 1.0e-13  # Accepted roundoff is masked before covariance construction.

    model = RHIMEGaussianMultiscale.from_native_grid(
        G,
        prior_flux,
        [0.5, 0.8],
        coarsen_factor=2,
        relative_prior_sd=0.4,
        flux_tolerance=1.0e-3,
    )
    tree = model.design.tree
    leaf_state = PartitionState(frozenset(tree.leaf_ids))
    expected_coarse_support = np.array([[2, 0, 1], [0, 1, 1]])

    assert tree.shape == (2, 3)
    for node_id in tree.leaf_ids:
        tile = tree.tile(node_id)
        assert model.support_by_node[node_id] == expected_coarse_support[tile.row_start, tile.col_start]

    reduced_design, variance_diag = model.reduced_design_and_variance(leaf_state)
    assert model.effective_region_count(leaf_state) == 4
    assert reduced_design.shape == (2, 4)
    np.testing.assert_allclose(np.sort(variance_diag), np.sort(0.4**2 / np.array([2, 1, 1, 1])))
    zero_support_nodes = np.flatnonzero(model.support_by_node == 0)
    np.testing.assert_array_equal(model.prior_variance_by_node[zero_support_nodes], 0.0)
    np.testing.assert_array_equal(model.tile_scores[zero_support_nodes], 0.0)

    masked_G = G.copy()
    masked_G[:, ~support] = 0.0
    expected_full_signal = 0.4**2 * (masked_G.reshape(2, -1) @ masked_G.reshape(2, -1).T)
    np.testing.assert_allclose(model.full_signal_covariance, expected_full_signal, atol=1e-14)


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (lambda G, flux, r, kwargs: (G[0], flux, r, kwargs), ValueError, "G must have shape"),
        (
            lambda G, flux, r, kwargs: (np.empty((0, 2, 2)), flux, np.empty(0), kwargs),
            ValueError,
            "non-empty",
        ),
        (lambda G, flux, r, kwargs: (G, flux.reshape(-1), r, kwargs), ValueError, "prior_flux must have"),
        (lambda G, flux, r, kwargs: (G, flux[:, :1], r, kwargs), ValueError, "spatial shape"),
        (
            lambda G, flux, r, kwargs: (G, np.where(flux == flux.flat[0], np.inf, flux), r, kwargs),
            ValueError,
            "finite",
        ),
        (lambda G, flux, r, kwargs: (G, flux, r[:2], kwargs), ValueError, "length"),
        (lambda G, flux, r, kwargs: (G, flux, np.array([1.0, 0.0, 1.0]), kwargs), ValueError, "positive"),
        (lambda G, flux, r, kwargs: (G, flux, np.full(3, np.inf), kwargs), ValueError, "finite"),
        (
            lambda G, flux, r, kwargs: (G, flux, r, {**kwargs, "relative_prior_sd": 0.0}),
            ValueError,
            "positive",
        ),
        (
            lambda G, flux, r, kwargs: (G, flux, r, {**kwargs, "relative_prior_sd": np.inf}),
            ValueError,
            "finite",
        ),
        (
            lambda G, flux, r, kwargs: (G, flux, r, {**kwargs, "flux_tolerance": -1.0}),
            ValueError,
            "non-negative",
        ),
        (
            lambda G, flux, r, kwargs: (G, flux, r, {**kwargs, "flux_tolerance": np.nan}),
            ValueError,
            "finite",
        ),
        (
            lambda G, flux, r, kwargs: (G, flux, r, {**kwargs, "coarsen_factor": 0}),
            ValueError,
            "positive",
        ),
        (
            lambda G, flux, r, kwargs: (G, flux, r, {**kwargs, "coarsen_factor": 1.5}),
            TypeError,
            "integer",
        ),
        (
            lambda G, flux, r, kwargs: (np.where(G == G.flat[0], np.nan, G), flux, r, kwargs),
            ValueError,
            "finite",
        ),
    ],
)
def test_invalid_dimensions_values_and_hyperparameters_are_rejected(
    mutate: Callable[
        [np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
    ],
    error: type[Exception],
    message: str,
) -> None:
    """Invalid dimensions, finite values, and scalar parameters should fail clearly."""
    G, prior_flux, r_diag, _ = _native_example()
    args = mutate(G.copy(), prior_flux.copy(), r_diag.copy(), {"coarsen_factor": 1})

    with pytest.raises(error, match=message):
        RHIMEGaussianMultiscale.from_native_grid(*args[:3], **args[3])


def test_nonzero_design_outside_flux_support_is_rejected() -> None:
    """Material design contributions at excluded native cells should be rejected."""
    G, prior_flux, r_diag, _ = _native_example()
    prior_flux[0, 0] = 0.0

    with pytest.raises(ValueError, match="approximately zero"):
        RHIMEGaussianMultiscale.from_native_grid(G, prior_flux, r_diag, coarsen_factor=1)


def test_state_and_split_validation_errors_are_preserved() -> None:
    """Public state-dependent methods should reject invalid frontiers and cell splits."""
    model = _model()
    tree = model.design.tree

    with pytest.raises(ValueError, match="at least one"):
        model.score(PartitionState(frozenset()))
    with pytest.raises(ValueError, match="cannot be split"):
        model.split_gain(tree.leaf_ids[0])
    with pytest.raises(KeyError, match="Unknown"):
        model.split_gain(999)


@pytest.mark.parametrize(
    ("innovations", "message"),
    [
        ([1.0, 2.0], "one value per observation"),
        ([1.0, np.nan, 2.0], "finite"),
        ([[1.0, 2.0, 3.0]], "one value per observation"),
    ],
)
def test_innovation_dependent_methods_reject_invalid_vectors(
    innovations: npt.ArrayLike,
    message: str,
) -> None:
    """Observation-dependent scores and posterior summaries should validate input."""
    model = _model()

    with pytest.raises(ValueError, match=message):
        model.data_dependent_tile_scores(innovations)
    with pytest.raises(ValueError, match=message):
        model.native_posterior_marginals(innovations)


@pytest.mark.parametrize(
    ("chunk_size", "error", "message"),
    [
        (0, ValueError, "positive"),
        (1.5, TypeError, "integer"),
    ],
)
def test_native_posterior_rejects_invalid_chunk_sizes(
    chunk_size: Any,
    error: type[Exception],
    message: str,
) -> None:
    """Native posterior chunk sizes should be positive integers."""
    with pytest.raises(error, match=message):
        _model().native_posterior_marginals([0.1, 0.2, 0.3], chunk_size=chunk_size)
