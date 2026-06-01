import numpy as np
import pytest

from openghg_inversions.basis.algorithms import _experimental_dyadic as dyadic


def test_dyadic_weight_array_precomputes_multiscale_sums():
    """The prototype-style weight array should contain every dyadic tile sum."""
    weights = np.arange(1.0, 17.0).reshape(4, 4)

    dyadic_weights = dyadic.make_dyadic_weight_array(weights)

    assert dyadic_weights.values.shape == (7, 7)
    assert dyadic_weights.values[0, 0] == weights[0, 0]
    assert dyadic_weights.weight(dyadic.Tile(0, 2, 0, 2)) == weights[:2, :2].sum()
    assert dyadic_weights.weight(dyadic.Tile(0, 4, 0, 4)) == weights.sum()


def test_dyadic_threshold_basis_covers_non_square_grid():
    """Threshold bisection should return compact labels for the original grid."""
    weights = np.array(
        [
            [9.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 8.0, 0.0],
        ]
    )

    labels = dyadic.dyadic_threshold_basis(weights, threshold=5.0)

    assert labels.shape == weights.shape
    assert labels.min() == 1
    assert labels.max() > 1
    assert set(np.unique(labels)) == set(range(1, int(labels.max()) + 1))


def test_dyadic_target_basis_gets_exact_power_of_two_case():
    """The threshold search can hit simple dyadic target counts exactly."""
    weights = np.ones((4, 4))

    labels = dyadic.dyadic_target_basis(weights, target_regions=4)

    assert labels.shape == weights.shape
    assert labels.max() == 4


def test_numba_threshold_basis_matches_python():
    """The numba bisection kernel should match the Python reference path."""
    if not dyadic._HAS_NUMBA:
        pytest.skip("numba is not installed")

    weights = np.arange(1.0, 17.0).reshape(4, 4)

    python_labels = dyadic.dyadic_threshold_basis(weights, threshold=20.0)
    numba_labels = dyadic.dyadic_threshold_basis(weights, threshold=20.0, use_numba=True)

    np.testing.assert_array_equal(numba_labels, python_labels)


def test_anneal_dyadic_basis_runs_and_keeps_best_energy():
    """The local-search refinement should run deterministically with a seed."""
    weights = np.array(
        [
            [9.0, 8.0, 1.0, 0.0],
            [7.0, 6.0, 1.0, 0.0],
            [0.0, 1.0, 5.0, 4.0],
            [0.0, 1.0, 4.0, 5.0],
        ]
    )

    result = dyadic.anneal_dyadic_basis(
        weights,
        initial_threshold=8.0,
        target_regions=5,
        iterations=30,
        temperature=0.2,
        region_penalty=0.5,
        seed=123,
    )

    assert result.labels.shape == weights.shape
    assert result.labels.min() == 1
    assert result.final_regions == int(result.labels.max())
    assert result.accepted_moves >= 0
    assert result.best_energy <= result.initial_energy
