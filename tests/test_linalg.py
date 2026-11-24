"""Shared dense factor, condition and log-determinant helpers."""

import numpy as np
import pytest

from openghg_inversions._linalg import cholesky_logdet, conditioned_cholesky

THRESHOLD = float(np.sqrt(np.finfo(np.float64).eps))


def test_conditioned_cholesky_returns_the_lower_factor_and_its_logdet() -> None:
    """A well-conditioned matrix gives NumPy's lower factor and the exact log-determinant."""
    rng = np.random.default_rng(0)
    root = rng.standard_normal((4, 4))
    matrix = root @ root.T + 4.0 * np.eye(4)

    factor = conditioned_cholesky(matrix, THRESHOLD)

    assert factor is not None
    np.testing.assert_array_equal(factor, np.linalg.cholesky(matrix))
    assert cholesky_logdet(factor) == pytest.approx(np.linalg.slogdet(matrix)[1], rel=1e-12)


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param(np.diag([1.0, 1e-12]), id="factors-but-ill-conditioned"),
        pytest.param(np.array([[1.0, 2.0], [2.0, 1.0]]), id="indefinite"),
        pytest.param(np.array([[1.0, np.nan], [np.nan, 1.0]]), id="non-finite"),
    ],
)
def test_conditioned_cholesky_rejects_unsafe_matrices(matrix: np.ndarray) -> None:
    """A factorization that succeeds is still rejected when its condition estimate is too small."""
    assert conditioned_cholesky(matrix, THRESHOLD) is None
