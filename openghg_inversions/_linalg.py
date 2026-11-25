"""Small dense linear-algebra helpers shared by covariance code."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor

# scipy-stubs 1.15.3 omits the runtime LAPACK wrappers, including dpocon.
from scipy.linalg.lapack import dpocon  # type: ignore[attr-defined]


def cholesky_with_condition(matrix: np.ndarray) -> tuple[tuple[np.ndarray, bool], float]:
    """Cholesky-factor a symmetric matrix and estimate its reciprocal condition.

    A successful Cholesky factorization does not prove that a matrix is
    numerically nonsingular: some LAPACK builds, such as Apple Accelerate,
    accept an exactly singular matrix with a tiny positive last pivot. The
    LAPACK ``dpocon`` estimate reuses the factor, so callers can reject such a
    matrix without an eigendecomposition. The estimate is not a bound: do not
    use it in place of an eigenvalue in an inequality that must hold. See
    ``docs/development/validation_and_xarray.rst``.

    Args:
        matrix: Real symmetric matrix expected to be positive definite.

    Returns:
        The lower Cholesky factor in :func:`scipy.linalg.cho_factor` form,
        and the estimated reciprocal 1-norm condition number.

    Raises:
        numpy.linalg.LinAlgError: If the factorization or condition estimate
            fails.
    """
    factor = cho_factor(matrix, lower=True, check_finite=True)
    reciprocal_condition, condition_info = dpocon(factor[0], np.linalg.norm(matrix, ord=1), uplo="L")
    if condition_info != 0:
        raise np.linalg.LinAlgError(f"LAPACK condition estimation failed with info={condition_info}")
    return factor, float(reciprocal_condition)
