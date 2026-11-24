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
    return factor, _reciprocal_condition(factor[0], matrix)


def conditioned_cholesky(matrix: np.ndarray, minimum_reciprocal_condition: float) -> np.ndarray | None:
    """Return a lower Cholesky factor only if the matrix is safely invertible.

    This is the non-raising form of :func:`cholesky_with_condition` for
    likelihood kernels that fall back to a slower, more stable route. The
    factor is accepted only when the matrix is finite, the factorization
    succeeds, and the reciprocal condition estimate from that factor strictly
    exceeds ``minimum_reciprocal_condition``. Choose the threshold for the
    calculation that consumes the factor.

    Args:
        matrix: Real symmetric matrix expected to be positive definite.
        minimum_reciprocal_condition: Largest rejected reciprocal 1-norm
            condition estimate.

    Returns:
        The lower-triangular factor ``L`` with zero upper triangle and
        ``L @ L.T == matrix``, or ``None`` when any screen fails.
    """
    if not np.isfinite(matrix).all():
        return None
    try:
        cholesky = np.linalg.cholesky(matrix)
        reciprocal_condition = _reciprocal_condition(cholesky, matrix)
    except np.linalg.LinAlgError:
        return None
    return cholesky if reciprocal_condition > minimum_reciprocal_condition else None


def cholesky_logdet(cholesky: np.ndarray) -> float:
    """Return ``log det(L @ L.T)`` from a triangular Cholesky factor ``L``.

    Args:
        cholesky: Lower or upper triangular factor; only its diagonal is read.

    Returns:
        Twice the sum of the logarithms of the factor's diagonal.
    """
    return float(2.0 * np.log(np.diag(cholesky)).sum())


def _reciprocal_condition(cholesky: np.ndarray, matrix: np.ndarray) -> float:
    """Return the LAPACK ``dpocon`` estimate for ``matrix`` from its lower factor."""
    reciprocal_condition, condition_info = dpocon(cholesky, np.linalg.norm(matrix, ord=1), uplo="L")
    if condition_info != 0:
        raise np.linalg.LinAlgError(f"LAPACK condition estimation failed with info={condition_info}")
    return float(reciprocal_condition)
