r"""Exact linear-Gaussian posteriors by Paige's generalised least squares.

For observations :math:`y = H x + e`, :math:`e \sim N(0, W)`, and a Gaussian
prior :math:`x \sim N(m, B)`, the posterior is Gaussian. This module computes
its mean and a covariance factor without inverting a covariance, a covariance
factor or forming the normal equations. The prior enters as extra rows,

.. math::

   \begin{bmatrix} H \\ I \end{bmatrix} x
   = \begin{bmatrix} y \\ m \end{bmatrix}
   + \begin{bmatrix} F_W & 0 \\ 0 & L_B \end{bmatrix} v,
   \qquad v \sim N(0, I),

where :math:`W = F_W F_W^T` and :math:`B = L_B L_B^T`. Paige's generalised QR
(Paige 1979; Golub and Van Loan, *Matrix Computations*, 4th ed., §6.1.2) then
solves the minimum-norm problem: a QR factorisation of the stacked design
:math:`A = Q [R_1; 0]` splits :math:`Q^T F` into :math:`F_1` and
:math:`F_2`, and a rank-revealing RQ factorisation :math:`F_2 Z = [S_2, 0]`
separates the noise directions fixed by the data from the free directions
:math:`Z_1`. The posterior mean solves :math:`R_1 \hat x = c_1 - F_1 Z_2 w`
and the posterior covariance is :math:`P P^T` with
:math:`P = R_1^{-1} F_1 Z_1`.

The system is first equilibrated by diagonal scalings: the state is written
as :math:`x = m + D z` with :math:`D` the row norms of :math:`L_B`, and each
observation row is divided by the row norm of :math:`F_W`. This balances
prior and noise scales of any relative size without inverting a factor.

Noise and prior factors may be rectangular or rank deficient. Each is first
compressed to its numerical column space by a pivoted QR factorisation of its
transpose, which yields a triangular factor of the same covariance without
forming it. The rank-revealing RQ factorisation is computed as a pivoted QR
factorisation of :math:`F_2^T`. Orthogonal factors stay implicit as LAPACK
Householder reflectors.

This formulation stays accurate when :math:`W` is very ill conditioned, for
example a small diagonal mismatch added to a rank-deficient aggregation
covariance, where whitening by :math:`W^{-1/2}` or a Cholesky precision form
loses digits in proportion to the condition number of :math:`W`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import lapack as _lapack, solve_triangular

# The LAPACK wrappers are generated at build time and are untyped.
lapack: Any = _lapack
_EPS = float(np.finfo(np.float64).eps)
_TINY = float(np.finfo(np.float64).tiny)


@dataclass(frozen=True)
class GlsDiagnostics:
    r"""Numerical ranks, condition estimates and residuals of one solve.

    Condition estimates are LAPACK ``trcon`` reciprocal 1-norm estimates of
    triangular factors of the equilibrated system (prior rows and noise rows
    scaled to unit factor norm); ``nan`` marks an empty factor.

    Attributes:
        n_observations: Number of observation rows.
        n_state: Number of state columns.
        noise_factor_columns: Columns of the supplied noise factor.
        noise_rank: Numerical rank of the noise factor, which is the rank of
            the noise covariance used.
        prior_factor_columns: Columns of the supplied prior factor.
        prior_rank: Numerical rank of the prior factor used.
        constraint_rank: Numerical rank of :math:`Q_2^T F`, the order of
            :math:`S_2`.
        rank_rtol: Relative tolerance used for every numerical rank.
        rcond_r1: Reciprocal condition estimate of :math:`R_1`.
        rcond_s2: Reciprocal condition estimate of :math:`S_2`.
        rcond_noise_factor: Reciprocal condition estimate of the compressed
            triangular noise factor, restricted to its numerical rank.
        noise_covariance_condition_estimate: ``rcond_noise_factor ** -2``, an
            estimate of the condition number of the noise covariance on its
            numerical range.
        generalized_residual_norm: :math:`\|\hat v\|`, the square root of the
            minimised objective.
        observation_residual_norm: Euclidean norm of ``y - design @ mean``.
        prior_residual_norm: Euclidean norm of ``mean - prior_mean``.
        constraint_residual_norm: Norm of the data component that the noise
            cannot reach; zero up to rounding for a consistent problem.
    """

    n_observations: int
    n_state: int
    noise_factor_columns: int
    noise_rank: int
    prior_factor_columns: int
    prior_rank: int
    constraint_rank: int
    rank_rtol: float
    rcond_r1: float
    rcond_s2: float
    rcond_noise_factor: float
    noise_covariance_condition_estimate: float
    generalized_residual_norm: float
    observation_residual_norm: float
    prior_residual_norm: float
    constraint_residual_norm: float

    def to_dict(self) -> dict[str, int | float]:
        """Return the diagnostics as a JSON-safe mapping."""
        return {name: (int(value) if isinstance(value, int) else float(value)) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class GaussianPosterior:
    """Exact Gaussian posterior moments returned by :func:`paige_gls`.

    Attributes:
        mean: Posterior mean, shape ``(n,)``.
        cov_factor: Square factor ``P`` with ``covariance = P @ P.T``, shape
            ``(n, n)``.
        diagnostics: Ranks, condition estimates and residual norms.
    """

    mean: NDArray[np.float64]
    cov_factor: NDArray[np.float64]
    diagnostics: GlsDiagnostics

    @property
    def covariance(self) -> NDArray[np.float64]:
        """Return the dense posterior covariance ``P @ P.T``."""
        return self.cov_factor @ self.cov_factor.T

    def draw(self, draws: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Return independent exact posterior draws.

        Args:
            draws: Number of draws.
            rng: Generator consumed for ``(draws, n)`` standard normals.

        Returns:
            Array of shape ``(draws, n)``.
        """
        standard = rng.standard_normal((draws, self.cov_factor.shape[1]))
        return self.mean[None, :] + standard @ self.cov_factor.T


@dataclass(frozen=True)
class PsdFactor:
    """A factor of a symmetric positive-semidefinite matrix and its repairs.

    Attributes:
        factor: Matrix ``L`` with ``L @ L.T`` equal to the clipped matrix; its
            columns are the retained eigenvectors scaled by root eigenvalues.
        diagnostics: JSON-safe eigenvalue clipping record: ``method``,
            ``size``, ``rank``, ``max_eigenvalue``, ``min_eigenvalue``,
            ``negative_count``, ``negative_clipped_max`` (largest magnitude
            of a negative eigenvalue set to zero), ``tolerance`` and
            ``dropped_positive_max`` (largest positive eigenvalue at or below
            the tolerance that was omitted), plus ``symmetry_max_abs``.
    """

    factor: NDArray[np.float64]
    diagnostics: dict[str, Any]


def psd_factor(matrix: ArrayLike, *, rtol: float = 0.0) -> PsdFactor:
    """Factor a symmetric positive-semidefinite matrix by clipped eigenvalues.

    The symmetric part is decomposed with ``numpy.linalg.eigh``. Negative
    eigenvalues are clipped to zero and eigenvalues at or below
    ``rtol * max_eigenvalue`` are omitted, so the factor has one column per
    retained mode. All repairs are recorded rather than hidden.

    The default keeps every positive eigenvalue. Do not drop small positive
    modes of a noise covariance: a mode below a relative tolerance can still
    exceed a small diagonal mismatch added to it, and omitting it would
    replace ``lambda + R`` by ``R`` in that direction. :func:`paige_gls`
    determines numerical rank itself.

    Args:
        matrix: Square symmetric matrix.
        rtol: Relative eigenvalue tolerance; the default ``0`` clips only
            negative eigenvalues.

    Returns:
        The factor and its clipping diagnostics.

    Raises:
        ValueError: If ``matrix`` is not square and finite, or not symmetric
            to a scale-relative tolerance of ``1e-10``.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"A PSD factor requires a square matrix; got shape {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("A PSD factor requires finite values.")
    size = values.shape[0]
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    asymmetry = float(np.max(np.abs(values - values.T))) if values.size else 0.0
    if asymmetry > 1e-10 * scale:
        raise ValueError("A PSD factor requires a symmetric matrix.")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (values + values.T))
    largest = float(eigenvalues[-1]) if size else 0.0
    tolerance = float(rtol) * max(largest, 0.0)
    keep = eigenvalues > tolerance
    negative = eigenvalues < 0
    dropped_positive = (~keep) & ~negative
    factor = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
    return PsdFactor(
        factor=factor[:, ::-1].copy(),
        diagnostics={
            "method": "eigh_clip",
            "size": size,
            "rank": int(keep.sum()),
            "max_eigenvalue": largest,
            "min_eigenvalue": float(eigenvalues[0]) if size else 0.0,
            "negative_count": int(negative.sum()),
            "negative_clipped_max": float(-eigenvalues[negative].min()) if negative.any() else 0.0,
            "negative_clipped_sum": float(-eigenvalues[negative].sum()),
            "tolerance": tolerance,
            "dropped_positive_max": float(eigenvalues[dropped_positive].max()) if dropped_positive.any() else 0.0,
            "symmetry_max_abs": asymmetry,
        },
    )


def _workspace(query: tuple[Any, ...]) -> int:
    """Return the optimal LAPACK workspace from a ``lwork=-1`` query result."""
    return max(1, int(np.real(query[-2][0] if np.ndim(query[-2]) else query[-2])))


def _apply_qt(reflectors: NDArray[np.float64], tau: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply ``Q.T`` from Householder ``reflectors`` to ``matrix`` from the left."""
    if matrix.shape[1] == 0 or tau.size == 0:
        return np.array(matrix, dtype=np.float64, order="F", copy=True)
    reflectors = reflectors[:, : tau.size]
    query = lapack.dormqr("L", "T", reflectors, tau, matrix, -1)
    result, _, info = lapack.dormqr("L", "T", reflectors, tau, matrix, _workspace(query))
    if info != 0:
        raise RuntimeError(f"LAPACK dormqr failed with info={info}.")
    return np.asarray(result)


def _pivoted_qr(matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.float64]]:
    """Return LAPACK ``geqp3`` reflectors, zero-based pivots and ``tau`` of a copy."""
    work = np.array(matrix, dtype=np.float64, order="F", copy=True)
    query = lapack.dgeqp3(work, -1)
    reflectors, pivots, tau, _, info = lapack.dgeqp3(work, _workspace(query), overwrite_a=True)
    if info != 0:
        raise RuntimeError(f"LAPACK dgeqp3 failed with info={info}.")
    return reflectors, np.asarray(pivots, dtype=np.intp) - 1, tau


def _numerical_rank(triangular: NDArray[np.float64], rtol: float) -> int:
    """Count pivoted-QR diagonal entries above ``rtol`` times the first."""
    diagonal = np.abs(np.diag(triangular))
    if diagonal.size == 0 or diagonal[0] == 0.0:
        return 0
    return int(np.count_nonzero(diagonal > rtol * diagonal[0]))


def _rcond(triangular: NDArray[np.float64]) -> float:
    """Return LAPACK's reciprocal 1-norm condition estimate of an upper-triangular matrix."""
    if triangular.size == 0:
        return float("nan")
    rcond, info = lapack.dtrcon(np.asfortranarray(triangular), norm="1", uplo="U", diag="N")
    if info != 0:
        raise RuntimeError(f"LAPACK dtrcon failed with info={info}.")
    return float(rcond)


def _compress_factor(
    factor: NDArray[np.float64], rtol: float,
) -> tuple[NDArray[np.float64], int, float]:
    """Return a numerically full-column-rank factor of the same covariance.

    ``factor.T[:, piv] = Z T`` gives ``factor[piv] = T.T Z.T`` and hence
    ``factor @ factor.T`` equals ``C @ C.T`` for ``C[piv] = T[:rank].T``.
    """
    rows, columns = factor.shape
    if columns == 0 or rows == 0:
        return np.zeros((rows, 0)), 0, float("nan")
    reflectors, pivots, _ = _pivoted_qr(factor.T)
    width = min(rows, columns)
    triangular = np.triu(reflectors[:width, :])
    rank = _numerical_rank(triangular, rtol)
    compressed = np.zeros((rows, rank))
    compressed[pivots] = triangular[:rank].T
    return compressed, rank, _rcond(triangular[:rank, :rank])


def paige_gls(
    y: ArrayLike,
    design: ArrayLike,
    noise_factor: ArrayLike,
    prior_mean: ArrayLike,
    prior_factor: ArrayLike,
    *,
    rank_rtol: float | None = None,
    consistency_rtol: float = 1e-6,
) -> GaussianPosterior:
    r"""Return the exact posterior of a linear-Gaussian model by Paige's method.

    The model is :math:`y = H x + F_W v_W` and :math:`x = m + L_B v_B`, with
    independent standard-normal :math:`v_W` and :math:`v_B`. Neither factor
    is inverted and no covariance or normal-equation matrix is formed.

    Args:
        y: Observations, shape ``(nobs,)``.
        design: Design matrix ``H``, shape ``(nobs, n)``.
        noise_factor: ``F_W`` with ``W = F_W F_W^T``, shape ``(nobs, k)``;
            any ``k`` including zero, and rank deficient factors are accepted.
        prior_mean: Prior mean ``m``, shape ``(n,)``.
        prior_factor: ``L_B`` with ``B = L_B L_B^T``, shape ``(n, q)``;
            rectangular or rank deficient factors are accepted.
        rank_rtol: Relative tolerance on pivoted-QR diagonals for every
            numerical rank. Defaults to ``max(shape) * eps`` of each
            factorised matrix.
        consistency_rtol: Largest accepted norm of the data component
            unreachable by the noise, relative to the projected data norm.

    Returns:
        Posterior mean, square covariance factor and diagnostics.

    Raises:
        ValueError: If shapes disagree, values are not finite, or the data
            are inconsistent with a rank-deficient noise model.
    """
    observations = np.asarray(y, dtype=np.float64)
    operator = np.asarray(design, dtype=np.float64)
    noise = np.asarray(noise_factor, dtype=np.float64)
    mean0 = np.asarray(prior_mean, dtype=np.float64)
    prior = np.asarray(prior_factor, dtype=np.float64)
    if operator.ndim != 2:
        raise ValueError(f"design must be two-dimensional; got shape {operator.shape}.")
    nobs, n = operator.shape
    if observations.shape != (nobs,):
        raise ValueError(f"y must have shape ({nobs},); got {observations.shape}.")
    if noise.ndim != 2 or noise.shape[0] != nobs:
        raise ValueError(f"noise_factor must have shape ({nobs}, k); got {noise.shape}.")
    if mean0.shape != (n,):
        raise ValueError(f"prior_mean must have shape ({n},); got {mean0.shape}.")
    if prior.ndim != 2 or prior.shape[0] != n:
        raise ValueError(f"prior_factor must have shape ({n}, q); got {prior.shape}.")
    for name, value in (("y", observations), ("design", operator), ("noise_factor", noise),
                        ("prior_mean", mean0), ("prior_factor", prior)):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values.")

    def tolerance(*shape: int) -> float:
        return float(rank_rtol) if rank_rtol is not None else max(shape + (1,)) * _EPS

    # Equilibrate: x = m + D z with D the prior row scales, and divide each
    # observation row by its noise scale. Both are diagonal, so no factor is
    # inverted, and prior and noise scales of any relative size are balanced.
    prior_scale = np.linalg.norm(prior, axis=1)
    prior_scale[prior_scale == 0] = 1.0
    noise_scale = np.linalg.norm(noise, axis=1)
    noise_scale[noise_scale == 0] = 1.0
    scaled_operator = operator * prior_scale[None, :] / noise_scale[:, None]
    innovation = (observations - operator @ mean0) / noise_scale
    noise_c, noise_rank, rcond_noise = _compress_factor(noise / noise_scale[:, None], tolerance(*noise.shape))
    prior_c, prior_rank, _ = _compress_factor(prior / prior_scale[:, None], tolerance(*prior.shape))

    # Stack the prior as pseudo-observations: A z = b + F v.
    rows = nobs + n
    stacked = np.zeros((rows, n), order="F")
    stacked[:nobs] = scaled_operator
    stacked[nobs:] = np.eye(n)
    rhs = np.concatenate([innovation, np.zeros(n)])[:, None]
    factor = np.zeros((rows, noise_rank + prior_rank), order="F")
    factor[:nobs, :noise_rank] = noise_c
    factor[nobs:, noise_rank:] = prior_c
    del noise_c, prior_c

    # QR of the stacked design; [I] rows make it full column rank.
    query = lapack.dgeqrf(stacked, -1)
    reflectors, tau, _, info = lapack.dgeqrf(stacked, _workspace(query), overwrite_a=True)
    if info != 0:
        raise RuntimeError(f"LAPACK dgeqrf failed with info={info}.")
    r1 = np.triu(reflectors[:n, :n])
    rotated = _apply_qt(reflectors, tau, factor)
    del factor
    projected = _apply_qt(reflectors, tau, np.asfortranarray(rhs))[:, 0]
    f1, f2 = rotated[:n], rotated[n:]
    c1, c2 = projected[:n], projected[n:]

    # Rank-revealing RQ of F2 as a pivoted QR of F2.T: F2.T[:, piv] = Z [T; 0].
    columns = f2.shape[1]
    if nobs and columns:
        z_reflectors, pivots, z_tau = _pivoted_qr(f2.T)
        triangular = np.triu(z_reflectors[: min(columns, nobs), :])
        constraint_rank = _numerical_rank(triangular, tolerance(columns, nobs))
        d = c2[pivots]
        t11 = triangular[:constraint_rank, :constraint_rank]
        w1 = solve_triangular(t11, d[:constraint_rank], trans="T", lower=False, check_finite=False)
        unreachable = d[constraint_rank:] - triangular[:constraint_rank, constraint_rank:].T @ w1
        rotated_f1 = _apply_qt(z_reflectors, z_tau, np.asfortranarray(f1.T))
    else:
        constraint_rank = 0
        t11 = np.zeros((0, 0))
        w1 = np.zeros(0)
        unreachable = c2.copy()
        rotated_f1 = np.asfortranarray(f1.T)
    constraint_residual = float(np.linalg.norm(unreachable))
    if constraint_residual > consistency_rtol * max(float(np.linalg.norm(projected)), _TINY):
        raise ValueError(
            "The observations are inconsistent with the rank-deficient noise model: "
            f"an unreachable data component of norm {constraint_residual:.3e} remains."
        )

    scaled_mean = solve_triangular(r1, c1 - rotated_f1[:constraint_rank].T @ w1, lower=False, check_finite=False)
    mean = mean0 + prior_scale * scaled_mean
    free = solve_triangular(r1, rotated_f1[constraint_rank:].T, lower=False, check_finite=False)
    free *= prior_scale[:, None]
    if free.shape[1] > n:
        # Compress P to a square factor with the same P P^T.
        upper = cast(NDArray[np.float64], np.linalg.qr(free.T, mode="r"))
        cov_factor = np.ascontiguousarray(upper.T)
    else:
        cov_factor = np.zeros((n, n))
        cov_factor[:, : free.shape[1]] = free

    diagnostics = GlsDiagnostics(
        n_observations=nobs,
        n_state=n,
        noise_factor_columns=noise.shape[1],
        noise_rank=noise_rank,
        prior_factor_columns=prior.shape[1],
        prior_rank=prior_rank,
        constraint_rank=constraint_rank,
        rank_rtol=tolerance(columns, nobs),
        rcond_r1=_rcond(r1),
        rcond_s2=_rcond(t11),
        rcond_noise_factor=rcond_noise,
        noise_covariance_condition_estimate=(
            rcond_noise ** -2.0 if rcond_noise > 0 else float("inf") if rcond_noise == 0 else float("nan")
        ),
        generalized_residual_norm=float(np.linalg.norm(w1)),
        observation_residual_norm=float(np.linalg.norm(observations - operator @ mean)),
        prior_residual_norm=float(np.linalg.norm(mean - mean0)),
        constraint_residual_norm=constraint_residual,
    )
    return GaussianPosterior(mean=mean, cov_factor=cov_factor, diagnostics=diagnostics)


__all__ = ["GaussianPosterior", "GlsDiagnostics", "PsdFactor", "paige_gls", "psd_factor"]
