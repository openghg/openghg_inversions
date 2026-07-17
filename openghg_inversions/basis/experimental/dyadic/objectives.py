"""Gaussian design objectives and explicitly labelled prototype tile scores.

The principal score is linear-Gaussian degrees of freedom for signal (DFS).
It is evaluated with state-space Cholesky solves and diagonal observation-error
covariance.  Partition covariance is never cached by the objective: callers
provide a builder that is invoked for each state/design evaluation.  For a
Bocquet-consistent construction that builder should return ``B_P = P B P.T``.
The included isotropic builder is only a proof-of-concept benchmark.

The historical quadratic tile score is retained separately.  It aggregates
fine-cell observation contributions before squaring, but it is a proxy and not
exact DFS unless additional covariance, prolongation, and normalization
assumptions are established.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .multiscale import MultiscaleDesign
from .state import PartitionState

CovarianceBuilder = Callable[[PartitionState, MultiscaleDesign], npt.ArrayLike]


@dataclass(frozen=True)
class IsotropicRegionCovariance:
    """Build isotropic region covariance as a proof-of-concept benchmark.

    This benchmark returns ``tau**2 * I`` for each active partition.  It is not
    the Bocquet-consistent aggregate ``B_P = P B P.T`` and should not be labelled
    as such.

    Args:
        tau: Positive finite prior standard deviation for every active regional
            coefficient.

    Raises:
        ValueError: If ``tau`` is not positive and finite.
    """

    tau: float

    def __post_init__(self) -> None:
        """Validate the benchmark prior scale."""
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be positive and finite.")

    def __call__(self, state: PartitionState, design: MultiscaleDesign) -> np.ndarray:
        """Return isotropic covariance for the state's active regions.

        Args:
            state: Partition whose active-region count sets the matrix size.
            design: Associated design, accepted to satisfy the covariance-builder
                interface and intentionally unused by this benchmark.

        Returns:
            Positive-definite covariance with shape ``(K, K)``.
        """
        del design
        active_count = len(state.ordered_active())
        return self.tau**2 * np.eye(active_count)


@dataclass(frozen=True)
class GaussianDFSObjective:
    """Evaluate Gaussian DFS with covariance rebuilt for every partition.

    Args:
        r_diag: Positive diagonal of the observation-error covariance ``R``.
        covariance_builder: Callable receiving ``(state, design)`` and returning
            the active covariance ``B_P``.  A Bocquet-consistent implementation
            should construct ``B_P = P B P.T`` for that partition; it must not
            assume one fixed ``B`` applies to every ``P``.

    Raises:
        ValueError: If ``r_diag`` is not a positive finite one-dimensional
            array.
    """

    r_diag: npt.ArrayLike
    covariance_builder: CovarianceBuilder

    def __post_init__(self) -> None:
        """Validate and freeze the observation covariance diagonal."""
        r_diag = _positive_vector(self.r_diag, name="r_diag")
        r_diag.setflags(write=False)
        object.__setattr__(self, "r_diag", r_diag)
        if not callable(self.covariance_builder):
            raise TypeError("covariance_builder must be callable.")

    def score(self, state: PartitionState, design: MultiscaleDesign) -> float:
        """Evaluate DFS for one partition/design pair.

        Args:
            state: Active partition to score.
            design: Multiscale design from which active columns are gathered.

        Returns:
            Gaussian degrees of freedom for signal.

        Raises:
            ValueError: If the gathered design, covariance, or observation
                covariance has incompatible or invalid values.
        """
        design_matrix = design.gather(state)
        covariance = self.covariance_builder(state, design)
        return gaussian_dfs(design_matrix, covariance, self.r_diag)

    def __call__(self, state: PartitionState, design: MultiscaleDesign) -> float:
        """Delegate callable objective evaluation to :meth:`score`."""
        return self.score(state, design)


def gaussian_dfs(H: npt.ArrayLike, B: npt.ArrayLike, r_diag: npt.ArrayLike) -> float:
    """Compute Gaussian DFS with stable coefficient-space Cholesky solves.

    The evaluated identity is
    ``K - tr(A^-1 B^-1)``, where
    ``A = B^-1 + H.T R^-1 H`` and ``R = diag(r_diag)``.  Cholesky whitening by
    ``B`` avoids explicitly forming either matrix inverse.

    Args:
        H: Observation-by-state design matrix with shape ``(N, K)``.
        B: Symmetric positive-definite state covariance with shape ``(K, K)``.
        r_diag: Positive observation covariance diagonal with shape ``(N,)``.

    Returns:
        Gaussian degrees of freedom for signal as a scalar.

    Raises:
        ValueError: If dimensions are incompatible, inputs are non-finite,
            ``r_diag`` is not positive, or ``B`` is not symmetric positive
            definite.
    """
    design = _finite_float_array(H, name="H")
    covariance = _finite_float_array(B, name="B")
    errors = _positive_vector(r_diag, name="r_diag")
    _validate_gaussian_shapes(design, covariance, errors)

    covariance_cholesky = _positive_definite_cholesky(covariance, name="B")
    whitened_design = design @ covariance_cholesky
    information = np.eye(design.shape[1]) + (whitened_design.T / errors[np.newaxis, :]) @ whitened_design
    information_cholesky = _positive_definite_cholesky(information, name="state information")
    inverse_information = _cholesky_solve(information_cholesky, np.eye(design.shape[1]))
    return float(design.shape[1] - np.trace(inverse_information))


def direct_observation_space_dfs(
    H: npt.ArrayLike,
    B: npt.ArrayLike,
    r_diag: npt.ArrayLike,
) -> float:
    """Compute DFS directly in observation space for small parity tests.

    This independent helper evaluates
    ``tr((H B H.T + R)^-1 H B H.T)``.  Its ``N``-by-``N`` factorization is useful
    as a test oracle but is not the preferred implementation when observations
    outnumber active state dimensions.

    Args:
        H: Observation-by-state design matrix with shape ``(N, K)``.
        B: Symmetric positive-definite state covariance with shape ``(K, K)``.
        r_diag: Positive observation covariance diagonal with shape ``(N,)``.

    Returns:
        Gaussian degrees of freedom for signal as a scalar.

    Raises:
        ValueError: If dimensions or numerical validity requirements are not
            satisfied.
    """
    design = _finite_float_array(H, name="H")
    covariance = _finite_float_array(B, name="B")
    errors = _positive_vector(r_diag, name="r_diag")
    _validate_gaussian_shapes(design, covariance, errors)
    _positive_definite_cholesky(covariance, name="B")

    projected_covariance = design @ covariance @ design.T
    innovation = projected_covariance + np.diag(errors)
    innovation_cholesky = _positive_definite_cholesky(innovation, name="innovation covariance")
    averaging_kernel = _cholesky_solve(innovation_cholesky, projected_covariance)
    return float(np.trace(averaging_kernel))


def prototype_quadratic_tile_scores(
    H: npt.ArrayLike,
    observation_precision: npt.ArrayLike,
    support_normalization: npt.ArrayLike,
) -> np.ndarray:
    """Compute the historical sum-then-square tile-score proxy.

    For already aggregated candidate columns ``h_iv``, this evaluates
    ``score_v = sum_i precision_i * h_iv**2 / support_v``.  ``H`` must therefore
    be formed by summing fine-cell contributions before this function is called.
    The result is explicitly a prototype quadratic proxy, not exact DFS.

    Args:
        H: Aggregated observation-by-tile columns with shape ``(N, V)``.
        observation_precision: Explicit positive values ``1 / variance_i`` with
            shape ``(N,)``.
        support_normalization: Explicit positive physical support or area for
            each tile, with shape ``(V,)``.

    Returns:
        One non-negative proxy score per candidate tile.

    Raises:
        ValueError: If inputs are non-finite, have incompatible dimensions, or
            precision/support values are not positive.
    """
    design = _finite_float_array(H, name="H")
    precision = _positive_vector(observation_precision, name="observation_precision")
    support = _positive_vector(support_normalization, name="support_normalization")
    if design.ndim != 2:
        raise ValueError("H must be a two-dimensional observation-by-tile matrix.")
    if precision.shape[0] != design.shape[0]:
        raise ValueError("observation_precision length must match H observations.")
    if support.shape[0] != design.shape[1]:
        raise ValueError("support_normalization length must match H tiles.")
    return np.sum(precision[:, np.newaxis] * np.square(design), axis=0) / support


def _finite_float_array(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Convert an array-like input to a finite floating-point array."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_vector(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Validate a finite one-dimensional vector with strictly positive values."""
    vector = _finite_float_array(values, name=name)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    return vector


def _validate_gaussian_shapes(H: np.ndarray, B: np.ndarray, r_diag: np.ndarray) -> None:
    """Validate dimensions shared by the two Gaussian DFS implementations."""
    if H.ndim != 2:
        raise ValueError("H must be a two-dimensional observation-by-state matrix.")
    if H.shape[0] == 0 or H.shape[1] == 0:
        raise ValueError("H dimensions must both be non-empty.")
    if B.shape != (H.shape[1], H.shape[1]):
        raise ValueError("B must have shape (H states, H states).")
    if r_diag.shape[0] != H.shape[0]:
        raise ValueError("r_diag length must match H observations.")
    if not np.allclose(B, B.T, rtol=1e-12, atol=1e-12):
        raise ValueError("B must be symmetric.")


def _positive_definite_cholesky(matrix: np.ndarray, *, name: str) -> np.ndarray:
    """Return a Cholesky factor or raise a domain-specific ``ValueError``."""
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite.") from exc


def _cholesky_solve(cholesky: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    """Solve a positive-definite system from its lower Cholesky factor."""
    intermediate = np.linalg.solve(cholesky, right_hand_side)
    return np.linalg.solve(cholesky.T, intermediate)
