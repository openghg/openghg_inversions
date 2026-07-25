"""Low-rank Gaussian closure for conditional aggregation error.

This module defines a bounded, auditable approximation to the aggregation
error that appears when native positive masses are represented only by
partition totals.  Native-cell proportions have one common additive
Dirichlet base measure.  Conditional on the active mass of each region, the
hidden proportions within distinct regions are independent Dirichlet random
vectors.  Their observation-space mean and covariance are therefore
available analytically.

Here ``H`` is response per unit physical native-cell mass, matching
``GammaBetaTreeProblem.sensitivity``; it is not an unconverted RHIME
``fp_x_flux`` array. For diagonal measurement covariance ``D`` and a fixed
orthonormal basis ``B`` in error-whitened observation space, the
low-dimensional design is

``W = B.T @ D**(-1/2) @ H``.

The exact conditional covariance ``S`` of the projected aggregation residual
is computed region by region.  The hybrid likelihood is the normalized
Gaussian with covariance

``D + D**(1/2) @ B @ S @ B.T @ D**(1/2)``.

Its log density is evaluated with a small Cholesky factor of ``I + S`` and a
Woodbury correction; no dense observation covariance is needed.  This is a
moment closure, not an exact marginal likelihood unless the hidden
aggregation residual is Gaussian.  The exact Dirichlet moments and the
Gaussian normalizer are nevertheless part of the public contract.

All public value objects own read-only arrays and validate inputs eagerly.
The first implementation intentionally uses transparent region loops.  It is
an oracle-quality baseline for later prefix-sum or sparse acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log, pi
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "AdditiveDirichletAggregation",
    "PartitionMassState",
    "low_rank_gaussian_log_likelihood",
]

_LOG_TWO_PI = log(2.0 * pi)


def _readonly_float(values: ArrayLike, *, name: str) -> FloatArray:
    """Return a finite owned read-only ``float64`` array.

    Args:
        values: Numerical input.
        name: Name used in validation errors.

    Returns:
        Owned read-only array.

    Raises:
        ValueError: If any value is non-finite.
    """
    result = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _noise_vector(noise_sd: ArrayLike, observation_count: int) -> FloatArray:
    """Return validated positive independent observation standard deviations."""
    scale = np.asarray(noise_sd, dtype=np.float64)
    if scale.ndim == 0:
        scale = np.full(observation_count, float(scale), dtype=np.float64)
    if scale.shape != (observation_count,):
        raise ValueError("noise_sd must be scalar or have one entry per observation.")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("noise_sd must contain only finite strictly positive values.")
    return _readonly_float(scale, name="noise_sd")


def _orthonormal_basis(values: ArrayLike, observation_count: int) -> FloatArray:
    """Return a validated error-whitened orthonormal summary basis."""
    basis = _readonly_float(values, name="summary_basis")
    if basis.ndim != 2 or basis.shape[0] != observation_count:
        raise ValueError("summary_basis must have shape (number_of_observations, summary_dimension).")
    if basis.shape[1] > observation_count:
        raise ValueError("summary_basis cannot have more columns than observations.")
    gram = basis.T @ basis
    tolerance = float(256.0 * np.finfo(np.float64).eps * max(basis.shape))
    if not np.allclose(
        gram,
        np.eye(basis.shape[1], dtype=np.float64),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("summary_basis columns must be orthonormal in whitened observation space.")
    return basis


def _symmetric_positive_semidefinite(
    values: ArrayLike,
    dimension: int,
) -> FloatArray:
    """Return a validated symmetric positive-semidefinite matrix."""
    matrix = _readonly_float(values, name="summary_covariance")
    if matrix.shape != (dimension, dimension):
        raise ValueError("summary_covariance shape must match the summary dimension.")
    if dimension == 0:
        return matrix
    scale = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = float(512.0 * np.finfo(np.float64).eps * max(1, dimension) * scale)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise ValueError("summary_covariance must be symmetric.")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    minimum_eigenvalue = float(eigenvalues[0])
    if minimum_eigenvalue < -tolerance:
        raise ValueError("summary_covariance must be positive semidefinite.")
    if minimum_eigenvalue < 0.0:
        eigenvalues = np.maximum(eigenvalues, 0.0)
        symmetric = (eigenvectors * eigenvalues[np.newaxis, :]) @ eigenvectors.T
        symmetric = 0.5 * (symmetric + symmetric.T)
    symmetric.setflags(write=False)
    return cast(FloatArray, symmetric)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class PartitionMassState:
    """Partition labels and positive active region masses.

    Labels are contiguous zero-based region identifiers.  Their numerical
    values have no scientific meaning: a simultaneous permutation of labels
    and masses represents the same state.

    Args:
        labels: Integer native-cell labels of any non-empty spatial shape.
        masses: One finite strictly positive total for each active label.

    Attributes:
        labels: Owned read-only ``int64`` labels.
        masses: Owned read-only ``float64`` region totals.

    Raises:
        TypeError: If labels are not an integer array.
        ValueError: If labels are empty, negative, non-contiguous, omit an
            active region, or masses are malformed.
    """

    labels: IntArray = field(init=False)
    masses: FloatArray = field(init=False)

    def __init__(self, labels: ArrayLike, masses: ArrayLike) -> None:
        """Validate, own, and freeze labels and masses.

        Args:
            labels: Integer native-cell labels of any non-empty spatial shape.
            masses: One finite strictly positive total for each active label.

        Raises:
            TypeError: If labels are not an integer array.
            ValueError: If labels or masses violate the class invariants.
        """
        raw_labels = np.asarray(labels)
        if not np.issubdtype(raw_labels.dtype, np.integer) or np.issubdtype(
            raw_labels.dtype,
            np.bool_,
        ):
            raise TypeError("labels must be an integer array.")
        labels = np.array(raw_labels, dtype=np.int64, copy=True)
        if labels.ndim == 0 or labels.size == 0:
            raise ValueError("labels must be a non-empty array.")
        if np.any(labels < 0):
            raise ValueError("labels must be non-negative.")
        unique = np.unique(labels)
        if not np.array_equal(unique, np.arange(unique.size, dtype=np.int64)):
            raise ValueError("labels must use every contiguous identifier from zero.")

        owned_masses = _readonly_float(masses, name="masses")
        if owned_masses.shape != (unique.size,):
            raise ValueError("masses must have exactly one entry per active label.")
        if np.any(owned_masses <= 0.0):
            raise ValueError("masses must be strictly positive.")
        labels.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "masses", owned_masses)

    @property
    def region_count(self) -> int:
        """Return the number of active regions."""
        return int(self.masses.size)


def low_rank_gaussian_log_likelihood(
    observation: ArrayLike,
    mean: ArrayLike,
    noise_sd: ArrayLike,
    summary_basis: ArrayLike,
    summary_covariance: ArrayLike,
) -> float:
    """Evaluate one normalized diagonal-plus-low-rank Gaussian log density.

    The supplied basis is orthonormal after division by ``noise_sd``.  If
    ``r = D**(-1/2) @ (observation - mean)`` and ``z = B.T @ r``, the density
    uses a Cholesky factor of ``I + S``. The quadratic is evaluated as the
    sum of the squared residual orthogonal to ``B`` and
    ``z.T @ (I + S)**(-1) @ z``. This avoids subtracting two large nearly
    equal quadratic forms when aggregation variance is large.

    Args:
        observation: Finite one-dimensional observation vector.
        mean: Finite vector with one entry per observation.
        noise_sd: Positive scalar or independent standard deviations.
        summary_basis: Fixed matrix ``B`` with orthonormal columns in
            whitened observation space.
        summary_covariance: Symmetric positive-semidefinite matrix ``S`` in
            summary coordinates.

    Returns:
        Fully normalized Gaussian log density, including determinant terms.

    Raises:
        ValueError: If shapes, values, orthonormality, or positive
            semidefiniteness are invalid, or the small Cholesky system is not
            numerically positive definite.
    """
    observed = _readonly_float(observation, name="observation")
    expected = _readonly_float(mean, name="mean")
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("observation must be a non-empty one-dimensional vector.")
    if expected.shape != observed.shape:
        raise ValueError("mean must have one entry per observation.")
    scale = _noise_vector(noise_sd, observed.size)
    basis = _orthonormal_basis(summary_basis, observed.size)
    covariance = _symmetric_positive_semidefinite(
        summary_covariance,
        basis.shape[1],
    )

    whitened_residual = (observed - expected) / scale
    summary_residual = basis.T @ whitened_residual
    small_covariance = np.eye(basis.shape[1], dtype=np.float64) + covariance
    try:
        cholesky = np.linalg.cholesky(small_covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("I + summary_covariance must be numerically positive definite.") from error
    solved_summary = np.linalg.solve(
        cholesky.T,
        np.linalg.solve(cholesky, summary_residual),
    )
    orthogonal_residual = whitened_residual - basis @ summary_residual
    orthogonal_quadratic = float(orthogonal_residual @ orthogonal_residual)
    summary_quadratic = float(summary_residual @ solved_summary)
    log_determinant_correction = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    return -0.5 * (
        observed.size * _LOG_TWO_PI
        + 2.0 * float(np.sum(np.log(scale)))
        + orthogonal_quadratic
        + summary_quadratic
        + log_determinant_correction
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class AdditiveDirichletAggregation:
    """Conditional aggregation moments under one additive Dirichlet measure.

    Native-cell concentrations may have any non-empty spatial shape.  The
    final dimension of ``design`` follows their row-major flattening.  For
    each active region ``r``, hidden normalized cell masses follow
    ``Dirichlet(alpha[c] for c in r)`` independently conditional on the
    region total.  This class computes the exact first two moments of the
    resulting aggregation residual.

    Args:
        cell_alphas: Strictly positive finite native-cell concentrations.
        design: Finite observation-by-native-cell design matrix ``H``.
        noise_sd: Positive scalar or vector defining diagonal covariance
            ``D``.
        summary_basis: Fixed orthonormal basis ``B`` in error-whitened
            observation space.

    Attributes:
        cell_alphas: Owned read-only native concentration array.
        design: Owned read-only design matrix.
        noise_sd: Owned read-only observation-error standard deviations.
        summary_basis: Owned read-only whitened orthonormal basis.

    Raises:
        ValueError: If arrays are empty, non-finite, non-positive where
            required, shape-incompatible, or the basis is not orthonormal.
    """

    cell_alphas: FloatArray = field(init=False)
    design: FloatArray = field(init=False)
    noise_sd: FloatArray = field(init=False)
    summary_basis: FloatArray = field(init=False)
    _summary_design: FloatArray = field(init=False, repr=False)

    def __init__(
        self,
        cell_alphas: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        summary_basis: ArrayLike,
    ) -> None:
        """Validate, own, and freeze the common native model.

        Args:
            cell_alphas: Strictly positive native-cell concentrations.
            design: Observation-by-native-cell matrix.
            noise_sd: Positive scalar or per-observation standard deviations.
            summary_basis: Error-whitened orthonormal summary basis.

        Raises:
            ValueError: If inputs violate the class invariants.
        """
        alphas = _readonly_float(cell_alphas, name="cell_alphas")
        if alphas.ndim == 0 or alphas.size == 0 or np.any(alphas <= 0.0):
            raise ValueError("cell_alphas must be a non-empty array of strictly positive values.")
        with np.errstate(over="ignore"):
            alpha_total = float(np.sum(alphas))
        if not np.isfinite(alpha_total):
            raise ValueError("cell_alphas must have a finite additive total.")
        matrix = _readonly_float(design, name="design")
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != alphas.size:
            raise ValueError("design must be a non-empty matrix with one column per native cell.")
        scale = _noise_vector(noise_sd, matrix.shape[0])
        basis = _orthonormal_basis(summary_basis, matrix.shape[0])
        summary_design = basis.T @ (matrix / scale[:, np.newaxis])
        summary_design = _readonly_float(summary_design, name="summary_design")
        object.__setattr__(self, "cell_alphas", alphas)
        object.__setattr__(self, "design", matrix)
        object.__setattr__(self, "noise_sd", scale)
        object.__setattr__(self, "summary_basis", basis)
        object.__setattr__(self, "_summary_design", summary_design)

    @property
    def cell_shape(self) -> tuple[int, ...]:
        """Return the native spatial shape used by partition labels."""
        return self.cell_alphas.shape

    @property
    def summary_design(self) -> FloatArray:
        """Return read-only ``W = B.T @ D**(-1/2) @ H``."""
        return self._summary_design

    def _validated_state(self, state: PartitionMassState) -> tuple[IntArray, FloatArray]:
        """Return flattened labels and concentrations after model validation."""
        if not isinstance(state, PartitionMassState):
            raise TypeError("state must be a PartitionMassState.")
        if state.labels.shape != self.cell_shape:
            raise ValueError("state labels must have the same shape as cell_alphas.")
        return state.labels.reshape(-1), self.cell_alphas.reshape(-1)

    def conditional_native_mean(self, state: PartitionMassState) -> FloatArray:
        """Return the conditional mean native mass for one partition state.

        Args:
            state: Active labels and region totals.

        Returns:
            Owned read-only native mass array with :attr:`cell_shape`.

        Raises:
            TypeError: If ``state`` has the wrong type.
            ValueError: If label and native-grid shapes differ.
        """
        labels, alphas = self._validated_state(state)
        result = np.empty(alphas.size, dtype=np.float64)
        for region, mass in enumerate(state.masses):
            selected = labels == region
            region_alphas = alphas[selected]
            result[selected] = float(mass) * region_alphas / float(region_alphas.sum())
        return _readonly_float(result.reshape(self.cell_shape), name="conditional_native_mean")

    def conditional_observation_mean(self, state: PartitionMassState) -> FloatArray:
        """Return ``H`` times the conditional mean native mass.

        Args:
            state: Active labels and region totals.

        Returns:
            Owned read-only vector with one value per observation.
        """
        native_mean = self.conditional_native_mean(state).reshape(-1)
        return _readonly_float(self.design @ native_mean, name="conditional_observation_mean")

    def _residual_covariance(
        self,
        columns: FloatArray,
        state: PartitionMassState,
    ) -> FloatArray:
        """Compute exact Dirichlet residual covariance for selected columns."""
        labels, alphas = self._validated_state(state)
        covariance = np.zeros((columns.shape[0], columns.shape[0]), dtype=np.float64)
        for region, mass in enumerate(state.masses):
            selected = labels == region
            region_alphas = alphas[selected]
            concentration = float(region_alphas.sum())
            proportions = region_alphas / concentration
            region_columns = columns[:, selected]
            region_mean = region_columns @ proportions
            centered = region_columns - region_mean[:, np.newaxis]
            covariance += (
                float(mass) ** 2
                / (concentration + 1.0)
                * ((centered * proportions[np.newaxis, :]) @ centered.T)
            )
        covariance = 0.5 * (covariance + covariance.T)
        return _readonly_float(covariance, name="aggregation_residual_covariance")

    def observation_residual_covariance(self, state: PartitionMassState) -> FloatArray:
        """Return exact conditional aggregation covariance in observation space.

        Args:
            state: Active labels and region totals.

        Returns:
            Owned read-only matrix with shape
            ``(number_of_observations, number_of_observations)``.
        """
        return self._residual_covariance(self.design, state)

    def summary_residual_covariance(self, state: PartitionMassState) -> FloatArray:
        """Return exact conditional aggregation covariance in summary space.

        Args:
            state: Active labels and region totals.

        Returns:
            Owned read-only ``S`` with one row and column per summary.
        """
        return self._residual_covariance(self.summary_design, state)

    def dense_hybrid_covariance(self, state: PartitionMassState) -> FloatArray:
        """Materialize the diagonal-plus-low-rank observation covariance.

        This method is intended for tests and small diagnostics.  Production
        likelihood evaluation should use :meth:`hybrid_log_likelihood`, which
        factors only the summary-space matrix.

        Args:
            state: Active labels and region totals.

        Returns:
            Owned read-only covariance
            ``D + D**(1/2) B S B.T D**(1/2)``.
        """
        summary_covariance = self.summary_residual_covariance(state)
        lifted = self.noise_sd[:, np.newaxis] * self.summary_basis
        covariance = np.diag(np.square(self.noise_sd))
        covariance += lifted @ summary_covariance @ lifted.T
        covariance = 0.5 * (covariance + covariance.T)
        return _readonly_float(covariance, name="dense_hybrid_covariance")

    def hybrid_log_likelihood(
        self,
        observation: ArrayLike,
        state: PartitionMassState,
        *,
        mean_offset: ArrayLike | None = None,
    ) -> float:
        """Evaluate the normalized conditional Gaussian hybrid likelihood.

        Args:
            observation: Finite vector with one entry per observation row.
            state: Active labels and region totals.
            mean_offset: Optional fixed observation-space contribution added
                to the conditional aggregation mean.

        Returns:
            Fully normalized low-rank Gaussian log density.

        Raises:
            TypeError: If ``state`` has the wrong type.
            ValueError: If model-state or observation inputs are malformed.
        """
        mean = np.array(self.conditional_observation_mean(state), copy=True)
        if mean_offset is not None:
            offset = _readonly_float(mean_offset, name="mean_offset")
            if offset.shape != mean.shape:
                raise ValueError("mean_offset must have one entry per observation.")
            mean += offset
        return low_rank_gaussian_log_likelihood(
            observation,
            mean,
            self.noise_sd,
            self.summary_basis,
            self.summary_residual_covariance(state),
        )
