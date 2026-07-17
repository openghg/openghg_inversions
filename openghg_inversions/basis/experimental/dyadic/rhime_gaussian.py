"""Bocquet-consistent Gaussian multiscale model for RHIME scaling columns.

RHIME's native design grid already contains footprint multiplied by prior flux,
with shape ``(observation, row, column)``.  A regional relative-scaling anomaly
therefore has an observation column equal to the sum of its native cell
columns.  For independent native relative-scaling errors with common standard
deviation ``s``, the ``B^-1``-weighted left inverse of the regional
prolongation gives a regional variance ``s**2 / n_v``, where ``n_v`` is the
number of supported native cells in region ``v``.

The search tree may be built over a sum-coarsened grid, but native cells remain
the fine Gaussian state.  Consequently, this module computes the full signal
covariance from native columns and uses coarsening only to construct candidate
regional columns and their supported-cell counts.  All covariance matrices are
formed in observation space; no native-cell-by-native-cell matrix is
materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np
import numpy.typing as npt

from .multiscale import MultiscaleDesign, sum_coarsen_grid
from .state import PartitionState
from .tree import DyadicTree, NodeId


@dataclass(frozen=True)
class RHIMEGaussianMultiscale:
    """Precomputed Gaussian covariances and DFS scores for one RHIME grid.

    Construct instances with :meth:`from_native_grid`.  Candidate columns and
    node-indexed arrays use the contiguous node ordering of ``design.tree``.
    Nodes with no supported native cells have zero prior variance and score;
    they are removed from returned reduced state vectors.

    Attributes:
        design: Sum-preserving candidate observation columns on the coarse
            dyadic search grid.
        support_by_node: Supported native-cell count for every tree node.
        prior_variance_by_node: Regional relative-scaling variance
            ``relative_prior_sd**2 / support`` for supported nodes, otherwise
            zero.
        full_signal_covariance: Native fine-state signal covariance
            ``relative_prior_sd**2 * G G.T`` after excluding unsupported cells.
        innovation_covariance: Partition-invariant covariance
            ``diag(r_diag) + full_signal_covariance``.
        tile_scores: Additive DFS contribution for every candidate node.
        full_grid_dfs: DFS of the supported native fine state.
        relative_prior_sd: Common native relative-scaling prior standard
            deviation.
        r_diag: Positive diagonal of the non-aggregation observation covariance.
        native_design: Supported native RHIME contribution columns retained for
            cancellation-resistant aggregation-covariance calculations.
        native_support: Boolean native-cell support used by every partition.
        coarsen_factor: Square native-cell block width defining one search leaf.
    """

    design: MultiscaleDesign
    support_by_node: np.ndarray
    prior_variance_by_node: np.ndarray
    full_signal_covariance: np.ndarray
    innovation_covariance: np.ndarray
    tile_scores: np.ndarray
    full_grid_dfs: float
    relative_prior_sd: float
    r_diag: np.ndarray
    native_design: np.ndarray
    native_support: np.ndarray
    coarsen_factor: int

    @classmethod
    def from_native_grid(
        cls,
        G: npt.ArrayLike,
        prior_flux: npt.ArrayLike,
        r_diag: npt.ArrayLike,
        *,
        coarsen_factor: int,
        relative_prior_sd: float = 1.0,
        flux_tolerance: float = 0.0,
    ) -> RHIMEGaussianMultiscale:
        """Build a Bocquet-consistent model from native RHIME scaling columns.

        Support is defined by ``abs(prior_flux) > flux_tolerance``.  Design
        values at excluded cells must already be zero to normal floating-point
        precision (within ``1e-12`` of the supported-design scale); accepted
        residual roundoff is explicitly set to zero before constructing either
        native or regional covariances.  Coarse leaf support is the number of
        supported native cells in its block, including partial boundary blocks.

        Args:
            G: Native RHIME design contributions with shape ``(N, row, col)``.
                Values already include multiplication by ``prior_flux``.
            prior_flux: Native prior flux with shape ``(row, col)``.
            r_diag: Strictly positive diagonal of the base observation
                covariance, with shape ``(N,)``.
            coarsen_factor: Positive integer width of square sum-coarsening
                blocks used to define the dyadic search grid.
            relative_prior_sd: Positive common standard deviation ``s`` of
                independent native relative-scaling errors.
            flux_tolerance: Non-negative absolute prior-flux threshold used to
                select supported native cells.

        Returns:
            An immutable container of the candidate design, covariance terms,
            and additive DFS scores.

        Raises:
            TypeError: If ``coarsen_factor`` is not an integer.
            ValueError: If dimensions are incompatible, an array contains
                non-finite or complex values, ``r_diag`` or
                ``relative_prior_sd`` is not positive, ``flux_tolerance`` is
                negative, excluded design values are materially nonzero, or a
                finite positive-definite innovation covariance cannot be built.
        """
        native_design = _finite_float_array(G, name="G")
        flux = _finite_float_array(prior_flux, name="prior_flux")
        errors = _positive_vector(r_diag, name="r_diag")
        relative_sd = _positive_scalar(relative_prior_sd, name="relative_prior_sd")
        tolerance = _nonnegative_scalar(flux_tolerance, name="flux_tolerance")

        if native_design.ndim != 3:
            raise ValueError("G must have shape (observation, row, column).")
        if any(extent == 0 for extent in native_design.shape):
            raise ValueError("G dimensions must all be non-empty.")
        if flux.ndim != 2:
            raise ValueError("prior_flux must have shape (row, column).")
        if flux.shape != native_design.shape[1:]:
            raise ValueError("prior_flux spatial shape must match G.")
        if errors.shape[0] != native_design.shape[0]:
            raise ValueError("r_diag length must match G observations.")

        support = np.abs(flux) > tolerance
        excluded_values = native_design[:, ~support]
        reference_values = native_design[:, support] if np.any(support) else native_design
        exclusion_scale = float(np.max(np.abs(reference_values)))
        exclusion_atol = 1e-12 * exclusion_scale
        if not np.allclose(excluded_values, 0.0, rtol=0.0, atol=exclusion_atol):
            raise ValueError("G must be approximately zero where prior_flux support is excluded.")

        supported_design = native_design.copy()
        supported_design[:, ~support] = 0.0
        coarse_design = sum_coarsen_grid(supported_design, coarsen_factor)
        coarse_support = sum_coarsen_grid(support[np.newaxis, ...], coarsen_factor)
        if not np.all(np.isfinite(coarse_design.values)):
            raise ValueError("coarsening G must produce finite values.")

        search_shape = (int(coarse_design.values.shape[1]), int(coarse_design.values.shape[2]))
        tree = DyadicTree.from_shape(search_shape)
        design = MultiscaleDesign.from_grid(coarse_design.values, tree)
        support_design = MultiscaleDesign.from_grid(coarse_support.values, tree)
        support_by_node = support_design.values[0].astype(np.int64)

        with np.errstate(over="ignore", invalid="ignore"):
            relative_variance = float(np.square(relative_sd))
        if not np.isfinite(relative_variance):
            raise ValueError("relative_prior_sd is too large to produce a finite variance.")
        prior_variance_by_node = np.zeros(len(tree.nodes), dtype=float)
        supported_nodes = support_by_node > 0
        prior_variance_by_node[supported_nodes] = relative_variance / support_by_node[supported_nodes]

        native_columns = supported_design.reshape(native_design.shape[0], -1)
        with np.errstate(over="ignore", invalid="ignore"):
            scaled_native_columns = relative_sd * native_columns
            full_signal_covariance = scaled_native_columns @ scaled_native_columns.T
        full_signal_covariance = _symmetrize(full_signal_covariance)
        if not np.all(np.isfinite(full_signal_covariance)):
            raise ValueError("inputs must produce a finite full signal covariance.")
        innovation_covariance = full_signal_covariance + np.diag(errors)
        if not np.all(np.isfinite(innovation_covariance)):
            raise ValueError("inputs must produce a finite innovation covariance.")
        innovation_cholesky = _positive_definite_cholesky(
            innovation_covariance,
            name="innovation covariance",
        )

        solved_candidates = _cholesky_solve(innovation_cholesky, design.values)
        quadratic_forms = np.einsum("ij,ij->j", design.values, solved_candidates)
        tile_scores = prior_variance_by_node * quadratic_forms
        score_tolerance = 1e-12 * max(1.0, float(np.max(np.abs(tile_scores))))
        if np.any(tile_scores < -score_tolerance):
            raise ValueError("numerical failure produced a negative tile DFS score.")
        tile_scores = np.maximum(tile_scores, 0.0)

        solved_full_signal = _cholesky_solve(innovation_cholesky, full_signal_covariance)
        full_grid_dfs = float(np.trace(solved_full_signal))
        if full_grid_dfs < -1e-12:
            raise ValueError("numerical failure produced a negative full-grid DFS.")
        full_grid_dfs = max(full_grid_dfs, 0.0)

        for array in (
            design.values,
            support_by_node,
            prior_variance_by_node,
            full_signal_covariance,
            innovation_covariance,
            tile_scores,
            errors,
            supported_design,
            support,
        ):
            array.setflags(write=False)

        return cls(
            design=design,
            support_by_node=support_by_node,
            prior_variance_by_node=prior_variance_by_node,
            full_signal_covariance=full_signal_covariance,
            innovation_covariance=innovation_covariance,
            tile_scores=tile_scores,
            full_grid_dfs=full_grid_dfs,
            relative_prior_sd=relative_sd,
            r_diag=errors,
            native_design=supported_design,
            native_support=support,
            coarsen_factor=int(coarsen_factor),
        )

    def score(self, state: PartitionState) -> float:
        """Return additive Gaussian DFS for a valid partition state.

        Args:
            state: Exact active frontier over ``design.tree``.

        Returns:
            Sum of precomputed active-node DFS contributions.  Zero-support
            regions contribute zero.

        Raises:
            ValueError: If ``state`` is not valid for this model's tree.
            ArithmeticError: If numerical error makes the reduced-state score
                exceed the full-native-grid DFS by more than roundoff.
        """
        state.validate(self.design.tree)
        value = float(np.sum(self.tile_scores[list(state.ordered_active())]))
        bound_tolerance = 1e-10 * max(1.0, abs(self.full_grid_dfs))
        if value > self.full_grid_dfs + bound_tolerance:
            raise ArithmeticError("partition DFS exceeds full-grid DFS beyond numerical tolerance.")
        return value

    def reduced_design_and_variance(self, state: PartitionState) -> tuple[np.ndarray, np.ndarray]:
        """Gather supported active columns and diagonal regional variances.

        Args:
            state: Exact active frontier over ``design.tree``.

        Returns:
            ``(H_P, variance_diag)`` in stable active-node order after pruning
            all regions containing no supported native cells.

        Raises:
            ValueError: If ``state`` is not valid for this model's tree.
        """
        state.validate(self.design.tree)
        active = np.asarray(state.ordered_active(), dtype=np.int64)
        supported_active = active[self.support_by_node[active] > 0]
        return (
            self.design.values[:, supported_active],
            self.prior_variance_by_node[supported_active],
        )

    def reduced_signal_covariance(self, state: PartitionState) -> np.ndarray:
        """Return ``C_P`` for the supported active regional state.

        Args:
            state: Exact active frontier over ``design.tree``.

        Returns:
            Symmetric observation-space covariance
            ``sum_v (s**2 / n_v) c_v c_v.T``.

        Raises:
            ValueError: If ``state`` is not valid for this model's tree.
        """
        design, variances = self.reduced_design_and_variance(state)
        weighted_design = design * np.sqrt(variances)
        return _symmetrize(weighted_design @ weighted_design.T)

    def aggregation_error_covariance(self, state: PartitionState) -> np.ndarray:
        """Return unresolved covariance from stable within-region scatter.

        In exact arithmetic this is ``C_full - C_P``. It is computed directly
        as the sum of centered native-column scatter matrices within active
        regions, avoiding catastrophic cancellation when resolved and full
        signal covariances are large and nearly equal. No eigenvalue clipping
        is performed.

        Args:
            state: Exact active frontier over ``design.tree``.

        Returns:
            Symmetric observation-space aggregation covariance.

        Raises:
            ValueError: If ``state`` is not valid for this model's tree.
            ArithmeticError: If a materially negative covariance mode is
                produced despite the centered-scatter construction.
        """
        state.validate(self.design.tree)
        covariance = np.zeros_like(self.full_signal_covariance)
        for node_id in state.ordered_active():
            native_indices = self._supported_native_indices(node_id)
            covariance += _centered_scatter(self.native_design, native_indices)
        covariance *= self.relative_prior_sd**2
        covariance = _symmetrize(covariance)
        _validate_positive_semidefinite(covariance, name="aggregation error covariance")
        return covariance

    def effective_observation_covariance(self, state: PartitionState) -> np.ndarray:
        """Return ``R_P = diag(r_diag) + C_full - C_P`` for one state.

        Args:
            state: Exact active frontier over ``design.tree``.

        Returns:
            Symmetric effective observation covariance containing base
            observation error and unresolved aggregation error.

        Raises:
            ValueError: If ``state`` is not valid for this model's tree.
            ArithmeticError: If the resulting effective covariance is not
                positive definite.
        """
        covariance = _symmetrize(np.diag(self.r_diag) + self.aggregation_error_covariance(state))
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ArithmeticError("effective observation covariance is not positive definite.") from exc
        return covariance

    def _supported_native_indices(self, node_id: NodeId) -> np.ndarray:
        """Return flattened supported native-cell indices covered by one node."""
        tile = self.design.tree.tile(node_id)
        row_start = tile.row_start * self.coarsen_factor
        row_stop = min(tile.row_stop * self.coarsen_factor, self.native_support.shape[0])
        col_start = tile.col_start * self.coarsen_factor
        col_stop = min(tile.col_stop * self.coarsen_factor, self.native_support.shape[1])

        rows = np.arange(row_start, row_stop, dtype=np.int64)[:, np.newaxis]
        cols = np.arange(col_start, col_stop, dtype=np.int64)[np.newaxis, :]
        native_indices = (rows * self.native_support.shape[1] + cols).ravel()
        flat_support = self.native_support.ravel()
        return native_indices[flat_support[native_indices]]

    def effective_region_count(self, state: PartitionState) -> int:
        """Count active regions that contain at least one supported native cell.

        Args:
            state: Exact active frontier over ``design.tree``.

        Returns:
            Number of active coefficients retained by
            :meth:`reduced_design_and_variance`.

        Raises:
            ValueError: If ``state`` is not valid for this model's tree.
        """
        state.validate(self.design.tree)
        active = np.asarray(state.ordered_active(), dtype=np.int64)
        return int(np.count_nonzero(self.support_by_node[active]))

    def split_gain(self, node_id: NodeId) -> float:
        """Return the additive DFS change from splitting one candidate node.

        Args:
            node_id: Candidate parent node in ``design.tree``.  The node need
                not currently be active.

        Returns:
            Sum of child tile scores minus the parent tile score.

        Raises:
            KeyError: If ``node_id`` is not in the tree.
            ValueError: If ``node_id`` is a cell and therefore cannot split.
        """
        children = self.design.tree.children(node_id)
        if not children:
            raise ValueError(f"Cell node {node_id!r} cannot be split.")
        return float(np.sum(self.tile_scores[list(children)]) - self.tile_scores[index(node_id)])


def _finite_float_array(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Convert an array-like input to a finite real floating-point array."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_vector(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Validate a finite one-dimensional vector containing positive values."""
    vector = _finite_float_array(values, name=name)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    return vector.copy()


def _positive_scalar(value: float, *, name: str) -> float:
    """Validate and return one finite strictly positive real scalar."""
    scalar = _real_scalar(value, name=name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return scalar


def _nonnegative_scalar(value: float, *, name: str) -> float:
    """Validate and return one finite non-negative real scalar."""
    scalar = _real_scalar(value, name=name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return scalar


def _real_scalar(value: float, *, name: str) -> float:
    """Validate and return one finite real scalar."""
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a finite real scalar.")
    scalar = float(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar.")
    return scalar


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


def _centered_scatter(
    native_design: np.ndarray,
    native_indices: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Return stable observation-space scatter over selected native columns.

    Args:
        native_design: Native contribution grid with shape
            ``(observation, row, column)``.
        native_indices: Flattened spatial indices in one active region.
        chunk_size: Maximum native columns materialized at once.

    Returns:
        Sum of outer products around the region's column mean.
    """
    observations = native_design.shape[0]
    scatter = np.zeros((observations, observations), dtype=float)
    mean = np.zeros(observations, dtype=float)
    count = 0
    flat_design = native_design.reshape(observations, -1)

    for start in range(0, native_indices.size, chunk_size):
        indices = native_indices[start : start + chunk_size]
        batch = flat_design[:, indices]
        batch_count = batch.shape[1]
        if batch_count == 0:
            continue
        batch_mean = batch.mean(axis=1)
        centered = batch - batch_mean[:, np.newaxis]
        batch_scatter = centered @ centered.T
        if count == 0:
            mean = batch_mean
            scatter = batch_scatter
            count = batch_count
            continue

        total_count = count + batch_count
        mean_difference = batch_mean - mean
        scatter += batch_scatter
        scatter += (count * batch_count / total_count) * np.outer(mean_difference, mean_difference)
        mean += (batch_count / total_count) * mean_difference
        count = total_count

    return scatter


def _validate_positive_semidefinite(matrix: np.ndarray, *, name: str) -> None:
    """Raise when a symmetric covariance has a materially negative mode."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -1e-10 * scale:
        raise ArithmeticError(f"{name} is not positive semidefinite within numerical tolerance.")


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Remove floating-point asymmetry without altering covariance eigenvalues."""
    return 0.5 * (matrix + matrix.T)


__all__ = ["RHIMEGaussianMultiscale"]
