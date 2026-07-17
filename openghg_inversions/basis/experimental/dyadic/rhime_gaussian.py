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
from scipy.linalg import solve_triangular

from .multiscale import MultiscaleDesign, sum_coarsen_grid
from .state import PartitionState
from .tree import DyadicTree, NodeId


@dataclass(frozen=True, slots=True)
class NativePosteriorMarginals:
    """Posterior marginal summaries for native relative-scaling anomalies.

    Attributes:
        mean_increment: Posterior mean increment on the native spatial grid.
            The prior mean of the anomaly is zero.
        marginal_variance: Diagonal of the native posterior covariance on the
            spatial grid.
        support: Boolean grid identifying native locations that contribute to
            the observation design. Unsupported locations retain their prior
            mean and variance.
    """

    mean_increment: np.ndarray
    marginal_variance: np.ndarray
    support: np.ndarray


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
        innovation_cholesky: Cached lower Cholesky factor of
            ``innovation_covariance``.
        tile_scores: Additive DFS contribution for every candidate node.
        fisher_tile_scores: Additive base-error Fisher contribution for every
            candidate node.
        full_grid_dfs: DFS of the supported native fine state.
        full_grid_fisher: Base-error Fisher criterion for the supported native
            fine state.
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
    innovation_cholesky: np.ndarray
    tile_scores: np.ndarray
    fisher_tile_scores: np.ndarray
    full_grid_dfs: float
    full_grid_fisher: float
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

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            whitened_candidates = (
                design.values
                / np.sqrt(errors)[:, np.newaxis]
                * np.sqrt(prior_variance_by_node)[np.newaxis, :]
            )
            fisher_tile_scores = np.sum(np.square(whitened_candidates), axis=0)
            whitened_native = relative_sd * native_columns / np.sqrt(errors)[:, np.newaxis]
            full_grid_fisher = float(np.sum(np.square(whitened_native)))
        if not np.all(np.isfinite(fisher_tile_scores)) or not np.isfinite(full_grid_fisher):
            raise ValueError("inputs must produce finite Fisher criteria.")
        fisher_tolerance = 1e-12 * max(1.0, float(np.max(np.abs(fisher_tile_scores))))
        if np.any(fisher_tile_scores < -fisher_tolerance) or full_grid_fisher < -fisher_tolerance:
            raise ValueError("numerical failure produced a negative Fisher criterion.")
        fisher_tile_scores = np.maximum(fisher_tile_scores, 0.0)
        full_grid_fisher = max(full_grid_fisher, 0.0)

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
            innovation_cholesky,
            tile_scores,
            fisher_tile_scores,
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
            innovation_cholesky=innovation_cholesky,
            tile_scores=tile_scores,
            fisher_tile_scores=fisher_tile_scores,
            full_grid_dfs=full_grid_dfs,
            full_grid_fisher=full_grid_fisher,
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

    def fisher_score(self, state: PartitionState) -> float:
        """Return the additive base-error Fisher criterion for one partition.

        This is Bocquet et al.'s weak-signal criterion using the declared base
        observation covariance ``diag(r_diag)``. It does not include the
        partition-dependent aggregation covariance used by the
        aggregation-aware Fisher criterion.

        Args:
            state: Exact active frontier over ``design.tree``.

        Returns:
            Sum of precomputed active-node Fisher contributions.

        Raises:
            ValueError: If ``state`` is not valid for this model's tree.
            ArithmeticError: If numerical error makes the partition criterion
                exceed the native-grid criterion beyond roundoff.
        """
        state.validate(self.design.tree)
        value = float(np.sum(self.fisher_tile_scores[list(state.ordered_active())]))
        bound_tolerance = 1e-10 * max(1.0, abs(self.full_grid_fisher))
        if value > self.full_grid_fisher + bound_tolerance:
            raise ArithmeticError("partition Fisher criterion exceeds its native-grid bound.")
        return value

    def data_dependent_tile_scores(self, innovations: npt.ArrayLike) -> np.ndarray:
        """Return additive Equation 45 scores for one realized innovation.

        The input must already be centered on the native prior prediction. The
        returned score omits the factor ``1/2`` used in some conventions and is

        ``(s**2 / n_v) * (h_v.T @ S**-1 @ innovations)**2``.

        Args:
            innovations: Finite vector with one centered value per observation.

        Returns:
            Read-only non-negative score for every candidate tree node.

        Raises:
            ValueError: If ``innovations`` has an incompatible shape or
                contains non-finite values.
            ArithmeticError: If numerical error produces a negative or
                non-finite score.
        """
        residual = self._validated_innovations(innovations)
        solved_residual = _cholesky_solve(self.innovation_cholesky, residual)
        projected_update = self.design.values.T @ solved_residual
        with np.errstate(over="ignore", invalid="ignore"):
            scores = self.prior_variance_by_node * np.square(projected_update)
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
            raise ArithmeticError("Equation 45 tile scores must be finite and non-negative.")
        scores.setflags(write=False)
        return scores

    def data_dependent_score(
        self,
        state: PartitionState,
        innovations: npt.ArrayLike,
    ) -> float:
        """Return the Equation 45 posterior-mean criterion for one partition.

        Args:
            state: Exact active frontier over ``design.tree``.
            innovations: Finite observation vector centered on the native prior
                prediction.

        Returns:
            Additive Equation 45 score without a factor ``1/2``.

        Raises:
            ValueError: If the state or innovation vector is invalid.
            ArithmeticError: If the projected score exceeds its native-grid
                bound beyond numerical roundoff.
        """
        state.validate(self.design.tree)
        scores = self.data_dependent_tile_scores(innovations)
        value = float(np.sum(scores[list(state.ordered_active())]))
        full_value = self.full_grid_data_dependent_score(innovations)
        bound_tolerance = 1e-10 * max(1.0, abs(full_value))
        if value > full_value + bound_tolerance:
            raise ArithmeticError("partition Equation 45 score exceeds its native-grid bound.")
        return value

    def full_grid_data_dependent_score(self, innovations: npt.ArrayLike) -> float:
        """Return the native-grid Equation 45 score for one innovation vector.

        Args:
            innovations: Finite observation vector centered on the native prior
                prediction.

        Returns:
            Squared prior-precision norm of the native posterior mean increment,
            without a factor ``1/2``.

        Raises:
            ValueError: If ``innovations`` is invalid.
            ArithmeticError: If the calculation is non-finite.
        """
        residual = self._validated_innovations(innovations)
        solved_residual = _cholesky_solve(self.innovation_cholesky, residual)
        native_updates = self.native_design.reshape(residual.size, -1).T @ solved_residual
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(self.relative_prior_sd**2 * np.dot(native_updates, native_updates))
        if not np.isfinite(value) or value < 0.0:
            raise ArithmeticError("native-grid Equation 45 score must be finite and non-negative.")
        return value

    def native_posterior_marginals(
        self,
        innovations: npt.ArrayLike,
        *,
        chunk_size: int = 4096,
    ) -> NativePosteriorMarginals:
        """Compute native posterior means and marginal variances in chunks.

        The native state is the relative-scaling anomaly with prior
        ``N(0, s**2 I)``. Unsupported grid locations have zero design columns,
        so they retain posterior mean zero and variance ``s**2``. Only the
        diagonal of the posterior covariance is returned; cross-location
        covariance must be queried through a separate operator or projected
        calculation.

        Args:
            innovations: Finite observation vector centered on the native prior
                prediction.
            chunk_size: Positive maximum number of native design columns
                transformed at once.

        Returns:
            Read-only posterior mean, marginal variance, and support grids.

        Raises:
            TypeError: If ``chunk_size`` is not an integer.
            ValueError: If the innovations or chunk size are invalid.
            ArithmeticError: If posterior variances are materially negative or
                any output is non-finite.
        """
        residual = self._validated_innovations(innovations)
        try:
            chunk = index(chunk_size)
        except TypeError as exc:
            raise TypeError("chunk_size must be an integer.") from exc
        if chunk < 1:
            raise ValueError("chunk_size must be positive.")

        solved_residual = _cholesky_solve(self.innovation_cholesky, residual)
        flat_design = self.native_design.reshape(residual.size, -1)
        prior_variance = self.relative_prior_sd**2
        posterior_mean = prior_variance * (flat_design.T @ solved_residual)
        posterior_variance = np.empty(flat_design.shape[1], dtype=float)

        for start in range(0, flat_design.shape[1], chunk):
            stop = min(start + chunk, flat_design.shape[1])
            transformed = solve_triangular(
                self.innovation_cholesky,
                flat_design[:, start:stop],
                lower=True,
                check_finite=False,
            )
            posterior_variance[start:stop] = prior_variance - (
                prior_variance**2 * np.sum(np.square(transformed), axis=0)
            )

        resolution_floor = 16.0 * np.finfo(float).eps * prior_variance
        supported_flat = self.native_support.ravel()
        if np.any(posterior_variance[supported_flat] <= resolution_floor):
            posterior_variance = _native_posterior_variance_svd(
                flat_design,
                self.r_diag,
                relative_prior_sd=self.relative_prior_sd,
            )

        variance_tolerance = 1e-10 * max(1.0, prior_variance)
        if float(np.min(posterior_variance)) < -variance_tolerance:
            raise ArithmeticError("native posterior marginal variance is materially negative.")
        posterior_variance = np.maximum(posterior_variance, 0.0)
        if not np.all(np.isfinite(posterior_mean)) or not np.all(np.isfinite(posterior_variance)):
            raise ArithmeticError("native posterior marginals must be finite.")

        spatial_shape = self.native_support.shape
        mean_grid = posterior_mean.reshape(spatial_shape)
        variance_grid = posterior_variance.reshape(spatial_shape)
        support = self.native_support.copy()
        for array in (mean_grid, variance_grid, support):
            array.setflags(write=False)
        return NativePosteriorMarginals(
            mean_increment=mean_grid,
            marginal_variance=variance_grid,
            support=support,
        )

    def _validated_innovations(self, innovations: npt.ArrayLike) -> np.ndarray:
        """Return a finite innovation vector matching the observation count."""
        residual = _finite_float_array(innovations, name="innovations")
        if residual.ndim != 1 or residual.shape[0] != self.native_design.shape[0]:
            raise ValueError("innovations must contain one value per observation.")
        return residual

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
    intermediate = solve_triangular(
        cholesky,
        right_hand_side,
        lower=True,
        check_finite=False,
    )
    return solve_triangular(
        cholesky.T,
        intermediate,
        lower=False,
        check_finite=False,
    )


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


def _native_posterior_variance_svd(
    native_design: np.ndarray,
    r_diag: np.ndarray,
    *,
    relative_prior_sd: float,
) -> np.ndarray:
    """Return cancellation-resistant native marginal variances via thin SVD.

    This fallback evaluates the diagonal of
    ``s**2 (I + Z.T Z)**-1`` with
    ``Z = diag(r_diag)**-1/2 @ (s * native_design)``. It is used only when the
    faster observation-space subtraction approaches floating-point resolution
    for at least one supported grid location.

    Args:
        native_design: Observation-by-native design matrix.
        r_diag: Positive diagonal of the base observation covariance.
        relative_prior_sd: Positive native relative-scaling prior standard
            deviation ``s``.

    Returns:
        Posterior marginal variance for every native grid location.
    """
    whitened_design = (
        relative_prior_sd * native_design / np.sqrt(r_diag)[:, np.newaxis]
    )
    _, singular_values, right_vectors_t = np.linalg.svd(
        whitened_design,
        full_matrices=False,
    )
    squared_right_vectors = np.square(right_vectors_t)
    row_space_mass = np.sum(squared_right_vectors, axis=0)
    unresolved_mass = np.maximum(1.0 - row_space_mass, 0.0)
    informed_mass = np.sum(
        squared_right_vectors / (1.0 + np.square(singular_values))[:, np.newaxis],
        axis=0,
    )
    return relative_prior_sd**2 * (unresolved_mass + informed_mass)


def _validate_positive_semidefinite(matrix: np.ndarray, *, name: str) -> None:
    """Raise when a symmetric covariance has a materially negative mode."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -1e-10 * scale:
        raise ArithmeticError(f"{name} is not positive semidefinite within numerical tolerance.")


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Remove floating-point asymmetry without altering covariance eigenvalues."""
    return 0.5 * (matrix + matrix.T)


__all__ = ["NativePosteriorMarginals", "RHIMEGaussianMultiscale"]
