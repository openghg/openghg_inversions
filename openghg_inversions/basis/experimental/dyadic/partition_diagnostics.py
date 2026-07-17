"""Projection-consistent Gaussian diagnostics for labelled search partitions.

This module evaluates arbitrary positive integer-labelled partitions on the
search grid of :class:`RHIMEGaussianMultiscale`.  Search labels are expanded
back to native RHIME cells before support is counted or aggregation error is
computed.  A retained regional scaling coefficient therefore has design
column equal to the sum of its supported native columns and prior variance
``tau**2 / n_supported``.

The unresolved aggregation covariance is formed directly from centered
native-column scatter within each labelled region.  This avoids subtracting
the reduced signal covariance from a potentially much larger full covariance.
The resulting reduced signal and effective observation covariances close to
the partition-invariant innovation covariance, up to floating-point error.

The posterior helper implements only conjugate linear-Gaussian conditioning.
It does not mutate the source model or diagnostics and does not depend on a
production inversion API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .rhime_gaussian import RHIMEGaussianMultiscale


@dataclass(frozen=True, slots=True)
class GaussianPartitionDiagnostics:
    """Gaussian model and diagnostics for one labelled search-grid partition.

    Arrays are copied during construction and marked read-only.  Region-indexed
    arrays follow ascending ``supported_region_ids`` order.  Labels whose
    expanded native cells have no flux support remain visible in ``label_grid``
    but do not create coefficient directions.

    Attributes:
        label_grid: Positive integer labels on the model's two-dimensional
            search grid.
        supported_region_ids: Label values with at least one supported native
            flux cell.
        supported_native_counts: Supported native-cell count for each retained
            region.
        regional_design: Summed regional design matrix ``H_P`` with shape
            ``(observation, retained_region)``.
        prior_variances: Diagonal regional prior variances
            ``tau**2 / supported_native_counts``.
        reduced_signal_covariance: Observation-space retained signal covariance
            ``C_P = H_P B_P H_P.T``.
        aggregation_error_covariance: Direct within-region native-column
            scatter covariance ``C_agg``.
        effective_observation_covariance: Base observation covariance plus
            ``C_agg``.
        dfs: Degrees of freedom for signal, ``trace(S^-1 C_P)``, where ``S``
            is the source model's invariant innovation covariance.
    """

    label_grid: np.ndarray
    supported_region_ids: np.ndarray
    supported_native_counts: np.ndarray
    regional_design: np.ndarray
    prior_variances: np.ndarray
    reduced_signal_covariance: np.ndarray
    aggregation_error_covariance: np.ndarray
    effective_observation_covariance: np.ndarray
    dfs: float

    @classmethod
    def from_search_labels(
        cls,
        model: RHIMEGaussianMultiscale,
        labels: npt.ArrayLike,
    ) -> GaussianPartitionDiagnostics:
        """Construct diagnostics from labels on a model's search grid.

        Args:
            model: Existing RHIME Gaussian multiscale model whose native
                columns, support, prior scale, and observation errors define
                the Gaussian problem.
            labels: Finite, integral, strictly positive labels with spatial
                shape equal to ``model.design.tree.shape``.  Labels need not be
                contiguous.

        Returns:
            Immutable diagnostics for all labels with native flux support.

        Raises:
            TypeError: If ``model`` is not a
                :class:`RHIMEGaussianMultiscale`.
            ValueError: If labels are not a valid complete search-grid
                partition or the source model has incompatible dimensions.
            ArithmeticError: If direct covariance construction does not close
                to the source innovation covariance, produces a materially
                negative aggregation mode, or produces a non-positive-definite
                effective observation covariance.
        """
        return build_partition_diagnostics(model, labels)


@dataclass(frozen=True, slots=True)
class GaussianPartitionObjectives:
    """Named Gaussian representation criteria for one labelled partition.

    Attributes:
        dfs: Degrees of freedom for signal from Bocquet et al. Equation 38.
        fisher: Base-error Fisher trace using ``diag(r_diag)``.
        aggregation_aware_fisher: Fisher trace using the partition's effective
            observation covariance, including aggregation error.
        equation45: Squared prior-precision norm of the retained posterior-mean
            increment. This convention omits the factor ``1/2``.
        bayesian_information_gain: KL divergence from the projected posterior
            to its projected prior, including the conventional factor ``1/2``.
    """

    dfs: float
    fisher: float
    aggregation_aware_fisher: float
    equation45: float
    bayesian_information_gain: float


def build_partition_diagnostics(
    model: RHIMEGaussianMultiscale,
    labels: npt.ArrayLike,
) -> GaussianPartitionDiagnostics:
    """Build projection-consistent diagnostics for arbitrary search labels.

    Search labels are repeated over each model coarsening block and cropped to
    the native spatial shape.  Only native cells selected by
    ``model.native_support`` contribute to regional columns, support counts,
    and centered aggregation scatter.  Consequently, an entirely unsupported
    labelled region is retained in the label image but omitted from the
    reduced coefficient state.

    Args:
        model: Existing Gaussian multiscale model defining native RHIME design
            columns and covariance hyperparameters.
        labels: Complete positive integer labelling of the two-dimensional
            model search grid.  Arbitrary non-contiguous label values are
            accepted.

    Returns:
        Immutable labelled-partition diagnostics.

    Raises:
        TypeError: If ``model`` is not a
            :class:`RHIMEGaussianMultiscale`.
        ValueError: If label values or dimensions are invalid, or model arrays
            have incompatible dimensions.
        ArithmeticError: If covariance closure, positive semidefiniteness, or
            positive definiteness fails beyond numerical tolerance.
    """
    if not isinstance(model, RHIMEGaussianMultiscale):
        raise TypeError("model must be an RHIMEGaussianMultiscale.")

    label_grid = _positive_integer_labels(labels, expected_shape=model.design.tree.shape)
    native_design, native_support, r_diag = _validated_model_arrays(model)
    expanded_labels = np.repeat(
        np.repeat(label_grid, model.coarsen_factor, axis=0),
        model.coarsen_factor,
        axis=1,
    )[: native_support.shape[0], : native_support.shape[1]]

    region_ids, supported_counts, regional_design = _group_supported_search_leaves(
        model,
        label_grid,
        observations=native_design.shape[0],
    )

    observations = native_design.shape[0]
    flat_support = native_support.ravel()
    supported_native_indices = np.flatnonzero(flat_support)
    supported_labels = expanded_labels.ravel()[supported_native_indices]
    native_region_ids, inverse, native_counts = np.unique(
        supported_labels,
        return_inverse=True,
        return_counts=True,
    )
    if not np.array_equal(region_ids, native_region_ids) or not np.array_equal(
        supported_counts,
        native_counts,
    ):
        raise ValueError("model leaf support is inconsistent with expanded native support.")

    aggregation_scatter = np.zeros((observations, observations), dtype=float)
    if region_ids.size:
        grouped_order = np.argsort(inverse, kind="stable")
        grouped_indices = supported_native_indices[grouped_order]
        group_starts = np.concatenate(([0], np.cumsum(supported_counts[:-1], dtype=np.int64)))
        for start, count in zip(group_starts, supported_counts, strict=True):
            native_indices = grouped_indices[start : start + count]
            aggregation_scatter += _centered_scatter(native_design, native_indices)

    tau_squared = float(np.square(model.relative_prior_sd))
    prior_variances = tau_squared / supported_counts.astype(float)
    weighted_design = regional_design * np.sqrt(prior_variances)
    reduced_signal_covariance = _symmetrize(weighted_design @ weighted_design.T)
    aggregation_error_covariance = _symmetrize(tau_squared * aggregation_scatter)
    _validate_positive_semidefinite(
        aggregation_error_covariance,
        name="aggregation error covariance",
    )

    effective_observation_covariance = _symmetrize(
        np.diag(r_diag) + aggregation_error_covariance,
    )
    _positive_definite_cholesky(
        effective_observation_covariance,
        name="effective observation covariance",
        error_type=ArithmeticError,
    )

    innovation, innovation_cholesky = _validated_model_innovation(model, observations=observations)
    reconstructed_innovation = effective_observation_covariance + reduced_signal_covariance
    closure_scale = max(1.0, float(np.max(np.abs(innovation))))
    if not np.allclose(reconstructed_innovation, innovation, rtol=1e-10, atol=1e-12 * closure_scale):
        raise ArithmeticError("partition covariances do not close to the model innovation covariance.")

    solved_signal = _cholesky_solve(innovation_cholesky, reduced_signal_covariance)
    dfs = float(np.trace(solved_signal))
    dfs_tolerance = 1e-10 * max(1.0, abs(model.full_grid_dfs))
    if dfs < -dfs_tolerance or dfs > model.full_grid_dfs + dfs_tolerance:
        raise ArithmeticError("partition DFS lies outside its projection-consistent bounds.")
    dfs = max(dfs, 0.0)

    arrays = (
        label_grid,
        region_ids,
        supported_counts,
        regional_design,
        prior_variances,
        reduced_signal_covariance,
        aggregation_error_covariance,
        effective_observation_covariance,
    )
    for array in arrays:
        array.setflags(write=False)

    return GaussianPartitionDiagnostics(
        label_grid=label_grid,
        supported_region_ids=region_ids,
        supported_native_counts=supported_counts,
        regional_design=regional_design,
        prior_variances=prior_variances,
        reduced_signal_covariance=reduced_signal_covariance,
        aggregation_error_covariance=aggregation_error_covariance,
        effective_observation_covariance=effective_observation_covariance,
        dfs=dfs,
    )


def gaussian_partition_objectives(
    model: RHIMEGaussianMultiscale,
    partition: GaussianPartitionDiagnostics,
    innovations: npt.ArrayLike,
) -> GaussianPartitionObjectives:
    """Evaluate DFS, Fisher, Equation 45, and Bayesian KL consistently.

    The innovation vector must be centered on the native prior prediction. The
    projected posterior is the exact Gaussian restriction induced by the source
    model, so all criteria use the same partition-invariant innovation
    covariance. The Bayesian information gain is evaluated eigenwise from the
    prior-whitened averaging kernel to retain weak covariance information and
    avoid determinant cancellation between differently scaled regional
    variances.

    Args:
        model: Source independent-relative-error Gaussian model.
        partition: Labelled partition diagnostics constructed from ``model``.
        innovations: Finite centered observation vector.

    Returns:
        Immutable named objective values with explicit normalization
        conventions.

    Raises:
        TypeError: If ``model`` or ``partition`` has the wrong type.
        ValueError: If arrays, dimensions, or the innovation vector are invalid
            or if the partition does not match the model.
        ArithmeticError: If an objective is non-finite or materially negative,
            or an averaging-kernel mode is incompatible with a positive-
            definite projected posterior covariance.
    """
    if not isinstance(model, RHIMEGaussianMultiscale):
        raise TypeError("model must be an RHIMEGaussianMultiscale.")
    if not isinstance(partition, GaussianPartitionDiagnostics):
        raise TypeError("partition must be GaussianPartitionDiagnostics.")

    regional_design, prior_variances, effective_covariance = _validated_partition_arrays(partition)
    _, _, r_diag = _validated_model_arrays(model)
    residual = _finite_float_array(innovations, name="innovations")
    observations = regional_design.shape[0]
    expected_shape = (observations, observations)
    if residual.ndim != 1 or residual.shape[0] != observations:
        raise ValueError("innovations must contain one value per observation.")
    innovation, innovation_cholesky = _validated_model_innovation(model, observations=observations)
    if effective_covariance.shape != expected_shape:
        raise ValueError("model and partition covariance shapes must match their observation count.")
    if r_diag.shape[0] != observations:
        raise ValueError("model and partition observation counts must match.")
    if not np.allclose(
        effective_covariance + partition.reduced_signal_covariance,
        innovation,
        rtol=1e-10,
        atol=1e-12 * max(1.0, float(np.max(np.abs(innovation)))),
    ):
        raise ValueError("partition covariance terms are incompatible with the source model.")

    signal_covariance = partition.reduced_signal_covariance
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        fisher = float(np.sum(np.diag(signal_covariance) / r_diag))
    effective_cholesky = _positive_definite_cholesky(
        effective_covariance,
        name="partition effective observation covariance",
        error_type=ArithmeticError,
    )
    aggregation_aware_fisher = float(np.trace(_cholesky_solve(effective_cholesky, signal_covariance)))

    solved_residual = _cholesky_solve(innovation_cholesky, residual)
    projected_linear_update = regional_design.T @ solved_residual
    equation45 = float(np.sum(prior_variances * np.square(projected_linear_update)))

    if prior_variances.size:
        whitened_design = regional_design * np.sqrt(prior_variances)
        solved_design = _cholesky_solve(innovation_cholesky, whitened_design)
        averaging_kernel = _symmetrize(whitened_design.T @ solved_design)
        covariance_information_gain = _covariance_kl_from_averaging_kernel(averaging_kernel)
        bayesian_information_gain = covariance_information_gain + 0.5 * equation45
    else:
        bayesian_information_gain = 0.0

    values = (
        partition.dfs,
        fisher,
        aggregation_aware_fisher,
        equation45,
        bayesian_information_gain,
    )
    tolerance = 1e-10 * max(1.0, *(abs(value) for value in values))
    if not all(np.isfinite(value) for value in values):
        raise ArithmeticError("Gaussian partition objectives must be finite.")
    if any(value < -tolerance for value in values):
        raise ArithmeticError("Gaussian partition objectives must be non-negative.")
    return GaussianPartitionObjectives(*(max(value, 0.0) for value in values))


def gaussian_posterior_mean(
    partition: GaussianPartitionDiagnostics,
    observations: npt.ArrayLike,
    *,
    emission_prior_mean: npt.ArrayLike = 1.0,
    baseline_design: npt.ArrayLike | None = None,
    baseline_prior_mean: npt.ArrayLike | None = None,
    baseline_prior_variances: npt.ArrayLike | None = None,
    training_subset: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Condition regional emissions and optional baselines on observations.

    Emission coefficients use the partition's diagonal prior variances and a
    configurable prior mean.  Baseline coefficients, when requested, are
    appended after emission coefficients and require explicit prior means and
    strictly positive diagonal variances.  Conditioning uses the selected
    principal submatrix of the partition's effective observation covariance;
    returned predictions use the posterior-mean retained components for every
    observation, including observations held out from fitting.

    Args:
        partition: Labelled Gaussian diagnostics defining emission design,
            prior variances, and effective observation covariance.
        observations: Finite observation vector.
        emission_prior_mean: Scalar mean broadcast over retained emission
            regions, or one finite mean per retained region.  Defaults to one.
        baseline_design: Optional finite matrix with one row per observation
            and one column per baseline coefficient.
        baseline_prior_mean: Explicit finite one-dimensional baseline prior
            mean.  Required exactly when ``baseline_design`` is supplied.
        baseline_prior_variances: Explicit finite, strictly positive diagonal
            baseline prior variances.  Required exactly when
            ``baseline_design`` is supplied.
        training_subset: Optional non-empty Boolean mask of observation length
            or one-dimensional array of unique, non-negative integer indices.
            Defaults to all observations.

    Returns:
        ``(coefficient_posterior_mean, retained_predictions)``.  Coefficients
        are ordered as retained emission regions followed by baseline columns;
        predictions have one value per original observation.

    Raises:
        TypeError: If ``partition`` is not
            :class:`GaussianPartitionDiagnostics`.
        ValueError: If observations, means, variances, baseline inputs, or the
            training subset have incompatible dimensions or invalid values, or
            if conditioning does not produce a finite positive-definite
            Gaussian system.
    """
    if not isinstance(partition, GaussianPartitionDiagnostics):
        raise TypeError("partition must be GaussianPartitionDiagnostics.")

    regional_design, emission_variances, effective_covariance = _validated_partition_arrays(partition)
    observation_vector = _finite_float_array(observations, name="observations")
    if observation_vector.ndim != 1 or observation_vector.shape[0] != regional_design.shape[0]:
        raise ValueError("observations must be one-dimensional with one value per design row.")

    emission_means = _broadcast_mean(
        emission_prior_mean,
        size=regional_design.shape[1],
        name="emission_prior_mean",
    )
    baseline_matrix, baseline_means, baseline_variances = _validated_baseline_inputs(
        baseline_design,
        baseline_prior_mean,
        baseline_prior_variances,
        observations=regional_design.shape[0],
    )
    training_indices = _training_indices(training_subset, observations=regional_design.shape[0])

    design = np.column_stack((regional_design, baseline_matrix))
    prior_mean = np.concatenate((emission_means, baseline_means))
    prior_variances = np.concatenate((emission_variances, baseline_variances))
    training_design = design[training_indices]
    training_covariance = effective_covariance[np.ix_(training_indices, training_indices)]

    weighted_training_design = training_design * prior_variances
    conditioning_covariance = _symmetrize(
        training_covariance + weighted_training_design @ training_design.T,
    )
    if not np.all(np.isfinite(conditioning_covariance)):
        raise ValueError("posterior conditioning covariance must contain only finite values.")
    conditioning_cholesky = _positive_definite_cholesky(
        conditioning_covariance,
        name="posterior conditioning covariance",
        error_type=ValueError,
    )
    residual = observation_vector[training_indices] - training_design @ prior_mean
    solved_residual = _cholesky_solve(conditioning_cholesky, residual)
    posterior_mean = prior_mean + prior_variances * (training_design.T @ solved_residual)
    predictions = design @ posterior_mean
    if not np.all(np.isfinite(posterior_mean)) or not np.all(np.isfinite(predictions)):
        raise ValueError("posterior mean calculation must produce only finite values.")
    return posterior_mean, predictions


def emissions_compression_quality(
    model: RHIMEGaussianMultiscale,
    partition: GaussianPartitionDiagnostics,
    *,
    observation_subset: npt.ArrayLike | None = None,
) -> float:
    """Measure retained emissions information on an observation subset.

    This diagnostic is deliberately emissions-only.  It uses the base model's
    diagonal observation covariance ``R`` and computes
    ``Q = 1 - trace(R^-1 C_agg) / trace(R^-1 C_full)`` on the selected
    principal submatrices.  Baseline designs and baseline priors are neither
    accepted nor included.  If the selected full-signal trace is exactly zero,
    the function returns ``1.0``: the subset contains no emissions information
    that the partition could lose.

    Args:
        model: Source model providing ``r_diag`` and the full native emissions
            signal covariance.
        partition: Labelled diagnostics built from the source model.
        observation_subset: Optional non-empty Boolean mask of observation
            length or one-dimensional array of unique, non-negative integer
            indices.  Defaults to all observations.

    Returns:
        Finite compression-quality fraction in the closed interval ``[0, 1]``.
        Values within numerical roundoff of either endpoint are clipped.

    Raises:
        TypeError: If model or partition has the wrong type.
        ValueError: If source arrays or the subset are invalid, or the model
            and partition dimensions/base covariance are incompatible.
        ArithmeticError: If weighted covariance traces are non-finite,
            inconsistent at a zero denominator, or imply a quality outside
            ``[0, 1]`` beyond numerical tolerance.
    """
    if not isinstance(model, RHIMEGaussianMultiscale):
        raise TypeError("model must be an RHIMEGaussianMultiscale.")
    if not isinstance(partition, GaussianPartitionDiagnostics):
        raise TypeError("partition must be GaussianPartitionDiagnostics.")

    _, _, r_diag = _validated_model_arrays(model)
    full_signal = _finite_float_array(model.full_signal_covariance, name="model.full_signal_covariance")
    aggregation = _finite_float_array(
        partition.aggregation_error_covariance,
        name="partition.aggregation_error_covariance",
    )
    effective = _finite_float_array(
        partition.effective_observation_covariance,
        name="partition.effective_observation_covariance",
    )
    expected_shape = (r_diag.size, r_diag.size)
    if full_signal.shape != expected_shape or aggregation.shape != expected_shape:
        raise ValueError("model and partition covariance shapes must match the observation count.")
    if effective.shape != expected_shape or not np.allclose(
        effective,
        np.diag(r_diag) + aggregation,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("partition effective covariance is incompatible with the model r_diag.")

    indices = _subset_indices(
        observation_subset,
        observations=r_diag.size,
        name="observation_subset",
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        weighted_full = np.diag(full_signal)[indices] / r_diag[indices]
        weighted_aggregation = np.diag(aggregation)[indices] / r_diag[indices]
        denominator = float(np.sum(weighted_full))
        numerator = float(np.sum(weighted_aggregation))
    if not np.isfinite(denominator) or not np.isfinite(numerator):
        raise ArithmeticError("compression-quality covariance traces must be finite.")
    if denominator == 0.0:
        if numerator != 0.0:
            raise ArithmeticError("aggregation trace must vanish when the full-signal trace is zero.")
        return 1.0
    if denominator < 0.0:
        raise ArithmeticError("full-signal weighted trace cannot be negative.")

    quality = 1.0 - numerator / denominator
    endpoint_tolerance = 1e-10
    if quality < -endpoint_tolerance or quality > 1.0 + endpoint_tolerance:
        raise ArithmeticError("compression quality lies outside [0, 1] beyond numerical tolerance.")
    return float(np.clip(quality, 0.0, 1.0))


def _positive_integer_labels(values: npt.ArrayLike, *, expected_shape: tuple[int, int]) -> np.ndarray:
    """Return a copied positive ``int64`` label grid of the required shape.

    Args:
        values: Candidate label array.
        expected_shape: Required two-dimensional search-grid shape.

    Returns:
        Validated positive integer labels, detached from the input.

    Raises:
        ValueError: If dimensions, shape, finiteness, integrality, positivity,
            or the ``int64`` range requirement is violated.
    """
    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError("labels must be a two-dimensional search-grid array.")
    if raw.shape != expected_shape:
        raise ValueError(f"labels shape {raw.shape} must match search-grid shape {expected_shape}.")
    if raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise ValueError("labels must contain integral positive integer values.")

    if raw.dtype.kind in "iu":
        if np.any(raw <= 0) or np.any(raw > np.iinfo(np.int64).max):
            raise ValueError("labels must contain positive values representable as int64.")
        return np.asarray(raw, dtype=np.int64).copy()

    numeric = _finite_float_array(raw, name="labels")
    if np.any(numeric <= 0.0):
        raise ValueError("labels must cover every search cell with a positive value.")
    if np.any(numeric != np.floor(numeric)):
        raise ValueError("labels must contain only integral values.")
    if np.any(numeric > np.iinfo(np.int64).max):
        raise ValueError("labels must be representable as int64.")
    return numeric.astype(np.int64)


def _validated_model_arrays(
    model: RHIMEGaussianMultiscale,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return native design, support, and observation errors.

    Args:
        model: Source model whose native arrays and coarsening factor are
            checked for internal dimensional consistency.

    Returns:
        Finite native design, Boolean native support, and positive observation
        error diagonal.

    Raises:
        ValueError: If a source array or the coarsening factor is invalid.
    """
    native_design = _finite_float_array(model.native_design, name="model.native_design")
    native_support = np.asarray(model.native_support)
    r_diag = _finite_float_array(model.r_diag, name="model.r_diag")
    if native_design.ndim != 3:
        raise ValueError("model.native_design must have three dimensions.")
    if native_support.dtype.kind != "b" or native_support.shape != native_design.shape[1:]:
        raise ValueError("model.native_support must be a Boolean native spatial grid.")
    if r_diag.ndim != 1 or r_diag.shape[0] != native_design.shape[0] or np.any(r_diag <= 0.0):
        raise ValueError("model.r_diag must be positive with one value per observation.")
    if isinstance(model.coarsen_factor, bool) or not isinstance(model.coarsen_factor, (int, np.integer)):
        raise ValueError("model.coarsen_factor must be a positive integer.")
    if model.coarsen_factor <= 0:
        raise ValueError("model.coarsen_factor must be a positive integer.")
    return native_design, native_support, r_diag


def _group_supported_search_leaves(
    model: RHIMEGaussianMultiscale,
    label_grid: np.ndarray,
    *,
    observations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate supported search-leaf columns and counts by label.

    Args:
        model: Source model providing search-tree leaf columns and native
            support counts.
        label_grid: Validated labels with the source search-grid shape.
        observations: Expected number of design rows.

    Returns:
        Ascending supported region IDs, native support counts by region, and
        summed regional design columns in matching order.

    Raises:
        ValueError: If model leaf geometry, design columns, or support counts
            are inconsistent with the source model dimensions.
    """
    design = _finite_float_array(model.design.values, name="model.design.values")
    support_by_node = np.asarray(model.support_by_node)
    node_count = len(model.design.tree.nodes)
    if design.shape != (observations, node_count):
        raise ValueError("model.design.values shape must match its observations and tree nodes.")
    if support_by_node.shape != (node_count,) or support_by_node.dtype.kind not in "iu":
        raise ValueError("model.support_by_node must contain one integer count per tree node.")
    if np.any(support_by_node < 0):
        raise ValueError("model.support_by_node cannot contain negative counts.")

    leaf_ids = np.asarray(model.design.tree.leaf_ids, dtype=np.int64)
    leaf_labels = np.empty(leaf_ids.size, dtype=np.int64)
    for leaf_index, leaf_id in enumerate(leaf_ids):
        tile = model.design.tree.tile(int(leaf_id))
        if not tile.is_cell:
            raise ValueError("model search-tree leaves must each cover one search cell.")
        leaf_labels[leaf_index] = label_grid[tile.row_start, tile.col_start]

    leaf_support = np.asarray(support_by_node[leaf_ids], dtype=np.int64)
    retained = leaf_support > 0
    region_ids, inverse = np.unique(leaf_labels[retained], return_inverse=True)
    supported_counts = np.zeros(region_ids.size, dtype=np.int64)
    np.add.at(supported_counts, inverse, leaf_support[retained])

    regional_design = np.zeros((observations, region_ids.size), dtype=float)
    retained_leaf_ids = leaf_ids[retained]
    for region_index in range(region_ids.size):
        regional_design[:, region_index] = design[:, retained_leaf_ids[inverse == region_index]].sum(axis=1)
    return region_ids, supported_counts, regional_design


def _validated_partition_arrays(
    partition: GaussianPartitionDiagnostics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return arrays used by conjugate conditioning.

    Args:
        partition: Diagnostics whose regional design, prior variances, and
            effective covariance are checked.

    Returns:
        Finite regional design, positive prior variances, and a
        positive-definite effective observation covariance.

    Raises:
        ValueError: If dimensions, values, or positive definiteness is invalid.
    """
    design = _finite_float_array(partition.regional_design, name="partition.regional_design")
    variances = _finite_float_array(partition.prior_variances, name="partition.prior_variances")
    covariance = _finite_float_array(
        partition.effective_observation_covariance,
        name="partition.effective_observation_covariance",
    )
    if design.ndim != 2:
        raise ValueError("partition.regional_design must be two-dimensional.")
    if variances.ndim != 1 or variances.shape[0] != design.shape[1] or np.any(variances <= 0.0):
        raise ValueError("partition.prior_variances must be positive with one value per design column.")
    if covariance.shape != (design.shape[0], design.shape[0]):
        raise ValueError("partition effective covariance shape must match its observation count.")
    _positive_definite_cholesky(
        covariance,
        name="partition effective observation covariance",
        error_type=ValueError,
    )
    return design, variances, covariance


def _validated_baseline_inputs(
    baseline_design: npt.ArrayLike | None,
    baseline_prior_mean: npt.ArrayLike | None,
    baseline_prior_variances: npt.ArrayLike | None,
    *,
    observations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate optional baseline design and its explicit diagonal prior.

    Args:
        baseline_design: Optional observation-by-baseline design matrix.
        baseline_prior_mean: Required one-dimensional means when a baseline
            design is supplied.
        baseline_prior_variances: Required positive diagonal variances when a
            baseline design is supplied.
        observations: Required number of baseline design rows.

    Returns:
        Baseline design, means, and variances.  When baselines are omitted,
        each returned array has a zero-length coefficient dimension.

    Raises:
        ValueError: If baseline inputs are incomplete, non-finite,
            non-positive where required, or dimensionally incompatible.
    """
    if baseline_design is None:
        if baseline_prior_mean is not None or baseline_prior_variances is not None:
            raise ValueError("baseline priors require baseline_design.")
        return (
            np.empty((observations, 0), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )
    if baseline_prior_mean is None or baseline_prior_variances is None:
        raise ValueError("baseline_design requires explicit baseline prior mean and variances.")

    design = _finite_float_array(baseline_design, name="baseline_design")
    if design.ndim != 2 or design.shape[0] != observations or design.shape[1] == 0:
        raise ValueError("baseline_design must be a non-empty matrix with one row per observation.")
    means = _finite_float_array(baseline_prior_mean, name="baseline_prior_mean")
    variances = _finite_float_array(baseline_prior_variances, name="baseline_prior_variances")
    expected_shape = (design.shape[1],)
    if means.shape != expected_shape:
        raise ValueError("baseline_prior_mean must have one value per baseline column.")
    if variances.shape != expected_shape or np.any(variances <= 0.0):
        raise ValueError("baseline_prior_variances must be positive with one value per baseline column.")
    return design, means, variances


def _broadcast_mean(values: npt.ArrayLike, *, size: int, name: str) -> np.ndarray:
    """Return a finite scalar-broadcast or correctly sized mean vector.

    Args:
        values: Scalar or one-dimensional candidate means.
        size: Required output vector length.
        name: Input name used in validation errors.

    Returns:
        Copied finite mean vector of length ``size``.

    Raises:
        ValueError: If values are non-finite or have an incompatible shape.
    """
    means = _finite_float_array(values, name=name)
    if means.ndim == 0:
        return np.full(size, float(means), dtype=float)
    if means.shape != (size,):
        raise ValueError(f"{name} must be a scalar or have one value per retained emission region.")
    return means.copy()


def _training_indices(values: npt.ArrayLike | None, *, observations: int) -> np.ndarray:
    """Normalize a Boolean mask or unique integer training subset to indices."""
    return _subset_indices(values, observations=observations, name="training_subset")


def _subset_indices(
    values: npt.ArrayLike | None,
    *,
    observations: int,
    name: str,
) -> np.ndarray:
    """Normalize a named Boolean mask or unique integer subset to indices.

    Args:
        values: Optional Boolean mask or integer index array.  ``None`` selects
            every observation.
        observations: Total number of available observations.
        name: Input name used in validation errors.

    Returns:
        Non-empty one-dimensional integer observation indices.

    Raises:
        ValueError: If the subset has the wrong type or dimensions, is empty,
            contains duplicates, or references an unavailable observation.
    """
    if values is None:
        return np.arange(observations, dtype=np.int64)
    subset = np.asarray(values)
    if subset.dtype.kind == "b":
        if subset.shape != (observations,):
            raise ValueError(f"Boolean {name} must have one value per observation.")
        indices = np.flatnonzero(subset)
    elif subset.dtype.kind in "iu":
        if subset.ndim != 1:
            raise ValueError(f"Integer {name} must be one-dimensional.")
        if np.any(subset < 0) or np.any(subset >= observations):
            raise ValueError(f"{name} indices are outside the observation range.")
        indices = np.asarray(subset, dtype=np.int64)
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} indices must be unique.")
    else:
        raise ValueError(f"{name} must be a Boolean mask or integer indices.")
    if indices.size == 0:
        raise ValueError(f"{name} must select at least one observation.")
    return indices


def _finite_float_array(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Convert an array-like input to a finite real floating-point array."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validated_model_innovation(
    model: RHIMEGaussianMultiscale,
    *,
    observations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the invariant covariance and its validated stored factor.

    Args:
        model: Source model containing the invariant innovation arrays.
        observations: Required extent of both stored square arrays.

    Returns:
        Innovation covariance and its cached lower Cholesky factor.

    Raises:
        ValueError: If either array has the wrong shape or the cached factor is
            not finite, lower triangular, positive on its diagonal, and a
            factorization of the stored innovation covariance.
    """
    innovation = _finite_float_array(model.innovation_covariance, name="model.innovation_covariance")
    cholesky = _finite_float_array(model.innovation_cholesky, name="model.innovation_cholesky")
    expected_shape = (observations, observations)
    if innovation.shape != expected_shape:
        raise ValueError("model.innovation_covariance shape must match its observation count.")
    if cholesky.shape != expected_shape:
        raise ValueError("model.innovation_cholesky shape must match its observation count.")
    triangular_tolerance = float(
        64.0
        * np.finfo(float).eps
        * max(
            1.0,
            float(np.max(np.abs(cholesky), initial=0.0)),
        )
    )
    if not np.allclose(cholesky, np.tril(cholesky), rtol=0.0, atol=triangular_tolerance):
        raise ValueError("model.innovation_cholesky must be lower triangular.")
    if np.any(np.diag(cholesky) <= 0.0):
        raise ValueError("model.innovation_cholesky must have a positive diagonal.")
    covariance_scale = float(max(1.0, np.max(np.abs(innovation), initial=0.0)))
    if not np.allclose(
        cholesky @ cholesky.T,
        innovation,
        rtol=1e-10,
        atol=1e-12 * covariance_scale,
    ):
        raise ValueError(
            "model.innovation_cholesky must factor model.innovation_covariance."
        )
    return innovation, cholesky


def _covariance_kl_from_averaging_kernel(averaging_kernel: np.ndarray) -> float:
    """Return covariance-only Gaussian KL from averaging-kernel eigenvalues.

    Args:
        averaging_kernel: Symmetric prior-whitened averaging kernel whose
            eigenvalues are theoretically in the half-open interval ``[0, 1)``.

    Returns:
        Covariance KL with the conventional factor ``1/2``.

    Raises:
        ArithmeticError: If a mode is non-finite or outside the admissible
            positive-definite Gaussian range beyond numerical roundoff.
    """
    eigenvalues = np.linalg.eigvalsh(averaging_kernel)
    if not np.all(np.isfinite(eigenvalues)):
        raise ArithmeticError("averaging-kernel eigenvalues must be finite.")
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    tolerance = 64.0 * np.finfo(float).eps * scale
    if np.any(eigenvalues < -tolerance) or np.any(eigenvalues > 1.0 + tolerance):
        raise ArithmeticError("averaging-kernel eigenvalues must lie in [0, 1).")
    eigenvalues = np.clip(eigenvalues, 0.0, np.nextafter(1.0, 0.0))
    return 0.5 * float(np.sum(_negative_log1p_minus_identity(eigenvalues)))


def _negative_log1p_minus_identity(values: np.ndarray) -> np.ndarray:
    """Evaluate ``-log1p(-x) - x`` without losing weak quadratic terms."""
    result = np.empty_like(values)
    small = values < 1e-4
    small_values = values[small]
    result[small] = np.square(small_values) * (
        0.5 + small_values * (1.0 / 3.0 + small_values * (0.25 + small_values * (0.2 + small_values / 6.0)))
    )
    regular_values = values[~small]
    result[~small] = -np.log1p(-regular_values) - regular_values
    return result


def _centered_scatter(
    native_design: np.ndarray,
    native_indices: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Return stable observation-space scatter for selected native columns.

    Args:
        native_design: Native contribution grid with shape
            ``(observation, row, column)``.
        native_indices: Flattened native spatial indices in one region.
        chunk_size: Maximum number of selected columns centered at once.

    Returns:
        Sum of outer products of native columns around their regional mean.
    """
    observations = native_design.shape[0]
    scatter = np.zeros((observations, observations), dtype=float)
    mean = np.zeros(observations, dtype=float)
    count = 0
    flat_design = native_design.reshape(observations, -1)

    for start in range(0, native_indices.size, chunk_size):
        batch = flat_design[:, native_indices[start : start + chunk_size]]
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


def _positive_definite_cholesky(
    matrix: np.ndarray,
    *,
    name: str,
    error_type: type[ValueError] | type[ArithmeticError],
) -> np.ndarray:
    """Return a Cholesky factor or raise the requested domain error.

    Args:
        matrix: Candidate positive-definite matrix.
        name: Matrix name used in the exception message.
        error_type: ``ValueError`` or ``ArithmeticError`` class appropriate to
            the caller's validation domain.

    Returns:
        Lower-triangular Cholesky factor.

    Raises:
        ValueError: If factorization fails and ``error_type`` is ``ValueError``.
        ArithmeticError: If factorization fails and ``error_type`` is
            ``ArithmeticError``.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise error_type(f"{name} must be positive definite.") from exc


def _cholesky_solve(cholesky: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    """Solve a positive-definite system from its lower Cholesky factor."""
    intermediate = np.linalg.solve(cholesky, right_hand_side)
    return np.linalg.solve(cholesky.T, intermediate)


def _validate_positive_semidefinite(matrix: np.ndarray, *, name: str) -> None:
    """Raise when a symmetric covariance has a materially negative mode."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -1e-10 * scale:
        raise ArithmeticError(f"{name} is not positive semidefinite within numerical tolerance.")


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Remove floating-point asymmetry from a covariance matrix."""
    return 0.5 * (matrix + matrix.T)


__all__ = [
    "GaussianPartitionDiagnostics",
    "GaussianPartitionObjectives",
    "build_partition_diagnostics",
    "emissions_compression_quality",
    "gaussian_partition_objectives",
    "gaussian_posterior_mean",
]
