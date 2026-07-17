"""Reusable experimental diagnostics for dyadic resolution sweeps.

This module contains two independent diagnostic helpers used by controlled
dyadic basis experiments.  Temporal selection constructs blocked holdouts from
site identifiers and timestamps alone, with an optional symmetric exclusion
buffer and a shared wall-clock thinning lattice.  Spatial resolution helpers
decompose the full native-grid degrees of freedom for signal (DFS) into native
cell contributions and compare that upper bound with the exact partition made
from every leaf of a coarsened dyadic search grid.

The routines are deliberately NumPy-only and side-effect free.  They do not
inspect observed mole fractions, fitted baselines, or production inversion
state.  Returned temporal masks are copied and made read-only, and summary
records contain only immutable scalar metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import cast

import numpy as np
import numpy.typing as npt

from .rhime_gaussian import RHIMEGaussianMultiscale


@dataclass(frozen=True, slots=True)
class TemporalSelection:
    """Immutable masks and metadata for one blocked temporal holdout.

    The holdout interval is half-open: ``[holdout_start, holdout_stop)``.
    Buffer rows are excluded immediately before and after that interval, and
    thinning is applied only to otherwise eligible training rows.  When a
    stride is requested, ``stride_anchor`` is the earliest input timestamp and
    every retained training timestamp differs from it by an exact multiple of
    the stride.

    Attributes:
        training_mask: Read-only Boolean mask selecting rows used for training.
        holdout_mask: Read-only Boolean mask selecting held-out rows.
        holdout_start: Inclusive start of the holdout as nanosecond datetime.
        holdout_stop: Exclusive end of the holdout as nanosecond datetime.
        buffer_hours: Symmetric exclusion-buffer width in hours.
        thinning_hours: Optional positive integer training stride in hours.
        stride_anchor: Shared wall-clock thinning anchor, or ``None`` when
            thinning is disabled.
        buffer_excluded_count: Number of non-holdout rows excluded by the
            symmetric buffer before thinning.
        thinning_excluded_count: Number of otherwise eligible training rows
            excluded by the wall-clock stride.
    """

    training_mask: np.ndarray
    holdout_mask: np.ndarray
    holdout_start: np.datetime64
    holdout_stop: np.datetime64
    buffer_hours: float
    thinning_hours: int | None
    stride_anchor: np.datetime64 | None
    buffer_excluded_count: int
    thinning_excluded_count: int

    def __post_init__(self) -> None:
        """Defensively freeze mask arrays stored in the immutable record."""
        training = np.asarray(self.training_mask)
        holdout = np.asarray(self.holdout_mask)
        if training.ndim != 1 or holdout.shape != training.shape:
            raise ValueError("training_mask and holdout_mask must be equally shaped one-dimensional arrays.")
        if training.dtype != np.bool_ or holdout.dtype != np.bool_:
            raise ValueError("training_mask and holdout_mask must be Boolean arrays.")
        if np.any(training & holdout):
            raise ValueError("training and holdout masks must not overlap.")
        if not np.any(training) or not np.any(holdout):
            raise ValueError("training and holdout masks must both be nonempty.")

        frozen_training = training.copy()
        frozen_holdout = holdout.copy()
        frozen_training.setflags(write=False)
        frozen_holdout.setflags(write=False)
        object.__setattr__(self, "training_mask", frozen_training)
        object.__setattr__(self, "holdout_mask", frozen_holdout)

    @property
    def training_count(self) -> int:
        """Return the number of selected training rows."""
        return int(np.count_nonzero(self.training_mask))

    @property
    def training(self) -> np.ndarray:
        """Return the read-only Boolean training mask."""
        return self.training_mask

    @property
    def holdout_count(self) -> int:
        """Return the number of selected holdout rows."""
        return int(np.count_nonzero(self.holdout_mask))

    @property
    def holdout(self) -> np.ndarray:
        """Return the read-only Boolean holdout mask."""
        return self.holdout_mask

    @property
    def total_count(self) -> int:
        """Return the total number of input rows represented by the masks."""
        return int(self.training_mask.size)


def blocked_temporal_selection(
    sites: npt.ArrayLike,
    timestamps: npt.ArrayLike,
    *,
    holdout_start: np.datetime64,
    holdout_stop: np.datetime64,
    buffer_hours: float = 0.0,
    thinning_hours: int | None = None,
) -> TemporalSelection:
    """Select a buffered temporal holdout without inspecting observed values.

    All sites use the same half-open holdout and buffer intervals.  Optional
    thinning is based on elapsed wall-clock time from the earliest timestamp,
    rather than row number or a separate per-site origin, so coincident rows at
    different sites remain aligned.

    Args:
        sites: One-dimensional site identifiers, one per observation row.
        timestamps: One-dimensional NumPy datetime array, one per row.
        holdout_start: Inclusive start of the blocked holdout.
        holdout_stop: Exclusive end of the blocked holdout.
        buffer_hours: Non-negative symmetric buffer width in hours.  Rows in
            the buffer are selected by neither returned mask.
        thinning_hours: Optional positive integer wall-clock stride in hours,
            applied after holdout and buffer exclusion.

    Returns:
        Immutable temporal masks, counts, and normalized interval metadata.

    Raises:
        TypeError: If ``thinning_hours`` is not an integer or ``sites``
            contains an unhashable identifier.
        ValueError: If inputs have incompatible shapes, timestamps are not
            finite datetimes, a site/timestamp pair is duplicated, interval or
            buffer/stride values are invalid, masks overlap, or either final
            selection is empty.
    """
    site_array = np.asarray(sites)
    time_array = np.asarray(timestamps)
    if site_array.ndim != 1 or time_array.ndim != 1:
        raise ValueError("sites and timestamps must be one-dimensional arrays.")
    if site_array.shape != time_array.shape:
        raise ValueError("sites and timestamps must have the same shape.")
    if site_array.size == 0:
        raise ValueError("sites and timestamps must be nonempty.")
    if time_array.dtype.kind != "M":
        raise ValueError("timestamps must have a NumPy datetime dtype.")

    normalized_times = time_array.astype("datetime64[ns]")
    if np.any(np.isnat(normalized_times)):
        raise ValueError("timestamps must not contain NaT.")
    start = _datetime_scalar(holdout_start, name="holdout_start")
    stop = _datetime_scalar(holdout_stop, name="holdout_stop")
    if start >= stop:
        raise ValueError("holdout_start must be earlier than holdout_stop.")

    buffer = _nonnegative_real(buffer_hours, name="buffer_hours")
    buffer_delta = _hours_to_timedelta(buffer, name="buffer_hours")
    stride = _optional_positive_integer(thinning_hours, name="thinning_hours")
    _validate_unique_site_times(site_array, normalized_times)

    holdout_mask = (normalized_times >= start) & (normalized_times < stop)
    buffer_start = start - buffer_delta
    buffer_stop = stop + buffer_delta
    if np.isnat(buffer_start) or np.isnat(buffer_stop):
        raise ValueError("buffer_hours produces an unrepresentable datetime interval.")
    buffer_mask = ~holdout_mask & (normalized_times >= buffer_start) & (normalized_times < buffer_stop)
    eligible_training = ~(holdout_mask | buffer_mask)

    stride_anchor: np.datetime64 | None = None
    thinning_excluded_count = 0
    training_mask = eligible_training
    if stride is not None:
        stride_anchor = np.min(normalized_times)
        stride_delta = _hours_to_timedelta(float(stride), name="thinning_hours")
        on_stride = (normalized_times - stride_anchor) % stride_delta == np.timedelta64(0, "ns")
        training_mask = eligible_training & on_stride
        thinning_excluded_count = int(np.count_nonzero(eligible_training & ~on_stride))

    if np.any(training_mask & holdout_mask):
        raise ValueError("training and holdout selections must not overlap.")
    if not np.any(holdout_mask):
        raise ValueError("blocked holdout selection is empty.")
    if not np.any(training_mask):
        raise ValueError("training selection is empty after buffer exclusion and thinning.")

    return TemporalSelection(
        training_mask=training_mask,
        holdout_mask=holdout_mask,
        holdout_start=start,
        holdout_stop=stop,
        buffer_hours=buffer,
        thinning_hours=stride,
        stride_anchor=stride_anchor,
        buffer_excluded_count=int(np.count_nonzero(buffer_mask)),
        thinning_excluded_count=thinning_excluded_count,
    )


@dataclass(frozen=True, slots=True)
class CoarseningResolutionSummary:
    """Immutable summary of native DFS hidden inside search-grid leaves.

    One ordinary search leaf represents
    ``ordinary_block_width * ordinary_block_width`` native cells.  The final
    search row and column may represent smaller native dimensions, recorded by
    ``partial_final_row_height`` and ``partial_final_column_width``.

    Attributes:
        search_shape: Exact two-dimensional dyadic search-grid shape.
        ordinary_block_width: Native-cell width represented by an ordinary
            interior search leaf.
        partial_final_row_height: Native rows represented by the final search
            row; equal to the ordinary width when there is no remainder.
        partial_final_column_width: Native columns represented by the final
            search column; equal to the ordinary width when there is no
            remainder.
        full_grid_dfs: DFS of the supported native fine state.
        all_search_leaves_dfs: Exact DFS of the partition containing every
            search leaf, summed from node-indexed model scores.
        unresolved_dfs: Native DFS unavailable to the all-search-leaf
            partition.
        all_leaf_retained_fraction: All-search-leaf DFS divided by native DFS.
        top_native_cell_row: Native row index of the largest cell DFS.
        top_native_cell_column: Native column index of the largest cell DFS.
        top_native_cell_dfs: Largest native-cell DFS contribution.
        top_native_cell_fraction: Largest cell contribution divided by total
            native DFS.
        top_ten_native_cell_fraction: Sum of the ten largest cell contributions
            (or all cells when fewer than ten) divided by native DFS.
        block_dominant_cell_fraction: Sum of each search block's largest native
            cell contribution divided by total native DFS.
        maximum_within_nonzero_block_cell_fraction: Largest ratio of one cell's
            DFS to the native-cell DFS sum of its nonzero search block.
        supported_native_cell_count: Number of cells selected by the model's
            native support mask.
    """

    search_shape: tuple[int, int]
    ordinary_block_width: int
    partial_final_row_height: int
    partial_final_column_width: int
    full_grid_dfs: float
    all_search_leaves_dfs: float
    unresolved_dfs: float
    all_leaf_retained_fraction: float
    top_native_cell_row: int
    top_native_cell_column: int
    top_native_cell_dfs: float
    top_native_cell_fraction: float
    top_ten_native_cell_fraction: float
    block_dominant_cell_fraction: float
    maximum_within_nonzero_block_cell_fraction: float
    supported_native_cell_count: int


def native_cell_dfs(
    model: RHIMEGaussianMultiscale,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute native-cell DFS contributions in bounded column batches.

    For native design column ``g_i``, the returned value is
    ``tau**2 * g_i.T @ S**-1 @ g_i``, where ``S`` is the model's invariant
    innovation covariance.  A Cholesky factor of ``S`` is reused for each
    observation-by-batch solve, and unsupported cells remain exactly zero.

    Args:
        model: Gaussian multiscale model defining native columns, support,
            prior scale, and invariant innovation covariance.
        batch_size: Positive maximum number of native columns solved together.

    Returns:
        Two-dimensional native-grid array of non-negative DFS contributions.

    Raises:
        TypeError: If ``model`` has the wrong type or ``batch_size`` is not an
            integer.
        ValueError: If model dimensions or values are invalid, ``batch_size``
            is non-positive, or the innovation covariance is not positive
            definite.
        ArithmeticError: If contributions are invalid or their sum does not
            close to ``model.full_grid_dfs`` within numerical tolerance.
    """
    _validate_model_type(model)
    normalized_batch_size = _positive_integer(batch_size, name="batch_size")
    native_design, native_support, innovation = _validated_native_model_arrays(model)

    try:
        cholesky = np.linalg.cholesky(innovation)
    except np.linalg.LinAlgError as error:
        raise ValueError("model.innovation_covariance must be positive definite.") from error

    observations, rows, columns = native_design.shape
    flattened_design = native_design.reshape(observations, rows * columns)
    flattened_support = native_support.ravel()
    contributions = np.zeros(rows * columns, dtype=float)
    tau_squared = float(np.square(model.relative_prior_sd))
    if not np.isfinite(tau_squared) or tau_squared <= 0.0:
        raise ValueError("model.relative_prior_sd must define a finite positive variance.")

    for start in range(0, flattened_design.shape[1], normalized_batch_size):
        stop = min(start + normalized_batch_size, flattened_design.shape[1])
        batch_support = flattened_support[start:stop]
        if not np.any(batch_support):
            continue
        batch_indices = start + np.flatnonzero(batch_support)
        batch = flattened_design[:, batch_indices]
        whitened = np.linalg.solve(cholesky, batch)
        contributions[batch_indices] = tau_squared * np.einsum(
            "ij,ij->j",
            whitened,
            whitened,
        )

    if not np.all(np.isfinite(contributions)) or np.any(contributions < 0.0):
        raise ArithmeticError("native-cell DFS calculation produced invalid contributions.")
    total = float(np.sum(contributions))
    tolerance = _dfs_tolerance(model.full_grid_dfs)
    if not np.isclose(total, model.full_grid_dfs, rtol=1e-10, atol=tolerance):
        raise ArithmeticError("native-cell DFS contributions do not close to model.full_grid_dfs.")
    return contributions.reshape(rows, columns)


def summarize_coarsening_resolution(
    model: RHIMEGaussianMultiscale,
    cell_dfs: npt.ArrayLike,
) -> CoarseningResolutionSummary:
    """Summarize exact search-leaf geometry and native DFS concentration.

    The all-search-leaves score is read directly from ``model.tile_scores`` at
    ``model.design.tree.leaf_ids``.  It is therefore the exact additive model
    score for that partition, not a sum or approximation derived from native
    cell diagnostics.

    Args:
        model: Gaussian multiscale model whose coarsening geometry and exact
            node scores are summarized.
        cell_dfs: Two-dimensional native-cell contributions, normally returned
            by :func:`native_cell_dfs` for the same model.

    Returns:
        Immutable geometry, resolution-loss, concentration, and support
        diagnostics.

    Raises:
        TypeError: If ``model`` has the wrong type.
        ValueError: If model or contribution arrays have invalid dimensions,
            values, support, geometry, or a non-positive native DFS total.
        ArithmeticError: If native contributions fail DFS closure or exact
            all-leaf scores violate projection-consistent numerical bounds.
    """
    _validate_model_type(model)
    _, native_support, _ = _validated_native_model_arrays(model)
    contributions = _finite_real_array(cell_dfs, name="cell_dfs")
    if contributions.shape != native_support.shape:
        raise ValueError("cell_dfs shape must match model.native_support.")

    full_dfs = _finite_nonnegative_scalar(model.full_grid_dfs, name="model.full_grid_dfs")
    tolerance = _dfs_tolerance(full_dfs)
    if np.any(contributions < -tolerance):
        raise ValueError("cell_dfs must contain non-negative contributions.")
    contributions = np.maximum(contributions, 0.0)
    unsupported_scale = max(1.0, full_dfs)
    if not np.allclose(contributions[~native_support], 0.0, rtol=0.0, atol=1e-12 * unsupported_scale):
        raise ValueError("cell_dfs must be zero outside model.native_support.")
    cell_total = float(np.sum(contributions))
    if not np.isclose(cell_total, full_dfs, rtol=1e-10, atol=tolerance):
        raise ArithmeticError("cell_dfs does not close to model.full_grid_dfs.")
    if full_dfs <= tolerance:
        raise ValueError("model.full_grid_dfs must be positive to summarize DFS fractions.")

    factor = _positive_integer(model.coarsen_factor, name="model.coarsen_factor")
    native_rows, native_columns = native_support.shape
    search_shape = (
        int(model.design.tree.shape[0]),
        int(model.design.tree.shape[1]),
    )
    expected_search_shape = (
        (native_rows + factor - 1) // factor,
        (native_columns + factor - 1) // factor,
    )
    if search_shape != expected_search_shape:
        raise ValueError("model search shape is inconsistent with native shape and coarsen_factor.")
    partial_row_height = native_rows - factor * (search_shape[0] - 1)
    partial_column_width = native_columns - factor * (search_shape[1] - 1)

    tree = model.design.tree
    scores = _finite_real_array(model.tile_scores, name="model.tile_scores")
    if scores.shape != (len(tree.nodes),):
        raise ValueError("model.tile_scores must contain one value per tree node.")
    if np.any(scores < -tolerance):
        raise ValueError("model.tile_scores must be non-negative within numerical tolerance.")
    leaf_ids = np.asarray(tree.leaf_ids, dtype=np.int64)
    if leaf_ids.shape != (search_shape[0] * search_shape[1],):
        raise ValueError("model tree leaves must partition every search-grid cell exactly once.")
    all_leaves_dfs = float(np.sum(np.maximum(scores[leaf_ids], 0.0)))
    if all_leaves_dfs > full_dfs + tolerance:
        raise ArithmeticError("all-search-leaves DFS exceeds model.full_grid_dfs.")
    all_leaves_dfs = min(all_leaves_dfs, full_dfs)
    unresolved_dfs = max(full_dfs - all_leaves_dfs, 0.0)

    flat_contributions = contributions.ravel()
    top_flat_index = int(np.argmax(flat_contributions))
    top_row, top_column = np.unravel_index(top_flat_index, contributions.shape)
    top_dfs = float(flat_contributions[top_flat_index])
    top_count = min(10, flat_contributions.size)
    top_10_dfs = float(np.partition(flat_contributions, -top_count)[-top_count:].sum())
    block_dominant_sum, maximum_within_block = _block_concentration(contributions, factor)

    fractions = (
        all_leaves_dfs / full_dfs,
        top_dfs / full_dfs,
        top_10_dfs / full_dfs,
        block_dominant_sum / full_dfs,
        maximum_within_block,
    )
    if any(not np.isfinite(value) or value < -1e-12 or value > 1.0 + 1e-10 for value in fractions):
        raise ArithmeticError("coarsening resolution fractions lie outside numerical bounds.")

    return CoarseningResolutionSummary(
        search_shape=search_shape,
        ordinary_block_width=factor,
        partial_final_row_height=partial_row_height,
        partial_final_column_width=partial_column_width,
        full_grid_dfs=full_dfs,
        all_search_leaves_dfs=all_leaves_dfs,
        unresolved_dfs=unresolved_dfs,
        all_leaf_retained_fraction=float(np.clip(fractions[0], 0.0, 1.0)),
        top_native_cell_row=int(top_row),
        top_native_cell_column=int(top_column),
        top_native_cell_dfs=top_dfs,
        top_native_cell_fraction=float(np.clip(fractions[1], 0.0, 1.0)),
        top_ten_native_cell_fraction=float(np.clip(fractions[2], 0.0, 1.0)),
        block_dominant_cell_fraction=float(np.clip(fractions[3], 0.0, 1.0)),
        maximum_within_nonzero_block_cell_fraction=float(np.clip(fractions[4], 0.0, 1.0)),
        supported_native_cell_count=int(np.count_nonzero(native_support)),
    )


def _validate_unique_site_times(sites: np.ndarray, timestamps: np.ndarray) -> None:
    """Reject duplicate site/timestamp pairs using exact site equality.

    Args:
        sites: One-dimensional site identifiers.
        timestamps: Normalized nanosecond timestamps with matching shape.

    Raises:
        TypeError: If a site identifier is not hashable.
        ValueError: If any site/timestamp pair occurs more than once.
    """
    seen: set[tuple[object, object, int]] = set()
    integer_times = timestamps.astype(np.int64)
    for row_index in range(sites.size):
        raw_site = sites[row_index]
        site = raw_site.item() if isinstance(raw_site, np.generic) else raw_site
        timestamp = int(integer_times[row_index])
        key = (type(site), site, timestamp)
        try:
            duplicate = key in seen
        except TypeError as error:
            raise TypeError("site identifiers must be hashable.") from error
        if duplicate:
            raise ValueError("site/timestamp pairs must be unique.")
        seen.add(key)


def _datetime_scalar(value: np.datetime64, *, name: str) -> np.datetime64:
    """Validate and normalize one finite NumPy datetime scalar.

    Args:
        value: Candidate scalar datetime.
        name: Parameter name used in validation errors.

    Returns:
        Nanosecond-resolution NumPy datetime scalar.

    Raises:
        ValueError: If the value is not one finite datetime scalar.
    """
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind != "M":
        raise ValueError(f"{name} must be a NumPy datetime scalar.")
    normalized = cast(np.datetime64, array.astype("datetime64[ns]")[()])
    if np.isnat(normalized):
        raise ValueError(f"{name} must not be NaT.")
    return normalized


def _hours_to_timedelta(value: float, *, name: str) -> np.timedelta64:
    """Convert finite hours to a nanosecond timedelta without overflow.

    Args:
        value: Validated non-negative hours.
        name: Parameter name used in validation errors.

    Returns:
        Equivalent nanosecond-resolution timedelta.

    Raises:
        ValueError: If the hour value cannot be represented in nanoseconds.
    """
    nanoseconds = value * 3_600_000_000_000.0
    if not np.isfinite(nanoseconds) or nanoseconds > np.iinfo(np.int64).max:
        raise ValueError(f"{name} is too large to represent as a timedelta.")
    return np.timedelta64(int(round(nanoseconds)), "ns")


def _optional_positive_integer(value: int | None, *, name: str) -> int | None:
    """Validate an optional positive integer parameter."""
    if value is None:
        return None
    return _positive_integer(value, name=name)


def _positive_integer(value: int, *, name: str) -> int:
    """Validate and normalize one positive integer parameter."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer.")
    try:
        normalized = index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a positive integer.") from error
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def _nonnegative_real(value: float, *, name: str) -> float:
    """Validate and normalize one finite non-negative real scalar."""
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a finite non-negative scalar.")
    normalized = float(array)
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar.")
    return normalized


def _validate_model_type(model: RHIMEGaussianMultiscale) -> None:
    """Require the concrete experimental Gaussian multiscale model."""
    if not isinstance(model, RHIMEGaussianMultiscale):
        raise TypeError("model must be an RHIMEGaussianMultiscale.")


def _validated_native_model_arrays(
    model: RHIMEGaussianMultiscale,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate model arrays needed by native resolution diagnostics.

    Args:
        model: Concrete Gaussian multiscale model.

    Returns:
        Native design, Boolean native support, and symmetric innovation
        covariance as finite NumPy arrays.

    Raises:
        ValueError: If dimensions, dtypes, values, support masking, or
            covariance symmetry are invalid.
    """
    native_design = _finite_real_array(model.native_design, name="model.native_design")
    native_support = np.asarray(model.native_support)
    innovation = _finite_real_array(model.innovation_covariance, name="model.innovation_covariance")
    if native_design.ndim != 3 or any(extent == 0 for extent in native_design.shape):
        raise ValueError("model.native_design must have nonempty (observation, row, column) shape.")
    if native_support.dtype != np.bool_ or native_support.shape != native_design.shape[1:]:
        raise ValueError("model.native_support must be Boolean and match the native spatial shape.")
    observations = native_design.shape[0]
    if innovation.shape != (observations, observations):
        raise ValueError("model.innovation_covariance shape must match native observations.")
    covariance_scale = max(1.0, float(np.max(np.abs(innovation))))
    if not np.allclose(innovation, innovation.T, rtol=1e-12, atol=1e-12 * covariance_scale):
        raise ValueError("model.innovation_covariance must be symmetric.")
    if np.any(native_design[:, ~native_support] != 0.0):
        raise ValueError("model.native_design must be zero outside native support.")
    return native_design, native_support, innovation


def _finite_real_array(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Convert an array-like input to a finite real floating-point array."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _finite_nonnegative_scalar(value: float, *, name: str) -> float:
    """Validate and normalize one finite non-negative real scalar."""
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a finite non-negative scalar.")
    normalized = float(array)
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar.")
    return normalized


def _dfs_tolerance(reference: float) -> float:
    """Return an absolute tolerance scaled to one DFS reference value."""
    return 1e-12 * max(1.0, abs(float(reference)))


def _block_concentration(contributions: np.ndarray, factor: int) -> tuple[float, float]:
    """Aggregate dominant-cell diagnostics over native coarsening blocks.

    Args:
        contributions: Validated non-negative native two-dimensional DFS.
        factor: Positive native-cell width of ordinary search blocks.

    Returns:
        Sum of each block maximum and the largest maximum-to-sum ratio among
        blocks with positive native DFS.
    """
    rows, columns = contributions.shape
    dominant_sum = 0.0
    maximum_fraction = 0.0
    for row_start in range(0, rows, factor):
        for column_start in range(0, columns, factor):
            block = contributions[
                row_start : min(row_start + factor, rows),
                column_start : min(column_start + factor, columns),
            ]
            block_sum = float(np.sum(block))
            if block_sum <= 0.0:
                continue
            block_maximum = float(np.max(block))
            dominant_sum += block_maximum
            maximum_fraction = max(maximum_fraction, block_maximum / block_sum)
    return dominant_sum, maximum_fraction
