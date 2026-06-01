"""Experimental dyadic basis-generation examples.

This module is intentionally not part of the public basis-generation API. It is
a cleaned-up reference extracted from exploratory IPython histories in
``~/Documents/basis_functions`` so that issue discussions can point at runnable
code instead of scratch notebooks.

The prototype material had three reusable ideas:

1. A thresholded dyadic bisection scheme backed by a precomputed multiscale
   weight array. This is the same core idea as the existing weighted basis
   algorithm in
   :mod:`openghg_inversions.basis.algorithms._weighted`: split a 2D weight grid
   along the longer axis until each region is below a bucket/threshold value.
   The example here keeps the geometry and threshold search small and explicit.
   ``DyadicWeightArray.values`` follows the prototype's ``make_multi`` idea:
   rows and columns are dyadic intervals at every scale, so a tile weight is an
   array lookup instead of a fresh grid slice sum. This deliberately omits
   production details such as land/sea splitting.
2. A post-bisection local search. The existing quadtree algorithm uses
   ``scipy.optimize.dual_annealing`` to tune the quadtree threshold. The
   prototype explored a different follow-up step: start from a dyadic tiling and
   propose split/merge moves, accepting some worse moves according to a
   simulated-annealing-style temperature.
3. A numba implementation of the bisection loop. This is a compiled version of
   the core weighted-basis split operation, not a compiled version of the full
   ``nregion_landsea_basis`` wrapper. It keeps xarray, land/sea masks, and
   threshold optimization outside the jitted kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import exp
from typing import Any, Iterable, cast

import numpy as np
import numpy.typing as npt

try:  # pragma: no cover - exercised only when numba is unavailable.
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    njit = None  # type: ignore[assignment]
    _HAS_NUMBA = False


ArrayLike2D = npt.ArrayLike
ThresholdKernel = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True, order=True)
class Tile:
    """Rectangular tile in padded row/column index space."""

    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    @property
    def height(self) -> int:
        """Number of rows covered by the tile."""
        return self.row_stop - self.row_start

    @property
    def width(self) -> int:
        """Number of columns covered by the tile."""
        return self.col_stop - self.col_start

    @property
    def area(self) -> int:
        """Number of grid cells covered by the tile."""
        return self.height * self.width

    def intersects_shape(self, shape: tuple[int, int]) -> bool:
        """Return whether this padded tile intersects an unpadded shape."""
        nrow, ncol = shape
        return self.row_start < nrow and self.col_start < ncol

    def clipped(self, shape: tuple[int, int]) -> Tile | None:
        """Clip this tile to an unpadded shape."""
        if not self.intersects_shape(shape):
            return None
        row_stop = min(self.row_stop, shape[0])
        col_stop = min(self.col_stop, shape[1])
        if self.row_start >= row_stop or self.col_start >= col_stop:
            return None
        return Tile(self.row_start, row_stop, self.col_start, col_stop)


@dataclass(frozen=True)
class AnnealingResult:
    """Result returned by :func:`anneal_dyadic_basis`."""

    labels: np.ndarray
    initial_energy: float
    best_energy: float
    accepted_moves: int
    initial_regions: int
    final_regions: int


@dataclass(frozen=True)
class DyadicWeightArray:
    """Precomputed sums for all dyadic row/column interval pairs.

    ``values[row_interval, col_interval]`` stores the total weight for that
    dyadic rectangle. Intervals are ordered from finest to coarsest scale: all
    length-1 intervals, then all length-2 intervals, and so on up to the root.
    """

    values: np.ndarray
    intervals: tuple[tuple[int, int], ...]
    interval_index: dict[tuple[int, int], int]
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]

    def weight(self, tile: Tile) -> float:
        """Return the precomputed weight for a dyadic tile."""
        row_idx = self.interval_index[(tile.row_start, tile.row_stop)]
        col_idx = self.interval_index[(tile.col_start, tile.col_stop)]
        return float(self.values[row_idx, col_idx])


# -----------------------------------------------------------------------------
# Idea 1: weighted-style dyadic bisection and threshold search
# -----------------------------------------------------------------------------


def dyadic_threshold_basis(
    weights: ArrayLike2D,
    threshold: float,
    *,
    use_numba: bool = False,
) -> np.ndarray:
    """Create a dyadic rectangular basis by thresholded recursive bisection.

    Args:
        weights: Non-negative 2D weight field.
        threshold: Split tiles while their total weight is greater than this
            value and they can still be bisected.
        use_numba: If True, use the numba implementation of the split loop.

    Returns:
        Integer labels, starting at 1, with the same shape as ``weights``.
    """
    weight_array = _as_2d_weight_array(weights)
    padded = _pad_to_power_of_two_square(weight_array)

    if use_numba:
        if not _HAS_NUMBA:
            raise ImportError("numba is required for use_numba=True.")
        labels = _dyadic_threshold_labels_numba(padded, float(threshold))
        labels = labels[: weight_array.shape[0], : weight_array.shape[1]]
        return _relabel_positive(labels)

    dyadic_weights = make_dyadic_weight_array(weight_array)
    tiles = _dyadic_threshold_tiles_from_weights(dyadic_weights, threshold)
    return tiles_to_labels(tiles, weight_array.shape)


def make_dyadic_weight_array(weights: ArrayLike2D) -> DyadicWeightArray:
    """Precompute weights for every dyadic tile in a padded square grid."""
    weight_array = _as_2d_weight_array(weights)
    padded = _pad_to_power_of_two_square(weight_array)
    values = _make_multiscale_axis(_make_multiscale_axis(padded, axis=0), axis=1)
    intervals = _dyadic_intervals(padded.shape[0])
    return DyadicWeightArray(
        values=values,
        intervals=intervals,
        interval_index={interval: idx for idx, interval in enumerate(intervals)},
        original_shape=weight_array.shape,
        padded_shape=padded.shape,
    )


def dyadic_threshold_tiles(weights: ArrayLike2D, threshold: float) -> list[Tile]:
    """Return leaf tiles from thresholded dyadic bisection."""
    dyadic_weights = make_dyadic_weight_array(weights)
    return _dyadic_threshold_tiles_from_weights(dyadic_weights, threshold)


def _dyadic_threshold_tiles_from_weights(
    dyadic_weights: DyadicWeightArray,
    threshold: float,
) -> list[Tile]:
    root = Tile(0, dyadic_weights.padded_shape[0], 0, dyadic_weights.padded_shape[1])
    stack = [root]
    leaves: list[Tile] = []

    while stack:
        tile = stack.pop()
        tile_weight = _tile_weight(dyadic_weights, tile)
        children = split_tile(tile)
        if tile_weight <= threshold or children is None:
            leaves.append(tile)
            continue

        # Push second child first so the first child is processed first.
        first, second = children
        stack.append(second)
        stack.append(first)

    return leaves


def dyadic_target_basis(
    weights: ArrayLike2D,
    target_regions: int,
    *,
    max_iter: int = 32,
    use_numba: bool = False,
) -> np.ndarray:
    """Approximate ``target_regions`` by searching over split thresholds.

    The region count is a step function of the threshold, so exact targets are
    not always attainable. This helper returns the closest count found during a
    bounded binary search.
    """
    if target_regions < 1:
        raise ValueError("target_regions must be at least 1.")

    weight_array = _as_2d_weight_array(weights)
    low = 0.0
    high = float(weight_array.sum())
    dyadic_weights = make_dyadic_weight_array(weight_array)
    if use_numba:
        best_labels = dyadic_threshold_basis(weight_array, high, use_numba=True)
    else:
        best_labels = _dyadic_threshold_basis_from_weights(dyadic_weights, high)
    best_error = abs(int(best_labels.max()) - target_regions)

    for _ in range(max_iter):
        threshold = (low + high) / 2.0
        if use_numba:
            labels = dyadic_threshold_basis(weight_array, threshold, use_numba=True)
        else:
            labels = _dyadic_threshold_basis_from_weights(dyadic_weights, threshold)
        nregions = int(labels.max())
        error = abs(nregions - target_regions)
        if error < best_error:
            best_labels = labels
            best_error = error
            if error == 0:
                break

        if nregions > target_regions:
            low = threshold
        else:
            high = threshold

    return best_labels


def _dyadic_threshold_basis_from_weights(
    dyadic_weights: DyadicWeightArray,
    threshold: float,
) -> np.ndarray:
    tiles = _dyadic_threshold_tiles_from_weights(dyadic_weights, threshold)
    return tiles_to_labels(tiles, dyadic_weights.original_shape)


# -----------------------------------------------------------------------------
# Idea 2: simulated-annealing-style refinement after bisection
# -----------------------------------------------------------------------------


def anneal_dyadic_basis(
    weights: ArrayLike2D,
    *,
    initial_threshold: float,
    target_regions: int,
    iterations: int = 100,
    temperature: float = 0.1,
    region_penalty: float = 1.0,
    seed: int | None = None,
) -> AnnealingResult:
    """Run a small simulated-annealing-style refinement of dyadic tiles.

    This is a compact demonstration of the split/merge local-search idea from
    the prototype. The score is deliberately simple: it favours tile sets with
    high weighted information density while penalising region counts away from
    ``target_regions``.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if target_regions < 1:
        raise ValueError("target_regions must be at least 1.")

    weight_array = _as_2d_weight_array(weights)
    dyadic_weights = make_dyadic_weight_array(weight_array)
    rng = np.random.default_rng(seed)

    current = _dyadic_threshold_tiles_from_weights(dyadic_weights, initial_threshold)
    current_energy = _tiling_energy(current, dyadic_weights, target_regions, region_penalty)
    initial_energy = current_energy
    best = list(current)
    best_energy = current_energy
    accepted = 0

    for _ in range(iterations):
        proposal = _propose_tile_move(current, rng)
        if proposal is None:
            continue

        proposal_energy = _tiling_energy(proposal, dyadic_weights, target_regions, region_penalty)
        delta = proposal_energy - current_energy

        accept = delta <= 0.0
        if not accept and temperature > 0.0:
            accept = rng.random() < exp(-delta / temperature)

        if not accept:
            continue

        current = proposal
        current_energy = proposal_energy
        accepted += 1
        if current_energy < best_energy:
            best = list(current)
            best_energy = current_energy

    labels = tiles_to_labels(best, weight_array.shape)
    return AnnealingResult(
        labels=labels,
        initial_energy=initial_energy,
        best_energy=best_energy,
        accepted_moves=accepted,
        initial_regions=len(_dyadic_threshold_tiles_from_weights(dyadic_weights, initial_threshold)),
        final_regions=int(labels.max()),
    )


# -----------------------------------------------------------------------------
# Shared tile and scoring helpers used by the examples above
# -----------------------------------------------------------------------------


def split_tile(tile: Tile) -> tuple[Tile, Tile] | None:
    """Bisect a tile along its longer dyadic axis."""
    if tile.height <= 1 and tile.width <= 1:
        return None

    if tile.height >= tile.width and tile.height > 1:
        midpoint = tile.row_start + tile.height // 2
        return (
            Tile(tile.row_start, midpoint, tile.col_start, tile.col_stop),
            Tile(midpoint, tile.row_stop, tile.col_start, tile.col_stop),
        )

    midpoint = tile.col_start + tile.width // 2
    return (
        Tile(tile.row_start, tile.row_stop, tile.col_start, midpoint),
        Tile(tile.row_start, tile.row_stop, midpoint, tile.col_stop),
    )


def tiles_to_labels(tiles: Iterable[Tile], shape: tuple[int, int]) -> np.ndarray:
    """Render a tile collection as a compact positive integer label array."""
    labels = np.zeros(shape, dtype=np.int64)
    label = 1
    for tile in tiles:
        clipped = tile.clipped(shape)
        if clipped is None:
            continue
        labels[clipped.row_start : clipped.row_stop, clipped.col_start : clipped.col_stop] = label
        label += 1
    return labels


def _as_2d_weight_array(weights: ArrayLike2D) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("weights must be a 2D array.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("weights must not be empty.")
    if not np.isfinite(arr).all():
        raise ValueError("weights must be finite.")
    if (arr < 0.0).any():
        raise ValueError("weights must be non-negative.")
    return np.ascontiguousarray(arr)


def _next_power_of_two(value: int) -> int:
    size = 1
    while size < value:
        size *= 2
    return size


def _pad_to_power_of_two_square(weights: np.ndarray) -> np.ndarray:
    size = _next_power_of_two(max(weights.shape))
    padded = np.zeros((size, size), dtype=np.float64)
    padded[: weights.shape[0], : weights.shape[1]] = weights
    return padded


def _dyadic_intervals(size: int) -> tuple[tuple[int, int], ...]:
    intervals: list[tuple[int, int]] = []
    block_size = 1
    while block_size <= size:
        intervals.extend((start, start + block_size) for start in range(0, size, block_size))
        block_size *= 2
    return tuple(intervals)


def _sum_adjacent_pairs(arr: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(arr, axis, -1)
    shape = moved.shape
    summed = moved.reshape(*shape[:-1], shape[-1] // 2, 2).sum(axis=-1)
    return np.moveaxis(summed, -1, axis)


def _make_multiscale_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    levels = [arr]
    current = arr
    while current.shape[axis] > 1:
        current = _sum_adjacent_pairs(current, axis=axis)
        levels.append(current)
    return np.concatenate(levels, axis=axis)


def _tile_weight(weights: DyadicWeightArray, tile: Tile) -> float:
    return weights.weight(tile)


def _tile_information_score(weights: DyadicWeightArray, tile: Tile) -> float:
    total = _tile_weight(weights, tile)
    # A toy DOF-like score: splitting a high-weight tile can improve score by
    # reducing the square-root area normalisation.
    return total / np.sqrt(max(tile.area, 1))


def _tiling_energy(
    tiles: list[Tile],
    weights: DyadicWeightArray,
    target_regions: int,
    region_penalty: float,
) -> float:
    score = sum(_tile_information_score(weights, tile) for tile in tiles)
    region_error = len(tiles) - target_regions
    return -score + region_penalty * float(region_error * region_error)


def _propose_tile_move(tiles: list[Tile], rng: np.random.Generator) -> list[Tile] | None:
    split_candidates = [idx for idx, tile in enumerate(tiles) if split_tile(tile) is not None]
    merge_candidates = _mergeable_pairs(tiles)

    if split_candidates and (not merge_candidates or rng.random() < 0.5):
        idx = int(rng.choice(split_candidates))
        children = split_tile(tiles[idx])
        if children is None:
            return None
        proposal = list(tiles)
        proposal[idx : idx + 1] = list(children)
        return proposal

    if merge_candidates:
        pair_idx = int(rng.integers(0, len(merge_candidates)))
        idx1, idx2, merged = merge_candidates[pair_idx]
        proposal = [tile for idx, tile in enumerate(tiles) if idx not in {idx1, idx2}]
        proposal.append(merged)
        proposal.sort()
        return proposal

    return None


def _mergeable_pairs(tiles: list[Tile]) -> list[tuple[int, int, Tile]]:
    pairs: list[tuple[int, int, Tile]] = []
    for idx1, first in enumerate(tiles):
        for idx2 in range(idx1 + 1, len(tiles)):
            second = tiles[idx2]
            merged = _merge_tiles(first, second)
            if merged is not None:
                pairs.append((idx1, idx2, merged))
    return pairs


def _merge_tiles(first: Tile, second: Tile) -> Tile | None:
    same_rows = first.row_start == second.row_start and first.row_stop == second.row_stop
    adjacent_cols = first.col_stop == second.col_start or second.col_stop == first.col_start
    equal_width = first.width == second.width
    if same_rows and adjacent_cols and equal_width:
        return Tile(
            first.row_start,
            first.row_stop,
            min(first.col_start, second.col_start),
            max(first.col_stop, second.col_stop),
        )

    same_cols = first.col_start == second.col_start and first.col_stop == second.col_stop
    adjacent_rows = first.row_stop == second.row_start or second.row_stop == first.row_start
    equal_height = first.height == second.height
    if same_cols and adjacent_rows and equal_height:
        return Tile(
            min(first.row_start, second.row_start),
            max(first.row_stop, second.row_stop),
            first.col_start,
            first.col_stop,
        )

    return None


def _relabel_positive(labels: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    out = np.zeros_like(labels, dtype=np.int64)
    for new_label, old_label in enumerate(unique_labels, start=1):
        out[labels == old_label] = new_label
    return out


# -----------------------------------------------------------------------------
# Idea 3: numba-compiled core of the weighted-style bisection loop
# -----------------------------------------------------------------------------


def _dyadic_threshold_labels_kernel(weights: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.zeros(weights.shape, dtype=np.int64)
    max_stack = weights.shape[0] * weights.shape[1] * 2
    stack = np.empty((max_stack, 4), dtype=np.int64)
    top = 0
    stack[top, 0] = 0
    stack[top, 1] = weights.shape[0]
    stack[top, 2] = 0
    stack[top, 3] = weights.shape[1]
    top += 1
    label = 1

    while top > 0:
        top -= 1
        row_start = stack[top, 0]
        row_stop = stack[top, 1]
        col_start = stack[top, 2]
        col_stop = stack[top, 3]

        total = 0.0
        for row in range(row_start, row_stop):
            for col in range(col_start, col_stop):
                total += weights[row, col]

        height = row_stop - row_start
        width = col_stop - col_start
        can_split = height > 1 or width > 1

        if total <= threshold or not can_split:
            for row in range(row_start, row_stop):
                for col in range(col_start, col_stop):
                    labels[row, col] = label
            label += 1
            continue

        if height >= width and height > 1:
            midpoint = row_start + height // 2
            stack[top, 0] = midpoint
            stack[top, 1] = row_stop
            stack[top, 2] = col_start
            stack[top, 3] = col_stop
            top += 1
            stack[top, 0] = row_start
            stack[top, 1] = midpoint
            stack[top, 2] = col_start
            stack[top, 3] = col_stop
            top += 1
        else:
            midpoint = col_start + width // 2
            stack[top, 0] = row_start
            stack[top, 1] = row_stop
            stack[top, 2] = midpoint
            stack[top, 3] = col_stop
            top += 1
            stack[top, 0] = row_start
            stack[top, 1] = row_stop
            stack[top, 2] = col_start
            stack[top, 3] = midpoint
            top += 1

    return labels


def _missing_numba_threshold_labels(weights: np.ndarray, threshold: float) -> np.ndarray:
    raise ImportError("numba is required for use_numba=True.")


if _HAS_NUMBA:
    _njit = cast(Any, njit)
    _dyadic_threshold_labels_numba: ThresholdKernel = cast(
        ThresholdKernel,
        _njit(cache=True)(_dyadic_threshold_labels_kernel),
    )
else:
    _dyadic_threshold_labels_numba = _missing_numba_threshold_labels


__all__ = [
    "AnnealingResult",
    "DyadicWeightArray",
    "Tile",
    "anneal_dyadic_basis",
    "dyadic_target_basis",
    "dyadic_threshold_basis",
    "dyadic_threshold_tiles",
    "make_dyadic_weight_array",
    "split_tile",
    "tiles_to_labels",
]
