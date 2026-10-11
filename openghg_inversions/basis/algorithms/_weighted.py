"""Weighted bucket algorithms for basis-region construction.

If the total (sum) of the input data in a region exceeds a certain threshold, then the region
is split in two. This continues recursively until we have a collection of regions whose totals
are all below the threshold.

The threshold is optimised to create a specific number of regions.
:class:`AxisAlignedWeightedSplitStrategy` adapts this recursion to one Boolean
basis-group mask without loading land/sea data. All land/sea-aware kernels
receive an explicitly resolved mask. Resource selection and file reading happen
at the application boundary.
"""

from dataclasses import dataclass
import logging

import numpy as np



logger = logging.getLogger(__name__)


class OptimizationError(Exception): ...



def bucket_value_split(
    grid: np.ndarray,
    bucket: float,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[tuple]:
    """Algorithm that will split the input grid (e.g. fp * flux).

    Split such that the sum of each basis function region will equal the bucket value
    or by a single array element.

    The number of regions will be determined by the bucket value:
    i.e. smaller bucket value ==> more regions, larger bucket value ==> fewer regions.

    Args:
        grid: 2D grid of footprints * flux, or whatever grid you want to split.
            Could be: population data, spatial distribution of bakeries, you chose!
        bucket: Maximum value for each basis function region.
        offset_x: Start index of the region on first axis of the grid. Default 0.
        offset_y: Start index of the region on second axis of the grid. Default 0.

    Returns:
        list: List of tuples that define the indices for each basis function region
            [(ymin0, ymax0, xmin0, xmax0), ..., (yminN, ymaxN, xminN, xmaxN)]
    """
    if np.sum(grid) <= bucket or grid.shape == (1, 1):
        return [(offset_y, offset_y + grid.shape[0], offset_x, offset_x + grid.shape[1])]

    # grid total too large; split on longer axis
    if grid.shape[0] >= grid.shape[1]:
        half_y = grid.shape[0] // 2
        return bucket_value_split(grid[0:half_y, :], bucket, offset_x, offset_y) + bucket_value_split(
            grid[half_y:, :], bucket, offset_x, offset_y + half_y
        )

    # else: grid.shape[0] < grid.shape[1]:
    half_x = grid.shape[1] // 2
    return bucket_value_split(grid[:, 0:half_x], bucket, offset_x, offset_y) + bucket_value_split(
        grid[:, half_x:], bucket, offset_x + half_x, offset_y
    )


def _labels_for_bucket(weights: np.ndarray, class_mask: np.ndarray, bucket: float) -> np.ndarray:
    """Apply the rectangular weighted splitter and mask labels to one class.

    Args:
        weights: Class-local weight field, with cells outside the class normally
            set to zero.
        class_mask: Boolean mask selecting cells in the current class.
        bucket: Maximum rectangular bucket weight passed to
            :func:`bucket_value_split`.

    Returns:
        Integer label array with positive labels only inside ``class_mask``.
    """
    labels = np.zeros(weights.shape, dtype=np.int64)
    label = 1
    for ymin, ymax, xmin, xmax in bucket_value_split(weights, bucket):
        region_mask = class_mask[ymin:ymax, xmin:xmax]
        if not region_mask.any():
            continue
        label_slice = labels[ymin:ymax, xmin:xmax]
        label_slice[region_mask] = label
        label += 1
    return labels


def _count_positive_labels(labels: np.ndarray) -> int:
    """Return the number of positive labels in an integer label array."""
    return int(np.count_nonzero(np.unique(labels) > 0))


@dataclass(frozen=True)
class AxisAlignedWeightedSplitStrategy:
    """Class-local strategy derived from recursive weighted bucket splitting.

    This applies the existing bucket splitter independently to the masked
    weights for one class. It recursively splits rectangles along the longer
    axis until each rectangle is below a searched threshold. It is not a
    compatibility implementation of the legacy weighted land/sea pipeline,
    which optimizes the bucket layout before applying the land/sea split.
    The default region-constrained strategy is
    :class:`~openghg_inversions.basis.algorithms.GreedySplitStrategy` composed
    with :class:`~openghg_inversions.basis.algorithms.AxisParallelSplitStep`
    instead.

    Attributes:
        max_iter: Maximum number of threshold-search iterations.
    """

    max_iter: int = 32

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Return class-local labels using recursive weighted bucket splits.

        Args:
            weights: Non-negative weight field for the full grid.
            class_mask: Boolean mask selecting the cells in the class being
                split.
            target_regions: Target number of local labels for this class.

        Returns:
            Integer label array with positive class-local labels inside
            ``class_mask`` and zero outside it.

        Raises:
            ValueError: If ``target_regions`` is less than one.
        """
        if target_regions < 1:
            raise ValueError("target_regions must be at least 1.")
        if not class_mask.any():
            return np.zeros(weights.shape, dtype=np.int64)

        class_weights = np.where(class_mask, weights, 0.0)
        total_weight = float(class_weights.sum())
        if total_weight == 0.0:
            class_weights = class_mask.astype(np.float64)
            total_weight = float(class_weights.sum())

        low = 0.0
        high = total_weight
        best = _labels_for_bucket(class_weights, class_mask, bucket=high)
        best_error = abs(_count_positive_labels(best) - target_regions)

        for _ in range(self.max_iter):
            bucket = (low + high) / 2.0
            labels = _labels_for_bucket(class_weights, class_mask, bucket=bucket)
            nregions = _count_positive_labels(labels)
            error = abs(nregions - target_regions)
            if error < best_error:
                best = labels
                best_error = error
                if error == 0:
                    break

            if nregions > target_regions:
                low = bucket
            else:
                high = bucket

        return best


def get_nregions(
    bucket: float,
    grid: np.ndarray,
    domain: str,
    *,
    landsea_indices: np.ndarray | None = None,
) -> int:
    """Count the land/sea-separated regions produced by one bucket threshold.

    Args:
        bucket:
            Maximum value for each basis function region
        grid:
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you choose!
        domain:
            Domain label used in shape-error diagnostics; selects no resource.
        landsea_indices:
            Required dense class array with the same shape and axis order as
            ``grid``. Classes 0 and 1 are split separately; other cells remain 0.

    Returns:
        Number of positive basis labels at the supplied bucket threshold.
    """
    return np.max(
        bucket_split_landsea_basis(
            grid,
            bucket,
            domain,
            landsea_indices=landsea_indices,
        )
    )


def optimize_nregions(
    bucket: float,
    grid: np.ndarray,
    nregion: int,
    tol: int,
    domain: str,
    *,
    landsea_indices: np.ndarray | None = None,
) -> float:
    """Search for a bucket threshold producing ``nregion`` labels within ``tol``.

    Each search uses already-resolved dense arrays. If no threshold converges,
    the tolerance increases up to nine times before OptimizationError is raised.

    Args:
        bucket:
            Maximum value for each basis function region
        grid:
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you choose!
        nregion:
            Number of desired basis function regions
        tol:
            Tolerance to find number of basis function regions.
            i.e. optimizes nregions to +/- tol
        domain:
            Domain label used in shape-error diagnostics; selects no resource.
        landsea_indices:
            Required dense class array with the same shape and axis order as
            ``grid``. No file discovery or coordinate alignment is performed.

    Returns:
        Bucket threshold achieving the target within the final tolerance.
    """
    current_bucket = bucket
    current_tol = tol

    # outer loop over tol; increase by 1 each time inner loops fails
    for _ in range(10):
        # try 1000 iterations
        for j in range(1000):
            current_nregion = get_nregions(
                current_bucket,
                grid,
                domain,
                landsea_indices=landsea_indices,
            )

            if current_nregion <= nregion + current_tol and current_nregion >= nregion - current_tol:
                print(
                    f"optimize_nregions found optimal bucket value {current_bucket} after {j} iterations with current_tolerance {current_tol}."
                )
                return current_bucket

            if current_nregion < nregion + current_tol:
                current_bucket *= 0.995
            else:
                current_bucket *= 1.005

        # if no convergence, increase tol
        current_tol += 1

    raise OptimizationError(
        f"optimize_nregions failed to converge for all tolerances from {tol} to {current_tol}. Try the 'quadtree' algorithm."
    )


def bucket_split_landsea_basis(
    grid: np.ndarray,
    bucket: float,
    domain: str,
    *,
    landsea_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Split weighted rectangles, assigning separate labels to classes 0 and 1.

    Inputs are eager arrays and remain unchanged. Cells with other class values
    retain label 0. Resolve and align the map before calling this kernel.

    Args:
        grid:
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you choose!
        bucket:
            Maximum value for each basis function region
        domain:
            Domain label used in shape-error diagnostics; selects no resource.
        landsea_indices:
            Required dense class array with the same shape and axis order as
            ``grid``; None or a mismatched shape raises ValueError.

    Returns:
        Array on the input grid with consecutive positive region labels and
        zero for cells outside classes 0 and 1.

    """
    if landsea_indices is None:
        raise ValueError("Resolve landsea_indices before calling the weighted numerical algorithm.")
    if landsea_indices.shape != grid.shape:
        raise ValueError(
            f"Land-sea mask shape {landsea_indices.shape} does not match basis grid shape "
            f"{grid.shape} for domain {domain}."
        )
    myregions = bucket_value_split(grid, bucket)

    mybasis_function = np.zeros(shape=grid.shape)

    for i in range(len(myregions)):
        ymin, ymax = myregions[i][0], myregions[i][1]
        xmin, xmax = myregions[i][2], myregions[i][3]

        inds_y0, inds_x0 = np.where(landsea_indices[ymin:ymax, xmin:xmax] == 0)
        inds_y1, inds_x1 = np.where(landsea_indices[ymin:ymax, xmin:xmax] == 1)

        count = np.max(mybasis_function)

        if len(inds_y0) != 0:
            count += 1
            for j in range(len(inds_y0)):
                mybasis_function[inds_y0[j] + ymin, inds_x0[j] + xmin] = count

        if len(inds_y1) != 0:
            count += 1
            for j in range(len(inds_y1)):
                mybasis_function[inds_y1[j] + ymin, inds_x1[j] + xmin] = count

    return mybasis_function


def nregion_landsea_basis(
    grid: np.ndarray,
    bucket: float = 1,
    nregion: int = 100,
    tol: int = 1,
    domain: str = "EUROPE",
    *,
    landsea_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Fit an eager land/sea-separated partition near the requested region count.

    The search may relax ``tol`` if the requested count cannot be found and
    raises OptimizationError after exhausting its search. Inputs are borrowed;
    this kernel never reads map files or aligns coordinates.

    Args:
        grid:
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you choose!
        bucket:
            Initial bucket value for each basis function region.
            Defaults to 1
        nregion:
            Number of desired basis function regions
            Defaults to 100
        tol:
            Tolerance to find number of basis function regions.
            i.e. optimizes nregions to +/- tol
            Defaults to 1
        domain:
            Domain label used in shape-error diagnostics; selects no resource.
        landsea_indices:
            Required dense class array with the same shape and axis order as
            ``grid``. No file discovery or coordinate alignment is performed.

    Returns:
        basis_function: 2D basis function array
    """
    bucket_opt = optimize_nregions(
        bucket,
        grid,
        nregion,
        tol,
        domain,
        landsea_indices=landsea_indices,
    )
    basis_function = bucket_split_landsea_basis(
        grid,
        bucket_opt,
        domain,
        landsea_indices=landsea_indices,
    )
    return basis_function
