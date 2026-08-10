"""Weighted bucket algorithms for basis-region construction.

If the total (sum) of the input data in a region exceeds a certain threshold, then the region
is split in two. This continues recursively until we have a collection of regions whose totals
are all below the threshold.

The threshold is optimised to create a specific number of regions.
:class:`AxisAlignedWeightedSplitStrategy` adapts this recursion to one Boolean
basis-group mask without loading land/sea data. The legacy
:func:`nregion_landsea_basis` path instead uses :func:`load_landsea_indices`,
which reads and caches a NetCDF land/sea field.
"""

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path

import numpy as np
import xarray as xr


logger = logging.getLogger(__name__)


class OptimizationError(Exception): ...


@lru_cache
def load_landsea_indices(domain: str, country_directory: str | None = None) -> np.ndarray:
    """Load array with indices that separate land and sea regions in specified domain.

    Args:
        domain: Domain for which to load landsea indices.
        country_directory: Directory containing land-sea files. If None, will use default files.

    Returns:
        np.ndarray: Array containing 0 (where there is sea) and 1 (where there is land).
    """
    default_dir = Path(__file__).parent
    if country_directory is not None:
        landsea_path = Path(country_directory) / f"country-land-sea_{domain}.nc"
    elif domain == "EUROPE":
        landsea_path = default_dir / "country-EUROPE-UKMO-landsea-2023.nc"
    else:
        landsea_path = default_dir / f"country-land-sea_{domain}.nc"
        if not landsea_path.exists():
            logger.warning(
                f"No land-sea file found for domain {domain}. Defaulting to EUROPE (country-EUROPE-UKMO-landsea-2023.nc)"
            )
            landsea_path = default_dir / "country-EUROPE-UKMO-landsea-2023.nc"
    return xr.open_dataset(landsea_path)["country"].values


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
    which optimizes the bucket layout before applying the land/sea split. New
    the default region-constrained strategy is
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


def get_nregions(bucket: float, grid: np.ndarray, domain: str, country_directory: str | None = None) -> int:
    """Optimize bucket value to number of desired regions.

    Args:
        bucket:
            Maximum value for each basis function region
        grid:
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you choose!
        domain:
            Domain across which to calculate basis functions.
        country_directory:
            Directory containing land-sea files. If None, will use default files.

    Return :
        number of basis functions for bucket value
    """
    return np.max(bucket_split_landsea_basis(grid, bucket, domain, country_directory))


def optimize_nregions(
    bucket: float, grid: np.ndarray, nregion: int, tol: int, domain: str, country_directory: str | None = None
) -> float:
    """Optimize bucket value to obtain nregion basis functions
    within +/- tol.

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
            Domain across which to calculate basis functions.
        country_directory:
            Directory containing land-sea files. If None, will use default files.

    Return :
        Optimized bucket value
    """
    current_bucket = bucket
    current_tol = tol

    # outer loop over tol; increase by 1 each time inner loops fails
    for _ in range(10):
        # try 1000 iterations
        for j in range(1000):
            current_nregion = get_nregions(current_bucket, grid, domain, country_directory)

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
    grid: np.ndarray, bucket: float, domain: str, country_directory: str | None = None
) -> np.ndarray:
    """Same as bucket_split_basis but includes
    land-sea split. i.e. basis functions cannot overlap sea and land.

    Args:
        grid:
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you choose!
        bucket:
            Maximum value for each basis function region
        domain:
            Domain across which to calculate basis functions.
        country_directory:
            Directory containing land-sea files. If None, will use default files.

    Returns:
        2D array with basis function values

    """
    landsea_indices = load_landsea_indices(domain, country_directory=country_directory)
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
    country_directory: str | None = None,
) -> np.ndarray:
    """Obtain basis function with nregions (for land-sea split).

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
            Domain across which to calculate basis functions.
        country_directory:
            Directory containing land-sea files. If None, will use default files.

    Returns:
        basis_function: 2D basis function array
    """
    bucket_opt = optimize_nregions(bucket, grid, nregion, tol, domain, country_directory)
    basis_function = bucket_split_landsea_basis(grid, bucket_opt, domain, country_directory)
    return basis_function
