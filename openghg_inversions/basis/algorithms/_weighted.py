"""Module to create basis regions used in the inversion using an algorithm that splits regions
by input data.

If the total (sum) of the input data in a region exceeds a certain threshold, then the region
is split in two. This continues recursively until we have a collection of regions whose totals
are all below the threshold.

The threshold is optimised to create a specific number of regions.
"""

from functools import lru_cache
import logging
from pathlib import Path

import numpy as np
import xarray as xr


logger = logging.getLogger(__name__)


class OptimizationError(Exception): ...


@lru_cache
def load_landsea_indices(domain: str) -> np.ndarray:
    """Load array with indices that separate land and sea regions in specified domain.

    Args:
        domain: Domain for which to load landsea indices. Currently only "EASTASIA" or "EUROPE".

    Returns:
        np.ndarray: Array containing 0 (where there is sea) and 1 (where there is land).
    """
    if domain == "EASTASIA":
        landsea_indices = xr.open_dataset(Path(__file__).parent / "country-land-sea_EASTASIA.nc")
    elif domain == "EUROPE":
        landsea_indices = xr.open_dataset(Path(__file__).parent / "country-EUROPE-UKMO-landsea-2023.nc")
    else:
        logger.warning(
            f"No land-sea file found for domain {domain}. Defaulting to EUROPE (country-EUROPE-UKMO-landsea-2023.nc)"
        )
        landsea_indices = xr.open_dataset(Path(__file__).parent / "country-EUROPE-UKMO-landsea-2023.nc")
    return landsea_indices["country"].values


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


def get_nregions(bucket: float, grid: np.ndarray, domain: str) -> int:
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
            Currently limited to "EUROPE" or "EASTASIA"

    Return :
        number of basis functions for bucket value
    """
    return np.max(bucket_split_landsea_basis(grid, bucket, domain))


def optimize_nregions(bucket: float, grid: np.ndarray, nregion: int, tol: int, domain: str) -> float:
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
            Currently limited to "EUROPE" or "EASTASIA"

    Return :
        Optimized bucket value
    """
    current_bucket = bucket
    current_tol = tol

    # outer loop over tol; increase by 1 each time inner loops fails
    for _ in range(10):
        # try 1000 iterations
        for j in range(1000):
            current_nregion = get_nregions(current_bucket, grid, domain)

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


def bucket_split_landsea_basis(grid: np.ndarray, bucket: float, domain: str) -> np.ndarray:
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
            Currently limited to "EUROPE" or "EASTASIA"

    Returns:
        2D array with basis function values

    """
    landsea_indices = load_landsea_indices(domain)
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
    grid: np.ndarray, bucket: float = 1, nregion: int = 100, tol: int = 1, domain: str = "EUROPE"
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
            Currently limited to "EUROPE" or "EASTASIA"

    Returns:
        basis_function: 2D basis function array
    """
    bucket_opt = optimize_nregions(bucket, grid, nregion, tol, domain)
    basis_function = bucket_split_landsea_basis(grid, bucket_opt, domain)
    return basis_function
