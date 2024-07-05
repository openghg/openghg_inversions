"""
Module to create basis regions used in the inversion using an algorihtm that splits region
by input data.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import xarray as xr


# BUCKET BASIS FUNCTIONS
def load_landsea_indices() -> np.ndarray:
    """
    Load UKMO array with indices that separate
    land and sea regions in EUROPE domain
    
    Returns :
        Array containing 0 (where there is sea) 
        and 1 (where there is land).
    """
    landsea_indices = xr.open_dataset(Path(__file__).parent / "country-EUROPE-UKMO-landsea-2023.nc")
    return landsea_indices["country"].values


def bucket_value_split(grid : np.ndarray, 
                       bucket : float, 
                       offset_x : Optional[int]=0, 
                       offset_y : Optional[int]=0) -> list[tuple]:
    """
    Algorithm that will split the input grid (e.g. fp * flux)
    such that the sum of each basis function region will
    equal the bucket value or by a single array element.

    The number of regions will be determined by the bucket value
    i.e. smaller bucket value ==> more regions
         larger bucket value ==> fewer regions
         
    Args:
        grid:
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you chose!
        bucket:
            Maximum value for each basis function region
        offset_x:
            Start index of the region on first axis of the grid
            Default 0
        offset_y:
            Start index of the region on second axis of the grid
            Default 0

    Returns:
        list of tuples that define the indices for each basis function region
        [(ymin0, ymax0, xmin0, xmax0), ..., (yminN, ymaxN, xminN, xmaxN)]
    """

    if np.sum(grid) <= bucket or grid.shape == (1, 1):
        return [(offset_y, offset_y + grid.shape[0], offset_x, offset_x + grid.shape[1])]

    else:
        if grid.shape[0] >= grid.shape[1]:
            half_y = grid.shape[0] // 2
            return bucket_value_split(grid[0:half_y, :], bucket, offset_x, offset_y) + bucket_value_split(
                grid[half_y:, :], bucket, offset_x, offset_y + half_y
            )

        elif grid.shape[0] < grid.shape[1]:
            half_x = grid.shape[1] // 2
            return bucket_value_split(grid[:, 0:half_x], bucket, offset_x, offset_y) + bucket_value_split(
                grid[:, half_x:], bucket, offset_x + half_x, offset_y
            )


def get_nregions(bucket : float, 
                 grid : np.ndarray) -> int:
    """Optimize bucket value to number of desired regions.
      
    Args:
        bucket:
            Maximum value for each basis function region
        grid:
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you chose!

    Return :
        no. of basis functions for bucket value
    """
    return np.max(bucket_split_landsea_basis(grid, bucket))


def optimize_nregions(bucket : float, 
                      grid : np.ndarray, 
                      nregion : int, 
                      tol : int) -> float:
    """
    Optimize bucket value to obtain nregion basis functions
    within +/- tol.
    
    Args:
        bucket:
            Maximum value for each basis function region
        grid:
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you chose!
        nregion:
            Number of desired basis function regions
        tol:
            Tolerance to find number of basis function regions.
            i.e. optimizes nregions to +/- tol

    Return :
        Optimized bucket value
    """
    # print(bucket, get_nregions(bucket, grid))
    if get_nregions(bucket, grid) <= nregion + tol and get_nregions(bucket, grid) >= nregion - tol:
        return bucket

    if get_nregions(bucket, grid) < nregion + tol:
        bucket = bucket * 0.995
        return optimize_nregions(bucket, grid, nregion, tol)

    elif get_nregions(bucket, grid) > nregion - tol:
        bucket = bucket * 1.005
        return optimize_nregions(bucket, grid, nregion, tol)


def bucket_split_landsea_basis(grid : np.ndarray, 
                               bucket : float) -> np.ndarray:
    """
    Same as bucket_split_basis but includes
    land-sea split. i.e. basis functions cannot overlap sea and land
    
    Args:
        grid:
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you chose!
        bucket:
            Maximum value for each basis function region

    Returns:
        2D array with basis function values

    """
    landsea_indices = load_landsea_indices()
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
            for i in range(len(inds_y0)):
                mybasis_function[inds_y0[i] + ymin, inds_x0[i] + xmin] = count

        if len(inds_y1) != 0:
            count += 1
            for i in range(len(inds_y1)):
                mybasis_function[inds_y1[i] + ymin, inds_x1[i] + xmin] = count

    return mybasis_function


def nregion_landsea_basis(grid : np.ndarray,
                          bucket : float = 1,
                          nregion : int = 100, 
                          tol : int = 1) -> np.ndarray:
    """
    Obtain basis function with nregions (for land-sea split)
    
    Args:
        grid:
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you chose!
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

    Returns:
        basis_function: 2D basis function array
    """
    bucket_opt = optimize_nregions(bucket, grid, nregion, tol)
    basis_function = bucket_split_landsea_basis(grid, bucket_opt)
    return basis_function
