# *****************************************************************************
# Created: 7 Nov. 2022
# Author: Eric Saboya, School of Geographical Sciences, University of Bristol
# Contact: eric.saboya@bristol.ac.uk
# *****************************************************************************
# About
#   Basis functions for used for HBMCMC inversions. Originally created by
#   Anita Ganesan and updated, here, by Eric Saboya.
#   HBMCMC uses the QuadTree algorithm for creating basis functions for
#   the inversion runs.
# *****************************************************************************
from typing import Optional

import numpy as np
import scipy.optimize
import xarray as xr


class quadTreeNode:
    def __init__(self, xStart, xEnd, yStart, yEnd):
        self.xStart = xStart
        self.xEnd = xEnd
        self.yStart = yStart
        self.yEnd = yEnd

        self.child1 = None  # top left
        self.child2 = None  # top right
        self.child3 = None  # bottom left
        self.child4 = None  # bottom right

    def isLeaf(self):
        if self.child1 or self.child2 or self.child3 or self.child4:
            return False
        else:
            return True

    def createChildren(self, grid, limit):
        value = np.sum(grid[self.xStart : self.xEnd, self.yStart : self.yEnd])  # .values

        # stop subdividing if finest resolution or bucket level reached
        if value < limit or (self.xEnd - self.xStart < 2) or (self.yEnd - self.yStart < 2):
            return

        dx = self.xEnd - self.xStart
        dy = self.yEnd - self.yStart

        # create 4 children for subdivison
        self.child1 = quadTreeNode(self.xStart, self.xStart + dx // 2, self.yStart, self.yStart + dy // 2)
        self.child2 = quadTreeNode(
            self.xStart + dx // 2, self.xStart + dx, self.yStart, self.yStart + dy // 2
        )
        self.child3 = quadTreeNode(
            self.xStart, self.xStart + dx // 2, self.yStart + dy // 2, self.yStart + dy
        )
        self.child4 = quadTreeNode(
            self.xStart + dx // 2, self.xStart + dx, self.yStart + dy // 2, self.yStart + dy
        )

        # apply recursion on all child nodes
        self.child1.createChildren(grid, limit)
        self.child2.createChildren(grid, limit)
        self.child3.createChildren(grid, limit)
        self.child4.createChildren(grid, limit)

    def appendLeaves(self, leafList):
        # recursively append all leaves/end nodes to leafList
        if self.isLeaf():
            leafList.append(self)
        else:
            self.child1.appendLeaves(leafList)
            self.child2.appendLeaves(leafList)
            self.child3.appendLeaves(leafList)
            self.child4.appendLeaves(leafList)


def quadTreeGrid(grid, limit):
    """
    Apply quadtree division algorithm.

    Args:
      grid (array):
        2d numpy array to apply quadtree division to
      limit (float):
        Use value as bucket level for defining maximum subdivision

    Returns:
      outputGrid (array):
        2d numpy grid, same shape as grid, with values correpsonding to
        each  box from boxList
    """
    # start with a single node the size of the entire input grid:
    parentNode = quadTreeNode(0, grid.shape[0], 0, grid.shape[1])
    parentNode.createChildren(grid, limit)

    leafList = []
    parentNode.appendLeaves(leafList)

    outputGrid = np.zeros_like(grid)

    for i, leaf in enumerate(leafList):
        outputGrid[leaf.xStart : leaf.xEnd, leaf.yStart : leaf.yEnd] = i

    return outputGrid


def get_quadtree_basis(fps: np.ndarray, nbasis: int, seed: Optional[int] = None) -> np.ndarray:
    """Given an array and a specified number of basis functions, return basis regions specified by
    the quadtree algorithm.

    Args:
        fps: array (mean flux times mean footprints) to use to calculate basis regions
        nbasis: target number of basis regions
        seed: optional random seed to use (for testing or reproducing results)

    Returns:
        2D numpy array with positive integer values representing basis regions.
    """

    def qtoptim(x):
        basisQuad = quadTreeGrid(fps, x)
        return (nbasis - np.max(basisQuad) - 1) ** 2

    cost = 1e6
    pwr = 0
    while cost > 3.0:
        optim = scipy.optimize.dual_annealing(
            qtoptim, np.expand_dims([0, 100 / 10**pwr], axis=0), seed=seed
        )
        cost = np.sqrt(optim.fun)
        pwr += 1
        if pwr > 10:
            raise RuntimeError("Quadtree did not converge after max iterations.")

    return quadTreeGrid(fps, optim.x[0]) + 1



# BUCKET BASIS FUNCTIONS
def load_landsea_indices():
    """
    Load UKMO array with indices that separate
    land and sea regions in EUROPE domain
    --------------
    land = 1
    sea = 0
    """
    landsea_indices = xr.open_dataset("../countries/country-EUROPE-UKMO-landsea-2023.nc")
    return landsea_indices["country"].values


def bucket_value_split(grid, bucket, offset_x=0, offset_y=0):
    """
    Algorithm that will split the input grid (e.g. fp * flux)
    such that the sum of each basis function region will
    equal the bucket value or by a single array element.

    The number of regions will be determined by the bucket value
    i.e. smaller bucket value ==> more regions
         larger bucket value ==> fewer regions
    ------------------------------------
    Args:
        grid: np.array
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you chose!

        bucket: float
            Maximum value for each basis function region

    Returns:
        array of tuples that define the indices for each basis function region
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


# Optimize bucket value to number of desired regions
def get_nregions(bucket, grid):
    """Returns no. of basis functions for bucket value"""
    return np.max(bucket_split_landsea_basis(grid, bucket))


def optimize_nregions(bucket, grid, nregion, tol):
    """
    Optimize bucket value to obtain nregion basis functions
    within +/- tol.
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


def bucket_split_landsea_basis(grid, bucket):
    """
    Same as bucket_split_basis but includes
    land-sea split. i.e. basis functions cannot overlap sea and land
     ------------------------------------
    Args:
        grid: np.array
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you chose!

        bucket: float
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


def nregion_landsea_basis(grid, bucket=1, nregion=100, tol=1):
    """
    Obtain basis function with nregions (for land-sea split)
    ------------------------------------
    Args:
        grid: np.array
            2D grid of footprints * flux, or whatever
            grid you want to split. Could be: population
            data, spatial distribution of bakeries, you chose!

        bucket: float
            Initial bucket value for each basis function region.
            Defaults to 1

        nregion: int
            Number of desired basis function regions
            Defaults to 100

        tol: int
            Tolerance to find number of basis function regions.
            i.e. optimizes nregions to +/- tol
            Defaults to 1

    Returns:
        basis_function np.array
        2D basis function array

    """
    bucket_opt = optimize_nregions(bucket, grid, nregion, tol)
    basis_function = bucket_split_landsea_basis(grid, bucket_opt)
    return basis_function
