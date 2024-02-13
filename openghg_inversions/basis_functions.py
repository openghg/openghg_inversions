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

import getpass
import os
from typing import Optional

import numpy as np
import pandas as pd
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
    Apply quadtree division algorithm
    -----------------------------------
    Args:
      grid (array):
        2d numpy array to apply quadtree division to
      limit (float):
        Use value as bucket level for defining maximum subdivision

    Returns:
      outputGrid (array):
        2d numpy grid, same shape as grid, with values correpsonding to
        each  box from boxList
      boxList: (list of lists)
        Each sublist describes the corners of a quadtree leaf
    -----------------------------------
    """
    # start with a single node the size of the entire input grid:
    parentNode = quadTreeNode(0, grid.shape[0], 0, grid.shape[1])
    parentNode.createChildren(grid, limit)

    leafList = []
    boxList = []
    parentNode.appendLeaves(leafList)

    outputGrid = np.zeros_like(grid)

    for i, leaf in enumerate(leafList):
        outputGrid[leaf.xStart : leaf.xEnd, leaf.yStart : leaf.yEnd] = i
        boxList.append([leaf.xStart, leaf.xEnd, leaf.yStart, leaf.yEnd])

    return outputGrid, boxList


def quadtreebasisfunction(
    emissions_name: list[str],
    fp_all: dict,
    sites: list[str],
    start_date: str,
    domain: str,
    species: str,
    outputname: Optional[str] = None,
    outputdir: Optional[str] = None,
    nbasis: int = 100,
    abs_flux: bool = False,
    seed: Optional[int] = None,
) -> xr.Dataset:
    """
    Creates a basis function with nbasis grid cells using a quadtree algorithm.
    The domain is split with smaller grid cells for regions which contribute
    more to the a priori (above basline) mole fraction. This is based on the
    average footprint over the inversion period and the a priori emissions field.
    Output is a netcdf file saved to /Temp/<domain> in the current directory
    if no outputdir is specified or to outputdir if specified.
    The number of basis functions is optimised using dual annealing. Probably
    not the best or fastest method as there should only be one minima, but doesn't
    require the Jacobian or Hessian for optimisation.
    -----------------------------------
    Args:
      emissions_name (list):
        List of "source" key words as used for retrieving specific emissions
        from the object store.
      fp_all (dict):
        Output from footprints_data_merge() function. Dictionary of datasets.
      sites (list):
        List of site names (This could probably be found elsewhere)
      start_date (str):
        String of start date of inversion
      domain (str):
        The inversion domain
      species (str):
        Atmospheric trace gas species of interest (e.g. 'co2')
      outputname (str):
        Identifier or run name
      outputdir (str, optional):
        Path to output directory where the basis function file will be saved.
        Basis function will automatically be saved in outputdir/DOMAIN
        Default of None makes a temp directory.
      nbasis (int):
        Number of basis functions that you want. This will optimise to
        closest value that fits with quadtree splitting algorithm,
        i.e. nbasis % 4 = 1.
      abs_flux (bool):
        If True this will take the absolute value of the flux
      seed:
        Optional seed to pass to scipy.optimize.dual_annealing. Used for testing.

    Returns:
        xr.Dataset with lat/lon dimensions and basis regions encoded by integers.
        If outputdir is not None, then saves the basis function in outputdir.
    -----------------------------------
    """
    if abs_flux:
        print("Using absolute values of flux array")
    if emissions_name is None:
        flux = (
            np.absolute(fp_all[".flux"]["all"].data.flux.values)
            if abs_flux
            else fp_all[".flux"]["all"].data.flux.values
        )
        meanflux = np.squeeze(flux)
    else:
        if isinstance(fp_all[".flux"][emissions_name[0]], dict):
            arr = fp_all[".flux"][emissions_name[0]]
            flux = np.absolute(arr.data.flux.values) if abs_flux else arr.data.flux.values
            meanflux = np.squeeze(flux)
        else:
            flux = (
                np.absolute(fp_all[".flux"][emissions_name[0]].data.flux.values)
                if abs_flux
                else fp_all[".flux"][emissions_name[0]].data.flux.values
            )
            meanflux = np.squeeze(flux)
    meanfp = np.zeros((fp_all[sites[0]].fp.shape[0], fp_all[sites[0]].fp.shape[1]))
    div = 0
    for site in sites:
        meanfp += np.sum(fp_all[site].fp.values, axis=2)
        div += fp_all[site].fp.shape[2]
    meanfp /= div

    if meanflux.shape != meanfp.shape:
        meanflux = np.mean(meanflux, axis=2)
    fps = meanfp * meanflux

    def qtoptim(x):
        basisQuad, boxes = quadTreeGrid(fps, x)
        return (nbasis - np.max(basisQuad) - 1) ** 2

    cost = 1e6
    pwr = 0
    while cost > 3.0:
        optim = scipy.optimize.dual_annealing(qtoptim, np.expand_dims([0, 100 / 10**pwr], axis=0), seed=seed)
        cost = np.sqrt(optim.fun)
        pwr += 1
        if pwr > 10:
            raise Exception("Quadtree did not converge after max iterations.")
    basisQuad, boxes = quadTreeGrid(fps, optim.x[0])

    lon = fp_all[sites[0]].lon.values
    lat = fp_all[sites[0]].lat.values

    base = np.expand_dims(basisQuad + 1, axis=2)

    time = [pd.to_datetime(start_date)]
    newds = xr.Dataset(
        {"basis": (["lat", "lon", "time"], base)},
        coords={"time": (["time"], time), "lat": (["lat"], lat), "lon": (["lon"], lon)},
    )
    newds.lat.attrs["long_name"] = "latitude"
    newds.lon.attrs["long_name"] = "longitude"
    newds.lat.attrs["units"] = "degrees_north"
    newds.lon.attrs["units"] = "degrees_east"
    newds.attrs["creator"] = getpass.getuser()
    newds.attrs["date created"] = str(pd.Timestamp.today())

    if outputdir is not None:
        basisoutpath = os.path.join(outputdir, domain)
        if outputname is None:
            outputname = "output_name"
        if not os.path.exists(basisoutpath):
            os.makedirs(basisoutpath)
        newds.to_netcdf(
            os.path.join(
                basisoutpath, f"quadtree_{species}-{outputname}_{domain}_{start_date.split('-')[0]}.nc"
            ),
            mode="w",
        )

    return newds


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


def bucketbasisfunction(
    emissions_name,
    fp_all,
    sites,
    start_date,
    domain,
    species,
    outputname,
    outputdir=None,
    nbasis=100,
    abs_flux=False,
) -> xr.Dataset:
    """
    Basis functions calculated using a weighted region approach
    where each basis function / scaling region contains approximately
    the same value
    -----------------------------------
    Args:
      emissions_name (str/list):
        List of keyword "source" args used for retrieving emissions files
        from the Object store.
      fp_all (dict):
        fp_all dictionary object as produced from get_data functions
      sites (str/list):
        List of measurements sites being used.
      start_date (str):
        Start date of period of inference
      domain (str):
        Name of model domain
      species (str):
        Name of atmospheric species of interest
      outputname (str):
        Name of inversion run
      outputdir (str):
        Directory where inversion run outputs are saved
      nbasis (int):
        Desired number of basis function regions
      abs_flux (bool):
        When set to True uses absolute values of a flux array

    Returns:
        xr.Dataset with lat/lon dimensions and basis regions encoded by integers.
        If outputdir is not None, then saves the basis function in outputdir.
    """
    if abs_flux:
        print("Using absolute values of flux array")
    if emissions_name is None:
        flux = (
            np.absolute(fp_all[".flux"]["all"].data.flux.values)
            if abs_flux
            else fp_all[".flux"]["all"].data.flux.values
        )
        meanflux = np.squeeze(flux)
    else:
        if isinstance(fp_all[".flux"][emissions_name[0]], dict):
            arr = fp_all[".flux"][emissions_name[0]]
            flux = np.absolute(arr.data.flux.values) if abs_flux else arr.data.flux.values
            meanflux = np.squeeze(flux)
        else:
            flux = (
                np.absolute(fp_all[".flux"][emissions_name[0]].data.flux.values)
                if abs_flux
                else fp_all[".flux"][emissions_name[0]].data.flux.values
            )
            meanflux = np.squeeze(flux)
    meanfp = np.zeros((fp_all[sites[0]].fp.shape[0], fp_all[sites[0]].fp.shape[1]))
    div = 0
    for site in sites:
        meanfp += np.sum(fp_all[site].fp.values, axis=2)
        div += fp_all[site].fp.shape[2]
    meanfp /= div

    if meanflux.shape != meanfp.shape:
        meanflux = np.mean(meanflux, axis=2)
    fps = meanfp * meanflux

    bucket_basis = nregion_landsea_basis(fps, 1, nbasis)

    lon = fp_all[sites[0]].lon.values
    lat = fp_all[sites[0]].lat.values

    base = np.expand_dims(bucket_basis, axis=2)

    time = [pd.to_datetime(start_date)]
    newds = xr.Dataset(
        {"basis": (["lat", "lon", "time"], base)},
        coords={"time": (["time"], time), "lat": (["lat"], lat), "lon": (["lon"], lon)},
    )
    newds.lat.attrs["long_name"] = "latitude"
    newds.lon.attrs["long_name"] = "longitude"
    newds.lat.attrs["units"] = "degrees_north"
    newds.lon.attrs["units"] = "degrees_east"
    newds.attrs["creator"] = getpass.getuser()
    newds.attrs["date created"] = str(pd.Timestamp.today())

    if outputdir is not None:
        basisoutpath = os.path.join(outputdir, domain)
        if not os.path.exists(basisoutpath):
            os.makedirs(basisoutpath)
        newds.to_netcdf(
            os.path.join(
                basisoutpath,
                f"weighted_{species}-{outputname}_{domain}_{start_date.split('-')[0]}{start_date.split('-')[1]}.nc",
            ),
            mode="w",
        )

    return newds
