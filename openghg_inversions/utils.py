# ****************************************************************************
# Created: 7 Nov. 2022
# Author: Eric Saboya, School of Geographical Sciences, University of Bristol
# Contact: ericsaboya@bristol.ac.uk
# ****************************************************************************
# About
# Script containing common Python functions that can be called for running
# HBMCMC and other  inversion models.
# Most functions have been copied form the acrg repo (e.g. acrg.name)
#
# ****************************************************************************
import glob
import os
import re
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import xarray as xr

from openghg_inversions.config.paths import Paths

openghginv_path = Paths.openghginv


def open_ds(path, chunks=None, combine=None):
    """
    Function efficiently opens xarray datasets.
    -----------------------------------
    Args:
      path (str):
      chunks (dict, optional):
        size of chunks for each dimension
        e.g. {'lat': 50, 'lon': 50}
        opens dataset with dask, such that it is opened 'lazily'
        and all of the data is not loaded into memory
        defaults to None - dataset is opened with out dask
      combine (str, optional):
        Way in which the data should be combined (if using chunks), either:
        'by_coords': order the datasets before concatenating (default)
        'nested': concatenate datasets in the order supplied

    Returns:
      ds (xarray)
    -----------------------------------
    """
    if chunks is not None:
        combine = "by_coords" if combine is None else combine
        ds = xr.open_mfdataset(path, chunks=chunks, combine=combine)
    else:
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            ds.load()

    return ds


def read_netcdfs(files, dim="time", chunks=None, verbose=True):
    """
    The read_netcdfs function uses xarray to open sequential netCDF files and
    and concatenates them along the specified dimension.
    Note: this function makes sure that file is closed after open_dataset call.
    -----------------------------------
    Args:
      files (list):
        List of netCDF filenames.
      dim (str, optional):
        Dimension of netCDF to use for concatenating the files.
        Default = "time".
      chunks (dict):
        size of chunks for each dimension
        e.g. {'lat': 50, 'lon': 50}
        opens dataset with dask, such that it is opened 'lazily'
        and all of the data is not loaded into memory
        defaults to None - dataset is opened with out dask

    Returns:
      xarray.Dataset:
        All files open as one concatenated xarray.Dataset object
    -----------------------------------
    """
    if verbose:
        print("Reading and concatenating files ...")
        for fname in files:
            print(fname)

    datasets = [open_ds(p, chunks=chunks) for p in sorted(files)]

    # reindex all of the lat-lon values to a common one to prevent floating point error differences
    with xr.open_dataset(files[0]) as temp:
        fields_ds = temp.load()
    fp_lat = fields_ds["lat"].values
    fp_lon = fields_ds["lon"].values

    datasets = [
        ds.reindex(indexers={"lat": fp_lat, "lon": fp_lon}, method="nearest", tolerance=1e-5)
        for ds in datasets
    ]

    combined = xr.concat(datasets, dim)

    return combined


def get_country(domain, country_file=None):
    if country_file is None:
        if not os.path.exists(os.path.join(openghginv_path, "countries/")):
            os.makedirs(os.path.join(openghginv_path, "countries/"))
            raise FileNotFoundError(
                "Country definition file not found." f" Please add to {openghginv_path}/countries/"
            )
        else:
            country_directory = os.path.join(openghginv_path, "countries/")

        filenames = glob.glob(os.path.join(country_directory, f"country_{domain}.nc"))
        filename = filenames[0]
    else:
        filename = country_file

    with xr.open_dataset(filename) as f:
        lon = f.variables["lon"][:].values
        lat = f.variables["lat"][:].values

        # Get country indices and names
        if "country" in f.variables:
            country = f.variables["country"][:, :]
        elif "region" in f.variables:
            country = f.variables["region"][:, :]
        else:
            raise ValueError(f"Variables 'country' or 'region' not found in country file {filename}.")

        #         if (ukmo is True) or (uk_split is True):
        #             name_temp = f.variables['name'][:]
        #             f.close()
        #             name=np.asarray(name_temp)

        #         else:
        name = f.variables["name"].values.astype(str)

    result = dict(
        lon=lon,
        lat=lat,
        lonmax=np.max(lon),
        lonmin=np.min(lon),
        latmax=np.max(lat),
        latmin=np.min(lat),
        country=np.asarray(country),
        name=name,
    )
    return SimpleNamespace(**result)


def areagrid(lat, lon):
    """
    Calculates grid of areas (m2) given arrays of latitudes and longitudes
    -------------------------------------
    Args:
      lat (array):
        1D array of latitudes
      lon (array):
        1D array of longitudes

    Returns:
      area (array):
        2D array of areas of of size lat x lon
    -------------------------------------
    Example:
      import utils.areagrid
      lat=np.arange(50., 60., 1.)
      lon=np.arange(0., 10., 1.)
      area=utils.areagrid(lat, lon)
    """

    re = 6367500.0  # radius of Earth in m

    dlon = abs(np.mean(lon[1:] - lon[0:-1])) * np.pi / 180.0
    dlat = abs(np.mean(lat[1:] - lat[0:-1])) * np.pi / 180.0
    theta = np.pi * (90.0 - lat) / 180.0

    area = np.zeros((len(lat), len(lon)))

    for latI in range(len(lat)):
        if theta[latI] == 0.0 or np.isclose(theta[latI], np.pi):
            area[latI, :] = (re**2) * abs(np.cos(dlat / 2.0) - np.cos(0.0)) * dlon
        else:
            lat1 = theta[latI] - dlat / 2.0
            lat2 = theta[latI] + dlat / 2.0
            area[latI, :] = (re**2) * (np.cos(lat1) - np.cos(lat2)) * dlon

    return area


def indexesMatch(dsa, dsb):
    """
    Check if two datasets need to be reindexed_like for combine_datasets
    -----------------------------------
    Args:
      dsa (xarray.Dataset) :
        First dataset to check
      dsb (xarray.Dataset) :
        Second dataset to check

    Returns:
      boolean:
        True if indexes match, False if datasets must be reindexed
    -----------------------------------
    """

    commonIndicies = [key for key in dsa.indexes.keys() if key in dsb.indexes.keys()]

    # test if each comon index is the same
    for index in commonIndicies:
        # first check lengths are the same to avoid error in second check
        if not len(dsa.indexes[index]) == len(dsb.indexes[index]):
            return False

        # check number of values that are not close (testing for equality with floating point)
        if index == "time":
            # for time iverride the default to have ~ second precision
            rtol = 1e-10
        else:
            rtol = 1e-5

        num_not_close = np.sum(
            ~np.isclose(
                dsa.indexes[index].values.astype(float),
                dsb.indexes[index].values.astype(float),
                rtol=rtol,
            )
        )
        if num_not_close > 0:
            return False

    return True


def combine_datasets(dsa, dsb, method="nearest", tolerance: Optional[float] = None) -> xr.Dataset:
    """
    Merge two datasets, re-indexing to the first dataset (within an optional tolerance).

    If "fp" variable is found within the combined dataset, the "time" values where the "lat", "lon"
    dimensions didn't match are removed.

    Example:
        ds = combine_datasets(dsa, dsb)

    Args:
      dsa (xarray.Dataset):
        First dataset to merge
      dsb (xarray.Dataset):
        Second dataset to merge
      method: One of {None, ‘nearest’, ‘pad’/’ffill’, ‘backfill’/’bfill’}
        See xarray.DataArray.reindex_like for list of options and meaning.
        Default = "ffill" (forward fill)
      tolerance: Maximum allowed (absolute) tolerance between matches.

    Returns:
      xarray.Dataset: combined dataset indexed to dsa
    """
    # merge the two datasets within a tolerance and remove times that are NaN (i.e. when FPs don't exist)

    if not indexesMatch(dsa, dsb):
        dsb_temp = dsb.load().reindex_like(dsa, method, tolerance=tolerance)
    else:
        dsb_temp = dsb

    ds_temp = dsa.merge(dsb_temp)

    if "fp" in ds_temp:
        mask = np.isfinite(ds_temp.fp.sum(dim=["lat", "lon"], skipna=False))
        ds_temp = ds_temp.where(mask.as_numpy(), drop=True)  # .as_numpy() in case mask is chunked

    return ds_temp
