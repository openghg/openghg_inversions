"""Script containing common Python functions that can be called for running
HBMCMC and other inversion models.

The main functions are related to applying basis functions to the flux and boundary
conditions, and their sensitivities.

Many functions in this submodule originated in the ACRG code base (in `acrg.name`).

"""

from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import xarray as xr

from openghg.analyse import combine_datasets as openghg_combine_datasets

from openghg_inversions.config.paths import Paths


openghginv_path = Paths.openghginv


def combine_datasets(
    dataset_a: xr.Dataset,
    dataset_b: xr.Dataset,
    method: str | None = "nearest",
    tolerance: float | None = None,
) -> xr.Dataset:
    """Merges two datasets and re-indexes to the first dataset.

    If "fp" variable is found within the combined dataset,
    the "time" values where the "lat", "lon" dimensions didn't match are removed.

    NOTE: this is temporary solution while waiting for `.load()` to be added to openghg version of combine_datasets

    Args:
        dataset_a: First dataset to merge
        dataset_b: Second dataset to merge
        method: One of None, nearest, ffill, bfill.
                See xarray.DataArray.reindex_like for list of options and meaning.
                Defaults to ffill (forward fill)
        tolerance: Maximum allowed tolerance between matches.

    Returns:
        xarray.Dataset: Combined dataset indexed to dataset_a
    """
    return openghg_combine_datasets(dataset_a, dataset_b.load(), method=method, tolerance=tolerance)


def open_ds(
    path: str | Path,
    chunks: dict | None = None,
    combine: Literal["by_coords", "nested"] = "by_coords",
) -> xr.Dataset:
    """Function efficiently opens xarray Datasets.

    Args:
      path: path to file to open
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
        xarray Dataset
    """
    if chunks is not None:
        ds = xr.open_mfdataset(path, chunks=chunks, combine=combine)
    else:
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            ds.load()

    return ds


def read_netcdfs(
    files: list[str] | list[Path],
    dim: str = "time",
    chunks: dict | None = None,
    verbose: bool = True,
) -> xr.Dataset:
    """The read_netcdfs function uses xarray to open sequential netCDF files and
    and concatenates them along the specified dimension.
    Note: this function makes sure that file is closed after open_dataset call.

    Args:
        files: List of netCDF filenames.
        dim: Dimension of netCDF to use for concatenating the files. Default = "time".
        chunks: size of chunks for each dimension
            e.g. {'lat': 50, 'lon': 50}
            opens dataset with dask, such that it is opened 'lazily'
            and all of the data is not loaded into memory
            defaults to None - dataset is opened with out dask

    Returns:
        xarray.Dataset: All files open as one concatenated xarray.Dataset object

    # TODO: this could be done more efficiently with xr.open_mfdataset (most likely)
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


def get_country_file_path(country_file: str | Path | None = None, domain: str | None = None):
    if isinstance(country_file, Path):
        return country_file

    if isinstance(country_file, str):
        return Path(country_file)

    if domain is None:
        raise ValueError("If `country_file` is None, then `domain` must be specified.")

    # try to find country file in default location
    country_directory = openghginv_path / "countries"

    if not country_directory.exists():
        country_directory.mkdir()

        raise FileNotFoundError(
            "Country definition file not found." f" Please add to {openghginv_path}/countries/"
        )

    filenames = list(country_directory.glob(f"country_{domain}.nc"))
    filename = filenames[0]
    return Path(filename)


def get_country(domain: str, country_file: str | Path | None = None):
    """Open country file for given domain and return as a SimpleNamespace.

    NOTE: a SimpleNamespace is a like dict with class like attribute access

    Args:
        domain: domain of inversion
        country_file: optional string or Path to country file. If `None`, then the first file found in
            `openghg_inversions/countries/` is used.

    Returns:
        SimpleNamespace with attributes: lon, lat, lonmax, lonmin, latmax, latmin, country, and name
    """
    filename = get_country_file_path(country_file=country_file, domain=domain)

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


def areagrid(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Calculates grid of areas (m^2), given arrays of latitudes and longitudes.

    Args:
        lat: 1D array of latitudes
        lon: 1D array of longitudes

    Returns:
        area: 2D array of areas of of size lat x lon

    Examples:
        >>> import utils.areagrid
        >>> lat = np.arange(50., 60., 1.)
        >>> lon = np.arange(0., 10., 1.)
        >>> area = utils.areagrid(lat, lon)
    """
    rad_earth = 6367500.0  # radius of Earth in m

    dlon = abs(np.mean(lon[1:] - lon[0:-1])) * np.pi / 180.0
    dlat = abs(np.mean(lat[1:] - lat[0:-1])) * np.pi / 180.0
    theta = np.pi * (90.0 - lat) / 180.0

    area = np.zeros((len(lat), len(lon)))

    for latI in range(len(lat)):
        if theta[latI] == 0.0 or np.isclose(theta[latI], np.pi):
            area[latI, :] = (rad_earth**2) * abs(np.cos(dlat / 2.0) - np.cos(0.0)) * dlon
        else:
            lat1 = theta[latI] - dlat / 2.0
            lat2 = theta[latI] + dlat / 2.0
            area[latI, :] = (rad_earth**2) * (np.cos(lat1) - np.cos(lat2)) * dlon

    return area
