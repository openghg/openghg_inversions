"""Functions to create basis datasets from fluxes and footprints."""

import os

import getpass
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import cast

import pandas as pd
import xarray as xr

from .algorithms import quadtree_algorithm, weighted_algorithm

from openghg_inversions.config.paths import Paths
from openghg_inversions.utils import read_netcdfs


openghginv_path = Paths.openghginv


def basis(domain: str, basis_case: str, basis_directory: str | None = None) -> xr.Dataset:
    """Read in basis function(s) from file given basis case and domain, and return as an
    xarray Dataset.

    The basis function files should be stored as on paths of the form:
        <basis_directory>/<domain>/<basis_case>_<domain>*.nc

    For instance: domain = EUROPE, basis_directory = /group/chem/acrg/LPDM/basis_functions,
    and basis_case = sub_transd would find files such as:

        /group/chem/acrg/LPDM/basis_functions/EUROPE/sub_transd_EUROPE_2014.nc

    Basis functions created by algorithms in OpenGHG inversions will be stored using
    this path format.

    Args:
        domain: domain name. The basis files should be sub-categorised by the domain.
        basis_case: basis case to read in. Examples of basis cases are "voronoi", "sub-transd",
            "sub-country_mask", "INTEM".
        basis_directory: basis_directory can be specified if files are not in the default
            directory (i.e. `openghg_inversions/basis_functions`). Must point to a directory that
            contains subfolders organized by domain.

    Returns:
        xarray.Dataset: combined dataset of matching basis functions
    """
    if basis_directory is None:
        basis_path = openghginv_path / "basis_functions"
        if not basis_path.exists():
            basis_path.mkdir()
            raise ValueError(
                f"Default basis directory {basis_path} was empty. " "Add basis files or specify `basis_path`."
            )
    else:
        basis_path = Path(basis_directory)

    file_path = (basis_path / domain).glob(f"{basis_case}_{domain}*.nc")
    files = sorted(list(file_path))

    if len(files) == 0:
        raise FileNotFoundError(
            f"Can't find basis function files for domain '{domain}'" f"and basis_case '{basis_case}' "
        )

    basis_ds = read_netcdfs(files)

    return basis_ds


def basis_boundary_conditions(domain: str, basis_case: str, bc_basis_directory: str | None = None):
    """Read in basis function(s) from file given basis case and domain, and return as an
    xarray Dataset.

    The basis function files should be stored as on paths of the form:
        <bc_basis_directory>/<domain>/<basis_case>_<domain>*.nc

    For instance: domain = "EUROPE", bc_basis_directory = /group/chem/acrg/LPDM/bc_basis_functions,
    and basis_case = "NESW" would find files such as:

        /group/chem/acrg/LPDM/bc_basis_functions/EUROPE/NESW_EUROPE_2014.nc

    Args:
        domain: domain name. The basis files should be sub-categorised by the domain.
        basis_case: basis case to read in. Examples of BC basis cases are "NESW", "stratgrad".
        bc_basis_directory: bc_basis_directory can be specified if files are not in the default
            directory (i.e. `openghg_inversions/bc_basis_functions`). Must point to a directory that
            contains subfolders organized by domain.

    Returns:
        xarray.Dataset: combined dataset of matching basis functions
    """
    if bc_basis_directory is None:
        bc_basis_path = openghginv_path / "bc_basis_functions"
        if not bc_basis_path.exists():
            bc_basis_path.mkdir()
            raise ValueError(
                f"Default BC basis directory {bc_basis_path} was empty. "
                "Add basis files or specify `bc_basis_path`."
            )
    else:
        bc_basis_path = Path(bc_basis_directory)

    file_path = (bc_basis_path / domain).glob(f"{basis_case}_{domain}*.nc")
    files = sorted(list(file_path))

    # check for files that we can't access
    # NOTE: Hannah added this in 2021 to the ACRG code.
    # I don't know why it is only for BC boundary conditions -- BM, 2024
    file_no_acc = [ff for ff in files if not os.access(ff, os.R_OK)]
    if len(file_no_acc) > 0:
        print(
            "Warning: unable to read all boundary conditions basis function files which match this criteria:"
        )
        print("\n".join(map(str, file_no_acc)))

    # only use files we can access
    files = [ff for ff in files if ff not in file_no_acc]

    if len(files) == 0:
        raise FileNotFoundError(
            f"Can't find BC basis function files for domain '{domain}'" f"and bc_basis_case '{basis_case}' "
        )

    basis_ds = read_netcdfs(files)

    return basis_ds


def _flux_fp_from_fp_all(
    fp_all: dict, emissions_name: list[str] | None = None
) -> tuple[xr.DataArray, list[xr.DataArray]]:
    """Get flux and list of footprints from `fp_all` dictionary and optional list of emissions names.

    Args:
      fp_all (dict):
        Output from footprints_data_merge() function. Dictionary of datasets.
      emissions_name (list):
        List of "source" key words as used for retrieving specific emissions
        from the object store.

    Returns:
      flux (xarray.DataArray):
        Array containing the flux data.
      footprints (list):
        List of xarray DataArray containing the footprints of each sites.
    """
    if emissions_name is not None:
        flux = fp_all[".flux"][emissions_name[0]].data.flux
    else:
        first_flux = next(iter(fp_all[".flux"].values()))
        flux = first_flux.data.flux

    flux = cast(xr.DataArray, flux)

    footprints: list[xr.DataArray] = [v.fp for k, v in fp_all.items() if not k.startswith(".")]

    return flux, footprints


def _mean_fp_times_mean_flux(
    flux: xr.DataArray,
    footprints: list[xr.DataArray],
    abs_flux: bool = False,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Multiply mean flux by mean of footprints, optionally restricted to a Boolean mask.

    Args :
      flux (xarray.DataArray):
        Array containing the flux data.
      footprints (list):
        List of xarray DataArray containing the footprints of each sites.
      abs_flux (bool):
        If True this will take the absolute value of the flux in the multiplication.
      mask (xarray.DataArray):
        Boolean mask on lat/lon coordinates, indicates the spatial area kept during
        the multiplication.

    Return:
      xarray DataArray containing temporal mean flux multiplied by temporal mean of footprints
    """
    if abs_flux is True:
        print("Using absolute value of flux array.")
        flux = abs(flux)

    mean_flux = flux.mean("time")

    # get total times before aligning
    n_measure = sum(len(fp.time) for fp in footprints)

    # align so that all times are used
    footprints = xr.align(*footprints, join="outer", fill_value=0.0)  # type: ignore  the docs say scalars are accepted as fill values, but type hints don't
    fp_total = sum(footprints)  # this seems to be faster than concatentating and summing over new axis

    fp_total = cast(xr.DataArray, fp_total)  # otherwise mypy complains about the next line
    mean_fp = fp_total.sum("time") / n_measure

    if mask is not None:
        # align to footprint lat/lon
        mean_fp, mean_flux, mask = xr.align(mean_fp, mean_flux, mask, join="override")
        return (mean_fp * mean_flux).where(mask, drop=True)

    mean_fp, mean_flux = xr.align(mean_fp, mean_flux, join="override")
    return mean_fp * mean_flux


def quadtreebasisfunction(
    fp_all: dict,
    start_date: str,
    emissions_name: list[str] | None = None,
    nbasis: int = 100,
    abs_flux: bool = False,
    seed: int | None = None,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Creates a basis function with nbasis grid cells using a quadtree algorithm.

    The domain is split with smaller grid cells for regions which contribute
    more to the a priori (above basline) mole fraction. This is based on the
    average footprint over the inversion period and the a priori emissions field.

    The number of basis functions is optimised using dual annealing. Probably
    not the best or fastest method as there should only be one minima, but doesn't
    require the Jacobian or Hessian for optimisation.

    Args:
      fp_all (dict):
        fp_all dictionary of datasets as produced from get_data functions
      start_date (str):
        Start date of period of inversion
      emissions_name (list):
        List of keyword "source" args used for retrieving emissions files
        from the Object store
        Default None
      nbasis (int):
        Desired number of basis function regions
        Default 100
      abs_flux (bool):
        When set to True uses absolute values of a flux array
        Default False
      seed (int):
        Optional seed to pass to scipy.optimize.dual_annealing. Used for testing.
        Default None
      mask (xarray.DataArray):
        Boolean mask on lat/lon coordinates. Used to find basis on sub-region
        Default None

    Returns:
      quad_basis (xarray.DataArray):
        Array with lat/lon dimensions and basis regions encoded by integers.
    """
    flux, footprints = _flux_fp_from_fp_all(fp_all, emissions_name)
    fps = _mean_fp_times_mean_flux(flux, footprints, abs_flux=abs_flux, mask=mask).as_numpy()

    # use xr.apply_ufunc to keep xarray coords
    func = partial(quadtree_algorithm, nbasis=nbasis, seed=seed)
    quad_basis = xr.apply_ufunc(func, fps)

    quad_basis = quad_basis.expand_dims({"time": [pd.to_datetime(start_date)]}, axis=-1)
    quad_basis = quad_basis.rename("basis")  # this will be used in merges

    quad_basis.attrs["creator"] = getpass.getuser()
    quad_basis.attrs["date created"] = str(pd.Timestamp.today())

    return quad_basis


def bucketbasisfunction(
    fp_all: dict,
    start_date: str,
    emissions_name: list[str] | None = None,
    nbasis: int = 100,
    abs_flux: bool = False,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Basis functions calculated using a weighted region approach
    where each basis function / scaling region contains approximately
    the same value.

    Args:
      fp_all (dict):
        fp_all dictionary of datasets as produced from get_data functions
      start_date (str):
        Start date of period of inversion
      emissions_name (list):
        List of keyword "source" args used for retrieving emissions files
        from the Object store
        Default None
      nbasis (int):
        Desired number of basis function regions
        Default 100
      abs_flux (bool):
        When set to True uses absolute values of a flux array
        Default False
      mask (xarray.DataArray):
        Boolean mask on lat/lon coordinates. Used to find basis on sub-region
        Default None

    Returns:
      bucket_basis (xarray.DataArray):
        Array with lat/lon dimensions and basis regions encoded by integers.
    """
    flux, footprints = _flux_fp_from_fp_all(fp_all, emissions_name)
    fps = _mean_fp_times_mean_flux(flux, footprints, abs_flux=abs_flux, mask=mask).as_numpy()

    # use xr.apply_ufunc to keep xarray coords
    func = partial(weighted_algorithm, nregion=nbasis, bucket=1)
    bucket_basis = xr.apply_ufunc(func, fps)

    bucket_basis = bucket_basis.expand_dims({"time": [pd.to_datetime(start_date)]}, axis=-1)
    bucket_basis = bucket_basis.rename("basis")  # this will be used in merges

    bucket_basis.attrs["creator"] = getpass.getuser()
    bucket_basis.attrs["date created"] = str(pd.Timestamp.today())

    return bucket_basis


# dict to retrieve basis function and description by algorithm name
BasisFunction = namedtuple("BasisFunction", ["description", "algorithm"])
basis_functions = {
    "quadtree": BasisFunction("quadtree algorithm", quadtreebasisfunction),
    "weighted": BasisFunction("weighted by data algorithm", bucketbasisfunction),
}


def fixed_outer_regions_basis(
    fp_all: dict,
    start_date: str,
    basis_algorithm: str,
    emissions_name: list[str] | None = None,
    nbasis: int = 100,
    abs_flux: bool = False,
) -> xr.DataArray:
    """Fix outer region of basis functions to InTEM regions, and fit the inner regions using `basis_algorithm`.

    Args:
      fp_all (dict):
        fp_all dictionary object as produced from get_data functions
      start_date (str):
        Start date of period of inference
      basis_algorithm (str):
        Name of the basis algorithm used. Options are "quadtree", "weighted"
      emissions_name (list):
        List of keyword "source" args used for retrieving emissions files
        from the Object store.
      nbasis (int):
        Desired number of basis function regions
        Default 100
      abs_flux:
        When set to True uses absolute values of a flux array
        Default False

    Returns:
        basis (xarray.DataArray) :
          Array with lat/lon dimensions and basis regions encoded by integers.
    """
    intem_regions_path = Path(__file__).parent / "intem_region_definition.nc"
    intem_regions = xr.open_dataset(intem_regions_path).region

    # force intem_regions to use flux coordinates
    flux, _ = _flux_fp_from_fp_all(fp_all, emissions_name)
    _, intem_regions = xr.align(flux, intem_regions, join="override")

    mask = intem_regions == 6

    basis_function = basis_functions[basis_algorithm].algorithm
    inner_region = basis_function(fp_all, start_date, emissions_name, nbasis, abs_flux, mask=mask)

    basis = intem_regions.rename("basis")

    loc_dict = {
        "lat": slice(inner_region.lat.min(), inner_region.lat.max() + 0.1),
        "lon": slice(inner_region.lon.min(), inner_region.lon.max() + 0.1),
    }
    basis.loc[loc_dict] = (inner_region + 5).squeeze().values

    basis += 1  # intem_region_definitions.nc regions start at 0, not 1

    basis = basis.expand_dims({"time": [pd.to_datetime(start_date)]})

    return basis
