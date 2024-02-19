"""
Functions to create basis datasets from fluxes and footprints.
"""
import getpass
from functools import partial
from pathlib import Path
from typing import cast, Optional

import pandas as pd
import xarray as xr

from .algorithms import quadtree_algorithm, weighted_algorithm


def _flux_fp_from_fp_all(
    fp_all: dict, emissions_name: Optional[list[str]] = None
) -> tuple[xr.DataArray, list[xr.DataArray]]:
    """Get flux and list of footprints from `fp_all` dictionary and optional list of emissions names."""
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
    mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Multiply mean flux by mean of footprints, optionally restricted to a Boolean mask."""
    if abs_flux is True:
        print("Using absolute value of flux array.")
        flux = abs(flux)

    mean_flux = flux.mean("time")

    fp_total = sum(footprints)  # this seems to be faster than concatentating and summing over new axis
    n_measure = sum(len(fp.time) for fp in footprints)

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
    emissions_name: Optional[list[str]] = None,
    nbasis: int = 100,
    abs_flux: bool = False,
    seed: Optional[int] = None,
    mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Creates a basis function with nbasis grid cells using a quadtree algorithm.

    The domain is split with smaller grid cells for regions which contribute
    more to the a priori (above basline) mole fraction. This is based on the
    average footprint over the inversion period and the a priori emissions field.

    The number of basis functions is optimised using dual annealing. Probably
    not the best or fastest method as there should only be one minima, but doesn't
    require the Jacobian or Hessian for optimisation.

    Args:
      fp_all:
        Output from footprints_data_merge() function. Dictionary of datasets.
      start_date:
        String of start date of inversion
      emissions_name:
        List of "source" key words as used for retrieving specific emissions
        from the object store.
      nbasis:
        Number of basis functions that you want. This will optimise to
        closest value that fits with quadtree splitting algorithm,
        i.e. nbasis % 4 = 1.
      abs_flux:
        If True this will take the absolute value of the flux
      seed:
        Optional seed to pass to scipy.optimize.dual_annealing. Used for testing.
      mask:
        Boolean mask on lat/lon coordinates. Used to find basis on sub-region.

    Returns:
        xr.DataArray with lat/lon dimensions and basis regions encoded by integers.
    """
    flux, footprints = _flux_fp_from_fp_all(fp_all, emissions_name)
    fps = _mean_fp_times_mean_flux(flux, footprints, abs_flux=abs_flux, mask=mask)

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
    emissions_name: Optional[list[str]] = None,
    nbasis: int = 100,
    abs_flux: bool = False,
    mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Basis functions calculated using a weighted region approach
    where each basis function / scaling region contains approximately
    the same value

    Args:
      fp_all:
        fp_all dictionary object as produced from get_data functions
      start_date:
        Start date of period of inference
      emissions_name:
        List of keyword "source" args used for retrieving emissions files
        from the Object store.
      nbasis:
        Desired number of basis function regions
      abs_flux:
        When set to True uses absolute values of a flux array
      mask:
        Boolean mask on lat/lon coordinates. Used to find basis on sub-region.

    Returns:
        xr.DataArray with lat/lon dimensions and basis regions encoded by integers.
    """
    flux, footprints = _flux_fp_from_fp_all(fp_all, emissions_name)
    fps = _mean_fp_times_mean_flux(flux, footprints, abs_flux=abs_flux, mask=mask)

    # use xr.apply_ufunc to keep xarray coords
    func = partial(weighted_algorithm, nregion=nbasis, bucket=1)
    bucket_basis = xr.apply_ufunc(func, fps)

    bucket_basis = bucket_basis.expand_dims({"time": [pd.to_datetime(start_date)]}, axis=-1)
    bucket_basis = bucket_basis.rename("basis")  # this will be used in merges

    bucket_basis.attrs["creator"] = getpass.getuser()
    bucket_basis.attrs["date created"] = str(pd.Timestamp.today())

    return bucket_basis
