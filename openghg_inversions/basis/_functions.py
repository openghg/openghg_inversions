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


def _save_basis(
    basis: xr.Dataset,
    basis_algorithm: str,
    output_dir: str,
    domain: str,
    species: str,
    output_name: Optional[str] = None,
) -> None:
    """Save basis functions to netCDF.

    Args:
        basis: basis dataset to save
        basis_algorithm: name of basis algorithm (e.g. "quadtree" or "weighted")
        output_dir: root directory to save basis functions
        domain: domain of inversion; basis is saved in a "domain" directory inside `output_dir`
        species: species of inversion
        output_name
    """
    basis_out_path = Path(output_dir, domain.upper())

    if not basis_out_path.exists():
        basis_out_path.mkdir(parents=True)

    start_date = str(basis.time.min().values)[:7]  # year and month

    if output_name is None:
        output_name = f"{basis_algorithm}_{species}_{domain}_{start_date}.nc"
    else:
        output_name = f"{basis_algorithm}_{species}-{output_name}_{domain}_{start_date}.nc"

    basis.to_netcdf(basis_out_path / output_name, mode="w")


def quadtreebasisfunction(
    fp_all: dict,
    start_date: str,
    domain: str,
    species: str,
    emissions_name: Optional[list[str]] = None,
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
    """
    flux, footprints = _flux_fp_from_fp_all(fp_all, emissions_name)
    fps = _mean_fp_times_mean_flux(flux, footprints, abs_flux=abs_flux)

    # use xr.apply_ufunc to keep xarray coords
    func = partial(quadtree_algorithm, nbasis=nbasis, seed=seed)
    quad_basis = xr.apply_ufunc(func, fps)

    quad_basis = quad_basis.expand_dims({"time": [pd.to_datetime(start_date)]}, axis=-1)

    new_ds = xr.Dataset({"basis": quad_basis})
    new_ds.attrs["creator"] = getpass.getuser()
    new_ds.attrs["date created"] = str(pd.Timestamp.today())

    if outputdir is not None:
        _save_basis(
            basis=new_ds,
            basis_algorithm="quadtree",
            output_dir=outputdir,
            domain=domain,
            species=species,
            output_name=outputname,
        )

    return new_ds


def bucketbasisfunction(
    fp_all: dict,
    start_date: str,
    domain: str,
    species: str,
    emissions_name: Optional[list[str]] = None,
    outputname: Optional[str] = None,
    outputdir: Optional[str] = None,
    nbasis: int = 100,
    abs_flux: bool = False,
) -> xr.Dataset:
    """
    Basis functions calculated using a weighted region approach
    where each basis function / scaling region contains approximately
    the same value

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
    flux, footprints = _flux_fp_from_fp_all(fp_all, emissions_name)
    fps = _mean_fp_times_mean_flux(flux, footprints, abs_flux=abs_flux)

    # use xr.apply_ufunc to keep xarray coords
    func = partial(weighted_algorithm, nregion=nbasis, bucket=1)
    bucket_basis = xr.apply_ufunc(func, fps)

    bucket_basis = bucket_basis.expand_dims({"time": [pd.to_datetime(start_date)]}, axis=-1)

    new_ds = xr.Dataset({"basis": bucket_basis})
    new_ds.attrs["creator"] = getpass.getuser()
    new_ds.attrs["date created"] = str(pd.Timestamp.today())

    if outputdir is not None:
        _save_basis(
            basis=new_ds,
            basis_algorithm="weighted",
            output_dir=outputdir,
            domain=domain,
            species=species,
            output_name=outputname,
        )

    return new_ds
