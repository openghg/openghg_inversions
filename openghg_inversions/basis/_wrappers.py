"""
Functions to calling basis function algorithsm and applying basis functions to data.
"""
from typing import cast, Optional

import numpy as np
import openghg_inversions.basis_functions as basis
import xarray as xr
from openghg_inversions import utils


def basis_functions_wrapper(
    fp_all: dict,
    species: str,
    sites: list[str],
    domain: str,
    start_date: str,
    emissions_name: list[str],
    nbasis: int,
    use_bc: bool,
    basis_algorithm: Optional[str] = None,
    fp_basis_case: Optional[str] = None,
    bc_basis_case: Optional[str] = None,
    basis_directory: Optional[str] = None,
    bc_basis_directory: Optional[str] = None,
    outputname: Optional[str] = None,
    output_path: Optional[str] = None,
):
    """
    Wrapper function for selecting basis function
    algorithm.
    -----------------------------------
    Args:
      basis_algorithm (str):
        One of "quadtree" (for using Quadtree algorithm) or
        "weighted" (for using an algorihtm that splits region
        by input data).

        NB. Land-sea separation is not imposed in the quadtree
        basis functions, but is imposed by default in "weighted"

    nbasis (int):
      Number of basis function regions to calculated in domain
    fp_basis_case (str):
      Name of basis function to use for emissions.
    bc_basis_case (str, optional):
      Name of basis case type for boundary conditions (NOTE, I don't
      think that currently you can do anything apart from scaling NSEW
      boundary conditions if you want to scale these monthly.)
    basis_directory (str, optional):
      Directory containing the basis function
      if not default.
    bc_basis_directory (str, optional):
      Directory containing the boundary condition basis functions
      (e.g. files starting with "NESW")
    use_bc (bool):
      Option to include/exclude boundary conditions in inversion
    fp_all (dict):
      Dictionary object produced from get_data functions
    species (str):
      Atmospheric trace gas species of interest
    sites (str/list):
      List of sites of interest
    domain (str):
      Model domain
    start_date (str):
      Start date of period of inference
    emissions_name (str/list):
      Emissions dataset key words for retrieving from object store
    outputname (str):
      File output name
    output_path (str):
      Passed to `outputdir` argument of `quadtreebasisfunction`. Used for testing.


    Returns:
      fp_data (dict):
        Dictionary object similar to fp_all but with information
        on basis functions and sensitivities
    """
    if fp_basis_case is not None:
        if basis_algorithm:
            print(
                f"Basis algorithm {basis_algorithm} and basis case {fp_basis_case} supplied; using {fp_basis_case}."
            )
        basis_func = utils.basis(domain=domain, basis_case=fp_basis_case, basis_directory=basis_directory)

    elif basis_algorithm is None:
        raise ValueError("One of `fp_basis_case` or `basis_algorithm` must be specified.")

    elif basis_algorithm == "quadtree":
        print("Using Quadtree algorithm to derive basis functions")
        basis_func = basis.quadtreebasisfunction(
            fp_all,
            sites,
            start_date,
            domain,
            species,
            emissions_name,
            outputname,
            nbasis=nbasis,
            outputdir=output_path,
        )

    elif basis_algorithm == "weighted":
        print("Using weighted by data algorithm to derive basis functions")
        basis_func = basis.bucketbasisfunction(
            emissions_name,
            fp_all,
            sites,
            start_date,
            domain,
            species,
            outputname,
            outputdir=output_path,
            nbasis=nbasis,
        )

    else:
        raise ValueError(
            "Basis algorithm not recognised. Please use either 'quadtree' or 'weighted', or input a basis function file"
        )

    fp_data = utils.fp_sensitivity(fp_all, basis_func=basis_func)

    if use_bc is True:
        fp_data = utils.bc_sensitivity(
            fp_data,
            domain=domain,
            basis_case=bc_basis_case,
            bc_basis_directory=bc_basis_directory,
        )

    return fp_data


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
) -> np.ndarray:
    if abs_flux is True:
        print("Using absolute value of flux array.")
        flux = abs(flux)

    mean_flux = flux.mean("time")

    fp_total = sum(footprints)
    n_measure = sum(len(fp.time) for fp in footprints)

    fp_total = cast(xr.DataArray, fp_total)  # otherwise mypy complains about the next line
    mean_fp = fp_total.sum("time") / n_measure

    if mask is not None:
        mean_fp, mean_flux, mask = xr.align(mask, mean_fp, mean_flux, join="override")
        result = mean_fp * mean_flux
        return result.where(mask, drop=True).values

    mean_fp, mean_flux = xr.align(mean_fp, mean_flux, join="override")
    return (mean_fp * mean_flux).values
