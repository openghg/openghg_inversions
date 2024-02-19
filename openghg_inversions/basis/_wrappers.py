"""
Functions to calling basis function algorithms and applying basis functions to data.
"""
from collections import namedtuple
from pathlib import Path
from typing import Optional

import xarray as xr

from ._functions import bucketbasisfunction, quadtreebasisfunction
from .. import utils


# dict to retrieve basis function and description by algorithm name
BasisFunction = namedtuple("BasisFunction", ["description", "algorithm"])
basis_functions = {
    "quadtree": BasisFunction("quadtree algorithm", quadtreebasisfunction),
    "weighted": BasisFunction("weighted by data algorithm", bucketbasisfunction),
}


def basis_functions_wrapper(
    fp_all: dict,
    species: str,
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
        basis_func = utils.basis(
            domain=domain, basis_case=fp_basis_case, basis_directory=basis_directory
        ).basis

    elif basis_algorithm is None:
        raise ValueError("One of `fp_basis_case` or `basis_algorithm` must be specified.")

    else:
        try:
            basis_function = basis_functions[basis_algorithm]
        except KeyError as e:
            raise ValueError(
                "Basis algorithm not recognised. Please use either 'quadtree' or 'weighted', or input a basis function file"
            ) from e
        else:
            print(f"Using {basis_function.description} to derive basis functions.")
            basis_func = basis_function.algorithm(fp_all, start_date, emissions_name, nbasis)

    fp_data = utils.fp_sensitivity(fp_all, basis_func=basis_func)

    if use_bc is True:
        fp_data = utils.bc_sensitivity(
            fp_data,
            domain=domain,
            basis_case=bc_basis_case,
            bc_basis_directory=bc_basis_directory,
        )

    if output_path is not None and basis_algorithm is not None and fp_basis_case is None:
        _save_basis(
            basis=basis_func,
            basis_algorithm=basis_algorithm,
            output_dir=output_path,
            domain=domain,
            species=species,
            output_name=outputname,
        )

    return fp_data


def _save_basis(
    basis: xr.DataArray,
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

    Returns:
        None. Saves basis dataset to netCDF.
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

