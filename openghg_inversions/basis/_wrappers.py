"""
Functions to calling basis function algorithsm and applying basis functions to data.
"""
from typing import Optional

import openghg_inversions.basis_functions as basis
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
