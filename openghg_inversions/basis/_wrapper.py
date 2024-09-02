"""Functions to calling basis function algorithms and applying basis functions to data."""

from pathlib import Path

import xarray as xr

from ._functions import basis_functions, fixed_outer_regions_basis, basis
from ._helpers import fp_sensitivity, bc_sensitivity


def basis_functions_wrapper(
    fp_all: dict,
    species: str,
    domain: str,
    start_date: str,
    emissions_name: list[str] | None,
    nbasis: int,
    use_bc: bool,
    basis_algorithm: str | None = None,
    fix_outer_regions: bool = False,
    fp_basis_case: str | None = None,
    bc_basis_case: str | None = None,
    basis_directory: str | None = None,
    bc_basis_directory: str | None = None,
    outputname: str | None = None,
    output_path: str | None = None,
):
    """Wrapper function for selecting basis function
    algorithm.

    Args:
      fp_all (dict):
        Dictionary object produced from get_data functions
      species (str):
        Atmospheric trace gas species of interest
      domain (str):
        Model domain
      start_date (str):
        Start date of period of inference
      emissions_name (str/list):
        Emissions dataset key words for retrieving from object store
      nbasis (int):
        Number of basis function regions to calculated in domain
      use_bc (bool):
        Option to include/exclude boundary conditions in inversion
      basis_algorithm (str, optional):
        One of "quadtree" (for using Quadtree algorithm) or
        "weighted" (for using an algorihtm that splits region
        by input data). Land-sea separation is not imposed in the
        quadtree basis functions, but is imposed by default in "weighted"
        Default None
      fixed_outer_region (bool):
        When set to True uses InTEM regions to derive basis functions for inner region
        Default False
      fp_basis_case (str):
        Name of basis function to use for emissions.
        Default None
      bc_basis_case (str, optional):
        Name of basis case type for boundary conditions (NOTE, I don't
        think that currently you can do anything apart from scaling NSEW
        boundary conditions if you want to scale these monthly.)
        Default None
      basis_directory (str, optional):
        Directory containing the basis function if not default.
        Default None
      bc_basis_directory (str, optional):
        Directory containing the boundary condition basis functions
        (e.g. files starting with "NESW")
        Default None
      outputname (str, optional):
        File output name
        Default None
      output_path (str, optional):
        Passed to `outputdir` argument of `quadtreebasisfunction`. Used for testing.
        Default None

    Returns:
      fp_data (dict):
        Dictionary object similar to fp_all but with information
        on basis functions and sensitivities
    """
    if use_bc is True and bc_basis_case is None:
        raise ValueError("If `use_bc` is True, you must specify `bc_basis_case`.")

    if fp_basis_case is not None:
        if basis_algorithm:
            print(
                f"Basis algorithm {basis_algorithm} and basis case {fp_basis_case} supplied; using {fp_basis_case}."
            )
        basis_data_array = basis(
            domain=domain, basis_case=fp_basis_case, basis_directory=basis_directory
        ).basis

    elif basis_algorithm is None:
        raise ValueError("One of `fp_basis_case` or `basis_algorithm` must be specified.")

    elif fix_outer_regions is True:
        try:
            basis_data_array = fixed_outer_regions_basis(
                fp_all, start_date, basis_algorithm, domain, emissions_name, nbasis
            )
        except KeyError as e:
            raise ValueError(
                "Basis algorithm not recognised. Please use either 'quadtree' or 'weighted', or input a basis function file"
            ) from e
        print(f"Using InTEM regions with {basis_algorithm} to derive basis functions for inner region.")

    else:
        try:
            basis_function = basis_functions[basis_algorithm]
        except KeyError as e:
            raise ValueError(
                "Basis algorithm not recognised. Please use either 'quadtree' or 'weighted', or input a basis function file"
            ) from e
        print(f"Using {basis_function.description} to derive basis functions.")
        basis_data_array = basis_function.algorithm(fp_all, start_date, domain, emissions_name, nbasis)

    fp_data = fp_sensitivity(fp_all, basis_func=basis_data_array)

    if use_bc is True:
        fp_data = bc_sensitivity(
            fp_data,
            domain=domain,
            basis_case=bc_basis_case,  # type: ignore ...check ensures bc_basis_case not None if use_bc True
            bc_basis_directory=bc_basis_directory,
        )

    if output_path is not None and basis_algorithm is not None and fp_basis_case is None:
        _save_basis(
            basis=basis_data_array,
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
    output_name: str | None = None,
) -> None:
    """Save basis functions to netCDF.

    Args:
      basis (xarray.DataArray):
        basis dataset to save
      basis_algorithm (str):
        name of basis algorithm (e.g. "quadtree" or "weighted")
      output_dir (str):
        root directory to save basis functions
      domain (str):
        domain of inversion; basis is saved in a "domain" directory inside `output_dir`
      species (str):
        species of inversion
      output_name (str,optional):
        File output name
        Default None

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
