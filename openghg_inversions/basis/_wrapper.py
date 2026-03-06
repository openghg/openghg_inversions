"""Functions to calling basis function algorithms and applying basis functions to data."""

from pathlib import Path
from time import time
from typing import Literal

import xarray as xr
from openghg.analyse._utils import match_dataset_dims, stack_datasets

from openghg_inversions.array_ops import force_align
from .basis_functions import BasisFunctions
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
    country_directory: str | None = None,
    outputname: str | None = None,
    output_path: str | None = None,
    return_basis_objects: bool = False,
    basis_output_format: Literal["legacy", "datatree"] = "legacy",
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
      return_basis_objects (bool, optional):
        If True, return a tuple ``(fp_data, basis_objects)`` where
        ``basis_objects["emissions"]`` is a ``BasisFunctions`` object constructed
        from the basis used in this wrapper.
        Default False
      basis_output_format (str, optional):
        Format to use when saving basis output with ``output_path``:
        - ``"legacy"``: save legacy flat basis netCDF (default)
        - ``"datatree"``: save BasisFunctions DataTree netCDF
        Default "legacy"

    Returns:
      fp_data (dict) or tuple[dict, dict[str, BasisFunctions]]:
        By default, returns a dictionary object similar to fp_all but with information
        on basis functions and sensitivities.

        If ``return_basis_objects=True``, returns ``(fp_data, basis_objects)`` where
        ``basis_objects`` contains an ``"emissions"`` key with a ``BasisFunctions``
        object that wraps the basis operator and representative flux.
    """
    if use_bc is True and bc_basis_case is None:
        raise ValueError("If `use_bc` is True, you must specify `bc_basis_case`.")

    basis_start = time()

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
        print("Using fixed outer regions for basis functions.")
        try:
            basis_data_array = fixed_outer_regions_basis(
                fp_all, start_date, basis_algorithm, domain, emissions_name, nbasis, country_directory
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
        basis_data_array = basis_function.algorithm(fp_all, start_date, domain, emissions_name, nbasis, country_directory=country_directory)

    print(f"Computing basis took {time() - basis_start}s.")

    fp_sens_start = time()
    fp_data = fp_sensitivity(fp_all, basis_func=basis_data_array)
    print(f"Computing fp sensitivity took {time() - fp_sens_start}s.")

    basis_objects: dict[str, BasisFunctions] = {}
    needs_basis_object = return_basis_objects or (output_path is not None and basis_output_format == "datatree")

    if needs_basis_object:
        basis_objects["emissions"] = _make_basis_functions_object(
            fp_all=fp_all,
            basis=basis_data_array,
        )

    if use_bc is True:
        bc_sens_start = time()
        fp_data = bc_sensitivity(
            fp_data,
            domain=domain,
            basis_case=bc_basis_case,  # type: ignore ...check ensures bc_basis_case not None if use_bc True
            bc_basis_directory=bc_basis_directory,
        )
        print(f"Computing bc sensitivity took {time() - bc_sens_start}s.")

    if output_path is not None and basis_algorithm is not None and fp_basis_case is None:
        if basis_output_format == "legacy":
            _save_basis(
                basis=basis_data_array,
                basis_algorithm=basis_algorithm,
                output_dir=output_path,
                domain=domain,
                species=species,
                output_name=outputname,
            )
        elif basis_output_format == "datatree":
            _save_basis_datatree(
                basis_functions=basis_objects["emissions"],
                basis=basis_data_array,
                basis_algorithm=basis_algorithm,
                output_dir=output_path,
                domain=domain,
                species=species,
                output_name=outputname,
            )
        else:
            raise ValueError(
                f"Unknown basis_output_format '{basis_output_format}'. "
                "Expected one of: 'legacy', 'datatree'."
            )

    if return_basis_objects:
        return fp_data, basis_objects

    return fp_data


def _make_basis_functions_object(fp_all: dict, basis: xr.DataArray) -> BasisFunctions:
    """Construct a BasisFunctions object from wrapper inputs.

    The current wrapper computes a single emissions basis array. For non-sector
    workflows, this helper combines all flux sources into one representative flux
    using the same alignment/summing behavior as ``ModelScenario.combine_flux_sources``.
    For sector-split workflows, fluxes are preserved along a ``source`` dimension.
    """
    if ".flux" not in fp_all or not fp_all[".flux"]:
        raise ValueError("Cannot construct BasisFunctions object: fp_all['.flux'] is missing or empty.")

    flux_entries = fp_all[".flux"]
    flux_arrays = {key: _extract_flux_dataarray(value, flux_key=key) for key, value in flux_entries.items()}

    # Follow existing ModelScenario behavior:
    # - single-source workflows combine fluxes by summing over flux entries
    # - multi-source/sector workflows keep per-source fluxes keyed by source
    if _is_multi_source_workflow(fp_all):
        flux = _stack_flux_sources_with_alignment(flux_arrays)
    else:
        flux = _combine_flux_sources_like_modelscenario(flux_arrays)

    return BasisFunctions.from_basis_flat(
        basis_flat=basis,
        flux=flux,
        operator_kwargs={"state_dim": "region"},
    )


def _is_multi_source_workflow(fp_all: dict) -> bool:
    """Determine multi-source/sector mode from explicit fp_all metadata."""
    split_by_sectors = fp_all.get(".split_by_sectors")
    if split_by_sectors is None:
        return False
    return bool(split_by_sectors)


def _extract_flux_dataarray(flux_entry: object, flux_key: str) -> xr.DataArray:
    """Extract a DataArray named ``flux`` from supported flux entry containers."""
    if hasattr(flux_entry, "data") and isinstance(flux_entry.data, xr.Dataset) and "flux" in flux_entry.data:
        return flux_entry.data["flux"]
    if isinstance(flux_entry, xr.Dataset) and "flux" in flux_entry:
        return flux_entry["flux"]
    if isinstance(flux_entry, xr.DataArray):
        return flux_entry

    raise TypeError(
        "Could not extract a flux DataArray from fp_all['.flux']. "
        f"Got type {type(flux_entry)!r} for flux entry {flux_key!r}."
    )


def _combine_flux_sources_like_modelscenario(flux_arrays: dict[str, xr.DataArray]) -> xr.DataArray:
    """Combine fluxes as in ModelScenario.combine_flux_sources."""
    flux_datasets = [
        arr.rename("flux").to_dataset() if arr.name != "flux" else arr.to_dataset()
        for arr in flux_arrays.values()
    ]

    if len(flux_datasets) == 1:
        return flux_datasets[0]["flux"]

    dims = [dim for dim in flux_datasets[0].dims if dim != "time"]
    flux_datasets = match_dataset_dims(flux_datasets, dims=dims)
    if "time" in flux_datasets[0].dims:
        flux_stacked = stack_datasets(flux_datasets, dim="time", method="ffill")
    else:
        flux_stacked = sum(flux_datasets)

    return flux_stacked["flux"]


def _stack_flux_sources_with_alignment(flux_arrays: dict[str, xr.DataArray]) -> xr.DataArray:
    """Stack fluxes along `source`, validating structural coordinate alignment."""
    first_key = next(iter(flux_arrays))
    reference = flux_arrays[first_key]
    dims_to_align = [dim for dim in reference.dims if dim != "time"]

    aligned_flux = {}
    for key, arr in flux_arrays.items():
        aligned_flux[key] = force_align(arr, reference=reference, dims=dims_to_align)

    return xr.concat(
        [arr.expand_dims({"source": [key]}) for key, arr in aligned_flux.items()],
        dim="source",
        join="outer",
    )


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


def _save_basis_datatree(
    basis_functions: BasisFunctions,
    basis: xr.DataArray,
    basis_algorithm: str,
    output_dir: str,
    domain: str,
    species: str,
    output_name: str | None = None,
) -> None:
    """Save BasisFunctions object to netCDF DataTree.

    This is an opt-in serialization path. The legacy flat basis writer remains the
    default to preserve backwards compatibility.
    """
    basis_out_path = Path(output_dir, domain.upper())

    if not basis_out_path.exists():
        basis_out_path.mkdir(parents=True)

    start_date = str(basis.time.min().values)[:7]  # year and month

    if output_name is None:
        output_name = f"{basis_algorithm}_{species}_{domain}_{start_date}_basis_datatree.nc"
    else:
        output_name = f"{basis_algorithm}_{species}-{output_name}_{domain}_{start_date}_basis_datatree.nc"

    dt = basis_functions.to_datatree()
    dt.to_netcdf(basis_out_path / output_name)
