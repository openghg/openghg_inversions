"""Functions to calling basis function algorithms and applying basis functions to data."""

from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Literal, cast

import xarray as xr

from .basis_functions import (
    BASIS_ARTIFACT_SOURCE_ATTR,
    BasisFunctions,
    basis_functions_from_fp_all_flat_basis,
    flux_from_fp_all,
)
from ._functions import basis_functions, fixed_outer_regions_basis, basis, openghginv_path
from ._helpers import apply_basis_functions_sensitivity, bc_sensitivity


def make_basis_functions(
    *,
    fp_all: dict,
    species: str,
    domain: str,
    start_date: str,
    emissions_name: list[str] | None,
    nbasis: int,
    basis_algorithm: str | None = None,
    fix_outer_regions: bool = False,
    fp_basis_case: str | None = None,
    basis_directory: str | None = None,
    country_directory: str | None = None,
    outputname: str | None = None,
    output_path: str | None = None,
    basis_output_format: Literal["legacy", "datatree"] = "legacy",
) -> BasisFunctions:
    """Create or load retained emissions basis functions.

    This helper owns basis artifact generation/loading without applying the
    legacy fixedbasis ``fp_data`` side channels.
    """
    basis_data_array: xr.DataArray | Mapping[str, xr.DataArray] | None = None
    basis_start = time()

    if fp_basis_case is not None:
        if basis_algorithm:
            print(
                f"Basis algorithm {basis_algorithm} and basis case {fp_basis_case} supplied; using {fp_basis_case}."
            )
        basis_functions_object = load_basis_functions(
            fp_all=fp_all,
            domain=domain,
            basis_case=fp_basis_case,
            basis_directory=basis_directory,
        )

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
        print("Using generated in-memory basis artifact.")
        basis_functions_object = basis_functions_from_fp_all_flat_basis(
            fp_all=fp_all,
            basis_flat=basis_data_array,
            metadata={BASIS_ARTIFACT_SOURCE_ATTR: "generated"},
        )

    else:
        try:
            basis_function = basis_functions[basis_algorithm]
        except KeyError as e:
            raise ValueError(
                "Basis algorithm not recognised. Please use either 'quadtree' or 'weighted', or input a basis function file"
            ) from e
        print(f"Using {basis_function.description} to derive basis functions.")
        basis_candidate = basis_function.algorithm(
            fp_all, start_date, domain, emissions_name, nbasis, country_directory=country_directory
        )
        if not isinstance(basis_candidate, (xr.DataArray, Mapping)):
            raise TypeError(
                f"Basis algorithm {basis_algorithm!r} returned unsupported basis data "
                f"{type(basis_candidate)!r}."
            )
        basis_data_array = cast(xr.DataArray | Mapping[str, xr.DataArray], basis_candidate)
        print("Using generated in-memory basis artifact.")
        basis_functions_object = basis_functions_from_fp_all_flat_basis(
            fp_all=fp_all,
            basis_flat=basis_data_array,
            metadata={BASIS_ARTIFACT_SOURCE_ATTR: "generated"},
        )

    print(f"Computing basis took {time() - basis_start}s.")

    if output_path is not None and basis_algorithm is not None and fp_basis_case is None:
        if not isinstance(basis_data_array, xr.DataArray):
            raise TypeError("Saving generated basis output currently requires a single flat basis DataArray.")
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
                basis_functions=basis_functions_object,
                basis=basis_data_array,
                basis_algorithm=basis_algorithm,
                output_dir=output_path,
                domain=domain,
                species=species,
                output_name=outputname,
            )
        else:
            raise ValueError(
                f"Unknown basis_output_format '{basis_output_format}'. Expected one of: 'legacy', 'datatree'."
            )

    return basis_functions_object


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

    basis_functions_object = make_basis_functions(
        fp_all=fp_all,
        species=species,
        domain=domain,
        start_date=start_date,
        emissions_name=emissions_name,
        nbasis=nbasis,
        basis_algorithm=basis_algorithm,
        fix_outer_regions=fix_outer_regions,
        fp_basis_case=fp_basis_case,
        basis_directory=basis_directory,
        country_directory=country_directory,
        outputname=outputname,
        output_path=output_path,
        basis_output_format=basis_output_format,
    )

    fp_sens_start = time()
    fp_data = apply_basis_functions_sensitivity(fp_all, basis_functions_object)
    print(f"Computing fp sensitivity took {time() - fp_sens_start}s.")

    basis_objects: dict[str, BasisFunctions] = {}
    if return_basis_objects:
        basis_objects["emissions"] = basis_functions_object

    if use_bc is True:
        bc_sens_start = time()
        fp_data = bc_sensitivity(
            fp_data,
            domain=domain,
            basis_case=bc_basis_case,  # type: ignore ...check ensures bc_basis_case not None if use_bc True
            bc_basis_directory=bc_basis_directory,
        )
        print(f"Computing bc sensitivity took {time() - bc_sens_start}s.")

    if return_basis_objects:
        return fp_data, basis_objects

    return fp_data


def load_basis_functions(
    *,
    fp_all: dict,
    domain: str,
    basis_case: str,
    basis_directory: str | Path | None = None,
) -> BasisFunctions:
    """Load a saved basis artifact as retained ``BasisFunctions``.

    DataTree artifacts are preferred when the matching file carries the
    ``openghg_inversions.flux_weighted_basis`` schema and are loaded through
    ``BasisFunctions.load``. Otherwise the existing legacy flat artifact loader
    is used and a retained basis object is built from runtime flux in
    ``fp_all``.

    Args:
        fp_all: Legacy merged-data dictionary used to build runtime flux when
            adapting legacy flat artifacts or replacing serialized DataTree flux.
        domain: Inversion domain used in the artifact path convention.
        basis_case: Basis case prefix used in the artifact path convention.
        basis_directory: Optional root directory containing per-domain basis
            artifact subdirectories.

    Returns:
        Loaded retained basis object with ``basis_artifact_source`` metadata.

    Raises:
        FileNotFoundError: If no matching artifact files are found.
        ValueError: If more than one matching DataTree artifact is found.
    """
    files = _basis_artifact_files(domain=domain, basis_case=basis_case, basis_directory=basis_directory)
    datatree_files = [file for file in files if _is_basis_datatree_artifact(file)]

    if datatree_files:
        if len(datatree_files) > 1:
            files_text = "\n".join(f"  - {file}" for file in datatree_files)
            raise ValueError(
                "DataTree basis artifact loading currently supports one matching file, but found "
                f"{len(datatree_files)} for basis_case={basis_case!r}, domain={domain!r}:\n"
                f"{files_text}\n"
                "Use a more specific basis_case or remove/rename stale DataTree basis artifacts."
            )
        basis_functions = BasisFunctions.load(datatree_files[0])
        print(f"Loaded DataTree basis artifact: {datatree_files[0]}")
        current_flux = flux_from_fp_all(fp_all)
        basis_functions = basis_functions.with_flux(current_flux).with_metadata(
            {BASIS_ARTIFACT_SOURCE_ATTR: "datatree"}
        )
        if "source" in current_flux.dims:
            basis_functions = basis_functions.select_sources(
                [str(source) for source in current_flux.source.values]
            )
        return basis_functions

    basis_data_array = basis(
        domain=domain,
        basis_case=basis_case,
        basis_directory=str(basis_directory) if isinstance(basis_directory, Path) else basis_directory,
    ).basis
    print(f"Loaded legacy flat basis artifact for basis_case={basis_case!r}, domain={domain!r}.")
    return basis_functions_from_fp_all_flat_basis(
        fp_all=fp_all,
        basis_flat=basis_data_array,
        metadata={BASIS_ARTIFACT_SOURCE_ATTR: "legacy_flat"},
    )


def _basis_artifact_files(
    *,
    domain: str,
    basis_case: str,
    basis_directory: str | Path | None = None,
) -> list[Path]:
    """Find basis artifact files using the legacy basis directory convention."""
    basis_path = Path(basis_directory) if basis_directory is not None else openghginv_path / "basis_functions"
    files = sorted((basis_path / domain).glob(f"{basis_case}_{domain}*.nc"))
    if not files:
        raise FileNotFoundError(
            f"Can't find basis function files for domain '{domain}' and basis_case '{basis_case}' "
        )
    return files


def _is_basis_datatree_artifact(path: Path) -> bool:
    """Return true when a file contains the BasisFunctions DataTree schema."""
    try:
        with xr.open_datatree(path) as dt:
            return dt.attrs.get("schema") == "openghg_inversions.flux_weighted_basis"
    except (OSError, ValueError, KeyError):
        return False


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
    """Save ``BasisFunctions`` using the wrapper's DataTree file convention.

    This is an opt-in serialization path around ``BasisFunctions.save``. The
    legacy flat basis writer remains the default to preserve backwards
    compatibility.
    """
    basis_out_path = Path(output_dir, domain.upper())

    if not basis_out_path.exists():
        basis_out_path.mkdir(parents=True)

    start_date = str(basis.time.min().values)[:7]  # year and month

    if output_name is None:
        output_name = f"{basis_algorithm}_{species}_{domain}_{start_date}_basis_datatree.nc"
    else:
        output_name = f"{basis_algorithm}_{species}-{output_name}_{domain}_{start_date}_basis_datatree.nc"

    basis_functions.save(basis_out_path / output_name)
