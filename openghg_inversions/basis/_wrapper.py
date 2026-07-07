"""Functions to calling basis function algorithms and applying basis functions to data."""

from pathlib import Path
from time import time

import xarray as xr

from ._functions import basis_functions, fixed_outer_regions_basis, basis
from ._helpers import fp_sensitivity, bc_sensitivity


def _nested_domain(domain: str, inner_domain: str) -> str:
    return f"{domain}-{inner_domain}"


def _fp_x_flux_variable(dataset: xr.Dataset) -> str:
    if "fp_x_flux" in dataset.data_vars:
        return "fp_x_flux"
    if "fp_x_flux_sectoral" in dataset.data_vars:
        return "fp_x_flux_sectoral"
    raise ValueError("Could not find fp_x_flux or fp_x_flux_sectoral in merged scenario data.")


def _total_fp_x_flux_sensitivity(data_array: xr.DataArray) -> float:
    """Return total fp_x_flux sensitivity magnitude."""
    return float(abs(data_array).fillna(0.0).sum().compute().values)


def _auto_distribute_nested_nbasis(fp_all: dict, total_nbasis: int) -> tuple[int, int]:
    """Split a total nested basis budget using a damped fp_x_flux sensitivity ratio."""
    if total_nbasis < 2:
        raise ValueError("Nested auto basis distribution requires nbasis >= 2.")

    outer_total = 0.0
    inner_total = 0.0

    for site, site_entry in fp_all.items():
        if site.startswith(".") or not isinstance(site_entry, xr.DataTree):
            continue
        if "standard" not in site_entry.children or "inner" not in site_entry.children:
            continue

        standard_ds = site_entry["standard"].ds
        inner_ds = site_entry["inner"].ds
        outer_total += _total_fp_x_flux_sensitivity(standard_ds[_fp_x_flux_variable(standard_ds)])
        inner_total += _total_fp_x_flux_sensitivity(inner_ds[_fp_x_flux_variable(inner_ds)])

    raw_total = outer_total + inner_total
    if raw_total > 0.0:
        raw_inner_share = inner_total / raw_total
    else:
        raw_inner_share = 0.5

    damped_outer_total = outer_total ** 0.5
    damped_inner_total = inner_total ** 0.5
    damped_total = damped_outer_total + damped_inner_total
    if damped_total > 0.0:
        damped_inner_share = damped_inner_total / damped_total
    else:
        damped_inner_share = 0.5

    inner_share = min(0.60, max(0.35, damped_inner_share))

    inner_nbasis = max(1, min(total_nbasis - 1, int(round(total_nbasis * inner_share))))
    outer_nbasis = total_nbasis - inner_nbasis

    print(
        "DIAGNOSTIC basis_budget | "
        f"mode=auto_damped total_nbasis={total_nbasis} outer_nbasis={outer_nbasis} inner_nbasis={inner_nbasis} "
        f"outer_fp_x_flux_total={outer_total:.6e} inner_fp_x_flux_total={inner_total:.6e} "
        f"raw_inner_share={raw_inner_share:.3f} damped_inner_share={damped_inner_share:.3f} "
        f"bounded_inner_share={inner_share:.3f}",
        flush=True,
    )
    return outer_nbasis, inner_nbasis


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
    outer_region_definition_file: str | Path | None = None,
    outputname: str | None = None,
    output_path: str | None = None,
    inner_domain: str | None = None,
    inner_nbasis: int | None = None,
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
      outer_region_definition_file (str/Path, optional):
        InTEM outer-region definition file to use when `fix_outer_regions`
        is True. If None, the default `outer_region_definition_<domain>.nc`
        lookup is used.
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
      inner_nbasis (int, optional):
        Number of basis regions to use for the inner domain. If not set for
        nested inversions, `nbasis` is treated as the total outer+inner basis
        budget and split by a damped, bounded inner/outer fp_x_flux sensitivity ratio.

    Returns:
      fp_data (dict):
        Dictionary object similar to fp_all but with information
        on basis functions and sensitivities
    """
    inner_basis_data_array = None
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
            if inner_domain is not None:
                if inner_nbasis is None:
                    nbasis, inner_nbasis = _auto_distribute_nested_nbasis(fp_all, nbasis)
                else:
                    print(
                        "DIAGNOSTIC basis_budget | "
                        f"mode=explicit outer_nbasis={nbasis} inner_nbasis={inner_nbasis}",
                        flush=True,
                    )

                basis_function = basis_functions[basis_algorithm]
                inner_basis_data_array = basis_function.algorithm(
                    fp_all=fp_all,
                    start_date=start_date,
                    domain=_nested_domain(domain, inner_domain),
                    emissions_name=emissions_name,
                    nbasis=inner_nbasis,
                    country_directory=country_directory,
                    scenario="inner",
                )
                print(
                    f"Using {basis_function.description} to derive inner-domain basis functions."
                )

            basis_data_array = fixed_outer_regions_basis(
                fp_all,
                start_date,
                basis_algorithm,
                domain,
                emissions_name,
                nbasis,
                country_directory,
                region_definition_file=outer_region_definition_file,
            )
        except KeyError as e:
            raise ValueError(
                "Basis algorithm not recognised. Please use either 'quadtree' or 'weighted', or input a basis function file"
            ) from e
        print(
            f"Using InTEM regions with {basis_algorithm} to derive standard-domain fixed-region basis functions."
        )

    else:
        try:
            basis_function = basis_functions[basis_algorithm]
        except KeyError as e:
            raise ValueError(
                "Basis algorithm not recognised. Please use either 'quadtree' or 'weighted', or input a basis function file"
            ) from e
        print(f"Using {basis_function.description} to derive basis functions.")

        if inner_domain is not None:
            if inner_nbasis is None:
                nbasis, inner_nbasis = _auto_distribute_nested_nbasis(fp_all, nbasis)
            else:
                print(
                    "DIAGNOSTIC basis_budget | "
                    f"mode=explicit outer_nbasis={nbasis} inner_nbasis={inner_nbasis}",
                    flush=True,
                )
            inner_basis_data_array = basis_function.algorithm(
                fp_all=fp_all,
                start_date=start_date,
                domain=f"{domain}-{inner_domain}",
                emissions_name=emissions_name,
                nbasis=inner_nbasis,
                country_directory=country_directory,
                scenario="inner",
            )

            print(f"Computing inner basis took {time() - basis_start}s.")

        basis_data_array = basis_function.algorithm(
            fp_all=fp_all,
            start_date=start_date,
            domain=domain,
            emissions_name=emissions_name,
            nbasis=nbasis,
            country_directory=country_directory,
        )

    print(f"Computing basis took {time() - basis_start}s.")

    fp_sens_start = time()
    if basis_data_array is not None:
        fp_data = fp_sensitivity(
            fp_all,
            basis_func=basis_data_array,
            inner_basis_func=inner_basis_data_array,
        )
        print(f"Computing fp sensitivity took {time() - fp_sens_start}s.")

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
        _save_basis(
            basis=basis_data_array,
            basis_algorithm=basis_algorithm,
            output_dir=output_path,
            domain=domain,
            species=species,
            output_name=outputname,
        )
        if inner_basis_data_array is not None and inner_domain is not None:
            _save_basis(
                basis=inner_basis_data_array,
                basis_algorithm=basis_algorithm,
                output_dir=output_path,
                domain=f"{domain}-{inner_domain}",
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
