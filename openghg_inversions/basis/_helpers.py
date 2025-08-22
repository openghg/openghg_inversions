"""Functions to create fit basis functiosn and apply to data."""

from openghg.util import get_species_info, synonyms
import xarray as xr
import numpy as np


from openghg_inversions import convert
from openghg_inversions.utils import combine_datasets
from openghg_inversions.array_ops import get_xr_dummies, sparse_xr_dot
from ._functions import basis_boundary_conditions


def fp_sensitivity(fp_and_data: dict, basis_func: xr.DataArray | dict[str, xr.DataArray]) -> dict:
    """Add a sensitivity matrix, H, to each site xr.Dataset in fp_and_data.

    The sensitivity matrix H takes the footprint sensitivities (the `fp` variable),
    multiplies it by the flux files, then aggregates over the basis regions.

    The basis functions can have one of two forms:
    - a xr.DataArray with lat/lon coordinates, and positive integer values, where all
      lat/lon pairs with value == i form the i-th basis region
    - a xr.DataArray with coordinates: lat, lon, region. For each fixed region value, there is
      a lat-lon grid with 1 in region and 0 outside region.

    Region numbering must start from 1

    TODO: describe output coordinates?

    Args:
        fp_and_data: output from `data_processing_surface_notracer`; contains "combined scenarios" keyed by
            site code, as well as fluxes.
        basis_func: basis functions to use; output from `utils.basis` or basis functions in `basis` submodule.
        verbose: if True, print info messages.

    Returns:
        dict in same format as fp_and_data with sensitivity matrix and basis functions added.
    """
    sites = [key for key in list(fp_and_data.keys()) if key[0] != "."]

    flux_sources = list(fp_and_data[".flux"].keys())

    if len(flux_sources) == 1:
        if not isinstance(basis_func, xr.DataArray):
            basis_func = next(iter(basis_func.values()))

        fp_x_flux_name = "fp_x_flux"

    else:
        # multi-sector case
        fp_x_flux_name = "fp_x_flux_sectoral"

        if isinstance(basis_func, dict):
            if len(basis_func) == 1:
                basis_func = next(iter(basis_func.values()))
            elif all(fs in basis_func for fs in flux_sources):
                # concat along sources
                basis_func = xr.concat(
                    [bf.expand_dims({"source": [k]}) for k, bf in basis_func.items()],
                    dim="source",
                    join="outer",
                )
            else:
                raise ValueError(
                    "There should either only be one basis_func, or it should be a dictionary keyed by sources."
                )

    if "time" in basis_func.dims:
        basis_func = basis_func.squeeze("time")

    fp_and_data[".basis"] = basis_func

    for site in sites:
        sensitivity = apply_fp_basis_functions(
            fp_x_flux=fp_and_data[site][fp_x_flux_name],
            basis_func=basis_func,
        )
        fp_and_data[site]["H"] = sensitivity

    return fp_and_data


def apply_fp_basis_functions(
    fp_x_flux: xr.DataArray,
    basis_func: xr.DataArray,
) -> xr.DataArray:
    """Computes sensitivity matrix `H` for one site. See `fp_sensitivity` for
    more info about the sensitivity matrix.

    # TODO: accept more complex basis functions
    # TODO: accept time varying basis functions?

    Args:
        fp_x_flux: xr.DataArray from `ModelScenario.footprints_data_merge`, e.g. `fp_all["TAC"].fp_x_flux` or
            `fp_all["TAC"].fp_x_flux_sectoral`.
        basis_func: basis functions with integer values in lat/lon grid cells

    Returns:
        sensitivity ("H") xr.DataArray
    """
    _, basis_aligned = xr.align(fp_x_flux.isel(time=0), basis_func, join="override")
    basis_mat = get_xr_dummies(basis_aligned, cat_dim="region")
    sensitivity = sparse_xr_dot(basis_mat, fp_x_flux.fillna(0.0), dim=["lat", "lon"]).transpose("region", "time", ...)
    return sensitivity.as_numpy()


def bc_sensitivity(
    fp_and_data: dict, domain: str, basis_case: str, bc_basis_directory: str | None = None
) -> dict:
    """Add boundary conditions sensitivity matrix `H_bc` to each site xr.Dataframe in fp_and_data.

    Args:
        fp_and_data: dict containing xr.Datasets output by `ModelScenario.footprints_data_merge`
            keyed by site code.
        domain: inversion domain. For instance "EUROPE"
        basis_case: BC basis case to read in. Examples of basis cases are "NESW","stratgrad".
        bc_basis_directory: bc_basis_directory can be specified if files are not in the default
            directory. Must point to a directory which contains subfolders organized
            by domain. (optional)

    Returns:
        dict of xr.Datasets in same format as fp_and_data with `H_bc` sensitivity matrix added.

    """
    sites = [key for key in list(fp_and_data.keys()) if key[0] != "."]

    if basis_case.lower() == "nesw":
        for site in sites:
            ds = fp_and_data[site]
            bc_ds = ds[[f"bc_{d}" for d in "nesw"]].rename({f"bc_{d}": d for d in "nesw"})
            sensitivity = bc_ds.sum(["lat", "lon", "height"]).to_dataarray(dim="bc_region").transpose("bc_region", ...)
            fp_and_data[site]["H_bc"] = sensitivity

        return fp_and_data

    basis_func = basis_boundary_conditions(
        domain=domain, basis_case=basis_case, bc_basis_directory=bc_basis_directory
    )

    # drop time if there is only one value
    if basis_func.sizes.get("time", -1) == 1:
        basis_func = basis_func.squeeze("time")
    else:
        basis_func = basis_func.sortby("time")

    # align basis data var names with baseline sensitivity data var names from ModelScenario
    bc_basis = basis_func.rename({dv: str(dv).replace("basis_", "") for dv in basis_func.data_vars})

    for site in sites:
        ds = fp_and_data[site]
        bc_ds = ds[[f"bc_{d}" for d in "nesw"]]
        sensitivity = (bc_ds * bc_basis).sum(["lat", "lon", "height"]).to_dataarray(dim="__newdim__").sum("__newdim__")
        sensitivity = sensitivity.rename(region="bc_region").transpose("bc_region", ...)
        fp_and_data[site]["H_bc"] = sensitivity

    return fp_and_data
