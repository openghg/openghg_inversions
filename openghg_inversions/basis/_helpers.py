"""Functions to create fit basis functiosn and apply to data."""

import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies, sparse_xr_dot, to_dense
from openghg_inversions.basis.basis_functions import BasisFunctions
from ._functions import basis_boundary_conditions


def fp_sensitivity(fp_and_data: dict, basis_func: xr.DataArray | dict[str, xr.DataArray]) -> dict:
    """Add a sensitivity matrix, H, to each site xr.Dataset in fp_and_data.

    Deprecated:
        Prefer ``BasisFunctions.sensitivity`` for new code. This legacy helper
        is retained for callers that still pass flat basis arrays directly.

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
    split_by_sectors = bool(fp_and_data.get(".split_by_sectors", len(flux_sources) > 1))

    if not split_by_sectors:
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

    if "time" in basis_func.dims and basis_func.sizes["time"] <= 1:
        basis_func = basis_func.squeeze("time")

    fp_and_data[".basis"] = basis_func

    for site in sites:
        sensitivity = apply_fp_basis_functions(
            fp_x_flux=fp_and_data[site][fp_x_flux_name],
            basis_func=basis_func,
        )
        fp_and_data[site]["H"] = sensitivity

    return fp_and_data


def apply_basis_functions_sensitivity(fp_and_data: dict, basis_functions: BasisFunctions) -> dict:
    """Add sensitivity matrices using ``BasisFunctions.sensitivity``.

    This is the wrapper-internal replacement for ``fp_sensitivity``. It still
    records the flat ``.basis`` side channel because legacy fixedbasisMCMC
    postprocessing reads that field directly.
    """
    sites = [key for key in list(fp_and_data.keys()) if key[0] != "."]
    flux_sources = list(fp_and_data[".flux"].keys())
    split_by_sectors = bool(fp_and_data.get(".split_by_sectors", len(flux_sources) > 1))
    fp_x_flux_name = "fp_x_flux_sectoral" if split_by_sectors else "fp_x_flux"

    fp_and_data[".basis"] = _legacy_flat_basis_for_fp_data(
        basis_functions=basis_functions,
        flux_sources=flux_sources,
        split_by_sectors=split_by_sectors,
    )

    for site in sites:
        sensitivity = basis_functions.sensitivity(fp_and_data[site][fp_x_flux_name])
        state_dim = basis_functions.operator.meta.state_dim
        if state_dim != "region" and state_dim in sensitivity.dims:
            sensitivity = sensitivity.rename({state_dim: "region"})
            state_dim = "region"
        if split_by_sectors:
            sensitivity = _legacy_multisource_h_if_needed(
                sensitivity,
                state_dim=state_dim,
                flux_sources=flux_sources,
            )
        fp_and_data[site]["H"] = sensitivity

    return fp_and_data


def _legacy_flat_basis_for_fp_data(
    *,
    basis_functions: BasisFunctions,
    flux_sources: list[str],
    split_by_sectors: bool,
) -> xr.DataArray:
    """Return the flat basis side channel expected by legacy postprocessing."""
    basis_func = basis_functions.flat_basis()

    if not split_by_sectors:
        if not isinstance(basis_func, xr.DataArray):
            basis_func = next(iter(basis_func.values()))
    elif isinstance(basis_func, dict):
        if len(basis_func) == 1:
            basis_func = next(iter(basis_func.values()))
        elif all(fs in basis_func for fs in flux_sources):
            basis_func = xr.concat(
                [bf.expand_dims({"source": [key]}) for key, bf in basis_func.items()],
                dim="source",
                join="outer",
            )
        else:
            raise ValueError(
                "There should either only be one basis_func, or it should be a dictionary keyed by sources."
            )

    if "time" in basis_func.dims and basis_func.sizes["time"] <= 1:
        basis_func = basis_func.squeeze("time")

    return basis_func


def _legacy_multisource_h_if_needed(
    sensitivity: xr.DataArray,
    *,
    state_dim: str,
    flux_sources: list[str],
) -> xr.DataArray:
    """Convert gathered multi-source H to legacy ``(region, time, source)`` shape.

    ``MultiSourceBucketBasisOperator.sensitivity`` returns a gathered MultiIndex
    state dimension. Current multisector model builders and legacy output code
    still expect separate ``region`` and ``source`` dimensions, so keep this
    translation at the wrapper boundary until downstream code accepts gathered H.
    """
    source_dim = "source"
    region_in_source_dim = "region_in_source"
    if source_dim in sensitivity.dims:
        return sensitivity
    if state_dim not in sensitivity.dims or source_dim not in sensitivity.coords:
        return sensitivity

    legacy_h = sensitivity.unstack(state_dim).fillna(0)
    if region_in_source_dim in legacy_h.dims:
        legacy_h = legacy_h.rename({region_in_source_dim: "region"})
    if source_dim not in legacy_h.dims or "region" not in legacy_h.dims:
        return sensitivity

    legacy_h = legacy_h.reindex({source_dim: flux_sources}).fillna(0)
    dim_order = ["region"]
    if "time" in legacy_h.dims:
        dim_order.append("time")
    dim_order.append(source_dim)
    dim_order.extend(str(dim) for dim in legacy_h.dims if str(dim) not in dim_order)
    return legacy_h.transpose(*dim_order)


def apply_fp_basis_functions(
    fp_x_flux: xr.DataArray,
    basis_func: xr.DataArray,
) -> xr.DataArray:
    """Computes sensitivity matrix `H` for one site.

    See `fp_sensitivity` for more info about the sensitivity matrix.

    # TODO: accept more complex basis functions
    # TODO: accept time varying basis functions?

    Args:
        fp_x_flux: xr.DataArray from `ModelScenario.footprints_data_merge`, e.g. `fp_all["TAC"].fp_x_flux` or
            `fp_all["TAC"].fp_x_flux_sectoral`.
        basis_func: basis functions with integer values in lat/lon grid cells

    Returns:
        sensitivity ("H") xr.DataArray
    """
    # add squeeze just in case this function is used directly
    if "time" in basis_func.dims and basis_func.sizes["time"] <= 1:
        basis_func = basis_func.squeeze("time")

    _, basis_aligned = xr.align(fp_x_flux.isel(time=0), basis_func, join="override")
    basis_mat = get_xr_dummies(basis_aligned, cat_dim="region")
    sensitivity = sparse_xr_dot(basis_mat, fp_x_flux.fillna(0.0), dim=["lat", "lon"])

    if sensitivity.dims[:2] != ("region", "time"):
        sensitivity = sensitivity.transpose("region", "time", ...)

    return to_dense(sensitivity)


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
            sensitivity = bc_ds.sum(["lat", "lon", "height"]).to_dataarray(dim="bc_region")
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
        sensitivity = (
            (bc_ds * bc_basis).sum(["lat", "lon", "height"]).to_dataarray(dim="__newdim__").sum("__newdim__")
        )
        sensitivity = sensitivity.rename(region="bc_region")
        fp_and_data[site]["H_bc"] = sensitivity

    return fp_and_data
