"""Functions to create fit basis functiosn and apply to data."""

import numpy as np
import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies, sparse_xr_dot, to_dense
from ._functions import basis_boundary_conditions


def _inner_extent_mask_to_grid(inner_data: xr.Dataset, lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    """Create a target-grid mask covering the full inner-domain lat/lon extent."""
    lat_min = float(inner_data.lat.min())
    lat_max = float(inner_data.lat.max())
    lon_min = float(inner_data.lon.min())
    lon_max = float(inner_data.lon.max())

    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    target = xr.DataArray(np.zeros((lat.size, lon.size), dtype=bool), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    return (lat_mask & lon_mask).broadcast_like(target)


def _check_basis_covers_sensitivity(
    *,
    basis_func: xr.DataArray,
    fp_x_flux: xr.DataArray,
    label: str,
) -> None:
    """Reject basis value 0 where there is non-zero footprint sensitivity."""
    basis_check = basis_func.squeeze("time") if basis_func.sizes.get("time", 0) == 1 else basis_func
    _, basis_check = xr.align(fp_x_flux.isel(time=0), basis_check, join="override")
    zero_basis = basis_check == 0
    zero_count = int(zero_basis.sum().compute())

    if zero_count:
        print(f"{label} basis contains {zero_count} grid cells with basis value 0.")

    active_on_zero_basis = fp_x_flux.where(zero_basis).fillna(0.0) != 0.0
    active_times = active_on_zero_basis.any(dim=[dim for dim in active_on_zero_basis.dims if dim != "time"])
    active_times = active_times.compute()

    if bool(active_times.any()):
        missing_index = fp_x_flux.time.where(active_times, drop=True).to_index()
        missing_preview = ", ".join(str(t) for t in missing_index[:5])
        extra = "" if missing_index.size <= 5 else f", ... ({missing_index.size} total)"
        raise ValueError(
            f"{label} basis has value 0 in cells with non-zero footprint sensitivity for: "
            f"{missing_preview}{extra}. This would leave sensitivity outside the inversion basis."
        )


def fp_sensitivity(
    fp_and_data: dict,
    basis_func: xr.DataArray | dict[str, xr.DataArray],
    inner_basis_func: xr.DataArray | None = None,
) -> dict:
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

    if "time" in basis_func.dims and basis_func.sizes["time"] <= 1:
        basis_func = basis_func.squeeze("time")

    if inner_basis_func is not None and "time" in inner_basis_func.dims and inner_basis_func.sizes["time"] <= 1:
        inner_basis_func = inner_basis_func.squeeze("time")

    fp_and_data[".basis"] = basis_func
    if inner_basis_func is not None:
        fp_and_data[".basis_inner"] = inner_basis_func

    if inner_basis_func is not None:
        invalid_inner_sites = [
            site
            for site in sites
            if not (
                isinstance(fp_and_data[site], xr.DataTree) and "inner" in fp_and_data[site].children
            )
        ]
        if invalid_inner_sites:
            raise ValueError(
                "Inner basis supplied, but some sites are not DataTree entries with an inner child: "
                f"{invalid_inner_sites}."
            )

    for site in sites:
        entry = fp_and_data[site]


        # extract root fp_x_flux (already masked if inner domain exists)
        if isinstance(entry, xr.DataTree):
            if "standard" in entry.children:
                root_ds = entry["standard"].ds
            else:
                root_ds = entry.ds
            fp_x_flux_outer = root_ds[fp_x_flux_name]

            if "inner" in entry.children:
                inner_ds = entry["inner"].ds
                inner_on_outer = _inner_extent_mask_to_grid(
                    inner_ds,
                    lat=fp_x_flux_outer.lat,
                    lon=fp_x_flux_outer.lon,
                )
                fp_x_flux_outer = fp_x_flux_outer.where(~inner_on_outer, other=0.0)
                root_ds = root_ds.assign({fp_x_flux_name: fp_x_flux_outer})

            # Compute outer H from the (already masked) fp_x_flux
            _check_basis_covers_sensitivity(
                basis_func=basis_func,
                fp_x_flux=fp_x_flux_outer,
                label=f"{site} standard",
            )
            sensitivity = apply_fp_basis_functions(
                fp_x_flux=fp_x_flux_outer,
                basis_func=basis_func,
            )

            # Compute H_inner from the inner child's fp_x_flux (its own lat/lon grid)
            if "inner" in entry.children:
                if inner_basis_func is None:
                    raise ValueError("Inner-domain data exists but no inner basis function was provided.")

                inner_fp_x_flux = entry["inner"].ds[fp_x_flux_name]
                _check_basis_covers_sensitivity(
                    basis_func=inner_basis_func,
                    fp_x_flux=inner_fp_x_flux,
                    label=f"{site} inner",
                )
                H_inner = apply_fp_basis_functions(
                    fp_x_flux=inner_fp_x_flux,
                    basis_func=inner_basis_func,
                )
                # Write both back into the DataTree
                new_root = root_ds.assign({"H": sensitivity})
                new_inner = entry["inner"].ds.assign({"H_inner": H_inner})
                fp_and_data[site] = xr.DataTree.from_dict({
                    "/standard": new_root,
                    "/inner": new_inner,
                })
            else:
                fp_and_data[site] = xr.DataTree.from_dict({
                    "/standard": root_ds.assign({"H": sensitivity})
                })

        else:
            if inner_basis_func is not None:
                raise ValueError(
                    "Inner-domain inversion requires DataTree site entries with an inner child. "
                    f"Site '{site}' is a plain Dataset."
                )

            # Legacy: plain xr.Dataset path — unchanged
            _check_basis_covers_sensitivity(
                basis_func=basis_func,
                fp_x_flux=entry[fp_x_flux_name],
                label=f"{site} standard",
            )
            sensitivity = apply_fp_basis_functions(
                fp_x_flux=entry[fp_x_flux_name],
                basis_func=basis_func,
            )
            fp_and_data[site]["H"] = sensitivity

    return fp_and_data


def combine_inner_outer_fp_x_flux(
    inner_fp_x_flux: xr.DataArray,
    outer_fp_x_flux: xr.DataArray,
) -> xr.DataArray:
    """Merge inner (6km) and outer (EUROPE) fp_x_flux."""
    # Regrid inner fp_x_flux to the same grid as outer fp_x_flux, and then patch it in where the inner domain mask is True.
    # regrid inner to EUROPE lat/lon coords
    inner_regridded = inner_fp_x_flux.interp(lat=outer_fp_x_flux.lat, lon=outer_fp_x_flux.lon, method="nearest")

    # force coordinates to exactly match outer (avoids float precision
    # mismatches that prevent xr.align / xr.where from working correctly)
    inner_regridded = inner_regridded.assign_coords(lat=outer_fp_x_flux.lat, lon=outer_fp_x_flux.lon)

    # fill NaN (points outside the inner domain extent) with 0
    inner_regridded = inner_regridded.fillna(0.0)

    # True where the inner domain contributed non-zero values at any timestep
    inner_has_coverage = (inner_regridded != 0).any("time")

    # Both arrays are now on the EUROPE grid so xr.where is safe
    return xr.where(inner_has_coverage, inner_regridded, outer_fp_x_flux)


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

    if "time" in fp_x_flux.dims:
        non_time_dims = [dim for dim in fp_x_flux.dims if dim != "time"]
        missing_times = fp_x_flux.isnull().all(dim=non_time_dims) if non_time_dims else fp_x_flux.isnull()
        missing_times = missing_times.compute()
        if bool(missing_times.any()):
            missing_index = fp_x_flux.time.where(missing_times, drop=True).to_index()
            missing_preview = ", ".join(str(t) for t in missing_index[:5])
            extra = "" if missing_index.size <= 5 else f", ... ({missing_index.size} total)"
            raise ValueError(
                "Footprint sensitivity input is entirely missing for one or more observation times: "
                f"{missing_preview}{extra}. Refusing to convert missing footprint data to zero sensitivity."
            )

    _, basis_aligned = xr.align(fp_x_flux.isel(time=0), basis_func, join="override")
    basis_mat = get_xr_dummies(basis_aligned, cat_dim="region")
    spatial_dims = ["lat", "lon"]

    sensitivity = sparse_xr_dot(basis_mat, fp_x_flux.fillna(0.0), dim=spatial_dims)

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
    def _outer_ds(entry):
        if isinstance(entry, xr.DataTree):
            if "standard" in entry.children:
                return entry["standard"].ds
            return entry.ds
        return entry

    def _with_updated_outer(entry, updated_outer_ds):
        if not isinstance(entry, xr.DataTree):
            return updated_outer_ds

        if "standard" in entry.children:
            tree_dict = {f"/{name}": child.ds for name, child in entry.children.items() if child.ds is not None}
            tree_dict["/standard"] = updated_outer_ds
            return xr.DataTree.from_dict(tree_dict)

        return xr.DataTree(dataset=updated_outer_ds, children=dict(entry.children))

    sites = [key for key in list(fp_and_data.keys()) if key[0] != "."]

    if basis_case.lower() == "nesw":
        for site in sites:
            entry = fp_and_data[site]
            outer_ds = _outer_ds(entry)
            bc_ds = outer_ds[[f"bc_{d}" for d in "nesw"]].rename({f"bc_{d}": d for d in "nesw"})
            sensitivity = bc_ds.sum(["lat", "lon", "height"]).to_dataarray(dim="bc_region")
            fp_and_data[site] = _with_updated_outer(entry, outer_ds.assign({"H_bc": sensitivity}))

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
        entry = fp_and_data[site]
        outer_ds = _outer_ds(entry)
        bc_ds = outer_ds[[f"bc_{d}" for d in "nesw"]]
        sensitivity = (bc_ds * bc_basis).sum(["lat", "lon", "height"]).to_dataarray(dim="__newdim__").sum("__newdim__")
        sensitivity = sensitivity.rename(region="bc_region")
        fp_and_data[site] = _with_updated_outer(entry, outer_ds.assign({"H_bc": sensitivity}))

    return fp_and_data
