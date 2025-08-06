"""Functions to create fit basis functiosn and apply to data."""

from openghg.util import get_species_info, synonyms
import xarray as xr
import numpy as np


from openghg_inversions import convert
from openghg_inversions.utils import combine_datasets
from openghg_inversions.array_ops import get_xr_dummies
from ._functions import basis_boundary_conditions


def fp_sensitivity(
    fp_and_data: dict, basis_func: xr.DataArray | dict[str, xr.DataArray]
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
                basis_func = xr.concat([bf.expand_dims({"source": [k]}) for k, bf in basis_func.items()], dim="source", join="outer")
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
    sensitivity = (basis_mat * fp_x_flux.fillna(0.0)).sum(["lat", "lon"]).transpose("region", "time", ...)
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

    basis_func = basis_boundary_conditions(
        domain=domain, basis_case=basis_case, bc_basis_directory=bc_basis_directory
    )
    # sort basis_func into time order
    ind = basis_func.time.argsort()
    timenew = basis_func.time[ind]
    basis_func = basis_func.reindex({"time": timenew})

    species_info = get_species_info()

    species = fp_and_data[".species"]
    species = synonyms(species, lower=False)

    for site in sites:
        # ES commented out line below as .bc not attribute.
        # Also assume openghg adds all relevant particle data to file.  TODO: what does this mean? BM, 2024
        #        if fp_and_data[site].bc.chunks is not None:
        for particles in [
            "particle_locations_n",
            "particle_locations_e",
            "particle_locations_s",
            "particle_locations_w",
        ]:
            fp_and_data[site][particles] = fp_and_data[site][particles].compute()

        # compute any chemical loss to the BCs, use lifetime or else set loss to 1 (no loss)
        if "lifetime" in species_info[species]:
            lifetime = species_info[species]["lifetime"]
            lifetime_hrs_list_or_float = convert.convert_to_hours(lifetime)

            # calculate the lifetime_hrs associated with each time point in fp_and_data
            # this is because lifetime can be a list of monthly values

            time_month = fp_and_data[site].time.dt.month
            if isinstance(lifetime_hrs_list_or_float, list):
                lifetime_hrs = [lifetime_hrs_list_or_float[item - 1] for item in time_month.values]
            else:
                lifetime_hrs = lifetime_hrs_list_or_float

            loss_n = np.exp(-1 * fp_and_data[site].mean_age_particles_n / lifetime_hrs).rename("loss_n")
            loss_e = np.exp(-1 * fp_and_data[site].mean_age_particles_e / lifetime_hrs).rename("loss_e")
            loss_s = np.exp(-1 * fp_and_data[site].mean_age_particles_s / lifetime_hrs).rename("loss_s")
            loss_w = np.exp(-1 * fp_and_data[site].mean_age_particles_w / lifetime_hrs).rename("loss_w")
        else:
            loss_n = fp_and_data[site].particle_locations_n.copy()
            loss_e = fp_and_data[site].particle_locations_e.copy()
            loss_s = fp_and_data[site].particle_locations_s.copy()
            loss_w = fp_and_data[site].particle_locations_w.copy()
            loss_n[:] = 1
            loss_e[:] = 1
            loss_s[:] = 1
            loss_w[:] = 1

        DS_particle_loc = xr.Dataset({
            "particle_locations_n": fp_and_data[site]["particle_locations_n"],
            "particle_locations_e": fp_and_data[site]["particle_locations_e"],
            "particle_locations_s": fp_and_data[site]["particle_locations_s"],
            "particle_locations_w": fp_and_data[site]["particle_locations_w"],
            "loss_n": loss_n,
            "loss_e": loss_e,
            "loss_s": loss_s,
            "loss_w": loss_w,
        })
        #                                 "bc":fp_and_data[site]["bc"]})

        DS_temp = combine_datasets(DS_particle_loc, fp_and_data[".bc"].data, method="ffill")

        DS = combine_datasets(DS_temp, basis_func, method="ffill")

        DS = DS.transpose("height", "lat", "lon", "region", "time")

        part_loc = np.hstack([
            DS.particle_locations_n,
            DS.particle_locations_e,
            DS.particle_locations_s,
            DS.particle_locations_w,
        ])

        loss = np.hstack([DS.loss_n, DS.loss_e, DS.loss_s, DS.loss_w])

        vmr_ed = np.hstack([DS.vmr_n, DS.vmr_e, DS.vmr_s, DS.vmr_w])

        bf = np.hstack([DS.bc_basis_n, DS.bc_basis_e, DS.bc_basis_s, DS.bc_basis_w])

        H_bc = np.zeros((len(DS.coords["region"]), len(DS["particle_locations_n"]["time"])))

        for i in range(len(DS.coords["region"])):
            reg = bf[:, :, i, :]
            H_bc[i, :] = np.nansum((part_loc * loss * vmr_ed * reg), axis=(0, 1))

        sensitivity = xr.Dataset(
            {"H_bc": (["region_bc", "time"], H_bc)},
            coords={"region_bc": (DS.coords["region"].values), "time": (DS.coords["time"])},
        )

        fp_and_data[site] = fp_and_data[site].merge(sensitivity)

    return fp_and_data
