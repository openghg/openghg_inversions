"""
Functions to create basis datasets from fluxes and footprints.
"""

from typing import Optional, Union

from openghg.dataobjects import FluxData
from openghg.util import get_species_info, synonyms
import xarray as xr
import dask.array as da
import numpy as np
from tqdm import tqdm

from openghg_inversions import convert
from openghg_inversions.utils import combine_datasets, read_netcdfs
from openghg_inversions.array_ops import get_xr_dummies, sparse_xr_dot
from ._functions import basis_boundary_conditions


def fp_sensitivity(
    fp_and_data: dict, basis_func: Union[xr.DataArray, dict[str, xr.DataArray]], verbose: bool = True
) -> dict:
    """
    Add a sensitivity matrix, H, to each site xr.Dataset in fp_and_data.

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

    if not isinstance(basis_func, dict):
        if len(flux_sources) == 1:
            basis_func = {flux_sources[0]: basis_func}
        else:
            basis_func = {"all": basis_func}

    if len(list(basis_func.keys())) != len(flux_sources):
        if len(list(basis_func.keys())) == 1:
            print(f"Using {basis_func[list(basis_func.keys())[0]]} as the basis case for all sources")
        else:
            print(
                "There should either only be one basis_func, or it should be a dictionary the same length\
                  as the number of sources."
            )
            return None

    for site in sites:
        site_sensitivities = []
        for source in flux_sources:
            if source in list(basis_func.keys()):
                current_basis_func = basis_func[source]
            else:
                current_basis_func = basis_func["all"]

            sensitivity, site_bf = fp_sensitivity_single_site_basis_func(
                scenario=fp_and_data[site],
                flux=fp_and_data[".flux"][source],
                source=source,
                basis_func=current_basis_func,
                verbose=verbose,
            )

            site_sensitivities.append(sensitivity)

        fp_and_data[site]["H"] = xr.concat(site_sensitivities, dim="region")
        fp_and_data[".basis"] = (
            current_basis_func.squeeze("time") if site_bf is None else site_bf.basis[:, :, 0]
        )
        # TODO: this will only contain the last value in the loop...

    return fp_and_data


def fp_sensitivity_single_site_basis_func(
    scenario: xr.Dataset,
    flux: Union[FluxData, dict[str, FluxData]],
    basis_func: xr.DataArray,
    source: str = "all",
    verbose: bool = True,
) -> tuple[xr.DataArray, Optional[xr.Dataset]]:
    """
    Computes sensitivity matrix `H` for one site. See `fp_sensitivity` for
    more info about the sensitivity matrix.

    Args:
        scenario: xr.Dataset from `ModelScenario.footprints_data_merge`, e.g. `fp_all["TAC"]`
        flux: FluxData object or dictionary of FluxData objects with flux values; a dictionary
            should only be passed for "high time resolution" or "time resolved" footprints.
        source: name of flux source; used for naming the region labels in the output coordinates.
        basis_func:

    Returns:
        sensitivity ("H") xr.DataArray and site_bf xr.Dataset containing basis functions and
        flux * footprint if "region" present in basis_func, otherwise, None
    """
    if isinstance(flux, dict):
        if "fp_HiTRes" in list(scenario.keys()):
            site_bf = xr.Dataset({"fp_HiTRes": scenario["fp_HiTRes"], "fp": scenario["fp"]})

            fp_time = (scenario.time[1] - scenario.time[0]).values.astype("timedelta64[h]").astype(int)

            # calculate the H matrix
            H_all = timeseries_HiTRes(
                fp_HiTRes_ds=site_bf,
                flux_dict=flux,
                output_TS=False,
                output_fpXflux=True,
                output_type="DataArray",
                time_resolution=f"{fp_time}H",
                verbose=verbose,
            )
        else:
            raise ValueError(
                "fp_and_data needs the variable fp_HiTRes to use the "
                "emissions dictionary with high_freq and low_freq emissions."
            )

    else:
        site_bf = combine_datasets(scenario["fp"].to_dataset(), flux.data, method="nearest")
        H_all = site_bf.fp * site_bf.flux

    H_all_v = H_all.values.reshape((len(site_bf.lat) * len(site_bf.lon), len(site_bf.time)))

    if "region" in basis_func.dims:
        if "time" in basis_func.dims:
            basis_func = basis_func.isel(time=0)

        site_bf = xr.merge([site_bf, basis_func])

        H = np.zeros((len(site_bf.region), len(site_bf.time)))

        base_v = site_bf.basis.values.reshape((len(site_bf.lat) * len(site_bf.lon), len(site_bf.region)))

        for i in range(len(site_bf.region)):
            H[i, :] = np.nansum(H_all_v * base_v[:, i, np.newaxis], axis=0)

        if source == "all":
            region_name = site_bf.region.decode("ascii")
        else:
            region_name = [source + "-" + reg.decode("ascii") for reg in site_bf.region.values]

        sensitivity = xr.DataArray(H, coords=[("region", region_name), ("time", scenario.coords["time"])])

    else:
        _, basis_aligned = xr.align(H_all.isel(time=0), basis_func, join="override")
        basis_mat = get_xr_dummies(basis_aligned.squeeze("time"), cat_dim="region")
        sensitivity = sparse_xr_dot(basis_mat, H_all.fillna(0.0)).transpose("region", "time")
        # TODO: use same region names as alternate method above
        site_bf = None

    return sensitivity, site_bf


def bc_sensitivity(
    fp_and_data: dict, domain: str, basis_case: str, bc_basis_directory: Optional[str] = None
) -> dict:
    """
    Add boundary conditions sensitivity matrix `H_bc` to each site xr.Dataframe in fp_and_data.

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
        if "lifetime" in species_info[species].keys():
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

        DS_particle_loc = xr.Dataset(
            {
                "particle_locations_n": fp_and_data[site]["particle_locations_n"],
                "particle_locations_e": fp_and_data[site]["particle_locations_e"],
                "particle_locations_s": fp_and_data[site]["particle_locations_s"],
                "particle_locations_w": fp_and_data[site]["particle_locations_w"],
                "loss_n": loss_n,
                "loss_e": loss_e,
                "loss_s": loss_s,
                "loss_w": loss_w,
            }
        )
        #                                 "bc":fp_and_data[site]["bc"]})

        DS_temp = combine_datasets(DS_particle_loc, fp_and_data[".bc"].data, method="ffill")

        DS = combine_datasets(DS_temp, basis_func, method="ffill")

        DS = DS.transpose("height", "lat", "lon", "region", "time")

        part_loc = np.hstack(
            [
                DS.particle_locations_n,
                DS.particle_locations_e,
                DS.particle_locations_s,
                DS.particle_locations_w,
            ]
        )

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


def timeseries_HiTRes(
    flux_dict,
    fp_HiTRes_ds=None,
    fp_file=None,
    output_TS=True,
    output_fpXflux=True,
    output_type="Dataset",
    output_file=None,
    verbose=False,
    chunks=None,
    time_resolution="1H",
):
    """
    The timeseries_HiTRes function computes flux * HiTRes footprints.

    HiTRes footprints record the footprint at each 2 hour period back
    in time for the first 24 hours. Need a high time resolution flux
    to multiply the first 24 hours back of footprints. Need a residual
    flux to multiply the residual integrated footprint for the remainder
    of the 30 day period.
    -----------------------------------
    Args:
      fp_HiTRes_ds (xarray.Dataset)
        Dataset of high time resolution footprints. HiTRes footprints
        record the footprint at each timestep back in time for a given
        amount of time (e.g. hourly time steps back in time for the first
        24 hours).
      domain (str)
        Domain name. The footprint files should be sub-categorised by the domain.
      flux_dict (dict)
        This should be a dictionary of the form output in the format
        flux_dict: {'high_freq': flux_dataset, 'low_freq': flux_dataset}.
        This is because this function needs two time resolutions of fluxes as
        explained in the header.

        If there are multiple sectors, the format should be:
        flux_dict: {sector1 : {'high_freq' : flux_dataset, 'low_freq'  : flux_dataset},
                    sector2 : {'high_freq' : flux_dataset, 'low_freq'  : flux_dataset}}
      output_TS (bool)
        Output the timeseries. Default is True.
      output_fpXflux (bool)
        Output the sensitivity map. Default is True.
      verbose (bool)
        Show progress bar throughout loop
      chunks (dict)
       Size of chunks for each dimension
       e.g. {'lat': 50, 'lon': 50}
       opens dataset with dask, such that it is opened 'lazily'
       and all of the data is not loaded into memory
       defaults to None - dataset is opened with out dask

    Returns:
      xarray.Dataset or dict
        Same format as flux_dict['high_freq']:
        If flux_dict['high_freq'] is an xarray.Dataset then an xarray.Dataset is returned
        If flux_dict['high_freq'] is a dict of xarray.Datasets then a dict of xarray.Datasets
        is returned (an xarray.Dataset for each sector)

        If output_TS is True:
          Outputs the timeseries
        If output_fpXflux is True:
          Outputs the sensitivity map
    -----------------------------------
    """
    if verbose:
        print(f"\nCalculating timeseries with {time_resolution} resolution, this might take a few minutes")
    # get the high time res footprint
    if fp_HiTRes_ds is None and fp_file is None:
        print("Must provide either a footprint Dataset or footprint filename")
        return None
    elif fp_HiTRes_ds is None:
        fp_HiTRes_ds = read_netcdfs(fp_file, chunks=chunks)
        fp_HiTRes = fp_HiTRes_ds.fp_HiTRes
    else:
        if isinstance(fp_HiTRes_ds, xr.DataArray):
            fp_HiTRes = fp_HiTRes_ds
        elif fp_HiTRes_ds.chunks is None and chunks is not None:
            fp_HiTRes = fp_HiTRes_ds.fp_HiTRes.chunk(chunks)
        else:
            fp_HiTRes = fp_HiTRes_ds.fp_HiTRes

    # resample fp to match the required time resolution
    fp_HiTRes = fp_HiTRes.resample(time=time_resolution).ffill()

    # get the H_back timestep and max number of hours back
    H_back_hour_diff = int(fp_HiTRes["H_back"].diff(dim="H_back").values.mean())
    max_H_back = int(fp_HiTRes["H_back"].values[-2])

    # create time array to loop through, with the required resolution
    time_array = fp_HiTRes.time.values
    # extract as a dask array to make the loop quicker
    fp_HiTRes = da.array(fp_HiTRes)

    # this is the number of hours over which the H_back will be resampled to match time_resolution
    H_resample = (
        int(time_resolution[0])
        if H_back_hour_diff == 1
        else 1 if H_back_hour_diff == int(time_resolution[0]) else None
    )
    if H_resample is None:
        print("Cannot resample H_back")
        return None
    # reverse the H_back coordinate to be chronological, and resample to match time_resolution
    fp_HiTRes = fp_HiTRes[:, :, :, ::-H_resample]

    # convert fluxes to dictionaries with sectors as keys
    flux = (
        {"total": flux_dict}
        if any([ff in list(flux_dict.keys()) for ff in ["high_freq", "low_freq"]])
        else flux_dict
    )
    flux = {
        sector: {freq: None if flux_freq is None else flux_freq for freq, flux_freq in flux_sector.items()}
        for sector, flux_sector in flux.items()
    }

    # extract the required time data
    flux = {
        sector: {
            freq: (
                flux_freq.sel(
                    time=slice(fp_HiTRes_ds.time[0] - np.timedelta64(max_H_back, "h"), fp_HiTRes_ds.time[-1])
                ).flux
                if flux_freq is not None
                else None
            )
            for freq, flux_freq in flux_sector.items()
        }
        for sector, flux_sector in flux.items()
    }

    for sector, flux_sector in flux.items():
        if "high_freq" in flux_sector.keys() and flux_sector["high_freq"] is not None:
            # reindex the high frequency data to match the fp
            time_flux = np.arange(
                fp_HiTRes_ds.time[0].values - np.timedelta64(max_H_back, "h"),
                fp_HiTRes_ds.time[-1].values + np.timedelta64(time_resolution[0], time_resolution[1].lower()),
                time_resolution[0],
                dtype=f"datetime64[{time_resolution[1].lower()}]",
            )
            flux_sector["high_freq"] = flux_sector["high_freq"].reindex(time=time_flux, method="ffill")
        else:
            print(
                f"\nWarning: no high frequency flux data for {sector}, estimating a timeseries using the low frequency data"
            )
            flux_sector["high_freq"] = None

        if "low_freq" not in flux_sector.keys() or flux_sector["low_freq"] is None:
            print(f"\nWarning: no low frequency flux data for {sector}, resampling from high frequency data")
            flux_sector["low_freq"] = flux_sector["high_freq"].resample(time="MS").mean()

    # convert to array to use in numba loop
    flux = {
        sector: {
            freq: (
                None
                if flux_freq is None
                else flux_freq.values if flux_freq.chunks is None else da.array(flux_freq)
            )
            for freq, flux_freq in flux_sector.items()
        }
        for sector, flux_sector in flux.items()
    }

    # Set up a numpy array to calculate the product of the footprint (H matrix) with the fluxes
    if output_fpXflux:
        fpXflux = {
            sector: da.zeros((len(fp_HiTRes_ds.lat), len(fp_HiTRes_ds.lon), len(time_array)))
            for sector in flux.keys()
        }

    elif output_TS:
        timeseries = {sector: da.zeros(len(time_array)) for sector in flux.keys()}

    # month and year of the start of the data - used to index the low res data
    start = {dd: getattr(np.datetime64(time_array[0], "h").astype(object), dd) for dd in ["month", "year"]}

    # put the time array into tqdm if we want a progress bar to show throughout the loop
    iters = tqdm(time_array) if verbose else time_array
    # iterate through the time coord to get the total mf at each time step using the H back coord
    # at each release time we disaggregate the particles backwards over the previous 24hrs
    for tt, time in enumerate(iters):
        # get 4 dimensional chunk of high time res footprint for this timestep
        # units : mol/mol/mol/m2/s
        fp_time = fp_HiTRes[:, :, tt, :]

        # get the correct index for the low res data
        # estimated using the difference between the current and start month and year
        current = {dd: getattr(np.datetime64(time, "h").astype(object), dd) for dd in ["month", "year"]}
        tt_low = current["month"] - start["month"] + 12 * (current["year"] - start["year"])

        # select the high res emissions for the corresponding 24 hours
        # if there aren't any high frequency data it will select from the low frequency data
        # this is so that we can compare emissions data with different resolutions e.g. ocean species
        emissions = {
            sector: (
                flux_sector["high_freq"][:, :, tt : tt + fp_time.shape[2] - 1]
                if flux_sector["high_freq"] is not None
                else flux_sector["low_freq"][:, :, tt_low]
            )
            for sector, flux_sector in flux.items()
        }
        # add an axis if the emissions is array is 2D so that it can be multiplied by the fp
        emissions = {
            sector: em_sec[:, :, np.newaxis] if len(em_sec.shape) == 2 else em_sec
            for sector, em_sec in emissions.items()
        }
        # select average monthly emissions for the start of the month
        emissions_end = {
            sector: flux_sector["low_freq"][:, :, tt_low] for sector, flux_sector in flux.items()
        }

        # Multiply the HiTRes footprint with the HiTRes emissions to give mf
        # we take all but the slice for H_back==24 as these are the hourly disaggregated fps
        # flux units : mol/m2/s;       fp units : mol/mol/mol/m2/s
        # --> mol/mol/mol/m2/s * mol/m2/s === mol / mol
        fpXflux_time = {sector: em_sec * fp_time[:, :, 1:] for sector, em_sec in emissions.items()}
        # multiply the monthly flux by the residual fp, at H_back==24
        fpXflux_end = {sector: em_end * fp_time[:, :, 0] for sector, em_end in emissions_end.items()}
        # append the residual emissions
        fpXflux_time = {
            sector: np.dstack((fp_fl, fpXflux_end[sector])) for sector, fp_fl in fpXflux_time.items()
        }

        for sector, fp_fl in fpXflux_time.items():
            if output_fpXflux:
                # Sum over time (H back) to give the total mf at this timestep
                fpXflux[sector][:, :, tt] = np.nansum(fp_fl, axis=2)

            elif output_TS:
                # work out timeseries by summing over lat, lon, & time (24 hrs)
                timeseries[sector][tt] = np.nansum(fp_fl)

    if output_fpXflux and output_TS:
        # if not already done then calculate the timeseries
        timeseries = {sector: fp_fl.sum(axis=(0, 1)) for sector, fp_fl in fpXflux.items()}

    if output_fpXflux:
        fpXflux = (
            {sec: (["lat", "lon", "time"], ff_sec) for sec, ff_sec in fpXflux.items()}
            if output_type.lower() == "dataset"
            else fpXflux
        )

        fpXflux = (
            xr.Dataset(
                fpXflux,
                coords={"lat": fp_HiTRes_ds.lat.values, "lon": fp_HiTRes_ds.lon.values, "time": time_array},
            )
            if output_type.lower() == "dataset"
            else (
                {
                    sector: xr.DataArray(
                        data=fpXflux_sector,
                        dims=["lat", "lon", "time"],
                        coords={
                            "lat": fp_HiTRes_ds.lat.values,
                            "lon": fp_HiTRes_ds.lon.values,
                            "time": time_array,
                        },
                    )
                    for sector, fpXflux_sector in fpXflux.items()
                }
                if output_type.lower() == "dataarray"
                else fpXflux
            )
        )

        if output_type.lower() == "dataset":
            fpXflux = fpXflux if fpXflux.chunks is None else fpXflux.compute()
        else:
            fpXflux = {sec: ff if ff.chunks is None else ff.compute() for sec, ff in fpXflux.items()}

        if output_type.lower() == "dataarray" and list(flux.keys()) == ["total"]:
            fpXflux = fpXflux["total"]

    if output_fpXflux and not output_TS:
        return fpXflux

    else:
        # for each sector create a tuple of ['time'] (coord name) and the timeseries
        # if the output required is a dataset
        timeseries = (
            {sec: (["time"], ts_sec) for sec, ts_sec in timeseries.items()}
            if output_type.lower() == "dataset"
            else timeseries
        )
        timeseries = (
            xr.Dataset(timeseries, coords={"time": time_array})
            if output_type.lower() == "dataset"
            else (
                {
                    sector: xr.DataArray(data=ts_sector, dims=["time"], coords={"time": time_array})
                    for sector, ts_sector in timeseries.items()
                }
                if output_type.lower() == "dataarray"
                else timeseries
            )
        )

        if output_type.lower() == "dataset":
            timeseries = timeseries if timeseries.chunks is None else timeseries.compute()
        else:
            timeseries = {sec: tt if tt.chunks is None else tt.compute() for sec, tt in timeseries.items()}

        if output_type.lower() == "dataarray" and list(flux.keys()) == ["total"]:
            timeseries = timeseries["total"]

        if output_file is not None and output_type.lower() == "dataset":
            print(f"Saving to {output_file}")
            timeseries.to_netcdf(output_file)
        elif output_file is not None:
            print("output type must be dataset to save to file")

        if output_fpXflux:
            return timeseries, fpXflux
        elif output_TS:
            return timeseries
