# *****************************************************************************
# get_data.py
# Author: Atmospheric Chemistry Research Group, University of Bristol
"""Functions for retrieving observations and datasets for creating forward
simulations.

Current data processing options include:
- "data_processing_surface_notracer": Surface based measurements, without tracers

Future data processing options will include:
- "data_processing_surface_tracer": Surface based measurements, with tracers

This module also includes functions for saving and loading "merged data" created
by the data processing functions.
"""

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, cast, Literal
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from openghg.analyse import ModelScenario
from openghg.dataobjects import ObsData, BoundaryConditionsData, FluxData, FootprintData
from openghg.retrieve import get_bc, get_flux, get_footprint, get_obs_surface, search_footprints
from openghg.types import SearchError
from openghg.util import timestamp_now


logger = logging.getLogger(__name__)


def add_obs_error(sites: list[str], fp_all: dict, add_averaging_error: bool = True) -> None:
    """Create `mf_error` variable that contains either `mf_repeatablility`, `mf_variability`
    or the square root of the sum of the squares of both, if `add_averaging_error` is True.

    This function modifies `fp_all` in place, adding `mf_error` and making sure that both
    `mf_repeatability` and `mf_variability` are present.

    Note: if `averaging_period` is specified in `data_processing_surface_notracer`, then OpenGHG
    will add an `mf_variability` variable with the standard deviation of the obs over the specified
    period. If `mf_variability` is already present (for instance, for Picarro data), then the existing
    variable is over-written. If the `averaging_period` matches the frequency of the data, this will
    make `mf_variability` zero (since the stdev of one value is 0).

    Args:
        sites: list of site names to process
        fp_all: dictionary of `ModelScenario` objects, keyed by site names
        add_averaging_error: if True, combine repeatability and variability to make `mf_error`
            variable. Otherwise, `mf_error` will equal `mf_repeatability` if it is present, otherwise
            it will equal `mf_variability`.

    Returns:
        None, modifies `fp_all` in place.
    """
    # TODO: do we want to fill missing values in repeatability or variability?
    for site in sites:
        ds = fp_all[site]

        variability_missing = False
        if "mf_variability" not in ds:
            ds["mf_variability"] = xr.zeros_like(ds.mf)
            variability_missing = True

        if "mf_repeatability" not in ds:
            if variability_missing:
                raise ValueError(f"Obs data for site {site} is missing both repeatability and variability.")

            ds["mf_repeatability"] = xr.zeros_like(ds.mf_variability)
            ds["mf_error"] = ds["mf_variability"]

            if add_averaging_error:
                logger.info(
                    "`mf_repeatability` not present; using `mf_variability` for `mf_error` at site %s", site
                )

        elif add_averaging_error:
            ds["mf_error"] = np.sqrt(ds["mf_repeatability"] ** 2 + ds["mf_variability"] ** 2)
        else:
            ds["mf_error"] = ds["mf_repeatability"]

        # warnings/info for debugging
        err0 = ds["mf_error"] == 0

        if err0.any():
            percent0 = 100 * err0.mean()
            logger.warning("`mf_error` is zero for %.0f percent of times at site %s.", percent0, site)
            info_msg = (
                "If `averaging_period` matches the frequency of the obs data, then `mf_variability` "
                "will be zero. Try setting `averaging_period = None`."
            )
            logger.info(info_msg)


def _convert_inlets_to_float(inlets):
    return np.array(list(map(lambda x: float(x[:-1]), inlets)))


def get_fp_indexer(obs: xr.Dataset, fp: xr.Dataset, averaging_period: str):
    obs_idx = obs.indexes["time"]
    fp_idx = fp.indexes["time"]

    period = pd.Timedelta(averaging_period)
    tol = period / 2  # type: ignore

    fp_reidx = fp_idx.get_indexer(obs_idx, method="nearest", tolerance=tol)
    int_idx = pd.IntervalIndex.from_arrays(fp_idx[fp_reidx], fp_idx[fp_reidx] + period, closed="left")
    return pd.DatetimeIndex([t for t in fp_idx if int_idx.contains(t).any()])


def get_footprint_to_match(
    obs: ObsData,
    domain: str,
    model: str | None = None,
    met_model: str = "not_set",
    fp_species: str | None = None,
    store: str | None = None,
    averaging_period: str | None = None,
    tolerance: float = 10.0,
) -> FootprintData:
    site = obs.metadata["site"]
    species = fp_species or obs.metadata.get("species", "inert")
    if store is None:
        store = Path(obs.metadata.get("object_store", "")).name or "user"
    start_date = obs._start_date
    end_date = obs._end_date

    # get available footprint heights
    fp_kwargs = {
        "site": site,
        "species": species,
        "domain": domain,
        "model": model,
        "met_model": met_model,
        "store": store,
        "start_date": start_date,
        "end_date": end_date,
    }
    results = search_footprints(**fp_kwargs)
    fp_heights_strs = list(results.results.inlet.unique())
    fp_heights = _convert_inlets_to_float(fp_heights_strs)

    # special case: only a single inlet height
    if "inlet" not in obs.data.data_vars:
        inlet = float(obs.metadata["inlet"][:-1])

        if np.min(np.abs(fp_heights - inlet)) > tolerance:
            raise SearchError("No footprints found with inlet heights matching given obs.")

        fp_height_idx = np.argmin(np.abs(fp_heights - inlet))
        fp_height = fp_heights_strs[fp_height_idx]
        return get_footprint(**fp_kwargs, inlet=fp_height)

    # get inlet values
    inlets = obs.data.inlet.values

    # match inlets to fp heights
    distances = np.abs(inlets.reshape((-1, 1)) - fp_heights.reshape((1, -1)))
    inlets_to_heights = np.argmin(distances, axis=1)

    # check tolerance
    inlet_tolerance_passed = np.min(distances, axis=1) <= tolerance
    if (s := np.sum(inlet_tolerance_passed)) > 0:
        warnings.warn(f"{s} times where obs. inlet height was not within {tolerance}m of a footprint height.")
        inlets_to_heights = inlets_to_heights[inlet_tolerance_passed]

    # footprint heights to load
    matched_fp_heights = [fp_heights_strs[i] for i in np.unique(inlets_to_heights)]
    footprints = []

    for fp_height in matched_fp_heights:
        fp_data = get_footprint(**fp_kwargs, inlet=fp_height)
        footprints.append(fp_data)

    if not footprints:
        raise SearchError("No footprints found with inlet heights matching given obs.")

    # select footprints to match inlets
    # note: we need to take into account the difference between the obs frequency
    # and the footprint frequency
    try:
        averaging_period = averaging_period or obs.metadata["averaged_period_str"]
    except KeyError:
        if "sampling_period" in obs.metadata and "sampling_period_unit" in obs.metadata:
            averaging_period = f"{obs.metadata['sampling_period']}{obs.metadata['sampling_period_unit']}"
        else:
            raise ValueError("`averaging_period` could not be inferred from ObsData; please provide a value.")

    # select times from footprints to match with obs
    for i, fp in zip(np.unique(inlets_to_heights), footprints):
        i_idx = np.where(inlets_to_heights == i)[0]
        fp_idx = get_fp_indexer(obs.data.isel(time=i_idx), fp.data, averaging_period=averaging_period)
        fp.data = fp.data.sel(time=fp_idx)

    # make FootprintData to return
    metadata = footprints[0].metadata

    if len(footprints) > 1:
        metadata["inlet"] = "varies"
        metadata["height"] = "varies"

    data = xr.concat([fp.data for fp in footprints], dim="time").sortby("time")

    return FootprintData(data=data, metadata=metadata)


def data_processing_surface_notracer(
    species: str,
    sites: list | str,
    domain: str,
    averaging_period: list[str | None] | str | None,
    start_date: str,
    end_date: str,
    obs_data_level: list[str | None] | str | None = None,
    inlet: list[str | None] | str | None = None,
    instrument: list[str | None] | str | None = None,
    calibration_scale: str | None = None,
    met_model: list[str | None] | str | None = None,
    fp_model: str | None = None,
    fp_height: list[str | None | Literal["auto"]] | Literal["auto"] | str | None = None,
    fp_species: str | None = None,
    emissions_name: list | None = None,
    use_bc: bool = True,
    bc_input: str | None = None,
    bc_store: str | None = None,
    obs_store: str | list[str] | None = None,
    footprint_store: str | list[str] | None = None,
    emissions_store: str | None = None,
    averagingerror: bool = True,
    save_merged_data: bool = False,
    merged_data_name: str | None = None,
    merged_data_dir: str | None = None,
    output_name: str | None = None,
) -> tuple[dict, list, list, list, list, list]:
    """Retrieve and prepare fixed-surface datasets from specified OpenGHG object stores.

    Use for forward simulations and model-data comparisons that do not
    use tracers.

    Args:
        species:
            Atmospheric trace gas species of interest
            e.g. "co2"
        sites:
            List of strings containing measurement
            station/site abbreviations
            e.g. ["MHD", "TAC"]
        domain:
            Model domain region of interest
            e.g. "EUROPE"
        averaging_period:
            List of averaging periods to apply to
            mole fraction data. NB. len(averaging_period)==len(sites)
            e.g. ["1H", "1H"]
        start_date:
            Date from which to gather data
            e.g. "2020-01-01"
        end_date:
            Date until which to gather data
            e.g. "2020-02-01"
        obs_data_level:
            ICOS observations data level. For non-ICOS sites
            use "None"
        inlet:
            Specific inlet height for the site observations
            (length must match number of sites)
        instrument:
            Specific instrument for the site
            (length must match number of sites)
        calibration_scale:
            Convert measurements to defined calibration scale
        met_model:
            Meteorological model used in the LPDM. List must be same length as number of sites.
        fp_model:
            LPDM used for generating footprints.
        fp_height:
            Inlet height used in footprints for corresponding sites.
        fp_species:
            Species name associated with footprints in the object store
        emissions_name:
            List of keywords args associated with emissions files
            in the object store.
        use_bc:
            Option to include boundary conditions in model
        bc_input:
            Variable for calling BC data from 'bc_store' - equivalent of
            'emissions_name' for fluxes
        bc_store:
            Name of object store to retrieve boundary conditions data from
        obs_store:
            Name of object store to retrieve observations data from
        footprint_store:
            Name of object store to retrieve footprints data from
        emissions_store:
            Name of object store to retrieve emissions data from
        averagingerror:
            Adds the variability in the averaging period to the measurement
            error if set to True.
        save_merged_data:
            Save forward simulations data and observations
        merged_data_name:
            Filename for saved forward simulations data and observations
        merged_data_dir:
            Directory path for for saved forward simulations data and observations
        output_name:
            Optional name used to create merged data name.

    Returns:
        fp_all:
            dictionnary containing flux data (key ".flux"), bc data (key ".bc"),
            and observations data (site short name as key)
        sites:
            Updated list of sites. All put in upper case and if data was not extracted
            correctly for any sites, drop these from the rest of the inversion.
        inlet:
            List of inlet height for the updated list of sites
        fp_height:
            List of footprint height for the updated list of sites
        instrument:
            List of instrument for the updated list of sites
        averaging_period:
            List of averaging_period for the updated list of sites

    """
    sites = [site.upper() for site in sites]

    # Convert 'None' args to list
    nsites = len(sites)
    if inlet is None or isinstance(inlet, str):
        inlet = [inlet] * nsites
    if instrument is None or isinstance(instrument, str):
        instrument = [instrument] * nsites
    if fp_height is None or isinstance(fp_height, str):
        fp_height = [fp_height] * nsites
    if obs_data_level is None or isinstance(obs_data_level, str):
        obs_data_level = [obs_data_level] * nsites
    if met_model is None or isinstance(met_model, str):
        met_model = [met_model] * nsites
    if averaging_period is None or isinstance(averaging_period, str):
        averaging_period = [averaging_period] * nsites

    fp_all = {}
    fp_all[".species"] = species.upper()

    # Get flux data and add to dict.
    flux_dict = {}
    for source in emissions_name:
        logging.Logger.disabled = True  # suppress confusing OpenGHG warnings
        try:
            get_flux_data = get_flux(
                species=species,
                domain=domain,
                source=source,
                start_date=start_date,
                end_date=end_date,
                store=emissions_store,
            )

            # fix to prevent empty time coordinate:
            if len(get_flux_data.data.time) == 0:
                raise SearchError
        except SearchError:
            print(f"No flux data found between {start_date} and {end_date}.")
            print(f"Searching for flux data from before {start_date}.")

            # re-try without start date
            try:
                get_flux_data = get_flux(
                    species=species,
                    domain=domain,
                    source=source,
                    start_date=None,
                    end_date=end_date,
                    store=emissions_store,
                )
            except SearchError as e:
                raise SearchError(f"No flux data found before {start_date}") from e
            else:
                get_flux_data.data = get_flux_data.data.isel(time=[-1])
                print(f"Using flux data from {str(get_flux_data.data.time.values[0]).split(':')[0]}.")

        logging.Logger.disabled = False  # resume confusing OpenGHG warnings

        flux_dict[source] = get_flux_data
    fp_all[".flux"] = flux_dict

    footprint_dict = {}
    scales = {}
    check_scales = set()
    site_indices_to_keep = []

    if not isinstance(obs_store, list):
        obs_store = [obs_store]

    if not isinstance(footprint_store, list):
        footprint_store = [footprint_store]

    for i, site in enumerate(sites):
        # Get observations data
        obs_found = False
        for store in obs_store:
            try:
                site_data = get_obs_surface(
                    site=site,
                    species=species.lower(),
                    inlet=inlet[i],
                    start_date=start_date,
                    end_date=end_date,
                    icos_data_level=obs_data_level[i],  # NB. Variable name may be later updated in OpenGHG
                    average=averaging_period[i],
                    instrument=instrument[i],
                    calibration_scale=calibration_scale,
                    store=store,
                )
            except SearchError:
                print(
                    f"\nNo obs data found for {site} with inlet {inlet[i]} and instrument {instrument[i]} in store {store}."
                )
                continue  # skip this site
            except AttributeError:
                print(f"\nNo data found for {site} between {start_date} and {end_date} in store {store}.")
                continue  # skip this site
            else:
                if site_data is None:
                    print(f"\nNo data found for {site} between {start_date} and {end_date} in store {store}.")
                    continue  # skip this site
                else:
                    obs_found = True
                    break
        if not obs_found:
            print("No obs. found, continuing model run without {site}.\n")
            continue
        else:
            unit = float(site_data[site].mf.units)

        # Get footprints data
        footprint_found = False
        for store in footprint_store:
            try:
                if fp_height[i] == "auto":
                    get_fps = get_footprint_to_match(
                        site_data,
                        domain=domain,
                        model=fp_model,
                        met_model=met_model[i],
                        store=store,
                        fp_species=fp_species,
                        averaging_period=averaging_period,
                    )
                else:
                    get_fps = get_footprint(
                        site=site,
                        height=fp_height[i],
                        domain=domain,
                        model=fp_model,
                        met_model=met_model[i],
                        start_date=start_date,
                        end_date=end_date,
                        store=store,
                        species=fp_species,
                    )
                if get_fps.data.time.size == 0:
                    raise SearchError
            except SearchError:
                continue  # try next store
            else:
                footprint_dict[site] = get_fps
                footprint_found = True
                break  # stop checking stores

        if not footprint_found:
            print(
                f"\nNo footprint data found for {site} with inlet/height {fp_height[i]}, model {fp_model}, and domain {domain}.",
                f"Check these values.\nContinuing model run without {site}.\n",
            )
            continue  # skip this site
        else:
            footprint_dict[site] = get_fps

        try:
            if use_bc is True:
                # Get boundary conditions data
                get_bc_data = get_bc(
                    species=species,
                    domain=domain,
                    bc_input=bc_input,
                    start_date=start_date,
                    end_date=end_date,
                    store=bc_store,
                )

                # Divide by trace gas species units
                # See if R+G can include this 'behind the scenes'
                get_bc_data.data.vmr_n.values /= unit
                get_bc_data.data.vmr_e.values /= unit
                get_bc_data.data.vmr_s.values /= unit
                get_bc_data.data.vmr_w.values /= unit
                my_bc = BoundaryConditionsData(
                    data=get_bc_data.data.transpose("height", "lat", "lon", "time"),
                    metadata=get_bc_data.metadata,
                )
                fp_all[".bc"] = my_bc

            else:
                my_bc = None

            # Create ModelScenario object for all emissions_sectors
            # and combine into one object
            model_scenario = ModelScenario(
                site=site,
                species=species,
                inlet=inlet[i],
                start_date=start_date,
                end_date=end_date,
                obs=site_data,
                footprint=footprint_dict[site],
                flux=flux_dict,
                bc=my_bc,
            )

            if len(emissions_name) == 1:
                scenario_combined = model_scenario.footprints_data_merge()
                if use_bc is True:
                    scenario_combined.bc_mod.values *= unit

            elif len(emissions_name) > 1:
                # Create model scenario object for each flux sector
                model_scenario_dict = {}

                for source in emissions_name:
                    scenario_sector = model_scenario.footprints_data_merge(sources=source, recalculate=True)

                    if species.lower() == "co2":
                        model_scenario_dict["mf_mod_high_res_" + source] = scenario_sector["mf_mod_high_res"]
                    else:
                        model_scenario_dict["mf_mod_" + source] = scenario_sector["mf_mod"]

                scenario_combined = model_scenario.footprints_data_merge(recalculate=True)

                for k, v in model_scenario_dict.items():
                    scenario_combined[k] = v
                    if use_bc is True:
                        scenario_combined.bc_mod.values *= unit

            fp_all[site] = scenario_combined

            scales[site] = scenario_combined.scale
            check_scales.add(scenario_combined.scale)

            site_indices_to_keep.append(i)

        except SearchError:
            print(
                f"\nError in reading in BC or flux data for {site}.\nContinuing model run without {site}.\n"
            )

    if len(site_indices_to_keep) == 0:
        raise SearchError("No site data found. Exiting process.")

    # If data was not extracted correctly for any sites, drop these from the rest of the inversion
    if len(site_indices_to_keep) < len(sites):
        sites = [sites[s] for s in site_indices_to_keep]
        inlet = [inlet[s] for s in site_indices_to_keep]
        fp_height = [fp_height[s] for s in site_indices_to_keep]
        instrument = [instrument[s] for s in site_indices_to_keep]
        averaging_period = [averaging_period[s] for s in site_indices_to_keep]

    # check for consistency of calibration scales
    if len(check_scales) > 1:
        msg = f"Not all sites using the same calibration scale: {len(check_scales)} scales found."
        logger.warning(msg)

    fp_all[".scales"] = scales

    fp_all[".units"] = float(scenario_combined.mf.units)

    # create `mf_error`
    add_obs_error(sites, fp_all, add_averaging_error=averagingerror)

    if save_merged_data:
        if merged_data_dir is None:
            print("`merged_data_dir` not specified; could not save merged data")
        else:
            _save_merged_data(
                fp_all,
                merged_data_dir,
                merged_data_name=merged_data_name,
                species=species,
                start_date=start_date,
                output_name=output_name,
            )
            print(f"\nfp_all saved in {merged_data_dir}\n")

    return fp_all, sites, inlet, fp_height, instrument, averaging_period


def _make_merged_data_name(species: str, start_date: str, output_name: str) -> str:
    return f"{species}_{start_date}_{output_name}_merged-data"


def _save_merged_data(
    fp_all: dict,
    merged_data_dir: str | Path,
    species: str | None = None,
    start_date: str | None = None,
    output_name: str | None = None,
    merged_data_name: str | None = None,
    output_format: Literal["pickle", "netcdf", "zarr", "zarr.zip"] = "zarr.zip",
) -> None:
    """Save `fp_all` dictionary to `merged_data_dir`.

    The name of the pickle file can be specified using `merged_data_name`, or
    a standard name will be created given `species`, `start_date`, and `output_name`.

    If `merged_data_name` is not given, then `species`, `start_date`, and `output_name` must be provided.

    The default output format is a zarr store. If zarr is not installed, then netCDF is used.
    Alternatively, "pickle" can be specified.

    Args:
        fp_all: dictionary of merged data to save
        merged_data_dir: path to directory where merged data will be saved
        species: species of inversion
        start_date: start date of inversion period
        output_name: output name parameter used for inversion run
        merged_data_name: name to use for saved data.
        output_format: format to save merged data to (default: "zarr").

    Returns:
        None
    """
    if merged_data_name is None:
        if any(arg is None for arg in [species, start_date, output_name]):
            raise ValueError(
                "If `merged_date_name` isn't given, then "
                "`species`, `start_date`, and `output_name` must be provided."
            )
        merged_data_name = _make_merged_data_name(species, start_date, output_name)  # type: ignore

    if isinstance(merged_data_dir, str):
        merged_data_dir = Path(merged_data_dir)

    # write to specified output
    if output_format == "pickle":
        with open(merged_data_dir / (merged_data_name + ".pickle"), "wb") as f:
            pickle.dump(fp_all, f)
    elif output_format in {"netcdf", "zarr", "zarr.zip"}:
        ds = make_combined_scenario(fp_all)

        if "zarr" in output_format:
            try:
                import zarr
            except ModuleNotFoundError:
                # zarr not found
                ds.to_netcdf(merged_data_dir / (merged_data_name + ".nc"))
            else:
                if output_format == "zarr":
                    ds.to_zarr(merged_data_dir / (merged_data_name + ".zarr"), mode="w")
                else:
                    with zarr.ZipStore(merged_data_dir / (merged_data_name + ".zarr.zip"), mode="w") as store:
                        ds.to_zarr(store, mode="w")
        else:
            ds.to_netcdf(merged_data_dir / (merged_data_name + ".nc"))
    else:
        raise ValueError(
            f"Output format should be 'pickle', 'netcdf', 'zarr', or 'zarr.zip'. Given '{output_format}'."
        )


def load_merged_data(
    merged_data_dir: str | Path,
    species: str | None = None,
    start_date: str | None = None,
    output_name: str | None = None,
    merged_data_name: str | None = None,
    output_format: Literal["pickle", "netcdf", "zarr", "zarr.zip"] | None = None,
) -> dict:
    """Load `fp_all` dictionary from a file in `merged_data_dir`.

    The name of the pickle file can be specified using `merged_data_name`, or
    a standard name will be created given `species`, `start_date`, and `output_name`.

    If `merged_data_name` is not given, then `species`, `start_date`, and `output_name` must be provided.

    This function tries to automatically find a compatible format of merged data, if a format is not specified.
    First, it checks for data in "zarr" format, then in netCDF, and finally in pickle.

    Args:
        merged_data_dir: path to directory where merged data will be saved
        species: species of inversion
        start_date: start date of inversion period
        output_name: output name parameter used for inversion run
        merged_data_name: name to use for saved data.
        output_format: format of data to load (if not specified, this will be inferred).

    Returns:
        `fp_all` dictionary
    """
    if isinstance(merged_data_dir, str):
        merged_data_dir = Path(merged_data_dir)

    if merged_data_name is not None:
        err_msg = (
            f"No merged data with file name {merged_data_name} in merged data directory {merged_data_dir}"
        )
    elif any(arg is None for arg in [species, start_date, output_name]):
        raise ValueError(
            "If `merged_date_name` isn't given, then "
            "`species`, `start_date`, and `output_name` must be provided."
        )
    else:
        merged_data_name = _make_merged_data_name(species, start_date, output_name)  # type: ignore
        err_msg = (
            f"No merged data for species {species}, start date {start_date}, and "
            f"output name {output_name} found in merged data directory {merged_data_dir}"
        )

    if output_format is not None:
        ext = output_format
        merged_data_file = merged_data_dir / (merged_data_name + "." + ext)
        if not merged_data_file.exists():
            raise ValueError(f"No merged data found at {merged_data_file}.")
    else:
        for ext in ["zarr.zip", "zarr", "nc", "pickle"]:
            # skip "zarr" if zarr not installed...
            if "zarr" in ext:
                try:
                    import zarr
                except ModuleNotFoundError:
                    continue

            merged_data_file = merged_data_dir / (merged_data_name + "." + ext)
            if merged_data_file.exists():
                break
        else:
            # no `break` occurred, so no file found
            raise ValueError(err_msg)

    # load merged data
    if merged_data_file.suffix == "pickle":
        with open(merged_data_file, "rb") as f:
            fp_all = pickle.load(f)
    else:
        if merged_data_file.suffixes == [".zarr", ".zip"]:
            import zarr

            with zarr.ZipStore(merged_data_file, mode="r") as store:
                ds = xr.open_zarr(store).load()
        elif merged_data_file.suffix == ".zarr":
            ds = xr.open_zarr(merged_data_file)
        else:
            # suffix is probably ".nc", but could be something else if name passed directly
            # try `open_dataset`
            ds = xr.open_dataset(merged_data_file)

        fp_all = fp_all_from_dataset(ds)

    return fp_all


list_keys = [
    "site",
    "inlet",
    "instrument",
    "sampling_period",
    "sampling_period_unit",
    "averaged_period_str",
    "scale",
    "network",
    "data_owner",
    "data_owner_email",
]


def combine_scenario_attrs(attrs_list: list[dict[str, Any]], context) -> dict[str, Any]:
    """Combine attributes when concatenating scenarios from different sites.

    The `ModelScenario.scenario`s in `get_combined_scenario` have the key "scenario" added
    to their attributes as a flag so this function can process the dataset attributes and
    the data variable attributes differently.

    TODO: add 'time_period', 'high_time/spatial_resolution', 'short_lifetime', 'heights'?
        Is 'time_period' from the footprint? Need to check model scenario...

    Args:
        attrs_list: list of attributes from datasets being concatenated
        context: additional parameter supplied by concatenate (this is required/supplied by xarray)

    Returns:
        dict that will be used as attributes for concatenated dataset
    """
    single_keys = [
        "species",
        "start_date",
        "end_date",
        "model",
        "metmodel",
        "domain",
        "max_longitude",
        "min_longitude",
        "max_latitude",
        "min_latitude",
    ]

    # take attributes from first element of attrs_list if key "scenario" is not in attributes
    # this is a flag set in `get_combined_scenarios` to facilitate combining attributes
    if "scenario" not in attrs_list[0]:
        return attrs_list[0]

    # processing for scenarios
    single_attrs = {
        k: attrs_list[0].get(k, "None") for k in single_keys
    }  # NoneType can't be saved to netCDF, use string instead
    list_attrs = defaultdict(list)
    for attrs in attrs_list:
        for key in list_keys:
            list_attrs[key].append(attrs.get(key, "None"))

    list_attrs = cast(dict, list_attrs)
    list_attrs.update(single_attrs)
    list_attrs["file_created"] = str(timestamp_now())
    return list_attrs


def make_combined_scenario(fp_all: dict) -> xr.Dataset:
    """Combine scenarios and merge in fluxes and boundary conditions.

    If fluxes and boundary conditions only have one coordinate for their
    "time" dimension, then "time" will be dropped.

    Otherwise, it is assumed that the time axis for fluxes and boundary conditions
    have the same length as the time axis for the model scenarios.

    """
    # combine scenarios by site
    scenarios = [v.expand_dims({"site": [k]}) for k, v in fp_all.items() if not k.startswith(".")]

    # add flag to top level attributes to help combine scenario attributes, without combining the
    # attributes of every data variable
    for scenario in scenarios:
        scenario.attrs["scenario"] = True

    combined_scenario = xr.concat(scenarios, dim="site", combine_attrs=combine_scenario_attrs)

    # make dtype of 'site' coordinate "<U3" (little-endian Unicode string of length 3)
    combined_scenario = combined_scenario.assign_coords(site=combined_scenario.site.astype(np.dtype("<U3")))

    # concat fluxes over source before merging into combined scenario
    fluxes = [v.data.expand_dims({"source": [k]}) for k, v in fp_all[".flux"].items()]
    combined_fluxes = xr.concat(fluxes, dim="source")

    if "time" in combined_fluxes.dims and combined_fluxes.sizes["time"] == 1:
        combined_fluxes = combined_fluxes.squeeze("time")

    # merge with override in case coordinates slightly off
    # (data should already be aligned by `ModelScenario`)
    combined_scenario = combined_scenario.merge(combined_fluxes, join="override")

    # merge in boundary conditions
    if ".bc" in fp_all:
        bc = fp_all[".bc"].data
        if "time" in bc.dims and bc.sizes["time"] == 1:
            bc = bc.squeeze("time")
        bc = bc.reindex_like(combined_scenario, method="nearest")
        combined_scenario = combined_scenario.merge(bc)

    return combined_scenario


def fp_all_from_dataset(ds: xr.Dataset) -> dict:
    """Recover "fp_all" dictionary from "combined scenario" dataset.

    This is the inverse of `make_combined_scenario`, except that the attributes of the
    scenarios, fluxes, and boundary conditions may be different.

    Args:
        ds: dataset created by `make_combined_scenario`

    Returns:
        dictionary containing model scenarios keyed by site, as well as flux and boundary conditions.
    """
    fp_all = {}

    # we'll get scales as we get scenarios
    fp_all[".scales"] = {}

    # get scenarios
    bc_vars = ["vmr_n", "vmr_e", "vmr_s", "vmr_w"]

    for i, site in enumerate(ds.site.values):
        scenario = (
            ds.sel(site=site, drop=True).drop_vars(["flux", *bc_vars], errors="ignore").drop_dims("source")
        )

        # extract attributes that were gathered into a list
        for k in list_keys:
            try:
                val = scenario.attrs[k][i]
            except (ValueError, IndexError):
                val = "None"

            if k == "scale":
                fp_all[".scales"][site] = val
            else:
                scenario.attrs[k] = val

        fp_all[site] = scenario.dropna("time")

    # get fluxes
    fp_all[".flux"] = {}

    for i, source in enumerate(ds.source.values):
        flux_ds = (
            ds[["flux"]]  # double brackets to get dataset
            .sel(source=source, drop=True)
            .expand_dims({"time": [ds.time.min().values]})
            .transpose(..., "time")
        )

        # extract attributes that were gathered into a list
        for k in list_keys:
            try:
                val = flux_ds.attrs[k][i]
            except (ValueError, IndexError):
                val = "None"
            flux_ds.attrs[k] = val

        fp_all[".flux"][source] = FluxData(data=flux_ds, metadata={})

    try:
        bc_ds = ds[bc_vars]
    except KeyError:
        pass
    else:
        bc_ds = bc_ds.expand_dims({"time": [ds.time.min().values]}).transpose(..., "time")
        fp_all[".bc"] = BoundaryConditionsData(data=bc_ds, metadata={})

    species = ds.attrs.get("species", None)
    if species is not None:
        species = species.upper()
    fp_all[".species"] = species

    try:
        fp_all[".units"] = float(ds.mf.attrs.get("units", 1.0))
    except ValueError:
        # conversion to float failed
        fp_all[".units"] = 1.0

    return fp_all
