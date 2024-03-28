# *****************************************************************************
# get_data.py
# Author: Atmospheric Chemistry Research Group, University of Bristol
# Created: Nov. 2023
# *****************************************************************************
# About
# Functions for retrieving observations and datasets for creating forward
# simulations
#
# Current options include:
# - "data_processing_surface_notracer": Surface based measurements, without tracers
# - "data_processing_surface_tracer"  : Surface based measurements, with tracers
#
# *****************************************************************************

import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, cast, Optional, Union

import numpy as np
from openghg.retrieve import get_obs_surface, get_flux
from openghg.retrieve import get_bc, get_footprint
from openghg.analyse import ModelScenario
from openghg.dataobjects import BoundaryConditionsData
from openghg.types import SearchError
from openghg.util import timestamp_now
import xarray as xr

import openghg_inversions.hbmcmc.inversionsetup as setup


def data_processing_surface_notracer(
    species,
    sites,
    domain,
    averaging_period,
    start_date,
    end_date,
    obs_data_level=None,
    inlet=None,
    instrument=None,
    calibration_scale=None,
    met_model=None,
    fp_model=None,
    fp_height=None,
    emissions_name=None,
    use_bc=True,
    bc_input=None,
    bc_store=None,
    obs_store=None,
    footprint_store=None,
    emissions_store=None,
    averagingerror=True,
    save_merged_data=False,
    merged_data_name=None,
    merged_data_dir=None,
):
    """
    Retrieve and prepare fixed-surface datasets from
    specified OpenGHG object stores for forward
    simulations and model-data comparisons that do not
    use tracers
    ---------------------------------------------
    Args:
        species (str):
            Atmospheric trace gas species of interest
            e.g. "co2"
        sites (list/str):
            List of strings containing measurement
            station/site abbreviations
            e.g. ["MHD", "TAC"]
        domain (str):
            Model domain region of interest
            e.g. "EUROPE"
        averaging_period (list/str):
            List of averaging periods to apply to
            mole fraction data. NB. len(averaging_period)==len(sites)
            e.g. ["1H", "1H"]
        start_date (str):
            Date from which to gather data
            e.g. "2020-01-01"
        end_date (str):
            Date until which to gather data
            e.g. "2020-02-01"
        obs_data_level (list/str):
            ICOS observations data level. For non-ICOS sites
            use "None"
        inlet (list/str/opt):
            Specific inlet height for the site observations
            (length must match number of sites)
        instrument (list/str/opt):
            Specific instrument for the site
            (length must match number of sites)
        calibration_scale (str):
            Convert measurements to defined calibration scale
        met_model (str/opt):
            Meteorological model used in the LPDM.
        fp_model (str):
            LPDM used for generating footprints.
        fp_height (list/str):
            Inlet height used in footprints for corresponding sites.
        emissions_name (list):
            List of keywords args associated with emissions files
            in the object store.
        use_bc (bool):
            Option to include boundary conditions in model
        bc_store (str):
            Name of object store to retrieve boundary conditions data from
        obs_store (str):
            Name of object store to retrieve observations data from
        footprint_store (str):
            Name of object store to retrieve footprints data from
        emissions_store (str):
            Name of object store to retrieve emissions data from
        averagingerror (bool/opt):
            Adds the variability in the averaging period to the measurement
            error if set to True.
        save_merged_data (bool/opt, default=False):
            Save forward simulations data and observations
        merged_data_name (str/opt):
            Filename for saved forward simulations data and observations
        merged_data_dir (str/opt):
            Directory path for for saved forward simulations data and observations
    """

    for i, site in enumerate(sites):
        sites[i] = site.upper()

    # Convert 'None' args to list
    nsites = len(sites)
    if inlet is None:
        inlet = [None] * nsites
    if instrument is None:
        instrument = [None] * nsites
    if fp_height is None:
        fp_height = [None] * nsites
    if obs_data_level is None:
        obs_data_level = [None] * nsites

    fp_all = {}
    fp_all[".species"] = species.upper()

    # Get flux data and add to dict.
    flux_dict = {}
    for source in emissions_name:
        get_flux_data = get_flux(
            species=species,
            domain=domain,
            source=source,
            start_date=start_date,
            end_date=end_date,
            store=emissions_store,
        )

        flux_dict[source] = get_flux_data
    fp_all[".flux"] = flux_dict

    footprint_dict = {}
    scales = {}
    check_scales = []
    site_indices_to_keep = []

    for i, site in enumerate(sites):
        # Get observations data
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
                store=obs_store,
            )
        except SearchError:
            print(
                f"\nNo obs data found for {site} with inlet {inlet[i]} and instrument {instrument[i]}. Check these values.\nContinuing model run without {site}.\n"
            )
            continue  # skip this site
        except AttributeError:
            print(
                f"\nNo data found for {site} between {start_date} and {end_date}.\nContinuing model run without {site}.\n"
            )
            continue  # skip this site
        else:
            if site_data is None:
                print(
                    f"\nNo data found for {site} between {start_date} and {end_date}.\nContinuing model run without {site}.\n"
                )
                continue  # skip this site
            unit = float(site_data[site].mf.units)

        # Get footprints data
        try:
            # Ensure HiTRes CO2 footprints are obtained if
            # using CO2
            if species.lower() == "co2":
                get_fps = get_footprint(
                    site=site,
                    height=fp_height[i],
                    domain=domain,
                    model=fp_model,
                    start_date=start_date,
                    end_date=end_date,
                    store=footprint_store,
                    species=species.lower(),
                )

            else:
                get_fps = get_footprint(
                    site=site,
                    height=fp_height[i],
                    domain=domain,
                    model=fp_model,
                    start_date=start_date,
                    end_date=end_date,
                    store=footprint_store,
                )
        except SearchError:
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
                get_bc_data.data.vmr_n.values = get_bc_data.data.vmr_n.values / unit
                get_bc_data.data.vmr_e.values = get_bc_data.data.vmr_e.values / unit
                get_bc_data.data.vmr_s.values = get_bc_data.data.vmr_s.values / unit
                get_bc_data.data.vmr_w.values = get_bc_data.data.vmr_w.values / unit
                my_bc = BoundaryConditionsData(
                    get_bc_data.data.transpose("height", "lat", "lon", "time"), get_bc_data.metadata
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
                    scenario_combined.bc_mod.values = scenario_combined.bc_mod.values * unit

            elif len(emissions_name) > 1:
                # Create model scenario object for each flux sector
                model_scenario_dict = {}

                for source in emissions_name:
                    scenario_sector = model_scenario.footprints_data_merge(sources=source, recalculate=True)

                    if species.lower() == "co2":
                        model_scenario_dict["mf_mod_high_res_" + source] = scenario_sector["mf_mod_high_res"]
                    elif species.lower() != "co2":
                        model_scenario_dict["mf_mod_" + source] = scenario_sector["mf_mod"]

                scenario_combined = model_scenario.footprints_data_merge(recalculate=True)

                for key in model_scenario_dict.keys():
                    scenario_combined[key] = model_scenario_dict[key]
                    if use_bc is True:
                        scenario_combined.bc_mod.values = scenario_combined.bc_mod.values * unit

            fp_all[site] = scenario_combined

            # Check consistency of measurement scales between sites
            check_scales += [scenario_combined.scale]
            if not all(s == check_scales[0] for s in check_scales):
                rt = []
                for i in check_scales:
                    if isinstance(i, list):
                        rt.extend(flatten(i))
                else:
                    rt.append(i)
                scales[site] = rt
            else:
                scales[site] = check_scales[0]

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

    fp_all[".scales"] = scales
    fp_all[".units"] = float(scenario_combined.mf.units)

    # If site contains measurement errors given as repeatability and variability,
    # use variability to replace missing repeatability values, then drop variability
    for site in sites:
        if "mf_variability" in fp_all[site] and "mf_repeatability" in fp_all[site]:
            fp_all[site]["mf_repeatability"][np.isnan(fp_all[site]["mf_repeatability"])] = fp_all[site][
                "mf_variability"
            ][
                np.logical_and(
                    np.isfinite(fp_all[site]["mf_variability"]), np.isnan(fp_all[site]["mf_repeatability"])
                )
            ]
            fp_all[site] = fp_all[site].drop_vars("mf_variability")

    # Add measurement variability in averaging period to measurement error
    if averagingerror:
        fp_all = setup.addaveragingerror(
            fp_all,
            sites,
            species,
            start_date,
            end_date,
            averaging_period,
            inlet=inlet,
            instrument=instrument,
            store=obs_store,
        )

    if save_merged_data:
        if merged_data_dir is None:
            print("`merged_data_dir` not specified; could not save merged data")
        else:
            save_merged_data_func(fp_all, merged_data_dir, merged_data_name=merged_data_name)
            print(f"\nfp_all saved in {merged_data_dir}\n")

    return fp_all, sites, inlet, fp_height, instrument, averaging_period


def _make_merged_data_name(species: str, start_date: str, output_name: str) -> str:
    return f"{species}_{start_date}_{output_name}_merged-data.pickle"


# NOTE: the _func at the end of the name is to distinguish from the local variable in the previous function
def save_merged_data_func(
    fp_all: dict,
    merged_data_dir: Union[str, Path],
    species: Optional[str] = None,
    start_date: Optional[str] = None,
    output_name: Optional[str] = None,
    merged_data_name: Optional[str] = None,
) -> None:
    """Save `fp_all` dictionary as a pickle file in `merged_data_dir`.

    The name of the pickle file can be specified using `merged_data_name`, or
    a standard name will be created given `species`, `start_date`, and `output_name`.

    If `merged_data_name` is not given, then `species`, `start_date`, and `output_name` must be provided.

    Args:
        fp_all: dictionary of merged data to save
        merged_data_dir: path to directory where merged data will be saved
        species: species of inversion
        start_date: start date of inversion period
        output_name: output name parameter used for inversion run
        merged_data_name: name to use for saved data.

    Returns:
        None
    """
    if merged_data_name is None:
        if any(arg is None for arg in [species, start_date, output_name]):
            raise ValueError(
                "If `merged_date_name` isn't given, then "
                "`species`, `start_date`, and `output_name` must be provided."
            )
        else:
            merged_data_name = _make_merged_data_name(species, start_date, output_name)  # type: ignore

    if isinstance(merged_data_dir, str):
        merged_data_dir = Path(merged_data_dir)

    with open(merged_data_dir / merged_data_name, "wb") as f:
        pickle.dump(fp_all, f)


def load_merged_data(
    merged_data_dir: Union[str, Path],
    species: Optional[str] = None,
    start_date: Optional[str] = None,
    output_name: Optional[str] = None,
    merged_data_name: Optional[str] = None,
) -> dict:
    """Load `fp_all` dictionary from a pickle file in `merged_data_dir`.

    The name of the pickle file can be specified using `merged_data_name`, or
    a standard name will be created given `species`, `start_date`, and `output_name`.

    If `merged_data_name` is not given, then `species`, `start_date`, and `output_name` must be provided.

    Args:
        merged_data_dir: path to directory where merged data will be saved
        species: species of inversion
        start_date: start date of inversion period
        output_name: output name parameter used for inversion run
        merged_data_name: name to use for saved data.

    Returns:
        `fp_all` dictionary
    """
    any_args_none = any(arg is None for arg in [species, start_date, output_name])
    if merged_data_name is None:
        if any_args_none:
            raise ValueError(
                "If `merged_date_name` isn't given, then "
                "`species`, `start_date`, and `output_name` must be provided."
            )
        else:
            merged_data_name = _make_merged_data_name(species, start_date, output_name)  # type: ignore

    if isinstance(merged_data_dir, str):
        merged_data_dir = Path(merged_data_dir)

    merged_data_file = merged_data_dir / merged_data_name

    if not merged_data_file.exists():
        if any_args_none:
            raise ValueError(
                f"No merged data with file name {merged_data_name} "
                f"found in merged data directory {merged_data_dir}"
            )
        else:
            raise ValueError(
                f"No merged data for species {species}, start date {start_date}, and "
                f"output name {output_name} found in merged data directory {merged_data_dir}"
            )

    with open(merged_data_file, "rb") as f:
        fp_all = pickle.load(f)

    return fp_all


def combine_scenario_attrs(attrs_list: list[dict[str, Any]], context) -> dict[str, Any]:
    """Combine attributes when concatenating scenarios from different sites.

    The `ModelScenario.scenario`s in `get_combined_scenario` have the key "scenario" added
    to their attributes as a flag so this function can process the dataset attributes and
    the data variable attributes differently.

    TODO: add 'time_period', 'high_time/spatial_resolution', 'short_lifetime', 'heights'?
        Is 'time_period' from the footprint? Need to check model scenario...

    Args:
        attrs_list: list of attributes from datasets being concatenated
        context: additional parameter supplied by concatenate

    Returns:
        dict that will be used as attributes for concatenated dataset
    """
    list_keys = [
        "species",
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
    single_keys = [
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


def combined_flux_attrs(attrs_list: list[dict[str, Any]], context) -> dict[str, Any]:
    """Combine attributes when concatenating fluxes from different sources.

    Currently just keeps a list of sources.

    NOTE: This assumes that the 'source' in OpenGHG metadata is the same as the 'source'
    in the attributes of the retrieved flux data.

    Args:
        attrs_list: list of attributes from datasets being concatenated
        context: additional parameter supplied by concatenate

    Returns:
        dict that will be used as attributes for concatenated dataset
    """
    if "source" in attrs_list[0]:
        return {"source": [attrs.get("source", "None") for attrs in attrs_list]}
    else:
        return attrs_list[0]


def make_combined_scenario(fp_all):
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


def recover_fp_all(ds: xr.Dataset) -> dict:
    """Recover "fp_all" dictionary from "combined scenario" dataset.

    Args:
        ds: dataset created by `make_combined_scenario`

    Returns:
        dictionary containing model scenarios keyed by site, as well as flux and boundary conditions.
    """
