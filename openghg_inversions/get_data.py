# *****************************************************************************
# Created: 30 Nov. 2023
# Author: Eric Saboya, School of Geographical Sciences, University of Bristol
# Contact: eric.saboya@bristol.ac.uk
# *****************************************************************************
# About
# Different functions for retrieving appropriate datasets for forward
# simulations
#
# Current options include:
# - "data_processing_surface_notracer": Surface based measurements, without tracers
# - "data_processing_surface_tracer"  : Surface based measurements, with tracers
#
# *****************************************************************************

import os
import sys
import shutil
import numpy as np
import pandas as pd
import pickle

import openghg_inversions.hbmcmc.inversionsetup as setup
from openghg.retrieve import get_obs_surface, get_flux
from openghg.retrieve import get_bc, get_footprint
from openghg.analyse import ModelScenario
from openghg.dataobjects import BoundaryConditionsData
from openghg.types import SearchError


def data_processing_surface_notracer(
    species,
    sites,
    domain,
    averaging_period,
    start_date,
    end_date,
    met_model=None,
    fp_model="NAME",
    fp_height=None,
    emissions_name=None,
    inlet=None,
    instrument=None,
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
    simulations that do not use tracers
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
        met_model (str/opt):
            Meteorological model used in the LPDM.
        fp_model (str):
            LPDM used for generating footprints.
        fp_height (list/str):
            Inlet height used in footprints for corresponding sites.
        emissions_name (list):
            List of keywords args associated with emissions files
            in the object store.
        inlet (str/list, optional):
            Specific inlet height for the site observations.
            (length must match number of sites)
        instrument (str/list, optional):
            Specific instrument for the site
            (length must match number of sites).
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

    """

    # Change list of sites to upper case equivalent as
    # most functions use upper case notation
    for i, site in enumerate(sites):
        sites[i] = site.upper()

    fp_all = {}
    fp_all[".species"] = species.upper()

    # Get fluxes
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
        try:
            # Get observations
            site_data = get_obs_surface(
                site=site,
                species=species.lower(),
                inlet=inlet[i],
                start_date=start_date,
                end_date=end_date,
                average=averaging_period[i],
                instrument=instrument[i],
                store=obs_store,
            )

            unit = float(site_data[site].mf.units)

            # Get footprints
            get_fps = get_footprint(
                site=site,
                height=fp_height[i],
                domain=domain,
                model=fp_model,
                start_date=start_date,
                end_date=end_date,
                store=footprint_store,
            )
            footprint_dict[site] = get_fps

            # Get boundary conditions
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

            # Create ModelScenario object for all emissions_sectors
            # and combine into one object
            if len(emissions_name) == 1:
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

                scenario_combined = model_scenario.footprints_data_merge()
                scenario_combined.bc_mod.values = scenario_combined.bc_mod.values * unit

            elif len(emissions_name) > 1:
                model_scenario_dict = {}

                for source in emissions_name:
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

                    scenario_sector = model_scenario.footprints_data_merge(sources=source)
                    model_scenario_dict["mf_mod_high_res_" + source] = scenario_sector["mf_mod_high_res"]

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

                scenario_combined = model_scenario.footprints_data_merge()

                for key in model_scenario_dict.keys():
                    scenario_combined[key] = model_scenario_dict[key]

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
                f"\nError in reading in data for {site}, possibly because there is no obs for this time period."
                + f"\nContinuing model run without {site}.\n"
            )

    # if data was not extracted correctly for any sites, drop these from the rest of the inversion
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

    if save_merged_data == True:
        with open(os.path.join(merged_data_dir, merged_data_name), "wb") as fp_out:
            pickle.dump(fp_all, fp_out)
            fp_out.close()

        print(f"\nfp_all saved in {merged_data_dir}\n")

    return fp_all, sites, inlet, fp_height, instrument, averaging_period
