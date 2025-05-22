"""Functions for retrieving observations and datasets for creating forward simulations.

Current data processing options include:
- "data_processing_surface_notracer": Surface based measurements, without tracers

Future data processing options will include:
- "data_processing_surface_tracer": Surface based measurements, with tracers

This module also includes functions for saving and loading "merged data" created
by the data processing functions.
"""

import logging
from typing import Literal

import numpy as np
import xarray as xr

from openghg.retrieve import get_bc
from openghg.types import SearchError

from openghg_inversions.inversion_data.getters import (
    convert_bc_units,
    get_flux_data,
    get_footprint_data,
    get_obs_data,
)
from openghg_inversions.inversion_data.scenario import merged_scenario_data
from openghg_inversions.inversion_data.serialise import _save_merged_data


logger = logging.getLogger(__name__)


def add_obs_error(sites: list[str], fp_all: dict, add_averaging_error: bool = True) -> None:
    """Create `mf_error` variable.

    The `mf_error` variables contains either `mf_repeatablility`, `mf_variability`
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
            ds["mf_variability"].attrs["long_name"] = ds.mf.attrs.get("long_name", "") + "_variability"
            variability_missing = True

        if "mf_repeatability" not in ds:
            if variability_missing:
                raise ValueError(f"Obs data for site {site} is missing both repeatability and variability.")

            ds["mf_repeatability"] = xr.zeros_like(ds.mf_variability)
            ds["mf_repeatability"].attrs["long_name"] = ds.mf.attrs.get("long_name", "") + "_repeatability"

            ds["mf_error"] = ds["mf_variability"]

            if add_averaging_error:
                logger.info(
                    "`mf_repeatability` not present; using `mf_variability` for `mf_error` at site %s", site
                )

        elif add_averaging_error:
            ds["mf_error"] = np.sqrt(ds["mf_repeatability"] ** 2 + ds["mf_variability"] ** 2)
        else:
            ds["mf_error"] = ds["mf_repeatability"]

        ds["mf_error"].attrs["long_name"] = ds.mf.attrs.get("long_name", "") + "_error"
        ds["mf_error"].attrs["units"] = ds.mf.attrs.get("units", None)

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


def convert_to_list(x: list[str | None] | str | None, length: int, name: str | None = None) -> list[str | None]:
    """Convert variable that might be list, str, or None to a list of the expected size.

    Args:
        x: variable to convert to list
        length: length of the output list
        name: name to use for error message

    Returns:
        list of specified length; either the original list, or a list containing
        repeats of the input value

    Raises:
        ValueError: if input is a list and its length differs from the specified
        length.

    """
    if x is None or isinstance(x, str):
        return [x] * length

    if len(x) != length:
        msg_name = name or x  # display entire list if name is not given
        raise ValueError(f"List {msg_name} does not have specified length: {len(x)} != {length}.")
    return x


def data_processing_surface_notracer(
    species: str,
    sites: list | str,
    domain: str,
    averaging_period: list[str | None] | str | None,
    start_date: str,
    end_date: str,
    obs_data_level: list[str | None] | str | None = None,
    platform: list[str | None] | str | None = None,
    inlet: list[str | None] | str | None = None,
    instrument: list[str | None] | str | None = None,
    max_level: int | None = None,
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
            NOTE: for satellite, pass as "satellitename-obs_region" eg "GOSAT-BRAZIL" and pass corresponding platform as "satellite"
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
      max_level:
        Maximum atmospheric level to extract. Only needed if using 
        satellite data. Must be an int 
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
    inlet = convert_to_list(inlet, nsites, "inlet")
    instrument = convert_to_list(instrument, nsites, "instrument")
    fp_height = convert_to_list(fp_height, nsites, "fp_height")
    obs_data_level = convert_to_list(obs_data_level, nsites, "obs_data_level")
    met_model = convert_to_list(met_model, nsites, "met_model")
    averaging_period = convert_to_list(averaging_period, nsites, "averaging_period")
    platform = convert_to_list(platform, nsites, "platform")

    fp_all = {}
    fp_all[".species"] = species.upper()

    # Get flux data
    if emissions_name is None:
        raise ValueError("`emissions_name` must be specified")

    flux_dict = get_flux_data(
        sources=emissions_name,
        species=species,
        domain=domain,
        start_date=start_date,
        end_date=end_date,
        store=emissions_store,
    )
    fp_all[".flux"] = flux_dict

    # Get BC data
    if use_bc is True:
        try:
            bc_data = get_bc(
                species=species,
                domain=domain,
                bc_input=bc_input,
                start_date=start_date,
                end_date=end_date,
                store=bc_store,
            )
        except SearchError as e:
            raise SearchError("Could not find matching boundary conditions.") from e
        else:
            fp_all[".bc"] = convert_bc_units(
                bc_data, 1.0
            )  # transpose coordinates, keep for consistency with old format
    else:
        bc_data = None

    # get obs and footprints, and make scenarios for each site
    scales = {}
    check_scales = set()
    site_indices_to_keep = []

    for i, site in enumerate(sites):
        # Get observations data
        # TODO: update this to get column data if platform is satellite
        site_data = get_obs_data(
            site=site,
            species=species,
            platform=platform[i],
            inlet=inlet[i],
            start_date=start_date,
            end_date=end_date,
            data_level=obs_data_level[i],
            average=averaging_period[i],
            instrument=instrument[i],
            calibration_scale=calibration_scale,
            max_level=max_level,
            stores=obs_store,
        )

        if site_data is None:
            print(f"No obs. found, continuing model run without {site}.\n")
            continue

        # Get footprints data

        print(site, domain, fp_height[i], start_date, end_date, fp_model, met_model, fp_species, averaging_period, site_data, footprint_store)
        footprint_data = get_footprint_data(
            site=site,
            domain=domain,
            platform=platform[i],
            fp_height=fp_height[i],
            start_date=start_date,
            end_date=end_date,
            model=fp_model,
            met_model=met_model[i],
            fp_species=fp_species,
            averaging_period=averaging_period[i],
            obs_data=site_data,
            stores=footprint_store,
        )
        if footprint_data is None:
            print(
                f"\nNo footprint data found for {site} with inlet/height {fp_height[i]}, model {fp_model}, and domain {domain}.",
                f"Check these values.\nContinuing model run without {site}.\n",
            )
            continue  # skip this site
        scenario_combined = merged_scenario_data(site_data, footprint_data, flux_dict, bc_data, platform=platform)
        fp_all[site] = scenario_combined

        scales[site] = scenario_combined.scale
        check_scales.add(scenario_combined.scale)

        site_indices_to_keep.append(i)
    if "satellite" not in footprint_data.metadata:
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
