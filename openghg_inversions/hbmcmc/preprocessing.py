from typing import Any, cast, Optional

import numpy as np
import pandas as pd
import xarray as xr

from openghg.analyse import ModelScenario
from openghg.dataobjects import BoundaryConditionsData, ObsData
from openghg.retrieve import get_obs_surface, get_flux, get_bc, get_footprint
from openghg.types import SearchError
from openghg.util import timestamp_now

import openghg_inversions.hbmcmc.inversionsetup as setup
import openghg_inversions.basis_functions as basis
from openghg_inversions import utils


def get_combined_data(
    species: str,
    sites: list[str],
    domain: str,
    measurement_periods: list[str],
    start_date: str,
    end_date: str,
    sources: list[str],
    fp_heights: list[str],
    fp_model: str = "NAME",
    met_model: Optional[str] = None,
    inlets: Optional[list[str | None]] = None,
    instruments: Optional[list[str | None]] = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Args:
      species: Atmospheric trace gas species of interest (e.g. 'co2')
      sites: List of site names
      domain: Inversion spatial domain.
      measurement_periods: Averaging period of measurements (must match number of sites).
      start_date: Start time of inversion "YYYY-mm-dd"
      end_date: End time of inversion "YYYY-mm-dd"
      met_model: Meteorological model used in the LPDM.
      fp_model: LPDM used for generating footprints.
      fp_height: height of footprint (must match number of sites). This allows for
        a slight mis-match between inlet height and footprint height.
      sources: List of keyword "source" args used for retrieving emissions files
        from the Object store.
      inlet: Specific inlet height for the site (must match number of sites)
      instrument: Specific instrument for the site (must match number of sites).

    Returns:
        dict: Dictionary with keys: '.species' (mapping to species), '.flux' (mapping to retrieved flux files),
              '.bc' (mapping to retrieve boundary condition files), '.units' (mapping to units), '.scales' (mapping to scales?)
              and a key for each site (mapping to combined footprints)
    """
    fp_all: dict[str, Any] = {}
    fp_all[".species"] = species.upper()

    Y_error: dict[str, Any] = {}  # error in Y obs
    averaging_error: dict[str, Any] = {}  # error due to resampling Y obs during `footprints_data_merge`

    # Get fluxes
    flux_dict = {}
    for source in sources:
        try:
            flux_data_result = get_flux(
                species=species, domain=domain, source=source, start_date=start_date, end_date=end_date
            )
        except SearchError as e:
            raise SearchError(f"Flux file with source '{source}' could not be retrieved.") from e
        else:
            flux_dict[source] = flux_data_result

    fp_all[".flux"] = flux_dict

    footprint_dict = {}
    scales = {}
    check_scales = []

    if inlets is None:
        inlets = cast(list[str | None], [None] * len(sites))
    if instruments is None:
        instruments = cast(list[str | None], [None] * len(sites))

    for site, inlet, instrument, average, fp_height in zip(
        sites, inlets, instruments, measurement_periods, fp_heights
    ):
        # Get observations
        try:
            site_data = get_obs_surface(
                site=site,
                species=species.lower(),
                inlet=inlet,
                start_date=start_date,
                end_date=end_date,
                average=average,
                instrument=instrument,
            )
        except SearchError as e:
            raise SearchError(
                f"Observation data for {site} between {start_date} to {end_date} could not be retrieved."
            ) from e
        else:
            site_data = cast(ObsData, site_data)  # get_obs_surface can't return none for local search...
            unit = float(site_data[site].mf.units)
            fp_all[".units"] = unit

        # Get footprints
        try:
            footprint_result = get_footprint(
                site=site,
                height=fp_height,
                domain=domain,
                model=fp_model,
                start_date=start_date,
                end_date=end_date,
            )
        except SearchError as e:
            raise SearchError(
                f"Footprint data for {site} between {start_date} to {end_date} could not be retrieved."
            ) from e
        else:
            footprint_dict[site] = footprint_result

        # Get boundary conditions
        if ".bc" not in fp_all:  # bc not loaded yet TODO: if new_fp_all.boundary_conditions is None
            try:
                bc_result = get_bc(species=species, domain=domain, start_date=start_date, end_date=end_date)
            except SearchError as e:
                raise SearchError(
                    f"Boundary condition data between {start_date} and {end_date} could not be retrieved."
                ) from e
            else:
                # Divide by trace gas species units
                # TODO See if this can be included 'behind the scenes'
                bc_result.data.vmr_n.values = bc_result.data.vmr_n.values / unit
                bc_result.data.vmr_e.values = bc_result.data.vmr_e.values / unit
                bc_result.data.vmr_s.values = bc_result.data.vmr_s.values / unit
                bc_result.data.vmr_w.values = bc_result.data.vmr_w.values / unit
                fp_all[".bc"] = BoundaryConditionsData(
                    bc_result.data.transpose("height", "lat", "lon", "time"), bc_result.metadata
                )

        # Create ModelScenario object
        model_scenario = ModelScenario(
            site=site,
            species=species,
            inlet=inlet,
            domain=domain,
            model=fp_model,
            metmodel=met_model,
            start_date=start_date,
            end_date=end_date,
            obs=site_data,
            footprint=footprint_dict[site],
            flux=flux_dict,
            bc=fp_all[".bc"],
        )

        scenario_combined = model_scenario.footprints_data_merge()
        scenario_combined.bc_mod.values = scenario_combined.bc_mod.values * unit
        scenario_combined.attrs["Domain"] = domain

        fp_all[site] = scenario_combined

        # create Y error vector and store averaging error.
        if "mf_repeatability" in scenario_combined:
            Y_error[site] = scenario_combined.mf_repeatability.fillna(np.median(scenario_combined.mf_repeatability.values)).rename("error")
            Y_error[site].attrs["long_name"] = "Y error as mole fraction"
            averaging_error[site] = scenario_combined.mf_variability
        else:
            Y_error[site] = scenario_combined.mf_variability.fillna(np.median(scenario_combined.mf_variability.values)).rename("error")
            Y_error[site].attrs["long_name"] = "Y error as mole fraction"
            averaging_error[site] = xr.zeros_like(scenario_combined.mf_variability)


        # Check consistency of measurement scales between sites
        check_scales += [scenario_combined.scale]
        if not all(s == check_scales[0] for s in check_scales):
            rt = []
            for scale in check_scales:
                if isinstance(scale, list):
                    rt.extend(
                        scale
                    )  # NOTE: removed call to non-existent 'flatten' function, so apparently this point has never been reached
                else:
                    rt.append(scale)
            scales[site] = rt
        else:
            scales[site] = check_scales[0]

    fp_all[".scales"] = scales

    return fp_all, Y_error, averaging_error


def get_combined_scenario(
    species: str,
    sites: list[str],
    domain: str,
    resampling_periods: list[str],
    start_date: str,
    end_date: str,
    sources: list[str],
    fp_heights: list[str],
    fp_model: str = "NAME",
    met_model: Optional[str] = None,
    inlets: Optional[list[str | None]] = None,
    instruments: Optional[list[str | None]] = None,
    units: float = 1.0
) -> xr.Dataset:
    """
    Args:
      species: Atmospheric trace gas species of interest (e.g. 'co2')
      sites: List of site names
      domain: Inversion spatial domain.
      resampling_periods: Averaging period of measurements (must match number of sites).
      start_date: Start time of inversion "YYYY-mm-dd"
      end_date: End time of inversion "YYYY-mm-dd"
      met_model: Meteorological model used in the LPDM.
      fp_model: LPDM used for generating footprints.
      fp_height: height of footprint (must match number of sites). This allows for
        a slight mis-match between inlet height and footprint height.
      sources: List of keyword "source" args used for retrieving emissions files
        from the Object store.
      inlet: Specific inlet height for the site (must match number of sites)
      instrument: Specific instrument for the site (must match number of sites).
      units: Units for observations.

    Returns:
        xarray Dataset with model scenarios for the given parameters concatenated along a 'site' dimension, along
    with boundary conditions and fluxes, which are concatenated along a 'source' dimension.

    The attributes are combined so that the 'site' attribute contains the sites in the order they
    were concatenated, and the 'source' attribute of the 'flux' data variable contains the sources
    in the order they were concatenated.

    For instance, if `fp_all = {'TAC': ds1, 'MHD': ds2, ...}` and `result = make_combined_scenario(fp_all)`,
    then `result.isel(site=0)` will select the scenario for 'TAC', and `result.isel(site=1)` will select the
    scenario for 'MHD'. Further, `result.attrs['site']` is equal to `['TAC', 'MHD']`.

    Similarly, if `fp_all['.flux'] = {'waste': v1, 'agric': v2}`, then `result.flux.attrs['source']` is equal
    to `['waste', 'agric']` and `result.flux.isel(source=0)` will select 'waste', and `result.`

    NOTE: it would be nicer to use strings in the 'site' and 'source' coordinates, but this makes it harder to
    compress the Dataset.

    TODO: what happens if some obs have mf_repeatability and some do not?

    TODO: how should attributes for individual data variables be handled? Currently the same parser for the 'scenario'
    attrs is also used on each data variable.
    """
    units_set = False  # flag, set to True when units found in observation data
    bc_set = False  # flag, set to True when boundary conditions retrieved
    bc: Optional[BoundaryConditionsData] = None
    scenarios: list[xr.Dataset] = []

    flux_dict = {}
    for source in sources:
        flux_dict[source] = get_flux(
                species=species, domain=domain, source=source, start_date=start_date, end_date=end_date
            )


    if inlets is None:
        inlets = cast(list[str | None], [None] * len(sites))
    if instruments is None:
        instruments = cast(list[str | None], [None] * len(sites))

    for site, inlet, instrument, average, fp_height in zip(
        sites, inlets, instruments, resampling_periods, fp_heights
    ):
        # Get observations
        site_data = get_obs_surface(
                site=site,
                species=species.lower(),
                inlet=inlet,
                start_date=start_date,
                end_date=end_date,
                average=average,
                instrument=instrument,
            )
        site_data = cast(ObsData, site_data)  # get_obs_surface can't return none for local search...

        if not units_set:
            units = float(site_data[site].mf.units)
            units_set = True

        # Get footprints
        footprint_result = get_footprint(
                site=site,
                height=fp_height,
                domain=domain,
                model=fp_model,
                start_date=start_date,
                end_date=end_date,
            )

        # Get boundary conditions
        if not bc_set:
            bc_result = get_bc(species=species, domain=domain, start_date=start_date, end_date=end_date)

            # Divide by trace gas species units
            # TODO See if this can be included 'behind the scenes'
            bc_result.data.vmr_n.values = bc_result.data.vmr_n.values / units
            bc_result.data.vmr_e.values = bc_result.data.vmr_e.values / units
            bc_result.data.vmr_s.values = bc_result.data.vmr_s.values / units
            bc_result.data.vmr_w.values = bc_result.data.vmr_w.values / units

            bc = BoundaryConditionsData(
                bc_result.data.transpose("height", "lat", "lon", "time"), bc_result.metadata
            )
            bc = cast(BoundaryConditionsData, bc)
            bc_set = True

        # Create ModelScenario object
        model_scenario = ModelScenario(
            site=site,
            species=species,
            inlet=inlet,
            domain=domain,
            model=fp_model,
            metmodel=met_model,
            start_date=start_date,
            end_date=end_date,
            obs=site_data,
            footprint=footprint_result,
            flux=flux_dict,
            bc=bc,
        )

        scenario = model_scenario.footprints_data_merge()
        scenario.bc_mod.values = scenario.bc_mod.values * units

        scenarios.append(scenario)

    # combine scenarios, fluxes, and bc into single xr.Dataset
    scenarios = [scenario.expand_dims({"site": [i]}) for i, scenario in enumerate(scenarios)]
    combined_scenario = xr.concat(scenarios, dim="site", combine_attrs=combine_scenario_attrs)

    # concat fluxes over source before merging into combined scenario
    fluxes = [v.data.expand_dims({"source": [i]}) for i, v in enumerate(flux_dict.values())]
    combined_fluxes = xr.concat(fluxes, dim="source", combine_attrs=combined_flux_attrs)

    # merge with override in case coordinates slightly off
    # squeeze to remove single time coordinate
    combined_scenario = combined_scenario.merge(combined_fluxes.squeeze(dim="time"), join="override")

    if bc:
        combined_scenario = combined_scenario.merge(bc.data.squeeze(dim="time"), join="override")

    return combined_scenario


def combine_scenario_attrs(attrs_list: list[dict[str, Any]], context) -> dict[str, Any]:
    """Combine attributes when concatenating scenarios from different sites.

    TODO: add 'time_period', 'high_time/spatial_resolution', 'short_lifetime', 'heights'?
        Is 'time_period' from the footprint? Need to check model scenario...

    TODO: add parsing for each data variable?

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
    single_keys = ["start_date", "end_date", "model", "metmodel", "domain", "max_longitude", "min_longitude", "max_latitude", "min_latitude"]

    single_attrs = {k: attrs_list[0].get(k, None) for k in single_keys}
    list_attrs = defaultdict(list)
    for attrs in attrs_list:
        for key in list_keys:
            list_attrs[key].append(attrs.get(key, None))

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
    return {"source": [attrs.get('source', None) for attrs in attrs_list]}


def make_combined_scenario(fp_all):
    """Combine scenarios and merge in fluxes and boundary conditions.

    If fluxes and boundary conditions only have one coordinate for their
    "time" dimension, then "time" will be dropped.

    Otherwise, it is assumed that the time axis for fluxes and boundary conditions
    have the same length as the time axis for the model scenarios.

    TODO: incorporate errors
    """
    scenarios = {k: v for k, v in fp_all.items() if not k.startswith(".")}
    site_codes = {}

    for i, site in enumerate(scenarios):
        scenarios[site] = scenarios[site].expand_dims({"site": [i]})
        site_codes[i] = site

    combined_scenario = xr.concat(scenarios.values(), dim="site")

    # concat fluxes over source before merging into combined scenario
    fluxes = {k: v.data for k, v in fp_all[".flux"].items()}
    flux_codes = {}

    for i, source in enumerate(fluxes):
        fluxes[source] = fluxes[source].expand_dims({"source": [i]})
        flux_codes[i] = source

    combined_fluxes = xr.concat(fluxes.values(), dim="source")

    # merge with override in case coordinates slightly off
    # squeeze to remove single time coordinate
    combined_scenario = combined_scenario.merge(combined_fluxes.squeeze(dim="time"), join="override")

    # merge in boundary conditions
    bc = fp_all[".bc"].data
    combined_scenario = combined_scenario.merge(bc.squeeze(dim="time"), join="override")

    return combined_scenario, site_codes, flux_codes


def fixedbasisMCMC_preprocessing(
    species,
    sites,
    domain,
    meas_period,
    start_date,
    end_date,
    outputpath,
    outputname,
    fp_height: list[str],
    emissions_name: list[str],
    fp_model="NAME",
    met_model=None,
    inlet=None,
    instrument=None,
    fp_basis_case=None,
    basis_directory=None,
    bc_basis_case="NESW",
    bc_basis_directory=None,
    quadtree_basis=True,
    nbasis=100,
    save_quadtree_to_outputpath=False,
    project_to_basis=True,
    filters=[],
    averagingerror=True,
    bc_freq=None,
    sigma_freq=None,
):
    """
    Script to run hierarchical Bayesian
    MCMC for inference of emissions using
    PyMC3 to solve the inverse problem.
    -----------------------------------
    Args:
      species (str):
        Atmospheric trace gas species of interest (e.g. 'co2')
      sites (list):
        List of site names
      domain (str):
        Inversion spatial domain.
      meas_period (list):
        Averaging period of measurements
      start_date (str):
        Start time of inversion "YYYY-mm-dd"
      end_date (str):
        End time of inversion "YYYY-mm-dd"
      outputname (str):
        Unique identifier for output/run name.
      outputpath (str):
        Path to where output should be saved.
      met_model (str):
        Meteorological model used in the LPDM.
      fp_model (str):
        LPDM used for generating footprints.
      xprior (dict):
        Dictionary containing information about the prior PDF for emissions.
        The entry "pdf" is the name of the analytical PDF used, see
        https://docs.pymc.io/api/distributions/continuous.html for PDFs
        built into pymc3, although they may have to be coded into the script.
        The other entries in the dictionary should correspond to the shape
        parameters describing that PDF as the online documentation,
        e.g. N(1,1**2) would be: xprior={pdf:"normal", "mu":1, "sd":1}.
        Note that the standard deviation should be used rather than the
        precision. Currently all variables are considered iid.
      bcprior (dict):
        Same as above but for boundary conditions.
      sigprior (dict):
        Same as above but for model error.
      offsetprior (dict):
        Same as above but for bias offset. Only used is addoffset=True.
      nit (int):
        Number of iterations for MCMC
      burn (int):
        Number of iterations to burn in MCMC
      tune (int):
        Number of iterations to use to tune step size
      nchain (int):
        Number of independent chains to run (there is no way at all of
        knowing whether your distribution has converged by running only
        one chain)
      emissions_name (list):
        List of keyword "source" args used for retrieving emissions files
        from the Object store.
      inlet (str/list, optional):
        Specific inlet height for the site (must match number of sites)
      instrument (str/list, optional):
        Specific instrument for the site (must match number of sites).
      fp_basis_case (str, optional):
        Name of basis function to use for emissions.
      bc_basis_case (str, optional):
        Name of basis case type for boundary conditions (NOTE, I don't
        think that currently you can do anything apart from scaling NSEW
        boundary conditions if you want to scale these monthly.)
      basis_directory (str, optional):
        Directory containing the basis function
        if not default.
      bc_basis_directory (str, optional):
        Directory containing the boundary condition basis functions
        (e.g. files starting with "NESW")
      country_file (str, optional):
        Path to the country definition file
      max_level (int, optional):
        The maximum level for a column measurement to be used for getting obs data
      quadtree_basis (bool, optional):
        Creates a basis function file for emissions on the fly using a
        quadtree algorithm based on the a priori contribution to the mole
        fraction if set to True.
      nbasis (int):
        Number of basis functions that you want if using quadtree derived
        basis function. This will optimise to closest value that fits with
        quadtree splitting algorithm, i.e. nbasis % 4 = 1.
      filters (list, optional):
        list of filters to apply from name.filtering. Defaults to empty list
      averagingerror (bool, optional):
        Adds the variability in the averaging period to the measurement
        error if set to True.
      bc_freq (str, optional):
        The perdiod over which the baseline is estimated. Set to "monthly"
        to estimate per calendar month; set to a number of days,
        as e.g. "30D" for 30 days; or set to None to estimate to have one
        scaling for the whole inversion period.
      sigma_freq (str, optional):
        as bc_freq, but for model sigma
      sigma_per_site (bool):
        Whether a model sigma value will be calculated for each site
        independantly (True) or all sites together (False).
        Default: True
      country_unit_prefix ('str', optional)
        A prefix for scaling the country emissions. Current options are:
       'T' will scale to Tg, 'G' to Gg, 'M' to Mg, 'P' to Pg.
        To add additional options add to convert.prefix
        Default is none and no scaling will be applied (output in g).
      add_offset (bool):
        Add an offset (intercept) to all sites but the first in the site list.
        Default False.

    Returns:
        Saves an output from the inversion code using inferpymc3_postprocessouts.

    -----------------------------------
    """
    fp_all, Y_error, averaging_error = get_combined_data(
        species=species,
        sites=sites,
        domain=domain,
        measurement_periods=meas_period,
        sources=emissions_name,
        fp_heights=fp_height,
        fp_model=fp_model,
        met_model=met_model,
        inlets=inlet,
        instruments=instrument,
        start_date=start_date,
        end_date=end_date,
    )

    if averagingerror == True:
        Y_err_final = {}
        for site in sites:
            Y_err_final[site] = np.sqrt(Y_error[site]**2 + averaging_error[site]**2)
    else:
        Y_err_final = Y_error

    if project_to_basis:
        # Create basis function using quadtree algorithm if needed
        tempdir = None
        if quadtree_basis:
            if fp_basis_case != None:
                print("Basis case %s supplied but quadtree_basis set to True" % fp_basis_case)
                print("Assuming you want to use %s " % fp_basis_case)
            else:
                if save_quadtree_to_outputpath:
                    outputdir = outputpath
                else:
                    outputdir = None

                tempdir = basis.quadtreebasisfunction(
                    emissions_name,
                    fp_all,
                    sites,
                    start_date,
                    domain,
                    species,
                    outputname,
                    outputdir=outputdir,
                    nbasis=nbasis,
                )
                fp_basis_case = "quadtree_" + species + "-" + outputname
                basis_directory = tempdir
        else:
            basis_directory = basis_directory

        fp_data = utils.fp_sensitivity(
            fp_all, domain=domain, basis_case=fp_basis_case, basis_directory=basis_directory
        )

        fp_data = utils.bc_sensitivity(
            fp_data, domain=domain, basis_case=bc_basis_case, bc_basis_directory=bc_basis_directory
        )
    else:
        for site in sites:
            site_time = fp_all[site].time
            H_vals = list(fp_all['.flux'].values())[0].data.squeeze().flux.values.reshape((-1, 1)) * fp_all[site]["fp"].values.reshape((-1, len(site_time)))
            H = xr.DataArray(H_vals, coords=[("region", np.arange(H_vals.shape[0])), ("time", site_time.data)])
            fp_all[site]['H'] = H

            Hbcs = []
            for d in ["n", "e", "s", "w"]:
                bc = fp_all['.bc'].data[f"vmr_{d}"].squeeze().values.reshape((-1, 1))
                Hbc_temp = fp_all[site][f"particle_locations_{d}"].values.reshape((-1, len(site_time)))
                Hbcs.append(Hbc_temp * bc)
            Hbc_vals = np.vstack(Hbcs)
            Hbc = xr.DataArray(Hbc_vals, coords=[("region",np.arange(Hbc_vals.shape[0])), ("time", site_time.data)])
            fp_all[site]['H_bc'] = Hbc

        fp_data = fp_all


    # Apply named filters to the data
    fp_data = utils.filtering(fp_data, filters)

    # Get inputs ready
    Y = np.concatenate([fp_data[site].mf.values for site in sites])
    Ytime = np.concatenate([fp_data[site].time.values for site in sites])
    error = np.concatenate([Y_err_final[site] for site in sites])
    siteindicator = np.concatenate([i * np.ones_like(fp_data[site].mf.values) for i, site in enumerate(sites)])

    Hx = np.hstack([fp_data[site].H.values for site in sites])

    if bc_freq == "monthly":
        Hbc = np.hstack([setup.monthly_bcs(start_date, end_date, site, fp_data) for site in sites])
    elif bc_freq == None:
        Hbc = np.hstack([fp_data[site].H_bc.values for site in sites])
    else:
        Hbc = np.hstack([setup.create_bc_sensitivity(start_date, end_date, site, fp_data, bc_freq) for site in sites])

    sigma_freq_index = setup.sigma_freq_indicies(Ytime, sigma_freq)

    return Hx, Hbc, Y, Ytime, error, siteindicator, sigma_freq_index, fp_data, Y_error, averaging_error
