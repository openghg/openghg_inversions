import numpy as np

import openghg_inversions.hbmcmc.inversionsetup as setup
from openghg.retrieve import get_obs_surface, get_flux
from openghg.retrieve import get_bc, get_footprint
from openghg.analyse import ModelScenario
from openghg.dataobjects import BoundaryConditionsData
import openghg_inversions.basis_functions as basis
from openghg_inversions import utils


def fixedbasisMCMC(
    species,
    sites,
    domain,
    meas_period,
    start_date,
    end_date,
    outputpath,
    outputname,
    met_model=None,
    fp_model="NAME",
    fp_height=None,
    emissions_name=None,
    inlet=None,
    instrument=None,
    fp_basis_case=None,
    basis_directory=None,
    bc_basis_case="NESW",
    bc_basis_directory=None,
    country_file=None,
    max_level=None,
    quadtree_basis=True,
    nbasis=100,
    filters=[],
    averagingerror=True,
    bc_freq=None,
    sigma_freq=None,
    sigma_per_site=True,
    country_unit_prefix=None,
    add_offset=False,
    verbose=False,
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
    # Change list of sites to upper case equivalent as
    # most acrg functions copied across use upper case notation
    for i, site in enumerate(sites):
        sites[i] = site.upper()

    fp_all = {}
    fp_all[".species"] = species.upper()

    # Get fluxes
    flux_dict = {}
    for source in emissions_name:
        try:
            print(f"Attempting to retrieve '{source}' fluxes" " from object store ...\n")

            get_flux_data = get_flux(
                species=species, domain=domain, source=source, start_date=start_date, end_date=end_date
            )

            print("Sucessfully retrieved flux file" f" '{source}' from objectstore.")

            flux_dict[source] = get_flux_data
        except:
            raise FileNotFoundError(
                f"Flux file '{source}' not found" " in object store. Please add file to object store."
            )
    fp_all[".flux"] = flux_dict

    footprint_dict = {}
    scales = {}
    check_scales = []

    for i, site in enumerate(sites):
        # Get observations
        try:
            print(
                f"Attempting to retrieve {species.upper()} measurements"
                f" for {site.upper()} between {start_date} and"
                f" {end_date} from object store ...\n"
            )
            site_data = get_obs_surface(
                site=site,
                species=species.lower(),
                inlet=inlet[i],
                start_date=start_date,
                end_date=end_date,
                average=meas_period[i],
                instrument=instrument[i],
            )
            print(
                f"Successfully retrieved {species.upper()} measurement"
                f" data for {site.upper()} from object store.\n"
            )
            unit = float(site_data[site].mf.units)
        except:
            raise FileNotFoundError(
                f"Observation data for {site}"
                f" between {start_date} to {end_date} was"
                f" not found in the object store."
            )

        # Get footprints
        try:
            print(f"Attempting to retrieve {site.upper()} {domain} footprint" " data from object store ...\n")
            get_fps = get_footprint(
                site=site,
                height=fp_height[i],
                domain=domain,
                model=fp_model,
                start_date=start_date,
                end_date=end_date,
            )
            footprint_dict[site] = get_fps
            print(f"Successfully retrieved {site.upper()} {domain} footprints" " from object store.\n")
        except:
            raise FileNotFoundError(
                f"Footprint data for {site.upper()}"
                f" between {start_date} to {end_date}"
                f" was not found in the object store."
            )

        # Get boundary conditions
        try:
            print(
                "Attempting to retrieve boundary condition data"
                f" between {start_date} to {end_date} from object store ...\n"
            )
            get_bc_data = get_bc(species=species, domain=domain, start_date=start_date, end_date=end_date)
            print(
                "Successfully retrieved boundary condition data between"
                f" {start_date} and {end_date} from object store.\n"
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
        except:
            raise FileNotFoundError(
                f"Boundary condition data between {start_date}"
                f" and {end_date} not found in object store. Please add"
                " boundary condition data to object store. Exiting process.\n"
            )

        # Create ModelScenario object
        model_scenario = ModelScenario(
            site=site,
            species=species,
            inlet=inlet[i],
            domain=domain,
            model=fp_model,
            metmodel=met_model,
            start_date=start_date,
            end_date=end_date,
            obs=site_data,
            footprint=footprint_dict[site],
            flux=flux_dict,
            bc=my_bc,
        )

        scenario_combined = model_scenario.footprints_data_merge()
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

    fp_all[".scales"] = scales
    fp_all[".units"] = float(scenario_combined.mf.units)

    print(f"Running for {start_date} to {end_date}")

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
            fp_all, sites, species, start_date, end_date, meas_period, inlet=inlet, instrument=instrument
        )

    # Create basis function using quadtree algorithm if needed
    if quadtree_basis:
        if fp_basis_case != None:
            print("Basis case %s supplied but quadtree_basis set to True" % fp_basis_case)
            print("Assuming you want to use %s " % fp_basis_case)
        else:
            tempdir = basis.quadtreebasisfunction(
                emissions_name, fp_all, sites, start_date, domain, species, outputname, nbasis=nbasis
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

    # Apply named filters to the data
    fp_data = utils.filtering(fp_data, filters)

    for si, site in enumerate(sites):
        fp_data[site].attrs["Domain"] = domain

    # Get inputs ready
    error = np.zeros(0)
    Hbc = np.zeros(0)
    Hx = np.zeros(0)
    Y = np.zeros(0)
    Ytime = np.zeros(0)
    siteindicator = np.zeros(0)
    for si, site in enumerate(sites):
        if "mf_repeatability" in fp_data[site]:
            error = np.concatenate((error, fp_data[site].mf_repeatability.values))
        if "mf_variability" in fp_data[site]:
            error = np.concatenate((error, fp_data[site].mf_variability.values))

        Y = np.concatenate((Y, fp_data[site].mf.values))
        siteindicator = np.concatenate((siteindicator, np.ones_like(fp_data[site].mf.values) * si))
        if si == 0:
            Ytime = fp_data[site].time.values
        else:
            Ytime = np.concatenate((Ytime, fp_data[site].time.values))

        if bc_freq == "monthly":
            Hmbc = setup.monthly_bcs(start_date, end_date, site, fp_data)
        elif bc_freq == None:
            Hmbc = fp_data[site].H_bc.values
        else:
            Hmbc = setup.create_bc_sensitivity(start_date, end_date, site, fp_data, bc_freq)

        if si == 0:
            Hbc = np.copy(Hmbc)  # fp_data[site].H_bc.values
            Hx = fp_data[site].H.values
        else:
            Hbc = np.hstack((Hbc, Hmbc))
            Hx = np.hstack((Hx, fp_data[site].H.values))

    sigma_freq_index = setup.sigma_freq_indicies(Ytime, sigma_freq)

    return Hx, Hbc, Y, Ytime, error, siteindicator, sigma_freq_index, fp_data
