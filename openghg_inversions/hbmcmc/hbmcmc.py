"""
Author: Atmospheric Chemistry Research Group, University of Bristol
Created: Nov.2022

About
  Originally created by Luke Western

  Modules for running an MCMC inversion using PyMC. There are also functions
  to dynamically create a basis function grid based on the a priori sensitivity,
  and some other functionality for setting up the inputs to this (or any)
  inverse method.

  If not using on an HPC in the terminal you should do:
    export OPENBLAS_NUM_THREADS=XX
 and/or
   export OMP_NUM_THREADS=XX
 where XX is the number of chains you are running.

 If running in Spyder do this before launching Spyder, else you will use every
 available thread. Apart from being annoying it will also slow down your run
 due to unnecessary forking.

 RHIME updated to use openghg as a dependency replacing (most) of the acrg
 modules previously used. See example input file for how input variables
 have chanegd.

 Note. RHIME with OpenGHG expects ALL data to already be included in the
 object stores and for the paths to object stores to already be set in
 the users .openghg config file

"""
from pathlib import Path
from typing import Union

import numpy as np
import openghg_inversions.hbmcmc.inversion_pymc as mcmc
import openghg_inversions.hbmcmc.inversionsetup as setup
from openghg_inversions import get_data, utils
from openghg_inversions.basis import basis_functions_wrapper


def fixedbasisMCMC(
    species,
    sites,
    domain,
    averaging_period,
    start_date,
    end_date,
    outputpath,
    outputname,
    bc_store="user",  # Do we want to set defaults for the object stores?
    obs_store="user",
    footprint_store="user",
    emissions_store="user",
    met_model=None,
    fp_model=None,  # Changed to none. When "NAME" specified FPs are not found
    fp_height=None,
    emissions_name=None,
    inlet=None,
    instrument=None,
    calibration_scale=None,
    obs_data_level=None,
    use_tracer=False,
    use_bc=True,
    fp_basis_case=None,
    basis_directory=None,
    bc_basis_case="NESW",
    bc_basis_directory=None,
    country_file=None,
    bc_input=None,
    basis_algorithm="weighted",
    nbasis=100,
    filters=[],
    xprior={"pdf": "truncatednormal", "mu": 1.0, "sigma": 1.0, "lower": 0.0},
    bcprior={"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.1, "lower": 0.0},
    sigprior={"pdf": "uniform", "lower": 0.1, "upper": 3},
    offsetprior={"pdf": "normal", "mu": 0, "sd": 1},
    nit=2.5e5,
    burn=50000,
    tune=1.25e5,
    nchain=2,
    fix_basis_outer_regions: bool = False,
    averaging_error=True,
    bc_freq=None,
    sigma_freq=None,
    sigma_per_site=True,
    country_unit_prefix=None,
    add_offset=False,
    verbose=False,
    reload_merged_data=False,
    save_merged_data=False,
    merged_data_dir=None,
    merged_data_name=None,
    basis_output_path=None,
    save_trace: Union[str, Path, bool] = False,
    skip_postprocessing: bool = False,
    merged_data_only: bool = False,
    **kwargs,
):
    """
    Script to run hierarchical Bayesian MCMC (RHIME) for inference
    of emissions using PyMC to solve the inverse problem.
    -----------------------------------------------------------------
    Args:
      species (str):
        Atmospheric trace gas species of interest (e.g. 'co2')

      sites (list):
        List of measurement site names

      domain (str):
        Model domain. (NB. Does not necessarily correspond to the inversion domain)

      averaging_period (list):
        Averaging period of observations (must match number of sites)

      start_date (str):
        Start time of inversion: "YYYY-mm-dd"

      end_date (str):
        End time of inversion: "YYYY-mm-dd"

      outputname (str):
        Unique identifier for output/run name

      outputpath (str):
        Path to where output should be saved

      bc_store (str):
        Name of object store containing boundary conditions files

      obs_store (str):
        Name of object store containing measurements files

      footprint_store (str):
        Name of object store containing footprints files

      emissions_store (str):
        Name of object store containing emissions/flux files

      met_model (str):
        Meteorological model used in the LPDM (e.g. 'ukv')

      fp_model (str):
        LPDM used for generating footprints (e.g. 'NAME')

      fp_height (list):
        Inlet height modelled for sites in LPDM (must match number of sites)

      emissions_name (list):
        List of keyword "source" args used for retrieving emissions files
        from 'emissions_store'.

      inlet (list, optional):
        Specific inlet height for the site (must match number of sites)

      instrument (str/list, optional):
        Specific instrument for the site (must match number of sites)

      calibration_scale (str):
        Calibration scale to use for measurements data

      obs_data_level (list):
        Data quality level for measurements data. (must match number of sites)

      use_tracer (bool):
        Option to use inverse model that uses tracers of species
        (e.g. d13C, CO, C2H4)

      fp_basis_case (str, optional):
        Name of basis function to use for emission

      basis_directory (str, optional):
        Directory containing the basis function

      bc_basis_case (str, optional):
        Name of basis case type for boundary conditions (NOTE, I don't
        think that currently you can do anything apart from scaling NSEW
        boundary conditions if you want to scale these monthly.)

      bc_basis_directory (str, optional):
        Directory containing the boundary condition basis functions
        (e.g. files starting with "NESW")

      bc_input (str):
        Variable for calling BC data from 'bc_store' - equivalent of
        'emissions_name' for fluxes

      country_file (str, optional):
        Path to the country definition file

      max_level (int, optional):
        The maximum level for a column measurement to be used for getting obs data

      basis_algorithm (str, optional):
        Select basis function algorithm for creating basis function file
        for emissions on the fly. Options include "quadtree" or "weighted".
        Defaults to "weighted" which distinguishes between land-sea regions

      nbasis (int):
        Number of basis functions that you want if using quadtree derived
        basis function. This will optimise to closest value that fits with
        quadtree splitting algorithm, i.e. nbasis % 4 = 1

      filters (list, optional):
        list of filters to apply from name.filtering. Defaults to empty list

      xprior (dict):
        Dictionary containing information about the prior PDF for emissions.
        The entry "pdf" is the name of the analytical PDF used, see
        https://docs.pymc.io/api/distributions/continuous.html for PDFs
        built into pymc3, although they may have to be coded into the script.
        The other entries in the dictionary should correspond to the shape
        parameters describing that PDF as the online documentation,
        e.g. N(1,1**2) would be: xprior={pdf:"normal", "mu":1, "sd":1}.
        Note that the standard deviation should be used rather than the
        precision. Currently all variables are considered iid

      bcprior (dict):
        Same as above but for boundary conditions.

      sigprior (dict):
        Same as above but for model error.

      offsetprior (dict):
        Same as above but for bias offset. Only used is addoffset=True.

      nit (int):
        Number of iterations for MCMC

      burn (int):
        Number of iterations to burn/discard in MCMC

      tune (int):
        Number of iterations to use to tune step size

      nchain (int):
        Number of independent chains to run (there is no way at all of
        knowing whether your distribution has converged by running only
        one chain)

      averaging_error (bool, optional):
        Adds the variability in the averaging period to the measurement
        error if set to True

      bc_freq (str, optional):
        The perdiod over which the baseline is estimated. Set to "monthly"
        to estimate per calendar month; set to a number of days,
        as e.g. "30D" for 30 days; or set to None to estimate to have one
        scaling for the whole inversion period

      sigma_freq (str, optional):
        as bc_freq, but for model sigma

      sigma_per_site (bool):
        Whether a model sigma value will be calculated for each site
        independantly (True) or all sites together (False)
        Default: True

      country_unit_prefix ('str', optional)
        A prefix for scaling the country emissions. Current options are:
       'T' will scale to Tg, 'G' to Gg, 'M' to Mg, 'P' to Pg.
        To add additional options add to convert.prefix
        Default is none and no scaling will be applied (output in g)

      add_offset (bool):
        Add an offset (intercept) to all sites but the first in the site list
        Default False

      reload_merged_data (bool):
        If True, reads fp_all object from a pickle file, instead of rerunning get_data

      save_merged_data (bool):
        If True, saves the merged data object (fp_all) as a pickle file

      merged_data_dir (str):
        Path to a directory of merged data objects. For saving to or reading from

      basis_output_path (Optional, str):
        If set, save the basis functions to this path. Used for testing

      save_trace: if True, save arviz `InferenceData` trace to `outputpath`. Alternatively,
        A file path (including file name and extension) can be passed, and the trace will be
        saved there.

      skip_post_processing: if True, return raw trace from sampling.

      merged_data_only: if True, save merged data, and do nothing else.

    Returns:
        Saves an output from the inversion code using inferpymc_postprocessouts.

    -----------------------------------------------------------------
    """
    rerun_merge = True

    if merged_data_only:
        reload_merged_data = False
        save_merged_data = True

    if reload_merged_data is True and merged_data_dir is not None:
        try:
            fp_all = get_data.load_merged_data(
                merged_data_dir, species, start_date, outputname, merged_data_name
            )
        except ValueError as e:
            # couldn't find merged data
            print(f"{e}, re-running data merge.")
        else:
            print("Successfully read in merged data.\n")
            rerun_merge = False

            # check if sites were dropped when merged data was saved
            sites_merged = [s for s in fp_all.keys() if "." not in s]

            if len(sites) != len(sites_merged):
                keep_i = [i for i, s in enumerate(sites) if s in sites_merged]
                s_dropped = [s for s in sites if s not in sites_merged]

                sites = [s for i, s in enumerate(sites) if i in keep_i]
                inlet = [s for i, s in enumerate(inlet) if i in keep_i]
                fp_height = [s for i, s in enumerate(fp_height) if i in keep_i]
                instrument = [s for i, s in enumerate(instrument) if i in keep_i]
                averaging_period = [s for i, s in enumerate(averaging_period) if i in keep_i]

                print(f"\nDropping {s_dropped} sites as they are not included in the merged data object.\n")

    if reload_merged_data is True and merged_data_dir is None:
        print("Cannot reload merged data without a value for `merged_data_dir`; re-running data merge.")

    # Get datasets for forward simulations
    if rerun_merge:
        if not use_tracer:
            (
                fp_all,
                sites,
                inlet,
                fp_height,
                instrument,
                averaging_period,
            ) = get_data.data_processing_surface_notracer(
                species,
                sites,
                domain,
                averaging_period,
                start_date,
                end_date,
                obs_data_level=obs_data_level,
                met_model=met_model,
                fp_model=fp_model,
                fp_height=fp_height,
                emissions_name=emissions_name,
                inlet=inlet,
                instrument=instrument,
                calibration_scale=calibration_scale,
                use_bc=use_bc,
                bc_input=bc_input,
                bc_store=bc_store,
                obs_store=obs_store,
                footprint_store=footprint_store,
                emissions_store=emissions_store,
                averagingerror=averaging_error,
                save_merged_data=save_merged_data,
                merged_data_name=merged_data_name,
                merged_data_dir=merged_data_dir,
                output_name = outputname,
            )

        elif use_tracer:
            raise ValueError("Model does not currently include tracer model. Watch this space")

        if merged_data_only:
            return None

    # Basis function regions and sensitivity matrices
    fp_data = basis_functions_wrapper(
        basis_algorithm=basis_algorithm,
        nbasis=nbasis,
        fp_basis_case=fp_basis_case,
        bc_basis_case=bc_basis_case,
        basis_directory=basis_directory,
        bc_basis_directory=bc_basis_directory,
        fp_all=fp_all,
        use_bc=use_bc,
        species=species,
        domain=domain,
        start_date=start_date,
        fix_outer_regions=fix_basis_outer_regions,
        emissions_name=emissions_name,
        outputname=outputname,
        output_path=basis_output_path,
    )

    # Apply named filters to the data
    fp_data = utils.filtering(fp_data, filters)

    for si, site in enumerate(sites):
        fp_data[site].attrs["Domain"] = domain

    # Inverse models
    if use_tracer is False:
        # Get inputs ready
        error = np.zeros(0)
        Hx = np.zeros(0)
        Y = np.zeros(0)
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

            if si == 0:
                Hx = fp_data[site].H.values
            else:
                Hx = np.hstack((Hx, fp_data[site].H.values))

        sigma_freq_index = setup.sigma_freq_indicies(Ytime, sigma_freq)

        # Path to save trace
        if isinstance(save_trace, (str, Path)):
            trace_path = save_trace
        elif save_trace is True:
            trace_path = Path(outputpath) / (outputname + f"{start_date}_trace.nc")
        else:
            trace_path = None

        mcmc_args = {
            "Hx": Hx,
            "Y": Y,
            "error": error,
            "siteindicator": siteindicator,
            "sigma_freq_index": sigma_freq_index,
            "xprior": xprior,
            "sigprior": sigprior,
            "nit": nit,
            "burn": burn,
            "tune": tune,
            "nchain": nchain,
            "sigma_per_site": sigma_per_site,
            "offsetprior": offsetprior,
            "add_offset": add_offset,
            "verbose": verbose,
            "save_trace": trace_path,
        }

        if use_bc is True:
            Hbc = np.zeros(0)

            for si, site in enumerate(sites):
                if bc_freq == "monthly":
                    Hmbc = setup.monthly_bcs(start_date, end_date, site, fp_data)
                elif bc_freq is None:
                    Hmbc = fp_data[site].H_bc.values
                else:
                    Hmbc = setup.create_bc_sensitivity(start_date, end_date, site, fp_data, bc_freq)

                if si == 0:
                    Hbc = np.copy(Hmbc)  # fp_data[site].H_bc.values
                else:
                    Hbc = np.hstack((Hbc, Hmbc))

            mcmc_args["Hbc"] = Hbc
            mcmc_args["bcprior"] = bcprior
            mcmc_args["use_bc"] = True
        else:
            mcmc_args["use_bc"] = False

        post_process_args = {
            "Ytime": Ytime,
            "domain": domain,
            "species": species,
            "sites": sites,
            "start_date": start_date,
            "end_date": end_date,
            "outputname": outputname,
            "outputpath": outputpath,
            "country_unit_prefix": country_unit_prefix,
            "fp_data": fp_data,
            "emissions_name": emissions_name,
            "emissions_store": emissions_store,
            "country_file": country_file,
        }

        # add mcmc_args to post_process_args
        # and delete a few we don't need
        post_process_args.update(mcmc_args)
        del post_process_args["nit"]
        del post_process_args["verbose"]
        del post_process_args["save_trace"]

        # add any additional kwargs to mcmc_args (these aren't needed for post processing)
        mcmc_args.update(kwargs)

        # Run PyMC inversion
        mcmc_results = mcmc.inferpymc(**mcmc_args)

        if skip_postprocessing:
            return mcmc_results

        # Process and save inversion output
        post_process_args.update(mcmc_results)
        out = mcmc.inferpymc_postprocessouts(**post_process_args)

    elif use_tracer:
        raise ValueError("Model does not currently include tracer model. Watch this space")

    print("---- Inversion completed ----")

    return out


def rerun_output(input_file, outputname, outputpath, verbose=False):
    """
    Rerun the MCMC code by taking the inputs from a previous output
    using this code and rewrite a new output. This allows reproducibility
    of results without the need to transfer all raw input files.
    -----------------------------------
    Args:
      input_file (str):
        Full path to previously written ncdf file
      outputname (list):
        Unique identifier new for output/run name.
      outputpath (str):
        Path to where output should be saved.

    Returns:
      Saves an output from the inversion code using inferpymc3_postprocessouts.

    Note: At the moment fluxapriori in the output is the mean apriori flux
          over the inversion period and so will not be identical to the
          original a priori flux, if it varies over the inversion period.
    -----------------------------------
    """

    def isFloat(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    ds_in = setup.opends(input_file)

    # Read inputs from ncdf output
    start_date = ds_in.attrs["Start date"]
    end_date = ds_in.attrs["End date"]
    Hx = ds_in.xsensitivity.values.T
    Hbc = ds_in.bcsensitivity.values.T
    Y = ds_in.Yobs.values
    Ytime = ds_in.Ytime.values
    error = ds_in.Yerror.values
    siteindicator = ds_in.siteindicator.values
    sigma_freq_index = ds_in.sigmafreqindex.values
    xprior_string = ds_in.attrs["Emissions Prior"].split(",")
    xprior = {k: float(v) if isFloat(v) else v for k, v in zip(xprior_string[::2], xprior_string[1::2])}
    bcprior_string = ds_in.attrs["BCs Prior"].split(",")
    bcprior = {k: float(v) if isFloat(v) else v for k, v in zip(bcprior_string[::2], bcprior_string[1::2])}
    sigprior_string = ds_in.attrs["Model error Prior"].split(",")
    sigprior = {k: float(v) if isFloat(v) else v for k, v in zip(sigprior_string[::2], sigprior_string[1::2])}
    if "Offset Prior" in ds_in.attrs.keys():
        offsetprior_string = ds_in.attrs["Offset Prior"].split(",")
        offsetprior = {
            k: float(v) if isFloat(v) else v
            for k, v in zip(offsetprior_string[::2], offsetprior_string[1::2])
        }
        add_offset = True
    else:
        add_offset = False
        offsetprior = None
    nit = len(ds_in.steps)
    burn = int(ds_in.attrs["Burn in"])
    tune = int(ds_in.attrs["Tuning steps"])
    nchain = int(ds_in.attrs["Number of chains"])
    if ds_in.attrs["Error for each site"] == "True":
        sigma_per_site = True
    else:
        sigma_per_site = False
    sites = ds_in.sitenames.values

    file_list = input_file.split("/")[-1].split("_")
    species = file_list[0]
    domain = file_list[1]
    if ds_in.countrymean.attrs["units"] != "g":
        country_unit_prefix = ds_in.countrymean.attrs["units"][0]
    else:
        country_unit_prefix = None

    (
        xouts,
        bcouts,
        sigouts,
        Ytrace,
        YBCtrace,
        convergence,
        step1,
        step2,
    ) = mcmc.inferpymc(
        Hx,
        Hbc,
        Y,
        error,
        siteindicator,
        sigma_freq_index,
        xprior,
        bcprior,
        sigprior,
        nit,
        burn,
        tune,
        nchain,
        sigma_per_site,
        offsetprior=offsetprior,
        add_offset=add_offset,
        verbose=verbose,
    )

    mcmc.inferpymc_postprocessouts(
        xouts,
        bcouts,
        sigouts,
        convergence,
        Hx,
        Hbc,
        Y,
        error,
        Ytrace,
        YBCtrace,
        step1,
        step2,
        xprior,
        bcprior,
        sigprior,
        offsetprior,
        Ytime,
        siteindicator,
        sigma_freq_index,
        domain,
        species,
        sites,
        start_date,
        end_date,
        outputname,
        outputpath,
        country_unit_prefix,
        burn,
        tune,
        nchain,
        sigma_per_site,
        add_offset=add_offset,
        rerun_file=ds_in,
    )
