"""Contains functions for running all steps of the MCMC inversion using PyMC:
getting data, filtering, applying basis functions, sampling, and processing
the outputs.

If not using on an HPC in the terminal you should do:
export OPENBLAS_NUM_THREADS=XX
and/or
export OMP_NUM_THREADS=XX
where XX is the number of chains you are running.

If running in Spyder do this before launching Spyder, else you will use every
available thread. Apart from being annoying it will also slow down your run
due to unnecessary forking.

Note. RHIME with OpenGHG expects ALL data to already be included in the
object stores and for the paths to object stores to already be set in
the users OpenGHG config file (default location: ~/.openghg/openghg.conf).
"""

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

import openghg_inversions.hbmcmc.inversion_pymc as mcmc
import openghg_inversions.hbmcmc.inversionsetup as setup
from openghg_inversions import get_data
from openghg_inversions.basis import basis_functions_wrapper
from openghg_inversions.filters import filtering
from openghg_inversions.model_error import residual_error_method, percentile_error_method, setup_min_error


def fixedbasisMCMC(
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: list[str],
    start_date: str,
    end_date: str,
    outputpath: str,
    outputname: str,
    bc_store: str = "user",  # Do we want to set defaults for the object stores?
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: list | None = None,
    fp_model: str | None = None,  # Changed to none. When "NAME" specified FPs are not found
    fp_height: list[str] | None = None,
    fp_species: str | None = None,
    emissions_name: list[str] | None = None,
    inlet: list[str] | None = None,
    instrument: list[str] | None = None,
    calibration_scale: str | None = None,
    obs_data_level: list | None = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    fp_basis_case: str | None = None,
    basis_directory: str | None = None,
    bc_basis_case: str = "NESW",
    bc_basis_directory: str | None = None,
    country_file: str | None = None,
    bc_input: str | None = None,
    basis_algorithm: str = "weighted",
    nbasis: int = 100,
    xprior: dict = {"pdf": "truncatednormal", "mu": 1.0, "sigma": 1.0, "lower": 0.0},
    bcprior: dict = {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.1, "lower": 0.0},
    sigprior: dict = {"pdf": "uniform", "lower": 0.1, "upper": 3},
    offsetprior: dict = {"pdf": "normal", "mu": 0, "sd": 1},
    nit: int = int(2.5e5),
    burn: int = 50000,
    tune: int = int(1.25e5),
    nchain: int = 2,
    filters: None | list | dict[str, list[str] | None] = None,
    fix_basis_outer_regions: bool = False,
    averaging_error: bool = True,
    bc_freq: str | None = None,
    sigma_freq: str | None = None,
    sigma_per_site: bool = True,
    country_unit_prefix: str | None = None,
    add_offset: bool = False,
    verbose: bool = False,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    basis_output_path: str | None = None,
    save_trace: str | Path | bool = False,
    skip_postprocessing: bool = False,
    merged_data_only: bool = False,
    calculate_min_error: Literal["percentile", "residual"] | None = None,
    min_error_options: dict | None = None,
    return_inv_out: bool = False,  # for testing new postprocessing
    new_postprocessing: bool = False,  # for testing new postprocessing
    paris_postprocessing: bool = False,
    paris_postprocessing_kwargs: dict | None = None,
    **kwargs,
) -> xr.Dataset | dict:
    """Script to run hierarchical Bayesian MCMC (RHIME) for inference
    of emissions using PyMC to solve the inverse problem.
    Saves an output from the inversion code using inferpymc_postprocessouts.

    Args:
      species:
        Atmospheric trace gas species of interest (e.g. 'co2')
      sites:
        List of measurement site names
      domain:
        Model domain. (NB. Does not necessarily correspond to the inversion domain)
      averaging_period:
        Averaging period of observations (must match number of sites)
      start_date:
        Start time of inversion: "YYYY-mm-dd"
      end_date:
        End time of inversion: "YYYY-mm-dd"
      outputname:
        Unique identifier for output/run name
      outputpath:
        Path to where output should be saved
      bc_store:
        Name of object store containing boundary conditions files
      obs_store:
        Name of object store containing measurements files
      footprint_store:
        Name of object store containing footprints files
      emissions_store:
        Name of object store containing emissions/flux files
      met_model:
        Meteorological model used in the LPDM (e.g. 'ukv')
      fp_model:
        LPDM used for generating footprints (e.g. 'NAME')
      fp_height:
        Inlet height modelled for sites in LPDM (must match number of sites)
      fp_species:
        Species name associated with footprints in the object store
      emissions_name:
        List of keyword "source" args used for retrieving emissions files
        from 'emissions_store'.
      inlet:
        Specific inlet height for the site (must match number of sites)
      instrument:
        Specific instrument for the site (must match number of sites)
      calibration_scale:
        Calibration scale to use for measurements data
      obs_data_level:
        Data quality level for measurements data. (must match number of sites)
      use_tracer:
        Option to use inverse model that uses tracers of species
        (e.g. d13C, CO, C2H4)
      use_bc:
        When True, use and infer boundary conditions.
      fp_basis_case:
        Name of basis function to use for emission
      basis_directory:
        Directory containing the basis function
      bc_basis_case:
        Name of basis case type for boundary conditions (NOTE, I don't
        think that currently you can do anything apart from scaling NSEW
        boundary conditions if you want to scale these monthly.)
      bc_basis_directory:
        Directory containing the boundary condition basis functions
        (e.g. files starting with "NESW")
      country_file:
        Path to the country definition file
      bc_input:
        Variable for calling BC data from 'bc_store' - equivalent of
        'emissions_name' for fluxes
      basis_algorithm:
        Select basis function algorithm for creating basis function file
        for emissions on the fly. Options include "quadtree" or "weighted".
        Defaults to "weighted" which distinguishes between land-sea regions
      nbasis:
        Number of basis functions that you want if using quadtree derived
        basis function. This will optimise to closest value that fits with
        quadtree splitting algorithm, i.e. nbasis % 4 = 1
      xprior:
        Dictionary containing information about the prior PDF for emissions.
        The entry "pdf" is the name of the analytical PDF used, see
        https://docs.pymc.io/api/distributions/continuous.html for PDFs
        built into pymc3, although they may have to be coded into the script.
        The other entries in the dictionary should correspond to the shape
        parameters describing that PDF as the online documentation,
        e.g. N(1,1**2) would be: xprior={pdf:"normal", "mu":1, "sd":1}.
        Note that the standard deviation should be used rather than the
        precision. Currently all variables are considered iid
      bcprior:
        Same as xrior but for boundary conditions.
      sigprior:
        Same as xrior but for model error.
      offsetprior:
        Same as xrior but for bias offset. Only used is addoffset=True.
      nit:
        Number of iterations for MCMC
      burn:
        Number of iterations to burn/discard in MCMC
      tune:
        Number of iterations to use to tune step size
      nchain:
        Number of independent chains to run (there is no way at all of
        knowing whether your distribution has converged by running only
        one chain)
      filters (list, or dictionary of lists, optional):
        list of filters to apply to all sites, or dictionary with sites as keys
        and a list of filters for each site, e.g. filters = {"MHD": ["pblh_inlet_diff", "pblh_min"], "JFJ": None}
      fix_basis_outer_regions:
        When set to True uses InTEM regions to derive basis functions for inner region
        Default False
      averaging_error:
        Adds the variability in the averaging period to the measurement
        error if set to True
      bc_freq:
        The perdiod over which the baseline is estimated. Set to "monthly"
        to estimate per calendar month; set to a number of days,
        as e.g. "30D" for 30 days; or set to None to estimate to have one
        scaling for the whole inversion period
      sigma_freq:
        as bc_freq, but for model sigma
      sigma_per_site:
        Whether a model sigma value will be calculated for each site
        independantly (True) or all sites together (False)
        Default: True
      country_unit_prefix ('str', optional)
        A prefix for scaling the country emissions. Current options are:
       'T' will scale to Tg, 'G' to Gg, 'M' to Mg, 'P' to Pg.
        To add additional options add to convert.prefix
        Default is none and no scaling will be applied (output in g)
      add_offset:
        Add an offset (intercept) to all sites but the first in the site list
        Default False
      verbose:
        When True, prints progress bar of mcmc.inferpymc
      reload_merged_data:
        If True, reads fp_all object from a pickle file, instead of rerunning get_data
      save_merged_data:
        If True, saves the merged data object (fp_all) as a pickle file
      merged_data_dir:
        Path to a directory of merged data objects. For saving to or reading from
      merged_data_name:
        Name of files in which are the merged data objects. For saving to or reading from
      basis_output_path:
        If set, save the basis functions to this path. Used for testing
      save_trace:
        If True, save arviz `InferenceData` trace to `outputpath`. Alternatively,
        A file path (including file name and extension) can be passed, and the trace will be
        saved there.
      skip_post_processing:
        If True, return raw trace from sampling.
      merged_data_only:
        If True, save merged data, and do nothing else.
      calculate_min_error:
        If None, use value in `kwargs[min_error]`. Otherwise, compute min model error
        using the "residual" method or the "percentile" method. (See `openghg_inversions.model_error.py` for
        details.)
      min_error_options:
        Dictionary of additional arguments to pass the the function used to calculate min. model
        error (as specified by `calculate_min_error`).

    Return:
      Results from the inversion in a Dataset if skip_post_processing==False, in a dictionnary if True
    """
    rerun_merge = True

    if merged_data_only:
        reload_merged_data = False

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
            sites_merged = [s for s in fp_all if "." not in s]

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
                fp_species=fp_species,
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
                output_name=outputname,
            )

        elif use_tracer:
            raise ValueError("Model does not currently include tracer model. Watch this space")

        if merged_data_only:
            return fp_all # type: ignore

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
    if filters is not None:
        fp_data = filtering(fp_data, filters)

    # check for sites dropped by filtering
    s_dropped = []
    for site in sites:
        # check if some datasets are empty due to filtering
        if fp_data[site].time.values.shape[0] == 0:
            s_dropped.append(site)
            del fp_data[site]

    if len(s_dropped) != 0:
        sites = [s for i, s in enumerate(sites) if s not in s_dropped]
        print(f"\nDropping {s_dropped} sites as no data passed the filtering.\n")

    for si, site in enumerate(sites):
        fp_data[site].attrs["Domain"] = domain

    # Inverse models
    if use_tracer is False:
        # Get inputs ready
        error = np.zeros(0)
        obs_repeatability = np.zeros(0)
        obs_variability = np.zeros(0)
        Hx = np.zeros(0)
        Y = np.zeros(0)
        siteindicator = np.zeros(0)

        for si, site in enumerate(sites):
            # select variables to drop NaNs from
            drop_vars = []
            for var in ["H", "H_bc", "mf", "mf_error", "mf_variability", "mf_repeatability"]:
                if var in fp_data[site].data_vars:
                    drop_vars.append(var)

            # pymc doesn't like NaNs, so drop them for the variables used below
            fp_data[site] = fp_data[site].dropna("time", subset=drop_vars)

            # repeatability/variability chosen/combined into mf_error in `get_data.py`
            error = np.concatenate((error, fp_data[site].mf_error.values))

            # make repeatability and variability for outputs (not used directly in inversions)
            obs_repeatability = np.concatenate((obs_repeatability, fp_data[site].mf_repeatability.values))
            obs_variability = np.concatenate((obs_variability, fp_data[site].mf_variability.values))

            Y = np.concatenate((Y, fp_data[site].mf.values))
            siteindicator = np.concatenate((siteindicator, np.ones_like(fp_data[site].mf.values) * si))
            if si == 0:
                Ytime = fp_data[site].time.values
            else:
                Ytime = np.concatenate((Ytime, fp_data[site].time.values))

            Hx = fp_data[site].H.values if si == 0 else np.hstack((Hx, fp_data[site].H.values))

        # Calculate min error
        if calculate_min_error == "residual":
            if min_error_options is not None:
                min_error = residual_error_method(fp_data, **min_error_options)
            else:
                min_error = residual_error_method(fp_data)

            # if "by_site" is True, align min_error via siteindicator
            if min_error_options and min_error_options.get("by_site", False):
                min_error = setup_min_error(min_error, siteindicator)

            kwargs["min_error"] = min_error  # currently `min_error` is passed via kwargs to `infer_pymc`
        elif calculate_min_error == "percentile":
            min_error = percentile_error_method(fp_data)
            min_error = setup_min_error(min_error, siteindicator)
            kwargs["min_error"] = min_error  # currently `min_error` is passed via kwargs to `infer_pymc`

        elif calculate_min_error is None:
            pass
        else:
            raise ValueError(
                "`calculate_min_error` must have values: 'residual', 'percentile', or `None`;"
                f" {calculate_min_error} not recognised."
            )

        sigma_freq_index = setup.sigma_freq_indicies(Ytime, sigma_freq)


        # check if lognormal mu and sigma need to be calculated
        def update_log_normal_prior(prior):
            if prior["pdf"].lower() == "lognormal" and "stdev" in prior:
                stdev = float(prior["stdev"])
                mean = float(prior.get("mean", 1.0))

                mu, sigma = mcmc.lognormal_mu_sigma(mean, stdev)
                prior["mu"] = mu
                prior["sigma"] = sigma

                del prior["stdev"]
                if "mean" in prior:
                    del prior["mean"]

        update_log_normal_prior(xprior)
        update_log_normal_prior(bcprior)

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
            "country_file": country_file,
            "obs_repeatability": obs_repeatability,
            "obs_variability": obs_variability,
        }

        # add mcmc_args to post_process_args
        # and delete a few we don't need
        post_process_args.update(mcmc_args)
        del post_process_args["nit"]
        del post_process_args["verbose"]

        # pass min model error to post-processing
        post_process_args["min_error"] = kwargs.get("min_error", 0.0)

        # add any additional kwargs to mcmc_args (these aren't needed for post processing)
        mcmc_args.update(kwargs)

        # Run PyMC inversion
        mcmc_results = mcmc.inferpymc(**mcmc_args)  # type: ignore

        if skip_postprocessing:
            return mcmc_results

        if return_inv_out:
            from ..postprocessing.inversion_output import make_inv_out

            return make_inv_out(
                fp_data=fp_data,
                Y=Y,
                Ytime=Ytime,
                error=error,
                obs_repeatability=obs_repeatability,
                obs_variability=obs_variability,
                site_indicator=siteindicator,
                site_names=sites,
                mcmc_results=mcmc_results,
                start_date=start_date,
                end_date=end_date,
                species=species,
                domain=domain,
            )


        if new_postprocessing:
            from ..postprocessing.inversion_output import make_inv_out
            from ..postprocessing.make_outputs import basic_output

            inv_out = make_inv_out(
                fp_data=fp_data,
                Y=Y,
                Ytime=Ytime,
                error=error,
                obs_repeatability=obs_repeatability,
                obs_variability=obs_variability,
                site_indicator=siteindicator,
                site_names=sites,
                mcmc_results=mcmc_results,
                start_date=start_date,
                end_date=end_date,
                species=species,
                domain=domain,
            )

            outputs = basic_output(inv_out, country_file=country_file)
            return outputs

        if paris_postprocessing:
            from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename
            from openghg_inversions.postprocessing.inversion_output import make_inv_out
            from openghg_inversions.postprocessing.make_paris_outputs import make_paris_outputs

            inv_out = make_inv_out(
                fp_data=fp_data,
                Y=Y,
                Ytime=Ytime,
                error=error,
                obs_repeatability=obs_repeatability,
                obs_variability=obs_variability,
                site_indicator=siteindicator,
                site_names=sites,
                mcmc_results=mcmc_results,
                start_date=start_date,
                end_date=end_date,
                species=species,
                domain=domain,
            )

            obs_avg_period = averaging_period[0] or "1h"
            paris_postprocessing_kwargs = paris_postprocessing_kwargs or {}
            flux_outs, conc_outs = make_paris_outputs(inv_out, country_file=country_file, obs_avg_period=obs_avg_period, **paris_postprocessing_kwargs)

            conc_output_filename = define_output_filename(outputpath, species, domain, outputname + "_conc", start_date, ext=".nc")
            flux_output_filename = define_output_filename(outputpath, species, domain, outputname + "_flux", start_date, ext=".nc")
            Path(outputpath).mkdir(parents=True, exist_ok=True)

            conc_outs.to_netcdf(conc_output_filename, unlimited_dims=["time"], mode="w")
            flux_outs.to_netcdf(flux_output_filename, unlimited_dims=["time"], mode="w")

            return xr.merge([conc_outs, flux_outs.rename(time="flux_time")])

        # get trace and model: for future updates
        trace = mcmc_results.pop("trace")
        model = mcmc_results.pop("model")

        # Path to save trace
        if save_trace:
            if isinstance(save_trace, str | Path):
                trace_path = save_trace
            else:
                trace_path = Path(outputpath) / (outputname + f"{start_date}_trace.nc")

            trace.to_netcdf(str(trace_path), engine="netcdf4")

        # Process and save inversion output
        post_process_args.update(mcmc_results)
        out = mcmc.inferpymc_postprocessouts(**post_process_args)

    elif use_tracer:
        raise ValueError("Model does not currently include tracer model. Watch this space")

    print("---- Inversion completed ----")

    return out


def rerun_output(input_file: str, outputname: str, outputpath: str, verbose: bool = False) -> None:
    """Rerun the MCMC code by taking the inputs from a previous output
    using this code and rewrite a new output. This allows reproducibility
    of results without the need to transfer all raw input files.

    Args:
      input_file:
        Full path to previously written ncdf file
      outputname:
        Unique identifier new for output/run name.
      outputpath:
        Path to where output should be saved.
      verbose:
        When True, prints progress bar of mcmc.inferpymc

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

    ds_in = xr.load_dataset(input_file)

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
    if "Offset Prior" in ds_in.attrs:
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
    sigma_per_site = ds_in.attrs["Error for each site"] == "True"
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
        Hx=Hx,
        Hbc=Hbc,
        Y=Y,
        error=error,
        siteindicator=siteindicator,
        sigma_freq_index=sigma_freq_index,
        xprior=xprior,
        bcprior=bcprior,
        sigprior=sigprior,
        nit=nit,
        burn=burn,
        tune=tune,
        nchain=nchain,
        sigma_per_site=sigma_per_site,
        offsetprior=offsetprior,
        add_offset=add_offset,
        verbose=verbose,
    )

    mcmc.inferpymc_postprocessouts(
        xouts=xouts,
        bcouts=bcouts,
        sigouts=sigouts,
        convergence=convergence,
        Hx=Hx,
        Hbc=Hbc,
        Y=Y,
        error=error,
        Ytrace=Ytrace,
        YBCtrace=YBCtrace,
        step1=step1,
        step2=step2,
        xprior=xprior,
        bcprior=bcprior,
        sigprior=sigprior,
        offsetprior=offsetprior,
        Ytime=Ytime,
        siteindicator=siteindicator,
        sigma_freq_index=sigma_freq_index,
        domain=domain,
        species=species,
        sites=sites,
        start_date=start_date,
        end_date=end_date,
        outputname=outputname,
        outputpath=outputpath,
        country_unit_prefix=country_unit_prefix,
        burn=burn,
        tune=tune,
        nchain=nchain,
        sigma_per_site=sigma_per_site,
        add_offset=add_offset,
        rerun_file=ds_in,
    )
