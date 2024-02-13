# *****************************************************************************
# inversion_pymc.py
# Created: 7 Nov. 2022
# Author: Atmospheric Chemistry Research Group, University of Bristol
# *****************************************************************************
# About
#   Functions for performing RHIME inversion.
# *****************************************************************************

import re
import sys
import numpy as np
import pymc as pm
import pandas as pd
import xarray as xr
import getpass
from scipy import stats
from pathlib import Path
from typing import Optional, Union


from openghg_inversions import convert
from openghg_inversions import utils
from openghg_inversions.hbmcmc.inversionsetup import offset_matrix
from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename
from openghg_inversions.config.version import code_version


def parseprior(name, prior_params, shape=()):
    """
    Parses all continuous distributions for PyMC 3.8:
    https://docs.pymc.io/api/distributions/continuous.html
    This format requires updating when the PyMC distributions update,
    but is safest for code execution
    -----------------------------------
    Args:
      name (str):
        name of variable in the pymc model
      prior_params (dict):
        dict of parameters for the distribution,
        including 'pdf' for the distribution to use
      shape (array):
        shape of distribution to be created.
        Default shape = () is the same as used by PyMC3
    -----------------------------------
    """
    functiondict = {
        "uniform": pm.Uniform,
        "flat": pm.Flat,
        "halfflat": pm.HalfFlat,
        "normal": pm.Normal,
        "truncatednormal": pm.TruncatedNormal,
        "halfnormal": pm.HalfNormal,
        "skewnormal": pm.SkewNormal,
        "beta": pm.Beta,
        "kumaraswamy": pm.Kumaraswamy,
        "exponential": pm.Exponential,
        "laplace": pm.Laplace,
        "studentt": pm.StudentT,
        "halfstudentt": pm.HalfStudentT,
        "cauchy": pm.Cauchy,
        "halfcauchy": pm.HalfCauchy,
        "gamma": pm.Gamma,
        "inversegamma": pm.InverseGamma,
        "weibull": pm.Weibull,
        "lognormal": pm.Lognormal,
        "chisquared": pm.ChiSquared,
        "wald": pm.Wald,
        "pareto": pm.Pareto,
        "exgaussian": pm.ExGaussian,
        "vonmises": pm.VonMises,
        "triangular": pm.Triangular,
        "gumbel": pm.Gumbel,
        "rice": pm.Rice,
        "logistic": pm.Logistic,
        "logitnormal": pm.LogitNormal,
        "interpolated": pm.Interpolated,
    }

    pdf = prior_params["pdf"]
    # Get a dictionary of the pdf arguments
    params = {x: prior_params[x] for x in prior_params if x != "pdf"}
    return functiondict[pdf.lower()](name, shape=shape, **params)


def inferpymc(
    Hx,
    Hbc,
    basis_region_mask,
    Y,
    error,
    siteindicator,
    sigma_freq_index,
    xprior={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
    bcprior={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
    sigprior={"pdf": "uniform", "lower": 0.1, "upper": 3.0},
    nit=2.5e5,
    burn=50000,
    tune=1.25e5,
    nchain=2,
    sigma_per_site=True,
    offsetprior={"pdf": "normal", "mu": 0, "sigma": 1},
    add_offset=False,
    verbose=False,
    min_model_error=0.0,
    save_trace=False,
):
    """
    Uses PyMC module for Bayesian inference for emissions field, boundary
    conditions and (currently) a single model error value.
    This uses a Normal likelihood but the (hyper)prior PDFs can selected by user.
    -----------------------------------
    Args:
      Hx (array):
        Transpose of the sensitivity matrix to map emissions to measurement.
        This is the same as what is given from fp_data[site].H.values, where
        fp_data is the output from e.g. footprint_data_merge, but where it
        has been stacked for all sites.
      Hbc (array):
        Same as above but for boundary conditions
      basis_region_mask (array):
        Mask to indicate which Hx elements are for each flux sector
      Y (array):
        Measurement vector containing all measurements
      error (arrray):
        Measurement error vector, containg a value for each element of Y.
      siteindicator (array):
        Array of indexing integers that relate each measurement to a site
      sigma_freq_index (array):
        Array of integer indexes that converts time into periods
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
      sigma_per_site (bool):
        Whether a model sigma value will be calculated for each site independantly (True) or all sites together (False).
        Default: True
      add_offset (bool):
        Add an offset (intercept) to all sites but the first in the site list. Default False.
      min_model_error (float):
        Minimum model error to impose on species baseline 
      verbose:
        When True, prints progress bar

    Returns:
      outs (array):
        MCMC chain for emissions scaling factors for each basis function.
      bcouts (array):
        MCMC chain for boundary condition scaling factors.
      sigouts (array):
        MCMC chain for model error.
      Ytrace (array):
        MCMC chain for modelled obs.
      YBCtrace (array):
        MCMC chain for modelled boundary condition.
      convergence (str):
        Passed/Failed convergence test as to whether mutliple chains
        have a Gelman-Rubin diagnostic value <1.05
      step1 (str):
        Type of MCMC sampler for emissions and boundary condition updates.
        Currently it's hardwired to NUTS (probably wouldn't change this
        unless you're doing something obscure).
      step2 (str):
        Type of MCMC sampler for model error updates.
        Currently it's hardwired to a slice sampler. This parameter is low
        dimensional and quite simple with a slice sampler, although could
        easily be changed.

    TO DO:
       - Allow non-iid variables
    -----------------------------------
    """
    burn = int(burn)

    hx = Hx.T
    hbc = Hbc.T
    nx = hx.shape[1]
    nbc = hbc.shape[1]
    ny = len(Y)

    nit = int(nit)
    nflux = len(list(xprior.keys()))

    # convert siteindicator into a site indexer
    if sigma_per_site:
        sites = siteindicator.astype(int)
        nsites = np.amax(sites) + 1
    else:
        sites = np.zeros_like(siteindicator).astype(int)
        nsites = 1
    nsigmas = np.amax(sigma_freq_index) + 1

    if add_offset:
        B = offset_matrix(siteindicator)

    with pm.Model() as model:
        xbc = parseprior("xbc", bcprior, shape=nbc)
        sig = parseprior("sig", sigprior, shape=(nsites, nsigmas))

        x_conc = []
        #hx_dot_x = []
        for i, key in enumerate(xprior.keys()):
            sector_len = len(np.squeeze(np.where(basis_region_mask == i + 1)))
            sector_mask = np.squeeze(np.where(basis_region_mask == i + 1))
            print(key, "has ", sector_len, " basis functions")
            x = parseprior(key, xprior[key], shape=sector_len)
            x_conc.append(x)
            #hx_dot_x.append([pm.math.dot(hx[:, sector_mask], x)])
            if i == 0:
                hx_dot_x = pm.math.dot(hx[:, sector_mask], x)
            else:
                hx_dot_x += pm.math.dot(hx[:, sector_mask], x)
          
        #CHANGES: removed .concatenate lines and replaced with sums, as 
        # dimensions of modelled mf weren't correct
        # or, could add the .concatenate back in and use a .sum too:
        # mu = pm.math.sum(pm.math.concatenate(hx_dot_x),axis=0)

        if add_offset:
            offset = parseprior("offset", offsetprior, shape=nsites - 1)
            offset_vec = pm.math.concatenate((np.array([0]), offset), axis=0)
            #mu = pm.math.concatenate(hx_dot_x) + pm.math.dot(hbc, xbc) + pm.math.dot(B, offset_vec) 
            mu = hx_dot_x + pm.math.dot(hbc, xbc) + pm.math.dot(B, offset_vec) 

        else:
            #mu = pm.math.concatenate(hx_dot_x) + pm.math.dot(hbc, xbc)
            mu = hx_dot_x + pm.math.dot(hbc, xbc)

        # Calculate model error
        #model_error = pm.math.abs(pm.math.concatenate(hx_dot_x)) * sig[sites, sigma_freq_index]
        model_error = pm.math.abs(hx_dot_x) * sig[sites, sigma_freq_index]
                
        epsilon = pm.math.sqrt(error**2 + model_error**2 + min_model_error**2)
        y = pm.Normal("y", mu=mu, sigma=epsilon, observed=Y, shape=ny)

        # Append BCs
        x_conc.append(xbc)
        
        step1 = pm.NUTS(vars=x_conc)
        step2 = pm.Slice(vars=[sig])
        
        trace = pm.sample(nit, 
                          tune=int(tune), 
                          chains=nchain,
                          step=[step1, step2], 
                          progressbar=verbose, 
                          cores=nchain) #step=pm.Metropolis())#  #target_accept=0.8,

        # Collate trace outputs for each sector. Keep first chain only.
        outs = {}
        for key in xprior.keys():
            outs[key] = trace.posterior[key][0, burn:nit]       
 
        bcouts = trace.posterior["xbc"][0, burn:nit]
        sigouts = trace.posterior["sig"][0, burn:nit]

        # Check for convergence
        convergence = {}
        for key in xprior.keys():
            gelrub = pm.rhat(trace)[key].max()
            if gelrub > 1.05:
                print(f"{key} Failed Gelman-Rubin at 1.05")
                convergence[key] = "Failed"
            else:
                convergence[key] = "Passed"

        if add_offset:
            offsetouts = trace.posterior["offset"][0, burn:nit]
            offset_trace = np.hstack([np.zeros((int(nit - burn), 1)), offsetouts])
            YBCtrace = np.dot(Hbc.T, bcouts.T) + np.dot(B, offset_trace.T)
            OFFSETtrace = np.dot(B, offset_trace.T)   
        else:
            YBCtrace = np.dot(Hbc.T, bcouts.T)
            offsetouts = outs[key] * 0 
            OFFSETtrace =  YBCtrace * 0
 
        hx_dot_x = []
        for i, key in enumerate(xprior.keys()):
            sector_mask = np.squeeze(np.where(basis_region_mask == i + 1)) 
            hx_sector = hx[:, sector_mask]
            hx_dot_x.append([np.dot(hx_sector, outs[key].values.T)])

        Ytrace = np.squeeze(np.sum(np.array(hx_dot_x), axis=0) + YBCtrace) 

        #if save_trace:
        #    trace.to_netcdf(str(save_trace), engine="netcdf4")
 
        return outs, bcouts, sigouts, offsetouts, Ytrace, YBCtrace, OFFSETtrace, convergence, step1, step2


def inferpymc_postprocessouts(
    xouts,
    bcouts,
    sigouts,
    offsetouts,
    convergence,
    Hx,
    Hbc,
    Y,
    error,
    Ytrace,
    YBCtrace,
    OFFSETtrace,
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
    emissions_name,
    emissions_store,
    fp_data=None,
    basis_directory=None,
    country_file=None,
    add_offset=False,
    rerun_file=None,
):
    """
    Takes the output from inferpymc3 function, along with some other input
    information, and places it all in a netcdf output. This function also
    calculates the mean posterior emissions for the countries in the
    inversion domain and saves it to netcdf.
    Note that the uncertainties are defined by the highest posterior
    density (HPD) region and NOT percentiles (as the tdMCMC code).
    The HPD region is defined, for probability content (1-a), as:
        1) P(x \in R | y) = (1-a)
        2) for x1 \in R and x2 \notin R, P(x1|y)>=P(x2|y)
    -------------------------------
    Args:
      xouts (dict of array):
        Dictionary with key = flux source and entry the 
        MCMC chain for emissions scaling factors for each basis function.
      bcouts (array):
        MCMC chain for boundary condition scaling factors.
      sigouts (array):
        MCMC chain for model error.
      convergence (dict of str):
        Passed/Failed convergence test as to whether mutliple chains
        have a Gelman-Rubin diagnostic value <1.05 for each flux source
      Hx (array):
        Transpose of the sensitivity matrix to map emissions to measurement.
        This is the same as what is given from fp_data[site].H.values, where
        fp_data is the output from e.g. footprint_data_merge, but where it
        has been stacked for all sites.
      Hbc (array):
        Same as above but for boundary conditions
      Y (array):
        Measurement vector containing all measurements
      error (arrray):
        Measurement error vector, containg a value for each element of Y.
      Ytrace (array):
        Trace of modelled y values calculated from mcmc outputs and H matrices
      YBCtrace (array):
        Trace of modelled boundary condition values calculated from mcmc outputs and Hbc matrices
      step1 (str):
        Type of MCMC sampler for emissions and boundary condition updates.
      step2 (str):
        Type of MCMC sampler for model error updates.
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
      Ytime (pandas datetime array):
        Time stamp of measurements as used by the inversion.
      siteindicator (array):
        Numerical indicator of which site the measurements belong to,
        same length at Y.
      sigma_freq_index (array):
        Array of integer indexes that converts time into periods
      domain (str):
        Inversion spatial domain.
      species (str):
        Species of interest
      sites (list):
        List of sites in inversion
      start_date (str):
        Start time of inversion "YYYY-mm-dd"
      end_date (str):
        End time of inversion "YYYY-mm-dd"
      outputname (str):
        Unique identifier for output/run name.
      outputpath (str):
        Path to where output should be saved.
      country_unit_prefix ('str', optional)
        A prefix for scaling the country emissions. Current options are:
        'T' will scale to Tg, 'G' to Gg, 'M' to Mg, 'P' to Pg.
        To add additional options add to acrg_convert.prefix
        Default is none and no scaling will be applied (output in g).
      burn (int):
        Number of iterations burned in MCMC
      tune (int):
        Number of iterations used to tune step size
      nchain (int):
        Number of independent chains run
      sigma_per_site (bool):
        Whether a model sigma value was be calculated for each site independantly (True)
        or all sites together (False).
      fp_data (dict, optional):
        Output from footprints_data_merge + sensitivies
      emissions_name (list, optional):
        Update: Now a list with "source" values as used when adding emissions data to
        the OpenGHG object store.
      basis_directory (str, optional):
        Directory containing basis function file
      country_file (str, optional):
        Path of country definition file
      add_offset (bool):
        Add an offset (intercept) to all sites but the first in the site list. Default False.
      rerun_file (xarray dataset, optional):
        An xarray dataset containing the ncdf output from a previous run of the MCMC code.

    Returns:
        netdf file containing results from inversion
    -------------------------------
    TO DO:
        - Look at compressability options for netcdf output
        - I'm sure the number of inputs can be cut down or found elsewhere.
        - Currently it can only work out the country total emissions if
          the a priori emissions are constant over the inversion period
          or else monthly (and inversion is for less than one calendar year).
    """

    print("Post-processing output")

    # Get parameters for output file
    sectors = list(xouts.keys())         # Emissions sectors
    nit = xouts[sectors[0]].shape[0]     # No. of MCMC samples
    #nx = Hx.shape[0]                     # No. of stacked scaling regions / basis functions
    ny = len(Y)                          # No. of collated atmospheric measurements 
    nbc = Hbc.shape[0]                   # No. of BCs
    noff = offsetouts.shape[0]
    nbasis = [xouts[s].shape[1] for s in sectors]

    nui = np.arange(2)
    steps = np.arange(nit)
    nmeasure = np.arange(ny)
    #nparam = np.arange(nx)
    nBC = np.arange(nbc)
    nOFF = np.arange(noff)
    # YBCtrace = np.dot(Hbc.T,bcouts.T)

    # OFFSET HYPERPARAMETER
    # Calculates the mean/median/mode of the posterior simulated bias
    YmodmuOFF = np.mean(OFFSETtrace, axis=1)           # Mean scaling
    YmodmedOFF = np.median(OFFSETtrace, axis=1)        # Median scaling
    YmodmodeOFF = np.zeros(shape=OFFSETtrace.shape[0]) # Mode scaling

    for i in range(0, OFFSETtrace.shape[0]):
        # If sufficient no. of MCMC iterations, uses a KDE
        # to calculate mode. Else, mean value used in lieu
        if np.nanmax(OFFSETtrace[i, :]) > np.nanmin(OFFSETtrace[i, :]):
            # NB. len/step-size of xes_off should be reviewed
            xes_off = np.linspace(np.nanmin(OFFSETtrace[i, :]), np.nanmax(OFFSETtrace[i, :]), 200)
            kde = stats.gaussian_kde(OFFSETtrace[i, :]).evaluate(xes_off)
            YmodmodeOFF[i] = xes_off[kde.argmax()]
        else:
            YmodmodeOFF[i] = np.mean(OFFSETtrace[i, :])

    Ymod95OFF = pm.stats.hdi(OFFSETtrace.T, 0.95)
    Ymod68OFF = pm.stats.hdi(OFFSETtrace.T, 0.68)

    # Y-BC HYPERPARAMETER
    # Calculates the mean/median/mode of the posterior simulated boundary conditions
    YmodmuBC = np.mean(YBCtrace, axis=1)            # Mean scaling
    YmodmedBC = np.median(YBCtrace, axis=1)         # Median scaling
    YmodmodeBC = np.zeros(shape=YBCtrace.shape[0])  # Mode scaling

    for i in range(0, YBCtrace.shape[0]):
        # If sufficient no. of MCMC iterations, uses a KDE
        # to calculate mode. Else, mean value used in lieu
        if np.nanmax(YBCtrace[i, :]) > np.nanmin(YBCtrace[i, :]):
            # NB. len/step-size of xes_bc should be reviewed
            xes_bc = np.linspace(np.nanmin(YBCtrace[i, :]), np.nanmax(YBCtrace[i, :]), 200)
            kde = stats.gaussian_kde(YBCtrace[i, :]).evaluate(xes_bc)
            YmodmodeBC[i] = xes_bc[kde.argmax()]
        else:
            YmodmodeBC[i] = np.mean(YBCtrace[i, :])

    Ymod95BC = pm.stats.hdi(YBCtrace.T, 0.95)
    Ymod68BC = pm.stats.hdi(YBCtrace.T, 0.68)
    YaprioriBC = np.sum(Hbc, axis=0)

    # Y-VALUES HYPERPARAMETER (XOUTS * H + YBCtrace)
    Ymodmu = np.mean(Ytrace, axis=1)            # Mean scaling
    Ymodmed = np.median(Ytrace, axis=1)         # Median scaling
    Ymodmode = np.zeros(shape=Ytrace.shape[0])  # Mode scaling

    for i in range(0, Ytrace.shape[0]):
        # If sufficient no. of MCMC iterations, uses a KDE
        # to calculate mode. Else, mean value used in lieu
        if np.nanmax(Ytrace[i, :]) > np.nanmin(Ytrace[i, :]):
            # NB. len/step size of xes should be reviewed
            xes = np.arange(np.nanmin(Ytrace[i, :]), np.nanmax(Ytrace[i, :]), 0.5)
            kde = stats.gaussian_kde(Ytrace[i, :]).evaluate(xes)
            Ymodmode[i] = xes[kde.argmax()]
        else:
            Ymodmode[i] = np.mean(Ytrace[i, :])

    Ymod95 = pm.stats.hdi(Ytrace.T, 0.95)
    Ymod68 = pm.stats.hdi(Ytrace.T, 0.68)
    Yapriori = np.sum(Hx.T, axis=1) + np.sum(Hbc.T, axis=1)
    sitenum = np.arange(len(sites))

    if fp_data is None and rerun_file is not None:
        lon = rerun_file.lon.values
        lat = rerun_file.lat.values
        site_lat = rerun_file.sitelats.values
        site_lon = rerun_file.sitelons.values
        bfds = rerun_file.basisfunctions
    else:
        lon = fp_data[sites[0]].lon.values
        lat = fp_data[sites[0]].lat.values
        site_lat = np.zeros(len(sites))
        site_lon = np.zeros(len(sites))
        for si, site in enumerate(sites):
            site_lat[si] = fp_data[site].release_lat.values[0]
            site_lon[si] = fp_data[site].release_lon.values[0]
        basis_time_index = np.where(fp_data['.basis'].time.values == np.datetime64(start_date))[0][0]
        bfds = fp_data[".basis"][:,:,:,basis_time_index]

    # Calculate mean and mode posterior scale map and flux field
    scalemap_mu = np.zeros_like(bfds.values)
    scalemap_mode = np.zeros_like(bfds.values)

    for i in range(len(sectors)):
        bfds_i = bfds[i, :, :] 

        for npm in range(0, int(np.max(bfds_i))):
            scalemap_mu[i][bfds_i.values == (npm + 1)] = np.mean(xouts[sectors[i]][:, npm])
            # If sufficient no. of MCMC iterations, uses a KDE
            # to calculate mode. Else, mean value used in lieu
            if np.nanmax(xouts[sectors[i]][:, npm]) > np.nanmin(xouts[sectors[i]][:, npm]):
                xes = np.arange(np.nanmin(xouts[sectors[i]][:, npm]), np.nanmax(xouts[sectors[i]][:, npm]), 0.01)
                kde = stats.gaussian_kde(xouts[sectors[i]][:, npm]).evaluate(xes)
                scalemap_mode[i][bfds_i.values == (npm + 1)] = xes[kde.argmax()]
            else:
                scalemap_mode[i][bfds_i.values == (npm + 1)] = np.mean(xouts[sectors[i]][:, npm])

    if rerun_file is not None:
        flux_array_all = np.expand_dims(rerun_file.fluxapriori.values, 2)
    else:
        if emissions_name is None:
            raise ValueError("Emissions name not provided.")
        else:
            ax0 = len(list(fp_data[".flux"].keys()))                              # nsector axis
            ax1 = fp_data[".flux"][emissions_name[0]].data.flux.values.shape[0]   # lat axis
            ax2 = fp_data[".flux"][emissions_name[0]].data.flux.values.shape[1]   # lon axis
            ax3 = fp_data[".flux"][emissions_name[0]].data.flux.values.shape[2]   # time axis

            flux_array_all = np.zeros(shape=(ax0, ax1, ax2, ax3))
            for i, key in enumerate(fp_data[".flux"].keys()):
               flux_array_all[i] = fp_data[".flux"][key].data.flux.values

    if flux_array_all.shape[3] == 1:
        print("\nAssuming flux prior is annual and extracting first index of flux array.")
        apriori_flux = flux_array_all[:, :, :, 0]
    else:
        print("\nAssuming flux prior is monthly.")
        print(f"Extracting weighted average flux prior from {start_date} to {end_date}")
        allmonths = pd.date_range(start_date, end_date).month[:-1].values
        allmonths -= 1  # to align with zero indexed array

        apriori_flux = np.zeros_like(flux_array_all[:, :, :, 0])

        # calculate the weighted average flux across the whole inversion period
        for m in np.unique(allmonths):
            apriori_flux += flux_array_all[:, :, :, m] * np.sum(allmonths == m) / len(allmonths)

    #flux = np.squeeze(scalemap_mode,axis=-1) * np.squeeze(apriori_flux,axis=-1)
    flux = scalemap_mode * apriori_flux

    # Basis functions to save
    #bfarray = np.squeeze(bfds.values - 1)
    bfarray = bfds.values - 1

    # Calculate country totals
    area = utils.areagrid(lat, lon)
    if not rerun_file:
        c_object = utils.get_country(domain, country_file=country_file)
        cntryds = xr.Dataset(
            {"country": (["lat", "lon"], c_object.country), "name": (["ncountries"], c_object.name)},
            coords={"lat": (c_object.lat), "lon": (c_object.lon)},
        )
        cntrynames = cntryds.name.values
        cntrygrid = cntryds.country.values
    else:
        cntrynames = rerun_file.countrynames.values
        cntrygrid = rerun_file.countrydefinition.values

    cntrymean = np.zeros((len(sectors), len(cntrynames)))
    cntrymedian = np.zeros((len(sectors), len(cntrynames)))
    cntrymode = np.zeros((len(sectors), len(cntrynames)))
    cntry68 = np.zeros((len(sectors), len(cntrynames), len(nui)))
    cntry95 = np.zeros((len(sectors), len(cntrynames), len(nui)))
    cntrysd = np.zeros((len(sectors), len(cntrynames)))
    cntryprior = np.zeros((len(sectors), len(cntrynames)))

    molarmass = convert.molar_mass(species)

    unit_factor = convert.prefix(country_unit_prefix)
    if country_unit_prefix is None:
        country_unit_prefix = ""
    country_units = country_unit_prefix + "g"
    if rerun_file is not None:
        obs_units = rerun_file.Yobs.attrs["units"].split(" ")[0]
    else:
        obs_units = str(fp_data[".units"])

    # WARNING! Triple nested for-loops. Might want to see if there's a better way to code this block
    for i in range(len(sectors)):
        for ci, cntry in enumerate(cntrynames):
            cntrytottrace = np.zeros(len(steps))
            cntrytotprior = 0
            for bf in range(int(np.max(bfarray[i])) + 1):
                bothinds = np.logical_and(cntrygrid == ci, bfarray[i] == bf)

                cntrytottrace += (
                    np.sum(area[bothinds].ravel() * apriori_flux[i][bothinds].ravel() * 3600 * 24 * 365 * molarmass)
                    * xouts[sectors[i]][:, bf]
                    / unit_factor
                )

                cntrytotprior += (
                    np.sum(area[bothinds].ravel() * apriori_flux[i][bothinds].ravel() * 3600 * 24 * 365 * molarmass)
                    / unit_factor
                )

            cntrymean[i, ci] = np.mean(cntrytottrace)
            cntrymedian[i, ci] = np.median(cntrytottrace)

            if np.nanmax(cntrytottrace) > np.nanmin(cntrytottrace):
                xes = np.linspace(np.nanmin(cntrytottrace), np.nanmax(cntrytottrace), 200)
                kde = stats.gaussian_kde(cntrytottrace).evaluate(xes)
                cntrymode[i, ci] = xes[kde.argmax()]
            else:
                cntrymode[i, ci] = np.mean(cntrytottrace)

        cntrysd[i, ci] = np.std(cntrytottrace)
        cntry68[i, ci, :] = pm.stats.hdi(cntrytottrace.values, 0.68)
        cntry95[i, ci, :] = pm.stats.hdi(cntrytottrace.values, 0.95)
        cntryprior[i, ci] = cntrytotprior

    # Make output netcdf file
    outds = xr.Dataset(
        {
            "Yobs": (["nmeasure"], Y),
            "Yerror": (["nmeasure"], error),
            "Ytime": (["nmeasure"], Ytime),
            "Yapriori": (["nmeasure"], Yapriori),
            "Ymodmean": (["nmeasure"], Ymodmu),
            "Ymodmedian": (["nmeasure"], Ymodmed),
            "Ymodmode": (["nmeasure"], Ymodmode),
            "Ymod95": (["nmeasure", "nUI"], Ymod95),
            "Ymod68": (["nmeasure", "nUI"], Ymod68),
            "Yoffmean": (["nmeasure"], YmodmuOFF),
            "Yoffmedian": (["nmeasure"], YmodmedOFF),
            "Yoffmode": (["nmeasure"], YmodmodeOFF),
            "Yoff68": (["nmeasure", "nUI"], Ymod68OFF),
            "Yoff95": (["nmeasure", "nUI"], Ymod95OFF),
            "YaprioriBC": (["nmeasure"], YaprioriBC),
            "YmodmeanBC": (["nmeasure"], YmodmuBC),
            "YmodmedianBC": (["nmeasure"], YmodmedBC),
            "YmodmodeBC": (["nmeasure"], YmodmodeBC),
            "Ymod95BC": (["nmeasure", "nUI"], Ymod95BC),
            "Ymod68BC": (["nmeasure", "nUI"], Ymod68BC),
            #"xtrace": (["fluxsectors", "steps", "nparam"], np.array(list(xouts.values()))),
            "bctrace": (["steps", "nBC"], bcouts.values),
            "sigtrace": (["steps", "nsigma_site", "nsigma_time"], sigouts.values),
            "siteindicator": (["nmeasure"], siteindicator),
            "sigmafreqindex": (["nmeasure"], sigma_freq_index),
            "sitenames": (["nsite"], sites),
            "sitelons": (["nsite"], site_lon),
            "sitelats": (["nsite"], site_lat),
            "fluxapriori": (["fluxsectors", "lat", "lon"], apriori_flux),
            "fluxmode": (["fluxsectors", "lat", "lon"], flux),
            "scalingmean": (["fluxsectors", "lat", "lon"], scalemap_mu),
            "scalingmode": (["fluxsectors", "lat", "lon"], scalemap_mode),
            "basisfunctions": (["fluxsectors","lat", "lon"], bfarray),

            "countrymean": (["fluxsectors", "countrynames"], cntrymean),
            "countrymedian": (["fluxsectors", "countrynames"], cntrymedian),
            "countrymode": (["fluxsectors", "countrynames"], cntrymode),
            "countrysd": (["fluxsectors", "countrynames"], cntrysd),
            "country68": (["fluxsectors", "countrynames", "nUI"], cntry68),
            "country95": (["fluxsectors", "countrynames", "nUI"], cntry95),
            "countryapriori": (["fluxsectors", "countrynames"], cntryprior),
            "countrydefinition": (["lat", "lon"], cntrygrid),
            #"xsensitivity": (["nmeasure", "nparam"], Hx.T),
            "bcsensitivity": (["nmeasure", "nBC"], Hbc.T),
        },
        coords={
            "stepnum": (["steps"], steps),
            #"paramnum": (["nlatent"], nparam),
            "numBC": (["nBC"], nBC),
            "measurenum": (["nmeasure"], nmeasure),
            "UInum": (["nUI"], nui),
            "nsites": (["nsite"], sitenum),
            "nsigma_time": (["nsigma_time"], np.unique(sigma_freq_index)),
            "nsigma_site": (["nsigma_site"], np.arange(sigouts.shape[1]).astype(int)),
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "countrynames": (["countrynames"], cntrynames),
            "fluxsectors": (["fluxsectors"], np.array(sectors)),
        },
    )

    Hx_split_array = np.split(Hx,np.cumsum(nbasis)[:-1],axis=0) #split full stacked H matrix into sectors
    Hx_split = dict(zip(sectors,Hx_split_array)) #turn into dictionary
    
    for i,s in enumerate(sectors):
        
        outds.coords[f'nbasis_{s}'] = ([f'nbasis_{s}'],np.arange(nbasis[i]))
        outds[f'xtrace_{s}'] = (['steps',f'nbasis_{s}'],xouts[s].values)
        outds[f'xtrace_{s}'].attrs["longname"] = f"trace of unitless scaling factors for {s} emissions parameters"
        
        outds[f'xsensitivity_{s}'] = (['nmeasure',f'nbasis_{s}'],Hx_split[s].T)
        outds[f'xsensitivity_{s}'].attrs["units"] = obs_units + " " + "mol/mol"
        outds[f'xsensitivity_{s}'].attrs["longname"] = f"{s} emissions sensitivity timeseries"

    outds.fluxmode.attrs["units"] = "mol/m2/s"
    outds.fluxapriori.attrs["units"] = "mol/m2/s"
    outds.Yobs.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yapriori.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymodmean.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymodmedian.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymodmode.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymod95.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymod68.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yoffmean.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yoffmedian.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yoffmode.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yoff95.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yoff68.attrs["units"] = obs_units + " " + "mol/mol"
    outds.YmodmeanBC.attrs["units"] = obs_units + " " + "mol/mol"
    outds.YmodmedianBC.attrs["units"] = obs_units + " " + "mol/mol"
    outds.YmodmodeBC.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymod95BC.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymod68BC.attrs["units"] = obs_units + " " + "mol/mol"
    outds.YaprioriBC.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yerror.attrs["units"] = obs_units + " " + "mol/mol"
    outds.countrymean.attrs["units"] = country_units
    outds.countrymedian.attrs["units"] = country_units
    outds.countrymode.attrs["units"] = country_units
    outds.country68.attrs["units"] = country_units
    outds.country95.attrs["units"] = country_units
    outds.countrysd.attrs["units"] = country_units
    outds.countryapriori.attrs["units"] = country_units
    #outds.xsensitivity.attrs["units"] = obs_units + " " + "mol/mol"
    outds.bcsensitivity.attrs["units"] = obs_units + " " + "mol/mol"
    outds.sigtrace.attrs["units"] = obs_units + " " + "mol/mol"

    outds.Yobs.attrs["longname"] = "observations"
    outds.Yerror.attrs["longname"] = "measurement error"
    outds.Ytime.attrs["longname"] = "time of measurements"
    outds.Yapriori.attrs["longname"] = "a priori simulated measurements"
    outds.Ymodmean.attrs["longname"] = "mean of posterior simulated measurements"
    outds.Ymodmedian.attrs["longname"] = "median of posterior simulated measurements"
    outds.Ymodmode.attrs["longname"] = "mode of posterior simulated measurements"
    outds.Ymod68.attrs["longname"] = " 0.68 Bayesian credible interval of posterior simulated measurements"
    outds.Ymod95.attrs["longname"] = " 0.95 Bayesian credible interval of posterior simulated measurements"
    outds.Yoffmean.attrs["longname"] = "mean of posterior simulated offset between measurements"
    outds.Yoffmedian.attrs["longname"] = "median of posterior simulated offset between measurements"
    outds.Yoffmode.attrs["longname"] = "mode of posterior simulated offset between measurements"
    outds.Yoff68.attrs[
        "longname"
    ] = " 0.68 Bayesian credible interval of posterior simulated offset between measurements"
    outds.Yoff95.attrs[
        "longname"
    ] = " 0.95 Bayesian credible interval of posterior simulated offset between measurements"
    outds.YaprioriBC.attrs["longname"] = "a priori simulated boundary conditions"
    outds.YmodmeanBC.attrs["longname"] = "mean of posterior simulated boundary conditions"
    outds.YmodmedianBC.attrs["longname"] = "median of posterior simulated boundary conditions"
    outds.YmodmodeBC.attrs["longname"] = "mode of posterior simulated boundary conditions"
    outds.Ymod68BC.attrs[
        "longname"
    ] = " 0.68 Bayesian credible interval of posterior simulated boundary conditions"
    outds.Ymod95BC.attrs[
        "longname"
    ] = " 0.95 Bayesian credible interval of posterior simulated boundary conditions"
    #outds.xtrace.attrs["longname"] = "trace of unitless scaling factors for emissions parameters"
    outds.bctrace.attrs["longname"] = "trace of unitless scaling factors for boundary condition parameters"
    outds.sigtrace.attrs["longname"] = "trace of model error parameters"
    outds.siteindicator.attrs["longname"] = "index of site of measurement corresponding to sitenames"
    outds.sigmafreqindex.attrs["longname"] = "perdiod over which the model error is estimated"
    outds.sitenames.attrs["longname"] = "site names"
    outds.sitelons.attrs["longname"] = "site longitudes corresponding to site names"
    outds.sitelats.attrs["longname"] = "site latitudes corresponding to site names"
    outds.fluxapriori.attrs["longname"] = "mean a priori flux over period"
    outds.fluxmode.attrs["longname"] = "mode posterior flux over period"
    outds.scalingmean.attrs["longname"] = "mean scaling factor field over period"
    outds.scalingmode.attrs["longname"] = "mode scaling factor field over period"
    outds.basisfunctions.attrs["longname"] = "basis function field"
    outds.countrymean.attrs["longname"] = "mean of ocean and country totals"
    outds.countrymedian.attrs["longname"] = "median of ocean and country totals"
    outds.countrymode.attrs["longname"] = "mode of ocean and country totals"
    outds.country68.attrs["longname"] = "0.68 Bayesian credible interval of ocean and country totals"
    outds.country95.attrs["longname"] = "0.95 Bayesian credible interval of ocean and country totals"
    outds.countrysd.attrs["longname"] = "standard deviation of ocean and country totals"
    outds.countryapriori.attrs["longname"] = "prior mean of ocean and country totals"
    outds.countrydefinition.attrs["longname"] = "grid definition of countries"
    #outds.xsensitivity.attrs["longname"] = "emissions sensitivity timeseries"
    outds.bcsensitivity.attrs["longname"] = "boundary conditions sensitivity timeseries"

    outds.attrs["Start date"] = start_date
    outds.attrs["End date"] = end_date
    outds.attrs["Latent sampler"] = str(step1)[20:33]
    outds.attrs["Hyper sampler"] = str(step2)[20:33]
    outds.attrs["Burn in"] = str(int(burn))
    outds.attrs["Tuning steps"] = str(int(tune))
    outds.attrs["Number of chains"] = str(int(nchain))
    outds.attrs["Error for each site"] = str(sigma_per_site)
    outds.attrs["Emissions Prior"] = "".join(["{0},{1},".format(k, v) for k, v in xprior.items()])[:-1]
    outds.attrs["Model error Prior"] = "".join(["{0},{1},".format(k, v) for k, v in sigprior.items()])[:-1]
    outds.attrs["BCs Prior"] = "".join(["{0},{1},".format(k, v) for k, v in bcprior.items()])[:-1]
    if add_offset:
        outds.attrs["Offset Prior"] = "".join(["{0},{1},".format(k, v) for k, v in offsetprior.items()])[:-1]
    outds.attrs["Creator"] = getpass.getuser()
    outds.attrs["Date created"] = str(pd.Timestamp("today"))
    outds.attrs["Convergence"] = str(convergence)
    try:
        outds.attrs["Repository version"] = code_version()
    except:
        print('Cannot find code version, check this function.')

    # variables with variable length data types shouldn't be compressed
    # e.g. object ("O") or unicode ("U") type
    do_not_compress = []
    dtype_pat = re.compile(r"[<>=]?[UO]")  # regex for Unicode and Object dtypes
    for dv in outds.data_vars:
        if dtype_pat.match(outds[dv].data.dtype.str):
            do_not_compress.append(dv)

    # setting compression levels for data vars in outds
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in outds.data_vars if var not in do_not_compress}

    output_filename = define_output_filename(outputpath, species, domain, outputname, start_date, ext=".nc")
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    outds.to_netcdf(output_filename, encoding=encoding, mode="w")

    return outds
