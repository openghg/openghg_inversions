# *****************************************************************************
# Created: 7 Nov. 2022
# Author: Eric Saboya, School of Geographical Sciences, University of Bristol
# Contact: eric.saboya@bristol.ac.uk
# *****************************************************************************
# About
#   Originally created by Luke Western (ACRG) and updated, here, by Eric Saboya
#   Functions for performing MCMC inversion.
#   PyMC library used for Bayesian modelling. Updated from PyMc3
# *****************************************************************************

import numpy as np
import pymc as pm
import pandas as pd
import xarray as xr
import getpass
from scipy import stats
from pathlib import Path
import pytensor.tensor as pt

from openghg.retrieve import get_flux

from openghg_inversions import convert 
from openghg_inversions import utils
from openghg_inversions.hbmcmc.inversionsetup import opends, offset_matrix
from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename
from openghg_inversions.config.version import code_version

def parseprior(name, prior_params, shape = ()):
    '''
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
    '''
    functiondict = {"uniform":pm.Uniform,
                    "flat":pm.Flat,
                    "halfflat":pm.HalfFlat,
                    "normal":pm.Normal,
                    "truncatednormal":pm.TruncatedNormal,
                    "halfnormal":pm.HalfNormal,
                    "skewnormal":pm.SkewNormal,
                    "beta":pm.Beta,
                    "kumaraswamy":pm.Kumaraswamy,
                    "exponential":pm.Exponential,
                    "laplace":pm.Laplace,
                    "studentt":pm.StudentT,
                    "halfstudentt":pm.HalfStudentT,
                    "cauchy":pm.Cauchy,
                    "halfcauchy":pm.HalfCauchy,
                    "gamma":pm.Gamma,
                    "inversegamma":pm.InverseGamma,
                    "weibull":pm.Weibull,
                    "lognormal":pm.Lognormal,
                    "chisquared":pm.ChiSquared,
                    "wald":pm.Wald,
                    "pareto":pm.Pareto,
                    "exgaussian":pm.ExGaussian,
                    "vonmises":pm.VonMises,
                    "triangular":pm.Triangular,
                    "gumbel":pm.Gumbel,
                    "rice":pm.Rice,
                    "logistic":pm.Logistic,
                    "logitnormal":pm.LogitNormal,
                    "interpolated":pm.Interpolated}
    
    pdf = prior_params["pdf"]
    #Get a dictionary of the pdf arguments
    params = {x: prior_params[x] for x in prior_params if x != "pdf"}
    return functiondict[pdf.lower()](name, shape=shape, **params)

def inferpymc(Hx, Hbc, Y, error, siteindicator, sigma_freq_index,
              xprior={'all':{"pdf":"lognormal", "mu":1, "sigma":1}},
              bcprior={"pdf":"lognormal", "mu":0.004, "sigma":0.02},
              sigprior={"pdf":"uniform", "lower":0.5, "upper":3},
              nit=2.5e5, burn=50000, tune=1.25e5, nchain=2, 
              sigma_per_site = True, 
              offsetprior={"pdf":"normal", "mu":0, "sigma":1},
              add_offset = False, verbose=False,emissions_name=['all']):       
    '''
    Uses PyMC module for Bayesian inference for emissions field, boundary 
    conditions and (currently) a single model error value.
    This uses a Normal likelihood but the (hyper)prior PDFs can selected by user.
    -----------------------------------
    Args:
      Hx (dict of arrays):
        Transpose of the sensitivity matrix to map emissions to measurement.
        This is the same as what is given from fp_data[site].H.values, where
        fp_data is the output from e.g. footprint_data_merge, but where it
        has been stacked for all sites. Dictionary of arrays, indexed by source.
      Hbc (array):
        Same as above but for boundary conditions
      Y (array):
        Measurement vector containing all measurements
      error (arrray):
        Measurement error vector, containg a value for each element of Y.
      siteindicator (array):
        Array of indexing integers that relate each measurement to a site
      sigma_freq_index (array):
        Array of integer indexes that converts time into periods
      xprior (dict of dict):
        Dictionary of dictionaries containing information about the prior PDF for emissions for each sector. 
        The emissions names in the ModelScenario object are the dictionary keys.
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
      verbose:
        When True, prints progress bar
      emissions_name (list of str):
        Name for each emissions source file.
      
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
    '''

    burn = int(burn)  
           
    #hx = Hx.T 
    hx = {source:Hx[source].T for source in emissions_name}
    hbc = Hbc.T
    nx = {source:hx[source].shape[1] for source in emissions_name}
    #nx = hx.shape[1]    
    nbc = hbc.shape[1]
    ny = len(Y)

    nit = int(nit)    
    
    #convert siteindicator into a site indexer
    if sigma_per_site:
        sites = siteindicator.astype(int)
        nsites = np.amax(sites)+1
    else:
        sites = np.zeros_like(siteindicator).astype(int)
        nsites = 1
    nsigmas = np.amax(sigma_freq_index)+1
    
    if add_offset:
        B = offset_matrix(siteindicator)

    with pm.Model() as model:
        #x = []
        #for source in emissions_name:
        #    if xprior[source] is 'fixed':
        #        x.append(pm.ConstantData(f'x_{source}',np.ones(nbasis_actual)))
        #        mu_test = pm.mat.dot(hx[:,:nbasis_actual])
        #    else:
        #        x.append(parseprior(f"x_{source}", xprior[source], shape=nbasis_actual))#
#
#        print(x)
        
        xall = []
        
        
        sig = parseprior("sig", sigprior, shape=(nsites, nsigmas))
        
        for s,source in enumerate(emissions_name):
            if xprior[source] is 'fixed':
                #x = pm.ConstantData(f'x_{source}',np.ones(nbasis_actual))
                x = pm.Deterministic(f'x_{source}',pt.as_tensor_variable(np.ones(nx[source])))
            else:
                x = parseprior(f"x_{source}", xprior[source], shape=nx[source])
                xall.append(x)
                
            mod_source_mf = pm.math.dot(hx[source],x)
                
            if s == 0:
                mu = mod_source_mf
                model_error = np.abs(mod_source_mf) * sig[sites, sigma_freq_index] 
            else:
                mu += mod_source_mf
                model_error += np.abs(mod_source_mf) * sig[sites, sigma_freq_index] 
                        
        xbc = parseprior("xbc", bcprior, shape=nbc)
        mu += pm.math.dot(hbc,xbc)
        
        if add_offset:
            offset = parseprior("offset", offsetprior, shape=np.amax(nsites)) 
            offset_vec = pm.math.concatenate( (np.array([0]), offset), axis=0)
            mu += pm.math.dot(B, offset_vec)
            
        model_error = np.ones(error.shape[0])
        epsilon = pm.math.sqrt(error**2 + model_error**2)
        y = pm.Normal('y', mu=mu, sigma=epsilon, observed=Y, shape=ny)
                
        #x_allsources = pm.Deterministic('x_allsources',pm.math.stack(x))
        
        # keep editing from here, try out above code that calculates Hx for each 
        # sector separately 
        
        #if add_offset:
        #    offset = parseprior("offset", offsetprior, shape=nsites-1) 
        #    offset_vec = pm.math.concatenate( (np.array([0]), offset), axis=0)
        #    mu = pm.math.dot(hx,x_allsources) + pm.math.dot(hbc,xbc) + pm.math.dot(B, offset_vec)
        #else:
        #    mu = pm.math.dot(hx,x_allsources) + pm.math.dot(hbc,xbc)      

        #model_error = np.abs(pm.math.dot(hx,x_allsources)) * sig[sites, sigma_freq_index] 
        #epsilon = pm.math.sqrt(error**2 + model_error**2)
        #y = pm.Normal('y', mu=mu, sigma=epsilon, observed=Y, shape=ny)
        
        #xall_sampled = pm.Deterministic('xall',pm.math.stack(xall))
        
        xall.append(xbc)
        
        #print(xall)
        
        step1 = pm.NUTS(vars=xall)
        #step2 = pm.NUTS(vars=[xbc])
        step2 = pm.Slice(vars=[sig])
        
        trace = pm.sample(nit, tune=int(tune), chains=nchain,
                          step=[step1,step2], 
                          progressbar=verbose, cores=nchain)#step=pm.Metropolis())#  #target_accept=0.8,
       
        outs = {}
        convergence = {}
        
        for source in emissions_name:
            outs[source] = trace.posterior[f'x_{source}'][0, burn:nit].values
            #Check for convergence
            gelrub = pm.rhat(trace)[f'x_{source}'].max()
            if gelrub > 1.05:
                print(f'Failed Gelman-Rubin at 1.05 for {source}')
                convergence[source] = "Failed"
            else:
                convergence[source] = "Passed"
       
        #outs = trace.posterior['xall'][0, burn:nit]
        bcouts = trace.posterior['xbc'][0, burn:nit]
        sigouts = trace.posterior['sig'][0, burn:nit]

        #outs = trace.get_values(x, burn=burn)[0:int((nit)-burn)]
        #bcouts = trace.get_values(xbc, burn=burn)[0:int((nit)-burn)]
        #sigouts = trace.get_values(sig, burn=burn)[0:int((nit)-burn)]
        
        #Check for convergence
        #gelrub = pm.rhat(trace)['xall'].max()
        #if gelrub > 1.05:
        #    print('Failed Gelman-Rubin at 1.05')
        #    convergence = "Failed"
        #else:
        #    convergence = "Passed"
        
        offset_outs = {}
        
        if add_offset:
            offset_outs = trace.posterior['offset'][0, burn:nit]
            #offset_outs = trace.get_values(offset, burn=burn)[0:int((nit)-burn)]
            offset_trace = np.hstack([np.zeros((int(nit-burn),1)), offset_outs])
            YBCtrace = np.dot(Hbc.T,bcouts.T) + np.dot(B, offset_trace.T)
            OFFtrace = np.dot(B, offset_trace.T)   
        else:
            YBCtrace = np.dot(Hbc.T,bcouts.T)
            #offset_outs = outs * 0 
            offset_outs = np.zeros(nx[emissions_name[0]]) #TODO how does this work for multiple sectors?
            #offset_trace = np.hstack([np.zeros((int(nit-burn),1)), offset_outs])
            OFFtrace =  YBCtrace * 0

        Ytrace = YBCtrace.copy()

        for source in emissions_name:
            
            Ytrace += np.dot(Hx[source].T,outs[source].T)
        
        return outs, bcouts, sigouts, offset_outs, Ytrace, YBCtrace, OFFtrace, convergence, step1, step2

def inferpymc_postprocessouts(xouts,bcouts, sigouts, offset_outs, convergence, 
                               Hx, Hbc, Y, error, Ytrace, YBCtrace, offset_trace, 
                               step1, step2, 
                               xprior, bcprior, sigprior, offsetprior, Ytime, siteindicator, sigma_freq_index,
                               domain, species, sites,
                               start_date, end_date, outputname, outputpath,
                               country_unit_prefix,
                               burn, tune, nchain, sigma_per_site,
                               emissions_name, emissions_store, fp_data=None, 
                               basis_directory=None, country_file=None,
                               add_offset=False, rerun_file=None):

        '''
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
            xouts (array):
                MCMC chain for emissions scaling factors for each basis function.
            bcouts (array):
                MCMC chain for boundary condition scaling factors.
            sigouts (array):
                MCMC chain for model error.
            convergence (str):
                Passed/Failed convergence test as to whether mutliple chains
                have a Gelman-Rubin diagnostic value <1.05
            Hx (dict array):
                Transpose of the sensitivity matrix to map emissions to measurement.
                This is the same as what is given from fp_data[site].H.values, where
                fp_data is the output from e.g. footprint_data_merge, but where it
                has been stacked for all sites. Dictionary of arrays, indexed by source.
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
            I   nversion spatial domain.
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
        '''
            
        print("Post-processing output")
            
        # Get parameters for output file 
        nit = xouts[emissions_name[0]].shape[0]
        #nx = Hx.shape[0]
        nx = {source:Hx[source].shape[0] for source in emissions_name}
        ny = len(Y)
        nbc = Hbc.shape[0]
        noff = offset_outs.shape[0]

        nui = np.arange(2)
        steps = np.arange(nit)
        nmeasure = np.arange(ny)
        #nparam = np.arange(nx)
        nparam = {source:np.arange(nx[source]) for source in emissions_name}
        nBC = np.arange(nbc)
        nOFF = np.arange(noff)
        #YBCtrace = np.dot(Hbc.T,bcouts.T)

        # OFFSET HYPERPARAMETER
        YmodmuOFF = np.mean(offset_trace,axis=1) # mean
        YmodmedOFF = np.median(offset_trace,axis=1) # median
        YmodmodeOFF = np.zeros(shape=offset_trace.shape[0]) # mode

        for i in range(0, offset_trace.shape[0]):
            # if sufficient no. of iterations use a KDE to calculate mode
            # else, mean value used in lieu
            if np.nanmax(offset_trace[i,:]) > np.nanmin(offset_trace[i,:]):
                xes_off = np.linspace(np.nanmin(offset_trace[i,:]), np.nanmax(offset_trace[i,:]), 200)
                kde = stats.gaussian_kde(offset_trace[i,:]).evaluate(xes_off)
                YmodmodeOFF[i] = xes_off[kde.argmax()]
            else:
                YmodmodeOFF[i] = np.mean(offset_trace[i,:])

        Ymod95OFF = pm.stats.hdi(offset_trace.T, 0.95)
        Ymod68OFF = pm.stats.hdi(offset_trace.T, 0.68)
        
        # Y-BC HYPERPARAMETER
        YmodmuBC = np.mean(YBCtrace, axis=1)
        YmodmedBC = np.median(YBCtrace, axis=1)
        YmodmodeBC = np.zeros(shape=YBCtrace.shape[0])

        for i in range(0, YBCtrace.shape[0]):
            # if sufficient no. of iterations use a KDE to calculate mode
            # else, mean value used in lieu
            if np.nanmax(YBCtrace[i,:]) > np.nanmin(YBCtrace[i,:]):
                xes_bc = np.linspace(np.nanmin(YBCtrace[i,:]), np.nanmax(YBCtrace[i,:]), 200)
                kde = stats.gaussian_kde(YBCtrace[i,:]).evaluate(xes_bc)
                YmodmodeBC[i] = xes_bc[kde.argmax()]
            else:
                YmodmodeBC[i] = np.mean(YBCtrace[i,:])

        Ymod95BC = pm.stats.hdi(YBCtrace.T, 0.95)
        Ymod68BC = pm.stats.hdi(YBCtrace.T, 0.68)
        YaprioriBC = np.sum(Hbc, axis=0)

        # Y-VALUES HYPERPARAMETER (XOUTS * H)
        Ymodmu = np.mean(Ytrace, axis=1)
        Ymodmed = np.median(Ytrace, axis=1)
        Ymodmode = np.zeros(shape=Ytrace.shape[0])

        for i in range(0, Ytrace.shape[0]):
            # if sufficient no. of iterations use a KDE to calculate mode
            # else, mean value used in lieu
            if np.nanmax(Ytrace[i,:]) > np.nanmin(Ytrace[i,:]):
                xes = np.arange(np.nanmin(Ytrace[i,:]), np.nanmax(Ytrace[i,:]), 0.5)
                kde = stats.gaussian_kde(Ytrace[i,:]).evaluate(xes)
                Ymodmode[i] = xes[kde.argmax()]
            else:
                Ymodmode[i] = np.mean(Ytrace[i,:])

        Ymod95 = pm.stats.hdi(Ytrace.T, 0.95)
        Ymod68 = pm.stats.hdi(Ytrace.T, 0.68)
        Yapriori = np.zeros(Hx[emissions_name[0]].shape[1])
        for source in emissions_name:
            Yapriori += np.sum(Hx[source].T, axis=1) + np.sum(Hbc.T, axis=1)
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
            bfds = fp_data[".basis"]
            #bfds = fp_data[source'][".basis"] #TODO extract basis per sector

        #Calculate mean  and mode posterior scale map and flux field
        scalemap_mu = {source:np.zeros_like(bfds.values) for source in emissions_name}
        scalemap_mode = {source:np.zeros_like(bfds.values) for source in emissions_name}
        
        for source in emissions_name:
            for npm in nparam[source]:
                scalemap_mu[source][bfds.values == (npm+1)] = np.mean(xouts[source][:,npm])
                if np.nanmax(xouts[source][:,npm]) > np.nanmin(xouts[source][:,npm]):
                    xes = np.arange(np.nanmin(xouts[source][:,npm]), np.nanmax(xouts[source][:,npm]), 0.01)
                    kde = stats.gaussian_kde(xouts[source][:,npm]).evaluate(xes)
                    scalemap_mode[source][bfds.values == (npm+1)] = xes[kde.argmax()]
                else:
                    scalemap_mode[source][bfds.values == (npm+1)] = np.mean(xouts[source][:,npm])

        emissions_flux = {}
        flux = {}
        apriori_flux = {}
        
        for source in emissions_name:
            print(f'\n{source}')

            if rerun_file is not None:
                emissions_flux[source] = rerun_file[f'flux_apriori_{source}'].values
            else:
                emds = fp_data['.flux'][source]
                flux_array_all = emds.data.flux.values
                
            if flux_array_all.shape[2] == 1:
                print('Assuming flux prior is annual and extracting first index of flux array.')
                apriori_flux[source] = flux_array_all[:,:,0]
            else:
                print(f'Assuming flux prior is monthly.')
                print(f'Extracting weighted average flux from {start_date} to {end_date}')
                allmonths = pd.date_range(start_date, end_date).month[:-1].values
                allmonths -= 1 #to align with zero indexed array

                apriori_flux[source] = np.zeros_like(flux_array_all[:,:,0])

                #calculate the weighted average flux across the whole inversion period
                for m in np.unique(allmonths):
                    apriori_flux[source] += flux_array_all[:,:,m] * np.sum(allmonths == m)/len(allmonths)

            flux[source] = scalemap_mode[source]*apriori_flux[source]
        
        #Basis functions to save
        bfarray = bfds.values-1

        #Calculate country totals   
        area = utils.areagrid(lat, lon)
        if not rerun_file:
            c_object = utils.get_country(domain, country_file=country_file)
            cntryds = xr.Dataset({'country': (['lat','lon'], c_object.country), 
                                    'name' : (['ncountries'],c_object.name)},
                                    coords = {'lat': (c_object.lat),
                                            'lon': (c_object.lon)})
            cntrynames = cntryds.name.values
            cntrygrid = cntryds.country.values
        else:
            cntrynames = rerun_file.countrynames.values
            cntrygrid = rerun_file.countrydefinition.values

        cntrymean = {source:np.zeros((len(cntrynames))) for source in emissions_name}
        cntrymedian = {source:np.zeros((len(cntrynames))) for source in emissions_name}
        cntrymode = {source:np.zeros((len(cntrynames))) for source in emissions_name}
        cntry68 = {source:np.zeros((len(cntrynames), len(nui))) for source in emissions_name}
        cntry95 = {source:np.zeros((len(cntrynames), len(nui))) for source in emissions_name}
        cntrysd = {source:np.zeros(len(cntrynames)) for source in emissions_name}
        cntryprior = {source:np.zeros(len(cntrynames)) for source in emissions_name}
        
        molarmass = convert.molar_mass(species)
        unit_factor = convert.prefix(country_unit_prefix)
        
        if country_unit_prefix is None:
            country_unit_prefix=''
        country_units = country_unit_prefix + 'g'
        if rerun_file is not None:
            obs_units = rerun_file.Yobs.attrs["units"].split(" ")[0]
        else:
            obs_units = str(fp_data[".units"])
            
        for source in emissions_name:
        
            for ci, cntry in enumerate(cntrynames):
                cntrytottrace = np.zeros(len(steps))
                cntrytotprior = 0
                for bf in range(int(np.max(bfarray))+1):
                    bothinds = np.logical_and(cntrygrid == ci, bfarray==bf)
                    cntrytottrace += np.sum(area[bothinds].ravel()*apriori_flux[source][bothinds].ravel()* \
                                3600*24*365*molarmass)*xouts[source][:,bf]/unit_factor
                    cntrytotprior += np.sum(area[bothinds].ravel()*apriori_flux[source][bothinds].ravel()* \
                                3600*24*365*molarmass)/unit_factor
                cntrymean[source][ci] = np.mean(cntrytottrace)
                cntrymedian[source][ci] = np.median(cntrytottrace)

                if np.nanmax(cntrytottrace) > np.nanmin(cntrytottrace): 
                    xes = np.linspace(np.nanmin(cntrytottrace), np.nanmax(cntrytottrace), 200)           
                    kde = stats.gaussian_kde(cntrytottrace).evaluate(xes)
                    cntrymode[source][ci] = xes[kde.argmax()]
                else:
                    cntrymode[source][ci] = np.mean(cntrytottrace)

                cntrysd[source][ci] = np.std(cntrytottrace)
                cntry68[source][ci, :] = pm.stats.hdi(cntrytottrace, 0.68)
                cntry95[source][ci, :] = pm.stats.hdi(cntrytottrace, 0.95)
                cntryprior[source][ci] = cntrytotprior

            
        #Make output netcdf file
        outds = xr.Dataset({'Yobs':(['nmeasure'], Y),
                            'Yerror' :(['nmeasure'], error),                          
                            'Ytime':(['nmeasure'],Ytime),
                            'Yapriori':(['nmeasure'],Yapriori),
                            'Ymodmean':(['nmeasure'], Ymodmu),
                            'Ymodmedian':(['nmeasure'], Ymodmed),
                            'Ymodmode': (['nmeasure'], Ymodmode), 
                            'Ymod95':(['nmeasure','nUI'], Ymod95),
                            'Ymod68':(['nmeasure','nUI'], Ymod68),
                            'Yoffmean':(['nmeasure'], YmodmuOFF),
                            'Yoffmedian':(['nmeasure'], YmodmedOFF),
                            'Yoffmode':(['nmeasure'], YmodmodeOFF),
                            'Yoff68':(['nmeasure','nUI'], Ymod68OFF),
                            'Yoff95':(['nmeasure','nUI'], Ymod95OFF),
                            'YaprioriBC':(['nmeasure'],YaprioriBC),                            
                            'YmodmeanBC':(['nmeasure'], YmodmuBC),
                            'YmodmedianBC':(['nmeasure'], YmodmedBC),
                            'YmodmodeBC':(['nmeasure'], YmodmodeBC),
                            'Ymod95BC':(['nmeasure','nUI'], Ymod95BC),
                            'Ymod68BC':(['nmeasure','nUI'], Ymod68BC),    
                            'bctrace':(['steps','nBC'],bcouts.values),
                            'sigtrace':(['steps', 'nsigma_site', 'nsigma_time'], sigouts.values),
                            'siteindicator':(['nmeasure'],siteindicator),
                            'sigmafreqindex':(['nmeasure'],sigma_freq_index),
                            'sitenames':(['nsite'],sites),
                            'sitelons':(['nsite'],site_lon),
                            'sitelats':(['nsite'],site_lat),
                            'basisfunctions':(['lat','lon'],bfarray),
                            'countrydefinition':(['lat','lon'], cntrygrid),
                            'bcsensitivity':(['nmeasure', 'nBC'],Hbc.T)},
                        coords={'stepnum' : (['steps'], steps),
                                    f'paranum_'
                                    'numBC' : (['nBC'], nBC),
                                    'measurenum' : (['nmeasure'], nmeasure), 
                                    'UInum' : (['nUI'], nui),
                                    'nsites': (['nsite'], sitenum),
                                    'nsigma_time': (['nsigma_time'], np.unique(sigma_freq_index)),
                                    'nsigma_site': (['nsigma_site'], np.arange(sigouts.shape[1]).astype(int)),
                                    'lat':(['lat'],lat),
                                    'lon':(['lon'],lon),
                                    'countrynames':(['countrynames'],cntrynames)})
        
        for source in emissions_name:
            outds.coords[f'nparam_{source}'] = nparam[source]
            outds[f'xtrace_{source}'] = (['steps',f'nparam_{source}'],xouts[source])
            outds[f'scalingmean_{source}'] = (['lat','lon'],scalemap_mu[source])
            outds[f'scalingmode_{source}'] = (['lat','lon'],scalemap_mode[source])
            outds[f'fluxmode_{source}'] = (['lat','lon'],flux[source])
            outds[f'fluxapriori_{source}'] = (['lat','lon'],apriori_flux[source])
            outds[f'countrymean_{source}'] = (['countrynames'],cntrymean[source])
            outds[f'countrymedian_{source}'] = (['countrynames'],cntrymedian[source])
            outds[f'countrymode_{source}'] = (['countrynames'],cntrymode[source])
            outds[f'countrysd_{source}'] = (['countrynames'],cntrysd[source])
            outds[f'country68_{source}'] = (['countrynames','nUI'],cntry68[source])
            outds[f'country95_{source}'] = (['countrynames','nUI'],cntry95[source])
            outds[f'countryapriori_{source}'] = (['countrynames'],cntryprior[source])
            outds[f'xsensitivity_{source}'] = (['nmeasure',f'nparam_{source}'],Hx[source].T)
            
            outds[f'countrymean_{source}'].attrs["units"] = country_units
            outds[f'countrymedian_{source}'].attrs["units"] = country_units
            outds[f'countrymode_{source}'].attrs["units"] = country_units
            outds[f'country68_{source}'].attrs["units"] = country_units
            outds[f'country95_{source}'].attrs["units"] = country_units
            outds[f'countrysd_{source}'].attrs["units"] = country_units
            outds[f'countryapriori_{source}'].attrs["units"] = country_units
            outds[f'xsensitivity_{source}'].attrs["units"] = obs_units+" "+"mol/mol"
            
            outds[f'fluxapriori_{source}'].attrs["longname"] = "mean a priori flux over period"
            outds[f'fluxmode_{source}'].attrs["longname"] = "mode posterior flux over period"
            outds[f'scalingmean_{source}'].attrs["longname"] = "mean scaling factor field over period"
            outds[f'scalingmode_{source}'].attrs["longname"] = "mode scaling factor field over period"
            outds[f'countrymean_{source}'].attrs["longname"] = "mean of ocean and country totals"
            outds[f'countrymedian_{source}'].attrs["longname"] = "median of ocean and country totals"
            outds[f'countrymode_{source}'].attrs["longname"] = "mode of ocean and country totals"
            outds[f'country68_{source}'].attrs["longname"] = "0.68 Bayesian credible interval of ocean and country totals"
            outds[f'country95_{source}'].attrs["longname"] = "0.95 Bayesian credible interval of ocean and country totals"        
            outds[f'countrysd_{source}'].attrs["longname"] = "standard deviation of ocean and country totals" 
            outds[f'countryapriori_{source}'].attrs["longname"] = "prior mean of ocean and country totals"
            outds[f'xsensitivity_{source}'].attrs["longname"] = "emissions sensitivity timeseries"   
            outds[f'xouts_{source}'].attrs["longname"] = "trace of unitless scaling factors for emissions parameters"
            
            outds.attrs[f'Convergence_{source}'] = convergence[source]
            
        outds.Yobs.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Yapriori.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Ymodmean.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Ymodmedian.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Ymodmode.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Ymod95.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Ymod68.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Yoffmean.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Yoffmedian.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Yoffmode.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Yoff95.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Yoff68.attrs["units"] = obs_units+" "+"mol/mol"
        outds.YmodmeanBC.attrs["units"] = obs_units+" "+"mol/mol"
        outds.YmodmedianBC.attrs["units"] = obs_units+" "+"mol/mol"
        outds.YmodmodeBC.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Ymod95BC.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Ymod68BC.attrs["units"] = obs_units+" "+"mol/mol"
        outds.YaprioriBC.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Yerror.attrs["units"] = obs_units+" "+"mol/mol"
        outds.bcsensitivity.attrs["units"] = obs_units+" "+"mol/mol"
        outds.sigtrace.attrs["units"] = obs_units+" "+"mol/mol"
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
        outds.Yoff68.attrs["longname"] = " 0.68 Bayesian credible interval of posterior simulated offset between measurements"
        outds.Yoff95.attrs["longname"] = " 0.95 Bayesian credible interval of posterior simulated offset between measurements"
        outds.YaprioriBC.attrs["longname"] = "a priori simulated boundary conditions"
        outds.YmodmeanBC.attrs["longname"] = "mean of posterior simulated boundary conditions"
        outds.YmodmedianBC.attrs["longname"] = "median of posterior simulated boundary conditions"
        outds.YmodmodeBC.attrs["longname"] = "mode of posterior simulated boundary conditions"
        outds.Ymod68BC.attrs["longname"] = " 0.68 Bayesian credible interval of posterior simulated boundary conditions"
        outds.Ymod95BC.attrs["longname"] = " 0.95 Bayesian credible interval of posterior simulated boundary conditions"
        outds.bctrace.attrs["longname"] = "trace of unitless scaling factors for boundary condition parameters"
        outds.sigtrace.attrs["longname"] = "trace of model error parameters"
        outds.siteindicator.attrs["longname"] = "index of site of measurement corresponding to sitenames"
        outds.sigmafreqindex.attrs["longname"] = "perdiod over which the model error is estimated"
        outds.sitenames.attrs["longname"] = "site names"
        outds.sitelons.attrs["longname"] = "site longitudes corresponding to site names"
        outds.sitelats.attrs["longname"] = "site latitudes corresponding to site names"
        outds.basisfunctions.attrs["longname"] = "basis function field"
        outds.countrydefinition.attrs["longname"] = "grid definition of countries" 
        outds.bcsensitivity.attrs["longname"] = "boundary conditions sensitivity timeseries"  
        
        outds.attrs['emissions_name'] = emissions_name
        outds.attrs['Start date'] = start_date
        outds.attrs['End date'] = end_date
        outds.attrs['Latent sampler'] = str(step1)[20:33]
        outds.attrs['Hyper sampler'] = str(step2)[20:33]
        outds.attrs['Burn in'] = str(int(burn))
        outds.attrs['Tuning steps'] = str(int(tune))
        outds.attrs['Number of chains'] = str(int(nchain))
        outds.attrs['Error for each site'] = str(sigma_per_site)
        outds.attrs['Emissions Prior'] = ''.join(['{0},{1},'.format(k, v) for k,v in xprior.items()])[:-1]
        outds.attrs['Model error Prior'] = ''.join(['{0},{1},'.format(k, v) for k,v in sigprior.items()])[:-1]
        outds.attrs['BCs Prior'] = ''.join(['{0},{1},'.format(k, v) for k,v in bcprior.items()])[:-1]
        if add_offset:
            outds.attrs['Offset Prior'] = ''.join(['{0},{1},'.format(k, v) for k,v in offsetprior.items()])[:-1]
        outds.attrs['Creator'] = getpass.getuser()
        outds.attrs['Date created'] = str(pd.Timestamp('today'))
        outds.attrs['Repository version'] = code_version()
        
        #comp = dict(zlib=True, complevel=5)
        #encoding = {var: comp for var in outds.data_vars}
        output_filename = define_output_filename(outputpath,species,domain,outputname,start_date,ext=".nc")
        Path(outputpath).mkdir(parents=True, exist_ok=True)
        outds.to_netcdf(output_filename,mode="w")

        #print(outds)

        return outds