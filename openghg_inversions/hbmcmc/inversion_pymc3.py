# *****************************************************************************
# Created: 7 Nov. 2022
# Author: Eric Saboya, School of Geographical Sciences, University of Bristol
# Contact: eric.saboya@bristol.ac.uk
# *****************************************************************************
# About
#   Originally created by Luke Western (ACRG) and updated, here, by Eric Saboya
#   Functions for performing MCMC inversion.
# *****************************************************************************

import numpy as np
import pymc3 as pm
import pandas as pd
import xarray as xr
import getpass
from pathlib import Path

from openghg.retrieve import get_flux

from openghg_inversions import convert 
from openghg_inversions import utils
from openghg_inversions.hbmcmc.inversionsetup import opends, offset_matrix
from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename
from openghg_inversions.config.version import code_version

#from openghg_inversions.config.paths import Paths
#data_path = Paths.data

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

def inferpymc3(Hx, Hbc, Y, error, siteindicator, sigma_freq_index,
               xprior={"pdf":"lognormal", "mu":1, "sd":1},
               bcprior={"pdf":"lognormal", "mu":0.004, "sd":0.02},
               sigprior={"pdf":"uniform", "lower":0.5, "upper":3},
               nit=2.5e5, burn=50000, tune=1.25e5, nchain=2, 
               sigma_per_site = True, 
               offsetprior={"pdf":"normal", "mu":0, "sd":1},
               add_offset = False, verbose=False):       
    '''
    Uses PyMC3 module for Bayesian inference for emissions field, boundary 
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
    '''
    burn = int(burn)         
    
    hx = Hx.T 
    hbc = Hbc.T
    nx = hx.shape[1]
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
        x = parseprior("x", xprior, shape=nx)
        xbc = parseprior("xbc", bcprior, shape=nbc)
        sig = parseprior("sig", sigprior, shape=(nsites, nsigmas))
        if add_offset:
            offset = parseprior("offset", offsetprior, shape=nsites-1) 
            offset_vec = pm.math.concatenate( (np.array([0]), offset), axis=0)
            mu = pm.math.dot(hx,x) + pm.math.dot(hbc,xbc) + pm.math.dot(B, offset_vec)
        else:
            mu = pm.math.dot(hx,x) + pm.math.dot(hbc,xbc)       
        epsilon = pm.math.sqrt(error**2 + sig[sites, sigma_freq_index]**2)
        y = pm.Normal('y', mu = mu, sd=epsilon, observed=Y, shape = ny)
        
        step1 = pm.NUTS(vars=[x,xbc])
        step2 = pm.Slice(vars=[sig])
        
        trace = pm.sample(nit, tune=int(tune), chains=nchain,
                          step=[step1,step2], progressbar=verbose, cores=nchain)#step=pm.Metropolis())#  #target_accept=0.8,
        
        outs = trace.get_values(x, burn=burn)[0:int((nit)-burn)]
        bcouts = trace.get_values(xbc, burn=burn)[0:int((nit)-burn)]
        sigouts = trace.get_values(sig, burn=burn)[0:int((nit)-burn)]
        
        #Check for convergence
        gelrub = pm.rhat(trace)['x'].max()
        if gelrub > 1.05:
            print('Failed Gelman-Rubin at 1.05')
            convergence = "Failed"
        else:
            convergence = "Passed"
        
        if add_offset:
            offset_outs = trace.get_values(offset, burn=burn)[0:int((nit)-burn)]
            offset_trace = np.hstack([np.zeros((int(nit-burn),1)), offset_outs])
            YBCtrace = np.dot(Hbc.T,bcouts.T) + np.dot(B, offset_trace.T)   
        else:
            YBCtrace = np.dot(Hbc.T,bcouts.T)
        Ytrace = np.dot(Hx.T,outs.T) + YBCtrace
        
        return outs, bcouts, sigouts, Ytrace, YBCtrace, convergence, step1, step2

def inferpymc3_postprocessouts(xouts,bcouts, sigouts, convergence, 
                               Hx, Hbc, Y, error, Ytrace, YBCtrace,
                               step1, step2, 
                               xprior, bcprior, sigprior, offsetprior, Ytime, siteindicator, sigma_freq_index,
                               domain, species, sites,
                               start_date, end_date, outputname, outputpath,
                               country_unit_prefix,
                               burn, tune, nchain, sigma_per_site,
                               emissions_name, fp_data=None, 
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
        '''
        
        print("Post-processing output")
         
        #Get parameters for output file 
        nit = xouts.shape[0]
        nx = Hx.shape[0]
        ny = len(Y)
        nbc = Hbc.shape[0]
        nui = np.arange(2)
        steps = np.arange(nit)
        nmeasure = np.arange(ny)
        nparam = np.arange(nx)
        nBC = np.arange(nbc)
        #YBCtrace = np.dot(Hbc.T,bcouts.T)
        YmodBC = np.mean(YBCtrace, axis=1)
        Ymod95BC = pm.stats.hdi(YBCtrace.T, 0.95)
        Ymod68BC = pm.stats.hdi(YBCtrace.T, 0.68)
        YaprioriBC = np.sum(Hbc, axis=0)
        Ymod = np.mean(Ytrace, axis=1)
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
            bfds = fp_data[".basis"]


        #Calculate mean posterior scale map and flux field
        scalemap = np.zeros_like(bfds.values)
        
        for npm in nparam:
            scalemap[bfds.values == (npm+1)] = np.mean(xouts[:,npm]) 
        
        if rerun_file is not None:
            # Note, at the moment fluxapriori in the output is the mean apriori flux over the 
            # inversion period and so will not be identical to the original a priori flux, if 
            # it varies over the inversion period
            emissions_flux = np.expand_dims(rerun_file.fluxapriori.values,2)
        else:
            if emissions_name == None:
                raise NameError("Emissions name not provided.")
            else:
                emds=get_flux(species=species,
                              domain=domain,
                              source=emissions_name[0],
                              start_date=start_date,
                              end_date=end_date)

            emissions_flux = emds.data.flux.values
        flux = scalemap*emissions_flux[:,:,0]
        
        #Basis functions to save
        bfarray = bfds.values-1
    
        #Calculate country totals   
        area = utils.areagrid(lat, lon)
        if not rerun_file:
            c_object = utils.get_country(domain, country_file=country_file)
            cntryds = xr.Dataset({'country': (['lat','lon'], c_object.country), 
                            'name' : (['ncountries'],c_object.name) },
                                            coords = {'lat': (c_object.lat),
                                            'lon': (c_object.lon)})
            cntrynames = cntryds.name.values
            cntrygrid = cntryds.country.values
        else:
            cntrynames = rerun_file.countrynames.values
            cntrygrid = rerun_file.countrydefinition.values
        cntrymean = np.zeros((len(cntrynames)))
        cntry68 = np.zeros((len(cntrynames), len(nui)))
        cntry95 = np.zeros((len(cntrynames), len(nui)))
        cntrysd = np.zeros(len(cntrynames))
        cntryprior = np.zeros(len(cntrynames))
        molarmass = convert.molar_mass(species)

        unit_factor = convert.prefix(country_unit_prefix)
        if country_unit_prefix is None:
            country_unit_prefix=''
        country_units = country_unit_prefix + 'g'
        if rerun_file is not None:
            obs_units = rerun_file.Yobs.attrs["units"].split(" ")[0]
        else:
            obs_units = str(fp_data[".units"])

        # Not sure how it's best to do this if multiple months in emissions 
        # file. Now it scales a weighted average of a priori emissions
        # If a priori emissions have frequency of more than monthly then this
        # needs chaning.
        aprioriflux = np.zeros_like(area)
        if emissions_flux.shape[2] > 1:
            print("Assuming the inversion is over a year or less and emissions file is monthly")
            allmonths = pd.date_range(start_date, end_date).month[:-1].values
            allmonths -= np.min(allmonths)
            for mi in allmonths:
                aprioriflux += emissions_flux[:,:,mi]*np.sum(allmonths == mi)/len(allmonths)
        else:
            aprioriflux = np.squeeze(emissions_flux)
        for ci, cntry in enumerate(cntrynames):
            cntrytottrace = np.zeros(len(steps))
            cntrytotprior = 0
            for bf in range(int(np.max(bfarray))+1):
                bothinds = np.logical_and(cntrygrid == ci, bfarray==bf)
                cntrytottrace += np.sum(area[bothinds].ravel()*aprioriflux[bothinds].ravel()* \
                               3600*24*365*molarmass)*xouts[:,bf]/unit_factor
                cntrytotprior += np.sum(area[bothinds].ravel()*aprioriflux[bothinds].ravel()* \
                               3600*24*365*molarmass)/unit_factor
            cntrymean[ci] = np.mean(cntrytottrace)
            cntrysd[ci] = np.std(cntrytottrace)
            cntry68[ci, :] = pm.stats.hdi(cntrytottrace, 0.68)
            cntry95[ci, :] = pm.stats.hdi(cntrytottrace, 0.95)
            cntryprior[ci] = cntrytotprior
            
    
        #Make output netcdf file
        outds = xr.Dataset({'Yobs':(['nmeasure'], Y),
                            'Yerror' :(['nmeasure'], error),                          
                            'Ytime':(['nmeasure'],Ytime),
                            'Yapriori':(['nmeasure'],Yapriori),
                            'Ymodmean':(['nmeasure'], Ymod), 
                            'Ymod95':(['nmeasure','nUI'], Ymod95),
                            'Ymod68':(['nmeasure','nUI'], Ymod68),
                            'YaprioriBC':(['nmeasure'],YaprioriBC),                            
                            'YmodmeanBC':(['nmeasure'], YmodBC),
                            'Ymod95BC':(['nmeasure','nUI'], Ymod95BC),
                            'Ymod68BC':(['nmeasure','nUI'], Ymod68BC),                        
                            'xtrace':(['steps','nparam'], xouts),
                            'bctrace':(['steps','nBC'],bcouts),
                            'sigtrace':(['steps', 'nsigma_site', 'nsigma_time'], sigouts),
                            'siteindicator':(['nmeasure'],siteindicator),
                            'sigmafreqindex':(['nmeasure'],sigma_freq_index),
                            'sitenames':(['nsite'],sites),
                            'sitelons':(['nsite'],site_lon),
                            'sitelats':(['nsite'],site_lat),
                            'fluxapriori':(['lat','lon'], aprioriflux), #NOTE this is the mean a priori flux over the inversion period
                            'fluxmean':(['lat','lon'], flux),                            
                            'scalingmean':(['lat','lon'],scalemap),
                            'basisfunctions':(['lat','lon'],bfarray),
                            'countrymean':(['countrynames'], cntrymean),
                            'countrysd':(['countrynames'], cntrysd),
                            'country68':(['countrynames', 'nUI'],cntry68),
                            'country95':(['countrynames', 'nUI'],cntry95),
                            'countryapriori':(['countrynames'],cntryprior),
                            'countrydefinition':(['lat','lon'], cntrygrid),
                            'xsensitivity':(['nmeasure','nparam'], Hx.T),
                            'bcsensitivity':(['nmeasure', 'nBC'],Hbc.T)},
                        coords={'stepnum' : (['steps'], steps), 
                                   'paramnum' : (['nlatent'], nparam),
                                   'numBC' : (['nBC'], nBC),
                                   'measurenum' : (['nmeasure'], nmeasure), 
                                   'UInum' : (['nUI'], nui),
                                   'nsites': (['nsite'], sitenum),
                                   'nsigma_time': (['nsigma_time'], np.unique(sigma_freq_index)),
                                   'nsigma_site': (['nsigma_site'], np.arange(sigouts.shape[1]).astype(int)),
                                   'lat':(['lat'],lat),
                                   'lon':(['lon'],lon),
                                   'countrynames':(['countrynames'],cntrynames)})
        
        outds.fluxmean.attrs["units"] = "mol/m2/s"
        outds.fluxapriori.attrs["units"] = "mol/m2/s"
        outds.Yobs.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Yapriori.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Ymodmean.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Ymod95.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Ymod68.attrs["units"] = obs_units+" "+"mol/mol"
        outds.YmodmeanBC.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Ymod95BC.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Ymod68BC.attrs["units"] = obs_units+" "+"mol/mol"
        outds.YaprioriBC.attrs["units"] = obs_units+" "+"mol/mol"
        outds.Yerror.attrs["units"] = obs_units+" "+"mol/mol"
        outds.countrymean.attrs["units"] = country_units
        outds.country68.attrs["units"] = country_units
        outds.country95.attrs["units"] = country_units
        outds.countrysd.attrs["units"] = country_units
        outds.countryapriori.attrs["units"] = country_units
        outds.xsensitivity.attrs["units"] = obs_units+" "+"mol/mol"
        outds.bcsensitivity.attrs["units"] = obs_units+" "+"mol/mol"
        outds.sigtrace.attrs["units"] = obs_units+" "+"mol/mol"
        
        outds.Yobs.attrs["longname"] = "observations"
        outds.Yerror.attrs["longname"] = "measurement error"
        outds.Ytime.attrs["longname"] = "time of measurements"
        outds.Yapriori.attrs["longname"] = "a priori simulated measurements"
        outds.Ymodmean.attrs["longname"] = "mean of posterior simulated measurements"
        outds.Ymod68.attrs["longname"] = " 0.68 Bayesian credible interval of posterior simulated measurements"
        outds.Ymod95.attrs["longname"] = " 0.95 Bayesian credible interval of posterior simulated measurements"
        outds.YaprioriBC.attrs["longname"] = "a priori simulated boundary conditions"
        outds.YmodmeanBC.attrs["longname"] = "mean of posterior simulated boundary conditions"
        outds.Ymod68BC.attrs["longname"] = " 0.68 Bayesian credible interval of posterior simulated boundary conditions"
        outds.Ymod95BC.attrs["longname"] = " 0.95 Bayesian credible interval of posterior simulated boundary conditions"
        outds.xtrace.attrs["longname"] = "trace of unitless scaling factors for emissions parameters"
        outds.bctrace.attrs["longname"] = "trace of unitless scaling factors for boundary condition parameters"
        outds.sigtrace.attrs["longname"] = "trace of model error parameters"
        outds.siteindicator.attrs["longname"] = "index of site of measurement corresponding to sitenames"
        outds.sigmafreqindex.attrs["longname"] = "perdiod over which the model error is estimated"
        outds.sitenames.attrs["longname"] = "site names"
        outds.sitelons.attrs["longname"] = "site longitudes corresponding to site names"
        outds.sitelats.attrs["longname"] = "site latitudes corresponding to site names"
        outds.fluxapriori.attrs["longname"] = "mean a priori flux over period"
        outds.fluxmean.attrs["longname"] = "mean posterior flux over period"
        outds.scalingmean.attrs["longname"] = "mean scaling factor field over period"
        outds.basisfunctions.attrs["longname"] = "basis function field"
        outds.countrymean.attrs["longname"] = "mean of ocean and country totals"
        outds.country68.attrs["longname"] = "0.68 Bayesian credible interval of ocean and country totals"
        outds.country95.attrs["longname"] = "0.95 Bayesian credible interval of ocean and country totals"        
        outds.countrysd.attrs["longname"] = "standard deviation of ocean and country totals" 
        outds.countryapriori.attrs["longname"] = "prior mean of ocean and country totals"
        outds.countrydefinition.attrs["longname"] = "grid definition of countries"
        outds.xsensitivity.attrs["longname"] = "emissions sensitivity timeseries"   
        outds.bcsensitivity.attrs["longname"] = "boundary conditions sensitivity timeseries"  
        
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
        outds.attrs['Convergence'] = convergence
        outds.attrs['Repository version'] = code_version()
        
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in outds.data_vars}
        output_filename = define_output_filename(outputpath,species,domain,outputname,start_date,ext=".nc")
        Path(outputpath).mkdir(parents=True, exist_ok=True)
        outds.to_netcdf(output_filename, encoding=encoding, mode="w")
