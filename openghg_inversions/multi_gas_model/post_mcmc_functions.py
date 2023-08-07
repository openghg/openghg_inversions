#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 7 11:10:00 2023

@author: cv18710

Functions to process and analyse mcmc outputs.

post_to_latlon():
    - Converts array of shape [n_basis_functions] to [lat,lon], based on basis_function file.
            
process_mcmc_traces():
    - Turns MCMC traces into posterior PDFs.
    
produce_model_outputs():
    - Processes posterior PDFs from a MCMC run into posterior fluxes, countryfluxes and 
      modelled observations.
"""

import numpy as np
import xarray as xr
from acrg.name.name import get_country
from acrg.grid.areagrid import areagrid
import openghg_inversions.convert as convert
from openghg_inversions.multi_gas_model import isotopes as isotopes
import glob

def post_to_latlon(post,basis):
    """ 
    Converts array of shape [n_basis_functions] to [lat,lon]
    E.g. to convert posterior mean scaling factors to a spatial grid.
    Inputs:
        post (array): Posterior mean values of the parameter.
        basis (array): Lat/lon grid of basis function values.
    Returns:
        out_grid (array): Post, extrapolated out to shape lat/lon.
    """
    
    out_grid = np.zeros((basis.shape))
    
    for i in range(post.shape[0]):
        out_grid[basis == (i+1)] = post[i]
        
    return out_grid

def process_mcmc_traces(species,sectors,basis,num_basis,post_av,xem_trace,xbc_trace=None,
                        y_sig_trace=None,R_trace=None,Rbc_trace=None,R_trace_allsectors=None,
                        n_trace_samples=100):
    """
    Turns MCMC traces into posterior PDFs.
    Based on create_mcmc_output in data_functions_all.py
    
    Inputs:
        species (list of str):
            List of species.
        sectors (list of list of str):
            Sectors for each species.
        basis (array):
            Lat/lon grid of basis function values.
        num_basis (int):
            Number of basis functions.
        post_av (float):
            Proportion of the MCMC trace used to produce the posterior.
        xem_trace (array):
            Trace of emission ratio scaling factors of shape (n_iter,n_basis*n_sector).
        xbc_trace (dict array):
            Trace of boundary condition scaling factors of shape (n_iter,n_bc) for each gas.
        y_sig_trace (dict of array):
            Trace of model uncertainty of shape (n_iter,n_obs) for each gas.
        R_trace (dict array):
            Trace of emission ratios of shape (n_iter,n_R) for each gas for each sector.
        Rbc_trace (dict array):
            Trace of boundary condition emission ratios of shape (n_iter,n_bc) for each gas.
        R_trace_allsectors (dict array):
            Trace of emission ratios extrapolated out to match the shape of xem_trace.
        n_trace_samples (int):
            Subsampling of the MCMC trace. E.g. if 100, every 100th value is kept to produce the posterior PDF.
    Returns:
        x_post (array):
            Posterior emission scaling factors of shape (n_iter*post_av[::n_trace_samples],n_basis*n_sector).
        x_post_dict (dict of array):
            x_post split into sector.
        x_post_latlon (array):
            Posterior emission scaling factors, extrapolated out to the lat/lon shape of the array.
        x_post_mu (array):
            Posterior mean emissions scaling factors, of shape (n_basis*n_sector).
        x_post_mu_dict (dict of array):
            Posterior mean emission scaling factors, split by sector.
        x_post_std (array):
            Posterior std dev emissions scaling factors, of shape (n_basis*n_sector)
        x_post_25perc (array):
            Posterior 2.5 percentile emissions scaling factors, of shape (n_basis*n_sector)
        x_post_975perc (array):
            Posterior 97.5 percentile emissions scaling factors, of shape (n_basis*n_sector).
        x_prior_latlon (array):
            A priori mean emission scaling factors, extrapolated out to the lat/lon shape of the array.
        x_post_mu_latlon (array):
            Posterior mean emission scaling factors, extrapolated out to the lat/lon shape of the array.
        x_post_std_latlon (array):
            Posterior std dev emission scaling factors, extrapolated out to the lat/lon shape of the array.
        +
        Same set out outputs for xbc, R, Rbc and y_sig.
    """
    
    #---------------------------------------------------------------------------------
    # Posterior x --------------------------------------------------------------------
    
    x_traceout = xem_trace[::n_trace_samples,:]
    x_post = x_traceout[-int(post_av*x_traceout.shape[0]):,:]
        
    x_post_mu = np.mean(x_post,axis=0)
    x_post_std = np.std(x_post,axis=0)
    x_post_25perc = np.percentile(x_post,2.5,axis=0)
    x_post_975perc = np.percentile(x_post,97.5,axis=0)

    x_post_mu_latlon = {se:np.array([]) for se in sectors[0]}
    x_post_std_latlon = {se:np.array([]) for se in sectors[0]}
    x_prior_latlon = {se:np.array([]) for se in sectors[0]}

    x_post_latlon = {se:np.zeros((x_post.shape[0],basis.shape[0],basis.shape[1])) for se in sectors[0]}

    count = 0 
    x_post_mu_dict,x_post_dict = {},{}

    for s,sector_name in enumerate(sectors[0]):

        if s == 0:
            for i in range(x_post.shape[0]):
                x_post_latlon[sector_name][i] = post_to_latlon(x_post[i,:num_basis],basis)
            x_post_mu_dict[sector_name] = x_post_mu[:num_basis]
            x_post_dict[sector_name] = x_post[:,:num_basis]
            x_prior_latlon[sector_name] = post_to_latlon(xem_trace[0,:num_basis],basis)
            x_post_mu_latlon[sector_name] = post_to_latlon(x_post_mu[:num_basis],basis)
            x_post_std_latlon[sector_name] = post_to_latlon(x_post_std[:num_basis],basis)
        else:
            for i in range(x_post.shape[0]):
                x_post_latlon[sector_name][i] = post_to_latlon(x_post[i,num_basis*count:num_basis*(count+1)],basis)
            x_post_mu_dict[sector_name] = x_post_mu[num_basis*count:num_basis*(count+1)]
            x_post_dict[sector_name] = x_post[:,num_basis*count:num_basis*(count+1)]
            x_prior_latlon[sector_name] = post_to_latlon(xem_trace[0,num_basis*count:num_basis*(count+1)],basis)
            x_post_mu_latlon[sector_name] = post_to_latlon(x_post_mu[num_basis*count:num_basis*(count+1)],basis)
            x_post_std_latlon[sector_name] = post_to_latlon(x_post_std[num_basis*count:num_basis*(count+1)],basis)
        
        count += 1
        
    #---------------------------------------------------------------------------------
    # Posterior xbc --------------------------------------------------------------------
    
    if xbc_trace is not None:
        
        xbc_traceout,xbc_post,xbc_post_mu,xbc_post_std = {},{},{},{}
        xbc_post_25perc,xbc_post_975perc = {},{}
        
        for s,s_name in enumerate(xbc_trace.keys()):
            
            xbc_traceout[s_name] = xbc_trace[s_name][::n_trace_samples,:]
            xbc_post[s_name] = xbc_traceout[s_name][-int(post_av*xbc_traceout[s_name].shape[0]):,:]
            
            xbc_post_mu[s_name] = np.mean(xbc_post[s_name],axis=0)
            xbc_post_std[s_name] = np.std(xbc_post[s_name],axis=0)
            xbc_post_25perc[s_name] = np.percentile(xbc_post[s_name],2.5,axis=0)
            xbc_post_975perc[s_name] = np.percentile(xbc_post[s_name],97.5,axis=0)
            
    else:
        
        xbc_traceout,xbc_post,xbc_post_mu,xbc_post_std = None,None,None,None
        xbc_post_25perc,xbc_post_975perc = None,None
        
    #---------------------------------------------------------------------------------
    # Posterior R -------------------------------------------------------------------- 
        
    if R_trace is not None:
        
        R_traceout,R_post = {sp:{} for sp in species[1:]},{sp:{} for sp in species[1:]}
        R_traceout_allsectors,R_post_allsectors = {sp:{} for sp in species[1:]},{sp:{} for sp in species[1:]}
        
        R_post_mu_allsectors = {sp:np.empty([0,0]) for sp in species[1:]}
        R_post_mu,R_post_std = {sp:{} for sp in species[1:]},{sp:{} for sp in species[1:]}
        R_post_25perc,R_post_975perc = {sp:{} for sp in species[1:]},{sp:{} for sp in species[1:]}
        R_post_mu_latlon = {sp:{} for sp in species[1:]}
        
        for s,s_name in enumerate(R_trace.keys()):
            
            R_traceout_allsectors[s_name] = R_trace_allsectors[s_name][::n_trace_samples,:]
            R_post_allsectors[s_name] = R_traceout_allsectors[s_name][-int(post_av*R_traceout_allsectors[s_name].shape[0]):,:]
            
            for a,sector_name in enumerate(R_trace[s_name].keys()):
                R_traceout[s_name][sector_name] = R_trace[s_name][sector_name][::n_trace_samples,:]
                R_post[s_name][sector_name] = R_traceout[s_name][sector_name][-int(post_av*R_traceout[s_name][sector_name].shape[0]):,:]
            
                R_post_mu[s_name][sector_name] = np.mean(R_post[s_name][sector_name],axis=0)
                R_post_std[s_name][sector_name] = np.std(R_post[s_name][sector_name],axis=0)
                R_post_25perc[s_name][sector_name] = np.percentile(R_post[s_name][sector_name],2.5,axis=0)
                R_post_975perc[s_name][sector_name] = np.percentile(R_post[s_name][sector_name],97.5,axis=0)
                
                if R_post[s_name][sector_name].shape[1] != num_basis:
                    
                    R_post_mu_latlon[s_name][sector_name] = post_to_latlon(np.ones(num_basis) * R_post_mu[s_name][sector_name],
                                                                             basis)
                    
                    #if a == 0:
                    #    R_post_mu_allsectors[s_name] = np.ones((R_post[s_name][sector_name].shape[0],num_basis)) * R_post[s_name][sector_name][0]
                    #else:
                    #    R_post_mu_allsectors[s_name] = np.hstack((R_post_mu_allsectors[s_name],
                    #                                    np.ones((R_post[s_name][sector_name].shape[0],num_basis)) * R_post[s_name][sector_name][0]))
                else:
                    
                    R_post_mu_latlon[s_name][sector_name] = post_to_latlon(R_post_mu[s_name][sector_name],basis)
                    
                    #if a == 0:
                    #    R_post_mu_allsectors[s_name] = R_post[s_name][sector_name]
                    #else:
                    #    R_post_mu_allsectors[s_name] = np.hstack((R_post_mu_allsectors[s_name],R_post[s_name][sector_name]))
                        
            R_post_mu_allsectors[s_name] = np.mean(R_post_allsectors[s_name],axis=0)
    else:
        
        R_traceout,R_post,R_post_mu,R_post_std = None,None,None,None
        R_post_mu_allsectors,R_traceout_allsectors,R_post_allsectors = None,None,None
        R_post_25perc,R_post_975perc,R_post_mu_latlon = None,None,None
                
    #---------------------------------------------------------------------------------
    # Posterior Rbc ------------------------------------------------------------------
    
    if Rbc_trace is not None:
        
        Rbc_traceout,Rbc_post,Rbc_post_mu,Rbc_post_std = {},{},{},{}
        Rbc_post_25perc,Rbc_post_975perc = {},{}
        
        for s,s_name in enumerate(Rbc_trace.keys()):
            
            Rbc_traceout[s_name] = Rbc_trace[s_name][::n_trace_samples,:]
            Rbc_post[s_name] = Rbc_traceout[s_name][-int(post_av*Rbc_traceout[s_name].shape[0]):,:]
            
            Rbc_post_mu[s_name] = np.mean(Rbc_post[s_name],axis=0)
            Rbc_post_std[s_name] = np.std(Rbc_post[s_name],axis=0)
            Rbc_post_25perc[s_name] = np.percentile(Rbc_post[s_name],2.5,axis=0)
            Rbc_post_975perc[s_name] = np.percentile(Rbc_post[s_name],97.5,axis=0)
            
    else:
        
        Rbc_traceout,Rbc_post,Rbc_post_mu,Rbc_post_std = None,None,None,None
        Rbc_post_25perc,Rbc_post_975perc = None,None
        
    #---------------------------------------------------------------------------------
    # Posterior sig_y ----------------------------------------------------------------
    
    if y_sig_trace is not None:
        
        y_sig_traceout,y_sig_post,y_sig_post_mu,y_sig_post_std = {},{},{},{}
        y_sig_post_25perc,y_sig_post_975perc = {},{}
        
        for s,s_name in enumerate(y_sig_trace.keys()):
        
            y_sig_traceout[s_name] = y_sig_trace[s_name][::n_trace_samples,:]
            y_sig_post[s_name] = y_sig_traceout[s_name][-int(post_av*y_sig_traceout[s_name].shape[0]):,:]

            y_sig_post_mu[s_name] = np.mean(y_sig_post[s_name],axis=0)
            y_sig_post_std[s_name] = np.std(y_sig_post[s_name],axis=0)
            y_sig_post_25perc[s_name] = np.percentile(y_sig_post[s_name],2.5,axis=0)
            y_sig_post_975perc[s_name] = np.percentile(y_sig_post[s_name],97.5,axis=0)
        
    else:

        y_sig_post,y_sig_post_mu,y_sig_post_std,y_sig_post_25perc,y_sig_post_975perc = None,None,None,None,None
        
    return (x_post,x_post_dict,x_post_latlon,x_post_mu,x_post_mu_dict,x_post_std,x_post_25perc,x_post_975perc,
            x_prior_latlon,x_post_mu_latlon,x_post_std_latlon,
            xbc_post,xbc_post_mu,xbc_post_std,xbc_post_25perc,xbc_post_975perc,
            R_post,R_post_allsectors,R_post_mu_allsectors,R_post_mu,R_post_std,R_post_25perc,R_post_975perc,
            Rbc_post,Rbc_post_mu,Rbc_post_std,Rbc_post_25perc,Rbc_post_975perc,
            R_post_mu_latlon,
            y_sig_post,y_sig_post_mu,y_sig_post_std,y_sig_post_25perc,y_sig_post_975perc)

def produce_model_outputs(n_trace_samples,domain,species,species_type,sectors_dict,basis,lat,lon,
                         fp_data_H_all,x_post_latlon,x_post_mu_latlon,x_prior_latlon,
                         y_mod_trace,xem_trace,x_post,H,y_times,sites,sites_array_all,
                         use_bc=True,xbc_trace=None,xbc_post=None,Hbc=None,
                         R_trace_allsectors=None,R_post_allsectors=None,
                         Rbc_trace=None,Rbc_post=None,
                         fp_data_H_prior_all=None,countrymask=None,countries=None):
    """
    Processes posterior PDFs from a MCMC run into posterior fluxes, countryfluxes and 
    modelled observations.
    Based on process_mcmc_output in data_functions_all.py
    Inputs:
        n_trace_samples (int):
            Subsampling of the MCMC trace. E.g. if 100, every 100th value is kept to produce the posterior PDF.
        domain (str):
            Study domain.
        species (list of str):
            List of species.
        species_type (list of str):
            Type of each species, e.g. ['mf','delta_value'].
        sectors_dict (dict of list of str):
            Sectors for each species, e.g. {'ch4':['FF','nonFF],'c2h6':['FF',None]}.
        basis (array):
            Lat/lon grid of basis function values.
        lat,lon (arrays):
            Latitudes and longtides of domain.
        fp_data_H_all (dict of datasets):
            All merged fp_data_H objects, for each gas.
        x_post_latlon (array):
            Posterior trace of emissions scaling factors, of shape (num_post_iter,lat,lon).
        x_post_mu_latlon (array):
            Posterior mean emissions scaling factors, extrapolated out to the whole domain lat/lon shape.
        x_prior_latlon (array):
            A priori mean emissions scaling factors, extrapolated out to the whole domain lat/lon shape.
        y_mod_trace (dict of arrays):
            Modelled y from the MCMC trace.
        xem_trace (array):
            Full emissions scaling factor trace, of shape (n_iter,n_basis*n_sector).
        x_post (array):
            Posterior trace of emissions scaling factors, of shape  (n_post_iter,n_basis*n_sector).
        H (dict of array):
            Merged footprints and emissions for each gas.
        sites (list of list of str):
            List of observation sites for each gas.
        sites_array_all (dict of array):
            Corresponding site for each observation.
        use_bc (bool):
            If True, use boundary conditions.
        xbc_trace (array):
            Full boundary conditions scaling factor trace, of shape (n_iter,n_bc).
        xbc_post (array):
            Posterior trace of boundary conditions scaling factors, of shape  (n_post_iter,n_bc).
        Hbc (dict of array):
            Merged footprints and boundary conditions for each gas.
        R_trace_allsectors (dict array):
            Trace of emission ratios extrapolated out to match the shape of xem_trace for each gas.
        R_post_allsectors (dict array):
            Posterior trace of emission ratios extrapolated out to match the shape of xem_trace for each gas.
        Rbc_trace (dict of arrays):
            Full boundary conditions ratios trace, of shape (n_iter,n_bc) for each gas.
        Rbc_post (dict of arrays):
            Posterior trace of boundary condition emission ratios, of shape  (n_post_iter,n_bc) for each gas.
        fp_data_H_prior_all (dict of arrays):
            Merged fp_data_H object for each gas, using a priori emissions.
        countrymask (str):
            Countrymask file over which to estimate country fluxes.
        countries (list of str):
            If specified, only estimates emissions for this list of countries.
    Outputs:
        flux_apriori (dict of arrays):
            A priori mean fluxes for each sector for each mf gas.
        flux_post_mu (dict of arrays):
            Posterior mean fluxes for each sector for each mf gas.
        c_out (list of str):
            Countrynames for which country fluxes were calculated.
        countrygrid (array):
            Lat/lon grid of country definitions from countrymask file.
        country_prior (dict of dict of arrays):
            A priori fluxes for each country for each sector for each gas.
        country_post (dict of dict of arrays):
            Posterior trace of fluxes for each country for each sector for each gas, of shape (n_post_iter,n_countries).
        country_post_mu (dict of dict of arrays):
            Posterior mean fluxes for each country for each sector for each gas.
        country_post_25perc,country_post_975perc,country_post_std (dict of dict of arrays):
            Posterior std dev, 2.5 and 97.5 percentiles of fluxes for each country for each sector for each gas.
        country_total_prior (dict of arrays):
            Total a priori fluxes for each country for each gas.
        country_total_post (dict of arrays):
            Posterior trace of total a priori flux for each country for each gas, of shape (n_post_iter,n_countries).
        country_total_post_mu (dict of arrays):
            Posterior mean of total a priori fluxes for each country for each gas.
        country_total_post_25perc,country_total_post_975perc,country_total_post_std (dict of arrays):
            Posterior std dev, 2.5 and 97.5 percentiles of the total a priori flux for each country for each gas.
        y_prior (dict of arrays):
            A priori modelled observations for each gas.
        y_post (dict of arrays):
            Posterior trace of modelled observations for each gas, of shape (n_post_iter,n_obs).
        y_post_mu (dict of arrays):
            Posterior mean modelled observations for each gas.
        y_post_25perc,y_post_975perc,y_post_std (dict of arrays):
            Posterior std dev, 2.5 and 97.5 percentiles of modelled observations for each gas.
        ybc_prior,ybc_post,ybc_post_mu,ybc_post_25perc,ybc_post_975perc,ybc_post_std:
            Same as above, for the modelled boundary condition component of each total observation.
    """
    
    mf_species = [species[i] for i,e in enumerate(species_type) if e == 'mf']
    delta_species = [species[i] for i,e in enumerate(species_type) if e == 'delta_value']
    num_post_iter = x_post_latlon[sectors_dict[species[0]][0]].shape[0]
    
    #---------------------------------------------------------------------------------
    # Posterior fluxes ---------------------------------------------------------------
    
    flux_apriori,flux_post_mu = {sp:{} for sp in species},{sp:{} for sp in species}
    
    for s_name in species:
        for sector_name in sectors_dict[s_name]:
            
            if fp_data_H_prior_all[s_name] is not None:
                flux_apriori[s_name][sector_name] = fp_data_H_prior_all[s_name]['.flux'][f'{s_name}_{sector_name}'].flux.values[:,:,0]
            else:
                flux_apriori[s_name][sector_name] = fp_data_H_all[s_name]['.flux'][f'{s_name}_{sector_name}'].flux.values[:,:,0]
            if sector_name is not None:
                flux_post_mu[s_name][sector_name] = x_post_mu_latlon[sector_name] * flux_apriori[s_name][sector_name]
            
    #---------------------------------------------------------------------------------
    # Posterior country fluxes -------------------------------------------------------
    
    # I HAVE NOT INCLUDED CALCULATION OF BASIS FLUXES HERE, THIS WAS INCLUDED IN THE OLDER CODE VERSION
    
    if domain == 'EUROPE' and countrymask == None:
        countrymask = '/user/home/cv18710/work_shared/LPDM/countries/country-ukmo_EUROPE.nc'
        print(f'\nUsing countrymask: {countrymask}.')
    
    c_object = get_country(domain, country_file=countrymask)
    countryds = xr.Dataset({'country': (['lat','lon'], c_object.country), 
                            'name' : (['ncountries'],c_object.name) },
                                            coords = {'lat': (c_object.lat),
                                            'lon': (c_object.lon)})
    countrynames = countryds.name.values
    countrygrid = countryds.country.values
    c_names_nums_dict = dict(zip(countrynames,np.arange(0,countrynames.shape[0]+1)))
    
    area = areagrid(lat,lon)
    s_in_y = 3600 * 24 * 365
    '''
    prior_basis_flux = {sp:{} for sp in mf_species}
    
    for s_name in mf_species:
        
        molmass = convert.molar_mass(s_name)
        unit_modifier = convert.prefix('T') # hardcoded at Tg here, but could add options for different units later
        
        for sector_name in sectors_dict[s_name]:
            
            prior_grid = post_to_latlon(x_prior_latlon[sector_name],basis)
            
            for bf in range(int(np.max(basis)+1)):
                basis_cells = np.where(basis == (bf+1))
                prior_basis_flux[sector_name][bf] = np.nansum(prior_grid[basis_cells] * 
                                                              flux_apriori[s_name][sector_name][basis_cells] * 
                                                              area[basis_cells] * s_in_y * molmass) / unit_modifier
    '''
    if countries == None:
        print(f'\nCalculating posterior country fluxes for all countries in {countrymask}.')
        c_out = countrynames
    else:
        c_out = np.array(countries)
        print(f'Calculating countryfluxes for {c_out} in {countrymask}.')
        
    n_c = len(c_out)

    country_post = {sp:{se:np.zeros((num_post_iter,n_c)) for se in sectors_dict[sp]} for sp in mf_species}
    country_total_post = {sp:np.zeros((num_post_iter,n_c)) for sp in mf_species}

    country_total_post_mu = {sp:np.zeros(n_c) for sp in mf_species}
    country_total_post_25perc = {sp:np.zeros(n_c) for sp in mf_species}
    country_total_post_975perc = {sp:np.zeros(n_c) for sp in mf_species}
    country_total_post_std = {sp:np.zeros(n_c) for sp in mf_species}
    country_total_prior = {sp:np.zeros(n_c) for sp in mf_species}

    country_post_mu = {sp:{se:np.zeros(n_c) for se in sectors_dict[sp]} for sp in mf_species}
    country_post_25perc = {sp:{se:np.zeros(n_c) for se in sectors_dict[sp]} for sp in mf_species}
    country_post_975perc = {sp:{se:np.zeros(n_c) for se in sectors_dict[sp]} for sp in mf_species}
    country_post_std = {sp:{se:np.zeros(n_c) for se in sectors_dict[sp]} for sp in mf_species}
    country_prior = {sp:{se:np.zeros(n_c) for se in sectors_dict[sp]} for sp in mf_species}
    
    for s_name in mf_species:
    
        molmass = convert.molar_mass(s_name)
        unit_modifier = convert.prefix('T') # hardcoded at Tg here, but could add options for different units later
        
        for sector_name in sectors_dict[s_name]:
            if sector_name is not None:
            
                for c,country in enumerate(c_out):
                    
                    for bf in range(int(np.max(basis)+1)):
                    
                        country_and_basis = np.logical_and(countrygrid==c_names_nums_dict[country],basis==bf)
                        
                        country_prior[s_name][sector_name][c] += np.sum(x_prior_latlon[sector_name][country_and_basis] * 
                                                            flux_apriori[s_name][sector_name][country_and_basis] * 
                                                            area[country_and_basis] * s_in_y * molmass) / unit_modifier
                        
                        for i in range(num_post_iter):
                            
                            country_post[s_name][sector_name][i,c] += np.sum(x_post_latlon[sector_name][i][country_and_basis] * 
                                                                            flux_apriori[s_name][sector_name][country_and_basis] * 
                                                                            area[country_and_basis] * s_in_y * molmass) / unit_modifier
                            
                    country_total_prior[s_name][c] += country_prior[s_name][sector_name][c]
                                                
                    country_post_mu[s_name][sector_name][c] = np.mean(country_post[s_name][sector_name][:,c],axis=0)
                    country_post_25perc[s_name][sector_name][c] = np.percentile(country_post[s_name][sector_name][:,c],2.5,axis=0)
                    country_post_975perc[s_name][sector_name][c] = np.percentile(country_post[s_name][sector_name][:,c],97.5,axis=0)
                    country_post_std[s_name][sector_name][c] = np.std(country_post[s_name][sector_name][:,c],axis=0)

    for s_name in mf_species:
        for c,country in enumerate(c_out):
            
            for sector_name in sectors_dict[s_name]:
                if sector_name is not None:
                    for i in range(num_post_iter):
                        country_total_post[s_name][i,c] += country_post[s_name][sector_name][i,c]
            
            country_total_post_mu[s_name][c] = np.mean(country_total_post[s_name][:,c],axis=0)
            country_total_post_25perc[s_name][c] = np.percentile(country_total_post[s_name][:,c],2.5,axis=0)
            country_total_post_975perc[s_name][c] = np.percentile(country_total_post[s_name][:,c],97.5,axis=0)
            country_total_post_std[s_name][c] = np.std(country_total_post[s_name][:,c],axis=0)
            
    #---------------------------------------------------------------------------------
    # Posterior y --------------------------------------------------------------------
    
    y_prior,y_post,y_post_mu,y_post_25perc,y_post_975perc,y_post_std = {},{},{},{},{},{}
    
    for s,s_name in enumerate(species):
        
        y_post[s_name] = y_mod_trace[s_name][::n_trace_samples,:][x_post.shape[0]:,:]
        if species_type[s] == 'mf':
            y_prior[s_name] = np.sum(H[s_name],axis=1)
            if use_bc == True:
                y_prior[s_name] += np.sum(Hbc[s_name],axis=1)
        else:
            y_prior[s_name] = y_mod_trace[s_name][0,:]
            
        
        y_post_mu[s_name] = np.mean(y_post[s_name],axis=0)
        y_post_std[s_name] = np.std(y_post[s_name],axis=0)
        y_post_25perc[s_name] = np.percentile(y_post[s_name],2.5,axis=0)
        y_post_975perc[s_name] = np.percentile(y_post[s_name],97.5,axis=0)
            
    if use_bc == True:
        
        ybc_prior,ybc_post_mu,ybc_post_25perc,ybc_post_975perc,ybc_post_std = {},{},{},{},{}
        ybc_post = {sp:np.zeros((num_post_iter,Hbc[sp].shape[0])) for sp in species}
        
        for s,s_name in enumerate(species):
            
            if species_type[s] == 'mf':
                
                ybc_prior[s_name] = np.matmul(Hbc[s_name],xbc_trace[s_name][0,:])
                
                for i in range(num_post_iter):
                    ybc_post[s_name][i] = np.matmul(Hbc[s_name],xbc_post[s_name][i,:])
                
            elif species_type[s] == 'delta_value':
                
                R_sample,Rbc_sample = {},{}
                
                for a,sp in enumerate(species[1:]):
                    R_sample[sp] = R_trace_allsectors[sp][0,:]
                    if species_type[a+1] == 'delta_value':
                        Rbc_sample[sp] = Rbc_trace[sp][0,:]
                    
                y_mod0,y_mod2,ybc_mod0,\
                ybc_prior[s_name] = isotopes.modelled_ch4_delta_obs(s_name,R_sample,xem_trace[0,:],
                                                                    H[species[0]],sites[0],sites[s],
                                                                    sites_array_all[species[0]],sites_array_all[s_name],
                                                                    y_times[species[0]],y_times[s_name],
                                                                    delta_bc_sample=Rbc_sample,
                                                                    xbc_sample=xbc_trace[species[0]][0,:],
                                                                    Hbc_sample=Hbc[species[0]],
                                                                    return_bc_separately=True)
                
                for i in range(num_post_iter):
                    
                    R_sample,Rbc_sample = {},{}
                    
                    for a,sp in enumerate(species[1:]):
                        R_sample[sp] = R_post_allsectors[sp][i,:]
                        if species_type[a+1] == 'delta_value':
                            Rbc_sample[sp] = Rbc_post[sp][0,:]
                    
                    y_mod0,y_mod2,ybc_mod0,\
                    ybc_post[s_name][i] = isotopes.modelled_ch4_delta_obs(s_name,R_sample,x_post[i,:],
                                                                        H[species[0]],sites[0],sites[s],
                                                                    sites_array_all[species[0]],sites_array_all[s_name],
                                                                        y_times[species[0]],y_times[s_name],
                                                                        delta_bc_sample=Rbc_sample,
                                                                        xbc_sample=xbc_post[species[0]][i,:],
                                                                        Hbc_sample=Hbc[species[0]],
                                                                        return_bc_separately=True)
        
            ybc_post_mu[s_name] = np.mean(ybc_post[s_name],axis=0)
            ybc_post_std[s_name] = np.std(ybc_post[s_name],axis=0)
            ybc_post_25perc[s_name] = np.percentile(ybc_post[s_name],2.5,axis=0)
            ybc_post_975perc[s_name] = np.percentile(ybc_post[s_name],97.5,axis=0)
        
    else:
        ybc_prior,ybc_post,ybc_post_mu = None,None,None
        ybc_post_25perc,ybc_post_975perc,ybc_post_std = None,None,None
        return_bc_separately = False
                   
    return (flux_apriori,flux_post_mu,c_out,countrygrid,
            country_prior,country_post,country_post_mu,country_post_25perc,country_post_975perc,country_post_std,
            country_total_prior,country_total_post,country_total_post_mu,
            country_total_post_25perc,country_total_post_975perc,country_total_post_std,
            y_prior,y_post,y_post_mu,y_post_25perc,y_post_975perc,y_post_std,
            ybc_prior,ybc_post,ybc_post_mu,ybc_post_25perc,ybc_post_975perc,ybc_post_std)


def recalculate_small_area_flux_wrapper(basis,countrygrid,country_index,species,area,x_post,flux_apriori,
                                        x_post2=None,flux_apriori2=None):
    """
    Used to recalculate country fluxes from a smaller/different area, after the inverse model has been run.
    Can be run for single x_post values (e.g. x_prior) rather than a whole trace of values, by inputting
    x_post with an extra dimention: np.expand_dims(f['x_post_nonFF'].values[0,:],0) 
    Inputs:
        basis (array):
            Lat, lon grid of basis functions used in the inversion.
        countrygrid (array):
            New country mask used to recalculate fluxes.
        country_index (int,float,bool):
            Value in countrygrid that you want to estimate fluxes over.
        species (str):
            Gas species.
        area (array):
            Lat, lon grid of area per cell. Can be found using acrg.areagrid.areagrid(lat,lon).
        x_post (array):
            Trace of posterior x scaling values, of shape (n_post_iterations).
        flux_apriori (array):
            Prior fluxes that are scaled by x_post, of shape (lat,lon).
        x_post2 (array) (optional):
            Trace of posterior x scaling values, for a second sector, if you want to calculate total fluxes.
        flux_apriori2 (array) (optional):
            Prior fluxes that are scaled by x_post2.
    Returns:
        country_mean, country_25perc, country_975perc: The mean 2.5th and 97.5th percentiles of the trace
                                                        of 'country' fluxes.
    """
    
    s_in_y = 3600 * 24 * 365
    unit_modifier = convert.prefix('T')
    molmass = convert.molar_mass(species)
    
    country_post_in = np.zeros(x_post.shape[0])
    
    x_post_latlon = np.zeros((x_post.shape[0],basis.shape[0],basis.shape[1]))
    for i in range(x_post.shape[0]):
        x_post_latlon[i,:,:] = post_to_latlon(x_post[i],basis)
        
    if x_post2 is not None:
        x_post_latlon2 = np.zeros((x_post2.shape[0],basis.shape[0],basis.shape[1]))
        for i in range(x_post2.shape[0]):
            x_post_latlon2[i,:,:] = post_to_latlon(x_post2[i],basis)
    else:
        x_post_latlon2 = None
    
    country_post = recalculate_small_area_flux(basis,countrygrid,country_index,area,x_post_latlon,flux_apriori,
                                               country_post_in,s_in_y,unit_modifier,molmass,
                                               x_post_latlon2=x_post_latlon2,flux_apriori2=flux_apriori2)
    
    country_mean = np.round(np.mean(country_post),4)
    country_25perc = np.round(np.percentile(country_post,2.5),4)
    country_975perc = np.round(np.percentile(country_post,97.5),4)
    
    return country_mean,country_25perc,country_975perc

def recalculate_small_area_flux(basis,countrygrid,country_index,area,x_post_latlon,flux_apriori,
                                country_post_in,s_in_y,unit_modifier,molmass,
                                x_post_latlon2=None,flux_apriori2=None):
    """
    Loopy part of recalculate_small_area_flux that could potentially be sped up using numba.
    Need to work on this.
    """
    
    for bf in range(int(np.max(basis)+1)):
            
        country_and_basis = np.logical_and(countrygrid==country_index,basis==bf)
    
        for i in range(x_post_latlon.shape[0]):
            
            country_post_in[i] += np.sum(x_post_latlon[i,:,:][country_and_basis] * flux_apriori[country_and_basis] * 
                                area[country_and_basis] * s_in_y * molmass) / unit_modifier
            
            if x_post_latlon2 is not None:
                
                country_post_in[i] += np.sum(x_post_latlon2[i,:,:][country_and_basis] * flux_apriori2[country_and_basis] * 
                                area[country_and_basis] * s_in_y * molmass) / unit_modifier
        
    return country_post_in

def recalc_country_post(filepath):
    
    with xr.open_dataset(filepath) as f:
        
        num_post_iter = np.shape(f.n_post_iter.values)[0]
        basis = f['fp_basis'].values
        countrygrid = f['countrymask'].values
        x_post = f['x_post'].values
        flux_apriori = {'FF':f['flux_apriori_ch4_FF'].values,
                        'nonFF':f['flux_apriori_ch4_nonFF'].values}
        lat = f.lat.values
        lon = f.lon.values
        
    area = areagrid(lat,lon)
        
    x_post_latlon = {'FF':np.zeros((x_post.shape[0],basis.shape[0],basis.shape[1])),
                     'nonFF':np.zeros((x_post.shape[0],basis.shape[0],basis.shape[1]))}
    
    for i in range(x_post.shape[0]):
        x_post_latlon['FF'][i,:,:] = post_to_latlon(x_post[i,:int(np.max(basis))],basis)
        x_post_latlon['nonFF'][i,:,:] = post_to_latlon(x_post[i,int(np.max(basis)):],basis)
    
    country_post = {'FF':np.zeros((num_post_iter)),
                    'nonFF':np.zeros((num_post_iter))}
    country_total_post = np.zeros((num_post_iter))
    
    s_in_y = 3600 * 24 * 365
    molmass = convert.molar_mass('ch4')
    unit_modifier = convert.prefix('T') # hardcoded at Tg here, but could add options for different units later
    
    for sector_name in ['FF','nonFF']:

        for bf in range(int(np.max(basis)+1)):
        
            country_and_basis = np.logical_and(countrygrid==19,basis==bf)
            
            for i in range(num_post_iter):
                
                country_post[sector_name][i] += np.sum(x_post_latlon[sector_name][i,:,:][country_and_basis] * 
                                                                flux_apriori[sector_name][country_and_basis] * 
                                                                area[country_and_basis] * s_in_y * molmass) / unit_modifier

        for i in range(num_post_iter):
            country_total_post[i] += country_post[sector_name][i]

    return country_post,country_total_post

def recalculate_small_area_flux_run():
    """
    Runs recalulate_small_area_flux_wrapper for a range of output files.
    New country masks need to be created using the function in create_flux_bc_R_priors.ipynb.
    """
    
    country_type = 'hfd'

    with xr.open_dataset('/user/home/cv18710/work/LPDM/countries/country-ukmo-hfd-only_EUROPE.nc') as f:
        country = f.country.values
        lat = f.lat.values
        lon = f.lon.values
        
    area = areagrid(lat,lon)
    
    post_dir = '/user/home/cv18710/work/posterior/'
    '''
    output_types = {
                'model4':f'methane_d13_dD_varR_real_data_filtered',
                'model5':f'methane_d13_dD_varR_real_data_filtered_flat_sector_prior',
                'model6':f'methane_d13_dD_varR_real_data_filtered_flat_prior'}

    output_dirs = {
               'model4': post_dir + 'methane_d13_dD/real_data_filtered/100_basis/',
               'model5': post_dir + 'methane_d13_dD/real_data_flat_sector_prior/obs-filtering_var-sigma/',
               'model6': post_dir + 'methane_d13_dD/real_data_flat_prior/obs-filtering_var-sigma/'}

    output_names = {
                'model4': output_dirs['model4'] + '*varR*',
                'model5': output_dirs['model5'] + '*varR*',
                'model6': output_dirs['model6'] + '*varR*'}
    
    
    output_types = {'model1':'methane_only_real_data_filtered',
                'model2':f'methane_only_real_data_filtered_flat_sector_prior',
                'model3':f'methane_only_real_data_filtered_flat_prior',
                'model4':f'methane_d13_dD_varR_real_data_filtered',
                'model5':f'methane_d13_dD_varR_real_data_filtered_flat_sector_prior',
                'model6':f'methane_d13_dD_varR_real_data_filtered_flat_prior'}

    output_dirs = {'model1': post_dir + 'methane_only/real_data_filtered/',
               'model2': post_dir + 'methane_only/real_data_flat_sector_prior/ch4-filtering_var-ysig/',
               'model3': post_dir + 'methane_only/real_data_flat_prior/obs-filtering_var-sigma/',
               'model4': post_dir + 'methane_d13_dD/real_data_filtered/100_basis/',
               'model5': post_dir + 'methane_d13_dD/real_data_flat_sector_prior/obs-filtering_var-sigma/',
               'model6': post_dir + 'methane_d13_dD/real_data_flat_prior/obs-filtering_var-sigma/'}
    
    output_names = {'model1': output_dirs['model1'] + '*100basis*',
                'model2': output_dirs['model2'] + '*100basis*',
                'model3': output_dirs['model3'] + '*100basis*',
                'model4': output_dirs['model4'] + '*varR*',
                'model5': output_dirs['model5'] + '*varR*',
                'model6': output_dirs['model6'] + '*varR*'}
    '''
    output_types = {#'model1':'methane_only_real_data_filtered',
                #'model2':f'methane_d13-dD-fixedR_real_data_filtered',
                'model3':f'methane_d13-dD-varR_real_data_filtered',
                #'model4':f'methane_c2h6-fixedR_real_data_filtered',
                #'model5':f'methane_c2h6-varR_real_data_filtered',
                #'model6':f'methane_d13-dD-c2h6-fixedR_real_data_filtered',
                #'model7':f'methane_d13-dD-c2h6-varR_real_data_filtered'
                }
    
    output_dirs = {#'model1': post_dir + 'methane_only/real_data_filtered/',
               #'model2': post_dir + 'methane_d13_dD/real_data_filtered/',
               'model3': post_dir + 'methane_d13_dD/real_data_filtered/',
               #'model4': post_dir + 'methane_ethane/real_data_filtered/',
               #'model5': post_dir + 'methane_ethane/real_data_filtered/',
               #'model6': post_dir + 'methane_d13_dD_ethane/real_data_filtered/',
               #'model7': post_dir + 'methane_d13_dD_ethane/real_data_filtered/'
               }

    output_names = {#'model1': output_dirs['model1'] + '*100basis*',
                #'model2': output_dirs['model2'] + '*fixedR*',
                'model3': output_dirs['model3'] + '*varR*',
                #'model4': output_dirs['model4'] + '*fixedR*',
                #'model5': output_dirs['model5'] + '*varR*',
                #'model6': output_dirs['model6'] + '*fixedR*',
                #'model7': output_dirs['model7'] + '*varR*'
                }
    
    model_names = list(output_dirs.keys())
        
    n_models = len(model_names)

    output_files = {}

    for m in model_names:
        output_files[m] = sorted(glob.glob(output_names[m]+'_post.nc'))
        
    n_months = len(output_files[model_names[0]])

    month_list = {m: np.array([]).astype('datetime64[ns]') for m in model_names}
    country_prior_FF = {m: np.array([]) for m in model_names}
    country_prior_nonFF = {m: np.array([]) for m in model_names}
    country_prior_total = {m: np.array([]) for m in model_names}
    country_post_FF = {m: np.array([]) for m in model_names}
    country_post_nonFF = {m: np.array([]) for m in model_names}
    country_post_total = {m: np.array([]) for m in model_names}
    country_25perc_FF = {m: np.array([]) for m in model_names}
    country_25perc_nonFF = {m: np.array([]) for m in model_names}
    country_25perc_total = {m: np.array([]) for m in model_names}
    country_975perc_FF = {m: np.array([]) for m in model_names}
    country_975perc_nonFF = {m: np.array([]) for m in model_names}
    country_975perc_total = {m: np.array([]) for m in model_names}

    for m in model_names:
        for file in output_files[m]:
            print(file)
            with xr.open_dataset(file) as f:
                month_list[m] = np.hstack((month_list[m],f['y_times_ch4'].values[0]))
                country_prior_FF_i,country_prior_FF_ia,country_prior_FF_ib = recalculate_small_area_flux_wrapper(f['fp_basis'].values,country,
                                                                                                19.0,'ch4',area,
                                                                                                np.expand_dims(np.ones(f['x_post_mu_FF'].values.shape[0]),0),
                                                                                                f['flux_apriori_ch4_FF'].values)
                country_prior_FF[m] = np.hstack((country_prior_FF[m],country_prior_FF_i))
                
                country_prior_nonFF_i,country_prior_nonFF_ia,country_prior_nonFF_ib = recalculate_small_area_flux_wrapper(f['fp_basis'].values,country,
                                                                                                19.0,'ch4',area,
                                                                                                np.expand_dims(np.ones(f['x_post_mu_nonFF'].values.shape[0]),0),
                                                                                                f['flux_apriori_ch4_nonFF'].values)
                country_prior_nonFF[m] = np.hstack((country_prior_nonFF[m],country_prior_nonFF_i))
                
                country_prior_total_i,country_prior_total_ia,country_prior_total_ib = recalculate_small_area_flux_wrapper(f['fp_basis'].values,country,
                                                                                                19.0,'ch4',area,
                                                                                                np.expand_dims(np.ones(f['x_post_mu_FF'].values.shape[0]),0),
                                                                                                f['flux_apriori_ch4_FF'].values,
                                                                                                np.expand_dims(np.ones(f['x_post_mu_nonFF'].values.shape[0]),0),
                                                                                                f['flux_apriori_ch4_nonFF'].values)
                country_prior_total[m] = np.hstack((country_prior_total[m],country_prior_total_i))
                
                country_post_FF_i,country_25perc_FF_i,country_975perc_FF_i = recalculate_small_area_flux_wrapper(f['fp_basis'].values,country,
                                                                                                19.0,'ch4',area,
                                                                                                f['x_post_FF'].values,
                                                                                                f['flux_apriori_ch4_FF'].values)
                country_post_FF[m] = np.hstack((country_post_FF[m],country_post_FF_i))
                country_25perc_FF[m] = np.hstack((country_25perc_FF[m],country_25perc_FF_i))
                country_975perc_FF[m] = np.hstack((country_975perc_FF[m],country_975perc_FF_i))
                
                country_post_nonFF_i,country_25perc_nonFF_i,country_975perc_nonFF_i = recalculate_small_area_flux_wrapper(f['fp_basis'].values,country,
                                                                                                19.0,'ch4',area,
                                                                                                f['x_post_nonFF'].values,
                                                                                                f['flux_apriori_ch4_nonFF'].values)
                country_post_nonFF[m] = np.hstack((country_post_nonFF[m],country_post_nonFF_i))
                country_25perc_nonFF[m] = np.hstack((country_25perc_nonFF[m],country_25perc_nonFF_i))
                country_975perc_nonFF[m] = np.hstack((country_975perc_nonFF[m],country_975perc_nonFF_i))
                
                country_post_total_i,country_25perc_total_i,country_975perc_total_i = recalculate_small_area_flux_wrapper(f['fp_basis'].values,country,
                                                                                                19.0,'ch4',area,
                                                                                                f['x_post_FF'].values,
                                                                                                f['flux_apriori_ch4_FF'].values,
                                                                                                f['x_post_nonFF'].values,
                                                                                                f['flux_apriori_ch4_nonFF'].values)
                country_post_total[m] = np.hstack((country_post_total[m],country_post_total_i))
                country_25perc_total[m] = np.hstack((country_25perc_total[m],country_25perc_total_i))
                country_975perc_total[m] = np.hstack((country_975perc_total[m],country_975perc_total_i))
                
    all_model_names = ['prior'] + model_names
    
    for m in model_names:
        output_ds = xr.Dataset({'country_post_FF':(['month'],country_post_FF[m]),
                                'country_post_nonFF':(['month'],country_post_nonFF[m]),
                                'country_post_total':(['month'],country_post_total[m]),
                                'country_25perc_FF':(['month'],country_25perc_FF[m]),
                                'country_25perc_nonFF':(['month'],country_25perc_nonFF[m]),
                                'country_25perc_total':(['month'],country_25perc_total[m]),
                                'country_975perc_FF':(['month'],country_975perc_FF[m]),
                                'country_975perc_nonFF':(['month'],country_975perc_nonFF[m]),
                                'country_975perc_total':(['month'],country_975perc_total[m]),
                                'country_prior_FF':(['month'],country_prior_FF[m]),
                                'country_prior_nonFF':(['month'],country_prior_nonFF[m]),
                                'country_prior_total':(['month'],country_prior_total[m]),
                                },
                            coords={'months':(['months'],month_list[m].astype('datetime64[M]').astype('datetime64[D]').astype(str))})
    
        output_ds.attrs['country_mask_type'] = country_type
        
        output_ds.to_netcdf(f'/user/home/cv18710/work/posterior/countryfluxes_small_areas/{output_types[m]}_{country_type}.nc')
    
def main():
    recalculate_small_area_flux_run()
    
if __name__ == "__main__":
    main()