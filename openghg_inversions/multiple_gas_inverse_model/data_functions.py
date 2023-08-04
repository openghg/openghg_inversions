#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 15 Feb 08:45:00 2023

@author: cv18710

Functions for processing and merging data for multi-gas inverse model runs. 
Uses functions from the ACRG code respository.

mcmc_input_errors_warnings()
    - Stops inverse model at the start of the run and prints errors if inputs are 
      not in the correct format.
    - Prints warnings about model setup choices.
    
merge_fp_data_flux_bc()
    - Extracts observations, corresponding footprints, fluxes and boundary conditions,
      creates sensitivity matrix H by merging footprints and fluxes. Option to save the 
      merged data as a pickle file.
      
filtering()
    - Filtering of observations (and associated merged footprints etc.) with options for 
      localness, PBLH (for both site and aircraft obs) and height.
      
extract_from_merged()
    - Extract and stack data from the merged footprints, prior and observations.
    
create_sig_index()
    - Creates index values used to group timeseries into sections with the same model error.
    
correlated_noise()
    - Creates correlated noise for synthetic observations from the same sites.
    
create_sites_array()
    - Creates an array with entries giving the site for each observation.
      Used to form the model-measurement covariance matrix.
      
calc_time_correlated_uncertainty()
    - Fills in off-diagonal terms of the model-measurement covariance matrix with
      time-correlated uncertainty.
      
find_covariance()
    - Creates example covariance for synthetic data tests with CH4 and 12CH4.
    
create_model_measurement_uncertainty_matrix()
    - Combines model and measurement errors to produce the obs uncertainty matrix.
    
create_mcmc_inputs()
    - Extracts inputs from the merged fp_data_H object, creates pseudo observations (if needed)
      and uncertainty matrices.
      
"""

import numpy as np
import pandas as pd
import numba
import xarray as xr
import os
import glob
import pickle
import io
from acrg.obs import read
from acrg.name import name
from acrg.name import basis_functions as basis_func
import mcmc_functions as mcmc
import isotopes as isotopes

def mcmc_input_errors(species,sectors,domain,countrymask,countries,sites,
                      y_sig_mu,y_cov,y_sig_pdf,y_sig_range,y_sig_freq,post_save_dir):
    
    errors = []
    warnings = []
    
    ### Countrymask and countrynames -------------------------------------------------
    
    if countrymask == None:
        if domain != 'EUROPE':
            errors.append(f'You must specify a countrymask for domain {domain}.')
        else:
            countrymask = '/user/home/cv18710/work_shared/LPDM/countries/country-ukmo_EUROPE.nc'

    c_object = name.get_country(domain, country_file=countrymask)
    countryds = xr.Dataset({'country': (['lat','lon'], c_object.country), 
                            'name' : (['ncountries'],c_object.name) },
                                            coords = {'lat': (c_object.lat),
                                            'lon': (c_object.lon)})
    countrynames = countryds.name.values
    
    if countries is not None:
        for c in countries:
            if c not in countrynames:
                errors.append(f'Countryname {c} not in {countrymask}.')
                
    if countries == None:
        warnings.append('\nWARNING: Solving for countryfluxes for all countries in countrymask.')
        warnings.append('This may be slow for large countrymask files.') 
        
    ### Other -------------------------------------------------------------------------
    
    if y_sig_pdf is not None:
        
        for s,site_iter in enumerate(sites):
            
            if len(site_iter) != len(y_sig_mu[s]):
                errors.append(f'Length of y_sig_mu does not match length of sites for {species[s]}.')
                
        if len(sites) != len(y_sig_range):
            errors.append(f'Length of y_sig_range does not match length of species.')
                
        if y_sig_freq != 'monthly':
            warnings.append(('\nWARNING: y_sig_freq keyword has not been fully tested.'+
                            'This may not work for non-continuous observations.'))
    
    if len(species) != len(sectors):
        errors.append(f'Length of sectors does not match species length.')
        
    if y_cov == True:
        for s,s_name in species[1:]:
            if sites[0][0] != sites[s+1][0]:
                errors.append((f'When using off-diagonal covariances, site lists for '+
                               '{species[0]} and {species[s+1]} should begin with the same site.'))
                
    if os.path.exists(post_save_dir) == False:
        errors.append(f'Output directory: {post_save_dir} does not exist.')
    
    return errors,warnings

def merge_fp_data_flux_bc(sites,heights,network,species,domain,start_date,end_date,
                          average,sectors,flux_name,
                          obs_dir,fp_dir,flux_dir,fp_basis_savename=None,fp_basis_dir=None,nquadtreebasis=None,
                          bc_dir=None,bc_basis_dir=None,bc_basis_name=None,use_bc=False,
                          save_dir=None,save_name=None,fp_basis_search_name=None,species_name=None):
    
    ''' 
    Extracts observations for given time period using read.get_obs.
    Merges observations,footprints and fluxes to produce sensitivity matrix H.
    Option to save the merged data as a pickle file.
    Returns merged data as a dataset.
    Assumes use of UKV footprints for all sites for EUROPE.
    species_name (str, optional) is used in pseudo data mode for delta value secondary gases,
                                 to name the fluxes correctly.
    '''
    
    #-------------------------------------------------------------------------
    # Observations -----------------------------------------------------------
    
    data = read.get_obs(sites=sites,species=species,start_date=start_date,end_date=end_date,
                        average=average,network=network,inlet=heights,keep_missing=False,
                        data_directory=obs_dir)
    
    #-------------------------------------------------------------------------
    # Setup ------------------------------------------------------------------
    
    #remove keys of sites with no data
    for i,site in enumerate(sites):
        if data[site] is None:
            del data[site]
    
    if heights is not None:
        heights_dict = dict(zip(sites,heights[:-1]))
    else:
        heights_dict = None
    
    if domain == 'EUROPE':
        met_model = 'UKV'
    else:
        met_model = None
        
    emissions_name,fp_basis_case = {},{}
        
    for s,sector in enumerate(sectors):
        emissions_name[f'{species_name}_{sector}'] = flux_name[s]
        fp_basis_case[f'{species_name}_{sector}'] = fp_basis_search_name
    
    #-------------------------------------------------------------------------
    # Merge fp, obs, flux etc ------------------------------------------------
    
    if use_bc is True:
        
        fp_data = name.footprints_data_merge(data,domain=domain,emissions_name=emissions_name,
                                             fp_directory=fp_dir,flux_directory=flux_dir,bc_directory=bc_dir,
                                             #site_modifier=site_modifier,
                                             met_model=met_model,
                                             calc_timeseries=True,load_flux=True,
                                             load_bc=True,calc_bc=True)
        
        # Create basis func file if it doesn't exist
        if nquadtreebasis is not None:
            if 'quadtree' not in fp_basis_search_name:
                print(f"Basis case {fp_basis_search_name} supplied but nquadtreebasis has also been set.")
                print(f"Assuming you want to use {fp_basis_search_name}.")
                
            elif len(glob.glob(os.path.join(fp_basis_dir,domain,f'{fp_basis_search_name}_EUROPE_{start_date[:4]}.nc'))) == 0:
                print(f'No file named {fp_basis_search_name}*.nc in {fp_basis_dir}, so creating basis function file.')

                basis = basis_func.quadtreebasisfunction(emissions_name=emissions_name,
                                                        fp_all=fp_data,sites=sites,
                                                        start_date=start_date,domain=domain,
                                                        species=species,
                                                        outputname=fp_basis_savename,
                                                        outputdir=fp_basis_dir,
                                                        nbasis=nquadtreebasis)
    
        fp_data_H = name.fp_sensitivity(fp_and_data=fp_data,domain=domain,
                                            basis_case=fp_basis_case,
                                            basis_directory=fp_basis_dir)

        fp_data_H = name.bc_sensitivity(fp_and_data=fp_data_H,domain=domain,basis_case=bc_basis_name,
                                            bc_basis_directory=bc_basis_dir)
            
        if save_name is not None:
            fp_out = open(save_dir+save_name+'_H.pickle','wb')
            pickle.dump(fp_data_H,fp_out)
            fp_out.close()

            print(f'\n fp_data_H saved in {save_dir}')
    
        return fp_data_H     

    else:
        
        fp_data = name.footprints_data_merge(data,domain=domain,emissions_name=emissions_name,
                                             fp_directory=fp_dir,flux_directory=flux_dir,
                                             met_model=met_model,
                                             calc_timeseries=True,load_flux=True,
                                             load_bc=False,calc_bc=False)
        
        # Create basis func file if it doesn't exist
        if nquadtreebasis is not None:
            if 'quadtree' not in fp_basis_search_name:
                print(f"Basis case {fp_basis_search_name} supplied but nquadtreebasis has also been set.")
                print(f"Assuming you want to use {fp_basis_search_name}.")
                
            elif len(glob.glob(os.path.join(fp_basis_dir,domain,f'{fp_basis_search_name}*.nc'))) == 0:
                print(f'No file named {fp_basis_search_name}*.nc in {fp_basis_dir}, so creating basis function file.')

                basis = basis_func.quadtreebasisfunction(emissions_name=emissions_name,
                                                        fp_all=fp_data,sites=sites,
                                                        start_date=start_date,domain=domain,
                                                        species=species,
                                                        outputname=fp_basis_savename,
                                                        outputdir=fp_basis_dir,
                                                        nbasis=nquadtreebasis)
        
        fp_data_H = name.fp_sensitivity(fp_and_data=fp_data,domain=domain,
                                            basis_case=fp_basis_case,
                                            basis_directory=fp_basis_dir)
            
        if save_name is not None:
            fp_out = open(save_dir+save_name+'_H.pickle','wb')
            pickle.dump(fp_data_H,fp_out)
            fp_out.close()

            print(f'\n fp_data_H saved in {save_dir}\n')
    
        return fp_data_H
    
def filtering(fp_data_H,sites,heights,species,filtering_types,obs_dir,
              network=None,secondary_heights=None,
              start_date=None,end_date=None,average=None):
    """
    Filters observations in fp_data_H dataset, to account for local influence
    and PBLH thresholds.
    Based on Mark Lunt's code and advice from Alistair Manning.
    Input variables after 'filtering_types' only required for height filtering.
    
    filtering_types = ['localness','pblh','height']
    """
    
    @numba.jit()
    def assign_localness(fp_data_H_site_fp,wh_rlat,wh_rlon):
            
        local_sum=np.zeros(fp_data_H_site_fp.shape[2])
            
        for ti in range(fp_data_H_site_fp.shape[2]):
            
            fp_data_H_site_local = fp_data_H_site_fp[wh_rlat[0]-2:wh_rlat[0]+3,
                                                     wh_rlon[0]-2:wh_rlon[0]+3,ti]
            
            local_sum[ti] = np.sum(fp_data_H_site_local)/np.sum(fp_data_H_site_fp[:,:,ti])
        
        return local_sum
        
    def define_localness(fp_data_H,site):
        """
        Define the localness of each time point for each site.
        Sum up the 25 grid boxes surrounding the site.
        """
    
        release_lon=fp_data_H[site].release_lon[0].values
        release_lat=fp_data_H[site].release_lat[0].values
            
        dlon=fp_data_H[site].lon[1].values-fp_data_H[site].lon[0].values
        dlat=fp_data_H[site].lat[1].values-fp_data_H[site].lat[0].values
            
        wh_rlon = np.where(abs(fp_data_H[site].lon.values-release_lon) < dlon/2.)[0]
        wh_rlat = np.where(abs(fp_data_H[site].lat.values-release_lat) < dlat/2.)[0]
        
        fp_data_H_site_fp = fp_data_H[site].fp.values

        local_sum = assign_localness(fp_data_H_site_fp,wh_rlat,wh_rlon)
    
        local_ds = xr.Dataset({'local_ratio': (['time'], local_sum)},
                                coords = {'time' : (fp_data_H[site].coords['time'])})
                
        fp_data_H[site] = fp_data_H[site].merge(local_ds)
            
        return fp_data_H
    
    def localness_filter(dataset,site,keep_missing=False):
        """
        Subset for times when local influence is below threshold.       
        Local influence expressed as a fraction of the sum of entire footprint domain.
        """
        
        pc = 0.1      #localness 'ratio' limit
        lr = dataset.local_ratio
        
        ti = [i for i, local_ratio in enumerate(lr) if local_ratio <= pc]
        if keep_missing is True: 
            mf_data_array = dataset.mf            
            dataset_temp = dataset.drop('mf')

            dataarray_temp = mf_data_array[dict(time = ti)]   

            mf_ds = xr.Dataset({'mf': (['time'], dataarray_temp)}, 
                                  coords = {'time' : (dataarray_temp.coords['time'])})

            dataset_out = name.combine_datasets(dataset_temp, mf_ds, method=None)
            
            return dataset_out
        
        else:
            
            return dataset[dict(time = ti)]
    
    def pblh_filter(dataset,site,keep_missing=False):
        """
        Subset for times when boundary layer height > threshold.
        Threshold needs to be set in dataset as pblh_threshold.
        Only works for sites as height is variable for aicraft.
        """
        
        threshold = dataset.pblh_threshold.values
        ti = [i for i, pblh in enumerate(dataset.PBLH) if pblh > threshold]
        
        if keep_missing:
            
            mf_data_array = dataset.mf
            
            dataset_temp = dataset.drop('mf')
            dataarray_temp = mf_data_array[dict(time = ti)]
            
            mf_ds = xr.Dataset({'mf': (['time'], dataarray_temp)},
                                   coords = {'time' : (dataarray_temp.coords['time'])})
            
            dataset_out = name.combine_datasets(dataset_temp, mf_ds, method=None)
            
            return dataset_out
        
        else:
            
            return dataset[dict(time = ti)]   
        
    def pblh_aircraft_filter(dataset):
        """
        Subset for times when boundary layer height > aircraft height.
        """
        
        ti = [i for i,pblh in enumerate(dataset.PBLH) if pblh > dataset['sample_alt'].values[i]]
        
        ''' 
        # for testing if NAME BL height matches obs BL flags
        flags = [dataset['bl_flag'].values[i] for i,pblh in enumerate(dataset.PBLH) if pblh > dataset['sample_alt'].values[i]]

        n_2 = len([i for i in flags if i == 2.0])
        n_not2 = len([i for i in flags if i != 2.0])
        
        perc_match = n_not2/(n_not2+n_2)*100
        print(f' No. of BLH that do not match: {n_2}')
        print(f' % of BLH that match: {perc_match}')
        '''
        
        return dataset[dict(time=ti)]
        
    def height_filter(fp_data_H,site,network,heights,secondary_heights,
                     start_date,end_date,average,species,obs_dir):
        """
        Filters obs by comparing mf observed from multiple inlets at the same height.
        If mole fractions observed at all heights are within 20 ppm (ppb for C2H6) 
        then the observation is kept.
        No filtering applied to sites with only one height.
        """

        fp_data_H_out = fp_data_H.copy()

        if secondary_heights is not None:

            obs1 = fp_data_H.mf.values
            times1 = fp_data_H.time.values

            with io.capture_output() as captured:

                data = read.get_obs(sites=[site],species=species,start_date=start_date,end_date=end_date,
                                average=average,network=[network],inlet=[secondary_heights],keep_missing=False,
                                data_directory=obs_dir)

            obs2 = data[site][0].mf.values
            times2 = data[site][0].time.values

            keep_index = []

            for t,time in enumerate(times1):
                for t2,time2 in enumerate(times2):
                    if time == time2:
                        if np.abs(obs1[t] - obs2[t2]) < 20.:
                            keep_index.append(t)

            fp_data_H_out = fp_data_H[dict(time=keep_index)]  

        else:
            print(f'{site} obs not filtered with height comparison as no second inlet height provided.')

        return fp_data_H_out
    
    n_obs = []
    n_obs_filtered = []
        
    for i,site in enumerate(sites):
        print(f'{site}:')
        
        n_obs.append(fp_data_H[site].mf.values.shape[0])
        
        if len(filtering_types) == len(sites):
            
            filter_site = filtering_types[i]
            
        else:
            filter_site = filtering_types
        
        if 'localness' in filter_site:
            print('Applying localness filtering')
            print('WARNING: localness filter should only be used for static sites, not moving instruments.')
        
            fp_data_H = define_localness(fp_data_H,site)
        
            fp_data_H[site] = localness_filter(fp_data_H[site],site)
        
        if 'pblh_site' in filter_site:
            print('Applying PBLH filtering')

            fp_data_H[site]["pblh_threshold"] = max(int(heights[i][:-1])+100,250)

            fp_data_H[site] = pblh_filter(fp_data_H[site],site)
            
        if 'pblh_aircraft' in filter_site:
            print('Applying PBLH filtering for aircraft data')

            fp_data_H[site] = pblh_aircraft_filter(fp_data_H[site])
            
        if 'height' in filter_site:
            print('Appling height filtering based on obs from other heights')

            fp_data_H[site] = height_filter(fp_data_H[site],site,network[i],heights[i],secondary_heights[i],
                                      start_date,end_date,average,species,obs_dir)
        
        n_obs_filtered.append(fp_data_H[site].mf.values.shape[0])
        
    perc_filtered = np.round((np.array(n_obs) - np.array(n_obs_filtered))/np.array(n_obs)*100,2)
    print(f'% of {species} filtered: {perc_filtered}')   
    
    return fp_data_H,perc_filtered

def extract_from_merged(fp_data_H,sites,use_bc,fp_data_H_prior):
    """
    Extract and stack data from the merged footprints, prior and observations.
    """
    
    for i,site in enumerate(sites):

        if i == 0:
            H_data = fp_data_H[site].H.values
            y_times = fp_data_H[site].time.values
            nsite_obs = np.array(fp_data_H[site].mf.values.shape)
            y = fp_data_H[site].mf.values
            
            if use_bc == True:
                Hbc_data = fp_data_H[site].H_bc.values
                
            if 'mf_variability' in fp_data_H[site] and fp_data_H[site].mf_variability.values.all() != 0:  
                print(f'Using mf_variability for {site} uncertainty.')
                y_err = fp_data_H[site].mf_variability.values
            elif 'mf_repeatability' in fp_data_H[site] :
                print(f'Using repeatability for {site} uncertainty.')
                y_err = fp_data_H[site].mf_repeatability.values
            else:
                print('No uncertainty in data file so assuming y1_err of 5 ppb/ppm.')
                y_err = np.ones(fp_data_H[site].mf.values.shape) * 5.
                
        else:
            H_data = np.hstack((H_data,fp_data_H[site].H.values))
            y_times = np.hstack((y_times,fp_data_H[site].time.values))
            nsite_obs = np.hstack((nsite_obs,fp_data_H[site].mf.values.shape))
            y = np.hstack((y,fp_data_H[site].mf.values))
            
            if use_bc == True:
                Hbc_data = np.hstack((Hbc_data,fp_data_H[site].H_bc.values))
                
            if 'mf_variability' in fp_data_H[site] and fp_data_H[site].mf_variability.values.all() != 0:  
                print(f'Using mf_variability for {site} uncertainty.')
                y_err = np.hstack((y_err,fp_data_H[site].mf_variability.values))
            elif 'mf_repeatability' in fp_data_H[site]:
                print(f'Using repeatability for {site} uncertainty.')
                y_err = np.hstack((y_err,fp_data_H[site].mf_repeatability.values))
            else:
                print('No uncertainty in data file so assuming y1_err of 5 ppb/ppm.')
                y_err = np.hstack((y_err,np.ones(fp_data_H[site].mf.values.shape) * 5.))
                
    if use_bc == True:
        Hbc = np.transpose(Hbc_data)
    else:
        Hbc = None

    H = np.transpose(H_data)
    
    if fp_data_H_prior is not None:
        
        for i,site in enumerate(sites):
            if i == 0:
                H_data_prior = fp_data_H_prior[site].H.values
            else:
                H_data_prior = np.hstack((H_data_prior,fp_data_H_prior[site].H.values))
                
        H_prior = np.transpose(H_data_prior)
        
    else:
        H_prior = None
    
    return H,Hbc,H_prior,y,y_err,y_times,nsite_obs

def create_sig_index(y_sig_freq,num_y,nsite_obs,y_times):
    """
    Creates index values used to group timeseries into sections with the same model error.
    """           
     
    if y_sig_freq == 'monthly':
    
        y_sig_index = np.cumsum(nsite_obs)
        
        return y_sig_index
            
    else:
        time_delta = pd.to_timedelta(y_sig_freq)
        y1_dt = pd.to_datetime(y_times)
            
        date_lim = y1_dt[0].floor('d') + time_delta

        y_sig_index = []
            
        for i,t in enumerate(y1_dt):
    
            if y1_dt[i] < y1_dt[i-1]:            # loop until hit next site 

                y_sig_index.append(i)
                # restart 5 day limit based on 00:00:00 of first data point of each site
                date_lim = y1_dt[i].floor('d') + time_delta

            if t > date_lim:                     # loop until date reaches 5 day limit

                y_sig_index.append(i)
                # add to create next 5 day limit
                date_lim += time_delta
            
            if i == y1_dt.shape[0]-1:             # loop until end of timeseries
                
                y_sig_index.append(i+1)
            
        if y_sig_index[0] == 0:      #removes 0 index
            y_sig_index.pop(0)
        
        y_sig_index[-1] = y_sig_index[-1] + 1
        y_sig_index = np.array(y_sig_index)
        
        return y_sig_index
    
def create_synthetic_obs(x_true,num_x,num_sectors,H,y,y_sig_mu,noise_type=None):
    """
    Uses a priori fluxes and footprints to create synthetic observations.
    """

    x_true_all = np.array([])

    for s in range(num_sectors):
    
        x_true_all = np.hstack((x_true_all,np.ones(num_x)*x_true[s]))
    
    perfect_y = np.matmul(H,x_true_all)
        
    n1 = np.random.RandomState(5)
    #noise1 = n1.normal(loc=0.,scale=perfect_y1*0.05,size=perfect_y1.shape[0])
    noise = n1.normal(loc=0.,scale=y_sig_mu,size=perfect_y.shape[0])
    
    y = perfect_y + noise
    
    if noise_type == 'systematic':
            
        noise_sys = perfect_y * 0.1
        
        y = y + noise_sys

    return y,noise
    
def correlated_noise(sites1,sites2,noise1,noise2,nsite_obs1):
    """
    Creates correlated noise for synthetic observations from the same sites.
    """
        
    noise1_for_gas2 = np.array([])
    
    nsite_obs1_sum = np.insert(np.cumsum(nsite_obs1),0,0.)

    for i,site in enumerate(sites1):
        i += 1
        if site in sites2:
            noise1_for_gas2 = np.hstack((noise1_for_gas2,(noise1[nsite_obs1_sum[i-1]:nsite_obs1_sum[i]])))
            
    return noise1_for_gas2

def create_sites_array(sites,nsite_obs):
    """
    Creates an array with entries giving the site for each observation.
    Used to form the model-measurement covariance matrix.
    """
    
    for s,site in enumerate(sites):
        if s == 0:
            sites_all = np.array([site]*nsite_obs[s])
        else:
            sites_all = np.hstack((sites_all,np.array([site]*nsite_obs[s])))
    
    return sites_all

@numba.jit(cache=True)
def calc_time_correlated_uncertainty(ac_timescale,sites_all,y_times_all,Q,num_y):
    """
    Fills in off-diagonal terms of the model-measurement covariance matrix with
    time-correlated uncertainty. 
    
    Inputs:
        ac_timescale_hours (int): Auto-correlation timescale in hours.
        sites_all (numpy array of str): Array of shape (num_y) containing entries giving
                                        the site of each observation.
        y_times_all (numpy array of datetime64): Time of each observation.
        Q (numpy array of floats): Diagonal matrix of shape (num_y,num_y) containing the combined
                                   model-measurement uncertainty of each observation.
                                   
    Returns:
        Q (numpy array of floats): Model-measurment uncertainty matrix, now containing off-diagonal terms.
    """
    
    # fill in off-diagonal terms for observations at the same site     
    for i in range(num_y):
        for j in range(num_y):
            if i != j and sites_all[i] == sites_all[j]:
                time_diff = np.abs(y_times_all[j] - y_times_all[i])
                Q[i,j] = np.sqrt(Q[i,i]) * np.sqrt(Q[j,j]) * np.exp((-time_diff)/ac_timescale)
    
    return Q

def find_covariance(mean1,mean2,std1,std2):
    """
    Creates example covariance for synthetic data tests with CH4 and 12CH4.
    mean1 and std1 are for CH4
    mean2 and std2 are for R (derived from delta13C-CH4)
    """
    
    samples1 = np.random.normal(mean1,std1,100000)
    samples2 = np.random.normal(mean2,std2,100000)
    
    for i,sample in enumerate(samples2):
        abs_fraction_ch4c12,abs_fraction_ch4c13 = isotopes.abs_fraction_ch4c12_ch4c13(sample)
        if i == 0:
            ch4c12 = abs_fraction_ch4c12 * samples1[i]
        else:
            ch4c12 = np.hstack((ch4c12,abs_fraction_ch4c12 * samples1[i]))
        
    cov_matrix = np.cov(samples1,ch4c12)
    
    return cov_matrix        

def create_model_measurement_uncertainty_matrix(mcmc_type,y,y_times,y_err,y_sig_mu_all,
                                                y_cov=None,ac_timescale_hours=None,y_sig_pdf=None,
                                                sites_all=None):
    """
    Takes the model error (fixed or hyperparameter) and measurement error
    and combines these to produce an uncertainty matrix for the observations.
    Includes options for including off-digonal covariances and correlated uncertainties.
    
    sites_all - required for auto-correlation (correlation between obs of the 
                same species from the same site) or to calc covariance, between 
                observations of two gases made at the same time (e.g. for ch4c12 mole fraction obs).
    """
    
    num_y = y.shape[0]
    
    if 'real' in mcmc_type:
        epsilon = np.sqrt(y_err**2 + y_sig_mu_all**2)
    elif 'pseudo' in mcmc_type:
        epsilon = y_sig_mu_all
        
    Q = np.zeros((num_y,num_y))
    np.fill_diagonal(Q,epsilon**2)
    
    if y_sig_pdf is not None:
        
        y_sig_sigma = y_sig_mu_all * 0.5
        
        Q_sig = np.zeros((y_sig_mu_all.shape[0],y_sig_mu_all.shape[0]))
        np.fill_diagonal(Q_sig,y_sig_sigma**2)
        
    else:
        
        Q_sig = None
        
    if ac_timescale_hours is not None:
        print(f'\nIncluding off-diagonal covariance terms using autocorrelation timescale of {ac_timescale_hours} hours.')

        Q = calc_time_correlated_uncertainty(ac_timescale=np.timedelta64(ac_timescale_hours,'h'),
                                             sites_all=sites_all,y_times_all=y_times,Q=Q,num_y=num_y)
        
    if y_cov == True:
        print('\nIncluding off-diagonal covariance terms by searching for matching time points.')
        print('WARNING: THIS HAS NOT BEEN TESTED WITH THIS VERSION OF THE MODEL CODE.')
        
        # loop through lists of sites and times to find matching points
        # calculate the covariance of each pair of measurements
        for i in range(num_y):
            for j in range(num_y):
                if sites_all[i] == sites_all[j] and y_times[i] == y_times[j]:
                    y_cov_matrix = find_covariance(y[i],y[j],y_err[i],y_err[j])
                    Q[i,j] = y_cov_matrix[0,1]
                    Q[j,i] = y_cov_matrix[0,1]
        
    return Q,Q_sig

def create_mcmc_inputs(mcmc_type,use_bc,species,species_type,sectors,sites,fp_data_H_all,x_sig,xbc_sig=None,
                       x_true=None,x_mu_factor=None,x_sig_factor=None,
                       y_sig_mu=None,noise_type=None,y_cov=None,
                       y_sig_pdf=None,y_sig_freq=None,ac_timescale_hours=None,
                       fp_data_H_prior_all=None,R_pdf=None,R_mu=None,R_sig=None,
                       Rbc_pdf=None,Rbc_mu=None,Rbc_sig=None,
                       spatialR=True):
    """
    Extracts or creates observations and uncertainty matrices.
    Stacks H, y and other matrices, into format used by mcmc code.
    Creates step size arrays for each optimised parameter.
    Inputs:
        mcmc_type (str):
            e.g. 'real-varR' or 'pseudo-fixedR'.
        use_bc (bool):
            If True, creates inputs for boundary conditions for emissions and emission ratios.
        species (list of str):
            List of species.
        species_type (list of str):
            Type of observations, e.g. ['mf','delta_value'].
        sectors (list of list of str):
            List of sectors for each species, e.g. [['FF','nonFF'],['FF','None']]
        sites (list of list of str):
            E.g. [['MHD','TAC','HFD'],['HFD']]
        fp_data_H_all (dict of datasets):
            Dictionary of merged fp_data_H objects for each gas.
        x_sig (list of floats):
            A priori emissions uncertainty for each sector for the primary gas.
        xbc_sig (list of floats):
            A priori boundary conditions uncertaity for each gas.
        x_true (list of floats):
            True emissions scaling factors values, used in pseudo data tests.
        x_mu_factor (list of floats):
            A priori emissions mean scaling factors, used in pseudo data tests.
        x_sig_factor (list of floats):
            A priori emissions uncertainty scaling factors, used in pseudo data tests.
        y_sig_mu (list of list of floats):
            Mean model error uncertainty for each gas for each site.
        noise_type (str):
            If 'systematic', adds systematic noise to pseudo obs. 
            If 'correlated, adds correlated noise to e.g. 12CH4 and CH4 pseudo obs.
        y_cov (bool):
            If True, calculates the uncertainty covariance between observations of the primary and seconary gases.
            Untested in this version of code, but was used with CH4+12CH4 model runs.
        y_sig_pdf (str):
            Model error prior PDF. If None, uses a fixed model error parameter.
        y_sig_freq (str):
            If 'monthly' estimates one value of model error per site per gas per month.
            If e.g. '2D' estimates one value of model error per site for each time period specified. 
            Untested in this verion of the code.
        ac_timescale_hours (int):
            Autocorrelation timescale in number of hours. Used for model-measurement uncertainty matrices with
            off-diagonal correlation between obs from the same site.
        fp_data_H_prior_all (dict of arrays):
            Dictionary of merged fp_data_H objects for each gas. Used in pseudo data tests when one set of fluxes are 
            used to produce the pseudo observations and another are given to the model as the flux prior.
        R_pdf (list of str):
            Emission ratio prior PDFs, one per secondary gas.
        R_mu (list of list of float):
            Emission ratio apriori means for each secondary gas for each sector.
        R_sig (list of list of floats):
            Emission ratio aprior standard deviations for each secondary gas for each sector.
        Rbc_pdf (list of str):
            Boundary condition prior PDFs, one per secondary gas.
        Rbc_mu (list of float):
            Boundary condition apriori means for each secondary gas.
        Rbc_sig (list of floats):
            Boundary condition aprior standard deviations for each secondary gas.
        spatialR (bool):
            If True, use one emission ratio per sector basis function. 
            If False, use just one emission ratio for each sector across the whole domain.
        
    Outputs:
        num_basis (int):
            Number of basis functions.
        H (dict of arrays):
            Dictionary of merged footprints and a priori emissions, for each gas.
        Hbc (dict of arrays):
            Dictionary of merged footprints and a priori boundary conditions for each gas.
        H_prior (dict of arrays):
            Dictionary of merged footprints and a priori emissions, for each gas. In pseudo data runs,
            these matrices are used as the prior and H is used to produce observations.
        y (dict of arrays):
            Observations of each gas.
        y_err (dict of arrays):
            Observational error for each gas.
        y_times (dict of arrays):
            Times of each observation, for each gas.
        nsite_obs (dict of lists):
            Number of observations from each site, for each gas.
        y_sig_index (dict of lists):
            Indices used to extrapolate out model uncertainties out to the full size of the
            model-measurment unceratinty matrix, for each gas.
        y_sig_mu_dict (dict of lists):
            A priori mean model error for each gas for each site.
        y_sig_mu_all (dict of lists):
            A priori mean model error for each gas for each site, extrapolated out to the full size
            of the model-measurement uncertainty matrix diagonal.
        sectors_dict (dict of lists):
            Sectors for each gas.
        xem_mu_all (array):
            A priori mean emissions scaling factors, extrapolated out to size of (num_basis*num_sectors).
        xem_std_all (array):
            A priori standard deviation of emissions scaling factors, extrapolated out to size of (num_basis*num_sectors).
        step_size_xem (array):
            Emissions scaling factor step size.
        Pem (array):
            Emissions apriori uncertainty matrix.
        xbc_mu_all (array):
            A priori mean of boundary condition scaling factors, extrapolated out to size of (num_basis*num_sectors).
        xbc_sig_all (array):
            A priori standard deviation of boundary condition scaling factors, extrapolated out to size of (num_basis*num_sectors).
        step_size_xbc (array):
            Boundary conditions scaling factor step size.
        Pbc(array):
            Boundary conditions apriori uncertainty matrix.
        xall_mu_all (array):
            A priori mean emissions and boundary conditions scaling factors, stacked together.
        Hall (dict of arrays):
            Dictionary of merged footprints and a priori emissions, stacked with merged footprints
            and boundary conditions.
        Q (dict of arrays):
            Model-measurement uncertainty matrix for each gas.
        Q_sig (dict of arrays):
            Model uncertainty uncertainty matrix for each gas.
        sites_array_all (dict of arrays):
            Sites, extrapolated out to full size of the model-measurement uncertainty matrix diagonal, for each gas. 
        step_size_y_sig (array):
            Model uncertainty step size.
        R_mu_dict (dict of lists):
            A priori mean emission ratio for each gas for each sector.
        R_mu_all (dict of lists):
            A priori mean emission ratio for each gas for each sector, extrapolated out to the full size
            of the x scaling factor array.
        R_mu_all (dict of lists):
            A priori mean emission ratio for each gas stacked for all sectors, extrapolated out to the full size
            of the x scaling factor array.
        Rbc_mu_all (dict of lists):
            A priori mean boundary condition for each gas for each site, extrapolated out to the full size
            of the xbc scaling factor array.
        step_size_R (dict of array):
            Emission ratio step sizes for each gas.
        step_size_Rbc (dict of array):
            Boundary condition ratio step sizes, for each gas.
        PR (dict of arrays):
            Emission ratio a priori uncertainty matrix for each gas.
        PRbc:
            Boundary condition ratio a priori uncertainty matrix for each gas.
    """
    
    delta_indices = [i for i,e in enumerate(species_type) if e == 'delta_value']
    delta_species = delta_species = [species[i] for i,e in enumerate(species_type) if e == 'delta_value']
    
    #---------------------------------------------------------------------------------
    # Extract and stack fp, obs etc. -------------------------------------------------
    
    sectors_dict = {}
        
    for s,s_name in enumerate(species):
        sectors_dict[s_name] = sectors[s]
    
    H,Hbc,H_prior,y,y_err,y_times,nsite_obs = {},{},{},{},{},{},{}
    num_y,y_sig_index,y_sig_mu_all,sites_array_all = {},{},{},{}

    for s,s_name in enumerate(species):
        
        print(s_name)
        H[s_name],Hbc[s_name],H_prior[s_name],y[s_name],y_err[s_name],y_times[s_name],\
        nsite_obs[s_name] = extract_from_merged(fp_data_H_all[s_name],sites[s],
                                                                        use_bc,fp_data_H_prior_all[s_name])
        
        sites_array_all[s_name] = create_sites_array(sites[s],nsite_obs[s_name])
        
        if species_type[s] == 'delta_value':
            n_before_resampling = y[s_name].shape[0]
            
            species0_match_times = isotopes.resample_y2_mod_to_y2_times(y_times1=y_times[species[0]],y_times2=y_times[s_name],
                                                                    sites_array1=sites_array_all[species[0]],
                                                                    sites_array2=sites_array_all[s_name])
            
            test_times = y_times[species[0]][species0_match_times]
            resample_indices = np.where(np.isin(y_times[s_name],test_times)==True)[0]
            
            sites_array_all[s_name] = sites_array_all[s_name][resample_indices]
            nsite_obs[s_name] = isotopes.resample_nsite_obs(sites[s],sites_array_all[s_name])
            
            y[s_name] = y[s_name][resample_indices]
            y_err[s_name] = y_err[s_name][resample_indices]
            y_times[s_name] = y_times[s_name][resample_indices]
            H[s_name] = H[s_name][resample_indices]
            if Hbc[s_name] is not None:
                Hbc[s_name] = Hbc[s_name][resample_indices]
            if H_prior[s_name] is not None:
                H_prior[s_name] = H_prior[s_name][resample_indices]
            
            print(f'Resampling {s_name} isotope obs down to match timestamps of {species[0]}, dropping {n_before_resampling-y[s_name].shape[0]} observations.')
            
        num_y[s_name] = y[s_name].shape[0]
            
        y_sig_index[s_name] = create_sig_index(y_sig_freq=y_sig_freq,num_y=num_y[s_name],nsite_obs=nsite_obs[s_name],
                                                y_times=y_times[s_name])
        
        y_sig_mu_all[s_name] = mcmc.y_sig_all(y_sig_index[s_name],y_sig_mu[s],num_y[s_name])
        
        y_sig_mu_all[s_name]
            
        if 'prop-obs-uncert' in mcmc_type: # for use with pseudo data tests, where model error is a % of each pollution event
            print('Using model error proportional to the size of each observation above background (the mean obs for each site).')
            y_sig_mu_all[s_name] = np.abs(np.mean(np.abs(y[s_name])) - np.abs(y[s_name])) * y_sig_mu_all[s_name]
        
        num_x = H[species[0]].shape[1]
        num_basis = fp_data_H_all[species[0]][sites[0][0]].H.values.shape[0]//len(sectors_dict[s_name])
        num_sectors = len(sectors_dict[s_name])
        
    #---------------------------------------------------------------------------------
    # Create synthetic obs -----------------------------------------------------------
    
    if 'pseudo' in mcmc_type:
        
        noise = {}
        
        if len(delta_indices) > 0:
            
            R_true_all = {sp:np.array([]) for sp in delta_species}
            
            for s in delta_indices:
                for a,sector_name in enumerate(sectors_dict[species[s]]):
                    R_true_all[species[s]] = np.hstack((R_true_all[species[s]],np.ones(num_basis) * R_mu[s-1][a]))
                    
            x_true_all = np.array([])
                
            for a in range(num_sectors):
                x_true_all = np.hstack((x_true_all,np.ones(num_basis)*x_mu_factor[a]))
                
        for s,s_name in enumerate(species):
            
            if species_type[s] == 'mf':
            
                y[s_name],noise[s_name] = create_synthetic_obs(x_true,num_basis,num_sectors,H[s_name],y[s_name],y_sig_mu_all[s_name],
                                                                noise_type)
                
                if s != 0 and noise_type == 'correlated':
                    y[s_name] = y[s_name] + correlated_noise(sites[0],sites[s],noise[species[0]],noise[s_name],
                                                             nsite_obs[s_name])
                
                negative_y = np.where(y[s_name] < 0.)
                y[s_name][negative_y] = 0.
                
            elif species_type[s] == 'delta_value':
                
                y_mod_sample,y[s_name],\
                nsite_obs[s_name] = isotopes.modelled_ch4_delta_obs(s_name,R_true_all,x_true_all,H[species[0]],
                                                                    sites[0],sites[s],sites_array_all[species[0]],sites_array_all[s_name],
                                                                    y_times[species[0]],y_times[s_name],
                                                                    delta_bc_sample=None,
                                                                    xbc_sample=None,Hbc_sample=None,
                                                                    return_bc_separately=False,add_noise=True,
                                                                    y_sig_mu=y_sig_mu_all[s_name],noise_type=noise_type,
                                                                    nsite_obs_delta=nsite_obs[s_name])
    else:
        
        noise = None
                
    #---------------------------------------------------------------------------------
    # Create emissions and boundary conditions uncertainty matrices ------------------

    step_size_xem = np.ones(num_x) * 0.01

    if 'pseudo' in mcmc_type:
        
        xem_mu_all,x_sig_all = np.array([]),np.array([])
        
        for s in range(num_sectors):

            xem_mu_all = np.hstack((xem_mu_all,np.ones(num_basis) * x_true[s] * np.array(x_mu_factor[s])))
            xem_sig_all = np.hstack((x_sig_all,np.ones(num_basis) * x_true[s] * np.array(x_sig_factor[s])))
        
    if 'real' in mcmc_type:
        
        xem_mu_all = np.ones(num_basis * num_sectors)
        
        xem_sig_all = np.array([])
        
        for s in range(num_sectors):

            xem_sig_all = np.hstack((xem_sig_all,np.ones(num_basis) * np.array(x_sig[s])))
        
    Pem = np.zeros((num_x,num_x))
    np.fill_diagonal(Pem,xem_sig_all**2)
    
    xall_mu_all = {}
    Hall = {}
    Hall_prior = {}

    if use_bc == True:
        
        xbc_mu_all = {}
        xbc_sig_all = {}
        step_size_xbc = {}
        Pbc = {}
            
        for s,s_name in enumerate(species):
            if species_type[s] == 'mf':
                num_xbc = Hbc[s_name].shape[1]
                xbc_mu_all[s_name] = np.ones(num_xbc)
                xbc_sig_all[s_name] = np.ones(num_xbc) * xbc_sig[s]
                step_size_xbc[s_name] = np.ones(num_xbc) * 0.01
                
                Pbc[s_name] = np.zeros((xbc_mu_all[s_name].shape[0],xbc_mu_all[s_name].shape[0]))
                np.fill_diagonal(Pbc[s_name],xbc_sig_all[s_name]**2)
                
                xall_mu_all[s_name] = np.hstack((xem_mu_all,xbc_mu_all[s_name]))
                Hall[s_name] = np.hstack((H[s_name],Hbc[s_name]))
                if H_prior[s_name] is not None:
                    Hall_prior[s_name] = np.hstack((H_prior[s_name],Hbc[s_name]))
                else:
                    Hall_prior[s_name] = None
    else:
        
        for s_name in species:
            
            xall_mu_all[s_name] = xem_mu_all
            Hall[s_name] = H[s_name]
            if H_prior[s_name] is not None:
                Hall_prior[s_name] = H_prior[s_name]
            else:
                Hall_prior[s_name] = None
        
        xbc_mu_all,step_size_xbc,xbc_sig_all,Pbc = None,None,None,None
        
    #---------------------------------------------------------------------------------
    # Create obs uncertainty matrices (model-measurment error) -----------------------
    
    Q,Q_sig = {},{}
    
    for s,s_name in enumerate(species):
        
        Q[s_name],Q_sig[s_name] = create_model_measurement_uncertainty_matrix(mcmc_type,y[s_name],y_times[s_name],
                                                                                      y_err[s_name],y_sig_mu_all[s_name],
                                                                                      y_cov,ac_timescale_hours,y_sig_pdf,
                                                                                      sites_array_all[s_name])
 
    if y_sig_pdf is not None:
        
        step_size_y_sig = {}
        y_sig_mu_dict = {}
        
        for s,s_name in enumerate(species):
            
            step_size_y_sig[s_name] = np.ones(len(y_sig_mu[s])) * (y_sig_mu[s][0] * 0.5)
            
            y_sig_mu_dict[s_name] = np.array(y_sig_mu[s])
            
    else:
        step_size_y_sig,y_sig_mu_dict = None,None

    #---------------------------------------------------------------------------------
    # Create R uncertainty matrix and step sizes -------------------------------------
                         
    if len(species) > 1:
        
        R_mu_dict,R_sig_dict = {sp:{} for sp in species[1:]},{sp:{} for sp in species[1:]}
        R_mu_all,R_sig_all = {sp:{} for sp in species[1:]}, {sp:{} for sp in species[1:]}
        R_mu_allsectors = {sp:np.array([]) for sp in species[1:]}
        
        if use_bc == True:
            Rbc_mu_all = {sp:{} for sp in delta_species}
        else:
            Rbc_mu_all = None
        
        for s,s_name in enumerate(species[1:]):
            for a,sector_name in enumerate(sectors_dict[s_name]):
                
                #if species_type[s+1] == 'delta_value' and 'pseudo' in mcmc_type:
                #    R_mu_all[s_name][sector_name] = R_pert[s_name][sector_name]
                #else:
                R_mu_all[s_name][sector_name] = np.ones(num_basis) * R_mu[s][a]
                R_mu_allsectors[s_name] = np.hstack((R_mu_allsectors[s_name],R_mu_all[s_name][sector_name]))
                
                if 'uniform' not in R_pdf[s]:
                    R_sig_all[s_name][sector_name] = np.ones(num_basis) * R_sig[s][a]
                else:
                    
                    R_sig_all[s_name][sector_name] = None
                    
                if spatialR == True:
                    #if species_type[s+1] == 'delta_value' and 'pseudo' in mcmc_type:
                    #    R_mu_dict[s_name][sector_name] = R_pert[s_name][sector_name]
                    #else:
                    R_mu_dict[s_name][sector_name] = np.ones(num_basis) * R_mu[s][a]
                        
                    if 'uniform' not in R_pdf[s]:
                        R_sig_dict[s_name][sector_name] = np.ones(num_basis) * R_sig[s][a]
                    else:
                        R_sig_dict[s_name][sector_name] = None
                    
                elif spatialR == False:
                    R_mu_dict[s_name][sector_name] = np.array([R_mu[s][a]])
                    if 'uniform' not in R_pdf[s]:
                        R_sig_dict[s_name][sector_name] = np.array([R_sig[s][a]])
                    else:
                        R_sig_dict[s_name][sector_name] = None

            if use_bc == True and species_type[s+1] == 'delta_value':
                Rbc_mu_all[s_name] = np.ones(Hbc[s_name].shape[1]) * Rbc_mu[s]

        if 'varR' in mcmc_type:
            
            step_size_R = {sp:{} for sp in species[1:]}
            PR = {sp:{} for sp in species[1:]}
            
            if use_bc == True:
                step_size_Rbc = {sp:{} for sp in species[1:]}
                PRbc = {sp:{} for sp in species[1:]}
            else:
                step_size_Rbc,PRbc = None,None
                
            for s,s_name in enumerate(species[1:]):
                
                if use_bc == True and species_type[s+1] == 'delta_value':
                    step_size_value = 0.1
                    step_size_Rbc[s_name] = np.ones(Rbc_mu_all[s_name].shape[0]) * step_size_value
                
                for a,sector_name in enumerate(sectors_dict[s_name]):
                    if sector_name is not None:
                
                        if species_type[s+1] == 'mf':
                            step_size_value = 0.01
                        elif species_type[s+1] == 'delta_value':
                            step_size_value = 0.1
                            
                        step_size_R[s_name][sector_name] = np.ones(R_mu_dict[s_name][sector_name].shape[0]) * step_size_value
                
                        if 'uniform' in R_pdf[s]:
                            PR[s_name][sector_name] = None
                        else:
                            PR[s_name][sector_name] = np.zeros((R_mu_dict[s_name][sector_name].shape[0],R_mu_dict[s_name][sector_name].shape[0]))
                            np.fill_diagonal(PR[s_name][sector_name],R_sig_dict[s_name][sector_name]**2)
            
            if use_bc == True and Rbc_pdf is not None: 
                for s,s_name in enumerate(species[1:]):
                    if 'uniform' in Rbc_pdf:
                        PRbc = None
                    elif species_type[s+1] == 'delta_value':
                        PRbc[s_name] = np.zeros((Rbc_mu_all[s_name].shape[0],Rbc_mu_all[s_name].shape[0]))
                        np.fill_diagonal(PRbc[s_name],Rbc_sig[s]**2)

            else:
                PRbc = None
            
        else:
            PR,PRbc,step_size_R,step_size_Rbc = None,None,None,None
            
    else:
        
        R_mu_dict,R_mu_all,Rbc_mu_all,PR,PRbc,step_size_R,step_size_Rbc = None,None,None,None,None,None,None
        R_mu_allsectors = None
            
    return (num_basis,H,Hbc,H_prior,y,y_err,y_times,nsite_obs,y_sig_index,y_sig_mu_dict,y_sig_mu_all,
            sectors_dict,
            xem_mu_all,xem_sig_all,step_size_xem,Pem,
            xbc_mu_all,xbc_sig_all,step_size_xbc,Pbc,
            xall_mu_all,Hall,Hall_prior,
            Q,Q_sig,sites_array_all,step_size_y_sig,
            R_mu_dict,R_mu_all,R_mu_allsectors,Rbc_mu_all,step_size_R,step_size_Rbc,PR,PRbc)