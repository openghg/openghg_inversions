#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:20:00 2023

@author: cv18710

Functions to use isotopic ratios and observations in an inverse model.

ratio_from_delta():
    - Calculates the isotopic ratio using the delta notation for isotopic composition.
    
delta_from_ratio():
    - Calculates isotopic delta value from an isotopic ratio.
    
abs_fraction_ch4c12_ch4c13_ch3d():
    - Calculates the absolute fraction of 12C-CH4 and 13C-CH4 12C-CH3D in the total mole fraction of CH4,
      using delta-13 and delta-D CH4 ratios.
      Based on atomic ratio calculations from Griffith and Rennick.
      
abs_fraction_ch4c12_ch4c13_ch3d_molecular():
    - Same as above, but derived from molecular ratios. This method makes the assumption that
      there is only 3 isotopes in all the gas.
      
resample_nsite_obs():
    - Recreates the nsite_obs object, which gives the number of obs per site.

resample_y2_mod_to_y2_times_onesite():
    - Creates an array of indices that can be used to resample a modelled delta value dataset
      down to timestamps only with matching observations of delta values.

resample_y2_mod_to_y2_times():
    - Resampling of y_times of a secondary gas, down to match observation times of the secondary gas.

modelled_ch4_delta_obs():
    - Creates modelled methane delta values.

"""

import numpy as np

ch4c13_molar_mass = 17.035
ch4c12_molar_mass = 16.032
ch3d_molar_mass = 17.049

def ratio_from_delta(delta,isotope='dch4c13'):
    """
    Calculates the isotopic ratio using the delta notation for isotopic composition.
    
    Inputs:
        delta (float): Delta value.
        isotope (str): 'dch4c13' or 'dch4d'
    Outputs:
        ratio (float): Isotopic ratio.
    """
    
    if isotope == 'dch4c13':
        ratio_standard = 0.01118
    if isotope == 'dch4d':
        ratio_standard = 0.00015575
        
    ratio = ratio_standard * ((delta/1000) + 1)
    
    return ratio

def delta_from_ratio(ratio,isotope='dch4c13'):
    """
    Calculates isotopic delta value from an isotopic ratio.

    Inputs:
        ratio (float): Isotopic ratio.
        isotope (str): 'dch4c13' or 'dch4d'.
    Outputs:
        delta (float): Delta value.
    """
    
    if isotope == 'dch4c13':
        ratio_standard = 0.01118
    elif isotope == 'dch4d':
        ratio_standard = 0.00015575
    
    delta = ((ratio/ratio_standard) - 1 ) * 1000
    
    return delta

def abs_fraction_ch4c12_ch4c13_ch3d(ch4c13_ratio=None,ch4d_ratio=None,
                                    dch4c13_delta=None,dch4d_delta=None):
    """
    Calculates the absolute fraction of 12C-CH4 and 13C-CH4 12C-CH3D in the total mole fraction of CH4,
    using delta-13 and delta-D CH4 ratios.
    Option to use ratio or delta value inputs.
    NEW ATOMIC RATIO CALCULATIONS FROM GRIFFITH + RENNICK.
    Inputs:
        ch4c13_ratio (float): ch4c13 atomic ratio.
        ch4d_ratio (float): ch4d atomic ratio.
        OR
        dch4c13_delta (float): dch4c13 delta value. This is converted to an isotopic ratio.
        dch4d_delta (float): dch4c4 delta_value. This is converted to an isotopic ratio.
    Returns:
        fraction_ch4c13, fraction_ch4c13, fraction_ch4d (floats) absolute fractions of each isotope.
    """
    
    if dch4c13_delta is not None:
        ch4c13_ratio = ratio_from_delta(dch4c13_delta,isotope='dch4c13')
    if dch4d_delta is not None:
        ch4d_ratio = ratio_from_delta(dch4d_delta,isotope='dch4d')
    
    denominator = (1+ch4c13_ratio) * ((1+ch4d_ratio)**4)
    
    fraction_ch4c12 = 1/denominator

    fraction_ch4c13 = ch4c13_ratio / denominator

    fraction_ch4d = (4 * ch4d_ratio) / denominator
    
    return fraction_ch4c12,fraction_ch4c13,fraction_ch4d

def abs_fraction_ch4c12_ch4c13_ch3d_molecular(ch4c13_ratio,ch4d_ratio):
    """
    Calculates the absolute fraction of 12C-CH4 and 13C-CH4 12C-CH3D in the total mole fraction of CH4,
    using delta-13 and delta-D CH4 ratios.
    BASED ON OLDER METHOD THAT USES ON MOLECULAR RATIOS
    Inputs:
        ch4c13_ratio (float): ch4c13 atomic ratio.
        ch4d_ratio (float): ch4d atomic ratio.
    Returns:
        fraction_ch4c13, fraction_ch4c13, fraction_ch4d (floats) absolute fractions of each isotope.
    """
    
    fraction_ch4c12 = 1 / (1 + ch4c13_ratio + ch4d_ratio)

    fraction_ch4c13 = 1 / ( (1/ch4c13_ratio) + 1 + (ch4d_ratio/ch4c13_ratio) ) 

    fraction_ch4d = 1 / ( (1/ch4d_ratio) + (ch4c13_ratio/ch4d_ratio) + 1 )
    
    return fraction_ch4c12,fraction_ch4c13,fraction_ch4d

def resample_nsite_obs(sites,sites_array):
    """
    Recreates the nsite_obs object, which gives the number of obs per site.
    Used after resample_y2_mod_to_y2_times is run the first time, to create 
    modelled delta value observations from mole fraction observations.
    Inputs:
        sites (list): Site codes for one species.
        sites_array (array): Array of corresponding sites for each observation.
    Returns:
        nsite_obs (array): Number of observations per site.
    """
    
    nsite_obs = np.array([])

    for s,site in enumerate(sites):
        count = np.where(sites_array == site)[0].shape[0]
        nsite_obs = np.hstack((nsite_obs,count))
                
    return nsite_obs.astype(int)

def resample_y2_mod_to_y2_times_onesite(y1_times,y2_times):
    """
    Creates an array of indices that can be used to resample a modelled delta value dataset
    down to timestamps only with matching observations of delta values.
    Simple version of the function, which can be used to find the indexes of matching times
    from two sets of observations from the same site.
    Inputs:
        y1_times,y2_times (arrays): 
            Arrays of observation times for the primary and secondary gas from one site.
    Returns:
        match_indices (array):
            Indices when there is observations of both gases at the same timestamp.
    """
    
    match_indices = np.where(np.isin(y2_times,y1_times) == True)[0]

    return match_indices

def resample_y2_mod_to_y2_times(y_times1,y_times2,sites_array1,sites_array2):
    """
    New resampling function using y_times in dictionary form.
    Creates an array of indices that can be used to resample a modelled delta value dataset
    down to timestamps only with matching observations of delta values.
    Inputs:
        y_times1 (array): Obs times for each site for the primary gas.
        y_times2 (array): Obs times for each site for the secondary gas.
        sites_array1 (list): Sites for the primary gas.
        sites_array2 (list): Sites for the secondary gas.
    Returns:
        keep_indices (array): Indices of matching timestamps for the secondary gas.
    """
    
    keep_indices = []

    match_times = np.where(np.isin(y_times1,y_times2) == True)[0]
    match_sites = np.where(np.isin(sites_array1,sites_array2) == True)[0]

    for m in match_times:
        if m in match_sites:
            keep_indices.append(m)
        
    return keep_indices
    
def modelled_ch4_delta_obs(species,delta_sample,xem_sample,H_sample,
                           sites_ch4,sites_delta,sites_array_ch4,sites_array_delta,
                           y_times_ch4,y_times_delta,y_mod_sample_out=None,
                           delta_bc_sample=None,xbc_sample=None,Hbc_sample=None,ybc_mod_sample_out=None,
                           return_bc_separately=False,add_noise=False,y_sig_mu=None,noise_type=None,
                           nsite_obs_delta=None):
    """
    Creates modelled methane mole fraction and methane isotope delta value observations from
    a sample of emissions scaling factors, delta samples, boundary condition scaling factors and 
    boundary condition delta values.
    Inputs:
        species (str): 
            'dch4c13' or 'dch4d'.
        delta_sample (dict of arrays):
            Sample of delta values for each gas for all sectors, extrapolated out to the same shape as xem_sample.
        xem_sample (array):
            Sample of emission scaling factors.
        H_sample (array):
            Merged footprints and emissions for the primary gas.
        sites_ch4 (list):
            List of sites for the primary gas.
        sites_delta (list):
            List of sites for the secondary gas.
        y_times_ch4 (array):
            Observation times for each site for the primary gas.
        y_times_delta (array):
            Observation times for each site for the secondary gas.
        y_mod_sample_out (array):
            Current values of modelled CH4.
        delta_bc_sample (dict of array):
            Sample of boundary condition delta values for all secondary gases.
        xbc_sample (array):
            Sample of boundary condition scaling factors for the primary gas.
        Hbc_sample (dict of array):
            Dictionary of merged footprints and boundary conditions, for the primary gas.
        ybc_mod_sample_out (array):
            Current values of modelled CH4 boundary conditions.
        return_bc_separately (bool):
            If True, also returns the modelled bc observations.
        add_noise (bool):
            If True, adds noise to modelled observations. Used when creating pseudo observations.
        y_sig_mu (array):
            Model error uncertainty for the secondary gas, used to add noise to observations.
        noise_type (str):
            If 'systematic' adds systematic noise to both the primary and secondary modelled obs.
        nsite_obs_delta (array):
            Number of observations per site for the secondary gas.
    Returns:
        y_mod_sample (array):
            Modelled observations of the primary gas.
        y2_mod_sample (array):
            Modelled observations of the secondary gas.
        ybc_mod_sample (array):
            Modelled boundary condition component of the primary gas observations.
        y2bc_mod_sample (array):
            Modelled boundary condition component of the primary gas observations.
        nsite_obs_delta_out (array):
            Number of observations per site for the secondary gas.
            
        """
        
    background_values = {'dch4c13':-47.3,'dch4d':-95.5}
    
    R_sample_all = {}
    
    for s in ['dch4c13','dch4d']:
        if s not in delta_sample.keys():
            delta_sample[s] = np.ones(xem_sample.shape[0]) * background_values[s]
    
        R_sample_all[s] = ratio_from_delta(delta_sample[s],s)

    x12_frac,x13_frac,x2_frac = abs_fraction_ch4c12_ch4c13_ch3d_molecular(ch4c13_ratio=R_sample_all['dch4c13'],
                                                                ch4d_ratio=R_sample_all['dch4d'])
    
    y_mod12_sample = np.matmul(H_sample,xem_sample * x12_frac)
    y_mod13_sample = np.matmul(H_sample,xem_sample * x13_frac)
    y_mod2_sample = np.matmul(H_sample,xem_sample * x2_frac)
    contains_data = []
    
    if delta_bc_sample is not None:
        
        for v in delta_bc_sample.values():
            if type(v) == np.ndarray:
                contains_data.append('yes')
    
    if len(contains_data) > 0:
        
        Rbc_sample_all = {}
    
        for s in ['dch4c13','dch4d']:
            if s not in delta_bc_sample.keys():
                delta_bc_sample[s] = np.ones(xbc_sample.shape[0]) * background_values[s]
        
            Rbc_sample_all[s] = ratio_from_delta(delta_bc_sample[s],s)
            
        #extract only ch4 bc values - not needed as dict now
        #xbc_sample = xbc_sample[:delta_bc_sample['dch4c13'].shape[0]]
        
        xbc12_frac,xbc13_frac,xbc2_frac = abs_fraction_ch4c12_ch4c13_ch3d_molecular(ch4c13_ratio=Rbc_sample_all['dch4c13'],
                                                                    ch4d_ratio=Rbc_sample_all['dch4d'])
        
        ybc_mod12_sample = np.matmul(Hbc_sample,xbc_sample * xbc12_frac)
        ybc_mod13_sample = np.matmul(Hbc_sample,xbc_sample * xbc13_frac)
        ybc_mod2_sample = np.matmul(Hbc_sample,xbc_sample * xbc2_frac)

        y_mod12_sample += ybc_mod12_sample
        y_mod13_sample += ybc_mod13_sample
        y_mod2_sample += ybc_mod2_sample
        
        if species == 'dch4c13':
            y2bc_mod_sample_all = delta_from_ratio(ybc_mod13_sample/ybc_mod12_sample,
                                                   isotope='dch4c13')
        elif species == 'dch4d':
            y2bc_mod_sample_all = delta_from_ratio(ybc_mod2_sample/ybc_mod12_sample,
                                                   isotope='dch4d')
        
        match_indices = resample_y2_mod_to_y2_times(y_times1=y_times_ch4,y_times2=y_times_delta,
                                                sites_array1=sites_array_ch4,sites_array2=sites_array_delta)
        
        y2bc_mod_sample = y2bc_mod_sample_all[match_indices]
        
        if ybc_mod_sample_out is not None:
            
            ybc_mod_sample = ybc_mod12_sample + ybc_mod13_sample + ybc_mod2_sample
            
            gas1_match_indices = resample_y2_mod_to_y2_times(y_times1=y_times_delta,y_times2=y_times_ch4, #probably incorrect, but don't use anymore
                                                sites_array1=sites_array_delta,sites_array2=sites_array_ch4)
            for i in gas1_match_indices:
                ybc_mod_sample_out[i] = ybc_mod_sample[i]

    else:
    
        ybc_mod_sample,y2bc_mod_sample = None,None
        
    y_mod12_sample[np.where(y_mod12_sample == 0.)] = np.min(y_mod12_sample[np.nonzero(y_mod12_sample)])
    y_mod13_sample[np.where(y_mod13_sample == 0.)] = np.min(y_mod13_sample[np.nonzero(y_mod13_sample)])
    y_mod2_sample[np.where(y_mod2_sample == 0.)] = np.min(y_mod2_sample[np.nonzero(y_mod2_sample)])

    if species == 'dch4c13':
        y2_mod_sample_all = delta_from_ratio(y_mod13_sample/y_mod12_sample,
                                                   isotope='dch4c13')
    elif species == 'dch4d':
        y2_mod_sample_all = delta_from_ratio(y_mod2_sample/y_mod12_sample,
                                                   isotope='dch4d')
    
    match_indices = resample_y2_mod_to_y2_times(y_times1=y_times_ch4,y_times2=y_times_delta,
                                                sites_array1=sites_array_ch4,sites_array2=sites_array_delta)
    
    if nsite_obs_delta is not None:
        nsite_obs_match_indices = resample_y2_mod_to_y2_times(y_times1=y_times_delta,y_times2=y_times_ch4,
                                                sites_array1=sites_array_delta,sites_array2=sites_array_ch4)
        nsite_obs_delta_out = resample_nsite_obs(sites_delta,sites_array_delta[nsite_obs_match_indices])
                                
    y2_mod_sample = y2_mod_sample_all[match_indices]
    
    # also resample modelled CH4, so only CH4 with timestamps matching those of the delta value obs are edited
    if y_mod_sample_out is not None:
        y_mod_sample = y_mod12_sample + y_mod13_sample + y_mod2_sample
        gas1_match_indices = resample_y2_mod_to_y2_times(y_times1=y_times_delta,y_times2=y_times_ch4, #probably incorrect, but don't use anymore
                                                sites_array1=sites_array_delta,sites_array2=sites_array_ch4)
        for i in gas1_match_indices:
            y_mod_sample_out[i] = y_mod_sample[i]
    
    if add_noise == True:
        print('Adding noise to modelled isotope observations.')
        
        n1 = np.random.RandomState(5)
        noise = n1.normal(loc=0.,scale=y_sig_mu,size=y2_mod_sample.shape[0])
        
        if noise_type == 'systematic':
            
            noise_sys = y2_mod_sample * 0.1

            y2_mod_sample = y2_mod_sample + noise_sys
            
        y2_mod_sample = y2_mod_sample + noise
        
    else:
        noise = None
        
    if return_bc_separately == True:
        
        if nsite_obs_delta is not None:
            return y_mod_sample_out,y2_mod_sample,ybc_mod_sample_out,y2bc_mod_sample,nsite_obs_delta_out
    
        else:
            return y_mod_sample_out,y2_mod_sample,ybc_mod_sample_out,y2bc_mod_sample

    else:
        
        if nsite_obs_delta is not None:
            return y_mod_sample_out,y2_mod_sample,nsite_obs_delta_out
        else:
            return y_mod_sample_out,y2_mod_sample