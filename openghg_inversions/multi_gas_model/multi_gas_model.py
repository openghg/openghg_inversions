#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 14 Feb 10:30:00 2023

Multi-gas inverse model for estimating sector-level methane emissions.

The multi_gas_model() function takes inputs from a .ini file and uses 
functions from the ACRG repository to merge observations, priors and footprints.
A MCMC process compares the prior probability of parameters to be optimised,
relative to the likelihood of these parameters in respect to the observations,
to produce posterior most-likely esimtates for these parameters.
The MCMC traces are processed to produce posterior emissions and country 
fluxes, with options to save this output as a netCDF file.

This version was created with the option to include multiple secondary gases.

Example secondary gas setups:
    - ethane (c2h6) with emission ratios for the fossil fuel (FF) sector.
    - methane isotopes, delta13c-ch4 (dch4c13) and/or delta2H-ch4 (dch4d) with emission ratios for each of the 
      FF and nonFF sectors.
    - the modelled c12 component of methane (ch4c12) with emission ratios for each 
      of the FF and nonFF sectors (untested).
    
Options for the model setup:
    - pseudo data, where mole fraction observations are created by combining flux/boundary 
      condition priors and transport footprints.
    - real data, using mole fraction observations.
    - fixed R, where the emission ratios/source signatures are fixed at their apriori
      values, throughout the mcmc process.
    - variable R, where the inversion also solves for posterior emission ratios/source
      signatures.

By specifying inputs in the .ini file, other options can be used or turned on/off:
    - mole fraction or isotope secondary gases.
    - different prior PDFs for all input parameters.
    - model-error hyper-parameters.
    - model-error covariances and off-diagonal terms in the uncertainty matrix.
    - use of boundary conditions.
    - filtering of observations.
    - spatial emission ratios at the same resolution as the emissions, or a single 
      emission ratio per sector.
    - use of different basis functions, to solve for emissions at different resolutions.
    - save the merged footprints, fluxes, observations etc. as a pickle file, so the 
      inverse model can be rerun with the same inputs, and reduce running time.
"""

import os
import glob
import numpy as np
import xarray as xr
import pickle
import version
from pandas import Timestamp
import getpass

from openghg_inversions.multi_gas_model import data_functions as data_func
from openghg_inversions.multi_gas_model import mcmc_functions as mcmc_func
from openghg_inversions.multi_gas_model import post_mcmc_functions as post_func

from openghg_inversions import utils

def multi_gas_model(mcmc_type,input_type,new_setup,full_output,data_path,directories,num_iter,post_av,
                    start_date,end_date,domain,species,species_type,
                    sites,heights,network,average,data_filtering,
                    sectors,flux_name,fp_basis_name,output_name,nquadtreebasis=None,
                    countrymask=None,countries=None,filtering_types=None,
                    y_sig_mu=None,y_sig_range=None,y_sig_pdf=None,
                    y_sig_freq='monthly',y_cov=False,ac_timescale_hours=None,
                    x_pdf=None,x_range=None,x_sig=None,
                    flux_name_prior=None,
                    use_bc=False,bc_basis_name=None,xbc_pdf=None,xbc_range=None,xbc_sig=None,
                    x_true=None,x_mu_factor=None,x_sig_factor=None,noise_type=None,
                    R_pdf=None,R_min=None,R_max=None,R_sig=None,R_mu=None,
                    Rbc_pdf=None,Rbc_min=None,Rbc_max=None,Rbc_sig=None,Rbc_mu=None,spatialR=True,
                    post_save_name=None,post_save_dir=None,n_trace_samples=None):
    
    """
    Inputs:
        See model_inputs_template.ini for detail on all inputs.
    
    Outputs:
        output_ds (xarray datatset)
            Contains MCMC traces, posteriors and selected inputs.
    """
    
    #---------------------------------------------------------------------------------
    # Directory paths ----------------------------------------------------------------
    
    if data_path == None:
        print('No data_path specified, cannot find inputs.')
        return None
    
    if directories['obs'] is not None: obs_dir = directories['obs']
    else: obs_dir = os.path.join(data_path,"obs/")
        
    if directories['fp'] is not None: fp_dir = directories['fp']
    else: fp_dir = os.path.join(data_path,"fp_NAME/")
        
    if directories['flux'] is not None: flux_dir = directories['flux']
    else: flux_dir = os.path.join(data_path,"LPDM/emissions/")
    
    if directories['bc'] is not None: bc_dir = directories['bc']
    else: bc_dir = os.path.join(data_path,"LPDM/bc/")
    
    if directories['fp_basis'] is not None: fp_basis_dir = directories['fp_basis']
    else: fp_basis_dir = os.path.join(data_path,"LPDM/basis_functions/")
       
    if directories['bc_basis'] is not None: bc_basis_dir = directories['bc_basis']
    else: bc_basis_dir = os.path.join(data_path,"LPDM/bc_basis_functions/")
    
    if directories['merged_data'] is not None: merged_data_dir = directories['merged_data']
    else: merged_data_dir = os.path.join(data_path,"fp_data_H/")
        
    #---------------------------------------------------------------------------------
    # Check array sizes --------------------------------------------------------------

    errors,warnings = data_func.mcmc_input_errors(species,sectors,domain,countrymask,countries,sites,
                      y_sig_mu,y_cov,y_sig_pdf,y_sig_range,y_sig_freq,post_save_dir)
    
    if len(warnings) > 0:
        for w in warnings:
            print(w)
    if len(errors) > 0:
        for e in errors:
            print(e)
        return None
    
    # work around for using % symbol files for heights, as this character cannot be read from .ini file
    if 'various' in heights:
        heights = ['%' if i == 'various' else i for i in heights]
        
    #---------------------------------------------------------------------------------
    # Rename flux and basis files and output -----------------------------------------
    
    year = start_date[:4]
    month = start_date[5:7]
    day = start_date[8:10]
    
    # output names for season USA runs
    if (np.datetime64(end_date) - np.datetime64(start_date)) > np.timedelta64(32,'D'):
        print(f'\nRunning for more than one month.\nSo assuming average fluxes etc between {start_date} and {end_date}...\n')
        
        if domain == 'USA':
            season_dict = {'2017-02-01':'winter',
                           '2017-10-01':'autumn',
                           '2018-04-01':'spring',
                           '2019-06-01':'summer'}
            
            if start_date in season_dict.keys():
                season = season_dict[start_date]    
    
            dateout = f'{year}{season}'
        
    else:
        print(f'\nRunning for one month between {start_date} and {end_date}...\n')    
        dateout = f'{year}{month}{day}'
        
    # slight work around to allow code to find a quadtree basis function file or save it correctly
    if nquadtreebasis is None:
        fp_basis_savename = None
        fp_basis_search_name = fp_basis_name
        
    else:
        fp_basis_savename = f'{nquadtreebasis}-{dateout}-{output_name}'
        fp_basis_search_name = f'quadtree_{species[0]}-{fp_basis_savename}'
        
    # output name 
    merged_name = f'{dateout}-{output_name}'
    
    fp_data_H_savename_all = [f'{merged_name}_{s}' for s in species]
    
    if flux_name_prior is not None:
        print('\nTO NOTE: synthetic data test is using real fluxes to create observations.')
        print(f'Then using {flux_name_prior} as the emission priors.\n')
        
        fp_data_H_prior_savename_all = [f'{merged_name}_prior_{s}' for s in species]
    
    if post_save_name:
        
        #TODO ADD SOMETHING HERE TO PRINT R VALUE TO OUTPUT NAME TOO?
        
        species_out = ''
        for s in species:
            species_out += f'{s}_'
        
        output_name = f'{mcmc_type}_{species_out}{merged_name}_{domain}_{post_save_name}'
           
        print(f'Will save output to {output_name}.\n')
    
    print(os.path.join(fp_basis_dir,f'{domain}/{fp_basis_search_name}_{domain}_*.nc'))
    
    
    #---------------------------------------------------------------------------------
    # Produce sensitivity matrices or reload -----------------------------------------
    
    fp_data_H_all = {}
    
    fp_data_H_prior_all = {}
    
    if new_setup == True:
        
        for s,s_name in enumerate(species):
            
            if 'pseudo' in mcmc_type and species_type[s] == 'delta_value':
                print(f"Running in 'pseudo data isotope' mode, so extracting {species[0]} obs \n",
                      f"for sites specified for {s_name} and using these to produce synthetic {s_name} obs.\n")
                merge_species = species[0]
            else:
                merge_species = s_name
                
            if input_type == 'acrg':
                print('Using ACRG repo functions and .nc files to produce model inputs.\n')
            
                fp_data_H_all[s_name] = data_func.merge_fp_data_flux_bc(sites=sites[s],heights=heights[s],network=network[s],
                                                                                    species=merge_species,domain=domain,
                                                                                start_date=start_date,end_date=end_date,
                                                                                average=average[s],sectors=sectors[s],flux_name=flux_name[s],
                                                                                fp_basis_savename=fp_basis_savename,
                                                                                obs_dir=obs_dir,fp_dir=fp_dir,flux_dir=flux_dir,
                                                                                fp_basis_dir=fp_basis_dir,nquadtreebasis=nquadtreebasis,bc_dir=bc_dir,
                                                                                bc_basis_dir=bc_basis_dir,bc_basis_name=bc_basis_name,
                                                                                use_bc=use_bc,save_dir=merged_data_dir,
                                                                                save_name=fp_data_H_savename_all[s],fp_basis_search_name=fp_basis_search_name,
                                                                                species_name=s_name)
            elif input_type == 'openghg':
                print('Using OpenGHG repo functions and an object store to produce model inputs.\n')
                
                os.environ["OPENGHG_PATH"] = data_path
                
                meas_period = [average[s]] * len(sites[s])
                
                fp_data_H_all[s_name] = utils.merge_fp_data_flux_bc_openghg(species=s_name,domain=domain,sites=sites[s],start_date=start_date,
                                                                            end_date=end_date,meas_period=meas_period,emissions_name=flux_name[s],
                                inlet=heights[s],network=network[s],instrument=None,fp_model=None,met_model=None,
                                fp_basis_case=fp_basis_name,fp_basis_search_name=fp_basis_search_name,fp_basis_savename=fp_basis_savename,
                                bc_basis_case=bc_basis_name,
                                basis_directory=fp_basis_dir,bc_basis_directory=bc_basis_dir,
                                nquadtreebasis=nquadtreebasis,outputname=output_name,filters=filtering_types[s],averagingerror=False,
                                save_directory=merged_data_dir)

            if flux_name_prior is not None:
                
                if input_type == 'acrg':
                    print('Using ACRG repo functions and .nc files to produce model inputs.\n')
                
                    fp_data_H_prior_all[s_name] =  data_func.merge_fp_data_flux_bc(sites=sites[s],heights=heights[s],network=network[s],
                                                    species=merge_species,domain=domain,
                                                    start_date=start_date,end_date=end_date,
                                                    average=average[s],sectors=sectors[s],flux_name=flux_name_prior[s],
                                                            fp_basis_savename=fp_basis_savename,
                                                    obs_dir=obs_dir,fp_dir=fp_dir,flux_dir=flux_dir,
                                                    fp_basis_dir=fp_basis_dir,nquadtreebasis=nquadtreebasis,use_bc=use_bc,save_dir=merged_data_dir,
                                                    save_name=fp_data_H_prior_savename_all[s],fp_basis_search_name=fp_basis_search_name,
                                                    species_name=s_name)
                    
                elif input_type == 'openghg':
                    print('Using OpenGHG repo functions and an object store to produce model inputs.\n')
                    
                    meas_period = average[s] * len(sites[s])
                    
                    fp_data_H_all[s_name] = utils.merge_fp_data_flux_bc_openghg(species=s_name,domain=domain,sites=sites[s],start_date=start_date,
                                                                                end_date=end_date,meas_period=meas_period,emissions_name=flux_name[s],
                                  inlet=heights[s],network=network[s],instrument=None,fp_model=None,met_model=None,
                                  fp_basis_case=fp_basis_search_name,bc_basis_case=bc_basis_name,
                                  basis_directory=fp_basis_dir,bc_basis_directory=bc_basis_dir,
                                  nquadtreebasis=nquadtreebasis,outputname=output_name,filters=filtering_types[s],averagingerror=False,
                                  save_directory=merged_data_dir)
            else:
                
                fp_data_H_prior_all[s_name] = None
                
    else:
        
        for s,s_name in enumerate(species):
        
            print(f'\nReading merged footprints from {fp_data_H_savename_all[s]}')
            fp_in = open(os.path.join(merged_data_dir,f'{fp_data_H_savename_all[s]}_H.pickle'),'rb')
            fp_data_H_all[s_name] = pickle.load(fp_in)
            fp_in.close()
            
            if flux_name_prior is not None:
                
                print(f'\nReading merged footprints from {fp_data_H_prior_savename_all[s]}')
                fp_in = open(os.path.join(merged_data_dir,f'{fp_data_H_prior_savename_all[s]}_H.pickle'),'rb')
                fp_data_H_prior_all[s_name] = pickle.load(fp_in)
                fp_in.close()
                
            else:
                
                fp_data_H_prior_all[s_name] = None

    #---------------------------------------------------------------------------------
    # Filter obs ---------------------------------------------------------------------
    
    perc_filtered = {sp:None for sp in species}
    
    if data_filtering == True and input_type == 'acrg':
         
        for s,s_name in enumerate(species):
            if all(i is None for i in filtering_types[s]) != True:
                print(f'\n{s_name} filtering:')
                fp_data_H_all[s_name],perc_filtered[s_name] = data_func.filtering(fp_data_H_all[s_name],sites[s],heights[s],
                                                                                  s_name,filtering_types[s],obs_dir)
                
                if 'pseudo' in mcmc_type:
                
                    fp_data_H_prior_all[s_name],perc_filtered[s_name] = data_func.filtering(fp_data_H_prior_all[s_name],sites[s],heights[s],
                                                                                  s_name,filtering_types[s],obs_dir)
                
    #---------------------------------------------------------------------------------
    # Stack matrices from fp_data_H, create uncertainty matrices and set up mcmc -----
    
    print('\nExtracting and stacking matrices from fp_data_H:')
        
    num_basis,H,Hbc,H_prior,y,y_err,y_times,nsite_obs,y_sig_index,y_sig_mu_dict,y_sig_mu_all,\
    sectors_dict,xem_mu_all,xem_sig_all,step_size_xem,Pem,xbc_mu_all,xbc_sig_all,\
    step_size_xbc,Pbc,xall_mu_all,Hall,Hall_prior,Q,Q_sig,sites_array_all,step_size_y_sig,R_mu_dict,R_mu_all,\
    R_mu_allsectors,Rbc_mu_all,step_size_R,step_size_Rbc,PR,\
    PRbc = data_func.create_mcmc_inputs(mcmc_type,use_bc,species,species_type,sectors,sites,
                                         fp_data_H_all,x_sig,xbc_sig,x_true,x_mu_factor,x_sig_factor,
                                         y_sig_mu,noise_type,y_cov,y_sig_pdf,y_sig_freq,ac_timescale_hours,
                                         fp_data_H_prior_all,R_pdf,R_mu,R_sig,Rbc_pdf,Rbc_mu,Rbc_sig,
                                         spatialR)
    
    #---------------------------------------------------------------------------------
    # Run mcmc -----------------------------------------------------------------------
    
    if flux_name_prior is not None:
        input_H = H_prior
        input_Hall = Hall_prior
    else:
        input_H = H
        input_Hall = Hall
    
    y_mod_trace,xem_trace,xbc_trace,R_trace,R_trace_allsectors,Rbc_trace,y_sig_trace,prob_trace_em,prob_trace_bc,\
    prob_trace_R,prob_trace_Rbc,prob_trace_y_sig = mcmc_func.mcmc(mcmc_type=mcmc_type,species=species,species_type=species_type,
                                                                  sectors_dict=sectors_dict,total_iter=num_iter,sites=sites,
                                                                  x_pdf=x_pdf,xem_range=x_range,xem_mu_all=xem_mu_all,Pem=Pem,
                                                                  step_size_xem=step_size_xem,xall_mu_all=xall_mu_all,Hall=input_Hall,
                                                                  y=y,y_times=y_times,y_err=y_err,
                                                                  Q=Q,H=input_H,Hbc=Hbc,
                                                                  xbc_pdf=xbc_pdf,xbc_range=xbc_range,xbc_mu_all=xbc_mu_all,
                                                                  Pbc=Pbc,step_size_xbc=step_size_xbc,R_pdf=R_pdf,R_min=R_min,
                                                                  R_max=R_max,R_mu_dict=R_mu_dict,
                                                                  R_mu_all=R_mu_all,R_mu_allsectors=R_mu_allsectors,PR=PR,
                                                                  step_size_R=step_size_R,Rbc_pdf=Rbc_pdf,Rbc_min=Rbc_min,
                                                                  Rbc_max=Rbc_max,Rbc_mu_all=Rbc_mu_all,PRbc=PRbc,
                                                                  step_size_Rbc=step_size_Rbc,y_sig_pdf=y_sig_pdf,
                                                                  y_sig_range=y_sig_range,y_sig_mu_dict=y_sig_mu_dict,
                                                                  y_sig_mu_all=y_sig_mu_all,step_size_y_sig=step_size_y_sig,
                                                                  y_sig_index=y_sig_index,Q_sig=Q_sig,sites_array_all=sites_array_all,
                                                                  spatialR=spatialR,y_cov=y_cov,ac_timescale_hours=ac_timescale_hours)
    
    #---------------------------------------------------------------------------------
    # Process mcmc outputs -----------------------------------------------------------
    
    fp_basis_search = glob.glob(os.path.join(fp_basis_dir,f'{domain}/{fp_basis_search_name}_{domain}_*.nc'))[0]
    
    with xr.open_dataset(fp_basis_search) as bf_file:
        basis = bf_file.basis.values[:,:,0]
        lat = bf_file.lat.values
        lon = bf_file.lon.values
        
    x_post,x_post_dict,x_post_latlon,x_post_mu,x_post_mu_dict,x_post_std,x_post_25perc,x_post_975perc,\
    x_prior_latlon,x_post_mu_latlon,x_post_std_latlon,\
    xbc_post,xbc_post_mu,xbc_post_std,xbc_post_25perc,xbc_post_975perc,\
    R_post,R_post_allsectors,R_post_mu_allsectors,R_post_mu,R_post_std,R_post_25perc,R_post_975perc,\
    Rbc_post,Rbc_post_mu,Rbc_post_std,Rbc_post_25perc,Rbc_post_975perc,\
    R_post_mu_latlon,\
    y_sig_post,y_sig_post_mu,y_sig_post_std,y_sig_post_25perc,\
    y_sig_post_975perc = post_func.process_mcmc_traces(species,sectors,basis,num_basis,post_av,xem_trace,xbc_trace,
                        y_sig_trace,R_trace,Rbc_trace,R_trace_allsectors,
                        n_trace_samples)
    
    flux_apriori,flux_post_mu,c_out,countrygrid,\
    country_prior,country_post,country_post_mu,country_post_25perc,country_post_975perc,\
    country_post_std,country_total_prior,country_total_post,country_total_post_mu,\
    country_total_post_25perc,country_total_post_975perc,country_total_post_std,\
    y_prior,y_post,y_post_mu,y_post_25perc,y_post_975perc,y_post_std,\
    ybc_prior,ybc_post,ybc_post_mu,ybc_post_25perc,ybc_post_975perc,\
    ybc_post_std = post_func.produce_model_outputs(n_trace_samples,domain,species,species_type,sectors_dict,basis,lat,lon,
                            fp_data_H_all,x_post_latlon,x_post_mu_latlon,x_prior_latlon,
                            y_mod_trace,xem_trace,x_post,H,y_times,sites,sites_array_all,
                            use_bc,xbc_trace,xbc_post,Hbc,
                            R_trace_allsectors,R_post_allsectors,
                            Rbc_trace,Rbc_post,
                            fp_data_H_prior_all,countrymask,countries,flux_name)
    
    #---------------------------------------------------------------------------------
    # Save outputs -------------------------------------------------------------------

    mf_species = [species[i] for i,e in enumerate(species_type) if e == 'mf']
    delta_species = [species[i] for i,e in enumerate(species_type) if e == 'delta_value']
    
    output_ds = xr.Dataset({'x_post':(['n_post_iter','n_x'],x_post),
                            'x_post_mu':(['n_x'],x_post_mu),
                            'x_post_25perc':(['n_x'],x_post_25perc),
                            'x_post_975perc':(['n_x'],x_post_975perc),
                            'fp_basis':(['lat','lon'],basis),
                            'x_sig':(['n_sectors'],np.array(x_sig)),
                            'countrymask':(['lat','lon'],countrygrid)
                            },
                           coords={'n_iter':(['n_iter'],np.arange(xem_trace.shape[0])),
                                   'n_post_iter':(['n_post_iter'],np.arange(x_post.shape[0])),
                                   'n_x':(['n_x'],np.arange(xem_trace.shape[1])),
                                   'n_basis':(['n_basis'],np.arange(x_post_mu_dict[sectors[0][0]].shape[0])),
                                   'lat':(['lat'],lat),
                                   'lon':(['lon'],lon),
                                   'n_sectors':(['n_sectors'],np.arange(len(x_sig))),
                                   'country':(['country'],np.arange(c_out.shape[0]))
                                   })
    
    if full_output == True:
        output_ds.attrs['step_size_xem']=step_size_xem[0]
        output_ds['x_trace'] = (['n_iter','n_x'],xem_trace)

    output_ds.attrs['step_size_xem']=step_size_xem[0]
    output_ds.attrs['inversion_type']=mcmc_type
    output_ds.attrs['date_created']=str(Timestamp.now())
    output_ds.attrs['code_version']=version.code_version()
    output_ds.attrs['start_date']=start_date                                     
    output_ds.attrs['end_date']=end_date
    output_ds.attrs['obs_av_period']=str(average)
    output_ds.attrs['Creator'] = getpass.getuser()
    
    output_ds.attrs['x_pdf']=x_pdf
    output_ds.attrs['basis_name']=fp_basis_search_name
    output_ds.attrs['country']=c_out
    
    if ac_timescale_hours is not None:
        output_ds.attrs['ac_timescale_hours'] = f'autocorrelation timescale of {ac_timescale_hours} hours'
    
    if x_true is not None:
        
        output_ds[f'x_true'] = (['n_sector'],np.array(x_true))
        output_ds[f'x_mu_factor'] = (['n_sector'],np.array(x_mu_factor))
        output_ds[f'x_sig_factor'] = (['n_sector'],np.array(x_sig_factor))
    
    if full_output == True:
        for sector in sectors[0]:
            output_ds[f'x_post_{sector}'] = (['n_post_iter','n_basis'],x_post_dict[sector])
            output_ds[f'x_post_mu_{sector}'] = (['n_basis'],x_post_mu_dict[sector])
            output_ds[f'x_post_mu_latlon_{sector}'] = (['lat','lon'],x_post_mu_latlon[sector])
        
    for s,s_name in enumerate(species):
        
        output_ds.attrs[f'flux_names_{s_name}'] = str(flux_name[s])
        output_ds.attrs[f'sectors_{s_name}'] = str(sectors[s])
        output_ds.coords[f'n_y_{s_name}'] = np.arange(y[s_name].shape[0])
        output_ds.coords[f'n_sites_{s_name}'] = np.arange(len(sites[s]))
        
        output_ds[f'sites_{s_name}'] = ([f'n_sites_{s_name}'],np.array(sites[s]))
        output_ds[f'nsite_obs_{s_name}'] = ([f'n_sites_{s_name}'],nsite_obs[s_name])
        output_ds[f'sites_array_all_{s_name}'] = ([f'n_y_{s_name}'],sites_array_all[s_name])
        output_ds[f'y_{s_name}'] = ([f'n_y_{s_name}'],y[s_name])
        output_ds[f'y_err_{s_name}'] = ([f'n_y_{s_name}'],y_err[s_name])
        output_ds[f'y_sig_mu_{s_name}'] = ([f'n_y_{s_name}'],y_sig_mu_all[s_name])
        output_ds[f'y_times_{s_name}'] = ([f'n_y_{s_name}'],y_times[s_name])
        output_ds[f'y_prior_{s_name}'] = ([f'n_y_{s_name}'],y_prior[s_name])
        output_ds[f'y_post_mu_{s_name}'] = ([f'n_y_{s_name}'],y_post_mu[s_name])
        output_ds[f'y_post_25perc_{s_name}'] = ([f'n_y_{s_name}'],y_post_25perc[s_name])
        output_ds[f'y_post_975perc_{s_name}'] = ([f'n_y_{s_name}'],y_post_975perc[s_name])
        
        if full_output == True:
            output_ds[f'H_{s_name}'] = ([f'n_y_{s_name}','n_x'],H[s_name])
            output_ds[f'Q_{s_name}'] = ([f'n_y_{s_name}',f'n_y_{s_name}'],Q[s_name])
            output_ds[f'y_post_{s_name}'] = (['n_post_iter',f'n_y_{s_name}'],y_post[s_name])
            
            if species_type[s] == 'mf':
                for a,site in enumerate(sites[s]):
                    
                    output_ds.coords[f'n_y_{s_name}_{site}'] = np.arange(nsite_obs[s_name][a])
                    output_ds[f'fp_{s_name}_{site}'] = (['lat','lon',f'n_y_{s_name}_{site}'],fp_data_H_all[s_name][site]['fp'].values)
        
            if perc_filtered[s_name] is not None:
                
                output_ds.coords[f'n_sites_{s_name}'] = np.arange(len(sites[s]))
                output_ds[f'perc_filtered_{s_name}'] = ([f'n_sites_{s_name}'],perc_filtered[s_name])
        
            if H_prior[s_name] is not None:
                
                output_ds[f'H_prior_{s_name}'] = ([f'n_y_{s_name}','n_x'],H_prior[s_name])
        
        if y_sig_post is not None:
            
            output_ds.attrs['y_sig_pdf']=y_sig_pdf
            output_ds.attrs['y_sig_freq']=y_sig_freq
            output_ds.coords[f'n_y_sig_{s_name}'] = np.arange(y_sig_post[s_name].shape[1])
            
            output_ds[f'y_sig_prior_{s_name}'] = ([f'n_y_sig_{s_name}'],y_sig_trace[s_name][0,:])
            output_ds[f'y_sig_post_{s_name}'] = (['n_post_iter',f'n_y_sig_{s_name}'],y_sig_post[s_name])
            output_ds[f'y_sig_post_mu_{s_name}'] = ([f'n_y_sig_{s_name}'],y_sig_post_mu[s_name])
            output_ds[f'y_sig_post_25perc_{s_name}'] = ([f'n_y_sig_{s_name}'],y_sig_post_25perc[s_name]) 
            output_ds[f'y_sig_post_975perc_{s_name}'] = ([f'n_y_sig_{s_name}'],y_sig_post_975perc[s_name]) 
            
            if full_output == True:
                output_ds.attrs['step_size_y_sig_{s_name}']=step_size_y_sig[s_name]
                output_ds[f'y_sig_trace_{s_name}'] = (['n_iter',f'n_y_sig_{s_name}'],y_sig_trace[s_name])
                output_ds[f'y_sig_index_{s_name}'] = ([f'n_y_sig_{s_name}'],y_sig_index[s_name])
                
        if ybc_post is not None:
            
            output_ds[f'ybc_prior_{s_name}'] = ([f'n_y_{s_name}'],ybc_prior[s_name])
            output_ds[f'ybc_post_{s_name}'] = (['n_post_iter',f'n_y_{s_name}'],ybc_post[s_name])
            output_ds[f'ybc_post_mu_{s_name}'] = ([f'n_y_{s_name}'],ybc_post_mu[s_name])
            output_ds[f'ybc_post_25perc_{s_name}'] = ([f'n_y_{s_name}'],ybc_post_25perc[s_name])
            output_ds[f'ybc_post_975perc_{s_name}'] = ([f'n_y_{s_name}'],ybc_post_975perc[s_name])

        if xbc_post is not None:
            if s_name in xbc_post.keys():
            
                output_ds.attrs['xbc_pdf']=xbc_pdf  
                output_ds.coords[f'n_bc'] = np.arange(xbc_post[s_name].shape[1])
                output_ds[f'xbc_post_{s_name}'] = (['n_post_iter','n_bc'],xbc_post[s_name])
                output_ds[f'xbc_post_mu_{s_name}'] = (['n_bc'],xbc_post_mu[s_name])
                output_ds[f'xbc_post_25perc_{s_name}'] = (['n_bc'],xbc_post_25perc[s_name])
                output_ds[f'xbc_post_975perc_{s_name}'] = (['n_bc'],xbc_post_975perc[s_name])
                
                if full_output == True:
                    output_ds.attrs[f'step_size_xbc_{s_name}']=step_size_xbc[s_name]
                    output_ds[f'Hbc_{s_name}'] = ([f'n_y_{s_name}','n_bc'],Hbc[s_name])
                    output_ds[f'xbc_trace_{s_name}'] = (['n_iter','n_bc'],xbc_trace[s_name])
                    
        for sector in sectors_dict[s_name]:
            if sector is not None:
            
                output_ds[f'flux_apriori_{s_name}_{sector}'] = (['lat','lon'],flux_apriori[s_name][sector])
                output_ds[f'flux_post_mu_{s_name}_{sector}'] = (['lat','lon'],flux_post_mu[s_name][sector])
            
    for s,s_name in enumerate(species[1:]):
        
        if R_post is not None:

            output_ds.attrs['R_pdf_s_name'] = str(R_pdf[s])
            
            if full_output == True:
                output_ds[f'R_post_allsectors_{s_name}'] = (['n_post_iter','n_x'],R_post_allsectors[s_name])
                output_ds[f'R_post_mu_allsectors_{s_name}'] = (['n_x'],R_post_mu_allsectors[s_name])
            
            for sector in sectors_dict[s_name]:
                if sector is not None and 'varR' in mcmc_type:
                
                    output_ds.coords[f'n_R'] = np.arange(R_post[s_name][sector].shape[1])

                    output_ds[f'R_post_{s_name}_{sector}'] = (['n_post_iter','n_R'],R_post[s_name][sector])
                    output_ds[f'R_post_mu_{s_name}_{sector}'] = (['n_R'],R_post_mu[s_name][sector])
                    output_ds[f'R_post_25perc_{s_name}_{sector}'] = (['n_R'],R_post_25perc[s_name][sector]) 
                    output_ds[f'R_post_975perc_{s_name}_{sector}'] = (['n_R'],R_post_975perc[s_name][sector]) 
                    output_ds[f'R_post_mu_latlon_{s_name}_{sector}'] = (['lat','lon'],R_post_mu_latlon[s_name][sector]) 
                    
                    if full_output == True:
                        output_ds.attrs[f'step_size_R_{s_name}_{sector}']=step_size_R[s_name][sector][0]
                        output_ds[f'R_trace_{s_name}_{sector}'] = (['n_iter','n_R'],R_trace[s_name][sector])
                        
    for s_name in delta_species:
        
        if Rbc_post is not None:

            output_ds.attrs['Rbc_pdf'] = str(Rbc_pdf)
            output_ds[f'Rbc_post_{s_name}'] = (['n_post_iter','n_bc'],Rbc_post[s_name])
            output_ds[f'Rbc_post_mu_{s_name}'] = (['n_bc'],Rbc_post_mu[s_name])   
            output_ds[f'Rbc_post_25perc_{s_name}'] = (['n_bc'],Rbc_post_25perc[s_name])    
            output_ds[f'Rbc_post_975_perc_{s_name}'] = (['n_bc'],Rbc_post_975perc[s_name])   
            
            if full_output == True:
                if step_size_Rbc is not None:
                    output_ds.attrs[f'step_size_Rbc_{s_name}']=step_size_Rbc[s_name][0]
                output_ds[f'Rbc_trace_{s_name}'] = (['n_iter','n_bc'],Rbc_trace[s_name])
                
    for s_name in mf_species:
        
        output_ds[f'country_total_prior_{s_name}'] = ([f'country'],country_total_prior[s_name])
        output_ds[f'country_total_post_mu_{s_name}'] = ([f'country'],country_total_post_mu[s_name])
        output_ds[f'country_total_post_25perc_{s_name}'] = ([f'country'],country_total_post_25perc[s_name])
        output_ds[f'country_total_post_975perc_{s_name}'] = ([f'country'],country_total_post_975perc[s_name])
        
        for sector in sectors_dict[s_name]:
            
            output_ds[f'country_prior_{s_name}_{sector}'] = ([f'country'],country_prior[s_name][sector])
            output_ds[f'country_post_mu_{s_name}_{sector}'] = ([f'country'],country_post_mu[s_name][sector])
            output_ds[f'country_post_25perc_{s_name}_{sector}'] = ([f'country'],country_post_25perc[s_name][sector])
            output_ds[f'country_post_975perc_{s_name}_{sector}'] = ([f'country'],country_post_975perc[s_name][sector])
            
        if full_output == True:
            output_ds[f'country_total_post_{s_name}'] = (['n_post_iter',f'country'],country_total_post[s_name])
            for sector in sectors_dict[s_name]:
                output_ds[f'country_post_{s_name}_{sector}'] = (['n_post_iter',f'country'],country_post[s_name][sector])
    
    if post_save_name is not None:
        
        output_path = os.path.join(post_save_dir,output_name+'_post.nc')
        
        output_ds.to_netcdf(output_path)
        print(f'\nMCMC output and input variables saved to {post_save_dir}:\n{output_name+"_post.nc"}')
        
    return output_ds