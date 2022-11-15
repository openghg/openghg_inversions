#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:13:21 2020

@author: lw13938

Modules for running an MCMC inversion using PyMC3. There are also functions
to dynamically create a basis function grid based on the a priori sensitivity,
and some other functionality for setting up the inputs to this (or any) inverse
method.

If not using on an HPC (i.e. on Snowy), in the terminal you should do:
export OPENBLAS_NUM_THREADS=XX
and/or
export OMP_NUM_THREADS=XX
where XX is the number of chains you are running. If running in Spyer do this
before launching Spyder, else you will use every available thread. Apart from
being annoying it will also slow down your run due to unnecessary forking.

*******************************************************************************
Update: Nov. 2022: Eric Saboya (eric.saboya@bristol.ac.uk)

HBMCMC updated to use openghg as a dependency replacing (most) of the acrg
modules previously used. HBMCMC inputs remain the same except for the
inclusion of the "fp_model" kwarg.

Note. This version expects all data to already be included in the
object store and for the path to the object store to already be set.

HBMCMC currently uses PyMC3. The next step will be to update this
to use PyMC v4.0. Watch this space!

*******************************************************************************
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import openghg_inversions.hbmcmc.inversionsetup as setup
import openghg_inversions.hbmcmc.inversion_pymc3 as mcmc

from openghg.retrieve import search_surface, get_obs_surface, get_flux
from openghg.retrieve import get_bc, get_footprint
#from openghg.standardise import standardise_surface, standardise_flux
#from openghg.standardise import standardise_footprint, standardise_bc
from openghg.analyse import ModelScenario


import acrg.name.name as name

import openghg_inversions.basis_functions as basis
from openghg_inversions import utils
from openghg_inversions.config.paths import Paths

acrg_path = Paths.acrg
data_path = Paths.data


def fixedbasisMCMC(species, sites, domain, meas_period, start_date,
                   end_date, outputpath, outputname,
                   met_model = None, fp_model="NAME",
                   xprior={"pdf":"lognormal", "mu":1, "sd":1},
                   bcprior={"pdf":"lognormal", "mu":0.004, "sd":0.02},
                   sigprior={"pdf":"uniform", "lower":0.5, "upper":3},
                   offsetprior={"pdf":"normal", "mu":0, "sd":1},
                   nit=2.5e5, burn=50000, tune=1.25e5, nchain=2,
                   emissions_name=None, inlet=None, fpheight=None, instrument=None,
                   fp_basis_case=None, basis_directory = None, bc_basis_case="NESW",
                   country_file = None,
                   max_level=None,
                   quadtree_basis=True,nbasis=100,
                   filters = [],
                   averagingerror=True, bc_freq=None, sigma_freq=None, sigma_per_site=True,
                   country_unit_prefix=None, add_offset = False,
                   verbose = False):

    """
    --------------------------------------------------------------------------

    Script to run hierarchical Bayesian MCMC for inference of emissions using
    pymc3 to solve the inverse problem.

    --------------------------------------------------------------------------
    Args:
        species (str):
            Species of interest
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
        emissions_name (dict, optional):
            Allows emissions files with filenames that are longer than just the species name
            to be read in (e.g. co2-ff-mth_EUROPE_2014.nc). This should be a dictionary
            with {source_name: emissions_file_identifier} (e.g. {'anth':'co2-ff-mth'}). This way
            multiple sources can be read in simultaneously if they are added as separate entries to
            the emissions_name dictionary.
        inlet (str/list, optional):
            Specific inlet height for the site (must match number of sites)
        fpheight (dict, optional):
            Specific release height for the sites' footprints.
            E.g. fpheight={"TAC":"185m"}(must match number of sites).
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
            Whether a model sigma value will be calculated for each site independantly (True) or all sites together (False).
            Default: True
        country_unit_prefix ('str', optional)
            A prefix for scaling the country emissions. Current options are:
            'T' will scale to Tg, 'G' to Gg, 'M' to Mg, 'P' to Pg.
            To add additional options add to acrg_convert.prefix
            Default is none and no scaling will be applied (output in g).
        add_offset (bool):
            Add an offset (intercept) to all sites but the first in the site list. Default False.
    
    --------------------------------------------------------------------------

    Returns:
        Saves an output from the inversion code using inferpymc3_postprocessouts.

    --------------------------------------------------------------------------    
    """
    # Change list of sites to upper case equivalent as 
    # most acrg functions copied across use upper case notation
    for i, site in enumerate(sites): sites[i]=site.upper()


    fp_all={}
    fp_all['.species']=species.upper()

    # ******** Get fluxes ********
    flux_dict={}
    basestring = (str, bytes)
    for source, emi_source in emissions_name.items():
        if isinstance(emissions_name[source], basestring):
            try:
                print(f"Attempting to retrieve '{emi_source}' fluxes"
                       " from object store ...\n") 
                get_flux_data = get_flux(species=species,
                                         domain=domain,
                                         source=list(emissions_name.keys())[0],
                                         start_date=start_date,
                                         end_date=end_date)

                print("Sucessfully retrieved flux file"
                     f" '{emi_source}' from objectstore.")

                flux_dict[source]=get_flux_data
            except:
                 print(f"-*- Warning -*-: Flux file '{emi_source}' not found"
                        " in object store. Please add file to object store."
                        " Exiting process.")
                 sys.exit(0)

    fp_all['.flux']=flux_dict

    # ******** Get boundary conditions ********
    try:
        print("Attempting to retrieve boundary condition data"
              f" between {start_date} to {end_date} from object store ...\n")
        get_bc_data = get_bc(species=species,
                             domain=domain)#,
                             #start_date=start_date,
                             #end_date=end_date)
        print("Successfully retrieved boundary condition data between"
              f" {start_date} and {end_date} from object store.\n")

    except ValueError:
        print(f"-*- Warning -*-: Boundary condition data between {start_date}"
              f" and {end_date} not found in object store. Please add"
                " boundary condition data to object store. Exiting process.\n")
        sys.exit(0)

    fp_all['.bc']=get_bc_data

    # ******** Get observations and footprints ********
    #   Check site data exists in object store
    search_objectstore=search_surface(site=sites,
                                      species=species,
                                      inlet=inlet,
                                      start_date=start_date,
                                      end_date=end_date)
    data={}
    footprint_dict={}
    scales={}
    check_scales=[]

    for i, site in enumerate(sites):
        print(f"Attempting to retrieve {species.upper()} measurements"
              f" for {site.upper()} between {start_date} and"
              f" {end_date} from object store ...\n")

        if site.lower() in search_objectstore.results.keys():
        #   Get obs from objectstore
            site_data=get_obs_surface(site=site,
                                      species=species,
                                      inlet=inlet[i],
                                      start_date=start_date,
                                      end_date=end_date,
                                      average=meas_period[i],
                                      instrument=instrument[i])

            print(f"Successfully retrieved {species.upper()} measurement"
                  f" data for {site.upper()} from object store.\n")
            data[site]=site_data[site]

            try:
        #   Attempt to retrieve footprints from objectstore
                print(f"Attempting to retrieve {site.upper()} {domain} footprint"
                       " data from object store ...\n")
                get_fps = get_footprint(site=site,
                                        height=inlet[i],
                                        domain=domain,
                                        model=fp_model,
                                        start_date=start_date,
                                        end_date=end_date)

                footprint_dict[site] = get_fps
                print(f"Successfully retrieved {site.upper()} {domain} footprints"
                       " from object store.\n")

            except:
                print(f"-* -Warning -*-: Footprint data for {site.upper()}"
                      f" between {start_date} to {end_date}"
                      f" was not found in the object store. Exiting process.")
                sys.exit(1)

        #   Create ModelScenario object
            model_scenario=ModelScenario(site=site,
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
                                         bc=get_bc_data)

            scenario_combined=model_scenario.footprints_data_merge()
            fp_all[site]=scenario_combined

        #    Check consistency of measurement scales between sites
            check_scales+=[scenario_combined.scale]
            if not all (s==check_scales[0] for s in check_scales):
                rt=[]
                for i in check_scales:
                    if isinstance(i, list): rt.extend(flatten(i))
                else:
                    rt.append(j)
                scales[site]=rt
            else:
                scales[site]=check_scales[0]

        else:
            print(f"-*- Warning -*-: {species.upper()} obs data for {site.upper()}"
                  f" between {start_date} to {end_date} was not found in the"
                   " object store. Please add measurement data to object store."
                   " Exiting process.")
            sys.exit(1)

    fp_all['.scales']=scales
    fp_all['.units']=float(scenario_combined.mf.units)

    print('Running for %s to %s' % (start_date, end_date))

    # If site contains measurement errors given as repeatability and variability,
    # use variability to replace missing repeatability values, then drop variability
    for site in sites:
        if "mf_variability" in fp_all[site] and "mf_repeatability" in fp_all[site]:
            fp_all[site]["mf_repeatability"][np.isnan(fp_all[site]["mf_repeatability"])] = \
                fp_all[site]["mf_variability"][np.logical_and(np.isfinite(fp_all[site]["mf_variability"]),np.isnan(fp_all[site]["mf_repeatability"]) )]
            fp_all[site] = fp_all[site].drop_vars("mf_variability")

    # Add measurement variability in averaging period to measurement error
    if averagingerror:
        fp_all = setup.addaveragingerror(fp_all, 
                                         sites, 
                                         species, 
                                         start_date, 
                                         end_date, 
                                         meas_period, 
                                         inlet=inlet, 
                                         instrument=instrument)

    # Create basis function using quadtree algorithm if needed
    if quadtree_basis:
        if fp_basis_case != None:
            print("Basis case %s supplied but quadtree_basis set to True" % fp_basis_case)
            print("Assuming you want to use %s " % fp_basis_case)
        else:
            tempdir = basis.quadtreebasisfunction(emissions_name, 
                                                  fp_all, 
                                                  sites,
                                                  start_date, 
                                                  domain, 
                                                  species, 
                                                  outputname,
                                                  nbasis=nbasis)

            fp_basis_case= "quadtree_"+species+"-"+outputname
            basis_directory = tempdir
    else:
        basis_directory = basis_directory


    fp_data = utils.fp_sensitivity(fp_all, 
                                   domain=domain, 
                                   basis_case=fp_basis_case,
                                   basis_directory=basis_directory)

    fp_data = utils.bc_sensitivity(fp_data, 
                                   domain=domain,
                                   basis_case=bc_basis_case)

    # Apply named filters to the data
    fp_data = utils.filtering(fp_data, filters)

    for si, site in enumerate(sites):
        fp_data[site].attrs['Domain']=domain

    # Get inputs ready
    error = np.zeros(0)
    Hbc = np.zeros(0)
    Hx = np.zeros(0)
    Y = np.zeros(0)
    siteindicator = np.zeros(0)
    for si, site in enumerate(sites):
        if 'mf_repeatability' in fp_data[site]:
            error = np.concatenate((error, fp_data[site].mf_repeatability.values))
        if 'mf_variability' in fp_data[site]:
            error = np.concatenate((error, fp_data[site].mf_variability.values))

        Y = np.concatenate((Y,fp_data[site].mf.values))
        siteindicator = np.concatenate((siteindicator, np.ones_like(fp_data[site].mf.values)*si))
        if si == 0:
            Ytime=fp_data[site].time.values
        else:
            Ytime = np.concatenate((Ytime,fp_data[site].time.values ))

        if bc_freq == "monthly":
            Hmbc = setup.monthly_bcs(start_date, end_date, site, fp_data)
        elif bc_freq == None:
            Hmbc = fp_data[site].H_bc.values
        else:
            Hmbc = setup.create_bc_sensitivity(start_date, end_date, site, fp_data, bc_freq)

        if si == 0:
            Hbc = np.copy(Hmbc) #fp_data[site].H_bc.values
            Hx = fp_data[site].H.values
        else:
            Hbc = np.hstack((Hbc, Hmbc))
            Hx = np.hstack((Hx, fp_data[site].H.values))

    sigma_freq_index = setup.sigma_freq_indicies(Ytime, sigma_freq)

    # Run Pymc3 inversion
    xouts, bcouts, sigouts, Ytrace, YBCtrace, convergence, step1, step2 = mcmc.inferpymc3(Hx, Hbc, Y, error, siteindicator, sigma_freq_index,
           xprior,bcprior, sigprior, nit, burn, tune, nchain, sigma_per_site, offsetprior=offsetprior, add_offset=add_offset, verbose=verbose)

    # Process and save inversion output
    mcmc.inferpymc3_postprocessouts(xouts,bcouts, sigouts, convergence,
                               Hx, Hbc, Y, error, Ytrace, YBCtrace,
                               step1, step2,
                               xprior, bcprior, sigprior, offsetprior, Ytime, siteindicator, sigma_freq_index,
                               domain, species, sites,
                               start_date, end_date, outputname, outputpath,
                               country_unit_prefix,
                               burn, tune, nchain, sigma_per_site,
                               fp_data=fp_data, emissions_name=emissions_name,
                               basis_directory=basis_directory, country_file=country_file,
                               add_offset=add_offset)

    if quadtree_basis is True:
        # remove the temporary basis function directory
        shutil.rmtree(tempdir)

    print("---- Inversion completed ----")

def rerun_output(input_file, outputname, outputpath, verbose=False):
    """
    Rerun the MCMC code by taking the inputs from a previous output using this code
    and rewrite a new output. This allows reproducibility of results without the need
    to transfer all raw input files.

    Args:
        input_file (str):
            Full path to previously written ncdf file
        outputname (list):
            Unique identifier new for output/run name.
        outputpath (str):
            Path to where output should be saved.

    Returns:
        Saves an output from the inversion code using inferpymc3_postprocessouts.

    Note: At the moment fluxapriori in the output is the mean apriori flux over the
          inversion period and so will not be identical to the original a priori flux, if
          it varies over the inversion period.
    """
    def isFloat(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    ds_in = setup.opends(input_file)

    # Read inputs from ncdf output
    start_date = ds_in.attrs['Start date']
    end_date = ds_in.attrs['End date']
    Hx = ds_in.xsensitivity.values.T
    Hbc = ds_in.bcsensitivity.values.T
    Y = ds_in.Yobs.values
    Ytime = ds_in.Ytime.values
    error = ds_in.Yerror.values
    siteindicator = ds_in.siteindicator.values
    sigma_freq_index = ds_in.sigmafreqindex.values
    xprior_string = ds_in.attrs["Emissions Prior"].split(",")
    xprior = {k:float(v) if isFloat(v) else v for k,v in zip(xprior_string[::2], xprior_string[1::2])}
    bcprior_string = ds_in.attrs["BCs Prior"].split(",")
    bcprior = {k:float(v) if isFloat(v) else v for k,v in zip(bcprior_string[::2], bcprior_string[1::2])}
    sigprior_string = ds_in.attrs["Model error Prior"].split(",")
    sigprior = {k:float(v) if isFloat(v) else v for k,v in zip(sigprior_string[::2], sigprior_string[1::2])}
    if 'Offset Prior' in ds_in.attrs.keys():
        offsetprior_string = ds_in.attrs["Offset Prior"].split(",")
        offsetprior = {k:float(v) if isFloat(v) else v for k,v in zip(offsetprior_string[::2], offsetprior_string[1::2])}
        add_offset = True
    else:
        add_offset = False
        offsetprior = None
    nit = len(ds_in.steps)
    burn = int(ds_in.attrs["Burn in"])
    tune = int(ds_in.attrs["Tuning steps"])
    nchain = int(ds_in.attrs['Number of chains'])
    if ds_in.attrs['Error for each site'] == "True":
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

    xouts, bcouts, sigouts, Ytrace, YBCtrace, convergence, step1, step2 = \
            mcmc.inferpymc3(Hx, Hbc, Y, error, siteindicator, sigma_freq_index,
                               xprior,bcprior, sigprior, nit, burn,
                               tune, nchain, sigma_per_site, offsetprior=offsetprior,
                               add_offset=add_offset, verbose=verbose)

    mcmc.inferpymc3_postprocessouts(xouts,bcouts, sigouts, convergence,
                                   Hx, Hbc, Y, error, Ytrace, YBCtrace,
                                   step1, step2,
                                   xprior, bcprior, sigprior, offsetprior, Ytime, siteindicator, sigma_freq_index,
                                   domain, species, sites,
                                   start_date, end_date, outputname, outputpath, country_unit_prefix,
                                   burn, tune, nchain, sigma_per_site,
                                   add_offset=add_offset, rerun_file=ds_in)
