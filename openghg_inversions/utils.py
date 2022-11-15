# ****************************************************************************
# Created: 7 Nov. 2022
# Author: Eric Saboya, School of Geographical Sciences, University of Bristol
# Contact: ericsaboya@bristol.ac.uk
# ****************************************************************************
# About
# Script containing common Python functions that can be called for running 
# HBMCMC and InTEM inversion models. 
# Most functions have been copied form the acrg repo (e.g. acrg.name)
#
# 
# List of Functions (updated 9 Nov. 2022)
# -----------------
# - open_ds: function for opening xarray datasets
# - filenames: function for providing name footprint filenames in specified
#              period
# - get_country (class)
# - filtering
# - areagrid
# - boundary_conditions: reads in data for domain to compute BCs 
# - read_netcdfs
# ****************************************************************************

import os 
import sys
import glob
import json
import bisect
import calendar
import subprocess
import pandas as pd
import datetime as dt
import numpy as np 
import xarray as xr
import dateutil.relativedelta

from tqdm import tqdm
import dask.array as da

from os.path import join
from collections import OrderedDict

from openghg_inversions import convert_time as convert
from openghg_inversions.config.paths import Paths

acrg_path = Paths.acrg
data_path = Paths.data


with open(acrg_path / "data/site_info.json") as f:
    site_info=json.load(f,object_pairs_hook=OrderedDict)

with open(acrg_path / "data/species_info.json") as f:
    species_info=json.load(f)

def open_ds(path, chunks=None, combine=None):
    """
    Function efficiently opens xarray datasets.

    Args:
        path (str)
        chunks (dict, optional)
            size of chunks for each dimension
            e.g. {'lat': 50, 'lon': 50}
            opens dataset with dask, such that it is opened 'lazily'
            and all of the data is not loaded into memory
            defaults to None - dataset is opened with out dask
        combine (str, optional)
            Way in which the data should be combined (if using chunks), either:
            'by_coords': order the datasets before concatenating (default)
            'nested': concatenate datasets in the order supplied
    """
    if chunks is not None:
        combine = 'by_coords' if combine is None else combine
        ds = xr.open_mfdataset(path, chunks=chunks, combine=combine)
    else:
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            ds.load()
    return ds


def read_netcdfs(files, dim = "time", chunks=None, verbose=True):
    """
    The read_netcdfs function uses xarray to open sequential netCDF files and 
    and concatenates them along the specified dimension.
    Note: this function makes sure that file is closed after open_dataset call.
    
    Args:
        files (list) : 
            List of netCDF filenames.
        dim (str, optional) : 
            Dimension of netCDF to use for concatenating the files.
            Default = "time".
        chunks (dict)
            size of chunks for each dimension
            e.g. {'lat': 50, 'lon': 50}
            opens dataset with dask, such that it is opened 'lazily'
            and all of the data is not loaded into memory
            defaults to None - dataset is opened with out dask
    
    Returns:
        xarray.Dataset : 
            All files open as one concatenated xarray.Dataset object    
    """
    if verbose:
        print("Reading and concatenating files: ")
        for fname in files:
            print(fname)

    datasets = [open_ds(p, chunks=chunks) for p in sorted(files)]

    # reindex all of the lat-lon values to a common one to prevent floating point error differences
    with xr.open_dataset(files[0]) as temp:
        fields_ds = temp.load()
    fp_lat = fields_ds["lat"].values
    fp_lon = fields_ds["lon"].values

    datasets = [ds.reindex(indexers={"lat":fp_lat, "lon":fp_lon}, method="nearest", tolerance=1e-5) for ds in datasets]

    combined = xr.concat(datasets, dim)
    return combined


def synonyms(search_string, info, alternative_label = "alt"):
    '''
    Check to see if there are other names that we should be using for
    a particular input. E.g. If CFC-11 or CFC11 was input, go on to use cfc-11,
    as this is used in species_info.json
    
    Args:
        search_string (str): Input string that you're trying to match
        info (dict): Dictionary whose keys are the "default" values, and an
             variable that contains other possible names
    Returns:
        corrected string
    '''

    keys=list(info.keys())

    #First test whether site matches keys (case insensitive)
    out_strings = \
        [k for k in keys if k.upper() == search_string.upper()]

    #If not found, search synonyms
    if len(out_strings) == 0:
        for k in keys:
            matched_strings = \
                [s for s in info[k][alternative_label] \
                    if s.upper() == search_string.upper()]
            if len(matched_strings) != 0:
                out_strings = [k]
                break

    if len(out_strings) == 1:
        out_string = out_strings[0]
    else:
        out_string = None

    return out_string


def boundary_conditions(domain, species, start = None, end = None, bc_directory=None):
    """
    The boundary_conditions function reads in the files with the global model vmrs at the domain edges 
    to give the boundary conditions as an xarray Dataset.

    Expect filenames of the form:
        [bc_directory]/domain/species.lower()_*.nc
        e.g. [/data/shared/LPDM/bc]/EUROPE/ch4_EUROPE_201301.nc

    Args:
        domain (str) : 
            Domain name. The boundary condition files should be sub-categorised by the domain.
        species (str) : 
            Species name. All species names are defined data/species_info.json.
        start (str, optional) : 
            Start date in format "YYYY-MM-DD" to output only a time slice of all the flux files.
            The start date used will be the first of the input month. I.e. if "2014-01-06" is input,
            "2014-01-01" will be used.  This is to mirror the time slice functionality of the filenames 
            function.
        end (str, optional): 
            End date in same format as start to output only a time slice of all the flux files.
            The end date used will be the first of the input month and the timeslice will go up
            to, but not include, this time. I.e. if "2014-02-25' is input, "2014-02-01" will be used.
            This is to mirror the time slice functionality of the filenames function.
        bc_directory (str, optional) : 
            bc_directory can be specified if files are not in the default directory. 
            Must point to a directory which contains subfolders organized by domain.

    Returns:
        xarray.Dataset : 
            Combined dataset of matching boundary conditions files
    """

    if bc_directory is None:
        bc_directory = join(data_path, 'LPDM/bc/')

    filenames = os.path.join(bc_directory,domain,f"{species.lower()}_*.nc")

    files = sorted(glob.glob(filenames))
    file_no_acc = [ff for ff in files if not os.access(ff, os.R_OK)]
    files = [ff for ff in files if os.access(ff, os.R_OK)]

    if len(file_no_acc)>0:
        print('Warning: unable to read all boundary conditions files which match this criteria:')
        [print(ff) for ff in file_no_acc]

    if len(files) == 0:
        print("Cannot find boundary condition files in {}".format(filenames))
        raise IOError(f"\nError: Cannot find boundary condition files for domain '{domain}' and species '{species}' ")

    bc_ds = read_netcdfs(files)

    if start == None or end == None:
        print("To get boundary conditions for a certain time period you must specify an end date.")
        return bc_ds
    else:
        #Change timeslice to be the beginning and end of months in the dates specified.
        start = pd.to_datetime(start)
        month_start = dt.datetime(start.year, start.month, 1, 0, 0)

        end = pd.to_datetime(end)
        month_end = dt.datetime(end.year, end.month, 1, 0, 0) - \
                    dt.timedelta(seconds = 1)

        bc_timeslice = bc_ds.sel(time=slice(month_start, month_end))
        if len(bc_timeslice.time)==0:
            bc_timeslice = bc_ds.sel(time=start, method = 'ffill')
            bc_timeslice = bc_timeslice.expand_dims('time',axis=-1)
            print(f"No boundary conditions available during the time period specified so outputting\
                    boundary conditions from {bc_timeslice.time.values[0]}")
        return bc_timeslice


class get_country(object):
  def __init__(self, domain, country_file=None):

        if country_file is None:
            filename=glob.glob(join(data_path,'LPDM/countries/',f"country_{domain}.nc"))
            f = xr.open_dataset(filename[0])
        else:
            filename = country_file
            f = xr.open_dataset(filename)

        lon = f.variables['lon'][:].values
        lat = f.variables['lat'][:].values

        #Get country indices and names
        if "country" in f.variables:
            country = f.variables['country'][:, :]
        elif "region" in f.variables:
            country = f.variables['region'][:, :]

#         if (ukmo is True) or (uk_split is True):
#             name_temp = f.variables['name'][:]  
#             f.close()
#             name=np.asarray(name_temp)

#         else:
        name_temp = f.variables['name'].values
        f.close()

        name_temp = np.ma.filled(name_temp,fill_value=None)

        name=[]
        for ii in range(len(name_temp)):
            if type(name_temp[ii]) is not str:
                name.append(''.join(name_temp[ii].decode("utf-8")))
            else:
                name.append(''.join(name_temp[ii]))
        name=np.asarray(name)


        self.lon = lon
        self.lat = lat
        self.lonmax = np.max(lon)
        self.lonmin = np.min(lon)
        self.latmax = np.max(lat)
        self.latmin = np.min(lat)
        self.country = np.asarray(country)
        self.name = name


def filtering(datasets_in, filters, keep_missing=False):
    """
    Applies filtering (in time dimension) to entire dataset.
    Filters supplied in a list and then applied in order. For example if you wanted a daily, daytime 
    average, you could do this:
    
        datasets_dictionary = filtering(datasets_dictionary, 
                                    ["daytime", "daily_median"])
    
    The order of the filters reflects the order they are applied, so for 
    instance when applying the "daily_median" filter if you only wanted
    to look at daytime values the filters list should be 
    ["daytime","daily_median"]                

    Args:
        datasets_in         : Output from footprints_data_merge(). Dictionary of datasets.
        filters (list)      : Which filters to apply to the datasets. 
                              All options are:
                                 "daytime"           : selects data between 1100 and 1500 local solar time
                                 "daytime9to5"       : selects data between 0900 and 1700 local solar time
                                 "nighttime"         : Only b/w 23:00 - 03:00 inclusive
                                 "noon"              : Only 12:00 fp and obs used
                                 "daily_median"      : calculates the daily median
                                 "pblh_gt_threshold" : 
                                 "local_influence"   : Only keep times when localness is low
                                 "six_hr_mean"       :
                                 "local_lapse"       :
        keep_missing (bool) : Whether to reindex to retain missing data.
    
    Returns:
       Same format as datasets_in : Datasets with filters applied. 
    """

    if type(filters) is not list:
        filters = [filters]

    datasets = datasets_in.copy()

    def local_solar_time(dataset):
        """
        Returns hour of day as a function of local solar time
        relative to the Greenwich Meridian. 
        """
        sitelon = dataset.release_lon.values[0]
        # convert lon to [-180,180], so time offset is negative west of 0 degrees
        if sitelon > 180:
            sitelon = sitelon - 360.
        dataset["time"] = dataset.time + pd.Timedelta(minutes=float(24*60*sitelon/360.))
        hours = dataset.time.to_pandas().index.hour
        return hours

    def local_ratio(dataset):
        """
        Calculates the local ratio in the surrounding grid cells
        """
        release_lons = dataset.release_lon[0].values
        release_lats = dataset.release_lat[0].values
        dlon = dataset.lon[1].values - dataset.lon[0].values
        dlat = dataset.lat[1].values-dataset.lat[0].values
        local_sum=np.zeros((len(dataset.mf)))

        for ti in range(len(dataset.mf)):
            release_lon=dataset.release_lon[ti].values
            release_lat=dataset.release_lat[ti].values
            wh_rlon = np.where(abs(dataset.lon.values-release_lon) < dlon/2.)
            wh_rlat = np.where(abs(dataset.lat.values-release_lat) < dlat/2.)
            if np.any(wh_rlon[0]) and np.any(wh_rlat[0]):
                local_sum[ti] = np.sum(dataset.fp[wh_rlat[0][0]-2:wh_rlat[0][0]+3,wh_rlon[0][0]-2:wh_rlon[0][0]+3,ti].values)/\
                                np.sum(dataset.fp[:,:,ti].values)
            else:
                local_sum[ti] = 0.0

        return local_sum

    # Filter functions
    def daily_median(dataset, keep_missing=False):
        """ Calculate daily median """
        if keep_missing:
            return dataset.resample(indexer={'time':"1D"}).median()
        else:
            return dataset.resample(indexer={'time':"1D"}).median().dropna(dim="time")

    def six_hr_mean(dataset, keep_missing=False):
        """ Calculate six-hour median """
        if keep_missing:
            return dataset.resample(indexer={'time':"6H"}).mean()
        else:
            return dataset.resample(indexer={'time':"6H"}).mean().dropna(dim="time")


    def daytime(dataset, site,keep_missing=False):
        """ Subset during daytime hours (11:00-15:00) """
        hours = local_solar_time(dataset)
        ti = [i for i, h in enumerate(hours) if h >= 11 and h <= 15]

        if keep_missing:
            dataset_temp = dataset[dict(time = ti)]
            dataset_out = dataset_temp.reindex_like(dataset)
            return dataset_out
        else:
            return dataset[dict(time = ti)]

    def daytime9to5(dataset, site,keep_missing=False):
        """ Subset during daytime hours (9:00-17:00) """
        hours = local_solar_time(dataset)
        ti = [i for i, h in enumerate(hours) if h >= 9 and h <= 17]

        if keep_missing:
            dataset_temp = dataset[dict(time = ti)]
            dataset_out = dataset_temp.reindex_like(dataset)
            return dataset_out
        else:
            return dataset[dict(time = ti)]

    def nighttime(dataset, site,keep_missing=False):
        """ Subset during nighttime hours (23:00 - 03:00) """
        hours = local_solar_time(dataset)
        ti = [i for i, h in enumerate(hours) if h >= 23 or h <= 3]

        if keep_missing:
            dataset_temp = dataset[dict(time = ti)]
            dataset_out = dataset_temp.reindex_like(dataset)
            return dataset_out
        else:
            return dataset[dict(time = ti)]

    def noon(dataset, site,keep_missing=False):
        """ Select only 12pm data """
        hours = local_solar_time(dataset)
        ti = [i for i, h in enumerate(hours) if h == 12]

        if keep_missing:
            dataset_temp = dataset[dict(time = ti)]
            dataset_out = dataset_temp.reindex_like(dataset)
            return dataset_out
        else:
            return dataset[dict(time = ti)]

    def local_influence(dataset,site, keep_missing=False):
        """
        Subset for times when local influence is below threshold.       
        Local influence expressed as a fraction of the sum of entire footprint domain.
        """
        if not dataset.filter_by_attrs(standard_name="local_ratio"):
            lr = local_ratio(dataset)
        else:
            lr = dataset.local_ratio

        pc = 0.1
        ti = [i for i, local_ratio in enumerate(lr) if local_ratio <= pc]
        if keep_missing is True:
            mf_data_array = dataset.mf
            dataset_temp = dataset.drop('mf')

            dataarray_temp = mf_data_array[dict(time = ti)]

            mf_ds = xr.Dataset({'mf': (['time'], dataarray_temp)},
                                  coords = {'time' : (dataarray_temp.coords['time'])})

            dataset_out = combine_datasets(dataset_temp, mf_ds, method=None)
            return dataset_out
        else:
            return dataset[dict(time = ti)]


    filtering_functions={"daily_median":daily_median,
                         "daytime":daytime,
                         "daytime9to5":daytime9to5,
                         "nighttime":nighttime,
                         "noon":noon,
                         "local_influence":local_influence,
                         "six_hr_mean":six_hr_mean}


    # Get list of sites
    sites = [key for key in list(datasets.keys()) if key[0] != '.']

    # Do filtering
    for site in sites:

            for filt in filters:
                if filt == "daily_median" or filt == "six_hr_mean":
                    datasets[site] = filtering_functions[filt](datasets[site], keep_missing=keep_missing)
                else:
                    datasets[site] = filtering_functions[filt](datasets[site], site, keep_missing=keep_missing)

    return datasets


def areagrid(lat, lon):
  """Calculates grid of areas (m2) given arrays of latitudes and longitudes

  Args:
      lat (array): 
          1D array of latitudes
      lon (array): 
          1D array of longitudes
        
  Returns:
      area (array): 
          2D array of areas of of size lat x lon
      
  Example:
    import acrg_grid
    lat=np.arange(50., 60., 1.)
    lon=np.arange(0., 10., 1.)
    area=acrg_grid.areagrid(lat, lon)
    
  """

  re=6367500.0  #radius of Earth in m

  dlon=abs(np.mean(lon[1:] - lon[0:-1]))*np.pi/180.
  dlat=abs(np.mean(lat[1:] - lat[0:-1]))*np.pi/180.
  theta=np.pi*(90.-lat)/180.

  area=np.zeros((len(lat), len(lon)))

  for latI in range(len(lat)):
    if theta[latI] == 0. or np.isclose(theta[latI], np.pi):
      area[latI, :]=(re**2)*abs(np.cos(dlat/2.)-np.cos(0.))*dlon
    else:
      lat1=theta[latI] - dlat/2.
      lat2=theta[latI] + dlat/2.
      area[latI, :]=((re**2)*(np.cos(lat1)-np.cos(lat2))*dlon)

  return area


def filenames(site, domain, start, end, height, fp_directory, met_model = None, network=None, species=None):
    """
    The filenames function outputs a list of available footprint file names,
    for given site, domain, directory and date range.
    
    Expect filenames of the form:
        [fp_directory]/domain/site*-height-species*domain*ym*.nc or [fp_directory]/domain/site*-height_domain*ym*.nc
        e.g. /data/shared/LPDM/fp_NAME/EUROPE/HFD-UKV-100magl-rn_EUROPE_202012.nc or /data/shared/LPDM/fp_NAME/EUROPE/MHD-10magl_EUROPE_201401.nc 
    
    Args:
        site (str) : 
            Site name. Full list of site names should be defined within data/site_info.json
        domain (str) : 
            Domain name. The footprint files should be sub-categorised by the NAME domain name.
        start (str) : 
            Start date in format "YYYY-MM-DD" for range of files to find.
        end (str) : 
            End date in same format as start for range of files to find.
        height (str) : 
            Height related to input data. 
        fp_directory (str) :
            fp_directory can be specified if files are not in the default directory must point to a directory 
            which contains subfolders organized by domain.
        met_model (str):
            Met model used to run NAME
            Default is None and implies the standard global met model
            Alternates include 'UKV' and this met model must be in the outputfolder NAME        
        network (str, optional) : 
            Network for site. 
            If not specified, first entry in data/site_info.json file will be used (if there are multiple).
        species (str, optional)
            If specified, will search for species specific footprint files.
    Returns:
        list (str): matched filenames
    """

    # Read site info for heights
    if height is None:
        if not site in list(site_info.keys()):
            print("Site code not found in data/site_info.json to get height information. " + \
                  "Check that site code is as intended. "+ \
                  "If so, either add new site to file or input height manually.")
            return None
        if network is None:
            network = list(site_info[site].keys())[0]
        height = site_info[site][network]["height_name"][0]

    if species:
        species_obs = synonyms(species, species_info)    

        if 'lifetime' in species_info[species_obs].keys():
            lifetime = species_info[species_obs]["lifetime"]
            lifetime_hrs = convert.convert_to_hours(lifetime)
            # if a monthly lifetime is a list, use the minimum lifetime 
            # in the list to determine whether a species specific footprint is needed
            if type(lifetime) == list:
                lifetime_hrs = min(lifetime_hrs)
        else:
            lifetime_hrs = None
    else:
        lifetime_hrs = None

    # Convert into time format
    months = pd.date_range(start = start, end = end, freq = "M").to_pydatetime()
    yearmonth = [str(d.year) + str(d.month).zfill(2) for d in months]

    # first search for species specific footprint files.
    # if does not exist, use integrated files if lifetime of species is over 2 months
    # if lifetime under 2 months and no species specific file exists, fail

    files = []
    for ym in yearmonth:

        if species:

            if met_model:
                f=glob.glob(join(fp_directory,domain,f"{site}-{height}_{met_model}_{species}_{domain}_{ym}*.nc"))
            else:
                f=glob.glob(join(fp_directory,domain,f"{site}-{height}_{species}_{domain}_{ym}*.nc"))


        else:
            #manually create empty list if no species specified
            f = []

        if len(f) == 0:

            if met_model:
                glob_path = join(fp_directory,domain,f"{site}-{height}_{met_model}_{domain}_{ym}*.nc")
            else:
                glob_path = join(fp_directory,domain,f"{site}-{height}_{domain}_{ym}*.nc")

            if lifetime_hrs is None:
                print("No lifetime defined in species_info.json or species not defined. WARNING: 30-day integrated footprint used without chemical loss.")
                f=glob.glob(glob_path)
            elif lifetime_hrs <= 1440:
                print("This is a short-lived species. Footprints must be species specific. Re-process in process.py with lifetime")
                return []
            else:
                print("Treating species as long-lived.")
                f=glob.glob(glob_path)

        if len(f) > 0:
            files += f

    files.sort()

    if len(files) == 0:
        print(f"Can't find footprints file: {glob_path}")
    return files


def basis(domain, basis_case, basis_directory = None):
    """
    The basis function reads in the all matching files for the basis case and domain as an xarray Dataset.
    
    Expect filenames of the form:
        [basis_directory]/domain/"basis_case"_"domain"*.nc
        e.g. [/data/shared/LPDM/basis_functions]/EUROPE/sub_transd_EUROPE_2014.nc

    TODO: More info on options for basis functions.

    Args:
        domain (str) : 
            Domain name. The basis files should be sub-categorised by the domain.
        basis_case (str) : 
            Basis case to read in. Examples of basis cases are "voroni","sub-transd",
            "sub-country_mask","INTEM".
        basis_directory (str, optional) : 
            basis_directory can be specified if files are not in the default directory. 
            Must point to a directory which contains subfolders organized by domain.
    
    Returns:
        xarray.Dataset : combined dataset of matching basis functions
    """
    if basis_directory is None:
        basis_directory = os.path.join(data_path, 'LPDM/basis_functions/')

    file_path = os.path.join(basis_directory,domain,f"{basis_case}_{domain}*.nc")
    files = sorted(glob.glob(file_path))

    if len(files) == 0:
        raise IOError(f"\nError: Can't find basis function files for domain '{domain}' \
                          and basis_case '{basis_case}' ")

    basis_ds = read_netcdfs(files)

    return basis_ds


def basis_boundary_conditions(domain, basis_case, bc_basis_directory = None):
    """
    The basis_boundary_conditions function reads in all matching files for the boundary conditions 
    basis case and domain as an xarray Dataset.
    
    Expect filesnames of the form:
        [bc_basis_directory]/domain/"basis_case"_"domain"*.nc
        e.g. [/data/shared/LPDM/bc_basis_directory]/EUROPE/NESW_EUROPE_2013.nc

    TODO: More info on options for basis functions.
    
    Args:
        domain (str) : 
            Domain name. The basis files should be sub-categorised by the domain.
        basis_case (str) : 
            Basis case to read in. Examples of basis cases are "NESW","stratgrad".
        bc_basis_directory (str, optional) : 
            bc_basis_directory can be specified if files are not in the default directory. 
            Must point to a directory which contains subfolders organized by domain.
    
    Returns:
        xarray.Datset : combined dataset of matching basis functions
    """
    if bc_basis_directory is None:
        bc_basis_directory = os.path.join(data_path,'LPDM/bc_basis_functions/')

    file_path = os.path.join(bc_basis_directory,domain,f"{basis_case}_{domain}*.nc")

    files = sorted(glob.glob(file_path))
    file_no_acc = [ff for ff in files if not os.access(ff, os.R_OK)]
    files = [ff for ff in files if os.access(ff, os.R_OK)]

    if len(file_no_acc)>0:
        print('Warning: unable to read all boundary conditions basis function files which match this criteria:')
        [print(ff) for ff in file_no_acc]

    if len(files) == 0:
        raise IOError("\nError: Can't find boundary condition basis function files for domain '{0}' "
              "and basis_case '{1}' ".format(domain,basis_case))

    basis_ds = read_netcdfs(files)

    return basis_ds


def indexesMatch(dsa, dsb):
    """
    Check if two datasets need to be reindexed_like for combine_datasets
    
    Args:
        dsa (xarray.Dataset) : 
            First dataset to check
        dsb (xarray.Dataset) : 
            Second dataset to check
            
    Returns:
        boolean:
            True if indexes match, False if datasets must be reindexed
    """

    commonIndicies  = [key for key in dsa.indexes.keys() if key in dsb.indexes.keys()]

    #test if each comon index is the same
    for index in commonIndicies:
        #first check lengths are the same to avoid error in second check
        if not len(dsa.indexes[index])==len(dsb.indexes[index]):
            return False

        #check number of values that are not close (testing for equality with floating point)
        if index == "time":
            #for time iverride the default to have ~ second precision
            rtol = 1e-10
        else:
            rtol = 1e-5
        if not np.sum(~np.isclose(dsa.indexes[index].values.astype(float),dsb.indexes[index].values.astype(float), rtol=rtol ))==0:
            return False

    return True


def combine_datasets(dsa, dsb, method = "ffill", tolerance = None):
    """
    The combine_datasets function merges two datasets and re-indexes to the FIRST dataset.
    If "fp" variable is found within the combined dataset, the "time" values where the "lat","lon"
    dimensions didn't match are removed.
    
    Example:
        ds = combine_datasets(dsa, dsb)

    Args:
        dsa (xarray.Dataset) : 
            First dataset to merge
        dsb (xarray.Dataset) : 
            Second dataset to merge
        method (str, optional) : 
            One of {None, ‘nearest’, ‘pad’/’ffill’, ‘backfill’/’bfill’}
            See xarray.DataArray.reindex_like for list of options and meaning.
            Default = "ffill" (forward fill)
        tolerance (int/float??) : 
            Maximum allowed tolerance between matches.

    Returns:
        xarray.Dataset: 
            Combined dataset indexed to dsa
    """
    # merge the two datasets within a tolerance and remove times that are NaN (i.e. when FPs don't exist)

    if not indexesMatch(dsa, dsb):
        dsb_temp = dsb.reindex_like(dsa, method, tolerance = tolerance)
    else:
        dsb_temp = dsb

    ds_temp = dsa.merge(dsb_temp)
    if 'fp' in list(ds_temp.keys()):
        flag = np.where(np.isfinite(ds_temp.fp.mean(dim=["lat","lon"]).values))
        ds_temp = ds_temp[dict(time = flag[0])]
    return ds_temp


def timeseries_HiTRes(flux_dict, fp_HiTRes_ds=None, fp_file=None, output_TS = True, output_fpXflux = True,
                      output_type='Dataset', output_file=None, verbose=False, chunks=None,
                      time_resolution='1H'):
    """
    The timeseries_HiTRes function computes flux * HiTRes footprints.
    
    HiTRes footprints record the footprint at each 2 hour period back in time for the first 24 hours.
    Need a high time resolution flux to multiply the first 24 hours back of footprints.
    Need a residual flux to multiply the residual integrated footprint for the remainder of the 30 
    day period.
    
    Args:
        fp_HiTRes_ds (xarray.Dataset)
            Dataset of high time resolution footprints. HiTRes footprints record the footprint at 
            each timestep back in time for a given amount of time
            (e.g. hourly time steps back in time for the first 24 hours).
        domain (str)
            Domain name. The footprint files should be sub-categorised by the domain.
        flux_dict (dict)
            This should be a dictionary of the form output in the format
            flux_dict: {'high_freq': flux_dataset, 'low_freq': flux_dataset}.
            This is because this function needs two time resolutions of fluxes as
            explained in the header.
            
            If there are multiple sectors, the format should be:
            flux_dict: {sector1 : {'high_freq' : flux_dataset, 'low_freq'  : flux_dataset},
                        sector2 : {'high_freq' : flux_dataset, 'low_freq'  : flux_dataset}}
        output_TS (bool)
            Output the timeseries. Default is True.
        output_fpXflux (bool)
            Output the sensitivity map. Default is True.
        verbose (bool)
            show progress bar throughout loop
        chunks (dict)
            size of chunks for each dimension
            e.g. {'lat': 50, 'lon': 50}
            opens dataset with dask, such that it is opened 'lazily'
            and all of the data is not loaded into memory
            defaults to None - dataset is opened with out dask
    
    Returns:
        xarray.Dataset or dict
            Same format as flux_dict['high_freq']:
                If flux_dict['high_freq'] is an xarray.Dataset then an xarray.Dataset is returned
                If flux_dict['high_freq'] is a dict of xarray.Datasets then a dict of xarray.Datasets
                is returned (an xarray.Dataset for each sector)
        
            If output_TS is True:
                Outputs the timeseries
            If output_fpXflux is True:
                Outputs the sensitivity map   
    """
    if verbose:
        print(f'\nCalculating timeseries with {time_resolution} resolution, this might take a few minutes')
    ### get the high time res footprint
    if fp_HiTRes_ds is None and fp_file is None:
        print('Must provide either a footprint Dataset or footprint filename')
        return None
    elif fp_HiTRes_ds is None:
        fp_HiTRes_ds = read_netcdfs(fp_file, chunks=chunks)
        fp_HiTRes    = fp_HiTRes_ds.fp_HiTRes
    else:
        fp_HiTRes = fp_HiTRes_ds if type(fp_HiTRes_ds)==xr.core.dataarray.DataArray else \
                    fp_HiTRes_ds.fp_HiTRes.chunk(chunks) \
                    if fp_HiTRes_ds.chunks is None and chunks is not None else fp_HiTRes_ds.fp_HiTRes

    # resample fp to match the required time resolution
    fp_HiTRes  = fp_HiTRes.resample(time=time_resolution).ffill()

    # get the H_back timestep and max number of hours back
    H_back_hour_diff = int(fp_HiTRes["H_back"].diff(dim="H_back").values.mean())
    max_H_back = int(fp_HiTRes["H_back"].values[-2])

    # create time array to loop through, with the required resolution
    time_array = fp_HiTRes.time.values
    # extract as a dask array to make the loop quicker
    fp_HiTRes  = da.array(fp_HiTRes)

    # this is the number of hours over which the H_back will be resampled to match time_resolution
    H_resample = int(time_resolution[0]) if H_back_hour_diff == 1 else \
                 1 if H_back_hour_diff == int(time_resolution[0]) else None
    if H_resample is None:
        print('Cannot resample H_back')
        return None
    # reverse the H_back coordinate to be chronological, and resample to match time_resolution
    fp_HiTRes   = fp_HiTRes[:,:,:,::-H_resample]

    # convert fluxes to dictionaries with sectors as keys
    flux = {'total': flux_dict} if any([ff in list(flux_dict.keys()) for ff in ['high_freq', 'low_freq']]) else flux_dict
    flux = {sector: {freq: None if flux_freq is None else flux_freq
                   for freq, flux_freq in flux_sector.items()}
            for sector, flux_sector in flux.items()}

    # extract the required time data
    flux = {sector: {freq: flux_freq.sel(time=slice(fp_HiTRes_ds.time[0] - np.timedelta64(max_H_back, 'h'),
                                                    fp_HiTRes_ds.time[-1])).flux
                     if flux_freq is not None else None
                     for freq, flux_freq in flux_sector.items() }
            for sector, flux_sector in flux.items()}

    for sector, flux_sector in flux.items():
        if 'high_freq' in flux_sector.keys() and flux_sector['high_freq'] is not None:
            # reindex the high frequency data to match the fp
            time_flux = np.arange(fp_HiTRes_ds.time[0].values - np.timedelta64(max_H_back, 'h'),
                                  fp_HiTRes_ds.time[-1].values + np.timedelta64(time_resolution[0],
                                                                                time_resolution[1].lower()),
                                  time_resolution[0], dtype=f'datetime64[{time_resolution[1].lower()}]')
            flux_sector['high_freq'] = flux_sector['high_freq'].reindex(time=time_flux, method='ffill')
        else:
            print(f'\nWarning: no high frequency flux data for {sector}, estimating a timeseries using the low frequency data')
            flux_sector['high_freq'] = None

        if 'low_freq' not in flux_sector.keys() or flux_sector['low_freq'] is None:
            print(f'\nWarning: no low frequency flux data for {sector}, resampling from high frequency data')
            flux_sector['low_freq'] = flux_sector['high_freq'].resample(time='MS').mean()

    # convert to array to use in numba loop
    flux = {sector: {freq: None if flux_freq is None else
                           flux_freq.values if flux_freq.chunks is None else
                           da.array(flux_freq)
                     for freq, flux_freq in flux_sector.items()}
            for sector, flux_sector in flux.items()}

    # Set up a numpy array to calculate the product of the footprint (H matrix) with the fluxes
    if output_fpXflux:
        fpXflux   = {sector: da.zeros((len(fp_HiTRes_ds.lat),
                                       len(fp_HiTRes_ds.lon),
                                       len(time_array)))
                     for sector in flux.keys()}

    elif output_TS:
        timeseries = {sector: da.zeros(len(time_array)) for sector in flux.keys()}

    # month and year of the start of the data - used to index the low res data
    start = {dd: getattr(np.datetime64(time_array[0], 'h').astype(object), dd)
             for dd in ['month', 'year']}

    # put the time array into tqdm if we want a progress bar to show throughout the loop
    iters = tqdm(time_array) if verbose else time_array
    ### iterate through the time coord to get the total mf at each time step using the H back coord
    # at each release time we disaggregate the particles backwards over the previous 24hrs
    for tt, time in enumerate(iters):
        # get 4 dimensional chunk of high time res footprint for this timestep
        # units : mol/mol/mol/m2/s
        fp_time = fp_HiTRes[:,:,tt,:]

        # get the correct index for the low res data
        # estimated using the difference between the current and start month and year
        current = {dd: getattr(np.datetime64(time, 'h').astype(object), dd)
                   for dd in ['month', 'year']}
        tt_low = current['month'] - start['month'] + 12*(current['year']-start['year'])

        # select the high res emissions for the corresponding 24 hours
        # if there aren't any high frequency data it will select from the low frequency data
        # this is so that we can compare emissions data with different resolutions e.g. ocean species
        emissions     = {sector: flux_sector['high_freq'][:,:,tt:tt+fp_time.shape[2]-1]
                                 if flux_sector['high_freq'] is not None else
                                 flux_sector['low_freq'][:,:,tt_low]
                         for sector, flux_sector in flux.items()}
        # add an axis if the emissions is array is 2D so that it can be multiplied by the fp
        emissions     = {sector: em_sec[:,:,np.newaxis] if len(em_sec.shape)==2 else em_sec
                        for sector, em_sec in emissions.items()}
        # select average monthly emissions for the start of the month
        emissions_end = {sector: flux_sector['low_freq'][:,:,tt_low]
                         for sector, flux_sector in flux.items()}

        # Multiply the HiTRes footprint with the HiTRes emissions to give mf
        # we take all but the slice for H_back==24 as these are the hourly disaggregated fps
        # flux units : mol/m2/s;       fp units : mol/mol/mol/m2/s
        # --> mol/mol/mol/m2/s * mol/m2/s === mol / mol
        fpXflux_time  = {sector: em_sec * fp_time[:,:,1:] for sector, em_sec in emissions.items()}
        # multiply the monthly flux by the residual fp, at H_back==24
        fpXflux_end   = {sector: em_end * fp_time[:,:,0] for sector, em_end in emissions_end.items()}
        # append the residual emissions
        fpXflux_time  = {sector: np.dstack((fp_fl, fpXflux_end[sector]))
                         for sector, fp_fl in fpXflux_time.items()}

        for sector, fp_fl in fpXflux_time.items():
            if output_fpXflux:
                # Sum over time (H back) to give the total mf at this timestep
                fpXflux[sector][:,:,tt] = np.nansum(fp_fl, axis=2)

            elif output_TS:
                # work out timeseries by summing over lat, lon, & time (24 hrs)
                timeseries[sector][tt] = np.nansum(fp_fl)

    if output_fpXflux and output_TS:
        # if not already done then calculate the timeseries
        timeseries = {sector: fp_fl.sum(axis=(0,1)) for sector, fp_fl in fpXflux.items()}

    if output_fpXflux:
        fpXflux = {sec : (['lat', 'lon', 'time'], ff_sec)
                   for sec, ff_sec in fpXflux.items()} \
                  if output_type.lower()=='dataset' else fpXflux

        fpXflux = xr.Dataset(fpXflux,
                             coords={'lat'  : fp_HiTRes_ds.lat.values,
                                     'lon'  : fp_HiTRes_ds.lon.values,
                                     'time' : time_array}) \
                  if output_type.lower()=='dataset' else \
                  {sector: xr.DataArray(data = fpXflux_sector,
                                        dims = ['lat', 'lon', 'time'],
                                        coords = {'lat' : fp_HiTRes_ds.lat.values,
                                                  'lon'  : fp_HiTRes_ds.lon.values,
                                                  'time' : time_array})
                   for sector, fpXflux_sector in fpXflux.items()} \
                  if output_type.lower()=='dataarray' else fpXflux

        if output_type.lower()=='dataset':
            fpXflux = fpXflux if fpXflux.chunks is None else fpXflux.compute()
        else:
            fpXflux = {sec: ff if ff.chunks is None else ff.compute() for sec, ff in fpXflux.items()}

        if output_type.lower()=='dataarray' and list(flux.keys())==['total']:
            fpXflux = fpXflux['total']

    if output_fpXflux and not output_TS:
        return fpXflux

    else:
        # for each sector create a tuple of ['time'] (coord name) and the timeseries
        # if the output required is a dataset
        timeseries = {sec : (['time'], ts_sec) for sec, ts_sec in timeseries.items()} \
                     if output_type.lower()=='dataset' else timeseries
        timeseries = xr.Dataset(timeseries,
                                coords={'time' : time_array}) \
                     if output_type.lower()=='dataset' else \
                     {sector: xr.DataArray(data = ts_sector,
                                           dims = ['time'],
                                           coords = {'time' : time_array})
                      for sector, ts_sector in timeseries.items()} \
                     if output_type.lower()=='dataarray' else timeseries

        if output_type.lower()=='dataset':
            timeseries = timeseries if timeseries.chunks is None else timeseries.compute()
        else:
            timeseries = {sec: tt if tt.chunks is None else tt.compute() for sec, tt in timeseries.items()}

        if output_type.lower()=='dataarray' and list(flux.keys())==['total']:
            timeseries = timeseries['total']

        if output_file is not None and output_type.lower()=='dataset':
            print(f'Saving to {output_file}')
            timeseries.to_netcdf(output_file)
        elif output_file is not None:
            print(f'output type must be dataset to save to file')

        if output_fpXflux:
            return timeseries, fpXflux
        elif output_TS:
            return timeseries


def fp_sensitivity(fp_and_data, domain, basis_case,
                   basis_directory = None, verbose=True):

    """
    The fp_sensitivity function adds a sensitivity matrix, H, to each site xarray dataframe in fp_and_data.

    Basis function data in an array: lat, lon, no. regions. In each 'region'
    element of array there is a lt lon grid with 1 in region and 0 outside region.
    
    Region numbering must start from 1
    
    Args:
        fp_and_data (dict)    : Output from footprints_data_merge() function. Dictionary of datasets.
        domain (str)          : Domain name. The footprint files should be sub-categorised by the domain.
        basis_case            : Basis case to read in. Examples of basis cases are "NESW","stratgrad".
                                String if only one basis case is required. Dict if there are multiple
                                sources that require separate basis cases. In which case, keys in dict should
                                reflect keys in emissions_name dict used in fp_data_merge.
        basis_directory (str) : basis_directory can be specified if files are not in the default 
                                directory. Must point to a directory which contains subfolders organized 
                                by domain. (optional)
    
    Returns:
        dict (xarray.Dataset) : Same format as fp_and_data with sensitivity matrix and basis function grid added.
    """

    sites = [key for key in list(fp_and_data.keys()) if key[0] != '.']

    flux_sources = list(fp_and_data['.flux'].keys())

    if type(basis_case) is not dict:
        if len(flux_sources) == 1:
            basis_case = {flux_sources[0]:basis_case}
        else:
            basis_case = {'all':basis_case}

    if len(list(basis_case.keys())) != len(flux_sources):
        if len(list(basis_case.keys())) == 1:
            print("Using %s as the basis case for all sources" %basis_case[list(basis_case.keys())[0]])
        else:
            print("There should either only be one basis_case, or it should be a dictionary the same length\
                  as the number of sources.")
            return None

    for site in sites:
        for si, source in enumerate(flux_sources):
            if source in list(basis_case.keys()):
                basis_func = basis(domain=domain, basis_case=basis_case[source], basis_directory=basis_directory)
            else:
                basis_func = basis(domain=domain, basis_case=basis_case['all'], basis_directory=basis_directory)

            if type(fp_and_data['.flux'][source]) == dict:
                if 'fp_HiTRes' in list(fp_and_data[site].keys()):
                    site_bf = xr.Dataset({"fp_HiTRes":fp_and_data[site]["fp_HiTRes"],
                                          "fp":fp_and_data[site]["fp"]})

                    fp_time = (fp_and_data[site].time[1] - fp_and_data[site].time[0]).values.astype('timedelta64[h]').astype(int)

                    # calculate the H matrix
                    H_all = timeseries_HiTRes(fp_HiTRes_ds = site_bf, flux_dict = fp_and_data['.flux'][source], output_TS = False,
                                              output_fpXflux = True, output_type = 'DataArray',
                                              time_resolution = f'{fp_time}H', verbose = verbose)
                else:
                    print("fp_and_data needs the variable fp_HiTRes to use the emissions dictionary with high_freq and low_freq emissions.")

            else:
                site_bf = combine_datasets(fp_and_data[site]["fp"].to_dataset(), fp_and_data['.flux'][source].data)
                H_all=site_bf.fp*site_bf.flux

            H_all_v=H_all.values.reshape((len(site_bf.lat)*len(site_bf.lon),len(site_bf.time)))


            if 'region' in list(basis_func.dims.keys()):

                if 'time' in basis_func.basis.dims:
                    basis_func = basis_func.isel(time=0)

                site_bf = xr.merge([site_bf, basis_func])

                H = np.zeros((len(site_bf.region),len(site_bf.time)))

                base_v = site_bf.basis.values.reshape((len(site_bf.lat)*len(site_bf.lon), len(site_bf.region)))

                for i in range(len(site_bf.region)):
                    H[i,:] = np.sum(H_all_v*base_v[:,i,np.newaxis], axis = 0)

                if source == all:
                    if (sys.version_info < (3,0)):
                        region_name = site_bf.region
                    else:
                        region_name = site_bf.region.decode('ascii')
                else:
                    if (sys.version_info < (3,0)):
                        region_name = [source+'-'+reg for reg in site_bf.region.values]
                    else:
                        region_name = [source+'-'+reg.decode('ascii') for reg in site_bf.region.values]

                sensitivity = xr.DataArray(H,
                                           coords=[('region', region_name),
                                                   ('time', fp_and_data[site].coords['time'])])

            else:
                print("Warning: Using basis functions without a region dimension may be deprecated shortly.")

                site_bf = combine_datasets(site_bf,basis_func, method='ffill')

                H = np.zeros((int(np.max(site_bf.basis)),len(site_bf.time)))

                basis_scale = xr.Dataset({'basis_scale': (['lat','lon','time'],
                                                    np.zeros(np.shape(site_bf.basis)))},
                                       coords = site_bf.coords)
                site_bf = site_bf.merge(basis_scale)

                base_v = np.ravel(site_bf.basis.values[:,:,0])
                for i in range(int(np.max(site_bf.basis))):
                    wh_ri = np.where(base_v == i+1)
                    H[i,:]=np.sum(H_all_v[wh_ri[0],:], axis = 0)

                if source == all:
                    region_name = list(range(1,np.max(site_bf.basis.values)+1))
                else:
                    region_name = [source+'-'+str(reg) for reg in range(1,int(np.max(site_bf.basis.values)+1))]

                sensitivity = xr.DataArray(H.data,
                                    coords=[('region', region_name),
                                        ('time', fp_and_data[site].coords['time'].data)])

            if si == 0:
                concat_sensitivity = sensitivity
            else:
                concat_sensitivity = xr.concat((concat_sensitivity,sensitivity), dim='region')

            sub_basis_cases = 0

            if basis_case[source].startswith('sub'):
                """
                To genrate sub_lon and sub_lat grids basis case must start with 'sub'
                e.g.
                'sub-transd', 'sub_transd', sub-intem' will work
                'transd' or 'transd-sub' won't work
                """
                sub_basis_cases += 1
                if sub_basis_cases > 1:
                    print("Can currently only use a sub basis case for one source. Skipping...")
                else:
                    sub_fp_temp = site_bf.fp.sel(lon=site_bf.sub_lon, lat=site_bf.sub_lat,
                                                 method="nearest")
                    sub_fp = xr.Dataset({'sub_fp': (['sub_lat','sub_lon','time'], sub_fp_temp.data)},
                                           coords = {'sub_lat': (site_bf.coords['sub_lat'].data),
                                                     'sub_lon': (site_bf.coords['sub_lon'].data),
                                                     'time' : (fp_and_data[site].coords['time'].data)})

                    sub_H_temp = H_all.sel(lon=site_bf.sub_lon, lat=site_bf.sub_lat,
                                           method="nearest")
                    sub_H = xr.Dataset({'sub_H': (['sub_lat','sub_lon','time'], sub_H_temp.data)},
                                          coords = {'sub_lat': (site_bf.coords['sub_lat'].data),
                                                    'sub_lon': (site_bf.coords['sub_lon'].data),
                                                    'time' : (fp_and_data[site].coords['time'].data)},
                                          attrs = {'flux_source_used_to_create_sub_H':source})

                    fp_and_data[site] = fp_and_data[site].merge(sub_fp)
                    fp_and_data[site] = fp_and_data[site].merge(sub_H)

        fp_and_data[site]['H'] = concat_sensitivity
        fp_and_data['.basis'] = site_bf.basis[:,:,0]

    return fp_and_data


def bc_sensitivity(fp_and_data, domain, basis_case, bc_basis_directory = None):

    """
    The bc_sensitivity adds H_bc to the sensitivity matrix, to each site xarray dataframe in fp_and_data.
    
    Args:
        fp_and_data (dict)       : Output from footprints_data_merge() function. Dictionary of datasets.
        domain (str)             : Domain name. The footprint files should be sub-categorised by the domain.
        basis_case (str)         : Basis case to read in. Examples of basis cases are "NESW","stratgrad".
        bc_basis_directory (str) : bc_basis_directory can be specified if files are not in the default 
                                   directory. Must point to a directory which contains subfolders organized 
                                   by domain. (optional)
    
    Returns:
        dict (xarray.Dataset) : Same format as fp_and_data with sensitivity matrix added.
    """

    sites = [key for key in list(fp_and_data.keys()) if key[0] != '.']

    basis_func = basis_boundary_conditions(domain = domain,
                                           basis_case = basis_case, bc_basis_directory=bc_basis_directory)
# sort basis_func into time order    
    ind = basis_func.time.argsort()
    timenew = basis_func.time[ind]
    basis_func = basis_func.reindex({"time":timenew})

    species = fp_and_data[".species"]
    species = synonyms(species, species_info)

    for site in sites:
# ES commented out line below as .bc not attribute. Also assume openghg adds all relevant particle data to file.
#        if fp_and_data[site].bc.chunks is not None:
        for particles in ['particle_locations_n', 'particle_locations_e',
                          'particle_locations_s', 'particle_locations_w']:
            fp_and_data[site][particles] = fp_and_data[site][particles].compute()

        # compute any chemical loss to the BCs, use lifetime or else set loss to 1 (no loss)
        if 'lifetime' in species_info[species].keys():
            lifetime = species_info[species]["lifetime"]
            lifetime_hrs_list_or_float = convert.convert_to_hours(lifetime)

            # calculate the lifetime_hrs associated with each time point in fp_and_data
            # this is because lifetime can be a list of monthly values

            time_month = fp_and_data[site].time.dt.month
            if type(lifetime_hrs_list_or_float) is list:
                lifetime_hrs = [lifetime_hrs_list_or_float[item-1] for item in time_month.values]
            else:
                lifetime_hrs = lifetime_hrs_list_or_float

            loss_n = np.exp(-1*fp_and_data[site].mean_age_particles_n/lifetime_hrs).rename('loss_n')
            loss_e = np.exp(-1*fp_and_data[site].mean_age_particles_e/lifetime_hrs).rename('loss_e')
            loss_s = np.exp(-1*fp_and_data[site].mean_age_particles_s/lifetime_hrs).rename('loss_s')
            loss_w = np.exp(-1*fp_and_data[site].mean_age_particles_w/lifetime_hrs).rename('loss_w')
        else:
            loss_n = fp_and_data[site].particle_locations_n.copy()
            loss_e = fp_and_data[site].particle_locations_e.copy()
            loss_s = fp_and_data[site].particle_locations_s.copy()
            loss_w = fp_and_data[site].particle_locations_w.copy()
            loss_n[:]=1
            loss_e[:]=1
            loss_s[:]=1
            loss_w[:]=1

        DS_particle_loc = xr.Dataset({"particle_locations_n":fp_and_data[site]["particle_locations_n"],
                                "particle_locations_e":fp_and_data[site]["particle_locations_e"],
                                "particle_locations_s":fp_and_data[site]["particle_locations_s"],
                                "particle_locations_w":fp_and_data[site]["particle_locations_w"],
                                "loss_n":loss_n,
                                "loss_e":loss_e,
                                "loss_s":loss_s,
                                "loss_w":loss_w})
#                                 "bc":fp_and_data[site]["bc"]})

        DS_temp = combine_datasets(DS_particle_loc, fp_and_data[".bc"].data, method='ffill')

        DS = combine_datasets(DS_temp, basis_func, method='ffill')

        DS = DS.transpose('height','lat','lon','region','time')

        part_loc = np.hstack([DS.particle_locations_n,
                                DS.particle_locations_e,
                                DS.particle_locations_s,
                                DS.particle_locations_w])

        loss = np.hstack([DS.loss_n,
                                DS.loss_e,
                                DS.loss_s,
                                DS.loss_w])

        vmr_ed = np.hstack([DS.vmr_n,
                           DS.vmr_e,
                           DS.vmr_s,
                           DS.vmr_w])

        bf = np.hstack([DS.bc_basis_n,
                        DS.bc_basis_e,
                        DS.bc_basis_s,
                        DS.bc_basis_w])

        H_bc = np.zeros((len(DS.coords['region']),len(DS["particle_locations_n"]["time"])))

        for i in range(len(DS.coords['region'])):
            reg = bf[:,:,i,:]
            H_bc[i,:] = np.sum((part_loc*loss*vmr_ed*reg), axis=(0,1))

        sensitivity = xr.Dataset({'H_bc': (['region_bc','time'], H_bc)},
                                    coords = {'region_bc': (DS.coords['region'].values),
                                              'time' : (DS.coords['time'])})

        fp_and_data[site] = fp_and_data[site].merge(sensitivity)

    return fp_and_data


