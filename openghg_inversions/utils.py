# ****************************************************************************
# Created: 7 Nov. 2022
# Author: Eric Saboya, School of Geographical Sciences, University of Bristol
# Contact: ericsaboya@bristol.ac.uk
# ****************************************************************************
# About
# Script containing common Python functions that can be called for running 
# HBMCMC and InTEM inversion models. 
# Many function have been copied form the acrg repo (e.g. acrg.name)
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

from os.path import join
from collections import OrderedDict


# modules from name.py
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
#from matplotlib import ticker
#import cartopy.crs as ccrs
#import cartopy


from acrg.time import convert
import acrg.obs as obs
from openghg_inversions.config.paths import Paths
#from acrg.utils import is_number

#from tqdm import tqdm
#import dask.array as da
acrg_path = Paths.acrg
data_path = Paths.data




# **** Functions from name.py ****
with open(acrg_path / "data/site_info.json") as f:
    site_info=json.load(f,object_pairs_hook=OrderedDict)

with open(acrg_path / "data/species_info.json") as f:
    species_info=json.load(f)

def open_ds(path, chunks=None, combine=None):
    """
    Function efficiently opens xray datasets.

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
        species_obs = obs.read.synonyms(species, species_info)   # <--------- Change obs.read.syn HERE! 

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

















# Everything below here is good! 

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

                                                                                                                                                                                                                 
