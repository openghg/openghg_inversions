# ****************************************************************************
# Created: 7 Nov. 2022
# Author: Eric Saboya, School of Geographical Sciences, University of Bristol
# Contact: eric.saboya@bristol.ac.uk
# ****************************************************************************
# About
#   Common Python functions that are called for converting time and gas species
#   units as part of data processing.  
#   Most functions have been copied from the University of Bristol
#   Atmospheric Chemistry Research Group (ACRG) repository and merged, here,
#   into one file.
# ****************************************************************************
# List of functions included (as of Dec. 2022):
#   molar_mass: Extracts the molar mass of a trace gas species
#   mol2g: Converts a value in moles to grams
#   prefix: Converts a prefix to its magnitude (e.g. T (Terra) -> 1e12)
#   concentration: Converts mol/mol to parts-per-X
#   convert_lons_0360: Converts longitudes to a 0-360 range from -180 to 180
#   check_iter:
#   return_iter:
#   reftime:
#   sec2time:
#   min2time:
#   hours2time:
#   day2time:
#   time2sec:
#   time2decimal:
#   decimal2time:
#   julian2time: Converts Julian dates to datetime
#   convert_to_hours: Converts an input time unit to its equivalent in hours
#
# ****************************************************************************
#  TODO: Add descriptions time-conversion functions
import os
import json
import calendar 
import dateutil
import time as tm
import datetime as dt
from matplotlib.dates import (julian2num, num2date)
from openghg_inversions.utils import synonyms
from openghg_inversions.config.paths import Paths

openghginv_path = Paths.openghginv

def molar_mass(species):
    '''
    Extracts the molar mass of a species from the species_info.json file.
    -----------------------------------
    Args:
      species (str):
        Atmospheric trace gas species of interest (e.g. 'co2')
    Returns:
        Molar mass of species (float)
    -----------------------------------
    '''
    species_info_file = os.path.join(openghginv_path,'data/species_info.json')
    with open(species_info_file) as f:
        species_info=json.load(f)
    species_key = synonyms(species,species_info)
    molmass = float(species_info[species_key]['mol_mass'])
    return molmass

def mol2g(value,species):
    ''' 
    Convert a value in moles to grams
    -----------------------------------
    Args:
      value (float):
        Value associated with the number of moles of species
      species (str)
        Atmospheric trace gas species of interest (e.g. 'co2')
    Returns:
      Corresponding value of trace gas species in grams (float)
    -----------------------------------
    '''
    molmass = molar_mass(species)
    return value*molmass

def prefix(units):
    ''' 
    Convert unit prefix to magnitude.
    -----------------------------------
    Args:
      units (str):
        Unit prefix of some quantity (e.g. T (Terra) --> 1e12)
    Returns:
      Unit magnitude (float)
    -----------------------------------
    '''
    if units is None:
        unit_factor = 1.
    elif units == 'T':
        unit_factor=1.e12
    elif units == 'G': 
        unit_factor=1.e9
    elif units == 'P': 
        unit_factor=1.e15
    elif units == 'M': 
        unit_factor=1.e6
    else:
        print('Undefined prefix: outputting in g/yr')
        unit_factor=1.

    return unit_factor

def concentration(units):
    '''
    Conversion between mol/mol to parts-per- units
    -----------------------------------
    Args:
      units (str):
        Numerical factor used for describing mole fraction
        e.g. (ppm, ppb, ppt)
    Returns:
      unit_factor (float)
        Numerical prefix magnitude
    -----------------------------------
    '''
    unit_factor = 1e-12 if units.lower() == 'ppt' else \
                  1e-9 if units.lower()  == 'ppb' else \
                  1e-6 if units.lower()  == 'ppm' else \
                  1
    if unit_factor==1:
        print('Undefined prefix')

    return unit_factor

def convert_lons_0360(lons):
    '''
    Convert longitude values onto a 0-360 range from -180-180 range. 
    Uses floored division. 
    ----------------------------------- 
    Args:
      lons (arr):
        1D array of longitudes.            

    Returns:
      lons (arr):
        Longitudes on 0-360 range. 
    -----------------------------------          
    '''
    div = lons // 360

    return lons - div*360

def check_iter(var):
    '''
    '''
    if not hasattr(var, '__iter__'):
        var = [var]
        notIter = True
    else:
        notIter = False

    return var, notIter

def return_iter(var, notIter):
    '''
    '''
    if notIter:
        return var[0]
    else:
        return var

def reftime(time_reference):
    '''
    '''
    time_reference, notIter = check_iter(time_reference)
    time_reference = return_iter(time_reference, notIter)
    #If reference time is a string, assume it's in CF convention 
    # and convert to datetime
    #if type(time_reference[0]) is str or type(time_reference[0]) is str:
    if isinstance(time_reference,str):
        time_ref=dateutil.parser.parse(time_reference)
    else:
        time_ref=time_reference

    return time_ref

def sec2time(seconds, time_reference):
    '''
    '''
    seconds, notIter = check_iter(seconds)

    time_ref = reftime(time_reference)

    return return_iter([time_ref +
        dt.timedelta(seconds=int(s)) for s in seconds], notIter)

def min2time(minutes, time_reference):
    '''
    '''
    minutes, notIter = check_iter(minutes)

    time_ref = reftime(time_reference)

    return return_iter([time_ref +
        dt.timedelta(minutes=m) for m in minutes], notIter)

def hours2time(hours, time_reference):
    '''
    '''
    hours, notIter = check_iter(hours)

    time_ref = reftime(time_reference)

    return return_iter([time_ref +
        dt.timedelta(hours=m) for m in hours], notIter)

def day2time(days, time_reference):
    '''
    '''
    days, notIter = check_iter(days)

    time_ref = reftime(time_reference)

    return return_iter([time_ref + dt.timedelta(days=d) for d in days],
                        notIter)

def time2sec(time, time_reference=None):
    '''
    '''
    time, notIter = check_iter(time)

    if time_reference is None:
        time_reference=dt.datetime(min(time).year, 1, 1, 0, 0)

    time_seconds=[\
        (t.replace(tzinfo=None)-time_reference).total_seconds() \
        for t in time]

    return return_iter(time_seconds, notIter), time_reference

def time2decimal(dates):
    '''
    '''
    def sinceEpoch(date): # returns seconds since epoch
        return tm.mktime(date.timetuple())
    s = sinceEpoch

    dates, notIter = check_iter(dates)

    frac=[]
    for date in dates:
        year = date.year
        startOfThisYear = dt.datetime(year=year, month=1, day=1)
        startOfNextYear = dt.datetime(year=year+1, month=1, day=1)

        yearElapsed = s(date) - s(startOfThisYear)
        yearDuration = s(startOfNextYear) - s(startOfThisYear)
        fraction = yearElapsed/yearDuration

        frac.append(date.year + fraction)

    return return_iter(frac, notIter)

def decimal2time(frac):
    '''
    '''
    frac, notIter = check_iter(frac)

    dates = []
    for f in frac:
        year = int(f)
        yeardatetime = dt.datetime(year, 1, 1)
        daysPerYear = 365 + calendar.leapdays(year, year+1)
        dates.append(yeardatetime + dt.timedelta(days = daysPerYear*(f - year)))

    return return_iter(dates, notIter)

def julian2time(dates):
    '''
    Convert Julian dates (e.g. from IDL) to datetime
    '''
    dates, notIter = check_iter(dates)

    dates_julian = []
    for date in dates:
        dates_julian.append(num2date(julian2num(date)))

    return return_iter(dates_julian, notIter)

def convert_to_hours(time):
    '''
    Convert to hours
    
    Returns in the input provided, float or list of floats
    '''
    hours_per_unit = {"H": 1, "D": 24, "W": 168, "M": 732, "Y":8760}
    if type(time) is list:
        time_hrs_list = []
        for ii in range(len(time)):
            time_hrs = float(time[ii][:-1]) * hours_per_unit[time[ii][-1]]
            time_hrs_list.append(time_hrs)
        return time_hrs_list
    else:
        time_hrs = float(time[:-1]) * hours_per_unit[time[-1]]
        return time_hrs

