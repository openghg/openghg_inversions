# *****************************************************************************
# Created: 7 Nov. 2022
# Author: Eric Saboya, School of Geographical Sciences, University of Bristol
# Contact: eric.saboya@bristol.ac.uk
# *****************************************************************************
# About
#   Common Python functions that are called for converting time and gas species
#   units as part of data processing.
#   Most functions have been copied from the University of Bristol
#   Atmospheric Chemistry Research Group (ACRG) repository and merged, here,
#   into one file.
# *****************************************************************************
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
# *****************************************************************************
#  TODO: Add descriptions time-conversion functions

from openghg.util import molar_mass

from openghg_inversions.config.paths import Paths


def mol2g(value, species):
    """
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
    """
    molmass = molar_mass(species)
    return value * molmass


def prefix(units):
    """
    Convert unit prefix to magnitude.
    -----------------------------------
    Args:
      units (str):
        Unit prefix of some quantity (e.g. T (Terra) --> 1e12)

    Returns:
      Unit magnitude (float)
    -----------------------------------
    """
    if units is None:
        unit_factor = 1.0
    elif units == "T":
        unit_factor = 1.0e12
    elif units == "G":
        unit_factor = 1.0e9
    elif units == "P":
        unit_factor = 1.0e15
    elif units == "M":
        unit_factor = 1.0e6
    else:
        print("Undefined prefix: outputting in g/yr")
        unit_factor = 1.0

    return unit_factor


def convert_to_hours(time):
    """
    Convert to hours

    Returns in the input provided, float or list of floats
    """
    hours_per_unit = {"H": 1, "D": 24, "W": 168, "M": 732, "Y": 8760}
    if type(time) is list:
        time_hrs_list = []
        for ii in range(len(time)):
            time_hrs = float(time[ii][:-1]) * hours_per_unit[time[ii][-1]]
            time_hrs_list.append(time_hrs)
        return time_hrs_list
    else:
        time_hrs = float(time[:-1]) * hours_per_unit[time[-1]]
        return time_hrs
