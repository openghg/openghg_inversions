"""Common Python functions that are called for converting time and gas species
units as part of data processing.
"""


from openghg.util import molar_mass


def mol2g(value: float, species: str) -> float:
    """Convert a value in moles to grams.

    Args:
      value: Value associated with the number of moles of species
      species: Atmospheric trace gas species of interest (e.g. 'co2')

    Returns:
        Corresponding value of trace gas species in grams (as float)
    """
    molmass = molar_mass(species)
    return value * molmass


def prefix(units: str) -> float:
    """Convert unit prefix to magnitude.

    Args:
        units: Unit prefix of some quantity (e.g. T (Terra) --> 1e12)

    Returns:
        Unit magnitude (float)
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
        print(f"Undefined prefix: {units}. Outputting in g/yr")
        unit_factor = 1.0

    return unit_factor


def convert_to_hours(time: str | list[str]) -> float | list[float]:
    """Convert (list of) times to hours.

    Args:
        time: time or list of times as a string (e.g. "4D" or ["2D", "1W"])

    Returns: times from input converted to hours; output is is a list if the input
        is a list.
    """
    hours_per_unit = {"H": 1, "D": 24, "W": 168, "M": 732, "Y": 8760}

    if isinstance(time, list):
        time_hrs_list = [float(t[:-1]) * hours_per_unit[t[-1]] for t in time]
        return time_hrs_list
    else:
        time_hrs = float(time[:-1]) * hours_per_unit[time[-1]]
        return time_hrs
