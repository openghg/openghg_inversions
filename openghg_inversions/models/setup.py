"""Functions used for setting up inversions components."""

import numpy as np
import pandas as pd


def sigma_freq_indicies(ytime: np.ndarray, sigma_freq: str | None) -> np.ndarray:
    """Create an index that splits times into given periods.

    Args:
      ytime:
        concatenated array of time values for observations
      sigma_freq:
        either "monthly", a pandas format string ("30D"), or None
        this is the period of time to divide the time array into

    Returns:
      output:
        index array that defines periods against time
    """
    ydt = pd.to_datetime(ytime)
    output = np.zeros(shape=len(ytime)).astype(int)
    if sigma_freq is None:
        # output already all 0's as expected for this setting
        pass
    elif sigma_freq.lower() == "monthly":
        months = ydt.month
        years = ydt.year
        months_u = np.unique(months)
        years_u = np.unique(years)

        # incrementally set sigma indicies for each month in each year
        count = 0
        for y in years_u:
            for m in months_u:
                indicies = (years == y) & (months == m)
                output[indicies] = count
                count += 1
    else:
        # divide the time between t0 and ti by sigma_freq, then floor
        # to calculate number of integer intervals the calculation is
        # performed in seconds as division by pd time_delta is not allowed
        time_delta = pd.to_timedelta(sigma_freq)
        fractional_freq_time = (ydt - np.amin(ydt)).total_seconds() / time_delta.total_seconds()
        output[:] = np.floor(fractional_freq_time.values).astype(int)

    return output
