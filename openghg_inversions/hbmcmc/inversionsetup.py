"""Functions used for setting up HBMCMC inversions."""

import numpy as np
import pandas as pd


def monthly_bcs(start_date: str, end_date: str, site: str, fp_data: dict) -> np.ndarray:
    """Creates a sensitivity matrix (H-matrix) for the boundary
    conditions, which will map monthly boundary condition
    scalings to the observations. This is for a single site.

    Args:
      start_date:
        Start time of inversion "YYYY-mm-dd"
      end_date:
        End time of inversion "YYYY-mm-dd"
      site:
        Site that you're creating it for
      fp_data:
        Output from utils..bc_sensitivity

    Returns:
      hmbc:
        Sensitivity matrix by month for observations
    """
    allmonth = pd.date_range(start_date, end_date, freq="MS")[:-1]
    nmonth = len(allmonth)
    curtime = pd.to_datetime(fp_data[site].time.values).to_period("M")
    pmonth = pd.to_datetime(fp_data[site].resample(time="MS").mean().time.values)
    hmbc = np.zeros((4 * nmonth, len(fp_data[site].time.values)))
    count = 0
    for cord in range(4):
        for m in range(0, nmonth):
            if allmonth[m] not in pmonth:
                count += 1
                continue
            mnth = allmonth[m].month
            yr = allmonth[m].year
            mnthloc = np.where(np.logical_and(curtime.month == mnth, curtime.year == yr))[0]
            hmbc[count, mnthloc] = fp_data[site].H_bc.values[cord, mnthloc]
            count += 1

    return hmbc


def create_bc_sensitivity(start_date: str, end_date: str, site: str, fp_data: dict, freq: str) -> np.ndarray:
    """Creates a sensitivity matrix (H-matrix) for the boundary
    conditions, which will map boundary condition scalings to
    the observations. This is for a single site. The frequency
    that the boundary condition sensitivity is specified over
    must be given in days. Currently only works for a
    boundary condition from each cardinal direction.

    Args:
      start_date:
        Start time of inversion "YYYY-mm-dd"
      end_date:
        End time of inversion "YYYY-mm-dd"
      site:
        Site that you're creating it for
      fp_data:
        Output from ModelScenario()
        Should be a dictionary of xr.Dataset/DataArray
      freq:
        Length-scale over which boundary condition sensitivities are
        specified over. Specified as in pandas, e.g. "30D".

    Returns:
      hmbc:
        Sensitivity matrix by for observations to boundary conditions
    """
    dys = int("".join([s for s in freq if s.isdigit()]))
    alldates = pd.date_range(
        pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.DateOffset(days=dys), freq=freq
    )
    ndates = np.sum(alldates < pd.to_datetime(end_date))
    curdates = fp_data[site].time.values
    hmbc = np.zeros((4 * ndates, len(fp_data[site].time.values)))
    count = 0
    for cord in range(4):
        for m in range(0, ndates):
            dateloc = np.where(
                np.logical_and(
                    curdates >= alldates[m].to_datetime64(), curdates < alldates[m + 1].to_datetime64()
                )
            )[0]
            if len(dateloc) == 0:
                count += 1
                continue
            hmbc[count, dateloc] = fp_data[site].H_bc.values[cord, dateloc]
            count += 1

    return hmbc
