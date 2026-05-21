"""Functions used for setting up HBMCMC inversions."""

import numpy as np
import pandas as pd
import xarray as xr


def _site_ds(fp_data: dict, site: str):
  entry = fp_data[site]

  if isinstance(entry, xr.DataTree):
    if "standard" in entry.children and entry["standard"].ds is not None:
      return entry["standard"].ds

    if entry.ds is not None:
      return entry.ds

    # Fallback: use first non-empty child dataset if present.
    for child in entry.children.values():
      if child.ds is not None:
        return child.ds

    raise ValueError(f"Site '{site}' DataTree does not contain a dataset in root or child nodes.")

  return entry


def _site_hbc(fp_data: dict, site: str) -> xr.DataArray:
  site_ds = _site_ds(fp_data, site)
  if "H_bc" not in site_ds.data_vars:
    raise ValueError(
      f"Boundary-condition sensitivity 'H_bc' is missing for site '{site}'. "
      f"Available variables: {list(site_ds.data_vars)}"
    )
  return site_ds["H_bc"]


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
    site_ds = _site_ds(fp_data, site)
    h_bc = _site_hbc(fp_data, site).values
    allmonth = pd.date_range(start_date, end_date, freq="MS")[:-1]
    nmonth = len(allmonth)
    curtime = pd.to_datetime(site_ds.time.values).to_period("M")
    pmonth = pd.to_datetime(site_ds.resample(time="MS").mean().time.values)
    nregions = site_ds.sizes["bc_region"]
    hmbc = np.zeros((nregions * nmonth, len(site_ds.time.values)))
    count = 0
    for cord in range(nregions):
        for m in range(0, nmonth):
            if allmonth[m] not in pmonth:
                count += 1
                continue
            mnth = allmonth[m].month
            yr = allmonth[m].year
            mnthloc = np.where(np.logical_and(curtime.month == mnth, curtime.year == yr))[0]
            hmbc[count, mnthloc] = h_bc[cord, mnthloc]
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
    site_ds = _site_ds(fp_data, site)
    h_bc = _site_hbc(fp_data, site).values
    dys = int("".join([s for s in freq if s.isdigit()]))
    alldates = pd.date_range(
        pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.DateOffset(days=dys), freq=freq
    )
    ndates = np.sum(alldates < pd.to_datetime(end_date))
    curdates = site_ds.time.values
    nregions = site_ds.sizes["bc_region"]
    hmbc = np.zeros((nregions * ndates, len(site_ds.time.values)))
    count = 0
    for cord in range(nregions):
        for m in range(0, ndates):
            dateloc = np.where(
                np.logical_and(
                    curdates >= alldates[m].to_datetime64(), curdates < alldates[m + 1].to_datetime64()
                )
            )[0]
            if len(dateloc) == 0:
                count += 1
                continue
            hmbc[count, dateloc] = h_bc[cord, dateloc]
            count += 1

    return hmbc


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
                if not np.any(indicies):
                  continue
                else:
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
