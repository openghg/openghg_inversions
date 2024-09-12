"""Functions used for setting up HBMCMC inversions."""

import numpy as np
import pandas as pd
from typing import Union, Optional

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


def offset_matrix(siteindicator: np.ndarray) -> np.ndarray:
    """Set up a matrix that can be used to add an offset to each site.
    This will anchor to the first site (i.e. first site has no offset).

    Args:
      siteindicator:
        Array of values used for indicating the indices associated
        with each site used in the inversion

    Returns:
      2D array
    """
    b = np.zeros((int(len(siteindicator)), int(max(siteindicator)) + 1))
    for i in range(int(max(siteindicator) + 1)):
        b[siteindicator == i, i] = 1.0

    return b


def monthly_h(start_date : str, 
                end_date : str,
                site : str, 
                fp_data : dict) -> np.ndarray:
    
    """
    Creates a sensitivity matrix (H-matrix) for the emissions, 
    which will map monthly flux scalings to the observations. 
    This is for a single site.

    Args:
      start_date:
        Start time of inversion "YYYY-mm-dd"
      end_date:
        End time of inversion "YYYY-mm-dd"
      site:
        Site that you're creating it for
      fp_data:
        Output from utils..bc_sensitivity
      nbasis:
        Number of basis functions in inversion

    Returns:
      hx:
        Sensitivity matrix by month for observations
    """
    nbasis = fp_data[site].coords["region"].shape[0]
    allmonth = pd.date_range(start_date, end_date, freq="MS")[:-1]
    nmonth = len(allmonth)
    curtime = pd.to_datetime(fp_data[site].time.values).to_period("M")
    pmonth = pd.to_datetime(fp_data[site].resample(time="MS").mean().time.values)
    hx = np.zeros((nbasis * nmonth, len(fp_data[site].time.values)))
    count = 0
    for m in range(nmonth):
        if allmonth[m] not in pmonth:
            count += nbasis
            continue
        mnth = allmonth[m].month
        yr = allmonth[m].year
        mnthloc = np.where(np.logical_and(curtime.month == mnth, curtime.year == yr))[0]
        for basis in range(nbasis):
            hx[count, mnthloc] = fp_data[site].H.values[basis, mnthloc]
            count += 1

    return hx, nbasis, nmonth


def create_h_sensitivity(start_date : str, 
                          end_date : str, 
                          site : str, 
                          fp_data : dict, 
                          freq : str) -> np.ndarray:
    """
    Creates a sensitivity matrix (H-matrix) for the emissions, 
    which will map emission scalings to the observations. This 
    is for a single site. The frequency that the emission 
    sensitivity is specified over must be given in days. 

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
        Length-scale over which emissions sensitivities are
        specified over. Specified as in pandas, e.g. "30D".

    Returns:
      hx:
        Sensitivity matrix by for observations to emissions
    """
    nbasis = fp_data[site].coords["region"].shape[0]
    dys = int("".join([s for s in freq if s.isdigit()]))
    alldates = pd.date_range(
        pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.DateOffset(days=dys), freq=freq
    )
    ndates = np.sum(alldates < pd.to_datetime(end_date))
    curdates = fp_data[site].time.values
    hx = np.zeros((nbasis * ndates, len(fp_data[site].time.values)))
    count = 0
    for m in range(0, ndates):
        dateloc = np.where(
            np.logical_and(
                curdates >= alldates[m].to_datetime64(), curdates < alldates[m + 1].to_datetime64()
            )
        )[0]
        if len(dateloc) == 0:
            count += nbasis
            continue
        for basis in range(nbasis):
            hx[count, dateloc] = fp_data[site].H.values[basis, dateloc]
            count += 1

    return hx, nbasis, int(ndates)


def xprior_covariance(nperiod : int,
                      nbasis : int,
                      decay_tau : float,
                      sigma_period: Optional[float]=1.0,
                      sigma_space: Optional[float]=1.0,
                      ) -> np.ndarray:
    """
    Introduces a covariance matrix (with non-zero off-diagonal values) to allow for 
    multivariate correlation between state vector (x) parameters. Currently only 
    temporal correlation is considered. Basis functions are considered iid. Spatial 
    and temporal covariances combined using the Kronecker product.

    Args:
        nperiod (int): 
            The number of temporal periods in the inversion. If nperiod=4, the inversion
            is divided into 4 (nearly) equal time periods.
        nbasis (int):
            The number of basis functions in the inversion.
        sigma_time (float):
            The standard deviation of the prior distribution of each temporal parameter.
        mu_time (float):
            The mean of the prior distribution of each temporal parameter.
        sigma_space (float):
            The standard deviation of the prior distribution of each spatial parameter.
        mu_space (float):
            The mean of the prior distribution of each spatial parameter.
        decay_tau (float):
            The time constant of the expontential decay of the temporal correlation.

    Returns:
        numpy array:
            Precision matrix to be inserted directly into pymc inversion.
    
    """

    if decay_tau == 0:
        
        covariance_matrix = np.eye(int(nbasis*nperiod))
        
        precision_matrix = np.eye(int(nbasis*nperiod))

    range_time = np.arange(nperiod)  # period indexes 

    cov_period = sigma_period**2 * np.eye(nperiod)  # standard deviation of distribution for each period parameter

    cov_period_offdiag = []  # initialisation of array for of diagonal components of covariance matrix

    for i in np.arange(nperiod - 1) + 1:
        for j in np.arange(i):
            
            dt = range_time[i] - range_time[j]  # delta t for each time period

            rho_time = np.exp(-dt/decay_tau)  # calculation of correlation coefficent

            covariance_ij = cov_period[i, i] * cov_period[j, j] * rho_time  # calculation of correlation; cov(X, Y) = rho(X, Y) * var(X) * var(Y)

            cov_period_offdiag.append(covariance_ij)  # append covaraiance to off diagonal array

    cov_period[np.tril_indices(n=nperiod, k=-1)] = cov_period_offdiag  # assign off-diagonal values to lower left corner of matrix

    cov_period = cov_period + np.tril(cov_period, k=-1).T  # assign off-diagonal values to upper right corner of matrix
    
    inv_cov_period = np.linalg.inv(cov_period)  # calculate the inverse of the covariance matrix

    cov_space = sigma_space**2 * np.eye(nbasis)  # construct covariance matrix with off-diagonal zeros and diagonal variance; cov(X, X) = var(X)**2
    
    inv_cov_space = np.linalg.inv(cov_space)  # calculate the inverse of the covariance matrix

    covariance_matrix = np.kron(cov_period, cov_space)  # combine covariance matrices using the Kronecker product

    precision_matrix = np.kron(inv_cov_period, inv_cov_space)  # calculate the precision matrix; precision = cov^-1 = kron( cov_p^-1, cov_b^-1 )

    return covariance_matrix, precision_matrix


def covariance_extension(x_covariance: np.ndarray,
                         nbc: int,) -> np.ndarray:
    
    nx = x_covariance.shape[0]
    bc_sig = 0.1

    bc_var = np.ones(nbc)*bc_sig**2
    cov_extended = np.zeros((nx+nbc, nx+nbc))

    cov_extended[:nx, :nx] = x_covariance
    cov_extended[nx:, nx:] = np.diag(bc_var)

    return cov_extended