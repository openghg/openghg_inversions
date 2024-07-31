# ****************************************************************************
# Created: 1 May 2024
# Author: Stephen Pearson, School of Geographical Sciences, University of Bristol
# Contact: s.pearson@bristol.ac.uk
# ****************************************************************************
# About
# The functions required to enable correlated state vector variables in the
# openghg_inversions hbmcmc.py inversion.
#
# ****************************************************************************

import numpy as np
import pandas as pd
import scipy


def xprior_covariance(nperiod,
                      nbasis,
                      decay_time,
                      sigma_time=1.0,
                      sigma_space=1.0,
                      ):
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
        decay_time (float):
            The time constant of the expontential decay of the temporal correlation.

    Returns:
        numpy array:
            Precision matrix to be inserted directly into pymc inversion.
    
    """

    range_time = np.arange(nperiod)  # period indexes 

    cov_time_diag = sigma_time * np.ones(nperiod)  # standard deviation of distribution for each period parameter

    cov_time_offdiag = []  # initialisation of array for of diagonal components of covariance matrix

    for i in np.arange(nperiod - 1) + 1:
        for j in np.arange(i):
            
            dt = range_time[i] - range_time[j]  # delta t for each time period

            rho_time = np.exp(-dt/decay_time)  # calculation of correlation coefficent

            covariance_ij = cov_time_diag[i] * cov_time_diag[j] * rho_time  # calculation of correlation; cov(X, Y) = rho(X, Y) * var(X) * var(Y)

            cov_time_offdiag.append(covariance_ij)  # append covaraiance to off diagonal array


    cov_period = np.diag(cov_time_diag**2)  # construct covariance matrix with off-diagonal zeros and diagonal variance; cov(X, X) = var(X)**2

    cov_period[np.tril_indices(n=nperiod, k=-1)] = cov_time_offdiag  # assign off-diagonal values to lower left corner of matrix

    cov_period = cov_period + np.tril(cov_period, k=-1).T  # assign off-diagonal values to upper right corner of matrix
    
    inv_cov_period = np.linalg.inv(cov_period)  # calculate the inverse of the covariance matrix

    cov_space_diag = sigma_space * np.ones(nbasis)  # standard deviation of distribution for each basis function parameter

    cov_space = np.diag(cov_space_diag**2)  # construct covariance matrix with off-diagonal zeros and diagonal variance; cov(X, X) = var(X)**2
    
    inv_cov_space = np.linalg.inv(cov_space)  # calculate the inverse of the covariance matrix

    covariance_matrix = np.kron(cov_period, cov_space)  # combine covariance matrices using the Kronecker product

    precision_matrix = np.kron(inv_cov_period, inv_cov_space)  # calculate the precision matrix; precision = cov^-1 = kron( cov_p^-1, cov_b^-1 )

    return covariance_matrix, precision_matrix


def period_dates(x_freq,
                 start_date,
                 end_date,
                 ):
    
    """
    Determines the start date of each period within a temporally correlated state
    vector. Also calculates the number of days within each period to allow for period
    flux calculation in inferpymc_postprocessouts.

    Args:
        xfreq (str):
            The period over which temporal correlation is to be considered. Currently 
            designed to consider "weekly" or "monthly" values. If "monthly", the inversion
            will calculate full monthly emissions, even if all of the days in the month are
            not included in the period. i.e. if the end date is 2024-04-05, then emissions
            for all of April will be inferred from the 4 days considered.

        start_date (str):
            The start date of the inversion. YYYY-MM-DD

        end_date (str):
            The end date of the inversion. YYYY-MM-DD. Up-to and not including last date

    Returns:

    """
    
    if x_freq is None:
        period_dates = pd.DatetimeIndex([pd.to_datetime(start_date)])
        days_in_period = (pd.to_datetime(end_date, yearfirst=True) - pd.to_datetime(start_date, yearfirst=True)).days
        nperiod = 1

    elif x_freq == "weekly":
        datetime_freq = "W-" + pd.to_datetime(start_date, yearfirst=True).strftime('%a')
        first_date = start_date    
        s = pd.to_datetime(start_date, yearfirst=True).day_of_week
        f = pd.to_datetime(end_date, yearfirst=True).day_of_week
        if s != f:
            # last_date = pd.to_datetime(end_date, yearfirst=True) + pd.DateOffset(days=(f-s+7)%7)
            last_date = pd.to_datetime(end_date, yearfirst=True) - pd.DateOffset(days=(f-s))
            print("Final week of inversion is incomplete ...")
        else:    
            last_date = pd.to_datetime(end_date, yearfirst=True) - pd.DateOffset(days=(7))
        
        period_dates = pd.date_range(start=first_date, end=last_date, freq=datetime_freq)
        final_period = (pd.to_datetime(end_date) - period_dates[-1]).days
        del(s,f)

    elif x_freq == "monthly":
        datetime_freq = "MS"
        if pd.to_datetime(start_date, yearfirst=True).day != 1:
            first_date = pd.to_datetime(start_date, yearfirst=True) - pd.offsets.MonthBegin(1)
            print("First month of inversion is incomplete ...")
        else:
            first_date = start_date

        if pd.to_datetime(end_date, yearfirst=True).day == 1:
            last_date = pd.to_datetime(end_date, yearfirst=True) - pd.DateOffset(days=1)
        else:
            last_date = end_date
            print("Final month of inversion is incomplete")
    
        period_dates = pd.date_range(start=first_date, end=last_date, freq=datetime_freq)
        final_period = pd.to_datetime(last_date).days_in_month
    
    else:
        raise ValueError("Inversion not set up for '{}' periods".format(x_freq))
    
    if x_freq is not None:
        nperiod = len(period_dates)
        
        # print("nperiod = {} \n nbasis = {}".format(nperiod, nbasis))

        days_in_period = period_dates.to_series().diff().dt.days
        days_in_period = days_in_period.iloc[1:].astype(int).values

        days_in_period = np.append(days_in_period, final_period)

        # print("Days in each period: {}".format(days_in_period))

    return period_dates, days_in_period, nperiod


def period_indices(data_time, period_dates, period, nperiod):
    
    """
    Determines the indices of fp_data that correspond to each period of the inversion.

    Args:
        data_time (array):
            The time array from fp_data[site]
        period_dates (pandas date_range):
            The start date of each period
        period:
            The index of the period being considered
    
    Returns:

    """

    if period == nperiod-1:
        period_ind = np.where(data_time >= period_dates[period])[0]
    else: 
        period_ind = np.where((data_time >= period_dates[period]) 
                            & (data_time < period_dates[period+1]))[0]
        
    return period_ind


def block_formation(H_blocks, Y_blocks, Ytime_blocks, error_blocks, siteindicator_blocks, Hx, Y, Ytime, error, nperiod, period_dates, si):
    
    siteindicator = np.ones_like(Y) * si

    if si == 0:

        for period in np.arange(nperiod):
        
            period_ind = period_indices(Ytime, period_dates, period, nperiod)

            H_blocks[period] = Hx[:, period_ind]
            Y_blocks[period] = Y[period_ind]
            Ytime_blocks[period] = Ytime[period_ind]
            error_blocks[period] = error[period_ind]
            siteindicator_blocks[period] = siteindicator[period_ind]
        
    else:

        for period in np.arange(nperiod):
        
            period_ind = period_indices(Ytime, period_dates, period, nperiod)

            H_blocks[period] = np.hstack((H_blocks[period], Hx[:, period_ind]))
            Y_blocks[period] = np.hstack((Y_blocks[period], Y[period_ind]))
            Ytime_blocks[period] = np.hstack((Ytime_blocks[period], Ytime[period_ind]))
            error_blocks[period] = np.hstack((error_blocks[period], error[period_ind]))
            siteindicator_blocks[period] = np.hstack((siteindicator_blocks[period], siteindicator[period_ind]))

    return H_blocks, Y_blocks, Ytime_blocks, error_blocks, siteindicator_blocks


def monthly_bcs_blocks(bcs_blocks, Hmbc, Ytime, period_dates, nperiod, si):

    if si ==0:

        for period in np.arange(nperiod):

            period_ind = period_indices(Ytime, period_dates, period, nperiod)

            bcs_blocks[period] = Hmbc[:, period_ind]

    else:
        
        for period in np.arange(nperiod):

            period_ind = period_indices(Ytime, period_dates, period, nperiod)

            bcs_blocks[period] = np.hstack(bcs_blocks[period], Hmbc[:, period_ind])

    return bcs_blocks


def block_diag_h(H_blocks):

    H_blocks = list(H_blocks.values())

    Hx = scipy.linalg.block_diag(*H_blocks)

    return Hx


def single_vector(vector_dict):

    vectors = list(vector_dict.values())

    single_vector = np.concatenate(vectors)

    return single_vector