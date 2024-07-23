import pandas as pd
import pandas.testing as pdt
import numpy as np
import pytest

@pytest.mark.parametrize(
    "nperiod, nbasis, decay_time, sigma_time, mu_time, sigma_space, mu_space, expected_output1, expected_output2",
    [
        (
            2, 
            2, 
            1.0, 
            1.0, 
            1.0, 
            1.0, 
            1.0,
            np.array([[1.0, 0, np.exp(-1), 0],
                      [0, 1.0, 0, np.exp(-1)],
                      [np.exp(-1), 0, 1.0, 0],
                      [0, np.exp(-1), 0, 1.0]]),
            np.linalg.inv(np.array([[1.0, 0, np.exp(-1), 0],
                                    [0, 1.0, 0, np.exp(-1)],
                                    [np.exp(-1), 0, 1.0, 0],
                                    [0, np.exp(-1), 0, 1.0]]))
        ),
        (
            3, 
            2, 
            1.0, 
            1.0, 
            1.0, 
            1.0, 
            1.0,
            np.array([[1.0, 0, np.exp(-1), 0, np.exp(-2), 0],
                      [0, 1.0, 0, np.exp(-1), 0, np.exp(-2)],
                      [np.exp(-1), 0, 1.0, 0, np.exp(-1), 0],
                      [0, np.exp(-1), 0, 1.0, 0, np.exp(-1)],
                      [np.exp(-2), 0, np.exp(-1), 0, 1.0, 0],
                      [0, np.exp(-2), 0, np.exp(-1), 0, 1.0]]),


            np.linalg.inv(np.array([[1.0, 0, np.exp(-1), 0, np.exp(-2), 0],
                                    [0, 1.0, 0, np.exp(-1), 0, np.exp(-2)],
                                    [np.exp(-1), 0, 1.0, 0, np.exp(-1), 0],
                                    [0, np.exp(-1), 0, 1.0, 0, np.exp(-1)],
                                    [np.exp(-2), 0, np.exp(-1), 0, 1.0, 0],
                                    [0, np.exp(-2), 0, np.exp(-1), 0, 1.0]]))
        )
    ])
def test_xprior_covariance(nperiod, nbasis, decay_time, sigma_time, mu_time, sigma_space, mu_space, expected_output1, expected_output2):
    from openghg_inversions.correlated_params import xprior_covariance
    output1, output2 = xprior_covariance(nperiod, nbasis, decay_time, sigma_time, mu_time, sigma_space, mu_space)
    assert np.array_equal(output1, expected_output1)
    assert np.array_equal(output2, expected_output2)


@pytest.mark.parametrize(
    "x_freq, start_date, end_date, expected_output1, expected_output2, expected_output3",
    [
        (
            "weekly",
            "2024-04-30",
            "2024-05-23",
            pd.date_range(start="2024-04-30", end="2024-05-21", freq="W-TUE" ),
            [7, 7, 7, 2],
            4
        ),
        (
            "monthly",
            "2024-02-23",
            "2024-06-07",
            pd.date_range(start="2024-02-01", end="2024-06-07", freq="MS" ),
            [29, 31, 30, 31, 30],
            5
        )
    ])
def test_period_dates(x_freq, start_date, end_date, expected_output1, expected_output2, expected_output3):
    from openghg_inversions.correlated_params import period_dates
    output1, output2, output3 = period_dates(x_freq, start_date, end_date)
    assert output1.equals(expected_output1)
    assert np.array_equal(output2, expected_output2)
    assert output3 == expected_output3


@pytest.mark.parametrize(
    "data_time, period_dates, period, nperiod, expected_output",
    [
        (
            np.array(np.datetime64("2024-04-01") + np.arange(20) * np.timedelta64(4, 'h'), dtype='datetime64[ns]'),
            pd.date_range(start="2024-04-01", end="2024-04-04", freq="D" ),
            0,
            4,
            np.arange(6)
        )
    ])
def test_period_indices(data_time, period_dates, period, nperiod, expected_output):
    from openghg_inversions.correlated_params import period_indices
    output = period_indices(data_time, period_dates, period, nperiod)
    assert np.array_equal(output, expected_output)