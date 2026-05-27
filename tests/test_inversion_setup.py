import pytest
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.hbmcmc.inversionsetup import sigma_freq_indicies, monthly_bcs
from openghg_inversions.hbmcmc.hbmcmc import make_inv_inputs

def test_sigma_freq_indicies():
    ytime = pd.date_range("2020-06-01", "2021-06-01", freq="4h")
    sigma_freq = "monthly"

    sigma_freq_index = sigma_freq_indicies(ytime, sigma_freq)
    nsigma_time = np.unique(sigma_freq_index)
    nsigma_site = [0]
    sigma = np.arange(len(nsigma_time)).reshape(1, -1)

    try:
        sigma[nsigma_site, sigma_freq_index]
    except IndexError:
        pytest.fail("Indexing sigma with nsigma_site and sigma_freq_index failed.")


def test_make_inv_inputs_with_empty_root_datatree():
    time = pd.date_range("2019-01-01", periods=3)

    standard = xr.Dataset(
        {
            "H": xr.DataArray(np.ones((2, 3)), dims=["region", "time"], coords={"time": time}),
            "H_bc": xr.DataArray(np.ones((4, 3)), dims=["bc_region", "time"], coords={"time": time}),
            "mf": xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["time"], coords={"time": time}),
            "mf_error": xr.DataArray(np.ones(3) * 0.1, dims=["time"], coords={"time": time}),
            "mf_repeatability": xr.DataArray(np.ones(3) * 0.05, dims=["time"], coords={"time": time}),
            "mf_variability": xr.DataArray(np.ones(3) * 0.02, dims=["time"], coords={"time": time}),
        },
        attrs={"inlet": "185m", "platform": "site"},
    )
    inner = xr.Dataset(
        {
            "H_inner": xr.DataArray(np.ones((1, 3)), dims=["region", "time"], coords={"time": time}),
        }
    )

    fp_data = {"TAC": xr.DataTree.from_dict({"/standard": standard, "/inner": inner})}

    mcmc_args, _ = make_inv_inputs(
        fp_data=fp_data,
        sites=["TAC"],
        dropped_sites=[],
        start_date="2019-01-01",
        end_date="2019-02-01",
        use_bc=True,
        bc_freq=None,
        sigma_freq=None,
        xprior={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        bcprior={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        sigprior={"pdf": "uniform", "lower": 0.1, "upper": 3.0},
        nit=20,
        burn=2,
        tune=5,
        nchain=1,
        sigma_per_site=True,
        offsetprior={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
        add_offset=False,
        verbose=False,
        min_error=0.0,
        calculate_min_error=None,
        min_error_options=None,
        offset_args=None,
        power=1.99,
    )

    assert mcmc_args["Hx"].shape == (2, 3)
    assert mcmc_args["Hx_inner"].shape == (1, 3)
    assert mcmc_args["Hbc"].shape == (4, 3)


def test_monthly_bcs_reads_standard_child_hbc():
    time = pd.date_range("2019-01-01", periods=6, freq="15D")
    hbc_vals = np.ones((4, len(time)))

    standard = xr.Dataset(
        {
            "H_bc": xr.DataArray(
                hbc_vals,
                dims=["bc_region", "time"],
                coords={"bc_region": [0, 1, 2, 3], "time": time},
            )
        }
    )
    fp_data = {"TAC": xr.DataTree.from_dict({"/standard": standard})}

    hmbc = monthly_bcs("2019-01-01", "2019-04-01", "TAC", fp_data)

    assert hmbc.shape[0] == 4 * 3
    assert hmbc.shape[1] == len(time)
