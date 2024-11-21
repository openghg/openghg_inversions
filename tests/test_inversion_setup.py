import pytest
import numpy as np
import pandas as pd

from openghg_inversions.hbmcmc.inversionsetup import sigma_freq_indicies

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
