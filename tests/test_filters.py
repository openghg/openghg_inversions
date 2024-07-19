import numpy as np
import pytest

from openghg_inversions.hbmcmc.hbmcmc import fixedbasisMCMC
from openghg_inversions.filters import filtering, filtering_functions
from openghg_inversions.get_data import load_merged_data


@pytest.fixture
def mcmc_args(tmp_path, tac_ch4_data_args, merged_data_dir, merged_data_file_name):
    mcmc_args = tac_ch4_data_args.copy()
    mcmc_args.update(
        {
            "outputname": "test_run",
            "outputpath": str(tmp_path),
            "basis_algorithm": "quadtree",
            "basis_output_path": str(tmp_path),
            "nbasis": 4,
            "nit": 1,
            "burn": 0,
            "tune": 0,
            "nchain": 1,
            "reload_merged_data": True,
            "merged_data_dir": merged_data_dir,
            "merged_data_name": merged_data_file_name,
        }
    )
    return mcmc_args


@pytest.fixture
def merged_data(merged_data_dir, merged_data_file_name):
    result = load_merged_data(
        merged_data_dir=merged_data_dir,
        species="ch4",
        start_date="2019-01-01",
        output_name="test_run",
        merged_data_name=merged_data_file_name,
    )
    return result


def test_pblh_filter_error(merged_data):
    with pytest.raises(NotImplementedError):
        filtering(merged_data, ["pblh"])


def test_all_filters(merged_data):
    for name in filtering_functions:
        if name != "pblh":
            filtering(merged_data, [name])


def test_filters_as_list(merged_data):
    filters = ["pblh_inlet_diff", "pblh_min"]
    filtering(merged_data, filters)


def test_filters_as_dict(merged_data):
    filters = {"TAC": ["pblh_inlet_diff", "pblh_min"]}
    filtering(merged_data, filters)
