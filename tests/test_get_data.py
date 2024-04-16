import pickle

import pytest
import xarray as xr
from openghg.types import SearchError

from openghg_inversions.get_data import data_processing_surface_notracer


def test_data_processing_surface_notracer(tac_ch4_data_args, raw_data_path):
    """
    Check that `data_processing_surface_notracer` produces the same output
    as v0.1 (test data frozen on 9 Feb 2024)
    """
    for k, v in tac_ch4_data_args.items():
        print(k, v)
    result = data_processing_surface_notracer(**tac_ch4_data_args)

    # check number of items returned
    assert len(result) == 6

    # check keys of "fp_all"
    assert list(result[0].keys()) == [".species", ".flux", ".bc", "TAC", ".scales", ".units"]

    # get combined scenario for TAC at time 2019-01-01 00:00:00
    expected_tac_combined_scenario = xr.open_dataset(
        raw_data_path / "merged_data_test_tac_combined_scenario.nc"
    )

    xr.testing.assert_allclose(result[0]["TAC"].isel(time=0), expected_tac_combined_scenario)


def test_save_load_merged_data(tac_ch4_data_args, merged_data_dir):
    merged_data_name = "test_save_load_merged_data"

    # make merged data dir
    merged_data_dir.mkdir(exist_ok=True)

    fp_all, *_ = data_processing_surface_notracer(
        save_merged_data=True,
        merged_data_dir=merged_data_dir,
        merged_data_name=merged_data_name,
        **tac_ch4_data_args,
    )

    with open(merged_data_dir / merged_data_name, "rb") as f:
        fp_all_reloaded = pickle.load(f)

    xr.testing.assert_allclose(fp_all["TAC"], fp_all_reloaded["TAC"])


def test_merged_data_vs_frozen_pickle_file(tac_ch4_data_args, merged_data_dir, pickled_data_file_name):
    fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)

    with open(merged_data_dir / pickled_data_file_name, "rb") as f:
        fp_all_reloaded = pickle.load(f)

    xr.testing.assert_allclose(fp_all["TAC"], fp_all_reloaded["TAC"])


def test_missing_data_at_one_site(tac_ch4_data_args):
    """Test that `fp_all` is created if one of two sites has missing data."""
    data_args = tac_ch4_data_args

    # add MHD as site... this won't be found
    data_args["sites"].append("MHD")
    data_args["inlet"].append("24m")
    data_args["instrument"].append("picarro")
    data_args["fp_height"].append("24m")
    data_args["averaging_period"].append("1H")

    fp_all, *_ = data_processing_surface_notracer(**data_args)

    assert "TAC" in fp_all
    assert "MHD" not in fp_all


def test_missing_data_at_all_sites():
    """Check that a SearchError is raised if data is missing from all sites."""
    data_args = {
        "species": "ch4",
        "sites": ["BSD", "MHD"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["185m", "185m"],
        "instrument": ["picarro", "picarro"],
        "domain": "EUROPE",
        "fp_height": ["185m", "185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        "met_model": "ukv",
        "averaging_period": ["1H", "1H"],
    }

    with pytest.raises(SearchError):
        data_processing_surface_notracer(**data_args)
