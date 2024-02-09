from pathlib import Path
import tempfile

import pytest
import xarray as xr

from openghg_inversions.get_data import data_processing_surface_notracer


def test_data_processing_surface_notracer(data_args, raw_data_path):
    """
    Check that `data_processing_surface_notracer` produces the same output
    as v0.1 (test data frozen on 9 Feb 2024)
    """
    result = data_processing_surface_notracer(**data_args)

    # check number of items returned
    assert len(result) == 6

    # check keys of "fp_all"
    assert list(result[0].keys()) == ['.species', '.flux', '.bc', 'TAC', '.scales', '.units']


    # get combined scenario for TAC at time 2019-01-01 00:00:00
    expected_tac_combined_scenario = xr.open_dataset(raw_data_path / "merged_data_test_tac_combined_scenario.nc")

    xr.testing.assert_allclose(result[0]['TAC'].isel(time=0), expected_tac_combined_scenario)


def test_save_load_merged_data(data_args):
    pass
