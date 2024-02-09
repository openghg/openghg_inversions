import pickle

import xarray as xr

from openghg_inversions.get_data import data_processing_surface_notracer


def test_data_processing_surface_notracer(data_args, raw_data_path):
    """
    Check that `data_processing_surface_notracer` produces the same output
    as v0.1 (test data frozen on 9 Feb 2024)
    """
    for k, v in data_args.items():
        print(k, v)
    result = data_processing_surface_notracer(**data_args)

    # check number of items returned
    assert len(result) == 6

    # check keys of "fp_all"
    assert list(result[0].keys()) == [".species", ".flux", ".bc", "TAC", ".scales", ".units"]

    # get combined scenario for TAC at time 2019-01-01 00:00:00
    expected_tac_combined_scenario = xr.open_dataset(
        raw_data_path / "merged_data_test_tac_combined_scenario.nc"
    )

    xr.testing.assert_allclose(result[0]["TAC"].isel(time=0), expected_tac_combined_scenario)


def test_save_load_merged_data(data_args, merged_data_dir):

    merged_data_name = "test_save_load_merged_data"

    # make merged data dir
    merged_data_dir.mkdir(exist_ok=True)

    fp_all, *_ = data_processing_surface_notracer(
        save_merged_data=True,
        merged_data_dir=merged_data_dir,
        merged_data_name=merged_data_name,
        **data_args,
    )

    with open(merged_data_dir / merged_data_name, "rb") as f:
        fp_all_reloaded = pickle.load(f)

    xr.testing.assert_allclose(fp_all["TAC"], fp_all_reloaded["TAC"])


def test_merged_data_vs_frozen_pickle_file(data_args, merged_data_dir, pickled_data_file_name):
    fp_all, *_ = data_processing_surface_notracer(**data_args)

    with open(merged_data_dir / pickled_data_file_name, "rb") as f:
        fp_all_reloaded = pickle.load(f)

    xr.testing.assert_allclose(fp_all["TAC"], fp_all_reloaded["TAC"])
