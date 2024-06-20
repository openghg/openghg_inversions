import copy
from unittest import mock

import numpy as np
import pytest
import xarray as xr
import openghg.retrieve
from openghg.dataobjects import ObsData
from openghg.retrieve import get_obs_surface
from openghg.types import SearchError

import openghg_inversions.get_data
from openghg_inversions.get_data import data_processing_surface_notracer, fp_all_from_dataset, make_combined_scenario, load_merged_data


def test_data_processing_surface_notracer(tac_ch4_data_args, raw_data_path, using_zarr_store):
    """
    Check that `data_processing_surface_notracer` produces the same output
    as v0.1, with test data frozen on 9 Feb 2024, or the same as v0.2, with test data frozen on
    15 Apr 2024 (using the zarr backend).
    """
    result = data_processing_surface_notracer(**tac_ch4_data_args)

    # check number of items returned
    assert len(result) == 6

    # check keys of "fp_all"
    assert list(result[0].keys()) == [".species", ".flux", ".bc", "TAC", ".scales", ".units"]

    if using_zarr_store:
        # get combined scenario for TAC at time 2019-01-01 00:00:00
        expected_tac_combined_scenario = xr.open_dataset(
            raw_data_path / "merged_data_test_tac_combined_scenario_v8.nc"
        )
        xr.testing.assert_allclose(result[0]["TAC"].isel(time=0).load(), expected_tac_combined_scenario.isel(time=0).isel(site=0, drop=True))
    else:
        # get combined scenario for TAC at time 2019-01-01 00:00:00
        expected_tac_combined_scenario = xr.open_dataset(
            raw_data_path / "merged_data_test_tac_combined_scenario.nc"
        )
        xr.testing.assert_allclose(result[0]["TAC"].isel(time=0), expected_tac_combined_scenario.isel(time=0).isel(site=0, drop=True), rtol=1e-2)


def test_save_load_merged_data(tac_ch4_data_args, merged_data_dir, using_zarr_store):
    merged_data_name = "test_save_load_merged_data"

    # make merged data dir
    merged_data_dir.mkdir(exist_ok=True)

    fp_all, *_ = data_processing_surface_notracer(
        save_merged_data=True,
        merged_data_dir=merged_data_dir,
        merged_data_name=merged_data_name,
        **tac_ch4_data_args,
    )

    fp_all_reloaded = load_merged_data(merged_data_dir=merged_data_dir, merged_data_name=merged_data_name)

    if using_zarr_store:
        xr.testing.assert_allclose(fp_all["TAC"].load(), fp_all_reloaded["TAC"])
    else:
        xr.testing.assert_allclose(fp_all["TAC"], fp_all_reloaded["TAC"])


def test_missing_data_at_one_site(tac_ch4_data_args):
    """Test that `fp_all` is created if one of two sites has missing data."""
    data_args = copy.deepcopy(tac_ch4_data_args)

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


def test_fp_all_to_dataset_and_back(tac_ch4_data_args):
    fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
    ds = make_combined_scenario(fp_all)
    fp_all_recovered = fp_all_from_dataset(ds)

    # check scenarios are the same
    xr.testing.assert_equal(fp_all["TAC"], fp_all_recovered["TAC"])

    print(fp_all[".bc"])
    print(fp_all_recovered[".bc"])

    for k, v in fp_all.items():
        if not k.startswith("."):
            continue

        assert k in fp_all_recovered

        v_recovered = fp_all_recovered[k]

        if k == ".flux":
            assert list(v.keys()) == list(v_recovered.keys())

            for flux_data1, flux_data2 in zip(v.values(), v_recovered.values()):
                xr.testing.assert_allclose(flux_data1.data, flux_data2.data, rtol=1e-3)

        elif k == ".bc":
            xr.testing.assert_allclose(v.data, v_recovered.data, rtol=1e-3)
        else:
            assert v == v_recovered


def test_add_averaging_error(tac_ch4_data_args):
    """Check that "add averaging error" adds variability to repeatability."""
    # we need to use "mock" to add mf_repeatability to our data
    # since our test data is from picarro and only has variability
    real_obs = get_obs_surface(site="tac", species="ch4", inlet="185m")
    real_obs_data = real_obs.data
    real_obs_metadata = real_obs.metadata
    real_obs_data["mf_repeatability"] = xr.ones_like(real_obs_data["mf_variability"])
    patched_obs = ObsData(data=real_obs_data, metadata=real_obs_metadata)

    with mock.patch.object(openghg_inversions.get_data, "get_obs_surface") as mock_obs:
        mock_obs.return_value = patched_obs

        # set up two scenarios, one with averaging, one without
        fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
        ds1 = fp_all["TAC"]

        tac_ch4_data_args["averagingerror"] = False
        fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
        ds2 = fp_all["TAC"]

        # check that "mf_error", "mf_repeatability", and "mf_variability" are present
        for var in ["mf_error", "mf_repeatability", "mf_variability"]:
            for ds in [ds1, ds2]:
                assert var in ds

        # averagingerror=True is default, so for ds1, "mf_error" should have repeatability
        # and variability added
        xr.testing.assert_allclose(ds1.mf_error, np.sqrt(ds1.mf_repeatability**2 + ds1.mf_variability**2))

        # ds2 should use repeatability for "mf_error", since we have set averagingerror=False
        xr.testing.assert_allclose(ds2.mf_error, ds2.mf_repeatability)
