import pytest
import pandas as pd
import xarray as xr

from openghg_inversions.filters import filtering, filtering_functions
from openghg_inversions.inversion_data.serialise import load_merged_data


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


def test_filters_as_none(merged_data):
    filters = None
    filtering(merged_data, filters)


def test_filters_as_str(merged_data):
    filters = "pblh_inlet_diff"
    filtering(merged_data, filters)


def test_filters_as_list(merged_data):
    filters = ["pblh_inlet_diff", "pblh_min"]
    filtering(merged_data, filters)


def test_filters_as_dict(merged_data):
    filters = {"TAC": ["pblh_inlet_diff", "pblh_min"]}
    filtering(merged_data, filters)


def test_filter_with_kwargs():
    time = pd.date_range("2020-01-01", periods=3, freq="h")
    datasets = {
        "MHD": xr.Dataset(
            {
                "mf": ("time", [1.0, 2.0, 3.0]),
                "PBLH": ("time", [75.0, 150.0, 250.0]),
            },
            coords={"time": time},
        )
    }

    filtered = filtering(datasets, {"MHD": [{"name": "pblh_min", "pblh_threshold": 100.0}]})
    assert filtered["MHD"].sizes["time"] == 2

    filtered = filtering(datasets, {"MHD": [{"pblh_min": {"pblh_threshold": 200.0}}]})
    assert filtered["MHD"].sizes["time"] == 1


def test_filter_datatree_keeps_inner_time_aligned():
    time = pd.date_range("2020-01-01", periods=3, freq="h")
    standard = xr.Dataset(
        {
            "mf": ("time", [1.0, 2.0, 3.0]),
            "PBLH": ("time", [75.0, 150.0, 250.0]),
        },
        coords={"time": time},
    )
    inner = xr.Dataset(
        {"H_inner": (("basis", "time"), [[10.0, 20.0, 30.0]])},
        coords={"basis": [0], "time": time},
    )
    datasets = {"MHD": xr.DataTree.from_dict({"/standard": standard, "/inner": inner})}

    filtered = filtering(datasets, {"MHD": [{"pblh_min": {"pblh_threshold": 100.0}}]})

    assert filtered["MHD"]["standard"].ds.sizes["time"] == 2
    xr.testing.assert_identical(
        filtered["MHD"]["inner"].ds.time,
        filtered["MHD"]["standard"].ds.time,
    )
    assert filtered["MHD"]["inner"].ds["H_inner"].values.tolist() == [[20.0, 30.0]]


def test_filter_datatree_uses_nearest_for_missing_inner_times():
    standard_time = pd.date_range("2020-01-01", periods=3, freq="h")
    inner_time = standard_time[:2]
    standard = xr.Dataset(
        {
            "mf": ("time", [1.0, 2.0, 3.0]),
            "PBLH": ("time", [75.0, 150.0, 250.0]),
        },
        coords={"time": standard_time},
    )
    inner = xr.Dataset(
        {"H_inner": (("basis", "time"), [[10.0, 20.0]])},
        coords={"basis": [0], "time": inner_time},
    )
    datasets = {"MHD": xr.DataTree.from_dict({"/standard": standard, "/inner": inner})}

    filtered = filtering(datasets, {"MHD": [{"pblh_min": {"pblh_threshold": 100.0}}]})

    assert filtered["MHD"]["standard"].ds.sizes["time"] == 2
    xr.testing.assert_identical(
        filtered["MHD"]["inner"].ds.time,
        filtered["MHD"]["standard"].ds.time,
    )
    assert filtered["MHD"]["inner"].ds["H_inner"].values.tolist() == [[20.0, 20.0]]


def test_filter_datatree_zero_fills_entirely_missing_inner_variable():
    standard_time = pd.date_range("2020-01-01", periods=2, freq="h")
    inner_time = standard_time[:1]
    standard = xr.Dataset(
        {
            "mf": ("time", [1.0, 2.0]),
            "PBLH": ("time", [150.0, 250.0]),
        },
        coords={"time": standard_time},
    )
    inner = xr.Dataset(
        {"H_inner": (("basis", "time"), [[float("nan")]])},
        coords={"basis": [0], "time": inner_time},
    )
    datasets = {"MHD": xr.DataTree.from_dict({"/standard": standard, "/inner": inner})}

    filtered = filtering(datasets, {"MHD": [{"pblh_min": {"pblh_threshold": 100.0}}]})

    assert filtered["MHD"]["inner"].ds["H_inner"].values.tolist() == [[0.0, 0.0]]


def test_filters_as_dict_with_missing_site(merged_data, capsys):
    filters = {"TAC": ["pblh_inlet_diff", "pblh_min"]}
    merged_data["MHD"] = "this will be skipped!"
    filtering(merged_data, filters)

    logs = capsys.readouterr().err
    assert "Missing entry for sites ['MHD'] in filters." in logs
