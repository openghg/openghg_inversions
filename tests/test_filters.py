import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions import filters as filter_module
from openghg_inversions.filters import filtering, filtering_functions
from openghg_inversions.inversion_data import SiteOptions
from openghg_inversions.rhime import filter_observations
from openghg_inversions.hbmcmc.legacy_data import load_merged_data


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


@pytest.mark.parametrize("longitude", [30.0, -30.0, 330.0])
@pytest.mark.parametrize("keep_missing", [False, True])
@pytest.mark.parametrize(
    "name,east_indices,west_indices",
    [
        ("daytime", list(range(9, 14)), list(range(13, 18))),
        ("daytime9to5", list(range(7, 16)), list(range(11, 20))),
        ("nighttime", [0, 1, 21, 22, 23], [1, 2, 3, 4, 5]),
        ("noon", [10], [14]),
    ],
)
def test_solar_filters_preserve_observations(name, east_indices, west_indices, longitude, keep_missing):
    """Local-hour selection keeps UTC labels and is repeatable on borrowed inputs."""
    dataset = xr.Dataset(
        {
            "mf": ("time", np.arange(24, dtype=float)),
            "release_lon": ("time", np.full(24, longitude)),
        },
        coords={"time": pd.date_range("2020-01-01", periods=24, freq="h")},
        attrs={"site": "TAC"},
    )
    dataset.time.attrs["long_name"] = "UTC observation time"
    original = dataset.copy(deep=True)
    inputs = {"TAC": dataset}
    selected = east_indices if longitude == 30.0 else west_indices
    expected = original.isel(time=selected)
    if keep_missing:
        expected = expected.reindex_like(original)

    result = filtering(inputs, name, keep_missing=keep_missing)
    repeated = filtering(inputs, name, keep_missing=keep_missing)

    xr.testing.assert_identical(dataset, original)
    xr.testing.assert_identical(result["TAC"], expected)
    xr.testing.assert_identical(repeated["TAC"], expected)


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


def test_filters_as_dict_with_missing_site(merged_data, capsys):
    filters = {"TAC": ["pblh_inlet_diff", "pblh_min"]}
    merged_data["MHD"] = "this will be skipped!"
    filtering(merged_data, filters)

    logs = capsys.readouterr().err
    assert "Missing entry for sites ['MHD'] in filters." in logs


@pytest.mark.parametrize("name", ["daily_median", "six_hr_mean"])
def test_retired_resampling_filters_are_not_available(name, capsys):
    assert not hasattr(filter_module, name)
    assert name not in filtering_functions
    filter_module.list_filters()
    assert name not in capsys.readouterr().out


@pytest.mark.parametrize("name", ["daily_median", "six_hr_mean"])
@pytest.mark.parametrize("request_kind", ["scalar", "list", "site_scalar", "site_list"])
@pytest.mark.parametrize("public_stage", [False, True])
def test_invalid_filter_requests_fail_before_any_data_execution(
    name, request_kind, public_stage, monkeypatch,
):
    """Validate all requested names before a preceding custom filter can compute."""
    calls = []

    @delayed
    def unused_observations():
        pytest.fail("Invalid filter request executed lazy observations")

    dataset = xr.Dataset(
        {"mf": ("time", da.from_delayed(unused_observations(), shape=(1,), dtype=float))},
        coords={"time": pd.date_range("2020-01-01", periods=1)},
    )
    monkeypatch.setattr(filter_module, "filtering_functions", filtering_functions.copy())

    @filter_module.register_filter
    def computing_filter(data, keep_missing=False):
        calls.append(1)
        return data.compute()

    request = {
        "scalar": name,
        "list": ["computing_filter", name],
        "site_scalar": {"TAC": "computing_filter", "MHD": name},
        "site_list": {"TAC": ["computing_filter"], "MHD": [name]},
    }[request_kind]
    data = {"TAC": dataset, "MHD": dataset}
    with pytest.raises(KeyError) as error:
        if public_stage:
            filter_observations(data, SiteOptions.from_inputs(sites=list(data), averaging_period=None), filters=request)
        else:
            filtering(data, request)

    assert calls == []
    message = str(error.value)
    assert name in message
    assert "removed resampling filter" in message
    assert "uncertainty-propagating resampling before observation selection" in message


def test_registered_custom_filter_remains_available(monkeypatch, capsys):
    monkeypatch.setattr(filter_module, "filtering_functions", filtering_functions.copy())

    @filter_module.register_filter
    def select_last(dataset, keep_missing=False):
        """Select the final observation."""
        return dataset.isel(time=[-1])

    dataset = xr.Dataset(
        {"mf": ("time", [1.0, 2.0])},
        coords={"time": pd.date_range("2020-01-01", periods=2)},
    )
    assert filter_module.filtering_functions["select_last"] is select_last
    result = filtering({"TAC": dataset}, "select_last")
    xr.testing.assert_identical(result["TAC"], dataset.isel(time=[-1]))
    filter_module.list_filters()
    assert "select_last" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["daily_median", "six_hr_mean"])
@pytest.mark.parametrize("data", [{}, {".metadata": "ignored"}])
def test_retired_global_filters_fail_without_observation_sites(name, data):
    with pytest.raises(KeyError, match="removed resampling filter"):
        filtering(data, name)


def test_unknown_filter_for_absent_site_is_ignored():
    assert filtering({}, {"ABSENT": "unknown_filter"}) == {}
