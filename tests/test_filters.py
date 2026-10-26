import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions import filters as filter_module
from openghg_inversions.filters import filtering, filtering_functions
from openghg_inversions.rhime.params import DataConfig
from openghg_inversions.rhime.preparation import acquire_merged_data
from openghg_inversions.inversion_data import SiteOptions
from openghg_inversions.rhime import filter_observations


@pytest.fixture
def merged_data(tac_ch4_data_args):
    return acquire_merged_data(
        DataConfig.from_params(tac_ch4_data_args, multisector=False)
    ).site_data


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


@pytest.mark.parametrize("name", ["daily_median", "six_hr_mean", "unknown_filter"])
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
    assert error.value.args == (name,)


@pytest.mark.parametrize("name", ["daily_median", "six_hr_mean", "unknown_filter", None])
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("data", [{}, {".metadata": "unchanged"}])
def test_global_filters_validate_registry_without_sites(name, as_list, data):
    if name is None and not as_list:
        assert filtering(data, None) is data
        return
    request = [name] if as_list else name
    with pytest.raises(KeyError) as error:
        filtering(data, request)
    assert error.value.args == (name,)


@pytest.mark.parametrize("name", ["daily_median", "six_hr_mean", "unknown_filter", None])
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("with_active_site", [False, True])
def test_absent_site_filter_entries_remain_deferred(name, as_list, with_active_site):
    dataset = xr.Dataset({"mf": ("time", [1.0])}, coords={"time": pd.date_range("2020-01-01", periods=1)})
    data = {"TAC": dataset} if with_active_site else {".metadata": "unchanged"}
    request = {"MHD": [name] if as_list else name}
    result = filtering(data, request)
    assert result.keys() == data.keys()
    for site in data:
        assert result[site] is data[site]


@pytest.mark.parametrize("filter_request", ["noon", ["noon"], {"MHD": "noon"}])
@pytest.mark.parametrize("data", [{}, {".metadata": "unchanged"}])
def test_supported_filters_accept_inputs_without_sites(filter_request, data):
    assert filtering(data, filter_request) == data


@pytest.mark.parametrize("site_mapping", [False, True])
def test_active_none_filter_rejected_before_prior_filter(monkeypatch, site_mapping):
    monkeypatch.setattr(filter_module, "filtering_functions", filtering_functions.copy())

    @filter_module.register_filter
    def unexpected_filter(dataset, keep_missing=False):
        pytest.fail("A preceding filter executed before invalid active None was rejected")

    dataset = xr.Dataset({"mf": ("time", [1.0])}, coords={"time": pd.date_range("2020-01-01", periods=1)})
    filters = {"TAC": ["unexpected_filter"], "MHD": [None]} if site_mapping else ["unexpected_filter", None]
    with pytest.raises(KeyError) as error:
        filtering({"TAC": dataset, "MHD": dataset}, filters)
    assert error.value.args == (None,)


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
