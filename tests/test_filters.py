import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.filters import filtering, filtering_functions
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
