import copy
import logging
from types import SimpleNamespace
from unittest import mock

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from openghg.dataobjects import ObsData
from openghg.retrieve import get_obs_surface
from openghg.types import SearchError

import openghg_inversions.inversion_data.get_data
import openghg_inversions.inversion_data.getters as getters_module
from openghg_inversions.flux_sanitization import FluxNonFiniteMetadata, NonFiniteFluxWarning
from openghg_inversions.inversion_data.get_data import (
    add_obs_error,
    data_processing_surface_notracer,
)
from openghg_inversions.inversion_data.getters import get_flux_data
from openghg_inversions.inversion_data.serialise import (
    fp_all_from_dataset,
    load_merged_data,
    make_combined_scenario,
)


def test_data_processing_surface_notracer(tac_ch4_data_args, merged_data_file_name, raw_data_path):
    """Check that `data_processing_surface_notracer` produces the same output
    as v0.1, with test data frozen on 9 Feb 2024, or the same as v0.2, with test data frozen on
    15 Apr 2024 (using the zarr backend).
    """
    result = data_processing_surface_notracer(**tac_ch4_data_args)

    # check number of items returned
    assert len(result) == 6

    # check keys of "fp_all"
    assert list(result[0].keys()) == [
        ".species",
        ".flux",
        ".split_by_sectors",
        ".bc",
        "TAC",
        ".scales",
        ".units",
    ]

    # variables to check (to avoid surprises from new variables added to data)
    check_vars = ["mf", "fp", "mf_mod", "bc_mod", "fp_x_flux", "bc_n"]

    # get combined scenario for TAC at time 2019-01-01 00:00:00; "frozen" data made
    # with OpenGHG 0.16 ModelScenario
    ds = xr.open_dataset(raw_data_path / (merged_data_file_name + ".nc"))
    expected_tac_combined_scenario = fp_all_from_dataset(ds)
    print(expected_tac_combined_scenario)
    xr.testing.assert_allclose(
        result[0]["TAC"][check_vars].isel(time=0).load(),
        expected_tac_combined_scenario["TAC"][check_vars].isel(time=0),
    )


def test_load_merged_data(merged_data_dir, merged_data_file_name):
    """This should pass by finding the merged data with .zarr suffix."""
    load_merged_data(merged_data_dir, merged_data_name=merged_data_file_name + "no_zip")


def test_load_merged_data_missing_data_error(merged_data_dir, merged_data_file_name):
    """This should pass by finding the merged data with .zarr suffix."""
    with pytest.raises(ValueError):
        load_merged_data(
            merged_data_dir, merged_data_name=merged_data_file_name + "abc123", output_format="netcdf"
        )


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

    fp_all_reloaded = load_merged_data(merged_data_dir=merged_data_dir, merged_data_name=merged_data_name)

    xr.testing.assert_allclose(fp_all["TAC"].load(), fp_all_reloaded["TAC"])


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


def test_missing_data_at_all_sites(openghg_test_store):
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


def test_combined_scenario_roundtrip_preserves_annual_flux_period_start() -> None:
    """Legacy merged data keeps a January annual flux period for June observations."""
    scenario = xr.Dataset(
        {"mf": ("time", [1.0])},
        coords={"time": pd.to_datetime(["2019-06-01"])},
        attrs={"species": "co2"},
    )
    annual_flux = xr.DataArray(
        np.ones((1, 1, 1)),
        dims=("lat", "lon", "time"),
        coords={
            "lat": [52.0],
            "lon": [1.0],
            "time": pd.to_datetime(["2019-01-01"]),
        },
        name="flux",
        attrs={"time_period": "1 year"},
    )
    fp_all = {
        "TAC": scenario,
        ".flux": {"annual": SimpleNamespace(data=annual_flux.to_dataset())},
        ".scales": {"TAC": "ppm"},
        ".species": "CO2",
    }

    recovered = fp_all_from_dataset(make_combined_scenario(fp_all))
    recovered_flux = recovered[".flux"]["annual"].data["flux"]

    np.testing.assert_array_equal(
        recovered_flux["time"].values,
        np.array(["2019-01-01"], dtype="datetime64[ns]"),
    )
    assert recovered_flux.attrs["time_period"] == "1 year"


@pytest.mark.parametrize("missing_period", ["", None, np.nan, pd.NaT])
def test_combined_scenario_uses_dataset_period_when_variable_period_is_missing(
    missing_period: object,
) -> None:
    """Legacy merged data falls back to valid dataset-level period metadata."""
    scenario = xr.Dataset(
        {"mf": ("time", [1.0])},
        coords={"time": pd.to_datetime(["2019-06-01"])},
        attrs={"species": "co2"},
    )
    flux = xr.DataArray(
        np.ones((1, 1, 1)),
        dims=("lat", "lon", "time"),
        coords={
            "lat": [52.0],
            "lon": [1.0],
            "time": pd.to_datetime(["2019-01-01"]),
        },
        name="flux",
        attrs={"time_period": missing_period},
    )
    flux_dataset = flux.to_dataset()
    flux_dataset.attrs["time_period"] = "monthly"
    fp_all = {
        "TAC": scenario,
        ".flux": {"monthly": SimpleNamespace(data=flux_dataset)},
        ".scales": {"TAC": "ppm"},
        ".species": "CO2",
    }

    recovered = fp_all_from_dataset(make_combined_scenario(fp_all))

    assert recovered[".flux"]["monthly"].data["flux"].attrs["time_period"] == "monthly"


def test_combined_scenario_roundtrip_preserves_source_specific_flux_times() -> None:
    """Legacy merged data removes outer-join padding from each flux source."""
    scenario = xr.Dataset(
        {"mf": ("time", [1.0, 2.0])},
        coords={"time": pd.to_datetime(["2019-06-01", "2019-07-01"])},
        attrs={"species": "co2"},
    )
    annual_flux = xr.DataArray(
        np.ones((1, 1, 1)),
        dims=("lat", "lon", "time"),
        coords={
            "lat": [52.0],
            "lon": [1.0],
            "time": pd.to_datetime(["2019-01-01"]),
        },
        name="flux",
        attrs={"time_period": "1 year"},
    )
    monthly_flux = xr.DataArray(
        np.array([[[2.0, 3.0]]]),
        dims=("lat", "lon", "time"),
        coords={
            "lat": [52.0],
            "lon": [1.0],
            "time": pd.to_datetime(["2019-06-01", "2019-07-01"]),
        },
        name="flux",
        attrs={"time_period": "monthly"},
    )
    fp_all = {
        "TAC": scenario,
        ".flux": {
            "annual": SimpleNamespace(data=annual_flux.to_dataset()),
            "monthly": SimpleNamespace(data=monthly_flux.to_dataset()),
        },
        ".scales": {"TAC": "ppm"},
        ".species": "CO2",
    }

    recovered_fluxes = fp_all_from_dataset(make_combined_scenario(fp_all))[".flux"]

    np.testing.assert_array_equal(
        recovered_fluxes["annual"].data["time"].values,
        np.array(["2019-01-01"], dtype="datetime64[ns]"),
    )
    np.testing.assert_array_equal(
        recovered_fluxes["monthly"].data["time"].values,
        np.array(["2019-06-01", "2019-07-01"], dtype="datetime64[ns]"),
    )
    assert recovered_fluxes["annual"].data["flux"].attrs["time_period"] == "1 year"
    assert recovered_fluxes["monthly"].data["flux"].attrs["time_period"] == "monthly"
    assert np.isfinite(recovered_fluxes["annual"].data["flux"]).all()
    assert np.isfinite(recovered_fluxes["monthly"].data["flux"]).all()


def test_combined_scenario_roundtrip_preserves_all_nan_flux_times() -> None:
    """Legitimate all-NaN flux slices are not mistaken for concat padding."""
    scenario = xr.Dataset(
        {"mf": ("time", [1.0, 2.0])},
        coords={"time": pd.to_datetime(["2019-06-01", "2019-07-01"])},
        attrs={"species": "co2"},
    )
    annual_flux = xr.DataArray(
        da.full((1, 1, 1), np.nan, chunks=(1, 1, 1)),
        dims=("lat", "lon", "time"),
        coords={
            "lat": [52.0],
            "lon": [1.0],
            "time": pd.to_datetime(["2019-01-01"]),
        },
        name="flux",
        attrs={"time_period": "1 year"},
    )
    monthly_flux = xr.DataArray(
        np.array([[[2.0, 3.0]]]),
        dims=("lat", "lon", "time"),
        coords={
            "lat": [52.0],
            "lon": [1.0],
            "time": pd.to_datetime(["2019-06-01", "2019-07-01"]),
        },
        name="flux",
        attrs={"time_period": "monthly"},
    )
    fp_all = {
        "TAC": scenario,
        ".flux": {
            "annual": SimpleNamespace(data=annual_flux.to_dataset()),
            "monthly": SimpleNamespace(data=monthly_flux.to_dataset()),
        },
        ".scales": {"TAC": "ppm"},
        ".species": "CO2",
    }

    recovered_fluxes = fp_all_from_dataset(make_combined_scenario(fp_all))[".flux"]
    recovered_annual = recovered_fluxes["annual"].data["flux"]

    assert recovered_annual.chunks is not None
    np.testing.assert_array_equal(
        recovered_annual["time"].values,
        np.array(["2019-01-01"], dtype="datetime64[ns]"),
    )
    assert recovered_annual.isnull().all()


def test_load_legacy_zarr_reads_lazy_source_period_metadata(tmp_path) -> None:
    """A lazy legacy Zarr leaf restores timestamp presence and period metadata."""
    scenario = xr.Dataset(
        {"mf": ("time", [1.0])},
        coords={"time": pd.to_datetime(["2019-06-01"])},
        attrs={"species": "co2"},
    )
    annual_flux = xr.DataArray(
        np.full((1, 1, 1), np.nan),
        dims=("lat", "lon", "time"),
        coords={
            "lat": [52.0],
            "lon": [1.0],
            "time": pd.to_datetime(["2019-01-01"]),
        },
        name="flux",
        attrs={"time_period": "1 year"},
    )
    fp_all = {
        "TAC": scenario,
        ".flux": {"annual": SimpleNamespace(data=annual_flux.to_dataset())},
        ".scales": {"TAC": "ppm"},
        ".species": "CO2",
    }
    make_combined_scenario(fp_all).to_zarr(tmp_path / "legacy.zarr")

    recovered = load_merged_data(
        merged_data_dir=tmp_path,
        merged_data_name="legacy",
        output_format="zarr",
    )

    recovered_flux = recovered[".flux"]["annual"].data["flux"]
    np.testing.assert_array_equal(
        recovered_flux["time"].values,
        np.array(["2019-01-01"], dtype="datetime64[ns]"),
    )
    assert recovered_flux.isnull().all()
    assert recovered_flux.attrs["time_period"] == "1 year"


@pytest.mark.parametrize(
    ("flux_times", "flux_values"),
    [
        (["2019-06-01"], [2.0]),
        (["2019-06-01", "2019-07-01"], [2.0, 3.0]),
    ],
)
def test_fp_all_from_dataset_reads_pre_presence_mask_flux_times(
    flux_times: list[str],
    flux_values: list[float],
) -> None:
    """Older merged datasets retain singleton and multi-time flux behavior."""
    scenario = xr.Dataset(
        {"mf": ("time", np.arange(1, len(flux_times) + 1, dtype=float))},
        coords={"time": pd.to_datetime(flux_times)},
        attrs={"species": "co2"},
    )
    flux = xr.DataArray(
        np.asarray(flux_values).reshape(1, 1, -1),
        dims=("lat", "lon", "time"),
        coords={"lat": [52.0], "lon": [1.0], "time": pd.to_datetime(flux_times)},
        name="flux",
        attrs={"time_period": "monthly"},
    )
    fp_all = {
        "TAC": scenario,
        ".flux": {"monthly": SimpleNamespace(data=flux.to_dataset())},
        ".scales": {"TAC": "ppm"},
        ".species": "CO2",
    }
    combined = make_combined_scenario(fp_all)
    legacy = combined.drop_vars(["flux", "flux_time_period", "flux_time_present"]).drop_dims("flux_time")
    if len(flux_times) == 1:
        legacy["flux"] = flux.isel(time=0, drop=True).expand_dims(source=["monthly"])
    else:
        legacy["flux"] = flux.expand_dims(source=["monthly"])

    recovered_flux = fp_all_from_dataset(legacy)[".flux"]["monthly"].data["flux"]

    np.testing.assert_array_equal(recovered_flux["time"].values, pd.to_datetime(flux_times).values)
    np.testing.assert_array_equal(recovered_flux.values.ravel(), flux_values)


def test_add_averaging_error(tac_ch4_data_args):
    """Check that "add averaging error" adds variability to repeatability."""
    # we need to use "mock" to add mf_repeatability to our data
    # since our test data is from picarro and only has variability
    real_obs = get_obs_surface(site="tac", species="ch4", inlet="185m")
    real_obs_data = real_obs.data
    real_obs_metadata = real_obs.metadata
    real_obs_data["mf_repeatability"] = xr.ones_like(real_obs_data["mf_variability"])
    patched_obs = ObsData(data=real_obs_data, metadata=real_obs_metadata)

    with mock.patch.object(openghg_inversions.inversion_data.getters, "get_obs_surface") as mock_obs:
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
                assert "number_of_observations" not in ds[var].attrs["long_name"]

        assert ds1.mf_error.attrs["long_name"] == ds1.mf.attrs["long_name"] + "_error"
        assert ds1.mf_repeatability.attrs["long_name"] == ds1.mf.attrs["long_name"] + "_repeatability"
        assert ds1.mf_variability.attrs["long_name"] == ds1.mf.attrs["long_name"] + "_variability"

        # averagingerror=True is default, so for ds1, "mf_error" should have repeatability
        # and variability added
        xr.testing.assert_allclose(ds1.mf_error, np.sqrt(ds1.mf_repeatability**2 + ds1.mf_variability**2))

        # ds2 should use repeatability for "mf_error", since we have set averagingerror=False
        xr.testing.assert_allclose(ds2.mf_error, ds2.mf_repeatability)


@pytest.mark.parametrize(
    "rept_vals, perc_missing",
    [([0] * 5 + [1] * 5, "50.00"), ([np.nan] * 5 + [1] * 5, "50.00"), ([0] + [1] * 999, "0.10")],
)
def test_add_obs_error_exceptions_warnings(rept_vals, perc_missing, caplog):
    n = len(rept_vals)
    ds = xr.Dataset()
    ds["mf"] = xr.DataArray([1] * n, dims="time")
    fp_all = {"TAC": ds}

    with pytest.raises(ValueError):
        add_obs_error(sites=["TAC"], fp_all=fp_all, add_averaging_error=True)

    # check for WARNING when `mf_error` contains zeros
    # plus INFO suggesting fix
    caplog.set_level(logging.INFO)

    ds["mf_repeatability"] = xr.DataArray(rept_vals, dims="time")
    ds["mf_variability"] = xr.DataArray([0] * n, dims="time")

    fp_all["TAC"] = ds.chunk(time=n // 3)

    add_obs_error(sites=["TAC"], fp_all=fp_all, add_averaging_error=False)

    output = caplog.text
    assert f"{perc_missing} percent" in output
    assert "Try setting `averaging_period = None`" in output


def test_add_obs_error_without_repeatability(caplog):
    """Check for logger info if repeatability isn't present and add_averaging_error is True."""
    ds = xr.Dataset()
    ds["mf"] = xr.DataArray([1] * 10, dims="time")
    ds["mf_variability"] = xr.DataArray([0] * 10, dims="time")
    fp_all = {"TAC": ds}

    # check for WARNING when `mf_error` contains zeros
    # plus INFO suggesting fix
    caplog.set_level(logging.INFO)

    add_obs_error(sites=["TAC"], fp_all=fp_all, add_averaging_error=True)

    output = caplog.text
    assert "`mf_repeatability` not present; using `mf_variability` for `mf_error` at site TAC" in output


def test_looking_older_flux_files(tac_ch4_data_args, capsys):
    """Check if an older flux file is found if no data is found for the specified start and end dates."""
    data_args = tac_ch4_data_args.copy()
    data_args["start_date"] = "2100-01-01"
    data_args["end_date"] = "2101-01-01"

    # we should get an error when trying to get obs data, but not when trying to get flux data
    with pytest.raises(SearchError):
        data_processing_surface_notracer(**data_args)

    stdout = capsys.readouterr().out

    # we find older flux data
    assert "Using flux data from 2019-01-01" in stdout


@pytest.mark.parametrize("end_date", ["2019-02-01", "2020-01-01", "2019-01-02"])
def test_flux_time_period_preserves_source_period(end_date, tac_ch4_data_args):
    """Retrieved annual period is independent of the inversion duration."""
    kwargs = {
        "sources": tac_ch4_data_args["emissions_name"],
        "species": tac_ch4_data_args["species"],
        "domain": tac_ch4_data_args["domain"],
        "start_date": "2019-01-01",
        "end_date": end_date,
    }
    flux_data = get_flux_data(**kwargs)

    source = tac_ch4_data_args["emissions_name"][0]
    assert flux_data[source].data.flux.attrs["time_period"] == "1 year"


@pytest.mark.parametrize(
    ("start_date", "time_period", "expected"),
    [
        ("2019-06-15", "1 year", "2019-01-01"),
        ("2019-06-15", "yearly", "2019-01-01"),
        ("2019-06-15", "annual", "2019-01-01"),
        ("2019-06-15", "1 month", "2019-06-01"),
        ("2019-06-15", "monthly", "2019-06-01"),
    ],
)
def test_adjust_flux_start_date_recognizes_period_variants(
    monkeypatch: pytest.MonkeyPatch,
    start_date: str,
    time_period: str,
    expected: str,
) -> None:
    """Flux retrieval starts at the governing annual or monthly interval."""
    search_result = SimpleNamespace(results=pd.DataFrame({"time_period": [time_period]}))
    monkeypatch.setattr(getters_module, "search_flux", lambda **kwargs: search_result)

    actual = getters_module.adjust_flux_start_date(
        start_date,
        species="co2",
        source="test-source",
        domain="EUROPE",
    )

    assert actual == pd.Timestamp(expected)


def test_adjust_flux_start_date_uses_longest_matching_source_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed search results use the longest period to include the governing slice."""
    search_result = SimpleNamespace(
        results=pd.DataFrame({"time_period": ["monthly", "1 year"]}),
    )
    monkeypatch.setattr(getters_module, "search_flux", lambda **kwargs: search_result)

    actual = getters_module.adjust_flux_start_date(
        "2019-06-15",
        species="co2",
        source="test-source",
        domain="EUROPE",
    )

    assert actual == pd.Timestamp("2019-01-01")


def test_get_flux_data_preserves_annual_period_for_midyear_inversion(
    tac_ch4_data_args,
) -> None:
    """A June monthly inversion retains a January annual prior and its period."""
    source = tac_ch4_data_args["emissions_name"][0]
    result = get_flux_data(
        sources=[source],
        species=tac_ch4_data_args["species"],
        domain=tac_ch4_data_args["domain"],
        start_date="2019-06-01",
        end_date="2019-07-01",
        store=tac_ch4_data_args["emissions_store"],
    )

    retained = result[source].data
    assert retained.attrs["time_period"] == "1 year"
    assert retained["flux"].attrs["time_period"] == "1 year"
    np.testing.assert_array_equal(retained["time"].values, np.array(["2019-01-01"], dtype="datetime64[ns]"))


def test_get_flux_data_preserves_variable_period_over_dataset_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Variable period metadata takes precedence over conflicting dataset metadata."""
    flux = xr.DataArray(
        np.ones((1, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.to_datetime(["2019-06-01"]),
            "lat": [52.0],
            "lon": [1.0],
        },
        name="flux",
        attrs={"time_period": "monthly"},
    )
    flux_data = SimpleNamespace(
        data=xr.Dataset({"flux": flux}, attrs={"time_period": "1 year"}),
    )
    monkeypatch.setattr(getters_module, "adjust_flux_start_date", lambda *args: pd.Timestamp("2019-06-01"))
    monkeypatch.setattr(getters_module, "get_flux", lambda **kwargs: flux_data)

    result = get_flux_data(
        sources=["mixed-metadata-source"],
        species="co2",
        domain="EUROPE",
        start_date="2019-06-01",
        end_date="2019-07-01",
    )

    assert result["mixed-metadata-source"].data["flux"].attrs["time_period"] == "monthly"


@pytest.mark.parametrize("missing_period", ["", np.nan, pd.NaT])
def test_get_flux_data_uses_dataset_period_when_variable_period_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    missing_period: object,
) -> None:
    """Missing variable metadata does not mask a valid dataset period."""
    flux = xr.DataArray(
        np.ones((1, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.to_datetime(["2019-01-01"]),
            "lat": [52.0],
            "lon": [1.0],
        },
        name="flux",
        attrs={"time_period": missing_period},
    )
    flux_data = SimpleNamespace(
        data=xr.Dataset({"flux": flux}, attrs={"time_period": "monthly"}),
    )
    monkeypatch.setattr(getters_module, "adjust_flux_start_date", lambda *args: pd.Timestamp("2019-01-01"))
    monkeypatch.setattr(getters_module, "get_flux", lambda **kwargs: flux_data)

    result = get_flux_data(
        sources=["missing-variable-period-source"],
        species="co2",
        domain="EUROPE",
        start_date="2019-01-01",
        end_date="2019-02-01",
    )

    assert result["missing-variable-period-source"].data["flux"].attrs["time_period"] == "monthly"


def test_get_flux_data_count_mode_audits_original_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrieval count mode records exact NaN and infinity replacements once."""
    flux = xr.DataArray(
        np.array([[[1.0, np.nan], [np.inf, 2.0]]]),
        dims=("time", "lat", "lon"),
        coords={"time": [np.datetime64("2019-01-01")]},
        name="flux",
    )
    flux_data = SimpleNamespace(data=xr.Dataset({"flux": flux}, attrs={"time_period": "1 year"}))
    monkeypatch.setattr(getters_module, "adjust_flux_start_date", lambda *args: np.datetime64("2019-01-01"))
    monkeypatch.setattr(getters_module, "get_flux", lambda **kwargs: flux_data)

    with pytest.warns(NonFiniteFluxWarning, match="contains 2 non-finite values"):
        result = get_flux_data(
            sources=["test-source"],
            species="co2",
            domain="EUROPE",
            start_date="2019-01-01",
            end_date="2020-01-01",
            flux_non_finite_check="count",
        )

    sanitized = result["test-source"].data["flux"]
    metadata = FluxNonFiniteMetadata.from_attrs(sanitized.attrs)
    assert metadata is not None
    assert metadata.count == 2
    assert metadata.total == 4
    assert np.isfinite(sanitized.values).all()
