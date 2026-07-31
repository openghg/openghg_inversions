import copy
import logging
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from openghg.dataobjects import ObsData
from openghg.retrieve import get_obs_surface
from openghg.types import SearchError

import openghg_inversions.inversion_data.get_data as get_data_module
import openghg_inversions.inversion_data.getters as getters_module
import openghg_inversions.inversion_data.scenario as scenario_module
from openghg_inversions.flux_sanitization import FluxNonFiniteMetadata, NonFiniteFluxWarning
from openghg_inversions.inversion_data._site_options import expand_site_option
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.inversion_data.get_data import (
    add_obs_error,
    convert_to_list,
    data_processing_surface_notracer,
)
from openghg_inversions.inversion_data.getters import get_flux_data
from openghg_inversions.inversion_data.serialise import (
    fp_all_from_dataset,
    load_merged_data,
    make_combined_scenario,
)


@pytest.mark.parametrize(
    ("raw_units", "expected"),
    [("1", 1.0), ("mol/mol", 1.0), ("ppb", 1e-9), ("1e-09 mol/mol", 1e-9)],
)
def test_mole_fraction_unit_scale_uses_openghg_registry(raw_units: str, expected: float) -> None:
    """OpenGHG unit expressions map directly to their mol/mol scale."""
    assert mole_fraction_unit_scale(raw_units, context="test observations") == pytest.approx(expected)


def test_data_processing_surface_notracer(tac_ch4_data_args, merged_data_file_name, raw_data_path):
    """Check that `data_processing_surface_notracer` produces the same output
    as v0.1, with test data frozen on 9 Feb 2024, or the same as v0.2, with test data frozen on
    15 Apr 2024 (using the zarr backend).
    """
    result = data_processing_surface_notracer(**tac_ch4_data_args)

    # check number of items returned
    assert len(result) == 6
    assert result[1:] == (["TAC"], ["185m"], ["185m"], ["picarro"], ["1h"])

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


def test_mixed_platforms_keep_surface_calibration_scale_per_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed surface and satellite data use each site's platform for scales."""
    observed_platforms: list[tuple[str, str | None, int | None]] = []
    scenario_platforms: list[str | None] = []

    def fake_get_obs_data(**kwargs: object) -> object:
        """Record each observation request's aligned platform and level."""
        observed_platforms.append(
            (
                str(kwargs["site"]),
                kwargs["platform"] if isinstance(kwargs["platform"], str) else None,
                kwargs["max_level"] if isinstance(kwargs["max_level"], int) else None,
            )
        )
        return object()

    def fake_merged_scenario_data(*args: object, **kwargs: object) -> xr.Dataset:
        """Return a minimal scenario with a platform-specific scale."""
        platform = kwargs["platform"] if isinstance(kwargs["platform"], str) else None
        scenario_platforms.append(platform)
        scale = "surface-scale" if platform == "surface" else "satellite-scale"
        mf = xr.DataArray([1.0], dims="time", attrs={"units": "1e-9"})
        return xr.Dataset({"mf": mf}, attrs={"scale": scale})

    monkeypatch.setattr(get_data_module, "get_flux_data", lambda **kwargs: {})
    monkeypatch.setattr(get_data_module, "get_obs_data", fake_get_obs_data)
    monkeypatch.setattr(get_data_module, "get_footprint_data", lambda **kwargs: object())
    monkeypatch.setattr(get_data_module, "merged_scenario_data", fake_merged_scenario_data)
    monkeypatch.setattr(get_data_module, "add_obs_error", lambda *args, **kwargs: None)

    result = data_processing_surface_notracer(
        species="ch4",
        sites=["TAC", "GOSAT-BRAZIL"],
        domain="EUROPE",
        averaging_period=["1h", "1h"],
        start_date="2019-01-01",
        end_date="2019-01-02",
        platform=["surface", "satellite"],
        max_level=[None, 17],
        emissions_name=["inventory"],
        use_bc=False,
    )

    assert observed_platforms == [
        ("TAC", "surface", None),
        ("GOSAT-BRAZIL", "satellite", 17),
    ]
    assert scenario_platforms == ["surface", "satellite"]
    assert result[0][".scales"] == {"TAC": "surface-scale"}
    assert result[0][".units"] == pytest.approx(1e-9)
    assert len(result) == 6


def test_get_obs_data_routes_site_column_platform_to_column_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented site-column platform uses site-based column retrieval."""
    captured: dict[str, object] = {}
    obs_data = SimpleNamespace(data=xr.Dataset(coords={"time": [np.datetime64("2019-01-01")]}))

    def fake_get_obs_column(**kwargs: object) -> SimpleNamespace:
        """Capture site-column lookup arguments."""
        captured.update(kwargs)
        return obs_data

    monkeypatch.setattr(getters_module, "get_obs_column", fake_get_obs_column)
    monkeypatch.setattr(
        getters_module,
        "get_obs_surface",
        lambda **kwargs: pytest.fail("site-column must not use surface retrieval"),
    )

    result = getters_module.get_obs_data(
        site="TAC",
        species="ch4",
        inlet=None,
        start_date="2019-01-01",
        end_date="2019-01-02",
        platform="site-column",
        max_level=17,
        stores="test",
    )

    assert result is obs_data
    assert captured["site"] == "TAC"
    assert captured["max_level"] == 17


@pytest.mark.parametrize("platform", [None, "surface"])
def test_column_inlet_labels_scenario_as_site_column(
    monkeypatch: pytest.MonkeyPatch,
    platform: str | None,
) -> None:
    """A column inlet overrides absent or contradictory surface platforms."""
    captured_platforms: list[str | None] = []

    def fake_scenario(*args: object, **kwargs: object) -> xr.Dataset:
        """Capture the normalized scenario platform."""
        platform = kwargs.get("platform")
        captured_platforms.append(platform if isinstance(platform, str) else None)
        return xr.Dataset(
            {
                "mf": ("time", [1.0], {"units": "ppb"}),
                "mf_repeatability": ("time", [0.1]),
            },
            attrs={"scale": "test-scale"},
        )

    monkeypatch.setattr(get_data_module, "get_flux_data", lambda **kwargs: {})
    monkeypatch.setattr(get_data_module, "get_obs_data", lambda **kwargs: object())
    monkeypatch.setattr(get_data_module, "get_footprint_data", lambda **kwargs: object())
    monkeypatch.setattr(get_data_module, "merged_scenario_data", fake_scenario)

    data_processing_surface_notracer(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period="1h",
        start_date="2019-01-01",
        end_date="2019-01-02",
        inlet="column",
        platform=platform,
        max_level=17,
        emissions_name=["inventory"],
        use_bc=False,
    )

    assert captured_platforms == ["site-column"]


def test_convert_to_list_accepts_numpy_integer_scalar() -> None:
    """NumPy scalars and arrays retain the legacy list-returning contract."""
    result = convert_to_list(np.int64(17), length=2, name="max_level")

    assert result == [17, 17]
    assert all(type(value) is int for value in result)
    assert convert_to_list(np.array(["a", "b"]), length=2, name="inlet") == ["a", "b"]


def test_data_processing_rejects_boolean_max_level_before_retrieval() -> None:
    """Boolean maximum levels are not accepted as integers."""
    with pytest.raises(ValueError, match="must be integers or None"):
        data_processing_surface_notracer(
            species="ch4",
            sites=["TAC"],
            domain="EUROPE",
            averaging_period="1h",
            start_date="2019-01-01",
            end_date="2019-01-02",
            max_level=[True],
            emissions_name=["inventory"],
            use_bc=False,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("column", ("column", "column")),
        (slice(1, 4), (slice(1, 4), slice(1, 4))),
        (None, (None, None)),
        ((1, 2), (1, 2)),
        (np.array([3, 4]), (3, 4)),
    ],
)
def test_expand_site_option_returns_immutable_site_aligned_values(
    value: Any,
    expected: tuple[object, object],
) -> None:
    """The shared broadcaster accepts scalar and array-like site options."""
    assert expand_site_option(value, nsites=2, name="option") == expected


def test_expand_site_option_rejects_misaligned_or_boolean_values() -> None:
    """The shared broadcaster rejects ambiguous or drifted option values."""
    with pytest.raises(ValueError, match="does not have specified length"):
        expand_site_option(["only-one"], nsites=2, name="option")
    with pytest.raises(ValueError, match="site-aligned iterable"):
        expand_site_option(True, nsites=2, name="option")
    with pytest.raises(ValueError, match="site-aligned iterable"):
        expand_site_option({"unordered", "values"}, nsites=2, name="option")
    with pytest.raises(ValueError, match="site-aligned iterable"):
        expand_site_option({"TAC": "value"}, nsites=1, name="option")


def test_data_processing_reuses_first_successful_observation_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later scenarios request the first successfully merged site's units."""
    requested_output_units: list[object] = []

    def fake_scenario(*args: object, **kwargs: object) -> xr.Dataset:
        """Return scenarios already converted by ModelScenario."""
        platform = kwargs["platform"]
        requested_output_units.append(kwargs.get("output_units"))
        return xr.Dataset(
            {
                "mf": ("time", [1000.0], {"units": "ppb"}),
                "mf_mod": ("time", [900.0], {"units": "ppb"}),
                "mf_repeatability": ("time", [2.0], {"units": "ppb"}),
                "mf_variability": ("time", [3.0], {"units": "ppb"}),
                "mf_number_of_observations": ("time", [10 if platform == "surface" else 20]),
            },
            attrs={"scale": "test-scale"},
        )

    monkeypatch.setattr(get_data_module, "get_flux_data", lambda **kwargs: {})
    monkeypatch.setattr(
        get_data_module,
        "get_obs_data",
        lambda **kwargs: None if kwargs["site"] == "BSD" else object(),
    )
    monkeypatch.setattr(get_data_module, "get_footprint_data", lambda **kwargs: object())
    monkeypatch.setattr(get_data_module, "merged_scenario_data", fake_scenario)

    fp_all, retained_sites, *_ = data_processing_surface_notracer(
        species="ch4",
        sites=["BSD", "TAC", "GOSAT-BRAZIL"],
        domain="EUROPE",
        averaging_period="1h",
        start_date="2019-01-01",
        end_date="2019-01-02",
        platform=["surface", "surface", "satellite"],
        emissions_name=["inventory"],
        use_bc=False,
    )

    assert requested_output_units == [None, "ppb"]
    assert retained_sites == ["TAC", "GOSAT-BRAZIL"]
    assert fp_all[".units"] == pytest.approx(1e-9)
    np.testing.assert_allclose(fp_all["TAC"]["mf"], [1000.0])
    np.testing.assert_allclose(fp_all["GOSAT-BRAZIL"]["mf"], [1000.0])
    np.testing.assert_allclose(fp_all["GOSAT-BRAZIL"]["mf_mod"], [900.0])
    np.testing.assert_allclose(fp_all["GOSAT-BRAZIL"]["mf_repeatability"], [2.0])
    np.testing.assert_allclose(fp_all["GOSAT-BRAZIL"]["mf_variability"], [3.0])
    np.testing.assert_allclose(fp_all["GOSAT-BRAZIL"]["mf_error"], [np.sqrt(13.0)])
    np.testing.assert_array_equal(fp_all["GOSAT-BRAZIL"]["mf_number_of_observations"], [20])
    assert fp_all["TAC"]["mf"].attrs["units"] == fp_all["GOSAT-BRAZIL"]["mf"].attrs["units"]


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_data_processing_reports_later_site_unit_conversion_failure(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    """A later ModelScenario unit failure identifies its site and retains the Pint cause."""
    scenario_calls = 0

    def fake_scenario(*args: object, **kwargs: object) -> xr.Dataset:
        """Return the target-unit scenario, then mimic a Pint conversion failure."""
        nonlocal scenario_calls
        scenario_calls += 1
        if scenario_calls == 1:
            assert kwargs["output_units"] is None
            return xr.Dataset(
                {"mf": ("time", [1000.0], {"units": "ppb"})},
                attrs={"scale": "test-scale"},
            )
        assert kwargs["output_units"] == "ppb"
        raise error_type("Pint could not convert ppm to ppb")

    monkeypatch.setattr(get_data_module, "get_flux_data", lambda **kwargs: {})
    monkeypatch.setattr(get_data_module, "get_obs_data", lambda **kwargs: object())
    monkeypatch.setattr(get_data_module, "get_footprint_data", lambda **kwargs: object())
    monkeypatch.setattr(get_data_module, "merged_scenario_data", fake_scenario)

    with pytest.raises(ValueError, match="site 'MHD'.*target observation units 'ppb'") as exc_info:
        data_processing_surface_notracer(
            species="ch4",
            sites=["TAC", "MHD"],
            domain="EUROPE",
            averaging_period="1h",
            start_date="2019-01-01",
            end_date="2019-01-02",
            platform="surface",
            emissions_name=["inventory"],
            use_bc=False,
        )

    assert isinstance(exc_info.value.__cause__, error_type)
    assert str(exc_info.value.__cause__) == "Pint could not convert ppm to ppb"


def test_merged_scenario_forwards_requested_output_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scenario boundary delegates common-unit conversion to ModelScenario."""
    merge_kwargs: dict[str, object] = {}
    expected = xr.Dataset({"mf": ("time", [1.0], {"units": "ppb"})})

    class FakeModelScenario:
        """Capture the arguments passed to the OpenGHG scenario object."""

        def __init__(self, **kwargs: object) -> None:
            """Record ModelScenario constructor arguments for inspection."""
            self.init_kwargs = kwargs

        def footprints_data_merge(self, **kwargs: object) -> xr.Dataset:
            """Capture merge options and return an already converted scenario."""
            merge_kwargs.update(kwargs)
            return expected

    monkeypatch.setattr(scenario_module, "ModelScenario", FakeModelScenario)

    result = scenario_module.merged_scenario_data(
        obs_data=object(),  # type: ignore[arg-type]
        footprint_data=object(),  # type: ignore[arg-type]
        flux_dict={},
        output_units="ppb",
    )

    assert result is expected
    assert merge_kwargs["output_units"] == "ppb"


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


def test_add_averaging_error(tac_ch4_data_args):
    """Check that "add averaging error" adds variability to repeatability."""
    # we need to use "mock" to add mf_repeatability to our data
    # since our test data is from picarro and only has variability
    real_obs = get_obs_surface(site="tac", species="ch4", inlet="185m")
    real_obs_data = real_obs.data
    real_obs_metadata = real_obs.metadata
    real_obs_data["mf_repeatability"] = xr.ones_like(real_obs_data["mf_variability"])
    patched_obs = ObsData(data=real_obs_data, metadata=real_obs_metadata)

    with mock.patch.object(getters_module, "get_obs_surface") as mock_obs:
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


@pytest.mark.parametrize(
    "end_date, time_period", [("2019-02-01", "monthly"), ("2020-01-01", "1 year"), ("2019-01-02", "1 year")]
)
def test_flux_time_period_inference(end_date, time_period, tac_ch4_data_args):
    kwargs = {
        "sources": tac_ch4_data_args["emissions_name"],
        "species": tac_ch4_data_args["species"],
        "domain": tac_ch4_data_args["domain"],
        "start_date": "2019-01-01",
        "end_date": end_date,
    }
    flux_data = get_flux_data(**kwargs)

    source = tac_ch4_data_args["emissions_name"][0]
    assert flux_data[source].data.flux.attrs["time_period"] == time_period


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
