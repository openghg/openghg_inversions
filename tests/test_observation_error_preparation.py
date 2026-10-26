"""Reported errors are derived on retained observation populations."""

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.inversion_data import MergedData, SiteOptions
from openghg_inversions.rhime import filter_observations, prepare_observation_errors


def _merged(dataset):
    return MergedData(
        site_data={"TAC": dataset}, flux_data={},
        site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"),
    )


@pytest.mark.parametrize("error_kind", ["derived", "custom", "complete"])
@pytest.mark.parametrize("lazy", [False, True])
def test_error_preparation_preserves_selector_order_over_mapping_order(error_kind, lazy):
    from openghg_inversions.inversion_inputs import make_inv_inputs

    site_data = {}
    for site, scale in (("MHD", 2.0), ("TAC", 1.0)):
        dataset = xr.Dataset(
            {
                "mf": ("time", np.array([10.0, 20.0]) * scale),
                "H": (("region", "time"), np.array([[1.0, 2.0]]) * scale),
            },
            coords={"time": pd.date_range("2020-01-01", periods=2, freq="h"), "region": [0]},
        )
        if error_kind in {"derived", "complete"}:
            dataset["mf_repeatability"] = ("time", np.array([3.0, 3.0]) * scale)
            dataset["mf_variability"] = ("time", np.array([4.0, 4.0]) * scale)
        if error_kind in {"custom", "complete"}:
            dataset["mf_error"] = ("time", np.array([7.0, 8.0]) * scale, {"source": "custom"})
        site_data[site] = dataset.chunk(time=1) if lazy else dataset
    originals = {site: dataset.copy(deep=True) for site, dataset in site_data.items()}
    merged = MergedData(
        site_data=site_data, flux_data={},
        site_options=SiteOptions.from_inputs(sites=["TAC", "MHD"], averaging_period=["1h", "2h"]),
    )

    selected, site_options = filter_observations(merged.site_data, merged.site_options)
    if error_kind == "derived":
        prepared = prepare_observation_errors(selected)
    else:
        with Callback(pretask=lambda *args: pytest.fail("supplied errors were computed")):
            prepared = prepare_observation_errors(selected)

    if error_kind == "complete":
        assert all(prepared[site] is merged.site_data[site] for site in prepared)
    assert tuple(prepared) == site_options.sites == ("TAC", "MHD")
    assert site_options.averaging_period == ("1h", "2h")
    gathered = make_inv_inputs(prepared, sites=list(site_options.sites))
    np.testing.assert_array_equal(gathered.site, ["TAC", "TAC", "MHD", "MHD"])
    expected_index = pd.MultiIndex.from_product(
        [["TAC", "MHD"], site_data["TAC"].time.values], names=["site", "time"]
    )
    pd.testing.assert_index_equal(gathered.indexes["nmeasure"], expected_index)
    assert {"site_indicator", "site_names"}.isdisjoint(gathered.variables)
    np.testing.assert_array_equal(gathered.mf, [10.0, 20.0, 20.0, 40.0])
    np.testing.assert_array_equal(gathered.H, [[1.0, 2.0, 2.0, 4.0]])
    expected_error = [5.0, 5.0, 10.0, 10.0] if error_kind == "derived" else [7.0, 8.0, 14.0, 16.0]
    np.testing.assert_allclose(gathered.mf_error, expected_error)
    if error_kind == "custom":
        assert gathered.mf_error.attrs == {"source": "custom"}
        for name in ("mf_repeatability", "mf_variability"):
            np.testing.assert_array_equal(gathered[name], np.zeros(4))
    for site, dataset in site_data.items():
        xr.testing.assert_identical(dataset, originals[site])
        assert prepared[site].mf.data is dataset.mf.data
        if error_kind != "derived":
            assert prepared[site].mf_error.data is dataset.mf_error.data
        if lazy:
            assert isinstance(prepared[site].mf_error.data, da.Array)
    assert tuple(merged.site_data) == ("MHD", "TAC")
    assert merged.sites == ("TAC", "MHD")


@pytest.mark.parametrize("lazy", [False, True])
def test_quadrature_preserves_borrowed_metadata(lazy):
    dataset = xr.Dataset(
        {
            "mf": ("time", [10.0, 20.0, 30.0], {"units": "1e-9", "long_name": "methane"}),
            "mf_repeatability": ("time", [1.0, 9.0, 3.0], {"method": "instrument"}),
            "mf_variability": ("time", [9.0, 1.0, 4.0], {"method": "window"}),
        },
        coords={"time": pd.date_range("2020-01-01", periods=3, freq="h")},
        attrs={"scale": "WMO", "footprint_transport_model": "NAME"},
    )
    if lazy:
        dataset = dataset.chunk(time=1)
    original = dataset.copy(deep=True)
    merged = _merged(dataset)
    prepared = prepare_observation_errors(merged.site_data)
    np.testing.assert_allclose(prepared["TAC"].mf_error, np.sqrt([82.0, 82.0, 25.0]))
    xr.testing.assert_identical(dataset, original)
    assert prepared["TAC"].mf.data is dataset.mf.data
    assert prepared["TAC"].attrs == dataset.attrs
    assert prepared["TAC"].mf_error.attrs == {"long_name": "methane_error", "units": "1e-9"}



@pytest.mark.parametrize("lazy", [False, True])
def test_zero_error_fallback_uses_retained_population_after_daytime_selection(lazy):
    dataset = xr.Dataset(
        {
            "mf": ("time", [100.0, 0.0, 10.0], {"units": "ppb", "long_name": "methane"}),
            "mf_repeatability": ("time", [1000.0, 0.0, 2.0], {"source": "instrument"}),
            "release_lon": ("time", [0.0, 0.0, 0.0]),
        },
        coords={"time": pd.to_datetime(["2020-01-01T00:00", "2020-01-01T12:00", "2020-01-01T13:00"])},
        attrs={"scale": "WMO"},
    )
    if lazy:
        dataset = dataset.chunk(time=1)
    original = dataset.copy(deep=True)
    merged = _merged(dataset)
    selected, options = filter_observations(merged.site_data, merged.site_options, filters="daytime")
    prepared = prepare_observation_errors(selected)
    np.testing.assert_allclose(prepared["TAC"].mf_error, [5.0, 2.0])
    np.testing.assert_array_equal(prepared["TAC"].time, dataset.time[1:])
    assert options.sites == ("TAC",)
    assert options.averaging_period == ("1h",)
    assert prepared["TAC"].attrs == dataset.attrs
    error_attrs = prepared["TAC"].mf_error.attrs
    assert {name: error_attrs[name] for name in ("long_name", "units")} == {
        "long_name": "methane_error", "units": "ppb",
    }
    xr.testing.assert_identical(dataset, original)


def test_custom_error_is_preserved_without_computing_or_requiring_components():
    dataset = xr.Dataset(
        {"mf_error": ("time", da.from_array([0.0, np.nan, 5.0], chunks=1), {"source": "custom"})},
        coords={"time": pd.date_range("2020-01-01", periods=3, freq="h")},
    )
    merged = _merged(dataset)
    with Callback(pretask=lambda *args: pytest.fail("custom error was computed")):
        prepared = prepare_observation_errors(merged.site_data)
        assert prepared["TAC"].mf_error.data is dataset.mf_error.data
    assert dataset.mf_error.attrs == {"source": "custom"}


@pytest.mark.parametrize("averaging_error", [False, True])
def test_variability_only_and_missing_components_remain_supported(averaging_error):
    dataset = xr.Dataset({"mf": ("time", [10.0, 20.0]), "mf_variability": ("time", [2.0, 3.0])})
    prepared = prepare_observation_errors(_merged(dataset).site_data, averaging_error=averaging_error)
    np.testing.assert_array_equal(prepared["TAC"].mf_error, [2.0, 3.0])
    np.testing.assert_array_equal(prepared["TAC"].mf_repeatability, [0.0, 0.0])
    with pytest.raises(ValueError, match="missing both"):
        prepare_observation_errors(_merged(dataset.drop_vars("mf_variability")).site_data)


@pytest.mark.parametrize("lazy", [False, True])
def test_nested_domains_prepare_errors_after_outer_selection_and_inner_alignment(monkeypatch, lazy):
    from openghg_inversions.rhime import nested, RhimeConfig

    # The error-ordering fixture uses a one-cell synthetic domain.
    monkeypatch.setattr(nested, "get_domain_info", lambda: {})
    time = pd.to_datetime(["2020-01-01T00:00", "2020-01-01T12:00", "2020-01-01T13:00"])
    outer = _merged(xr.Dataset(
        {"mf": ("time", [100.0, 0.0, 10.0]),
         "mf_repeatability": ("time", [1000.0, 0.0, 2.0]),
         "release_lon": ("time", [0.0, 0.0, 0.0])},
        coords={"time": time, "lat": [50.0], "lon": [-1.0]},
    ))
    inner = _merged(xr.Dataset(
        {"mf": ("time", [1000.0, 0.0, 20.0]), "mf_repeatability": ("time", [2000.0, 0.0, 4.0])},
        coords={"time": time, "lat": [50.0], "lon": [-1.0]},
    ))
    if lazy:
        outer = _merged(outer.site_data["TAC"].chunk(time=1))
        inner = _merged(inner.site_data["TAC"].chunk(time=1))
    config = RhimeConfig.from_params(dict(
        species="ch4", domain="EUROPE", sites=["TAC"], averaging_period="1h",
        start_date="2020-01-01", end_date="2020-01-02", output_name="nested-errors", mismatch_model="fixed_error",
        output_format="none", flux_sources=["inventory"], filters="daytime", use_bc=False,
    ), multisector=False)
    monkeypatch.setattr(MergedData, "from_options", lambda **kw: inner if kw["domain"] == "EUROPE-6km" else outer)
    monkeypatch.setattr(nested, "_mask_outer_for_aligned_inner_domain", lambda outer, inner: outer)
    captured = []

    class PreparedBothDomains(Exception):
        pass

    def reached_basis(merged, **kwargs):
        assert not isinstance(merged, MergedData)
        captured.append(merged)
        if len(captured) == 2:
            raise PreparedBothDomains

    monkeypatch.setattr(nested, "_prepare_one_domain", reached_basis)
    with pytest.raises(PreparedBothDomains):
        nested.run_nested(config=config, inner_domain="6km", inner_nbasis=2)
    np.testing.assert_allclose(captured[0].site_data["TAC"].mf_error, [5.0, 2.0])
    np.testing.assert_allclose(captured[1].site_data["TAC"].mf_error, [10.0, 4.0])
    np.testing.assert_array_equal(captured[1].site_data["TAC"].time, time[1:])
    assert "mf_error" not in outer.site_data["TAC"]
    assert "mf_error" not in inner.site_data["TAC"]


@pytest.mark.parametrize("component", ["mf_variability", "mf_repeatability", None])
def test_custom_error_with_missing_diagnostics_reaches_gathered_assembly(component):
    from openghg_inversions.inversion_inputs import make_inv_inputs

    dataset = xr.Dataset(
        {"mf": ("time", [10.0, 20.0]),
         "mf_error": ("time", [7.0, 8.0], {"source": "custom", "units": "ppb"}),
         "H": (("region", "time"), [[1.0, 2.0]])},
        coords={"time": pd.date_range("2020-01-01", periods=2, freq="h"), "region": [0]},
    )
    if component:
        dataset[component] = ("time", [2.0, 3.0])
    original = dataset.copy(deep=True)
    prepared = prepare_observation_errors(_merged(dataset).site_data)
    gathered = make_inv_inputs(prepared, sites=["TAC"])
    np.testing.assert_array_equal(gathered.mf_error, [7.0, 8.0])
    assert gathered.mf_error.attrs == {"source": "custom", "units": "ppb"}
    for name in ("mf_repeatability", "mf_variability"):
        np.testing.assert_array_equal(gathered[name], [2.0, 3.0] if name == component else [0.0, 0.0])
    xr.testing.assert_identical(dataset, original)


@pytest.mark.parametrize("lazy", [False, True])
def test_selection_drops_invalid_site_before_requiring_error_components(lazy):
    time = pd.to_datetime(["2020-01-01T00:00", "2020-01-01T12:00"])
    valid = xr.Dataset(
        {"mf": ("time", [100.0, 10.0]), "mf_repeatability": ("time", [1000.0, 2.0]),
         "release_lon": ("time", [0.0, 0.0])}, coords={"time": time},
    )
    invalid = valid.isel(time=[0]).drop_vars("mf_repeatability")
    site_data = {"BAD": invalid, "TAC": valid}
    if lazy:
        site_data = {site: ds.chunk(time=1) for site, ds in site_data.items()}
    options = SiteOptions.from_inputs(sites=["BAD", "TAC"], averaging_period=["2h", "1h"])
    selected, retained = filter_observations(site_data, options, filters="daytime")
    prepared = prepare_observation_errors(selected)
    assert retained.sites == ("TAC",)
    assert retained.averaging_period == ("1h",)
    np.testing.assert_array_equal(prepared["TAC"].mf_error, [2.0])
    with pytest.raises(ValueError, match="missing both"):
        prepare_observation_errors({"BAD": valid.isel(time=[1]).drop_vars("mf_repeatability")})
    assert all("mf_error" not in ds for ds in site_data.values())


def test_selected_lazy_error_derivation_never_executes_rejected_payload():
    from dask import delayed

    @delayed
    def rejected():
        pytest.fail("rejected concentration payload was computed")

    mf = da.concatenate([da.from_delayed(rejected(), shape=(1,), dtype=float), da.from_array([0.0, 10.0], chunks=1)])
    dataset = xr.Dataset(
        {"mf": ("time", mf), "mf_repeatability": ("time", da.from_array([1000.0, 0.0, 2.0], chunks=1)),
         "release_lon": ("time", [0.0, 0.0, 0.0])},
        coords={"time": pd.to_datetime(["2020-01-01T00:00", "2020-01-01T12:00", "2020-01-01T13:00"])},
    )
    merged = _merged(dataset)
    selected, _ = filter_observations(merged.site_data, merged.site_options, filters="daytime")
    prepared = prepare_observation_errors(selected)
    np.testing.assert_array_equal(prepared["TAC"].mf_error, [5.0, 2.0])
    assert isinstance(dataset.mf.data, da.Array)


@pytest.mark.parametrize("lazy", [False, True])
def test_selection_preserves_custom_zero_nan_and_repeatability_nan_policy(lazy):
    dataset = xr.Dataset(
        {"mf": ("time", [100.0, 0.0, 10.0]),
         "mf_error": ("time", [1000.0, 0.0, np.nan], {"source": "custom"}),
         "mf_repeatability": ("time", [1000.0, np.nan, 2.0]),
         "release_lon": ("time", [0.0, 0.0, 0.0])},
        coords={"time": pd.to_datetime(["2020-01-01T00:00", "2020-01-01T12:00", "2020-01-01T13:00"])},
    )
    if lazy:
        dataset = dataset.chunk(time=1)
    original = dataset.copy(deep=True)
    merged = _merged(dataset)
    selected, _ = filter_observations(merged.site_data, merged.site_options, filters="daytime")
    prepared = prepare_observation_errors(selected)
    np.testing.assert_allclose(prepared["TAC"].mf_error, [0.0, np.nan], equal_nan=True)
    assert prepared["TAC"].mf_error.attrs == {"source": "custom"}
    derived = prepare_observation_errors({"TAC": selected["TAC"].drop_vars("mf_error")}, averaging_error=False)
    np.testing.assert_allclose(derived["TAC"].mf_error, [np.nan, 2.0], equal_nan=True)
    xr.testing.assert_identical(dataset, original)
