"""Reported errors are derived on acquired populations before filtering."""

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.inversion_data import RhimeMergedData, SiteOptions
from openghg_inversions.rhime import filter_observations, prepare_observation_errors


def _merged(dataset):
    return RhimeMergedData(
        site_data={"TAC": dataset}, flux_data={},
        site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"),
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_quadrature_precedes_daily_median_and_preserves_borrowed_metadata(lazy):
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
    prepared = prepare_observation_errors(merged)
    filtered = filter_observations(prepared, filters="daily_median")
    # median(sqrt(r**2 + v**2)) != sqrt(median(r)**2 + median(v)**2).
    assert filtered.site_data["TAC"].mf_error.item() == pytest.approx(np.sqrt(82.0))
    assert np.sqrt(np.median(dataset.mf_repeatability.values) ** 2 + np.median(dataset.mf_variability.values) ** 2) == 5.0
    xr.testing.assert_identical(dataset, original)
    assert prepared.site_data["TAC"].mf.data is dataset.mf.data
    assert prepared.site_data["TAC"].attrs == dataset.attrs
    assert prepared.site_data["TAC"].mf_error.attrs == {"long_name": "methane_error", "units": "1e-9"}
    assert prepared.site_options == merged.site_options
    assert prepared.provenance == merged.provenance


def test_zero_error_fallback_uses_acquired_population_before_daytime_selection():
    dataset = xr.Dataset(
        {
            "mf": ("time", [100.0, 0.0, 10.0]),
            "mf_repeatability": ("time", [2.0, 0.0, 2.0]),
            "release_lon": ("time", [0.0, 0.0, 0.0]),
        },
        coords={"time": pd.to_datetime(["2020-01-01T00:00", "2020-01-01T12:00", "2020-01-01T13:00"])},
    ).chunk(time=1)
    filtered = filter_observations(prepare_observation_errors(_merged(dataset)), filters="daytime")
    np.testing.assert_allclose(filtered.site_data["TAC"].mf_error, [np.std([100.0, 0.0, 10.0]), 2.0])
    assert "mf_error" not in dataset
    assert "mf_variability" not in dataset
    np.testing.assert_array_equal(filtered.site_data["TAC"].time, dataset.time[1:])


def test_custom_error_is_preserved_without_computing_or_requiring_components():
    dataset = xr.Dataset(
        {"mf_error": ("time", da.from_array([0.0, np.nan, 5.0], chunks=1), {"source": "custom"})},
        coords={"time": pd.date_range("2020-01-01", periods=3, freq="h")},
    )
    merged = _merged(dataset)
    with Callback(pretask=lambda *args: pytest.fail("custom error was computed")):
        prepared = prepare_observation_errors(merged)
        assert prepared.site_data["TAC"].mf_error.data is dataset.mf_error.data
    assert dataset.mf_error.attrs == {"source": "custom"}


@pytest.mark.parametrize("averaging_error", [False, True])
def test_variability_only_and_missing_components_remain_supported(averaging_error):
    dataset = xr.Dataset({"mf": ("time", [10.0, 20.0]), "mf_variability": ("time", [2.0, 3.0])})
    prepared = prepare_observation_errors(_merged(dataset), averaging_error=averaging_error)
    np.testing.assert_array_equal(prepared.site_data["TAC"].mf_error, [2.0, 3.0])
    np.testing.assert_array_equal(prepared.site_data["TAC"].mf_repeatability, [0.0, 0.0])
    with pytest.raises(ValueError, match="missing both"):
        prepare_observation_errors(_merged(dataset.drop_vars("mf_variability")))


def test_nested_domains_prepare_errors_before_outer_aggregation_and_inner_alignment(monkeypatch):
    from openghg_inversions.rhime import nested, RhimeConfig

    time = pd.date_range("2020-01-01", periods=3, freq="h")
    outer = _merged(xr.Dataset(
        {"mf": ("time", [10.0, 20.0, 30.0]),
         "mf_repeatability": ("time", [1.0, 9.0, 3.0]),
         "mf_variability": ("time", [9.0, 1.0, 4.0])},
        coords={"time": time},
    ))
    inner = _merged(xr.Dataset(
        {"mf": ("time", [0.0, 10.0, 100.0]), "mf_repeatability": ("time", [0.0, 2.0, 2.0])},
        coords={"time": time},
    ).chunk(time=1))
    config = RhimeConfig.from_params(dict(
        species="ch4", domain="EUROPE", sites=["TAC"], averaging_period="1h",
        start_date="2020-01-01", end_date="2020-01-02", output_name="nested-errors",
        output_format="none", flux_sources=["inventory"], filters="daily_median", use_bc=False,
    ), multisector=False)
    monkeypatch.setattr(RhimeMergedData, "from_options", lambda **kw: inner if kw["domain"] == "EUROPE-6km" else outer)
    monkeypatch.setattr(nested, "mask_outer_merged_for_inner_domain", lambda outer, inner: outer)
    captured = []

    class PreparedBothDomains(Exception):
        pass

    def reached_basis(merged, **kwargs):
        captured.append(merged)
        if len(captured) == 2:
            raise PreparedBothDomains

    monkeypatch.setattr(nested, "_prepare_one_domain", reached_basis)
    with pytest.raises(PreparedBothDomains):
        nested.prepare_nested_rhime_inputs(config, inner_domain="6km", inner_nbasis=2)
    assert captured[0].site_data["TAC"].mf_error.item() == pytest.approx(np.sqrt(82.0))
    assert captured[1].site_data["TAC"].mf_error.item() == pytest.approx(np.std([0.0, 10.0, 100.0]))
    np.testing.assert_array_equal(captured[1].site_data["TAC"].time, time[:1])
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
    prepared = prepare_observation_errors(_merged(dataset))
    gathered = make_inv_inputs(prepared.site_data, sites=["TAC"])
    np.testing.assert_array_equal(gathered.mf_error, [7.0, 8.0])
    assert gathered.mf_error.attrs == {"source": "custom", "units": "ppb"}
    for name in ("mf_repeatability", "mf_variability"):
        np.testing.assert_array_equal(gathered[name], [2.0, 3.0] if name == component else [0.0, 0.0])
    xr.testing.assert_identical(dataset, original)
