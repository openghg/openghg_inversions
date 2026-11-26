"""Boundary retrieval retains native labels for the intervals covering releases."""

from types import SimpleNamespace
from unittest.mock import Mock

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback
from openghg.dataobjects import BoundaryConditionsData
from openghg.types import SearchError

from openghg_inversions.inversion_data import MergedData, SiteOptions, acquisition


@pytest.fixture
def boundary_provider(monkeypatch):
    """Mirror BaseData's label slicing of a multi-timestamp boundary product."""
    calls = []

    def install(native, period="monthly"):
        metadata = {"time_period": period, "uuid": "boundary-id", "latest_version": "v3"}
        catalog = Mock(return_value=SimpleNamespace(results=pd.DataFrame([metadata])))
        monkeypatch.setattr(acquisition, "search_bc", catalog)

        def retrieve(**kwargs):
            calls.append(kwargs)
            start = pd.Timestamp(kwargs["start_date"]) - pd.Timedelta(seconds=1)
            end = pd.Timestamp(kwargs["end_date"]) - pd.Timedelta(seconds=1)
            return BoundaryConditionsData(
                data=native.sel(time=slice(start, end)), metadata=metadata, sort=False,
            )

        monkeypatch.setattr(acquisition, "get_bc", retrieve)
        return calls, catalog, metadata

    return install


@pytest.mark.parametrize("period,labels,start,end,expected", [
    ("monthly", ["2020-01-01", "2020-02-01", "2020-03-01"], "2020-01-16", "2020-02-02", ["2020-01-01", "2020-02-01"]),
    ("monthly", ["2020-01-01", "2020-02-01", "2020-03-01"], "2020-02-01", "2020-03-01", ["2020-01-01", "2020-02-01"]),
    ("monthly", ["2020-01-15", "2020-02-15", "2020-03-15"], "2020-02-01", "2020-02-16", ["2020-01-15", "2020-02-15"]),
    ("2h", ["2020-01-01 01:00", "2020-01-01 03:00", "2020-01-01 05:00"], "2020-01-01 02:00", "2020-01-01 04:00", ["2020-01-01 01:00", "2020-01-01 03:00"]),
])
def test_boundary_getter_preserves_governing_native_interval(boundary_provider, period, labels, start, end, expected):
    native = xr.Dataset({"vmr_n": ("time", da.from_array([401., 402., 403.], chunks=1))},
                        coords={"time": pd.to_datetime(labels)})
    calls, catalog, metadata = boundary_provider(native, period)
    with Callback(pretask=lambda *args: pytest.fail("retrieval computed borrowed boundary payload")):
        boundary, identity = acquisition._get_boundary_data(
            use_bc=True, species="co2", domain="EUROPE", bc_input="CAMS",
            start_date=start, end_date=end, store="boundary-store",
        )
    assert boundary.data.indexes["time"].equals(pd.DatetimeIndex(expected))
    assert isinstance(boundary.data.vmr_n.data, da.Array)
    assert boundary.metadata is metadata
    assert native.indexes["time"].equals(pd.DatetimeIndex(labels))
    assert identity.uuid == "boundary-id" and identity.dataversion == "v3"
    assert identity.store == "boundary-store"
    assert calls[0]["end_date"] == end
    assert calls[0]["uuid"] == ["boundary-id"]
    assert catalog.call_args.kwargs == dict(
        species="co2", domain="EUROPE", bc_input="CAMS", start_date=start, end_date=end, store="boundary-store",
    )


def test_disabled_boundary_does_not_search_or_retrieve(monkeypatch):
    monkeypatch.setattr(acquisition, "search_bc", Mock(side_effect=AssertionError("searched disabled BC")))
    monkeypatch.setattr(acquisition, "get_bc", Mock(side_effect=AssertionError("retrieved disabled BC")))
    assert acquisition._get_boundary_data(
        use_bc=False, species="co2", domain="EUROPE", bc_input=None,
        start_date="2020-01-16", end_date="2020-02-01", store="user",
    ) == (None, None)


def test_boundary_without_declared_period_keeps_requested_start(monkeypatch):
    monkeypatch.setattr(acquisition, "search_bc", lambda **kw: SimpleNamespace(results=pd.DataFrame()))
    retrieve = Mock(return_value=SimpleNamespace(data=xr.Dataset(), metadata={}))
    monkeypatch.setattr(acquisition, "get_bc", retrieve)
    acquisition._get_boundary_data(
        use_bc=True, species="co2", domain="EUROPE", bc_input=None,
        start_date="2020-01-16", end_date="2020-02-01", store="user",
    )
    assert retrieve.call_args.kwargs["start_date"] == pd.Timestamp("2020-01-16")
    assert "uuid" not in retrieve.call_args.kwargs


def test_boundary_padding_keeps_original_catalog_selection(monkeypatch):
    metadata = pd.DataFrame({"uuid": ["current"], "time_period": ["monthly"]})
    monkeypatch.setattr(acquisition, "search_bc", lambda **kw: SimpleNamespace(results=metadata))

    def retrieve(**kwargs):
        assert kwargs["start_date"] == pd.Timestamp("2019-12-16")
        # The padded dates also overlap an expired, distinct product.
        candidates = {"current", "expired"}
        candidates.intersection_update(kwargs.get("uuid", candidates))
        if len(candidates) != 1:
            raise SearchError("Multiple entries found")
        return SimpleNamespace(data=xr.Dataset(), metadata={"uuid": candidates.pop()})

    monkeypatch.setattr(acquisition, "get_bc", retrieve)
    boundary, identity = acquisition._get_boundary_data(
        use_bc=True, species="co2", domain="EUROPE", bc_input=None,
        start_date="2020-01-16", end_date="2020-02-01", store="user",
    )
    assert identity.uuid == boundary.metadata["uuid"] == "current"


@pytest.mark.parametrize("uuids", [["current"], ["current", "also-current"]])
def test_boundary_mixed_periods_rewind_to_earliest_and_preserve_candidates(monkeypatch, uuids):
    catalog = pd.DataFrame({"uuid": [uuids[0], uuids[-1]], "time_period": ["monthly", "yearly"]})
    monkeypatch.setattr(acquisition, "search_bc", lambda **kw: SimpleNamespace(results=catalog))
    retrieve = Mock(return_value=SimpleNamespace(data=xr.Dataset(), metadata={}))
    monkeypatch.setattr(acquisition, "get_bc", retrieve)
    acquisition._get_boundary_data(
        use_bc=True, species="co2", domain="EUROPE", bc_input=None,
        start_date="2020-01-16", end_date="2020-02-01", store="user",
    )
    assert retrieve.call_args.kwargs["start_date"] == pd.Timestamp("2019-01-16")
    assert set(retrieve.call_args.kwargs["uuid"]) == set(uuids)


def test_missing_boundary_keeps_search_error(monkeypatch):
    monkeypatch.setattr(acquisition, "search_bc", Mock(side_effect=SearchError("missing catalog")))
    with pytest.raises(SearchError, match="Could not find matching boundary conditions"):
        acquisition._get_boundary_data(
            use_bc=True, species="co2", domain="EUROPE", bc_input=None,
            start_date="2020-01-16", end_date="2020-02-01", store="user",
        )


@pytest.fixture
def acquire_monthly_boundary(monkeypatch, boundary_provider):
    """Public acquisition with provider-like slicing and real boundary arithmetic."""
    def acquire(start, end, *, centered=False, scenario=False):
        times = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(hours=12), freq="12h")
        obs = xr.Dataset({"co2": ("time", np.full(len(times), 400.), {"units": "ppm"})},
                         coords={"time": times}, attrs={"scale": "WMO", "platform": "surface"})
        fp = xr.Dataset({"fp": (("lat", "lon", "time"), np.ones((1, 1, len(times))), {"units": "s m2 / mol"})},
                        coords={"lat": [0.], "lon": [0.], "height": [10.], "time": times})
        flux = xr.Dataset({"flux": (fp.fp.dims, np.ones(fp.fp.shape), {"units": "mol / m2 / s"})},
                          coords={name: fp[name] for name in fp.fp.dims})
        bc = xr.Dataset(coords={"height": fp.height, "lat": fp.lat, "lon": fp.lon,
                                "time": pd.date_range("2020-01-01", periods=3, freq="MS")},
                        attrs={"time_period": "monthly"})
        for direction in "nesw":
            side = "lon" if direction in "ns" else "lat"
            fp[f"particle_locations_{direction}"] = (("height", side, "time"), np.full((1, 1, len(times)), .25), {"units": "1"})
            bc[f"vmr_{direction}"] = (("height", side, "time"), np.array([[[401., 402., 403.]]]), {"units": "ppm"})
        boundary_provider(bc)
        monkeypatch.setattr(acquisition, "get_obs_data", lambda **kw: SimpleNamespace(data=obs, metadata={}))
        monkeypatch.setattr(acquisition, "get_footprint_data", lambda **kw: SimpleNamespace(data=fp, metadata={}))
        monkeypatch.setattr(acquisition, "get_flux_data", lambda **kw: {"a": SimpleNamespace(data=flux, metadata={})})
        if scenario:
            def merge(observations, footprint, fluxes, boundary, **kwargs):
                from openghg.analyse._modelled_baseline import baseline_sensitivities
                result = baseline_sensitivities(boundary.data, footprint.data, species="co2", allow_climatology=False)
                return obs.rename(co2="mf").assign(result)
            monkeypatch.setattr(acquisition, "merged_scenario_data", merge)
        return MergedData.from_options(
            species="co2", site_options=SiteOptions.from_inputs(sites=["MHD"], averaging_period="1h"),
            domain="EUROPE", start_date=start, end_date=end, flux_sources=["a"], use_bc=True,
            surface_transport=None if scenario else {"a": "integrated"},
            **({"boundary_representation": "centered", "reference": 400.,
                "reference_units": "ppm", "mass_tolerance": 1e-6} if centered else {}),
        )
    return acquire


@pytest.mark.parametrize("scenario", [False, True])
@pytest.mark.parametrize("start,end,expected", [
    ("2020-01-16", "2020-01-17", [401.]),
    ("2020-02-01", "2020-02-02", [402.]),
    ("2020-01-31", "2020-02-02", [401., 402.]),
])
def test_fresh_ordinary_acquisition_retains_monthly_boundary(acquire_monthly_boundary, scenario, start, end, expected):
    result = acquire_monthly_boundary(start, end, scenario=scenario)
    site = result.site_data["MHD"]
    np.testing.assert_allclose(sum(site[f"bc_{direction}"] for direction in "nesw").squeeze(), np.repeat(expected, 2))
    assert result.provenance.boundary.uuid == "boundary-id"
    assert result.provenance.boundary.dataversion == "v3"
