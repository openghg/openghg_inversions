"""Acquisition factories retain selectors and borrow lazy datasets."""

from types import SimpleNamespace

import dask.array as da
import pytest
import xarray as xr
from dask.callbacks import Callback

from openghg_inversions.inversion_data import MergedData, SiteOptions
from openghg_inversions.inversion_data import acquisition as get_data








def test_fresh_factory_retains_selectors_and_borrows_lazy_datasets(monkeypatch):
    options = SiteOptions.from_inputs(
        sites=["tac", "mhd", "rgl"],
        averaging_period=["1h", "2h", "3h"],
        inlet=["10m", "20m", "30m"],
        time_resolved=[None, False, True],
    )
    dataset = xr.Dataset(
        {"mf": ("time", da.from_array([1.0, 2.0]), {"units": "1e-9"})},
        attrs={"scale": "WMO"},
    )
    footprint_selectors = []

    def footprints(**kwargs):
        footprint_selectors.append((kwargs["site"], kwargs["time_resolved"]))
        return SimpleNamespace(data=dataset, metadata={})

    class CustomMergedData(MergedData):
        pass

    monkeypatch.setattr(get_data, "get_flux_data", lambda **kw: {})
    monkeypatch.setattr(
        get_data, "get_obs_data",
        lambda **kw: None if kw["site"] == "TAC" else SimpleNamespace(data=dataset, metadata={}),
    )
    monkeypatch.setattr(get_data, "get_footprint_data", footprints)
    monkeypatch.setattr(get_data, "merged_scenario_data", lambda *a, **kw: dataset)
    with Callback(pretask=lambda *args: pytest.fail("factory computed borrowed observations")):
        result = CustomMergedData.from_options(
            species="ch4",
            site_options=options,
            domain="EUROPE",
            start_date="2020-01-01",
            end_date="2020-02-01",

            flux_sources=["inventory"],
            use_bc=False,
        )
    assert result.site_options == options.select_indices([1, 2])
    assert isinstance(result, CustomMergedData)
    assert footprint_selectors == [("MHD", False), ("RGL", True)]
    assert result.site_data["RGL"] is dataset
    assert options.sites == ("TAC", "MHD", "RGL")





def test_retrieval_builds_modern_record_without_legacy_adapters(monkeypatch):
    """Fresh retrieval retains aligned selectors and identities without fp_all."""
    options = SiteOptions.from_inputs(
        sites=["TAC", "MHD"], averaging_period=["1h", "2h"], inlet=["10m", "20m"]
    )
    site = xr.Dataset(
        {"mf": ("time", da.from_array([1.0, 2.0]), {"units": "1e-9"})},
        attrs={"scale": "WMO"},
    )
    flux = xr.Dataset({"flux": ("time", da.from_array([3.0, 4.0]))})
    boundary = xr.Dataset({"bc": ("time", da.from_array([5.0, 6.0]))})

    def wrap(dataset, uuid):
        return SimpleNamespace(data=dataset, metadata={"uuid": uuid, "dataversion": "v2"})

    monkeypatch.setattr(get_data, "get_flux_data", lambda **kw: {"inventory": wrap(flux, "flux-id")})
    monkeypatch.setattr(get_data, "get_bc", lambda **kw: wrap(boundary, "boundary-id"))
    monkeypatch.setattr(
        get_data, "get_obs_data", lambda **kw: None if kw["site"] == "TAC" else wrap(site, "obs-id")
    )
    monkeypatch.setattr(get_data, "get_footprint_data", lambda **kw: wrap(site, "footprint-id"))
    monkeypatch.setattr(get_data, "merged_scenario_data", lambda *a, **kw: site)
    with Callback(pretask=lambda *a: pytest.fail("record construction computed borrowed data")):
        merged = MergedData.from_options(
            site_options=options,
            species="ch4",
            domain="EUROPE",
            start_date="2020-01-01",
            end_date="2020-02-01",

            flux_sources=["inventory"],
        )
    assert merged.site_options == options.select_indices([1])
    assert merged.site_data == {"MHD": site}
    assert merged.flux_data["inventory"] is flux
    assert merged.boundary_data is boundary
    assert merged.provenance.observations["MHD"].uuid == "obs-id"
    assert merged.provenance.footprints["MHD"].uuid == "footprint-id"
    assert merged.provenance.flux["inventory"].uuid == "flux-id"
    assert merged.provenance.boundary.uuid == "boundary-id"
    assert merged.acquisition.species == "ch4"
    assert merged.acquisition.start_date == "2020-01-01"


@pytest.mark.parametrize(("selectors", "store", "source_domain"), [
    ({}, "user", "FINE"),
    ({"flux_store": "selected", "flux_domain": "COARSE"}, "selected", "COARSE"),
])
def test_flux_selectors_retrieve_source_domain_and_preserve_footprint_domain(
    monkeypatch, selectors, store, source_domain
):
    """A source domain selects flux metadata while the footprint grid stays native."""
    import numpy as np
    import pandas as pd

    time = pd.date_range("2020-01-01", periods=1)
    observations = xr.Dataset(
        {"mf": ("time", [1.0], {"units": "1e-9"}), "mf_error": ("time", [0.2]),
         "mf_repeatability": ("time", [0.2])},
        coords={"time": time}, attrs={"scale": "WMO"},
    )
    flux = xr.Dataset(
        {"flux": (("lat", "lon", "time"), np.arange(4.0).reshape(2, 2, 1))},
        coords={"lat": [0., 1.], "lon": [0., 1.], "time": time},
    )
    footprint = xr.Dataset(
        {"fp": (("lat", "lon", "time"), np.ones((3, 3, 1)))},
        coords={"lat": [0., .5, 1.], "lon": [0., .5, 1.], "time": time},
    )

    def wrap(data):
        return SimpleNamespace(data=data, metadata={})

    def retrieve_flux(**kwargs):
        assert kwargs["store"] == store
        assert kwargs["domain"] == source_domain
        return {"inventory": wrap(flux)}

    def retrieve_footprint(**kwargs):
        assert kwargs["domain"] == "FINE"
        return wrap(footprint)

    monkeypatch.setattr(get_data, "get_flux_data", retrieve_flux)
    monkeypatch.setattr(get_data, "get_obs_data", lambda **kwargs: wrap(observations))
    monkeypatch.setattr(get_data, "get_footprint_data", retrieve_footprint)
    monkeypatch.setattr(get_data, "merged_scenario_data", lambda *args, **kwargs: observations)
    acquired = MergedData.from_options(
        species="ch4", site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"),
        domain="FINE", start_date="2020-01-01", end_date="2020-01-02",
        flux_sources=["inventory"], use_bc=False, **selectors,
    )
    assert acquired.acquisition.domain == "FINE"
    assert acquired.acquisition.flux_domain == selectors.get("flux_domain")
    if source_domain == "COARSE":
        assert acquired.flux_data["inventory"].sizes["lat"] == 3
        assert acquired.flux_data["inventory"].flux.sel(lat=1, lon=1).item() == 3.
        assert acquired.flux_data["inventory"].flux.sel(lat=.5, lon=.5).item() == 0.
    else:
        xr.testing.assert_identical(acquired.flux_data["inventory"], flux)
    assert flux.sizes["lat"] == 2


@pytest.mark.parametrize(("old", "new"), [
    ("emissions_store", "flux_store"), ("emissions_domain", "flux_domain"),
])
@pytest.mark.parametrize("value", [None, "same"])
def test_direct_acquisition_rejects_retired_flux_spellings(old, new, value):
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{old}'"):
        MergedData.from_options(
            species="ch4", site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"),
            domain="EUROPE", start_date="2020-01-01", end_date="2020-01-02",
            flux_sources=["inventory"], **{old: value, new: value},
        )
