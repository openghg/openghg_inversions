"""Acquisition factories retain selectors and borrow lazy datasets."""

import inspect
import warnings
from types import SimpleNamespace

import dask.array as da
import pytest
import xarray as xr
from dask.callbacks import Callback

from openghg_inversions.inversion_data import AcquisitionFacts, RhimeMergedData, SiteOptions
from openghg_inversions.inversion_data import acquisition as get_data
from openghg_inversions.hbmcmc import legacy_data


def test_legacy_retrieval_alias_preserves_signature_docs_and_forwarding(monkeypatch):
    modern = legacy_data.retrieve_inversion_data
    legacy = legacy_data.data_processing_surface_notracer
    assert inspect.signature(legacy) == inspect.signature(modern)
    assert legacy.__doc__ == modern.__doc__
    sentinel = object()
    monkeypatch.setattr(legacy_data, "retrieve_inversion_data", lambda *args, **kwargs: (sentinel, args, kwargs))
    with pytest.warns(DeprecationWarning, match="retrieve_inversion_data"):
        assert legacy("ch4", domain="EUROPE") == (sentinel, ("ch4",), {"domain": "EUROPE"})


def test_neutral_retrieval_does_not_emit_deprecation():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="flux_sources"):
            legacy_data.retrieve_inversion_data("ch4", ["TAC"], "EUROPE", "1h", "2020-01-01", "2020-02-01")
    assert not any(issubclass(item.category, DeprecationWarning) for item in caught)


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

    class CustomMergedData(RhimeMergedData):
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


@pytest.mark.parametrize("entrypoint", ["retrieve_inversion_data", "data_processing_surface_notracer"])
def test_public_retrieval_hides_private_provenance_transport(monkeypatch, entrypoint):
    """Both public six-tuple APIs retain their established mapping keys."""
    from contextlib import nullcontext

    site = xr.Dataset({"mf": ("time", [1.0]), "mf_error": ("time", [0.1]),
                       "mf_repeatability": ("time", [0.1]), "mf_variability": ("time", [0.0])})
    retained = (["TAC"], ["185m"], ["185m"], ["picarro"], ["1h"])
    options = SiteOptions.from_inputs(
        sites=["TAC"], averaging_period="1h", inlet="185m", fp_height="185m", instrument="picarro"
    )
    acquired = RhimeMergedData(
        site_data={"TAC": site},
        flux_data={},
        boundary_data=xr.Dataset(),
        site_options=options,
        acquisition=AcquisitionFacts(),
    )

    def retrieve(**kwargs):
        return acquired

    monkeypatch.setattr(RhimeMergedData, "from_options", retrieve)
    warning = pytest.warns(DeprecationWarning) if entrypoint == "data_processing_surface_notracer" else nullcontext()
    with warning:
        result = getattr(legacy_data, entrypoint)(
            "ch4", ["TAC"], "EUROPE", "1h", "2020-01-01", "2020-01-02"
        )
    assert len(result) == 6
    assert result[1:] == retained
    assert list(result[0]) == [".flux", ".split_by_sectors", ".bc", "TAC"]
    assert result[0]["TAC"] is site
    assert acquired.acquisition == AcquisitionFacts()


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
    monkeypatch.setattr(legacy_data, "from_legacy_fp_all", lambda *a, **kw: pytest.fail("used fp_all"))
    monkeypatch.setattr(legacy_data, "to_legacy_fp_all", lambda *a, **kw: pytest.fail("used fp_all"))
    with Callback(pretask=lambda *a: pytest.fail("record construction computed borrowed data")):
        merged = RhimeMergedData.from_options(
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
    ({"emissions_store": "selected", "emissions_domain": "COARSE"}, "selected", "COARSE"),
])
def test_flux_selectors_retrieve_source_domain_and_preserve_footprint_domain(
    monkeypatch, selectors, store, source_domain
):
    """A source domain selects flux metadata while the footprint grid stays native."""
    from contextlib import nullcontext
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
    warning = pytest.warns(DeprecationWarning, match=r"removed in 0\.9") if "emissions_store" in selectors else nullcontext()
    with warning:
        acquired = RhimeMergedData.from_options(
            species="ch4", site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"),
            domain="FINE", start_date="2020-01-01", end_date="2020-01-02",
            flux_sources=["inventory"], use_bc=False, **selectors,
        )
    assert acquired.acquisition.domain == "FINE"
    assert acquired.acquisition.flux_domain == selectors.get("flux_domain", selectors.get("emissions_domain"))
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
def test_direct_acquisition_rejects_duplicate_flux_spellings(old, new, value):
    with pytest.raises(ValueError, match="cannot be supplied together"):
        RhimeMergedData.from_options(
            species="ch4", site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"),
            domain="EUROPE", start_date="2020-01-01", end_date="2020-01-02",
            flux_sources=["inventory"], **{old: value, new: value},
        )
