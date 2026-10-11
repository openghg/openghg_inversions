"""Acquisition factories retain selectors and borrow lazy datasets."""

import inspect
import warnings

import dask.array as da
import pytest
import xarray as xr
from dask.callbacks import Callback

from openghg_inversions.inversion_data import AcquisitionFacts, RhimeMergedData, SiteOptions
from openghg_inversions.inversion_data import get_data


def test_legacy_retrieval_alias_preserves_signature_docs_and_forwarding(monkeypatch):
    modern = get_data.retrieve_inversion_data
    legacy = get_data.data_processing_surface_notracer
    assert inspect.signature(legacy) == inspect.signature(modern)
    assert legacy.__doc__ == modern.__doc__
    sentinel = object()
    monkeypatch.setattr(get_data, "retrieve_inversion_data", lambda *args, **kwargs: (sentinel, args, kwargs))
    with pytest.warns(DeprecationWarning, match="retrieve_inversion_data"):
        assert legacy("ch4", domain="EUROPE") == (sentinel, ("ch4",), {"domain": "EUROPE"})


def test_neutral_retrieval_does_not_emit_deprecation():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="emissions"):
            get_data.retrieve_inversion_data("ch4", ["TAC"], "EUROPE", "1h", "2020-01-01", "2020-02-01")
    assert not any(issubclass(item.category, DeprecationWarning) for item in caught)


def test_fresh_factory_retains_selectors_and_borrows_lazy_datasets(monkeypatch):
    options = SiteOptions.from_inputs(
        sites=["tac", "mhd", "rgl"],
        averaging_period=["1h", "2h", "3h"],
        inlet=["10m", "20m", "30m"],
        time_resolved=[None, False, True],
    )
    dataset = xr.Dataset({"mf": ("time", da.from_array([1.0, 2.0]))})

    def retrieve(**kwargs):
        assert kwargs["site_options"].sites == ("TAC", "MHD", "RGL")
        assert "save_merged_data" not in kwargs
        assert kwargs["site_options"].time_resolved == (None, False, True)
        return RhimeMergedData(
            site_data={"RGL": dataset, "MHD": dataset},
            flux_data={},
            site_options=options.select_indices([2, 1]),
            acquisition=AcquisitionFacts(),
        )

    monkeypatch.setattr(get_data, "_retrieve_inversion_data_from_options", retrieve)
    monkeypatch.setattr(RhimeMergedData, "from_legacy_fp_all", lambda *a, **kw: pytest.fail("fresh used fp_all"))
    monkeypatch.setattr(RhimeMergedData, "to_legacy_fp_all", lambda *a, **kw: pytest.fail("fresh used fp_all"))
    with Callback(pretask=lambda *args: pytest.fail("factory computed borrowed observations")):
        result = RhimeMergedData.from_options(
            species="ch4",
            site_options=options,
            domain="EUROPE",
            start_date="2020-01-01",
            end_date="2020-02-01",

            flux_sources=["inventory"],
        )
    assert result.site_options == options.select_indices([2, 1])
    assert result.site_data["RGL"] is dataset
    assert options.sites == ("TAC", "MHD", "RGL")


@pytest.mark.parametrize("entrypoint", ["retrieve_inversion_data", "data_processing_surface_notracer"])
def test_public_retrieval_hides_private_provenance_transport(monkeypatch, entrypoint):
    """Both public six-tuple APIs retain their established mapping keys."""
    from contextlib import nullcontext

    site = xr.Dataset({"mf": ("time", [1.0])})
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

    monkeypatch.setattr(get_data, "_retrieve_inversion_data_from_options", retrieve)
    warning = pytest.warns(DeprecationWarning) if entrypoint == "data_processing_surface_notracer" else nullcontext()
    with warning:
        result = getattr(get_data, entrypoint)(
            "ch4", ["TAC"], "EUROPE", "1h", "2020-01-01", "2020-01-02"
        )
    assert len(result) == 6
    assert result[1:] == retained
    assert list(result[0]) == [".flux", ".split_by_sectors", ".bc", "TAC"]
    assert result[0]["TAC"] is site
    assert acquired.acquisition == AcquisitionFacts()


def test_retrieval_builds_modern_record_without_legacy_adapters(monkeypatch):
    """Fresh retrieval retains aligned selectors and identities without fp_all."""
    from types import SimpleNamespace

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
    monkeypatch.setattr(get_data, "add_obs_error", lambda *a, **kw: None)
    monkeypatch.setattr(RhimeMergedData, "from_legacy_fp_all", lambda *a, **kw: pytest.fail("used fp_all"))
    monkeypatch.setattr(RhimeMergedData, "to_legacy_fp_all", lambda *a, **kw: pytest.fail("used fp_all"))
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
