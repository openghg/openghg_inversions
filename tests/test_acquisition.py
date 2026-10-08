"""Acquisition factories retain selectors and make cache failures explicit."""

import inspect
import warnings

import dask.array as da
import pytest
import xarray as xr
from dask.callbacks import Callback

from openghg_inversions.inversion_data import RhimeMergedData, SiteOptions
from openghg_inversions.inversion_data import acquisition, get_data
from openghg_inversions.inversion_data._site_options import convert_to_list


def test_legacy_retrieval_alias_preserves_signature_docs_and_forwarding(monkeypatch):
    modern = get_data.retrieve_inversion_data
    legacy = get_data.data_processing_surface_notracer
    assert inspect.signature(legacy) == inspect.signature(modern)
    assert legacy.__doc__ == modern.__doc__
    assert get_data.convert_to_list is convert_to_list
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
        assert kwargs["sites"] == ["TAC", "MHD", "RGL"]
        assert kwargs["save_merged_data"] is False
        assert kwargs["time_resolved"] == [None, False, True]
        return {"MHD": dataset, "RGL": dataset}, ["RGL", "MHD"], [], [], [], []

    monkeypatch.setattr(acquisition, "retrieve_inversion_data", retrieve)
    monkeypatch.setattr(acquisition, "load_merged_data", lambda *a, **kw: pytest.fail("fresh must not load"))
    with Callback(pretask=lambda *args: pytest.fail("factory computed borrowed observations")):
        result = RhimeMergedData.from_options(
            species="ch4",
            site_options=options,
            domain="EUROPE",
            start_date="2020-01-01",
            end_date="2020-02-01",
            output_name="test",
            flux_sources=["inventory"],
        )
    assert result.site_options == options.select_indices([2, 1])
    assert result.fp_all["RGL"] is dataset
    assert options.sites == ("TAC", "MHD", "RGL")


@pytest.mark.parametrize("artifact", ["absent.nc", "corrupt.nc"])
def test_missing_or_corrupt_artifact_never_retrieves(tmp_path, monkeypatch, artifact):
    if artifact == "corrupt.nc":
        (tmp_path / artifact).write_text("not a netCDF file")
    monkeypatch.setattr(acquisition, "retrieve_inversion_data", lambda **kw: pytest.fail("load retrieved"))
    options = SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h")
    with pytest.raises((ValueError, OSError)):
        RhimeMergedData.load(tmp_path, site_options=options, merged_data_name=artifact)


def test_supplied_handoff_bypasses_all_acquisition_and_saving(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("supplied data must bypass acquisition and writes")

    monkeypatch.setattr(RhimeMergedData, "load", fail)
    monkeypatch.setattr(RhimeMergedData, "from_options", fail)
    monkeypatch.setattr(RhimeMergedData, "save", fail)
    supplied = RhimeMergedData(
        {"TAC": xr.Dataset()}, SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h")
    )
    assert (
        acquisition.retrieve_or_reload_rhime_data(
            {"reload_merged_data": True, "save_merged_data": True},
            multisector=False,
            merged_data=supplied,
        )
        is supplied
    )


def test_current_codec_load_requires_explicit_selectors(merged_data_dir, merged_data_file_name, tmp_path):
    options = SiteOptions.from_inputs(sites=["TAC", "MHD"], averaging_period=["1h", "2h"])
    merged = RhimeMergedData.load(
        merged_data_dir,
        site_options=options,
        merged_data_name=merged_data_file_name,
    )
    assert merged.sites == ("TAC",)
    merged.save(tmp_path, merged_data_name="roundtrip.nc")
    reopened = RhimeMergedData.load(tmp_path, site_options=options, merged_data_name="roundtrip.nc")
    assert reopened.site_options == merged.site_options
    xr.testing.assert_equal(reopened.fp_all["TAC"], merged.fp_all["TAC"])
