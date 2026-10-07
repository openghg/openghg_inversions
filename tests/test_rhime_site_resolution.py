"""Site shorthand is translated once before shared retrieval executes."""

import pytest
import xarray as xr

from openghg_inversions.inversion_data import acquisition, get_data, preparation


def _site_inputs():
    return {
        "sites": ["tac", "MHD"],
        "averaging_period": "1h",
        "inlet": None,
        "fp_height": None,
        "instrument": None,
        "platform": None,
        "obs_data_level": None,
        "met_model": None,
        "max_level": None,
        "time_resolved": None,
    }


def _data_inputs():
    return {
        "species": "ch4",
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "output_name": "resolved",
        "flux_sources": ["total"],
        "use_bc": False,
    }


@pytest.mark.parametrize("boundary", ["retrieval", "acquisition", "preparation"])
def test_public_site_shorthand_expands_once_before_retrieval(monkeypatch, boundary):
    original = acquisition._SiteOptions.from_inputs
    resolved = []

    def resolve(cls, **kwargs):
        options = original(**kwargs)
        resolved.append(options)
        return options

    def stop(**kwargs):
        raise RuntimeError("retrieval reached")

    monkeypatch.setattr(acquisition._SiteOptions, "from_inputs", classmethod(resolve))
    monkeypatch.setattr(get_data, "get_flux_data", stop)
    kwargs = {**_site_inputs(), **_data_inputs()}
    with pytest.raises(RuntimeError, match="retrieval reached"):
        if boundary == "retrieval":
            kwargs["emissions_name"] = kwargs.pop("flux_sources")
            get_data.data_processing_surface_notracer(**kwargs)
        elif boundary == "acquisition":
            acquisition._retrieve_or_reload_merged_data(**kwargs)
        else:
            preparation.prepare_rhime_inputs(**kwargs)
    assert len(resolved) == 1
    assert resolved[0].sites == ("TAC", "MHD")
    assert resolved[0].averaging_period == ("1h", "1h")
    assert resolved[0].time_resolved == (None, None)


def test_canonical_acquisition_and_retrieval_never_expand_selectors(monkeypatch):
    options = acquisition._SiteOptions.from_inputs(**_site_inputs())

    def fail(*args, **kwargs):
        raise AssertionError("canonical selectors must not be expanded")

    def stop(**kwargs):
        raise RuntimeError("retrieval reached")

    monkeypatch.setattr(acquisition._SiteOptions, "from_inputs", classmethod(fail))
    monkeypatch.setattr(get_data, "expand_site_option", fail)
    monkeypatch.setattr(acquisition, "expand_site_option", fail)
    monkeypatch.setattr(get_data, "get_flux_data", stop)
    with pytest.raises(RuntimeError, match="retrieval reached"):
        acquisition._retrieve_or_reload_merged_data_from_options(site_options=options, **_data_inputs())


def test_canonical_reload_selects_options_without_expansion(monkeypatch, tmp_path):
    options = acquisition._SiteOptions.from_inputs(**_site_inputs())

    def fail(*args, **kwargs):
        raise AssertionError("reload must not expand selectors")

    monkeypatch.setattr(acquisition._SiteOptions, "from_inputs", classmethod(fail))
    monkeypatch.setattr(acquisition, "load_merged_data", lambda *args: {"MHD": xr.Dataset()})
    merged = acquisition._retrieve_or_reload_merged_data_from_options(
        site_options=options,
        **_data_inputs(),
        reload_merged_data=True,
        merged_data_dir=str(tmp_path),
    )
    assert merged.site_options == options.select_indices([1])
    assert options.sites == ("TAC", "MHD")


def test_canonical_supplied_data_keeps_its_options_and_tracer_guard():
    requested = acquisition._SiteOptions.from_inputs(**_site_inputs())
    merged = acquisition.RhimeMergedData({}, requested.select_indices([1]))
    assert (
        acquisition.retrieve_or_reload_rhime_data_from_options(
            requested, {}, multisector=False, merged_data=merged
        )
        is merged
    )
    with pytest.raises(ValueError, match="use_tracer=True.*not supported"):
        acquisition.retrieve_or_reload_rhime_data_from_options(
            requested, {"use_tracer": True}, multisector=False, merged_data=merged
        )
