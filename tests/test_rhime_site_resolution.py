"""Site shorthand is translated once before shared retrieval executes."""

from dataclasses import fields

import pytest

from openghg_inversions.inversion_data import SiteOptions, _site_options, acquisition, get_data


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
        "flux_sources": ["total"],
        "use_bc": False,
    }


def test_public_site_shorthand_expands_once_before_retrieval(monkeypatch):
    original = SiteOptions.from_inputs
    resolved = []

    def resolve(cls, **kwargs):
        options = original(**kwargs)
        resolved.append(options)
        return options

    def stop(**kwargs):
        raise RuntimeError("retrieval reached")

    monkeypatch.setattr(SiteOptions, "from_inputs", classmethod(resolve))
    monkeypatch.setattr(get_data, "get_flux_data", stop)
    kwargs = {**_site_inputs(), **_data_inputs()}
    with pytest.raises(RuntimeError, match="retrieval reached"):
        kwargs["emissions_name"] = kwargs.pop("flux_sources")
        get_data.retrieve_inversion_data(**kwargs)
    assert len(resolved) == 1
    assert resolved[0].sites == ("TAC", "MHD")
    assert resolved[0].averaging_period == ("1h", "1h")
    assert resolved[0].time_resolved == (None, None)


def test_canonical_acquisition_and_retrieval_never_expand_selectors(monkeypatch):
    options = SiteOptions.from_inputs(**_site_inputs())

    def fail(*args, **kwargs):
        raise AssertionError("canonical selectors must not be expanded")

    def stop(**kwargs):
        raise RuntimeError("retrieval reached")

    monkeypatch.setattr(SiteOptions, "from_inputs", classmethod(fail))
    monkeypatch.setattr(_site_options, "expand_site_option", fail)
    monkeypatch.setattr(get_data, "get_flux_data", stop)
    with pytest.raises(RuntimeError, match="retrieval reached"):
        acquisition.RhimeMergedData.from_options(site_options=options, **_data_inputs())


def test_site_options_public_factory_resolved_constructor_and_selection():

    options = SiteOptions.from_inputs(**_site_inputs())
    assert options == SiteOptions.from_inputs(
        **{**_site_inputs(), "averaging_period": ["1h", "1h"]}
    )
    assert SiteOptions is _site_options.SiteOptions
    values = {field.name: getattr(options, field.name) for field in fields(options)}
    assert SiteOptions(**values) == options
    selected = options.retain_sites(["mhd"], context="test")
    assert selected.sites == ("MHD",)
    assert selected.averaging_period == ("1h",)
    assert options.sites == ("TAC", "MHD")
    with pytest.raises(ValueError, match="same length"):
        SiteOptions(**{**values, "averaging_period": ("1h",)})


def test_site_options_preserve_unspecified_averaging_period():
    options = SiteOptions.from_inputs(sites=["TAC", "MHD"], averaging_period=None)
    assert options.averaging_period == (None, None)
