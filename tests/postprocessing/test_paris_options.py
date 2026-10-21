"""PARIS reporting options resolve at the owner of each output family."""

from dataclasses import FrozenInstanceError

import pytest

from openghg_inversions.postprocessing.make_paris_outputs import PARIS_LATEST_COUNTRIES
from openghg_inversions.postprocessing.paris_options import NestedParisOptions, ParisOptions
from openghg_inversions.rhime.co2.co2_outputs import _co2_paris_options


def test_family_defaults_and_explicit_none():
    standard = ParisOptions.from_params(None)
    nested = NestedParisOptions.from_params({})
    multi = ParisOptions.from_params({}, default_template_version="latest")
    co2 = _co2_paris_options(None)
    assert standard.template_version == nested.template_version == "legacy"
    assert multi.template_version == co2.template_version == "latest"
    assert standard.inversion_grid and multi.inversion_grid and not co2.inversion_grid
    assert standard.country_selections == multi.country_selections == PARIS_LATEST_COUNTRIES
    assert co2.country_selections is None
    params = {"country_selections": None}
    assert ParisOptions.from_params(params).country_selections is None
    assert params == {"country_selections": None}
    with pytest.raises(FrozenInstanceError):
        standard.report_mode = True


@pytest.mark.parametrize("key", ["domain", "country_file", "obs_avg_period", "unknown", "inner_country_file"])
def test_common_parser_rejects_unknown_and_recipe_owned_keys(key):
    with pytest.raises(ValueError, match=key):
        ParisOptions.from_params({key: None})


@pytest.mark.parametrize("params, message", [
    ({"template_version": "unknown"}, "template version"),
    ({"time_point": "end"}, "time_point"),
    ({"template_version": "latest", "time_point": "start"}, "midpoint"),
    ({"flux_frequency": "0h"}, "positive fixed duration"),
])
def test_validation_is_explicit_and_checks_scalar_choices(params, message):
    options = ParisOptions(**params)
    with pytest.raises(ValueError, match=message):
        options.validate()
    with pytest.raises(ValueError, match=message):
        ParisOptions.from_params(params)


def test_nested_parser_accepts_its_resources_and_validates_inner_file(tmp_path):
    inner = tmp_path / "inner.nc"
    options = NestedParisOptions(
        inner_country_file=inner, country_file_cache_dir=tmp_path / "cache",
    )
    with pytest.raises(FileNotFoundError):
        options.validate()
    inner.touch()
    options.validate()
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("params, message", [
    ({"template_version": "legacy"}, "template_version"),
    ({"time_point": "start"}, "midpoint"),
    ({"report_mode": True}, "report_mode"),
    ({"inversion_grid": True}, "inversion_grid"),
    ({"flux_frequency": "invalid"}, "positive fixed duration"),
])
def test_co2_keeps_its_product_restrictions(params, message):
    with pytest.raises(ValueError, match=message):
        _co2_paris_options(params)
