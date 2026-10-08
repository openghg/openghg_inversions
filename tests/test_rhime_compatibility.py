"""INI decoding and legacy translation boundaries."""

import os
import subprocess
import sys
import warnings

import pytest

from openghg_inversions.hbmcmc.compatibility import (
    fixedbasis_params_to_rhime,
    params_from_config,
    translate_rhime_aliases,
)
from openghg_inversions.rhime.ini import read_rhime_ini


def test_reader_preserves_file_names_values_and_first_occurrence(tmp_path):
    path = tmp_path / "rhime.ini"
    path.write_text(
        '[FIRST]\noutputname = "first"\ndraws = "7"\n'
        'use_tracer = False\nspecial_recipe = {"x": [1, 2]}\n'
        '[SECOND]\noutputname = "second"\n'
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoded = read_rhime_ini(path)
    assert not caught
    assert decoded == {
        "outputname": "first",
        "draws": "7",
        "use_tracer": False,
        "special_recipe": {"x": [1, 2]},
    }


def test_dictionary_adapter_applies_overrides_before_normalizing(tmp_path):
    path = tmp_path / "rhime.ini"
    path.write_text('[RUN]\noutputpath = "old"\ndraws = "7"\nxprior = "invalid"\n')
    with pytest.warns(DeprecationWarning):
        decoded = params_from_config(path, normalise=False)
    assert decoded["draws"] == "7"
    with pytest.warns(DeprecationWarning):
        result = params_from_config(
            path,
            output_path="cli",
            start_date="2020-01-01",
            extra_kwargs={"output_path": "final", "x_prior": {"pdf": "normal"}},
        )
    assert result["output_path"] == "final"
    assert result["start_date"] == "2020-01-01"
    assert result["draws"] == 7
    assert result["x_prior"] == {"pdf": "normal"}
    assert "xprior" not in result


def test_alias_warnings_are_conditional_and_canonical_values_win():
    canonical = {"output_name": "run", "draws": 7, "output_format": "legacy"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert translate_rhime_aliases(canonical) == canonical
        fixedbasis_params_to_rhime(canonical)
    assert not caught
    raw = {**canonical, "outputname": "old", "use_tracer": False}
    with pytest.warns(DeprecationWarning) as caught:
        translated = translate_rhime_aliases(raw)
    assert len(caught) == 2
    assert translated == canonical
    assert raw["outputname"] == "old"
    with pytest.raises(ValueError, match="use_tracer=True"):
        translate_rhime_aliases({"use_tracer": True})


def test_fixedbasis_translation_preserves_borrowed_prior():
    prior = {"pdf": "lognormal", "mu": 0, "sigma": 1}
    with pytest.warns(DeprecationWarning):
        translated = fixedbasis_params_to_rhime(
            {"xprior": prior, "reparameterise_log_normal": True, "nit": "7"}
        )
    assert translated["x_prior"] == {**prior, "reparameterise": True}
    assert "reparameterise" not in prior
    assert translated["draws"] == "7"  # Canonical coercion belongs to the resolver.
    assert translated["output_filename_convention"] == "legacy"
    assert translated["save_inversion_output"] is False


def test_compatibility_import_and_translation_do_not_load_runners():
    code = """
import sys
from openghg_inversions.hbmcmc.compatibility import fixedbasis_params_to_rhime
assert fixedbasis_params_to_rhime({"draws": 7})["draws"] == 7
assert "openghg_inversions.rhime" not in sys.modules
assert "openghg_inversions.hbmcmc.run_hbmcmc" not in sys.modules
assert "pymc" not in sys.modules
assert "pytensor" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True)


def test_output_alias_and_public_adapter_import():
    from openghg_inversions.rhime import params_from_config as public_adapter

    assert public_adapter is params_from_config
    with pytest.warns(DeprecationWarning, match="output_format"):
        assert translate_rhime_aliases({"output_format": "HBMCMC"}) == {"output_format": "legacy"}
