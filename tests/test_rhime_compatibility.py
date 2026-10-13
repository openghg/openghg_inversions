"""INI decoding and legacy translation boundaries."""

import os
import subprocess
import sys
import warnings
from pathlib import Path

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
    assert result["output_path"] == Path.cwd() / "final"
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


def test_output_alias_and_legacy_only_adapter_import():
    import openghg_inversions.rhime as rhime

    assert not hasattr(rhime, "params_from_config")
    with pytest.warns(DeprecationWarning, match="output_format"):
        assert translate_rhime_aliases({"output_format": "HBMCMC"}) == {"output_format": "legacy"}


def test_dictionary_resource_translation_respects_raw_mode_and_winning_overrides(tmp_path, monkeypatch):
    path = tmp_path / "legacy.ini"
    path.write_text('[RUN]\ndomain = "EUROPE"\nbc_basis_case = "missing"\n'
                    'bc_basis_directory = "absent"\nfp_basis_case = "missing"\nbasis_directory = "absent"\n')
    overrides = {"bc_basis_case": "NESW", "fp_basis_case": None}
    original = dict(overrides)

    def no_glob(*args, **kwargs):
        pytest.fail("superseded missing cases must not be discovered")

    monkeypatch.setattr(Path, "glob", no_glob)
    with pytest.warns(DeprecationWarning):
        raw = params_from_config(path, normalise=False, extra_kwargs=overrides)
    assert raw["bc_basis_case"] == "NESW"
    assert raw["bc_basis_directory"] == "absent"
    with pytest.warns(DeprecationWarning):
        normalized = params_from_config(path, extra_kwargs=overrides)
    assert normalized == {"domain": "EUROPE"}
    assert overrides == original


def test_dictionary_resource_translation_selects_explicit_files(tmp_path):
    path = tmp_path / "legacy.ini"
    domain_dir = tmp_path / "EUROPE"
    domain_dir.mkdir()
    basis_file = domain_dir / "custom_EUROPE_2019.nc"
    basis_file.touch()  # Translation selects paths; loading belongs to the runner.
    path.write_text(f'[RUN]\ndomain = "EUROPE"\nbc_basis_case = "custom"\n'
                    f'bc_basis_directory = "{tmp_path}"\n')
    with pytest.warns(DeprecationWarning):
        normalized = params_from_config(path)
    assert normalized == {"domain": "EUROPE", "bc_basis_paths": (basis_file,)}


def test_dictionary_removed_country_directory_is_an_error(tmp_path):
    path = tmp_path / "legacy.ini"
    path.write_text('[RUN]\ncountry_directory = "old"\n')
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="country_file.*landsea_path.*outer_regions_path"):
        params_from_config(path)
