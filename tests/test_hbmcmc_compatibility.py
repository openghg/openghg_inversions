"""Resolved construction contains one warned legacy-spelling boundary."""

import subprocess
import sys
import warnings

import pytest

from openghg_inversions.hbmcmc import compatibility
from openghg_inversions.rhime import params as rhime_params


def _request(**overrides):
    return {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": "1h",
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-02-01",
        "output_name": "compatibility",
        "output_format": "none",
        "flux_sources": ["inventory"],
        **overrides,
    }


def test_translation_import_does_not_load_a_runner_or_scientific_backend():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import openghg_inversions.hbmcmc.compatibility; "
            "assert 'openghg_inversions.hbmcmc.run_hbmcmc' not in sys.modules; "
            "assert 'openghg_inversions.rhime.params' not in sys.modules; "
            "assert 'pymc' not in sys.modules",
        ],
        check=True,
    )


def test_factory_translates_once_and_canonical_coercion_is_silent(monkeypatch):
    calls = []

    def capture_translation(params):
        calls.append(params)
        return compatibility.translate_rhime_aliases(params)

    monkeypatch.setattr(rhime_params, "translate_rhime_aliases", capture_translation)
    params = _request(draws="12", output_format="NONE")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = rhime_params.RhimeConfig.from_params(params, multisector=False)

    assert calls == [params]
    assert resolved.sampler.draws == 12
    assert resolved.output.output_format == "none"
    assert not caught


@pytest.mark.parametrize("name", ["calculate_min_error", "reparameterise_log_normal", "mcmc_type"])
def test_factory_rejects_obsolete_fixedbasis_switches(name):
    with pytest.raises(ValueError, match=f"{name}.*not supported"):
        rhime_params.RhimeConfig.from_params(_request(**{name: None}), multisector=False)


def test_factory_warns_on_alias_removal_and_keeps_canonical_precedence():
    params = _request(outputname="ignored", outer_region_definition_file="outer.nc")
    with pytest.warns(DeprecationWarning) as caught:
        resolved = rhime_params.RhimeConfig.from_params(params, multisector=False)

    assert resolved.output_name == "compatibility"
    assert resolved.outer_regions_path == "outer.nc"
    assert len(caught) == 2
    assert params["outputname"] == "ignored"
    assert "outer_regions_path" not in params


def test_factory_accepts_legacy_names_and_warns_only_for_changed_inputs():
    params = _request(
        sigprior={"pdf": "halfnormal", "sigma": 2.0},
        output_format="hbmcmc",
        mismatch_model="pollution_event",
    )
    params["outputname"] = params.pop("output_name")
    params["output_path"] = "out"
    with pytest.warns(DeprecationWarning) as caught:
        resolved = rhime_params.RhimeConfig.from_params(params, multisector=False)

    assert len(caught) == 3
    assert resolved.output_name == "compatibility"
    assert resolved.output.output_format == "legacy"
    assert resolved.model.likelihood.sigma_prior == {"pdf": "halfnormal", "sigma": 2.0}
    assert "output_name" not in params


def test_dictionary_adapter_preserves_raw_and_normalized_modes(tmp_path):
    path = tmp_path / "request.ini"
    path.write_text('[RUN]\noutputname = "file"\ndraws = "invalid"\n', encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="params_from_config"):
        raw = rhime_params.params_from_config(path, normalise=False)
    assert raw == {"outputname": "file", "draws": "invalid"}

    with pytest.warns(DeprecationWarning) as caught:
        normalized = compatibility.params_from_config(
            path, start_date="2019-01-01", extra_kwargs={"draws": "17", "output_name": "override"}
        )

    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert any("outputname" in str(item.message) for item in caught)
    assert normalized == {"draws": 17, "start_date": "2019-01-01", "output_name": "override"}


def test_fixedbasis_aliases_warn_and_leave_value_coercion_to_factory():
    params = _request(nit="12", nchain=2, draws="17")
    with pytest.warns(DeprecationWarning) as caught:
        translated = compatibility.fixedbasis_params_to_rhime(params)

    assert len(caught) == 2
    assert translated["draws"] == "17"
    assert translated["chains"] == 2
    resolved = rhime_params.RhimeConfig.from_params(translated, multisector=False)
    assert resolved.sampler.draws == 17
    assert resolved.sampler.chains == 2
    assert params["nit"] == "12"
