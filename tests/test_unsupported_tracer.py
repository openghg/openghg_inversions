"""Unsupported generic tracer requests fail at public boundaries before I/O."""

from types import SimpleNamespace
import warnings

import pytest

from openghg_inversions.rhime import run_standard, run_rhime_multisector
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.requests import load_params
from openghg_inversions.hbmcmc.compatibility import params_from_config
from openghg_inversions.hbmcmc.compatibility import translate_rhime_aliases as legacy_options
from openghg_inversions.rhime._runner_compat import translate_rhime_aliases as runner_options


@pytest.mark.parametrize("multisector", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_option_resolution_rejects_tracer_before_required_options(multisector, enabled):
    with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
        RhimeConfig.from_params(params={"use_tracer": enabled}, multisector=multisector)


@pytest.mark.parametrize("translate", [legacy_options, runner_options])
@pytest.mark.parametrize("options", [{}, {"use_tracer": False}, {"use_tracer": True}])
def test_alias_translation_leaves_unsupported_options_for_validation(translate, options):
    original = options.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert translate(options) == original
    assert not caught
    assert options == original


@pytest.mark.parametrize("configured,override", [(True, None), (False, None), (True, False), (False, True)])
def test_ini_tracer_option_is_rejected_after_overrides(tmp_path, configured, override):
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(f"[RHIME.OPTIONS]\nuse_tracer = {configured}\n", encoding="utf-8")
    assert params_from_config(config_file, normalise=False)["use_tracer"] is configured
    extra_kwargs = {} if override is None else {"use_tracer": override}
    legacy_params = params_from_config(config_file, extra_kwargs=extra_kwargs)
    assert legacy_params["use_tracer"] is (configured if override is None else override)
    with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
        RhimeConfig.from_params(legacy_params, multisector=False)
    with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
        load_params(config_file=config_file, overrides=extra_kwargs)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("group", [None, "data", "model"])
def test_request_loader_rejects_flat_and_grouped_tracer_options(enabled, group):
    options = {"use_tracer": enabled}
    if group is not None:
        options = {group: options}
    with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
        load_params(overrides=options)


@pytest.mark.parametrize("multisector", [False, True])
def test_omitted_tracer_is_valid_and_not_forwarded_after_resolution(multisector):
    params = {
        "species": "ch4",
        "sites": ["TAC"],
        "domain": "EUROPE",
        "averaging_period": "1h",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "output_name": "test",
        "output_format": "none",
        "flux_sources": ["anthro", "natural"] if multisector else ["total"],
    }
    original = params.copy()
    config = RhimeConfig.from_params(params=load_params(overrides=params), multisector=multisector)
    assert not hasattr(config, "use_tracer")
    assert params == original


@pytest.mark.parametrize("runner", [run_standard, run_rhime_multisector])
@pytest.mark.parametrize("supplied_merged_data", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_rhime_runner_rejects_tracer_before_acquisition(runner, supplied_merged_data, enabled):
    merged = SimpleNamespace(fp_all={}) if supplied_merged_data else None
    with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
        runner(use_tracer=enabled, merged_data=merged)
