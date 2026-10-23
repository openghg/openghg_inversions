"""Unsupported generic tracer requests fail at public boundaries before I/O."""

from types import SimpleNamespace

import pytest

from openghg_inversions.rhime import run_standard, run_rhime_multisector
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.hbmcmc.compatibility import params_from_config
from openghg_inversions.hbmcmc.compatibility import translate_rhime_aliases


@pytest.mark.parametrize("multisector", [False, True])
@pytest.mark.parametrize("value", [False, True])
def test_option_resolution_rejects_tracer_before_required_options(multisector, value):
    with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
        RhimeConfig.from_params(params={"use_tracer": value}, multisector=multisector)


@pytest.mark.parametrize("tracer_options", [{}, {"use_tracer": False}])
def test_false_or_omitted_tracer_is_accepted_during_option_normalization(tracer_options):
    assert translate_rhime_aliases(tracer_options) == {}


@pytest.mark.parametrize(
    ("configured", "override", "effective"),
    [(True, None, True), (False, None, False), (True, False, False), (False, True, True)],
)
def test_ini_tracer_option_respects_overrides_before_normalization(tmp_path, configured, override, effective):
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(f"[RHIME.OPTIONS]\nuse_tracer = {configured}\n", encoding="utf-8")
    assert params_from_config(config_file, normalise=False)["use_tracer"] is configured
    extra_kwargs = {} if override is None else {"use_tracer": override}
    if effective:
        with pytest.raises(ValueError, match="use_tracer=True.*not supported"):
            params_from_config(config_file, extra_kwargs=extra_kwargs)
    else:
        assert "use_tracer" not in params_from_config(config_file, extra_kwargs=extra_kwargs)


@pytest.mark.parametrize("multisector", [False, True])
def test_omitted_tracer_is_valid_after_resolution(multisector):
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
    from openghg_inversions.rhime._runner_compat import translate_rhime_aliases as runner_options

    config = RhimeConfig.from_params(params=runner_options(params), multisector=multisector)
    assert not hasattr(config, "use_tracer")
    assert params == original


@pytest.mark.parametrize("runner", [run_standard, run_rhime_multisector])
@pytest.mark.parametrize("supplied_merged_data", [False, True])
@pytest.mark.parametrize("value", [False, True])
def test_rhime_runner_rejects_tracer_before_acquisition(runner, supplied_merged_data, value):
    merged = SimpleNamespace(fp_all={}) if supplied_merged_data else None
    with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
        runner(use_tracer=value, merged_data=merged)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("source", ["overrides", "ini", "json"])
def test_modern_loader_rejects_explicit_tracer_options(tmp_path, value, grouped, source):
    import json
    from openghg_inversions.rhime.requests import load_params

    params = {"data": {"use_tracer": value}} if grouped else {"use_tracer": value}
    kwargs = {"overrides": params}
    if source == "ini":
        path = tmp_path / "rhime.ini"
        path.write_text(f"[RHIME.OPTIONS]\nuse_tracer = {value}\n")
        kwargs = {"config_file": path}
    elif source == "json":
        path = tmp_path / "rhime.json"
        path.write_text(json.dumps(params))
        kwargs = {"params_file": path}
    with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
        load_params(**kwargs)


@pytest.mark.parametrize("value", [False, True])
def test_modern_alias_translation_preserves_tracer_for_validation(value):
    from openghg_inversions.rhime._runner_compat import translate_rhime_aliases

    assert translate_rhime_aliases({"use_tracer": value}) == {"use_tracer": value}
