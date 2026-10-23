"""Every explicit tracer option fails ordinary validation before scientific work."""

import json
import warnings
from types import SimpleNamespace

import pytest

from openghg_inversions.rhime import run_standard, run_multisector, run_nested
from openghg_inversions.rhime.ini import read_rhime_ini
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.requests import load_params


@pytest.mark.parametrize("value", [True, False, None, 0, "false"])
@pytest.mark.parametrize("multisector", [False, True])
def test_option_resolution_rejects_every_tracer_value_before_required_options(value, multisector):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
            RhimeConfig.from_params(params={"use_tracer": value}, multisector=multisector)


@pytest.mark.parametrize("value", [True, False, None, 0, "false"])
@pytest.mark.parametrize("group", [None, "data", "model", "sampler", "output"])
@pytest.mark.parametrize("phase_groups", [None, {"data"}, {"model", "sampler"}, {"output"}])
def test_request_loader_rejects_flat_grouped_and_saved_phase_tracer(value, group, phase_groups):
    params = {"use_tracer": value}
    if group is not None:
        params = {group: params}
    with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
        load_params(overrides=params, phase_groups=phase_groups)


@pytest.mark.parametrize("source_kind", ["ini", "json"])
@pytest.mark.parametrize("configured", [True, False, None])
@pytest.mark.parametrize("override", [{}, {"use_tracer": False}, {"use_tracer": True}])
def test_file_tracer_is_unsupported_even_with_an_override(tmp_path, source_kind, configured, override):
    path = tmp_path / f"rhime.{source_kind}"
    path.write_text(json.dumps({"use_tracer": configured}) if source_kind == "json" else
                    f"[RHIME.OPTIONS]\nuse_tracer = {configured!r}\n", encoding="utf-8")
    source = {"params_file" if source_kind == "json" else "config_file": path}
    with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
        load_params(**source, overrides=override)


@pytest.mark.parametrize("runner", [run_standard, run_multisector, run_nested])
@pytest.mark.parametrize("value", [True, False, None, 0, "false"])
@pytest.mark.parametrize("route", ["fresh", "prepared", "resolved"])
def test_runner_rejects_tracer_before_acquisition_or_supplied_inputs(runner, value, route):
    supplied = {} if route == "fresh" else {"prepared" if route == "prepared" else "config": SimpleNamespace()}
    with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
        runner(use_tracer=value, **supplied)


@pytest.mark.parametrize("runner", [run_standard, run_multisector])
@pytest.mark.parametrize("value", [True, False, None])
def test_runner_rejects_tracer_before_supplied_merged_data(runner, value):
    with pytest.raises(ValueError, match="Unsupported RHIME parameter.*use_tracer"):
        runner(use_tracer=value, merged_data=SimpleNamespace())


@pytest.mark.parametrize("multisector", [False, True])
def test_omitted_tracer_keeps_canonical_configuration(multisector):
    params = {
        "species": "ch4", "sites": ["TAC"], "domain": "EUROPE", "averaging_period": "1h",
        "start_date": "2019-01-01", "end_date": "2019-01-02", "output_name": "test",
        "output_format": "none", "flux_sources": ["anthro", "natural"] if multisector else ["total"],
    }
    original = params.copy()
    config = RhimeConfig.from_params(params=load_params(overrides=params), multisector=multisector)
    assert not hasattr(config, "use_tracer")
    assert params == original


def test_ini_reader_still_decodes_historical_inputs_without_translating(tmp_path):
    path = tmp_path / "rhime.ini"
    path.write_text('[FIRST]\noutputname = "first"\ndraws = "7"\n'
                    'use_tracer = False\nspecial_recipe = {"x": [1, 2]}\n'
                    '[SECOND]\noutputname = "second"\n', encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert read_rhime_ini(path) == {
            "outputname": "first", "draws": "7", "use_tracer": False, "special_recipe": {"x": [1, 2]},
        }
    assert not caught


@pytest.mark.parametrize("family", ["co2", "co2_o2"])
@pytest.mark.parametrize("phase", ["prepare", "sample", "postprocess"])
@pytest.mark.parametrize("value", [False, True, None])
@pytest.mark.parametrize("section", [None, "prepared_inputs", "likelihood", "sampling", "outputs"])
def test_co2_phases_reject_tracer_in_active_and_inactive_sections(tmp_path, family, phase, value, section):
    from openghg_inversions.cli import main

    options = {"use_tracer": value}
    if section is not None:
        options = {section: options}
    options_file = tmp_path / "choices.json"
    options_file.write_text(json.dumps(options), encoding="utf-8")
    arguments = [phase, "--model", family, "--params-file", str(options_file),
                 "--output-dir", str(tmp_path / "unopened")]
    if phase != "prepare":
        stage = "sample" if phase == "postprocess" else "prepare"
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"schema": f"openghg_inversions.{family}.{stage}"}), encoding="utf-8")
        arguments += ["--sample-manifest" if phase == "postprocess" else "--preparation-manifest", str(manifest)]
    with pytest.raises(ValueError, match="Unknown configuration option.*use_tracer"):
        main(arguments)
    assert not (tmp_path / "unopened").exists()


@pytest.mark.parametrize("family, section", [
    ("co2", "model"), ("co2", "model.boundary"), ("co2", "model.offset.prior"),
    ("co2_o2", "channels"), ("co2_o2", "channels.co2"),
    ("co2_o2", "channels.o2.boundary.activity"), ("co2_o2", "channels.co2.offset.prior"),
    ("co2", "outputs.paris_postprocessing_kwargs"), ("co2_o2", "outputs.paris_postprocessing_kwargs"),
])
@pytest.mark.parametrize("value", [False, True, None])
def test_co2_nested_option_names_are_checked_without_resolving_values(family, section, value):
    from openghg_inversions.rhime.co2.execution import load_co2_params
    from openghg_inversions.rhime.co2.co2_o2_configuration import load_co2_o2_params

    options = {"use_tracer": value}
    for name in reversed(section.split(".")):
        options = {name: options}
    loader = load_co2_params if family == "co2" else load_co2_o2_params
    with pytest.raises(ValueError, match="Unknown configuration option.*use_tracer"):
        loader(params=options)
    with pytest.raises(ValueError, match="Unknown configuration option.*use_tracer"):
        loader(params={}, overrides=options)
    # Replacing a whole section does not hide an explicitly unsupported source option.
    with pytest.raises(ValueError, match="Unknown configuration option.*use_tracer"):
        loader(params=options, overrides={section.split(".")[0]: None})


@pytest.mark.parametrize("family", ["co2", "co2_o2"])
def test_co2_request_names_do_not_reinterpret_scientific_labels_or_inactive_values(family):
    from openghg_inversions.rhime.co2.execution import load_co2_params
    from openghg_inversions.rhime.co2.co2_o2_configuration import load_co2_o2_params

    loader = load_co2_params if family == "co2" else load_co2_o2_params
    options = {
        "sampling": {"draws": "inactive-invalid"}, "prepared_inputs": {"path": 42},
        "likelihood": {"fixed_site_amplitudes": {"use_tracer": 0.5}},
        "outputs": {"output_format": "inactive-invalid"},
    }
    assert loader(params=options) == options


@pytest.mark.parametrize("value", [False, True, None])
@pytest.mark.parametrize("section", [None, "prepared_inputs", "likelihood", "sampling", "outputs"])
@pytest.mark.parametrize("family", ["co2", "co2_o2"])
def test_direct_co2_owners_reject_tracer_before_inputs_or_artifacts(family, section, value):
    from openghg_inversions.rhime.co2.configuration import Co2PreparationConfig, Co2ModelConfig
    from openghg_inversions.rhime.co2.execution import resolve_co2_output
    from openghg_inversions.rhime.co2.co2_o2_configuration import Co2O2PreparationConfig, Co2O2ModelConfig
    from openghg_inversions.rhime.co2.co2_o2_outputs import resolve_co2_o2_output

    owners = (Co2PreparationConfig.from_params, Co2ModelConfig.from_params, resolve_co2_output) if family == "co2" else (
        Co2O2PreparationConfig.from_params, Co2O2ModelConfig.from_params, resolve_co2_o2_output,
    )
    options = {"use_tracer": value}
    if section is not None:
        options = {section: options}
    for owner in owners:
        with pytest.raises(ValueError, match="Unknown configuration option.*use_tracer"):
            owner(options)


@pytest.mark.parametrize("value", [False, True, None])
def test_direct_modern_data_and_output_boundaries_reject_tracer(value):
    from openghg_inversions.inversion_data import MergedData, SiteOptions
    from openghg_inversions.rhime.params import DataConfig
    from openghg_inversions.rhime.specs import RhimeOutputSpec

    with pytest.raises(ValueError, match="Unsupported data parameter.*use_tracer"):
        DataConfig.from_params({"use_tracer": value})
    with pytest.raises(TypeError, match="unexpected keyword argument 'use_tracer'"):
        RhimeOutputSpec.from_params({"use_tracer": value}, multisector=False)
    with pytest.raises(TypeError, match="unexpected keyword argument 'use_tracer'"):
        MergedData.from_options(
            species="ch4", site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"),
            domain="EUROPE", start_date="2019-01-01", end_date="2019-01-02",
            flux_sources=["inventory"], use_tracer=value,
        )
