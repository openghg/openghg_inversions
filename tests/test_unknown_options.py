"""Unknown option names fail ordinary validation at every boundary, before work.

Removed and renamed options need no tests of their own: once a name leaves the
supported options, these generic checks reject it. The gates compare names
only, so each case passes ``None``, the value most likely to be mistaken for
an omitted option. Axes are kept only where they reach a distinct boundary.
"""

import json
import warnings
from types import SimpleNamespace

import pytest

from openghg_inversions.rhime import run_standard, run_multisector, run_nested
from openghg_inversions.rhime.ini import read_rhime_ini
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.requests import load_params

UNKNOWN = "not_an_option"


@pytest.mark.parametrize("multisector", [False, True])
def test_option_resolution_rejects_unknown_names_before_required_options(multisector):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match=f"Unsupported RHIME parameter.*{UNKNOWN}"):
            RhimeConfig.from_params(params={UNKNOWN: None}, multisector=multisector)


@pytest.mark.parametrize("group", ["data", "model"])
def test_grouped_option_resolution_reports_the_qualified_name(group):
    with pytest.raises(ValueError, match=f"Unsupported grouped RHIME parameter.*{group}.{UNKNOWN}"):
        RhimeConfig.from_params(params={group: {UNKNOWN: None}}, multisector=False)


@pytest.mark.parametrize(
    ("group", "phase_groups"),
    [
        *((group, None) for group in (None, "data", "model", "sampler", "output")),
        *((None, phase_groups) for phase_groups in ({"data"}, {"model", "sampler"}, {"output"})),
    ],
)
def test_request_loader_rejects_flat_grouped_and_saved_phase_names(group, phase_groups):
    params = {UNKNOWN: None}
    if group is not None:
        params = {group: params}
    with pytest.raises(ValueError, match=f"Unsupported RHIME parameter.*{UNKNOWN}"):
        load_params(overrides=params, phase_groups=phase_groups)


@pytest.mark.parametrize("source_kind", ["ini", "json"])
@pytest.mark.parametrize("override", [{}, {UNKNOWN: False}])
def test_file_option_is_rejected_even_with_an_override(tmp_path, source_kind, override):
    path = tmp_path / f"rhime.{source_kind}"
    path.write_text(json.dumps({UNKNOWN: None}) if source_kind == "json" else
                    f"[RHIME.OPTIONS]\n{UNKNOWN} = None\n", encoding="utf-8")
    source = {"params_file" if source_kind == "json" else "config_file": path}
    with pytest.raises(ValueError, match=f"Unsupported RHIME parameter.*{UNKNOWN}"):
        load_params(**source, overrides=override)


@pytest.mark.parametrize("runner", [run_standard, run_multisector, run_nested])
@pytest.mark.parametrize("route", ["fresh", "prepared", "resolved"])
def test_runner_rejects_unknown_names_before_acquisition_or_supplied_inputs(runner, route):
    supplied = {} if route == "fresh" else {"prepared" if route == "prepared" else "config": SimpleNamespace()}
    expected = "either resolved" if route == "resolved" else f"Unsupported RHIME parameter.*{UNKNOWN}"
    with pytest.raises(ValueError, match=expected):
        runner(**{UNKNOWN: None}, **supplied)


@pytest.mark.parametrize("runner", [run_standard, run_multisector])
def test_runner_rejects_unknown_names_before_supplied_merged_data(runner):
    with pytest.raises(ValueError, match=f"Unsupported RHIME parameter.*{UNKNOWN}"):
        runner(**{UNKNOWN: None}, merged_data=SimpleNamespace())


def test_ini_reader_decodes_unknown_names_without_translating(tmp_path):
    path = tmp_path / "rhime.ini"
    path.write_text('[FIRST]\noutputname = "first"\ndraws = "7"\n'
                    f'{UNKNOWN} = False\nspecial_recipe = {{"x": [1, 2]}}\n'
                    '[SECOND]\noutputname = "second"\n', encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert read_rhime_ini(path) == {
            "outputname": "first", "draws": "7", UNKNOWN: False, "special_recipe": {"x": [1, 2]},
        }
    assert not caught


@pytest.mark.parametrize("family", ["co2", "co2_o2"])
@pytest.mark.parametrize("phase", ["prepare", "sample", "postprocess"])
@pytest.mark.parametrize("section", [None, "prepared_inputs", "likelihood", "sampling", "outputs"])
def test_co2_phases_reject_unknown_names_in_active_and_inactive_sections(tmp_path, family, phase, section):
    from openghg_inversions.cli import main

    options = {UNKNOWN: None}
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
    with pytest.raises(ValueError, match=f"Unknown configuration option.*{UNKNOWN}"):
        main(arguments)
    assert not (tmp_path / "unopened").exists()


@pytest.mark.parametrize("family, section", [
    ("co2", "model"), ("co2", "model.boundary"), ("co2", "model.offset.prior"),
    ("co2_o2", "channels"), ("co2_o2", "channels.co2"),
    ("co2_o2", "channels.o2.boundary.activity"), ("co2_o2", "channels.co2.offset.prior"),
    ("co2", "outputs.paris_postprocessing_kwargs"), ("co2_o2", "outputs.paris_postprocessing_kwargs"),
])
def test_co2_nested_option_names_are_checked_without_resolving_values(family, section):
    from openghg_inversions.rhime.co2.execution import load_co2_params
    from openghg_inversions.rhime.co2.co2_o2_configuration import load_co2_o2_params

    options = {UNKNOWN: None}
    for name in reversed(section.split(".")):
        options = {name: options}
    loader = load_co2_params if family == "co2" else load_co2_o2_params
    with pytest.raises(ValueError, match=f"Unknown configuration option.*{UNKNOWN}"):
        loader(params=options)
    with pytest.raises(ValueError, match=f"Unknown configuration option.*{UNKNOWN}"):
        loader(params={}, overrides=options)
    # Replacing a whole section does not hide an explicitly unsupported source option.
    with pytest.raises(ValueError, match=f"Unknown configuration option.*{UNKNOWN}"):
        loader(params=options, overrides={section.split(".")[0]: None})


@pytest.mark.parametrize("family", ["co2", "co2_o2"])
def test_co2_request_names_do_not_reinterpret_scientific_labels_or_inactive_values(family):
    from openghg_inversions.rhime.co2.execution import load_co2_params
    from openghg_inversions.rhime.co2.co2_o2_configuration import load_co2_o2_params

    loader = load_co2_params if family == "co2" else load_co2_o2_params
    options = {
        "sampling": {"draws": "inactive-invalid"}, "prepared_inputs": {"path": 42},
        "likelihood": {"fixed_site_amplitudes": {UNKNOWN: 0.5}},
        "outputs": {"output_format": "inactive-invalid"},
    }
    assert loader(params=options) == options


@pytest.mark.parametrize("family", ["co2", "co2_o2"])
def test_direct_co2_components_reject_unknown_names_in_their_own_sections(family):
    from openghg_inversions.rhime.co2.configuration import Co2PreparationConfig, Co2ModelConfig
    from openghg_inversions.rhime.co2.execution import resolve_co2_output
    from openghg_inversions.rhime.co2.co2_o2_configuration import Co2O2PreparationConfig, Co2O2ModelConfig
    from openghg_inversions.rhime.co2.co2_o2_outputs import resolve_co2_o2_output

    if family == "co2":
        model_options = {"model": {UNKNOWN: None}}
        owners = (Co2PreparationConfig.from_params, Co2ModelConfig.from_params, resolve_co2_output)
    else:
        model_options = {"channels": {"co2": {"units": "ppm", UNKNOWN: None}}}
        owners = (Co2O2PreparationConfig.from_params, Co2O2ModelConfig.from_params, resolve_co2_o2_output)
    for owner, options in zip(owners, (
        {"prepared_inputs": {UNKNOWN: None}}, model_options, {"outputs": {UNKNOWN: None}},
    ), strict=True):
        with pytest.raises(ValueError, match=f"option.*{UNKNOWN}"):
            owner(options)


def test_direct_data_and_output_boundaries_reject_unknown_names():
    from openghg_inversions.inversion_data import MergedData, SiteOptions
    from openghg_inversions.rhime.params import DataConfig
    from openghg_inversions.rhime.specs import RhimeOutputSpec

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match=f"Unsupported data parameter.*{UNKNOWN}"):
            DataConfig.from_params({UNKNOWN: None})
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{UNKNOWN}'"):
        RhimeOutputSpec.from_params({UNKNOWN: None}, multisector=False)
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{UNKNOWN}'"):
        MergedData.from_options(
            species="ch4", site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"),
            domain="EUROPE", start_date="2019-01-01", end_date="2019-01-02",
            flux_sources=["inventory"], **{UNKNOWN: None},
        )
