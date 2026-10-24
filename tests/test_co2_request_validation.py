"""CO2 request readers own names; scientific components resolve their sections."""

from importlib import import_module

import pytest

from openghg_inversions.rhime.co2 import configuration, execution, co2_o2_configuration
from openghg_inversions.rhime.co2.co2_o2_outputs import resolve_co2_o2_output


def test_full_co2_request_names_are_checked_once_before_missing_preparation(monkeypatch):
    original = configuration._validate_section_option_names
    calls = []

    def check(requests, sections):
        calls.append(tuple(requests))
        original(calls[-1], sections)

    monkeypatch.setattr(configuration, "_validate_section_option_names", check)
    with pytest.raises(ValueError, match="Pass prepared inputs"):
        execution.run_co2(params={"likelihood": {"kind": "additive_sigma"}})
    assert len(calls) == 1


@pytest.mark.parametrize("family", ["co2", "co2_o2"])
def test_component_parsers_do_not_repeat_whole_request_name_validation(monkeypatch, family):
    def unexpected(*args, **kwargs):
        pytest.fail("a consuming component rescanned the whole request")

    monkeypatch.setattr(configuration, "_validate_section_option_names", unexpected)
    monkeypatch.setattr(co2_o2_configuration, "_validate_section_option_names", unexpected)
    if family == "co2":
        preparation = configuration.Co2PreparationConfig
        model = configuration.Co2ModelConfig
        seed = configuration.resolve_co2_prior_seed
        output = execution.resolve_co2_output
        choices = {"likelihood": {"kind": "additive_sigma", "no_model_error": True}}
    else:
        preparation = co2_o2_configuration.Co2O2PreparationConfig
        model = co2_o2_configuration.Co2O2ModelConfig
        seed = co2_o2_configuration.resolve_co2_o2_prior_seed
        output = resolve_co2_o2_output
        choices = {"channels": {"co2": {"units": "ppm"}, "o2": {"units": "ppm"}}}
    assert preparation.from_params({"outputs": {"unconsumed_option": object()}}).prepared_path is None
    model.from_params({**choices, "outputs": {"unconsumed_option": object()}})
    assert seed({"sampling": {"random_seed": 19, "draws": "inactive-invalid"}}) == 19
    assert output({"outputs": {"output_format": "none"}, "sampling": {"unconsumed_option": object()}})["output"].output_format == "none"


@pytest.mark.parametrize("family", ["co2", "co2_o2"])
@pytest.mark.parametrize("phase", ["sample", "prior_predictive"])
def test_raw_handoff_requests_validate_unknown_inactive_names_before_artifact_io(monkeypatch, tmp_path, family, phase):
    handoffs = import_module("openghg_inversions.rhime.co2." + (
        "handoffs" if family == "co2" else "co2_o2_handoffs"
    ))
    monkeypatch.setattr(handoffs, "load_preparation", lambda *a, **kw: pytest.fail("opened scientific payload"))
    with pytest.raises(ValueError, match="Unknown configuration option.*outputs.unconsumed_option"):
        getattr(handoffs, phase)(
            "missing.json", model={"outputs": {"unconsumed_option": False}}, output_dir=tmp_path / "unopened",
        )
    assert not (tmp_path / "unopened").exists()


@pytest.mark.parametrize("family", ["co2", "co2_o2"])
@pytest.mark.parametrize("phase", ["prepare", "sample", "prior-predictive", "postprocess"])
def test_cli_co2_request_names_have_one_outer_owner(monkeypatch, tmp_path, family, phase):
    import json
    from openghg_inversions.cli import main

    owners = configuration if family == "co2" else co2_o2_configuration
    original = owners._validate_section_option_names
    calls = []

    def check(requests, sections):
        calls.append(tuple(requests))
        original(calls[-1], sections)

    monkeypatch.setattr(owners, "_validate_section_option_names", check)
    handoffs = import_module("openghg_inversions.rhime.co2." + (
        "handoffs" if family == "co2" else "co2_o2_handoffs"
    ))
    model_type = configuration.Co2ModelConfig if family == "co2" else co2_o2_configuration.Co2O2ModelConfig

    def execute(*args, **kwargs):
        if phase in {"sample", "prior-predictive"}:
            assert isinstance(kwargs["model"], model_type)
        return {"status": "pass"}

    monkeypatch.setattr(handoffs, phase.replace("-", "_"), execute)
    options = {
        "prepared_inputs": {"path": "unopened.nc"},
        "outputs": {"output_format": "none"},
    }
    if family == "co2":
        options["likelihood"] = {"kind": "additive_sigma", "no_model_error": True}
    else:
        options["channels"] = {"co2": {"units": "ppm"}, "o2": {"units": "ppm"}}
    path = tmp_path / "choices.json"
    path.write_text(json.dumps(options))
    arguments = [phase, "--model", family, "--params-file", str(path), "--output-dir", str(tmp_path / "unopened")]
    if phase != "prepare":
        stage = "sample" if phase == "postprocess" else "prepare"
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"schema": f"openghg_inversions.{family}.{stage}"}))
        arguments += ["--sample-manifest" if phase == "postprocess" else "--preparation-manifest", str(manifest)]
    main(arguments)
    assert len(calls) == 1
    assert not (tmp_path / "unopened").exists()
