"""Public model selection preserves concrete recipe contracts."""

import pytest

from openghg_inversions.cli import build_parser, main, rhime_main
from openghg_inversions.inversion_data import MergedData, PreparedInputs, RhimeMergedData, RhimePreparedInputs
from openghg_inversions.rhime import (
    RhimeConfig,
    assemble_inputs,
    assemble_rhime_inputs,
    run,
    run_multisector,
    run_nested,
    run_rhime,
    run_rhime_multisector,
    run_rhime_nested,
    run_standard,
    with_prepared_rhime_sites,
    with_prepared_sites,
)


def test_established_imports_are_same_implementations():
    assert run_rhime is run_standard
    assert run_rhime_multisector is run_multisector
    assert run_rhime_nested is run_nested
    assert RhimeMergedData is MergedData
    assert RhimePreparedInputs is PreparedInputs
    assert assemble_rhime_inputs is assemble_inputs
    assert with_prepared_rhime_sites is with_prepared_sites


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unknown model 'missing'"):
        run(model="missing")


@pytest.mark.parametrize("model", ["co2", "co2_cached_sigma", "co2_o2", "co2_o2_cached_sigma"])
def test_carbon_recipes_require_their_actual_prepared_inputs(model):
    with pytest.raises(TypeError, match="prepared_inputs"):
        run(model=model)
    with pytest.raises(TypeError, match="config_file"):
        run(model=model, config_file="unsupported.ini")


@pytest.mark.parametrize("model", ["standard", "multisector"])
def test_selector_preserves_real_recipe_configuration_validation(model):
    config = RhimeConfig.from_params(
        dict(species="ch4", domain="EUROPE", sites=["TAC"], averaging_period="1h",
             start_date="2019-01-01", end_date="2019-02-01", output_name="api",
             flux_sources=["inventory", "ocean"] if model == "multisector" else ["inventory"],
             output_format="none"),
        multisector=model == "multisector",
    )
    with pytest.raises(ValueError, match="either resolved"):
        run(model=model, config=config, config_file="ambiguous.ini")


@pytest.mark.parametrize("entrypoint", [main, rhime_main])
@pytest.mark.parametrize("model", ["standard", "multisector", "nested"])
def test_both_executables_select_models_and_preserve_overrides(monkeypatch, entrypoint, model):
    calls = []
    monkeypatch.setattr("openghg_inversions.rhime.run", lambda **kw: calls.append(kw))
    entrypoint(["run", "2020-01-01", "2020-02-01", "-c", "run.ini", "--model", model,
                "--kwargs", '{"draws": 10}'])
    assert calls == [dict(model=model, config_file="run.ini", start_date="2020-01-01",
                          end_date="2020-02-01", draws=10)]


def test_cli_default_and_prepared_only_model_boundary():
    parser = build_parser(prog="rhime")
    assert parser.parse_args(["run", "-c", "run.ini"]).model == "standard"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "-c", "run.ini", "--model", "co2"])
