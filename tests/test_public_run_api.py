"""Public model selection preserves concrete recipe contracts."""

import pytest

from openghg_inversions.cli import build_parser, main, rhime_main
from openghg_inversions.inversion_data import MergedData, PreparedInputs
from openghg_inversions.rhime import (
    RhimeConfig,
    assemble_inputs,
    run,
    run_multisector,
    run_nested,
)


def test_canonical_imports_are_owned_implementations():
    from openghg_inversions.inversion_data import acquisition, prepared_inputs
    from openghg_inversions.rhime import multisector, nested, preparation

    assert run_multisector is multisector.run_multisector
    assert run_nested is nested.run_nested
    assert MergedData is acquisition.MergedData
    assert PreparedInputs is prepared_inputs.PreparedInputs
    assert assemble_inputs is preparation.assemble_inputs


@pytest.mark.parametrize("module_name,alias", [
    ("openghg_inversions.rhime", "run_rhime_multisector"),
    ("openghg_inversions.rhime.multisector", "run_rhime_multisector"),
    ("openghg_inversions.rhime", "run_rhime_nested"),
    ("openghg_inversions.rhime.nested", "run_rhime_nested"),
    ("openghg_inversions.rhime", "RhimeMergedData"),
    ("openghg_inversions.inversion_data", "RhimeMergedData"),
    ("openghg_inversions.inversion_data.acquisition", "RhimeMergedData"),
    ("openghg_inversions.inversion_data", "RhimePreparedInputs"),
    ("openghg_inversions.inversion_data.prepared_inputs", "RhimePreparedInputs"),
    ("openghg_inversions.rhime", "assemble_rhime_inputs"),
    ("openghg_inversions.rhime.preparation", "assemble_rhime_inputs"),
])
def test_retired_python_alias_is_absent(module_name, alias):
    import importlib

    module = importlib.import_module(module_name)
    assert not hasattr(module, alias)
    assert alias not in getattr(module, "__all__", [])




def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unknown model 'missing'"):
        run(model="missing")


@pytest.mark.parametrize("model", ["co2_o2", "co2_o2_cached_sigma"])
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
def test_cli_selects_models_and_preserves_overrides(monkeypatch, entrypoint, model):
    calls = []
    monkeypatch.setattr("openghg_inversions.rhime.run", lambda **kw: calls.append(kw))
    entrypoint(["run", "2020-01-01", "2020-02-01", "-c", "run.ini", "--model", model,
                "--kwargs", '{"draws": 10}'])
    assert calls == [dict(model=model, config_file="run.ini", start_date="2020-01-01",
                          end_date="2020-02-01", draws=10)]


@pytest.mark.parametrize("entrypoint", [main, rhime_main])
@pytest.mark.parametrize("command", [["run", "--model", "co2"]])
def test_configured_co2_cli_preserves_sections_and_run_overrides(monkeypatch, entrypoint, command):
    calls = []
    monkeypatch.setattr("openghg_inversions.rhime.co2.run_co2", lambda **kw: calls.append(kw))

    entrypoint([*command, "2020-01-01", "2020-02-01", "-c", "co2.toml",
                "--output-path", "products", "--kwargs", '{"sampling":{"draws":10}}'])
    assert calls == [dict(config_file="co2.toml", start_date="2020-01-01", end_date="2020-02-01",
                          output_path="products", overrides={"sampling": {"draws": 10}})]


def test_cli_default_and_prepared_only_model_boundary():
    parser = build_parser(prog="rhime")
    assert parser.parse_args(["run", "-c", "run.ini"]).model == "standard"
    assert parser.parse_args(["run", "-c", "co2.toml", "--model", "co2"]).model == "co2"
    for model in ("co2_cached_sigma", "co2_o2", "co2_o2_cached_sigma"):
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "-c", "run.ini", "--model", model])


def test_python_co2_selector_uses_the_canonical_product_runner(monkeypatch):
    calls = []
    output = object()

    def configured(**kwargs):
        calls.append(kwargs)
        return output

    monkeypatch.setattr("openghg_inversions.rhime.co2.run_co2", configured)
    assert run(model="co2", config_file="co2.toml", output_path="products") is output
    assert calls == [{"config_file": "co2.toml", "output_path": "products"}]


def test_removed_ordinary_runner_names_have_no_compatibility_aliases():
    from openghg_inversions.rhime import co2
    from openghg_inversions import rhime

    for name in ("run_rhime_co2", "build_rhime_co2", "run_co2_from_config", "run_rhime_co2_cached_sigma", "Co2RunSetup", "sample_co2_cached_prior_predictive"):
        assert not hasattr(co2, name)
        assert not hasattr(rhime, name)


def test_retired_run_envelopes_and_site_helpers_have_no_public_exports():
    from openghg_inversions import rhime

    for name in ("RhimeResult", "RhimeRunSpec", "with_prepared_sites", "with_prepared_rhime_sites"):
        assert not hasattr(rhime, name)
        assert name not in rhime.__all__
    assert not hasattr(RhimeConfig, "retained_run_spec")


def test_retired_cached_selector_has_no_compatibility_alias():
    with pytest.raises(ValueError, match="Unknown model 'co2_cached_sigma'"):
        run(model="co2_cached_sigma")
