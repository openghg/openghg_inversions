"""Public standard runner migration preserves CLI execution."""

from argparse import Namespace
import warnings
import pytest


@pytest.mark.parametrize("name", ["run_rhime", "build_standard_rhime_model", "build_standard_rhime_model_result"])
def test_removed_python_api_is_absent(name):
    import openghg_inversions.rhime as rhime

    assert not hasattr(rhime, name)
    with pytest.raises(AttributeError, match=name):
        getattr(rhime, name)


def test_canonical_cli_calls_standard_without_python_compatibility_warning(monkeypatch):
    from openghg_inversions.rhime import standard
    from openghg_inversions.cli import main

    calls = []
    monkeypatch.setattr(standard, "run_standard", lambda **kwargs: calls.append(kwargs))
    with warnings.catch_warnings(record=True) as observed:
        main(["run", "--model", "standard", "-c", "selected.ini"])
    assert calls == [{"config_file": "selected.ini"}]
    assert not observed


def test_older_scientific_view_none_is_noop_and_missing_capabilities_are_actionable():
    import xarray as xr
    from openghg_inversions.postprocessing.inversion_output import InversionOutput
    from openghg_inversions.rhime.outputs import postprocess_standard_outputs
    from openghg_inversions.rhime.specs import RhimeOutputSpec
    from tests.test_staged_workflow import _prepared

    prepared = _prepared()
    older = InversionOutput(trace=xr.DataTree(), inv_inputs=prepared.inv_inputs,
                            basis_functions=prepared.basis_functions)
    assert postprocess_standard_outputs(older, RhimeOutputSpec(output_format="none")) is None
    with pytest.raises(ValueError, match="lacks supported-output capabilities"):
        postprocess_standard_outputs(older, RhimeOutputSpec(output_format="inv_out", save_inversion_output=False))


def test_saved_standard_family_selector_rejects_multisector(tmp_path):
    import json
    from openghg_inversions.cli import _check_saved_family

    manifest = tmp_path / "prepare-manifest.json"
    manifest.write_text(json.dumps({"schema": "openghg_inversions.standard.prepare"}))
    with pytest.raises(ValueError, match="standard checkpoint.*multisector"):
        _check_saved_family(Namespace(command="sample", preparation_manifest=manifest, model="multisector"))


def test_shared_output_view_projects_capabilities_and_basic_is_explicit(monkeypatch, tmp_path):
    import xarray as xr
    from openghg_inversions.postprocessing.contracts import OutputContract
    from openghg_inversions.postprocessing.output_views import make_inversion_output
    from openghg_inversions.postprocessing import make_outputs
    from openghg_inversions.rhime.outputs import postprocess_standard_outputs, validate_output_resources
    from openghg_inversions.rhime.specs import RhimeOutputSpec
    from tests.test_staged_workflow import _prepared

    contract = OutputContract(variable_roles={"flux_scale": "x"}, supported_output_formats=("none", "basic"))
    inv_out = make_inversion_output(
        prepared=_prepared(), trace=xr.DataTree(), contract=contract,
        run_metadata={"start_date": "2019-01-01"}, model_metadata={"species": "ch4", "domain": "EUROPE"},
        output_metadata={}, provenance={},
    )
    assert inv_out.model_metadata["supported_output_formats"] == ["none", "basic"]
    monkeypatch.setattr(make_outputs, "basic_output", lambda *a, **k: pytest.fail("unused basic product computed"))
    country = tmp_path / "country.nc"
    country.touch()
    spec = RhimeOutputSpec(output_format="basic", country_file=country, save_inversion_output=False)
    validate_output_resources(spec)
    assert postprocess_standard_outputs(inv_out, spec) is None


@pytest.mark.parametrize("output_format", ["paris", "legacy"])
def test_product_processing_and_artifact_export_are_separate(tmp_path, monkeypatch, output_format):
    import xarray as xr
    from openghg_inversions.postprocessing.inversion_output import InversionOutput
    from openghg_inversions.rhime import outputs
    from openghg_inversions.rhime.specs import RhimeOutputSpec
    from tests.test_staged_workflow import _prepared

    prepared = _prepared()
    inv_out = InversionOutput(
        trace=xr.DataTree(), inv_inputs=prepared.inv_inputs, basis_functions=prepared.basis_functions,
        run_metadata={"start_date": "2019-01-01", "averaging_period": ["1h"]},
        model_metadata={"species": "ch4", "domain": "EUROPE", "use_bc": False,
                        "supported_output_formats": [output_format]},
    )
    country = tmp_path / "country.nc"
    country.touch()
    spec = RhimeOutputSpec(output_format=output_format, output_path=tmp_path, country_file=country, save_trace=True)
    outputs.validate_standard_output_resources(spec)
    events = []
    product = xr.Dataset()

    def compute(*args, **kwargs):
        events.append("compute")
        return (product, product) if output_format == "paris" else product

    monkeypatch.setattr("openghg_inversions.postprocessing.make_paris_outputs.make_paris_outputs", compute)
    monkeypatch.setattr("openghg_inversions.postprocessing.legacy_outputs.make_legacy_hbmcmc_output", compute)
    monkeypatch.setattr(outputs, "write_netcdf_preserving_bounds_attrs", lambda *a, **k: events.append("product"))
    monkeypatch.setattr(xr.Dataset, "to_netcdf", lambda *a, **k: events.append("product"))
    monkeypatch.setattr(InversionOutput, "save", lambda *a, **k: events.append("inversion"))
    monkeypatch.setattr(outputs, "save_trace", lambda *a, **k: events.append("trace"))

    outputs.postprocess_standard_outputs(inv_out, spec)
    assert events == ["compute"] + ["product"] * (2 if output_format == "paris" else 1)
    events.clear()
    outputs.save_inversion_output(inv_out, spec)
    assert events == ["inversion", "trace"]


@pytest.mark.parametrize("field", ["output_name", "start_date"])
def test_direct_artifact_export_rejects_filename_traversal(tmp_path, field):
    from types import SimpleNamespace
    from openghg_inversions.rhime.outputs import save_inversion_output
    from openghg_inversions.rhime.specs import RhimeOutputSpec

    destination = tmp_path / "outputs"
    inv_out = SimpleNamespace(run_metadata={"start_date": "../../outside" if field == "start_date" else "2019-01-01"})
    spec = RhimeOutputSpec(output_path=destination, output_name="../../outside" if field == "output_name" else "run")
    with pytest.raises(ValueError, match="filename component"):
        save_inversion_output(inv_out, spec)
    assert not destination.exists()
