"""Public standard runner migration preserves CLI execution."""

from argparse import Namespace
import warnings
import pytest


def test_removed_python_runner_explains_migration():
    with pytest.raises(ImportError, match="removed in 0.8.*run_standard.*InversionOutput.*CLI command is unchanged"):
        from openghg_inversions.rhime import run_rhime  # noqa: F401


def test_established_cli_calls_standard_without_python_compatibility_warning(monkeypatch):
    import openghg_inversions.rhime as rhime
    from openghg_inversions.cli import _run_rhime_command

    calls = []
    monkeypatch.setattr(rhime, "run_standard", lambda **kwargs: calls.append(kwargs))
    with warnings.catch_warnings(record=True) as observed:
        _run_rhime_command(Namespace(config="selected.ini", start=None, end=None, output_path=None, kwargs=None))
    assert calls == [{"config_file": "selected.ini"}]
    assert not observed


def test_older_scientific_view_none_is_noop_and_missing_capabilities_are_actionable():
    import xarray as xr
    from openghg_inversions.postprocessing.inversion_output import InversionOutput
    from openghg_inversions.rhime.outputs import write_standard_outputs
    from openghg_inversions.rhime.specs import RhimeOutputSpec
    from tests.test_staged_workflow import _prepared

    prepared = _prepared()
    older = InversionOutput(trace=xr.DataTree(), inv_inputs=prepared.inv_inputs,
                            basis_functions=prepared.basis_functions)
    assert write_standard_outputs(older, RhimeOutputSpec(output_format="none")) is None
    with pytest.raises(ValueError, match="lacks supported-output capabilities"):
        write_standard_outputs(older, RhimeOutputSpec(output_format="inv_out", save_inversion_output=False))


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
    from openghg_inversions.rhime.outputs import write_standard_outputs, validate_output_resources
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
    assert write_standard_outputs(inv_out, spec) is None
