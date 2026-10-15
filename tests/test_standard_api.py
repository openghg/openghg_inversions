"""Public standard runner migration preserves CLI execution."""

from argparse import Namespace
import warnings
import pytest


def test_removed_python_runner_explains_migration():
    with pytest.raises(ImportError, match="removed in 0.8.*run_standard.*InversionOutput.*CLI command is unchanged"):
        from openghg_inversions.rhime import run_rhime
        assert callable(run_rhime)


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
    assert write_standard_outputs(older, RhimeOutputSpec(output_format="none")) == {}
    with pytest.raises(ValueError, match="lacks supported-output capabilities"):
        write_standard_outputs(older, RhimeOutputSpec(output_format="inv_out", save_inversion_output=False))
