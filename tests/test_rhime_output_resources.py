"""Reporting resources belong to output execution, independently of preparation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import xarray as xr

from examples.rhime_customisation import runner as copied_runner
from openghg_inversions.rhime import (
    RhimeConfig,
    _standard_stages,
    multisector,
    nested,
    outputs,
    preparation,
    prepared,
    standard,
    validate_output_resources,
)
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime.specs import RhimeOutputSpec
from tests.helpers import make_trace
from tests.integration.test_custom_basis_rhime_runner import custom_basis_runner
from tests.test_staged_workflow import _params, _prepared


@pytest.mark.parametrize("output_format", ["none", "inv_out", "basic", "paris", "legacy"])
def test_reporting_resource_policy(tmp_path, output_format):
    spec = RhimeOutputSpec(output_format=output_format)
    if output_format in ("basic", "paris"):
        with pytest.raises(ValueError, match="country_file must be supplied"):
            validate_output_resources(spec)
    else:
        validate_output_resources(spec)
    for path in (tmp_path / "missing.nc", tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_output_resources(replace(spec, country_file=str(path)))
    country = tmp_path / "country.nc"
    country.touch()
    # Resource validation checks the file's presence, without opening its contents.
    validate_output_resources(replace(spec, country_file=str(country)))
    assert spec.country_file is None


@pytest.mark.parametrize("country_kind", ["omitted", "missing", "directory"])
@pytest.mark.parametrize("route", [
    "standard", "multisector", "nested", "copied", "copied_basis",
    "prepared", "prepared_multi", "prepared_nested", "staged",
    "output_multi", "output_nested",
])
def test_reporting_routes_reject_before_work_or_output_directory(tmp_path, monkeypatch, country_kind, route):
    destination = tmp_path / "outputs"
    country = {"omitted": None, "missing": str(tmp_path / "missing.nc"), "directory": str(tmp_path)}[country_kind]
    params = _params(output_format="paris", country_file=country, output_path=str(destination))
    is_multi = route in ("multisector", "prepared_multi")
    if is_multi:
        params["flux_sources"] = ["inventory", "ocean"]
    config = RhimeConfig.from_params(params, multisector=is_multi)
    monkeypatch.setattr(
        standard.RhimeMergedData, "from_options", lambda **kwargs: pytest.fail("acquired before preflight"),
    )
    monkeypatch.setattr(
        prepared, "materialize_pymc_inputs", lambda *args, **kwargs: pytest.fail("materialized before preflight"),
    )
    monkeypatch.setattr(
        nested, "materialize_pymc_inputs", lambda *args, **kwargs: pytest.fail("materialized before preflight"),
    )
    monkeypatch.setattr(
        outputs, "_make_inversion_output", lambda **kwargs: pytest.fail("constructed products before preflight"),
    )
    if route == "staged":
        # Saved standard samples replay without a graph; only output choices change.
        preparation = _standard_stages.save_preparation(_prepared(), data_config=config.data, output_dir=tmp_path / "prep")
        posterior = make_trace(posterior=xr.Dataset(
            {"x": (("chain", "draw", "region"), [[[1.0]]])}, coords={"chain": [0], "draw": [0], "region": [0]},
        ))
        posterior.attrs["rhime_priors"] = "{}"
        sampled = _standard_stages.save_samples(
            preparation, posterior,
            model=config.model, actual_inference=config.sampler.describe(),
            output_contract=OutputContract(variable_roles={"flux_scale": "x"}, supported_output_formats=("none", "paris")),
            output_dir=tmp_path / "samples",
            preparation_generation=_standard_stages.load_preparation(preparation)[2],
        )
    error = ValueError if country is None else FileNotFoundError
    with pytest.raises(error, match="country_file" if country is None else None):
        if route == "standard":
            standard.run_standard(config=config)
        elif route == "multisector":
            multisector.run_rhime_multisector(config=config)
        elif route == "nested":
            nested.run_nested(**params, inner_domain="6km")
        elif route == "copied":
            copied_runner.run_custom_rhime(**params)
        elif route == "copied_basis":
            custom_basis_runner.run_custom_rhime(**params)
        elif route in ("prepared", "prepared_multi"):
            inputs = _prepared()
            if is_multi:
                inputs = replace(inputs, inv_inputs=inputs.inv_inputs.assign(
                    H=inputs.inv_inputs.H.expand_dims(source=["inventory", "ocean"]),
                ))
            if is_multi:
                prepared.run_rhime_from_prepared_inputs(
                    prepared_inputs=inputs, run_spec=config.retained_run_spec(inputs),
                )
            else:
                standard.run_standard(prepared=inputs, config=config)
        elif route == "prepared_nested":
            nested.run_rhime_nested_from_prepared_inputs(
                prepared_inputs=None, run_spec=config.retained_run_spec(_prepared()),
            )
        elif route == "staged":
            _standard_stages.postprocess(sampled, output=config.output)
        else:
            result = SimpleNamespace(output_spec=config.output)
            if route == "output_nested":
                nested._write_nested_paris_outputs(SimpleNamespace(rhime_result=result))
            else:
                outputs.make_multisector_rhime_outputs(result=result, prepared=None)
    assert not destination.exists()


def test_resource_helpers_are_public_and_indexed():
    from pathlib import Path

    assert "resolve_basis_inputs" in preparation.__all__
    index = (Path(__file__).parents[1] / "docs/reference/index.rst").read_text()
    assert "openghg_inversions.rhime.preparation.resolve_basis_inputs" in index
    assert "openghg_inversions.rhime.outputs.validate_output_resources" in index
