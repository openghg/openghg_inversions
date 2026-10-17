"""Reporting resources belong to output execution, independently of preparation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import xarray as xr

from examples.rhime_customisation import runner as copied_runner
from openghg_inversions.rhime import (
    RhimeConfig,
    _standard_stages,
    multisector_stages,
    multisector,
    nested,
    outputs,
    preparation,
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
    "prepared", "prepared_multi", "prepared_nested", "staged", "staged_multi",
    "output_multi", "output_nested",
])
def test_reporting_routes_reject_before_work_or_output_directory(tmp_path, monkeypatch, country_kind, route):
    destination = tmp_path / "outputs"
    country = {"omitted": None, "missing": str(tmp_path / "missing.nc"), "directory": str(tmp_path)}[country_kind]
    params = _params(output_format="paris", country_file=country, output_path=str(destination))
    is_multi = route in ("multisector", "prepared_multi", "staged_multi", "output_multi")
    if is_multi:
        params["flux_sources"] = ["inventory", "ocean"]
    config = RhimeConfig.from_params(params, multisector=is_multi)
    monkeypatch.setattr(
        standard.RhimeMergedData, "from_options", lambda **kwargs: pytest.fail("acquired before preflight"),
    )
    monkeypatch.setattr(
        multisector, "materialize_multisector_inputs", lambda *args, **kwargs: pytest.fail("materialized before preflight"),
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
                multisector.run_multisector(prepared=inputs, config=config)
            else:
                standard.run_standard(prepared=inputs, config=config)
        elif route == "prepared_nested":
            nested.run_rhime_nested_from_prepared_inputs(
                prepared_inputs=None, run_spec=config.retained_run_spec(_prepared()),
            )
        elif route == "staged":
            _standard_stages.postprocess(sampled, output=config.output)
        elif route == "staged_multi":
            multisector_stages.postprocess(tmp_path / "absent-manifest.json", output=config.output)
        else:
            result = SimpleNamespace(output_spec=config.output)
            if route == "output_nested":
                nested._write_nested_paris_outputs(SimpleNamespace(rhime_result=result))
            else:
                outputs.postprocess_multisector_outputs(
                    SimpleNamespace(model_metadata={"supported_output_formats": ["paris"]}), config.output,
                )
    assert not destination.exists()


def test_resource_helpers_are_public_and_indexed():
    from pathlib import Path

    assert "resolve_basis_inputs" in preparation.__all__
    index = (Path(__file__).parents[1] / "docs/reference/index.rst").read_text()
    assert "openghg_inversions.rhime.preparation.resolve_basis_inputs" in index
    assert "openghg_inversions.rhime.outputs.validate_output_resources" in index


@pytest.mark.parametrize("output_format", ["paris", "legacy"])
@pytest.mark.parametrize("route", ["standard", "prepared", "staged", "copied", "copied_basis"])
def test_standard_file_products_require_destination_before_work(tmp_path, monkeypatch, output_format, route):
    country = tmp_path / "country.nc"
    country.touch()
    params = _params(output_format=output_format, country_file=country, output_path=None,
                     save_inversion_output=False)
    config = RhimeConfig.from_params(params, multisector=False)
    monkeypatch.setattr(
        standard.RhimeMergedData, "from_options", lambda **kwargs: pytest.fail("acquired before preflight"),
    )
    monkeypatch.setattr(
        standard, "sample_rhime_model", lambda *args, **kwargs: pytest.fail("sampled before preflight"),
    )
    monkeypatch.setattr(
        standard, "materialize_standard_inputs", lambda *args, **kwargs: pytest.fail("materialized before preflight"),
    )
    with pytest.raises(ValueError, match="output_path.*required.*standard"):
        if route == "standard":
            standard.run_standard(config=config)
        elif route == "prepared":
            standard.run_standard(config=config, prepared=_prepared())
        elif route == "staged":
            _standard_stages.postprocess(tmp_path / "absent-manifest.json", output=config.output)
        elif route == "copied":
            copied_runner.run_custom_rhime(**params)
        else:
            custom_basis_runner.run_custom_rhime(**params)


@pytest.mark.parametrize("output_format", ["paris", "legacy"])
def test_shared_resource_validation_preserves_deferred_in_memory_products(tmp_path, output_format):
    country = tmp_path / "country.nc"
    country.touch()
    spec = RhimeOutputSpec(output_format=output_format, country_file=country, save_inversion_output=False)
    validate_output_resources(spec)
    with pytest.raises(ValueError, match="output_path"):
        outputs.validate_standard_output_resources(spec)


@pytest.mark.parametrize("route", ["full", "prepared", "staged"])
def test_multisector_paris_requires_destination_before_work(tmp_path, monkeypatch, route):
    country = tmp_path / "country.nc"
    country.touch()
    config = RhimeConfig.from_params(
        _params(flux_sources=["inventory", "ocean"], output_format="paris",
                country_file=country, output_path=None, save_inversion_output=False),
        multisector=True,
    )
    monkeypatch.setattr(
        multisector, "acquire_merged_data", lambda *args, **kwargs: pytest.fail("acquired before preflight"),
    )
    monkeypatch.setattr(
        multisector, "materialize_multisector_inputs",
        lambda *args, **kwargs: pytest.fail("materialized before preflight"),
    )
    with pytest.raises(ValueError, match="output_path.*required.*multisector"):
        if route == "full":
            multisector.run_multisector(config=config)
        elif route == "prepared":
            multisector.run_multisector(config=config, prepared=_prepared())
        else:
            multisector_stages.postprocess(tmp_path / "absent-manifest.json", output=config.output)
