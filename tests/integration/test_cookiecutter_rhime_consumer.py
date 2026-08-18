"""Acceptance tests for the package-shaped cookiecutter RHIME consumer."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import xarray as xr

from examples.rhime_cookiecutter.my_inversion import likelihoods
from examples.rhime_cookiecutter.my_inversion import runner as consumer_runner
import openghg_inversions.rhime.standard as rhime_runner


def test_consumer_runs_public_acquisition_to_supported_output(  # noqa: C901, PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the downstream wrapper through controlled library-owned stages."""
    run_spec = SimpleNamespace(
        model=SimpleNamespace(aggregation_error_mode="diagonal"),
        output=SimpleNamespace(output_format="inv_out"),
    )
    sampler = object()
    setup = SimpleNamespace(
        data_args={"species": "ch4"},
        run_spec=run_spec,
        sampler=sampler,
    )
    merged = object()
    filtered = object()
    basis = object()
    site_data = object()
    prepared = SimpleNamespace(
        inv_inputs=xr.Dataset(
            {"mf": ("nmeasure", [1.0])},
            coords={"region": [0], "source": ["inventory"]},
        ),
        sites=("MHD",),
        basis_artifact_source="controlled-test-basis",
    )
    model_inputs = xr.Dataset({"mf": ("nmeasure", [1.0])})
    build_result = object()
    idata = object()
    expected = object()
    calls: list[str] = []

    def resolve(*, params: dict[str, Any], multisector: bool) -> Any:
        assert params == {"species": "ch4", "output_format": "inv_out"}
        assert multisector is False
        calls.append("resolve")
        return setup

    def retrieve(data_args: dict[str, Any], *, multisector: bool) -> Any:
        assert data_args is setup.data_args
        assert multisector is False
        calls.append("retrieve")
        return merged

    def filter_observations(actual: Any, data_args: dict[str, Any]) -> Any:
        assert actual is merged
        assert data_args is setup.data_args
        calls.append("filter")
        return filtered

    def build_basis(actual: Any, data_args: dict[str, Any]) -> Any:
        assert actual is filtered
        assert data_args is setup.data_args
        calls.append("basis")
        return basis

    def build_sensitivities(
        actual: Any,
        actual_basis: Any,
        data_args: dict[str, Any],
        *,
        multisector: bool,
    ) -> Any:
        assert actual is filtered
        assert actual_basis is basis
        assert data_args is setup.data_args
        assert multisector is False
        calls.append("sensitivities")
        return site_data

    def assemble(
        actual: Any,
        actual_basis: Any,
        actual_site_data: Any,
        data_args: dict[str, Any],
    ) -> Any:
        assert (actual, actual_basis, actual_site_data) == (filtered, basis, site_data)
        assert data_args is setup.data_args
        calls.append("assemble")
        return prepared

    def align(actual_spec: Any, actual_prepared: Any) -> Any:
        assert actual_prepared is prepared
        calls.append("align")
        return actual_spec

    def materialize(actual: Any, *, aggregation_error_mode: str) -> Any:
        assert actual is prepared
        assert aggregation_error_mode == "diagonal"
        calls.append("materialize")
        return model_inputs

    def build(**kwargs: Any) -> Any:
        assert kwargs == {
            "prepared": prepared,
            "model_inputs": model_inputs,
            "run_spec": run_spec,
            "likelihood_builder": likelihoods.likelihood_builder,
        }
        calls.append("build")
        return build_result

    def sample(*args: Any, **kwargs: Any) -> Any:
        assert args == (build_result, sampler)
        assert kwargs == {"use_variable_roles": True}
        calls.append("sample")
        return idata

    def make_result(**kwargs: Any) -> Any:
        assert kwargs["run_spec"].output.output_format == "inv_out"
        assert kwargs["likelihood_builder"] is likelihoods.likelihood_builder
        assert kwargs["model_build_result"] is build_result
        assert kwargs["idata"] is idata
        calls.append("output")
        return expected

    monkeypatch.setattr(rhime_runner, "resolve_rhime_options", resolve)
    monkeypatch.setattr(rhime_runner, "retrieve_or_reload_rhime_data", retrieve)
    monkeypatch.setattr(rhime_runner, "filter_rhime_observations", filter_observations)
    monkeypatch.setattr(rhime_runner, "build_rhime_basis", build_basis)
    monkeypatch.setattr(rhime_runner, "build_rhime_sensitivities", build_sensitivities)
    monkeypatch.setattr(rhime_runner, "assemble_rhime_inputs", assemble)
    monkeypatch.setattr(rhime_runner, "with_prepared_rhime_sites", align)
    monkeypatch.setattr(rhime_runner, "materialize_pymc_inputs", materialize)
    monkeypatch.setattr(rhime_runner, "build_standard_rhime_model", build)
    monkeypatch.setattr(rhime_runner, "sample_rhime_model", sample)
    monkeypatch.setattr(rhime_runner, "make_standard_rhime_result", make_result)

    result = consumer_runner.run(species="ch4", output_format="inv_out")

    assert result is expected
    assert calls == [
        "resolve",
        "retrieve",
        "filter",
        "basis",
        "sensitivities",
        "assemble",
        "align",
        "materialize",
        "build",
        "sample",
        "output",
    ]


def test_consumer_cli_routes_to_the_same_project_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The module and optional project script entry point share ``main``."""
    config_file = tmp_path / "inversion.ini"
    expected = object()
    seen: dict[str, Any] = {}

    def run(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(consumer_runner, "run", run)
    result = consumer_runner.main(
        [str(config_file), "--kwargs", '{"output_format": "inv_out", "draws": 10}']
    )

    assert result is expected
    assert seen == {
        "config_file": config_file,
        "output_format": "inv_out",
        "draws": 10,
    }


@pytest.mark.parametrize("module_path", [likelihoods.__file__, consumer_runner.__file__])
def test_consumer_imports_only_the_public_rhime_package(module_path: str | None) -> None:
    """Consumer modules do not reach into private or non-RHIME library paths."""
    assert module_path is not None
    source = Path(module_path).read_text(encoding="utf-8")
    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    for node in imports:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("openghg_inversions"):
            assert node.module == "openghg_inversions.rhime"
            assert all(not alias.name.startswith("_") for alias in node.names)
        elif isinstance(node, ast.Import):
            assert all(not alias.name.startswith("openghg_inversions") for alias in node.names)
