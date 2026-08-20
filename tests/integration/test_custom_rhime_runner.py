"""Integration tests for the executable RHIME customisation example."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import xarray as xr

from examples.rhime_customisation import likelihoods
from examples.rhime_customisation import runner as custom_runner
from examples.rhime_customisation import run_with_likelihood as short_runner


def test_short_and_full_examples_share_likelihood_and_supported_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The preferred form selects the full runner's likelihood and output mode."""
    config_file = tmp_path / "rhime.ini"
    expected = object()
    seen: dict[str, Any] = {}

    def run_rhime(**kwargs: Any) -> Any:
        """Capture the preferred one-call example without executing RHIME."""
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(short_runner, "run_rhime", run_rhime)
    result = short_runner.run_with_likelihood(
        config_file=config_file,
        output_format="none",
    )

    assert result is expected
    assert short_runner.likelihood_builder is likelihoods.likelihood_builder
    assert custom_runner.likelihood_builder is likelihoods.likelihood_builder
    assert seen == {
        "config_file": config_file,
        "likelihood_builder": custom_runner.likelihood_builder,
        "output_format": "none",
    }


@pytest.mark.parametrize("reload_merged_data", [False, True])
def test_custom_runner_uses_supported_stages_for_acquisition_and_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reload_merged_data: bool,
) -> None:
    """Carry ordinary acquisition and reload requests through every public stage."""
    config_file = tmp_path / "rhime.ini"
    config_file.write_text('[RHIME.OUTPUT]\noutput_format = "none"\n', encoding="utf-8")
    overrides = {"reload_merged_data": reload_merged_data, "draws": 3}
    parsed_params = {"from_config": True, **overrides}
    data_args = {"reload_merged_data": reload_merged_data}
    sampler = object()
    initial_spec = SimpleNamespace(model=SimpleNamespace(aggregation_error_mode="diagonal"))
    aligned_spec = SimpleNamespace(model=SimpleNamespace(aggregation_error_mode="low_rank"))
    setup = SimpleNamespace(data_args=data_args, run_spec=initial_spec, sampler=sampler)

    merged = object()
    filtered = object()
    basis = object()
    site_data = object()
    prepared = SimpleNamespace(
        inv_inputs=xr.Dataset({"mf": ("nmeasure", [1.0])}),
        sites=("MHD",),
        basis_artifact_source="test-basis",
    )
    model_inputs = xr.Dataset({"mf": ("nmeasure", [1.0])})
    build_result = object()
    idata = object()
    expected_result = object()
    calls: list[str] = []

    def parse_config(
        actual_config: str | Path,
        *,
        extra_kwargs: dict[str, Any],
        normalise: bool,
    ) -> dict[str, Any]:
        """Record configuration parsing at the workflow boundary."""
        assert actual_config == config_file
        assert extra_kwargs == overrides
        assert normalise is False
        return parsed_params

    def resolve(*, params: dict[str, Any], multisector: bool) -> Any:
        """Record public option resolution."""
        assert params is parsed_params
        assert multisector is False
        calls.append("resolve")
        return setup

    def retrieve(actual_data_args: dict[str, Any], *, multisector: bool) -> Any:
        """Record ordinary acquisition or controlled merged-data reload."""
        assert actual_data_args is data_args
        assert actual_data_args["reload_merged_data"] is reload_merged_data
        assert multisector is False
        calls.append("retrieve")
        return merged

    def filter_observations(actual: Any, actual_data_args: dict[str, Any]) -> Any:
        """Record public observation filtering."""
        assert actual is merged
        assert actual_data_args is data_args
        calls.append("filter")
        return filtered

    def build_basis(actual: Any, actual_data_args: dict[str, Any]) -> Any:
        """Record public basis construction."""
        assert actual is filtered
        assert actual_data_args is data_args
        calls.append("basis")
        return basis

    def build_sensitivities(
        actual: Any,
        actual_basis: Any,
        actual_data_args: dict[str, Any],
        *,
        multisector: bool,
    ) -> Any:
        """Record public sensitivity construction."""
        assert actual is filtered
        assert actual_basis is basis
        assert actual_data_args is data_args
        assert multisector is False
        calls.append("sensitivities")
        return site_data

    def assemble(
        actual: Any,
        actual_basis: Any,
        actual_site_data: Any,
        actual_data_args: dict[str, Any],
    ) -> Any:
        """Record public labelled-input assembly."""
        assert actual is filtered
        assert actual_basis is basis
        assert actual_site_data is site_data
        assert actual_data_args is data_args
        calls.append("assemble")
        return prepared

    def align(actual_spec: Any, actual_prepared: Any) -> Any:
        """Record retained-site alignment."""
        assert actual_spec is initial_spec
        assert actual_prepared is prepared
        calls.append("align")
        return aligned_spec

    def materialize(actual: Any, *, variable_names: tuple[str, ...]) -> xr.Dataset:
        """Record the explicit eager model-input boundary."""
        assert actual is prepared
        assert set(variable_names) >= {"H", "mf", "mf_error", "min_error"}
        calls.append("materialize")
        return model_inputs

    def build(**kwargs: Any) -> Any:
        """Record the project-owned Student-t likelihood handoff."""
        assert kwargs["prepared"] is prepared
        assert kwargs["model_inputs"] is model_inputs
        assert kwargs["run_spec"] is aligned_spec
        assert kwargs["likelihood_builder"] is custom_runner.likelihood_builder
        calls.append("build")
        return build_result

    def sample(*args: Any, **kwargs: Any) -> Any:
        """Record public sampling."""
        assert args == (build_result, sampler)
        assert kwargs == {}
        calls.append("sample")
        return idata

    def make_result(**kwargs: Any) -> Any:
        """Record the supported output stage and its complete handoff."""
        assert kwargs["prepared"] is prepared
        assert kwargs["run_spec"] is aligned_spec
        assert kwargs["sampler"] is sampler
        assert kwargs["model_build_result"] is build_result
        assert kwargs["idata"] is idata
        assert kwargs["likelihood_builder"] is custom_runner.likelihood_builder
        assert kwargs["build_and_sample_seconds"] >= 0.0
        calls.append("result")
        return expected_result

    def make_outputs(**kwargs: Any) -> None:
        assert kwargs == {"result": expected_result, "prepared": prepared}
        calls.append("outputs")

    monkeypatch.setattr(custom_runner, "params_from_config", parse_config)
    monkeypatch.setattr(custom_runner, "resolve_rhime_options", resolve)
    monkeypatch.setattr(custom_runner, "retrieve_or_reload_rhime_data", retrieve)
    monkeypatch.setattr(custom_runner, "filter_rhime_observations", filter_observations)
    monkeypatch.setattr(custom_runner, "build_rhime_basis", build_basis)
    monkeypatch.setattr(custom_runner, "build_rhime_sensitivities", build_sensitivities)
    monkeypatch.setattr(custom_runner, "assemble_rhime_inputs", assemble)
    monkeypatch.setattr(custom_runner, "with_prepared_rhime_sites", align)
    monkeypatch.setattr(
        custom_runner,
        "standard_model_input_names",
        lambda _actual, _model: ("H", "mf", "mf_error", "min_error"),
    )
    monkeypatch.setattr(custom_runner, "materialize_pymc_inputs", materialize)
    monkeypatch.setattr(custom_runner, "build_standard_rhime_model_result", build)
    monkeypatch.setattr(custom_runner, "sample_rhime_model", sample)
    monkeypatch.setattr(custom_runner, "make_standard_rhime_result", make_result)
    monkeypatch.setattr(custom_runner, "make_standard_rhime_outputs", make_outputs)

    result = custom_runner.run_custom_rhime(config_file=config_file, **overrides)

    assert result is expected_result
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
        "result",
        "outputs",
    ]


def test_custom_runner_main_forwards_cli_config_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return the run result after forwarding typed CLI and JSON overrides."""
    config_file = tmp_path / "rhime.ini"
    output_path = tmp_path / "outputs"
    expected_result = object()
    seen: dict[str, Any] = {}

    def run_custom_rhime(*, config_file: str | Path | None, **kwargs: Any) -> Any:
        """Capture the command-line handoff without starting a real inversion."""
        seen["config_file"] = config_file
        seen["kwargs"] = kwargs
        return expected_result

    monkeypatch.setattr(custom_runner, "run_custom_rhime", run_custom_rhime)

    result = custom_runner.main(
        [
            str(config_file),
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2019-02-01",
            "--output-path",
            str(output_path),
            "--output-name",
            "cli-name",
            "--draws",
            "5",
            "--tune",
            "2",
            "--chains",
            "1",
            "--kwargs",
            '{"draws": 99, "species": "ch4", "reload_merged_data": true}',
        ]
    )

    assert result is expected_result
    assert seen == {
        "config_file": config_file,
        "kwargs": {
            "species": "ch4",
            "reload_merged_data": True,
            "start_date": "2019-01-01",
            "end_date": "2019-02-01",
            "output_path": output_path,
            "output_name": "cli-name",
            "draws": 5,
            "tune": 2,
            "chains": 1,
        },
    }
