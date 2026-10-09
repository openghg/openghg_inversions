"""Integration tests for the executable RHIME customisation example."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import xarray as xr

from openghg_inversions.inversion_data import RhimeMergedData
from openghg_inversions.rhime.params import RhimeConfig

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
        "mismatch_model": None,
        "output_format": "none",
    }


def test_custom_runner_uses_supported_stages_for_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Carry ordinary acquisition requests through every public stage."""
    config_file = tmp_path / "rhime.ini"
    config_file.write_text('[RHIME.OUTPUT]\noutput_format = "none"\n', encoding="utf-8")
    overrides = {"draws": 3}
    config = RhimeConfig.from_params(
        params=dict(species="ch4", sites=["TAC", "MHD"], domain="EUROPE",
                    averaging_period="1h", start_date="2019-01-01", end_date="2019-02-01",
                    output_name="example", output_format="none", mismatch_model=None, flux_sources=["inventory"],
                    draws=3),
        multisector=False,
    )
    sampler = config.sampler

    merged = RhimeMergedData(
        site_data={site: xr.Dataset() for site in config.site_options.sites},
        flux_data={}, site_options=config.site_options,
    )
    filtered = object()
    basis = object()
    site_data = object()
    prepared = SimpleNamespace(
        inv_inputs=xr.Dataset({"mf": ("nmeasure", [1.0])}),
        sites=("MHD",),
        averaging_period=("1h",),
        basis_artifact_source="test-basis",
    )
    model_inputs = xr.Dataset({"mf": ("nmeasure", [1.0])})
    build_result = object()
    idata = object()
    expected_result = object()
    calls: list[str] = []

    def parse_config(actual_config: str | Path) -> dict[str, Any]:
        assert actual_config == config_file
        return {"output_format": "none"}

    def resolve(cls, params: dict[str, Any], *, multisector: bool) -> Any:
        assert params == {"output_format": "none", **overrides, "mismatch_model": None}
        assert multisector is False
        calls.append("resolve")
        return config

    def retrieve(**kwargs: Any) -> Any:
        assert kwargs["site_options"] is config.site_options
        assert kwargs["split_by_sectors"] is False
        assert "project_basis_path" not in kwargs
        calls.append("retrieve")
        return merged

    def filter_observations(actual: Any, *, filters: Any) -> Any:
        assert actual is merged
        assert filters is config.filters
        calls.append("filter")
        return filtered

    def build_basis(actual: Any, **kwargs: Any) -> Any:
        assert actual is filtered
        assert kwargs["domain"] == config.domain
        assert kwargs["nbasis"] == config.nbasis
        calls.append("basis")
        return basis

    def build_sensitivities(actual: Any, actual_basis: Any, **kwargs: Any) -> Any:
        assert actual is filtered
        assert actual_basis is basis
        assert kwargs["domain"] == config.domain
        assert kwargs["flux_sources"] == config.flux_sources
        assert kwargs["multisector"] is False
        calls.append("sensitivities")
        return site_data

    def assemble(actual: Any, actual_basis: Any, actual_site_data: Any, **kwargs: Any) -> Any:
        assert actual is filtered
        assert actual_basis is basis
        assert actual_site_data is site_data
        assert kwargs["min_error_options"] == config.min_error_options
        assert kwargs["start_date"] == config.start_date
        calls.append("assemble")
        return prepared


    def materialize(actual: Any, *, variable_names: tuple[str, ...]) -> xr.Dataset:
        """Record the explicit eager model-input boundary."""
        assert actual is prepared
        assert set(variable_names) >= {"H", "mf", "mf_error"}
        assert "min_error" not in variable_names
        calls.append("materialize")
        return model_inputs

    def build(**kwargs: Any) -> Any:
        """Record the project-owned Student-t likelihood handoff."""
        assert kwargs["prepared"] is prepared
        assert kwargs["model_inputs"] is model_inputs
        assert kwargs["run_spec"].sites == ("MHD",)
        assert kwargs["run_spec"].model is config.model
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
        assert kwargs["run_spec"].sites == ("MHD",)
        assert kwargs["run_spec"].model is config.model
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

    monkeypatch.setattr(custom_runner, "read_rhime_ini", parse_config)
    monkeypatch.setattr(custom_runner.RhimeConfig, "from_params", classmethod(resolve))
    monkeypatch.setattr(custom_runner.RhimeMergedData, "from_options", retrieve)
    monkeypatch.setattr(custom_runner, "filter_rhime_observations", filter_observations)
    monkeypatch.setattr(custom_runner, "build_rhime_basis", build_basis)
    monkeypatch.setattr(custom_runner, "build_rhime_sensitivities", build_sensitivities)
    monkeypatch.setattr(custom_runner, "assemble_rhime_inputs", assemble)
    monkeypatch.setattr(
        custom_runner,
        "standard_model_input_names",
        lambda _actual, _model: ("H", "mf", "mf_error"),
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
            '{"draws": 99, "species": "ch4"}',
        ]
    )

    assert result is expected_result
    assert seen == {
        "config_file": config_file,
        "kwargs": {
            "species": "ch4",
            "start_date": "2019-01-01",
            "end_date": "2019-02-01",
            "output_path": output_path,
            "output_name": "cli-name",
            "draws": 5,
            "tune": 2,
            "chains": 1,
        },
    }
