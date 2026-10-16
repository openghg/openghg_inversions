"""Integration tests for the executable RHIME customisation example."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import xarray as xr

from openghg_inversions.inversion_data import RhimeMergedData
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.requests import load_params
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.contracts import OutputContract

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

    def run_standard(**kwargs: Any) -> Any:
        """Capture the preferred one-call example without executing RHIME."""
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(short_runner, "run", run_standard)
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
        site_data={site: xr.Dataset({"mf_error": ("time", [1.0]), "mf_repeatability": ("time", [0.0]), "mf_variability": ("time", [0.0])}) for site in config.data.site_options.sites},
        flux_data={"inventory": xr.Dataset({"flux": (("lat", "lon"), [[2.]])},
                                           coords={"lat": [50.], "lon": [0.]})}, site_options=config.data.site_options,
    )
    filtered = {"MHD": merged.site_data["MHD"]}
    filtered_options = object()
    basis = object()
    site_data = object()
    prepared = SimpleNamespace(
        inv_inputs=xr.Dataset({"mf": ("nmeasure", [1.0])}),
        sites=("MHD",),
        averaging_period=("1h",),
        basis_artifact_source="test-basis",
    )
    output_contract = OutputContract(variable_roles={"flux_scale": "x"}, supported_output_formats=("none", "inv_out"))
    build_result = SimpleNamespace(output_contract=output_contract)
    idata = xr.DataTree()
    model_inputs = prepared.inv_inputs
    priors = {"x": {"pdf": "normal", "mu": 1.0, "sigma": 0.2}}
    expected_result = InversionOutput(trace=idata, inv_inputs=prepared.inv_inputs, basis_functions=basis)
    calls: list[str] = []

    def parse_config(actual_config: str | Path) -> dict[str, Any]:
        assert actual_config == config_file
        return {"output_format": "none"}

    def resolve(cls, params: dict[str, Any], *, multisector: bool) -> Any:
        assert params == load_params(overrides={"output_format": "none", **overrides, "mismatch_model": None})
        assert multisector is False
        calls.append("resolve")
        return config

    resolved_basis = {"landsea_classes": object(), "outer_regions": None}

    def resolve_basis_inputs(actual: RhimeConfig) -> dict[str, Any]:
        assert actual == config.data
        calls.append("basis-inputs")
        return resolved_basis

    def retrieve(**kwargs: Any) -> Any:
        assert kwargs["site_options"] is config.data.site_options
        assert kwargs["split_by_sectors"] is False
        assert "project_basis_path" not in kwargs
        calls.append("retrieve")
        return merged

    def filter_observations(actual: Any, site_options: Any, *, filters: Any) -> Any:
        assert actual.keys() == merged.site_data.keys()
        assert site_options is merged.site_options
        assert filters is config.data.filters
        calls.append("filter")
        return filtered, filtered_options

    def build_basis(**kwargs: Any) -> Any:
        assert kwargs["site_data"] is filtered
        assert kwargs["domain"] == config.data.domain
        assert kwargs["nbasis"] == config.data.nbasis
        assert kwargs["landsea_classes"] is resolved_basis["landsea_classes"]
        assert "output_path" not in kwargs
        calls.append("basis")
        return basis

    def build_sensitivities(actual: Any, actual_basis: Any, **kwargs: Any) -> Any:
        assert actual is filtered
        assert actual_basis is basis
        assert kwargs["bc_basis"] is None
        assert kwargs["flux_sources"] == config.data.flux_sources
        assert kwargs["multisector"] is False
        calls.append("sensitivities")
        return site_data

    def assemble(actual: Any, actual_basis: Any, **kwargs: Any) -> Any:
        assert actual is site_data
        assert actual_basis is basis
        assert kwargs["site_options"] is filtered_options
        assert kwargs["min_error_options"] == config.data.min_error_options
        assert kwargs["start_date"] == config.data.start_date
        calls.append("assemble")
        return prepared


    def materialize(actual: Any, model_spec: Any) -> Any:
        """Keep selected numerical inputs and priors at the explicit boundary."""
        assert actual is prepared
        assert model_spec is config.model
        calls.append("materialize")
        return model_inputs, priors

    def build(actual: Any, model_spec: Any, **kwargs: Any) -> Any:
        """Record the eager model boundary after project preparation."""
        assert actual is model_inputs
        assert model_spec is config.model
        assert kwargs["priors"] is priors
        assert kwargs["likelihood_builder"] is custom_runner.likelihood_builder
        calls.append("build")
        return build_result

    def sample(*args: Any, **kwargs: Any) -> Any:
        """Record public sampling."""
        assert args == (build_result, sampler)
        assert kwargs == {}
        calls.append("sample")
        return idata

    def annotate(actual: Any, posterior: Any, **kwargs: Any) -> None:
        """Record prior and custom-likelihood provenance once after sampling."""
        assert actual is build_result
        assert posterior is idata
        assert kwargs["likelihood_builder"] is custom_runner.likelihood_builder
        calls.append("annotate")

    def make_result(**kwargs: Any) -> InversionOutput:
        """Construct the scientific view without output policy or live references."""
        assert kwargs["prepared"] is prepared
        assert kwargs["model_spec"] is config.model
        assert kwargs["posterior"] is idata
        assert kwargs["output_contract"] is output_contract
        assert kwargs["start_date"] == config.data.start_date
        assert kwargs["end_date"] == config.data.end_date
        assert kwargs["actual_inference"] == sampler.describe()
        assert kwargs["split_by_sectors"] is False
        assert "run_spec" not in kwargs
        assert "sampler" not in kwargs
        calls.append("result")
        return expected_result

    def write_outputs(actual: InversionOutput, output_spec: Any) -> None:
        assert actual is expected_result
        assert output_spec is config.output
        calls.append("output")

    monkeypatch.setattr(custom_runner, "read_rhime_ini", parse_config)
    monkeypatch.setattr(custom_runner.RhimeConfig, "from_params", classmethod(resolve))
    monkeypatch.setattr(custom_runner, "resolve_basis_inputs", resolve_basis_inputs)
    monkeypatch.setattr(RhimeMergedData, "from_options", retrieve)
    monkeypatch.setattr(custom_runner, "filter_observations", filter_observations)
    monkeypatch.setattr(custom_runner, "prepare_basis", build_basis)
    monkeypatch.setattr(custom_runner, "build_sensitivities", build_sensitivities)
    monkeypatch.setattr(custom_runner, "assemble_rhime_inputs", assemble)
    monkeypatch.setattr(custom_runner, "materialize_standard_inputs", materialize)
    monkeypatch.setattr(custom_runner, "build_standard_model", build)
    monkeypatch.setattr(custom_runner, "sample_rhime_model", sample)
    monkeypatch.setattr(custom_runner, "annotate_sample_trace", annotate)
    monkeypatch.setattr(custom_runner, "make_model_output", make_result)
    monkeypatch.setattr(custom_runner, "save_inversion_output", write_outputs)

    result = custom_runner.run_custom_rhime(config_file=config_file, **overrides)

    assert result is expected_result
    assert calls == [
        "resolve",
        "basis-inputs",
        "retrieve",
        "filter",
        "basis",
        "sensitivities",
        "assemble",
        "materialize",
        "build",
        "sample",
        "annotate",
        "result",
        "output",
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
