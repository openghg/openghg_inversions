"""Acceptance tests for the package-shaped cookiecutter RHIME consumer."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import xarray as xr

from openghg_inversions.inversion_data import RhimeMergedData
from openghg_inversions.rhime.params import RhimeConfig

from examples.rhime_cookiecutter.my_inversion import likelihoods
from examples.rhime_cookiecutter.my_inversion import runner as consumer_runner
import openghg_inversions.rhime.standard as rhime_runner


def test_consumer_runs_public_acquisition_to_supported_output(  # noqa: C901, PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the downstream wrapper through controlled library-owned stages."""
    config = RhimeConfig.from_params(
        params=dict(species="ch4", sites=["TAC", "MHD"], domain="EUROPE",
                    averaging_period="1h", start_date="2019-01-01", end_date="2019-02-01",
                    output_name="example", output_format="inv_out", save_inversion_output=False,
                    mismatch_model=None, flux_sources=["inventory"],
                    draws=3),
        multisector=False,
    )
    sampler = config.sampler
    merged = RhimeMergedData(
        site_data={
            site: xr.Dataset({
                "mf_error": ("time", [1.0]),
                "mf_repeatability": ("time", [0.0]),
                "mf_variability": ("time", [0.0]),
            })
            for site in config.data.site_options.sites
        },
        flux_data={"inventory": xr.Dataset({"flux": (("lat", "lon"), [[2.]])},
                                           coords={"lat": [50.], "lon": [0.]})},
        site_options=config.data.site_options,
    )
    filtered = {"MHD": merged.site_data["MHD"]}
    filtered_options = object()
    basis = object()
    site_data = object()
    prepared = SimpleNamespace(
        inv_inputs=xr.Dataset(
            {"mf": ("nmeasure", [1.0])},
            coords={"region": [0], "source": ["inventory"]},
        ),
        sites=("MHD",),
        averaging_period=("1h",),
        basis_artifact_source="controlled-test-basis",
    )
    model_inputs = xr.Dataset({"mf": ("nmeasure", [1.0])})
    build_result = object()
    idata = object()
    expected = object()
    calls: list[str] = []

    def resolve(cls, params: dict[str, Any], *, multisector: bool) -> Any:
        assert params == {
            "data": {"species": "ch4"},
            "model": {"mismatch_model": None},
            "sampler": {},
            "output": {"output_format": "inv_out"},
        }
        assert multisector is False
        calls.append("resolve")
        return config

    def retrieve(**kwargs: Any) -> Any:
        assert kwargs["site_options"] is config.data.site_options
        assert kwargs["species"] == config.data.species
        assert kwargs["split_by_sectors"] is False
        assert "merged_data" not in kwargs
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
        assert kwargs["weighting_flux"].data is merged.flux_data[config.data.flux_sources[0]]["flux"].data
        xr.testing.assert_equal(kwargs["runtime_flux"], merged.flux_data[config.data.flux_sources[0]]["flux"])
        assert kwargs["domain"] == config.data.domain
        calls.append("basis")
        return basis

    def build_sensitivities(actual: Any, actual_basis: Any, **kwargs: Any) -> Any:
        assert actual is filtered
        assert actual_basis is basis
        assert kwargs["multisector"] is False
        calls.append("sensitivities")
        return site_data

    def assemble(actual: Any, actual_basis: Any, **kwargs: Any) -> Any:
        assert (actual, actual_basis) == (site_data, basis)
        assert kwargs["site_options"] is filtered_options
        assert kwargs["min_error_options"] == config.data.min_error_options
        calls.append("assemble")
        return prepared



    def materialize(actual: Any, *, variable_names: tuple[str, ...]) -> Any:
        assert actual is prepared
        assert set(variable_names) == {"H", "mf", "mf_error"}
        calls.append("materialize")
        return model_inputs

    def build(**kwargs: Any) -> Any:
        assert kwargs == {
            "prepared": prepared,
            "model_inputs": model_inputs,
            "run_spec": config.retained_run_spec(prepared),
            "model_builder": None,
            "likelihood_builder": likelihoods.likelihood_builder,
            "likelihood_kwargs": None,
            "preserve_legacy_likelihood": False,
            "legacy_unused_sigma_settings": None,
            "legacy_minimum_error_floor": False,
        }
        calls.append("build")
        return build_result

    def sample(*args: Any, **kwargs: Any) -> Any:
        assert args == (build_result, sampler)
        assert kwargs == {}
        calls.append("sample")
        return idata

    def make_result(**kwargs: Any) -> Any:
        assert kwargs["run_spec"].output.output_format == "inv_out"
        assert kwargs["likelihood_builder"] is likelihoods.likelihood_builder
        assert kwargs["model_build_result"] is build_result
        assert kwargs["idata"] is idata
        calls.append("result")
        return expected

    def make_outputs(**kwargs: Any) -> None:
        assert kwargs == {"result": expected, "prepared": prepared, "compatibility_output_chain": None}
        calls.append("output")
        return kwargs["result"]

    monkeypatch.setattr(rhime_runner.RhimeConfig, "from_params", classmethod(resolve))
    monkeypatch.setattr(rhime_runner.RhimeMergedData, "from_options", retrieve)
    monkeypatch.setattr(rhime_runner, "filter_observations", filter_observations)
    monkeypatch.setattr(rhime_runner, "prepare_basis", build_basis)
    monkeypatch.setattr(rhime_runner, "build_sensitivities", build_sensitivities)
    monkeypatch.setattr(rhime_runner, "assemble_rhime_inputs", assemble)
    monkeypatch.setattr(
        rhime_runner,
        "standard_model_input_names",
        lambda _actual, _model, **_kwargs: ("H", "mf", "mf_error"),
    )
    monkeypatch.setattr(rhime_runner, "materialize_pymc_inputs", materialize)
    monkeypatch.setattr(rhime_runner, "build_standard_rhime_model_result", build)
    monkeypatch.setattr(rhime_runner, "sample_rhime_model", sample)
    monkeypatch.setattr(rhime_runner, "make_standard_rhime_result", make_result)
    monkeypatch.setattr(rhime_runner, "make_standard_rhime_outputs", make_outputs)

    result = consumer_runner.run(species="ch4", output_format="inv_out")

    assert result is expected
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
def test_consumer_imports_only_public_supported_modules(module_path: str | None) -> None:
    """Consumer modules use public recipe or model-component modules."""
    assert module_path is not None
    source = Path(module_path).read_text(encoding="utf-8")
    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    for node in imports:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("openghg_inversions"):
            assert node.module in {
                "openghg_inversions.models.additive_sigma",
                "openghg_inversions.models.pollution_event",
                "openghg_inversions.observation_error",
                "openghg_inversions.rhime",
                "openghg_inversions.sigma",
            }
            assert all(not alias.name.startswith("_") for alias in node.names)
        elif isinstance(node, ast.Import):
            assert all(not alias.name.startswith("openghg_inversions") for alias in node.names)
