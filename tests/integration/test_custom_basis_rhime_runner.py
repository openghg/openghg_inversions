"""Integration tests for the executable custom-basis RHIME example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.inversion_data import RhimeMergedData
from openghg_inversions.rhime.params import RhimeConfig

from openghg_inversions.basis.basis_functions import BasisFunctions


_RUNNER_PATH = Path(__file__).parents[2] / "examples" / "rhime_customisation" / "custom_basis_runner.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("custom_basis_rhime_runner", _RUNNER_PATH)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
custom_basis_runner = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(custom_basis_runner)


def _basis_functions(*, artifact_source: str = "project-generated") -> BasisFunctions:
    """Build a small supported basis object for orchestration tests."""
    basis_flat = xr.DataArray(
        [[1, 1], [2, 2]],
        dims=("lat", "lon"),
        coords={"lat": [50.0, 51.0], "lon": [-2.0, -1.0]},
        name="basis",
    )
    flux = xr.DataArray(
        np.ones((2, 2)),
        dims=("lat", "lon"),
        coords=basis_flat.coords,
        name="flux",
    )
    return BasisFunctions.from_flat_basis(
        basis_flat=basis_flat,
        flux=flux,
        metadata={"openghg_inversions:basis_artifact_source": artifact_source},
    )


def test_custom_basis_runner_replaces_only_basis_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Carry in-memory acquisition through output with a custom basis stage."""
    config_file = tmp_path / "rhime.ini"
    config_file.write_text('[RHIME.OUTPUT]\noutput_format = "none"\n', encoding="utf-8")
    project_basis_path = tmp_path / "external-basis.nc"
    overrides = {"draws": 3}
    parsed_params = {
        "from_config": True,
        "max_child_pca_eccentricity": 6.5,
        **overrides,
    }
    config = RhimeConfig.from_params(
        params=dict(species="ch4", sites=["TAC", "MHD"], domain="EUROPE",
                    averaging_period="1h", start_date="2019-01-01", end_date="2019-02-01",
                    output_name="example", output_format="none", flux_sources=["inventory"],
                    draws=3),
        multisector=False,
    )
    sampler = config.sampler

    merged = RhimeMergedData(
        site_data={site: xr.Dataset() for site in config.site_options.sites},
        flux_data={}, site_options=config.site_options,
    )
    filtered = object()
    basis = _basis_functions()
    site_data = object()
    prepared = SimpleNamespace(
        inv_inputs=xr.Dataset({"mf": ("nmeasure", [1.0])}),
        basis_functions=basis,
        sites=("MHD",),
        averaging_period=("1h",),
        basis_artifact_source=basis.basis_artifact_source,
    )
    model_inputs = xr.Dataset({"mf": ("nmeasure", [1.0])})
    build_result = object()
    idata = object()
    expected_result = object()
    calls: list[str] = []

    def parse_config(actual_config: str | Path) -> dict[str, Any]:
        """Record configuration decoding at the workflow boundary."""
        assert actual_config == config_file
        return parsed_params

    def resolve(cls, params: dict[str, Any], *, multisector: bool) -> Any:
        """Record recipe resolution after project options have been consumed."""
        assert params == {"from_config": True, **overrides}
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

    def build_project_basis(actual: Any, **kwargs: Any) -> BasisFunctions:
        assert actual is filtered
        assert kwargs["domain"] == config.domain
        assert kwargs["flux_sources"] == config.flux_sources
        assert kwargs["nbasis"] == config.nbasis
        assert kwargs["project_basis_path"] == tmp_path / "external-basis.nc"
        assert kwargs["max_child_pca_eccentricity"] == 6.5
        calls.append("project-basis")
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
        assert set(variable_names) >= {"H", "mf", "mf_error", "min_error"}
        calls.append("materialize")
        return model_inputs

    def build(**kwargs: Any) -> Any:
        """Record the unchanged standard model stage."""
        assert kwargs == {
            "prepared": prepared,
            "model_inputs": model_inputs,
            "run_spec": config.retained_run_spec(prepared),
        }
        calls.append("build")
        return build_result

    def sample(*args: Any, **kwargs: Any) -> Any:
        """Record unchanged public sampling."""
        assert args == (build_result, sampler)
        assert kwargs == {}
        calls.append("sample")
        return idata

    def make_result(**kwargs: Any) -> Any:
        """Record the supported output stage and retained custom basis."""
        assert kwargs["prepared"] is prepared
        assert kwargs["prepared"].basis_functions is basis
        assert kwargs["run_spec"].sites == ("MHD",)
        assert kwargs["run_spec"].model is config.model
        assert kwargs["sampler"] is sampler
        assert kwargs["model_build_result"] is build_result
        assert kwargs["idata"] is idata
        assert kwargs["build_and_sample_seconds"] >= 0.0
        assert set(kwargs) == {
            "prepared",
            "run_spec",
            "sampler",
            "model_build_result",
            "idata",
            "build_and_sample_seconds",
        }
        calls.append("result")
        return expected_result

    def make_outputs(**kwargs: Any) -> None:
        assert kwargs == {"result": expected_result, "prepared": prepared}
        calls.append("outputs")

    monkeypatch.setattr(custom_basis_runner, "read_rhime_ini", parse_config)
    monkeypatch.setattr(custom_basis_runner.RhimeConfig, "from_params", classmethod(resolve))
    monkeypatch.setattr(custom_basis_runner.RhimeMergedData, "from_options", retrieve)
    monkeypatch.setattr(custom_basis_runner, "filter_rhime_observations", filter_observations)
    monkeypatch.setattr(custom_basis_runner, "build_project_basis", build_project_basis)
    monkeypatch.setattr(custom_basis_runner, "build_rhime_sensitivities", build_sensitivities)
    monkeypatch.setattr(custom_basis_runner, "assemble_rhime_inputs", assemble)
    monkeypatch.setattr(
        custom_basis_runner,
        "standard_model_input_names",
        lambda _actual, _model: ("H", "mf", "mf_error", "min_error"),
    )
    monkeypatch.setattr(custom_basis_runner, "materialize_pymc_inputs", materialize)
    monkeypatch.setattr(custom_basis_runner, "build_standard_rhime_model_result", build)
    monkeypatch.setattr(custom_basis_runner, "sample_rhime_model", sample)
    monkeypatch.setattr(custom_basis_runner, "make_standard_rhime_result", make_result)
    monkeypatch.setattr(custom_basis_runner, "make_standard_rhime_outputs", make_outputs)

    result = custom_basis_runner.run_custom_rhime(
        config_file=config_file,
        project_basis_path=project_basis_path,
        **overrides,
    )

    assert result is expected_result
    assert calls == [
        "resolve",
        "retrieve",
        "filter",
        "project-basis",
        "sensitivities",
        "assemble",
        "materialize",
        "build",
        "sample",
        "result",
        "outputs",
    ]


def test_project_basis_artifact_bypasses_calculation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Load a self-contained artifact without calculating a new basis."""
    stored = _basis_functions(artifact_source="external-project")
    artifact_path = tmp_path / "project-basis.nc"
    stored.save(artifact_path)
    merged = SimpleNamespace(
        fp_all={
            ".flux": {"current-inventory": 2.0 * stored.flux},
            ".split_by_sectors": False,
        }
    )

    def fail_calculation(actual: Any, **kwargs: Any) -> BasisFunctions:
        """Fail if the cached ingress route tries to calculate a basis."""
        raise AssertionError(f"unexpected basis calculation for {actual!r} with {kwargs!r}")

    monkeypatch.setattr(custom_basis_runner, "_guarded_basis", fail_calculation)

    loaded = custom_basis_runner.build_project_basis(
        merged,
        domain="EUROPE",
        project_basis_path=artifact_path,
    )

    assert isinstance(loaded, BasisFunctions)
    assert loaded.basis_artifact_source == "external-project"
    xr.testing.assert_identical(loaded.operator.basis_matrix, stored.operator.basis_matrix)
    xr.testing.assert_identical(loaded.flux, stored.flux)


def test_generated_project_basis_uses_guarded_connected_inertial_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a retained basis from the guarded connected-inertial policy."""
    weights = xr.DataArray(
        [[np.nan, 2.0], [4.0, 1.0]],
        dims=("lat", "lon"),
        coords={"lat": [50.0, 51.0], "lon": [-1.0, 1.0]},
        name="weight",
    )
    country_classes = xr.DataArray(
        [[0, 4], [9, 0]],
        dims=weights.dims,
        coords=weights.coords,
        name="country",
    )
    generated_labels = xr.DataArray(
        [[0, 1], [2, 0]],
        dims=weights.dims,
        coords=weights.coords,
        name="raw_labels",
    )
    expected_basis = _basis_functions()
    from openghg_inversions.inversion_data import RhimeMergedData, SiteOptions
    merged = RhimeMergedData(site_data={"TAC": xr.Dataset()}, flux_data={},
        site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"))
    data_args = {
        "species": "ch4",
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "flux_sources": ["inventory"],
        "nbasis": 8,
        "country_directory": Path("/project/country-classes"),
    }

    def basis_weights(
        site_data: Any,
        flux_data: Any,
        emissions_name: list[str],
        *,
        abs_flux: bool,
    ) -> xr.DataArray:
        """Return deterministic weights at the public custom-basis boundary."""
        assert site_data is merged.site_data
        assert flux_data is merged.flux_data
        assert emissions_name == ["inventory"]
        assert abs_flux is True
        return weights

    def load_classes(domain: str, country_directory: str | Path | None) -> xr.DataArray:
        """Return deterministic country codes at the public class-map boundary."""
        assert domain == "EUROPE"
        assert country_directory == Path("/project/country-classes")
        return country_classes

    def build_labels(
        normalized_weights: xr.DataArray,
        region_classes: xr.DataArray,
        nbasis: int,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Validate the guarded connected-inertial algorithm composition."""
        xr.testing.assert_identical(
            normalized_weights,
            xr.DataArray(
                [[0.0, 0.5], [1.0, 0.25]],
                dims=weights.dims,
                coords=weights.coords,
                name="weight",
            ),
        )
        assert region_classes.name == "basis_class"
        np.testing.assert_array_equal(region_classes, [["ocean", "land"], ["land", "ocean"]])
        assert nbasis == 8
        assert kwargs.keys() == {"allocation", "min_regions_per_class", "split_strategy"}
        assert kwargs["allocation"] == "weight"
        assert kwargs["min_regions_per_class"] == 1

        strategy = kwargs["split_strategy"]
        assert isinstance(strategy, custom_basis_runner.ConnectedComponentSplitStrategy)
        assert strategy.connectivity == 1
        greedy = strategy.split_strategy
        assert isinstance(greedy, custom_basis_runner.GreedySplitStrategy)
        partition_step = greedy.split_step
        assert isinstance(partition_step, custom_basis_runner.ConnectedComponentPartitionStep)
        assert partition_step.connectivity == 1
        inertial_step = partition_step.split_step
        assert isinstance(inertial_step, custom_basis_runner.InertialSplitStep)
        assert inertial_step.balanced is True
        assert isinstance(inertial_step.geometry, custom_basis_runner.LatLonGridGeometry)
        guard = greedy.split_acceptance
        assert isinstance(guard, custom_basis_runner.MaxChildPCAEccentricity)
        assert guard.max_child_pca_eccentricity == 7.5
        assert guard.geometry is inertial_step.geometry
        return generated_labels

    def retain_basis(**kwargs: Any) -> BasisFunctions:
        """Validate conversion of flat labels to the supported retained object."""
        expected_metadata = {
            "openghg_inversions:basis_artifact_source": "project-guarded",
            "openghg_inversions:project_basis_strategy": ("connected_component_balanced_inertial"),
            "openghg_inversions:project_basis_connectivity": 1,
            "openghg_inversions:project_basis_max_child_pca_eccentricity": 7.5,
            "openghg_inversions:project_basis_class_policy": "land_ocean",
            "openghg_inversions:project_basis_weights": ("basis_weights_from_data_abs_flux_normalized"),
        }
        assert kwargs["flux_data"] is merged.flux_data
        assert kwargs["split_by_sectors"] is merged.split_by_sectors
        assert kwargs["basis_flat"].name == "basis"
        assert kwargs["basis_flat"].dtype == np.dtype(np.int16)
        xr.testing.assert_equal(kwargs["basis_flat"], generated_labels.astype(np.int16).rename("basis"))
        assert kwargs["basis_flat"].attrs == expected_metadata
        assert kwargs["metadata"] == expected_metadata
        return expected_basis

    monkeypatch.setattr(custom_basis_runner, "basis_weights_from_data", basis_weights)
    monkeypatch.setattr(custom_basis_runner, "load_country_region_classes", load_classes)
    monkeypatch.setattr(custom_basis_runner, "region_constrained_basis", build_labels)
    monkeypatch.setattr(custom_basis_runner, "basis_functions_from_flat_basis", retain_basis)

    actual = custom_basis_runner.build_project_basis(
        merged,
        domain=data_args["domain"],
        flux_sources=data_args["flux_sources"],
        nbasis=data_args["nbasis"],
        country_directory=data_args["country_directory"],
        max_child_pca_eccentricity=7.5,
    )

    assert actual is expected_basis
    assert isinstance(actual, BasisFunctions)
    assert bool(weights.isnull().any())


def test_incompatible_project_basis_failure_remains_owned_by_sensitivity_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let the unchanged downstream stage explain an incompatible custom basis."""
    config = RhimeConfig.from_params(
        params=dict(species="ch4", sites=["TAC", "MHD"], domain="EUROPE",
                    averaging_period="1h", start_date="2019-01-01", end_date="2019-02-01",
                    output_name="example", output_format="none", flux_sources=["inventory"],
                    draws=3),
        multisector=False,
    )
    merged = RhimeMergedData(
        site_data={site: xr.Dataset() for site in config.site_options.sites},
        flux_data={}, site_options=config.site_options,
    )
    filtered = object()
    incompatible_basis = object()

    monkeypatch.setattr(
        custom_basis_runner.RhimeConfig,
        "from_params",
        classmethod(lambda cls, params, *, multisector: config),
    )
    monkeypatch.setattr(custom_basis_runner.RhimeMergedData, "from_options",
        lambda **kwargs: merged,
    )
    monkeypatch.setattr(
        custom_basis_runner,
        "filter_rhime_observations",
        lambda actual, *, filters: filtered,
    )
    monkeypatch.setattr(
        custom_basis_runner,
        "build_project_basis",
        lambda actual, **kwargs: incompatible_basis,
    )

    def reject_incompatible_basis(actual: Any, actual_basis: Any, **kwargs: Any) -> Any:
        assert actual is filtered
        assert actual_basis is incompatible_basis
        raise TypeError("build_rhime_sensitivities requires compatible BasisFunctions")

    monkeypatch.setattr(
        custom_basis_runner,
        "build_rhime_sensitivities",
        reject_incompatible_basis,
    )

    with pytest.raises(
        TypeError,
        match="build_rhime_sensitivities requires compatible BasisFunctions",
    ):
        custom_basis_runner.run_custom_rhime(species="ch4")


@pytest.mark.parametrize("from_file", [False, True])
@pytest.mark.parametrize("algorithm", [None, "project-only"])
def test_custom_basis_runner_resolves_before_loading_project_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, from_file: bool, algorithm: str | None,
) -> None:
    """A project artifact bypasses unused built-in basis validation and fitting."""
    stored = _basis_functions(artifact_source="external-project")
    artifact_path = tmp_path / "project-basis.nc"
    stored.save(artifact_path)
    params = dict(
        species="ch4", sites=["TAC"], domain="EUROPE", averaging_period="1h",
        start_date="2019-01-01", end_date="2019-02-01", output_name="artifact",
        output_format="none", flux_sources=["inventory"],
        project_basis_path=str(artifact_path), basis_algorithm=algorithm, fp_basis_case=None,
        max_child_pca_eccentricity=7.0,
    )
    merged = SimpleNamespace(fp_all={".flux": {"inventory": stored.flux}, ".split_by_sectors": False})
    monkeypatch.setattr(custom_basis_runner.RhimeMergedData, "from_options", lambda **kwargs: merged)
    monkeypatch.setattr(custom_basis_runner, "filter_rhime_observations", lambda value, **kwargs: value)
    monkeypatch.setattr(
        custom_basis_runner, "_guarded_basis", lambda *args, **kwargs: pytest.fail("unexpected fitting"),
    )

    class ReachedSensitivities(Exception):
        pass

    def check_loaded(actual, basis, **kwargs):
        assert actual is merged
        xr.testing.assert_identical(basis.operator.basis_matrix, stored.operator.basis_matrix)
        assert basis.basis_artifact_source == "external-project"
        raise ReachedSensitivities

    monkeypatch.setattr(custom_basis_runner, "build_rhime_sensitivities", check_loaded)
    if from_file:
        config_file = tmp_path / "project.ini"
        config_file.write_text("[PROJECT]\n" + "\n".join(f"{k} = {v!r}" for k, v in params.items()))
        request = {"config_file": config_file}
    else:
        request = params
    with pytest.raises(ReachedSensitivities):
        custom_basis_runner.run_custom_rhime(**request)
