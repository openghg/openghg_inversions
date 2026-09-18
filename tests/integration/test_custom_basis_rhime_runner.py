"""Integration tests for the executable custom-basis RHIME example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr

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


@pytest.mark.parametrize("reload_merged_data", [False, True])
def test_custom_basis_runner_replaces_only_basis_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reload_merged_data: bool,
) -> None:
    """Carry acquisition and reload through output with only a custom basis stage."""
    config_file = tmp_path / "rhime.ini"
    config_file.write_text('[RHIME.OUTPUT]\noutput_format = "none"\n', encoding="utf-8")
    project_basis_path = tmp_path / "external-basis.nc"
    overrides = {"reload_merged_data": reload_merged_data, "draws": 3}
    parsed_params = {
        "from_config": True,
        "max_child_pca_eccentricity": 6.5,
        **overrides,
    }
    resolved_data_args = {"reload_merged_data": reload_merged_data}
    sampler = object()
    initial_spec = SimpleNamespace(model=SimpleNamespace(aggregation_error_mode="diagonal"))
    aligned_spec = SimpleNamespace(model=SimpleNamespace(aggregation_error_mode="low_rank"))
    setup = SimpleNamespace(data_args=resolved_data_args, run_spec=initial_spec, sampler=sampler)

    merged = object()
    filtered = object()
    basis = _basis_functions()
    site_data = object()
    prepared = SimpleNamespace(
        inv_inputs=xr.Dataset({"mf": ("nmeasure", [1.0])}),
        basis_functions=basis,
        sites=("MHD",),
        basis_artifact_source=basis.basis_artifact_source,
    )
    model_inputs = xr.Dataset({"mf": ("nmeasure", [1.0])})
    build_result = object()
    idata = object()
    expected_result = object()
    calls: list[str] = []
    workflow_data_args: dict[str, Any] | None = None

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
        assert "max_child_pca_eccentricity" not in params
        assert multisector is False
        calls.append("resolve")
        return setup

    def retrieve(actual_data_args: dict[str, Any], *, multisector: bool) -> Any:
        """Record ordinary acquisition or controlled merged-data reload."""
        nonlocal workflow_data_args
        workflow_data_args = actual_data_args
        assert actual_data_args is resolved_data_args
        assert actual_data_args["reload_merged_data"] is reload_merged_data
        assert "project_basis_path" not in actual_data_args
        assert multisector is False
        calls.append("retrieve")
        return merged

    def filter_observations(actual: Any, actual_data_args: dict[str, Any]) -> Any:
        """Record unchanged public observation filtering."""
        assert actual is merged
        assert actual_data_args is workflow_data_args
        calls.append("filter")
        return filtered

    def build_project_basis(
        actual: Any,
        actual_data_args: dict[str, Any],
        *,
        project_basis_path: str | Path | None,
        max_child_pca_eccentricity: float,
    ) -> BasisFunctions:
        """Record the sole custom replacement stage and its supported result."""
        assert actual is filtered
        assert actual_data_args == workflow_data_args
        assert actual_data_args is not workflow_data_args
        assert "project_basis_path" not in actual_data_args
        assert "max_child_pca_eccentricity" not in actual_data_args
        assert project_basis_path == tmp_path / "external-basis.nc"
        assert max_child_pca_eccentricity == 6.5
        calls.append("project-basis")
        return basis

    def build_sensitivities(
        actual: Any,
        actual_basis: BasisFunctions,
        actual_data_args: dict[str, Any],
        *,
        multisector: bool,
    ) -> Any:
        """Record unchanged sensitivity construction with the custom basis."""
        assert actual is filtered
        assert actual_basis is basis
        assert isinstance(actual_basis, BasisFunctions)
        assert actual_data_args is workflow_data_args
        assert multisector is False
        calls.append("sensitivities")
        return site_data

    def assemble(
        actual: Any,
        actual_basis: BasisFunctions,
        actual_site_data: Any,
        actual_data_args: dict[str, Any],
    ) -> Any:
        """Record unchanged labelled-input assembly with the custom basis."""
        assert actual is filtered
        assert actual_basis is basis
        assert actual_site_data is site_data
        assert actual_data_args is workflow_data_args
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
        """Record the unchanged standard model stage."""
        assert kwargs == {
            "prepared": prepared,
            "model_inputs": model_inputs,
            "run_spec": aligned_spec,
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
        assert kwargs["run_spec"] is aligned_spec
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

    monkeypatch.setattr(custom_basis_runner, "params_from_config", parse_config)
    monkeypatch.setattr(custom_basis_runner, "resolve_rhime_options", resolve)
    monkeypatch.setattr(custom_basis_runner, "retrieve_or_reload_rhime_data", retrieve)
    monkeypatch.setattr(custom_basis_runner, "filter_rhime_observations", filter_observations)
    monkeypatch.setattr(custom_basis_runner, "build_project_basis", build_project_basis)
    monkeypatch.setattr(custom_basis_runner, "build_rhime_sensitivities", build_sensitivities)
    monkeypatch.setattr(custom_basis_runner, "assemble_rhime_inputs", assemble)
    monkeypatch.setattr(custom_basis_runner, "with_prepared_rhime_sites", align)
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
        "align",
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

    def fail_calculation(actual: Any, data_args: dict[str, Any]) -> BasisFunctions:
        """Fail if the cached ingress route tries to calculate a basis."""
        raise AssertionError(f"unexpected basis calculation for {actual!r} with {data_args!r}")

    monkeypatch.setattr(custom_basis_runner, "_guarded_basis", fail_calculation)

    loaded = custom_basis_runner.build_project_basis(
        merged,
        {},
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
    fp_all = object()
    merged = SimpleNamespace(fp_all=fp_all)
    data_args = {
        "species": "ch4",
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "flux_sources": ["inventory"],
        "nbasis": 8,
        "country_directory": Path("/project/country-classes"),
    }

    def basis_weights(
        actual_fp_all: Any,
        emissions_name: list[str],
        *,
        abs_flux: bool,
    ) -> xr.DataArray:
        """Return deterministic weights at the public custom-basis boundary."""
        assert actual_fp_all is fp_all
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
            "openghg_inversions:project_basis_weights": ("basis_weights_from_fp_all_abs_flux_normalized"),
        }
        assert kwargs["fp_all"] is fp_all
        assert kwargs["basis_flat"].name == "basis"
        assert kwargs["basis_flat"].dtype == np.dtype(np.int16)
        xr.testing.assert_equal(kwargs["basis_flat"], generated_labels.astype(np.int16).rename("basis"))
        assert kwargs["basis_flat"].attrs == expected_metadata
        assert kwargs["metadata"] == expected_metadata
        return expected_basis

    monkeypatch.setattr(custom_basis_runner, "basis_weights_from_fp_all", basis_weights)
    monkeypatch.setattr(custom_basis_runner, "load_country_region_classes", load_classes)
    monkeypatch.setattr(custom_basis_runner, "region_constrained_basis", build_labels)
    monkeypatch.setattr(custom_basis_runner, "basis_functions_from_fp_all_flat_basis", retain_basis)

    actual = custom_basis_runner.build_project_basis(
        merged,
        data_args,
        max_child_pca_eccentricity=7.5,
    )

    assert actual is expected_basis
    assert isinstance(actual, BasisFunctions)
    assert bool(weights.isnull().any())


def test_incompatible_project_basis_failure_remains_owned_by_sensitivity_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let the unchanged downstream stage explain an incompatible custom basis."""
    setup = SimpleNamespace(data_args={}, run_spec=object(), sampler=object())
    merged = object()
    filtered = object()
    incompatible_basis = object()

    monkeypatch.setattr(
        custom_basis_runner,
        "resolve_rhime_options",
        lambda *, params, multisector: setup,
    )
    monkeypatch.setattr(
        custom_basis_runner,
        "retrieve_or_reload_rhime_data",
        lambda data_args, *, multisector: merged,
    )
    monkeypatch.setattr(
        custom_basis_runner,
        "filter_rhime_observations",
        lambda actual, data_args: filtered,
    )
    monkeypatch.setattr(
        custom_basis_runner,
        "build_project_basis",
        lambda actual, data_args, *, project_basis_path, max_child_pca_eccentricity: incompatible_basis,
    )

    def reject_incompatible_basis(
        actual: Any,
        actual_basis: Any,
        data_args: dict[str, Any],
        *,
        multisector: bool,
    ) -> Any:
        """Represent validation owned by the public sensitivity stage."""
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
