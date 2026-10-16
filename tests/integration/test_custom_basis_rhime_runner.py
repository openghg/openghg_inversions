"""Integration tests for the executable custom-basis RHIME example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.inversion_data import RhimeMergedData, SiteOptions
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.requests import load_params
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.contracts import OutputContract

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
    project_basis_path.touch()
    overrides = {"draws": 3}
    parsed_params = {
        "species": "ch4",
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
        site_data={site: xr.Dataset({"mf_error": ("time", [1.0]), "mf_repeatability": ("time", [0.0]), "mf_variability": ("time", [0.0])}) for site in config.data.site_options.sites},
        flux_data={"inventory": _basis_functions().flux.to_dataset(name="flux")},
        site_options=config.data.site_options,
    )
    filtered = {"MHD": merged.site_data["MHD"]}
    filtered_options = object()
    basis = _basis_functions()
    site_data = object()
    prepared = SimpleNamespace(
        inv_inputs=xr.Dataset({"mf": ("nmeasure", [1.0])}),
        basis_functions=basis,
        sites=("MHD",),
        averaging_period=("1h",),
        basis_artifact_source=basis.basis_artifact_source,
    )
    output_contract = OutputContract(variable_roles={"flux_scale": "x"}, supported_output_formats=("none", "inv_out"))
    build_result = SimpleNamespace(output_contract=output_contract)
    idata = xr.DataTree()
    model_inputs = prepared.inv_inputs
    priors = {"x": {"pdf": "normal", "mu": 1.0, "sigma": 0.2}}
    expected_result = InversionOutput(trace=idata, inv_inputs=prepared.inv_inputs, basis_functions=basis)
    calls: list[str] = []

    def parse_config(actual_config: str | Path) -> dict[str, Any]:
        """Record configuration decoding at the workflow boundary."""
        assert actual_config == config_file
        return parsed_params

    def resolve(cls, params: dict[str, Any], *, multisector: bool) -> Any:
        """Record recipe resolution after project options have been consumed."""
        assert params == load_params(overrides={"species": "ch4", **overrides})
        assert multisector is False
        calls.append("resolve")
        return config

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

    def build_project_basis(**kwargs: Any) -> BasisFunctions:
        assert kwargs["site_data"] is filtered
        assert kwargs["domain"] == config.data.domain
        xr.testing.assert_equal(kwargs["weighting_flux"], merged.flux_data["inventory"].flux)
        assert kwargs["nbasis"] == config.data.nbasis
        assert kwargs["project_basis_path"] == tmp_path / "external-basis.nc"
        assert kwargs["max_child_pca_eccentricity"] == 6.5
        assert kwargs["region_classes"] is None
        calls.append("project-basis")
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
        calls.append("build")
        return build_result

    def sample(*args: Any, **kwargs: Any) -> Any:
        """Record unchanged public sampling."""
        assert args == (build_result, sampler)
        assert kwargs == {}
        calls.append("sample")
        return idata

    def annotate(actual: Any, posterior: Any, **kwargs: Any) -> None:
        """Record prior and custom-likelihood provenance once after sampling."""
        assert actual is build_result
        assert posterior is idata
        assert kwargs["likelihood_builder"] is None
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

    monkeypatch.setattr(custom_basis_runner, "read_rhime_ini", parse_config)
    monkeypatch.setattr(custom_basis_runner.RhimeConfig, "from_params", classmethod(resolve))
    monkeypatch.setattr(RhimeMergedData, "from_options", retrieve)
    monkeypatch.setattr(custom_basis_runner, "filter_observations", filter_observations)
    monkeypatch.setattr(custom_basis_runner, "build_project_basis", build_project_basis)
    monkeypatch.setattr(custom_basis_runner, "build_sensitivities", build_sensitivities)
    monkeypatch.setattr(custom_basis_runner, "assemble_rhime_inputs", assemble)
    monkeypatch.setattr(custom_basis_runner, "materialize_standard_inputs", materialize)
    monkeypatch.setattr(custom_basis_runner, "build_standard_model", build)
    monkeypatch.setattr(custom_basis_runner, "sample_rhime_model", sample)
    monkeypatch.setattr(custom_basis_runner, "annotate_sample_trace", annotate)
    monkeypatch.setattr(custom_basis_runner, "make_rhime_output", make_result)
    monkeypatch.setattr(custom_basis_runner, "save_inversion_output", write_outputs)

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
        "annotate",
        "result",
        "output",
    ]


def test_project_basis_artifact_bypasses_calculation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Load a self-contained artifact without calculating a new basis."""
    stored = _basis_functions(artifact_source="external-project")
    artifact_path = tmp_path / "project-basis.nc"
    stored.save(artifact_path)
    def fail_calculation(actual: Any, **kwargs: Any) -> BasisFunctions:
        """Fail if the cached ingress route tries to calculate a basis."""
        raise AssertionError(f"unexpected basis calculation for {actual!r} with {kwargs!r}")

    monkeypatch.setattr(
        custom_basis_runner, "load_basis_support_map",
        lambda *args, **kwargs: pytest.fail("artifact reuse must not load generation maps"),
    )
    monkeypatch.setattr(custom_basis_runner, "_guarded_basis", fail_calculation)

    loaded = custom_basis_runner.build_project_basis(
        site_data={}, runtime_flux=2.0 * stored.flux, weighting_flux=stored.flux,
        domain="EUROPE",
        project_basis_path=artifact_path,
    )

    assert isinstance(loaded, BasisFunctions)
    assert loaded.basis_artifact_source == "external-project"
    xr.testing.assert_identical(loaded.operator.basis_matrix, stored.operator.basis_matrix)
    xr.testing.assert_equal(loaded.flux, 2.0 * stored.flux)
    xr.testing.assert_equal(BasisFunctions.load(artifact_path).flux, stored.flux)
    state = xr.ones_like(loaded.operator.basis_matrix.isel(lat=0, lon=0, drop=True))
    reconstructed = loaded.state_to_flux(state)
    expected = 2.0 * stored.flux
    xr.testing.assert_equal(reconstructed, expected.transpose(*reconstructed.dims))
    response = 5.0 * expected
    xr.testing.assert_equal(loaded.sensitivity(response), stored.sensitivity(response))
    xr.testing.assert_equal(loaded.sensitivity(response), 2.0 * stored.sensitivity(5.0 * stored.flux))



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
    }

    def basis_weights(
        site_data: Any,
        weighting_flux: Any,
        *,
        abs_flux: bool,
    ) -> xr.DataArray:
        """Return deterministic weights at the public custom-basis boundary."""
        assert site_data is merged.site_data
        assert weighting_flux is expected_basis.flux
        assert abs_flux is True
        return weights

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
        assert kwargs["flux"] is expected_basis.flux
        assert kwargs["basis_flat"].name == "basis"
        assert kwargs["basis_flat"].dtype == np.dtype(np.int16)
        xr.testing.assert_equal(kwargs["basis_flat"], generated_labels.astype(np.int16).rename("basis"))
        assert kwargs["basis_flat"].attrs == expected_metadata
        assert kwargs["metadata"] == expected_metadata
        return expected_basis

    monkeypatch.setattr(custom_basis_runner, "basis_weights_from_data", basis_weights)
    monkeypatch.setattr(custom_basis_runner, "region_constrained_basis", build_labels)
    monkeypatch.setattr(custom_basis_runner, "basis_functions_from_flat_basis", retain_basis)

    actual = custom_basis_runner.build_project_basis(
        site_data=merged.site_data, runtime_flux=expected_basis.flux, weighting_flux=expected_basis.flux,
        domain=data_args["domain"],
        nbasis=data_args["nbasis"],
        region_classes=country_classes,
        max_child_pca_eccentricity=7.5,
    )

    assert actual is expected_basis
    assert isinstance(actual, BasisFunctions)
    assert bool(weights.isnull().any())
    xr.testing.assert_identical(
        country_classes,
        xr.DataArray([[0, 4], [9, 0]], dims=weights.dims, coords=weights.coords, name="country"),
    )


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
        site_data={site: xr.Dataset({"mf_error": ("time", [1.0]), "mf_repeatability": ("time", [0.0]), "mf_variability": ("time", [0.0])}) for site in config.data.site_options.sites},
        flux_data={"inventory": _basis_functions().flux.to_dataset(name="flux")},
        site_options=config.data.site_options,
    )
    filtered = {"MHD": merged.site_data["MHD"]}
    filtered_options = object()
    incompatible_basis = object()

    monkeypatch.setattr(
        custom_basis_runner.RhimeConfig,
        "from_params",
        classmethod(lambda cls, params, *, multisector: config),
    )
    monkeypatch.setattr(RhimeMergedData, "from_options",
        lambda **kwargs: merged,
    )
    monkeypatch.setattr(
        custom_basis_runner,
        "filter_observations",
        lambda actual, site_options, *, filters: (filtered, filtered_options),
    )
    monkeypatch.setattr(
        custom_basis_runner,
        "build_project_basis",
        lambda **kwargs: incompatible_basis,
    )

    def reject_incompatible_basis(actual: Any, actual_basis: Any, **kwargs: Any) -> Any:
        assert actual is filtered
        assert actual_basis is incompatible_basis
        raise TypeError("build_sensitivities requires compatible BasisFunctions")

    monkeypatch.setattr(
        custom_basis_runner,
        "build_sensitivities",
        reject_incompatible_basis,
    )

    with pytest.raises(
        TypeError,
        match="build_sensitivities requires compatible BasisFunctions",
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
        project_basis_path=str(artifact_path), basis_algorithm=algorithm,
        max_child_pca_eccentricity=7.0,
    )
    merged = RhimeMergedData(
        site_data={"TAC": xr.Dataset({
            "mf_error": ("time", [1.0]),
            "mf_repeatability": ("time", [0.0]),
            "mf_variability": ("time", [0.0]),
        })},
        flux_data={"inventory": stored.flux.to_dataset(name="flux")},
        site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"),
    )
    monkeypatch.setattr(RhimeMergedData, "from_options", lambda **kwargs: merged)
    monkeypatch.setattr(
        custom_basis_runner, "filter_observations", lambda value, site_options, **kwargs: (value, site_options),
    )
    monkeypatch.setattr(
        custom_basis_runner, "_guarded_basis", lambda *args, **kwargs: pytest.fail("unexpected fitting"),
    )

    class ReachedSensitivities(Exception):
        pass

    def check_loaded(actual, basis, **kwargs):
        assert actual.keys() == merged.site_data.keys()
        xr.testing.assert_identical(basis.operator.basis_matrix, stored.operator.basis_matrix)
        assert basis.basis_artifact_source == "external-project"
        raise ReachedSensitivities

    monkeypatch.setattr(custom_basis_runner, "build_sensitivities", check_loaded)
    if from_file:
        config_file = tmp_path / "project.ini"
        config_file.write_text("[PROJECT]\n" + "\n".join(f"{k} = {v!r}" for k, v in params.items()))
        request = {"config_file": config_file}
    else:
        request = params
    with pytest.raises(ReachedSensitivities):
        custom_basis_runner.run_custom_rhime(**request)


def test_custom_basis_missing_selected_map_fails_before_acquisition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The project validates its selected support map before store access."""
    missing_map = tmp_path / "missing-classes.nc"
    monkeypatch.setattr(
        RhimeMergedData,
        "from_options",
        lambda **kwargs: pytest.fail("a missing selected map must fail before acquisition"),
    )
    with pytest.raises(FileNotFoundError):
        custom_basis_runner.run_custom_rhime(
            species="ch4", sites=["TAC"], domain="EUROPE", averaging_period="1h",
            start_date="2019-01-01", end_date="2019-02-01", output_name="missing-map",
            output_format="none", flux_sources=["inventory"], landsea_path=missing_map,
        )
