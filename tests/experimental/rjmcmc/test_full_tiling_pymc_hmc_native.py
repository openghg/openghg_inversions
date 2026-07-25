"""Tests for the frozen-input mobile PyMC HMC native driver."""

from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from typing import Any, Literal

import numpy as np
import pytest
import xarray as xr


_DRIVER_PATH = Path(__file__).parents[3] / "examples" / "rjmcmc" / "full_tiling_pymc_hmc_native.py"
_X64_CHILD_ENV = "OPENGHG_INVERSIONS_PYMC_HMC_NATIVE_DRIVER_X64_CHILD"
_IS_X64_CHILD = os.environ.get(_X64_CHILD_ENV) == "1"
_requires_x64_child = pytest.mark.skipif(
    not _IS_X64_CHILD,
    reason="PyMC driver assertions execute in the isolated float64 child",
)


def _pytensor_flags_with_float64(flags: str, *, compiledir: Path) -> str:
    """Return PyTensor flags with isolated float64 compilation."""
    retained = []
    for item in flags.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name not in {"base_compiledir", "floatX", "warn_float64"}:
            retained.append(stripped)
    return ",".join(
        (
            "floatX=float64",
            "warn_float64=ignore",
            f"base_compiledir={compiledir}",
            *retained,
        )
    )


def _run_x64_test_file(tmp_path: Path) -> None:
    """Run all PyMC driver assertions in one fresh float64 subprocess."""
    environment = os.environ.copy()
    environment[_X64_CHILD_ENV] = "1"
    environment["PYTENSOR_FLAGS"] = _pytensor_flags_with_float64(
        environment.get("PYTENSOR_FLAGS", ""),
        compiledir=tmp_path / "pytensor",
    )
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(Path(__file__).resolve())],
        cwd=Path(__file__).parents[3],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        "isolated native PyMC HMC driver tests failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.skipif(_IS_X64_CHILD, reason="parent-only subprocess dispatch")
def test_driver_cases_use_a_fresh_float64_subprocess(tmp_path: Path) -> None:
    """Driver cases cannot inherit process-global float32 from other tests."""
    _run_x64_test_file(tmp_path)


@pytest.fixture(scope="module")
def hmc_driver() -> ModuleType:
    """Load the example driver without invoking its command-line entry point."""
    specification = importlib.util.spec_from_file_location(
        "full_tiling_pymc_hmc_native",
        _DRIVER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the native PyMC HMC example.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_frozen_input(path: Path) -> None:
    """Write exact-closure data whose first fresh mass is not a log/exp fixed point."""
    sensitivity = np.arange(1.0, 13.0).reshape(3, 2, 2)
    outer = np.arange(18.0).reshape(3, 6) / 8.0
    boundary = np.array([4.0, 5.0, 6.0])
    dataset = xr.Dataset(
        {
            "fp_x_flux": (
                ("lon", "nmeasure", "lat"),
                sensitivity.transpose(2, 0, 1),
            ),
            "mf": (
                "nmeasure",
                boundary + sensitivity.sum(axis=(1, 2)) + outer.sum(axis=1),
            ),
            "mf_error": ("nmeasure", np.ones(3)),
            "nominal_weight": (
                ("lon", "lat"),
                np.array(
                    [
                        [1.0 / 16.0, 1.0 / 16.0],
                        [1.0 / 4.0, 5.0 / 8.0],
                    ],
                ).T,
            ),
            "outer_design": (("outer_region", "nmeasure"), outer.T),
            "YaprioriBC": ("nmeasure", boundary),
        },
        coords={
            "nmeasure": ["obs-a", "obs-b", "obs-c"],
            "lat": [50.0, 51.0],
            "lon": [-2.0, -1.0],
            "outer_region": [f"region-{index}" for index in range(6)],
        },
    )
    dataset.to_netcdf(path, engine="h5netcdf")


def _arguments(
    driver: ModuleType,
    input_path: Path,
    output_path: Path,
    *,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """Return the complete explicit tiny-run CLI contract."""
    calibration_path = input_path.with_name("calibration.json")
    development = [
        {
            "role": "development-nominal",
            "initializer": "largest-nominal",
            "topology_seed": None,
            "master_seed": 83003,
            "topology_sha256": "c" * 64,
            "topology_precision_sha256": "1" * 64,
            "sweeps": 500,
            "mean_acceptance": 0.75,
            "divergences": 0,
            "finite_scientific_endpoints": True,
            "finite_transformed_endpoints": True,
            "accepted_nonzero_displacement": True,
        },
        {
            "role": "development-a",
            "initializer": "random-recursive",
            "topology_seed": 42003,
            "master_seed": 83004,
            "topology_sha256": "d" * 64,
            "topology_precision_sha256": "2" * 64,
            "sweeps": 500,
            "mean_acceptance": 0.8,
            "divergences": 0,
            "finite_scientific_endpoints": True,
            "finite_transformed_endpoints": True,
            "accepted_nonzero_displacement": True,
        },
        {
            "role": "development-b",
            "initializer": "random-recursive",
            "topology_seed": 42004,
            "master_seed": 83005,
            "topology_sha256": "e" * 64,
            "topology_precision_sha256": "3" * 64,
            "sweeps": 500,
            "mean_acceptance": 0.78,
            "divergences": 0,
            "finite_scientific_endpoints": True,
            "finite_transformed_endpoints": True,
            "accepted_nonzero_displacement": True,
        },
    ]
    held_out = [
        {
            "role": "held-out-a",
            "initializer": "random-recursive",
            "topology_seed": 42005,
            "master_seed": 74003,
            "topology_sha256": "f" * 64,
            "topology_precision_sha256": "4" * 64,
            "sweeps": 500,
            "mean_acceptance": 0.76,
            "divergences": 0,
            "finite_scientific_endpoints": True,
            "finite_transformed_endpoints": True,
            "accepted_nonzero_displacement": True,
        },
        {
            "role": "held-out-b",
            "initializer": "random-recursive",
            "topology_seed": 42006,
            "master_seed": 74004,
            "topology_sha256": "a" * 64,
            "topology_precision_sha256": "5" * 64,
            "sweeps": 500,
            "mean_acceptance": 0.79,
            "divergences": 0,
            "finite_scientific_endpoints": True,
            "finite_transformed_endpoints": True,
            "accepted_nonzero_displacement": True,
        },
    ]
    candidate_grid = driver._calibration_candidate_grid()
    candidate_master_seeds = (73003, 73004, 73005)
    candidate_results = []
    for candidate_index, step_size in enumerate(candidate_grid["step_sizes"]):
        for leapfrog_steps in candidate_grid["leapfrog_steps"]:
            selected_candidate = step_size == 0.025 and leapfrog_steps == 3
            candidate_development = []
            for role_index, validation_trajectory in enumerate(development):
                candidate_development.append(
                    {
                        **{
                            name: validation_trajectory[name]
                            for name in (
                                "role",
                                "initializer",
                                "topology_seed",
                                "topology_sha256",
                                "topology_precision_sha256",
                            )
                        },
                        "master_seed": candidate_master_seeds[role_index],
                        "sweeps": 200,
                        "mean_acceptance": 0.75 + 0.01 * role_index,
                        "divergences": 0,
                        "finite_scientific_endpoints": True,
                        "finite_transformed_endpoints": True,
                        "accepted_nonzero_displacement": True,
                        "mean_mahalanobis_squared_displacement_per_gradient": (
                            10.0 - role_index if selected_candidate else 1.0 / (candidate_index + 2)
                        ),
                        "throughput_sweeps_per_second": 100.0 - role_index,
                    }
                )
            candidate_results.append(
                {
                    "step_size": step_size,
                    "leapfrog_steps": leapfrog_steps,
                    "development": candidate_development,
                    "development_admissible": True,
                }
            )
    validation = {
        "development": development,
        "held_out": held_out,
    }
    calibration = {
        "schema": driver.CALIBRATION_SCHEMA,
        "calibration_id": "tiny-topology-conditioned-hmc-calibration-v3",
        "fixed_k": 2,
        "input_sha256": driver._sha256_file(input_path),
        "target": {
            "concentration": 3.0,
            "root_variance": 0.25,
            "likelihood_power": 1.0,
            "fixed_prior_mean": [1.0] * 6,
            "fixed_prior_sd": [1.0] * 6,
            "nominal_weight_policy": "positive-native-mass-v1",
            "normalize_weights": True,
        },
        "kernel": {
            "step_size": 0.025,
            "leapfrog_steps": 3,
            "coordinate_layout_id": (driver.FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID),
            "metric_semantics_id": (driver.FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID),
            "metric_builder_id": driver.FULL_TILING_PYMC_HMC_METRIC_BUILDER_ID,
            "metric_reference_id": driver.FULL_TILING_PYMC_HMC_METRIC_REFERENCE_ID,
        },
        "evidence": {
            "code_revision": "test-revision",
            "input_sha256": driver._sha256_file(input_path),
            "candidate_grid": candidate_grid,
            "candidate_results": candidate_results,
            "development": development,
            "held_out": held_out,
            "selected": {
                "step_size": 0.025,
                "leapfrog_steps": 3,
                "selection_rule_id": driver.CALIBRATION_SELECTION_RULE_ID,
                "candidate_grid_sha256": driver._json_sha256(candidate_grid),
                "candidate_results_sha256": driver._json_sha256(candidate_results),
                "development_evidence_sha256": driver._json_sha256(development),
                "validation_evidence_sha256": driver._json_sha256(validation),
            },
            "excluded_production_topology_sha256": {
                item["role"]: item["topology_sha256"] for item in (*development, *held_out)
            },
            "source_artifact_sha256": {
                "candidate-grid": driver._json_sha256(candidate_grid),
                "candidate-results": driver._json_sha256(candidate_results),
                "development-validation": driver._json_sha256(development),
                "held-out-validation": driver._json_sha256(held_out),
            },
        },
    }
    calibration_path.write_text(
        driver._canonical_json(calibration),
        encoding="utf-8",
    )
    values = [
        "--input",
        str(input_path),
        "--output-directory",
        str(output_path),
        "--k",
        "2",
        "--sweeps",
        "1",
        "--seed",
        "812",
        "--chain-id",
        "tiny-chain",
        "--step-size",
        "0.025",
        "--leapfrog-steps",
        "3",
        "--calibration-id",
        "tiny-topology-conditioned-hmc-calibration-v3",
        "--calibration-file",
        str(calibration_path),
        "--calibration-sha256",
        driver._sha256_file(calibration_path),
        "--concentration",
        "3",
        "--root-variance",
        "0.25",
        "--fixed-prior-mean",
        "1",
        "--fixed-prior-sd",
        "1",
        "--input-id",
        "tiny-frozen-native-v1",
        "--expected-input-sha256",
        driver._sha256_file(input_path),
        "--code-revision",
        "test-revision",
        "--nominal-weight-policy",
        "positive-native-mass-v1",
    ]
    values.extend(extra)
    return values


def _replace_option(
    arguments: list[str],
    option: str,
    replacement: str,
) -> list[str]:
    """Return CLI arguments with one option value replaced."""
    changed = list(arguments)
    changed[changed.index(option) + 1] = replacement
    return changed


def _json(path: Path) -> dict[str, Any]:
    """Load one JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash_calibration_evidence(
    driver: ModuleType,
    calibration: dict[str, Any],
) -> None:
    """Refresh the v3 hashes after a test mutates trajectory evidence."""
    evidence = calibration["evidence"]
    candidate_grid = evidence["candidate_grid"]
    candidate_results = evidence["candidate_results"]
    development = evidence["development"]
    held_out = evidence["held_out"]
    evidence["selected"]["candidate_grid_sha256"] = driver._json_sha256(candidate_grid)
    evidence["selected"]["candidate_results_sha256"] = driver._json_sha256(candidate_results)
    evidence["selected"]["development_evidence_sha256"] = driver._json_sha256(development)
    evidence["selected"]["validation_evidence_sha256"] = driver._json_sha256(
        {
            "development": development,
            "held_out": held_out,
        }
    )
    evidence["excluded_production_topology_sha256"] = {
        item["role"]: item["topology_sha256"] for item in (*development, *held_out)
    }
    evidence["source_artifact_sha256"] = {
        "candidate-grid": driver._json_sha256(candidate_grid),
        "candidate-results": driver._json_sha256(candidate_results),
        "development-validation": driver._json_sha256(development),
        "held-out-validation": driver._json_sha256(held_out),
    }


@_requires_x64_child
@pytest.mark.parametrize(
    ("tie_break", "rival_index"), (("acceptance", 1), ("throughput", 1), ("step_size", 3))
)
def test_calibration_selection_recomputes_each_predeclared_tie_break(
    tmp_path: Path,
    hmc_driver: ModuleType,
    tie_break: str,
    rival_index: int,
) -> None:
    """Selection evidence is recomputed through every ordered tie-break."""
    input_path = tmp_path / f"{tie_break}.nc"
    _write_frozen_input(input_path)
    arguments = _arguments(hmc_driver, input_path, tmp_path / tie_break)
    calibration_path = Path(arguments[arguments.index("--calibration-file") + 1])
    calibration = _json(calibration_path)
    candidates = calibration["evidence"]["candidate_results"]
    for candidate in candidates:
        for trajectory in candidate["development"]:
            trajectory["mean_mahalanobis_squared_displacement_per_gradient"] = 0.1
    selected = candidates[0]["development"]
    rival = candidates[rival_index]["development"]
    for selected_trajectory, rival_trajectory in zip(selected, rival, strict=True):
        selected_trajectory["mean_mahalanobis_squared_displacement_per_gradient"] = 10.0
        rival_trajectory["mean_mahalanobis_squared_displacement_per_gradient"] = 10.0
        if tie_break in {"throughput", "step_size"}:
            selected_trajectory["mean_acceptance"] = 0.75
            rival_trajectory["mean_acceptance"] = 0.75
        if tie_break == "acceptance":
            selected_trajectory["mean_acceptance"] = 0.75
            rival_trajectory["mean_acceptance"] = 0.80
        if tie_break == "throughput":
            selected_trajectory["throughput_sweeps_per_second"] = 100.0
            rival_trajectory["throughput_sweeps_per_second"] = 90.0
        if tie_break == "step_size":
            selected_trajectory["throughput_sweeps_per_second"] = 100.0
            rival_trajectory["throughput_sweeps_per_second"] = 100.0
    _rehash_calibration_evidence(hmc_driver, calibration)
    parsed = hmc_driver.build_parser().parse_args(arguments)
    hmc_driver._validate_calibration_evidence(calibration["evidence"], parsed)


@_requires_x64_child
def test_dry_fresh_and_resumed_segments_are_exact_and_auditable(
    tmp_path: Path,
    hmc_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh canonical boundary publishes in order and resumes unchanged.

    The fresh state is canonicalized before draw zero; the resumed state uses
    stored authoritative coordinates unchanged. The same immutable chain also
    proves that calibration, provenance, and static-HMC setting changes are
    rejected by checkpoint manifest validation before continuation output is
    published.
    """
    input_path = tmp_path / "frozen.nc"
    dry_output = tmp_path / "dry-output"
    fresh_output = tmp_path / "fresh-output"
    resumed_output = tmp_path / "resumed-output"
    _write_frozen_input(input_path)

    dry_summary = hmc_driver.run(
        hmc_driver.build_parser().parse_args(
            [
                *_arguments(hmc_driver, input_path, dry_output),
                "--dry-run",
            ]
        )
    )
    assert dry_summary["status"] == "dry_run"
    assert dry_summary["closure"] == {
        "mass_coordinate_max_abs_error": 0.0,
        "prior_mean_total_max_abs_error": 0.0,
    }
    assert (
        abs(dry_summary["transformed_target_preflight"]["difference"])
        <= dry_summary["transformed_target_preflight"]["absolute_tolerance"]
    )
    assert dry_summary["runtime_identity"]["pytensor_float_x"] == "float64"
    assert not dry_output.exists()

    publication_order: list[str] = []
    original_write_text = hmc_driver._atomic_write_text
    original_write_trace = hmc_driver._atomic_write_trace
    original_save_checkpoint = hmc_driver.save_full_tiling_pymc_hmc_checkpoint

    def recording_write_text(path: Path, text: str) -> None:
        """Record JSON publication while preserving create-only writes."""
        publication_order.append(path.name)
        original_write_text(path, text)

    def recording_write_trace(
        dataset: xr.Dataset,
        path: Path,
        *,
        engine: str,
    ) -> None:
        """Record trace publication while preserving the real NetCDF write."""
        publication_order.append(path.name)
        original_write_trace(dataset, path, engine=engine)

    def recording_save_checkpoint(
        path: Path,
        checkpoint: Any,
        *,
        run_manifest: Any,
    ) -> None:
        """Record checkpoint publication while preserving strict checkpoint I/O."""
        publication_order.append(path.name)
        original_save_checkpoint(
            path,
            checkpoint,
            run_manifest=run_manifest,
        )

    monkeypatch.setattr(
        hmc_driver,
        "_atomic_write_text",
        recording_write_text,
    )
    monkeypatch.setattr(
        hmc_driver,
        "_atomic_write_trace",
        recording_write_trace,
    )
    monkeypatch.setattr(
        hmc_driver,
        "save_full_tiling_pymc_hmc_checkpoint",
        recording_save_checkpoint,
    )

    fresh_arguments = _arguments(
        hmc_driver,
        input_path,
        fresh_output,
    )
    fresh_summary = hmc_driver.run(hmc_driver.build_parser().parse_args(fresh_arguments))
    assert publication_order == [
        hmc_driver.MANIFEST_FILENAME,
        hmc_driver.TRACE_FILENAME,
        hmc_driver.SUMMARY_FILENAME,
        hmc_driver.CHECKPOINT_FILENAME,
        hmc_driver.COMPLETION_FILENAME,
    ]
    assert {path.name for path in fresh_output.iterdir()} == {
        hmc_driver.MANIFEST_FILENAME,
        hmc_driver.TRACE_FILENAME,
        hmc_driver.SUMMARY_FILENAME,
        hmc_driver.CHECKPOINT_FILENAME,
        hmc_driver.COMPLETION_FILENAME,
    }
    manifest = _json(fresh_output / hmc_driver.MANIFEST_FILENAME)
    completion = _json(fresh_output / hmc_driver.COMPLETION_FILENAME)
    assert manifest["schema"].endswith("native_manifest.v3")
    assert manifest["initialization"]["seed"] is None
    assert manifest["sampler"]["metric_semantics_id"] == (hmc_driver.FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID)
    assert manifest["sampler"]["metric_builder_id"] == (hmc_driver.FULL_TILING_PYMC_HMC_METRIC_BUILDER_ID)
    assert manifest["sampler"]["metric_reference_id"] == (hmc_driver.FULL_TILING_PYMC_HMC_METRIC_REFERENCE_ID)
    assert manifest["sampler"]["metric_rebuild_policy"] == (hmc_driver.TOPOLOGY_PRECISION_REBUILD_POLICY)
    assert manifest["sampler"]["topology_dependent_metric"] is True
    assert len(manifest["initialization"]["topology_precision_sha256"]) == 64
    assert (
        manifest["initialization"]["state_sha256"] == fresh_summary["lineage"]["segment_start_state_sha256"]
    )
    assert manifest["sampler"]["calibration"] == {
        "schema": hmc_driver.CALIBRATION_SCHEMA,
        "id": "tiny-topology-conditioned-hmc-calibration-v3",
        "sha256": hmc_driver._sha256_file(input_path.with_name("calibration.json")),
    }
    assert fresh_summary["schema"].endswith("native_summary.v3")
    assert "sampling_seconds" not in fresh_summary["performance"]
    assert fresh_summary["performance"]["kernel_setup_and_compile_seconds"] >= 0.0
    assert fresh_summary["performance"]["transition_sampling_seconds"] >= 0.0
    assert fresh_summary["performance"]["sweeps_per_second"] is not None
    assert (
        fresh_summary["performance"]["leapfrog_steps_per_second"]
        == 3 * fresh_summary["performance"]["sweeps_per_second"]
    )
    assert fresh_summary["run"] == {
        "fixed_k": 2,
        "schedule_id": hmc_driver.FULL_TILING_PYMC_HMC_SCHEDULE_ID,
        "segment_sweeps": 1,
        "segment_start_sweep": 0,
        "segment_end_sweep": 1,
        "retained_states": 2,
        "durable_checkpoint": True,
        "topology_precision_sha256": fresh_summary["run"]["topology_precision_sha256"],
    }
    assert len(fresh_summary["run"]["topology_precision_sha256"]) == 64
    assert fresh_summary["run"]["topology_precision_sha256"].islower()
    assert completion["schema"].endswith("native_completion.v3")
    assert completion["parent_checkpoint_sha256"] is None
    assert completion["parent_completion_sha256"] is None
    assert completion["parent_artifact_sha256"] is None
    assert fresh_summary["lineage"]["parent_completion_sha256"] is None
    assert fresh_summary["lineage"]["parent_artifact_sha256"] is None
    assert completion["segment_start_sweep"] == 0
    assert completion["segment_end_sweep"] == 1
    assert set(completion["sha256"]) == {
        hmc_driver.MANIFEST_FILENAME,
        hmc_driver.TRACE_FILENAME,
        hmc_driver.SUMMARY_FILENAME,
        hmc_driver.CHECKPOINT_FILENAME,
    }
    for filename, digest in completion["sha256"].items():
        assert hmc_driver._sha256_file(fresh_output / filename) == digest

    with np.load(
        fresh_output / hmc_driver.CHECKPOINT_FILENAME,
        allow_pickle=False,
    ) as checkpoint:
        assert set(checkpoint.files) == {
            "rectangle_bounds",
            "leaf_masses",
            "fixed_coefficients",
            "log_leaf_mass",
            "log_fixed_coefficient",
            "dynamic_prediction",
            "fixed_prediction",
            "prediction",
            "residual",
            "metadata",
            "metadata_sha256",
        }
        assert all(checkpoint[name].dtype != np.dtype(object) for name in checkpoint.files)
        checkpoint_metadata = json.loads(checkpoint["metadata"].tobytes().decode("utf-8"))
    assert checkpoint_metadata["schema_version"] == 3
    assert checkpoint_metadata["sweeps_completed"] == 1
    assert (
        checkpoint_metadata["topology_precision_sha256"]
        == (fresh_summary["run"]["topology_precision_sha256"])
    )
    assert checkpoint_metadata["runtime_identity"]["pytensor_float_x"] == ("float64")
    assert checkpoint_metadata["run_manifest_json"] == (hmc_driver._canonical_json(manifest).rstrip("\n"))

    with xr.open_dataset(
        fresh_output / hmc_driver.TRACE_FILENAME,
        engine="h5netcdf",
    ) as fresh_trace:
        fresh_trace.load()
        assert fresh_trace.attrs["schema"].endswith("native_trace.v3")
        assert fresh_trace.attrs["topology_dependent_metric"] == "true"
        assert fresh_trace.attrs["metric_builder_id"] == (hmc_driver.FULL_TILING_PYMC_HMC_METRIC_BUILDER_ID)
        assert fresh_trace.attrs["manifest_sha256"] == (
            hmc_driver._sha256_file(fresh_output / hmc_driver.MANIFEST_FILENAME)
        )
        assert fresh_trace.sizes == {
            "draw": 2,
            "region": 2,
            "bound": 4,
            "fixed_parameter": 6,
            "observation": 3,
            "sweep": 1,
            "lat": 2,
            "lon": 2,
        }
        np.testing.assert_array_equal(fresh_trace["sweep"], [1])
        np.testing.assert_array_equal(
            fresh_trace["state_sweep"],
            [0, 1],
        )
        assert (
            fresh_trace["log_target"].isel(draw=0).item()
            == (fresh_summary["target"]["chain_initial_log_target"])
        )
        assert (
            fresh_trace["log_target"].isel(draw=0).item()
            == (fresh_summary["target"]["segment_initial_log_target"])
        )
        np.testing.assert_array_equal(
            fresh_trace["leaf_mass"].isel(draw=0),
            np.exp(fresh_trace["log_leaf_mass"].isel(draw=0)),
        )
        np.testing.assert_array_equal(
            fresh_trace["fixed_coefficient"].isel(draw=0),
            np.exp(fresh_trace["log_fixed_coefficient"].isel(draw=0)),
        )
        assert fresh_trace["hmc_start_log_leaf_mass"].shape == (1, 2)
        assert fresh_trace["hmc_start_log_fixed_coefficient"].shape == (1, 6)
        assert fresh_trace["leaf_mass"].isel(draw=0, region=0).item() != 0.125
        assert fresh_trace["hmc_seed"].dtype == np.dtype(np.uint64)
        fresh_final_mass = fresh_trace["leaf_mass"].isel(draw=-1).values
        fresh_final_fixed = fresh_trace["fixed_coefficient"].isel(draw=-1).values

    checkpoint_path = fresh_output / hmc_driver.CHECKPOINT_FILENAME
    mismatch_cases = (
        (
            "--calibration-id",
            "different-calibration",
            "Calibration v3 identity",
        ),
        ("--chain-id", "different-chain", "manifest does not match"),
        ("--step-size", "0.002", "Calibration v3 identity"),
    )
    for option, replacement, message in mismatch_cases:
        mismatch_output = tmp_path / option.removeprefix("--")
        mismatched = _replace_option(
            _arguments(
                hmc_driver,
                input_path,
                mismatch_output,
                extra=(
                    "--resume-checkpoint",
                    str(checkpoint_path),
                ),
            ),
            option,
            replacement,
        )
        with pytest.raises(ValueError, match=message):
            hmc_driver.run(hmc_driver.build_parser().parse_args(mismatched))
        assert not mismatch_output.exists()

    parent_failure_cases = (
        ("missing-completion", "missing"),
        ("corrupt-summary", "corrupt"),
        ("incompatible-completion", "incompatible"),
    )
    for case_name, mutation in parent_failure_cases:
        parent_bundle = tmp_path / f"parent-{case_name}"
        shutil.copytree(fresh_output, parent_bundle)
        if mutation == "missing":
            (parent_bundle / hmc_driver.COMPLETION_FILENAME).unlink()
            expected_error = FileNotFoundError
            message = "completion certificate"
        elif mutation == "corrupt":
            with (parent_bundle / hmc_driver.SUMMARY_FILENAME).open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write("\n")
            expected_error = ValueError
            message = "SHA-256 does not match"
        else:
            certificate = _json(parent_bundle / hmc_driver.COMPLETION_FILENAME)
            certificate["checkpoint"] = "different-checkpoint.npz"
            (parent_bundle / hmc_driver.COMPLETION_FILENAME).write_text(
                hmc_driver._canonical_json(certificate),
                encoding="utf-8",
            )
            expected_error = ValueError
            message = "checkpoint name is incompatible"
        failed_output = tmp_path / f"child-{case_name}"
        failed_arguments = _arguments(
            hmc_driver,
            input_path,
            failed_output,
            extra=(
                "--resume-checkpoint",
                str(parent_bundle / hmc_driver.CHECKPOINT_FILENAME),
            ),
        )
        with pytest.raises(expected_error, match=message):
            hmc_driver.run(hmc_driver.build_parser().parse_args(failed_arguments))
        assert not failed_output.exists()

    publication_order.clear()
    resume_arguments = _arguments(
        hmc_driver,
        input_path,
        resumed_output,
        extra=("--resume-checkpoint", str(checkpoint_path)),
    )
    resumed_summary = hmc_driver.run(hmc_driver.build_parser().parse_args(resume_arguments))
    assert (resumed_output / hmc_driver.MANIFEST_FILENAME).read_bytes() == (
        fresh_output / hmc_driver.MANIFEST_FILENAME
    ).read_bytes()
    assert publication_order[-1] == hmc_driver.COMPLETION_FILENAME
    assert resumed_summary["run"]["segment_start_sweep"] == 1
    assert resumed_summary["run"]["segment_end_sweep"] == 2
    resumed_completion = _json(resumed_output / hmc_driver.COMPLETION_FILENAME)
    assert resumed_completion["parent_checkpoint_sha256"] == (hmc_driver._sha256_file(checkpoint_path))
    parent_completion_sha256 = hmc_driver._sha256_file(fresh_output / hmc_driver.COMPLETION_FILENAME)
    assert resumed_summary["lineage"]["parent_completion_sha256"] == (parent_completion_sha256)
    assert resumed_summary["lineage"]["parent_artifact_sha256"] == completion["sha256"]
    assert resumed_completion["parent_completion_sha256"] == (parent_completion_sha256)
    assert resumed_completion["parent_artifact_sha256"] == completion["sha256"]
    assert resumed_completion["segment_start_sweep"] == 1
    assert resumed_completion["segment_end_sweep"] == 2
    with xr.open_dataset(
        resumed_output / hmc_driver.TRACE_FILENAME,
        engine="h5netcdf",
    ) as resumed_trace:
        np.testing.assert_array_equal(resumed_trace["sweep"], [2])
        np.testing.assert_array_equal(
            resumed_trace["state_sweep"],
            [1, 2],
        )
        np.testing.assert_array_equal(
            resumed_trace["leaf_mass"].isel(draw=0),
            fresh_final_mass,
        )
        np.testing.assert_array_equal(
            resumed_trace["fixed_coefficient"].isel(draw=0),
            fresh_final_fixed,
        )


@_requires_x64_child
def test_driver_rejects_initialization_runtime_and_forbidden_outputs(
    tmp_path: Path,
    hmc_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialization, runtime, and PARIS output safeguards fail before writes."""
    input_path = tmp_path / "frozen.nc"
    _write_frozen_input(input_path)
    output_path = tmp_path / "output"

    common = _arguments(hmc_driver, input_path, output_path)
    with pytest.raises(ValueError, match="requires --initialization-seed"):
        hmc_driver.run(
            hmc_driver.build_parser().parse_args([*common, "--initialization", "random-recursive"])
        )
    with pytest.raises(ValueError, match="must differ"):
        hmc_driver.run(
            hmc_driver.build_parser().parse_args(
                [
                    *common,
                    "--initialization",
                    "random-recursive",
                    "--initialization-seed",
                    "812",
                ]
            )
        )
    malformed_calibration = _replace_option(
        common,
        "--calibration-sha256",
        "not-a-sha256",
    )
    with pytest.raises(ValueError, match="64 hexadecimal"):
        hmc_driver.run(hmc_driver.build_parser().parse_args(malformed_calibration))

    missing_calibration = _replace_option(
        _arguments(hmc_driver, input_path, output_path),
        "--calibration-file",
        str(tmp_path / "missing-calibration.json"),
    )
    with pytest.raises(FileNotFoundError, match="Calibration file"):
        hmc_driver.run(hmc_driver.build_parser().parse_args(missing_calibration))

    hash_mismatch = _arguments(hmc_driver, input_path, output_path)
    calibration_path = Path(hash_mismatch[hash_mismatch.index("--calibration-file") + 1])
    calibration_path.write_text(
        calibration_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Calibration file SHA-256"):
        hmc_driver.run(hmc_driver.build_parser().parse_args(hash_mismatch))

    duplicate_member = _arguments(hmc_driver, input_path, output_path)
    calibration_path.write_text(
        '{"schema":"first","schema":"second"}\n',
        encoding="utf-8",
    )
    duplicate_member = _replace_option(
        duplicate_member,
        "--calibration-sha256",
        hmc_driver._sha256_file(calibration_path),
    )
    with pytest.raises(ValueError, match="Duplicate JSON object member"):
        hmc_driver.run(hmc_driver.build_parser().parse_args(duplicate_member))

    incompatible_identity = _arguments(
        hmc_driver,
        input_path,
        output_path,
    )
    calibration = _json(calibration_path)
    calibration["fixed_k"] = 3
    calibration_path.write_text(
        hmc_driver._canonical_json(calibration),
        encoding="utf-8",
    )
    incompatible_identity = _replace_option(
        incompatible_identity,
        "--calibration-sha256",
        hmc_driver._sha256_file(calibration_path),
    )
    with pytest.raises(ValueError, match="identity does not exactly match"):
        hmc_driver.run(hmc_driver.build_parser().parse_args(incompatible_identity))
    assert not output_path.exists()

    common = _arguments(hmc_driver, input_path, output_path)
    runtime = hmc_driver.full_tiling_pymc_hmc_runtime_identity()
    monkeypatch.setattr(
        hmc_driver,
        "full_tiling_pymc_hmc_runtime_identity",
        lambda: replace(runtime, pytensor_float_x="float32"),
    )
    with pytest.raises(RuntimeError, match="floatX must be exactly float64"):
        hmc_driver.run(hmc_driver.build_parser().parse_args([*common, "--dry-run"]))
    assert not output_path.exists()

    paris_parent = tmp_path / "PARIS_inversions"
    paris_parent.mkdir()
    before = tuple(paris_parent.iterdir())
    forbidden_output = paris_parent / "must-not-exist"
    with pytest.raises(ValueError, match="PARIS_inversions"):
        hmc_driver.run(
            hmc_driver.build_parser().parse_args(
                _arguments(
                    hmc_driver,
                    input_path,
                    forbidden_output,
                )
            )
        )
    assert tuple(paris_parent.iterdir()) == before


@_requires_x64_child
@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("legacy-v1", "schemas v1 and v2"),
        ("legacy-v2", "schemas v1 and v2"),
        ("retired-scale-key", "identity does not exactly match"),
        ("candidate-grid", "does not exactly match the predeclared H2d grid"),
        ("omitted-candidate", "must contain every predeclared grid candidate"),
        ("candidate-order", "not in exact predeclared grid order"),
        ("candidate-sweeps", "must contain 200 sweeps"),
        ("candidate-master-seed", "master seeds"),
        ("candidate-admissibility", "development_admissible"),
        ("altered-result-winner", "recomputed candidate-grid winner"),
        ("development-count", "exactly three development"),
        ("role-order", "roles and initializers are out of order"),
        ("nominal-topology-seed", "largest-nominal calibration topology seed must be null"),
        ("duplicate-master-seed", "master seeds must be distinct"),
        ("duplicate-topology-hash", "topology hashes must be distinct"),
        ("duplicate-precision-hash", "topology precision hashes must be distinct"),
        ("sweeps-type", "must contain 500 sweeps"),
        ("divergences-type", "does not pass the frozen validation gates"),
        ("heldout-divergence", "does not pass the frozen validation gates"),
        ("heldout-acceptance", "does not pass the frozen validation gates"),
        ("heldout-nonfinite", "Non-standard JSON constant"),
        ("heldout-finite-failure", "does not pass the frozen validation gates"),
        ("heldout-transformed-failure", "does not pass the frozen validation gates"),
        ("heldout-displacement-failure", "does not pass the frozen validation gates"),
        ("selected-controls-mismatch", "do not match the requested HMC controls"),
        ("selected-leapfrog-type", "do not match the requested HMC controls"),
        ("selection-rule", "incompatible selection rule"),
        ("candidate-results-hash", "candidate-results SHA-256"),
        ("development-evidence-hash", "development evidence SHA-256"),
        ("validation-evidence-hash", "validation evidence SHA-256"),
        ("candidate-source-hash", "source artifact hashes"),
        ("excluded-topology", "must exactly match all five trajectories"),
    ),
)
def test_driver_rejects_invalid_v3_calibration_evidence(
    tmp_path: Path,
    hmc_driver: ModuleType,
    case: str,
    message: str,
) -> None:
    """Calibration v3 rejects retired schemas and malformed H2d evidence."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / case
    _write_frozen_input(input_path)
    arguments = _arguments(hmc_driver, input_path, output_path)
    calibration_path = Path(arguments[arguments.index("--calibration-file") + 1])
    calibration = _json(calibration_path)
    evidence = calibration["evidence"]
    heldout = evidence["held_out"][1]
    rehash = False

    if case == "legacy-v1":
        calibration["schema"] = "openghg_inversions.full_tiling_pymc_hmc_calibration.v1"
    elif case == "legacy-v2":
        calibration["schema"] = "openghg_inversions.full_tiling_pymc_hmc_calibration.v2"
    elif case == "retired-scale-key":
        kernel = calibration["kernel"]
        kernel["leaf_position_scale"] = 1.0
    elif case == "candidate-grid":
        evidence["candidate_grid"]["step_sizes"][0] = 0.026
        rehash = True
    elif case == "omitted-candidate":
        evidence["candidate_results"].pop()
        rehash = True
    elif case == "candidate-order":
        evidence["candidate_results"][0]["leapfrog_steps"] = 5
        rehash = True
    elif case == "candidate-sweeps":
        evidence["candidate_results"][0]["development"][0]["sweeps"] = 199
        rehash = True
    elif case == "candidate-master-seed":
        evidence["candidate_results"][1]["development"][0]["master_seed"] = -1
        rehash = True
    elif case == "candidate-admissibility":
        evidence["candidate_results"][0]["development_admissible"] = False
        rehash = True
    elif case == "altered-result-winner":
        for trajectory in evidence["candidate_results"][0]["development"]:
            trajectory["mean_mahalanobis_squared_displacement_per_gradient"] = 0.1
        for trajectory in evidence["candidate_results"][1]["development"]:
            trajectory["mean_mahalanobis_squared_displacement_per_gradient"] = 0.7
        rehash = True
    elif case == "development-count":
        evidence["development"].pop()
        rehash = True
    elif case == "role-order":
        evidence["development"][1]["role"] = "development-b"
        rehash = True
    elif case == "nominal-topology-seed":
        evidence["development"][0]["topology_seed"] = 42002
        rehash = True
    elif case == "duplicate-master-seed":
        heldout["master_seed"] = evidence["held_out"][0]["master_seed"]
        rehash = True
    elif case == "duplicate-topology-hash":
        heldout["topology_sha256"] = evidence["development"][0]["topology_sha256"]
        rehash = True
    elif case == "duplicate-precision-hash":
        heldout["topology_precision_sha256"] = evidence["development"][0]["topology_precision_sha256"]
        rehash = True
    elif case == "sweeps-type":
        heldout["sweeps"] = 500.0
        rehash = True
    elif case == "divergences-type":
        heldout["divergences"] = 0.0
        rehash = True
    elif case == "heldout-divergence":
        heldout["divergences"] = 1
        rehash = True
    elif case == "heldout-acceptance":
        heldout["mean_acceptance"] = 0.96
        rehash = True
    elif case == "heldout-finite-failure":
        heldout["finite_scientific_endpoints"] = False
        rehash = True
    elif case == "heldout-transformed-failure":
        heldout["finite_transformed_endpoints"] = False
        rehash = True
    elif case == "heldout-displacement-failure":
        heldout["accepted_nonzero_displacement"] = False
        rehash = True
    elif case == "selected-controls-mismatch":
        evidence["selected"]["step_size"] = 0.002
    elif case == "selected-leapfrog-type":
        evidence["selected"]["leapfrog_steps"] = 1.0
    elif case == "selection-rule":
        evidence["selected"]["selection_rule_id"] = "different-rule"
    elif case == "candidate-results-hash":
        evidence["selected"]["candidate_results_sha256"] = "0" * 64
    elif case == "development-evidence-hash":
        evidence["selected"]["development_evidence_sha256"] = "0" * 64
    elif case == "validation-evidence-hash":
        evidence["selected"]["validation_evidence_sha256"] = "0" * 64
    elif case == "candidate-source-hash":
        evidence["source_artifact_sha256"]["candidate-grid"] = "0" * 64
    elif case == "excluded-topology":
        evidence["excluded_production_topology_sha256"]["held-out-b"] = "0" * 64

    if rehash:
        _rehash_calibration_evidence(hmc_driver, calibration)

    calibration_text = hmc_driver._canonical_json(calibration)
    if case == "heldout-nonfinite":
        calibration_text = calibration_text.replace(
            '"mean_acceptance":0.79',
            '"mean_acceptance":NaN',
        )
    calibration_path.write_text(calibration_text, encoding="utf-8")
    arguments = _replace_option(
        arguments,
        "--calibration-sha256",
        hmc_driver._sha256_file(calibration_path),
    )

    with pytest.raises(ValueError, match=message):
        hmc_driver.run(hmc_driver.build_parser().parse_args(arguments))
    assert not output_path.exists()


@_requires_x64_child
def test_driver_rejects_production_topology_seen_during_calibration(
    tmp_path: Path,
    hmc_driver: ModuleType,
) -> None:
    """Retained production starts must be disjoint from H2d topology seeds."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "reused-calibration-topology"
    _write_frozen_input(input_path)
    arguments = _replace_option(
        _arguments(
            hmc_driver,
            input_path,
            output_path,
            extra=(
                "--initialization",
                "random-recursive",
                "--initialization-seed",
                "42050",
            ),
        ),
        "--k",
        "50",
    )

    with pytest.raises(ValueError, match="disjoint from H2d calibration topology seeds"):
        hmc_driver.run(hmc_driver.build_parser().parse_args(arguments))
    assert not output_path.exists()


@_requires_x64_child
@pytest.mark.parametrize("digest_character", ("c", "d", "e", "f", "a"))
def test_driver_rejects_exact_calibration_topology_hash_collision(
    tmp_path: Path,
    hmc_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    digest_character: str,
) -> None:
    """Every calibration topology identity is excluded from production."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "reused-calibration-topology-hash"
    _write_frozen_input(input_path)
    monkeypatch.setattr(
        hmc_driver,
        "_topology_sha256",
        lambda bounds: digest_character * 64,
    )

    with pytest.raises(ValueError, match="topology hash was used by H2d calibration"):
        hmc_driver.run(hmc_driver.build_parser().parse_args(_arguments(hmc_driver, input_path, output_path)))
    assert not output_path.exists()


@_requires_x64_child
def test_artifact_failures_never_publish_completion_certificate(
    tmp_path: Path,
    hmc_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinate-audit and checkpoint-stage failures leave bundles incomplete."""
    input_path = tmp_path / "frozen.nc"
    _write_frozen_input(input_path)
    original_write_trace = hmc_driver._atomic_write_trace
    original_save_checkpoint = hmc_driver.save_full_tiling_pymc_hmc_checkpoint

    def corrupting_write_trace(
        dataset: xr.Dataset,
        path: Path,
        *,
        engine: Literal["h5netcdf", "netcdf4", "scipy"],
    ) -> None:
        """Corrupt one coordinate after publication to exercise reopen audit."""
        original_write_trace(dataset, path, engine=engine)
        with xr.open_dataset(path, engine=engine) as opened:
            corrupted = opened.load()
        corrupted = corrupted.assign_coords(draw=np.asarray(corrupted["draw"].values) + 1)
        try:
            corrupted.to_netcdf(path, engine=engine, mode="w")
        finally:
            corrupted.close()

    coordinate_output = tmp_path / "coordinate-audit-failure"
    monkeypatch.setattr(
        hmc_driver,
        "_atomic_write_trace",
        corrupting_write_trace,
    )
    with pytest.raises(RuntimeError, match="coordinate draw does not match"):
        hmc_driver.run(
            hmc_driver.build_parser().parse_args(
                _arguments(
                    hmc_driver,
                    input_path,
                    coordinate_output,
                )
            )
        )
    assert coordinate_output.is_dir()
    assert not (coordinate_output / hmc_driver.COMPLETION_FILENAME).exists()

    def failing_save_checkpoint(*args: Any, **kwargs: Any) -> None:
        """Inject a checkpoint publication failure."""
        del args, kwargs
        raise RuntimeError("injected checkpoint publication failure")

    checkpoint_output = tmp_path / "checkpoint-failure"
    monkeypatch.setattr(
        hmc_driver,
        "_atomic_write_trace",
        original_write_trace,
    )
    monkeypatch.setattr(
        hmc_driver,
        "save_full_tiling_pymc_hmc_checkpoint",
        failing_save_checkpoint,
    )
    with pytest.raises(RuntimeError, match="injected checkpoint"):
        hmc_driver.run(
            hmc_driver.build_parser().parse_args(
                _arguments(
                    hmc_driver,
                    input_path,
                    checkpoint_output,
                )
            )
        )
    assert checkpoint_output.is_dir()
    assert not (checkpoint_output / hmc_driver.COMPLETION_FILENAME).exists()
    monkeypatch.setattr(
        hmc_driver,
        "save_full_tiling_pymc_hmc_checkpoint",
        original_save_checkpoint,
    )


@_requires_x64_child
def test_parser_and_kernel_settings_make_seed_and_metric_semantics_explicit(
    tmp_path: Path,
    hmc_driver: ModuleType,
) -> None:
    """The CLI exposes controls and metric identities but no metric knob."""
    input_path = tmp_path / "frozen.nc"
    _write_frozen_input(input_path)
    arguments = _arguments(
        hmc_driver,
        input_path,
        tmp_path / "output",
    )
    seed_index = arguments.index("--seed")
    without_seed = arguments[:seed_index] + arguments[seed_index + 2 :]
    with pytest.raises(SystemExit):
        hmc_driver.build_parser().parse_args(without_seed)
    calibration_index = arguments.index("--calibration-file")
    without_calibration = arguments[:calibration_index] + arguments[calibration_index + 2 :]
    with pytest.raises(SystemExit):
        hmc_driver.build_parser().parse_args(without_calibration)
    for option in (
        "--leaf-contrast-position-scale",
        "--leaf-total-position-scale",
        "--fixed-coefficient-position-scale",
    ):
        with pytest.raises(SystemExit):
            hmc_driver.build_parser().parse_args([*arguments, option, "1"])

    parsed = hmc_driver.build_parser().parse_args(arguments)
    settings = hmc_driver._requested_kernel_settings(parsed)
    assert asdict(settings) == {
        "fixed_k": 2,
        "step_size": 0.025,
        "leapfrog_steps": 3,
        "metric_builder_id": hmc_driver.FULL_TILING_PYMC_HMC_METRIC_BUILDER_ID,
        "metric_reference_id": hmc_driver.FULL_TILING_PYMC_HMC_METRIC_REFERENCE_ID,
    }
    assert "topology_reference_precision" in (hmc_driver.FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID)
    assert "is_cov_false" in hmc_driver.FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID
