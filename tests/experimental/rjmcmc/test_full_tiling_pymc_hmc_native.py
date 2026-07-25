"""Tests for the frozen-input mobile PyMC HMC native driver."""

from __future__ import annotations

from dataclasses import replace
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
    calibration = {
        "schema": driver.CALIBRATION_SCHEMA,
        "calibration_id": "tiny-static-hmc-calibration-v2",
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
            "step_size": 0.001,
            "leapfrog_steps": 1,
            "coordinate_layout_id": (driver.FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID),
            "metric_semantics_id": (driver.FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID),
            "leaf_contrast_position_scale": 1.75,
            "leaf_total_position_scale": 2.25,
            "fixed_coefficient_position_scale": [
                0.5,
                0.75,
                1.0,
                1.25,
                1.5,
                2.0,
            ],
        },
        "evidence": {
            "code_revision": "test-revision",
            "robust_variance_estimator": (driver.ROBUST_VARIANCE_ESTIMATOR),
            "leaf_metric_estimator": driver.LEAF_METRIC_ESTIMATOR,
            "clipping_bounds": [1.0e-4, 1.0e2],
            "development_initializers": [
                {
                    "role": "development-a",
                    "strategy": "random-recursive",
                    "seed": 41003,
                    "sampler_seed": 71003,
                    "sweeps": 200,
                },
                {
                    "role": "development-b",
                    "strategy": "random-recursive",
                    "seed": 41004,
                    "sampler_seed": 71004,
                    "sweeps": 200,
                },
            ],
            "candidate_grid": [{"step_size": 0.001, "leapfrog_steps": 1}],
            "decision_statistics": [
                {
                    "step_size": 0.001,
                    "leapfrog_steps": 1,
                    "development_a_mean_acceptance": 0.75,
                    "development_b_mean_acceptance": 0.8,
                    "divergences": 0,
                    "finite": True,
                    "median_hmc_log_displacement_per_leapfrog_step": 0.125,
                    "selected": True,
                }
            ],
            "selected_validation": {
                "step_size": 0.001,
                "leapfrog_steps": 1,
                "initializers": [
                    {
                        "role": "development-a",
                        "strategy": "random-recursive",
                        "seed": 41003,
                        "sampler_seed": 72003,
                        "sweeps": 500,
                        "mean_acceptance": 0.75,
                        "divergences": 0,
                        "finite": True,
                    },
                    {
                        "role": "development-b",
                        "strategy": "random-recursive",
                        "seed": 41004,
                        "sampler_seed": 72004,
                        "sweeps": 500,
                        "mean_acceptance": 0.8,
                        "divergences": 0,
                        "finite": True,
                    },
                    {
                        "role": "held-out",
                        "strategy": "random-recursive",
                        "seed": 41005,
                        "sampler_seed": 72005,
                        "sweeps": 500,
                        "mean_acceptance": 0.78,
                        "divergences": 0,
                        "finite": True,
                    },
                ],
            },
            "excluded_production_topology_sha256": {
                "metric_source": "c" * 64,
                "development_a": "d" * 64,
                "development_b": "e" * 64,
                "held_out": "f" * 64,
            },
            "source_artifact_sha256": {"tiny-nuts-reference": "b" * 64},
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
        "0.001",
        "--leapfrog-steps",
        "1",
        "--leaf-contrast-position-scale",
        "1.75",
        "--leaf-total-position-scale",
        "2.25",
        "--fixed-coefficient-position-scale",
        "0.5,0.75,1,1.25,1.5,2",
        "--calibration-id",
        "tiny-static-hmc-calibration-v2",
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
    assert manifest["schema"].endswith("native_manifest.v2")
    assert manifest["initialization"]["seed"] is None
    assert manifest["sampler"]["metric_semantics_id"] == (hmc_driver.FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID)
    assert manifest["sampler"]["leaf_contrast_position_scale"] == 1.75
    assert manifest["sampler"]["leaf_total_position_scale"] == 2.25
    assert manifest["sampler"]["fixed_coefficient_position_scale"] == [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    assert (
        manifest["initialization"]["state_sha256"] == fresh_summary["lineage"]["segment_start_state_sha256"]
    )
    assert manifest["sampler"]["calibration"] == {
        "schema": hmc_driver.CALIBRATION_SCHEMA,
        "id": "tiny-static-hmc-calibration-v2",
        "sha256": hmc_driver._sha256_file(input_path.with_name("calibration.json")),
    }
    assert fresh_summary["schema"].endswith("native_summary.v2")
    assert "sampling_seconds" not in fresh_summary["performance"]
    assert fresh_summary["performance"]["kernel_setup_and_compile_seconds"] >= 0.0
    assert fresh_summary["performance"]["transition_sampling_seconds"] >= 0.0
    assert fresh_summary["performance"]["sweeps_per_second"] is not None
    assert (
        fresh_summary["performance"]["leapfrog_steps_per_second"]
        == fresh_summary["performance"]["sweeps_per_second"]
    )
    assert fresh_summary["run"] == {
        "fixed_k": 2,
        "schedule_id": hmc_driver.FULL_TILING_PYMC_HMC_SCHEDULE_ID,
        "segment_sweeps": 1,
        "segment_start_sweep": 0,
        "segment_end_sweep": 1,
        "retained_states": 2,
        "durable_checkpoint": True,
    }
    assert completion["schema"].endswith("native_completion.v2")
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
    assert checkpoint_metadata["schema_version"] == 2
    assert checkpoint_metadata["sweeps_completed"] == 1
    assert checkpoint_metadata["runtime_identity"]["pytensor_float_x"] == ("float64")
    assert checkpoint_metadata["run_manifest_json"] == (hmc_driver._canonical_json(manifest).rstrip("\n"))

    with xr.open_dataset(
        fresh_output / hmc_driver.TRACE_FILENAME,
        engine="h5netcdf",
    ) as fresh_trace:
        fresh_trace.load()
        assert fresh_trace.attrs["schema"].endswith("native_trace.v2")
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
            "Calibration v2 identity",
        ),
        ("--chain-id", "different-chain", "manifest does not match"),
        ("--step-size", "0.002", "Calibration v2 identity"),
        (
            "--leaf-contrast-position-scale",
            "1.5",
            "Calibration v2 identity",
        ),
        (
            "--leaf-total-position-scale",
            "2.5",
            "Calibration v2 identity",
        ),
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
        ("legacy-schema", "retired diagonal metric"),
        ("legacy-leaf-key", "identity does not exactly match"),
        ("swapped-leaf-scales", "identity does not exactly match"),
        ("development-seed-mismatch", "development seeds are incompatible"),
        ("heldout-seed-equality", "must differ from development seeds"),
        ("validation-sampler-seed-equality", "sampler seeds must be distinct"),
        ("heldout-divergence", "does not pass the frozen acceptance gates"),
        ("heldout-acceptance", "does not pass the frozen acceptance gates"),
        ("heldout-nonfinite", "Non-standard JSON constant"),
        ("heldout-failure", "does not pass the frozen acceptance gates"),
        ("validation-candidate-mismatch", "does not match the selected decision"),
        ("duplicate-excluded-topology", "topology hashes must be distinct"),
    ),
)
def test_driver_rejects_invalid_v2_calibration_evidence(
    tmp_path: Path,
    hmc_driver: ModuleType,
    case: str,
    message: str,
) -> None:
    """Calibration v2 rejects retired metrics and invalid held-out evidence."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / case
    _write_frozen_input(input_path)
    arguments = _arguments(hmc_driver, input_path, output_path)
    calibration_path = Path(arguments[arguments.index("--calibration-file") + 1])
    calibration = _json(calibration_path)
    heldout = calibration["evidence"]["selected_validation"]["initializers"][2]

    if case == "legacy-schema":
        calibration["schema"] = hmc_driver.LEGACY_CALIBRATION_SCHEMA
    elif case == "legacy-leaf-key":
        kernel = calibration["kernel"]
        kernel["leaf_position_scale"] = kernel.pop("leaf_contrast_position_scale")
        kernel.pop("leaf_total_position_scale")
    elif case == "swapped-leaf-scales":
        kernel = calibration["kernel"]
        kernel["leaf_contrast_position_scale"], kernel["leaf_total_position_scale"] = (
            kernel["leaf_total_position_scale"],
            kernel["leaf_contrast_position_scale"],
        )
    elif case == "development-seed-mismatch":
        calibration["evidence"]["selected_validation"]["initializers"][1]["seed"] = 41006
    elif case == "heldout-seed-equality":
        heldout["seed"] = 41003
    elif case == "validation-sampler-seed-equality":
        heldout["sampler_seed"] = 72003
    elif case == "heldout-divergence":
        heldout["divergences"] = 1
    elif case == "heldout-acceptance":
        heldout["mean_acceptance"] = 0.91
    elif case == "heldout-failure":
        heldout["finite"] = False
    elif case == "validation-candidate-mismatch":
        calibration["evidence"]["selected_validation"]["step_size"] = 0.002
    elif case == "duplicate-excluded-topology":
        excluded = calibration["evidence"]["excluded_production_topology_sha256"]
        excluded["held_out"] = excluded["development_a"]

    calibration_text = hmc_driver._canonical_json(calibration)
    if case == "heldout-nonfinite":
        calibration_text = calibration_text.replace(
            '"mean_acceptance":0.78',
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
def test_driver_rejects_excessive_leaf_metric_condition_ratio(
    tmp_path: Path,
    hmc_driver: ModuleType,
) -> None:
    """The production CLI bounds the total/contrast eigenscale ratio."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "invalid-metric-ratio"
    _write_frozen_input(input_path)
    arguments = _replace_option(
        _arguments(hmc_driver, input_path, output_path),
        "--leaf-total-position-scale",
        "20000",
    )

    with pytest.raises(ValueError, match="maximum-to-minimum position-scale ratio"):
        hmc_driver.run(hmc_driver.build_parser().parse_args(arguments))
    assert not output_path.exists()


@_requires_x64_child
@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--leaf-contrast-position-scale", "200"),
        ("--leaf-total-position-scale", "200"),
        ("--fixed-coefficient-position-scale", "0.5,0.75,1,1.25,1.5,200"),
    ),
)
def test_driver_rejects_position_scales_outside_calibration_clipping_bounds(
    tmp_path: Path,
    hmc_driver: ModuleType,
    option: str,
    value: str,
) -> None:
    """Production scales must be possible outputs of the declared clipping rule."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / f"invalid-scale-{option}"
    _write_frozen_input(input_path)
    arguments = _replace_option(
        _arguments(hmc_driver, input_path, output_path),
        option,
        value,
    )

    with pytest.raises(ValueError, match="frozen clipping bounds"):
        hmc_driver.run(hmc_driver.build_parser().parse_args(arguments))
    assert not output_path.exists()


@_requires_x64_child
def test_driver_rejects_production_topology_seen_during_calibration(
    tmp_path: Path,
    hmc_driver: ModuleType,
) -> None:
    """Retained production starts must be disjoint from H2c topology seeds."""
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
                "41050",
            ),
        ),
        "--k",
        "50",
    )

    with pytest.raises(ValueError, match="disjoint from H2c calibration topology seeds"):
        hmc_driver.run(hmc_driver.build_parser().parse_args(arguments))
    assert not output_path.exists()


@_requires_x64_child
def test_driver_rejects_exact_calibration_topology_hash_collision(
    tmp_path: Path,
    hmc_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual topology identity, not only initializer seed, gates production."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "reused-calibration-topology-hash"
    _write_frozen_input(input_path)
    monkeypatch.setattr(
        hmc_driver,
        "_topology_sha256",
        lambda bounds: "c" * 64,
    )

    with pytest.raises(ValueError, match="topology hash was used by H2c calibration"):
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
    """The CLI requires a seed and resolves position-covariance diagonals."""
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
    ):
        option_index = arguments.index(option)
        without_metric_scale = arguments[:option_index] + arguments[option_index + 2 :]
        with pytest.raises(SystemExit):
            hmc_driver.build_parser().parse_args(without_metric_scale)
    for option, value in (
        ("--leaf-contrast-position-scale", "0"),
        ("--leaf-total-position-scale", "-1"),
    ):
        nonpositive = _replace_option(arguments, option, value)
        with pytest.raises(ValueError, match=f"{option} must be finite and positive"):
            hmc_driver.run(hmc_driver.build_parser().parse_args(nonpositive))

    parsed = hmc_driver.build_parser().parse_args(arguments)
    fixed_scales = hmc_driver._expand_values(
        parsed.fixed_coefficient_position_scale,
        size=6,
        name="fixed_coefficient_position_scale",
    )
    settings = hmc_driver._requested_kernel_settings(
        parsed,
        fixed_scales,
    )
    expected_matrix = np.diag([0.0, 0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    expected_matrix[:2, :2] = [[2.0, 0.25], [0.25, 2.0]]
    np.testing.assert_array_equal(settings.position_scale_matrix, expected_matrix)
    np.testing.assert_array_equal(
        settings.position_scale_diagonal,
        [2.0, 2.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    )
    eigendirections = np.eye(8)
    eigendirections[:, :2] = 0.0
    eigendirections[:2, 0] = np.array([1.0, 1.0]) / np.sqrt(2.0)
    eigendirections[:2, 1] = np.array([1.0, -1.0]) / np.sqrt(2.0)
    eigenvalues = np.array([2.25, 1.75, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    np.testing.assert_allclose(
        settings.position_scale_matrix @ eigendirections,
        eigendirections * eigenvalues,
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )
    assert "position_covariance" in (hmc_driver.FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID)
    assert "momentum_precision" in (hmc_driver.FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID)
