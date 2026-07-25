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
    """Write a tiny exact-closure native dataset with six labelled outers."""
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
                np.full((2, 2), 0.25).T,
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
        "calibration_id": "tiny-static-hmc-calibration-v1",
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
            "leaf_position_scale": 1.75,
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
            "clipping_bounds": [1.0e-4, 1.0e2],
            "pilot_initializers": [
                {
                    "strategy": "largest-nominal",
                    "seed": None,
                    "sweeps": 100,
                },
                {
                    "strategy": "random-recursive",
                    "seed": 51003,
                    "sweeps": 100,
                },
            ],
            "candidate_grid": [{"step_size": 0.001, "leapfrog_steps": 1}],
            "decision_statistics": [
                {
                    "step_size": 0.001,
                    "leapfrog_steps": 1,
                    "largest_nominal_mean_acceptance": 0.75,
                    "random_recursive_mean_acceptance": 0.8,
                    "divergences": 0,
                    "finite": True,
                    "median_log_displacement_per_leapfrog_step": 0.125,
                    "selected": True,
                }
            ],
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
        "--leaf-position-scale",
        "1.75",
        "--fixed-coefficient-position-scale",
        "0.5,0.75,1,1.25,1.5,2",
        "--calibration-id",
        "tiny-static-hmc-calibration-v1",
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
    """A synthetic chain dry-runs, publishes in order, and resumes exactly.

    The same immutable chain is also used to prove that calibration,
    provenance, and static-HMC setting changes are rejected by checkpoint
    manifest validation before any continuation output is published.
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
    assert manifest["schema"].endswith("native_manifest.v1")
    assert manifest["initialization"]["seed"] is None
    assert manifest["sampler"]["metric_semantics_id"] == (hmc_driver.FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID)
    assert manifest["sampler"]["leaf_position_scale"] == 1.75
    assert manifest["sampler"]["fixed_coefficient_position_scale"] == [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    assert manifest["sampler"]["calibration"] == {
        "schema": hmc_driver.CALIBRATION_SCHEMA,
        "id": "tiny-static-hmc-calibration-v1",
        "sha256": hmc_driver._sha256_file(input_path.with_name("calibration.json")),
    }
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
    assert completion["schema"].endswith("native_completion.v1")
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
    assert checkpoint_metadata["schema_version"] == 1
    assert checkpoint_metadata["sweeps_completed"] == 1
    assert checkpoint_metadata["runtime_identity"]["pytensor_float_x"] == ("float64")
    assert checkpoint_metadata["run_manifest_json"] == (hmc_driver._canonical_json(manifest).rstrip("\n"))

    with xr.open_dataset(
        fresh_output / hmc_driver.TRACE_FILENAME,
        engine="h5netcdf",
    ) as fresh_trace:
        fresh_trace.load()
        assert fresh_trace.attrs["schema"].endswith("native_trace.v1")
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
        assert fresh_trace["hmc_seed"].dtype == np.dtype(np.uint64)
        fresh_final_mass = fresh_trace["leaf_mass"].isel(draw=-1).values
        fresh_final_fixed = fresh_trace["fixed_coefficient"].isel(draw=-1).values

    checkpoint_path = fresh_output / hmc_driver.CHECKPOINT_FILENAME
    mismatch_cases = (
        (
            "--calibration-id",
            "different-calibration",
            "Calibration v1 identity",
        ),
        ("--chain-id", "different-chain", "manifest does not match"),
        ("--step-size", "0.002", "Calibration v1 identity"),
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
    np.testing.assert_array_equal(
        settings.position_scale_diagonal,
        [1.75, 1.75, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    )
    assert "position_covariance" in (hmc_driver.FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID)
    assert "momentum_precision" in (hmc_driver.FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID)
