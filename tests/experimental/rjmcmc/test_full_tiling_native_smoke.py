"""Tests for the frozen-input full-tiling native smoke driver."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="module")
def smoke_module() -> ModuleType:
    """Load the example driver as a module without requiring a package import."""
    path = Path(__file__).parents[3] / "examples" / "rjmcmc" / "full_tiling_native_smoke.py"
    specification = importlib.util.spec_from_file_location("full_tiling_native_smoke", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the full-tiling smoke-test example.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_frozen_input(path: Path, *, nominal_weight: float = 0.25) -> None:
    """Write a tiny exact-closure native dataset with six labelled outers."""
    sensitivity = np.arange(1.0, 13.0).reshape(3, 2, 2)
    outer = np.arange(18.0).reshape(3, 6) / 8.0
    boundary = np.array([4.0, 5.0, 6.0])
    dataset = xr.Dataset(
        {
            "fp_x_flux": (("lon", "nmeasure", "lat"), sensitivity.transpose(2, 0, 1)),
            "mf": ("nmeasure", boundary + sensitivity.sum(axis=(1, 2)) + outer.sum(axis=1)),
            "mf_error": ("nmeasure", np.ones(3)),
            "nominal_weight": (
                ("lon", "lat"),
                np.full((2, 2), nominal_weight).T,
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


def _arguments(input_path: Path, output_path: Path) -> list[str]:
    """Return the common explicit scientific and sampler CLI arguments."""
    return [
        "--input",
        str(input_path),
        "--output-directory",
        str(output_path),
        "--k",
        "2",
        "--cycles",
        "1",
        "--seed",
        "812",
        "--concentration",
        "3",
        "--root-variance",
        "0.25",
        "--fixed-prior-sd",
        "1",
        "--fixed-proposal-sd",
        "0.2",
        "--input-id",
        "tiny-frozen-native-v1",
        "--code-revision",
        "test-revision",
        "--nominal-weight-policy",
        "positive-native-mass-v1",
    ]


def test_dry_run_enforces_sha_and_reviewed_profile_gates(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Dry-run validates exact hashes and rejects a non-PARIS input profile."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "dry-output"
    _write_frozen_input(input_path)
    digest = smoke_module._sha256_file(input_path)

    summary = smoke_module.run(
        smoke_module.build_parser().parse_args(
            [
                *_arguments(input_path, output_path),
                "--dry-run",
                "--expected-input-sha256",
                digest,
            ]
        )
    )
    assert summary["status"] == "diagnostic_dry_run"
    assert summary["cycle_length"] == 14
    assert summary["requested_atomic_transitions"] == 14
    assert summary["input"]["sha256"] == digest
    assert np.isfinite(summary["target"]["initial_log_target"])
    assert summary["target"]["normalization"].startswith("fixed-K")
    assert summary["initial_leaf_prior_scaling_sd"]["minimum"] > 0.0
    assert summary["closure"] == {
        "mass_coordinate_max_abs_error": 0.0,
        "prior_mean_total_max_abs_error": 0.0,
    }
    assert not output_path.exists()

    with pytest.raises(ValueError, match="SHA-256"):
        smoke_module.run(
            smoke_module.build_parser().parse_args(
                [
                    *_arguments(input_path, output_path),
                    "--dry-run",
                    "--expected-input-sha256",
                    "0" * 64,
                ]
            )
        )
    with pytest.raises(ValueError, match="expected 1382 observations"):
        smoke_module.run(
            smoke_module.build_parser().parse_args(
                [
                    *_arguments(input_path, output_path),
                    "--dry-run",
                    "--require-paris-profile",
                    "--expected-input-sha256",
                    digest,
                    "--expected-outer-labels",
                    ",".join(f"region-{index}" for index in range(6)),
                ]
            )
        )


def test_dry_run_without_weight_normalization_closes_all_prior_mean_terms(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Nonunit native masses still close raw inner, BC, and outer predictions."""
    input_path = tmp_path / "nonunit-weights.nc"
    output_path = tmp_path / "dry-output"
    _write_frozen_input(input_path, nominal_weight=2.5)

    summary = smoke_module.run(
        smoke_module.build_parser().parse_args(
            [
                *_arguments(input_path, output_path),
                "--dry-run",
                "--no-normalize-weights",
            ]
        )
    )

    assert summary["status"] == "diagnostic_dry_run"
    assert summary["manifest"]["input"]["weight_normalization_factor"] == 10.0
    assert summary["manifest"]["input"]["contract"]["normalize_weights"] is False
    assert np.isfinite(summary["target"]["initial_log_target"])
    assert summary["closure"] == {
        "mass_coordinate_max_abs_error": 0.0,
        "prior_mean_total_max_abs_error": 0.0,
    }
    assert not output_path.exists()


def test_tiny_end_to_end_bundle_reopens_with_labels_attrs_and_hashes(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """One tiny cycle writes a complete auditable fourteen-attempt bundle."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "output"
    _write_frozen_input(input_path)

    summary = smoke_module.run(smoke_module.build_parser().parse_args(_arguments(input_path, output_path)))

    assert {path.name for path in output_path.iterdir()} == {
        "manifest.json",
        "trace.nc",
        "summary.json",
        "complete.json",
    }
    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((output_path / "complete.json").read_text(encoding="utf-8"))
    assert summary["run"]["atomic_transitions"] == 14
    assert manifest["kernel"]["cycle_length"] == 14
    assert completion["atomic_transitions"] == 14

    with xr.open_dataset(output_path / "trace.nc", engine="h5netcdf") as trace:
        assert trace.sizes["transition"] == 14
        assert trace.sizes["fixed_parameter"] == 6
        assert trace.sizes["region"] == 2
        np.testing.assert_array_equal(trace["global_transition"], np.arange(1, 15))
        np.testing.assert_array_equal(
            trace["fixed_parameter"],
            [f"region-{index}" for index in range(6)],
        )
        np.testing.assert_array_equal(
            trace["bound"],
            ["row_start", "row_stop", "col_start", "col_stop"],
        )
        np.testing.assert_array_equal(trace["lat"], [50.0, 51.0])
        np.testing.assert_array_equal(trace["lon"], [-2.0, -1.0])
        assert trace.attrs["diagnostic_only"] == "true"
        assert trace.attrs["convergence_claim"] == "none"
        assert trace.attrs["connectivity_proven"] == "false"
        assert trace.attrs["fixed_k"] == 2
        assert trace.attrs["input_sha256"] == smoke_module._sha256_file(input_path)
        assert trace.attrs["schedule_id"] == manifest["kernel"]["schedule_id"]
        np.testing.assert_array_equal(
            trace["log_structural_prior"],
            np.zeros(trace.sizes["draw"]),
        )
        assert trace.attrs["manifest_sha256"] == smoke_module._sha256_file(output_path / "manifest.json")

    assert set(completion["sha256"]) == {
        "manifest.json",
        "trace.nc",
        "summary.json",
    }
    for filename, digest in completion["sha256"].items():
        assert smoke_module._sha256_file(output_path / filename) == digest
