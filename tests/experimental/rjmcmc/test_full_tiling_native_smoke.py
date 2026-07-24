"""Tests for the frozen-input full-tiling native smoke driver."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
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


def _arguments(
    input_path: Path,
    output_path: Path,
    *,
    iterations: int | None = None,
) -> list[str]:
    """Return the common explicit scientific and sampler CLI arguments."""
    segment = ["--cycles", "1"] if iterations is None else ["--iterations", str(iterations)]
    return [
        "--input",
        str(input_path),
        "--output-directory",
        str(output_path),
        "--k",
        "2",
        *segment,
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
        "--chain-id",
        "tiny-chain",
    ]


def _checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load checkpoint arrays without pickle for exact artifact comparisons."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


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

    with pytest.raises(ValueError, match="cannot be combined"):
        smoke_module.run(
            smoke_module.build_parser().parse_args(
                [
                    *_arguments(input_path, output_path),
                    "--dry-run",
                    "--resume-checkpoint",
                    str(tmp_path / "missing-checkpoint.npz"),
                ]
            )
        )

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
        "checkpoint.npz",
        "complete.json",
    }
    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((output_path / "complete.json").read_text(encoding="utf-8"))
    assert summary["run"]["atomic_transitions"] == 14
    assert summary["schema"].endswith("summary.v2")
    assert manifest["kernel"]["cycle_length"] == 14
    assert manifest["schema"].endswith("manifest.v2")
    assert manifest["provenance"]["durable_checkpoint"] is True
    assert "path" not in manifest["input"]
    assert manifest["model"]["root_prior_shape"] == 4.0
    assert manifest["model"]["root_prior_rate"] == 4.0
    assert completion["atomic_transitions"] == 14
    assert completion["schema"].endswith("completion.v2")
    assert completion["durable_checkpoint"] is True
    assert completion["parent_checkpoint_sha256"] is None
    assert len(completion["segment_start_state_sha256"]) == 64

    with xr.open_dataset(output_path / "trace.nc", engine="h5netcdf") as trace:
        assert trace.sizes["transition"] == 14
        assert trace.sizes["fixed_parameter"] == 6
        assert trace.sizes["region"] == 2
        np.testing.assert_array_equal(trace["global_transition"], np.arange(1, 15))
        np.testing.assert_array_equal(
            trace["transition"],
            trace["global_transition"],
        )
        np.testing.assert_array_equal(trace["draw"], trace["state_transition"])
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
        assert trace.attrs["schema"].endswith("trace.v2")
        assert trace.attrs["movement_diagnostics_collected"] == "false"
        assert "slice_log_density_evaluations" not in trace

    assert set(completion["sha256"]) == {
        "manifest.json",
        "trace.nc",
        "summary.json",
        "checkpoint.npz",
    }
    for filename, digest in completion["sha256"].items():
        assert smoke_module._sha256_file(output_path / filename) == digest


def test_awkward_checkpoint_continuation_matches_direct_segment(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """A durable 5+9 transition run exactly replays one direct 14-transition run."""
    input_path = tmp_path / "frozen.nc"
    first_path = tmp_path / "first"
    resumed_path = tmp_path / "resumed"
    direct_path = tmp_path / "direct"
    _write_frozen_input(input_path)

    first = smoke_module.run(
        smoke_module.build_parser().parse_args(_arguments(input_path, first_path, iterations=5))
    )
    resumed_arguments = [
        *_arguments(input_path, resumed_path, iterations=9),
        "--resume-checkpoint",
        str(first_path / "checkpoint.npz"),
    ]
    resumed = smoke_module.run(smoke_module.build_parser().parse_args(resumed_arguments))
    direct = smoke_module.run(
        smoke_module.build_parser().parse_args(_arguments(input_path, direct_path, iterations=14))
    )

    assert resumed["run"]["segment_start_transition"] == 5
    assert resumed["run"]["segment_end_transition"] == 14
    assert resumed["run"]["schedule_phase_start"] == 5
    assert resumed["run"]["schedule_phase_end"] == 0
    assert direct["run"]["segment_start_transition"] == 0
    assert resumed["lineage"]["parent_checkpoint_sha256"] == (
        smoke_module._sha256_file(first_path / "checkpoint.npz")
    )
    assert len(resumed["lineage"]["segment_start_state_sha256"]) == 64
    assert resumed["target"]["segment_initial_log_target"] == first["target"]["final_log_target"]
    assert resumed["target"]["chain_initial_log_target"] == direct["target"]["chain_initial_log_target"]

    with (
        xr.open_dataset(first_path / "trace.nc", engine="h5netcdf") as first_trace,
        xr.open_dataset(resumed_path / "trace.nc", engine="h5netcdf") as resumed_trace,
        xr.open_dataset(direct_path / "trace.nc", engine="h5netcdf") as direct_trace,
    ):
        for name in (
            "global_transition",
            "slot",
            "move",
            "valid",
            "accepted",
            "log_acceptance_ratio",
            "invalid_reason",
        ):
            combined = np.concatenate((first_trace[name].values, resumed_trace[name].values))
            np.testing.assert_array_equal(combined, direct_trace[name].values)
        for name in (
            "rectangle_bounds",
            "leaf_mass",
            "root_total",
            "fixed_coefficient",
            "log_target",
            "state_transition",
        ):
            combined = np.concatenate((first_trace[name].values, resumed_trace[name].values))
            np.testing.assert_array_equal(combined, direct_trace[name].values)

    resumed_checkpoint = _checkpoint_arrays(resumed_path / "checkpoint.npz")
    direct_checkpoint = _checkpoint_arrays(direct_path / "checkpoint.npz")
    assert resumed_checkpoint.keys() == direct_checkpoint.keys()
    for name in resumed_checkpoint:
        np.testing.assert_array_equal(resumed_checkpoint[name], direct_checkpoint[name])


def test_output_only_diagnostics_preserve_trajectory_rng_and_persist_all_fields(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Diagnostics add aligned metrics without changing ordinary output or PCG64."""
    input_path = tmp_path / "frozen.nc"
    ordinary_path = tmp_path / "ordinary"
    diagnosed_path = tmp_path / "diagnosed"
    _write_frozen_input(input_path)
    common = _arguments(input_path, ordinary_path, iterations=14)
    smoke_module.run(smoke_module.build_parser().parse_args(common))
    diagnosed_arguments = _arguments(input_path, diagnosed_path, iterations=14)
    diagnosed_arguments.append("--collect-movement-diagnostics")
    summary = smoke_module.run(smoke_module.build_parser().parse_args(diagnosed_arguments))

    diagnostic_fields = {
        "proposal_elapsed_ns",
        "diagnostic_elapsed_ns",
        "source_merge_count",
        "destination_catalogue_size",
        "pair_catalogue_size",
        "design_cache_misses",
        "changed_native_cell_count",
        "changed_nominal_mass",
        "standardized_prediction_l2",
        "root_abs_displacement",
        "root_abs_log_displacement",
        "allocation_share_l1_displacement",
        "fixed_position",
        "fixed_abs_displacement",
        "fixed_abs_log_displacement",
        "slice_left_steps",
        "slice_right_steps",
        "slice_shrink_draws",
        "slice_log_density_evaluations",
    }
    with (
        xr.open_dataset(ordinary_path / "trace.nc", engine="h5netcdf") as ordinary,
        xr.open_dataset(diagnosed_path / "trace.nc", engine="h5netcdf") as diagnosed,
    ):
        assert ordinary.attrs["movement_diagnostics_collected"] == "false"
        assert diagnosed.attrs["movement_diagnostics_collected"] == "true"
        assert diagnostic_fields <= set(diagnosed.data_vars)
        assert diagnostic_fields.isdisjoint(ordinary.data_vars)
        for name in ordinary.data_vars:
            np.testing.assert_array_equal(ordinary[name].values, diagnosed[name].values)
        for name in diagnostic_fields:
            assert diagnosed[name].dims == ("transition",)
            assert diagnosed[name].attrs["diagnostic_role"] == "output_only"

    ordinary_checkpoint = _checkpoint_arrays(ordinary_path / "checkpoint.npz")
    diagnosed_checkpoint = _checkpoint_arrays(diagnosed_path / "checkpoint.npz")
    for name in ordinary_checkpoint:
        np.testing.assert_array_equal(ordinary_checkpoint[name], diagnosed_checkpoint[name])
    assert summary["movement_diagnostics"]["slice"]["attempts"] == 1
    assert summary["movement_diagnostics"]["slice"]["log_density_evaluations_total"] >= 2


@pytest.mark.parametrize(
    ("changed_argument", "changed_value"),
    [
        ("--seed", "813"),
        ("--root-slice-width", "2.0"),
    ],
)
def test_resume_rejects_chain_identity_mismatch(
    tmp_path: Path,
    smoke_module: ModuleType,
    changed_argument: str,
    changed_value: str,
) -> None:
    """Resume rejects seed and resolved-kernel manifest mismatches."""
    input_path = tmp_path / "frozen.nc"
    first_path = tmp_path / "first"
    output_path = tmp_path / "resume"
    _write_frozen_input(input_path)
    smoke_module.run(smoke_module.build_parser().parse_args(_arguments(input_path, first_path, iterations=5)))
    arguments = _arguments(input_path, output_path, iterations=9)
    if changed_argument in arguments:
        position = arguments.index(changed_argument)
        arguments[position + 1] = changed_value
    else:
        arguments.extend([changed_argument, changed_value])
    arguments.extend(["--resume-checkpoint", str(first_path / "checkpoint.npz")])
    with pytest.raises(ValueError, match="manifest"):
        smoke_module.run(smoke_module.build_parser().parse_args(arguments))


def test_resume_rejects_input_mismatch_and_output_never_overwrites(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Resume checks the frozen input identity and refuses an existing bundle path."""
    input_path = tmp_path / "frozen.nc"
    relocated_input_path = tmp_path / "relocated.nc"
    changed_input_path = tmp_path / "changed.nc"
    first_path = tmp_path / "first"
    relocated_output_path = tmp_path / "relocated-resume"
    output_path = tmp_path / "resume"
    _write_frozen_input(input_path)
    shutil.copyfile(input_path, relocated_input_path)
    _write_frozen_input(changed_input_path, nominal_weight=0.5)
    smoke_module.run(smoke_module.build_parser().parse_args(_arguments(input_path, first_path, iterations=5)))
    relocated_arguments = [
        *_arguments(relocated_input_path, relocated_output_path, iterations=1),
        "--resume-checkpoint",
        str(first_path / "checkpoint.npz"),
    ]
    relocated = smoke_module.run(smoke_module.build_parser().parse_args(relocated_arguments))
    assert relocated["run"]["segment_start_transition"] == 5
    assert relocated["run"]["segment_end_transition"] == 6
    arguments = [
        *_arguments(changed_input_path, output_path, iterations=9),
        "--resume-checkpoint",
        str(first_path / "checkpoint.npz"),
    ]
    with pytest.raises(ValueError, match="manifest"):
        smoke_module.run(smoke_module.build_parser().parse_args(arguments))

    marker = first_path / "complete.json"
    original = marker.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        smoke_module.run(
            smoke_module.build_parser().parse_args(_arguments(input_path, first_path, iterations=1))
        )
    assert marker.read_bytes() == original
