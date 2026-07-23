"""Tests for the frozen-input Gamma--Beta real-data smoke-test driver."""

from __future__ import annotations

import importlib.util
import errno
import json
from pathlib import Path
import shutil
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.experimental.rjmcmc.gamma_beta_io import (
    GAMMA_BETA_TRACE_SCHEMA_ID,
)

EXAMPLE_PATH = Path(__file__).resolve().parents[3] / "examples" / "rjmcmc" / "gamma_beta_native_smoke.py"


@pytest.fixture(scope="module")
def smoke_module() -> ModuleType:
    """Dynamically import the repository-root example module."""
    specification = importlib.util.spec_from_file_location(
        "gamma_beta_native_smoke_example",
        EXAMPLE_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load Gamma-Beta smoke-test example.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_frozen_input(path: Path) -> None:
    """Write a tiny exact-closure native-grid dataset with six outer columns."""
    nmeasure = np.arange(3, dtype=np.int64)
    lat = np.array([50.0, 51.0])
    lon = np.array([-2.0, -1.0])
    outer_region = np.arange(6, dtype=np.int64)
    sensitivity = np.arange(1.0, 13.0).reshape(3, 2, 2)
    outer_design = np.arange(18.0).reshape(3, 6) / 8.0
    fixed_offset = np.array([4.0, 5.0, 6.0])
    expected = fixed_offset + sensitivity.sum(axis=(1, 2)) + outer_design @ np.ones(6)
    dataset = xr.Dataset(
        {
            "fp_x_flux": (
                ("nmeasure", "lat", "lon"),
                sensitivity,
            ),
            "mf": ("nmeasure", expected),
            "mf_error": ("nmeasure", np.ones(3)),
            "nominal_weight": (("lat", "lon"), np.full((2, 2), 0.25)),
            "outer_design": (
                ("nmeasure", "outer_region"),
                outer_design,
            ),
            "YaprioriBC": ("nmeasure", fixed_offset),
        },
        coords={
            "nmeasure": nmeasure,
            "lat": lat,
            "lon": lon,
            "outer_region": outer_region,
        },
    )
    dataset.to_netcdf(path, engine="h5netcdf")


def _write_larger_frozen_input(path: Path) -> None:
    """Write a 4x4 exact-closure dataset with several random frontiers."""
    nmeasure = np.arange(3, dtype=np.int64)
    lat = np.arange(4, dtype=np.float64) + 48.0
    lon = np.arange(4, dtype=np.float64) - 3.0
    outer_region = np.arange(6, dtype=np.int64)
    sensitivity = np.arange(1.0, 49.0).reshape(3, 4, 4)
    outer_design = np.arange(18.0).reshape(3, 6) / 8.0
    fixed_offset = np.array([4.0, 5.0, 6.0])
    expected = fixed_offset + sensitivity.sum(axis=(1, 2)) + outer_design @ np.ones(6)
    dataset = xr.Dataset(
        {
            "fp_x_flux": (("nmeasure", "lat", "lon"), sensitivity),
            "mf": ("nmeasure", expected),
            "mf_error": ("nmeasure", np.ones(3)),
            "nominal_weight": (("lat", "lon"), np.full((4, 4), 1.0 / 16.0)),
            "outer_design": (("nmeasure", "outer_region"), outer_design),
            "YaprioriBC": ("nmeasure", fixed_offset),
        },
        coords={
            "nmeasure": nmeasure,
            "lat": lat,
            "lon": lon,
            "outer_region": outer_region,
        },
    )
    dataset.to_netcdf(path, engine="h5netcdf")


def _common_arguments(input_path: Path, output_path: Path) -> list[str]:
    """Return shared explicit scientific and sampler CLI arguments."""
    return [
        "--input",
        str(input_path),
        "--output-directory",
        str(output_path),
        "--k-min",
        "1",
        "--k-max",
        "4",
        "--start-k",
        "2",
        "--concentration",
        "3",
        "--root-variance",
        "0.25",
        "--fixed-prior-sd",
        "1",
        "--fixed-proposal-sd",
        "0.2",
        "--warmup",
        "0",
        "--thin",
        "1",
        "--seed",
        "812",
        "--chain-id",
        "test-chain-0",
        "--code-revision",
        "test-revision",
        "--input-id",
        "test-frozen-native-v1",
        "--nominal-weight-policy",
        "test-positive-native-mass-v1",
    ]


def test_fresh_complete_cycle_writes_auditable_durable_segment(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """One default cycle should write 16 labelled attempts and exact closure."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "fresh"
    _write_frozen_input(input_path)

    assert smoke_module.main([*_common_arguments(input_path, output_path), "--cycles", "1"]) == 0

    assert {path.name for path in output_path.iterdir()} == {
        "manifest.json",
        "checkpoint.npz",
        "trace.nc",
        "summary.json",
        "complete.json",
    }
    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_path / "summary.json").read_text(encoding="utf-8"))
    completion = json.loads((output_path / "complete.json").read_text(encoding="utf-8"))
    with xr.open_dataset(output_path / "trace.nc", engine="h5netcdf") as trace:
        assert trace.attrs["schema_id"] == GAMMA_BETA_TRACE_SCHEMA_ID
        assert trace.sizes["attempt"] == 16
        assert trace.sizes["fixed_parameter"] == 6
        np.testing.assert_array_equal(trace["global_transition"], np.arange(1, 17))
        np.testing.assert_array_equal(
            trace["slot"],
            (
                "structural",
                "structural",
                "relocation",
                "subtree_retile",
                "root",
                "fraction",
                "fraction",
                "fraction",
                "fraction",
                "fraction",
                "fixed",
                "fixed",
                "fixed",
                "fixed",
                "fixed",
                "fixed",
            ),
        )
        assert trace["move"].values[2] == "relocate"
        assert trace["move"].values[3] == "subtree_retile"
        assert np.all(trace["secondary_node_id"].values[trace["move"].values != "relocate"] == -1)
        assert np.all(trace["block_leaf_count"].values[trace["move"].values != "subtree_retile"] == -1)
        np.testing.assert_array_equal(trace["latitude"], [50.0, 51.0])
        np.testing.assert_array_equal(trace["longitude"], [-2.0, -1.0])
        np.testing.assert_array_equal(
            trace["fixed_parameter"],
            [str(value) for value in range(6)],
        )
        np.testing.assert_array_equal(
            trace["measurement"],
            [str(value) for value in range(3)],
        )
        assert "split_node_id" in trace
        assert "secondary_node_id" in trace
        assert "block_leaf_count" in trace
        assert "node_row_start" in trace
        assert trace.attrs["problem_sha256"] == manifest["problem_sha256"]

    assert manifest["schedule"]["cycle_length"] == 16
    assert manifest["schedule"]["relocation_slots"] == 1
    assert manifest["schedule"]["subtree_retile_slots"] == 1
    assert manifest["schedule"]["max_subtree_leaves"] == 8
    assert manifest["chain"]["id"] == "test-chain-0"
    assert manifest["chain"]["initial_k"] == 2
    assert summary["closure"] == {
        "mass_coordinate_max_abs_error": 0.0,
        "prior_mean_total_max_abs_error": 0.0,
    }
    assert summary["segment"]["atomic_transitions"] == 16
    assert summary["segment"]["schedule_phase_end"] == 0
    assert sum(move["attempts"] for move in summary["moves"].values()) == 16
    assert summary["moves"]["relocate"]["attempts"] == 1
    assert summary["moves"]["subtree_retile"]["attempts"] == 1
    assert [value["attempts"] for value in summary["fixed_coefficients"].values()] == [1] * 6
    assert completion["transitions_completed"] == 16
    assert set(completion["sha256"]) == {
        "manifest.json",
        "checkpoint.npz",
        "trace.nc",
        "summary.json",
    }
    for name, digest in completion["sha256"].items():
        assert smoke_module._sha256_file(output_path / name) == digest
    assert (output_path / "checkpoint.npz").is_file()

    with pytest.raises(FileExistsError, match="already exists"):
        smoke_module.main([*_common_arguments(input_path, output_path), "--cycles", "1"])


def test_resume_from_mid_cycle_preserves_global_attempt_coordinates(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """A durable partial-cycle resume should continue attempts and phase exactly."""
    input_path = tmp_path / "frozen.nc"
    first_output = tmp_path / "part-one"
    second_output = tmp_path / "part-two"
    direct_output = tmp_path / "direct"
    restaged_input = tmp_path / "restaged-frozen.nc"
    _write_frozen_input(input_path)
    shutil.copyfile(input_path, restaged_input)

    assert smoke_module.main([*_common_arguments(input_path, first_output), "--iterations", "5"]) == 0
    assert (
        smoke_module.main(
            [
                *_common_arguments(restaged_input, second_output),
                "--iterations",
                "11",
                "--resume-checkpoint",
                str(first_output / "checkpoint.npz"),
            ]
        )
        == 0
    )
    assert smoke_module.main([*_common_arguments(input_path, direct_output), "--cycles", "1"]) == 0

    first_summary = json.loads((first_output / "summary.json").read_text(encoding="utf-8"))
    second_summary = json.loads((second_output / "summary.json").read_text(encoding="utf-8"))
    with (
        xr.open_dataset(first_output / "trace.nc", engine="h5netcdf") as first,
        xr.open_dataset(second_output / "trace.nc", engine="h5netcdf") as second,
        xr.open_dataset(direct_output / "trace.nc", engine="h5netcdf") as direct,
    ):
        np.testing.assert_array_equal(first["global_transition"], np.arange(1, 6))
        np.testing.assert_array_equal(second["global_transition"], np.arange(6, 17))
        for name in (
            "slot",
            "move",
            "valid",
            "accepted",
            "node_id",
            "secondary_node_id",
            "block_leaf_count",
            "coefficient_id",
            "k_after",
        ):
            combined = np.concatenate((first[name].values, second[name].values))
            np.testing.assert_array_equal(combined, direct[name].values)
        for name in (
            "root_total",
            "fixed_coefficients",
            "k",
            "log_gaussian_likelihood",
            "log_likelihood",
            "log_root_prior",
            "log_fraction_prior",
            "log_partition_prior",
            "log_fixed_coefficient_prior",
            "log_target",
        ):
            combined = np.concatenate((first[name].values, second[name].values), axis=0)
            np.testing.assert_array_equal(combined, direct[name].values)
        np.testing.assert_array_equal(
            np.concatenate((first["state_transition"].values, second["state_transition"].values)),
            direct["state_transition"].values,
        )
        for segmented in (first, second):
            for name in (
                "node_row_start",
                "node_row_stop",
                "node_column_start",
                "node_column_stop",
                "node_depth",
                "node_parent_id",
                "node_first_child_id",
                "node_second_child_id",
            ):
                np.testing.assert_array_equal(segmented[name].values, direct[name].values)
        combined_frontiers = [
            tuple(dataset["frontier_node_id"].values[draw][dataset["frontier_active"].values[draw]].tolist())
            for dataset in (first, second)
            for draw in range(dataset.sizes["draw"])
        ]
        direct_frontiers = [
            tuple(direct["frontier_node_id"].values[draw][direct["frontier_active"].values[draw]].tolist())
            for draw in range(direct.sizes["draw"])
        ]
        assert combined_frontiers == direct_frontiers
        combined_splits = [
            (
                tuple(dataset["split_node_id"].values[draw][dataset["split_active"].values[draw]].tolist()),
                tuple(dataset["split_fraction"].values[draw][dataset["split_active"].values[draw]].tolist()),
            )
            for dataset in (first, second)
            for draw in range(dataset.sizes["draw"])
        ]
        direct_splits = [
            (
                tuple(direct["split_node_id"].values[draw][direct["split_active"].values[draw]].tolist()),
                tuple(direct["split_fraction"].values[draw][direct["split_active"].values[draw]].tolist()),
            )
            for draw in range(direct.sizes["draw"])
        ]
        assert combined_splits == direct_splits
    with (
        np.load(second_output / "checkpoint.npz", allow_pickle=False) as resumed,
        np.load(direct_output / "checkpoint.npz", allow_pickle=False) as direct,
    ):
        assert resumed.files == direct.files
        for name in resumed.files:
            np.testing.assert_array_equal(resumed[name], direct[name])
    assert first_summary["segment"]["schedule_phase_end"] == 5
    assert second_summary["segment"]["transitions_start"] == 5
    assert second_summary["segment"]["transitions_end"] == 16
    assert second_summary["segment"]["schedule_phase_end"] == 0


def test_resume_rejects_an_incomplete_or_changed_segment_bundle(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Resume must validate the completion marker and every artifact hash."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "source"
    next_output = tmp_path / "next"
    _write_frozen_input(input_path)
    assert smoke_module.main([*_common_arguments(input_path, output_path), "--iterations", "5"]) == 0

    (output_path / "complete.json").unlink()
    with pytest.raises(ValueError, match="no completed segment marker"):
        smoke_module.main(
            [
                *_common_arguments(input_path, next_output),
                "--iterations",
                "1",
                "--resume-checkpoint",
                str(output_path / "checkpoint.npz"),
            ]
        )

    completion = {
        "checkpoint": "checkpoint.npz",
        "sha256": {
            name: smoke_module._sha256_file(output_path / name)
            for name in ("manifest.json", "checkpoint.npz", "trace.nc", "summary.json")
        },
    }
    (output_path / "complete.json").write_text(
        json.dumps(completion),
        encoding="utf-8",
    )
    with (output_path / "summary.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="summary.json"):
        smoke_module.main(
            [
                *_common_arguments(input_path, next_output),
                "--iterations",
                "1",
                "--resume-checkpoint",
                str(output_path / "checkpoint.npz"),
            ]
        )


def test_dry_run_validates_without_creating_output(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Dry-run mode should audit a segment without creating its reserved path."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "dry-run"
    _write_frozen_input(input_path)

    arguments = smoke_module.build_parser().parse_args(
        [*_common_arguments(input_path, output_path), "--cycles", "100", "--dry-run"]
    )
    summary = smoke_module.run(arguments)

    assert not output_path.exists()
    assert summary["input"]["id"] == "test-frozen-native-v1"
    assert summary["input"]["sha256"] == smoke_module._sha256_file(input_path)
    assert summary["run_manifest"]["chain"]["initial_k"] == 2
    assert summary["run_manifest"]["chain"]["initial_state_sha256"]
    assert summary["run_manifest"]["k_prior"]["minimum"] == 1
    assert summary["run_manifest"]["k_prior"]["maximum"] == 4
    assert len(summary["run_manifest"]["k_prior"]["probability_by_k_sha256"]) == 64
    assert summary["cycle_length"] == 16
    assert summary["run_manifest"]["schedule"]["relocation_slots"] == 1
    assert summary["run_manifest"]["schedule"]["subtree_retile_slots"] == 1
    assert summary["run_manifest"]["schedule"]["max_subtree_leaves"] == 8

    bad_arguments = smoke_module.build_parser().parse_args(
        [
            *_common_arguments(input_path, output_path),
            "--cycles",
            "1",
            "--dry-run",
            "--expected-input-sha256",
            "0" * 64,
        ]
    )
    with pytest.raises(ValueError, match="SHA-256"):
        smoke_module.run(bad_arguments)


def test_random_initial_frontier_seed_replays_without_touching_sampler_seed(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """A separate PCG64 stream should replay topology and distinguish seeds."""
    input_path = tmp_path / "larger.nc"
    output_path = tmp_path / "unused"
    _write_larger_frozen_input(input_path)
    arguments = smoke_module.build_parser().parse_args(
        [
            *_common_arguments(input_path, output_path),
            "--cycles",
            "1",
            "--dry-run",
            "--start-k",
            "7",
            "--k-max",
            "16",
        ]
    )
    dataset = smoke_module._load_frozen_dataset(input_path, engine="h5netcdf")
    adapter = smoke_module._build_adapter(dataset, arguments)

    first = smoke_module._initial_state(adapter.problem, k=7, frontier_seed=41)
    replay = smoke_module._initial_state(adapter.problem, k=7, frontier_seed=41)
    different = smoke_module._initial_state(adapter.problem, k=7, frontier_seed=73)

    assert first.frontier == replay.frontier
    np.testing.assert_array_equal(first.active_fractions, replay.active_fractions)
    assert first.frontier != different.frontier
    assert arguments.seed == 812


def test_random_initial_frontier_preserves_closure_and_binds_manifest(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Random topology starts should preserve all-one closure and provenance."""
    input_path = tmp_path / "larger.nc"
    output_path = tmp_path / "dry-run"
    _write_larger_frozen_input(input_path)
    arguments = smoke_module.build_parser().parse_args(
        [
            *_common_arguments(input_path, output_path),
            "--cycles",
            "1",
            "--dry-run",
            "--start-k",
            "7",
            "--k-max",
            "16",
            "--initial-frontier-seed",
            "41",
        ]
    )

    summary = smoke_module.run(arguments)

    assert summary["closure"]["prior_mean_total_max_abs_error"] == pytest.approx(0.0, abs=1e-12)
    contract = json.loads(summary["run_manifest"]["inputs"]["input_variable_contract"]["identifier"])
    assert contract["initial_frontier"] == {
        "policy": "uniform_splittable_leaf_prior_mean_v1",
        "seed": 41,
    }


def test_absent_initial_frontier_seed_preserves_legacy_initialization(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Omitting the new option should retain the mass-greedy prior-mean start."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "dry-run"
    _write_frozen_input(input_path)
    arguments = smoke_module.build_parser().parse_args(
        [*_common_arguments(input_path, output_path), "--cycles", "1", "--dry-run"]
    )
    dataset = smoke_module._load_frozen_dataset(input_path, engine="h5netcdf")
    adapter = smoke_module._build_adapter(dataset, arguments)

    actual = smoke_module._initial_state(adapter.problem, k=2, frontier_seed=None)
    expected = smoke_module.initialize_gamma_beta_state(adapter.problem, k=2)

    assert arguments.initial_frontier_seed is None
    assert actual.frontier == expected.frontier
    np.testing.assert_array_equal(actual.active_fractions, expected.active_fractions)
    np.testing.assert_array_equal(actual.prediction, expected.prediction)
    summary = smoke_module.run(arguments)
    contract = json.loads(summary["run_manifest"]["inputs"]["input_variable_contract"]["identifier"])
    assert contract["initial_frontier"] == {
        "policy": "mass_greedy_prior_mean_v1",
        "seed": None,
    }


def test_fixed_k_topology_cli_defaults_help_and_manifest_are_configurable(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """CLI topology settings have reviewed defaults and bind the run manifest."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "dry-run"
    _write_frozen_input(input_path)
    parser = smoke_module.build_parser()
    defaults = parser.parse_args([*_common_arguments(input_path, output_path), "--cycles", "1", "--dry-run"])

    assert defaults.relocation_slots == 1
    assert defaults.subtree_retile_slots == 1
    assert defaults.max_subtree_leaves == 8
    help_text = parser.format_help()
    assert "--relocation-slots" in help_text
    assert "--subtree-retile-slots" in help_text
    assert "--max-subtree-leaves" in help_text

    configured = parser.parse_args(
        [
            *_common_arguments(input_path, output_path),
            "--cycles",
            "1",
            "--dry-run",
            "--relocation-slots",
            "2",
            "--subtree-retile-slots",
            "3",
            "--max-subtree-leaves",
            "4",
        ]
    )
    summary = smoke_module.run(configured)
    schedule = summary["run_manifest"]["schedule"]
    assert summary["cycle_length"] == 19
    assert schedule["relocation_slots"] == 2
    assert schedule["subtree_retile_slots"] == 3
    assert schedule["max_subtree_leaves"] == 4


def test_directory_sync_ignores_only_documented_unsupported_errors(
    tmp_path: Path,
    smoke_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared-filesystem directory-fsync rejection should not fail a segment."""
    monkeypatch.setattr(
        smoke_module.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError(errno.EINVAL, "unsupported")),
    )
    smoke_module._fsync_directory(tmp_path)

    monkeypatch.setattr(
        smoke_module.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError(errno.EIO, "I/O failure")),
    )
    with pytest.raises(OSError, match="I/O failure"):
        smoke_module._fsync_directory(tmp_path)


def test_broken_output_backend_fails_before_sampling(
    tmp_path: Path,
    smoke_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend preflight must fail before a chain or output directory starts."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "never-created"
    _write_frozen_input(input_path)
    sampler = Mock()
    monkeypatch.setattr(smoke_module, "sample_gamma_beta_compound", sampler)
    monkeypatch.setattr(
        smoke_module,
        "_preflight_output_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken backend")),
    )

    with pytest.raises(RuntimeError, match="broken backend"):
        smoke_module.main([*_common_arguments(input_path, output_path), "--cycles", "1"])
    sampler.assert_not_called()
    assert not output_path.exists()


def test_reviewed_profile_requires_exact_hash_and_outer_label_order(
    tmp_path: Path,
    smoke_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PARIS-profile validation must bind content and fixed-column semantics."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "dry-profile"
    _write_frozen_input(input_path)
    monkeypatch.setattr(smoke_module, "PARIS_OBSERVATIONS", 3)
    monkeypatch.setattr(smoke_module, "PARIS_GRID_SHAPE", (2, 2))
    monkeypatch.setattr(smoke_module, "PARIS_OUTER_COEFFICIENTS", 6)
    digest = smoke_module._sha256_file(input_path)
    profile_arguments = [
        *_common_arguments(input_path, output_path),
        "--cycles",
        "1",
        "--dry-run",
        "--require-paris-profile",
        "--expected-input-sha256",
        digest,
        "--expected-outer-labels",
        "0,1,2,3,4,5",
    ]

    summary = smoke_module.run(smoke_module.build_parser().parse_args(profile_arguments))
    assert summary["input"]["sha256"] == digest

    wrong_order = [*profile_arguments]
    wrong_order[-1] = "1,0,2,3,4,5"
    with pytest.raises(ValueError, match="labels/order"):
        smoke_module.run(smoke_module.build_parser().parse_args(wrong_order))


def test_resume_rejects_changed_variable_selection_with_identical_values(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """The manifest must bind configurable input-variable names and backend."""
    input_path = tmp_path / "frozen-with-alias.nc"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    _write_frozen_input(input_path)
    with xr.open_dataset(input_path, engine="h5netcdf") as opened:
        dataset = opened.load()
    dataset["fp_x_flux_alias"] = dataset["fp_x_flux"].copy(deep=True)
    dataset.to_netcdf(input_path, engine="h5netcdf", mode="w")

    assert smoke_module.main([*_common_arguments(input_path, first_output), "--iterations", "5"]) == 0
    with pytest.raises(ValueError, match="manifest"):
        smoke_module.main(
            [
                *_common_arguments(input_path, second_output),
                "--iterations",
                "1",
                "--sensitivity-name",
                "fp_x_flux_alias",
                "--resume-checkpoint",
                str(first_output / "checkpoint.npz"),
            ]
        )


def test_resume_rejects_changed_fixed_k_topology_schedule(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Resume rejects a fixed-K slot change bound into the run manifest."""
    input_path = tmp_path / "frozen.nc"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    _write_frozen_input(input_path)

    assert smoke_module.main([*_common_arguments(input_path, first_output), "--iterations", "5"]) == 0
    with pytest.raises(ValueError, match="manifest"):
        smoke_module.main(
            [
                *_common_arguments(input_path, second_output),
                "--iterations",
                "1",
                "--relocation-slots",
                "0",
                "--resume-checkpoint",
                str(first_output / "checkpoint.npz"),
            ]
        )


def test_resume_rejects_changed_initial_frontier_seed(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Resume must reject a changed topology-initialization stream contract."""
    input_path = tmp_path / "larger.nc"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    _write_larger_frozen_input(input_path)
    common = [
        *_common_arguments(input_path, first_output),
        "--start-k",
        "7",
        "--k-max",
        "16",
        "--initial-frontier-seed",
        "41",
    ]
    assert smoke_module.main([*common, "--iterations", "5"]) == 0

    resumed = [
        *_common_arguments(input_path, second_output),
        "--start-k",
        "7",
        "--k-max",
        "16",
        "--initial-frontier-seed",
        "73",
        "--iterations",
        "1",
        "--resume-checkpoint",
        str(first_output / "checkpoint.npz"),
    ]
    with pytest.raises(ValueError, match="manifest"):
        smoke_module.main(resumed)
