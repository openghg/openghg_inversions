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


def _fixed_basis_arguments(
    input_path: Path,
    output_path: Path,
    *,
    iterations: int | None = None,
) -> list[str]:
    """Return common arguments for the deterministic fixed-basis control."""
    return [
        *_arguments(input_path, output_path, iterations=iterations),
        "--structure-mode",
        "fixed-basis",
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
    assert manifest["schema"].endswith("manifest.v4")
    assert manifest["initialization"] == {
        "strategy": "largest-nominal",
        "seed": None,
        "rng_stream": "none",
    }
    assert manifest["state_space_scope"]["structure_mode"] == "mobile"
    assert manifest["state_space_scope"]["structural_target"] == (
        "uniform_over_unique_canonical_tilings_at_fixed_k"
    )
    assert manifest["state_space_scope"]["connectivity_proven"] is False
    assert manifest["kernel"]["structure_mode"] == "mobile"
    assert manifest["kernel"]["structural_slots"] == 2
    assert manifest["kernel"]["schedule_id"] == smoke_module.FULL_TILING_COMPOUND_SCHEDULE_ID
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


def test_fixed_basis_bundle_has_singleton_support_and_no_structural_attempts(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Fixed-basis mode runs one twelve-slot cycle on singleton topology support."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "fixed-basis"
    _write_frozen_input(input_path)

    summary = smoke_module.run(
        smoke_module.build_parser().parse_args(
            _fixed_basis_arguments(input_path, output_path),
        )
    )

    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    scope = manifest["state_space_scope"]
    kernel = manifest["kernel"]
    assert summary["run"]["atomic_transitions"] == 12
    assert manifest["schema"].endswith("manifest.v4")
    assert scope["structure_mode"] == "fixed-basis"
    assert scope["structural_target"] == "point_mass_at_recorded_deterministic_tiling"
    assert scope["structural_support"] == "singleton"
    assert scope["communication_component"] == "singleton_recorded_deterministic_basis"
    assert scope["connectivity_proven"] is True
    assert len(scope["initial_topology_sha256"]) == 64
    assert kernel["structure_mode"] == "fixed-basis"
    assert kernel["schedule_id"] == smoke_module.FIXED_BASIS_COMPOUND_SCHEDULE_ID
    assert kernel["cycle_length"] == 12
    assert kernel["structural_slots"] == 0
    assert kernel["pair_allocation_refresh_slots"] == 5
    assert manifest["initialization"] == {
        "strategy": "largest-nominal",
        "seed": None,
        "rng_stream": "none",
    }

    with xr.open_dataset(output_path / "trace.nc", engine="h5netcdf") as trace:
        assert trace.sizes["transition"] == 12
        np.testing.assert_array_equal(
            trace["slot"],
            ["root", *(["pair_allocation"] * 5), *(["fixed"] * 6)],
        )
        assert "structural" not in trace["slot"].values
        assert not np.isin(
            trace["move"].values,
            ["edge_flip", "resolution_relocation"],
        ).any()
        assert trace.attrs["schedule_id"] == smoke_module.FIXED_BASIS_COMPOUND_SCHEDULE_ID
        assert trace.attrs["connectivity_proven"] == "true"


def test_fixed_basis_seed_replay_is_exact(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """The fixed deterministic basis and sampler seed replay byte-exact state."""
    input_path = tmp_path / "frozen.nc"
    first_path = tmp_path / "first"
    replay_path = tmp_path / "replay"
    _write_frozen_input(input_path)

    smoke_module.run(
        smoke_module.build_parser().parse_args(
            _fixed_basis_arguments(input_path, first_path),
        )
    )
    smoke_module.run(
        smoke_module.build_parser().parse_args(
            _fixed_basis_arguments(input_path, replay_path),
        )
    )

    with (
        xr.open_dataset(first_path / "trace.nc", engine="h5netcdf") as first,
        xr.open_dataset(replay_path / "trace.nc", engine="h5netcdf") as replay,
    ):
        for name in first.data_vars:
            np.testing.assert_array_equal(first[name], replay[name])
    first_checkpoint = _checkpoint_arrays(first_path / "checkpoint.npz")
    replay_checkpoint = _checkpoint_arrays(replay_path / "checkpoint.npz")
    assert first_checkpoint.keys() == replay_checkpoint.keys()
    for name in first_checkpoint:
        np.testing.assert_array_equal(first_checkpoint[name], replay_checkpoint[name])


def test_fixed_basis_requires_the_deterministic_initialization(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Fixed-basis support rejects a random-recursive initial topology."""
    input_path = tmp_path / "frozen.nc"
    _write_frozen_input(input_path)
    arguments = [
        *_fixed_basis_arguments(input_path, tmp_path / "invalid", iterations=1),
        "--initialization",
        "random-recursive",
        "--initialization-seed",
        "41",
    ]

    with pytest.raises(ValueError, match="largest-nominal"):
        smoke_module.run(smoke_module.build_parser().parse_args(arguments))


@pytest.mark.parametrize(
    "initialization_arguments",
    [
        pytest.param([], id="largest-nominal"),
        pytest.param(
            [
                "--initialization",
                "random-recursive",
                "--initialization-seed",
                "41",
            ],
            id="random-recursive",
        ),
    ],
)
def test_awkward_checkpoint_continuation_matches_direct_segment(
    tmp_path: Path,
    smoke_module: ModuleType,
    initialization_arguments: list[str],
) -> None:
    """A durable 5+9 run replays a direct 14-transition run from either start."""
    input_path = tmp_path / "frozen.nc"
    first_path = tmp_path / "first"
    resumed_path = tmp_path / "resumed"
    direct_path = tmp_path / "direct"
    _write_frozen_input(input_path)

    first = smoke_module.run(
        smoke_module.build_parser().parse_args(
            [
                *_arguments(input_path, first_path, iterations=5),
                *initialization_arguments,
            ]
        )
    )
    resumed_arguments = [
        *_arguments(input_path, resumed_path, iterations=9),
        *initialization_arguments,
        "--resume-checkpoint",
        str(first_path / "checkpoint.npz"),
    ]
    resumed = smoke_module.run(smoke_module.build_parser().parse_args(resumed_arguments))
    direct = smoke_module.run(
        smoke_module.build_parser().parse_args(
            [
                *_arguments(input_path, direct_path, iterations=14),
                *initialization_arguments,
            ]
        )
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


def test_fixed_basis_awkward_checkpoint_matches_direct_cycle(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """A fixed-basis durable 5+7 run replays a direct twelve-transition cycle."""
    input_path = tmp_path / "frozen.nc"
    first_path = tmp_path / "first"
    resumed_path = tmp_path / "resumed"
    direct_path = tmp_path / "direct"
    _write_frozen_input(input_path)

    first = smoke_module.run(
        smoke_module.build_parser().parse_args(
            _fixed_basis_arguments(input_path, first_path, iterations=5),
        )
    )
    resumed = smoke_module.run(
        smoke_module.build_parser().parse_args(
            [
                *_fixed_basis_arguments(input_path, resumed_path, iterations=7),
                "--resume-checkpoint",
                str(first_path / "checkpoint.npz"),
            ]
        )
    )
    direct = smoke_module.run(
        smoke_module.build_parser().parse_args(
            _fixed_basis_arguments(input_path, direct_path, iterations=12),
        )
    )

    assert first["run"]["schedule_phase_end"] == 5
    assert resumed["run"]["segment_start_transition"] == 5
    assert resumed["run"]["segment_end_transition"] == 12
    assert resumed["run"]["schedule_phase_start"] == 5
    assert resumed["run"]["schedule_phase_end"] == 0
    assert direct["run"]["segment_start_transition"] == 0

    with (
        xr.open_dataset(first_path / "trace.nc", engine="h5netcdf") as first_trace,
        xr.open_dataset(resumed_path / "trace.nc", engine="h5netcdf") as resumed_trace,
        xr.open_dataset(direct_path / "trace.nc", engine="h5netcdf") as direct_trace,
    ):
        for name in direct_trace.data_vars:
            combined = np.concatenate(
                (first_trace[name].values, resumed_trace[name].values),
            )
            np.testing.assert_array_equal(combined, direct_trace[name].values)

    resumed_checkpoint = _checkpoint_arrays(resumed_path / "checkpoint.npz")
    direct_checkpoint = _checkpoint_arrays(direct_path / "checkpoint.npz")
    assert resumed_checkpoint.keys() == direct_checkpoint.keys()
    for name in resumed_checkpoint:
        np.testing.assert_array_equal(resumed_checkpoint[name], direct_checkpoint[name])


def test_fixed_basis_resume_without_retained_boundary_succeeds(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """A valid fixed segment may finish before the next cycle-boundary draw."""
    input_path = tmp_path / "frozen.nc"
    first_path = tmp_path / "first"
    resumed_path = tmp_path / "resumed"
    _write_frozen_input(input_path)
    smoke_module.run(
        smoke_module.build_parser().parse_args(
            _fixed_basis_arguments(input_path, first_path, iterations=5),
        )
    )
    summary = smoke_module.run(
        smoke_module.build_parser().parse_args(
            [
                *_fixed_basis_arguments(input_path, resumed_path, iterations=1),
                "--resume-checkpoint",
                str(first_path / "checkpoint.npz"),
            ]
        )
    )

    assert summary["run"]["segment_start_transition"] == 5
    assert summary["run"]["segment_end_transition"] == 6
    assert summary["run"]["retained_draws"] == 0
    assert summary["topology"]["unique_retained_topologies"] == 0
    assert (resumed_path / "complete.json").is_file()


def test_random_recursive_initialization_is_replayable_and_manifest_pinned(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Random starts use a dedicated seed recorded in checkpoint identity."""
    input_path = tmp_path / "frozen.nc"
    first_path = tmp_path / "random-first"
    replay_path = tmp_path / "random-replay"
    different_path = tmp_path / "random-different"
    _write_frozen_input(input_path)

    common = [
        "--initialization",
        "random-recursive",
        "--initialization-seed",
        "41",
    ]
    smoke_module.run(
        smoke_module.build_parser().parse_args([*_arguments(input_path, first_path, iterations=14), *common])
    )
    smoke_module.run(
        smoke_module.build_parser().parse_args([*_arguments(input_path, replay_path, iterations=14), *common])
    )
    smoke_module.run(
        smoke_module.build_parser().parse_args(
            [
                *_arguments(input_path, different_path, iterations=14),
                "--initialization",
                "random-recursive",
                "--initialization-seed",
                "1",
            ]
        )
    )

    first = json.loads((first_path / "manifest.json").read_text(encoding="utf-8"))
    replay = json.loads((replay_path / "manifest.json").read_text(encoding="utf-8"))
    different = json.loads((different_path / "manifest.json").read_text(encoding="utf-8"))
    assert first["initialization"] == {
        "strategy": "random-recursive",
        "seed": 41,
        "rng_stream": "dedicated_pcg64",
    }
    assert (
        first["state_space_scope"]["initial_topology_sha256"]
        == (replay["state_space_scope"]["initial_topology_sha256"])
    )
    assert (
        first["state_space_scope"]["initial_topology_sha256"]
        != (different["state_space_scope"]["initial_topology_sha256"])
    )
    first_checkpoint = _checkpoint_arrays(first_path / "checkpoint.npz")
    replay_checkpoint = _checkpoint_arrays(replay_path / "checkpoint.npz")
    assert first_checkpoint.keys() == replay_checkpoint.keys()
    for name in first_checkpoint:
        np.testing.assert_array_equal(first_checkpoint[name], replay_checkpoint[name])


def test_random_recursive_initialization_requires_an_exclusive_seed(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """The dedicated initialization stream cannot be missing or ignored."""
    input_path = tmp_path / "frozen.nc"
    _write_frozen_input(input_path)

    missing = smoke_module.build_parser().parse_args(
        [
            *_arguments(input_path, tmp_path / "missing", iterations=1),
            "--initialization",
            "random-recursive",
        ]
    )
    with pytest.raises(ValueError, match="requires --initialization-seed"):
        smoke_module.run(missing)

    ignored = smoke_module.build_parser().parse_args(
        [
            *_arguments(input_path, tmp_path / "ignored", iterations=1),
            "--initialization-seed",
            "41",
        ]
    )
    with pytest.raises(ValueError, match="only valid"):
        smoke_module.run(ignored)


@pytest.mark.parametrize("structure_mode", ["mobile", "fixed-basis"])
def test_output_only_diagnostics_preserve_trajectory_rng_and_persist_all_fields(
    tmp_path: Path,
    smoke_module: ModuleType,
    structure_mode: str,
) -> None:
    """Diagnostics preserve output and PCG64 under both structural schedules."""
    input_path = tmp_path / "frozen.nc"
    ordinary_path = tmp_path / "ordinary"
    diagnosed_path = tmp_path / "diagnosed"
    _write_frozen_input(input_path)
    arguments_builder = _fixed_basis_arguments if structure_mode == "fixed-basis" else _arguments
    iterations = 12 if structure_mode == "fixed-basis" else 14
    common = arguments_builder(input_path, ordinary_path, iterations=iterations)
    smoke_module.run(smoke_module.build_parser().parse_args(common))
    diagnosed_arguments = arguments_builder(
        input_path,
        diagnosed_path,
        iterations=iterations,
    )
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


def test_resume_rejects_random_initialization_seed_mismatch(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """A durable chain cannot resume under a different initial-tiling stream."""
    input_path = tmp_path / "frozen.nc"
    first_path = tmp_path / "first"
    output_path = tmp_path / "resume"
    _write_frozen_input(input_path)
    first_arguments = [
        *_arguments(input_path, first_path, iterations=5),
        "--initialization",
        "random-recursive",
        "--initialization-seed",
        "41",
    ]
    smoke_module.run(smoke_module.build_parser().parse_args(first_arguments))
    resumed_arguments = [
        *_arguments(input_path, output_path, iterations=9),
        "--initialization",
        "random-recursive",
        "--initialization-seed",
        "73",
        "--resume-checkpoint",
        str(first_path / "checkpoint.npz"),
    ]
    with pytest.raises(ValueError, match="manifest"):
        smoke_module.run(smoke_module.build_parser().parse_args(resumed_arguments))


@pytest.mark.parametrize(
    ("first_fixed", "resume_fixed"),
    [
        pytest.param(True, False, id="fixed-to-mobile"),
        pytest.param(False, True, id="mobile-to-fixed"),
    ],
)
def test_resume_rejects_structure_schedule_mismatch(
    tmp_path: Path,
    smoke_module: ModuleType,
    first_fixed: bool,
    resume_fixed: bool,
) -> None:
    """A checkpoint cannot cross fixed-basis and mobile schedule identity."""
    input_path = tmp_path / "frozen.nc"
    first_path = tmp_path / "first"
    output_path = tmp_path / "resume"
    _write_frozen_input(input_path)
    first_arguments = (
        _fixed_basis_arguments(input_path, first_path, iterations=5)
        if first_fixed
        else _arguments(input_path, first_path, iterations=5)
    )
    resume_arguments = (
        _fixed_basis_arguments(input_path, output_path, iterations=7)
        if resume_fixed
        else _arguments(input_path, output_path, iterations=7)
    )
    smoke_module.run(smoke_module.build_parser().parse_args(first_arguments))
    resume_arguments.extend(
        ["--resume-checkpoint", str(first_path / "checkpoint.npz")],
    )

    with pytest.raises(ValueError, match="manifest"):
        smoke_module.run(smoke_module.build_parser().parse_args(resume_arguments))


def test_fixed_basis_resume_rejects_changed_topology_support(
    tmp_path: Path,
    smoke_module: ModuleType,
) -> None:
    """Fixed-basis resume rejects a different K and deterministic topology."""
    input_path = tmp_path / "frozen.nc"
    first_path = tmp_path / "first"
    output_path = tmp_path / "resume"
    _write_frozen_input(input_path)
    smoke_module.run(
        smoke_module.build_parser().parse_args(
            _fixed_basis_arguments(input_path, first_path, iterations=5),
        )
    )
    arguments = _fixed_basis_arguments(input_path, output_path, iterations=7)
    arguments[arguments.index("--k") + 1] = "3"
    arguments.extend(["--resume-checkpoint", str(first_path / "checkpoint.npz")])

    with pytest.raises(ValueError, match="problem fingerprint|manifest"):
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
