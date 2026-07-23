"""Tests for durable Gamma--Beta manifests, checkpoints, and xarray output."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.experimental.rjmcmc.core import FixedDesignBlock
from openghg_inversions.experimental.rjmcmc.dyadic_tree import (
    CanonicalDyadicTree,
    DyadicFrontier,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_compound_sampling import (
    GammaBetaCompoundCheckpoint,
    GammaBetaCompoundConfig,
    GammaBetaCompoundTrace,
    continue_gamma_beta_compound,
    sample_gamma_beta_compound,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_io import (
    GAMMA_BETA_CHECKPOINT_SCHEMA_VERSION,
    GAMMA_BETA_MANIFEST_SCHEMA_VERSION,
    GAMMA_BETA_TRACE_SCHEMA_ID,
    GAMMA_BETA_TRACE_SCHEMA_VERSION,
    build_gamma_beta_run_manifest,
    canonical_gamma_beta_run_manifest,
    gamma_beta_compound_trace_to_dataset,
    gamma_beta_problem_fingerprint,
    gamma_beta_state_fingerprint,
    load_gamma_beta_checkpoint,
    save_gamma_beta_checkpoint,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_tree import (
    GammaBetaTreePrior,
    GammaBetaTreeProblem,
    GammaBetaTreeState,
    TreePartitionPrior,
    build_gamma_beta_tree_state,
)
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings


def _problem(*, observation_shift: float = 0.0) -> GammaBetaTreeProblem:
    """Build an equivalent reconstructible problem with two fixed columns."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    nominal_mass = np.array([1.0, 2.0, 3.0, 4.0])
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        nominal_mass,
        concentration=3.0,
        root_mean=10.0,
        root_variance=8.0,
    )
    fixed_block = FixedDesignBlock(
        design=np.array(
            [
                [0.2, 0.0],
                [0.0, 0.1],
                [0.3, 0.2],
                [0.1, 0.4],
            ]
        ),
        coefficient_prior_mean=np.array([1.0, 1.5]),
        coefficient_prior_sd=np.array([0.5, 0.8]),
    )
    return GammaBetaTreeProblem(
        observations=np.full(4, observation_shift),
        observation_sd=np.array([0.5, 0.7, 0.9, 1.1]),
        sensitivity=np.eye(4),
        prior=prior,
        partition_prior=TreePartitionPrior.uniform_k(tree),
        likelihood_power=0.25,
        fixed_offset=np.array([0.1, 0.2, 0.3, 0.4]),
        fixed_block=fixed_block,
    )


def _initial_state(problem: GammaBetaTreeProblem) -> GammaBetaTreeState:
    """Return the coarsest state at prior means."""
    return build_gamma_beta_tree_state(
        problem,
        frontier=DyadicFrontier.root(problem.tree),
        root_total=10.0,
        active_fractions=[],
        fixed_coefficients=np.array([1.0, 1.5]),
    )


def _manifest(
    problem: GammaBetaTreeProblem,
    checkpoint: GammaBetaCompoundCheckpoint,
    *,
    reverse_inputs: bool = False,
    chain_id: str = "chain-0",
    initial_state: GammaBetaTreeState | None = None,
) -> dict[str, object]:
    """Build a manifest, optionally reversing input insertion order."""
    names = ("observations", "footprints")
    if reverse_inputs:
        names = tuple(reversed(names))
    identifiers = {name: f"frozen:{name}" for name in names}
    hashes = {
        "observations": "a" * 64,
        "footprints": "b" * 64,
    }
    return build_gamma_beta_run_manifest(
        problem,
        checkpoint.kernel_settings,
        checkpoint.retention,
        chain_id=chain_id,
        initial_state=_initial_state(problem) if initial_state is None else initial_state,
        code_revision="0123456789abcdef",
        input_identifiers=identifiers,
        input_sha256={name: hashes[name] for name in names},
        nominal_weight_policy="positive-emissions-mass-v1",
        nominal_weight_normalization_factor=10.0,
        seed=481,
    )


def _assert_states_equal(actual: GammaBetaTreeState, expected: GammaBetaTreeState) -> None:
    """Assert exact equality of state coordinates and rebuilt caches."""
    assert actual.frontier == expected.frontier
    assert actual.root_total == expected.root_total
    for name in (
        "active_fractions",
        "active_node_masses",
        "fixed_coefficients",
        "dynamic_prediction",
        "fixed_prediction",
        "prediction",
        "residual",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in (
        "log_gaussian_likelihood",
        "log_likelihood",
        "log_root_prior",
        "log_fraction_prior",
        "log_partition_prior",
        "log_fixed_coefficient_prior",
        "log_target",
    ):
        assert getattr(actual, name) == getattr(expected, name)


def _assert_traces_equal(
    actual: GammaBetaCompoundTrace,
    expected: GammaBetaCompoundTrace,
) -> None:
    """Assert exact equality of retained and every-attempt trace values."""
    assert actual.frontiers == expected.frontiers
    for left, right in zip(actual.split_fractions, expected.split_fractions, strict=True):
        np.testing.assert_array_equal(left, right)
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
        "state_transition",
        "global_transition",
        "slot",
        "move",
        "valid",
        "accepted",
        "node_id",
        "secondary_node_id",
        "block_leaf_count",
        "coefficient_id",
        "k_before",
        "k_after",
        "log_acceptance_ratio",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def _rewrite_archive(
    path: Path,
    transform: Callable[[dict[str, np.ndarray[Any, Any]]], None],
) -> None:
    """Apply a transform and overwrite ``path`` with a recompressed archive."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    transform(arrays)
    with path.open("wb") as handle:
        savez = cast(Callable[..., None], np.savez_compressed)
        savez(handle, **arrays)


def _rewrite_metadata(
    path: Path,
    transform: Callable[[dict[str, Any]], None],
) -> None:
    """Rewrite checksum-protected JSON metadata for a consistency test."""

    def apply(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
        metadata = json.loads(arrays["metadata"].tobytes().decode("utf-8"))
        transform(metadata)
        payload = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        arrays["metadata"] = np.frombuffer(payload, dtype=np.uint8)
        arrays["metadata_sha256"] = np.frombuffer(
            sha256(payload).hexdigest().encode("ascii"),
            dtype=np.uint8,
        )

    _rewrite_archive(path, apply)


def test_checkpoint_round_trip_preserves_exact_continuation(tmp_path: Path) -> None:
    """A durable boundary resumes the identical PCG64 trajectory."""
    problem = _problem()
    retention = RetentionSettings(warmup_transitions=3, thin=4)
    first = sample_gamma_beta_compound(
        problem,
        _initial_state(problem),
        GammaBetaCompoundConfig(
            iterations=19,
            seed=481,
            relocation_slots=2,
            subtree_retile_slots=1,
            max_subtree_leaves=3,
            fixed_coefficient_proposal_sd=(0.3, 0.6),
        ),
        retention=retention,
    )
    manifest = _manifest(problem, first.checkpoint)
    path = tmp_path / "checkpoint.npz"
    save_gamma_beta_checkpoint(path, first.checkpoint, run_manifest=manifest)

    reconstructed_problem = _problem()
    loaded = load_gamma_beta_checkpoint(
        path,
        reconstructed_problem,
        expected_run_manifest=manifest,
    )
    direct = continue_gamma_beta_compound(problem, first.checkpoint, iterations=31)
    restored = continue_gamma_beta_compound(
        reconstructed_problem,
        loaded,
        iterations=31,
    )

    assert loaded.problem is reconstructed_problem
    assert loaded.rng_state == first.checkpoint.rng_state
    assert loaded.kernel_settings == first.checkpoint.kernel_settings
    assert loaded.kernel_settings.relocation_slots == 2
    assert loaded.kernel_settings.subtree_retile_slots == 1
    assert loaded.kernel_settings.max_subtree_leaves == 3
    assert loaded.retention == first.checkpoint.retention
    _assert_states_equal(loaded.state, first.checkpoint.state)
    _assert_states_equal(restored.final_state, direct.final_state)
    _assert_traces_equal(restored.trace, direct.trace)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype != np.dtype(object) for name in archive.files)
        checkpoint_metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
    assert checkpoint_metadata["schema_version"] == GAMMA_BETA_CHECKPOINT_SCHEMA_VERSION
    assert checkpoint_metadata["kernel"]["relocation_slots"] == 2
    assert checkpoint_metadata["kernel"]["subtree_retile_slots"] == 1
    assert checkpoint_metadata["kernel"]["max_subtree_leaves"] == 3
    assert manifest["schema_version"] == GAMMA_BETA_MANIFEST_SCHEMA_VERSION
    schedule = cast(dict[str, Any], manifest["schedule"])
    assert schedule["relocation_slots"] == 2
    assert schedule["subtree_retile_slots"] == 1
    assert schedule["max_subtree_leaves"] == 3


def test_manifest_binds_chain_and_initial_coordinates(tmp_path: Path) -> None:
    """Chain IDs and immutable initial coordinates remain distinct on resume."""
    problem = _problem()
    initial = _initial_state(problem)
    alternate_initial = build_gamma_beta_tree_state(
        problem,
        frontier=initial.frontier.split(problem.tree, 0),
        root_total=10.0,
        active_fractions=[0.4],
        fixed_coefficients=np.array([1.0, 1.5]),
    )
    result = sample_gamma_beta_compound(
        problem,
        initial,
        GammaBetaCompoundConfig(iterations=9, seed=481, fixed_coefficient_proposal_sd=0.4),
    )
    first = _manifest(problem, result.checkpoint, chain_id="chain-0", initial_state=initial)
    other_chain = _manifest(
        problem,
        result.checkpoint,
        chain_id="chain-1",
        initial_state=initial,
    )
    other_initial = _manifest(
        problem,
        result.checkpoint,
        chain_id="chain-0",
        initial_state=alternate_initial,
    )

    assert first != other_chain
    assert first != other_initial
    assert other_initial["chain"]["initial_k"] == 2  # type: ignore[index]
    assert first["chain"] == {
        "id": "chain-0",
        "initial_k": 1,
        "initial_state_sha256": gamma_beta_state_fingerprint(problem, initial),
    }
    assert (
        first["chain"]["initial_state_sha256"]  # type: ignore[index]
        != other_initial["chain"]["initial_state_sha256"]  # type: ignore[index]
    )

    path = tmp_path / "checkpoint.npz"
    save_gamma_beta_checkpoint(path, result.checkpoint, run_manifest=first)
    with pytest.raises(ValueError, match="manifest does not match"):
        load_gamma_beta_checkpoint(
            path,
            _problem(),
            expected_run_manifest=other_chain,
        )
    with pytest.raises(ValueError, match="manifest does not match"):
        load_gamma_beta_checkpoint(
            path,
            _problem(),
            expected_run_manifest=other_initial,
        )


def test_stale_cache_cannot_replace_existing_checkpoint(tmp_path: Path) -> None:
    """Save preflight rejects stale caches before replacing a good archive."""
    problem = _problem()
    result = sample_gamma_beta_compound(
        problem,
        _initial_state(problem),
        GammaBetaCompoundConfig(iterations=8, seed=481, fixed_coefficient_proposal_sd=0.4),
    )
    manifest = _manifest(problem, result.checkpoint)
    path = tmp_path / "checkpoint.npz"
    save_gamma_beta_checkpoint(path, result.checkpoint, run_manifest=manifest)
    original = path.read_bytes()
    stale_state = replace(
        result.checkpoint.state,
        prediction=result.checkpoint.state.prediction + 1.0,
    )
    stale_checkpoint = replace(result.checkpoint, state=stale_state)

    with pytest.raises(ValueError, match="stale or inconsistent cached arrays"):
        save_gamma_beta_checkpoint(path, stale_checkpoint, run_manifest=manifest)

    assert path.read_bytes() == original
    restored = load_gamma_beta_checkpoint(
        path,
        _problem(),
        expected_run_manifest=manifest,
    )
    _assert_states_equal(restored.state, result.checkpoint.state)


def test_checkpoint_rejects_tampered_arrays_problem_and_manifest(tmp_path: Path) -> None:
    """Integrity checks reject modified arrays and mismatched identities."""
    problem = _problem()
    result = sample_gamma_beta_compound(
        problem,
        _initial_state(problem),
        GammaBetaCompoundConfig(iterations=13, seed=481, fixed_coefficient_proposal_sd=0.4),
    )
    manifest = _manifest(problem, result.checkpoint)
    path = tmp_path / "checkpoint.npz"
    save_gamma_beta_checkpoint(path, result.checkpoint, run_manifest=manifest)

    with pytest.raises(ValueError, match="problem fingerprint"):
        load_gamma_beta_checkpoint(
            path,
            _problem(observation_shift=0.01),
            expected_run_manifest=manifest,
        )
    altered_manifest = dict(manifest)
    altered_manifest["code_revision"] = "different"
    with pytest.raises(ValueError, match="manifest does not match"):
        load_gamma_beta_checkpoint(
            path,
            _problem(),
            expected_run_manifest=altered_manifest,
        )

    def alter_frontier(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
        """Change one persisted coordinate without updating its digest."""
        arrays["frontier_node_ids"][0] += 1

    _rewrite_archive(path, alter_frontier)
    with pytest.raises(ValueError, match="frontier_node_ids SHA-256"):
        load_gamma_beta_checkpoint(
            path,
            _problem(),
            expected_run_manifest=manifest,
        )

    settings_path = tmp_path / "settings.npz"
    save_gamma_beta_checkpoint(
        settings_path,
        result.checkpoint,
        run_manifest=manifest,
    )

    def alter_settings(metadata: dict[str, Any]) -> None:
        """Change a checksummed kernel setting but not the manifest."""
        metadata["kernel"]["max_subtree_leaves"] = 2

    _rewrite_metadata(settings_path, alter_settings)
    with pytest.raises(ValueError, match="manifest schedule"):
        load_gamma_beta_checkpoint(
            settings_path,
            _problem(),
            expected_run_manifest=manifest,
        )

    missing_path = tmp_path / "missing-setting.npz"
    save_gamma_beta_checkpoint(
        missing_path,
        result.checkpoint,
        run_manifest=manifest,
    )

    def remove_new_setting(metadata: dict[str, Any]) -> None:
        """Remove one v2 kernel field from otherwise checksummed metadata."""
        del metadata["kernel"]["relocation_slots"]

    _rewrite_metadata(missing_path, remove_new_setting)
    with pytest.raises(ValueError, match="kernel has an invalid field set"):
        load_gamma_beta_checkpoint(
            missing_path,
            _problem(),
            expected_run_manifest=manifest,
        )

    old_checkpoint_path = tmp_path / "old-checkpoint.npz"
    save_gamma_beta_checkpoint(
        old_checkpoint_path,
        result.checkpoint,
        run_manifest=manifest,
    )

    def downgrade_checkpoint(metadata: dict[str, Any]) -> None:
        """Present a checksum-valid but unsupported v1 checkpoint."""
        metadata["schema_version"] = 1

    _rewrite_metadata(old_checkpoint_path, downgrade_checkpoint)
    with pytest.raises(ValueError, match="checkpoint schema is incompatible"):
        load_gamma_beta_checkpoint(
            old_checkpoint_path,
            _problem(),
            expected_run_manifest=manifest,
        )

    old_manifest = dict(manifest)
    old_manifest["schema_version"] = 2
    with pytest.raises(ValueError, match="run_manifest schema is incompatible"):
        save_gamma_beta_checkpoint(
            tmp_path / "old-manifest.npz",
            result.checkpoint,
            run_manifest=old_manifest,
        )


def test_manifest_is_deterministic_and_requires_auditable_inputs() -> None:
    """Canonical manifests ignore order and reject malformed audit metadata."""
    problem = _problem()
    result = sample_gamma_beta_compound(
        problem,
        _initial_state(problem),
        GammaBetaCompoundConfig(iterations=1, seed=481, fixed_coefficient_proposal_sd=0.4),
    )
    forward = _manifest(problem, result.checkpoint)
    reverse = _manifest(problem, result.checkpoint, reverse_inputs=True)

    assert forward == reverse
    assert canonical_gamma_beta_run_manifest(forward) == canonical_gamma_beta_run_manifest(reverse)
    assert json.loads(canonical_gamma_beta_run_manifest(forward)) == forward
    assert forward["problem_sha256"] == gamma_beta_problem_fingerprint(_problem())

    common = {
        "problem": problem,
        "kernel_settings": result.checkpoint.kernel_settings,
        "retention": result.checkpoint.retention,
        "chain_id": "chain-0",
        "initial_state": _initial_state(problem),
        "code_revision": "revision",
        "input_identifiers": {"input": "frozen:input"},
        "input_sha256": {"input": "a" * 64},
        "nominal_weight_policy": "emissions",
        "nominal_weight_normalization_factor": 1.0,
        "seed": 481,
    }
    with pytest.raises(ValueError, match="code_revision"):
        build_gamma_beta_run_manifest(**{**common, "code_revision": ""})
    with pytest.raises(ValueError, match="same nonempty key set"):
        build_gamma_beta_run_manifest(**{**common, "input_sha256": {}})
    with pytest.raises(ValueError, match="64-character"):
        build_gamma_beta_run_manifest(**{**common, "input_sha256": {"input": "bad"}})
    with pytest.raises(ValueError, match="nominal_weight_policy"):
        build_gamma_beta_run_manifest(**{**common, "nominal_weight_policy": ""})
    with pytest.raises(ValueError, match="finite and positive"):
        build_gamma_beta_run_manifest(**{**common, "nominal_weight_normalization_factor": 0.0})
    with pytest.raises(ValueError, match="chain_id"):
        build_gamma_beta_run_manifest(**{**common, "chain_id": ""})


def test_trace_dataset_has_padded_states_and_attempt_diagnostics(tmp_path: Path) -> None:
    """xarray output keeps variable dimensions and every atomic attempt."""
    problem = _problem()
    result = sample_gamma_beta_compound(
        problem,
        _initial_state(problem),
        GammaBetaCompoundConfig(iterations=37, seed=481, fixed_coefficient_proposal_sd=0.4),
        retention=RetentionSettings(thin=2),
    )
    dataset = gamma_beta_compound_trace_to_dataset(
        result.trace,
        problem,
        metadata={"chain": 2},
        latitudes=np.array([50.0, 51.0]),
        longitudes=np.array([-2.0, -1.0]),
        fixed_parameter_labels=["outer-west", "outer-east"],
        measurement_labels=["obs-0", "obs-1", "obs-2", "obs-3"],
    )

    assert dataset.attrs["schema_id"] == GAMMA_BETA_TRACE_SCHEMA_ID
    assert dataset.attrs["schema_version"] == GAMMA_BETA_TRACE_SCHEMA_VERSION
    assert dataset.attrs["chain"] == 2
    assert dataset.sizes["draw"] == len(result.trace.frontiers)
    assert dataset.sizes["attempt"] == 37
    assert dataset.sizes["fixed_parameter"] == 2
    assert dataset.attrs["problem_sha256"] == gamma_beta_problem_fingerprint(problem)
    assert dataset.attrs["likelihood_family"] == "independent_gaussian"
    assert dataset.attrs["likelihood_power"] == 0.25
    np.testing.assert_array_equal(dataset["latitude"], [50.0, 51.0])
    np.testing.assert_array_equal(dataset["longitude"], [-2.0, -1.0])
    np.testing.assert_array_equal(
        dataset["fixed_parameter"],
        ["outer-west", "outer-east"],
    )
    np.testing.assert_array_equal(
        dataset["measurement"],
        ["obs-0", "obs-1", "obs-2", "obs-3"],
    )
    np.testing.assert_array_equal(dataset["node"], np.arange(len(problem.tree.nodes)))
    np.testing.assert_array_equal(
        dataset["node_parent_id"],
        [-1, 0, 1, 1, 0, 4, 4],
    )
    np.testing.assert_array_equal(
        dataset["node_first_child_id"],
        [1, 2, -1, -1, 5, -1, -1],
    )
    np.testing.assert_array_equal(
        dataset["node_second_child_id"],
        [4, 3, -1, -1, 6, -1, -1],
    )
    np.testing.assert_array_equal(
        dataset["frontier_active"].sum("region_slot"),
        result.trace.k,
    )
    np.testing.assert_array_equal(
        dataset["split_active"].sum("split_slot"),
        result.trace.k - 1,
    )
    for draw, frontier in enumerate(result.trace.frontiers):
        split_count = len(frontier) - 1
        np.testing.assert_array_equal(
            dataset["split_node_id"][draw, :split_count],
            frontier.active_split_nodes(problem.tree),
        )
        np.testing.assert_array_equal(
            dataset["split_node_id"][draw, split_count:],
            -1,
        )
    np.testing.assert_array_equal(dataset["state_transition"], result.trace.state_transition)
    np.testing.assert_array_equal(dataset["global_transition"], result.trace.global_transition)
    for name in (
        "slot",
        "move",
        "valid",
        "accepted",
        "node_id",
        "secondary_node_id",
        "block_leaf_count",
        "coefficient_id",
        "k_before",
        "k_after",
        "log_acceptance_ratio",
    ):
        np.testing.assert_array_equal(dataset[name], getattr(result.trace, name))
    assert dataset["frontier_node_id"].attrs["inactive_sentinel"] == -1
    assert dataset["split_fraction"].attrs["mask"] == "split_active"
    assert dataset["log_likelihood"].attrs["likelihood_power_applied"] == 0.25
    assert dataset["node_parent_id"].attrs["absent_sentinel"] == -1
    assert dataset["secondary_node_id"].attrs["not_applicable_sentinel"] == -1
    assert dataset["block_leaf_count"].attrs["not_applicable_sentinel"] == -1
    assert dataset["block_leaf_count"].attrs["units"] == "active_regions"

    output = tmp_path / "trace.nc"
    dataset.to_netcdf(output, engine="h5netcdf")
    with xr.open_dataset(output, engine="h5netcdf") as reopened:
        loaded = reopened.load()
    assert loaded.attrs["problem_sha256"] == gamma_beta_problem_fingerprint(problem)
    np.testing.assert_array_equal(loaded["latitude"], [50.0, 51.0])
    np.testing.assert_array_equal(
        loaded["fixed_parameter"],
        ["outer-west", "outer-east"],
    )
    np.testing.assert_array_equal(loaded["node_parent_id"], dataset["node_parent_id"])
    np.testing.assert_array_equal(
        loaded["secondary_node_id"],
        result.trace.secondary_node_id,
    )
    np.testing.assert_array_equal(
        loaded["block_leaf_count"],
        result.trace.block_leaf_count,
    )
    assert loaded["log_likelihood"].attrs["likelihood_power_applied"] == 0.25


def test_trace_dataset_rejects_inconsistent_target_terms_and_labels() -> None:
    """Converter rejects traces or optional labels inconsistent with a problem."""
    problem = _problem()
    result = sample_gamma_beta_compound(
        problem,
        _initial_state(problem),
        GammaBetaCompoundConfig(iterations=12, seed=481, fixed_coefficient_proposal_sd=0.4),
    )
    bad_power = replace(
        result.trace,
        log_likelihood=result.trace.log_likelihood + 0.1,
    )
    with pytest.raises(ValueError, match="likelihood power"):
        gamma_beta_compound_trace_to_dataset(bad_power, problem)

    bad_target = replace(
        result.trace,
        log_target=result.trace.log_target + 0.1,
    )
    with pytest.raises(ValueError, match="target components"):
        gamma_beta_compound_trace_to_dataset(bad_target, problem)
    with pytest.raises(ValueError, match="latitudes must have shape"):
        gamma_beta_compound_trace_to_dataset(
            result.trace,
            problem,
            latitudes=[50.0],
        )
    with pytest.raises(ValueError, match="fixed_parameter_labels must have shape"):
        gamma_beta_compound_trace_to_dataset(
            result.trace,
            problem,
            fixed_parameter_labels=["only-one"],
        )


def test_trace_dataset_supports_zero_retained_draws() -> None:
    """Warmup-only segments retain diagnostics with zero-width state axes."""
    problem = _problem()
    result = sample_gamma_beta_compound(
        problem,
        _initial_state(problem),
        GammaBetaCompoundConfig(iterations=7, seed=481, fixed_coefficient_proposal_sd=0.4),
        retention=RetentionSettings(warmup_transitions=100),
    )
    dataset = gamma_beta_compound_trace_to_dataset(result.trace, problem)

    assert dataset.sizes["draw"] == 0
    assert dataset.sizes["region_slot"] == 0
    assert dataset.sizes["split_slot"] == 0
    assert dataset.sizes["fixed_parameter"] == 2
    assert dataset.sizes["attempt"] == 7
    assert dataset["fixed_coefficients"].shape == (0, 2)
    assert dataset["frontier_node_id"].shape == (0, 0)
    assert dataset["split_node_id"].shape == (0, 0)
    assert dataset["split_fraction"].shape == (0, 0)
    assert dataset["secondary_node_id"].shape == (7,)
    assert dataset["block_leaf_count"].shape == (7,)
    np.testing.assert_array_equal(
        dataset["secondary_node_id"],
        result.trace.secondary_node_id,
    )
    np.testing.assert_array_equal(
        dataset["block_leaf_count"],
        result.trace.block_leaf_count,
    )
