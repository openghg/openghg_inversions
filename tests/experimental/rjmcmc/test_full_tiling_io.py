"""Tests for strict durable fixed-``K`` full-tiling checkpoints."""

from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, cast
from zipfile import ZipFile

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.core import FixedDesignBlock
from openghg_inversions.experimental.rjmcmc.dyadic_tree import CanonicalDyadicTree
from openghg_inversions.experimental.rjmcmc.full_tiling_compound_sampling import (
    FULL_TILING_COMPOUND_SCHEDULE_ID,
    FullTilingCompoundConfig,
    FullTilingCompoundKernelSettings,
    FullTilingCompoundSamplingResult,
    continue_full_tiling_compound,
    sample_full_tiling_compound,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_io import (
    FULL_TILING_CHECKPOINT_SCHEMA_VERSION,
    canonical_full_tiling_run_manifest,
    full_tiling_problem_fingerprint,
    full_tiling_state_fingerprint,
    load_full_tiling_checkpoint,
    save_full_tiling_checkpoint,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    initialize_full_tiling_posterior_state,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_tree import (
    GammaBetaTreePrior,
    GammaBetaTreeProblem,
    TreePartitionPrior,
)
from openghg_inversions.experimental.rjmcmc.sampling import PCG64State


def _problem(
    *,
    observation_shift: float = 0.0,
    concentration: float = 3.0,
) -> FullTilingProblem:
    """Build an independently reconstructible full-tiling problem."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        np.array([1.0, 2.0, 3.0, 4.0]),
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
    base = GammaBetaTreeProblem(
        observations=np.full(4, observation_shift),
        observation_sd=np.array([0.5, 0.7, 0.9, 1.1]),
        sensitivity=np.eye(4),
        prior=prior,
        partition_prior=TreePartitionPrior.uniform_k(tree),
        likelihood_power=0.25,
        fixed_offset=np.array([0.1, 0.2, 0.3, 0.4]),
        fixed_block=fixed_block,
    )
    return FullTilingProblem(base=base, concentration=concentration)


def _manifest(*, revision: str = "0123456789abcdef") -> dict[str, object]:
    """Return a small caller-owned strict finite run manifest."""
    return {
        "chain": {"id": "chain-0", "seed": 481},
        "code_revision": revision,
        "inputs": {
            "frozen-input": {
                "identifier": "fixture:full-tiling",
                "sha256": "a" * 64,
            }
        },
    }


def _sample_boundary(problem: FullTilingProblem) -> FullTilingCompoundSamplingResult:
    """Return one awkward-phase checkpoint with nontrivial topology."""
    initial = initialize_full_tiling_posterior_state(problem, k=3)
    return sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(
            iterations=5,
            seed=481,
            pair_allocation_refresh_slots=2,
            fixed_coefficient_proposal_sd=(0.3, 0.6),
            root_slice_width=0.7,
            root_slice_max_steps=17,
            root_slice_max_shrink_steps=29,
        ),
    )


def _assert_states_equal(
    actual: FullTilingPosteriorState,
    expected: FullTilingPosteriorState,
) -> None:
    """Assert exact equality of coordinates and every posterior cache."""
    assert actual.allocation.tiling == expected.allocation.tiling
    for name in (
        "leaf_masses",
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
        "log_allocation_prior",
        "log_fixed_coefficient_prior",
        "log_target",
    ):
        assert getattr(actual, name) == getattr(expected, name)


def _assert_results_equal(
    actual: FullTilingCompoundSamplingResult,
    expected: FullTilingCompoundSamplingResult,
) -> None:
    """Assert exact equality of traces, states, and continuation metadata."""
    for field in fields(actual.trace):
        np.testing.assert_array_equal(
            getattr(actual.trace, field.name),
            getattr(expected.trace, field.name),
        )
    _assert_states_equal(actual.final_state, expected.final_state)
    assert actual.checkpoint.rng_state == expected.checkpoint.rng_state
    assert actual.checkpoint.transitions_completed == expected.checkpoint.transitions_completed
    assert actual.checkpoint.schedule_phase == expected.checkpoint.schedule_phase
    assert actual.checkpoint.kernel_settings == expected.checkpoint.kernel_settings
    assert actual.checkpoint.schedule_id == expected.checkpoint.schedule_id


def _rewrite_archive(
    path: Path,
    transform: Callable[[dict[str, np.ndarray[Any, Any]]], None],
) -> None:
    """Apply one test-only mutation and rewrite an NPZ archive."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    transform(arrays)
    with path.open("wb") as handle:
        save = cast(Callable[..., None], np.savez_compressed)
        save(handle, **arrays)


def _rewrite_metadata(
    path: Path,
    transform: Callable[[dict[str, Any]], None],
) -> None:
    """Rewrite checksum-protected metadata for an internal mismatch test."""

    def apply(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
        """Apply a metadata mutation and update its outer checksum."""
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


def test_checkpoint_round_trip_preserves_exact_continuation(
    tmp_path: Path,
) -> None:
    """A fresh equal problem resumes the exact cached PCG64 trajectory."""
    problem = _problem()
    first = _sample_boundary(problem)
    manifest = _manifest()
    path = tmp_path / "checkpoint.npz"
    save_full_tiling_checkpoint(path, first.checkpoint, run_manifest=manifest)

    reconstructed_problem = _problem()
    loaded = load_full_tiling_checkpoint(
        path,
        reconstructed_problem,
        expected_run_manifest=manifest,
    )
    direct = continue_full_tiling_compound(
        problem,
        first.checkpoint,
        iterations=23,
    )
    restored = continue_full_tiling_compound(
        reconstructed_problem,
        loaded,
        iterations=23,
    )

    assert loaded.problem is reconstructed_problem
    assert loaded.state.problem is reconstructed_problem
    assert loaded.rng_state == first.checkpoint.rng_state
    assert loaded.kernel_settings == first.checkpoint.kernel_settings
    assert loaded.schedule_id == FULL_TILING_COMPOUND_SCHEDULE_ID
    _assert_states_equal(loaded.state, first.checkpoint.state)
    _assert_results_equal(restored, direct)
    for name in (
        "leaf_masses",
        "fixed_coefficients",
        "dynamic_prediction",
        "fixed_prediction",
        "prediction",
        "residual",
    ):
        assert not getattr(loaded.state, name).flags.writeable

    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype != np.dtype(object) for name in archive.files)
        assert archive["metadata"].dtype == np.dtype(np.uint8)
        assert archive["metadata_sha256"].dtype == np.dtype(np.uint8)
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
    assert metadata["schema_version"] == FULL_TILING_CHECKPOINT_SCHEMA_VERSION
    assert metadata["schedule_id"] == FULL_TILING_COMPOUND_SCHEDULE_ID
    assert metadata["schedule_id"].endswith("_v2")
    frozen_v1_kernel_fields = frozenset(
        (
            "fixed_k",
            "pair_allocation_refresh_slots",
            "fixed_coefficient_proposal_sd",
            "root_slice_width",
            "root_slice_max_steps",
            "root_slice_max_shrink_steps",
        )
    )
    assert frozenset(metadata["kernel"]) == frozen_v1_kernel_fields
    assert (
        frozenset(field.name for field in fields(FullTilingCompoundKernelSettings)) == frozen_v1_kernel_fields
    )
    for name in (
        "root_slice_width",
        "root_slice_max_steps",
        "root_slice_max_shrink_steps",
    ):
        assert metadata["kernel"][name] == getattr(
            first.checkpoint.kernel_settings,
            name,
        )


def test_fingerprints_and_load_reject_changed_problem_or_manifest(
    tmp_path: Path,
) -> None:
    """Fingerprints are reconstructible and reject changed scientific identity."""
    problem = _problem()
    result = _sample_boundary(problem)
    manifest = _manifest()
    path = tmp_path / "checkpoint.npz"
    save_full_tiling_checkpoint(path, result.checkpoint, run_manifest=manifest)
    equal_problem = _problem()
    loaded = load_full_tiling_checkpoint(
        path,
        equal_problem,
        expected_run_manifest=manifest,
    )

    assert full_tiling_problem_fingerprint(problem) == (full_tiling_problem_fingerprint(equal_problem))
    assert full_tiling_state_fingerprint(problem, result.checkpoint.state) == (
        full_tiling_state_fingerprint(equal_problem, loaded.state)
    )
    assert canonical_full_tiling_run_manifest(
        {"second": 2, "first": 1}
    ) == canonical_full_tiling_run_manifest({"first": 1, "second": 2})
    assert full_tiling_problem_fingerprint(problem) != (
        full_tiling_problem_fingerprint(_problem(observation_shift=0.01))
    )
    assert full_tiling_problem_fingerprint(problem) != (
        full_tiling_problem_fingerprint(_problem(concentration=4.0))
    )

    with pytest.raises(ValueError, match="problem fingerprint"):
        load_full_tiling_checkpoint(
            path,
            _problem(observation_shift=0.01),
            expected_run_manifest=manifest,
        )
    with pytest.raises(ValueError, match="manifest does not match"):
        load_full_tiling_checkpoint(
            path,
            _problem(),
            expected_run_manifest=_manifest(revision="different"),
        )


def test_checkpoint_rejects_tampered_arrays_metadata_and_caches(
    tmp_path: Path,
) -> None:
    """Checksums and independent rebuilds reject altered checkpoint content."""
    problem = _problem()
    result = _sample_boundary(problem)
    manifest = _manifest()

    array_path = tmp_path / "array.npz"
    save_full_tiling_checkpoint(
        array_path,
        result.checkpoint,
        run_manifest=manifest,
    )

    def alter_mass(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
        """Change one coordinate without updating its recorded digest."""
        arrays["leaf_masses"][0] *= 1.01

    _rewrite_archive(array_path, alter_mass)
    with pytest.raises(ValueError, match="leaf_masses SHA-256"):
        load_full_tiling_checkpoint(
            array_path,
            _problem(),
            expected_run_manifest=manifest,
        )

    target_path = tmp_path / "target.npz"
    save_full_tiling_checkpoint(
        target_path,
        result.checkpoint,
        run_manifest=manifest,
    )

    def alter_target(metadata: dict[str, Any]) -> None:
        """Change one target cache while preserving outer metadata integrity."""
        metadata["state"]["log_target"] += 1.0

    _rewrite_metadata(target_path, alter_target)
    with pytest.raises(ValueError, match="log_target does not match"):
        load_full_tiling_checkpoint(
            target_path,
            _problem(),
            expected_run_manifest=manifest,
        )

    checksum_path = tmp_path / "metadata.npz"
    save_full_tiling_checkpoint(
        checksum_path,
        result.checkpoint,
        run_manifest=manifest,
    )

    def alter_metadata_digest(
        arrays: dict[str, np.ndarray[Any, Any]],
    ) -> None:
        """Corrupt one byte of the metadata checksum."""
        arrays["metadata_sha256"][0] ^= 1

    _rewrite_archive(checksum_path, alter_metadata_digest)
    with pytest.raises(ValueError, match="metadata SHA-256"):
        load_full_tiling_checkpoint(
            checksum_path,
            _problem(),
            expected_run_manifest=manifest,
        )

    cache_path = tmp_path / "cache.npz"
    save_full_tiling_checkpoint(
        cache_path,
        result.checkpoint,
        run_manifest=manifest,
    )

    def alter_cache_and_hashes(
        arrays: dict[str, np.ndarray[Any, Any]],
    ) -> None:
        """Forge outer checksums so the independent rebuild catches the cache."""
        arrays["prediction"][0] += 1.0
        metadata = json.loads(arrays["metadata"].tobytes().decode("utf-8"))
        digest = sha256()
        array = np.ascontiguousarray(arrays["prediction"])
        label = "prediction"
        descriptor = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        for framed_label, payload in (
            (f"{label}.descriptor", descriptor),
            (f"{label}.data", array.tobytes(order="C")),
        ):
            label_bytes = framed_label.encode("utf-8")
            digest.update(len(label_bytes).to_bytes(8, "big"))
            digest.update(label_bytes)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        metadata["array_sha256"]["prediction"] = digest.hexdigest()
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

    _rewrite_archive(cache_path, alter_cache_and_hashes)
    with pytest.raises(ValueError, match="prediction does not match persisted cache"):
        load_full_tiling_checkpoint(
            cache_path,
            _problem(),
            expected_run_manifest=manifest,
        )

    kernel_path = tmp_path / "kernel.npz"
    save_full_tiling_checkpoint(
        kernel_path,
        result.checkpoint,
        run_manifest=manifest,
    )

    def remove_root_setting(metadata: dict[str, Any]) -> None:
        """Remove one required v2 root-slice setting."""
        del metadata["kernel"]["root_slice_width"]

    _rewrite_metadata(kernel_path, remove_root_setting)
    with pytest.raises(ValueError, match="kernel has an invalid field set"):
        load_full_tiling_checkpoint(
            kernel_path,
            _problem(),
            expected_run_manifest=manifest,
        )

    numpy_path = tmp_path / "numpy-version.npz"
    save_full_tiling_checkpoint(
        numpy_path,
        result.checkpoint,
        run_manifest=manifest,
    )

    def alter_numpy_version(metadata: dict[str, Any]) -> None:
        """Make exact continuation provenance incompatible."""
        metadata["numpy_version"] = "0.0.0"

    _rewrite_metadata(numpy_path, alter_numpy_version)
    with pytest.raises(ValueError, match="NumPy version"):
        load_full_tiling_checkpoint(
            numpy_path,
            _problem(),
            expected_run_manifest=manifest,
        )

    duplicate_path = tmp_path / "duplicate-member.npz"
    save_full_tiling_checkpoint(
        duplicate_path,
        result.checkpoint,
        run_manifest=manifest,
    )
    with ZipFile(duplicate_path, mode="a") as archive:
        metadata_member = archive.read("metadata.npy")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("metadata.npy", metadata_member)
    with pytest.raises(ValueError, match="duplicate array members"):
        load_full_tiling_checkpoint(
            duplicate_path,
            _problem(),
            expected_run_manifest=manifest,
        )

    object_path = tmp_path / "object.npz"
    save_full_tiling_checkpoint(
        object_path,
        result.checkpoint,
        run_manifest=manifest,
    )

    def inject_object_array(
        arrays: dict[str, np.ndarray[Any, Any]],
    ) -> None:
        """Replace one numeric cache with a pickle-requiring object array."""
        arrays["prediction"] = np.array([object()], dtype=object)

    _rewrite_archive(object_path, inject_object_array)
    with pytest.raises(ValueError, match="pickle or object dtype"):
        load_full_tiling_checkpoint(
            object_path,
            _problem(),
            expected_run_manifest=manifest,
        )


def test_checkpoint_publication_is_create_only(tmp_path: Path) -> None:
    """A second publication cannot replace an existing valid generation."""
    problem = _problem()
    result = _sample_boundary(problem)
    manifest = _manifest()
    path = tmp_path / "checkpoint.npz"
    save_full_tiling_checkpoint(path, result.checkpoint, run_manifest=manifest)
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        save_full_tiling_checkpoint(
            path,
            result.checkpoint,
            run_manifest=manifest,
        )

    assert path.read_bytes() == original
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
    loaded = load_full_tiling_checkpoint(
        path,
        _problem(),
        expected_run_manifest=manifest,
    )
    _assert_states_equal(loaded.state, result.checkpoint.state)


def test_stale_state_does_not_publish_checkpoint(tmp_path: Path) -> None:
    """Save preflight rejects stale caches without leaving any archive."""
    problem = _problem()
    result = _sample_boundary(problem)
    stale_state = replace(
        result.checkpoint.state,
        prediction=result.checkpoint.state.prediction + 1.0,
    )
    stale_checkpoint = replace(result.checkpoint, state=stale_state)
    path = tmp_path / "checkpoint.npz"

    with pytest.raises(ValueError, match="stale or inconsistent cached arrays"):
        save_full_tiling_checkpoint(
            path,
            stale_checkpoint,
            run_manifest=_manifest(),
        )

    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_invalid_rng_does_not_publish_checkpoint(tmp_path: Path) -> None:
    """Save preflight rejects an unusable PCG64 state before publication."""
    problem = _problem()
    result = _sample_boundary(problem)
    invalid_rng = replace(
        result.checkpoint.rng_state,
        state=-1,
    )
    assert isinstance(invalid_rng, PCG64State)
    invalid_checkpoint = replace(result.checkpoint, rng_state=invalid_rng)
    path = tmp_path / "checkpoint.npz"

    with pytest.raises(ValueError, match="valid exact PCG64"):
        save_full_tiling_checkpoint(
            path,
            invalid_checkpoint,
            run_manifest=_manifest(),
        )

    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
