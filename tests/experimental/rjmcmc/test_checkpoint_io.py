"""Regression tests for strict durable RJMCMC checkpoints."""

from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc import checkpoint_io
from openghg_inversions.experimental.rjmcmc.checkpoint_io import load_checkpoint, save_checkpoint
from openghg_inversions.experimental.rjmcmc.core import (
    Backend,
    FixedDesignBlock,
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
    uniform_log_k_prior,
)
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings
from openghg_inversions.experimental.rjmcmc.sampling import (
    SamplerConfig,
    SamplingResult,
    continue_sample,
    sample,
)


def _problem(*, fixed: bool = False, observation_shift: float = 0.0) -> TransDimensionalProblem:
    """Return a fresh equivalent problem, optionally with fixed columns."""
    fixed_block = None
    fixed_offset = None
    if fixed:
        fixed_block = FixedDesignBlock(
            design=np.array(
                [
                    [0.4, -0.2],
                    [0.1, 0.7],
                    [-0.3, 0.5],
                ]
            ),
            coefficient_prior_mean=np.array([1.0, 0.8]),
            coefficient_prior_sd=np.array([0.3, 0.25]),
        )
        fixed_offset = np.array([0.05, -0.1, 0.2])
    return TransDimensionalProblem(
        observations=np.array([5.0 + observation_shift, 1.0, -0.5]),
        observation_sd=np.array([0.8, 1.2, 0.5]),
        sensitivities=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.5, 0.0, 1.0, 0.0],
                [-1.0, 2.0, 0.0, 1.0],
            ]
        ),
        grid_coordinates=np.arange(4, dtype=float)[:, np.newaxis],
        k_min=1,
        k_max=3,
        log_k_prior=uniform_log_k_prior(1, 3),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.4,
        fixed_offset=fixed_offset,
        fixed_block=fixed_block,
    )


def _initial_state(
    problem: TransDimensionalProblem,
    *,
    backend: Backend = "numpy",
) -> TransDimensionalState:
    """Build the same valid starting point for either problem shape."""
    fixed_coefficients = np.array([0.9, 1.1]) if problem.n_fixed_coefficients else None
    return build_state(
        problem,
        [0, 3],
        [0.8, 1.2],
        fixed_coefficients=fixed_coefficients,
        backend=backend,
    )


def _config(
    *,
    iterations: int,
    fixed: bool,
    backend: Backend = "numpy",
) -> SamplerConfig:
    """Return a seeded kernel configuration for round-trip comparisons."""
    return SamplerConfig(
        iterations=iterations,
        coefficient_proposal_sd=0.15,
        birth_proposal_sd=0.25,
        fixed_coefficient_proposal_sd=0.1 if fixed else None,
        seed=481,
        backend=backend,
        nucleus_move="local",
        local_move_scale=0.8,
    )


def _assert_state_equal(actual: TransDimensionalState, expected: TransDimensionalState) -> None:
    """Assert exact equality for every cached continuation field."""
    for state_field in fields(TransDimensionalState):
        actual_value = getattr(actual, state_field.name)
        expected_value = getattr(expected, state_field.name)
        if isinstance(actual_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def _assert_split_matches_full(
    full: SamplingResult,
    first: SamplingResult,
    continued: SamplingResult,
) -> None:
    """Require a persisted split chain to reproduce one uninterrupted chain."""
    for name in (
        "state_transition",
        "k",
        "nuclei",
        "coefficients",
        "fixed_coefficients",
        "log_target",
        "moves",
        "accepted",
        "log_acceptance_ratio",
    ):
        np.testing.assert_array_equal(
            np.concatenate((getattr(first.trace, name), getattr(continued.trace, name))),
            getattr(full.trace, name),
        )
    _assert_state_equal(continued.final_state, full.final_state)
    assert continued.checkpoint.rng_state == full.checkpoint.rng_state
    assert continued.checkpoint.transitions_completed == full.checkpoint.transitions_completed


def _archive_arrays(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    """Copy every array from a test archive without enabling pickle support."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _rewrite_archive(path: Path, arrays: dict[str, np.ndarray[Any, Any]]) -> None:
    """Replace a test archive with supplied arrays."""
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _rewrite_metadata(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Rewrite authenticated test metadata after applying ``mutate``."""
    arrays = _archive_arrays(path)
    metadata = json.loads(arrays["metadata"].tobytes().decode("utf-8"))
    assert isinstance(metadata, dict)
    mutate(metadata)
    payload = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    arrays["metadata"] = np.frombuffer(payload, dtype=np.uint8)
    arrays["metadata_sha256"] = np.frombuffer(sha256(payload).hexdigest().encode("ascii"), dtype=np.uint8)
    _rewrite_archive(path, arrays)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("fixed", [False, True], ids=["dynamic-only", "fixed-block"])
def test_durable_round_trip_continues_exact_chain(
    tmp_path: Path,
    fixed: bool,
    backend: Backend,
) -> None:
    """Loading against an equal fresh problem should exactly resume both schedules."""
    problem = _problem(fixed=fixed)
    initial = _initial_state(problem, backend=backend)
    retention = RetentionSettings(warmup_transitions=3, thin=2)
    manifest = {"schema_version": 1, "name": "checkpoint-test", "inputs": ["synthetic"]}

    full = sample(
        problem,
        initial,
        _config(iterations=19, fixed=fixed, backend=backend),
        retention,
    )
    first = sample(
        problem,
        initial,
        _config(iterations=7, fixed=fixed, backend=backend),
        retention,
    )
    path = tmp_path / "chain-checkpoint.npz"
    save_checkpoint(path, first.checkpoint, run_manifest=manifest)

    loaded_problem = _problem(fixed=fixed)
    loaded = load_checkpoint(path, loaded_problem, expected_run_manifest=manifest)
    continued = continue_sample(loaded_problem, loaded, iterations=12)

    assert loaded.problem is loaded_problem
    assert loaded.problem is not problem
    assert loaded.rng_state == first.checkpoint.rng_state
    assert loaded.kernel_settings == first.checkpoint.kernel_settings
    assert loaded.retention == first.checkpoint.retention
    assert loaded.schedule_id == first.checkpoint.schedule_id
    _assert_state_equal(loaded.state, first.checkpoint.state)
    _assert_split_matches_full(full, first, continued)
    assert not loaded.state.nuclei.flags.writeable
    assert not loaded.state.prediction.flags.writeable

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "nuclei",
            "coefficients",
            "labels",
            "design",
            "fixed_coefficients",
            "dynamic_prediction",
            "fixed_prediction",
            "prediction",
            "residual",
            "metadata",
            "metadata_sha256",
        }
        assert all(archive[name].dtype != np.dtype(object) for name in archive.files)
        assert archive["metadata"].dtype == np.dtype(np.uint8)


def test_same_shape_changed_problem_content_is_rejected(tmp_path: Path) -> None:
    """Array shapes alone must not allow continuation against changed observations."""
    problem = _problem()
    result = sample(problem, _initial_state(problem), _config(iterations=7, fixed=False))
    path = tmp_path / "problem-bound.npz"
    save_checkpoint(path, result.checkpoint)

    with pytest.raises(ValueError, match="problem fingerprint"):
        load_checkpoint(path, _problem(observation_shift=0.01))


def test_expected_run_manifest_must_match_exact_canonical_content(tmp_path: Path) -> None:
    """A different run identity must not silently inherit a checkpoint."""
    problem = _problem()
    result = sample(problem, _initial_state(problem), _config(iterations=7, fixed=False))
    path = tmp_path / "manifest-bound.npz"
    save_checkpoint(path, result.checkpoint, run_manifest={"name": "original", "seed": 481})

    with pytest.raises(ValueError, match="does not match the expected run manifest"):
        load_checkpoint(path, _problem(), expected_run_manifest={"name": "other", "seed": 481})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_id", "future.checkpoint", "Unsupported checkpoint schema"),
        ("schema_version", 2, "Unsupported checkpoint schema version"),
    ],
)
def test_unknown_schema_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Authenticated metadata still must use the one supported schema."""
    problem = _problem()
    result = sample(problem, _initial_state(problem), _config(iterations=7, fixed=False))
    path = tmp_path / "future-schema.npz"
    save_checkpoint(path, result.checkpoint)
    _rewrite_metadata(path, lambda metadata: metadata.__setitem__(field, value))

    with pytest.raises(ValueError, match=message):
        load_checkpoint(path, _problem())


def test_unsupported_rng_and_tampered_manifest_hash_are_rejected(tmp_path: Path) -> None:
    """Execution state and embedded provenance should both fail closed when altered."""
    problem = _problem()
    result = sample(problem, _initial_state(problem), _config(iterations=7, fixed=False))
    path = tmp_path / "strict-metadata.npz"
    save_checkpoint(path, result.checkpoint, run_manifest={"name": "strict"})

    def replace_rng(metadata: dict[str, Any]) -> None:
        """Replace the declared bit generator in test metadata."""
        metadata["rng"]["algorithm"] = "Philox"

    _rewrite_metadata(path, replace_rng)
    with pytest.raises(ValueError, match="Unsupported checkpoint RNG"):
        load_checkpoint(path, _problem())

    save_checkpoint(path, result.checkpoint, run_manifest={"name": "strict"})
    _rewrite_metadata(path, lambda metadata: metadata.__setitem__("run_manifest_sha256", "0" * 64))
    with pytest.raises(ValueError, match="run manifest SHA-256"):
        load_checkpoint(path, _problem())


def test_exact_replay_rejects_a_different_numpy_version(tmp_path: Path) -> None:
    """A checkpoint should not claim bitwise replay across NumPy versions."""
    problem = _problem()
    result = sample(problem, _initial_state(problem), _config(iterations=7, fixed=False))
    path = tmp_path / "other-numpy.npz"
    save_checkpoint(path, result.checkpoint)
    _rewrite_metadata(path, lambda metadata: metadata.__setitem__("numpy_version", "0.0.test"))

    with pytest.raises(ValueError, match="NumPy version.*incompatible"):
        load_checkpoint(path, _problem())


def test_numba_replay_rejects_a_different_numba_version(tmp_path: Path) -> None:
    """A Numba-backed checkpoint should require the saved compiler version."""
    problem = _problem()
    result = sample(
        problem,
        _initial_state(problem, backend="numba"),
        _config(iterations=7, fixed=False, backend="numba"),
    )
    path = tmp_path / "other-numba.npz"
    save_checkpoint(path, result.checkpoint)
    _rewrite_metadata(path, lambda metadata: metadata.__setitem__("numba_version", "0.0.test"))

    with pytest.raises(ValueError, match="Numba version.*incompatible"):
        load_checkpoint(path, _problem())


def test_numba_kernel_can_checkpoint_a_rejected_numpy_initial_state(tmp_path: Path) -> None:
    """Persist the actual NumPy cache backend when a Numba proposal is rejected."""
    problem = _problem()
    initial = _initial_state(problem, backend="numpy")
    result = None
    for seed in range(100):
        config = replace(_config(iterations=1, fixed=False, backend="numba"), seed=seed)
        candidate = sample(problem, initial, config)
        if candidate.final_state is initial:
            result = candidate
            break
    assert result is not None, "expected at least one deterministic rejected first proposal"

    path = tmp_path / "numpy-cache-numba-kernel.npz"
    save_checkpoint(path, result.checkpoint)
    loaded = load_checkpoint(path, _problem())

    _assert_state_equal(loaded.state, initial)
    assert loaded.kernel_settings.backend == "numba"


def test_corrupt_archive_and_state_array_fail_validation(tmp_path: Path) -> None:
    """Unreadable ZIP data and altered cached arrays must be rejected."""
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not a NumPy archive")
    with pytest.raises(ValueError, match="corrupt or unreadable"):
        load_checkpoint(corrupt, _problem())

    problem = _problem()
    result = sample(problem, _initial_state(problem), _config(iterations=7, fixed=False))
    altered = tmp_path / "altered.npz"
    save_checkpoint(altered, result.checkpoint)
    truncated = tmp_path / "truncated.npz"
    archive_bytes = altered.read_bytes()
    truncated.write_bytes(archive_bytes[: len(archive_bytes) // 2])
    with pytest.raises(ValueError, match="corrupt or unreadable"):
        load_checkpoint(truncated, _problem())

    arrays = _archive_arrays(altered)
    arrays["prediction"][0] += 0.01
    _rewrite_archive(altered, arrays)
    with pytest.raises(ValueError, match="prediction.*SHA-256"):
        load_checkpoint(altered, _problem())


def test_object_array_metadata_is_rejected_without_pickle(tmp_path: Path) -> None:
    """An object-array payload should never be unpickled while loading."""
    problem = _problem()
    result = sample(problem, _initial_state(problem), _config(iterations=7, fixed=False))
    path = tmp_path / "object-payload.npz"
    save_checkpoint(path, result.checkpoint)
    arrays = _archive_arrays(path)
    arrays["metadata"] = np.array([{"unexpected": "object"}], dtype=object)
    _rewrite_archive(path, arrays)

    with pytest.raises(ValueError, match="object arrays and pickle payloads are not permitted"):
        load_checkpoint(path, _problem())


def test_failed_atomic_replace_preserves_existing_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement failure should leave the previous archive byte-for-byte intact."""
    problem = _problem()
    result = sample(problem, _initial_state(problem), _config(iterations=7, fixed=False))
    path = tmp_path / "existing.npz"
    save_checkpoint(path, result.checkpoint, run_manifest={"generation": 1})
    previous = path.read_bytes()

    def fail_replace(source: Path, destination: Path) -> None:
        """Simulate an operating-system failure at the atomic boundary."""
        raise OSError("simulated replace failure")

    monkeypatch.setattr(checkpoint_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        save_checkpoint(path, result.checkpoint, run_manifest={"generation": 2})

    assert path.read_bytes() == previous
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
