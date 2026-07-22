"""Checkpoint-v3 tests for inferred OU and shared-hierarchy sampler state."""

from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc import checkpoint_io
from openghg_inversions.experimental.rjmcmc.checkpoint_io import load_checkpoint, save_checkpoint
from openghg_inversions.experimental.rjmcmc.core import (
    FixedDesignBlock,
    InferredOUErrorModel,
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
    uniform_log_k_prior,
)
from openghg_inversions.experimental.rjmcmc.hierarchy import SharedLognormalHierarchy
from openghg_inversions.experimental.rjmcmc.likelihood import IndependentSiteOUData
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings
from openghg_inversions.experimental.rjmcmc.sampling import (
    FIXED_BLOCK_SCHEDULE_ID,
    LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_ID,
    LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_PROFILE,
    KernelSettings,
    PCG64State,
    SamplerCheckpoint,
    continue_sample,
)


def _problem(
    *,
    optional_layers: bool,
    time_shift: float = 0.0,
    mismatch_upper_shift: float = 0.0,
    hierarchy_median_shift: float = 0.0,
) -> TransDimensionalProblem:
    """Return a six-fixed-column problem with optional OU and hierarchy layers."""
    observation_sd = np.array([0.3, 0.4, 0.5, 0.6])
    error_model = None
    hierarchy = None
    if optional_layers:
        error_model = InferredOUErrorModel(
            data=IndependentSiteOUData(
                observation_sd=observation_sd,
                observation_time=np.array([0.0, 1.0 + time_shift, 0.5, 2.5]),
                site_index=np.array([0, 0, 1, 1]),
                mismatch_group_index=np.array([0, 1, 0, 1]),
                site_tau_index=np.array([0, 0]),
            ),
            mismatch_sd_prior_lower=np.array([0.1, 0.2]),
            mismatch_sd_prior_upper=np.array([1.5 + mismatch_upper_shift, 1.7]),
            correlation_timescale_prior_lower=np.array([0.25]),
            correlation_timescale_prior_upper=np.array([8.0]),
        )
        hierarchy = SharedLognormalHierarchy(
            mean_hyperprior_median=1.0 + hierarchy_median_shift,
            mean_hyperprior_log_sd=0.4,
            sd_hyperprior_median=0.8,
            sd_hyperprior_log_sd=0.5,
        )
    return TransDimensionalProblem(
        observations=np.array([1.0, 0.6, -0.2, 0.4]),
        observation_sd=observation_sd,
        sensitivities=np.array(
            [
                [1.0, 0.2, -0.1, 0.4],
                [0.1, 0.7, 0.3, -0.2],
                [-0.4, 0.2, 0.8, 0.1],
                [0.5, -0.3, 0.2, 0.9],
            ]
        ),
        grid_coordinates=np.arange(4, dtype=np.float64)[:, np.newaxis],
        k_min=1,
        k_max=3,
        log_k_prior=uniform_log_k_prior(1, 3),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=1.0,
        fixed_offset=np.array([0.1, -0.1, 0.05, 0.0]),
        fixed_block=FixedDesignBlock(
            design=np.arange(24, dtype=np.float64).reshape(4, 6) / 50.0,
            coefficient_prior_mean=np.ones(6),
            coefficient_prior_sd=np.full(6, 0.5),
        ),
        error_model=error_model,
        coefficient_hierarchy=hierarchy,
    )


def _checkpoint(problem: TransDimensionalProblem) -> SamplerCheckpoint:
    """Build a valid deterministic checkpoint for either target shape."""
    optional_layers = problem.error_model is not None
    state = build_state(
        problem,
        [0, 3],
        [0.8, 1.3],
        fixed_coefficients=np.linspace(0.8, 1.3, 6),
        mismatch_sd=np.array([0.6, 0.9]) if optional_layers else None,
        correlation_timescale=np.array([2.0]) if optional_layers else None,
        coefficient_prior_mean=1.2 if optional_layers else None,
        coefficient_prior_sd=0.7 if optional_layers else None,
        backend="numpy",
    )
    if optional_layers:
        kernel = KernelSettings(
            coefficient_proposal_sd=0.2,
            birth_proposal_sd=0.3,
            fixed_coefficient_proposal_sd=0.15,
            backend="numpy",
            nucleus_move="local",
            local_move_scale=1.0,
            schedule_profile=LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_PROFILE,
            mismatch_sd_proposal_sd=0.1,
            correlation_timescale_proposal_sd=0.25,
            eta_proposal_sd=0.08,
            zeta_proposal_sd=0.09,
        )
        schedule_id = LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_ID
    else:
        kernel = KernelSettings(
            coefficient_proposal_sd=0.2,
            birth_proposal_sd=0.3,
            fixed_coefficient_proposal_sd=0.15,
            backend="numpy",
            nucleus_move="local",
            local_move_scale=1.0,
        )
        schedule_id = FIXED_BLOCK_SCHEDULE_ID
    return SamplerCheckpoint(
        problem=problem,
        state=state,
        rng_state=PCG64State.from_generator(np.random.default_rng(824)),
        transitions_completed=31,
        kernel_settings=kernel,
        retention=RetentionSettings(warmup_transitions=7, thin=3),
        schedule_id=schedule_id,
    )


def _assert_state_equal(actual: TransDimensionalState, expected: TransDimensionalState) -> None:
    """Require exact equality for every durable state field."""
    for state_field in fields(TransDimensionalState):
        actual_value = getattr(actual, state_field.name)
        expected_value = getattr(expected, state_field.name)
        if isinstance(actual_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def _archive_arrays(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    """Copy an NPZ archive without enabling pickle payloads."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _write_archive(path: Path, arrays: dict[str, np.ndarray[Any, Any]]) -> None:
    """Write a test-only archive replacement."""
    with path.open("wb") as handle:
        cast(Any, np.savez_compressed)(handle, **arrays)


def _rewrite_metadata(
    arrays: dict[str, np.ndarray[Any, Any]],
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    """Mutate metadata and refresh its top-level authentication digest."""
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
    digest = sha256(payload).hexdigest().encode("ascii")
    arrays["metadata_sha256"] = np.frombuffer(digest, dtype=np.uint8)


def _downgrade_archive(
    path: Path,
    problem: TransDimensionalProblem,
    *,
    version: int,
) -> None:
    """Convert a v3 test archive to the exact numeric v1/v2 field contract."""
    arrays = _archive_arrays(path)
    arrays.pop("mismatch_sd")
    arrays.pop("correlation_timescale")

    def downgrade(metadata: dict[str, Any]) -> None:
        """Remove every field introduced after the requested legacy schema."""
        metadata["schema_version"] = version
        metadata["problem_sha256"] = checkpoint_io._legacy_problem_sha256(problem)
        for name in (
            "mismatch_sd_proposal_sd",
            "correlation_timescale_proposal_sd",
            "eta_proposal_sd",
            "zeta_proposal_sd",
        ):
            metadata["kernel"].pop(name)
        if version == 1:
            metadata["kernel"].pop("schedule_profile")
        for name in (
            "eta",
            "zeta",
            "log_error_model_prior",
            "log_coefficient_hyperprior",
        ):
            metadata["state"].pop(name)
        metadata["array_sha256"].pop("mismatch_sd")
        metadata["array_sha256"].pop("correlation_timescale")

    _rewrite_metadata(arrays, downgrade)
    _write_archive(path, arrays)


def test_v3_round_trip_preserves_optional_state_and_kernel(tmp_path: Path) -> None:
    """V3 should exactly restore OU, hierarchy, kernel, and cache provenance state."""
    problem = _problem(optional_layers=True)
    checkpoint = _checkpoint(problem)
    path = tmp_path / "optional-v3.npz"
    save_checkpoint(path, checkpoint, run_manifest={"case": "ou-hierarchy"})

    equivalent_problem = _problem(optional_layers=True)
    loaded = load_checkpoint(
        path,
        equivalent_problem,
        expected_run_manifest={"case": "ou-hierarchy"},
    )

    assert loaded.problem is equivalent_problem
    assert loaded.kernel_settings == checkpoint.kernel_settings
    assert loaded.rng_state == checkpoint.rng_state
    _assert_state_equal(loaded.state, checkpoint.state)
    assert not loaded.state.mismatch_sd.flags.writeable
    assert not loaded.state.correlation_timescale.flags.writeable
    with np.load(path, allow_pickle=False) as archive:
        assert "mismatch_sd" in archive.files
        assert "correlation_timescale" in archive.files
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
    assert metadata["schema_version"] == checkpoint_io.CHECKPOINT_SCHEMA_VERSION == 3
    assert metadata["state"]["eta"] == checkpoint.state.eta
    assert metadata["kernel"]["eta_proposal_sd"] == 0.08


def test_v3_loaded_optional_checkpoint_continues_exactly(tmp_path: Path) -> None:
    """An unaligned optional-state restart must preserve trace, phase, and RNG exactly."""
    problem = _problem(optional_layers=True)
    checkpoint = _checkpoint(problem)
    expected = continue_sample(problem, checkpoint, iterations=29)
    path = tmp_path / "optional-continuation-v3.npz"
    save_checkpoint(path, checkpoint)

    loaded_problem = _problem(optional_layers=True)
    loaded = load_checkpoint(path, loaded_problem)
    actual = continue_sample(loaded_problem, loaded, iterations=29)

    for trace_field in fields(type(expected.trace)):
        np.testing.assert_array_equal(
            getattr(actual.trace, trace_field.name),
            getattr(expected.trace, trace_field.name),
        )
    _assert_state_equal(actual.final_state, expected.final_state)
    assert actual.checkpoint.rng_state == expected.checkpoint.rng_state
    assert actual.checkpoint.transitions_completed == expected.checkpoint.transitions_completed


@pytest.mark.parametrize(
    "changed_problem",
    [
        lambda: _problem(optional_layers=True, time_shift=0.01),
        lambda: _problem(optional_layers=True, mismatch_upper_shift=0.01),
        lambda: _problem(optional_layers=True, hierarchy_median_shift=0.01),
    ],
    ids=["raw-observation-time", "error-prior-bound", "hierarchy-config"],
)
def test_v3_problem_fingerprint_covers_optional_target_inputs(
    tmp_path: Path,
    changed_problem: Callable[[], TransDimensionalProblem],
) -> None:
    """Optional target inputs must be part of the v3 scientific fingerprint."""
    problem = _problem(optional_layers=True)
    path = tmp_path / "fingerprint-v3.npz"
    save_checkpoint(path, _checkpoint(problem))

    with pytest.raises(ValueError, match="problem fingerprint"):
        load_checkpoint(path, changed_problem())


def test_v3_rejects_authenticated_optional_array_tampering(tmp_path: Path) -> None:
    """Shape, prior support, and scalar cache tampering must fail after authentication."""
    problem = _problem(optional_layers=True)
    path = tmp_path / "tampered-v3.npz"
    save_checkpoint(path, _checkpoint(problem))

    arrays = _archive_arrays(path)
    arrays["mismatch_sd"] = np.array([0.6], dtype=np.float64)

    def replace_shape_digest(metadata: dict[str, Any]) -> None:
        """Authenticate the wrong-shaped array so structural validation sees it."""
        metadata["array_sha256"]["mismatch_sd"] = checkpoint_io._array_sha256(
            "mismatch_sd", arrays["mismatch_sd"]
        )

    _rewrite_metadata(arrays, replace_shape_digest)
    _write_archive(path, arrays)
    with pytest.raises(ValueError, match="mismatch_sd.*shape"):
        load_checkpoint(path, problem)

    save_checkpoint(path, _checkpoint(problem))
    arrays = _archive_arrays(path)
    arrays["correlation_timescale"] = np.array([20.0], dtype=np.float64)

    def replace_support_digest(metadata: dict[str, Any]) -> None:
        """Authenticate an out-of-support timescale for support validation."""
        metadata["array_sha256"]["correlation_timescale"] = checkpoint_io._array_sha256(
            "correlation_timescale", arrays["correlation_timescale"]
        )

    _rewrite_metadata(arrays, replace_support_digest)
    _write_archive(path, arrays)
    with pytest.raises(ValueError, match="correlation_timescale.*prior support"):
        load_checkpoint(path, problem)

    save_checkpoint(path, _checkpoint(problem))
    arrays = _archive_arrays(path)
    _rewrite_metadata(
        arrays,
        lambda metadata: metadata["state"].__setitem__("eta", metadata["state"]["eta"] + 0.1),
    )
    _write_archive(path, arrays)
    with pytest.raises(ValueError, match="cached state does not match"):
        load_checkpoint(path, problem)


@pytest.mark.parametrize("version", [1, 2])
def test_real_legacy_archive_contract_remains_loadable(tmp_path: Path, version: int) -> None:
    """Exact v1/v2 archives should use their legacy fields and problem hash."""
    problem = _problem(optional_layers=False)
    checkpoint = _checkpoint(problem)
    path = tmp_path / f"legacy-v{version}.npz"
    save_checkpoint(path, checkpoint)
    _downgrade_archive(path, problem, version=version)

    loaded = load_checkpoint(path, _problem(optional_layers=False))

    assert loaded.kernel_settings.schedule_profile == "default"
    assert loaded.kernel_settings.mismatch_sd_proposal_sd is None
    assert loaded.schedule_id == FIXED_BLOCK_SCHEDULE_ID
    _assert_state_equal(loaded.state, checkpoint.state)


def test_legacy_schema_cannot_load_optional_target(tmp_path: Path) -> None:
    """V1/v2 fingerprints must never bind to targets with unrepresented layers."""
    problem = _problem(optional_layers=True)
    checkpoint = _checkpoint(problem)
    path = tmp_path / "legacy-optional.npz"
    save_checkpoint(path, checkpoint)
    arrays = _archive_arrays(path)
    arrays.pop("mismatch_sd")
    arrays.pop("correlation_timescale")

    def forge_v2(metadata: dict[str, Any]) -> None:
        """Create an authenticated legacy-shaped file from an optional target."""
        metadata["schema_version"] = 2
        metadata["problem_sha256"] = checkpoint_io._legacy_problem_sha256(problem)
        metadata["schedule_id"] = FIXED_BLOCK_SCHEDULE_ID
        metadata["kernel"]["schedule_profile"] = "default"
        for name in (
            "mismatch_sd_proposal_sd",
            "correlation_timescale_proposal_sd",
            "eta_proposal_sd",
            "zeta_proposal_sd",
        ):
            metadata["kernel"].pop(name)
        for name in (
            "eta",
            "zeta",
            "log_error_model_prior",
            "log_coefficient_hyperprior",
        ):
            metadata["state"].pop(name)
        metadata["array_sha256"].pop("mismatch_sd")
        metadata["array_sha256"].pop("correlation_timescale")

    _rewrite_metadata(arrays, forge_v2)
    _write_archive(path, arrays)
    with pytest.raises(ValueError, match="cannot be loaded against inferred OU"):
        load_checkpoint(path, problem)


def test_v3_archive_field_set_is_version_specific(tmp_path: Path) -> None:
    """Metadata version and numeric array contract must agree exactly."""
    problem = _problem(optional_layers=False)
    path = tmp_path / "mismatched-version.npz"
    save_checkpoint(path, _checkpoint(problem))
    arrays = _archive_arrays(path)
    _rewrite_metadata(arrays, lambda metadata: metadata.__setitem__("schema_version", 2))
    _write_archive(path, arrays)

    with pytest.raises(ValueError, match="archive fields do not match its schema version"):
        load_checkpoint(path, problem)


def test_v3_kernel_scale_tampering_fails_closed(tmp_path: Path) -> None:
    """Missing, unused, or nonpositive optional kernel scales must be rejected."""
    problem = _problem(optional_layers=True)
    path = tmp_path / "kernel-scales-v3.npz"
    save_checkpoint(path, _checkpoint(problem))
    arrays = _archive_arrays(path)
    _rewrite_metadata(
        arrays,
        lambda metadata: metadata["kernel"].__setitem__("eta_proposal_sd", -0.1),
    )
    _write_archive(path, arrays)

    with pytest.raises(ValueError, match="eta_proposal_sd.*finite and positive"):
        load_checkpoint(path, problem)


def test_v3_save_rejects_inconsistent_optional_scalar_cache(tmp_path: Path) -> None:
    """Save-time rebuild validation must cover new optional scalar target terms."""
    problem = _problem(optional_layers=True)
    checkpoint = _checkpoint(problem)
    bad_state = replace(
        checkpoint.state,
        log_coefficient_hyperprior=checkpoint.state.log_coefficient_hyperprior + 1.0,
    )

    with pytest.raises(ValueError, match="cached state does not match"):
        save_checkpoint(tmp_path / "bad-cache.npz", replace(checkpoint, state=bad_state))
