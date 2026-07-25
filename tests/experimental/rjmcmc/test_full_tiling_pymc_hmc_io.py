"""Tests for strict durable full-tiling PyMC HMC checkpoints."""

from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, cast

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc import full_tiling_io
from openghg_inversions.experimental.rjmcmc.core import FixedDesignBlock
from openghg_inversions.experimental.rjmcmc.dyadic_tree import CanonicalDyadicTree
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    initialize_full_tiling_posterior_state,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_pymc_hmc import (
    FullTilingPyMCHMCConfig,
    FullTilingPyMCHMCSamplingResult,
    continue_full_tiling_pymc_hmc,
    sample_full_tiling_pymc_hmc,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_pymc_hmc_io import (
    load_full_tiling_pymc_hmc_checkpoint,
    save_full_tiling_pymc_hmc_checkpoint,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_tree import (
    GammaBetaTreePrior,
    GammaBetaTreeProblem,
    TreePartitionPrior,
)

_THIS_FILE = Path(__file__).resolve()
_X64_CHILD_ENV = "OPENGHG_INVERSIONS_PYMC_HMC_IO_X64_CHILD"
_IS_X64_CHILD = os.environ.get(_X64_CHILD_ENV) == "1"
_requires_x64_child = pytest.mark.skipif(
    not _IS_X64_CHILD,
    reason="checkpoint assertion executes in the isolated float64 child",
)
_ARCHIVE_FIELDS = {
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


def _pytensor_flags_with_float64(flags: str) -> str:
    """Return PyTensor flags with an unambiguous float64 configuration."""
    retained = []
    for item in flags.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name not in {"floatX", "warn_float64"}:
            retained.append(stripped)
    return ",".join(("floatX=float64", "warn_float64=ignore", *retained))


def _run_x64_test_file() -> None:
    """Run all checkpoint assertions in one fresh float64 subprocess."""
    environment = os.environ.copy()
    environment[_X64_CHILD_ENV] = "1"
    environment["PYTENSOR_FLAGS"] = _pytensor_flags_with_float64(
        environment.get("PYTENSOR_FLAGS", ""),
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(_THIS_FILE)],
        cwd=_THIS_FILE.parents[3],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        "isolated full-tiling PyMC HMC checkpoint tests failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.skipif(_IS_X64_CHILD, reason="parent-only subprocess dispatch")
def test_checkpoint_cases_use_a_fresh_float64_subprocess() -> None:
    """Checkpoint cases cannot inherit process-global float32 from RHIME tests."""
    _run_x64_test_file()


def _problem(*, observation_shift: float = 0.0) -> FullTilingProblem:
    """Build an independently reconstructible tiny posterior problem."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        np.array([1.0, 2.0, 3.0, 4.0]),
        concentration=3.0,
        root_mean=3.0,
        root_variance=1.0,
    )
    fixed_block = FixedDesignBlock(
        design=np.array(
            [
                [0.2, 0.0],
                [0.0, 0.1],
                [0.3, 0.2],
                [0.1, 0.4],
            ],
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
    return FullTilingProblem(base=base, concentration=3.0)


def _manifest(*, revision: str = "0123456789abcdef") -> dict[str, object]:
    """Return a strict finite caller-owned run manifest."""
    return {
        "chain": {"id": "chain-0", "seed": 481},
        "code_revision": revision,
        "input_sha256": "a" * 64,
    }


def _sample_boundary(problem: FullTilingProblem) -> FullTilingPyMCHMCSamplingResult:
    """Return one short non-cycle-aligned exact checkpoint."""
    initial = initialize_full_tiling_posterior_state(problem, k=3)
    return sample_full_tiling_pymc_hmc(
        problem,
        initial,
        FullTilingPyMCHMCConfig(
            iterations=2,
            step_size=0.002,
            leapfrog_steps=2,
            leaf_contrast_position_scale=1.4,
            leaf_total_position_scale=2.6,
            fixed_coefficient_position_scale=(0.7, 1.8),
            seed=481,
        ),
    )


def _assert_states_equal(
    actual: FullTilingPosteriorState,
    expected: FullTilingPosteriorState,
) -> None:
    """Assert exact coordinates, caches, and target components."""
    assert actual.tiling_state.tiling == expected.tiling_state.tiling
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
    actual: FullTilingPyMCHMCSamplingResult,
    expected: FullTilingPyMCHMCSamplingResult,
) -> None:
    """Assert exact traces, scientific states, and continuation metadata."""
    for field in fields(actual.trace):
        np.testing.assert_array_equal(
            getattr(actual.trace, field.name),
            getattr(expected.trace, field.name),
        )
    _assert_states_equal(actual.final_state, expected.final_state)
    for name in ("log_leaf_mass", "log_fixed_coefficient"):
        np.testing.assert_array_equal(
            getattr(actual.checkpoint, name),
            getattr(expected.checkpoint, name),
        )
    assert actual.checkpoint.rng_state == expected.checkpoint.rng_state
    assert actual.checkpoint.sweeps_completed == expected.checkpoint.sweeps_completed
    assert actual.checkpoint.kernel_settings == expected.checkpoint.kernel_settings
    assert actual.checkpoint.runtime_identity == expected.checkpoint.runtime_identity


def _rewrite_archive(
    path: Path,
    transform: Callable[[dict[str, np.ndarray[Any, Any]]], None],
) -> None:
    """Apply one test mutation and rewrite an NPZ archive."""
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
    """Rewrite metadata and its outer digest for an internal mismatch test."""

    def apply(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
        """Apply the mutation and recalculate the metadata digest."""
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


@_requires_x64_child
def test_save_load_continue_matches_uninterrupted_checkpoint_exactly(
    tmp_path: Path,
) -> None:
    """A durable equal-problem restart exactly reproduces direct continuation."""
    source_problem = _problem()
    restored_problem = _problem()
    first = _sample_boundary(source_problem)
    path = tmp_path / "checkpoint.npz"
    manifest = _manifest()
    save_full_tiling_pymc_hmc_checkpoint(
        path,
        first.checkpoint,
        run_manifest=manifest,
    )
    loaded = load_full_tiling_pymc_hmc_checkpoint(
        path,
        restored_problem,
        expected_run_manifest=manifest,
    )
    direct = continue_full_tiling_pymc_hmc(
        source_problem,
        first.checkpoint,
        iterations=2,
    )
    resumed = continue_full_tiling_pymc_hmc(
        restored_problem,
        loaded,
        iterations=2,
    )

    _assert_results_equal(resumed, direct)
    _assert_states_equal(loaded.state, first.checkpoint.state)
    np.testing.assert_array_equal(loaded.log_leaf_mass, first.checkpoint.log_leaf_mass)
    np.testing.assert_array_equal(
        loaded.log_fixed_coefficient,
        first.checkpoint.log_fixed_coefficient,
    )
    assert loaded.rng_state == first.checkpoint.rng_state
    assert loaded.kernel_settings == first.checkpoint.kernel_settings
    assert loaded.runtime_identity == first.checkpoint.runtime_identity


@_requires_x64_child
def test_v2_metadata_roundtrips_both_leaf_eigenscales(tmp_path: Path) -> None:
    """Schema v2 persists and restores distinct contrast and total eigenscales."""
    problem = _problem()
    result = _sample_boundary(problem)
    path = tmp_path / "metric-v2.npz"
    save_full_tiling_pymc_hmc_checkpoint(
        path,
        result.checkpoint,
        run_manifest=_manifest(),
    )

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
    loaded = load_full_tiling_pymc_hmc_checkpoint(
        path,
        _problem(),
        expected_run_manifest=_manifest(),
    )

    assert metadata["schema_version"] == 2
    assert metadata["kernel"]["leaf_contrast_position_scale"] == 1.4
    assert metadata["kernel"]["leaf_total_position_scale"] == 2.6
    assert loaded.kernel_settings.leaf_contrast_position_scale == 1.4
    assert loaded.kernel_settings.leaf_total_position_scale == 2.6


@_requires_x64_child
@pytest.mark.parametrize(
    "field",
    [
        "leaf_contrast_position_scale",
        "leaf_total_position_scale",
    ],
)
def test_loader_rejects_leaf_eigenscale_tampering(
    tmp_path: Path,
    field: str,
) -> None:
    """Invalid tampering of either persisted leaf eigenscale fails closed."""
    problem = _problem()
    result = _sample_boundary(problem)
    path = tmp_path / f"{field}.npz"
    save_full_tiling_pymc_hmc_checkpoint(
        path,
        result.checkpoint,
        run_manifest=_manifest(),
    )
    _rewrite_metadata(
        path,
        lambda metadata: metadata["kernel"].__setitem__(field, 0.0),
    )

    with pytest.raises(ValueError, match=rf"kernel\.{field} must be finite and positive"):
        load_full_tiling_pymc_hmc_checkpoint(
            path,
            problem,
            expected_run_manifest=_manifest(),
        )


@_requires_x64_child
def test_loader_rejects_v1_schema_and_old_scalar_leaf_scale_key(
    tmp_path: Path,
) -> None:
    """Old schema and scalar leaf-scale metadata cannot enter the v2 loader."""
    problem = _problem()
    result = _sample_boundary(problem)
    schema_path = tmp_path / "schema-v1.npz"
    save_full_tiling_pymc_hmc_checkpoint(
        schema_path,
        result.checkpoint,
        run_manifest=_manifest(),
    )
    _rewrite_metadata(
        schema_path,
        lambda metadata: metadata.__setitem__("schema_version", 1),
    )
    with pytest.raises(ValueError, match="schema version 1 uses the retired scalar"):
        load_full_tiling_pymc_hmc_checkpoint(
            schema_path,
            problem,
            expected_run_manifest=_manifest(),
        )

    key_path = tmp_path / "old-leaf-scale-key.npz"
    save_full_tiling_pymc_hmc_checkpoint(
        key_path,
        result.checkpoint,
        run_manifest=_manifest(),
    )

    def use_old_scalar_key(metadata: dict[str, Any]) -> None:
        """Replace the two v2 leaf scales with the rejected v1 scalar key."""
        kernel = metadata["kernel"]
        kernel["leaf_position_scale"] = kernel.pop("leaf_contrast_position_scale")
        kernel.pop("leaf_total_position_scale")

    _rewrite_metadata(key_path, use_old_scalar_key)
    with pytest.raises(ValueError, match="kernel has an invalid field set"):
        load_full_tiling_pymc_hmc_checkpoint(
            key_path,
            problem,
            expected_run_manifest=_manifest(),
        )


@_requires_x64_child
def test_archive_is_no_pickle_exact_field_set_and_create_only(tmp_path: Path) -> None:
    """Published archives have only numeric schema fields and are never replaced."""
    result = _sample_boundary(_problem())
    path = tmp_path / "checkpoint.npz"
    save_full_tiling_pymc_hmc_checkpoint(
        path,
        result.checkpoint,
        run_manifest=_manifest(),
    )
    original = path.read_bytes()

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == _ARCHIVE_FIELDS
        assert all(archive[name].dtype != np.dtype(object) for name in archive.files)
    with pytest.raises(FileExistsError):
        save_full_tiling_pymc_hmc_checkpoint(
            path,
            result.checkpoint,
            run_manifest=_manifest(),
        )
    assert path.read_bytes() == original


@_requires_x64_child
def test_loader_rejects_wrong_manifest_and_problem(tmp_path: Path) -> None:
    """Manifest content and scientific problem fingerprints fail closed."""
    problem = _problem()
    result = _sample_boundary(problem)
    path = tmp_path / "checkpoint.npz"
    save_full_tiling_pymc_hmc_checkpoint(
        path,
        result.checkpoint,
        run_manifest=_manifest(),
    )

    with pytest.raises(ValueError, match="problem fingerprint does not match"):
        load_full_tiling_pymc_hmc_checkpoint(
            path,
            _problem(observation_shift=0.1),
            expected_run_manifest=_manifest(),
        )
    with pytest.raises(ValueError, match="manifest does not match"):
        load_full_tiling_pymc_hmc_checkpoint(
            path,
            problem,
            expected_run_manifest=_manifest(revision="different"),
        )


@_requires_x64_child
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pymc_version", "0.0", "runtime identity does not match"),
        ("coordinate_layout_id", "future_layout_v2", "runtime identity does not match"),
        ("metric_semantics_id", "future_metric_v2", "runtime identity does not match"),
    ],
)
def test_loader_rejects_runtime_and_layout_identity_tampering(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    """Backend, coordinate-layout, and metric identities are exact."""
    problem = _problem()
    result = _sample_boundary(problem)
    path = tmp_path / f"{field}.npz"
    save_full_tiling_pymc_hmc_checkpoint(
        path,
        result.checkpoint,
        run_manifest=_manifest(),
    )
    _rewrite_metadata(
        path,
        lambda metadata: metadata["runtime_identity"].__setitem__(field, value),
    )

    with pytest.raises(ValueError, match=message):
        load_full_tiling_pymc_hmc_checkpoint(
            path,
            problem,
            expected_run_manifest=_manifest(),
        )


@_requires_x64_child
def test_loader_rejects_kernel_semantics_and_settings_tampering(
    tmp_path: Path,
) -> None:
    """Position-scale semantics and fixed-K settings cannot be relabelled."""
    problem = _problem()
    result = _sample_boundary(problem)
    semantics_path = tmp_path / "semantics.npz"
    save_full_tiling_pymc_hmc_checkpoint(
        semantics_path,
        result.checkpoint,
        run_manifest=_manifest(),
    )
    _rewrite_metadata(
        semantics_path,
        lambda metadata: metadata["kernel"].__setitem__(
            "position_scale_semantics",
            "future_position_scale_v2",
        ),
    )
    with pytest.raises(ValueError, match="position-scale semantics are incompatible"):
        load_full_tiling_pymc_hmc_checkpoint(
            semantics_path,
            problem,
            expected_run_manifest=_manifest(),
        )

    settings_path = tmp_path / "settings.npz"
    save_full_tiling_pymc_hmc_checkpoint(
        settings_path,
        result.checkpoint,
        run_manifest=_manifest(),
    )
    _rewrite_metadata(
        settings_path,
        lambda metadata: metadata["kernel"].__setitem__("fixed_k", 2),
    )
    with pytest.raises(ValueError, match="fixed K does not match"):
        load_full_tiling_pymc_hmc_checkpoint(
            settings_path,
            problem,
            expected_run_manifest=_manifest(),
        )


@_requires_x64_child
def test_loader_rejects_corruption_and_hash_tampering(tmp_path: Path) -> None:
    """Unreadable archives and altered arrays fail their integrity gates."""
    problem = _problem()
    result = _sample_boundary(problem)
    corrupt_path = tmp_path / "corrupt.npz"
    corrupt_path.write_bytes(b"not an npz checkpoint")
    with pytest.raises(ValueError, match="corrupt or unreadable"):
        load_full_tiling_pymc_hmc_checkpoint(
            corrupt_path,
            problem,
            expected_run_manifest=_manifest(),
        )

    hash_path = tmp_path / "hash.npz"
    save_full_tiling_pymc_hmc_checkpoint(
        hash_path,
        result.checkpoint,
        run_manifest=_manifest(),
    )
    _rewrite_archive(
        hash_path,
        lambda arrays: arrays["leaf_masses"].__setitem__(
            0,
            np.nextafter(arrays["leaf_masses"][0], np.inf),
        ),
    )
    with pytest.raises(ValueError, match="leaf_masses SHA-256 checksum does not match"):
        load_full_tiling_pymc_hmc_checkpoint(
            hash_path,
            problem,
            expected_run_manifest=_manifest(),
        )


@_requires_x64_child
def test_independent_cache_audit_rejects_forged_cache_hash(tmp_path: Path) -> None:
    """A forged prediction digest cannot bypass the scientific rebuild audit."""
    problem = _problem()
    result = _sample_boundary(problem)
    path = tmp_path / "cache.npz"
    save_full_tiling_pymc_hmc_checkpoint(
        path,
        result.checkpoint,
        run_manifest=_manifest(),
    )

    def forge_cache(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
        """Alter one cache and update both array and metadata digests."""
        arrays["prediction"][0] += 1.0e-5
        metadata = json.loads(arrays["metadata"].tobytes().decode("utf-8"))
        metadata["array_sha256"]["prediction"] = full_tiling_io._array_sha256(
            "prediction",
            arrays["prediction"],
        )
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

    _rewrite_archive(path, forge_cache)
    with pytest.raises(ValueError, match="stale or inconsistent cached arrays"):
        load_full_tiling_pymc_hmc_checkpoint(
            path,
            problem,
            expected_run_manifest=_manifest(),
        )


@_requires_x64_child
def test_loaded_arrays_restore_exact_caches_and_are_immutable(tmp_path: Path) -> None:
    """Loading restores persisted caches exactly as independent read-only arrays."""
    source_problem = _problem()
    result = _sample_boundary(source_problem)
    path = tmp_path / "immutable.npz"
    save_full_tiling_pymc_hmc_checkpoint(
        path,
        result.checkpoint,
        run_manifest=_manifest(),
    )
    loaded = load_full_tiling_pymc_hmc_checkpoint(
        path,
        _problem(),
        expected_run_manifest=_manifest(),
    )

    for name in (
        "leaf_masses",
        "fixed_coefficients",
        "dynamic_prediction",
        "fixed_prediction",
        "prediction",
        "residual",
    ):
        actual = getattr(loaded.state, name)
        np.testing.assert_array_equal(actual, getattr(result.checkpoint.state, name))
        assert not actual.flags.writeable
    for name in ("log_leaf_mass", "log_fixed_coefficient"):
        actual = getattr(loaded, name)
        np.testing.assert_array_equal(actual, getattr(result.checkpoint, name))
        assert not actual.flags.writeable
