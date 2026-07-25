"""Strict durable checkpoint I/O for the mobile full-tiling PyMC HMC kernel.

This module publishes create-only, atomic, no-pickle NPZ checkpoints for the
experimental structural-then-static-HMC sampler. The schema preserves the
canonical tiling, scientific coordinates, every posterior cache and log
component, the authoritative symmetric log coordinates, exact PCG64 state,
global sweep coordinate, and complete resolved HMC settings.

Loading fails closed unless the caller supplies the same fingerprinted
scientific problem and canonical run manifest and the current PyMC, PyTensor,
NumPy, Python-minor, platform, precision, coordinate-layout, schedule, and
kernel identities all match. It reconstructs the complete scientific state as
an independent audit. Small cache differences caused only by floating-point
summation order use the established full-tiling reconstruction tolerance;
after that audit, the persisted immutable caches are restored exactly so
continuation starts from the precise saved boundary.

Schema version 3 binds the topology-conditioned precision builder, reference,
semantics, and exact resolved precision hash without storing the dense
precision or a factorization. Versions 1 and 2 use retired static position
scales and are rejected explicitly; this module provides no converter.

SHA-256 digests detect accidental corruption but do not authenticate an
archive against a writer capable of replacing content and digests. Publication
uses a synced same-directory temporary file and an atomic hard link, so an
existing destination is never replaced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import errno
from hashlib import sha256
import hmac
from math import isfinite
import os
from pathlib import Path
import tempfile
from typing import Any, TypeAlias, cast
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray

from . import full_tiling_io as _full_tiling_io
from .full_tiling import LeafTiling, Rectangle, TilingState
from .full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    build_full_tiling_posterior_state,
)
from .full_tiling_pymc_hmc import (
    FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID,
    FULL_TILING_PYMC_HMC_SCHEDULE_ID,
    FullTilingPyMCHMCCheckpoint,
    FullTilingPyMCHMCKernelSettings,
    FullTilingPyMCHMCRuntimeIdentity,
    full_tiling_pymc_hmc_runtime_identity,
)
from .sampling import PCG64State

PathLike: TypeAlias = str | os.PathLike[str]
RunManifest: TypeAlias = Mapping[str, object]

FULL_TILING_PYMC_HMC_CHECKPOINT_SCHEMA_ID = (
    "openghg_inversions.experimental.rjmcmc.full_tiling_pymc_hmc_checkpoint"
)
"""Stable identifier for the durable full-tiling PyMC HMC checkpoint schema."""

FULL_TILING_PYMC_HMC_CHECKPOINT_SCHEMA_VERSION = 3
"""Current durable full-tiling PyMC HMC checkpoint schema version."""

_STATE_ARRAY_NAMES = (
    "rectangle_bounds",
    "leaf_masses",
    "fixed_coefficients",
    "log_leaf_mass",
    "log_fixed_coefficient",
    "dynamic_prediction",
    "fixed_prediction",
    "prediction",
    "residual",
)
_CACHE_ARRAY_NAMES = (
    "dynamic_prediction",
    "fixed_prediction",
    "prediction",
    "residual",
)
_STATE_LOG_FIELDS = (
    "log_gaussian_likelihood",
    "log_likelihood",
    "log_root_prior",
    "log_allocation_prior",
    "log_fixed_coefficient_prior",
    "log_target",
)
_KERNEL_FIELDS_V3 = (
    "fixed_k",
    "step_size",
    "leapfrog_steps",
    "metric_builder_id",
    "metric_reference_id",
)
_KERNEL_METADATA_NAMES = frozenset(
    (
        *_KERNEL_FIELDS_V3,
        "metric_semantics_id",
        "step_size_semantics",
    )
)
_STEP_SIZE_SEMANTICS = "requested_unscaled_integrator_step_size_v1"
_RUNTIME_IDENTITY_FIELDS_V3 = (
    "python_minor",
    "platform_system",
    "platform_machine",
    "numpy_version",
    "pymc_version",
    "pytensor_version",
    "pytensor_float_x",
    "coordinate_layout_id",
    "metric_semantics_id",
)
_RUNTIME_IDENTITY_NAMES = frozenset(
    {
        *_RUNTIME_IDENTITY_FIELDS_V3,
    }
)
_ARCHIVE_NAMES = frozenset((*_STATE_ARRAY_NAMES, "metadata", "metadata_sha256"))
_METADATA_NAMES = frozenset(
    {
        "schema_id",
        "schema_version",
        "runtime_identity",
        "schedule_id",
        "problem_sha256",
        "state_sha256",
        "topology_precision_sha256",
        "sweeps_completed",
        "rng",
        "kernel",
        "state",
        "run_manifest_json",
        "run_manifest_sha256",
        "array_sha256",
    }
)


def _require_mapping(value: object, *, location: str) -> Mapping[str, object]:
    """Return a string-keyed mapping or reject it.

    Args:
        value: Candidate decoded JSON value.
        location: Field path included in validation errors.

    Returns:
        The validated mapping.

    Raises:
        ValueError: If the value is not a string-keyed mapping.
    """
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _require_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    location: str,
) -> None:
    """Require exactly the declared mapping keys.

    Args:
        value: Mapping whose keys are checked.
        expected: Complete allowed and required key set.
        location: Field path included in validation errors.

    Raises:
        ValueError: If a key is missing or unexpected.
    """
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{location} has an invalid field set; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )


def _integer(
    value: object,
    *,
    location: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    """Return one bounded built-in integer metadata value.

    Args:
        value: Candidate decoded JSON scalar.
        location: Field path included in validation errors.
        minimum: Inclusive lower bound.
        maximum: Optional inclusive upper bound.

    Returns:
        Validated built-in integer.

    Raises:
        ValueError: If the value is Boolean, non-integral, or out of bounds.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer.")
    if value < minimum or (maximum is not None and value > maximum):
        description = f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ValueError(f"{location} must be {description}.")
    return value


def _finite_float(
    value: object,
    *,
    location: str,
    positive: bool = False,
) -> float:
    """Return one finite metadata float, optionally requiring positivity.

    Args:
        value: Candidate decoded JSON scalar.
        location: Field path included in validation errors.
        positive: Whether zero and negative values are invalid.

    Returns:
        Validated built-in float.

    Raises:
        ValueError: If the value is not a finite real scalar or violates the
            requested positivity constraint.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite real number.")
    result = float(value)
    if not isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{location} must be {qualifier}.")
    return result


def _nonempty_string(value: object, *, location: str) -> str:
    """Return one non-empty string metadata value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string.")
    return value


def _sha256_hex(value: object, *, location: str) -> str:
    """Return one canonical lowercase SHA-256 hexadecimal digest."""
    result = _nonempty_string(value, location=location)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{location} must be a lowercase SHA-256 hexadecimal digest.")
    return result


def _runtime_identity_metadata(
    identity: FullTilingPyMCHMCRuntimeIdentity,
) -> dict[str, str]:
    """Return strict metadata for the frozen v3 runtime identity.

    Args:
        identity: Exact immutable identity retained by the checkpoint.

    Returns:
        Strict-JSON mapping of backend, platform, precision, coordinate, and
        metric identities.

    Raises:
        TypeError: If ``identity`` has the wrong type or malformed fields.
        RuntimeError: If the identity dataclass changed without a schema
            version decision.
    """
    if not isinstance(identity, FullTilingPyMCHMCRuntimeIdentity):
        raise TypeError("runtime_identity must be a FullTilingPyMCHMCRuntimeIdentity.")
    current_fields = frozenset(field.name for field in fields(FullTilingPyMCHMCRuntimeIdentity))
    if current_fields != _RUNTIME_IDENTITY_NAMES:
        raise RuntimeError(
            "FullTilingPyMCHMCRuntimeIdentity changed without a durable checkpoint schema-version decision."
        )
    result = {name: getattr(identity, name) for name in _RUNTIME_IDENTITY_FIELDS_V3}
    if any(not isinstance(value, str) or not value for value in result.values()):
        raise TypeError("runtime_identity fields must be non-empty strings.")
    return result


def _checkpoint_arrays(
    checkpoint: FullTilingPyMCHMCCheckpoint,
) -> dict[str, NDArray[Any]]:
    """Return exact persisted scientific, cache, and log-coordinate arrays.

    Schema v3 deliberately excludes the dense topology precision and every
    factorization; those values are reconstructed in memory during loading.

    Args:
        checkpoint: Exact validated sampler boundary.

    Returns:
        Numeric NPZ members for scientific coordinates, authoritative log
        coordinates, and posterior caches.
    """
    state = checkpoint.state
    return {
        "rectangle_bounds": np.asarray(
            [
                (
                    leaf.row_start,
                    leaf.row_stop,
                    leaf.col_start,
                    leaf.col_stop,
                )
                for leaf in state.tiling_state.tiling.leaves
            ],
            dtype=np.int64,
        ),
        "leaf_masses": np.asarray(state.leaf_masses, dtype=np.float64),
        "fixed_coefficients": np.asarray(
            state.fixed_coefficients,
            dtype=np.float64,
        ),
        "log_leaf_mass": np.asarray(
            checkpoint.log_leaf_mass,
            dtype=np.float64,
        ),
        "log_fixed_coefficient": np.asarray(
            checkpoint.log_fixed_coefficient,
            dtype=np.float64,
        ),
        "dynamic_prediction": np.asarray(
            state.dynamic_prediction,
            dtype=np.float64,
        ),
        "fixed_prediction": np.asarray(
            state.fixed_prediction,
            dtype=np.float64,
        ),
        "prediction": np.asarray(state.prediction, dtype=np.float64),
        "residual": np.asarray(state.residual, dtype=np.float64),
    }


def _kernel_field_names() -> frozenset[str]:
    """Return the frozen v3 setting fields or require a schema decision."""
    current = frozenset(field.name for field in fields(FullTilingPyMCHMCKernelSettings))
    frozen = frozenset(_KERNEL_FIELDS_V3)
    if current != frozen:
        raise RuntimeError(
            "FullTilingPyMCHMCKernelSettings changed without a durable checkpoint schema-version decision."
        )
    return frozen


def _kernel_metadata(
    settings: FullTilingPyMCHMCKernelSettings,
) -> dict[str, object]:
    """Return strict metadata for resolved static HMC settings.

    Args:
        settings: Fully resolved in-memory kernel settings.

    Returns:
        Strict finite JSON metadata including explicit topology-precision and
        requested step-size semantics.

    Raises:
        TypeError: If a setting has an unsupported runtime type.
        RuntimeError: If the dataclass field set changed without a schema
            version decision.
    """
    _kernel_field_names()
    return {
        "fixed_k": settings.fixed_k,
        "step_size": settings.step_size,
        "leapfrog_steps": settings.leapfrog_steps,
        "metric_builder_id": settings.metric_builder_id,
        "metric_reference_id": settings.metric_reference_id,
        "metric_semantics_id": FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID,
        "step_size_semantics": _STEP_SIZE_SEMANTICS,
    }


def _checkpoint_metadata(
    checkpoint: FullTilingPyMCHMCCheckpoint,
    arrays: Mapping[str, NDArray[Any]],
    *,
    run_manifest_json: str,
) -> dict[str, object]:
    """Build strict metadata for one validated continuation boundary.

    Args:
        checkpoint: Exact validated sampler boundary.
        arrays: Numeric arrays that will be persisted.
        run_manifest_json: Canonical caller-owned run manifest.

    Returns:
        Strict finite JSON-compatible metadata with integrity digests.
    """
    state = checkpoint.state
    rng = checkpoint.rng_state
    return {
        "schema_id": FULL_TILING_PYMC_HMC_CHECKPOINT_SCHEMA_ID,
        "schema_version": FULL_TILING_PYMC_HMC_CHECKPOINT_SCHEMA_VERSION,
        "runtime_identity": _runtime_identity_metadata(checkpoint.runtime_identity),
        "schedule_id": checkpoint.schedule_id,
        "problem_sha256": _full_tiling_io.full_tiling_problem_fingerprint(checkpoint.problem),
        "state_sha256": _full_tiling_io.full_tiling_state_fingerprint(
            checkpoint.problem,
            state,
        ),
        "topology_precision_sha256": _sha256_hex(
            checkpoint.topology_precision_sha256,
            location="checkpoint.topology_precision_sha256",
        ),
        "sweeps_completed": checkpoint.sweeps_completed,
        "rng": {
            "algorithm": rng.algorithm,
            "state": rng.state,
            "increment": rng.increment,
            "has_uint32": rng.has_uint32,
            "uinteger": rng.uinteger,
        },
        "kernel": _kernel_metadata(checkpoint.kernel_settings),
        "state": {
            "k": state.k,
            "root_total": state.root_total,
            **{name: getattr(state, name) for name in _STATE_LOG_FIELDS},
        },
        "run_manifest_json": run_manifest_json,
        "run_manifest_sha256": sha256(run_manifest_json.encode("utf-8")).hexdigest(),
        "array_sha256": {name: _full_tiling_io._array_sha256(name, array) for name, array in arrays.items()},
    }


def _fsync_parent_directory(path: Path) -> None:
    """Persist one directory entry where directory fsync is supported.

    Args:
        path: Parent directory containing the new archive link.

    Raises:
        OSError: If opening or syncing fails for a supported reason.
    """
    unsupported = {
        errno.EACCES,
        errno.EBADF,
        errno.EINVAL,
        errno.EPERM,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno in unsupported:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def save_full_tiling_pymc_hmc_checkpoint(
    path: PathLike,
    checkpoint: FullTilingPyMCHMCCheckpoint,
    *,
    run_manifest: RunManifest,
) -> None:
    """Atomically publish one create-only PyMC HMC checkpoint.

    The archive retains authoritative ``log_leaf_mass`` and
    ``log_fixed_coefficient`` arrays in addition to their decoded scientific
    coordinates. Kernel metadata identifies the deterministic
    topology-precision builder, reference coordinates, and PyMC
    ``is_cov=False`` semantics. The resolved precision hash is retained, but
    neither the dense matrix nor a factorization is stored. The stored
    ``step_size`` is the requested unscaled integrator step; per-sweep
    effective values remain trace diagnostics and are not mutable continuation
    state.

    Args:
        path: New destination NPZ path whose parent directory already exists.
        checkpoint: Exact in-memory full-tiling PyMC HMC boundary.
        run_manifest: Caller-owned strict finite JSON manifest.

    Raises:
        TypeError: If the path, checkpoint, or manifest has the wrong type.
        ValueError: If the schedule or PCG64 state is incompatible, the state
            or log coordinates are stale, the metric identities or
            topology-precision digest are incompatible, or the manifest is
            invalid.
        ImportError: If the checkpoint backend is unavailable.
        RuntimeError: If PyTensor precision or the frozen settings schema is
            incompatible.
        FileExistsError: If ``path`` already exists.
        OSError: If writing, linking, or durable syncing fails.

    Notes:
        The archive is written and synced under a temporary same-directory
        name, then atomically hard-linked to the absent destination. Existing
        checkpoint generations are never replaced.
    """
    if not isinstance(checkpoint, FullTilingPyMCHMCCheckpoint):
        raise TypeError("checkpoint must be a FullTilingPyMCHMCCheckpoint.")
    if checkpoint.schedule_id != FULL_TILING_PYMC_HMC_SCHEDULE_ID:
        raise ValueError("checkpoint schedule is incompatible.")
    if checkpoint.runtime_identity != full_tiling_pymc_hmc_runtime_identity():
        raise ValueError("checkpoint runtime identity is incompatible.")
    try:
        checkpoint.rng_state.generator()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("checkpoint must contain a valid exact PCG64 continuation state.") from exc
    _full_tiling_io._rebuild_and_validate_state(
        checkpoint.problem,
        checkpoint.state,
        location="checkpoint state",
    )
    run_manifest_json = _full_tiling_io.canonical_full_tiling_run_manifest(run_manifest)
    arrays = _checkpoint_arrays(checkpoint)
    metadata = _checkpoint_metadata(
        checkpoint,
        arrays,
        run_manifest_json=run_manifest_json,
    )
    metadata_bytes = _full_tiling_io._canonical_json(
        metadata,
        location="checkpoint metadata",
    ).encode("utf-8")
    archive_arrays = {
        **arrays,
        "metadata": np.frombuffer(metadata_bytes, dtype=np.uint8),
        "metadata_sha256": np.frombuffer(
            sha256(metadata_bytes).hexdigest().encode("ascii"),
            dtype=np.uint8,
        ),
    }
    destination = Path(path)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **archive_arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, destination)
        temporary_path.unlink()
        temporary_path = None
        _fsync_parent_directory(destination.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_archive_arrays(path: Path) -> dict[str, NDArray[Any]]:
    """Load the exact numeric archive field set with pickle disabled.

    Args:
        path: Existing checkpoint archive.

    Returns:
        Owned arrays indexed by required schema names.

    Raises:
        OSError: If the path cannot be opened.
        ValueError: If the file is corrupt, is not NPZ, has wrong members, or
            contains an object array.
    """
    try:
        archive = np.load(path, allow_pickle=False)
    except (FileNotFoundError, PermissionError):
        raise
    except (BadZipFile, EOFError, ValueError) as exc:
        raise ValueError("Full-tiling PyMC HMC checkpoint archive is corrupt or unreadable.") from exc
    if isinstance(archive, np.ndarray):
        raise ValueError("Full-tiling PyMC HMC checkpoint must be an NPZ archive.")
    try:
        with archive:
            archive_names = archive.files
            if len(archive_names) != len(set(archive_names)):
                raise ValueError("Full-tiling PyMC HMC checkpoint contains duplicate array members.")
            names = frozenset(archive_names)
            if names != _ARCHIVE_NAMES:
                raise ValueError(
                    "Full-tiling PyMC HMC checkpoint has an invalid array set; "
                    f"missing={sorted(_ARCHIVE_NAMES - names)}, "
                    f"unexpected={sorted(names - _ARCHIVE_NAMES)}."
                )
            try:
                arrays = {name: np.array(archive[name], copy=True) for name in names}
            except ValueError as exc:
                raise ValueError(
                    "Full-tiling PyMC HMC checkpoint arrays cannot require pickle or object dtype."
                ) from exc
    except ValueError:
        raise
    except (BadZipFile, EOFError) as exc:
        raise ValueError("Full-tiling PyMC HMC checkpoint archive is corrupt or unreadable.") from exc
    if any(array.dtype == np.dtype(object) for array in arrays.values()):
        raise ValueError("Full-tiling PyMC HMC checkpoint arrays cannot use object dtype.")
    return arrays


def _metadata_from_arrays(
    arrays: Mapping[str, NDArray[Any]],
) -> dict[str, Any]:
    """Validate the metadata digest and decode strict JSON.

    Args:
        arrays: Loaded archive arrays including metadata bytes and digest.

    Returns:
        Parsed metadata object.

    Raises:
        ValueError: If byte layouts, checksums, or JSON are malformed.
    """
    metadata_bytes = arrays["metadata"]
    metadata_sha = arrays["metadata_sha256"]
    if metadata_bytes.dtype != np.dtype(np.uint8) or metadata_bytes.ndim != 1:
        raise ValueError("Checkpoint metadata must be a one-dimensional uint8 array.")
    if metadata_sha.dtype != np.dtype(np.uint8) or metadata_sha.shape != (64,):
        raise ValueError("Checkpoint metadata_sha256 must be a 64-byte uint8 array.")
    try:
        payload = metadata_bytes.tobytes().decode("utf-8")
        supplied = metadata_sha.tobytes().decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Checkpoint metadata is not valid UTF-8/ASCII.") from exc
    expected = sha256(metadata_bytes.tobytes()).hexdigest()
    if not hmac.compare_digest(supplied, expected):
        raise ValueError("Checkpoint metadata SHA-256 checksum does not match.")
    return _full_tiling_io._parse_json_object(
        payload,
        location="checkpoint metadata",
    )


def _state_array(
    arrays: Mapping[str, NDArray[Any]],
    name: str,
    *,
    dtype: np.dtype[Any],
    ndim: int,
) -> NDArray[Any]:
    """Validate one persisted state array's exact dtype and rank.

    Args:
        arrays: Loaded checkpoint arrays.
        name: Required state-array name.
        dtype: Exact required NumPy dtype.
        ndim: Exact required number of dimensions.

    Returns:
        Validated owned array.

    Raises:
        ValueError: If dtype or rank differs from the schema.
    """
    array = arrays[name]
    if array.dtype != dtype or array.ndim != ndim:
        raise ValueError(f"Checkpoint {name} must be {ndim}-dimensional {dtype}.")
    return array


def _readonly_float_copy(values: NDArray[Any]) -> NDArray[np.float64]:
    """Return one owned read-only ``float64`` array."""
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _kernel_settings_from_metadata(
    value: object,
) -> FullTilingPyMCHMCKernelSettings:
    """Reconstruct exact resolved HMC settings from strict metadata.

    Args:
        value: Parsed candidate kernel metadata.

    Returns:
        Fully validated immutable kernel settings.

    Raises:
        ValueError: If fields, semantics, types, or values are incompatible.
        RuntimeError: If the installed settings dataclass changed without a
            schema version decision.
    """
    kernel = _require_mapping(value, location="kernel")
    _kernel_field_names()
    _require_keys(kernel, _KERNEL_METADATA_NAMES, location="kernel")
    if kernel["metric_semantics_id"] != FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID:
        raise ValueError("Checkpoint HMC metric semantics are incompatible.")
    if kernel["step_size_semantics"] != _STEP_SIZE_SEMANTICS:
        raise ValueError("Checkpoint HMC step-size semantics are incompatible.")
    return FullTilingPyMCHMCKernelSettings(
        fixed_k=_integer(
            kernel["fixed_k"],
            location="kernel.fixed_k",
            minimum=1,
        ),
        step_size=_finite_float(
            kernel["step_size"],
            location="kernel.step_size",
            positive=True,
        ),
        leapfrog_steps=_integer(
            kernel["leapfrog_steps"],
            location="kernel.leapfrog_steps",
            minimum=1,
        ),
        metric_builder_id=_nonempty_string(
            kernel["metric_builder_id"],
            location="kernel.metric_builder_id",
        ),
        metric_reference_id=_nonempty_string(
            kernel["metric_reference_id"],
            location="kernel.metric_reference_id",
        ),
    )


def _runtime_identity_from_metadata(
    value: object,
) -> FullTilingPyMCHMCRuntimeIdentity:
    """Reconstruct and validate the exact continuation runtime identity.

    Args:
        value: Parsed candidate runtime-identity metadata.

    Returns:
        Immutable persisted identity after exact comparison with this process.

    Raises:
        ValueError: If fields or exact runtime identity do not match.
        ImportError: If PyMC or PyTensor is unavailable.
        RuntimeError: If PyTensor precision or the frozen identity schema is
            incompatible.
    """
    runtime = _require_mapping(value, location="runtime_identity")
    _require_keys(
        runtime,
        _RUNTIME_IDENTITY_NAMES,
        location="runtime_identity",
    )
    if any(not isinstance(runtime[name], str) or not runtime[name] for name in _RUNTIME_IDENTITY_NAMES):
        raise ValueError("Full-tiling PyMC HMC checkpoint runtime identity is malformed.")
    current = full_tiling_pymc_hmc_runtime_identity()
    _runtime_identity_metadata(current)
    persisted = FullTilingPyMCHMCRuntimeIdentity(
        **{name: cast(str, runtime[name]) for name in _RUNTIME_IDENTITY_FIELDS_V3}
    )
    if persisted != current:
        mismatches = [
            name for name in _RUNTIME_IDENTITY_FIELDS_V3 if getattr(persisted, name) != getattr(current, name)
        ]
        raise ValueError(
            f"Full-tiling PyMC HMC checkpoint runtime identity does not match for {mismatches[0]}."
        )
    return persisted


def _validate_manifest(
    metadata: Mapping[str, object],
    expected_run_manifest: RunManifest,
) -> None:
    """Require an intact canonical manifest equal to caller expectations.

    Args:
        metadata: Validated top-level checkpoint metadata.
        expected_run_manifest: Exact caller-owned expected manifest.

    Raises:
        ValueError: If manifest metadata, checksum, or canonical content is
            incompatible.
    """
    embedded_json = metadata["run_manifest_json"]
    embedded_sha = metadata["run_manifest_sha256"]
    if not isinstance(embedded_json, str) or not isinstance(embedded_sha, str):
        raise ValueError("Checkpoint run manifest metadata is malformed.")
    if not hmac.compare_digest(
        sha256(embedded_json.encode("utf-8")).hexdigest(),
        embedded_sha,
    ):
        raise ValueError("Checkpoint run manifest SHA-256 checksum does not match.")
    expected_json = _full_tiling_io.canonical_full_tiling_run_manifest(expected_run_manifest)
    if not hmac.compare_digest(embedded_json, expected_json):
        raise ValueError("Checkpoint run manifest does not match expected canonical content.")


def _validate_array_hashes(
    arrays: Mapping[str, NDArray[Any]],
    value: object,
) -> None:
    """Require every numeric array to match its exact-layout digest.

    Args:
        arrays: Loaded archive arrays.
        value: Parsed candidate digest mapping.

    Raises:
        ValueError: If fields or any SHA-256 digest are invalid.
    """
    hashes = _require_mapping(value, location="array_sha256")
    expected_names = frozenset(_STATE_ARRAY_NAMES)
    if frozenset(hashes) != expected_names:
        raise ValueError("Checkpoint array_sha256 has an invalid field set.")
    for name in _STATE_ARRAY_NAMES:
        supplied = hashes[name]
        if not isinstance(supplied, str) or not hmac.compare_digest(
            supplied,
            _full_tiling_io._array_sha256(name, arrays[name]),
        ):
            raise ValueError(f"Checkpoint {name} SHA-256 checksum does not match.")


def _rng_state_from_metadata(value: object) -> PCG64State:
    """Reconstruct and validate the exact PCG64 continuation state.

    Args:
        value: Parsed candidate RNG metadata.

    Returns:
        Valid exact PCG64 state.

    Raises:
        ValueError: If fields, bounds, algorithm, or NumPy restoration fail.
    """
    rng = _require_mapping(value, location="rng")
    _require_keys(
        rng,
        frozenset(("algorithm", "state", "increment", "has_uint32", "uinteger")),
        location="rng",
    )
    if rng["algorithm"] != "PCG64":
        raise ValueError("rng.algorithm must be exactly 'PCG64'.")
    result = PCG64State(
        state=_integer(
            rng["state"],
            location="rng.state",
            maximum=(1 << 128) - 1,
        ),
        increment=_integer(
            rng["increment"],
            location="rng.increment",
            maximum=(1 << 128) - 1,
        ),
        has_uint32=_integer(
            rng["has_uint32"],
            location="rng.has_uint32",
            maximum=1,
        ),
        uinteger=_integer(
            rng["uinteger"],
            location="rng.uinteger",
            maximum=(1 << 32) - 1,
        ),
        algorithm="PCG64",
    )
    try:
        result.generator()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Checkpoint PCG64 state is invalid.") from exc
    return result


def load_full_tiling_pymc_hmc_checkpoint(
    path: PathLike,
    problem: FullTilingProblem,
    *,
    expected_run_manifest: RunManifest,
) -> FullTilingPyMCHMCCheckpoint:
    """Load and independently validate a PyMC HMC continuation boundary.

    The current runtime must exactly match the checkpoint's PyMC, PyTensor,
    NumPy, Python-minor, platform, ``floatX``, symmetric-coordinate layout,
    topology-conditioned HMC semantics, and schedule identities. Scientific
    coordinates are fully rebuilt against ``problem`` and audited before exact
    persisted caches and authoritative log coordinates are attached to the
    returned checkpoint. Schema v3 does not persist a dense precision matrix
    or factorization. After rebuilding the state, checkpoint construction
    deterministically reconstructs the current topology precision, hashes its
    canonical encoding, and requires exact equality with
    ``topology_precision_sha256``.

    Args:
        path: Source NPZ archive.
        problem: Equal reconstructed scientific problem for the resumed run.
        expected_run_manifest: Exact caller-held manifest expected in the
            archive.

    Returns:
        Valid checkpoint attached to ``problem`` with exact caches, log
        coordinates, sweep count, settings, and PCG64 state restored.

    Raises:
        TypeError: If the path, ``problem``, or manifest has the wrong type.
        ValueError: If archive schema, hashes, backend, problem, manifest,
            topology, coordinates, caches, target terms, RNG, schedule, or
            resolved HMC settings disagree, including if the reconstructed
            topology-precision hash does not match.
        ImportError: If PyMC or PyTensor is unavailable.
        RuntimeError: If PyTensor precision or the frozen settings schema is
            incompatible.
        OSError: If the archive cannot be read.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    if not isinstance(expected_run_manifest, Mapping):
        raise TypeError("expected_run_manifest must be a mapping.")
    arrays = _load_archive_arrays(Path(path))
    metadata = _metadata_from_arrays(arrays)
    if metadata.get("schema_id") != FULL_TILING_PYMC_HMC_CHECKPOINT_SCHEMA_ID:
        raise ValueError("Full-tiling PyMC HMC checkpoint schema is incompatible.")
    schema_version = _integer(
        metadata.get("schema_version"),
        location="schema_version",
        minimum=1,
    )
    if schema_version == 1:
        raise ValueError(
            "Full-tiling PyMC HMC checkpoint schema version 1 uses the retired "
            "scalar diagonal leaf metric; schema version 3 is required and no "
            "converter is provided."
        )
    if schema_version == 2:
        raise ValueError(
            "Full-tiling PyMC HMC checkpoint schema version 2 uses retired "
            "static leaf-eigenscale and fixed position-scale settings; schema "
            "version 3 is required and no converter is provided."
        )
    if schema_version != FULL_TILING_PYMC_HMC_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Full-tiling PyMC HMC checkpoint schema version is incompatible.")
    _require_keys(metadata, _METADATA_NAMES, location="checkpoint metadata")
    runtime_identity = _runtime_identity_from_metadata(metadata["runtime_identity"])
    schedule_id = metadata["schedule_id"]
    if schedule_id != FULL_TILING_PYMC_HMC_SCHEDULE_ID:
        raise ValueError("Full-tiling PyMC HMC checkpoint schedule is incompatible.")
    problem_sha = _full_tiling_io.full_tiling_problem_fingerprint(problem)
    if metadata["problem_sha256"] != problem_sha:
        raise ValueError("Full-tiling PyMC HMC checkpoint problem fingerprint does not match.")
    _validate_manifest(metadata, expected_run_manifest)
    _validate_array_hashes(arrays, metadata["array_sha256"])

    rectangle_bounds = _state_array(
        arrays,
        "rectangle_bounds",
        dtype=np.dtype(np.int64),
        ndim=2,
    )
    leaf_masses = _state_array(
        arrays,
        "leaf_masses",
        dtype=np.dtype(np.float64),
        ndim=1,
    )
    fixed_coefficients = _state_array(
        arrays,
        "fixed_coefficients",
        dtype=np.dtype(np.float64),
        ndim=1,
    )
    log_leaf_mass = _state_array(
        arrays,
        "log_leaf_mass",
        dtype=np.dtype(np.float64),
        ndim=1,
    )
    log_fixed_coefficient = _state_array(
        arrays,
        "log_fixed_coefficient",
        dtype=np.dtype(np.float64),
        ndim=1,
    )
    caches = {
        name: _state_array(
            arrays,
            name,
            dtype=np.dtype(np.float64),
            ndim=1,
        )
        for name in _CACHE_ARRAY_NAMES
    }
    state_metadata = _require_mapping(metadata["state"], location="state")
    _require_keys(
        state_metadata,
        frozenset(("k", "root_total", *_STATE_LOG_FIELDS)),
        location="state",
    )
    k = _integer(state_metadata["k"], location="state.k", minimum=1)
    if rectangle_bounds.shape != (k, 4):
        raise ValueError("Checkpoint rectangle_bounds must have shape (K, 4).")
    if leaf_masses.shape != (k,) or log_leaf_mass.shape != (k,):
        raise ValueError("Checkpoint leaf mass and log-coordinate arrays must have shape (K,).")
    if log_fixed_coefficient.shape != fixed_coefficients.shape:
        raise ValueError("Checkpoint fixed coefficients and log coordinates must have matching shapes.")
    observation_shape = problem.observations.shape
    if any(cache.shape != observation_shape for cache in caches.values()):
        raise ValueError("Checkpoint posterior caches must match the observation shape.")
    if np.any(~np.isfinite(log_leaf_mass)) or np.any(~np.isfinite(log_fixed_coefficient)):
        raise ValueError("Checkpoint authoritative log coordinates must be finite.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if not np.array_equal(np.exp(log_leaf_mass), leaf_masses):
            raise ValueError("Checkpoint log_leaf_mass does not exactly encode leaf_masses.")
        if not np.array_equal(
            np.exp(log_fixed_coefficient),
            fixed_coefficients,
        ):
            raise ValueError("Checkpoint log_fixed_coefficient does not exactly encode fixed_coefficients.")
    try:
        rectangles = tuple(Rectangle(*(int(bound) for bound in row)) for row in rectangle_bounds.tolist())
        allocation = TilingState(
            LeafTiling(problem.shape, rectangles),
            leaf_masses,
        )
        rebuilt = build_full_tiling_posterior_state(
            problem,
            allocation=allocation,
            fixed_coefficients=fixed_coefficients,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint topology or state coordinates are invalid.") from exc

    _full_tiling_io._validate_rebuilt_caches(
        problem,
        rebuilt,
        caches,
        location="Rebuilt checkpoint state",
    )
    expected_root_total = _finite_float(
        state_metadata["root_total"],
        location="state.root_total",
        positive=True,
    )
    if rebuilt.root_total != expected_root_total:
        raise ValueError("Rebuilt state root_total does not match checkpoint metadata.")
    persisted_logs = {
        name: _finite_float(
            state_metadata[name],
            location=f"state.{name}",
        )
        for name in _STATE_LOG_FIELDS
    }
    cache_consistent = _full_tiling_io._cache_consistent_state(
        problem,
        rebuilt,
        caches,
    )
    _full_tiling_io._validate_cache_consistent_values(
        cache_consistent,
        caches,
        persisted_logs,
        location="Checkpoint state",
    )
    state = FullTilingPosteriorState(
        problem=problem,
        allocation=allocation,
        fixed_coefficients=_readonly_float_copy(fixed_coefficients),
        dynamic_prediction=_readonly_float_copy(caches["dynamic_prediction"]),
        fixed_prediction=_readonly_float_copy(caches["fixed_prediction"]),
        prediction=_readonly_float_copy(caches["prediction"]),
        residual=_readonly_float_copy(caches["residual"]),
        log_gaussian_likelihood=persisted_logs["log_gaussian_likelihood"],
        log_likelihood=persisted_logs["log_likelihood"],
        log_root_prior=persisted_logs["log_root_prior"],
        log_allocation_prior=persisted_logs["log_allocation_prior"],
        log_fixed_coefficient_prior=persisted_logs["log_fixed_coefficient_prior"],
    )
    if state.log_target != persisted_logs["log_target"]:
        raise ValueError("Persisted state log_target is inconsistent with its components.")
    supplied_state_sha = metadata["state_sha256"]
    rebuilt_state_sha = _full_tiling_io.full_tiling_state_fingerprint(
        problem,
        state,
    )
    if not isinstance(supplied_state_sha, str) or not hmac.compare_digest(
        supplied_state_sha,
        rebuilt_state_sha,
    ):
        raise ValueError("Full-tiling PyMC HMC checkpoint state fingerprint does not match.")

    settings = _kernel_settings_from_metadata(metadata["kernel"])
    if settings.fixed_k != k:
        raise ValueError("Checkpoint fixed K does not match the persisted state.")
    rng_state = _rng_state_from_metadata(metadata["rng"])
    topology_precision_sha256 = _sha256_hex(
        metadata["topology_precision_sha256"],
        location="topology_precision_sha256",
    )
    return FullTilingPyMCHMCCheckpoint(
        problem=problem,
        state=state,
        log_leaf_mass=_readonly_float_copy(log_leaf_mass),
        log_fixed_coefficient=_readonly_float_copy(log_fixed_coefficient),
        rng_state=rng_state,
        sweeps_completed=_integer(
            metadata["sweeps_completed"],
            location="sweeps_completed",
        ),
        kernel_settings=settings,
        runtime_identity=runtime_identity,
        topology_precision_sha256=topology_precision_sha256,
        schedule_id=cast(str, schedule_id),
    )


__all__ = [
    "FULL_TILING_PYMC_HMC_CHECKPOINT_SCHEMA_ID",
    "FULL_TILING_PYMC_HMC_CHECKPOINT_SCHEMA_VERSION",
    "load_full_tiling_pymc_hmc_checkpoint",
    "save_full_tiling_pymc_hmc_checkpoint",
]
