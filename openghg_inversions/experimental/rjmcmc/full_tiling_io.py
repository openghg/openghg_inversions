"""Strict durable checkpoint I/O for the fixed-``K`` full-tiling sampler.

This module fingerprints full-tiling scientific problems and state
coordinates, canonicalizes caller-owned run manifests, and publishes
checksummed no-pickle NPZ checkpoints. Checkpoints preserve the exact PCG64
state, resolved compound-kernel settings, irreducible geometry and allocation
coordinates, and every posterior cache required for an exact continuation
boundary.

Loading reconstructs rectangle topology and allocation coordinates against a
caller-supplied equal problem and independently rebuilds the complete
posterior state. The audit uses the sampler's strict numerical reconstruction
tolerances because incremental updates can differ from a full rebuild by
floating-point summation order; the original persisted caches are restored
exactly for bit-for-bit continuation. SHA-256 digests detect accidental
corruption but do not authenticate archives against a writer capable of
replacing both content and digests. Publication is atomic and create-only: an
existing destination is never replaced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import errno
from hashlib import sha256
import hmac
import json
from math import isfinite
import os
from pathlib import Path
import tempfile
from typing import Any, TypeAlias, cast
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray

from .full_tiling import LeafTiling, Rectangle, TilingState
from .full_tiling_compound_sampling import (
    FULL_TILING_COMPOUND_SCHEDULE_ID,
    FullTilingCompoundCheckpoint,
    FullTilingCompoundKernelSettings,
)
from .full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    build_full_tiling_posterior_state,
)
from .gamma_beta_io import gamma_beta_problem_fingerprint
from .sampling import PCG64State

PathLike: TypeAlias = str | os.PathLike[str]
RunManifest: TypeAlias = Mapping[str, object]

FULL_TILING_CHECKPOINT_SCHEMA_ID = "openghg_inversions.experimental.rjmcmc.full_tiling_checkpoint"
"""Stable identifier for the durable full-tiling checkpoint schema."""

FULL_TILING_CHECKPOINT_SCHEMA_VERSION = 1
"""Current durable full-tiling checkpoint schema version."""

_STATE_ARRAY_NAMES = (
    "rectangle_bounds",
    "leaf_masses",
    "fixed_coefficients",
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
_KERNEL_FIELDS_V1 = (
    "fixed_k",
    "pair_allocation_refresh_slots",
    "fixed_coefficient_proposal_sd",
    "root_slice_width",
    "root_slice_max_steps",
    "root_slice_max_shrink_steps",
)
_ARCHIVE_NAMES = frozenset((*_STATE_ARRAY_NAMES, "metadata", "metadata_sha256"))
_STATE_LOG_FIELDS = (
    "log_gaussian_likelihood",
    "log_likelihood",
    "log_root_prior",
    "log_allocation_prior",
    "log_fixed_coefficient_prior",
    "log_target",
)
_CACHE_REBUILD_ATOL = 5e-13
_TARGET_REBUILD_ATOL = 1e-9
_METADATA_NAMES = frozenset(
    {
        "schema_id",
        "schema_version",
        "numpy_version",
        "schedule_id",
        "problem_sha256",
        "state_sha256",
        "transitions_completed",
        "schedule_phase",
        "rng",
        "kernel",
        "state",
        "run_manifest_json",
        "run_manifest_sha256",
        "array_sha256",
    }
)


def _validate_json_value(value: object, *, location: str) -> None:
    """Reject values outside strict finite JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{location} must not contain non-finite floats.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, location=f"{location}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} object keys must be strings.")
            _validate_json_value(item, location=f"{location}.{key}")
        return
    raise ValueError(f"{location} contains unsupported value type {type(value).__name__!r}.")


def _canonical_json(value: object, *, location: str) -> str:
    """Return deterministic strict JSON for supported built-in containers."""
    _validate_json_value(value, location=location)

    def builtins_only(item: object) -> object:
        """Convert generic mappings into ordinary dictionaries."""
        if isinstance(item, Mapping):
            return {str(key): builtins_only(child) for key, child in item.items()}
        if isinstance(item, list):
            return [builtins_only(child) for child in item]
        return item

    return json.dumps(
        builtins_only(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_full_tiling_run_manifest(run_manifest: RunManifest) -> str:
    """Return the deterministic strict-JSON representation of a run manifest.

    The manifest schema is caller-owned. This function accepts any mapping
    whose complete recursive content is finite strict JSON.

    Args:
        run_manifest: Caller-owned scientific run manifest.

    Returns:
        Canonical JSON with sorted object keys and no insignificant
        whitespace.

    Raises:
        TypeError: If ``run_manifest`` is not a mapping.
        ValueError: If it contains unsupported or non-finite content.
    """
    if not isinstance(run_manifest, Mapping):
        raise TypeError("run_manifest must be a mapping.")
    return _canonical_json(run_manifest, location="run_manifest")


def _json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Checkpoint metadata repeats JSON key {key!r}.")
        result[key] = value
    return result


def _parse_json_object(payload: str, *, location: str) -> dict[str, Any]:
    """Parse one strict JSON object without constants or duplicate keys."""

    def reject_constant(value: str) -> None:
        """Reject the non-standard NaN and infinity JSON constants."""
        raise ValueError(f"{location} contains invalid JSON constant {value!r}.")

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=_json_object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location} is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{location} must contain a JSON object.")
    _validate_json_value(parsed, location=location)
    return cast(dict[str, Any], parsed)


def _require_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    location: str,
) -> None:
    """Require exactly the declared mapping keys."""
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{location} has an invalid field set; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )


def _require_mapping(value: object, *, location: str) -> Mapping[str, object]:
    """Return a string-keyed mapping or reject it."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _integer(
    value: object,
    *,
    location: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    """Return a bounded built-in integer metadata value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer.")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            description = f"at least {minimum}"
        else:
            description = f"between {minimum} and {maximum}"
        raise ValueError(f"{location} must be {description}.")
    return value


def _finite_float(
    value: object,
    *,
    location: str,
    positive: bool = False,
) -> float:
    """Return one finite metadata float, optionally requiring positivity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite real number.")
    result = float(value)
    if not isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{location} must be {qualifier}.")
    return result


def _update_hash_bytes(digest: Any, label: str, payload: bytes) -> None:
    """Add one length-framed labelled byte sequence to a digest."""
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _update_hash_array(digest: Any, label: str, values: object) -> None:
    """Add an array's dtype, shape, and C-order values to a digest."""
    array = np.ascontiguousarray(np.asarray(values))
    descriptor = _canonical_json(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        location=f"{label} descriptor",
    )
    _update_hash_bytes(digest, f"{label}.descriptor", descriptor.encode("utf-8"))
    _update_hash_bytes(digest, f"{label}.data", array.tobytes(order="C"))


def _array_sha256(label: str, values: object) -> str:
    """Return the framed exact-layout SHA-256 for one array."""
    digest = sha256()
    _update_hash_array(digest, label, values)
    return digest.hexdigest()


def full_tiling_problem_fingerprint(problem: FullTilingProblem) -> str:
    """Return a deterministic SHA-256 identity for a full-tiling problem.

    The fingerprint covers the complete wrapped Gamma--Beta observation
    problem and the full-tiling allocation concentration. Lazy rectangle
    design caches and other deterministic derived values are excluded.

    Args:
        problem: Immutable full-tiling scientific problem.

    Returns:
        Lower-case 64-character SHA-256 digest.

    Raises:
        TypeError: If ``problem`` has the wrong type.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    digest = sha256()
    _update_hash_bytes(digest, "fingerprint_schema", b"full_tiling_problem_v1")
    _update_hash_bytes(
        digest,
        "base_problem_sha256",
        gamma_beta_problem_fingerprint(problem.base).encode("ascii"),
    )
    _update_hash_array(
        digest,
        "allocation_concentration",
        np.asarray(problem.concentration, dtype=np.float64),
    )
    return digest.hexdigest()


def _rectangle_bounds(state: FullTilingPosteriorState) -> NDArray[np.int64]:
    """Return canonical rectangle bounds for one posterior state."""
    return np.asarray(
        [
            (
                leaf.row_start,
                leaf.row_stop,
                leaf.col_start,
                leaf.col_stop,
            )
            for leaf in state.allocation.tiling.leaves
        ],
        dtype=np.int64,
    )


def _rebuild_and_validate_state(
    problem: FullTilingProblem,
    state: object,
    *,
    location: str,
) -> FullTilingPosteriorState:
    """Independently rebuild and numerically validate one posterior state.

    Args:
        problem: Exact problem instance expected to own ``state``.
        state: Candidate posterior state.
        location: Field path used in validation errors.

    Returns:
        Independently rebuilt state with the same scientific coordinates.

    Raises:
        TypeError: If ``state`` has the wrong type.
        ValueError: If problem identity, topology, coordinates, caches, target
            support, or target components are inconsistent.
    """
    if not isinstance(state, FullTilingPosteriorState):
        raise TypeError(f"{location} must be a FullTilingPosteriorState.")
    if state.problem is not problem:
        raise ValueError(f"{location} must belong to the supplied problem instance.")
    rebuilt = build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(
            LeafTiling(
                state.allocation.tiling.shape,
                tuple(state.allocation.tiling.leaves),
            ),
            state.leaf_masses,
        ),
        fixed_coefficients=state.fixed_coefficients,
    )
    if not isfinite(rebuilt.log_target):
        raise ValueError(f"{location} must have finite target support.")
    if rebuilt.allocation.tiling != state.allocation.tiling:
        raise ValueError(f"{location} has inconsistent topology coordinates.")
    coordinate_fields = ("leaf_masses", "fixed_coefficients")
    if any(not np.array_equal(getattr(rebuilt, name), getattr(state, name)) for name in coordinate_fields):
        raise ValueError(f"{location} has inconsistent continuous coordinates.")
    if any(
        not np.allclose(
            getattr(rebuilt, name),
            getattr(state, name),
            rtol=0.0,
            atol=_CACHE_REBUILD_ATOL,
        )
        for name in _CACHE_ARRAY_NAMES
    ):
        raise ValueError(f"{location} contains stale or inconsistent cached arrays.")
    if any(
        not np.isclose(
            getattr(rebuilt, name),
            getattr(state, name),
            rtol=0.0,
            atol=_TARGET_REBUILD_ATOL,
        )
        for name in _STATE_LOG_FIELDS
    ):
        raise ValueError(f"{location} contains stale or inconsistent target components.")
    return rebuilt


def full_tiling_state_fingerprint(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
) -> str:
    """Return a canonical SHA-256 identity for state coordinates.

    The identity covers canonical rectangle bounds, aligned leaf masses, and
    fixed coefficients. Derived predictions and target components are
    excluded from the digest but are independently rebuilt and validated to
    the sampler's strict reconstruction tolerances before hashing.

    Args:
        problem: Exact problem instance to which ``state`` must belong.
        state: Full-tiling posterior state to identify.

    Returns:
        Lower-case 64-character SHA-256 digest.

    Raises:
        TypeError: If either argument has the wrong type.
        ValueError: If the state belongs to another problem or contains stale
            coordinates, caches, or target terms.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    validated = _rebuild_and_validate_state(problem, state, location="state")
    digest = sha256()
    _update_hash_bytes(
        digest,
        "fingerprint_schema",
        b"full_tiling_state_coordinates_v1",
    )
    _update_hash_array(digest, "rectangle_bounds", _rectangle_bounds(validated))
    _update_hash_array(digest, "leaf_masses", validated.leaf_masses)
    _update_hash_array(
        digest,
        "fixed_coefficients",
        validated.fixed_coefficients,
    )
    return digest.hexdigest()


def _checkpoint_arrays(
    state: FullTilingPosteriorState,
) -> dict[str, NDArray[Any]]:
    """Return exact coordinate and cache arrays for persistence."""
    return {
        "rectangle_bounds": _rectangle_bounds(state),
        "leaf_masses": np.asarray(state.leaf_masses, dtype=np.float64),
        "fixed_coefficients": np.asarray(
            state.fixed_coefficients,
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
    """Return the frozen v1 kernel field set or require a schema decision."""
    current_fields = frozenset(field.name for field in fields(FullTilingCompoundKernelSettings))
    frozen_fields = frozenset(_KERNEL_FIELDS_V1)
    if current_fields != frozen_fields:
        raise RuntimeError(
            "FullTilingCompoundKernelSettings changed without a durable checkpoint schema-version decision."
        )
    return frozen_fields


def _kernel_metadata(
    settings: FullTilingCompoundKernelSettings,
) -> dict[str, object]:
    """Return strict JSON metadata for the frozen v1 kernel schema."""
    _kernel_field_names()
    result: dict[str, object] = {}
    for name in _KERNEL_FIELDS_V1:
        value = getattr(settings, name)
        if isinstance(value, tuple):
            result[name] = list(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[name] = value
        else:
            raise TypeError(f"Unsupported kernel setting type for {name!r}: {type(value).__name__}.")
    return result


def _checkpoint_metadata(
    checkpoint: FullTilingCompoundCheckpoint,
    arrays: Mapping[str, NDArray[Any]],
    *,
    run_manifest_json: str,
) -> dict[str, object]:
    """Build strict metadata for one validated continuation boundary.

    Args:
        checkpoint: Exact validated sampler boundary.
        arrays: Coordinate and cache arrays that will be persisted.
        run_manifest_json: Canonical caller-owned run manifest.

    Returns:
        Strict finite JSON-compatible metadata with all integrity digests.
    """
    state = checkpoint.state
    rng = checkpoint.rng_state
    return {
        "schema_id": FULL_TILING_CHECKPOINT_SCHEMA_ID,
        "schema_version": FULL_TILING_CHECKPOINT_SCHEMA_VERSION,
        "numpy_version": np.__version__,
        "schedule_id": checkpoint.schedule_id,
        "problem_sha256": full_tiling_problem_fingerprint(checkpoint.problem),
        "state_sha256": full_tiling_state_fingerprint(
            checkpoint.problem,
            checkpoint.state,
        ),
        "transitions_completed": checkpoint.transitions_completed,
        "schedule_phase": checkpoint.schedule_phase,
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
        "array_sha256": {name: _array_sha256(name, array) for name, array in arrays.items()},
    }


def _fsync_parent_directory(path: Path) -> None:
    """Persist one directory entry where directory fsync is supported."""
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


def save_full_tiling_checkpoint(
    path: PathLike,
    checkpoint: FullTilingCompoundCheckpoint,
    *,
    run_manifest: RunManifest,
) -> None:
    """Atomically publish one create-only full-tiling checkpoint.

    Args:
        path: New destination NPZ path whose parent directory already exists.
        checkpoint: Exact in-memory full-tiling continuation boundary.
        run_manifest: Caller-owned strict finite JSON manifest.

    Raises:
        TypeError: If the checkpoint or manifest has the wrong type.
        ValueError: If the schedule or PCG64 state is incompatible, the state
            has stale coordinates, caches, or target components, or the
            manifest is not strict finite JSON.
        FileExistsError: If ``path`` already exists.
        OSError: If writing, linking, or durable syncing fails.

    Notes:
        The archive is written and synced under a temporary same-directory
        name, then atomically hard-linked to the absent destination. This
        create-only publication never replaces an existing generation.
    """
    if not isinstance(checkpoint, FullTilingCompoundCheckpoint):
        raise TypeError("checkpoint must be a FullTilingCompoundCheckpoint.")
    if checkpoint.schedule_id != FULL_TILING_COMPOUND_SCHEDULE_ID:
        raise ValueError("checkpoint schedule is incompatible.")
    try:
        checkpoint.rng_state.generator()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("checkpoint must contain a valid exact PCG64 continuation state.") from exc
    _rebuild_and_validate_state(
        checkpoint.problem,
        checkpoint.state,
        location="checkpoint state",
    )
    run_manifest_json = canonical_full_tiling_run_manifest(run_manifest)
    arrays = _checkpoint_arrays(checkpoint.state)
    metadata = _checkpoint_metadata(
        checkpoint,
        arrays,
        run_manifest_json=run_manifest_json,
    )
    metadata_bytes = _canonical_json(
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
        Owned arrays indexed by the required schema field names.

    Raises:
        OSError: If the path cannot be opened.
        ValueError: If the file is corrupt, is not an NPZ archive, has an
            invalid field set, or contains object arrays.
    """
    try:
        archive = np.load(path, allow_pickle=False)
    except (FileNotFoundError, PermissionError):
        raise
    except (BadZipFile, EOFError, ValueError) as exc:
        raise ValueError("Full-tiling checkpoint archive is corrupt or unreadable.") from exc
    if isinstance(archive, np.ndarray):
        raise ValueError("Full-tiling checkpoint must be an NPZ archive.")
    try:
        with archive:
            archive_names = archive.files
            if len(archive_names) != len(set(archive_names)):
                raise ValueError("Full-tiling checkpoint contains duplicate array members.")
            names = frozenset(archive_names)
            if names != _ARCHIVE_NAMES:
                raise ValueError(
                    "Full-tiling checkpoint has an invalid array set; "
                    f"missing={sorted(_ARCHIVE_NAMES - names)}, "
                    f"unexpected={sorted(names - _ARCHIVE_NAMES)}."
                )
            try:
                arrays = {name: np.array(archive[name], copy=True) for name in names}
            except ValueError as exc:
                raise ValueError(
                    "Full-tiling checkpoint arrays cannot require pickle or object dtype."
                ) from exc
    except ValueError:
        raise
    except (BadZipFile, EOFError) as exc:
        raise ValueError("Full-tiling checkpoint archive is corrupt or unreadable.") from exc
    if any(array.dtype == np.dtype(object) for array in arrays.values()):
        raise ValueError("Full-tiling checkpoint arrays cannot use object dtype.")
    return arrays


def _metadata_from_arrays(
    arrays: Mapping[str, NDArray[Any]],
) -> dict[str, Any]:
    """Validate the metadata digest and decode strict JSON.

    Args:
        arrays: Loaded archive arrays containing metadata bytes and digest.

    Returns:
        Parsed metadata with the exact schema field set.

    Raises:
        ValueError: If byte-array layouts, checksums, JSON, or fields are
            malformed.
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
    metadata = _parse_json_object(payload, location="checkpoint metadata")
    _require_keys(metadata, _METADATA_NAMES, location="checkpoint metadata")
    return metadata


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
        ValueError: If dtype or rank differs from the checkpoint schema.
    """
    array = arrays[name]
    if array.dtype != dtype or array.ndim != ndim:
        raise ValueError(f"Checkpoint {name} must be {ndim}-dimensional {dtype}.")
    return array


def _readonly_float_copy(values: NDArray[Any]) -> NDArray[np.float64]:
    """Return one owned read-only ``float64`` cache array."""
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _kernel_settings_from_metadata(
    value: object,
) -> FullTilingCompoundKernelSettings:
    """Reconstruct exact resolved kernel settings from strict metadata.

    Args:
        value: Parsed candidate kernel metadata.

    Returns:
        Fully validated immutable kernel settings.

    Raises:
        ValueError: If the field set, field types, or values are incompatible
            with the installed full-tiling sampler.
    """
    kernel = _require_mapping(value, location="kernel")
    field_names = _kernel_field_names()
    _require_keys(kernel, field_names, location="kernel")
    settings: dict[str, object] = {}
    for name in field_names:
        value = kernel[name]
        if name == "fixed_coefficient_proposal_sd":
            if not isinstance(value, list):
                raise ValueError("kernel fixed_coefficient_proposal_sd must be a JSON array.")
            settings[name] = tuple(
                _finite_float(
                    item,
                    location=f"kernel.{name}",
                    positive=True,
                )
                for item in value
            )
        elif name in {
            "fixed_k",
            "root_slice_max_steps",
            "root_slice_max_shrink_steps",
        }:
            settings[name] = _integer(
                value,
                location=f"kernel.{name}",
                minimum=1,
            )
        elif name == "pair_allocation_refresh_slots":
            settings[name] = _integer(
                value,
                location=f"kernel.{name}",
            )
        elif name == "root_slice_width":
            settings[name] = _finite_float(
                value,
                location=f"kernel.{name}",
                positive=True,
            )
        else:
            raise ValueError(f"Unsupported resolved kernel metadata field {name!r}.")
    return FullTilingCompoundKernelSettings(**settings)  # type: ignore[arg-type]


def load_full_tiling_checkpoint(
    path: PathLike,
    problem: FullTilingProblem,
    *,
    expected_run_manifest: RunManifest,
) -> FullTilingCompoundCheckpoint:
    """Load and independently validate a full-tiling continuation boundary.

    Args:
        path: Source NPZ archive.
        problem: Equal reconstructed scientific problem for the resumed run.
        expected_run_manifest: Exact caller-held manifest expected in the
            archive.

    Returns:
        Valid checkpoint attached to ``problem`` after independent cache
        audit, with exact persisted immutable caches and PCG64 continuation
        state restored.

    Raises:
        TypeError: If ``problem`` or the manifest has the wrong type.
        ValueError: If archive schema, hashes, problem, manifest, topology,
            coordinates, caches, target terms, RNG, schedule, or resolved
            kernel settings disagree.
        OSError: If the archive cannot be read.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    if not isinstance(expected_run_manifest, Mapping):
        raise TypeError("expected_run_manifest must be a mapping.")
    arrays = _load_archive_arrays(Path(path))
    metadata = _metadata_from_arrays(arrays)
    if (
        metadata["schema_id"] != FULL_TILING_CHECKPOINT_SCHEMA_ID
        or metadata["schema_version"] != FULL_TILING_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("Full-tiling checkpoint schema is incompatible.")
    if not isinstance(metadata["numpy_version"], str) or not metadata["numpy_version"]:
        raise ValueError("Full-tiling checkpoint NumPy provenance is malformed.")
    if metadata["numpy_version"] != np.__version__:
        raise ValueError(
            "Full-tiling checkpoint NumPy version does not match the running "
            "environment required for exact continuation."
        )
    if metadata["schedule_id"] != FULL_TILING_COMPOUND_SCHEDULE_ID:
        raise ValueError("Full-tiling checkpoint schedule is incompatible.")
    problem_sha = full_tiling_problem_fingerprint(problem)
    if metadata["problem_sha256"] != problem_sha:
        raise ValueError("Full-tiling checkpoint problem fingerprint does not match.")

    embedded_manifest_json = metadata["run_manifest_json"]
    embedded_manifest_sha = metadata["run_manifest_sha256"]
    if not isinstance(embedded_manifest_json, str) or not isinstance(
        embedded_manifest_sha,
        str,
    ):
        raise ValueError("Checkpoint run manifest metadata is malformed.")
    if not hmac.compare_digest(
        sha256(embedded_manifest_json.encode("utf-8")).hexdigest(),
        embedded_manifest_sha,
    ):
        raise ValueError("Checkpoint run manifest SHA-256 checksum does not match.")
    expected_manifest_json = canonical_full_tiling_run_manifest(expected_run_manifest)
    if not hmac.compare_digest(
        embedded_manifest_json,
        expected_manifest_json,
    ):
        raise ValueError("Checkpoint run manifest does not match expected canonical content.")

    array_hashes = _require_mapping(
        metadata["array_sha256"],
        location="array_sha256",
    )
    if frozenset(array_hashes) != frozenset(_STATE_ARRAY_NAMES):
        raise ValueError("Checkpoint array_sha256 has an invalid field set.")
    for name in _STATE_ARRAY_NAMES:
        supplied_hash = array_hashes[name]
        if not isinstance(supplied_hash, str) or not hmac.compare_digest(
            supplied_hash,
            _array_sha256(name, arrays[name]),
        ):
            raise ValueError(f"Checkpoint {name} SHA-256 checksum does not match.")

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
    if leaf_masses.shape != (k,):
        raise ValueError("Checkpoint leaf_masses must have shape (K,).")
    observation_shape = problem.observations.shape
    if any(cache.shape != observation_shape for cache in caches.values()):
        raise ValueError("Checkpoint posterior caches must match the observation shape.")
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

    for name, persisted in caches.items():
        if not np.allclose(
            getattr(rebuilt, name),
            persisted,
            rtol=0.0,
            atol=_CACHE_REBUILD_ATOL,
        ):
            raise ValueError(f"Rebuilt state {name} does not match persisted cache.")
    expected_root_total = _finite_float(
        state_metadata["root_total"],
        location="state.root_total",
        positive=True,
    )
    if rebuilt.root_total != expected_root_total:
        raise ValueError("Rebuilt state root_total does not match checkpoint metadata.")
    persisted_logs: dict[str, float] = {}
    for name in _STATE_LOG_FIELDS:
        expected_value = _finite_float(
            state_metadata[name],
            location=f"state.{name}",
        )
        persisted_logs[name] = expected_value
        if not np.isclose(
            getattr(rebuilt, name),
            expected_value,
            rtol=0.0,
            atol=_TARGET_REBUILD_ATOL,
        ):
            raise ValueError(f"Rebuilt state {name} does not match checkpoint metadata.")
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
    rebuilt_state_sha = full_tiling_state_fingerprint(problem, state)
    if not isinstance(supplied_state_sha, str) or not hmac.compare_digest(
        supplied_state_sha,
        rebuilt_state_sha,
    ):
        raise ValueError("Full-tiling checkpoint state fingerprint does not match.")

    settings = _kernel_settings_from_metadata(metadata["kernel"])
    rng_metadata = _require_mapping(metadata["rng"], location="rng")
    _require_keys(
        rng_metadata,
        frozenset(("algorithm", "state", "increment", "has_uint32", "uinteger")),
        location="rng",
    )
    algorithm = rng_metadata["algorithm"]
    if algorithm != "PCG64":
        raise ValueError("rng.algorithm must be exactly 'PCG64'.")
    rng_state = PCG64State(
        state=_integer(
            rng_metadata["state"],
            location="rng.state",
            maximum=(1 << 128) - 1,
        ),
        increment=_integer(
            rng_metadata["increment"],
            location="rng.increment",
            maximum=(1 << 128) - 1,
        ),
        has_uint32=_integer(
            rng_metadata["has_uint32"],
            location="rng.has_uint32",
            maximum=1,
        ),
        uinteger=_integer(
            rng_metadata["uinteger"],
            location="rng.uinteger",
            maximum=(1 << 32) - 1,
        ),
        algorithm=cast(str, algorithm),
    )
    try:
        rng_state.generator()
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint PCG64 state is invalid.") from exc
    return FullTilingCompoundCheckpoint(
        problem=problem,
        state=state,
        rng_state=rng_state,
        transitions_completed=_integer(
            metadata["transitions_completed"],
            location="transitions_completed",
        ),
        schedule_phase=_integer(
            metadata["schedule_phase"],
            location="schedule_phase",
        ),
        kernel_settings=settings,
        schedule_id=cast(str, metadata["schedule_id"]),
    )


__all__ = [
    "FULL_TILING_CHECKPOINT_SCHEMA_ID",
    "FULL_TILING_CHECKPOINT_SCHEMA_VERSION",
    "canonical_full_tiling_run_manifest",
    "full_tiling_problem_fingerprint",
    "full_tiling_state_fingerprint",
    "load_full_tiling_checkpoint",
    "save_full_tiling_checkpoint",
]
