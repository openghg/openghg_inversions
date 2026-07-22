"""Strict durable checkpoints for the experimental RJMCMC sampler.

The format is an atomic compressed NumPy archive containing only numeric
arrays. Metadata is canonical JSON encoded as a one-dimensional ``uint8``
array, with a separate SHA-256 digest. Loading never enables NumPy pickle
support and fails closed when the schema, problem, manifest, random generator,
or cached numerical state does not match.

Exact source-code identity is caller-owned: reproducible runs should embed a
run manifest containing a code revision and supply that manifest again when
loading. Archives are intended as trusted local scientific files; pickle is
disabled, but this first format does not impose an arbitrary file-size cap.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import hmac
import json
from math import isfinite
import os
from pathlib import Path
import tempfile
from typing import Any, TypeAlias, cast
from zipfile import BadZipFile

import numba
import numpy as np
from numpy.typing import NDArray

from openghg_inversions.experimental.rjmcmc.core import (
    Backend,
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
)
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings
from openghg_inversions.experimental.rjmcmc.sampling import (
    FIXED_BLOCK_SCHEDULE_ID,
    LUNT_OPPORTUNITY_MATCHED_SCHEDULE_PROFILE,
    SCHEDULE_ID,
    KernelSettings,
    PCG64State,
    SamplerCheckpoint,
    ScheduleProfile,
    _schedule_id,
)

PathLike: TypeAlias = str | os.PathLike[str]
RunManifest: TypeAlias = Mapping[str, object]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

CHECKPOINT_SCHEMA_ID = "openghg_inversions.experimental.rjmcmc.checkpoint"
CHECKPOINT_SCHEMA_VERSION = 2

_STATE_ARRAY_NAMES = (
    "nuclei",
    "coefficients",
    "labels",
    "design",
    "fixed_coefficients",
    "dynamic_prediction",
    "fixed_prediction",
    "prediction",
    "residual",
)
_ARCHIVE_NAMES = frozenset((*_STATE_ARRAY_NAMES, "metadata", "metadata_sha256"))
_LOG_FIELD_NAMES = (
    "log_likelihood",
    "log_coefficient_prior",
    "log_fixed_coefficient_prior",
    "log_k_prior",
    "log_nucleus_prior",
)
_METADATA_NAMES = frozenset(
    {
        "schema_id",
        "schema_version",
        "numpy_version",
        "numba_version",
        "schedule_id",
        "transitions_completed",
        "rng",
        "kernel",
        "retention",
        "state",
        "state_backend",
        "problem_sha256",
        "run_manifest_json",
        "run_manifest_sha256",
        "array_sha256",
    }
)


def _validate_json_value(value: object, *, location: str) -> None:
    """Reject values outside strict finite JSON before canonical encoding."""
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
    """Return a deterministic strict-JSON representation of ``value``."""
    _validate_json_value(value, location=location)

    def builtins_only(item: object) -> object:
        """Convert generic mappings to containers accepted by ``json.dumps``."""
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


def _json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Checkpoint metadata repeats JSON key {key!r}.")
        result[key] = value
    return result


def _parse_json_object(payload: str, *, location: str) -> dict[str, Any]:
    """Parse one strict JSON object and reject constants and duplicate keys."""

    def reject_constant(value: str) -> None:
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


def _require_keys(value: Mapping[str, object], expected: frozenset[str], *, location: str) -> None:
    """Require exactly the declared mapping keys."""
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"{location} has an invalid field set; missing={missing}, unexpected={unexpected}.")


def _require_mapping(value: object, *, location: str) -> Mapping[str, object]:
    """Return ``value`` as a string-keyed mapping or reject it."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _require_text(value: object, *, location: str) -> str:
    """Return a nonempty string metadata value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a nonempty string.")
    return value


def _require_integer(
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
        description = f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ValueError(f"{location} must be {description}.")
    return value


def _require_positive_float(value: object, *, location: str) -> float:
    """Return a finite positive float metadata value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be finite and positive.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{location} must be finite and positive.")
    return result


def _require_finite_float(value: object, *, location: str) -> float:
    """Return a finite float metadata value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be finite.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{location} must be finite.")
    return result


def _require_optional_positive_float(value: object, *, location: str) -> float | None:
    """Return ``None`` or a finite positive float metadata value."""
    if value is None:
        return None
    return _require_positive_float(value, location=location)


def _require_sha256(value: object, *, location: str) -> str:
    """Return a lowercase hexadecimal SHA-256 digest."""
    text = _require_text(value, location=location)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{location} must be a lowercase hexadecimal SHA-256 digest.")
    return text


def _update_hash_bytes(digest: Any, label: str, payload: bytes) -> None:
    """Add one unambiguously framed byte field to a SHA-256 digest."""
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _update_hash_array(digest: Any, label: str, array: NDArray[Any]) -> None:
    """Add an array's identity, dtype, shape, and C-order contents to a digest."""
    metadata = _canonical_json(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        location=f"{label} array metadata",
    ).encode("utf-8")
    _update_hash_bytes(digest, f"{label}.metadata", metadata)
    _update_hash_bytes(digest, f"{label}.contents", np.ascontiguousarray(array).tobytes(order="C"))


def _array_sha256(name: str, array: NDArray[Any]) -> str:
    """Return a framed digest for one persisted state array."""
    digest = sha256()
    _update_hash_array(digest, name, array)
    return digest.hexdigest()


def _problem_sha256(problem: TransDimensionalProblem) -> str:
    """Fingerprint the exact normalized numerical target consumed by the sampler."""
    digest = sha256()
    scalar_payload = _canonical_json(
        {
            "k_min": problem.k_min,
            "k_max": problem.k_max,
            "coefficient_prior_mean": problem.coefficient_prior_mean,
            "coefficient_prior_sd": problem.coefficient_prior_sd,
            "has_fixed_block": problem.fixed_block is not None,
        },
        location="problem scalars",
    ).encode("utf-8")
    _update_hash_bytes(digest, "problem.scalars", scalar_payload)
    for name in (
        "observations",
        "observation_sd",
        "sensitivities",
        "grid_coordinates",
        "log_k_prior",
        "fixed_offset",
    ):
        _update_hash_array(digest, f"problem.{name}", getattr(problem, name))
    if problem.fixed_block is not None:
        for name in ("design", "coefficient_prior_mean", "coefficient_prior_sd"):
            _update_hash_array(
                digest,
                f"problem.fixed_block.{name}",
                getattr(problem.fixed_block, name),
            )
    return digest.hexdigest()


def _expected_schedule(
    problem: TransDimensionalProblem,
    schedule_profile: ScheduleProfile,
) -> str:
    """Return the versioned transition schedule implied by the target/profile."""
    return _schedule_id(problem, schedule_profile)


def _validate_kernel(kernel: KernelSettings) -> None:
    """Validate settings whose frozen dataclass has no runtime checks."""
    if not isinstance(kernel, KernelSettings):
        raise TypeError("checkpoint.kernel_settings must be a KernelSettings instance.")
    _require_positive_float(kernel.coefficient_proposal_sd, location="kernel.coefficient_proposal_sd")
    _require_positive_float(kernel.birth_proposal_sd, location="kernel.birth_proposal_sd")
    _require_optional_positive_float(
        kernel.fixed_coefficient_proposal_sd,
        location="kernel.fixed_coefficient_proposal_sd",
    )
    if kernel.backend not in ("numpy", "numba"):
        raise ValueError("kernel.backend must be 'numpy' or 'numba'.")
    if kernel.nucleus_move not in ("global", "local"):
        raise ValueError("kernel.nucleus_move must be 'global' or 'local'.")
    local_scale = _require_optional_positive_float(
        kernel.local_move_scale, location="kernel.local_move_scale"
    )
    if kernel.nucleus_move == "local" and local_scale is None:
        raise ValueError("kernel.local_move_scale is required for local nucleus moves.")
    if kernel.schedule_profile not in ("default", LUNT_OPPORTUNITY_MATCHED_SCHEDULE_PROFILE):
        raise ValueError("kernel.schedule_profile is unsupported.")


def _validate_rng(rng_state: PCG64State) -> None:
    """Validate exact PCG64 state ranges and NumPy acceptance."""
    if not isinstance(rng_state, PCG64State):
        raise TypeError("checkpoint.rng_state must be a PCG64State instance.")
    if rng_state.algorithm != "PCG64":
        raise ValueError(f"Unsupported checkpoint RNG algorithm {rng_state.algorithm!r}.")
    _require_integer(rng_state.state, location="rng.state", maximum=2**128 - 1)
    _require_integer(rng_state.increment, location="rng.increment", maximum=2**128 - 1)
    _require_integer(rng_state.has_uint32, location="rng.has_uint32", maximum=1)
    _require_integer(rng_state.uinteger, location="rng.uinteger", maximum=2**32 - 1)
    try:
        rng_state.generator()
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint contains an invalid PCG64 state.") from exc


def _validate_state_against_rebuild(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    preferred_backend: Backend,
) -> Backend:
    """Return the backend that exactly rebuilds every persisted state cache."""
    if not isinstance(state, TransDimensionalState):
        raise TypeError("checkpoint.state must be a TransDimensionalState instance.")
    fixed_coefficients = state.fixed_coefficients if problem.n_fixed_coefficients else None
    alternative_backend: Backend = "numba" if preferred_backend == "numpy" else "numpy"
    mismatch_fields: dict[Backend, list[str]] = {}
    for backend in (preferred_backend, alternative_backend):
        try:
            rebuilt = build_state(
                problem,
                state.nuclei[: state.k],
                state.coefficients[: state.k],
                fixed_coefficients=fixed_coefficients,
                backend=backend,
            )
        except (TypeError, ValueError):
            mismatch_fields[backend] = ["state reconstruction"]
            continue

        mismatches = [
            name
            for name in _STATE_ARRAY_NAMES
            if not np.array_equal(getattr(state, name), getattr(rebuilt, name))
        ]
        if state.k != rebuilt.k:
            mismatches.append("k")
        mismatches.extend(name for name in _LOG_FIELD_NAMES if getattr(state, name) != getattr(rebuilt, name))
        if not mismatches:
            return backend
        mismatch_fields[backend] = mismatches
    raise ValueError(
        "Checkpoint cached state does not match an independent rebuild with either backend; "
        f"mismatches={mismatch_fields}."
    )


def _validate_checkpoint(checkpoint: SamplerCheckpoint) -> Backend:
    """Validate a checkpoint and return its exact state-cache backend."""
    if not isinstance(checkpoint, SamplerCheckpoint):
        raise TypeError("checkpoint must be a SamplerCheckpoint instance.")
    if not isinstance(checkpoint.problem, TransDimensionalProblem):
        raise TypeError("checkpoint.problem must be a TransDimensionalProblem instance.")
    transitions = checkpoint.transitions_completed
    if isinstance(transitions, bool) or not isinstance(transitions, (int, np.integer)) or transitions < 0:
        raise ValueError("checkpoint.transitions_completed must be a non-negative integer.")
    _validate_kernel(checkpoint.kernel_settings)
    expected_schedule = _expected_schedule(
        checkpoint.problem,
        checkpoint.kernel_settings.schedule_profile,
    )
    if checkpoint.schedule_id != expected_schedule:
        raise ValueError(
            f"Checkpoint schedule {checkpoint.schedule_id!r} is incompatible with its problem/profile; "
            f"expected {expected_schedule!r}."
        )
    if not isinstance(checkpoint.retention, RetentionSettings):
        raise TypeError("checkpoint.retention must be a RetentionSettings instance.")
    _validate_rng(checkpoint.rng_state)
    return _validate_state_against_rebuild(
        checkpoint.problem,
        checkpoint.state,
        preferred_backend=checkpoint.kernel_settings.backend,
    )


def _state_arrays(state: TransDimensionalState) -> dict[str, NDArray[Any]]:
    """Return the complete persisted array cache for ``state``."""
    return {name: np.asarray(getattr(state, name)) for name in _STATE_ARRAY_NAMES}


def _metadata(
    checkpoint: SamplerCheckpoint,
    arrays: Mapping[str, NDArray[Any]],
    *,
    run_manifest: RunManifest | None,
    state_backend: Backend,
) -> dict[str, object]:
    """Build canonical-JSON-compatible metadata for a validated checkpoint."""
    manifest_json = None
    manifest_sha256 = None
    if run_manifest is not None:
        if not isinstance(run_manifest, Mapping):
            raise TypeError("run_manifest must be a mapping or None.")
        manifest_json = _canonical_json(run_manifest, location="run_manifest")
        manifest_sha256 = sha256(manifest_json.encode("utf-8")).hexdigest()

    state = checkpoint.state
    kernel = checkpoint.kernel_settings
    rng = checkpoint.rng_state
    return {
        "schema_id": CHECKPOINT_SCHEMA_ID,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
        "schedule_id": checkpoint.schedule_id,
        "transitions_completed": int(checkpoint.transitions_completed),
        "rng": {
            "algorithm": rng.algorithm,
            "state": rng.state,
            "increment": rng.increment,
            "has_uint32": rng.has_uint32,
            "uinteger": rng.uinteger,
        },
        "kernel": {
            "coefficient_proposal_sd": kernel.coefficient_proposal_sd,
            "birth_proposal_sd": kernel.birth_proposal_sd,
            "fixed_coefficient_proposal_sd": kernel.fixed_coefficient_proposal_sd,
            "backend": kernel.backend,
            "nucleus_move": kernel.nucleus_move,
            "local_move_scale": kernel.local_move_scale,
            "schedule_profile": kernel.schedule_profile,
        },
        "retention": {
            "warmup_transitions": checkpoint.retention.warmup_transitions,
            "thin": checkpoint.retention.thin,
        },
        "state": {
            "k": state.k,
            **{name: getattr(state, name) for name in _LOG_FIELD_NAMES},
        },
        "state_backend": state_backend,
        "problem_sha256": _problem_sha256(checkpoint.problem),
        "run_manifest_json": manifest_json,
        "run_manifest_sha256": manifest_sha256,
        "array_sha256": {name: _array_sha256(name, array) for name, array in arrays.items()},
    }


def save_checkpoint(
    path: PathLike,
    checkpoint: SamplerCheckpoint,
    *,
    run_manifest: RunManifest | None = None,
) -> None:
    """Atomically save one strict durable sampler checkpoint.

    Args:
        path: Destination ``.npz`` path. Its parent directory must exist.
        checkpoint: Valid in-memory checkpoint to persist.
        run_manifest: Optional JSON-compatible run manifest embedded in
            canonical form and protected by its own SHA-256 digest.

    Raises:
        TypeError: If checkpoint components or the manifest have wrong types.
        ValueError: If the checkpoint is inconsistent or the manifest is not
            strict finite JSON.
        OSError: If the temporary archive cannot be written or atomically
            replaced. A pre-existing destination is left unchanged.
    """
    state_backend = _validate_checkpoint(checkpoint)
    arrays = _state_arrays(checkpoint.state)
    metadata = _metadata(
        checkpoint,
        arrays,
        run_manifest=run_manifest,
        state_backend=state_backend,
    )
    metadata_bytes = _canonical_json(metadata, location="checkpoint metadata").encode("utf-8")
    metadata_digest = sha256(metadata_bytes).hexdigest().encode("ascii")
    archive_arrays = {
        **arrays,
        "metadata": np.frombuffer(metadata_bytes, dtype=np.uint8),
        "metadata_sha256": np.frombuffer(metadata_digest, dtype=np.uint8),
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
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_archive_arrays(path: Path) -> dict[str, NDArray[Any]]:
    """Read the exact numeric archive field set with pickle support disabled."""
    try:
        archive = np.load(path, allow_pickle=False)
    except (FileNotFoundError, PermissionError):
        raise
    except (BadZipFile, EOFError, ValueError) as exc:
        raise ValueError("Checkpoint archive is corrupt or unreadable.") from exc
    if isinstance(archive, np.ndarray):
        raise ValueError("Checkpoint path must contain a compressed NPZ archive, not a single NPY array.")
    try:
        with archive:
            names = frozenset(archive.files)
            if names != _ARCHIVE_NAMES:
                missing = sorted(_ARCHIVE_NAMES - names)
                unexpected = sorted(names - _ARCHIVE_NAMES)
                raise ValueError(
                    "Checkpoint archive has an invalid array set; "
                    f"missing={missing}, unexpected={unexpected}."
                )
            try:
                return {name: np.array(archive[name], copy=True) for name in _ARCHIVE_NAMES}
            except ValueError as exc:
                raise ValueError(
                    "Checkpoint archive arrays could not be read safely; "
                    "object arrays and pickle payloads are not permitted."
                ) from exc
    except ValueError:
        raise
    except (BadZipFile, EOFError) as exc:
        raise ValueError("Checkpoint archive is corrupt or unreadable.") from exc


def _metadata_from_arrays(arrays: Mapping[str, NDArray[Any]]) -> dict[str, Any]:
    """Authenticate, decode, and structurally validate checkpoint metadata."""
    metadata_bytes_array = arrays["metadata"]
    metadata_digest_array = arrays["metadata_sha256"]
    if metadata_bytes_array.dtype != np.dtype(np.uint8) or metadata_bytes_array.ndim != 1:
        raise ValueError("Checkpoint metadata must be a one-dimensional uint8 array.")
    if metadata_digest_array.dtype != np.dtype(np.uint8) or metadata_digest_array.shape != (64,):
        raise ValueError("Checkpoint metadata_sha256 must be a 64-byte uint8 array.")
    metadata_bytes = metadata_bytes_array.tobytes()
    try:
        supplied_digest = metadata_digest_array.tobytes().decode("ascii")
        metadata_json = metadata_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Checkpoint metadata is not valid UTF-8/ASCII.") from exc
    expected_digest = sha256(metadata_bytes).hexdigest()
    if not hmac.compare_digest(supplied_digest, expected_digest):
        raise ValueError("Checkpoint metadata SHA-256 checksum does not match.")
    metadata = _parse_json_object(metadata_json, location="checkpoint metadata")
    _require_keys(metadata, _METADATA_NAMES, location="checkpoint metadata")
    return metadata


def _kernel_from_metadata(value: object, *, schema_version: int) -> KernelSettings:
    """Validate and reconstruct immutable kernel settings."""
    kernel = _require_mapping(value, location="checkpoint metadata.kernel")
    expected_fields = {
        "coefficient_proposal_sd",
        "birth_proposal_sd",
        "fixed_coefficient_proposal_sd",
        "backend",
        "nucleus_move",
        "local_move_scale",
    }
    if schema_version >= 2:
        expected_fields.add("schedule_profile")
    _require_keys(
        kernel,
        frozenset(expected_fields),
        location="checkpoint metadata.kernel",
    )
    backend = _require_text(kernel["backend"], location="checkpoint metadata.kernel.backend")
    nucleus_move = _require_text(
        kernel["nucleus_move"],
        location="checkpoint metadata.kernel.nucleus_move",
    )
    schedule_profile = (
        "default"
        if schema_version == 1
        else _require_text(
            kernel["schedule_profile"],
            location="checkpoint metadata.kernel.schedule_profile",
        )
    )
    settings = KernelSettings(
        coefficient_proposal_sd=_require_positive_float(
            kernel["coefficient_proposal_sd"],
            location="checkpoint metadata.kernel.coefficient_proposal_sd",
        ),
        birth_proposal_sd=_require_positive_float(
            kernel["birth_proposal_sd"],
            location="checkpoint metadata.kernel.birth_proposal_sd",
        ),
        fixed_coefficient_proposal_sd=_require_optional_positive_float(
            kernel["fixed_coefficient_proposal_sd"],
            location="checkpoint metadata.kernel.fixed_coefficient_proposal_sd",
        ),
        backend=cast(Any, backend),
        nucleus_move=cast(Any, nucleus_move),
        local_move_scale=_require_optional_positive_float(
            kernel["local_move_scale"],
            location="checkpoint metadata.kernel.local_move_scale",
        ),
        schedule_profile=cast(Any, schedule_profile),
    )
    _validate_kernel(settings)
    return settings


def _rng_from_metadata(value: object) -> PCG64State:
    """Validate and reconstruct an exact PCG64 state."""
    rng = _require_mapping(value, location="checkpoint metadata.rng")
    _require_keys(
        rng,
        frozenset({"algorithm", "state", "increment", "has_uint32", "uinteger"}),
        location="checkpoint metadata.rng",
    )
    state = PCG64State(
        algorithm=_require_text(rng["algorithm"], location="checkpoint metadata.rng.algorithm"),
        state=_require_integer(
            rng["state"],
            location="checkpoint metadata.rng.state",
            maximum=2**128 - 1,
        ),
        increment=_require_integer(
            rng["increment"],
            location="checkpoint metadata.rng.increment",
            maximum=2**128 - 1,
        ),
        has_uint32=_require_integer(
            rng["has_uint32"],
            location="checkpoint metadata.rng.has_uint32",
            maximum=1,
        ),
        uinteger=_require_integer(
            rng["uinteger"],
            location="checkpoint metadata.rng.uinteger",
            maximum=2**32 - 1,
        ),
    )
    _validate_rng(state)
    return state


def _retention_from_metadata(value: object) -> RetentionSettings:
    """Validate and reconstruct collection-time retention settings."""
    retention = _require_mapping(value, location="checkpoint metadata.retention")
    _require_keys(
        retention,
        frozenset({"warmup_transitions", "thin"}),
        location="checkpoint metadata.retention",
    )
    return RetentionSettings(
        warmup_transitions=_require_integer(
            retention["warmup_transitions"],
            location="checkpoint metadata.retention.warmup_transitions",
        ),
        thin=_require_integer(
            retention["thin"],
            location="checkpoint metadata.retention.thin",
            minimum=1,
        ),
    )


def _validated_state_arrays(
    arrays: Mapping[str, NDArray[Any]],
    metadata: Mapping[str, object],
    problem: TransDimensionalProblem,
) -> dict[str, NDArray[Any]]:
    """Validate state array checksums, exact dtypes, and problem-derived shapes."""
    digests = _require_mapping(metadata["array_sha256"], location="checkpoint metadata.array_sha256")
    _require_keys(digests, frozenset(_STATE_ARRAY_NAMES), location="checkpoint metadata.array_sha256")
    expected: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
        "nuclei": (np.dtype(np.int64), (problem.k_max,)),
        "coefficients": (np.dtype(np.float64), (problem.k_max,)),
        "labels": (np.dtype(np.int64), (problem.n_grid_cells,)),
        "design": (np.dtype(np.float64), (problem.n_observations, problem.k_max)),
        "fixed_coefficients": (np.dtype(np.float64), (problem.n_fixed_coefficients,)),
        "dynamic_prediction": (np.dtype(np.float64), (problem.n_observations,)),
        "fixed_prediction": (np.dtype(np.float64), (problem.n_observations,)),
        "prediction": (np.dtype(np.float64), (problem.n_observations,)),
        "residual": (np.dtype(np.float64), (problem.n_observations,)),
    }
    result: dict[str, NDArray[Any]] = {}
    for name, (dtype, shape) in expected.items():
        array = arrays[name]
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(
                f"Checkpoint state array {name!r} must have dtype {dtype} and shape {shape}; "
                f"received dtype {array.dtype} and shape {array.shape}."
            )
        supplied_digest = _require_sha256(
            digests[name],
            location=f"checkpoint metadata.array_sha256.{name}",
        )
        if not hmac.compare_digest(supplied_digest, _array_sha256(name, array)):
            raise ValueError(f"Checkpoint state array {name!r} SHA-256 checksum does not match.")
        owned = np.array(array, dtype=dtype, copy=True)
        owned.setflags(write=False)
        result[name] = owned
    return result


def _state_from_metadata(
    value: object,
    arrays: Mapping[str, NDArray[Any]],
) -> TransDimensionalState:
    """Construct the exact stored state after scalar metadata validation."""
    state = _require_mapping(value, location="checkpoint metadata.state")
    expected_names = frozenset(("k", *_LOG_FIELD_NAMES))
    _require_keys(state, expected_names, location="checkpoint metadata.state")
    return TransDimensionalState(
        k=_require_integer(state["k"], location="checkpoint metadata.state.k", minimum=1),
        nuclei=cast(IntArray, arrays["nuclei"]),
        coefficients=cast(FloatArray, arrays["coefficients"]),
        labels=cast(IntArray, arrays["labels"]),
        design=cast(FloatArray, arrays["design"]),
        fixed_coefficients=cast(FloatArray, arrays["fixed_coefficients"]),
        dynamic_prediction=cast(FloatArray, arrays["dynamic_prediction"]),
        fixed_prediction=cast(FloatArray, arrays["fixed_prediction"]),
        prediction=cast(FloatArray, arrays["prediction"]),
        residual=cast(FloatArray, arrays["residual"]),
        log_likelihood=_require_finite_float(
            state["log_likelihood"],
            location="checkpoint metadata.state.log_likelihood",
        ),
        log_coefficient_prior=_require_finite_float(
            state["log_coefficient_prior"],
            location="checkpoint metadata.state.log_coefficient_prior",
        ),
        log_fixed_coefficient_prior=_require_finite_float(
            state["log_fixed_coefficient_prior"],
            location="checkpoint metadata.state.log_fixed_coefficient_prior",
        ),
        log_k_prior=_require_finite_float(
            state["log_k_prior"],
            location="checkpoint metadata.state.log_k_prior",
        ),
        log_nucleus_prior=_require_finite_float(
            state["log_nucleus_prior"],
            location="checkpoint metadata.state.log_nucleus_prior",
        ),
    )


def _validate_manifest(
    metadata: Mapping[str, object],
    *,
    expected_run_manifest: RunManifest | None,
) -> None:
    """Validate the embedded manifest hash and an optional caller expectation."""
    manifest_json = metadata["run_manifest_json"]
    manifest_sha256 = metadata["run_manifest_sha256"]
    if manifest_json is None or manifest_sha256 is None:
        if manifest_json is not None or manifest_sha256 is not None:
            raise ValueError(
                "Checkpoint run manifest and SHA-256 must either both be present or both be null."
            )
        if expected_run_manifest is not None:
            raise ValueError("Checkpoint does not contain the expected run manifest.")
        return
    manifest_text = _require_text(manifest_json, location="checkpoint metadata.run_manifest_json")
    supplied_digest = _require_sha256(
        manifest_sha256,
        location="checkpoint metadata.run_manifest_sha256",
    )
    parsed_manifest = _parse_json_object(manifest_text, location="checkpoint run manifest")
    canonical_manifest = _canonical_json(parsed_manifest, location="checkpoint run manifest")
    if manifest_text != canonical_manifest:
        raise ValueError("Checkpoint run manifest is not in canonical JSON form.")
    actual_digest = sha256(manifest_text.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(supplied_digest, actual_digest):
        raise ValueError("Checkpoint run manifest SHA-256 checksum does not match.")
    if expected_run_manifest is not None:
        if not isinstance(expected_run_manifest, Mapping):
            raise TypeError("expected_run_manifest must be a mapping or None.")
        expected_json = _canonical_json(expected_run_manifest, location="expected_run_manifest")
        if (
            not hmac.compare_digest(
                sha256(expected_json.encode("utf-8")).hexdigest(),
                actual_digest,
            )
            or expected_json != manifest_text
        ):
            raise ValueError("Checkpoint run manifest does not match the expected run manifest.")


def load_checkpoint(
    path: PathLike,
    problem: TransDimensionalProblem,
    *,
    expected_run_manifest: RunManifest | None = None,
) -> SamplerCheckpoint:
    """Load and strictly validate a durable sampler checkpoint.

    The returned checkpoint is bound to the supplied ``problem`` object, so it
    can be passed directly to :func:`~openghg_inversions.experimental.rjmcmc.sampling.continue_sample`.
    The exact stored cache values are restored only after an independent
    :func:`~openghg_inversions.experimental.rjmcmc.core.build_state` produces
    identical fields.

    Args:
        path: Existing checkpoint archive.
        problem: Fully transformed numerical problem expected by the chain.
        expected_run_manifest: Optional manifest that must exactly match the
            embedded canonical manifest.

    Returns:
        A validated exact continuation checkpoint bound to ``problem``.

    Raises:
        TypeError: If ``problem`` or the expected manifest has a wrong type.
        ValueError: If any archive, schema, checksum, problem, manifest,
            schedule, RNG, or state validation fails.
        OSError: If the path cannot be accessed for reasons other than archive
            corruption.
    """
    if not isinstance(problem, TransDimensionalProblem):
        raise TypeError("problem must be a TransDimensionalProblem instance.")
    archive = _load_archive_arrays(Path(path))
    metadata = _metadata_from_arrays(archive)

    if metadata["schema_id"] != CHECKPOINT_SCHEMA_ID:
        raise ValueError(f"Unsupported checkpoint schema {metadata['schema_id']!r}.")
    version = _require_integer(
        metadata["schema_version"],
        location="checkpoint metadata.schema_version",
        minimum=1,
    )
    if version not in (1, CHECKPOINT_SCHEMA_VERSION):
        raise ValueError(f"Unsupported checkpoint schema version {version}.")
    saved_numpy_version = _require_text(
        metadata["numpy_version"],
        location="checkpoint metadata.numpy_version",
    )
    if saved_numpy_version != np.__version__:
        raise ValueError(
            f"Checkpoint NumPy version {saved_numpy_version!r} is incompatible with "
            f"the current exact-replay environment {np.__version__!r}."
        )
    saved_numba_version = _require_text(
        metadata["numba_version"],
        location="checkpoint metadata.numba_version",
    )
    supplied_problem_sha256 = _require_sha256(
        metadata["problem_sha256"],
        location="checkpoint metadata.problem_sha256",
    )
    if not hmac.compare_digest(supplied_problem_sha256, _problem_sha256(problem)):
        raise ValueError("Checkpoint numerical problem fingerprint does not match the supplied problem.")
    _validate_manifest(metadata, expected_run_manifest=expected_run_manifest)

    schedule_id = _require_text(metadata["schedule_id"], location="checkpoint metadata.schedule_id")
    if version == 1 and schedule_id not in (SCHEDULE_ID, FIXED_BLOCK_SCHEDULE_ID):
        raise ValueError(f"Checkpoint schema version 1 cannot use schedule {schedule_id!r}.")
    kernel = _kernel_from_metadata(metadata["kernel"], schema_version=version)
    expected_schedule = _expected_schedule(problem, kernel.schedule_profile)
    if schedule_id != expected_schedule:
        raise ValueError(
            f"Checkpoint schedule {schedule_id!r} is incompatible with the supplied problem/profile; "
            f"expected {expected_schedule!r}."
        )
    transitions_completed = _require_integer(
        metadata["transitions_completed"],
        location="checkpoint metadata.transitions_completed",
    )
    if kernel.backend == "numba" and saved_numba_version != numba.__version__:
        raise ValueError(
            f"Checkpoint Numba version {saved_numba_version!r} is incompatible with "
            f"the current exact-replay environment {numba.__version__!r}."
        )
    rng_state = _rng_from_metadata(metadata["rng"])
    retention = _retention_from_metadata(metadata["retention"])
    state_arrays = _validated_state_arrays(archive, metadata, problem)
    state = _state_from_metadata(metadata["state"], state_arrays)
    state_backend = _require_text(
        metadata["state_backend"],
        location="checkpoint metadata.state_backend",
    )
    if state_backend not in ("numpy", "numba"):
        raise ValueError("checkpoint metadata.state_backend must be 'numpy' or 'numba'.")
    rebuilt_backend = _validate_state_against_rebuild(
        problem,
        state,
        preferred_backend=cast(Backend, state_backend),
    )
    if rebuilt_backend != state_backend:
        raise ValueError(f"Checkpoint state does not match its declared cache backend {state_backend!r}.")

    return SamplerCheckpoint(
        problem=problem,
        state=state,
        rng_state=rng_state,
        transitions_completed=transitions_completed,
        kernel_settings=kernel,
        retention=retention,
        schedule_id=schedule_id,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_ID",
    "CHECKPOINT_SCHEMA_VERSION",
    "load_checkpoint",
    "save_checkpoint",
]
