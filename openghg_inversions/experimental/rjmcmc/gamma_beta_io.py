"""Durable manifests, checkpoints, and xarray output for Gamma--Beta RJMCMC.

This module builds canonical run manifests and scientific problem
fingerprints, atomically saves and validates resumable NPZ checkpoints, and
converts compound traces to padded xarray datasets. The checkpoint stores only
irreducible state coordinates; loading rebuilds every prediction and target
cache through :func:`build_gamma_beta_tree_state`.

Stored SHA-256 digests provide integrity checks, not proof of authenticity
against a malicious archive writer. Loading never enables NumPy pickle
support. Variable-length trace values use explicit padding and Boolean masks.
"""

from __future__ import annotations

from collections.abc import Mapping
import errno
from hashlib import sha256
import hmac
import json
from math import isfinite
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Any, TypeAlias, cast
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from .dyadic_tree import DyadicFrontier
from .gamma_beta_compound_sampling import (
    GAMMA_BETA_COMPOUND_SCHEDULE_ID,
    GammaBetaCompoundCheckpoint,
    GammaBetaCompoundKernelSettings,
    GammaBetaCompoundTrace,
)
from .gamma_beta_tree import (
    GammaBetaTreeProblem,
    GammaBetaTreeState,
    build_gamma_beta_tree_state,
)
from .retention import RetentionSettings
from .sampling import PCG64State

PathLike: TypeAlias = str | os.PathLike[str]
RunManifest: TypeAlias = Mapping[str, object]

GAMMA_BETA_CHECKPOINT_SCHEMA_ID = "openghg_inversions.experimental.rjmcmc.gamma_beta_checkpoint"
GAMMA_BETA_CHECKPOINT_SCHEMA_VERSION = 2
GAMMA_BETA_MANIFEST_SCHEMA_ID = "openghg_inversions.experimental.rjmcmc.gamma_beta_run"
GAMMA_BETA_MANIFEST_SCHEMA_VERSION = 3
GAMMA_BETA_TRACE_SCHEMA_ID = "openghg_inversions.experimental.rjmcmc.gamma_beta_trace"
GAMMA_BETA_TRACE_SCHEMA_VERSION = 2

_STATE_ARRAY_NAMES = ("frontier_node_ids", "active_fractions", "fixed_coefficients")
_ARCHIVE_NAMES = frozenset((*_STATE_ARRAY_NAMES, "metadata", "metadata_sha256"))
_MANIFEST_NAMES = frozenset(
    {
        "schema_id",
        "schema_version",
        "code_revision",
        "inputs",
        "nominal_weight",
        "problem_sha256",
        "tree",
        "k_prior",
        "likelihood",
        "gamma_beta_prior",
        "fixed_block",
        "schedule",
        "retention",
        "seed",
        "chain",
    }
)
_METADATA_NAMES = frozenset(
    {
        "schema_id",
        "schema_version",
        "numpy_version",
        "schedule_id",
        "problem_sha256",
        "transitions_completed",
        "schedule_phase",
        "rng",
        "kernel",
        "retention",
        "state",
        "run_manifest_json",
        "run_manifest_sha256",
        "array_sha256",
    }
)
_STATE_LOG_FIELDS = (
    "log_gaussian_likelihood",
    "log_likelihood",
    "log_root_prior",
    "log_fraction_prior",
    "log_partition_prior",
    "log_fixed_coefficient_prior",
    "log_target",
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
    """Return deterministic strict JSON for supported built-in containers.

    Args:
        value: Candidate finite JSON-compatible value.
        location: Field path used in validation errors.

    Returns:
        Canonical JSON string.

    Raises:
        ValueError: If ``value`` contains unsupported or non-finite content.
    """
    _validate_json_value(value, location=location)

    def builtins_only(item: object) -> object:
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


def canonical_gamma_beta_run_manifest(run_manifest: RunManifest) -> str:
    """Return the deterministic strict-JSON representation of a run manifest.

    Args:
        run_manifest: Manifest produced by
            :func:`build_gamma_beta_run_manifest`.

    Returns:
        Canonical JSON string suitable for UTF-8 encoding, with sorted keys
        and no insignificant whitespace.

    Raises:
        TypeError: If ``run_manifest`` is not a mapping.
        ValueError: If it contains a non-JSON or non-finite value.
    """
    if not isinstance(run_manifest, Mapping):
        raise TypeError("run_manifest must be a mapping.")
    return _canonical_json(run_manifest, location="run_manifest")


def _update_hash_bytes(digest: Any, label: str, payload: bytes) -> None:
    """Mutate a digest with one length-delimited labelled byte string.

    Args:
        digest: Hash object supporting ``update``.
        label: Stable field label.
        payload: Exact bytes to add.
    """
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "little"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _update_hash_array(digest: Any, label: str, values: object) -> None:
    """Mutate a digest with one array's exact C-order layout and values.

    Args:
        digest: Hash object supporting ``update``.
        label: Stable array field label.
        values: Values convertible to a NumPy array.
    """
    array = np.ascontiguousarray(np.asarray(values))
    descriptor = _canonical_json(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        location=f"{label} descriptor",
    )
    _update_hash_bytes(digest, f"{label}.descriptor", descriptor.encode("utf-8"))
    _update_hash_bytes(digest, f"{label}.data", array.tobytes(order="C"))


def _array_sha256(label: str, values: object) -> str:
    """Return the labelled exact-layout SHA-256 for one array."""
    digest = sha256()
    _update_hash_array(digest, label, values)
    return digest.hexdigest()


def gamma_beta_problem_fingerprint(problem: GammaBetaTreeProblem) -> str:
    """Return a deterministic SHA-256 identity for a Gamma--Beta problem.

    The identity covers observations, errors, response-per-unit-mass
    sensitivity, fixed offset and block, tree topology, both priors, and the
    likelihood power. Derived node-design caches are intentionally omitted
    because they are rebuilt deterministically from those inputs.

    Args:
        problem: Immutable Gamma--Beta scientific problem.

    Returns:
        Lower-case 64-character SHA-256 digest.

    Raises:
        TypeError: If ``problem`` has the wrong type.
    """
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be a GammaBetaTreeProblem.")
    digest = sha256()
    _update_hash_bytes(digest, "fingerprint_schema", b"gamma_beta_problem_v1")
    tree = problem.tree
    topology = np.asarray(
        [
            (
                node.node_id,
                node.row_start,
                node.row_stop,
                node.col_start,
                node.col_stop,
                node.depth,
                -1 if node.parent_id is None else node.parent_id,
                -1 if not node.child_ids else node.child_ids[0],
                -1 if not node.child_ids else node.child_ids[1],
            )
            for node in tree.nodes
        ],
        dtype=np.int64,
    )
    _update_hash_array(digest, "tree.shape", np.asarray(tree.shape, dtype=np.int64))
    _update_hash_array(digest, "tree.topology", topology)
    _update_hash_array(digest, "tree.leaf_ids", np.asarray(tree.leaf_ids, dtype=np.int64))
    _update_hash_array(
        digest,
        "tree.internal_node_ids",
        np.asarray(tree.internal_node_ids, dtype=np.int64),
    )
    for label, values in (
        ("observations", problem.observations),
        ("observation_sd", problem.observation_sd),
        ("sensitivity", problem.sensitivity),
        ("fixed_offset", problem.fixed_offset),
        ("prior.nominal_cell_mass", problem.prior.nominal_cell_mass),
        ("prior.beta_shape_by_node", problem.prior.beta_shape_by_node),
        (
            "partition_prior.marginal_probability_by_k",
            problem.partition_prior.marginal_probability_by_k,
        ),
    ):
        _update_hash_array(digest, label, values)
    scalars = {
        "likelihood_power": problem.likelihood_power,
        "root_shape": problem.prior.root_shape,
        "root_rate": problem.prior.root_rate,
        "fixed_block_present": problem.fixed_block is not None,
    }
    _update_hash_bytes(
        digest,
        "scalars",
        _canonical_json(scalars, location="problem scalars").encode("utf-8"),
    )
    if problem.fixed_block is not None:
        _update_hash_array(digest, "fixed_block.design", problem.fixed_block.design)
        _update_hash_array(
            digest,
            "fixed_block.coefficient_prior_mean",
            problem.fixed_block.coefficient_prior_mean,
        )
        _update_hash_array(
            digest,
            "fixed_block.coefficient_prior_sd",
            problem.fixed_block.coefficient_prior_sd,
        )
    return digest.hexdigest()


def _rebuild_and_validate_state(
    problem: GammaBetaTreeProblem,
    state: object,
    *,
    location: str,
) -> GammaBetaTreeState:
    """Rebuild and exactly validate one state's coordinates and caches."""
    if not isinstance(state, GammaBetaTreeState):
        raise TypeError(f"{location} must be a GammaBetaTreeState.")
    if state.problem is not problem:
        raise ValueError(f"{location} must belong to the supplied problem instance.")
    rebuilt = build_gamma_beta_tree_state(
        problem,
        frontier=state.frontier,
        root_total=state.root_total,
        active_fractions=state.active_fractions,
        fixed_coefficients=state.fixed_coefficients,
    )
    if not isfinite(rebuilt.log_target):
        raise ValueError(f"{location} must have finite target support.")
    if rebuilt.frontier != state.frontier or rebuilt.root_total != state.root_total:
        raise ValueError(f"{location} has inconsistent irreducible coordinates.")
    array_fields = (
        "active_fractions",
        "active_node_masses",
        "fixed_coefficients",
        "dynamic_prediction",
        "fixed_prediction",
        "prediction",
        "residual",
    )
    if any(not np.array_equal(getattr(rebuilt, name), getattr(state, name)) for name in array_fields):
        raise ValueError(f"{location} contains stale or inconsistent cached arrays.")
    if any(getattr(rebuilt, name) != getattr(state, name) for name in _STATE_LOG_FIELDS):
        raise ValueError(f"{location} contains stale or inconsistent target components.")
    return rebuilt


def gamma_beta_state_fingerprint(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
) -> str:
    """Return a canonical SHA-256 identity for irreducible state coordinates.

    The identity covers frontier node IDs, root total, active fractions, and
    fixed coefficients. It intentionally excludes derived caches and
    caller-owned labels or physical coordinate semantics. The state is fully
    rebuilt first, so a directly constructed or replaced stale state cannot
    acquire a valid identity.

    Args:
        problem: Exact problem instance to which ``state`` must belong.
        state: State whose irreducible coordinates define the identity.

    Returns:
        Lower-case 64-character SHA-256 digest.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If the state belongs to another problem or has stale
            coordinates, caches, or target terms.
    """
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be a GammaBetaTreeProblem.")
    validated = _rebuild_and_validate_state(problem, state, location="state")
    digest = sha256()
    _update_hash_bytes(digest, "fingerprint_schema", b"gamma_beta_state_coordinates_v1")
    _update_hash_array(
        digest,
        "frontier_node_ids",
        np.asarray(validated.frontier.node_ids, dtype=np.int64),
    )
    _update_hash_array(
        digest,
        "root_total",
        np.asarray(validated.root_total, dtype=np.float64),
    )
    _update_hash_array(digest, "active_fractions", validated.active_fractions)
    _update_hash_array(digest, "fixed_coefficients", validated.fixed_coefficients)
    return digest.hexdigest()


def _schedule_manifest(settings: GammaBetaCompoundKernelSettings) -> dict[str, object]:
    """Return canonical manifest fields for resolved kernel settings."""
    return {
        "id": GAMMA_BETA_COMPOUND_SCHEDULE_ID,
        "split_direction_probability": settings.split_direction_probability,
        "fraction_refresh_slots": settings.fraction_refresh_slots,
        "relocation_slots": settings.relocation_slots,
        "subtree_retile_slots": settings.subtree_retile_slots,
        "max_subtree_leaves": settings.max_subtree_leaves,
        "fixed_coefficient_proposal_sd": list(settings.fixed_coefficient_proposal_sd),
        "cycle_length": settings.cycle_length,
    }


def _retention_manifest(retention: RetentionSettings) -> dict[str, object]:
    """Return canonical manifest fields for global retention settings."""
    return {
        "warmup_transitions": retention.warmup_transitions,
        "thin": retention.thin,
    }


def build_gamma_beta_run_manifest(
    problem: GammaBetaTreeProblem,
    kernel_settings: GammaBetaCompoundKernelSettings,
    retention: RetentionSettings,
    *,
    chain_id: str,
    initial_state: GammaBetaTreeState,
    code_revision: str,
    input_identifiers: Mapping[str, str],
    input_sha256: Mapping[str, str],
    nominal_weight_policy: str,
    nominal_weight_normalization_factor: float,
    seed: int | None,
) -> dict[str, object]:
    """Build the canonical scientific manifest required by durable checkpoints.

    Input identifiers and hashes must use the same nonempty key set. The
    nominal-weight policy is caller-owned because choosing an emissions,
    area, or other base measure is a scientific modelling decision.

    Args:
        problem: Gamma--Beta scientific problem.
        kernel_settings: Fully resolved compound schedule settings.
        retention: Global warmup and thinning settings.
        chain_id: Nonempty caller-owned chain identifier. Independent chains
            must use different identifiers.
        initial_state: Exact immutable initial state for this chain. Its
            irreducible coordinates are fingerprinted independently of later
            checkpoint states.
        code_revision: Nonempty source revision supplied by the caller.
        input_identifiers: Stable identifiers for every frozen scientific
            input.
        input_sha256: SHA-256 digests for the same named inputs.
        nominal_weight_policy: Explicit description or versioned identifier
            for the native-cell base measure.
        nominal_weight_normalization_factor: Positive factor used to normalize
            the supplied native-cell weights.
        seed: Non-negative initial PCG64 seed, or ``None``.

    Returns:
        Strict-JSON-compatible manifest mapping.

    Raises:
        TypeError: If problem, settings, retention, initial state, or mappings
            have the wrong type.
        ValueError: If required metadata, hashes, normalization, or seed are
            malformed.
    """
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be a GammaBetaTreeProblem.")
    if not isinstance(kernel_settings, GammaBetaCompoundKernelSettings):
        raise TypeError("kernel_settings must be GammaBetaCompoundKernelSettings.")
    if not isinstance(retention, RetentionSettings):
        raise TypeError("retention must be RetentionSettings.")
    if not isinstance(chain_id, str) or not chain_id.strip():
        raise ValueError("chain_id must be a nonempty string.")
    if chain_id != chain_id.strip():
        raise ValueError("chain_id must not have leading or trailing whitespace.")
    initial_sha256 = gamma_beta_state_fingerprint(problem, initial_state)
    if not isinstance(code_revision, str) or not code_revision.strip():
        raise ValueError("code_revision must be a nonempty string.")
    if not isinstance(input_identifiers, Mapping) or not isinstance(input_sha256, Mapping):
        raise TypeError("input_identifiers and input_sha256 must be mappings.")
    identifier_keys = frozenset(input_identifiers)
    hash_keys = frozenset(input_sha256)
    if not identifier_keys or identifier_keys != hash_keys:
        raise ValueError("input identifiers and hashes must have the same nonempty key set.")
    if any(not isinstance(name, str) or not name for name in identifier_keys):
        raise ValueError("input names must be nonempty strings.")
    inputs: dict[str, object] = {}
    for name in sorted(identifier_keys):
        identifier = input_identifiers[name]
        digest = input_sha256[name]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"input_identifiers[{name!r}] must be a nonempty string.")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"input_sha256[{name!r}] must be a 64-character SHA-256.")
        try:
            bytes.fromhex(digest)
        except ValueError as exc:
            raise ValueError(f"input_sha256[{name!r}] must be hexadecimal.") from exc
        inputs[name] = {"identifier": identifier, "sha256": digest.lower()}
    if not isinstance(nominal_weight_policy, str) or not nominal_weight_policy.strip():
        raise ValueError("nominal_weight_policy must be a nonempty string.")
    if isinstance(nominal_weight_normalization_factor, bool):
        raise TypeError("nominal_weight_normalization_factor must be a real number.")
    normalization = float(nominal_weight_normalization_factor)
    if not isfinite(normalization) or normalization <= 0.0:
        raise ValueError("nominal_weight_normalization_factor must be finite and positive.")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a non-negative integer or None.")
        seed = int(seed)
    probabilities = problem.partition_prior.marginal_probability_by_k
    positive_k = np.flatnonzero(probabilities > 0.0)
    if positive_k.size == 0 or not np.array_equal(
        positive_k,
        np.arange(int(positive_k[0]), int(positive_k[-1]) + 1),
    ):
        raise ValueError("run manifests require contiguous positive p(K) support.")
    fixed_block: dict[str, object]
    if problem.fixed_block is None:
        fixed_block = {
            "n_coefficients": 0,
            "coefficient_prior_mean": [],
            "coefficient_prior_sd": [],
        }
    else:
        fixed_block = {
            "n_coefficients": problem.fixed_block.n_coefficients,
            "coefficient_prior_mean": problem.fixed_block.coefficient_prior_mean.tolist(),
            "coefficient_prior_sd": problem.fixed_block.coefficient_prior_sd.tolist(),
        }
    manifest: dict[str, object] = {
        "schema_id": GAMMA_BETA_MANIFEST_SCHEMA_ID,
        "schema_version": GAMMA_BETA_MANIFEST_SCHEMA_VERSION,
        "code_revision": code_revision.strip(),
        "inputs": inputs,
        "nominal_weight": {
            "policy": nominal_weight_policy.strip(),
            "normalization_factor": normalization,
        },
        "problem_sha256": gamma_beta_problem_fingerprint(problem),
        "tree": {
            "shape": list(problem.tree.shape),
            "n_nodes": len(problem.tree.nodes),
            "n_cells": len(problem.tree.leaf_ids),
        },
        "k_prior": {
            "minimum": int(positive_k[0]),
            "maximum": int(positive_k[-1]),
            "probability_by_k_sha256": _array_sha256(
                "partition_prior.marginal_probability_by_k",
                probabilities,
            ),
        },
        "likelihood": {
            "family": "independent_gaussian",
            "power": problem.likelihood_power,
            "n_observations": int(problem.observations.size),
        },
        "gamma_beta_prior": {
            "root_shape": problem.prior.root_shape,
            "root_rate": problem.prior.root_rate,
            "nominal_cell_mass_sha256": _array_sha256(
                "prior.nominal_cell_mass",
                problem.prior.nominal_cell_mass,
            ),
            "beta_shape_by_node_sha256": _array_sha256(
                "prior.beta_shape_by_node",
                problem.prior.beta_shape_by_node,
            ),
        },
        "fixed_block": fixed_block,
        "schedule": _schedule_manifest(kernel_settings),
        "retention": _retention_manifest(retention),
        "seed": seed,
        "chain": {
            "id": chain_id,
            "initial_k": initial_state.k,
            "initial_state_sha256": initial_sha256,
        },
    }
    canonical_gamma_beta_run_manifest(manifest)
    return manifest


def _require_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    location: str,
) -> None:
    """Require an exact metadata field set."""
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{location} has an invalid field set; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )


def _require_mapping(value: object, *, location: str) -> Mapping[str, object]:
    """Return a string-keyed mapping or reject it."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _parse_json_object(payload: str, *, location: str) -> dict[str, Any]:
    """Parse one strict JSON object while rejecting duplicate keys.

    Args:
        payload: JSON text to decode.
        location: Field path used in validation errors.

    Returns:
        Decoded built-in dictionary.

    Raises:
        ValueError: If the text is malformed, is not an object, repeats a key,
            or contains a non-finite constant.
    """

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{location} repeats JSON key {key!r}.")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{location} contains invalid JSON constant {value!r}.")

    try:
        result = json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location} is not valid JSON.") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{location} must contain a JSON object.")
    _validate_json_value(result, location=location)
    return cast(dict[str, Any], result)


def _validate_manifest_against_checkpoint(
    run_manifest: RunManifest,
    checkpoint: GammaBetaCompoundCheckpoint,
) -> str:
    """Validate required manifest fields against one checkpoint.

    Args:
        run_manifest: Candidate caller-owned manifest.
        checkpoint: Boundary defining required scientific and schedule
            metadata.

    Returns:
        Canonical validated manifest JSON.

    Raises:
        TypeError: If the manifest is not a mapping.
        ValueError: If required fields are malformed or disagree with the
            checkpoint.
    """
    manifest_json = canonical_gamma_beta_run_manifest(run_manifest)
    manifest = _parse_json_object(manifest_json, location="run_manifest")
    _require_keys(manifest, _MANIFEST_NAMES, location="run_manifest")
    if (
        manifest["schema_id"] != GAMMA_BETA_MANIFEST_SCHEMA_ID
        or manifest["schema_version"] != GAMMA_BETA_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("run_manifest schema is incompatible.")
    if manifest["problem_sha256"] != gamma_beta_problem_fingerprint(checkpoint.problem):
        raise ValueError("run_manifest problem fingerprint does not match checkpoint problem.")
    if manifest["schedule"] != _schedule_manifest(checkpoint.kernel_settings):
        raise ValueError("run_manifest schedule does not match checkpoint settings.")
    if manifest["retention"] != _retention_manifest(checkpoint.retention):
        raise ValueError("run_manifest retention does not match checkpoint settings.")
    code_revision = manifest["code_revision"]
    if not isinstance(code_revision, str) or not code_revision:
        raise ValueError("run_manifest code_revision must be nonempty.")
    inputs = _require_mapping(manifest["inputs"], location="run_manifest.inputs")
    if not inputs:
        raise ValueError("run_manifest inputs must be nonempty.")
    input_identifiers: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for name, value in inputs.items():
        item = _require_mapping(value, location=f"run_manifest.inputs.{name}")
        _require_keys(
            item,
            frozenset(("identifier", "sha256")),
            location=f"run_manifest.inputs.{name}",
        )
        identifier = item["identifier"]
        digest = item["sha256"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"run_manifest input {name!r} has an invalid identifier.")
        if not isinstance(digest, str):
            raise ValueError(f"run_manifest input {name!r} has an invalid SHA-256.")
        input_identifiers[name] = identifier
        input_hashes[name] = digest
    nominal_weight = _require_mapping(
        manifest["nominal_weight"],
        location="run_manifest.nominal_weight",
    )
    if (
        frozenset(nominal_weight) != frozenset(("policy", "normalization_factor"))
        or not isinstance(nominal_weight["policy"], str)
        or not nominal_weight["policy"]
        or isinstance(nominal_weight["normalization_factor"], bool)
        or not isinstance(nominal_weight["normalization_factor"], (int, float))
        or not isfinite(float(nominal_weight["normalization_factor"]))
        or float(nominal_weight["normalization_factor"]) <= 0.0
    ):
        raise ValueError("run_manifest nominal_weight metadata is malformed.")
    seed = manifest["seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("run_manifest seed must be a non-negative integer or null.")
    chain = _require_mapping(manifest["chain"], location="run_manifest.chain")
    _require_keys(
        chain,
        frozenset(("id", "initial_k", "initial_state_sha256")),
        location="run_manifest.chain",
    )
    chain_id = chain["id"]
    initial_k = chain["initial_k"]
    initial_sha256 = chain["initial_state_sha256"]
    if not isinstance(chain_id, str) or not chain_id or chain_id != chain_id.strip():
        raise ValueError("run_manifest chain id must be a nonempty trimmed string.")
    if isinstance(initial_k, bool) or not isinstance(initial_k, int) or initial_k < 1:
        raise ValueError("run_manifest chain initial_k must be a positive integer.")
    if (
        not isinstance(initial_sha256, str)
        or len(initial_sha256) != 64
        or initial_sha256 != initial_sha256.lower()
    ):
        raise ValueError("run_manifest chain initial_state_sha256 must be a 64-character SHA-256.")
    try:
        bytes.fromhex(initial_sha256)
    except ValueError as exc:
        raise ValueError("run_manifest chain initial_state_sha256 must be hexadecimal.") from exc
    probabilities = checkpoint.problem.partition_prior.marginal_probability_by_k
    if initial_k >= probabilities.size or probabilities[initial_k] <= 0.0:
        raise ValueError("run_manifest chain initial_k lies outside positive p(K) support.")
    expected_manifest = build_gamma_beta_run_manifest(
        checkpoint.problem,
        checkpoint.kernel_settings,
        checkpoint.retention,
        chain_id=chain_id,
        initial_state=checkpoint.state,
        code_revision=code_revision,
        input_identifiers=input_identifiers,
        input_sha256=input_hashes,
        nominal_weight_policy=cast(str, nominal_weight["policy"]),
        nominal_weight_normalization_factor=float(cast(float, nominal_weight["normalization_factor"])),
        seed=seed,
    )
    # The run manifest deliberately preserves the chain's immutable initial
    # coordinates across later segments; only compare the rebuilt scientific
    # portion against the current checkpoint.
    expected_manifest["chain"] = dict(chain)
    if manifest_json != canonical_gamma_beta_run_manifest(expected_manifest):
        raise ValueError("run_manifest scientific metadata does not match checkpoint.")
    return manifest_json


def _checkpoint_arrays(checkpoint: GammaBetaCompoundCheckpoint) -> dict[str, NDArray[Any]]:
    """Return irreducible numeric state coordinates for persistence."""
    return {
        "frontier_node_ids": np.asarray(checkpoint.state.frontier.node_ids, dtype=np.int64),
        "active_fractions": np.asarray(checkpoint.state.active_fractions, dtype=np.float64),
        "fixed_coefficients": np.asarray(checkpoint.state.fixed_coefficients, dtype=np.float64),
    }


def _checkpoint_metadata(
    checkpoint: GammaBetaCompoundCheckpoint,
    arrays: Mapping[str, NDArray[Any]],
    *,
    run_manifest_json: str,
) -> dict[str, object]:
    """Build strict checkpoint metadata from one validated boundary.

    Args:
        checkpoint: Exact compound continuation boundary.
        arrays: Irreducible state arrays to checksum.
        run_manifest_json: Canonical validated run manifest.

    Returns:
        Strict-JSON-compatible checkpoint metadata.
    """
    rng = checkpoint.rng_state
    settings = checkpoint.kernel_settings
    state = checkpoint.state
    return {
        "schema_id": GAMMA_BETA_CHECKPOINT_SCHEMA_ID,
        "schema_version": GAMMA_BETA_CHECKPOINT_SCHEMA_VERSION,
        "numpy_version": np.__version__,
        "schedule_id": checkpoint.schedule_id,
        "problem_sha256": gamma_beta_problem_fingerprint(checkpoint.problem),
        "transitions_completed": checkpoint.transitions_completed,
        "schedule_phase": checkpoint.schedule_phase,
        "rng": {
            "algorithm": rng.algorithm,
            "state": rng.state,
            "increment": rng.increment,
            "has_uint32": rng.has_uint32,
            "uinteger": rng.uinteger,
        },
        "kernel": {
            "split_direction_probability": settings.split_direction_probability,
            "fraction_refresh_slots": settings.fraction_refresh_slots,
            "relocation_slots": settings.relocation_slots,
            "subtree_retile_slots": settings.subtree_retile_slots,
            "max_subtree_leaves": settings.max_subtree_leaves,
            "fixed_coefficient_proposal_sd": list(settings.fixed_coefficient_proposal_sd),
        },
        "retention": _retention_manifest(checkpoint.retention),
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
    """Persist one directory entry where the platform supports directory fsync.

    Unsupported directory handles or ``fsync`` operations are ignored only
    for the documented platform error codes. Other I/O failures propagate.
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


def save_gamma_beta_checkpoint(
    path: PathLike,
    checkpoint: GammaBetaCompoundCheckpoint,
    *,
    run_manifest: RunManifest,
) -> None:
    """Atomically save one strict Gamma--Beta compound checkpoint.

    Args:
        path: Destination NPZ path whose parent directory already exists. A
            temporary file is created there and an existing destination is
            atomically replaced.
        checkpoint: Exact in-memory continuation boundary.
        run_manifest: Required canonical scientific run manifest.

    Raises:
        TypeError: If the checkpoint or manifest has the wrong type.
        ValueError: If checkpoint and manifest settings disagree.
        OSError: If writing, atomically replacing, or durably syncing the
            destination fails on a platform that supports directory syncing.

    Notes:
        The temporary archive contents and containing-directory entry are
        ``fsync``-ed. Platforms that explicitly reject directory handles or
        directory ``fsync`` with a documented unsupported-operation error
        retain atomic replacement semantics without a crash-durability
        guarantee for the directory entry.
    """
    if not isinstance(checkpoint, GammaBetaCompoundCheckpoint):
        raise TypeError("checkpoint must be GammaBetaCompoundCheckpoint.")
    _rebuild_and_validate_state(
        checkpoint.problem,
        checkpoint.state,
        location="checkpoint state",
    )
    run_manifest_json = _validate_manifest_against_checkpoint(run_manifest, checkpoint)
    arrays = _checkpoint_arrays(checkpoint)
    metadata = _checkpoint_metadata(
        checkpoint,
        arrays,
        run_manifest_json=run_manifest_json,
    )
    metadata_bytes = _canonical_json(metadata, location="checkpoint metadata").encode("utf-8")
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
        os.replace(temporary_path, destination)
        temporary_path = None
        _fsync_parent_directory(destination.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_archive_arrays(path: Path) -> dict[str, NDArray[Any]]:
    """Load the exact numeric archive field set with pickle disabled.

    Args:
        path: Source NPZ archive.

    Returns:
        Owned arrays indexed by the required schema field names.

    Raises:
        OSError: If the archive cannot be opened.
        ValueError: If the archive is corrupt, is an NPY file, has unexpected
            fields, or requires pickle/object loading.
    """
    try:
        archive = np.load(path, allow_pickle=False)
    except (FileNotFoundError, PermissionError):
        raise
    except (BadZipFile, EOFError, ValueError) as exc:
        raise ValueError("Gamma-Beta checkpoint archive is corrupt or unreadable.") from exc
    if isinstance(archive, np.ndarray):
        raise ValueError("Gamma-Beta checkpoint must be an NPZ archive.")
    try:
        with archive:
            names = frozenset(archive.files)
            if names != _ARCHIVE_NAMES:
                raise ValueError(
                    "Gamma-Beta checkpoint has an invalid array set; "
                    f"missing={sorted(_ARCHIVE_NAMES - names)}, "
                    f"unexpected={sorted(names - _ARCHIVE_NAMES)}."
                )
            try:
                return {name: np.array(archive[name], copy=True) for name in names}
            except ValueError as exc:
                raise ValueError(
                    "Gamma-Beta checkpoint arrays cannot require pickle or object dtype."
                ) from exc
    except ValueError:
        raise
    except (BadZipFile, EOFError) as exc:
        raise ValueError("Gamma-Beta checkpoint archive is corrupt or unreadable.") from exc


def _metadata_from_arrays(arrays: Mapping[str, NDArray[Any]]) -> dict[str, Any]:
    """Check the stored metadata digest and decode strict JSON.

    Args:
        arrays: Loaded archive arrays containing metadata bytes and digest.

    Returns:
        Decoded checkpoint metadata.

    Raises:
        ValueError: If byte layouts, digest, JSON, or fields are malformed.
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


def _integer(value: object, *, location: str, minimum: int = 0) -> int:
    """Return a bounded built-in metadata integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer at least {minimum}.")
    return value


def _finite_float(value: object, *, location: str, positive: bool = False) -> float:
    """Return a finite metadata float, optionally requiring positivity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite real number.")
    result = float(value)
    if not isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{location} must be {qualifier}.")
    return result


def _state_array(
    arrays: Mapping[str, NDArray[Any]],
    name: str,
    *,
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    """Validate one persisted one-dimensional state array.

    Args:
        arrays: Loaded checkpoint arrays.
        name: Required state-array name.
        dtype: Exact required NumPy dtype.

    Returns:
        The validated owned array.

    Raises:
        ValueError: If the dtype or dimensionality is wrong.
    """
    array = arrays[name]
    if array.dtype != dtype or array.ndim != 1:
        raise ValueError(f"Checkpoint {name} must be one-dimensional {dtype}.")
    return array


def load_gamma_beta_checkpoint(
    path: PathLike,
    problem: GammaBetaTreeProblem,
    *,
    expected_run_manifest: RunManifest,
) -> GammaBetaCompoundCheckpoint:
    """Load and fully rebuild one durable Gamma--Beta continuation boundary.

    Args:
        path: Source NPZ archive.
        problem: Reconstructed problem expected by the resumed run.
        expected_run_manifest: Exact caller-held manifest expected in the
            archive.

    Returns:
        Valid in-memory compound checkpoint attached to ``problem``.

    Raises:
        TypeError: If ``problem`` or the manifest has the wrong type.
        ValueError: If schema, hashes, problem, manifest, settings, state, or
            rebuilt target components disagree.
        OSError: If the archive cannot be read.

    Notes:
        ``numpy_version`` is recorded as provenance but is not a compatibility
        gate. The explicit PCG64 state must be accepted by the installed
        NumPy; callers making cross-version bit-for-bit replay claims should
        validate those claims in their run environment.
    """
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be GammaBetaTreeProblem.")
    if not isinstance(expected_run_manifest, Mapping):
        raise TypeError("expected_run_manifest must be a mapping.")
    arrays = _load_archive_arrays(Path(path))
    metadata = _metadata_from_arrays(arrays)
    if (
        metadata["schema_id"] != GAMMA_BETA_CHECKPOINT_SCHEMA_ID
        or metadata["schema_version"] != GAMMA_BETA_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("Gamma-Beta checkpoint schema is incompatible.")
    if not isinstance(metadata["numpy_version"], str) or not metadata["numpy_version"]:
        raise ValueError("Gamma-Beta checkpoint NumPy provenance is malformed.")
    if metadata["schedule_id"] != GAMMA_BETA_COMPOUND_SCHEDULE_ID:
        raise ValueError("Gamma-Beta checkpoint schedule is incompatible.")
    expected_problem_sha = gamma_beta_problem_fingerprint(problem)
    if metadata["problem_sha256"] != expected_problem_sha:
        raise ValueError("Gamma-Beta checkpoint problem fingerprint does not match.")

    embedded_manifest_json = metadata["run_manifest_json"]
    embedded_manifest_sha = metadata["run_manifest_sha256"]
    if not isinstance(embedded_manifest_json, str) or not isinstance(embedded_manifest_sha, str):
        raise ValueError("Checkpoint run manifest metadata is malformed.")
    if not hmac.compare_digest(
        sha256(embedded_manifest_json.encode("utf-8")).hexdigest(),
        embedded_manifest_sha,
    ):
        raise ValueError("Checkpoint run manifest SHA-256 checksum does not match.")
    expected_manifest_json = canonical_gamma_beta_run_manifest(expected_run_manifest)
    if not hmac.compare_digest(embedded_manifest_json, expected_manifest_json):
        raise ValueError("Checkpoint run manifest does not match expected canonical content.")

    array_hashes = _require_mapping(metadata["array_sha256"], location="array_sha256")
    if frozenset(array_hashes) != frozenset(_STATE_ARRAY_NAMES):
        raise ValueError("Checkpoint array_sha256 has an invalid field set.")
    for name in _STATE_ARRAY_NAMES:
        supplied_hash = array_hashes[name]
        if not isinstance(supplied_hash, str) or not hmac.compare_digest(
            supplied_hash,
            _array_sha256(name, arrays[name]),
        ):
            raise ValueError(f"Checkpoint {name} SHA-256 checksum does not match.")

    frontier_ids = _state_array(
        arrays,
        "frontier_node_ids",
        dtype=np.dtype(np.int64),
    )
    active_fractions = _state_array(
        arrays,
        "active_fractions",
        dtype=np.dtype(np.float64),
    )
    fixed_coefficients = _state_array(
        arrays,
        "fixed_coefficients",
        dtype=np.dtype(np.float64),
    )
    state_metadata = _require_mapping(metadata["state"], location="state")
    _require_keys(
        state_metadata,
        frozenset(("k", "root_total", *_STATE_LOG_FIELDS)),
        location="state",
    )
    k = _integer(state_metadata["k"], location="state.k", minimum=1)
    if frontier_ids.size != k:
        raise ValueError("Checkpoint frontier length does not match stored K.")
    root_total = _finite_float(
        state_metadata["root_total"],
        location="state.root_total",
        positive=True,
    )
    state = build_gamma_beta_tree_state(
        problem,
        frontier=DyadicFrontier(tuple(int(value) for value in frontier_ids)),
        root_total=root_total,
        active_fractions=active_fractions,
        fixed_coefficients=fixed_coefficients,
    )
    for name in _STATE_LOG_FIELDS:
        expected_value = _finite_float(state_metadata[name], location=f"state.{name}")
        if getattr(state, name) != expected_value:
            raise ValueError(f"Rebuilt state {name} does not match checkpoint metadata.")

    kernel = _require_mapping(metadata["kernel"], location="kernel")
    _require_keys(
        kernel,
        frozenset(
            (
                "split_direction_probability",
                "fraction_refresh_slots",
                "relocation_slots",
                "subtree_retile_slots",
                "max_subtree_leaves",
                "fixed_coefficient_proposal_sd",
            )
        ),
        location="kernel",
    )
    scales = kernel["fixed_coefficient_proposal_sd"]
    if not isinstance(scales, list):
        raise ValueError("kernel fixed_coefficient_proposal_sd must be a JSON array.")
    settings = GammaBetaCompoundKernelSettings(
        split_direction_probability=_finite_float(
            kernel["split_direction_probability"],
            location="kernel.split_direction_probability",
        ),
        fraction_refresh_slots=_integer(
            kernel["fraction_refresh_slots"],
            location="kernel.fraction_refresh_slots",
        ),
        relocation_slots=_integer(
            kernel["relocation_slots"],
            location="kernel.relocation_slots",
        ),
        subtree_retile_slots=_integer(
            kernel["subtree_retile_slots"],
            location="kernel.subtree_retile_slots",
        ),
        max_subtree_leaves=_integer(
            kernel["max_subtree_leaves"],
            location="kernel.max_subtree_leaves",
            minimum=1,
        ),
        fixed_coefficient_proposal_sd=tuple(
            _finite_float(value, location="kernel.fixed_coefficient_proposal_sd", positive=True)
            for value in scales
        ),
    )
    retention_metadata = _require_mapping(metadata["retention"], location="retention")
    _require_keys(
        retention_metadata,
        frozenset(("warmup_transitions", "thin")),
        location="retention",
    )
    retention = RetentionSettings(
        warmup_transitions=_integer(
            retention_metadata["warmup_transitions"],
            location="retention.warmup_transitions",
        ),
        thin=_integer(retention_metadata["thin"], location="retention.thin", minimum=1),
    )
    rng_metadata = _require_mapping(metadata["rng"], location="rng")
    _require_keys(
        rng_metadata,
        frozenset(("algorithm", "state", "increment", "has_uint32", "uinteger")),
        location="rng",
    )
    algorithm = rng_metadata["algorithm"]
    if not isinstance(algorithm, str):
        raise ValueError("rng.algorithm must be a string.")
    rng_state = PCG64State(
        state=_integer(rng_metadata["state"], location="rng.state"),
        increment=_integer(rng_metadata["increment"], location="rng.increment"),
        has_uint32=_integer(rng_metadata["has_uint32"], location="rng.has_uint32"),
        uinteger=_integer(rng_metadata["uinteger"], location="rng.uinteger"),
        algorithm=algorithm,
    )
    rng_state.generator()
    checkpoint = GammaBetaCompoundCheckpoint(
        problem=problem,
        state=state,
        rng_state=rng_state,
        transitions_completed=_integer(
            metadata["transitions_completed"],
            location="transitions_completed",
        ),
        schedule_phase=_integer(metadata["schedule_phase"], location="schedule_phase"),
        kernel_settings=settings,
        retention=retention,
        schedule_id=cast(str, metadata["schedule_id"]),
    )
    _validate_manifest_against_checkpoint(expected_run_manifest, checkpoint)
    return checkpoint


def _trace_coordinate(
    values: object | None,
    *,
    size: int,
    name: str,
    numeric: bool,
) -> NDArray[Any] | None:
    """Normalize one optional physical coordinate or label vector."""
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim != 1 or array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if numeric:
        try:
            result = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain finite numeric values.") from exc
        if np.any(~np.isfinite(result)):
            raise ValueError(f"{name} must contain finite numeric values.")
        return result
    items = array.tolist()
    if any(not isinstance(value, str) or not value for value in items):
        raise ValueError(f"{name} must contain nonempty strings.")
    return np.asarray(items, dtype=np.str_)


def _validate_trace_target_components(
    trace: GammaBetaCompoundTrace,
    problem: GammaBetaTreeProblem,
) -> None:
    """Require retained likelihood-power and target-decomposition identities."""

    def close(actual: NDArray[np.float64], expected: NDArray[np.float64]) -> bool:
        return bool(
            np.array_equal(actual, expected)
            or np.allclose(
                actual,
                expected,
                rtol=float(4.0 * np.finfo(np.float64).eps),
                atol=1e-12,
            )
        )

    powered = trace.log_gaussian_likelihood * problem.likelihood_power
    if not close(trace.log_likelihood, powered):
        raise ValueError("trace log_likelihood is inconsistent with the problem likelihood power.")
    decomposed = (
        trace.log_likelihood
        + trace.log_root_prior
        + trace.log_fraction_prior
        + trace.log_partition_prior
        + trace.log_fixed_coefficient_prior
    )
    if not close(trace.log_target, decomposed):
        raise ValueError("trace log_target is inconsistent with its target components.")


def gamma_beta_compound_trace_to_dataset(
    trace: GammaBetaCompoundTrace,
    problem: GammaBetaTreeProblem,
    *,
    metadata: Mapping[str, Any] | None = None,
    latitudes: object | None = None,
    longitudes: object | None = None,
    fixed_parameter_labels: object | None = None,
    measurement_labels: object | None = None,
) -> xr.Dataset:
    """Convert retained Gamma--Beta states and attempts to an xarray dataset.

    ``draw`` indexes retained states and ``attempt`` indexes every atomic
    transition. ``region_slot`` and ``split_slot`` hold padded frontier IDs
    and active-split node IDs/fractions; inactive IDs use ``-1``, inactive
    fractions use ``NaN``, and Boolean masks identify valid entries.
    ``fixed_parameter`` indexes the always-active block. Zero-retained-draw
    traces produce zero-width state dimensions while preserving attempts.
    The complete canonical tree geometry is stored on ``node``. Optional
    physical latitude/longitude coordinates and exact fixed/measurement
    labels describe the caller's scientific inputs, but are deliberately not
    included in :func:`gamma_beta_problem_fingerprint`; callers must preserve
    those semantics through the run manifest and frozen-input hashes. Caller
    metadata is copied to attributes, but reserved provenance attributes are
    overwritten.

    Args:
        trace: Valid compound sampler trace.
        problem: Problem whose fixed tree defines frontier and active-split
            node IDs.
        metadata: Optional caller-supplied dataset attributes.
        latitudes: Optional finite native-grid row coordinates.
        longitudes: Optional finite native-grid column coordinates.
        fixed_parameter_labels: Optional nonempty string label for each
            always-active fixed coefficient.
        measurement_labels: Optional nonempty string label for each
            observation.

    Returns:
        Dataset containing retained target components and every-attempt
        diagnostics.

    Raises:
        TypeError: If ``trace``, ``problem``, or ``metadata`` has the wrong
            type.
        ValueError: If dimensions, optional coordinates, a retained frontier,
            likelihood powering, or target decomposition are inconsistent
            with ``problem``.
    """
    if not isinstance(trace, GammaBetaCompoundTrace):
        raise TypeError("trace must be GammaBetaCompoundTrace.")
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be GammaBetaTreeProblem.")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None.")
    _validate_trace_target_components(trace, problem)
    n_draws = len(trace.frontiers)
    n_fixed = trace.fixed_coefficients.shape[1]
    if n_fixed != problem.n_fixed_coefficients:
        raise ValueError("trace fixed-coefficient width does not match problem.")
    rows, columns = problem.tree.shape
    latitude = _trace_coordinate(
        latitudes,
        size=rows,
        name="latitudes",
        numeric=True,
    )
    longitude = _trace_coordinate(
        longitudes,
        size=columns,
        name="longitudes",
        numeric=True,
    )
    fixed_labels = _trace_coordinate(
        fixed_parameter_labels,
        size=n_fixed,
        name="fixed_parameter_labels",
        numeric=False,
    )
    observation_labels = _trace_coordinate(
        measurement_labels,
        size=problem.observations.size,
        name="measurement_labels",
        numeric=False,
    )
    region_capacity = max((len(frontier) for frontier in trace.frontiers), default=0)
    split_capacity = max((fraction.size for fraction in trace.split_fractions), default=0)
    frontier_node_id = np.full((n_draws, region_capacity), -1, dtype=np.int64)
    frontier_active = np.zeros((n_draws, region_capacity), dtype=np.bool_)
    split_node_id = np.full((n_draws, split_capacity), -1, dtype=np.int64)
    split_fraction = np.full((n_draws, split_capacity), np.nan, dtype=np.float64)
    split_active = np.zeros((n_draws, split_capacity), dtype=np.bool_)
    for draw, (frontier, fractions) in enumerate(zip(trace.frontiers, trace.split_fractions, strict=True)):
        frontier.validate(problem.tree)
        k = len(frontier)
        frontier_node_id[draw, :k] = frontier.node_ids
        frontier_active[draw, :k] = True
        split_count = fractions.size
        split_node_id[draw, :split_count] = frontier.active_split_nodes(problem.tree)
        split_fraction[draw, :split_count] = fractions
        split_active[draw, :split_count] = True

    tree = problem.tree
    node_id = np.asarray([node.node_id for node in tree.nodes], dtype=np.int64)
    node_parent_id = np.asarray(
        [-1 if node.parent_id is None else node.parent_id for node in tree.nodes],
        dtype=np.int64,
    )
    node_first_child_id = np.asarray(
        [-1 if not node.child_ids else node.child_ids[0] for node in tree.nodes],
        dtype=np.int64,
    )
    node_second_child_id = np.asarray(
        [-1 if not node.child_ids else node.child_ids[1] for node in tree.nodes],
        dtype=np.int64,
    )
    attrs = {} if metadata is None else dict(metadata)
    attrs.update(
        {
            "schema_id": GAMMA_BETA_TRACE_SCHEMA_ID,
            "schema_version": GAMMA_BETA_TRACE_SCHEMA_VERSION,
            "problem_sha256": gamma_beta_problem_fingerprint(problem),
            "likelihood_family": "independent_gaussian",
            "likelihood_power": problem.likelihood_power,
            "tree_rows": rows,
            "tree_columns": columns,
            "problem_fingerprint_scope": (
                "Numerical problem arrays, priors, likelihood power, and canonical "
                "tree topology; caller labels and physical coordinate semantics are "
                "covered only by the run manifest and frozen-input hashes."
            ),
        }
    )
    coords: dict[str, Any] = {
        "draw": np.arange(n_draws, dtype=np.int64),
        "region_slot": np.arange(region_capacity, dtype=np.int64),
        "split_slot": np.arange(split_capacity, dtype=np.int64),
        "fixed_parameter": (np.arange(n_fixed, dtype=np.int64) if fixed_labels is None else fixed_labels),
        "measurement": (
            np.arange(problem.observations.size, dtype=np.int64)
            if observation_labels is None
            else observation_labels
        ),
        "tree_row": np.arange(rows, dtype=np.int64),
        "tree_column": np.arange(columns, dtype=np.int64),
        "node": node_id,
        "state_transition": ("draw", trace.state_transition),
        "attempt": np.arange(trace.global_transition.size, dtype=np.int64),
        "global_transition": ("attempt", trace.global_transition),
    }
    if latitude is not None:
        coords["latitude"] = ("tree_row", latitude)
    if longitude is not None:
        coords["longitude"] = ("tree_column", longitude)
    dataset = xr.Dataset(
        data_vars={
            "k": ("draw", trace.k),
            "frontier_node_id": (("draw", "region_slot"), frontier_node_id),
            "frontier_active": (("draw", "region_slot"), frontier_active),
            "split_node_id": (("draw", "split_slot"), split_node_id),
            "split_fraction": (("draw", "split_slot"), split_fraction),
            "split_active": (("draw", "split_slot"), split_active),
            "root_total": ("draw", trace.root_total),
            "fixed_coefficients": (
                ("draw", "fixed_parameter"),
                trace.fixed_coefficients,
            ),
            "log_gaussian_likelihood": ("draw", trace.log_gaussian_likelihood),
            "log_likelihood": ("draw", trace.log_likelihood),
            "log_root_prior": ("draw", trace.log_root_prior),
            "log_fraction_prior": ("draw", trace.log_fraction_prior),
            "log_partition_prior": ("draw", trace.log_partition_prior),
            "log_fixed_coefficient_prior": (
                "draw",
                trace.log_fixed_coefficient_prior,
            ),
            "log_target": ("draw", trace.log_target),
            "slot": ("attempt", trace.slot),
            "move": ("attempt", trace.move),
            "valid": ("attempt", trace.valid),
            "accepted": ("attempt", trace.accepted),
            "node_id": ("attempt", trace.node_id),
            "secondary_node_id": ("attempt", trace.secondary_node_id),
            "block_leaf_count": ("attempt", trace.block_leaf_count),
            "coefficient_id": ("attempt", trace.coefficient_id),
            "k_before": ("attempt", trace.k_before),
            "k_after": ("attempt", trace.k_after),
            "log_acceptance_ratio": ("attempt", trace.log_acceptance_ratio),
            "node_row_start": (
                "node",
                np.asarray([node.row_start for node in tree.nodes], dtype=np.int64),
            ),
            "node_row_stop": (
                "node",
                np.asarray([node.row_stop for node in tree.nodes], dtype=np.int64),
            ),
            "node_column_start": (
                "node",
                np.asarray([node.col_start for node in tree.nodes], dtype=np.int64),
            ),
            "node_column_stop": (
                "node",
                np.asarray([node.col_stop for node in tree.nodes], dtype=np.int64),
            ),
            "node_depth": (
                "node",
                np.asarray([node.depth for node in tree.nodes], dtype=np.int64),
            ),
            "node_parent_id": ("node", node_parent_id),
            "node_first_child_id": ("node", node_first_child_id),
            "node_second_child_id": ("node", node_second_child_id),
        },
        coords=coords,
        attrs=attrs,
    )
    dataset["frontier_node_id"].attrs.update(
        {
            "long_name": "active canonical tree node identifier",
            "inactive_sentinel": -1,
            "mask": "frontier_active",
        }
    )
    dataset["frontier_active"].attrs["long_name"] = "valid frontier slot mask"
    dataset["split_node_id"].attrs.update(
        {
            "long_name": "active split-node identifier",
            "inactive_sentinel": -1,
            "mask": "split_active",
        }
    )
    dataset["split_fraction"].attrs.update(
        {
            "long_name": "active first-child mass fraction",
            "inactive_sentinel": "NaN",
            "mask": "split_active",
        }
    )
    dataset["split_active"].attrs["long_name"] = "valid split slot mask"
    dataset["log_gaussian_likelihood"].attrs.update(
        {
            "long_name": "raw normalized independent-Gaussian log likelihood",
            "likelihood_power_applied": 0,
        }
    )
    dataset["log_likelihood"].attrs.update(
        {
            "long_name": "powered log likelihood used in the posterior target",
            "likelihood_power_applied": problem.likelihood_power,
            "identity": "log_likelihood = likelihood_power * log_gaussian_likelihood",
        }
    )
    dataset["log_target"].attrs.update(
        {
            "long_name": "complete powered posterior log target",
            "decomposition": (
                "log_likelihood + log_root_prior + log_fraction_prior + "
                "log_partition_prior + log_fixed_coefficient_prior"
            ),
        }
    )
    dataset["node_id"].attrs.update(
        {
            "long_name": "primary proposal tree node or subtree-block identifier",
            "not_applicable_sentinel": -1,
        }
    )
    dataset["secondary_node_id"].attrs.update(
        {
            "long_name": "relocation destination split-node identifier",
            "not_applicable_sentinel": -1,
        }
    )
    dataset["block_leaf_count"].attrs.update(
        {
            "long_name": "active region count inside a subtree-retile block",
            "not_applicable_sentinel": -1,
            "units": "active_regions",
        }
    )
    dataset["coefficient_id"].attrs.update(
        {
            "long_name": "proposal fixed-coefficient position",
            "not_applicable_sentinel": -1,
        }
    )
    for name in ("node_parent_id", "node_first_child_id", "node_second_child_id"):
        dataset[name].attrs["absent_sentinel"] = -1
    for name in (
        "node_row_start",
        "node_row_stop",
        "node_column_start",
        "node_column_stop",
    ):
        dataset[name].attrs["bounds_convention"] = "zero-based half-open native-grid indices"
    return dataset


__all__ = [
    "GAMMA_BETA_CHECKPOINT_SCHEMA_ID",
    "GAMMA_BETA_CHECKPOINT_SCHEMA_VERSION",
    "GAMMA_BETA_MANIFEST_SCHEMA_ID",
    "GAMMA_BETA_MANIFEST_SCHEMA_VERSION",
    "GAMMA_BETA_TRACE_SCHEMA_ID",
    "GAMMA_BETA_TRACE_SCHEMA_VERSION",
    "RunManifest",
    "build_gamma_beta_run_manifest",
    "canonical_gamma_beta_run_manifest",
    "gamma_beta_compound_trace_to_dataset",
    "gamma_beta_problem_fingerprint",
    "gamma_beta_state_fingerprint",
    "load_gamma_beta_checkpoint",
    "save_gamma_beta_checkpoint",
]
