"""Validate and report the four-chain Gamma--Beta Stage-3 HPC diagnostic.

The expected layout is ``RUN_ROOT/chain_0/segment_000`` through
``RUN_ROOT/chain_3/segment_009``.  Each immutable segment must contain the
five artifacts written by :mod:`gamma_beta_native_smoke`.  This postprocessor
validates completion hashes, canonical manifests, exact checkpoints, attempt
coordinates, and retained-draw continuity before calculating mobility,
performance, scientific-summary, and multi-chain convergence diagnostics.

Stage 3 is deliberately only 1,000 cycles per chain.  The generated report
therefore treats every result as a sampler diagnostic, never as a converged
scientific inversion.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import pairwise
from math import isfinite
from pathlib import Path
from typing import Any, cast

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from openghg_inversions.experimental.rjmcmc.dyadic_tree import DyadicFrontier
from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    GammaBetaRHIMEAdapterResult,
    gamma_beta_problem_from_rhime_inputs,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_io import (
    GAMMA_BETA_TRACE_SCHEMA_ID,
    canonical_gamma_beta_run_manifest,
    load_gamma_beta_checkpoint,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_tree import (
    GammaBetaTreeProblem,
    build_gamma_beta_tree_state,
    render_cell_mass,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

CHAIN_COUNT = 4
SEGMENT_COUNT = 10
CYCLES_PER_SEGMENT = 100
CYCLE_LENGTH = 14
TRANSITIONS_PER_SEGMENT = CYCLES_PER_SEGMENT * CYCLE_LENGTH
TRANSITIONS_PER_CHAIN = SEGMENT_COUNT * TRANSITIONS_PER_SEGMENT
WARMUP_TRANSITIONS = 2_800
THIN = 14
EXPECTED_START_K = (50, 250, 50, 250)
FIXED_COUNT = 6
ARTIFACT_NAMES = frozenset(
    ("manifest.json", "checkpoint.npz", "trace.nc", "summary.json")
)
TARGET_FIELDS = (
    "log_gaussian_likelihood",
    "log_likelihood",
    "log_root_prior",
    "log_fraction_prior",
    "log_partition_prior",
    "log_fixed_coefficient_prior",
    "log_target",
)
MOVE_NAMES = (
    "split",
    "merge",
    "root_refresh",
    "fraction_refresh",
    "fixed_coefficient",
)
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0
CH4_MOLAR_MASS_G_PER_MOL = 16.0425
PARIS_ROOT = Path("/group/chem/acrg/PARIS_inversions")


@dataclass(frozen=True)
class SegmentData:
    """Validated immutable segment and its loaded trace."""

    chain: int
    segment: int
    directory: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    completion: dict[str, Any]
    trace: xr.Dataset
    checkpoint_k: int
    checkpoint_root_total: float
    checkpoint_fixed_coefficients: FloatArray


@dataclass(frozen=True)
class ChainData:
    """Ten validated segments for one chain."""

    chain: int
    segments: tuple[SegmentData, ...]
    manifest: dict[str, Any]
    state_transition: IntArray
    k: IntArray
    root_total: FloatArray
    fixed_coefficients: FloatArray
    target_components: Mapping[str, FloatArray]


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one strict finite JSON object, rejecting duplicate keys."""

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{path} repeats JSON key {key!r}.")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains invalid JSON constant {value!r}.")

    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON.") from error
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return loaded


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _distribution(values: NDArray[Any]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "q05": None,
            "median": None,
            "q95": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _reject_protected_output(path: Path) -> None:
    resolved = path.resolve()
    protected = PARIS_ROOT.resolve()
    if resolved == protected or resolved.is_relative_to(protected):
        raise ValueError(f"Refusing to write beneath {PARIS_ROOT}.")


def _validate_paths(arguments: argparse.Namespace) -> None:
    if not arguments.run_root.is_dir():
        raise FileNotFoundError(f"Stage-3 run root is not a directory: {arguments.run_root}")
    if not arguments.input.is_file():
        raise FileNotFoundError(f"Frozen input is not a file: {arguments.input}")
    if arguments.output_directory.exists():
        raise FileExistsError(f"Output directory already exists: {arguments.output_directory}")
    if not arguments.output_directory.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent is not a directory: {arguments.output_directory.parent}"
        )
    _reject_protected_output(arguments.output_directory)


def _segment_directory(run_root: Path, chain: int, segment: int) -> Path:
    return run_root / f"chain_{chain}" / f"segment_{segment:03d}"


def _validate_exact_layout(run_root: Path) -> None:
    expected = {
        _segment_directory(run_root, chain, segment).resolve()
        for chain in range(CHAIN_COUNT)
        for segment in range(SEGMENT_COUNT)
    }
    actual = {
        path.parent.resolve()
        for path in run_root.glob("chain_*/segment_*/complete.json")
    }
    missing = sorted(str(path) for path in expected - actual)
    unexpected = sorted(str(path) for path in actual - expected)
    if missing or unexpected:
        raise ValueError(
            "Stage-3 layout must contain exactly four chains x ten segments; "
            f"missing={missing}, unexpected={unexpected}."
        )


def _validate_completion(directory: Path) -> dict[str, Any]:
    completion_path = directory / "complete.json"
    completion = _load_json(completion_path)
    if completion.get("checkpoint") != "checkpoint.npz":
        raise ValueError(f"{completion_path} does not identify checkpoint.npz.")
    hashes = completion.get("sha256")
    if not isinstance(hashes, dict) or frozenset(hashes) != ARTIFACT_NAMES:
        raise ValueError(f"{completion_path} has an invalid artifact hash set.")
    for name in sorted(ARTIFACT_NAMES):
        path = directory / name
        digest = hashes[name]
        if (
            not path.is_file()
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or _sha256_file(path) != digest
        ):
            raise ValueError(f"Completion SHA-256 validation failed for {path}.")
    return completion


def _scientific_manifest(manifest: Mapping[str, Any]) -> str:
    """Canonical manifest content common to independent chains."""
    common = deepcopy(dict(manifest))
    common.pop("seed", None)
    common.pop("chain", None)
    return json.dumps(
        common,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _input_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    try:
        item = manifest["inputs"]["input_variable_contract"]
        identifier = item["identifier"]
        expected_sha = item["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("Manifest has no frozen input-variable contract.") from error
    if not isinstance(identifier, str) or not isinstance(expected_sha, str):
        raise TypeError("Manifest input-variable contract is malformed.")
    actual_sha = sha256(identifier.encode("utf-8")).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("Manifest input-variable contract SHA-256 does not match.")
    try:
        contract = json.loads(identifier)
    except json.JSONDecodeError as error:
        raise ValueError("Manifest input-variable contract is not valid JSON.") from error
    if not isinstance(contract, dict):
        raise TypeError("Manifest input-variable contract must be an object.")
    return contract


def _rebuild_problem(
    input_path: Path,
    manifest: Mapping[str, Any],
    *,
    concentration: float,
) -> tuple[xr.Dataset, GammaBetaRHIMEAdapterResult]:
    """Rebuild the exact problem bound by the first chain manifest."""
    frozen = manifest["inputs"]["frozen_native_dataset"]
    expected_input_sha = frozen["sha256"]
    if _sha256_file(input_path) != expected_input_sha:
        raise ValueError("Frozen input SHA-256 does not match the run manifest.")
    contract = _input_contract(manifest)
    engine = str(contract["input_netcdf_engine"])
    with xr.open_dataset(input_path, engine=engine) as opened:
        dataset = opened.load()
    gamma_prior = manifest["gamma_beta_prior"]
    root_shape = float(gamma_prior["root_shape"])
    root_rate = float(gamma_prior["root_rate"])
    root_variance = root_shape / root_rate**2
    fixed = manifest["fixed_block"]
    adapter = gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=dataset[str(contract["nominal_weight_name"])],
        k_min=int(manifest["k_prior"]["minimum"]),
        k_max=int(manifest["k_prior"]["maximum"]),
        concentration=concentration,
        root_variance=root_variance,
        normalize_weights=bool(contract["normalize_weights"]),
        likelihood_power=float(manifest["likelihood"]["power"]),
        sensitivity_name=str(contract["sensitivity_name"]),
        observation_name=str(contract["observation_name"]),
        observation_sd_name=str(contract["observation_sd_name"]),
        fixed_design_name=str(contract["fixed_design_name"]),
        fixed_offset_name=str(contract["fixed_offset_name"]),
        fixed_coefficient_prior_mean=np.asarray(
            fixed["coefficient_prior_mean"],
            dtype=np.float64,
        ),
        fixed_coefficient_prior_sd=np.asarray(
            fixed["coefficient_prior_sd"],
            dtype=np.float64,
        ),
    )
    if adapter.problem.n_fixed_coefficients != FIXED_COUNT:
        raise ValueError("Stage 3 requires exactly six fixed outer coefficients.")
    if adapter.problem.tree.shape != tuple(manifest["tree"]["shape"]):
        raise ValueError("Rebuilt problem tree shape does not match the manifest.")
    if adapter.weight_normalization_factor != float(
        manifest["nominal_weight"]["normalization_factor"]
    ):
        raise ValueError("Rebuilt nominal-weight normalization does not match the manifest.")
    return dataset, adapter


def _required_trace_variables() -> frozenset[str]:
    return frozenset(
        (
            "k",
            "frontier_node_id",
            "frontier_active",
            "split_node_id",
            "split_fraction",
            "split_active",
            "root_total",
            "fixed_coefficients",
            *TARGET_FIELDS,
            "slot",
            "move",
            "valid",
            "accepted",
            "node_id",
            "coefficient_id",
            "k_before",
            "k_after",
            "log_acceptance_ratio",
        )
    )


def _validate_trace(
    trace: xr.Dataset,
    *,
    chain: int,
    segment: int,
    manifest: Mapping[str, Any],
) -> None:
    location = f"chain {chain} segment {segment}"
    missing = _required_trace_variables() - frozenset(trace.data_vars)
    if missing:
        raise ValueError(f"{location} trace is missing variables {sorted(missing)}.")
    if trace.attrs.get("schema_id") != GAMMA_BETA_TRACE_SCHEMA_ID:
        raise ValueError(f"{location} has an incompatible trace schema.")
    if trace.attrs.get("problem_sha256") != manifest["problem_sha256"]:
        raise ValueError(f"{location} trace problem fingerprint does not match.")
    if int(trace.sizes["attempt"]) != TRANSITIONS_PER_SEGMENT:
        raise ValueError(f"{location} must contain exactly {TRANSITIONS_PER_SEGMENT} attempts.")
    if int(trace.sizes["fixed_parameter"]) != FIXED_COUNT:
        raise ValueError(f"{location} must contain six fixed coefficients.")
    valid = np.asarray(trace["valid"].values, dtype=np.bool_)
    accepted = np.asarray(trace["accepted"].values, dtype=np.bool_)
    k_before = np.asarray(trace["k_before"].values, dtype=np.int64)
    k_after = np.asarray(trace["k_after"].values, dtype=np.int64)
    ratios = np.asarray(trace["log_acceptance_ratio"].values, dtype=np.float64)
    if np.any(accepted & ~valid):
        raise ValueError(f"{location} has accepted invalid proposals.")
    if np.any(~valid & (k_before != k_after)):
        raise ValueError(f"{location} invalid proposals changed K.")
    if np.any(np.isnan(ratios)) or np.any(np.isposinf(ratios)):
        raise ValueError(f"{location} has NaN or positive-infinite acceptance ratios.")
    if k_before.size > 1 and not np.array_equal(k_before[1:], k_after[:-1]):
        raise ValueError(f"{location} attempted K path is discontinuous.")
    for name in TARGET_FIELDS:
        if np.any(~np.isfinite(np.asarray(trace[name].values, dtype=np.float64))):
            raise ValueError(f"{location} retained {name} contains non-finite values.")
    likelihood_power = float(manifest["likelihood"]["power"])
    if not np.allclose(
        trace["log_likelihood"].values,
        trace["log_gaussian_likelihood"].values * likelihood_power,
        rtol=4.0 * np.finfo(np.float64).eps,
        atol=1e-12,
    ):
        raise ValueError(f"{location} likelihood-power identity failed.")
    target_sum = sum(
        np.asarray(trace[name].values, dtype=np.float64)
        for name in TARGET_FIELDS[1:-1]
    )
    if not np.allclose(
        trace["log_target"].values,
        target_sum,
        rtol=4.0 * np.finfo(np.float64).eps,
        atol=1e-12,
    ):
        raise ValueError(f"{location} target decomposition failed.")


def _summary_counts(trace: xr.Dataset, selected: BoolArray) -> dict[str, Any]:
    valid_values = np.asarray(trace["valid"].values, dtype=np.bool_)[selected]
    accepted_values = np.asarray(trace["accepted"].values, dtype=np.bool_)[selected]
    attempts = int(np.count_nonzero(selected))
    valid = int(np.count_nonzero(valid_values))
    accepted = int(np.count_nonzero(accepted_values))
    return {
        "attempts": attempts,
        "valid": valid,
        "accepted": accepted,
        "valid_fraction": _fraction(valid, attempts),
        "acceptance_fraction": _fraction(accepted, attempts),
        "acceptance_given_valid": _fraction(accepted, valid),
    }


def _attempt_summary(trace: xr.Dataset) -> dict[str, Any]:
    move = np.asarray(trace["move"].values).astype(str)
    coefficient_id = np.asarray(trace["coefficient_id"].values, dtype=np.int64)
    labels = np.asarray(trace.coords["fixed_parameter"].values).astype(str)
    return {
        "moves": {
            name: _summary_counts(trace, move == name)
            for name in MOVE_NAMES
        },
        "fixed_coefficients": {
            str(labels[position]): _summary_counts(
                trace,
                (move == "fixed_coefficient") & (coefficient_id == position),
            )
            for position in range(FIXED_COUNT)
        },
    }


def _validate_segment_summary(
    trace: xr.Dataset,
    summary: Mapping[str, Any],
    *,
    expected_start: int,
    expected_end: int,
) -> None:
    segment = summary.get("segment")
    if not isinstance(segment, Mapping):
        raise TypeError("Segment summary has no segment metadata.")
    expected = {
        "atomic_transitions": TRANSITIONS_PER_SEGMENT,
        "transitions_start": expected_start,
        "transitions_end": expected_end,
        "cycle_length": CYCLE_LENGTH,
        "schedule_phase_end": 0,
    }
    for key, value in expected.items():
        if int(segment[key]) != value:
            raise ValueError(f"Segment summary {key}={segment[key]!r}; expected {value}.")
    calculated = _attempt_summary(trace)
    for name in MOVE_NAMES:
        for field in ("attempts", "valid", "accepted"):
            if int(summary["moves"][name][field]) != calculated["moves"][name][field]:
                raise ValueError(f"Segment summary move count disagrees for {name}.{field}.")
    for position in range(FIXED_COUNT):
        for field in ("attempts", "valid", "accepted"):
            if (
                int(summary["fixed_coefficients"][str(position)][field])
                != list(calculated["fixed_coefficients"].values())[position][field]
            ):
                raise ValueError(
                    f"Segment fixed-coefficient count disagrees for position {position}.{field}."
                )


def _load_segment(
    directory: Path,
    *,
    chain: int,
    segment: int,
    problem: GammaBetaTreeProblem,
) -> SegmentData:
    completion = _validate_completion(directory)
    manifest = _load_json(directory / "manifest.json")
    canonical = canonical_gamma_beta_run_manifest(manifest)
    if (directory / "manifest.json").read_text(encoding="utf-8") != canonical:
        raise ValueError(f"{directory / 'manifest.json'} is not canonical.")
    summary = _load_json(directory / "summary.json")
    with xr.open_dataset(directory / "trace.nc") as opened:
        trace = opened.load()
    _validate_trace(trace, chain=chain, segment=segment, manifest=manifest)
    expected_start = segment * TRANSITIONS_PER_SEGMENT
    expected_end = (segment + 1) * TRANSITIONS_PER_SEGMENT
    global_transition = np.asarray(trace["global_transition"].values, dtype=np.int64)
    if not np.array_equal(
        global_transition,
        np.arange(expected_start + 1, expected_end + 1, dtype=np.int64),
    ):
        raise ValueError(f"Chain {chain} segment {segment} attempt coordinates are incomplete.")
    _validate_segment_summary(
        trace,
        summary,
        expected_start=expected_start,
        expected_end=expected_end,
    )
    checkpoint = load_gamma_beta_checkpoint(
        directory / "checkpoint.npz",
        problem,
        expected_run_manifest=manifest,
    )
    if (
        checkpoint.transitions_completed != expected_end
        or checkpoint.schedule_phase != 0
        or int(completion["transitions_completed"]) != expected_end
    ):
        raise ValueError(f"Chain {chain} segment {segment} checkpoint boundary is inconsistent.")
    if int(trace["k_after"].values[-1]) != checkpoint.state.k:
        raise ValueError(f"Chain {chain} segment {segment} final K disagrees with checkpoint.")
    if (
        trace.sizes["draw"]
        and int(trace["state_transition"].values[-1]) == expected_end
        and (
            int(trace["k"].values[-1]) != checkpoint.state.k
            or float(trace["root_total"].values[-1]) != checkpoint.state.root_total
            or not np.array_equal(
                trace["fixed_coefficients"].values[-1],
                checkpoint.state.fixed_coefficients,
            )
        )
    ):
        raise ValueError(
            f"Chain {chain} segment {segment} final retained state disagrees with checkpoint."
        )
    return SegmentData(
        chain=chain,
        segment=segment,
        directory=directory,
        manifest=manifest,
        summary=summary,
        completion=completion,
        trace=trace,
        checkpoint_k=checkpoint.state.k,
        checkpoint_root_total=checkpoint.state.root_total,
        checkpoint_fixed_coefficients=np.asarray(
            checkpoint.state.fixed_coefficients,
            dtype=np.float64,
        ),
    )


def _load_chain(
    run_root: Path,
    *,
    chain: int,
    problem: GammaBetaTreeProblem,
) -> ChainData:
    segments = tuple(
        _load_segment(
            _segment_directory(run_root, chain, segment),
            chain=chain,
            segment=segment,
            problem=problem,
        )
        for segment in range(SEGMENT_COUNT)
    )
    canonical = canonical_gamma_beta_run_manifest(segments[0].manifest)
    if any(canonical_gamma_beta_run_manifest(item.manifest) != canonical for item in segments[1:]):
        raise ValueError(f"Chain {chain} manifest changed across immutable segments.")
    manifest = segments[0].manifest
    if int(manifest["chain"]["initial_k"]) != EXPECTED_START_K[chain]:
        raise ValueError(f"Chain {chain} does not use the required alternating initial K.")
    if int(manifest["retention"]["warmup_transitions"]) != WARMUP_TRANSITIONS:
        raise ValueError(f"Chain {chain} warmup is not 2,800 transitions.")
    if int(manifest["retention"]["thin"]) != THIN:
        raise ValueError(f"Chain {chain} thinning is not every 14 transitions.")
    if int(manifest["schedule"]["cycle_length"]) != CYCLE_LENGTH:
        raise ValueError(f"Chain {chain} schedule is not the 14-slot profile.")

    for previous, following in pairwise(segments):
        if int(previous.trace["k_after"].values[-1]) != int(
            following.trace["k_before"].values[0]
        ):
            raise ValueError(
                f"Chain {chain} attempt K is discontinuous between segments "
                f"{previous.segment} and {following.segment}."
            )
    state_transition = np.concatenate(
        [np.asarray(item.trace["state_transition"].values, dtype=np.int64) for item in segments]
    )
    expected_draws = np.arange(
        WARMUP_TRANSITIONS,
        TRANSITIONS_PER_CHAIN + 1,
        THIN,
        dtype=np.int64,
    )
    if not np.array_equal(state_transition, expected_draws):
        raise ValueError(f"Chain {chain} retained-draw coordinates are incomplete.")
    k = np.concatenate([np.asarray(item.trace["k"].values, dtype=np.int64) for item in segments])
    root_total = np.concatenate(
        [np.asarray(item.trace["root_total"].values, dtype=np.float64) for item in segments]
    )
    fixed_coefficients = np.concatenate(
        [
            np.asarray(item.trace["fixed_coefficients"].values, dtype=np.float64)
            for item in segments
        ],
        axis=0,
    )
    target_components = {
        name: np.concatenate(
            [np.asarray(item.trace[name].values, dtype=np.float64) for item in segments]
        )
        for name in TARGET_FIELDS
    }
    return ChainData(
        chain=chain,
        segments=segments,
        manifest=manifest,
        state_transition=state_transition,
        k=k,
        root_total=root_total,
        fixed_coefficients=fixed_coefficients,
        target_components=target_components,
    )


def _chain_attempt_arrays(chain: ChainData) -> dict[str, NDArray[Any]]:
    names = (
        "global_transition",
        "move",
        "valid",
        "accepted",
        "node_id",
        "coefficient_id",
        "k_before",
        "k_after",
    )
    return {
        name: np.concatenate([np.asarray(segment.trace[name].values) for segment in chain.segments])
        for name in names
    }


def _edge_flow(attempts: Mapping[str, NDArray[Any]]) -> list[dict[str, Any]]:
    move = attempts["move"].astype(str)
    valid = attempts["valid"].astype(np.bool_)
    accepted = attempts["accepted"].astype(np.bool_)
    before = attempts["k_before"].astype(np.int64)
    rows: list[dict[str, Any]] = []
    lower = int(np.min(before))
    upper = int(np.max(before))
    for k in range(lower, upper + 1):
        upward = (move == "split") & (before == k)
        downward = (move == "merge") & (before == k + 1)
        if not np.any(upward) and not np.any(downward):
            continue
        up_attempts = int(np.count_nonzero(upward))
        up_valid = int(np.count_nonzero(upward & valid))
        up_accepted = int(np.count_nonzero(upward & accepted))
        down_attempts = int(np.count_nonzero(downward))
        down_valid = int(np.count_nonzero(downward & valid))
        down_accepted = int(np.count_nonzero(downward & accepted))
        rows.append(
            {
                "lower_k": k,
                "upper_k": k + 1,
                "up_attempts": up_attempts,
                "up_valid": up_valid,
                "up_accepted": up_accepted,
                "up_acceptance_given_valid": _fraction(up_accepted, up_valid),
                "down_attempts": down_attempts,
                "down_valid": down_valid,
                "down_accepted": down_accepted,
                "down_acceptance_given_valid": _fraction(down_accepted, down_valid),
                "realized_bidirectional_flow": min(up_accepted, down_accepted),
                "realized_total_flow": up_accepted + down_accepted,
            }
        )
    return rows


def _immediate_reversals(attempts: Mapping[str, NDArray[Any]]) -> dict[str, Any]:
    transition = attempts["global_transition"].astype(np.int64)
    move = attempts["move"].astype(str)
    accepted = attempts["accepted"].astype(np.bool_)
    node = attempts["node_id"].astype(np.int64)
    before = attempts["k_before"].astype(np.int64)
    after = attempts["k_after"].astype(np.int64)
    structural = accepted & np.isin(move, ("split", "merge"))
    positions = np.flatnonzero(structural)
    if positions.size < 2:
        return {
            "accepted_structural_events": int(positions.size),
            "adjacent_atomic_opposite_direction_reversals": 0,
            "exact_node_reversals": 0,
            "exact_node_reversal_fraction": _fraction(0, int(positions.size)),
        }
    first = positions[:-1]
    second = positions[1:]
    adjacent = transition[second] == transition[first] + 1
    opposite = move[first] != move[second]
    returns = (before[first] == after[second]) & (after[first] == before[second])
    immediate = adjacent & opposite & returns
    exact = immediate & (node[first] == node[second])
    exact_count = int(np.count_nonzero(exact))
    return {
        "accepted_structural_events": int(positions.size),
        "adjacent_atomic_opposite_direction_reversals": int(np.count_nonzero(immediate)),
        "exact_node_reversals": exact_count,
        "exact_node_reversal_fraction": _fraction(exact_count, int(positions.size)),
        "interpretation": (
            "An exact-node reversal is an accepted split followed immediately "
            "by the reverse merge, or vice versa, at the same node. With no "
            "intervening transition this restores the preceding deterministic prediction."
        ),
    }


def _first_passage(attempts: Mapping[str, NDArray[Any]], initial_k: int) -> dict[str, Any]:
    transition = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            attempts["global_transition"].astype(np.int64),
        )
    )
    k_path = np.concatenate(
        (
            np.asarray([initial_k], dtype=np.int64),
            attempts["k_after"].astype(np.int64),
        )
    )
    distance = np.abs(k_path - initial_k)
    milestones: dict[str, int | None] = {}
    for threshold in (1, 5, 10, 25, 50, 100, 200):
        hits = np.flatnonzero(distance >= threshold)
        milestones[str(threshold)] = None if hits.size == 0 else int(transition[hits[0]])
    return {
        "definition": (
            "First-passage distance is absolute displacement |K_t-K_0|. "
            "Entries give the first atomic transition reaching each distance."
        ),
        "first_transition_by_absolute_k_distance": milestones,
        "maximum_absolute_k_distance": int(np.max(distance)),
        "maximum_upward_displacement": int(np.max(k_path - initial_k)),
        "maximum_downward_displacement": int(np.max(initial_k - k_path)),
        "net_k_displacement": int(k_path[-1] - initial_k),
        "final_k": int(k_path[-1]),
        "minimum_visited_k": int(np.min(k_path)),
        "maximum_visited_k": int(np.max(k_path)),
    }


def _find_number(value: object, names: frozenset[str]) -> float | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in names and isinstance(child, (int, float)) and not isinstance(child, bool):
                result = float(child)
                if isfinite(result):
                    return result
        for child in value.values():
            found = _find_number(child, names)
            if found is not None:
                return found
    return None


def _resource_rss_bytes(segment: SegmentData) -> int | None:
    names_bytes = frozenset(
        ("maximum_rss_bytes", "max_rss_bytes", "maximum_sampled_rss_bytes")
    )
    found = _find_number(segment.summary, names_bytes)
    if found is not None:
        return int(found)
    resource_path = segment.directory / "resource_usage.json"
    if resource_path.is_file():
        resource = _load_json(resource_path)
        found = _find_number(resource, names_bytes)
        if found is not None:
            return int(found)
        found_kib = _find_number(
            resource,
            frozenset(
                (
                    "maximum_resident_set_size_kbytes",
                    "maximum_rss_kib",
                    "max_rss_kib",
                    "maxrss_kb",
                )
            ),
        )
        if found_kib is not None:
            return int(found_kib * 1024)
    return None


def _segment_performance(segment: SegmentData) -> dict[str, Any]:
    performance = segment.summary["performance"]
    output = segment.completion.get("performance", {})
    sampling = float(performance["sampling_seconds"])
    input_seconds = float(performance["input_hash_and_load_seconds"])
    setup = float(performance["problem_setup_seconds"])
    resume = float(performance["resume_validation_seconds"])
    output_seconds = float(output.get("output_before_completion_seconds", 0.0))
    io_seconds = input_seconds + resume + output_seconds
    accounted = setup + sampling + io_seconds
    rss_bytes = _resource_rss_bytes(segment)
    return {
        "input_hash_and_load_seconds": input_seconds,
        "problem_setup_seconds": setup,
        "resume_validation_seconds": resume,
        "sampling_seconds": sampling,
        "output_before_completion_seconds": output_seconds,
        "accounted_wall_seconds": accounted,
        "io_seconds": io_seconds,
        "io_fraction_of_accounted_wall": io_seconds / accounted if accounted > 0.0 else None,
        "atomic_transitions_per_sampling_second": (
            TRANSITIONS_PER_SEGMENT / sampling if sampling > 0.0 else None
        ),
        "maximum_rss_bytes": rss_bytes,
    }


def _performance_warnings(segments: tuple[SegmentData, ...]) -> dict[str, Any]:
    performance = [_segment_performance(segment) for segment in segments]
    warnings: list[str] = []
    io_bad = [
        index
        for index, values in enumerate(performance)
        if values["io_fraction_of_accounted_wall"] is not None
        and values["io_fraction_of_accounted_wall"] > 0.10
    ]
    if io_bad:
        warnings.append(f"I/O exceeded 10% of accounted wall time in segments {io_bad}.")
    warmed = np.asarray(
        [
            values["atomic_transitions_per_sampling_second"]
            for values in performance[2:]
        ],
        dtype=np.float64,
    )
    throughput_ratio = None
    falling_segments: list[int] = []
    if warmed.size:
        reference = float(warmed[0])
        throughput_ratio = float(np.min(warmed) / reference) if reference > 0.0 else None
        falling_segments = [
            index + 2
            for index, value in enumerate(warmed)
            if reference > 0.0 and value < 0.8 * reference
        ]
        if falling_segments:
            warnings.append(
                "Warmed-segment throughput fell more than 20% below segment 2 "
                f"in segments {falling_segments}."
            )
    rss_values = [values["maximum_rss_bytes"] for values in performance]
    rss_growth = None
    if all(value is not None for value in rss_values):
        first = int(rss_values[0])
        last = int(rss_values[-1])
        rss_growth = (last - first) / first if first > 0 else None
        if rss_growth is not None and rss_growth > 0.10:
            warnings.append("Recorded segment RSS grew by more than 10%.")
    else:
        warnings.append(
            "Per-segment RSS was unavailable, so the 10% RSS-growth warning could not be evaluated."
        )
    return {
        "segments": performance,
        "minimum_warmed_throughput_ratio_to_segment_2": throughput_ratio,
        "warmed_throughput_fall_segments": falling_segments,
        "rss_first_to_last_fractional_growth": rss_growth,
        "warnings": warnings,
    }


def _mobility_warnings(
    attempt_summary: Mapping[str, Any],
    edge_flow: list[dict[str, Any]],
    reversals: Mapping[str, Any],
) -> list[str]:
    warnings: list[str] = []
    for label, values in attempt_summary["fixed_coefficients"].items():
        if int(values["accepted"]) == 0:
            warnings.append(f"Fixed coefficient {label!r} had no accepted update.")
    accepted_by_edge = np.asarray(
        [row["realized_total_flow"] for row in edge_flow],
        dtype=np.float64,
    )
    total = float(np.sum(accepted_by_edge))
    if total > 0.0:
        top_three_share = float(np.sum(np.sort(accepted_by_edge)[-3:]) / total)
        if top_three_share > 0.75:
            warnings.append(
                "More than 75% of accepted structural movement was concentrated "
                "on three K edges."
            )
    exact = int(reversals["exact_node_reversals"])
    accepted = int(reversals["accepted_structural_events"])
    if exact >= 10 and exact / accepted >= 0.25:
        warnings.append(
            "At least 25% of accepted structural events participated in repeated "
            "immediate exact-node reversals."
        )
    return warnings


def _chain_report(chain: ChainData) -> dict[str, Any]:
    attempts = _chain_attempt_arrays(chain)
    move = attempts["move"].astype(str)
    coefficient = attempts["coefficient_id"].astype(np.int64)
    combined_trace = xr.Dataset(
        {
            name: ("attempt", values)
            for name, values in attempts.items()
            if name
            in {
                "move",
                "valid",
                "accepted",
                "coefficient_id",
            }
        },
        coords={
            "fixed_parameter": chain.segments[0].trace.coords[
                "fixed_parameter"
            ].values
        },
    )
    attempt_summary = {
        "moves": {
            name: _summary_counts(combined_trace, move == name)
            for name in MOVE_NAMES
        },
        "fixed_coefficients": {
            str(
                chain.segments[0].trace.coords["fixed_parameter"].values[position]
            ): _summary_counts(
                combined_trace,
                (move == "fixed_coefficient") & (coefficient == position),
            )
            for position in range(FIXED_COUNT)
        },
    }
    edge_flow = _edge_flow(attempts)
    reversals = _immediate_reversals(attempts)
    performance = _performance_warnings(chain.segments)
    mobility_warnings = _mobility_warnings(attempt_summary, edge_flow, reversals)
    labels = np.asarray(
        chain.segments[0].trace.coords["fixed_parameter"].values
    ).astype(str)
    return {
        "chain": chain.chain,
        "chain_id": chain.manifest["chain"]["id"],
        "initial_k": int(chain.manifest["chain"]["initial_k"]),
        "seed": chain.manifest["seed"],
        "segments_completed": len(chain.segments),
        "attempts": attempt_summary,
        "edge_flow": edge_flow,
        "immediate_reversals": reversals,
        "first_passage_and_displacement": _first_passage(
            attempts,
            int(chain.manifest["chain"]["initial_k"]),
        ),
        "retained_state_summaries": {
            "k": _distribution(chain.k),
            "root_total": _distribution(chain.root_total),
            "fixed_coefficients": {
                str(label): _distribution(chain.fixed_coefficients[:, position])
                for position, label in enumerate(labels)
            },
            "target_components": {
                name: _distribution(values)
                for name, values in chain.target_components.items()
            },
        },
        "segments": [
            {
                "segment": segment.segment,
                "transitions_start": segment.segment * TRANSITIONS_PER_SEGMENT,
                "transitions_end": (segment.segment + 1) * TRANSITIONS_PER_SEGMENT,
                "attempts": _attempt_summary(segment.trace),
                "retained_draws": int(segment.trace.sizes["draw"]),
                "k": {
                    "minimum": int(
                        min(
                            np.min(segment.trace["k_before"].values),
                            np.min(segment.trace["k_after"].values),
                        )
                    ),
                    "maximum": int(
                        max(
                            np.max(segment.trace["k_before"].values),
                            np.max(segment.trace["k_after"].values),
                        )
                    ),
                    "final": segment.checkpoint_k,
                },
                "checkpoint": {
                    "root_total": segment.checkpoint_root_total,
                    "fixed_coefficients": segment.checkpoint_fixed_coefficients.tolist(),
                },
                "performance": performance["segments"][segment.segment],
            }
            for segment in chain.segments
        ],
        "performance": performance,
        "warnings": [*performance["warnings"], *mobility_warnings],
    }


def _site_groups(dataset: xr.Dataset) -> dict[str, IntArray]:
    if "site" in dataset and dataset["site"].dims == ("nmeasure",):
        values = np.asarray(dataset["site"].values).astype(str)
    elif "nmeasure" in dataset.coords:
        values = np.asarray(
            [str(value).split("|", maxsplit=1)[0] for value in dataset["nmeasure"].values]
        )
    else:
        return {}
    return {
        label: np.flatnonzero(values == label).astype(np.int64)
        for label in sorted(set(values.tolist()))
    }


def _validate_flux_auxiliaries(
    dataset: xr.Dataset,
) -> tuple[FloatArray, FloatArray, dict[str, FloatArray]] | None:
    required = ("prior_flux", "grid_cell_area", "country_fraction")
    available = [name in dataset for name in required]
    if not any(available):
        return None
    if not all(available):
        missing = [name for name, present in zip(required, available, strict=True) if not present]
        raise ValueError(
            "Frozen input has a partial physical-flux auxiliary contract; "
            f"missing={missing}."
        )
    if set(dataset["prior_flux"].dims) != {"lat", "lon"}:
        raise ValueError("prior_flux must have dimensions (lat, lon).")
    if set(dataset["grid_cell_area"].dims) != {"lat", "lon"}:
        raise ValueError("grid_cell_area must have dimensions (lat, lon).")
    if set(dataset["country_fraction"].dims) != {"country", "lat", "lon"}:
        raise ValueError("country_fraction must have dimensions (country, lat, lon).")
    prior = np.asarray(
        dataset["prior_flux"].transpose("lat", "lon").values,
        dtype=np.float64,
    )
    area = np.asarray(
        dataset["grid_cell_area"].transpose("lat", "lon").values,
        dtype=np.float64,
    )
    countries = np.asarray(dataset.coords["country"].values).astype(str)
    if tuple(countries.tolist()) != ("GBR", "IRL"):
        raise ValueError("country_fraction labels/order must be exactly GBR, IRL.")
    fractions = np.asarray(
        dataset["country_fraction"].transpose("country", "lat", "lon").values,
        dtype=np.float64,
    )
    if (
        np.any(~np.isfinite(prior))
        or np.any(~np.isfinite(area))
        or np.any(~np.isfinite(fractions))
        or np.any(area <= 0.0)
        or np.any(fractions < 0.0)
        or np.any(fractions > 1.0)
    ):
        raise ValueError("Physical-flux auxiliaries contain invalid values.")
    units = str(dataset["prior_flux"].attrs.get("units", "")).lower()
    normalized_units = re.sub(r"[\s_]+", " ", units).strip()
    accepted_units = (
        "mol m-2 s-1",
        "mol m^-2 s^-1",
        "mol/m2/s",
        "mol m**-2 s**-1",
    )
    if normalized_units not in accepted_units:
        raise ValueError(
            "prior_flux units must explicitly denote mol m-2 s-1; "
            f"found {dataset['prior_flux'].attrs.get('units')!r}."
        )
    return (
        prior.reshape(-1, order="C"),
        area.reshape(-1, order="C"),
        {
            str(label): fractions[position].reshape(-1, order="C")
            for position, label in enumerate(countries)
        },
    )


def _draw_state(
    trace: xr.Dataset,
    draw: int,
    problem: GammaBetaTreeProblem,
) -> Any:
    frontier_mask = np.asarray(trace["frontier_active"].values[draw], dtype=np.bool_)
    split_mask = np.asarray(trace["split_active"].values[draw], dtype=np.bool_)
    frontier = DyadicFrontier(
        tuple(
            int(value)
            for value in np.asarray(
                trace["frontier_node_id"].values[draw][frontier_mask],
                dtype=np.int64,
            )
        )
    )
    state = build_gamma_beta_tree_state(
        problem,
        frontier=frontier,
        root_total=float(trace["root_total"].values[draw]),
        active_fractions=np.asarray(
            trace["split_fraction"].values[draw][split_mask],
            dtype=np.float64,
        ),
        fixed_coefficients=np.asarray(
            trace["fixed_coefficients"].values[draw],
            dtype=np.float64,
        ),
    )
    for name in TARGET_FIELDS:
        if not np.isclose(
            getattr(state, name),
            float(trace[name].values[draw]),
            rtol=4.0 * np.finfo(np.float64).eps,
            atol=1e-12,
        ):
            raise ValueError(f"Rebuilt retained state disagrees with trace field {name}.")
    return state


def _scientific_series(
    chains: list[ChainData],
    dataset: xr.Dataset,
    problem: GammaBetaTreeProblem,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Rebuild retained states and calculate prediction/physical-flux series."""
    site_groups = _site_groups(dataset)
    flux_aux = _validate_flux_auxiliaries(dataset)
    draw_count = chains[0].state_transition.size
    names = (
        "prediction_mean",
        "prediction_rmse",
        "prediction_mae",
        "dynamic_prediction_mean",
        "fixed_prediction_mean",
        *(f"prediction_site_mean::{label}" for label in site_groups),
    )
    series = {
        name: np.empty((CHAIN_COUNT, draw_count), dtype=np.float64)
        for name in names
    }
    if flux_aux is not None:
        for name in (
            "native_grid_flux_total_Gg_CH4_per_year",
            "GBR_flux_total_Gg_CH4_per_year",
            "IRL_flux_total_Gg_CH4_per_year",
        ):
            series[name] = np.empty((CHAIN_COUNT, draw_count), dtype=np.float64)

    nominal = problem.prior.nominal_cell_mass
    scaling_sum = np.zeros(nominal.size, dtype=np.float64)
    flux_sum = np.zeros(nominal.size, dtype=np.float64)
    total_draws = 0
    flux_factor = CH4_MOLAR_MASS_G_PER_MOL * SECONDS_PER_YEAR / 1.0e9
    for chain in chains:
        output_draw = 0
        for segment in chain.segments:
            trace = segment.trace
            for draw in range(int(trace.sizes["draw"])):
                state = _draw_state(trace, draw, problem)
                prediction = state.prediction
                residual = prediction - problem.observations
                series["prediction_mean"][chain.chain, output_draw] = float(
                    np.mean(prediction)
                )
                series["prediction_rmse"][chain.chain, output_draw] = float(
                    np.sqrt(np.mean(np.square(residual)))
                )
                series["prediction_mae"][chain.chain, output_draw] = float(
                    np.mean(np.abs(residual))
                )
                series["dynamic_prediction_mean"][chain.chain, output_draw] = float(
                    np.mean(state.dynamic_prediction)
                )
                series["fixed_prediction_mean"][chain.chain, output_draw] = float(
                    np.mean(state.fixed_prediction)
                )
                for label, positions in site_groups.items():
                    series[f"prediction_site_mean::{label}"][
                        chain.chain, output_draw
                    ] = float(np.mean(prediction[positions]))

                cell_mass = render_cell_mass(problem, state)
                scaling = cell_mass / nominal
                scaling_sum += scaling
                if flux_aux is not None:
                    prior_flux, area, country_fractions = flux_aux
                    cell_flux_gg_year = prior_flux * area * scaling * flux_factor
                    flux_sum += prior_flux * scaling
                    series["native_grid_flux_total_Gg_CH4_per_year"][
                        chain.chain, output_draw
                    ] = float(np.sum(cell_flux_gg_year))
                    for country, fraction in country_fractions.items():
                        series[f"{country}_flux_total_Gg_CH4_per_year"][
                            chain.chain, output_draw
                        ] = float(np.sum(cell_flux_gg_year * fraction))
                output_draw += 1
                total_draws += 1
        if output_draw != draw_count:
            raise ValueError(f"Chain {chain.chain} scientific draw count is incomplete.")

    summary: dict[str, Any] = {
        "definition": (
            "Prediction metrics are deterministic retained-state forward predictions, "
            "not posterior predictive draws with observation noise. Root total is the "
            "normalized area-weighted inner mass total. Native scaling is rendered "
            "cell mass divided by normalized nominal weight."
        ),
        "observation_space": {
            "units": str(dataset["mf"].attrs.get("units", "frozen mf units")),
            "metrics": {
                name: {
                    "pooled": _distribution(values.reshape(-1)),
                    "by_chain": [
                        _distribution(values[chain]) for chain in range(CHAIN_COUNT)
                    ],
                }
                for name, values in series.items()
                if "prediction" in name
            },
        },
        "flux_space": {
            "available": flux_aux is not None,
            "prior_flux_role": (
                "Reference field multiplied by sampled native scaling; it is not flux truth."
            ),
            "native_scaling_posterior_mean": (
                scaling_sum / total_draws
            ).reshape(problem.tree.shape, order="C").tolist(),
        },
    }
    if flux_aux is None:
        summary["flux_space"]["reason"] = (
            "prior_flux, grid_cell_area, and country_fraction auxiliaries were absent."
        )
    else:
        summary["flux_space"].update(
            {
                "units": "Gg CH4 yr-1",
                "conversion": (
                    "sum(prior_flux [mol m-2 s-1] * grid_cell_area [m2] * "
                    "native_scaling * country_fraction) * 16.0425 g mol-1 * "
                    "365 d yr-1 / 1e9 g Gg-1"
                ),
                "totals": {
                    name: {
                        "pooled": _distribution(values.reshape(-1)),
                        "by_chain": [
                            _distribution(values[chain])
                            for chain in range(CHAIN_COUNT)
                        ],
                    }
                    for name, values in series.items()
                    if "flux_total" in name
                },
                "native_flux_posterior_mean_mol_m2_s": (
                    flux_sum / total_draws
                ).reshape(problem.tree.shape, order="C").tolist(),
            }
        )
    return summary, series


def _finite_or_none(value: float) -> float | None:
    return value if isfinite(value) else None


def _xarray_scalar(value: Any) -> float:
    """Extract the scalar returned by an ArviZ diagnostic."""
    if isinstance(value, xr.Dataset):
        return float(value["value"].item())
    if isinstance(value, xr.DataArray):
        return float(value.item())
    return float(value)


def _convergence_metric(values: FloatArray) -> dict[str, Any]:
    if values.shape[0] != CHAIN_COUNT or values.shape[1] < 4:
        raise ValueError("Convergence series must contain four chains and at least four draws.")
    dataset = xr.Dataset({"value": (("chain", "draw"), values)})
    with np.errstate(all="ignore"):
        rhat = _xarray_scalar(az.rhat(dataset, method="rank"))
        bulk = _xarray_scalar(az.ess(dataset, method="bulk"))
        tail = _xarray_scalar(az.ess(dataset, method="tail"))
        mcse = _xarray_scalar(az.mcse(dataset, method="mean"))
    sd = float(np.std(values.reshape(-1), ddof=1))
    mcse_over_sd = mcse / sd if sd > 0.0 else None
    return {
        "chains": int(values.shape[0]),
        "draws_per_chain": int(values.shape[1]),
        "rank_normalized_split_rhat": _finite_or_none(rhat),
        "bulk_ess": _finite_or_none(bulk),
        "tail_ess": _finite_or_none(tail),
        "mean_mcse": _finite_or_none(mcse),
        "pooled_sd": _finite_or_none(sd),
        "mcse_over_sd": (
            _finite_or_none(mcse_over_sd) if mcse_over_sd is not None else None
        ),
        "promotion_thresholds_passed": bool(
            isfinite(rhat)
            and rhat <= 1.01
            and isfinite(bulk)
            and bulk >= 400
            and isfinite(tail)
            and tail >= 400
            and mcse_over_sd is not None
            and isfinite(mcse_over_sd)
            and mcse_over_sd <= 0.05
        ),
    }


def _convergence(
    chains: list[ChainData],
    scientific_series: Mapping[str, FloatArray],
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    labels = np.asarray(
        chains[0].segments[0].trace.coords["fixed_parameter"].values
    ).astype(str)
    matrices: dict[str, FloatArray] = {
        "k": np.stack([chain.k for chain in chains]).astype(np.float64),
        "root_total": np.stack([chain.root_total for chain in chains]),
        "log_target": np.stack(
            [chain.target_components["log_target"] for chain in chains]
        ),
    }
    for position, label in enumerate(labels):
        matrices[f"outer_coefficient::{label}"] = np.stack(
            [chain.fixed_coefficients[:, position] for chain in chains]
        )
    matrices.update(scientific_series)
    metrics = {
        name: _convergence_metric(values)
        for name, values in sorted(matrices.items())
    }
    failed = [
        name
        for name, values in metrics.items()
        if not values["promotion_thresholds_passed"]
    ]
    return (
        {
            "definition": (
                "ArviZ rank-normalized split R-hat, bulk/tail effective sample "
                "size (ESS), and mean Monte Carlo standard error (MCSE) use all "
                "801 globally post-warmup retained states per chain. MCSE/SD is "
                "mean MCSE divided by the pooled retained-state standard deviation."
            ),
            "promotion_thresholds": {
                "rank_normalized_split_rhat_maximum": 1.01,
                "bulk_ess_minimum": 400,
                "tail_ess_minimum": 400,
                "mcse_over_sd_maximum": 0.05,
            },
            "metrics": metrics,
            "all_numeric_thresholds_passed": not failed,
            "failed_metrics": failed,
            "scientific_promotion_allowed": False,
            "scientific_promotion_reason": (
                "The planned 1,000-cycle Stage-3 profile is a stability and "
                "mobility diagnostic, not a converged scientific inversion. "
                "Numerical thresholds alone cannot override the required "
                "overlapping K distributions and traversal/round-trip evidence."
            ),
        },
        matrices,
    )


def _plot_k(path: Path, chains: list[ChainData]) -> None:
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for chain in chains:
        axis.plot(
            chain.state_transition / CYCLE_LENGTH,
            chain.k,
            lw=1.0,
            label=f"chain {chain.chain}, start K={chain.manifest['chain']['initial_k']}",
        )
    axis.set_xlabel("Completed 14-slot cycles")
    axis.set_ylabel("Active tree regions, K")
    axis.set_title("Gamma–Beta Stage-3 retained K trajectories")
    axis.legend(ncol=2, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_rhat(path: Path, convergence: Mapping[str, Any]) -> None:
    rows = [
        (name, values["rank_normalized_split_rhat"])
        for name, values in convergence["metrics"].items()
        if values["rank_normalized_split_rhat"] is not None
    ]
    rows.sort(key=lambda item: float(item[1]), reverse=True)
    rows = rows[:25]
    figure, axis = plt.subplots(
        figsize=(10, max(5, 0.3 * len(rows))),
        constrained_layout=True,
    )
    positions = np.arange(len(rows))
    axis.barh(positions, [float(value) for _, value in rows], color="#4477AA")
    axis.set_yticks(positions, [name for name, _ in rows])
    axis.invert_yaxis()
    axis.axvline(1.01, color="#CC3311", ls="--", lw=1.0)
    axis.set_xlabel("Rank-normalized split R-hat")
    axis.set_title("Largest Stage-3 convergence diagnostics")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_native_scaling(
    path: Path,
    scientific: Mapping[str, Any],
    dataset: xr.Dataset,
) -> None:
    scaling = np.asarray(
        scientific["flux_space"]["native_scaling_posterior_mean"],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    if "lat" in dataset.coords and "lon" in dataset.coords:
        image = axis.pcolormesh(
            np.asarray(dataset["lon"].values, dtype=np.float64),
            np.asarray(dataset["lat"].values, dtype=np.float64),
            scaling,
            shading="auto",
            cmap="viridis",
        )
        axis.set_xlabel("Longitude [degrees east]")
        axis.set_ylabel("Latitude [degrees north]")
    else:
        image = axis.imshow(scaling, origin="lower", cmap="viridis", aspect="auto")
        axis.set_xlabel("Native longitude index")
        axis.set_ylabel("Native latitude index")
    figure.colorbar(image, ax=axis, label="Retained-state mean scaling [dimensionless]")
    axis.set_title("Stage-3 mean native-grid flux scaling")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _format_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _markdown_report(report: Mapping[str, Any]) -> str:
    chain_reports = report["chains"]
    convergence = report["convergence"]
    metrics = list(convergence["metrics"].items())
    metrics.sort(
        key=lambda item: (
            item[1]["rank_normalized_split_rhat"] is not None,
            item[1]["rank_normalized_split_rhat"] or -np.inf,
        ),
        reverse=True,
    )
    warnings = [
        f"Chain {chain['chain']}: {warning}"
        for chain in chain_reports
        for warning in chain["warnings"]
    ]
    lines = [
        "# Gamma–Beta RJMCMC Stage-3 HPC diagnostic",
        "",
        "## What Was Tested",
        "",
        (
            "Four independent fixed-direction Gamma–Beta tree RJMCMC chains were "
            "run for 1,000 complete 14-slot cycles (14,000 atomic transitions) "
            "each, in ten immutable 100-cycle segments. Starts alternated between "
            "`K=50` and `K=250`; global warmup was 200 cycles and retained states "
            "were thinned every cycle. The schedule used two independently mixed "
            "split/merge opportunities, one Gamma-root refresh, five active-fraction "
            "refreshes, and one update for each of six outer coefficients."
        ),
        "",
        (
            "The likelihood is independent Gaussian with fixed row-specific errors, "
            "`R = diag(mf_error_i²)`. These errors are supplied by the frozen input "
            "and are not inferred. The model has a fixed row-aligned boundary "
            "contribution and six inferred outer-region scaling coefficients."
        ),
        "",
        (
            f"All 40 immutable segment bundles under `{report['run_root']}` passed "
            "completion SHA-256, canonical-manifest, exact-checkpoint, target, "
            "attempt-coordinate, and retained-draw-continuity validation."
        ),
        "",
        "## Terminology And Truth",
        "",
        (
            "**Truth:** this real-data diagnostic has no flux truth. Observed CH₄ "
            "in the frozen `mf` product is the observation-space comparator. "
            "The frozen `prior_flux` is a flux-space reference field used to turn "
            "sampled native scaling into physical totals; it is not truth."
        ),
        "",
        (
            "**Deterministic retained-state prediction** is the model forward value "
            "at one retained state. It is not a posterior predictive draw and adds "
            "no observation noise. **First-passage distance** is `|K_t-K_0|`; the "
            "reported coordinate is the first atomic transition at which a distance "
            "threshold is reached. **Exact-node reversal** is an accepted split "
            "immediately undone by a merge at the same tree node, or the reverse."
        ),
        "",
        "## What Happened",
        "",
        (
            "The run is operationally complete, but its convergence diagnostics are "
            f"not a scientific promotion decision. {len(convergence['failed_metrics'])} "
            "reported scalar series failed at least one numerical promotion threshold. "
            "Regardless of those values, the planned 1,000-cycle profile is a sampler "
            "stability and mobility diagnostic, not a converged inversion."
        ),
        "",
        "## Key Results",
        "",
        (
            "The following table summarizes K mobility and structural outcomes. "
            "Acceptance is accepted structural proposals divided by valid structural "
            "proposals; I/O is input hashing/loading, resume validation, and durable "
            "output as a percentage of accounted wall time."
        ),
        "",
        (
            "| Chain | Start K | Final K | Visited K | Structural accepted/valid | "
            "Exact reversals | Max |K−K₀| | Net ΔK | Mean I/O % |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for chain in chain_reports:
        passage = chain["first_passage_and_displacement"]
        split = chain["attempts"]["moves"]["split"]
        merge = chain["attempts"]["moves"]["merge"]
        accepted = int(split["accepted"]) + int(merge["accepted"])
        valid = int(split["valid"]) + int(merge["valid"])
        io_values = [
            segment["performance"]["io_fraction_of_accounted_wall"]
            for segment in chain["segments"]
        ]
        mean_io = float(np.mean([value for value in io_values if value is not None]))
        lines.append(
            f"| {chain['chain']} | {chain['initial_k']} | {passage['final_k']} | "
            f"{passage['minimum_visited_k']}–{passage['maximum_visited_k']} | "
            f"{_format_number(_fraction(accepted, valid))} | "
            f"{chain['immediate_reversals']['exact_node_reversals']} | "
            f"{passage['maximum_absolute_k_distance']} | "
            f"{passage['net_k_displacement']:+d} | {100.0 * mean_io:.1f} |"
        )
    lines.extend(
        [
            "",
            (
                "The next table lists multi-chain diagnostics for every required "
                "state, deterministic observation-space, and flux-space summary. "
                "ESS is in retained draws; mean MCSE has the metric's own units, "
                "while MCSE/SD is dimensionless."
            ),
            "",
            "| Metric | R-hat | Bulk ESS | Tail ESS | MCSE/SD | Numeric gate |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for name, values in metrics:
        lines.append(
            f"| `{name}` | "
            f"{_format_number(values['rank_normalized_split_rhat'])} | "
            f"{_format_number(values['bulk_ess'], 1)} | "
            f"{_format_number(values['tail_ess'], 1)} | "
            f"{_format_number(values['mcse_over_sd'])} | "
            f"{'pass' if values['promotion_thresholds_passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Performance And Warnings",
            "",
            (
                "Segment timing is separated into frozen-input/hash time, problem "
                "setup, sampling, resume validation, and durable output. Per-segment "
                "RSS is used only when an external `resource_usage.json` or equivalent "
                "summary field was present; otherwise the RSS-growth gate is explicitly "
                "unevaluated."
            ),
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No configured performance or mobility warning fired.")
    lines.extend(
        [
            "",
            "## Primary Figures",
            "",
            "![Retained K trajectories](figure_1_k_trajectories.png)",
            "",
            (
                "**Figure 1.** Retained active-region count `K` against completed "
                "14-slot cycles for all chains. Low/high-start overlap and repeated "
                "movement across the same K range are required before scientific promotion."
            ),
            "",
            "![Convergence diagnostics](figure_2_rhat.png)",
            "",
            (
                "**Figure 2.** Largest rank-normalized split R-hat values. The dashed "
                "line is the 1.01 numerical threshold; this diagnostic-only run is not "
                "promoted even if a metric lies left of the line."
            ),
            "",
            "![Native scaling](figure_3_native_scaling.png)",
            "",
            (
                "**Figure 3.** Pooled retained-state mean native-grid scaling "
                "(dimensionless), reconstructed as cell mass divided by normalized "
                "nominal area weight. This is a flux-space state summary, not flux truth."
            ),
            "",
            "## Interpretation",
            "",
            (
                "These outputs diagnose durability, local split/merge mobility, "
                "continuous-kernel activity, computational stability, and whether four "
                "short chains begin to overlap. Physical GBR, IRL, and whole-inner-grid "
                "totals use `prior_flux × sampled scaling × grid-cell area` and are "
                "reported in Gg CH₄ yr⁻¹. They must not be interpreted as final national "
                "emissions estimates."
            ),
            "",
            "## Follow-Up",
            "",
            (
                "Profile every warning before lengthening the run. Scientific promotion "
                "requires R-hat ≤1.01, bulk and tail ESS ≥400, MCSE/SD ≤0.05, overlapping "
                "low/high-start K distributions, and repeated traversal or round-trip "
                "evidence appropriate to posterior support. If local fixed-direction "
                "mobility remains slow, compare fixed-K tree rearrangements or the full "
                "tiling move set."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _save_series(path: Path, matrices: Mapping[str, FloatArray]) -> None:
    arrays = {
        re.sub(r"[^A-Za-z0-9_]+", "__", name).strip("_"): np.asarray(
            values,
            dtype=np.float64,
        )
        for name, values in matrices.items()
    }
    with path.open("wb") as handle:
        np.savez_compressed(handle, **cast(dict[str, Any], arrays))


def _validate_chain_identities(
    chain_ids: Sequence[str],
    initial_k: Sequence[int],
    initial_hashes: Sequence[str],
) -> None:
    """Validate independent identities while allowing repeated prescribed starts."""
    if not (
        len(chain_ids) == len(initial_k) == len(initial_hashes) == CHAIN_COUNT
    ):
        raise ValueError("Stage-3 identity vectors must contain exactly four chains.")
    if len(set(chain_ids)) != CHAIN_COUNT:
        raise ValueError("Stage-3 chains require four distinct chain IDs.")

    hash_by_initial_k: dict[int, str] = {}
    for k_value, state_hash in zip(initial_k, initial_hashes, strict=True):
        previous = hash_by_initial_k.setdefault(k_value, state_hash)
        if previous != state_hash:
            raise ValueError("Repeated Stage-3 initial K values require identical state hashes.")
    if len(set(hash_by_initial_k.values())) != len(hash_by_initial_k):
        raise ValueError("Distinct Stage-3 initial K values require distinct state hashes.")


def run_report(arguments: argparse.Namespace) -> dict[str, Any]:
    """Validate Stage 3, write the report bundle, and return the report mapping."""
    _validate_paths(arguments)
    _validate_exact_layout(arguments.run_root)
    first_manifest = _load_json(
        _segment_directory(arguments.run_root, 0, 0) / "manifest.json"
    )
    dataset, adapter = _rebuild_problem(
        arguments.input,
        first_manifest,
        concentration=arguments.concentration,
    )
    chains = [
        _load_chain(arguments.run_root, chain=chain, problem=adapter.problem)
        for chain in range(CHAIN_COUNT)
    ]
    scientific_manifests = {_scientific_manifest(chain.manifest) for chain in chains}
    if len(scientific_manifests) != 1:
        raise ValueError("Scientific manifest settings differ across chains.")
    chain_ids = [str(chain.manifest["chain"]["id"]) for chain in chains]
    initial_hashes = [
        str(chain.manifest["chain"]["initial_state_sha256"]) for chain in chains
    ]
    initial_k = [int(chain.manifest["chain"]["initial_k"]) for chain in chains]
    _validate_chain_identities(chain_ids, initial_k, initial_hashes)
    seeds = [chain.manifest["seed"] for chain in chains]
    if any(seed is None for seed in seeds) or len(set(seeds)) != CHAIN_COUNT:
        raise ValueError("Stage-3 chains require four distinct explicit seeds.")

    chain_reports = [_chain_report(chain) for chain in chains]
    scientific, scientific_series = _scientific_series(
        chains,
        dataset,
        adapter.problem,
    )
    convergence, matrices = _convergence(chains, scientific_series)
    report: dict[str, Any] = {
        "schema": "openghg_inversions.gamma_beta_stage3_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "scientific_convergence_claim": False,
        "run_root": str(arguments.run_root.resolve()),
        "frozen_input": {
            "path": str(arguments.input.resolve()),
            "sha256": _sha256_file(arguments.input),
            "identifier": first_manifest["inputs"]["frozen_native_dataset"][
                "identifier"
            ],
        },
        "settings": {
            "chains": CHAIN_COUNT,
            "segments_per_chain": SEGMENT_COUNT,
            "cycles_per_segment": CYCLES_PER_SEGMENT,
            "cycles_per_chain": SEGMENT_COUNT * CYCLES_PER_SEGMENT,
            "transitions_per_chain": TRANSITIONS_PER_CHAIN,
            "warmup_transitions": WARMUP_TRANSITIONS,
            "thin_transitions": THIN,
            "retained_draws_per_chain": int(chains[0].k.size),
            "initial_k": list(EXPECTED_START_K),
            "fixed_coefficients": FIXED_COUNT,
            "concentration": arguments.concentration,
            "likelihood": (
                "Independent Gaussian with R=diag(mf_error_i^2); "
                "row-specific errors are fixed, not inferred."
            ),
            "nominal_weight": first_manifest["nominal_weight"],
            "code_revision": first_manifest["code_revision"],
        },
        "validation": {
            "segment_bundles_validated": CHAIN_COUNT * SEGMENT_COUNT,
            "completion_hashes": "passed",
            "canonical_manifests": "passed",
            "exact_checkpoint_reload": "passed",
            "attempt_coordinate_continuity": "passed",
            "retained_draw_continuity": "passed",
            "target_decomposition": "passed",
        },
        "truth": {
            "flux_truth": None,
            "observation_space_comparator": (
                "Frozen May 2014 mf observations; not a flux truth."
            ),
            "flux_reference": (
                "Frozen prior_flux over the native inner grid; reference only, not truth."
            ),
        },
        "chains": chain_reports,
        "scientific_summaries": scientific,
        "convergence": convergence,
    }

    arguments.output_directory.mkdir()
    _write_json(arguments.output_directory / "stage3_analysis.json", report)
    _save_series(arguments.output_directory / "stage3_series.npz", matrices)
    _plot_k(arguments.output_directory / "figure_1_k_trajectories.png", chains)
    _plot_rhat(arguments.output_directory / "figure_2_rhat.png", convergence)
    _plot_native_scaling(
        arguments.output_directory / "figure_3_native_scaling.png",
        scientific,
        dataset,
    )
    (arguments.output_directory / "stage3_report.md").write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    output_names = (
        "stage3_analysis.json",
        "stage3_series.npz",
        "stage3_report.md",
        "figure_1_k_trajectories.png",
        "figure_2_rhat.png",
        "figure_3_native_scaling.png",
    )
    artifact_manifest = {
        "schema": "openghg_inversions.gamma_beta_stage3_report_artifacts_v1",
        "artifacts": {
            name: {
                "sha256": _sha256_file(arguments.output_directory / name),
                "bytes": (arguments.output_directory / name).stat().st_size,
            }
            for name in output_names
        },
    }
    _write_json(
        arguments.output_directory / "stage3_report_manifest.json",
        artifact_manifest,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--concentration",
        type=float,
        default=2.0,
        help=(
            "Constant Beta concentration used to rebuild the frozen problem; "
            "Stage-3 default is 2 and fingerprint validation rejects disagreement."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_report(arguments)
    print(
        json.dumps(
            {
                "output_directory": str(arguments.output_directory.resolve()),
                "segments_validated": report["validation"][
                    "segment_bundles_validated"
                ],
                "diagnostic_only": report["diagnostic_only"],
                "failed_convergence_metrics": len(
                    report["convergence"]["failed_metrics"]
                ),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
