"""Run a durable Gamma--Beta RJMCMC segment from one frozen native-grid input.

The input must be a single explicitly frozen NetCDF dataset.  By default it
contains ``fp_x_flux(nmeasure, lat, lon)``, ``mf(nmeasure)``,
``mf_error(nmeasure)``, strictly positive ``nominal_weight(lat, lon)``,
``outer_design(nmeasure, outer_region)``, and the fixed concentration offset
``YaprioriBC(nmeasure)``.  Variable names are configurable, but none of these
scientific inputs is discovered, guessed, or silently floored.

``--cycles`` is the recommended segment-length interface.  With the default
fixed-``K`` topology settings, five fraction refreshes, and six outer
coefficients, one cycle is exactly 16 atomic transitions: two mixed
split/merge opportunities, one relocation, one bounded subtree retile, one
root refresh, five fraction refreshes, and six fixed-coefficient updates.
``--iterations`` is available for an explicitly requested partial-cycle
checkpoint.

Each successful segment creates a new output directory containing an
immutable run manifest, a durable exact-continuation checkpoint, a labelled
NetCDF trace, a compact JSON summary, and a completion marker written last.
Resume reconstructs the problem and manifest from the frozen input before
loading the checkpoint, so scientific or schedule mismatches are rejected.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    GammaBetaRHIMEAdapterResult,
    gamma_beta_problem_from_rhime_inputs,
    initialize_gamma_beta_state,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_compound_sampling import (
    GammaBetaCompoundConfig,
    GammaBetaCompoundKernelSettings,
    GammaBetaCompoundSamplingResult,
    continue_gamma_beta_compound,
    sample_gamma_beta_compound,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_io import (
    build_gamma_beta_run_manifest,
    canonical_gamma_beta_run_manifest,
    gamma_beta_compound_trace_to_dataset,
    load_gamma_beta_checkpoint,
    save_gamma_beta_checkpoint,
)
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings

FloatArray = NDArray[np.float64]
NetCDFEngine = Literal["h5netcdf", "netcdf4", "scipy"]

MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_FILENAME = "checkpoint.npz"
TRACE_FILENAME = "trace.nc"
SUMMARY_FILENAME = "summary.json"
COMPLETION_FILENAME = "complete.json"
PARIS_OBSERVATIONS = 1_382
PARIS_GRID_SHAPE = (183, 128)
PARIS_OUTER_COEFFICIENTS = 6
FRACTION_REFRESH_SLOTS = 5
RELOCATION_SLOTS = 1
SUBTREE_RETILE_SLOTS = 1
MAX_SUBTREE_LEAVES = 8
SPLIT_DIRECTION_PROBABILITY = 0.5
_CLOSURE_RTOL = 1.0e-12
_CLOSURE_ATOL = 1.0e-12


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it twice."""
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Flush one directory entry where the filesystem supports it."""
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
    except OSError as error:
        if error.errno in unsupported:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write and flush one UTF-8 text artifact."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_trace(
    dataset: xr.Dataset,
    path: Path,
    *,
    engine: NetCDFEngine,
) -> None:
    """Atomically write and flush one NetCDF trace."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        dataset.to_netcdf(temporary, engine=engine)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_output_backend(
    parent: Path,
    *,
    engine: NetCDFEngine,
) -> None:
    """Write and reopen a tiny NetCDF on the selected output filesystem."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".gamma-beta-netcdf-preflight-",
        suffix=".nc",
        dir=parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        probe = xr.Dataset({"probe": ("item", np.asarray([1.0]))})
        probe.to_netcdf(temporary, engine=engine)
        probe.close()
        with xr.open_dataset(temporary, engine=engine) as reopened:
            if reopened["probe"].values.tolist() != [1.0]:
                raise RuntimeError("NetCDF backend preflight did not round-trip probe data.")
    finally:
        temporary.unlink(missing_ok=True)


def _validate_resume_bundle(checkpoint_path: Path) -> None:
    """Require a completed driver bundle and verify all recorded file hashes."""
    completion_path = checkpoint_path.parent / COMPLETION_FILENAME
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint is not a file: {checkpoint_path}")
    if not completion_path.is_file():
        raise ValueError(f"Resume checkpoint has no completed segment marker: {completion_path}")
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Could not read completed segment marker: {completion_path}") from error
    if not isinstance(completion, dict) or completion.get("checkpoint") != checkpoint_path.name:
        raise ValueError("Completed segment marker does not identify the selected checkpoint.")
    expected_names = {
        MANIFEST_FILENAME,
        CHECKPOINT_FILENAME,
        TRACE_FILENAME,
        SUMMARY_FILENAME,
    }
    file_hashes = completion.get("sha256")
    if not isinstance(file_hashes, dict) or set(file_hashes) != expected_names:
        raise ValueError("Completed segment marker has an invalid file-hash set.")
    for name in sorted(expected_names):
        artifact = checkpoint_path.parent / name
        digest = file_hashes[name]
        if (
            not artifact.is_file()
            or not isinstance(digest, str)
            or len(digest) != 64
            or _sha256_file(artifact) != digest
        ):
            raise ValueError(f"Completed segment artifact failed SHA-256 validation: {name}")


def _positive_values(value: str) -> float | tuple[float, ...]:
    """Parse one scalar or comma-separated vector of positive floats."""
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a scalar or comma-separated floats") from error
    if not values or any(not np.isfinite(item) or item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("all values must be finite and strictly positive")
    return values[0] if len(values) == 1 else values


def _comma_separated_labels(value: str) -> tuple[str, ...]:
    """Parse a nonempty comma-separated sequence of unique labels."""
    labels = tuple(item.strip() for item in value.split(","))
    if not labels or any(not label for label in labels):
        raise argparse.ArgumentTypeError("labels must be nonempty")
    if len(set(labels)) != len(labels):
        raise argparse.ArgumentTypeError("labels must be unique")
    return labels


def _expand_values(
    values: float | tuple[float, ...],
    *,
    size: int,
    name: str,
) -> tuple[float, ...]:
    """Broadcast one positive scalar or validate one exact-width vector."""
    if not isinstance(values, tuple):
        return (float(values),) * size
    if len(values) != size:
        raise ValueError(f"{name} must be scalar or contain exactly {size} values.")
    return values


def _load_frozen_dataset(path: Path, *, engine: NetCDFEngine) -> xr.Dataset:
    """Eagerly load and close one frozen NetCDF input."""
    if not path.is_file():
        raise FileNotFoundError(f"Frozen input is not a file: {path}")
    with xr.open_dataset(path, engine=engine) as opened:
        return opened.load()


def _input_array(dataset: xr.Dataset, name: str) -> xr.DataArray:
    """Return one explicitly named data variable."""
    if name not in dataset.data_vars:
        raise ValueError(f"Frozen input is missing required data variable {name!r}.")
    return dataset[name]


def _fixed_values_for_adapter(
    values: float | tuple[float, ...],
    *,
    size: int,
    name: str,
) -> float | tuple[float, ...]:
    """Return a scalar or validate a per-fixed-column vector."""
    if isinstance(values, float):
        return values
    return _expand_values(values, size=size, name=name)


def _dimension_labels(dataset: xr.Dataset, dimension: str) -> np.ndarray:
    """Return nonempty string labels for one scientific dimension."""
    if dimension in dataset.coords:
        values = np.asarray(dataset.coords[dimension].values)
    else:
        values = np.arange(dataset.sizes[dimension], dtype=np.int64)
    return np.asarray([str(value) for value in values.tolist()], dtype=np.str_)


def _build_adapter(
    dataset: xr.Dataset,
    arguments: argparse.Namespace,
) -> GammaBetaRHIMEAdapterResult:
    """Build the explicit native-grid adapter from parsed CLI settings."""
    fixed_design = _input_array(dataset, arguments.fixed_design_name)
    if fixed_design.ndim != 2 or "nmeasure" not in fixed_design.dims:
        raise ValueError(
            f"{arguments.fixed_design_name!r} must have dimensions ('nmeasure', <fixed coefficient>)."
        )
    fixed_dimension = next(dimension for dimension in fixed_design.dims if dimension != "nmeasure")
    n_fixed = int(fixed_design.sizes[fixed_dimension])
    fixed_mean = _fixed_values_for_adapter(
        arguments.fixed_prior_mean,
        size=n_fixed,
        name="fixed_prior_mean",
    )
    fixed_sd = _fixed_values_for_adapter(
        arguments.fixed_prior_sd,
        size=n_fixed,
        name="fixed_prior_sd",
    )
    return gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=_input_array(dataset, arguments.nominal_weight_name),
        k_min=arguments.k_min,
        k_max=arguments.k_max,
        concentration=arguments.concentration,
        root_variance=arguments.root_variance,
        normalize_weights=arguments.normalize_weights,
        likelihood_power=arguments.likelihood_power,
        sensitivity_name=arguments.sensitivity_name,
        observation_name=arguments.observation_name,
        observation_sd_name=arguments.observation_sd_name,
        fixed_design_name=arguments.fixed_design_name,
        fixed_offset_name=arguments.fixed_offset_name,
        fixed_coefficient_prior_mean=fixed_mean,
        fixed_coefficient_prior_sd=fixed_sd,
    )


def _require_paris_profile(
    dataset: xr.Dataset,
    adapter: GammaBetaRHIMEAdapterResult,
    *,
    fixed_design_name: str,
    expected_outer_labels: tuple[str, ...],
) -> None:
    """Reject inputs that do not match the reviewed modern PARIS profile."""
    problem = adapter.problem
    actual = (
        int(problem.observations.size),
        adapter.spatial_shape,
        problem.n_fixed_coefficients,
    )
    expected = (
        PARIS_OBSERVATIONS,
        PARIS_GRID_SHAPE,
        PARIS_OUTER_COEFFICIENTS,
    )
    if actual != expected:
        raise ValueError(
            "--require-paris-profile expected "
            f"{expected[0]} observations, grid {expected[1]}, and "
            f"{expected[2]} fixed coefficients; found {actual}."
        )
    fixed_design = dataset[fixed_design_name]
    if set(fixed_design.dims) != {"nmeasure", "outer_region"}:
        raise ValueError(
            "--require-paris-profile requires fixed design dimensions 'nmeasure' and 'outer_region'."
        )
    if len(expected_outer_labels) != PARIS_OUTER_COEFFICIENTS:
        raise ValueError("--expected-outer-labels must contain exactly six reviewed labels.")
    actual_outer_labels = tuple(_dimension_labels(dataset, "outer_region").astype(str).tolist())
    if actual_outer_labels != expected_outer_labels:
        raise ValueError("Frozen outer_region labels/order do not match --expected-outer-labels.")
    if "nmeasure" not in dataset.coords:
        raise ValueError("--require-paris-profile requires explicit nmeasure labels.")
    measurement_labels = _dimension_labels(dataset, "nmeasure").astype(str).tolist()
    if len(set(measurement_labels)) != PARIS_OBSERVATIONS:
        raise ValueError("--require-paris-profile requires unique nmeasure labels.")


def _closure_audit(
    dataset: xr.Dataset,
    adapter: GammaBetaRHIMEAdapterResult,
    *,
    start_k: int,
    sensitivity_name: str,
    fixed_design_name: str,
    fixed_offset_name: str,
) -> dict[str, float]:
    """Verify mass-coordinate and prior-mean forward-model closure."""
    problem = adapter.problem
    sensitivity = dataset[sensitivity_name].transpose("nmeasure", "lat", "lon")
    scaling_prediction = np.asarray(sensitivity.values, dtype=np.float64).sum(axis=(1, 2))
    mass_prediction = problem.sensitivity @ problem.prior.nominal_cell_mass
    mass_error = np.asarray(mass_prediction - scaling_prediction, dtype=np.float64)
    if not np.allclose(
        mass_prediction,
        scaling_prediction,
        rtol=_CLOSURE_RTOL,
        atol=_CLOSURE_ATOL,
    ):
        raise ValueError(
            "Mass-coordinate closure failed: sensitivity_per_mass @ nominal_weight "
            "does not reproduce the all-one fp_x_flux prediction."
        )

    if problem.fixed_block is None or problem.fixed_offset is None:
        raise RuntimeError("The real-data driver requires explicit fixed design and offset terms.")
    fixed_design = dataset[fixed_design_name]
    fixed_dimension = next(dimension for dimension in fixed_design.dims if dimension != "nmeasure")
    fixed_values = np.asarray(
        fixed_design.transpose("nmeasure", fixed_dimension).values,
        dtype=np.float64,
    )
    offset = np.asarray(dataset[fixed_offset_name].transpose("nmeasure").values, dtype=np.float64)
    expected_total = offset + scaling_prediction + fixed_values @ problem.fixed_block.coefficient_prior_mean
    initial = initialize_gamma_beta_state(problem, k=start_k)
    total_error = np.asarray(initial.prediction - expected_total, dtype=np.float64)
    if not np.allclose(
        initial.prediction,
        expected_total,
        rtol=_CLOSURE_RTOL,
        atol=_CLOSURE_ATOL,
    ):
        raise ValueError(
            "Prior-mean total closure failed against fixed offset, all-one "
            "inner scaling, and fixed-coefficient prior means."
        )
    return {
        "mass_coordinate_max_abs_error": float(np.max(np.abs(mass_error), initial=0.0)),
        "prior_mean_total_max_abs_error": float(np.max(np.abs(total_error), initial=0.0)),
    }


def _kernel_settings(
    adapter: GammaBetaRHIMEAdapterResult,
    proposal_sd: float | tuple[float, ...],
    *,
    relocation_slots: int,
    subtree_retile_slots: int,
    max_subtree_leaves: int,
) -> GammaBetaCompoundKernelSettings:
    """Resolve the configurable fixed-``K`` topology schedule."""
    scales = _expand_values(
        proposal_sd,
        size=adapter.problem.n_fixed_coefficients,
        name="fixed_proposal_sd",
    )
    return GammaBetaCompoundKernelSettings(
        split_direction_probability=SPLIT_DIRECTION_PROBABILITY,
        fraction_refresh_slots=FRACTION_REFRESH_SLOTS,
        relocation_slots=relocation_slots,
        subtree_retile_slots=subtree_retile_slots,
        max_subtree_leaves=max_subtree_leaves,
        fixed_coefficient_proposal_sd=scales,
    )


def _input_contract(arguments: argparse.Namespace) -> tuple[str, str]:
    """Return canonical input-selection metadata and its SHA-256."""
    contract = {
        "input_netcdf_engine": arguments.input_netcdf_engine,
        "sensitivity_name": arguments.sensitivity_name,
        "observation_name": arguments.observation_name,
        "observation_sd_name": arguments.observation_sd_name,
        "nominal_weight_name": arguments.nominal_weight_name,
        "fixed_design_name": arguments.fixed_design_name,
        "fixed_offset_name": arguments.fixed_offset_name,
        "normalize_weights": arguments.normalize_weights,
    }
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return canonical, sha256(canonical.encode("utf-8")).hexdigest()


def _segment_iterations(arguments: argparse.Namespace, *, cycle_length: int) -> int:
    """Resolve the requested segment length to atomic transitions."""
    if arguments.cycles is not None:
        return int(arguments.cycles) * cycle_length
    return int(arguments.iterations)


def _move_summary(result: GammaBetaCompoundSamplingResult) -> dict[str, dict[str, Any]]:
    """Summarize attempts, validity, acceptance, and rates by concrete move."""
    trace = result.trace
    summaries: dict[str, dict[str, Any]] = {}
    for move in (
        "split",
        "merge",
        "relocate",
        "subtree_retile",
        "root_refresh",
        "fraction_refresh",
        "fixed_coefficient",
    ):
        selected = trace.move == move
        attempts = int(np.count_nonzero(selected))
        valid = int(np.count_nonzero(trace.valid[selected]))
        accepted = int(np.count_nonzero(trace.accepted[selected]))
        summaries[move] = {
            "attempts": attempts,
            "valid": valid,
            "accepted": accepted,
            "valid_rate": None if attempts == 0 else valid / attempts,
            "acceptance_rate": None if attempts == 0 else accepted / attempts,
            "acceptance_rate_given_valid": None if valid == 0 else accepted / valid,
        }
    return summaries


def _fixed_coefficient_summary(
    result: GammaBetaCompoundSamplingResult,
) -> dict[str, dict[str, Any]]:
    """Summarize deterministic fixed-coefficient slots by position."""
    trace = result.trace
    summaries: dict[str, dict[str, Any]] = {}
    for position in range(result.final_state.fixed_coefficients.size):
        selected = (trace.move == "fixed_coefficient") & (trace.coefficient_id == position)
        attempts = int(np.count_nonzero(selected))
        valid = int(np.count_nonzero(trace.valid[selected]))
        accepted = int(np.count_nonzero(trace.accepted[selected]))
        summaries[str(position)] = {
            "attempts": attempts,
            "valid": valid,
            "accepted": accepted,
            "acceptance_rate": None if attempts == 0 else accepted / attempts,
            "acceptance_rate_given_valid": None if valid == 0 else accepted / valid,
        }
    return summaries


def _segment_summary(
    result: GammaBetaCompoundSamplingResult,
    *,
    input_seconds: float,
    problem_setup_seconds: float,
    resume_validation_seconds: float,
    sampling_seconds: float,
    closure: dict[str, float],
    input_path: Path,
    input_digest: str,
    cycle_length: int,
) -> dict[str, Any]:
    """Build one strict-JSON-compatible segment summary."""
    trace = result.trace
    transitions_end = result.checkpoint.transitions_completed
    transitions_start = transitions_end - trace.global_transition.size
    visited_k = np.concatenate(
        (
            trace.k_before,
            trace.k_after,
            np.asarray([result.final_state.k], dtype=np.int64),
        )
    )
    return {
        "input": {
            "path": str(input_path.resolve()),
            "sha256": input_digest,
        },
        "closure": closure,
        "segment": {
            "atomic_transitions": int(trace.global_transition.size),
            "transitions_start": int(transitions_start),
            "transitions_end": int(transitions_end),
            "cycle_length": cycle_length,
            "schedule_phase_end": result.checkpoint.schedule_phase,
            "whole_cycle_equivalents": int(trace.global_transition.size // cycle_length),
            "global_cycle_boundaries_crossed": int(
                transitions_end // cycle_length - transitions_start // cycle_length
            ),
            "retained_draws": int(trace.k.size),
        },
        "k": {
            "minimum_visited": int(np.min(visited_k)),
            "maximum_visited": int(np.max(visited_k)),
            "final": result.final_state.k,
        },
        "moves": _move_summary(result),
        "fixed_coefficients": _fixed_coefficient_summary(result),
        "performance": {
            "input_hash_and_load_seconds": input_seconds,
            "problem_setup_seconds": problem_setup_seconds,
            "resume_validation_seconds": resume_validation_seconds,
            "sampling_seconds": sampling_seconds,
            "atomic_transitions_per_second": (
                float(trace.global_transition.size / sampling_seconds) if sampling_seconds > 0.0 else None
            ),
        },
    }


def _write_outputs(
    output_directory: Path,
    result: GammaBetaCompoundSamplingResult,
    *,
    run_manifest: dict[str, object],
    summary: dict[str, Any],
    netcdf_engine: NetCDFEngine,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    fixed_parameter_labels: np.ndarray,
    measurement_labels: np.ndarray,
) -> None:
    """Create one immutable segment directory and write its completion marker last."""
    output_started = perf_counter()
    output_directory.mkdir()
    canonical_manifest = canonical_gamma_beta_run_manifest(run_manifest)
    artifact_timings: dict[str, float] = {}
    started = perf_counter()
    _atomic_write_text(
        output_directory / MANIFEST_FILENAME,
        canonical_manifest,
    )
    artifact_timings["manifest_seconds"] = perf_counter() - started
    started = perf_counter()
    save_gamma_beta_checkpoint(
        output_directory / CHECKPOINT_FILENAME,
        result.checkpoint,
        run_manifest=run_manifest,
    )
    artifact_timings["checkpoint_seconds"] = perf_counter() - started
    trace_dataset = gamma_beta_compound_trace_to_dataset(
        result.trace,
        problem=result.checkpoint.problem,
        metadata={
            "run_manifest_sha256": sha256(canonical_manifest.encode("utf-8")).hexdigest(),
            "transitions_start": summary["segment"]["transitions_start"],
            "transitions_end": summary["segment"]["transitions_end"],
        },
        latitudes=latitudes,
        longitudes=longitudes,
        fixed_parameter_labels=fixed_parameter_labels,
        measurement_labels=measurement_labels,
    )
    try:
        started = perf_counter()
        _atomic_write_trace(
            trace_dataset,
            output_directory / TRACE_FILENAME,
            engine=netcdf_engine,
        )
        artifact_timings["trace_seconds"] = perf_counter() - started
    finally:
        trace_dataset.close()
    started = perf_counter()
    _atomic_write_text(
        output_directory / SUMMARY_FILENAME,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    artifact_timings["summary_seconds"] = perf_counter() - started
    artifact_hashes = {
        name: _sha256_file(output_directory / name)
        for name in (
            MANIFEST_FILENAME,
            CHECKPOINT_FILENAME,
            TRACE_FILENAME,
            SUMMARY_FILENAME,
        )
    }
    completion = {
        "checkpoint": CHECKPOINT_FILENAME,
        "manifest": MANIFEST_FILENAME,
        "summary": SUMMARY_FILENAME,
        "trace": TRACE_FILENAME,
        "transitions_completed": result.checkpoint.transitions_completed,
        "sha256": artifact_hashes,
        "performance": {
            **artifact_timings,
            "output_before_completion_seconds": perf_counter() - output_started,
        },
    }
    _atomic_write_text(
        output_directory / COMPLETION_FILENAME,
        json.dumps(completion, sort_keys=True, allow_nan=False) + "\n",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit frozen-input and durable-segment CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Frozen NetCDF input file.")
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="New per-segment directory; existing paths are refused.",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    length = parser.add_mutually_exclusive_group(required=True)
    length.add_argument(
        "--cycles",
        type=int,
        help=(
            "Complete compound cycles (recommended; 16 transitions with six "
            "outer columns and the default fixed-K topology slots)."
        ),
    )
    length.add_argument(
        "--iterations",
        type=int,
        help="Atomic transitions, including an explicitly requested partial cycle.",
    )
    parser.add_argument("--k-min", type=int, required=True)
    parser.add_argument("--k-max", type=int, required=True)
    parser.add_argument("--start-k", type=int, required=True)
    parser.add_argument("--concentration", type=float, required=True)
    parser.add_argument("--root-variance", type=float, required=True)
    parser.add_argument("--likelihood-power", type=float, default=1.0)
    parser.add_argument(
        "--fixed-prior-mean",
        type=_positive_values,
        default=1.0,
        metavar="VALUE[,VALUE...]",
    )
    parser.add_argument(
        "--fixed-prior-sd",
        type=_positive_values,
        required=True,
        metavar="VALUE[,VALUE...]",
    )
    parser.add_argument(
        "--fixed-proposal-sd",
        type=_positive_values,
        required=True,
        metavar="VALUE[,VALUE...]",
    )
    parser.add_argument(
        "--relocation-slots",
        type=int,
        default=RELOCATION_SLOTS,
        help="Fixed-K cherry-relocation opportunities per cycle (default: 1).",
    )
    parser.add_argument(
        "--subtree-retile-slots",
        type=int,
        default=SUBTREE_RETILE_SLOTS,
        help="Bounded fixed-K subtree-retile opportunities per cycle (default: 1).",
    )
    parser.add_argument(
        "--max-subtree-leaves",
        type=int,
        default=MAX_SUBTREE_LEAVES,
        help="Largest active-leaf count eligible for exact subtree retile (default: 8).",
    )
    parser.add_argument("--warmup", type=int, default=0, help="Global warmup atomic transitions.")
    parser.add_argument("--thin", type=int, default=1, help="Global thinning in atomic transitions.")
    parser.add_argument("--seed", type=int, required=True, help="Initial PCG64 seed.")
    parser.add_argument("--chain-id", required=True, help="Stable unique chain identifier.")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument(
        "--input-id",
        required=True,
        help="Stable logical frozen-input identifier, independent of scratch path.",
    )
    parser.add_argument(
        "--expected-input-sha256",
        help="Optional required SHA-256; mandatory with --require-paris-profile.",
    )
    parser.add_argument("--nominal-weight-policy", required=True)
    parser.add_argument(
        "--normalize-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sensitivity-name", default="fp_x_flux")
    parser.add_argument("--observation-name", default="mf")
    parser.add_argument("--observation-sd-name", default="mf_error")
    parser.add_argument("--nominal-weight-name", default="nominal_weight")
    parser.add_argument("--fixed-design-name", default="outer_design")
    parser.add_argument("--fixed-offset-name", default="YaprioriBC")
    parser.add_argument(
        "--require-paris-profile",
        action="store_true",
        help="Require 1382 observations, a 183x128 grid, and six outer coefficients.",
    )
    parser.add_argument(
        "--expected-outer-labels",
        type=_comma_separated_labels,
        help="Reviewed comma-separated outer_region labels in exact column order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input, closure, manifest, and schedule without sampling or writing.",
    )
    parser.add_argument(
        "--input-netcdf-engine",
        choices=("h5netcdf", "netcdf4", "scipy"),
        default="h5netcdf",
        help="Explicit xarray backend used to read the frozen input.",
    )
    parser.add_argument(
        "--netcdf-engine",
        choices=("h5netcdf", "netcdf4", "scipy"),
        default="h5netcdf",
        help="xarray NetCDF writer; h5netcdf avoids implicit backend selection.",
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject malformed cross-argument settings before expensive work."""
    if arguments.output_directory.exists():
        raise FileExistsError(f"Output path already exists: {arguments.output_directory}")
    if not arguments.output_directory.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory does not exist: {arguments.output_directory.parent}"
        )
    if arguments.cycles is not None and arguments.cycles < 1:
        raise ValueError("cycles must be positive.")
    if arguments.iterations is not None and arguments.iterations < 1:
        raise ValueError("iterations must be positive.")
    if arguments.relocation_slots < 0:
        raise ValueError("relocation_slots must be non-negative.")
    if arguments.subtree_retile_slots < 0:
        raise ValueError("subtree_retile_slots must be non-negative.")
    if arguments.max_subtree_leaves < 1:
        raise ValueError("max_subtree_leaves must be positive.")
    if arguments.dry_run and arguments.resume_checkpoint is not None:
        raise ValueError("--dry-run cannot be combined with --resume-checkpoint.")
    if arguments.expected_input_sha256 is not None:
        digest = arguments.expected_input_sha256
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise ValueError("--expected-input-sha256 must be exactly 64 hexadecimal characters.")
    if arguments.require_paris_profile and (
        arguments.expected_input_sha256 is None or arguments.expected_outer_labels is None
    ):
        raise ValueError(
            "--require-paris-profile requires --expected-input-sha256 and --expected-outer-labels."
        )


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    """Validate, run, and persist one fresh or resumed compound segment."""
    _validate_arguments(arguments)
    _preflight_output_backend(
        arguments.output_directory.parent,
        engine=arguments.netcdf_engine,
    )
    input_started = perf_counter()
    input_digest = _sha256_file(arguments.input)
    if (
        arguments.expected_input_sha256 is not None
        and input_digest.lower() != arguments.expected_input_sha256.lower()
    ):
        raise ValueError("Frozen input SHA-256 does not match --expected-input-sha256.")
    dataset = _load_frozen_dataset(
        arguments.input,
        engine=arguments.input_netcdf_engine,
    )
    input_seconds = perf_counter() - input_started
    problem_started = perf_counter()
    adapter = _build_adapter(dataset, arguments)
    if arguments.require_paris_profile:
        _require_paris_profile(
            dataset,
            adapter,
            fixed_design_name=arguments.fixed_design_name,
            expected_outer_labels=arguments.expected_outer_labels,
        )
    closure = _closure_audit(
        dataset,
        adapter,
        start_k=arguments.start_k,
        sensitivity_name=arguments.sensitivity_name,
        fixed_design_name=arguments.fixed_design_name,
        fixed_offset_name=arguments.fixed_offset_name,
    )
    settings = _kernel_settings(
        adapter,
        arguments.fixed_proposal_sd,
        relocation_slots=arguments.relocation_slots,
        subtree_retile_slots=arguments.subtree_retile_slots,
        max_subtree_leaves=arguments.max_subtree_leaves,
    )
    retention = RetentionSettings(
        warmup_transitions=arguments.warmup,
        thin=arguments.thin,
    )
    initial_state = initialize_gamma_beta_state(
        adapter.problem,
        k=arguments.start_k,
    )
    input_contract, input_contract_sha = _input_contract(arguments)
    run_manifest = build_gamma_beta_run_manifest(
        adapter.problem,
        settings,
        retention,
        chain_id=arguments.chain_id,
        initial_state=initial_state,
        code_revision=arguments.code_revision,
        input_identifiers={
            "frozen_native_dataset": arguments.input_id,
            "input_variable_contract": input_contract,
        },
        input_sha256={
            "frozen_native_dataset": input_digest,
            "input_variable_contract": input_contract_sha,
        },
        nominal_weight_policy=arguments.nominal_weight_policy,
        nominal_weight_normalization_factor=adapter.weight_normalization_factor,
        seed=arguments.seed,
    )
    iterations = _segment_iterations(arguments, cycle_length=settings.cycle_length)
    problem_setup_seconds = perf_counter() - problem_started
    if arguments.dry_run:
        return {
            "closure": closure,
            "cycle_length": settings.cycle_length,
            "requested_atomic_transitions": iterations,
            "input": {
                "id": arguments.input_id,
                "path": str(arguments.input.resolve()),
                "sha256": input_digest,
            },
            "problem_sha256": run_manifest["problem_sha256"],
            "run_manifest": run_manifest,
            "performance": {
                "input_hash_and_load_seconds": input_seconds,
                "problem_setup_seconds": problem_setup_seconds,
            },
        }

    resume_validation_seconds = 0.0
    if arguments.resume_checkpoint is None:
        started = perf_counter()
        result = sample_gamma_beta_compound(
            adapter.problem,
            initial_state,
            GammaBetaCompoundConfig(
                iterations=iterations,
                seed=arguments.seed,
                split_direction_probability=SPLIT_DIRECTION_PROBABILITY,
                fraction_refresh_slots=FRACTION_REFRESH_SLOTS,
                relocation_slots=arguments.relocation_slots,
                subtree_retile_slots=arguments.subtree_retile_slots,
                max_subtree_leaves=arguments.max_subtree_leaves,
                fixed_coefficient_proposal_sd=settings.fixed_coefficient_proposal_sd,
            ),
            retention=retention,
        )
        sampling_seconds = perf_counter() - started
    else:
        resume_started = perf_counter()
        _validate_resume_bundle(arguments.resume_checkpoint)
        checkpoint = load_gamma_beta_checkpoint(
            arguments.resume_checkpoint,
            adapter.problem,
            expected_run_manifest=run_manifest,
        )
        resume_validation_seconds = perf_counter() - resume_started
        started = perf_counter()
        result = continue_gamma_beta_compound(
            adapter.problem,
            checkpoint,
            iterations=iterations,
        )
        sampling_seconds = perf_counter() - started
    summary = _segment_summary(
        result,
        input_seconds=input_seconds,
        problem_setup_seconds=problem_setup_seconds,
        resume_validation_seconds=resume_validation_seconds,
        sampling_seconds=sampling_seconds,
        closure=closure,
        input_path=arguments.input,
        input_digest=input_digest,
        cycle_length=settings.cycle_length,
    )
    _write_outputs(
        arguments.output_directory,
        result,
        run_manifest=run_manifest,
        summary=summary,
        netcdf_engine=arguments.netcdf_engine,
        latitudes=adapter.latitudes,
        longitudes=adapter.longitudes,
        fixed_parameter_labels=_dimension_labels(
            dataset,
            next(
                dimension
                for dimension in dataset[arguments.fixed_design_name].dims
                if dimension != "nmeasure"
            ),
        ),
        measurement_labels=_dimension_labels(dataset, "nmeasure"),
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run one segment and print its compact JSON summary."""
    arguments = build_parser().parse_args(argv)
    summary = run(arguments)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
