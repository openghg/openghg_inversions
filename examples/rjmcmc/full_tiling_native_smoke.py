"""Run one uninterrupted fixed-``K`` full-tiling native-data smoke chain.

The input is one explicitly frozen NetCDF dataset.  By default it contains
``fp_x_flux(nmeasure, lat, lon)``, ``mf(nmeasure)``,
``mf_error(nmeasure)``, strictly positive ``nominal_weight(lat, lon)``,
``outer_design(nmeasure, outer_region)``, and ``YaprioriBC(nmeasure)``.
Variable names are configurable; no scientific input is guessed.

This is a diagnostic smoke driver, not a convergence workflow.  It runs one
single-process chain without durable continuation, retains complete cycle
boundaries, and writes a new immutable output directory.  Until connectivity
has been established independently, every result is explicitly restricted to
the communication component reached from its deterministic prior-mean start.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
import xarray as xr

from openghg_inversions.experimental.rjmcmc.full_tiling_compound_sampling import (
    FULL_TILING_COMPOUND_SCHEDULE_ID,
    FullTilingCompoundConfig,
    FullTilingCompoundSamplingResult,
    sample_full_tiling_compound,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    FullTilingPosteriorState,
    full_tiling_problem_from_gamma_beta_adapter,
    initialize_full_tiling_posterior_state,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    GammaBetaRHIMEAdapterResult,
    gamma_beta_problem_from_rhime_inputs,
)

NetCDFEngine = Literal["h5netcdf", "netcdf4", "scipy"]

MANIFEST_FILENAME = "manifest.json"
TRACE_FILENAME = "trace.nc"
SUMMARY_FILENAME = "summary.json"
COMPLETION_FILENAME = "complete.json"
PARIS_OBSERVATIONS = 1_382
PARIS_GRID_SHAPE = (183, 128)
PARIS_OUTER_COEFFICIENTS = 6
PAIR_ALLOCATION_REFRESH_SLOTS = 5
_CLOSURE_RTOL = 1.0e-12
_CLOSURE_ATOL = 1.0e-12
_COMMUNICATION_COMPONENT = "component_reachable_from_deterministic_prior_mean_start"


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Flush one directory entry where supported."""
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


def _preflight_output_backend(parent: Path, *, engine: NetCDFEngine) -> None:
    """Write and reopen a tiny NetCDF on the selected output filesystem."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".full-tiling-netcdf-preflight-",
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


def _dimension_labels(dataset: xr.Dataset, dimension: str) -> np.ndarray:
    """Return stable string labels for one dimension."""
    if dimension in dataset.coords:
        values = np.asarray(dataset.coords[dimension].values)
    else:
        values = np.arange(dataset.sizes[dimension], dtype=np.int64)
    return np.asarray([str(value) for value in values.tolist()], dtype=np.str_)


def _build_adapter(
    dataset: xr.Dataset,
    arguments: argparse.Namespace,
) -> GammaBetaRHIMEAdapterResult:
    """Build the existing RHIME-to-Gamma--Beta adapter explicitly."""
    fixed_design = _input_array(dataset, arguments.fixed_design_name)
    if fixed_design.ndim != 2 or "nmeasure" not in fixed_design.dims:
        raise ValueError(
            f"{arguments.fixed_design_name!r} must have dimensions ('nmeasure', <fixed coefficient>)."
        )
    fixed_dimension = next(dimension for dimension in fixed_design.dims if dimension != "nmeasure")
    n_fixed = int(fixed_design.sizes[fixed_dimension])
    fixed_mean = _expand_values(
        arguments.fixed_prior_mean,
        size=n_fixed,
        name="fixed_prior_mean",
    )
    fixed_sd = _expand_values(
        arguments.fixed_prior_sd,
        size=n_fixed,
        name="fixed_prior_sd",
    )
    return gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=_input_array(dataset, arguments.nominal_weight_name),
        k_min=arguments.k,
        k_max=arguments.k,
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
    """Reject inputs outside the reviewed modern PARIS profile."""
    actual = (
        int(adapter.problem.observations.size),
        adapter.spatial_shape,
        adapter.problem.n_fixed_coefficients,
    )
    expected = (PARIS_OBSERVATIONS, PARIS_GRID_SHAPE, PARIS_OUTER_COEFFICIENTS)
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
    actual_labels = tuple(_dimension_labels(dataset, "outer_region").astype(str).tolist())
    if actual_labels != expected_outer_labels:
        raise ValueError("Frozen outer_region labels/order do not match --expected-outer-labels.")
    if "nmeasure" not in dataset.coords:
        raise ValueError("--require-paris-profile requires explicit nmeasure labels.")
    labels = _dimension_labels(dataset, "nmeasure").astype(str).tolist()
    if len(set(labels)) != PARIS_OBSERVATIONS:
        raise ValueError("--require-paris-profile requires unique nmeasure labels.")


def _closure_audit(
    dataset: xr.Dataset,
    adapter: GammaBetaRHIMEAdapterResult,
    *,
    initial_state: FullTilingPosteriorState,
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
        raise RuntimeError("The native smoke driver requires fixed design and offset terms.")
    fixed_design = dataset[fixed_design_name]
    fixed_dimension = next(dimension for dimension in fixed_design.dims if dimension != "nmeasure")
    fixed_values = np.asarray(
        fixed_design.transpose("nmeasure", fixed_dimension).values,
        dtype=np.float64,
    )
    offset = np.asarray(
        dataset[fixed_offset_name].transpose("nmeasure").values,
        dtype=np.float64,
    )
    expected_total = offset + scaling_prediction + fixed_values @ problem.fixed_block.coefficient_prior_mean
    total_error = np.asarray(initial_state.prediction - expected_total, dtype=np.float64)
    if not np.allclose(
        initial_state.prediction,
        expected_total,
        rtol=_CLOSURE_RTOL,
        atol=_CLOSURE_ATOL,
    ):
        raise ValueError(
            "Prior-mean closure failed against raw fp_x_flux, fixed boundary "
            "offset, and all fixed-coefficient prior means."
        )
    return {
        "mass_coordinate_max_abs_error": float(np.max(np.abs(mass_error), initial=0.0)),
        "prior_mean_total_max_abs_error": float(np.max(np.abs(total_error), initial=0.0)),
    }


def _canonical_json(value: object) -> str:
    """Return deterministic strict JSON terminated by one newline."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def _input_contract(arguments: argparse.Namespace) -> dict[str, object]:
    """Return the complete explicit scientific input-variable contract."""
    return {
        "sensitivity": arguments.sensitivity_name,
        "observation": arguments.observation_name,
        "observation_sd": arguments.observation_sd_name,
        "nominal_weight": arguments.nominal_weight_name,
        "fixed_design": arguments.fixed_design_name,
        "fixed_offset": arguments.fixed_offset_name,
        "normalize_weights": bool(arguments.normalize_weights),
        "nominal_weight_policy": arguments.nominal_weight_policy,
    }


def _build_manifest(
    arguments: argparse.Namespace,
    adapter: GammaBetaRHIMEAdapterResult,
    *,
    input_digest: str,
    cycle_length: int,
    outer_labels: np.ndarray,
) -> dict[str, object]:
    """Build the immutable native full-tiling smoke manifest."""
    fixed_block = adapter.problem.fixed_block
    if fixed_block is None:
        raise RuntimeError("The native smoke driver requires a fixed design block.")
    fixed_mean = fixed_block.coefficient_prior_mean
    fixed_sd = fixed_block.coefficient_prior_sd
    manifest: dict[str, object] = {
        "schema": "openghg_inversions.full_tiling_native_smoke_manifest.v1",
        "status": "diagnostic_not_convergence_evidence",
        "state_space_scope": {
            "fixed_k": int(arguments.k),
            "structural_target": "uniform_over_unique_canonical_tilings_at_fixed_k",
            "communication_component": _COMMUNICATION_COMPONENT,
            "connectivity_proven": False,
        },
        "input": {
            "id": arguments.input_id,
            "path": str(arguments.input.resolve()),
            "sha256": input_digest,
            "contract": _input_contract(arguments),
            "weight_normalization_factor": adapter.weight_normalization_factor,
        },
        "model": {
            "grid_shape": list(adapter.spatial_shape),
            "observations": int(adapter.problem.observations.size),
            "fixed_coefficients": adapter.problem.n_fixed_coefficients,
            "outer_labels": outer_labels.astype(str).tolist(),
            "concentration": float(arguments.concentration),
            "root_variance": float(arguments.root_variance),
            "likelihood_power": float(arguments.likelihood_power),
            "fixed_prior_mean": fixed_mean.astype(float).tolist(),
            "fixed_prior_sd": fixed_sd.astype(float).tolist(),
        },
        "kernel": {
            "schedule_id": FULL_TILING_COMPOUND_SCHEDULE_ID,
            "cycle_length": cycle_length,
            "cycles": int(arguments.cycles),
            "atomic_transitions": int(arguments.cycles * cycle_length),
            "structural_slots": 2,
            "pair_allocation_refresh_slots": PAIR_ALLOCATION_REFRESH_SLOTS,
            "fixed_coefficient_proposal_sd": _expand_values(
                arguments.fixed_proposal_sd,
                size=adapter.problem.n_fixed_coefficients,
                name="fixed_proposal_sd",
            ),
        },
        "provenance": {
            "input_id": arguments.input_id,
            "code_revision": arguments.code_revision,
            "seed": int(arguments.seed),
            "single_process": True,
            "durable_checkpoint": False,
        },
    }
    manifest["manifest_payload_sha256"] = sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    return manifest


def _trace_to_dataset(
    result: FullTilingCompoundSamplingResult,
    *,
    adapter: GammaBetaRHIMEAdapterResult,
    outer_labels: np.ndarray,
    input_digest: str,
    manifest_digest: str,
    fixed_k: int,
) -> xr.Dataset:
    """Convert the complete retained and attempt traces to labelled xarray."""
    trace = result.trace
    draw = np.arange(trace.root_total.size, dtype=np.int64)
    transition = np.arange(trace.global_transition.size, dtype=np.int64)
    dataset = xr.Dataset(
        data_vars={
            "rectangle_bounds": (
                ("draw", "region", "bound"),
                trace.rectangle_bounds,
                {
                    "long_name": "canonical half-open native-grid rectangle bounds",
                },
            ),
            "leaf_mass": (
                ("draw", "region"),
                trace.leaf_masses,
                {"long_name": "positive mass aligned with canonical rectangles"},
            ),
            "root_total": (
                ("draw",),
                trace.root_total,
                {"long_name": "positive total inner-domain mass"},
            ),
            "fixed_coefficient": (
                ("draw", "fixed_parameter"),
                trace.fixed_coefficients,
                {"long_name": "positive always-active outer coefficient"},
            ),
            "log_gaussian_likelihood": (
                ("draw",),
                trace.log_gaussian_likelihood,
                {"long_name": "normalized unpowered Gaussian log likelihood"},
            ),
            "log_likelihood": (
                ("draw",),
                trace.log_likelihood,
                {"long_name": "likelihood-power-scaled log likelihood"},
            ),
            "log_root_prior": (
                ("draw",),
                trace.log_root_prior,
                {"long_name": "normalized Gamma root-total log density"},
            ),
            "log_allocation_prior": (
                ("draw",),
                trace.log_allocation_prior,
                {"long_name": "normalized additive-alpha Dirichlet-share log density"},
            ),
            "log_structural_prior": (
                ("draw",),
                trace.log_structural_prior,
                {
                    "long_name": "fixed-K uniform canonical-tiling log-ratio component",
                    "value": "zero by declaration",
                },
            ),
            "log_fixed_coefficient_prior": (
                ("draw",),
                trace.log_fixed_coefficient_prior,
                {"long_name": "normalized fixed-coefficient lognormal log density"},
            ),
            "log_target": (
                ("draw",),
                trace.log_target,
                {"long_name": "complete retained log target"},
            ),
            "state_transition": (
                ("draw",),
                trace.state_transition,
                {"long_name": "completed atomic-transition coordinate of retained state"},
            ),
            "global_transition": (
                ("transition",),
                trace.global_transition,
                {"long_name": "one-based global atomic-transition coordinate"},
            ),
            "slot": (
                ("transition",),
                trace.slot,
                {"long_name": "compound-schedule slot"},
            ),
            "move": (
                ("transition",),
                trace.move,
                {"long_name": "attempted proposal kernel"},
            ),
            "valid": (
                ("transition",),
                trace.valid,
                {"long_name": "proposal reached a Metropolis-Hastings decision"},
            ),
            "accepted": (
                ("transition",),
                trace.accepted,
                {"long_name": "proposal changed the visited state"},
            ),
            "log_acceptance_ratio": (
                ("transition",),
                trace.log_acceptance_ratio,
                {
                    "long_name": "untruncated Metropolis-Hastings log acceptance ratio",
                    "invalid_value": "-infinity",
                },
            ),
            "invalid_reason": (
                ("transition",),
                trace.invalid_reason,
                {"long_name": "empty for valid proposals; diagnostic reason otherwise"},
            ),
        },
        coords={
            "draw": draw,
            "region": np.arange(fixed_k, dtype=np.int64),
            "bound": np.asarray(
                ("row_start", "row_stop", "col_start", "col_stop"),
                dtype=np.str_,
            ),
            "transition": transition,
            "fixed_parameter": outer_labels,
            "lat": adapter.latitudes,
            "lon": adapter.longitudes,
        },
        attrs={
            "title": "Diagnostic fixed-K full-tiling native-data smoke trace",
            "diagnostic_only": "true",
            "convergence_claim": "none",
            "fixed_k": fixed_k,
            "schedule_id": FULL_TILING_COMPOUND_SCHEDULE_ID,
            "structural_target": ("uniform over unique canonical leaf tilings conditional on fixed K"),
            "continuous_target": (
                "Gamma root total, additive-alpha Dirichlet shares, and "
                "independent fixed-coefficient lognormal priors"
            ),
            "communication_component": _COMMUNICATION_COMPONENT,
            "connectivity_proven": "false",
            "input_sha256": input_digest,
            "manifest_sha256": manifest_digest,
            "nominal_weight_normalization_factor": (adapter.weight_normalization_factor),
            "allocation_concentration": result.final_state.problem.concentration,
            "root_prior_shape": adapter.problem.prior.root_shape,
            "root_prior_rate": adapter.problem.prior.root_rate,
            "root_prior_mean": (adapter.problem.prior.root_shape / adapter.problem.prior.root_rate),
            "root_prior_variance": (adapter.problem.prior.root_shape / adapter.problem.prior.root_rate**2),
            "likelihood_power": adapter.problem.likelihood_power,
        },
    )
    dataset["lat"].attrs["long_name"] = "native-grid latitude"
    dataset["lon"].attrs["long_name"] = "native-grid longitude"
    dataset["fixed_parameter"].attrs["long_name"] = "outer coefficient label"
    return dataset


def _move_summary(
    result: FullTilingCompoundSamplingResult,
) -> dict[str, dict[str, float | int | None]]:
    """Summarize validity and acceptance by concrete move."""
    trace = result.trace
    output: dict[str, dict[str, float | int | None]] = {}
    for move in (
        "edge_flip",
        "resolution_relocation",
        "root_total_refresh",
        "pair_allocation_refresh",
        "fixed_coefficient",
    ):
        selected = trace.move == move
        attempts = int(np.count_nonzero(selected))
        valid = int(np.count_nonzero(trace.valid[selected]))
        accepted = int(np.count_nonzero(trace.accepted[selected]))
        output[move] = {
            "attempts": attempts,
            "valid": valid,
            "accepted": accepted,
            "valid_rate": None if attempts == 0 else valid / attempts,
            "acceptance_rate": None if attempts == 0 else accepted / attempts,
            "acceptance_rate_given_valid": None if valid == 0 else accepted / valid,
        }
    return output


def _unique_topology_count(result: FullTilingCompoundSamplingResult) -> int:
    """Return the number of distinct retained canonical rectangle sets."""
    bounds = np.ascontiguousarray(result.trace.rectangle_bounds)
    return len({bounds[index].tobytes() for index in range(bounds.shape[0])})


def _initial_leaf_scaling_sd(
    initial_state: FullTilingPosteriorState,
) -> dict[str, float | str]:
    """Return additive-alpha prior scaling-SD diagnostics at the initial leaves."""
    problem = initial_state.problem
    root_mean = problem.base.prior.root_shape / problem.base.prior.root_rate
    root_variance = problem.base.prior.root_shape / problem.base.prior.root_rate**2
    relative_root_variance = root_variance / root_mean**2
    weights = np.asarray(
        [problem.rectangle_nominal_mass(rectangle) for rectangle in initial_state.allocation.tiling.leaves],
        dtype=np.float64,
    )
    concentration = problem.concentration
    variances = relative_root_variance + (
        (1.0 + relative_root_variance) * (1.0 - weights) / (weights * (concentration + 1.0))
    )
    standard_deviations = np.sqrt(variances)
    return {
        "minimum": float(np.min(standard_deviations)),
        "median": float(np.median(standard_deviations)),
        "maximum": float(np.max(standard_deviations)),
        "root_relative_variance": float(relative_root_variance),
        "formula": ("sqrt(v + (1 + v) * (1 - w) / (w * (kappa + 1)))"),
    }


def _summary(
    result: FullTilingCompoundSamplingResult,
    *,
    initial_state: FullTilingPosteriorState,
    closure: dict[str, float],
    input_path: Path,
    input_digest: str,
    cycles: int,
    cycle_length: int,
    input_seconds: float,
    setup_seconds: float,
    sampling_seconds: float,
) -> dict[str, Any]:
    """Build a strict-JSON-compatible diagnostic summary."""
    trace = result.trace
    invalid_reasons = Counter(str(reason) for reason in trace.invalid_reason[~trace.valid].tolist())
    attempts = int(trace.global_transition.size)
    valid = int(np.count_nonzero(trace.valid))
    accepted = int(np.count_nonzero(trace.accepted))
    return {
        "status": "diagnostic_not_convergence_evidence",
        "communication_component": _COMMUNICATION_COMPONENT,
        "connectivity_proven": False,
        "input": {
            "path": str(input_path.resolve()),
            "sha256": input_digest,
        },
        "closure": closure,
        "run": {
            "fixed_k": result.final_state.k,
            "cycles": cycles,
            "cycle_length": cycle_length,
            "atomic_transitions": attempts,
            "retained_draws": int(trace.root_total.size),
            "schedule_phase_end": result.checkpoint.schedule_phase,
        },
        "target": {
            "initial_log_target": float(initial_state.log_target),
            "final_log_target": float(result.final_state.log_target),
        },
        "initial_leaf_prior_scaling_sd": _initial_leaf_scaling_sd(initial_state),
        "topology": {
            "unique_retained_topologies": _unique_topology_count(result),
        },
        "attempts": {
            "valid": valid,
            "accepted": accepted,
            "valid_rate": valid / attempts,
            "acceptance_rate": accepted / attempts,
            "acceptance_rate_given_valid": None if valid == 0 else accepted / valid,
            "invalid_reasons": dict(sorted(invalid_reasons.items())),
        },
        "moves": _move_summary(result),
        "performance": {
            "input_hash_and_load_seconds": input_seconds,
            "problem_setup_seconds": setup_seconds,
            "sampling_seconds": sampling_seconds,
            "atomic_transitions_per_second": (
                attempts / sampling_seconds if sampling_seconds > 0.0 else None
            ),
        },
    }


def _write_outputs(
    output_directory: Path,
    result: FullTilingCompoundSamplingResult,
    *,
    manifest: dict[str, object],
    summary: dict[str, Any],
    trace_dataset: xr.Dataset,
    netcdf_engine: NetCDFEngine,
) -> None:
    """Write a new artifact bundle and completion marker last."""
    output_directory.mkdir()
    manifest_text = _canonical_json(manifest)
    _atomic_write_text(output_directory / MANIFEST_FILENAME, manifest_text)
    try:
        _atomic_write_trace(
            trace_dataset,
            output_directory / TRACE_FILENAME,
            engine=netcdf_engine,
        )
    finally:
        trace_dataset.close()
    summary_text = (
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    _atomic_write_text(output_directory / SUMMARY_FILENAME, summary_text)

    trace_path = output_directory / TRACE_FILENAME
    with xr.open_dataset(trace_path, engine=netcdf_engine) as reopened:
        loaded = reopened.load()
        if (
            loaded.sizes.get("region") != result.final_state.k
            or loaded.sizes.get("transition") != result.trace.global_transition.size
            or loaded.attrs.get("schedule_id") != FULL_TILING_COMPOUND_SCHEDULE_ID
        ):
            raise RuntimeError("Written full-tiling trace failed reopen validation.")
        loaded.close()

    artifact_hashes = {
        name: _sha256_file(output_directory / name)
        for name in (MANIFEST_FILENAME, TRACE_FILENAME, SUMMARY_FILENAME)
    }
    completion = {
        "schema": "openghg_inversions.full_tiling_native_smoke_completion.v1",
        "manifest": MANIFEST_FILENAME,
        "trace": TRACE_FILENAME,
        "summary": SUMMARY_FILENAME,
        "atomic_transitions": int(result.trace.global_transition.size),
        "sha256": artifact_hashes,
    }
    _atomic_write_text(
        output_directory / COMPLETION_FILENAME,
        _canonical_json(completion),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen-input fixed-``K`` smoke-test CLI.

    Returns:
        Argument parser exposing every scientific input and kernel setting
        required for a reproducible uninterrupted diagnostic run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--k",
        "--fixed-k",
        dest="k",
        type=int,
        required=True,
        help="Fixed active rectangle count.",
    )
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
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
    parser.add_argument("--input-id", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--expected-input-sha256")
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
        help="Require the reviewed 1382 by 183x128, six-outer profile.",
    )
    parser.add_argument(
        "--expected-outer-labels",
        type=_comma_separated_labels,
        help="Reviewed comma-separated outer_region labels in exact order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the frozen input, prior-mean closure, and schedule only.",
    )
    parser.add_argument(
        "--input-netcdf-engine",
        choices=("h5netcdf", "netcdf4", "scipy"),
        default="h5netcdf",
    )
    parser.add_argument(
        "--netcdf-engine",
        choices=("h5netcdf", "netcdf4", "scipy"),
        default="h5netcdf",
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject malformed CLI combinations before loading scientific data."""
    if arguments.output_directory.exists():
        raise FileExistsError(f"Output path already exists: {arguments.output_directory}")
    if not arguments.output_directory.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory does not exist: {arguments.output_directory.parent}"
        )
    if arguments.k < 1:
        raise ValueError("--k must be positive.")
    if arguments.cycles < 1:
        raise ValueError("--cycles must be positive.")
    if arguments.seed < 0:
        raise ValueError("--seed must be non-negative.")
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
    """Validate, run, and persist one uninterrupted diagnostic chain.

    Args:
        arguments: Namespace returned by :func:`build_parser`.

    Returns:
        Strict-JSON-compatible run summary. A dry run returns its validation
        summary without creating the output directory.

    Raises:
        FileExistsError: If the requested output path already exists.
        FileNotFoundError: If an input or output parent is absent.
        ValueError: If scientific inputs, settings, hashes, labels, or closure
            checks are invalid.
        RuntimeError: If sampling or artifact reopen validation fails.
    """
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

    setup_started = perf_counter()
    adapter = _build_adapter(dataset, arguments)
    if arguments.require_paris_profile:
        _require_paris_profile(
            dataset,
            adapter,
            fixed_design_name=arguments.fixed_design_name,
            expected_outer_labels=arguments.expected_outer_labels,
        )
    problem = full_tiling_problem_from_gamma_beta_adapter(
        adapter,
        concentration=arguments.concentration,
    )
    initial_state = initialize_full_tiling_posterior_state(problem, k=arguments.k)
    closure = _closure_audit(
        dataset,
        adapter,
        initial_state=initial_state,
        sensitivity_name=arguments.sensitivity_name,
        fixed_design_name=arguments.fixed_design_name,
        fixed_offset_name=arguments.fixed_offset_name,
    )
    fixed_design = dataset[arguments.fixed_design_name]
    fixed_dimension = next(dimension for dimension in fixed_design.dims if dimension != "nmeasure")
    outer_labels = _dimension_labels(dataset, fixed_dimension)
    fixed_scales = _expand_values(
        arguments.fixed_proposal_sd,
        size=adapter.problem.n_fixed_coefficients,
        name="fixed_proposal_sd",
    )
    cycle_length = 3 + PAIR_ALLOCATION_REFRESH_SLOTS + len(fixed_scales)
    manifest = _build_manifest(
        arguments,
        adapter,
        input_digest=input_digest,
        cycle_length=cycle_length,
        outer_labels=outer_labels,
    )
    setup_seconds = perf_counter() - setup_started

    if arguments.dry_run:
        dry_run_summary = {
            "status": "diagnostic_dry_run",
            "communication_component": _COMMUNICATION_COMPONENT,
            "closure": closure,
            "cycle_length": cycle_length,
            "requested_atomic_transitions": arguments.cycles * cycle_length,
            "input": {
                "id": arguments.input_id,
                "path": str(arguments.input.resolve()),
                "sha256": input_digest,
            },
            "manifest": manifest,
            "performance": {
                "input_hash_and_load_seconds": input_seconds,
                "problem_setup_seconds": setup_seconds,
            },
        }
        dataset.close()
        return dry_run_summary

    sampling_started = perf_counter()
    result = sample_full_tiling_compound(
        problem,
        initial_state,
        FullTilingCompoundConfig(
            iterations=arguments.cycles * cycle_length,
            seed=arguments.seed,
            pair_allocation_refresh_slots=PAIR_ALLOCATION_REFRESH_SLOTS,
            fixed_coefficient_proposal_sd=fixed_scales,
        ),
    )
    sampling_seconds = perf_counter() - sampling_started
    if (
        result.trace.global_transition.size != arguments.cycles * cycle_length
        or result.checkpoint.schedule_phase != 0
    ):
        raise RuntimeError("Sampler did not finish at the requested complete-cycle boundary.")
    summary = _summary(
        result,
        initial_state=initial_state,
        closure=closure,
        input_path=arguments.input,
        input_digest=input_digest,
        cycles=arguments.cycles,
        cycle_length=cycle_length,
        input_seconds=input_seconds,
        setup_seconds=setup_seconds,
        sampling_seconds=sampling_seconds,
    )
    manifest_text = _canonical_json(manifest)
    trace_dataset = _trace_to_dataset(
        result,
        adapter=adapter,
        outer_labels=outer_labels,
        input_digest=input_digest,
        manifest_digest=sha256(manifest_text.encode("utf-8")).hexdigest(),
        fixed_k=arguments.k,
    )
    _write_outputs(
        arguments.output_directory,
        result,
        manifest=manifest,
        summary=summary,
        trace_dataset=trace_dataset,
        netcdf_engine=arguments.netcdf_engine,
    )
    dataset.close()
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run one fixed-``K`` smoke chain and print its JSON summary.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Process status zero after successful validation and output writing.

    Raises:
        OSError: If input or output files cannot be read or written.
        ValueError: If the CLI or scientific data contract is invalid.
        RuntimeError: If sampling or artifact validation fails.
    """
    arguments = build_parser().parse_args(argv)
    summary = run(arguments)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
