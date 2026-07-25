"""Run or resume one topology-conditioned PyMC HMC full-tiling segment.

The driver consumes one explicitly frozen NetCDF snapshot and runs exactly one
complete-sweep segment of the experimental structural-then-HMC kernel. Every
structural outcome is followed by a deterministic full precision rebuild for
the selected topology and one non-adapting PyMC HamiltonianMC transition.
There is no user metric knob, online adaptation, or step-size randomization.

``--calibration-file`` is strict JSON schema v3. It binds the frozen target and
the selected step size/leapfrog count to the coordinate, metric-semantics,
precision-builder, and reference identities. Its evidence contains exactly
three ordered development validation trajectories (``development-nominal``,
``development-a``, ``development-b``) and two ordered held-out trajectories
(``held-out-a``, ``held-out-b``). Every trajectory records its initializer,
topology/master seeds, topology and topology-precision hashes, 500-sweep
acceptance evidence, and the finite/divergence/displacement gates. The selected
controls are certified against all 21 ordered grid candidates, each with three
200-sweep development trajectories. The decision maximizes the worst
Mahalanobis-squared displacement per gradient evaluation before applying the
documented acceptance, throughput, and step-size tie-breaks. Hashes bind the
grid, every candidate result, and the complete validation evidence.
Calibration schemas v1 and v2 are rejected.

Fresh chains may use the deterministic largest-nominal initializer or a
separately seeded random-recursive topology. All five calibration topology
hashes and their reviewed random-recursive topology seeds are excluded from
retained production starts. Resumed chains load the dedicated checksummed
no-pickle checkpoint and retain the original immutable run manifest.

PyTensor must use float64. Before a dry run, fresh segment, or resumed segment,
the compiled density is checked against the independently assembled scientific
target plus the symmetric log-coordinate Jacobians. Successful runs publish
``manifest.json``, ``trace.nc``, ``summary.json``, and ``checkpoint.npz`` in a
new output directory. Every artifact is reopened and audited before
``complete.json`` is hash-certified and written last. The driver refuses every
output path beneath a directory whose name contains ``PARIS_inversions``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import xarray as xr

from openghg_inversions.experimental.rjmcmc import (
    full_tiling_pymc_hmc as full_tiling_pymc_hmc_kernel,
)
from openghg_inversions.experimental.rjmcmc.full_tiling import (
    LeafTiling,
    Rectangle,
    TilingState,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_io import (
    full_tiling_state_fingerprint,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    build_full_tiling_posterior_state,
    full_tiling_problem_from_gamma_beta_adapter,
    initialize_full_tiling_posterior_state,
    initialize_random_full_tiling_posterior_state,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_pymc_hmc import (
    FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID,
    FULL_TILING_PYMC_HMC_METRIC_BUILDER_ID,
    FULL_TILING_PYMC_HMC_METRIC_REFERENCE_ID,
    FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID,
    FULL_TILING_PYMC_HMC_SCHEDULE_ID,
    FullTilingPyMCHMCCheckpoint,
    FullTilingPyMCHMCConfig,
    FullTilingPyMCHMCKernelSettings,
    FullTilingPyMCHMCSamplingResult,
    build_full_tiling_pymc_hmc_model,
    build_full_tiling_pymc_hmc_topology_precision,
    continue_full_tiling_pymc_hmc,
    full_tiling_pymc_hmc_runtime_identity,
    sample_full_tiling_pymc_hmc,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_pymc_hmc_io import (
    load_full_tiling_pymc_hmc_checkpoint,
    save_full_tiling_pymc_hmc_checkpoint,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    GammaBetaRHIMEAdapterResult,
    gamma_beta_problem_from_rhime_inputs,
)

NetCDFEngine = Literal["h5netcdf", "netcdf4", "scipy"]

MANIFEST_FILENAME = "manifest.json"
TRACE_FILENAME = "trace.nc"
SUMMARY_FILENAME = "summary.json"
CHECKPOINT_FILENAME = "checkpoint.npz"
COMPLETION_FILENAME = "complete.json"
CALIBRATION_SCHEMA = "openghg_inversions.full_tiling_pymc_hmc_calibration.v3"
LEGACY_CALIBRATION_SCHEMAS = frozenset(
    {
        "openghg_inversions.full_tiling_pymc_hmc_calibration.v1",
        "openghg_inversions.full_tiling_pymc_hmc_calibration.v2",
    }
)
COMPLETION_SCHEMA = "openghg_inversions.full_tiling_pymc_hmc_native_completion.v3"
CALIBRATION_SELECTION_RULE_ID = (
    "maximin_mahalanobis_squared_displacement_per_gradient_then_acceptance_"
    "deviation_0.75_throughput_step_size_v1"
)
TOPOLOGY_PRECISION_REBUILD_POLICY = "full_rebuild_after_every_structural_outcome_before_hmc_v1"
CALIBRATION_CANDIDATE_STEP_SIZES = (0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75)
CALIBRATION_CANDIDATE_LEAPFROG_STEPS = (3, 5, 8)
CALIBRATION_DEVELOPMENT_TOPOLOGY_SEEDS = {
    50: (None, 42050, 42051),
    250: (None, 42250, 42251),
}
CALIBRATION_DEVELOPMENT_MASTER_SEEDS = {
    50: (73050, 73051, 73052),
    250: (73250, 73251, 73252),
}
CALIBRATION_HELD_OUT_TOPOLOGY_SEEDS = {
    50: (42052, 42053),
    250: (42252, 42253),
}
CALIBRATION_VALIDATION_MASTER_SEEDS = {
    50: (83050, 83051, 83052, 74050, 74051),
    250: (83250, 83251, 83252, 74250, 74251),
}
PARIS_OBSERVATIONS = 1_382
PARIS_GRID_SHAPE = (183, 128)
PARIS_OUTER_COEFFICIENTS = 6
_CLOSURE_RTOL = 1.0e-12
_CLOSURE_ATOL = 1.0e-12
_TRANSFORMED_TARGET_ATOL = 5.0e-10
_COMMUNICATION_COMPONENT = "component_reachable_from_recorded_initial_tiling"
_BUNDLE_ARTIFACTS = (
    MANIFEST_FILENAME,
    TRACE_FILENAME,
    SUMMARY_FILENAME,
    CHECKPOINT_FILENAME,
)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    """Return deterministic strict JSON terminated by one newline."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    raise ValueError(f"Non-standard JSON constant {value!r} is forbidden.")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting duplicate member names."""
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON object member {name!r} is forbidden.")
        result[name] = value
    return result


def _load_strict_json_object(path: Path, *, description: str) -> dict[str, Any]:
    """Load one UTF-8 strict-JSON file whose root must be an object.

    Args:
        path: JSON file to load.
        description: Human-readable artifact name for validation errors.

    Returns:
        Parsed JSON object with duplicate keys and non-finite constants
        rejected at every nesting level.

    Raises:
        FileNotFoundError: If ``path`` is not a regular file.
        ValueError: If UTF-8 decoding, JSON parsing, or root-type validation
            fails.
        OSError: If the file cannot be read.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{description} is not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} is not valid UTF-8.") from error
    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is not valid strict JSON.") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{description} root must be a JSON object.")
    return parsed


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
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_trace(
    dataset: xr.Dataset,
    path: Path,
    *,
    engine: NetCDFEngine,
) -> None:
    """Publish one create-only NetCDF trace through a same-directory link.

    Args:
        dataset: Fully materialized trace dataset to serialize. The dataset is
            read but not closed or mutated.
        path: Final artifact path, whose parent must already exist. An
            existing target is never replaced.
        engine: Explicit xarray NetCDF backend used for serialization.

    Raises:
        FileExistsError: If ``path`` already exists when the temporary file is
            linked into place.
        OSError: If temporary-file creation, serialization, flushing, linking,
            cleanup, or directory synchronization fails.
        ValueError: If ``dataset`` cannot be represented by ``engine``.

    Notes:
        Serialization occurs in a uniquely named sibling temporary file. The
        file is flushed, linked create-only to ``path``, and followed by a
        parent-directory sync; best-effort temporary cleanup runs on every
        exit path.
    """
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        dataset.to_netcdf(temporary, engine=engine)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_output_backend(
    parent: Path,
    *,
    engine: NetCDFEngine,
) -> None:
    """Verify exact trace-like NetCDF round trips on the output filesystem.

    Args:
        parent: Existing directory in which the temporary probe is written.
        engine: Explicit xarray NetCDF backend to exercise.

    Raises:
        OSError: If the probe cannot be created, written, reopened, or removed.
        RuntimeError: If float64, int64, uint64, or string values change during
            the backend round trip.
        ValueError: If the selected backend cannot serialize the probe.

    Notes:
        The probe exercises the dtypes required by production traces. It is
        always removed and never creates the requested output directory.
    """
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".full-tiling-pymc-hmc-netcdf-preflight-",
        suffix=".nc",
        dir=parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        probe = xr.Dataset(
            {
                "float64": ("item", np.asarray([1.0], dtype=np.float64)),
                "int64": ("item", np.asarray([1], dtype=np.int64)),
                "uint64": ("item", np.asarray([2**63 + 1], dtype=np.uint64)),
                "string": ("item", np.asarray(["probe"], dtype=np.str_)),
            }
        )
        probe.to_netcdf(temporary, engine=engine)
        probe.close()
        with xr.open_dataset(temporary, engine=engine) as reopened:
            loaded = reopened.load()
        try:
            for name in ("float64", "int64", "uint64", "string"):
                if not np.array_equal(
                    np.asarray(loaded[name].values),
                    np.asarray(probe[name].values),
                ):
                    raise RuntimeError(f"NetCDF backend preflight changed {name} data.")
        finally:
            loaded.close()
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


def _validate_sha256(value: str, *, name: str) -> None:
    """Require one lower- or upper-case hexadecimal SHA-256 string."""
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(f"{name} must be exactly 64 hexadecimal characters.")


def _load_frozen_dataset(path: Path, *, engine: NetCDFEngine) -> xr.Dataset:
    """Eagerly load and close one immutable-on-entry NetCDF input."""
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
    return np.asarray(
        [str(value) for value in values.tolist()],
        dtype=np.str_,
    )


def _build_adapter(
    dataset: xr.Dataset,
    arguments: argparse.Namespace,
) -> GammaBetaRHIMEAdapterResult:
    """Build a fixed-``K`` Gamma--Beta adapter from explicit CLI controls.

    Args:
        dataset: Eagerly loaded frozen native input dataset.
        arguments: Validated CLI namespace naming every scientific input
            variable and supplying target, weight, and fixed-prior controls.

    Returns:
        Adapter result whose problem, spatial metadata, and normalization
        factor are derived from ``dataset`` without variable-name inference.

    Raises:
        ValueError: If a required variable is absent, the fixed design is not
            two-dimensional with ``nmeasure``, scalar/vector priors cannot be
            resolved to the fixed-block width, or adapter scientific
            validation fails.
    """
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
        nominal_weight=_input_array(
            dataset,
            arguments.nominal_weight_name,
        ),
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
    """Require the exact reviewed modern PARIS dimensions and labels.

    Args:
        dataset: Frozen dataset containing fixed-design and coordinate labels.
        adapter: Scientific adapter whose observation, grid, and fixed-block
            dimensions are checked.
        fixed_design_name: Explicit fixed-design variable name.
        expected_outer_labels: Reviewed six-label sequence in model order.

    Raises:
        KeyError: If the fixed design or a required dimension is absent.
        ValueError: If observation count, grid shape, fixed-block width,
            dimension names, outer-label order, or unique observation labels
            differ from the reviewed profile.
    """
    actual = (
        int(adapter.problem.observations.size),
        adapter.spatial_shape,
        adapter.problem.n_fixed_coefficients,
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
    """Verify mass-coordinate and complete prior-mean prediction closure.

    Args:
        dataset: Frozen dataset containing raw sensitivity, fixed design, and
            fixed offset arrays.
        adapter: Adapted Gamma--Beta problem whose mass-coordinate sensitivity
            and fixed prior are audited.
        initial_state: Prior-mean initialized scientific state.
        sensitivity_name: Raw gridded sensitivity variable name.
        fixed_design_name: Fixed coefficient design variable name.
        fixed_offset_name: Always-active observation-offset variable name.

    Returns:
        Maximum absolute mass-coordinate and complete prior-mean prediction
        errors, both computed in observation space.

    Raises:
        KeyError: If a named input variable or required dimension is absent.
        ValueError: If array dimensions are incompatible or either closure
            comparison exceeds the frozen relative/absolute tolerances.
        RuntimeError: If the adapted problem lacks required fixed design or
            offset terms.
    """
    problem = adapter.problem
    sensitivity = dataset[sensitivity_name].transpose(
        "nmeasure",
        "lat",
        "lon",
    )
    scaling_prediction = np.asarray(
        sensitivity.values,
        dtype=np.float64,
    ).sum(axis=(1, 2))
    mass_prediction = problem.sensitivity @ problem.prior.nominal_cell_mass
    mass_error = np.asarray(
        mass_prediction - scaling_prediction,
        dtype=np.float64,
    )
    if not np.allclose(
        mass_prediction,
        scaling_prediction,
        rtol=_CLOSURE_RTOL,
        atol=_CLOSURE_ATOL,
    ):
        raise ValueError(
            "Mass-coordinate closure failed: sensitivity_per_mass @ "
            "nominal_weight does not reproduce the all-one fp_x_flux "
            "prediction."
        )
    if problem.fixed_block is None or problem.fixed_offset is None:
        raise RuntimeError("The native PyMC HMC driver requires fixed design and offset terms.")
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
    total_error = np.asarray(
        initial_state.prediction - expected_total,
        dtype=np.float64,
    )
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


def _rectangle_bounds(state: FullTilingPosteriorState) -> list[list[int]]:
    """Return canonical rectangle bounds as strict-JSON integers."""
    return [
        [
            leaf.row_start,
            leaf.row_stop,
            leaf.col_start,
            leaf.col_stop,
        ]
        for leaf in state.tiling_state.tiling.leaves
    ]


def _topology_sha256(bounds: list[list[int]]) -> str:
    """Hash canonical rectangle bounds independently of sampler output."""
    return sha256(_canonical_json(bounds).encode("utf-8")).hexdigest()


def _json_sha256(value: object) -> str:
    """Hash one value using the calibration contract's canonical JSON."""
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _calibration_candidate_grid() -> dict[str, object]:
    """Return the exact ordered H2d candidate grid."""
    return {
        "step_sizes": list(CALIBRATION_CANDIDATE_STEP_SIZES),
        "leapfrog_steps": list(CALIBRATION_CANDIDATE_LEAPFROG_STEPS),
        "sweeps_per_development_trajectory": 200,
        "ordering": "step_size_major_then_leapfrog_steps_v1",
    }


def _topology_precision_sha256(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
) -> str:
    """Build and hash a topology precision using checkpoint byte semantics."""
    precision = build_full_tiling_pymc_hmc_topology_precision(problem, state)
    digest = sha256()
    digest.update(np.asarray(precision.shape, dtype="<i8").tobytes())
    digest.update(
        np.asarray(
            precision,
            dtype="<f8",
            order="C",
        ).tobytes(order="C")
    )
    return digest.hexdigest()


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


def _transformed_target_preflight(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
    *,
    log_leaf_mass: np.ndarray | None = None,
    log_fixed_coefficient: np.ndarray | None = None,
) -> dict[str, float]:
    """Compare compiled PyMC logp with the exact transformed target.

    Args:
        problem: Scientific problem owning ``state``.
        state: Supported scientific state to audit.
        log_leaf_mass: Optional authoritative symmetric leaf coordinates.
        log_fixed_coefficient: Optional authoritative fixed coordinates.

    Returns:
        Strict-JSON-compatible compiled, expected, Jacobian, difference, and
        tolerance values.

    Raises:
        ValueError: If coordinates do not exactly decode to ``state`` or the
            compiled target differs beyond the float64 audit tolerance.
        ImportError: If PyMC or PyTensor is unavailable.
        RuntimeError: If PyTensor is not configured for float64.
    """
    runtime = full_tiling_pymc_hmc_runtime_identity()
    if runtime.pytensor_float_x != "float64":
        raise RuntimeError("PyTensor floatX must be exactly float64.")
    x = (
        np.asarray(log_leaf_mass, dtype=np.float64)
        if log_leaf_mass is not None
        else np.log(state.leaf_masses)
    )
    y = (
        np.asarray(log_fixed_coefficient, dtype=np.float64)
        if log_fixed_coefficient is not None
        else np.log(state.fixed_coefficients)
    )
    if x.shape != state.leaf_masses.shape or y.shape != state.fixed_coefficients.shape:
        raise ValueError("Transformed-target preflight coordinates have invalid shapes.")
    if (
        np.any(~np.isfinite(x))
        or np.any(~np.isfinite(y))
        or not np.array_equal(np.exp(x), state.leaf_masses)
        or not np.array_equal(np.exp(y), state.fixed_coefficients)
    ):
        raise ValueError(
            "Transformed-target preflight coordinates do not exactly decode to the scientific state."
        )
    model = build_full_tiling_pymc_hmc_model(problem, state)
    point = model.initial_point()
    point["x"] = np.array(x, copy=True)
    if state.fixed_coefficients.size:
        point["y"] = np.array(y, copy=True)
    compiled = float(model.compile_logp()(point))
    log_total = float(np.logaddexp.reduce(x))
    leaf_jacobian = float(x.sum() - (state.k - 1) * log_total)
    fixed_jacobian = float(y.sum())
    expected = float(state.log_target + leaf_jacobian + fixed_jacobian)
    difference = float(compiled - expected)
    tolerance = float(
        max(
            _TRANSFORMED_TARGET_ATOL,
            128.0 * np.finfo(np.float64).eps * max(1.0, abs(compiled), abs(expected)),
        )
    )
    if not math.isfinite(compiled) or not math.isfinite(expected) or abs(difference) > tolerance:
        raise ValueError(
            "Compiled PyMC transformed target does not match the exact "
            f"scientific target plus Jacobians: difference={difference:.17g}, "
            f"tolerance={tolerance:.17g}."
        )
    return {
        "compiled_logp": compiled,
        "scientific_log_target": float(state.log_target),
        "leaf_log_coordinate_jacobian": leaf_jacobian,
        "fixed_log_coordinate_jacobian": fixed_jacobian,
        "expected_transformed_logp": expected,
        "difference": difference,
        "absolute_tolerance": tolerance,
    }


def _build_manifest(
    arguments: argparse.Namespace,
    adapter: GammaBetaRHIMEAdapterResult,
    *,
    initial_state: FullTilingPosteriorState,
    input_digest: str,
    outer_labels: np.ndarray,
    calibration: Mapping[str, Any],
    calibration_digest: str,
) -> dict[str, object]:
    """Build immutable checkpoint identity independent of segment reporting.

    The manifest intentionally excludes segment length, output paths, wall
    timings, and parent-checkpoint location so one logical chain can resume
    through any number of separately published complete-sweep segments.

    Args:
        arguments: Validated CLI arguments.
        adapter: Frozen RHIME-to-Gamma--Beta adapter result.
        initial_state: Deterministically reconstructed fresh-chain state.
        input_digest: Frozen input whole-file SHA-256.
        outer_labels: Ordered fixed-coefficient labels.
        calibration: Verified v3 calibration identity.
        calibration_digest: Verified whole-file calibration SHA-256.

    Returns:
        Strict-JSON-compatible immutable run identity.

    Raises:
        RuntimeError: If the required fixed block is absent.
    """
    fixed_block = adapter.problem.fixed_block
    if fixed_block is None:
        raise RuntimeError("The native PyMC HMC driver requires a fixed design block.")
    bounds = _rectangle_bounds(initial_state)
    initial_precision_digest = _topology_precision_sha256(
        initial_state.problem,
        initial_state,
    )
    runtime_identity = asdict(full_tiling_pymc_hmc_runtime_identity())
    manifest: dict[str, object] = {
        "schema": ("openghg_inversions.full_tiling_pymc_hmc_native_manifest.v3"),
        "status": "experimental_topology_conditioned_hmc_not_convergence_evidence",
        "input": {
            "id": arguments.input_id,
            "sha256": input_digest,
            "contract": _input_contract(arguments),
            "weight_normalization_factor": (adapter.weight_normalization_factor),
        },
        "model": {
            "fixed_k": int(arguments.k),
            "grid_shape": list(adapter.spatial_shape),
            "observations": int(adapter.problem.observations.size),
            "fixed_coefficients": adapter.problem.n_fixed_coefficients,
            "outer_labels": outer_labels.astype(str).tolist(),
            "concentration": float(arguments.concentration),
            "root_variance": float(arguments.root_variance),
            "root_prior_shape": float(adapter.problem.prior.root_shape),
            "root_prior_rate": float(adapter.problem.prior.root_rate),
            "likelihood_power": float(arguments.likelihood_power),
            "fixed_prior_mean": (fixed_block.coefficient_prior_mean.astype(float).tolist()),
            "fixed_prior_sd": (fixed_block.coefficient_prior_sd.astype(float).tolist()),
            "structural_target": ("uniform_over_unique_canonical_tilings_at_fixed_k"),
            "communication_component": _COMMUNICATION_COMPONENT,
            "connectivity_proven": False,
        },
        "initialization": {
            "strategy": arguments.initialization,
            "seed": arguments.initialization_seed,
            "rng_stream": ("dedicated_pcg64" if arguments.initialization == "random-recursive" else "none"),
            "rectangle_bounds": bounds,
            "topology_sha256": _topology_sha256(bounds),
            "topology_precision_sha256": initial_precision_digest,
            "state_sha256": full_tiling_state_fingerprint(
                initial_state.problem,
                initial_state,
            ),
        },
        "sampler": {
            "name": "full_tiling_structure_then_topology_conditioned_pymc_hmc",
            "schedule_id": FULL_TILING_PYMC_HMC_SCHEDULE_ID,
            "chains_per_invocation": 1,
            "step_size_requested": float(arguments.step_size),
            "leapfrog_steps": int(arguments.leapfrog_steps),
            "metric_semantics_id": (FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID),
            "metric_builder_id": FULL_TILING_PYMC_HMC_METRIC_BUILDER_ID,
            "metric_reference_id": FULL_TILING_PYMC_HMC_METRIC_REFERENCE_ID,
            "metric_rebuild_policy": TOPOLOGY_PRECISION_REBUILD_POLICY,
            "coordinate_layout_id": (FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID),
            "adapt_step_size": False,
            "step_size_randomization": False,
            "topology_dependent_metric": True,
            "runtime_identity": runtime_identity,
            "calibration": {
                "schema": calibration["schema"],
                "id": calibration["calibration_id"],
                "sha256": calibration_digest,
            },
        },
        "provenance": {
            "input_id": arguments.input_id,
            "code_revision": arguments.code_revision,
            "chain_id": arguments.chain_id,
            "original_master_seed": int(arguments.seed),
            "single_process": True,
            "durable_checkpoint": True,
        },
    }
    manifest["manifest_payload_sha256"] = sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    return manifest


def _expected_calibration_identity(
    arguments: argparse.Namespace,
    adapter: GammaBetaRHIMEAdapterResult,
    *,
    input_digest: str,
) -> dict[str, Any]:
    """Build the exact minimal v3 calibration identity for this invocation.

    Args:
        arguments: Validated driver arguments.
        adapter: Frozen scientific adapter with resolved fixed priors.
        input_digest: Verified frozen-input whole-file SHA-256.

    Returns:
        Strict-JSON-compatible calibration identity.

    Raises:
        RuntimeError: If the required fixed coefficient block is absent.
    """
    fixed_block = adapter.problem.fixed_block
    if fixed_block is None:
        raise RuntimeError("The native PyMC HMC driver requires a fixed design block.")
    return {
        "schema": CALIBRATION_SCHEMA,
        "calibration_id": arguments.calibration_id,
        "fixed_k": int(arguments.k),
        "input_sha256": input_digest,
        "target": {
            "concentration": float(arguments.concentration),
            "root_variance": float(arguments.root_variance),
            "likelihood_power": float(arguments.likelihood_power),
            "fixed_prior_mean": fixed_block.coefficient_prior_mean.astype(float).tolist(),
            "fixed_prior_sd": fixed_block.coefficient_prior_sd.astype(float).tolist(),
            "nominal_weight_policy": arguments.nominal_weight_policy,
            "normalize_weights": bool(arguments.normalize_weights),
        },
        "kernel": {
            "step_size": float(arguments.step_size),
            "leapfrog_steps": int(arguments.leapfrog_steps),
            "coordinate_layout_id": FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID,
            "metric_semantics_id": FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID,
            "metric_builder_id": FULL_TILING_PYMC_HMC_METRIC_BUILDER_ID,
            "metric_reference_id": FULL_TILING_PYMC_HMC_METRIC_REFERENCE_ID,
        },
    }


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    """Require one JSON object with exactly the expected member names."""
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)!r}.")
    return value


def _finite_json_number(value: object, *, name: str) -> float:
    """Return one finite JSON number while rejecting Booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite JSON number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite JSON number.")
    return result


def _validate_calibration_evidence(
    value: object,
    arguments: argparse.Namespace,
) -> None:
    """Validate the exact H2d selected-control and trajectory evidence.

    Args:
        value: Parsed strict-JSON calibration ``evidence`` value.
        arguments: Validated CLI namespace supplying the current code
            revision, fixed ``K``, step size, and leapfrog count.

    Raises:
        ValueError: If identities, ordered trajectory evidence, selected
            controls, evidence/source hashes, or exclusion hashes violate the
            v3 contract.

    Notes:
        The development hash covers the ordered development array. The
        validation hash covers one object containing both ordered development
        and held-out arrays. Production ``K=50``/``K=250`` evidence is also
        bound to the reviewed topology and master PCG64 seeds.
    """
    evidence = _require_exact_keys(
        value,
        {
            "code_revision",
            "input_sha256",
            "candidate_grid",
            "candidate_results",
            "development",
            "held_out",
            "selected",
            "excluded_production_topology_sha256",
            "source_artifact_sha256",
        },
        name="Calibration evidence",
    )
    if evidence["code_revision"] != arguments.code_revision:
        raise ValueError("Calibration evidence code_revision does not match --code-revision.")
    input_sha256 = evidence["input_sha256"]
    if not isinstance(input_sha256, str):
        raise ValueError("Calibration evidence input_sha256 must be a string.")
    _validate_sha256(input_sha256, name="Calibration evidence input SHA-256")
    if input_sha256.lower() != arguments.expected_input_sha256.lower():
        raise ValueError("Calibration evidence input_sha256 does not match the frozen input.")

    candidate_grid = evidence["candidate_grid"]
    expected_candidate_grid = _calibration_candidate_grid()
    if candidate_grid != expected_candidate_grid:
        raise ValueError("Calibration candidate_grid does not exactly match the predeclared H2d grid.")
    candidate_results = evidence["candidate_results"]
    if not isinstance(candidate_results, list):
        raise ValueError("Calibration candidate_results must be an ordered array.")
    expected_controls = [
        (step_size, leapfrog_steps)
        for step_size in CALIBRATION_CANDIDATE_STEP_SIZES
        for leapfrog_steps in CALIBRATION_CANDIDATE_LEAPFROG_STEPS
    ]
    if len(candidate_results) != len(expected_controls):
        raise ValueError("Calibration candidate_results must contain every predeclared grid candidate.")

    development = evidence["development"]
    held_out = evidence["held_out"]
    if not isinstance(development, list) or len(development) != 3:
        raise ValueError("Calibration evidence must contain exactly three development trajectories.")
    if not isinstance(held_out, list) or len(held_out) != 2:
        raise ValueError("Calibration evidence must contain exactly two held-out trajectories.")

    expected_roles = (
        ("development-nominal", "largest-nominal"),
        ("development-a", "random-recursive"),
        ("development-b", "random-recursive"),
        ("held-out-a", "random-recursive"),
        ("held-out-b", "random-recursive"),
    )
    reviewed_topology_seeds = (
        *CALIBRATION_DEVELOPMENT_TOPOLOGY_SEEDS.get(arguments.k, (None, None, None)),
        *CALIBRATION_HELD_OUT_TOPOLOGY_SEEDS.get(arguments.k, (None, None)),
    )
    reviewed_master_seeds = CALIBRATION_VALIDATION_MASTER_SEEDS.get(arguments.k)
    reviewed_candidate_master_seeds = CALIBRATION_DEVELOPMENT_MASTER_SEEDS.get(arguments.k)
    trajectories = [*development, *held_out]
    topology_hashes: dict[str, str] = {}
    topology_precision_hashes: list[str] = []
    topology_seeds: list[int] = []
    master_seeds: list[int] = []
    trajectory_keys = {
        "role",
        "initializer",
        "topology_seed",
        "master_seed",
        "topology_sha256",
        "topology_precision_sha256",
        "sweeps",
        "mean_acceptance",
        "divergences",
        "finite_scientific_endpoints",
        "finite_transformed_endpoints",
        "accepted_nonzero_displacement",
    }
    for index, ((expected_role, expected_initializer), expected_seed) in enumerate(
        zip(expected_roles, reviewed_topology_seeds, strict=True)
    ):
        trajectory = _require_exact_keys(
            trajectories[index],
            trajectory_keys,
            name=f"Calibration trajectory {index}",
        )
        if trajectory["role"] != expected_role or trajectory["initializer"] != expected_initializer:
            raise ValueError("Calibration trajectory roles and initializers are out of order.")
        topology_seed = trajectory["topology_seed"]
        if expected_initializer == "largest-nominal":
            if topology_seed is not None:
                raise ValueError("The largest-nominal calibration topology seed must be null.")
        else:
            if isinstance(topology_seed, bool) or not isinstance(topology_seed, int) or topology_seed < 0:
                raise ValueError("Random-recursive calibration topology seeds must be non-negative integers.")
            topology_seeds.append(topology_seed)
        if arguments.k in CALIBRATION_DEVELOPMENT_TOPOLOGY_SEEDS and topology_seed != expected_seed:
            raise ValueError(f"Calibration evidence for K={arguments.k} has incompatible topology seeds.")
        master_seed = trajectory["master_seed"]
        if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
            raise ValueError("Calibration master seeds must be non-negative integers.")
        master_seeds.append(master_seed)
        if reviewed_master_seeds is not None and master_seed != reviewed_master_seeds[index]:
            raise ValueError(f"Calibration evidence for K={arguments.k} has incompatible master seeds.")
        for field in ("topology_sha256", "topology_precision_sha256"):
            digest = trajectory[field]
            if not isinstance(digest, str):
                raise ValueError(f"Calibration trajectory {field} must be a string.")
            _validate_sha256(digest, name=f"Calibration trajectory {field}")
            if digest != digest.lower():
                raise ValueError(f"Calibration trajectory {field} must be lowercase.")
        topology_hashes[expected_role] = trajectory["topology_sha256"]
        topology_precision_hashes.append(trajectory["topology_precision_sha256"])
        if (
            isinstance(trajectory["sweeps"], bool)
            or not isinstance(trajectory["sweeps"], int)
            or trajectory["sweeps"] != 500
        ):
            raise ValueError("Every calibration validation trajectory must contain 500 sweeps.")
        mean_acceptance = _finite_json_number(
            trajectory["mean_acceptance"],
            name=f"Calibration trajectory {index}.mean_acceptance",
        )
        divergences = trajectory["divergences"]
        if (
            not 0.60 <= mean_acceptance <= 0.95
            or isinstance(divergences, bool)
            or not isinstance(divergences, int)
            or divergences != 0
            or trajectory["finite_scientific_endpoints"] is not True
            or trajectory["finite_transformed_endpoints"] is not True
            or trajectory["accepted_nonzero_displacement"] is not True
        ):
            raise ValueError("Calibration trajectory does not pass the frozen validation gates.")
    if len(set(topology_seeds)) != len(topology_seeds):
        raise ValueError("Calibration random-recursive topology seeds must be distinct.")
    if len(set(master_seeds)) != len(master_seeds):
        raise ValueError("Calibration master seeds must be distinct.")
    if len(set(topology_hashes.values())) != len(topology_hashes):
        raise ValueError("Calibration topology hashes must be distinct.")
    if len(set(topology_precision_hashes)) != len(topology_precision_hashes):
        raise ValueError("Calibration topology precision hashes must be distinct.")

    candidate_keys = {
        "step_size",
        "leapfrog_steps",
        "development",
        "development_admissible",
    }
    decision_trajectory_keys = {
        *trajectory_keys,
        "mean_mahalanobis_squared_displacement_per_gradient",
        "throughput_sweeps_per_second",
    }
    development_topology_identity: dict[str, tuple[str, str]] = {}
    admissible_candidates: list[tuple[tuple[float, float, float, float], float, int]] = []
    for candidate_index, expected_control in enumerate(expected_controls):
        candidate = _require_exact_keys(
            candidate_results[candidate_index],
            candidate_keys,
            name=f"Calibration candidate result {candidate_index}",
        )
        candidate_step_size = _finite_json_number(
            candidate["step_size"],
            name=f"Calibration candidate result {candidate_index}.step_size",
        )
        candidate_leapfrog_steps = candidate["leapfrog_steps"]
        if (
            candidate_step_size != expected_control[0]
            or isinstance(candidate_leapfrog_steps, bool)
            or not isinstance(candidate_leapfrog_steps, int)
            or candidate_leapfrog_steps != expected_control[1]
        ):
            raise ValueError("Calibration candidate_results are not in exact predeclared grid order.")
        candidate_development = candidate["development"]
        if not isinstance(candidate_development, list) or len(candidate_development) != 3:
            raise ValueError(
                "Every calibration candidate must contain exactly three development trajectories."
            )
        acceptance_rates: list[float] = []
        displacement_per_gradient: list[float] = []
        throughputs: list[float] = []
        passes = True
        for role_index, (expected_role, expected_initializer) in enumerate(expected_roles[:3]):
            trajectory = _require_exact_keys(
                candidate_development[role_index],
                decision_trajectory_keys,
                name=(f"Calibration candidate result {candidate_index} development trajectory {role_index}"),
            )
            if trajectory["role"] != expected_role or trajectory["initializer"] != expected_initializer:
                raise ValueError("Calibration candidate development roles and initializers are out of order.")
            expected_topology_seed = reviewed_topology_seeds[role_index]
            topology_seed = trajectory["topology_seed"]
            if expected_initializer == "largest-nominal":
                if topology_seed is not None:
                    raise ValueError("The largest-nominal candidate topology seed must be null.")
            elif isinstance(topology_seed, bool) or not isinstance(topology_seed, int) or topology_seed < 0:
                raise ValueError("Random-recursive candidate topology seeds must be non-negative integers.")
            if (
                arguments.k in CALIBRATION_DEVELOPMENT_TOPOLOGY_SEEDS
                and topology_seed != expected_topology_seed
            ):
                raise ValueError(
                    f"Calibration candidate evidence for K={arguments.k} has incompatible topology seeds."
                )
            master_seed = trajectory["master_seed"]
            if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
                raise ValueError("Calibration candidate master seeds must be non-negative integers.")
            if (
                reviewed_candidate_master_seeds is not None
                and master_seed != reviewed_candidate_master_seeds[role_index]
            ):
                raise ValueError(
                    f"Calibration candidate evidence for K={arguments.k} has incompatible master seeds."
                )
            for field in ("topology_sha256", "topology_precision_sha256"):
                digest = trajectory[field]
                if not isinstance(digest, str):
                    raise ValueError(f"Calibration candidate trajectory {field} must be a string.")
                _validate_sha256(digest, name=f"Calibration candidate trajectory {field}")
                if digest != digest.lower():
                    raise ValueError(f"Calibration candidate trajectory {field} must be lowercase.")
            identity = (
                trajectory["topology_sha256"],
                trajectory["topology_precision_sha256"],
            )
            previous_identity = development_topology_identity.setdefault(expected_role, identity)
            if identity != previous_identity:
                raise ValueError(
                    "Calibration candidate topology identities must be constant across the grid."
                )
            if (
                isinstance(trajectory["sweeps"], bool)
                or not isinstance(trajectory["sweeps"], int)
                or trajectory["sweeps"] != 200
            ):
                raise ValueError(
                    "Every calibration candidate development trajectory must contain 200 sweeps."
                )
            mean_acceptance = _finite_json_number(
                trajectory["mean_acceptance"],
                name=(
                    f"Calibration candidate result {candidate_index} trajectory {role_index}.mean_acceptance"
                ),
            )
            divergences = trajectory["divergences"]
            if isinstance(divergences, bool) or not isinstance(divergences, int) or divergences < 0:
                raise ValueError("Calibration candidate divergences must be non-negative integers.")
            displacement = _finite_json_number(
                trajectory["mean_mahalanobis_squared_displacement_per_gradient"],
                name=(
                    f"Calibration candidate result {candidate_index} trajectory "
                    f"{role_index}.mean_mahalanobis_squared_displacement_per_gradient"
                ),
            )
            throughput = _finite_json_number(
                trajectory["throughput_sweeps_per_second"],
                name=(
                    f"Calibration candidate result {candidate_index} "
                    f"trajectory {role_index}.throughput_sweeps_per_second"
                ),
            )
            if displacement <= 0.0 or throughput <= 0.0:
                raise ValueError(
                    "Calibration candidate displacement-per-gradient and throughput "
                    "must be strictly positive."
                )
            acceptance_rates.append(mean_acceptance)
            displacement_per_gradient.append(displacement)
            throughputs.append(throughput)
            passes = passes and (
                0.60 <= mean_acceptance <= 0.95
                and divergences == 0
                and trajectory["finite_scientific_endpoints"] is True
                and trajectory["finite_transformed_endpoints"] is True
                and trajectory["accepted_nonzero_displacement"] is True
            )
        if candidate["development_admissible"] is not passes:
            raise ValueError(
                "Calibration candidate development_admissible does not match its trajectory gates."
            )
        if passes:
            selection_key = (
                -min(displacement_per_gradient),
                max(abs(acceptance - 0.75) for acceptance in acceptance_rates),
                -min(throughputs),
                candidate_step_size,
            )
            admissible_candidates.append((selection_key, candidate_step_size, candidate_leapfrog_steps))
    if not admissible_candidates:
        raise ValueError("Calibration contains no development-admissible candidate.")
    winning_key = min(item[0] for item in admissible_candidates)
    winners = [item for item in admissible_candidates if item[0] == winning_key]
    if len(winners) != 1:
        raise ValueError("Calibration candidate selection rule does not produce a unique winner.")
    _, winning_step_size, winning_leapfrog_steps = winners[0]

    for role_index, role in enumerate(expected_roles[:3]):
        expected_identity = development_topology_identity[role[0]]
        actual_identity = (
            development[role_index]["topology_sha256"],
            development[role_index]["topology_precision_sha256"],
        )
        if actual_identity != expected_identity:
            raise ValueError(
                "Calibration development validation topology identity differs from grid evidence."
            )

    selected = _require_exact_keys(
        evidence["selected"],
        {
            "step_size",
            "leapfrog_steps",
            "selection_rule_id",
            "candidate_grid_sha256",
            "candidate_results_sha256",
            "development_evidence_sha256",
            "validation_evidence_sha256",
        },
        name="Calibration selected controls",
    )
    step_size = _finite_json_number(
        selected["step_size"],
        name="Calibration selected controls.step_size",
    )
    leapfrog_steps = selected["leapfrog_steps"]
    if (
        step_size != float(arguments.step_size)
        or isinstance(leapfrog_steps, bool)
        or not isinstance(leapfrog_steps, int)
        or leapfrog_steps != int(arguments.leapfrog_steps)
    ):
        raise ValueError("Calibration selected controls do not match the requested HMC controls.")
    if step_size != winning_step_size or leapfrog_steps != winning_leapfrog_steps:
        raise ValueError("Calibration selected controls do not match the recomputed candidate-grid winner.")
    if selected["selection_rule_id"] != CALIBRATION_SELECTION_RULE_ID:
        raise ValueError("Calibration selected controls use an incompatible selection rule.")
    for field in (
        "candidate_grid_sha256",
        "candidate_results_sha256",
        "development_evidence_sha256",
        "validation_evidence_sha256",
    ):
        digest = selected[field]
        if not isinstance(digest, str):
            raise ValueError(f"Calibration selected controls {field} must be a string.")
        _validate_sha256(digest, name=f"Calibration selected controls {field}")
        if digest != digest.lower():
            raise ValueError(f"Calibration selected controls {field} must be lowercase.")
    if selected["candidate_grid_sha256"] != _json_sha256(candidate_grid):
        raise ValueError("Calibration candidate-grid SHA-256 does not match.")
    if selected["candidate_results_sha256"] != _json_sha256(candidate_results):
        raise ValueError("Calibration candidate-results SHA-256 does not match.")
    if selected["development_evidence_sha256"] != _json_sha256(development):
        raise ValueError("Calibration development evidence SHA-256 does not match.")
    validation_evidence = {
        "development": development,
        "held_out": held_out,
    }
    if selected["validation_evidence_sha256"] != _json_sha256(validation_evidence):
        raise ValueError("Calibration validation evidence SHA-256 does not match.")

    excluded_topologies = _require_exact_keys(
        evidence["excluded_production_topology_sha256"],
        set(topology_hashes),
        name="Calibration excluded production topologies",
    )
    if excluded_topologies != topology_hashes:
        raise ValueError(
            "Calibration excluded production topology hashes must exactly match all five trajectories."
        )

    expected_sources = {
        "candidate-grid": _json_sha256(candidate_grid),
        "candidate-results": _json_sha256(candidate_results),
        "development-validation": _json_sha256(development),
        "held-out-validation": _json_sha256(held_out),
    }
    sources = _require_exact_keys(
        evidence["source_artifact_sha256"],
        set(expected_sources),
        name="Calibration source_artifact_sha256",
    )
    if sources != expected_sources:
        raise ValueError("Calibration source artifact hashes do not match their embedded exact evidence.")


def _load_verified_calibration(
    arguments: argparse.Namespace,
    adapter: GammaBetaRHIMEAdapterResult,
    *,
    input_digest: str,
) -> tuple[dict[str, Any], str]:
    """Load, hash, and exactly validate the invocation's calibration file.

    Args:
        arguments: Validated driver arguments.
        adapter: Frozen scientific adapter with resolved fixed priors.
        input_digest: Verified frozen-input whole-file SHA-256.

    Returns:
        Verified calibration object and lowercase whole-file SHA-256.

    Raises:
        FileNotFoundError: If the calibration file is absent.
        OSError: If the calibration file cannot be read.
        ValueError: If its hash, strict JSON, stability, or v3 identity does
            not exactly match the current invocation.
    """
    path = cast(Path, arguments.calibration_file)
    digest = _sha256_file(path)
    if digest.lower() != arguments.calibration_sha256.lower():
        raise ValueError("Calibration file SHA-256 does not match --calibration-sha256.")
    calibration = _load_strict_json_object(
        path,
        description="Calibration file",
    )
    if _sha256_file(path) != digest:
        raise ValueError("Calibration file changed while it was being validated.")
    if calibration.get("schema") in LEGACY_CALIBRATION_SCHEMAS:
        raise ValueError(
            "Calibration schemas v1 and v2 use retired metric semantics; calibration v3 is required."
        )
    expected = _expected_calibration_identity(
        arguments,
        adapter,
        input_digest=input_digest,
    )
    if set(calibration) != {*expected, "evidence"}:
        raise ValueError("Calibration v3 root keys are incompatible.")
    identity = {name: calibration[name] for name in expected}
    if _canonical_json(identity) != _canonical_json(expected):
        raise ValueError("Calibration v3 identity does not exactly match the current invocation.")
    _validate_calibration_evidence(calibration["evidence"], arguments)
    return calibration, digest


def _trace_states(
    problem: FullTilingProblem,
    result: FullTilingPyMCHMCSamplingResult,
) -> tuple[FullTilingPosteriorState, ...]:
    """Rebuild every retained trace state through the scientific oracle.

    Args:
        problem: Scientific problem for the completed segment.
        result: Completed HMC sampling segment.

    Returns:
        Rebuilt states in trace order.

    Raises:
        RuntimeError: If topology, coordinates, or retained log targets fail
            reconstruction.
    """
    states: list[FullTilingPosteriorState] = []
    for draw, bounds in enumerate(result.trace.rectangle_bounds):
        try:
            rectangles = tuple(Rectangle(*(int(value) for value in row)) for row in bounds.tolist())
            state = build_full_tiling_posterior_state(
                problem,
                allocation=TilingState(
                    LeafTiling(problem.shape, rectangles),
                    result.trace.leaf_masses[draw],
                ),
                fixed_coefficients=result.trace.fixed_coefficients[draw],
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Trace state {draw} failed scientific reconstruction.") from error
        target_difference = abs(float(state.log_target - result.trace.log_target[draw]))
        tolerance = max(
            _TRANSFORMED_TARGET_ATOL,
            128.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                abs(state.log_target),
                abs(float(result.trace.log_target[draw])),
            ),
        )
        if target_difference > tolerance:
            raise RuntimeError(
                f"Trace state {draw} log target differs from its scientific "
                f"rebuild by {target_difference:.17g}."
            )
        states.append(state)
    return tuple(states)


def _stack_state_field(
    states: tuple[FullTilingPosteriorState, ...],
    name: str,
) -> np.ndarray:
    """Stack one array-valued posterior-state field across retained draws."""
    return np.stack([np.asarray(getattr(state, name)) for state in states])


def _trace_to_dataset(
    result: FullTilingPyMCHMCSamplingResult,
    *,
    problem: FullTilingProblem,
    adapter: GammaBetaRHIMEAdapterResult,
    observation_labels: np.ndarray,
    outer_labels: np.ndarray,
    input_digest: str,
    manifest_digest: str,
) -> xr.Dataset:
    """Convert all state, target, and diagnostic fields to labelled xarray.

    Args:
        result: Completed full-tiling PyMC HMC segment.
        problem: Scientific problem used by the segment.
        adapter: Native-data adapter supplying grid coordinates.
        observation_labels: Stable labels in observation order.
        outer_labels: Fixed-coefficient labels in exact model order.
        input_digest: Frozen input whole-file SHA-256.
        manifest_digest: Canonical immutable-manifest SHA-256.

    Returns:
        Audited in-memory trace dataset ready for NetCDF serialization.
        Boundary variables use a ``draw`` dimension of ``sweeps + 1``;
        transition diagnostics use ``sweep`` of length ``sweeps``. The
        returned dataset owns no open file handle.

    Raises:
        RuntimeError: If any retained topology, scientific state, or log
            target cannot be reconstructed exactly enough for the audit.
        TypeError: If retained trace arrays cannot rebuild valid tilings or
            scientific states.
        ValueError: If labels, shapes, or coordinates are incompatible with
            the completed problem and adapter.

    Notes:
        Authoritative boundary-inclusive ``log_leaf_mass`` and
        ``log_fixed_coefficient`` arrays are copied from the sampler trace;
        they are not regenerated from decoded scientific values. Input and
        manifest digests are attached as immutable dataset identity metadata.
    """
    trace = result.trace
    states = _trace_states(problem, result)
    draws = len(states)
    sweeps = int(trace.global_sweep.size)
    log_fields = (
        "log_gaussian_likelihood",
        "log_likelihood",
        "log_root_prior",
        "log_allocation_prior",
        "log_fixed_coefficient_prior",
        "log_target",
    )
    data_vars: dict[str, Any] = {
        "state_sweep": (
            ("draw",),
            trace.state_sweep,
            {"long_name": "global completed compound-sweep coordinate"},
        ),
        "rectangle_bounds": (
            ("draw", "region", "bound"),
            trace.rectangle_bounds,
            {"long_name": "canonical half-open native-grid rectangle bounds"},
        ),
        "leaf_mass": (
            ("draw", "region"),
            trace.leaf_masses,
            {"long_name": "positive allocation in canonical leaf order"},
        ),
        "log_leaf_mass": (
            ("draw", "region"),
            trace.log_leaf_mass,
            {"long_name": "symmetric PyMC log leaf-mass coordinate"},
        ),
        "root_total": (
            ("draw",),
            np.asarray(
                [state.root_total for state in states],
                dtype=np.float64,
            ),
            {"long_name": "positive total allocation coordinate"},
        ),
        "fixed_coefficient": (
            ("draw", "fixed_parameter"),
            trace.fixed_coefficients,
            {"long_name": "positive always-active fixed coefficient"},
        ),
        "log_fixed_coefficient": (
            ("draw", "fixed_parameter"),
            trace.log_fixed_coefficient,
            {"long_name": "PyMC log fixed-coefficient coordinate"},
        ),
        "dynamic_prediction": (
            ("draw", "observation"),
            _stack_state_field(states, "dynamic_prediction"),
            {"long_name": "dynamic full-tiling prediction cache"},
        ),
        "fixed_prediction": (
            ("draw", "observation"),
            _stack_state_field(states, "fixed_prediction"),
            {"long_name": "fixed offset and outer prediction cache"},
        ),
        "prediction": (
            ("draw", "observation"),
            _stack_state_field(states, "prediction"),
            {"long_name": "complete observation-space prediction cache"},
        ),
        "residual": (
            ("draw", "observation"),
            _stack_state_field(states, "residual"),
            {"long_name": "prediction minus frozen observation"},
        ),
        "structural_move": (
            ("sweep",),
            trace.structural_move,
            {"long_name": "attempted structural proposal"},
        ),
        "structural_valid": (
            ("sweep",),
            trace.structural_valid,
            {"long_name": "structural proposal reached MH decision"},
        ),
        "structural_accepted": (
            ("sweep",),
            trace.structural_accepted,
            {"long_name": "structural proposal changed the state"},
        ),
        "structural_log_acceptance_ratio": (
            ("sweep",),
            trace.structural_log_acceptance_ratio,
            {"long_name": "log-boundary-corrected structural log acceptance ratio"},
        ),
        "structural_invalid_reason": (
            ("sweep",),
            trace.structural_invalid_reason,
            {"long_name": "empty for valid structural proposals"},
        ),
        "hmc_start_log_leaf_mass": (
            ("sweep", "region"),
            trace.hmc_start_log_leaf_mass,
            {"long_name": ("post-structure pre-HMC log leaf masses in post-HMC canonical order")},
        ),
        "hmc_start_log_fixed_coefficient": (
            ("sweep", "fixed_parameter"),
            trace.hmc_start_log_fixed_coefficient,
            {"long_name": "post-structure pre-HMC log fixed coefficients"},
        ),
        "hmc_accepted": (
            ("sweep",),
            trace.hmc_accepted,
            {"long_name": "PyMC accepted the HMC endpoint"},
        ),
        "hmc_acceptance_probability": (
            ("sweep",),
            trace.hmc_acceptance_probability,
            {"long_name": "HMC Metropolis acceptance probability"},
        ),
        "hmc_diverging": (
            ("sweep",),
            trace.hmc_diverging,
            {"long_name": "PyMC trajectory divergence flag"},
        ),
        "hmc_energy": (
            ("sweep",),
            trace.hmc_energy,
            {"long_name": "PyMC endpoint Hamiltonian"},
        ),
        "hmc_energy_error": (
            ("sweep",),
            trace.hmc_energy_error,
            {"long_name": "endpoint-minus-start Hamiltonian error"},
        ),
        "hmc_step_size": (
            ("sweep",),
            trace.hmc_step_size,
            {"long_name": "effective PyMC leapfrog step size"},
        ),
        "hmc_n_steps": (
            ("sweep",),
            trace.hmc_n_steps,
            {"long_name": "reported leapfrog step count"},
        ),
        "hmc_seed": (
            ("sweep",),
            trace.hmc_seed,
            {"long_name": "per-sweep uint64 PyMC reseed from master PCG64"},
        ),
    }
    for name in log_fields:
        data_vars[name] = (
            ("draw",),
            np.asarray(
                [getattr(state, name) for state in states],
                dtype=np.float64,
            ),
            {"long_name": name.replace("_", " ")},
        )
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "draw": np.arange(draws, dtype=np.int64),
            "sweep": trace.global_sweep,
            "region": np.arange(result.final_state.k, dtype=np.int64),
            "bound": np.asarray(
                ("row_start", "row_stop", "col_start", "col_stop"),
                dtype=np.str_,
            ),
            "fixed_parameter": outer_labels,
            "observation": observation_labels,
            "lat": adapter.latitudes,
            "lon": adapter.longitudes,
        },
        attrs={
            "schema": ("openghg_inversions.full_tiling_pymc_hmc_native_trace.v3"),
            "title": "Topology-conditioned PyMC HMC mobile full-tiling segment",
            "diagnostic_only": "true",
            "convergence_claim": "none",
            "fixed_k": result.final_state.k,
            "schedule_id": FULL_TILING_PYMC_HMC_SCHEDULE_ID,
            "coordinate_layout_id": (FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID),
            "metric_semantics_id": (FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID),
            "metric_builder_id": FULL_TILING_PYMC_HMC_METRIC_BUILDER_ID,
            "metric_reference_id": FULL_TILING_PYMC_HMC_METRIC_REFERENCE_ID,
            "metric_rebuild_policy": TOPOLOGY_PRECISION_REBUILD_POLICY,
            "topology_dependent_metric": "true",
            "communication_component": _COMMUNICATION_COMPONENT,
            "connectivity_proven": "false",
            "input_sha256": input_digest,
            "manifest_sha256": manifest_digest,
            "segment_sweeps": sweeps,
            "durable_checkpoint": "true",
        },
    )
    dataset["draw"].attrs["long_name"] = "segment-local retained-state index"
    dataset["sweep"].attrs["long_name"] = "global compound-sweep coordinate"
    return dataset


def _finite_summary(
    values: np.ndarray,
) -> dict[str, float | int | None]:
    """Summarize finite values while retaining non-finite counts."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
        "mean": None if not finite.size else float(np.mean(finite)),
        "minimum": None if not finite.size else float(np.min(finite)),
        "maximum": None if not finite.size else float(np.max(finite)),
    }


def _summary(
    result: FullTilingPyMCHMCSamplingResult,
    *,
    chain_initial_state: FullTilingPosteriorState,
    segment_initial_state: FullTilingPosteriorState,
    closure: dict[str, float],
    preflight: Mapping[str, float],
    input_path: Path,
    input_digest: str,
    parent_checkpoint_sha256: str | None,
    parent_completion_sha256: str | None,
    parent_artifact_sha256: Mapping[str, str] | None,
    input_seconds: float,
    problem_setup_seconds: float,
    preflight_seconds: float,
) -> dict[str, Any]:
    """Build a strict-JSON-compatible report for one completed segment.

    Args:
        result: Completed segment, including boundary trace, checkpoint, and
            kernel/transition timings.
        chain_initial_state: Reconstructed immutable start of the logical
            chain.
        segment_initial_state: Exact fresh or resumed state at this segment's
            boundary.
        closure: Scientific closure-audit metrics.
        preflight: Compiled transformed-target parity metrics for the segment
            boundary.
        input_path: Frozen native input path recorded for provenance.
        input_digest: Verified whole-file input SHA-256.
        parent_checkpoint_sha256: Certified parent checkpoint SHA-256 for a
            resumed segment, or ``None`` for a fresh segment.
        parent_completion_sha256: Verified parent ``complete.json`` SHA-256,
            or ``None`` for a fresh segment.
        parent_artifact_sha256: Complete certified parent artifact-hash map,
            or ``None`` for a fresh segment.
        input_seconds: Non-authoritative input hashing/loading duration.
        problem_setup_seconds: Non-authoritative scientific setup duration.
        preflight_seconds: Non-authoritative transformed-target compilation
            and parity-check duration.

    Returns:
        Nested summary containing lineage, target decomposition, structural
        and HMC diagnostics, and timing categories. Sweep and leapfrog
        throughput use transition execution time only.

    Notes:
        Timing values are reporting metadata and are excluded from immutable
        chain identity and exact-replay comparisons.
    """
    trace = result.trace
    invalid = Counter(
        str(reason) for reason in trace.structural_invalid_reason[~trace.structural_valid].tolist()
    )
    sweeps = int(trace.global_sweep.size)
    structural_valid = int(np.count_nonzero(trace.structural_valid))
    structural_accepted = int(np.count_nonzero(trace.structural_accepted))
    hmc_accepted = int(np.count_nonzero(trace.hmc_accepted))
    divergences = int(np.count_nonzero(trace.hmc_diverging))
    leapfrog_steps = int(np.sum(trace.hmc_n_steps))
    return {
        "schema": ("openghg_inversions.full_tiling_pymc_hmc_native_summary.v3"),
        "status": "experimental_topology_conditioned_hmc_not_convergence_evidence",
        "input": {
            "path": str(input_path.resolve()),
            "sha256": input_digest,
        },
        "closure": closure,
        "transformed_target_preflight": dict(preflight),
        "lineage": {
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "parent_completion_sha256": parent_completion_sha256,
            "parent_artifact_sha256": (
                None if parent_artifact_sha256 is None else dict(parent_artifact_sha256)
            ),
            "segment_start_state_sha256": full_tiling_state_fingerprint(
                segment_initial_state.problem,
                segment_initial_state,
            ),
        },
        "run": {
            "fixed_k": result.final_state.k,
            "schedule_id": result.checkpoint.schedule_id,
            "segment_sweeps": sweeps,
            "segment_start_sweep": int(trace.state_sweep[0]),
            "segment_end_sweep": result.checkpoint.sweeps_completed,
            "retained_states": int(trace.state_sweep.size),
            "durable_checkpoint": True,
            "topology_precision_sha256": result.checkpoint.topology_precision_sha256,
        },
        "target": {
            "chain_initial_log_target": float(chain_initial_state.log_target),
            "segment_initial_log_target": float(segment_initial_state.log_target),
            "final_log_gaussian_likelihood": float(result.final_state.log_gaussian_likelihood),
            "final_log_likelihood": float(result.final_state.log_likelihood),
            "final_log_root_prior": float(result.final_state.log_root_prior),
            "final_log_allocation_prior": float(result.final_state.log_allocation_prior),
            "final_log_fixed_coefficient_prior": float(result.final_state.log_fixed_coefficient_prior),
            "final_log_target": float(result.final_state.log_target),
            "normalization": ("fixed-K communication-component structural constant omitted"),
        },
        "structural": {
            "valid": structural_valid,
            "accepted": structural_accepted,
            "valid_rate": structural_valid / sweeps,
            "acceptance_rate": structural_accepted / sweeps,
            "acceptance_rate_given_valid": (
                None if structural_valid == 0 else structural_accepted / structural_valid
            ),
            "invalid_reasons": dict(sorted(invalid.items())),
        },
        "hmc": {
            "accepted": hmc_accepted,
            "acceptance_rate": hmc_accepted / sweeps,
            "divergences": divergences,
            "acceptance_probability": _finite_summary(trace.hmc_acceptance_probability),
            "energy": _finite_summary(trace.hmc_energy),
            "energy_error": _finite_summary(trace.hmc_energy_error),
            "effective_step_size": _finite_summary(trace.hmc_step_size),
            "reported_leapfrog_steps": {
                "minimum": int(np.min(trace.hmc_n_steps)),
                "maximum": int(np.max(trace.hmc_n_steps)),
            },
        },
        "performance": {
            "input_hash_and_load_seconds": input_seconds,
            "problem_setup_seconds": problem_setup_seconds,
            "transformed_target_preflight_compile_seconds": preflight_seconds,
            "kernel_setup_and_compile_seconds": result.kernel_setup_seconds,
            "transition_sampling_seconds": result.transition_seconds,
            "sweeps_per_second": (
                None if result.transition_seconds <= 0.0 else sweeps / result.transition_seconds
            ),
            "leapfrog_steps_per_second": (
                None if result.transition_seconds <= 0.0 else leapfrog_steps / result.transition_seconds
            ),
        },
    }


def _assert_checkpoint_equal(
    actual: FullTilingPyMCHMCCheckpoint,
    expected: FullTilingPyMCHMCCheckpoint,
) -> None:
    """Require exact equality of two durable continuation boundaries.

    Args:
        actual: Checkpoint reopened from the published no-pickle artifact.
        expected: In-memory checkpoint produced by the completed sampler.

    Raises:
        RuntimeError: If RNG state, sweep count, kernel/runtime identity,
            topology, scientific state arrays, target components, or
            authoritative log coordinates differ.

    Notes:
        Array comparisons are exact and do not use numerical tolerances,
        because the checkpoint is an exact replay boundary rather than a
        scientific equivalence artifact.
    """
    if (
        actual.rng_state != expected.rng_state
        or actual.sweeps_completed != expected.sweeps_completed
        or actual.kernel_settings != expected.kernel_settings
        or actual.runtime_identity != expected.runtime_identity
        or actual.topology_precision_sha256 != expected.topology_precision_sha256
        or actual.schedule_id != expected.schedule_id
        or actual.state.tiling_state.tiling != expected.state.tiling_state.tiling
    ):
        raise RuntimeError("Reopened checkpoint metadata or topology does not match.")
    array_fields = (
        "leaf_masses",
        "fixed_coefficients",
        "dynamic_prediction",
        "fixed_prediction",
        "prediction",
        "residual",
    )
    for name in array_fields:
        if not np.array_equal(
            getattr(actual.state, name),
            getattr(expected.state, name),
        ):
            raise RuntimeError(f"Reopened checkpoint state field {name} does not match.")
    for name in (
        "log_gaussian_likelihood",
        "log_likelihood",
        "log_root_prior",
        "log_allocation_prior",
        "log_fixed_coefficient_prior",
        "log_target",
    ):
        if getattr(actual.state, name) != getattr(expected.state, name):
            raise RuntimeError(f"Reopened checkpoint target field {name} does not match.")
    if not np.array_equal(
        actual.log_leaf_mass,
        expected.log_leaf_mass,
    ) or not np.array_equal(
        actual.log_fixed_coefficient,
        expected.log_fixed_coefficient,
    ):
        raise RuntimeError("Reopened checkpoint authoritative log coordinates do not match.")


def _audit_reopened_trace(
    expected: xr.Dataset,
    path: Path,
    *,
    engine: NetCDFEngine,
) -> None:
    """Reopen a trace and require exact content, coordinates, and identity.

    Args:
        expected: In-memory dataset passed to NetCDF serialization.
        path: Published trace artifact to reopen and hash.
        engine: Explicit xarray backend used for the reopen.

    Raises:
        OSError: If the trace cannot be hashed, opened, loaded, or closed.
        RuntimeError: If data-variable or coordinate sets, dimensions, values,
            identity attributes, or the file hash differ from expectations.

    Notes:
        Floating arrays compare exactly while treating NaNs in corresponding
        positions as equal. The whole-file SHA-256 is checked before and after
        the reopen audit to detect concurrent modification.
    """
    before = _sha256_file(path)
    with xr.open_dataset(path, engine=engine) as opened:
        actual = opened.load()
    try:
        if set(actual.data_vars) != set(expected.data_vars):
            raise RuntimeError("Reopened trace has an incompatible data-variable set.")
        if set(actual.coords) != set(expected.coords):
            raise RuntimeError("Reopened trace has an incompatible coordinate set.")
        if dict(actual.sizes) != dict(expected.sizes):
            raise RuntimeError("Reopened trace dimensions do not match.")
        for name in expected.data_vars:
            left = np.asarray(actual[name].values)
            right = np.asarray(expected[name].values)
            equal_nan = np.issubdtype(left.dtype, np.inexact) and np.issubdtype(
                right.dtype,
                np.inexact,
            )
            if not np.array_equal(left, right, equal_nan=equal_nan):
                raise RuntimeError(f"Reopened trace variable {name} does not match.")
        for name in expected.coords:
            if actual[name].dims != expected[name].dims:
                raise RuntimeError(f"Reopened trace coordinate {name} dimensions do not match.")
            left = np.asarray(actual[name].values)
            right = np.asarray(expected[name].values)
            equal_nan = np.issubdtype(left.dtype, np.inexact) and np.issubdtype(
                right.dtype,
                np.inexact,
            )
            if not np.array_equal(left, right, equal_nan=equal_nan):
                raise RuntimeError(f"Reopened trace coordinate {name} does not match.")
        for name in (
            "schema",
            "schedule_id",
            "coordinate_layout_id",
            "metric_semantics_id",
            "metric_builder_id",
            "metric_reference_id",
            "metric_rebuild_policy",
            "topology_dependent_metric",
            "input_sha256",
            "manifest_sha256",
        ):
            if actual.attrs.get(name) != expected.attrs.get(name):
                raise RuntimeError(f"Reopened trace attribute {name} does not match.")
    finally:
        actual.close()
    if _sha256_file(path) != before:
        raise RuntimeError("Trace changed while it was being reopened.")


def _certify_parent_segment_bundle(
    checkpoint_path: Path,
) -> tuple[str, dict[str, str]]:
    """Require a checkpoint to belong to one complete hash-certified bundle.

    Args:
        checkpoint_path: Requested parent checkpoint, which must be the
            canonical ``checkpoint.npz`` sibling named by ``complete.json``.

    Returns:
        Verified ``complete.json`` SHA-256 and exact certified hashes of the
        four parent segment artifacts.

    Raises:
        FileNotFoundError: If the completion certificate or any certified
            artifact is absent.
        OSError: If a parent artifact cannot be read.
        ValueError: If the completion certificate is malformed, incompatible,
            or does not certify the current bytes of every parent artifact.
    """
    if checkpoint_path.name != CHECKPOINT_FILENAME:
        raise ValueError(f"--resume-checkpoint must name the certified {CHECKPOINT_FILENAME} artifact.")
    completion_path = checkpoint_path.parent / COMPLETION_FILENAME
    if not completion_path.is_file():
        raise FileNotFoundError(f"Parent completion certificate is not a file: {completion_path}")
    completion_digest = _sha256_file(completion_path)
    completion = _load_strict_json_object(
        completion_path,
        description="Parent completion certificate",
    )
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise ValueError("Parent completion certificate schema is incompatible.")
    if completion.get("status") != "complete":
        raise ValueError("Parent completion certificate status is not complete.")
    if completion.get("durable_checkpoint") is not True:
        raise ValueError("Parent completion certificate is not durable.")
    if completion.get("schedule_id") != FULL_TILING_PYMC_HMC_SCHEDULE_ID:
        raise ValueError("Parent completion certificate schedule is incompatible.")
    names = {
        "manifest": MANIFEST_FILENAME,
        "trace": TRACE_FILENAME,
        "summary": SUMMARY_FILENAME,
        "checkpoint": CHECKPOINT_FILENAME,
    }
    for field, expected_name in names.items():
        if completion.get(field) != expected_name:
            raise ValueError(f"Parent completion certificate {field} name is incompatible.")
    certified = completion.get("sha256")
    if not isinstance(certified, dict) or set(certified) != set(_BUNDLE_ARTIFACTS):
        raise ValueError("Parent completion certificate artifact hashes are incomplete.")
    hashes: dict[str, str] = {}
    for name in _BUNDLE_ARTIFACTS:
        digest = certified.get(name)
        if not isinstance(digest, str):
            raise ValueError(f"Parent completion hash for {name} is not a string.")
        _validate_sha256(digest, name=f"parent {name} SHA-256")
        artifact = checkpoint_path.parent / name
        if not artifact.is_file():
            raise FileNotFoundError(f"Certified parent artifact is not a file: {artifact}")
        actual = _sha256_file(artifact)
        if actual.lower() != digest.lower():
            raise ValueError(f"Certified parent artifact SHA-256 does not match: {name}")
        hashes[name] = actual
    if _sha256_file(completion_path) != completion_digest:
        raise ValueError("Parent completion certificate changed while it was being validated.")
    return completion_digest, hashes


def _write_outputs(
    output_directory: Path,
    result: FullTilingPyMCHMCSamplingResult,
    *,
    problem: FullTilingProblem,
    manifest: dict[str, object],
    summary: dict[str, Any],
    trace_dataset: xr.Dataset,
    netcdf_engine: NetCDFEngine,
) -> None:
    """Publish one create-only bundle and write its certificate last.

    Args:
        output_directory: New directory for the segment bundle.
        result: Completed segment and continuation checkpoint.
        problem: Reconstructed scientific problem used for checkpoint reopen.
        manifest: Immutable chain identity used by durable checkpoint I/O.
        summary: Segment-local diagnostic report.
        trace_dataset: Audited full state/target/diagnostic trace.
        netcdf_engine: Selected output backend.

    Raises:
        FileExistsError: If the output directory or an artifact exists.
        OSError: If writing, syncing, or reopening fails.
        RuntimeError: If any reopened artifact differs.

    Notes:
        ``complete.json`` is written only after all four hashed artifacts pass
        their reopen audits.
    """
    output_directory.mkdir()
    _fsync_directory(output_directory.parent)
    manifest_path = output_directory / MANIFEST_FILENAME
    trace_path = output_directory / TRACE_FILENAME
    summary_path = output_directory / SUMMARY_FILENAME
    checkpoint_path = output_directory / CHECKPOINT_FILENAME
    _atomic_write_text(manifest_path, _canonical_json(manifest))
    _atomic_write_trace(
        trace_dataset,
        trace_path,
        engine=netcdf_engine,
    )
    _atomic_write_text(
        summary_path,
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )
    save_full_tiling_pymc_hmc_checkpoint(
        checkpoint_path,
        result.checkpoint,
        run_manifest=manifest,
    )

    with manifest_path.open(encoding="utf-8") as handle:
        reopened_manifest = json.load(handle)
    if reopened_manifest != manifest:
        raise RuntimeError("Reopened manifest does not match.")
    with summary_path.open(encoding="utf-8") as handle:
        reopened_summary = json.load(handle)
    if reopened_summary != summary:
        raise RuntimeError("Reopened summary does not match.")
    _audit_reopened_trace(
        trace_dataset,
        trace_path,
        engine=netcdf_engine,
    )
    checkpoint_hash = _sha256_file(checkpoint_path)
    reopened_checkpoint = load_full_tiling_pymc_hmc_checkpoint(
        checkpoint_path,
        problem,
        expected_run_manifest=manifest,
    )
    if _sha256_file(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed while it was being reopened.")
    _assert_checkpoint_equal(reopened_checkpoint, result.checkpoint)

    artifact_hashes = {
        name: _sha256_file(output_directory / name)
        for name in (
            MANIFEST_FILENAME,
            TRACE_FILENAME,
            SUMMARY_FILENAME,
            CHECKPOINT_FILENAME,
        )
    }
    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "manifest": MANIFEST_FILENAME,
        "trace": TRACE_FILENAME,
        "summary": SUMMARY_FILENAME,
        "checkpoint": CHECKPOINT_FILENAME,
        "schedule_id": FULL_TILING_PYMC_HMC_SCHEDULE_ID,
        "segment_sweeps": int(result.trace.global_sweep.size),
        "segment_start_sweep": int(result.trace.state_sweep[0]),
        "segment_end_sweep": result.checkpoint.sweeps_completed,
        "parent_checkpoint_sha256": summary["lineage"]["parent_checkpoint_sha256"],
        "parent_completion_sha256": summary["lineage"]["parent_completion_sha256"],
        "parent_artifact_sha256": summary["lineage"]["parent_artifact_sha256"],
        "durable_checkpoint": True,
        "sha256": artifact_hashes,
    }
    _atomic_write_text(
        output_directory / COMPLETION_FILENAME,
        _canonical_json(completion),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the restartable native full-tiling PyMC HMC CLI.

    Returns:
        Parser exposing all frozen scientific inputs, initialization identity,
        static HMC settings, calibration provenance, restart controls, and
        artifact backends.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Explicitly frozen native NetCDF snapshot.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="New artifact directory; existing paths are rejected.",
    )
    parser.add_argument(
        "--k",
        "--fixed-k",
        dest="k",
        type=int,
        required=True,
        help="Fixed active rectangle count.",
    )
    parser.add_argument(
        "--sweeps",
        type=int,
        required=True,
        help="Positive complete structural-then-HMC sweeps in this segment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Original non-negative master PCG64 seed for the logical chain.",
    )
    parser.add_argument(
        "--chain-id",
        required=True,
        help="Stable logical chain identifier.",
    )
    parser.add_argument(
        "--initialization",
        choices=("largest-nominal", "random-recursive"),
        default="largest-nominal",
        help="Fresh-chain fixed-K topology initializer.",
    )
    parser.add_argument(
        "--initialization-seed",
        type=int,
        help=("Required separate PCG64 seed for --initialization random-recursive."),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "Certified checkpoint.npz in a complete parent segment bundle "
            "whose complete.json hashes all four sibling artifacts."
        ),
    )
    parser.add_argument(
        "--step-size",
        "--hmc-step-size",
        dest="step_size",
        type=float,
        required=True,
        help="Requested unscaled static leapfrog step size.",
    )
    parser.add_argument(
        "--leapfrog-steps",
        type=int,
        required=True,
        help="Exact static leapfrog count per compound sweep.",
    )
    parser.add_argument(
        "--calibration-id",
        required=True,
        help="Stable identifier exactly repeated by the v3 calibration file.",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        required=True,
        help=(
            "Strict-JSON v3 calibration identity binding K, target and HMC "
            "controls, topology-conditioned metric identities, five ordered "
            "validation trajectories, and source/evidence hashes."
        ),
    )
    parser.add_argument(
        "--calibration-sha256",
        required=True,
        help="Required whole-file SHA-256 of --calibration-file.",
    )
    parser.add_argument(
        "--concentration",
        type=float,
        required=True,
        help="Positive additive-alpha Dirichlet concentration.",
    )
    parser.add_argument(
        "--root-variance",
        type=float,
        required=True,
        help="Positive Gamma root-total prior variance.",
    )
    parser.add_argument(
        "--likelihood-power",
        type=float,
        default=1.0,
        help="Non-negative Gaussian likelihood multiplier.",
    )
    parser.add_argument(
        "--fixed-prior-mean",
        type=_positive_values,
        default=1.0,
        metavar="VALUE[,VALUE...]",
        help="Arithmetic lognormal prior mean(s) for fixed coefficients.",
    )
    parser.add_argument(
        "--fixed-prior-sd",
        type=_positive_values,
        required=True,
        metavar="VALUE[,VALUE...]",
        help="Arithmetic lognormal prior SD(s) for fixed coefficients.",
    )
    parser.add_argument(
        "--input-id",
        required=True,
        help="Stable logical frozen-input identifier.",
    )
    parser.add_argument(
        "--expected-input-sha256",
        required=True,
        help="Required whole-file frozen-input SHA-256.",
    )
    parser.add_argument(
        "--code-revision",
        required=True,
        help="Clean source revision used for this chain.",
    )
    parser.add_argument(
        "--nominal-weight-policy",
        required=True,
        help="Reviewed strictly positive nominal-base-measure identifier.",
    )
    parser.add_argument(
        "--normalize-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize nominal weights to sum to one (default: true).",
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
        help=(
            "Require the reviewed 1382 by 183x128 PARIS shape, six outer "
            "coefficients, exact input hash, and ordered outer labels."
        ),
    )
    parser.add_argument(
        "--expected-outer-labels",
        type=_comma_separated_labels,
        help="Expected comma-separated outer_region labels in exact order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate frozen input, closure, float64 runtime, manifest, and "
            "compiled transformed target without sampling or publishing."
        ),
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
    """Validate CLI values and create-only paths before scientific I/O.

    Args:
        arguments: Namespace produced by :func:`build_parser`.

    Raises:
        FileExistsError: If the requested output directory already exists.
        FileNotFoundError: If the output parent, calibration file, or requested
            resume checkpoint is absent.
        ValueError: If output policy, seeds, initialization combinations,
            static-HMC controls, scientific scalars, required identifiers,
            SHA-256 syntax, or PARIS-profile arguments are invalid.

    Notes:
        Validation does not create the output directory or read scientific
        dataset contents. Create-only publication remains authoritative in
        :func:`_write_outputs`.
    """
    if arguments.output_directory.exists():
        raise FileExistsError(f"Output path already exists: {arguments.output_directory}")
    if not arguments.output_directory.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory does not exist: {arguments.output_directory.parent}"
        )
    resolved_output = arguments.output_directory.resolve(strict=False)
    if any("paris_inversions" in part.lower() for part in resolved_output.parts):
        raise ValueError("Output and NetCDF preflight writes beneath PARIS_inversions are forbidden.")
    if arguments.k < 2:
        raise ValueError("--k must be at least two for the structural kernel.")
    if arguments.sweeps < 1:
        raise ValueError("--sweeps must be positive.")
    if arguments.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if not arguments.chain_id:
        raise ValueError("--chain-id must be nonempty.")
    if arguments.initialization == "random-recursive":
        if arguments.initialization_seed is None:
            raise ValueError("--initialization random-recursive requires --initialization-seed.")
        if arguments.initialization_seed < 0:
            raise ValueError("--initialization-seed must be non-negative.")
        if arguments.initialization_seed == arguments.seed:
            raise ValueError("--initialization-seed must differ from the master --seed.")
        calibration_topology_seeds = (
            *CALIBRATION_DEVELOPMENT_TOPOLOGY_SEEDS.get(arguments.k, ()),
            *CALIBRATION_HELD_OUT_TOPOLOGY_SEEDS.get(arguments.k, ()),
        )
        if arguments.initialization_seed in calibration_topology_seeds:
            raise ValueError("--initialization-seed must be disjoint from H2d calibration topology seeds.")
    elif arguments.initialization_seed is not None:
        raise ValueError("--initialization-seed is only valid with --initialization random-recursive.")
    if arguments.dry_run and arguments.resume_checkpoint is not None:
        raise ValueError("--dry-run cannot be combined with --resume-checkpoint.")
    for name in ("step_size", "concentration", "root_variance"):
        value = float(getattr(arguments, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
    if arguments.leapfrog_steps < 1:
        raise ValueError("--leapfrog-steps must be positive.")
    if not np.isfinite(arguments.likelihood_power) or arguments.likelihood_power < 0.0:
        raise ValueError("--likelihood-power must be finite and non-negative.")
    if not arguments.input_id:
        raise ValueError("--input-id must be nonempty.")
    if not arguments.code_revision:
        raise ValueError("--code-revision must be nonempty.")
    if not arguments.calibration_id:
        raise ValueError("--calibration-id must be nonempty.")
    _validate_sha256(
        arguments.expected_input_sha256,
        name="--expected-input-sha256",
    )
    _validate_sha256(
        arguments.calibration_sha256,
        name="--calibration-sha256",
    )
    if not arguments.calibration_file.is_file():
        raise FileNotFoundError(f"Calibration file is not a file: {arguments.calibration_file}")
    if arguments.require_paris_profile and arguments.expected_outer_labels is None:
        raise ValueError("--require-paris-profile requires --expected-outer-labels.")
    if arguments.resume_checkpoint is not None and not arguments.resume_checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint is not a file: {arguments.resume_checkpoint}")


def _requested_kernel_settings(
    arguments: argparse.Namespace,
) -> FullTilingPyMCHMCKernelSettings:
    """Return the exact resolved v3 CLI kernel settings.

    Args:
        arguments: Validated namespace containing fixed ``K``, trajectory
            controls, and immutable metric builder/reference identities.

    Returns:
        Complete immutable topology-conditioned HMC kernel settings.
    """
    return FullTilingPyMCHMCKernelSettings(
        fixed_k=arguments.k,
        step_size=arguments.step_size,
        leapfrog_steps=arguments.leapfrog_steps,
    )


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    """Validate, run, and publish one fresh or resumed complete-sweep segment.

    A fresh initializer is canonicalized before closure, transformed-target
    preflight, manifest construction, and sampling. A resume instead uses the
    exact scientific/log-coordinate boundary stored in the checkpoint.

    Args:
        arguments: Namespace returned by :func:`build_parser`.

    Returns:
        Strict-JSON-compatible dry-run or completed-segment summary.

    Raises:
        FileExistsError: If a create-only output target already exists.
        FileNotFoundError: If an input, checkpoint, or output parent is absent.
        ImportError: If the required PyMC/PyTensor runtime is unavailable.
        OSError: If input, preflight, checkpoint, or artifact I/O fails.
        ValueError: If scientific inputs, hashes, settings, initialization,
            backend identity, closure, target, or checkpoint are incompatible.
        RuntimeError: If sampling or artifact reopen validation fails.
    """
    _validate_arguments(arguments)
    _preflight_output_backend(
        arguments.output_directory.parent,
        engine=arguments.netcdf_engine,
    )
    input_started = perf_counter()
    input_digest = _sha256_file(arguments.input)
    if input_digest.lower() != arguments.expected_input_sha256.lower():
        raise ValueError("Frozen input SHA-256 does not match --expected-input-sha256.")
    dataset = _load_frozen_dataset(
        arguments.input,
        engine=arguments.input_netcdf_engine,
    )
    try:
        if _sha256_file(arguments.input) != input_digest:
            raise ValueError("Frozen input changed while it was being loaded.")
        input_seconds = perf_counter() - input_started
        setup_started = perf_counter()
        adapter = _build_adapter(dataset, arguments)
        if arguments.require_paris_profile:
            _require_paris_profile(
                dataset,
                adapter,
                fixed_design_name=arguments.fixed_design_name,
                expected_outer_labels=cast(
                    tuple[str, ...],
                    arguments.expected_outer_labels,
                ),
            )
        problem = full_tiling_problem_from_gamma_beta_adapter(
            adapter,
            concentration=arguments.concentration,
        )
        if arguments.initialization == "largest-nominal":
            initial_state = initialize_full_tiling_posterior_state(
                problem,
                k=arguments.k,
            )
        else:
            initial_state = initialize_random_full_tiling_posterior_state(
                problem,
                k=arguments.k,
                seed=arguments.initialization_seed,
            )
        (
            initial_state,
            initial_log_leaf_mass,
            initial_log_fixed_coefficient,
        ) = full_tiling_pymc_hmc_kernel.canonicalize_full_tiling_pymc_hmc_fresh_state(
            problem,
            initial_state,
        )
        if not np.isfinite(initial_state.log_target):
            raise ValueError("Initial full-tiling log target is not finite.")
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
        observation_labels = _dimension_labels(dataset, "nmeasure")
        requested_settings = _requested_kernel_settings(arguments)
        calibration, calibration_digest = _load_verified_calibration(
            arguments,
            adapter,
            input_digest=input_digest,
        )
        initial_topology_sha256 = _topology_sha256(_rectangle_bounds(initial_state))
        excluded_topologies = calibration["evidence"]["excluded_production_topology_sha256"]
        if initial_topology_sha256 in excluded_topologies.values():
            raise ValueError(
                "Initial topology hash was used by H2d calibration and is excluded from retained production."
            )
        problem_setup_seconds = perf_counter() - setup_started
        preflight_started = perf_counter()
        initial_preflight = _transformed_target_preflight(
            problem,
            initial_state,
            log_leaf_mass=initial_log_leaf_mass,
            log_fixed_coefficient=initial_log_fixed_coefficient,
        )
        preflight_seconds = perf_counter() - preflight_started
        manifest = _build_manifest(
            arguments,
            adapter,
            initial_state=initial_state,
            input_digest=input_digest,
            outer_labels=outer_labels,
            calibration=calibration,
            calibration_digest=calibration_digest,
        )
        if arguments.dry_run:
            return {
                "schema": ("openghg_inversions.full_tiling_pymc_hmc_native_summary.v3"),
                "status": "dry_run",
                "input": {
                    "id": arguments.input_id,
                    "path": str(arguments.input.resolve()),
                    "sha256": input_digest,
                },
                "fixed_k": arguments.k,
                "requested_segment_sweeps": arguments.sweeps,
                "closure": closure,
                "transformed_target_preflight": initial_preflight,
                "runtime_identity": asdict(full_tiling_pymc_hmc_runtime_identity()),
                "manifest": manifest,
                "performance": {
                    "input_hash_and_load_seconds": input_seconds,
                    "problem_setup_seconds": problem_setup_seconds,
                    "transformed_target_preflight_compile_seconds": (preflight_seconds),
                },
            }

        parent_checkpoint_sha256: str | None
        parent_completion_sha256: str | None
        parent_checkpoint_path: Path | None
        parent_artifact_hashes: dict[str, str] | None
        parent_certification: tuple[str, dict[str, str]] | None
        if arguments.resume_checkpoint is None:
            parent_checkpoint_sha256 = None
            parent_completion_sha256 = None
            parent_checkpoint_path = None
            parent_artifact_hashes = None
            parent_certification = None
            segment_initial_state = initial_state
            preflight = initial_preflight
            result = sample_full_tiling_pymc_hmc(
                problem,
                initial_state,
                FullTilingPyMCHMCConfig(
                    iterations=arguments.sweeps,
                    step_size=arguments.step_size,
                    leapfrog_steps=arguments.leapfrog_steps,
                    seed=arguments.seed,
                ),
            )
        else:
            parent_checkpoint_path = cast(Path, arguments.resume_checkpoint)
            parent_certification = _certify_parent_segment_bundle(parent_checkpoint_path)
            parent_completion_sha256, parent_artifact_hashes = parent_certification
            parent_checkpoint_sha256 = parent_artifact_hashes[CHECKPOINT_FILENAME]
            checkpoint = load_full_tiling_pymc_hmc_checkpoint(
                parent_checkpoint_path,
                problem,
                expected_run_manifest=manifest,
            )
            if _certify_parent_segment_bundle(parent_checkpoint_path) != parent_certification:
                raise ValueError("Parent segment bundle changed while it was being loaded.")
            if checkpoint.kernel_settings != requested_settings:
                raise ValueError("Resume checkpoint HMC settings do not match the CLI.")
            segment_initial_state = checkpoint.state
            resumed_preflight_started = perf_counter()
            preflight = _transformed_target_preflight(
                problem,
                checkpoint.state,
                log_leaf_mass=checkpoint.log_leaf_mass,
                log_fixed_coefficient=(checkpoint.log_fixed_coefficient),
            )
            preflight_seconds += perf_counter() - resumed_preflight_started
            result = continue_full_tiling_pymc_hmc(
                problem,
                checkpoint,
                iterations=arguments.sweeps,
            )
        if (
            result.trace.global_sweep.size != arguments.sweeps
            or result.trace.state_sweep.size != arguments.sweeps + 1
            or result.checkpoint.sweeps_completed != int(result.trace.state_sweep[-1])
        ):
            raise RuntimeError("Sampler did not finish the requested complete-sweep segment.")
        if (
            result.checkpoint.kernel_settings != requested_settings
            or result.checkpoint.schedule_id != FULL_TILING_PYMC_HMC_SCHEDULE_ID
            or result.checkpoint.runtime_identity != full_tiling_pymc_hmc_runtime_identity()
        ):
            raise RuntimeError("Sampler checkpoint identity differs from requested settings.")
        if np.any(result.trace.hmc_n_steps != arguments.leapfrog_steps):
            raise RuntimeError("Trace HMC leapfrog counts differ from the frozen setting.")
        summary = _summary(
            result,
            chain_initial_state=initial_state,
            segment_initial_state=segment_initial_state,
            closure=closure,
            preflight=preflight,
            input_path=arguments.input,
            input_digest=input_digest,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            parent_completion_sha256=parent_completion_sha256,
            parent_artifact_sha256=parent_artifact_hashes,
            input_seconds=input_seconds,
            problem_setup_seconds=problem_setup_seconds,
            preflight_seconds=preflight_seconds,
        )
        manifest_text = _canonical_json(manifest)
        trace_dataset = _trace_to_dataset(
            result,
            problem=problem,
            adapter=adapter,
            observation_labels=observation_labels,
            outer_labels=outer_labels,
            input_digest=input_digest,
            manifest_digest=sha256(manifest_text.encode("utf-8")).hexdigest(),
        )
        try:
            if (
                parent_checkpoint_path is not None
                and parent_certification is not None
                and _certify_parent_segment_bundle(parent_checkpoint_path) != parent_certification
            ):
                raise ValueError("Parent segment bundle changed before child publication.")
            _write_outputs(
                arguments.output_directory,
                result,
                problem=problem,
                manifest=manifest,
                summary=summary,
                trace_dataset=trace_dataset,
                netcdf_engine=arguments.netcdf_engine,
            )
        finally:
            trace_dataset.close()
        return summary
    finally:
        dataset.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run one native HMC segment and print its machine-readable summary.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Zero after a successful dry run or completed artifact workflow.

    Raises:
        SystemExit: If command-line parsing fails.
        Exception: Propagates validation, sampling, and publication failures
            so batch jobs fail closed.

    Notes:
        On success, one indented strict-JSON summary is written to standard
        output. Non-dry runs publish their create-only artifact bundle before
        the summary is printed.
    """
    arguments = build_parser().parse_args(argv)
    summary = run(arguments)
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
