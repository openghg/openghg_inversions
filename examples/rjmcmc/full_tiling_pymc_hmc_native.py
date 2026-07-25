"""Run or resume one static PyMC HMC mobile full-tiling data segment.

The driver consumes one explicitly frozen NetCDF snapshot, constructs the
native Gamma--Dirichlet full-tiling problem without guessing scientific
variables, and runs exactly one complete-sweep segment of the experimental
structural-then-static-HMC kernel. Fresh chains may use the deterministic
largest-nominal initializer or a separately seeded random-recursive topology.
Fresh states are canonicalized in the HMC log chart before preflight,
fingerprinting, and sampling. Resumed chains load the dedicated checksummed
no-pickle checkpoint, preserve its authoritative coordinates unchanged, and
retain the original immutable run manifest.

PyTensor must use float64. Before a dry run, fresh segment, or resumed segment,
the compiled PyMC density is checked against the independently assembled
scientific target plus the symmetric log-coordinate Jacobians. Static HMC
settings are calibration outputs: no adaptation, online tuning, randomized
step size, or topology-dependent metric is enabled here.
The larger leaf-metric eigenvalue may be at most 10,000 times the smaller one.

``--calibration-file`` is a strict-JSON v2 calibration identity with exactly
this shape (values shown symbolically):

Calibration v1 used the retired scalar diagonal leaf metric and is rejected;
no automatic converter is provided.

.. code-block:: json

   {
     "schema": "openghg_inversions.full_tiling_pymc_hmc_calibration.v2",
     "calibration_id": "<--calibration-id>",
     "fixed_k": "<--k>",
     "input_sha256": "<--expected-input-sha256>",
     "target": {
       "concentration": "<--concentration>",
       "root_variance": "<--root-variance>",
       "likelihood_power": "<--likelihood-power>",
       "fixed_prior_mean": ["<resolved fixed prior means>"],
       "fixed_prior_sd": ["<resolved fixed prior SDs>"],
       "nominal_weight_policy": "<--nominal-weight-policy>",
       "normalize_weights": "<--normalize-weights>"
     },
     "kernel": {
       "step_size": "<--step-size>",
       "leapfrog_steps": "<--leapfrog-steps>",
       "coordinate_layout_id": "<driver coordinate layout ID>",
       "metric_semantics_id": "<driver metric semantics ID>",
       "leaf_contrast_position_scale": "<--leaf-contrast-position-scale>",
       "leaf_total_position_scale": "<--leaf-total-position-scale>",
       "fixed_coefficient_position_scale": ["<resolved fixed scales>"]
     },
     "evidence": {
       "code_revision": "<--code-revision>",
       "robust_variance_estimator": "squared_scaled_median_absolute_deviation_1.4826",
       "leaf_metric_estimator": "normalized_common_and_centered_contrast_scaled_mad_v1",
       "clipping_bounds": [0.0001, 100.0],
       "development_initializers": [
         {
           "role": "development-a",
           "strategy": "random-recursive",
           "seed": "<development topology seed A>",
           "sampler_seed": "<development sampler seed A>",
           "sweeps": 200
         },
         {
           "role": "development-b",
           "strategy": "random-recursive",
           "seed": "<development topology seed B>",
           "sampler_seed": "<development sampler seed B>",
           "sweeps": 200
         }
       ],
       "candidate_grid": [
         {"step_size": "<candidate>", "leapfrog_steps": "<candidate>"}
       ],
       "decision_statistics": [
         {
           "step_size": "<candidate>",
           "leapfrog_steps": "<candidate>",
           "development_a_mean_acceptance": "<finite value>",
           "development_b_mean_acceptance": "<finite value>",
           "divergences": "<non-negative integer>",
           "finite": "<Boolean>",
           "median_hmc_log_displacement_per_leapfrog_step": "<finite value>",
           "selected": "<Boolean>"
         }
       ],
       "selected_validation": {
         "step_size": "<selected candidate>",
         "leapfrog_steps": "<selected candidate>",
         "initializers": [
           {
             "role": "development-a",
             "strategy": "random-recursive",
             "seed": "<development topology seed A>",
             "sampler_seed": "<validation sampler seed A>",
             "sweeps": 500,
             "mean_acceptance": "<finite value>",
             "divergences": 0,
             "finite": true
           },
           {
             "role": "development-b",
             "strategy": "random-recursive",
             "seed": "<development topology seed B>",
             "sampler_seed": "<validation sampler seed B>",
             "sweeps": 500,
             "mean_acceptance": "<finite value>",
             "divergences": 0,
             "finite": true
           },
           {
             "role": "held-out",
             "strategy": "random-recursive",
             "seed": "<held-out topology seed>",
             "sampler_seed": "<held-out validation sampler seed>",
             "sweeps": 500,
             "mean_acceptance": "<finite value>",
             "divergences": 0,
             "finite": true
           }
         ]
       },
       "excluded_production_topology_sha256": {
         "metric_source": "<SHA-256>",
         "development_a": "<SHA-256>",
         "development_b": "<SHA-256>",
         "held_out": "<SHA-256>"
       },
       "source_artifact_sha256": {"<artifact ID>": "<SHA-256>"}
     }
   }

The file must contain no additional keys, duplicate object keys, or non-finite
JSON constants. Its bytes must match ``--calibration-sha256`` and every
identity value must match the current invocation. Candidate rows and decision
rows must correspond one-to-one; exactly one finite, zero-divergence decision
must be selected, must match the requested kernel, and must have both
development acceptance means in the inclusive interval 0.6--0.9. The selected
candidate must also have three ordered 500-sweep validation initializers with
finite acceptance in that interval and zero divergences. Production ``K=50``
and ``K=250`` evidence additionally requires reviewed, role-specific topology
and master PCG64 seeds. Candidate development runs reuse the same master seed
for a given topology across every candidate (common random numbers);
validation uses distinct streams, and all calibration topology seeds are
disjoint from retained-production starts. The driver also rejects exact
collisions between a proposed production topology hash and the four
calibration topology hashes, including the fixed-basis NUTS metric source.

Successful runs publish ``manifest.json``, ``trace.nc``, ``summary.json``, and
``checkpoint.npz`` inside a new output directory. Every artifact is reopened
and audited before ``complete.json`` is hash-certified and written last.
Interrupted directories therefore cannot masquerade as complete results.
``--dry-run`` performs input, closure, backend, manifest, and transformed
target checks without creating the output directory. The driver refuses every
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
    FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID,
    FULL_TILING_PYMC_HMC_SCHEDULE_ID,
    FullTilingPyMCHMCCheckpoint,
    FullTilingPyMCHMCConfig,
    FullTilingPyMCHMCKernelSettings,
    FullTilingPyMCHMCSamplingResult,
    build_full_tiling_pymc_hmc_model,
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
CALIBRATION_SCHEMA = "openghg_inversions.full_tiling_pymc_hmc_calibration.v2"
LEGACY_CALIBRATION_SCHEMA = "openghg_inversions.full_tiling_pymc_hmc_calibration.v1"
COMPLETION_SCHEMA = "openghg_inversions.full_tiling_pymc_hmc_native_completion.v2"
ROBUST_VARIANCE_ESTIMATOR = "squared_scaled_median_absolute_deviation_1.4826"
LEAF_METRIC_ESTIMATOR = "normalized_common_and_centered_contrast_scaled_mad_v1"
MAX_LEAF_METRIC_CONDITION_RATIO = 1.0e4
POSITION_SCALE_CLIPPING_BOUNDS = (1.0e-4, 1.0e2)
CALIBRATION_DEVELOPMENT_TOPOLOGY_SEEDS = {
    50: (41050, 41051),
    250: (41250, 41251),
}
CALIBRATION_HELD_OUT_TOPOLOGY_SEEDS = {
    50: 41052,
    250: 41252,
}
CALIBRATION_DEVELOPMENT_SAMPLER_SEEDS = {
    50: (71050, 71051),
    250: (71250, 71251),
}
CALIBRATION_VALIDATION_SAMPLER_SEEDS = {
    50: (72050, 72051, 72052),
    250: (72250, 72251, 72252),
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
    fixed_position_scales: tuple[float, ...],
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
        fixed_position_scales: Fully resolved fixed-coordinate position scale.
        calibration: Verified v2 calibration identity.
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
    runtime_identity = asdict(full_tiling_pymc_hmc_runtime_identity())
    manifest: dict[str, object] = {
        "schema": ("openghg_inversions.full_tiling_pymc_hmc_native_manifest.v2"),
        "status": "experimental_mobile_hmc_not_convergence_evidence",
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
            "state_sha256": full_tiling_state_fingerprint(
                initial_state.problem,
                initial_state,
            ),
        },
        "sampler": {
            "name": "full_tiling_structure_then_static_pymc_hmc",
            "schedule_id": FULL_TILING_PYMC_HMC_SCHEDULE_ID,
            "chains_per_invocation": 1,
            "step_size_requested": float(arguments.step_size),
            "leapfrog_steps": int(arguments.leapfrog_steps),
            "leaf_contrast_position_scale": float(arguments.leaf_contrast_position_scale),
            "leaf_total_position_scale": float(arguments.leaf_total_position_scale),
            "fixed_coefficient_position_scale": list(fixed_position_scales),
            "metric_semantics_id": (FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID),
            "coordinate_layout_id": (FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID),
            "adapt_step_size": False,
            "step_size_randomization": False,
            "topology_dependent_metric": False,
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
    fixed_position_scales: tuple[float, ...],
) -> dict[str, Any]:
    """Build the exact minimal v2 calibration identity for this invocation.

    Args:
        arguments: Validated driver arguments.
        adapter: Frozen scientific adapter with resolved fixed priors.
        input_digest: Verified frozen-input whole-file SHA-256.
        fixed_position_scales: Resolved fixed-coordinate position scales.

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
            "leaf_contrast_position_scale": float(arguments.leaf_contrast_position_scale),
            "leaf_total_position_scale": float(arguments.leaf_total_position_scale),
            "fixed_coefficient_position_scale": list(fixed_position_scales),
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


def _calibration_candidate(value: object, *, name: str) -> tuple[float, int]:
    """Validate and return one declared static-HMC candidate."""
    candidate = _require_exact_keys(
        value,
        {"step_size", "leapfrog_steps"},
        name=name,
    )
    step_size = _finite_json_number(candidate["step_size"], name=f"{name}.step_size")
    leapfrog_steps = candidate["leapfrog_steps"]
    if step_size <= 0.0:
        raise ValueError(f"{name}.step_size must be positive.")
    if isinstance(leapfrog_steps, bool) or not isinstance(leapfrog_steps, int):
        raise ValueError(f"{name}.leapfrog_steps must be a positive integer.")
    if leapfrog_steps < 1:
        raise ValueError(f"{name}.leapfrog_steps must be a positive integer.")
    return step_size, leapfrog_steps


def _validate_calibration_evidence(
    value: object,
    arguments: argparse.Namespace,
) -> None:
    """Validate calibration provenance and its selected static-HMC candidate.

    Args:
        value: Parsed strict-JSON calibration ``evidence`` value.
        arguments: Validated CLI namespace supplying the current code
            revision, fixed ``K``, step size, and leapfrog count.

    Raises:
        ValueError: If evidence keys, code revision, robust estimator, clipping
            bounds, development identities, candidate grid, decision
            statistics, selected candidate validation, acceptance gates, or
            source SHA-256 values violate the v2 contract.

    Notes:
        Candidate and decision rows must correspond one-to-one. Exactly one
        finite, zero-divergence candidate must be selected, and production
        ``K=50``/``K=250`` evidence is bound to the reviewed random-recursive
        development and held-out seeds.
    """
    evidence = _require_exact_keys(
        value,
        {
            "code_revision",
            "robust_variance_estimator",
            "leaf_metric_estimator",
            "clipping_bounds",
            "development_initializers",
            "candidate_grid",
            "decision_statistics",
            "selected_validation",
            "excluded_production_topology_sha256",
            "source_artifact_sha256",
        },
        name="Calibration evidence",
    )
    if evidence["code_revision"] != arguments.code_revision:
        raise ValueError("Calibration evidence code_revision does not match --code-revision.")
    if evidence["robust_variance_estimator"] != ROBUST_VARIANCE_ESTIMATOR:
        raise ValueError("Calibration evidence robust variance estimator is incompatible.")
    if evidence["leaf_metric_estimator"] != LEAF_METRIC_ESTIMATOR:
        raise ValueError("Calibration evidence leaf metric estimator is incompatible.")
    if _canonical_json(evidence["clipping_bounds"]) != _canonical_json(list(POSITION_SCALE_CLIPPING_BOUNDS)):
        raise ValueError("Calibration evidence clipping_bounds must be [0.0001, 100.0].")

    development_initializers = evidence["development_initializers"]
    if not isinstance(development_initializers, list) or len(development_initializers) != 2:
        raise ValueError("Calibration evidence must declare exactly two development initializers.")
    development_topology_seeds: list[int] = []
    development_sampler_seeds: list[int] = []
    for index in range(2):
        initializer = _require_exact_keys(
            development_initializers[index],
            {"role", "strategy", "seed", "sampler_seed", "sweeps"},
            name=f"Calibration development initializer {index}",
        )
        sweeps = initializer["sweeps"]
        if (
            initializer["role"] != f"development-{chr(ord('a') + index)}"
            or initializer["strategy"] != "random-recursive"
            or isinstance(sweeps, bool)
            or not isinstance(sweeps, int)
            or sweeps != 200
        ):
            raise ValueError(
                "Calibration development initializer strategies and sweep counts are incompatible."
            )
        seed = initializer["seed"]
        sampler_seed = initializer["sampler_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Calibration development topology seeds must be non-negative.")
        if isinstance(sampler_seed, bool) or not isinstance(sampler_seed, int) or sampler_seed < 0:
            raise ValueError("Calibration development sampler seeds must be non-negative.")
        development_topology_seeds.append(seed)
        development_sampler_seeds.append(sampler_seed)
    if len(set(development_topology_seeds)) != 2 or len(set(development_sampler_seeds)) != 2:
        raise ValueError("Calibration development topology and sampler seeds must be distinct.")
    required_development_topology_seeds = CALIBRATION_DEVELOPMENT_TOPOLOGY_SEEDS.get(arguments.k)
    if (
        required_development_topology_seeds is not None
        and tuple(development_topology_seeds) != required_development_topology_seeds
    ):
        raise ValueError(
            f"Calibration evidence for K={arguments.k} has incompatible development topology seeds."
        )
    required_development_sampler_seeds = CALIBRATION_DEVELOPMENT_SAMPLER_SEEDS.get(arguments.k)
    if (
        required_development_sampler_seeds is not None
        and tuple(development_sampler_seeds) != required_development_sampler_seeds
    ):
        raise ValueError(
            f"Calibration evidence for K={arguments.k} has incompatible development sampler seeds."
        )

    candidate_grid = evidence["candidate_grid"]
    if not isinstance(candidate_grid, list) or not candidate_grid:
        raise ValueError("Calibration candidate_grid must be a nonempty array.")
    candidates = [
        _calibration_candidate(item, name=f"Calibration candidate {index}")
        for index, item in enumerate(candidate_grid)
    ]
    if len(set(candidates)) != len(candidates):
        raise ValueError("Calibration candidate_grid entries must be unique.")

    decisions = evidence["decision_statistics"]
    if not isinstance(decisions, list) or len(decisions) != len(candidates):
        raise ValueError("Calibration decision_statistics must match candidate_grid length.")
    decision_candidates: list[tuple[float, int]] = []
    selected: list[tuple[tuple[float, int], dict[str, Any]]] = []
    decision_keys = {
        "step_size",
        "leapfrog_steps",
        "development_a_mean_acceptance",
        "development_b_mean_acceptance",
        "divergences",
        "finite",
        "median_hmc_log_displacement_per_leapfrog_step",
        "selected",
    }
    for index, item in enumerate(decisions):
        decision = _require_exact_keys(
            item,
            decision_keys,
            name=f"Calibration decision {index}",
        )
        candidate = _calibration_candidate(
            {
                "step_size": decision["step_size"],
                "leapfrog_steps": decision["leapfrog_steps"],
            },
            name=f"Calibration decision {index}",
        )
        decision_candidates.append(candidate)
        for field in (
            "development_a_mean_acceptance",
            "development_b_mean_acceptance",
            "median_hmc_log_displacement_per_leapfrog_step",
        ):
            _finite_json_number(decision[field], name=f"Calibration decision {index}.{field}")
        divergences = decision["divergences"]
        if isinstance(divergences, bool) or not isinstance(divergences, int) or divergences < 0:
            raise ValueError("Calibration decision divergences must be non-negative integers.")
        if not isinstance(decision["finite"], bool) or not isinstance(decision["selected"], bool):
            raise ValueError("Calibration decision finite and selected fields must be Booleans.")
        if decision["selected"]:
            selected.append((candidate, decision))
    if set(decision_candidates) != set(candidates) or len(decision_candidates) != len(
        set(decision_candidates)
    ):
        raise ValueError("Calibration decision candidates must exactly match candidate_grid.")
    if len(selected) != 1:
        raise ValueError("Calibration evidence must select exactly one candidate.")
    selected_candidate, selected_decision = selected[0]
    requested_candidate = (float(arguments.step_size), int(arguments.leapfrog_steps))
    if selected_candidate != requested_candidate:
        raise ValueError("Selected calibration candidate does not match requested HMC controls.")
    if (
        selected_decision["finite"] is not True
        or selected_decision["divergences"] != 0
        or not 0.6 <= float(selected_decision["development_a_mean_acceptance"]) <= 0.9
        or not 0.6 <= float(selected_decision["development_b_mean_acceptance"]) <= 0.9
    ):
        raise ValueError("Selected calibration decision does not pass the frozen acceptance gates.")

    selected_validation = _require_exact_keys(
        evidence["selected_validation"],
        {"step_size", "leapfrog_steps", "initializers"},
        name="Calibration selected_validation",
    )
    validation_candidate = _calibration_candidate(
        {
            "step_size": selected_validation["step_size"],
            "leapfrog_steps": selected_validation["leapfrog_steps"],
        },
        name="Calibration selected_validation",
    )
    if validation_candidate != selected_candidate:
        raise ValueError("Calibration selected_validation candidate does not match the selected decision.")
    validation_initializers = selected_validation["initializers"]
    if not isinstance(validation_initializers, list) or len(validation_initializers) != 3:
        raise ValueError("Calibration selected_validation must declare exactly three initializers.")
    expected_validation_identities = (
        ("development-a", development_topology_seeds[0]),
        ("development-b", development_topology_seeds[1]),
        ("held-out", None),
    )
    held_out_seed: int | None = None
    validation_sampler_seeds: list[int] = []
    for index, (expected_role, expected_seed) in enumerate(expected_validation_identities):
        initializer = _require_exact_keys(
            validation_initializers[index],
            {
                "role",
                "strategy",
                "seed",
                "sampler_seed",
                "sweeps",
                "mean_acceptance",
                "divergences",
                "finite",
            },
            name=f"Calibration selected validation initializer {index}",
        )
        sweeps = initializer["sweeps"]
        if (
            initializer["role"] != expected_role
            or initializer["strategy"] != "random-recursive"
            or isinstance(sweeps, bool)
            or not isinstance(sweeps, int)
            or sweeps != 500
        ):
            raise ValueError(
                "Calibration selected validation initializer identities and sweep counts are incompatible."
            )
        seed = initializer["seed"]
        if index < 2:
            if seed != expected_seed or isinstance(seed, bool):
                raise ValueError("Calibration selected validation development seeds are incompatible.")
        elif isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Calibration held-out random-recursive seed must be non-negative.")
        else:
            held_out_seed = seed
        sampler_seed = initializer["sampler_seed"]
        if isinstance(sampler_seed, bool) or not isinstance(sampler_seed, int) or sampler_seed < 0:
            raise ValueError("Calibration selected validation sampler seeds must be non-negative.")
        validation_sampler_seeds.append(sampler_seed)
        mean_acceptance = _finite_json_number(
            initializer["mean_acceptance"],
            name=f"Calibration selected validation initializer {index}.mean_acceptance",
        )
        divergences = initializer["divergences"]
        if (
            initializer["finite"] is not True
            or isinstance(divergences, bool)
            or not isinstance(divergences, int)
            or divergences != 0
            or not 0.6 <= mean_acceptance <= 0.9
        ):
            raise ValueError(
                "Calibration selected validation initializer does not pass the frozen acceptance gates."
            )
    if held_out_seed in development_topology_seeds:
        raise ValueError("Calibration held-out topology seed must differ from development seeds.")
    if len(set(validation_sampler_seeds)) != 3:
        raise ValueError("Calibration selected validation sampler seeds must be distinct.")
    if set(validation_sampler_seeds) & set(development_sampler_seeds):
        raise ValueError("Calibration validation sampler seeds must differ from development sampler seeds.")
    required_held_out_seed = CALIBRATION_HELD_OUT_TOPOLOGY_SEEDS.get(arguments.k)
    if required_held_out_seed is not None and held_out_seed != required_held_out_seed:
        raise ValueError(
            f"Calibration evidence for K={arguments.k} has an incompatible held-out topology seed."
        )
    required_validation_sampler_seeds = CALIBRATION_VALIDATION_SAMPLER_SEEDS.get(arguments.k)
    if (
        required_validation_sampler_seeds is not None
        and tuple(validation_sampler_seeds) != required_validation_sampler_seeds
    ):
        raise ValueError(
            f"Calibration evidence for K={arguments.k} has incompatible validation sampler seeds."
        )

    excluded_topologies = _require_exact_keys(
        evidence["excluded_production_topology_sha256"],
        {"metric_source", "development_a", "development_b", "held_out"},
        name="Calibration excluded production topologies",
    )
    for role, digest in excluded_topologies.items():
        _validate_sha256(
            digest,
            name=f"Calibration excluded topology {role!r} SHA-256",
        )
    if len(set(excluded_topologies.values())) != len(excluded_topologies):
        raise ValueError("Calibration excluded production topology hashes must be distinct.")

    sources = evidence["source_artifact_sha256"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Calibration source_artifact_sha256 must be a nonempty object.")
    for source, digest in sources.items():
        if not source or not isinstance(digest, str):
            raise ValueError("Calibration source artifact IDs and hashes must be strings.")
        _validate_sha256(digest, name=f"Calibration source artifact {source!r} SHA-256")


def _load_verified_calibration(
    arguments: argparse.Namespace,
    adapter: GammaBetaRHIMEAdapterResult,
    *,
    input_digest: str,
    fixed_position_scales: tuple[float, ...],
) -> tuple[dict[str, Any], str]:
    """Load, hash, and exactly validate the invocation's calibration file.

    Args:
        arguments: Validated driver arguments.
        adapter: Frozen scientific adapter with resolved fixed priors.
        input_digest: Verified frozen-input whole-file SHA-256.
        fixed_position_scales: Resolved fixed-coordinate position scales.

    Returns:
        Verified calibration object and lowercase whole-file SHA-256.

    Raises:
        FileNotFoundError: If the calibration file is absent.
        OSError: If the calibration file cannot be read.
        ValueError: If its hash, strict JSON, stability, or v2 identity does
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
    if calibration.get("schema") == LEGACY_CALIBRATION_SCHEMA:
        raise ValueError("Calibration v1 uses the retired diagonal metric; calibration v2 is required.")
    expected = _expected_calibration_identity(
        arguments,
        adapter,
        input_digest=input_digest,
        fixed_position_scales=fixed_position_scales,
    )
    if set(calibration) != {*expected, "evidence"}:
        raise ValueError("Calibration v2 root keys are incompatible.")
    identity = {name: calibration[name] for name in expected}
    if _canonical_json(identity) != _canonical_json(expected):
        raise ValueError("Calibration v2 identity does not exactly match the current invocation.")
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
            {"long_name": "raw structural log acceptance ratio"},
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
            "schema": ("openghg_inversions.full_tiling_pymc_hmc_native_trace.v2"),
            "title": "Static PyMC HMC mobile full-tiling segment",
            "diagnostic_only": "true",
            "convergence_claim": "none",
            "fixed_k": result.final_state.k,
            "schedule_id": FULL_TILING_PYMC_HMC_SCHEDULE_ID,
            "coordinate_layout_id": (FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID),
            "metric_semantics_id": (FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID),
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
        "schema": ("openghg_inversions.full_tiling_pymc_hmc_native_summary.v2"),
        "status": "experimental_not_convergence_evidence",
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
        "--leaf-contrast-position-scale",
        dest="leaf_contrast_position_scale",
        type=float,
        required=True,
        help=(
            "Positive PyMC position-covariance eigenvalue on normalized "
            "centered log-leaf contrasts; equivalently momentum precision. "
            "The contrast/total maximum-to-minimum ratio may not exceed 10000."
        ),
    )
    parser.add_argument(
        "--leaf-total-position-scale",
        dest="leaf_total_position_scale",
        type=float,
        required=True,
        help=(
            "Positive PyMC position-covariance eigenvalue on the normalized "
            "common log-leaf total direction; equivalently momentum precision. "
            "The contrast/total maximum-to-minimum ratio may not exceed 10000."
        ),
    )
    parser.add_argument(
        "--fixed-coefficient-position-scale",
        "--fixed-position-scales",
        dest="fixed_coefficient_position_scale",
        type=_positive_values,
        required=True,
        metavar="VALUE[,VALUE...]",
        help=(
            "Shared or ordered fixed-coordinate PyMC position-covariance "
            "diagonal; equivalently momentum precision."
        ),
    )
    parser.add_argument(
        "--calibration-id",
        required=True,
        help="Stable identifier exactly repeated by the v2 calibration file.",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        required=True,
        help=(
            "Strict-JSON v2 calibration identity binding K, input SHA, target "
            "controls, kernel controls, coordinate layout, metric semantics, "
            "resolved position scales, bounded-search decisions, source "
            "artifact hashes, and calibration code revision."
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
        raise ValueError("--k must be at least two for total/contrast calibration.")
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
            *(
                ()
                if arguments.k not in CALIBRATION_HELD_OUT_TOPOLOGY_SEEDS
                else (CALIBRATION_HELD_OUT_TOPOLOGY_SEEDS[arguments.k],)
            ),
        )
        if arguments.initialization_seed in calibration_topology_seeds:
            raise ValueError("--initialization-seed must be disjoint from H2c calibration topology seeds.")
    elif arguments.initialization_seed is not None:
        raise ValueError("--initialization-seed is only valid with --initialization random-recursive.")
    if arguments.dry_run and arguments.resume_checkpoint is not None:
        raise ValueError("--dry-run cannot be combined with --resume-checkpoint.")
    for name in (
        "step_size",
        "leaf_contrast_position_scale",
        "leaf_total_position_scale",
        "concentration",
        "root_variance",
    ):
        value = float(getattr(arguments, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
    leaf_metric_scales = (
        float(arguments.leaf_contrast_position_scale),
        float(arguments.leaf_total_position_scale),
    )
    if max(leaf_metric_scales) / min(leaf_metric_scales) > MAX_LEAF_METRIC_CONDITION_RATIO:
        raise ValueError(
            "The leaf metric maximum-to-minimum position-scale ratio must not exceed "
            f"{MAX_LEAF_METRIC_CONDITION_RATIO:g}."
        )
    position_scale_values = (
        *leaf_metric_scales,
        *(float(value) for value in arguments.fixed_coefficient_position_scale),
    )
    lower_scale, upper_scale = POSITION_SCALE_CLIPPING_BOUNDS
    if any(value < lower_scale or value > upper_scale for value in position_scale_values):
        raise ValueError(
            "All requested position scales must lie within the frozen clipping "
            f"bounds [{lower_scale:g}, {upper_scale:g}]."
        )
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
    fixed_position_scales: tuple[float, ...],
) -> FullTilingPyMCHMCKernelSettings:
    """Return the exact resolved v2 CLI kernel settings.

    Args:
        arguments: Validated namespace containing fixed ``K``, trajectory
            controls, and both leaf position-covariance eigenscales.
        fixed_position_scales: Ordered fixed-coefficient
            position-covariance diagonal.

    Returns:
        Complete immutable total/contrast HMC kernel settings.
    """
    return FullTilingPyMCHMCKernelSettings(
        fixed_k=arguments.k,
        step_size=arguments.step_size,
        leapfrog_steps=arguments.leapfrog_steps,
        leaf_contrast_position_scale=arguments.leaf_contrast_position_scale,
        leaf_total_position_scale=arguments.leaf_total_position_scale,
        fixed_coefficient_position_scale=fixed_position_scales,
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
        fixed_position_scales = _expand_values(
            arguments.fixed_coefficient_position_scale,
            size=adapter.problem.n_fixed_coefficients,
            name="fixed_coefficient_position_scale",
        )
        requested_settings = _requested_kernel_settings(
            arguments,
            fixed_position_scales,
        )
        calibration, calibration_digest = _load_verified_calibration(
            arguments,
            adapter,
            input_digest=input_digest,
            fixed_position_scales=fixed_position_scales,
        )
        initial_topology_sha256 = _topology_sha256(_rectangle_bounds(initial_state))
        excluded_topologies = calibration["evidence"]["excluded_production_topology_sha256"]
        if initial_topology_sha256 in excluded_topologies.values():
            raise ValueError(
                "Initial topology hash was used by H2c calibration and is excluded from retained production."
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
            fixed_position_scales=fixed_position_scales,
            calibration=calibration,
            calibration_digest=calibration_digest,
        )
        if arguments.dry_run:
            return {
                "schema": ("openghg_inversions.full_tiling_pymc_hmc_native_summary.v2"),
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
                    leaf_contrast_position_scale=(arguments.leaf_contrast_position_scale),
                    leaf_total_position_scale=arguments.leaf_total_position_scale,
                    fixed_coefficient_position_scale=(fixed_position_scales),
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
