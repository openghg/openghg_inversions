"""Sample one deterministic full-tiling basis with NumPyro NUTS via PyMC.

This executable is a continuous-sampler reference for the fixed-basis
Gamma--Dirichlet leaf marginal induced by the experimental Gamma--Beta
allocation. It consumes one explicitly frozen NetCDF file,
constructs the same deterministic ``largest-nominal`` basis as the fixed-basis
Metropolis-within-Gibbs comparison, and runs exactly one independently seeded
chain. Continuous coordinates may start at their exact prior means or at a
reproducible draw from every declared prior. Geometry is conditioned on the
recorded basis; this is not an RJMCMC run and makes no checkpoint/restart
claim.

Every scientific input variable and prior is explicit. The Gaussian
likelihood power is fixed at one. Before sampling, the driver requires
float64 from JAX and PyTensor and compares the PyMC model log density without
transform Jacobians against the existing immutable state's normalized log
target. NumPyro receives the explicitly recorded continuous initial point
without jitter.

Successful runs publish ``trace.nc``, ``manifest.json``, and ``summary.json``
inside a create-only output directory. ``complete.json`` is hash-certified
and written last. An interrupted or invalid run therefore cannot masquerade
as a complete result. ``--dry-run`` performs the input, backend, model, and
log-density checks without creating the output directory.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import errno
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import xarray as xr

from openghg_inversions.experimental.rjmcmc.fixed_basis_nuts import (
    FixedBasisNUTSData,
    build_fixed_basis_pymc_model,
    fixed_basis_nuts_initvals,
    prepare_fixed_basis_nuts,
    preflight_fixed_basis_nuts,
    require_fixed_basis_nuts_float64,
    sample_fixed_basis_nuts,
)
from openghg_inversions.experimental.rjmcmc.core import lognormal_mu_sigma
from openghg_inversions.experimental.rjmcmc.full_tiling import TilingState
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    FullTilingPosteriorState,
    build_full_tiling_posterior_state,
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
_CLOSURE_RTOL = 1.0e-12
_CLOSURE_ATOL = 1.0e-12
_TRACE_RTOL = 5.0e-12
_TRACE_ATOL = 5.0e-10
_ROOT_INTERPRETATION = "normalized nominal-weight aggregate scaling"
_SAMPLER = "pymc_numpyro_nuts"


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    """Return deterministic strict JSON with one trailing newline."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _fsync_directory(path: Path) -> None:
    """Flush a directory entry on filesystems that support it."""
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
    """Atomically publish and flush one UTF-8 text artifact."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _require_variable(
    dataset: xr.Dataset,
    *,
    group: str,
    name: str,
    dims: tuple[str, ...],
) -> xr.DataArray:
    """Return one required variable with its exact dimension order."""
    if name not in dataset.data_vars:
        raise RuntimeError(f"Trace group {group!r} is missing required variable {name!r}.")
    variable = dataset[name]
    if variable.dims != dims:
        raise RuntimeError(
            f"Trace variable {group}.{name} must have dimensions {dims}; found {variable.dims}."
        )
    return variable


def _require_coordinates(
    dataset: xr.Dataset,
    *,
    group: str,
    expected: Mapping[str, np.ndarray],
) -> None:
    """Require exact dimension coordinates for one trace group."""
    for dimension, expected_values in expected.items():
        if dimension not in dataset.coords:
            raise RuntimeError(f"Trace group {group!r} is missing required coordinate {dimension!r}.")
        actual = np.asarray(dataset.coords[dimension].values)
        if not np.array_equal(actual, expected_values):
            raise RuntimeError(f"Trace group {group!r} has incorrect {dimension!r} coordinate values.")


def _require_float64_finite(
    variable: xr.DataArray,
    *,
    qualified_name: str,
    positive: bool = False,
) -> np.ndarray:
    """Return a scientific trace array after strict precision/support checks."""
    values = np.asarray(variable.values)
    if values.dtype != np.dtype(np.float64):
        raise RuntimeError(f"Trace variable {qualified_name} must have dtype float64; found {values.dtype}.")
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Trace variable {qualified_name} contains non-finite values.")
    if positive and np.any(values <= 0.0):
        raise RuntimeError(f"Trace variable {qualified_name} must be strictly positive.")
    return values


def _maximum_absolute_difference(actual: np.ndarray, expected: np.ndarray) -> float:
    """Return a finite maximum absolute difference, including empty arrays."""
    return float(np.max(np.abs(actual - expected), initial=0.0))


def _require_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    qualified_name: str,
) -> float:
    """Require two scientific trace arrays to agree at float64 tolerance."""
    maximum_error = _maximum_absolute_difference(actual, expected)
    if not np.allclose(
        actual,
        expected,
        rtol=_TRACE_RTOL,
        atol=_TRACE_ATOL,
    ):
        raise RuntimeError(
            f"Trace variable {qualified_name} violates its deterministic identity; "
            f"maximum absolute error is {maximum_error:.17g}."
        )
    return maximum_error


def _validate_inference_data(
    inference_data: Any,
    *,
    data: FixedBasisNUTSData,
    expected_draws: int,
) -> dict[str, object]:
    """Validate a complete one-chain scientific NUTS trace.

    Validation covers the ArviZ group/schema contract, exact dimension
    coordinates, float64 scientific arrays, constrained support, deterministic
    mass and forward-model identities, stable NUTS diagnostics, and the
    pointwise Gaussian log likelihood. The same function is applied to the
    in-memory result and to the reopened NetCDF artifact.
    """
    import arviz as az

    if not isinstance(inference_data, az.InferenceData):
        raise RuntimeError("Sampler output must be an ArviZ InferenceData result.")
    required_groups = ("posterior", "sample_stats", "observed_data", "log_likelihood")
    available_groups = set(inference_data.groups())
    missing_groups = tuple(group for group in required_groups if group not in available_groups)
    if missing_groups:
        raise RuntimeError(
            "Trace is missing required InferenceData group(s): " + ", ".join(missing_groups) + "."
        )
    if expected_draws < 1:
        raise ValueError("expected_draws must be positive.")

    common_coords = {
        "chain": np.arange(1, dtype=np.int64),
        "draw": np.arange(expected_draws, dtype=np.int64),
    }
    leaf_coords = np.asarray(data.leaf_labels, dtype=np.str_)
    fixed_coords = np.asarray(
        [f"fixed_{position}" for position in range(data.n_fixed_coefficients)],
        dtype=np.str_,
    )
    observation_coords = np.arange(data.observations.size, dtype=np.int64)

    posterior: xr.Dataset = getattr(inference_data, "posterior")
    _require_coordinates(
        posterior,
        group="posterior",
        expected={
            **common_coords,
            "leaf": leaf_coords,
            "fixed": fixed_coords,
            "observation": observation_coords,
        },
    )
    root_total = _require_float64_finite(
        _require_variable(
            posterior,
            group="posterior",
            name="root_total",
            dims=("chain", "draw"),
        ),
        qualified_name="posterior.root_total",
        positive=True,
    )
    leaf_share = _require_float64_finite(
        _require_variable(
            posterior,
            group="posterior",
            name="leaf_share",
            dims=("chain", "draw", "leaf"),
        ),
        qualified_name="posterior.leaf_share",
        positive=True,
    )
    leaf_mass = _require_float64_finite(
        _require_variable(
            posterior,
            group="posterior",
            name="leaf_mass",
            dims=("chain", "draw", "leaf"),
        ),
        qualified_name="posterior.leaf_mass",
        positive=True,
    )
    leaf_scaling = _require_float64_finite(
        _require_variable(
            posterior,
            group="posterior",
            name="leaf_scaling",
            dims=("chain", "draw", "leaf"),
        ),
        qualified_name="posterior.leaf_scaling",
        positive=True,
    )
    fixed_coefficient = _require_float64_finite(
        _require_variable(
            posterior,
            group="posterior",
            name="fixed_coefficient",
            dims=("chain", "draw", "fixed"),
        ),
        qualified_name="posterior.fixed_coefficient",
        positive=True,
    )
    mean_observation = _require_float64_finite(
        _require_variable(
            posterior,
            group="posterior",
            name="mean_observation",
            dims=("chain", "draw", "observation"),
        ),
        qualified_name="posterior.mean_observation",
    )

    share_sums = np.sum(leaf_share, axis=-1)
    maximum_simplex_error = _maximum_absolute_difference(
        share_sums,
        np.ones_like(share_sums),
    )
    if not np.allclose(share_sums, 1.0, rtol=0.0, atol=5.0e-12):
        raise RuntimeError(
            "Trace variable posterior.leaf_share does not lie on the simplex; "
            f"maximum sum error is {maximum_simplex_error:.17g}."
        )
    maximum_mass_error = _require_close(
        leaf_mass,
        root_total[..., np.newaxis] * leaf_share,
        qualified_name="posterior.leaf_mass",
    )
    maximum_scaling_error = _require_close(
        leaf_scaling,
        leaf_mass / data.nominal_leaf_share,
        qualified_name="posterior.leaf_scaling",
    )

    flattened_mass = leaf_mass.reshape(expected_draws, data.k)
    flattened_fixed = fixed_coefficient.reshape(
        expected_draws,
        data.n_fixed_coefficients,
    )
    flattened_mean = mean_observation.reshape(expected_draws, data.observations.size)
    maximum_mean_error = 0.0
    for start in range(0, expected_draws, 128):
        stop = min(start + 128, expected_draws)
        expected_mean = (
            data.fixed_offset
            + flattened_mass[start:stop] @ data.dynamic_design.T
            + flattened_fixed[start:stop] @ data.fixed_design.T
        )
        maximum_mean_error = max(
            maximum_mean_error,
            _require_close(
                flattened_mean[start:stop],
                expected_mean,
                qualified_name="posterior.mean_observation",
            ),
        )

    sample_stats: xr.Dataset = getattr(inference_data, "sample_stats")
    _require_coordinates(
        sample_stats,
        group="sample_stats",
        expected=common_coords,
    )
    diverging = np.asarray(
        _require_variable(
            sample_stats,
            group="sample_stats",
            name="diverging",
            dims=("chain", "draw"),
        ).values
    )
    if diverging.dtype != np.dtype(bool):
        raise RuntimeError(f"Trace variable sample_stats.diverging must be Boolean; found {diverging.dtype}.")
    for name in ("n_steps", "tree_depth"):
        values = np.asarray(
            _require_variable(
                sample_stats,
                group="sample_stats",
                name=name,
                dims=("chain", "draw"),
            ).values
        )
        if not np.issubdtype(values.dtype, np.integer):
            raise RuntimeError(f"Trace variable sample_stats.{name} must have integer dtype.")
        if np.any(values < 1):
            raise RuntimeError(f"Trace variable sample_stats.{name} must be positive.")
    for name in ("acceptance_rate", "energy", "lp", "step_size"):
        values = _require_float64_finite(
            _require_variable(
                sample_stats,
                group="sample_stats",
                name=name,
                dims=("chain", "draw"),
            ),
            qualified_name=f"sample_stats.{name}",
            positive=name == "step_size",
        )
        if name == "acceptance_rate" and np.any((values < 0.0) | (values > 1.0)):
            raise RuntimeError("Trace variable sample_stats.acceptance_rate must lie in [0, 1].")

    observed_data: xr.Dataset = getattr(inference_data, "observed_data")
    _require_coordinates(
        observed_data,
        group="observed_data",
        expected={"observation": observation_coords},
    )
    observed = _require_float64_finite(
        _require_variable(
            observed_data,
            group="observed_data",
            name="observed",
            dims=("observation",),
        ),
        qualified_name="observed_data.observed",
    )
    if not np.array_equal(observed, data.observations):
        raise RuntimeError("Trace variable observed_data.observed does not match the frozen input.")

    log_likelihood: xr.Dataset = getattr(inference_data, "log_likelihood")
    _require_coordinates(
        log_likelihood,
        group="log_likelihood",
        expected={**common_coords, "observation": observation_coords},
    )
    pointwise = _require_float64_finite(
        _require_variable(
            log_likelihood,
            group="log_likelihood",
            name="observed",
            dims=("chain", "draw", "observation"),
        ),
        qualified_name="log_likelihood.observed",
    )
    residual = (data.observations - mean_observation) / data.observation_sd
    expected_pointwise = -0.5 * residual * residual - np.log(data.observation_sd) - 0.5 * np.log(2.0 * np.pi)
    maximum_log_likelihood_error = _require_close(
        pointwise,
        expected_pointwise,
        qualified_name="log_likelihood.observed",
    )
    return {
        "groups": list(required_groups),
        "chains": 1,
        "draws": expected_draws,
        "leaves": data.k,
        "fixed_coefficients": data.n_fixed_coefficients,
        "observations": int(data.observations.size),
        "maximum_leaf_share_simplex_error": maximum_simplex_error,
        "maximum_leaf_mass_identity_error": maximum_mass_error,
        "maximum_leaf_scaling_identity_error": maximum_scaling_error,
        "maximum_mean_observation_identity_error": maximum_mean_error,
        "maximum_pointwise_log_likelihood_error": maximum_log_likelihood_error,
    }


def _atomic_write_trace(
    inference_data: Any,
    path: Path,
    *,
    data: FixedBasisNUTSData,
    expected_draws: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate, serialize, reopen, and atomically publish one ArviZ trace."""
    temporary = path.with_name(f".{path.name}.tmp.nc")
    in_memory_audit = _validate_inference_data(
        inference_data,
        data=data,
        expected_draws=expected_draws,
    )
    try:
        inference_data.to_netcdf(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        import arviz as az

        reopened: Any = az.from_netcdf(temporary)
        try:
            reopened_audit = _validate_inference_data(
                reopened,
                data=data,
                expected_draws=expected_draws,
            )
        finally:
            reopened.close()
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return in_memory_audit, reopened_audit
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
    """Broadcast a scalar or validate an exact-width prior vector."""
    if not isinstance(values, tuple):
        return (float(values),) * size
    if len(values) != size:
        raise ValueError(f"{name} must be scalar or contain exactly {size} values.")
    return values


def _load_frozen_dataset(path: Path, *, engine: NetCDFEngine) -> xr.Dataset:
    """Eagerly load and close one immutable-on-entry NetCDF input."""
    if not path.is_file():
        raise FileNotFoundError(f"Frozen input is not a file: {path}")
    with xr.open_dataset(path, engine=engine) as opened:
        return opened.load()


def _input_array(dataset: xr.Dataset, name: str) -> xr.DataArray:
    """Return an explicitly named required data variable."""
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
    """Build the reviewed RHIME-to-Gamma--Beta adapter explicitly."""
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
        likelihood_power=1.0,
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
    """Reject input outside the reviewed modern PARIS dimensions and labels."""
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
    """Verify mass-coordinate and complete prior-mean forward-model closure."""
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
        raise RuntimeError("The fixed-basis NUTS driver requires fixed design and offset terms.")
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
            "offset, and fixed-coefficient prior means."
        )
    return {
        "mass_coordinate_max_abs_error": float(np.max(np.abs(mass_error), initial=0.0)),
        "prior_mean_total_max_abs_error": float(np.max(np.abs(total_error), initial=0.0)),
    }


def _rectangle_bounds(initial_state: FullTilingPosteriorState) -> list[list[int]]:
    """Return canonical half-open rectangle bounds in state order."""
    return [
        [leaf.row_start, leaf.row_stop, leaf.col_start, leaf.col_stop]
        for leaf in initial_state.allocation.tiling.leaves
    ]


def _continuous_initial_state(
    prior_mean_state: FullTilingPosteriorState,
    *,
    profile: str,
    seed: int | None,
) -> FullTilingPosteriorState:
    """Construct the audited continuous initial state on fixed geometry.

    Args:
        prior_mean_state: Deterministic basis with exact prior-mean
            coordinates.
        profile: ``"prior-mean"`` or ``"prior-draw"``.
        seed: Dedicated PCG64 seed required only for a prior draw.

    Returns:
        Fully rebuilt state on the unchanged deterministic tiling.

    Raises:
        ValueError: If the profile or seed contract is inconsistent.
    """
    if profile == "prior-mean":
        if seed is not None:
            raise ValueError(
                "--initialization-seed is only valid with --continuous-initialization prior-draw."
            )
        return prior_mean_state
    if profile != "prior-draw":
        raise ValueError("--continuous-initialization must be 'prior-mean' or 'prior-draw'.")
    if seed is None:
        raise ValueError("--continuous-initialization prior-draw requires --initialization-seed.")
    generator = np.random.Generator(np.random.PCG64(seed))
    problem = prior_mean_state.problem
    prior = problem.base.prior
    root_total = float(
        generator.gamma(
            shape=prior.root_shape,
            scale=1.0 / prior.root_rate,
        )
    )
    alphas = problem.allocation_prior.leaf_alphas(prior_mean_state.allocation.tiling)
    shares = generator.dirichlet(alphas)
    masses = root_total * shares
    fixed_block = problem.base.fixed_block
    if fixed_block is None:
        fixed_coefficients = np.empty(0, dtype=np.float64)
    else:
        fixed_coefficients = np.empty(
            fixed_block.n_coefficients,
            dtype=np.float64,
        )
        for index, (mean, standard_deviation) in enumerate(
            zip(
                fixed_block.coefficient_prior_mean,
                fixed_block.coefficient_prior_sd,
                strict=True,
            )
        ):
            mu, sigma = lognormal_mu_sigma(
                float(mean),
                float(standard_deviation),
            )
            fixed_coefficients[index] = generator.lognormal(mean=mu, sigma=sigma)
    return build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(
            prior_mean_state.allocation.tiling,
            masses,
        ),
        fixed_coefficients=fixed_coefficients,
    )


def _initialization_metadata(
    state: FullTilingPosteriorState,
    *,
    profile: str,
    seed: int | None,
) -> dict[str, object]:
    """Return the exact constrained coordinates passed to NumPyro."""
    return {
        "geometry": "deterministic_largest_nominal",
        "continuous_profile": profile,
        "continuous_initialization_seed": seed,
        "rng": "none" if seed is None else "dedicated_numpy_pcg64",
        "jitter": False,
        "root_total": float(state.root_total),
        "leaf_share": (
            np.asarray(state.allocation.leaf_masses, dtype=np.float64) / state.root_total
        ).tolist(),
        "fixed_coefficient": state.fixed_coefficients.astype(float).tolist(),
    }


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


def _package_versions() -> dict[str, str]:
    """Return versions of libraries that define the compiled sampler."""
    versions: dict[str, str] = {}
    for distribution in ("openghg_inversions", "pymc", "pytensor", "jax", "jaxlib", "numpyro"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unknown"
    return versions


def _backend_metadata() -> dict[str, object]:
    """Return required precision metadata plus the selected JAX backend."""
    metadata: dict[str, object] = dict(require_fixed_basis_nuts_float64())
    import jax

    metadata["jax_default_backend"] = str(jax.default_backend())
    metadata["jax_devices"] = [
        {
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        for device in jax.devices()
    ]
    return metadata


def _manifest(
    arguments: argparse.Namespace,
    adapter: GammaBetaRHIMEAdapterResult,
    *,
    input_digest: str,
    outer_labels: np.ndarray,
    initial_state: FullTilingPosteriorState,
    backend: Mapping[str, object],
) -> dict[str, object]:
    """Build the immutable scientific and computational run identity."""
    fixed_block = adapter.problem.fixed_block
    if fixed_block is None:
        raise RuntimeError("The fixed-basis NUTS driver requires a fixed design block.")
    bounds = _rectangle_bounds(initial_state)
    manifest: dict[str, object] = {
        "schema": "openghg_inversions.full_tiling_fixed_basis_nuts_manifest.v1",
        "status": "fixed_basis_continuous_sampler_reference",
        "input": {
            "id": arguments.input_id,
            "path": str(arguments.input.resolve()),
            "sha256": input_digest,
            "contract": _input_contract(arguments),
            "weight_normalization_factor": adapter.weight_normalization_factor,
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
            "root_interpretation": _ROOT_INTERPRETATION,
            "likelihood_power": 1.0,
            "fixed_prior_mean": fixed_block.coefficient_prior_mean.astype(float).tolist(),
            "fixed_prior_sd": fixed_block.coefficient_prior_sd.astype(float).tolist(),
            "structural_target": "point_mass_at_recorded_deterministic_tiling",
            "rectangle_bounds": bounds,
            "topology_sha256": _topology_sha256(bounds),
        },
        "sampler": {
            "name": _SAMPLER,
            "chains_per_invocation": 1,
            "draws": int(arguments.draws),
            "tune": int(arguments.tune),
            "target_accept": float(arguments.target_accept),
            "max_tree_depth": int(arguments.max_tree_depth),
            "dense_mass": bool(arguments.dense_mass),
            "initialization": _initialization_metadata(
                initial_state,
                profile=arguments.continuous_initialization,
                seed=arguments.initialization_seed,
            ),
            "backend": dict(backend),
        },
        "provenance": {
            "input_id": arguments.input_id,
            "code_revision": arguments.code_revision,
            "chain_id": arguments.chain_id,
            "seed": int(arguments.seed),
            "single_process": True,
            "checkpoint_or_restart_supported": False,
            "library_versions": _package_versions(),
        },
    }
    manifest["manifest_payload_sha256"] = sha256(_canonical_json(manifest).encode()).hexdigest()
    return manifest


def _sample_stat_summary(inference_data: Any) -> dict[str, object]:
    """Summarize stable NumPyro diagnostics available in ``sample_stats``."""
    stats = inference_data.sample_stats
    output: dict[str, object] = {}
    if "diverging" in stats:
        output["divergences"] = int(np.count_nonzero(np.asarray(stats["diverging"].values)))
    for name in ("n_steps", "tree_depth"):
        if name in stats:
            values = np.asarray(stats[name].values, dtype=np.float64)
            output[f"{name}_mean"] = float(np.mean(values))
            output[f"{name}_maximum"] = float(np.max(values))
    for name in ("acceptance_rate", "energy", "step_size"):
        if name in stats:
            values = np.asarray(stats[name].values, dtype=np.float64)
            output[f"{name}_mean"] = float(np.mean(values))
    return output


def _write_outputs(
    output_directory: Path,
    *,
    inference_data: Any,
    data: FixedBasisNUTSData,
    expected_draws: int,
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
) -> dict[str, object]:
    """Publish a create-only bundle and write its completion marker last.

    Args:
        output_directory: New directory in which to publish the bundle.
        inference_data: Serializable ArviZ inference data.
        data: Immutable scientific model data used to validate the trace.
        expected_draws: Exact retained draw count required in the trace.
        manifest: Immutable scientific and computational run identity.
        summary: Completed-run summary.

    Returns:
        Published summary with in-memory and reopened-NetCDF validation
        audits.

    Raises:
        FileExistsError: If ``output_directory`` already exists.
        OSError: If an artifact cannot be written, flushed, or reopened.
        RuntimeError: If the serialized trace fails its reopen validation.

    Note:
        Failure after directory creation deliberately leaves an incomplete
        directory without ``complete.json``; it cannot masquerade as a
        completed run.
    """
    output_directory.mkdir()
    _fsync_directory(output_directory.parent)
    in_memory_audit, reopened_audit = _atomic_write_trace(
        inference_data,
        output_directory / TRACE_FILENAME,
        data=data,
        expected_draws=expected_draws,
    )
    published_summary = dict(summary)
    published_summary["trace_validation"] = {
        "in_memory": in_memory_audit,
        "reopened_netcdf": reopened_audit,
    }
    _atomic_write_text(
        output_directory / MANIFEST_FILENAME,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write_text(
        output_directory / SUMMARY_FILENAME,
        json.dumps(published_summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    artifact_hashes = {
        name: _sha256_file(output_directory / name)
        for name in (TRACE_FILENAME, MANIFEST_FILENAME, SUMMARY_FILENAME)
    }
    completion = {
        "schema": "openghg_inversions.full_tiling_fixed_basis_nuts_completion.v1",
        "status": "complete",
        "sampler": _SAMPLER,
        "checkpoint_or_restart_supported": False,
        "trace": TRACE_FILENAME,
        "manifest": MANIFEST_FILENAME,
        "summary": SUMMARY_FILENAME,
        "sha256": artifact_hashes,
    }
    _atomic_write_text(output_directory / COMPLETION_FILENAME, _canonical_json(completion))
    return published_summary


def build_parser() -> argparse.ArgumentParser:
    """Build the one-chain fixed-basis NumPyro NUTS command-line interface.

    Returns:
        Parser exposing the complete scientific, sampler, and provenance
        contract.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Frozen NetCDF input snapshot.")
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="New output directory; existing paths are rejected.",
    )
    parser.add_argument("--k", "--fixed-k", dest="k", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True, help="Post-warmup draws for this chain.")
    parser.add_argument("--tune", type=int, required=True, help="NUTS warmup/adaptation draws.")
    parser.add_argument("--seed", type=int, required=True, help="Non-negative NumPyro chain seed.")
    parser.add_argument("--chain-id", required=True, help="Stable logical one-chain identifier.")
    parser.add_argument(
        "--continuous-initialization",
        choices=("prior-mean", "prior-draw"),
        required=True,
        help=(
            "Continuous start on the deterministic basis: exact prior means "
            "or one independently seeded draw from every declared prior."
        ),
    )
    parser.add_argument(
        "--initialization-seed",
        type=int,
        help="Required dedicated PCG64 seed for --continuous-initialization prior-draw.",
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
        help="Positive Gamma aggregate-scaling prior variance.",
    )
    parser.add_argument(
        "--fixed-prior-mean",
        type=_positive_values,
        required=True,
        metavar="VALUE[,VALUE...]",
        help="Arithmetic lognormal prior mean(s) for fixed outer coefficients.",
    )
    parser.add_argument(
        "--fixed-prior-sd",
        type=_positive_values,
        required=True,
        metavar="VALUE[,VALUE...]",
        help="Arithmetic lognormal prior SD(s) for fixed outer coefficients.",
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        required=True,
        help="NumPyro NUTS target acceptance probability.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        required=True,
        help="Positive maximum NumPyro NUTS tree depth.",
    )
    parser.add_argument(
        "--dense-mass",
        action=argparse.BooleanOptionalAction,
        default=None,
        required=True,
        help="Use or disable a dense adapted NUTS mass matrix.",
    )
    parser.add_argument(
        "--likelihood-power",
        type=float,
        default=1.0,
        help="Must be exactly 1; powered/tempered likelihoods are excluded.",
    )
    parser.add_argument("--input-id", required=True, help="Stable frozen-input identifier.")
    parser.add_argument(
        "--code-revision",
        required=True,
        help="Caller-supplied source revision verified by the external HPC preflight.",
    )
    parser.add_argument(
        "--expected-input-sha256",
        required=True,
        help="Required whole-file SHA-256 for the frozen input.",
    )
    parser.add_argument(
        "--nominal-weight-policy",
        required=True,
        help="Reviewed policy identifier for the positive nominal base measure.",
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
        help="Require the reviewed 1382-observation, 183x128, six-outer profile.",
    )
    parser.add_argument(
        "--expected-outer-labels",
        type=_comma_separated_labels,
        help="Expected comma-separated outer-region labels in exact order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input, float64 backend, model, and exact initial log density only.",
    )
    parser.add_argument(
        "--progressbar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show PyMC/NumPyro progress output (default: false for batch logs).",
    )
    parser.add_argument(
        "--input-netcdf-engine",
        choices=("h5netcdf", "netcdf4", "scipy"),
        default="h5netcdf",
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject malformed settings before loading scientific input."""
    if arguments.output_directory.exists():
        raise FileExistsError(f"Output path already exists: {arguments.output_directory}")
    if not arguments.output_directory.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory does not exist: {arguments.output_directory.parent}"
        )
    if arguments.k < 1:
        raise ValueError("--k must be positive.")
    if arguments.draws < 1:
        raise ValueError("--draws must be positive.")
    if arguments.tune < 1:
        raise ValueError("--tune must be positive.")
    if arguments.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if arguments.continuous_initialization == "prior-draw":
        if arguments.initialization_seed is None:
            raise ValueError("--continuous-initialization prior-draw requires --initialization-seed.")
        if arguments.initialization_seed < 0:
            raise ValueError("--initialization-seed must be non-negative.")
        if arguments.initialization_seed == arguments.seed:
            raise ValueError("--initialization-seed must differ from the sampler --seed.")
    elif arguments.initialization_seed is not None:
        raise ValueError("--initialization-seed is only valid with --continuous-initialization prior-draw.")
    if not arguments.chain_id:
        raise ValueError("--chain-id must be nonempty.")
    if not np.isfinite(arguments.concentration) or arguments.concentration <= 0.0:
        raise ValueError("--concentration must be finite and positive.")
    if not np.isfinite(arguments.root_variance) or arguments.root_variance <= 0.0:
        raise ValueError("--root-variance must be finite and positive.")
    if not np.isfinite(arguments.target_accept) or not 0.0 < arguments.target_accept < 1.0:
        raise ValueError("--target-accept must lie strictly between zero and one.")
    if arguments.max_tree_depth < 1:
        raise ValueError("--max-tree-depth must be positive.")
    if arguments.dense_mass is None:
        raise ValueError("Specify exactly one of --dense-mass or --no-dense-mass.")
    if arguments.likelihood_power != 1.0:
        raise ValueError("--likelihood-power must be exactly 1 for this reference.")
    if not arguments.normalize_weights:
        raise ValueError(
            "This reference requires --normalize-weights so root_total is a "
            "normalized nominal-weight aggregate scaling."
        )
    digest = arguments.expected_input_sha256
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise ValueError("--expected-input-sha256 must be exactly 64 hexadecimal characters.")
    if arguments.require_paris_profile and arguments.expected_outer_labels is None:
        raise ValueError("--require-paris-profile requires --expected-outer-labels.")


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    """Validate and execute one fixed-basis NumPyro NUTS chain.

    Args:
        arguments: Parsed namespace from :func:`build_parser`.

    Returns:
        Strict-JSON-compatible dry-run or completed-run summary.

    Raises:
        FileExistsError: If the create-only output target already exists.
        FileNotFoundError: If the input or output parent does not exist.
        ImportError: If the required PyMC/JAX/NumPyro runtime is unavailable.
        ValueError: If settings, input, profile, closure, backend, or initial
            log density violate the declared contract.
        RuntimeError: If sampling or artifact validation fails.

    Note:
        A non-dry run publishes a create-only artifact directory.
    """
    _validate_arguments(arguments)
    input_started = perf_counter()
    input_digest = _sha256_file(arguments.input)
    if input_digest.lower() != arguments.expected_input_sha256.lower():
        raise ValueError("Frozen input SHA-256 does not match --expected-input-sha256.")
    dataset = _load_frozen_dataset(arguments.input, engine=arguments.input_netcdf_engine)
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
                expected_outer_labels=arguments.expected_outer_labels,
            )
        problem = full_tiling_problem_from_gamma_beta_adapter(
            adapter,
            concentration=arguments.concentration,
        )
        prior_mean_state = initialize_full_tiling_posterior_state(
            problem,
            k=arguments.k,
        )
        initial_state = _continuous_initial_state(
            prior_mean_state,
            profile=arguments.continuous_initialization,
            seed=arguments.initialization_seed,
        )
        if not np.isfinite(initial_state.log_target):
            raise ValueError("Exact initial full-tiling log target is not finite.")
        closure = _closure_audit(
            dataset,
            adapter,
            initial_state=prior_mean_state,
            sensitivity_name=arguments.sensitivity_name,
            fixed_design_name=arguments.fixed_design_name,
            fixed_offset_name=arguments.fixed_offset_name,
        )
        fixed_design = dataset[arguments.fixed_design_name]
        fixed_dimension = next(dimension for dimension in fixed_design.dims if dimension != "nmeasure")
        outer_labels = _dimension_labels(dataset, fixed_dimension)
        backend = _backend_metadata()
        bridge: FixedBasisNUTSData = prepare_fixed_basis_nuts(
            problem,
            initial_state,
        )
        initvals = fixed_basis_nuts_initvals(bridge)
        model = build_fixed_basis_pymc_model(bridge)
        preflight = preflight_fixed_basis_nuts(
            bridge,
            model,
            initvals=initvals,
            expected_log_target=initial_state.log_target,
        )
        manifest = _manifest(
            arguments,
            adapter,
            input_digest=input_digest,
            outer_labels=outer_labels,
            initial_state=initial_state,
            backend=backend,
        )
        setup_seconds = perf_counter() - setup_started
        base_summary: dict[str, Any] = {
            "schema": "openghg_inversions.full_tiling_fixed_basis_nuts_summary.v1",
            "status": "dry_run" if arguments.dry_run else "complete",
            "input": {
                "id": arguments.input_id,
                "path": str(arguments.input.resolve()),
                "sha256": input_digest,
            },
            "fixed_k": int(arguments.k),
            "chain_id": arguments.chain_id,
            "root_interpretation": _ROOT_INTERPRETATION,
            "topology_sha256": _topology_sha256(_rectangle_bounds(initial_state)),
            "closure": closure,
            "preflight": dict(preflight),
            "target": {
                "existing_state_log_target": float(initial_state.log_target),
                "pymc_initial_logp_jacobian_false": float(preflight["constrained_log_target"]),
                "absolute_difference": abs(float(preflight["log_target_difference"])),
                "absolute_tolerance": float(preflight["log_target_absolute_tolerance"]),
                "likelihood_power": 1.0,
            },
            "performance": {
                "input_hash_and_load_seconds": input_seconds,
                "problem_setup_and_preflight_seconds": setup_seconds,
            },
        }
        if arguments.dry_run:
            base_summary["manifest"] = manifest
            return base_summary

        sampling_started = perf_counter()
        inference_data: Any = sample_fixed_basis_nuts(
            model,
            bridge,
            draws=arguments.draws,
            tune=arguments.tune,
            seed=arguments.seed,
            target_accept=arguments.target_accept,
            max_tree_depth=arguments.max_tree_depth,
            dense_mass=arguments.dense_mass,
            chains=1,
            cores=1,
            chain_method="parallel",
            progressbar=arguments.progressbar,
            initvals=initvals,
        )
        sampling_seconds = perf_counter() - sampling_started
        _validate_inference_data(
            inference_data,
            data=bridge,
            expected_draws=arguments.draws,
        )
        chains = 1
        draws = int(arguments.draws)
        base_summary["sampler_diagnostics"] = _sample_stat_summary(inference_data)
        base_summary["performance"]["sampling_seconds"] = sampling_seconds
        base_summary["performance"]["total_wall_seconds"] = input_seconds + setup_seconds + sampling_seconds
        base_summary["run"] = {
            "chains": chains,
            "draws": draws,
            "tune": int(arguments.tune),
            "checkpoint_or_restart_supported": False,
        }
        published_summary = _write_outputs(
            arguments.output_directory,
            inference_data=inference_data,
            data=bridge,
            expected_draws=arguments.draws,
            manifest=manifest,
            summary=base_summary,
        )
        return published_summary
    finally:
        dataset.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run one chain and print its machine-readable summary to stdout.

    Args:
        argv: Optional command-line arguments; defaults to ``sys.argv``.

    Returns:
        Zero after a successful dry run or completed chain.

    Raises:
        Exception: Propagates parser, validation, sampling, and artifact
            publication failures so batch jobs fail closed.
    """
    arguments = build_parser().parse_args(argv)
    summary = run(arguments)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
