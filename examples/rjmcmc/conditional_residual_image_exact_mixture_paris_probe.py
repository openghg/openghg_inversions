#!/usr/bin/env python3
r"""Measure the authenticated PARIS root residual spectrum and resource scale.

This bounded feasibility probe constructs the observation-blind analytic
Gamma--Dirichlet covariance spectrum for one global root on the frozen May
2014 PARIS input.  The conditional continuous law is an exact Gaussian
location mixture and its covariance spectrum is analytic.  Any later rank
projection, finite Sobol bank, Gaussian complement, or cluster compression is
an approximation and requires separate scientific validation.

The driver reuses the validated RHIME-to-physical-mass adapter, the globally
additive full-tiling bridge, :class:`RootResidualSpectrum`, and SciPy's
symmetric eigensolver.  It performs no posterior inference, does not use the
realized residual to select the spectrum, opens no protected catalogue, and
refuses output paths containing ``PARIS_inversions``.

For normalized nominal weights \(u\), the adapter defines
\(\alpha_i=\eta u_i\) and converts unit-scaling responses to the
physical-native-mass design \(H_i=F_i/u_i\).  With
\(A=\operatorname{diag}(\mathtt{mf_error})^{-1}
(H-Hu\mathbf 1^\mathsf T)\), the unit-root allocation covariance is

\[
S_0=A\{\operatorname{diag}(u)-uu^\mathsf T\}A^\mathsf T/(\eta+1),
\qquad S(T)=T^2S_0.
\]

Exactness is conditional on this single-root additive Dirichlet model and
fixed diagonal Gaussian measurement error.  A different base measure or
error covariance, including an AR(1) mismatch model, changes the spectrum.

The durable derivation is in the sibling ``inversions-knowledge`` repository:
``docs/derivations/non-gaussian-aggregation-error-by-marginalization.md``.
Related qualifications are in
``docs/source-notes/aggregation-error-and-priors.md`` and
``docs/research-questions/learning-non-gaussian-marginal-models.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import scipy
import xarray as xr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openghg_inversions.experimental.rjmcmc.aggregation_error_exact_mixture import (
    RootResidualSpectrum,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    aggregation_from_full_tiling_problem,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    full_tiling_problem_from_gamma_beta_adapter,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    gamma_beta_problem_from_rhime_inputs,
)

SCHEMA = "rjmcmc-exact-mixture-paris-root-resource-probe-v1"
PROTOCOL = "paris-root-analytic-spectrum-resource-probe-v1"
PARIS_INPUT_SCHEMA = "paris-may-2014-gamma-beta-native-v1"
PARIS_OBSERVATION_COUNT = 1_382
PARIS_GRID_SHAPE = (183, 128)
PARIS_OUTER_LABELS = tuple(f"intem_label_{index}" for index in range(6))
DEFAULT_VARIANCE_FRACTIONS = (0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999)
DEFAULT_RANKS = (16, 32, 64, 128, 256, 512, 1_024)
DEFAULT_SOURCE_SAMPLE_COUNT = 65_536
DEFAULT_COMPONENT_COUNT = 256
SOBOL_DIMENSION_BLOCK_LIMIT = 21_201
FloatArray = NDArray[np.float64]


def _canonical_json(value: object) -> str:
    """Return strict canonical ASCII JSON."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: NDArray[Any]) -> str:
    """Fingerprint one numerical array with its shape and dtype."""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        content = np.ascontiguousarray(array, dtype="<f8")
    elif np.issubdtype(array.dtype, np.integer):
        content = np.ascontiguousarray(array, dtype="<i8")
    else:
        raise TypeError("probe array hashes support only numeric arrays")
    digest = hashlib.sha256()
    digest.update(content.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(content.shape, dtype="<i8").tobytes())
    digest.update(content.tobytes())
    return digest.hexdigest()


def _full_revision(value: str) -> str:
    """Validate one complete lower-case Git revision."""
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("source_revision must be a 40-character lower-case Git SHA")
    return value


def _sha256(value: str, *, name: str) -> str:
    """Validate one lower-case SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a 64-character lower-case SHA-256")
    return value


def _positive_float(value: object, *, name: str) -> float:
    """Return one finite strictly positive non-Boolean float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    """Return one strictly positive non-Boolean integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _validated_fractions(values: Sequence[float]) -> tuple[float, ...]:
    """Return unique increasing retained-variance fractions in ``(0, 1)``."""
    fractions = tuple(float(value) for value in values)
    if (
        not fractions
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in fractions)
        or tuple(sorted(set(fractions))) != fractions
    ):
        raise ValueError("variance_fractions must be unique, increasing, and lie in (0, 1)")
    return fractions


def _validated_ranks(values: Sequence[int]) -> tuple[int, ...]:
    """Return unique increasing positive diagnostic ranks."""
    ranks = tuple(_positive_integer(value, name="diagnostic rank") for value in values)
    if tuple(sorted(set(ranks))) != ranks:
        raise ValueError("diagnostic_ranks must be unique and increasing")
    return ranks


def _load_frozen_subset(path: Path, *, engine: str | None) -> xr.Dataset:
    """Eagerly load and close only the explicit variables used by the probe.

    Args:
        path: Authenticated frozen NetCDF path.
        engine: Optional explicit xarray backend name.

    Returns:
        In-memory dataset containing the required variables and their
        coordinates and global attributes.

    Raises:
        ValueError: If any required variable is absent.
        OSError: If xarray or the selected backend cannot read the file.

    The whole-file digest authenticates the frozen NetCDF.  Values loaded are
    the unit-scaling response, observations required by the validated adapter,
    fixed independent errors, nominal weights, fixed outer design, and fixed
    boundary offset.  The realized observations never enter spectrum
    selection.
    """
    names = (
        "fp_x_flux",
        "mf",
        "mf_error",
        "nominal_weight",
        "outer_design",
        "YaprioriBC",
    )
    with xr.open_dataset(path, engine=engine) as opened:
        missing = sorted(set(names).difference(opened.data_vars))
        if missing:
            raise ValueError(f"frozen input is missing required variables: {missing}")
        return opened[list(names)].load()


def _require_profile(
    dataset: xr.Dataset,
    *,
    expected_shape: tuple[int, int, int],
    expected_outer_labels: tuple[str, ...],
    expected_schema: str | None,
) -> None:
    """Require exact dimensions, labels, schema, and numerical support.

    Args:
        dataset: Eager in-memory probe subset.
        expected_shape: Exact ``(nmeasure, lat, lon)`` sizes.
        expected_outer_labels: Exact ordered outer-region labels.
        expected_schema: Required global ``schema_id`` or ``None``.

    Raises:
        ValueError: If dimensions, sizes, labels, schema, numerical dtypes,
            finiteness, or strict positivity differ from the frozen contract.
    """
    observations, latitudes, longitudes = expected_shape
    actual_shape = (
        int(dataset.sizes.get("nmeasure", -1)),
        int(dataset.sizes.get("lat", -1)),
        int(dataset.sizes.get("lon", -1)),
    )
    if actual_shape != expected_shape:
        raise ValueError(f"frozen input shape {actual_shape} does not match {expected_shape}")
    expected_dims = {
        "fp_x_flux": ("nmeasure", "lat", "lon"),
        "mf": ("nmeasure",),
        "mf_error": ("nmeasure",),
        "nominal_weight": ("lat", "lon"),
        "outer_design": ("nmeasure", "outer_region"),
        "YaprioriBC": ("nmeasure",),
    }
    for name, dimensions in expected_dims.items():
        if dataset[name].dims != dimensions:
            raise ValueError(f"{name!r} must have dimensions {dimensions}")
    if dataset.sizes.get("outer_region") != len(expected_outer_labels):
        raise ValueError("outer_design has the wrong number of outer regions")
    outer_labels = tuple(str(value) for value in dataset["outer_region"].values.tolist())
    if outer_labels != expected_outer_labels:
        raise ValueError("outer_region labels or order do not match the frozen profile")
    measurement_labels = tuple(str(value) for value in dataset["nmeasure"].values.tolist())
    if len(set(measurement_labels)) != observations:
        raise ValueError("nmeasure labels must be unique")
    if expected_schema is not None and dataset.attrs.get("schema_id") != expected_schema:
        raise ValueError("frozen input schema_id does not match the expected profile")
    for name in ("fp_x_flux", "mf", "mf_error", "nominal_weight", "outer_design", "YaprioriBC"):
        try:
            values = np.asarray(dataset[name].values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name!r} must contain numerical values") from error
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name!r} must contain only finite values")
    if np.any(np.asarray(dataset["mf_error"].values, dtype=np.float64) <= 0.0):
        raise ValueError("mf_error must be strictly positive")
    if np.any(np.asarray(dataset["nominal_weight"].values, dtype=np.float64) <= 0.0):
        raise ValueError("nominal_weight must be strictly positive")
    if latitudes < 1 or longitudes < 1:
        raise ValueError("native spatial axes must be non-empty")


def _spectrum_diagnostics(
    spectrum: RootResidualSpectrum,
    *,
    native_cell_count: int,
    fractions: tuple[float, ...],
    diagnostic_ranks: tuple[int, ...],
) -> dict[str, Any]:
    """Summarize trace ranks and distributional omission bounds.

    Args:
        spectrum: Complete numerically retained analytic root spectrum.
        native_cell_count: Number of native simplex cells.
        fractions: Predeclared cumulative analytic-trace targets.
        diagnostic_ranks: Predeclared omission/mixture rank scenarios.

    Returns:
        JSON-compatible eigenvalue, effective-rank, trace-rank, and
        distributional-bound diagnostics. Eigenvalues and omitted variances
        are squared noise-whitened observation units per squared root mass.

    The reported KL and TV coefficients apply only if directions beyond a
    rank are omitted after convolution with unit Gaussian measurement noise.
    In the intended hybrid likelihood they may instead remain as a Gaussian
    complement; the coefficients are then diagnostics, not incurred error.
    """
    eigenvalues = np.asarray(spectrum.eigenvalues, dtype=np.float64)
    total = float(spectrum.total_variance)
    cumulative = np.cumsum(eigenvalues)
    fraction_records = []
    for fraction in fractions:
        uncapped_rank = (
            int(np.searchsorted(cumulative, fraction * total, side="left")) + 1
            if total > 0.0 and cumulative.size
            else 0
        )
        rank = min(uncapped_rank, spectrum.retained_rank)
        retained = float(cumulative[rank - 1]) if rank else 0.0
        actual_fraction = 1.0 if total == 0.0 else retained / total
        fraction_records.append(
            {
                "requested_fraction": fraction,
                "rank": rank,
                "actual_fraction": actual_fraction,
                "requested_fraction_met": actual_fraction >= fraction,
            }
        )
    rank_records = []
    rank_catalogue = tuple(dict.fromkeys((*diagnostic_ranks, spectrum.retained_rank)))
    for requested_rank in rank_catalogue:
        rank = min(requested_rank, spectrum.retained_rank)
        retained = float(cumulative[rank - 1]) if rank else 0.0
        tail = max(0.0, total - retained)
        rank_records.append(
            {
                "requested_rank": requested_rank,
                "rank": rank,
                "retained_variance_fraction": (1.0 if total == 0.0 else retained / total),
                "omitted_variance_per_squared_root_mass": tail,
                "projection_kl_upper_bound_per_squared_root_mass": 0.5 * tail,
                "projection_tv_upper_bound_per_absolute_root_mass": 0.5 * math.sqrt(tail),
            }
        )
    stable_rank = 0.0 if not eigenvalues.size else total / float(eigenvalues[0])
    retained_total = float(np.sum(eigenvalues))
    positive_weights = eigenvalues / retained_total if retained_total > 0.0 else eigenvalues
    entropy_rank = (
        0.0
        if not positive_weights.size
        else float(np.exp(-np.sum(positive_weights * np.log(positive_weights))))
    )
    return {
        "positive_numerical_rank": spectrum.retained_rank,
        "algebraic_rank_ceiling": min(
            spectrum.observation_mean_design.size,
            native_cell_count - 1,
        ),
        "eigenvalue_tolerance": spectrum.eigenvalue_tolerance,
        "total_variance_per_squared_root_mass": total,
        "eigenvalues_sha256": _array_sha256(eigenvalues),
        "eigenvalues": eigenvalues.tolist(),
        "stable_rank": stable_rank,
        "retained_spectrum_entropy_effective_rank": entropy_rank,
        "variance_fraction_ranks": fraction_records,
        "diagnostic_ranks": rank_records,
    }


def _resource_estimates(
    *,
    observation_count: int,
    native_cell_count: int,
    numerical_rank: int,
    diagnostic_ranks: tuple[int, ...],
    source_sample_count: int,
    component_count: int,
) -> dict[str, Any]:
    """Estimate persistent arrays and known current Sobol temporary arrays.

    Args:
        observation_count: Number of observation rows.
        native_cell_count: Number of native simplex cells.
        numerical_rank: Complete retained spectrum rank.
        diagnostic_ranks: Candidate non-Gaussian mixture ranks.
        source_sample_count: Frozen direct-bank component count.
        component_count: Frozen compressed-mixture component count.

    Returns:
        JSON-compatible deterministic byte formulas. They distinguish the
        current full-rank source layout, a hypothetical projected-source
        layout, compressed persistent arrays, spectrum work arrays, and a
        lower bound for the current balanced-Sobol builder.

    These are deterministic array-byte calculations, not process peak RSS.
    The current balanced-tree Sobol builder materializes one
    ``samples x native_cells`` share array and a largest joint Sobol block.
    Active tree masses and library workspaces add further memory.
    """
    eight = np.dtype(np.float64).itemsize
    source_full_rank_bytes = eight * (
        source_sample_count * numerical_rank
        + 2 * observation_count
        + observation_count * numerical_rank
        + 2 * native_cell_count
        + 1
    )
    largest_sobol_dimension = min(
        max(0, native_cell_count - 1),
        SOBOL_DIMENSION_BLOCK_LIMIT,
    )
    share_bytes = eight * source_sample_count * native_cell_count
    uniform_block_bytes = eight * source_sample_count * largest_sobol_dimension
    spectrum_work_arrays = {
        "physical_mass_design_bytes": eight * observation_count * native_cell_count,
        "centered_operator_bytes": eight * observation_count * native_cell_count,
        "covariance_factor_bytes": eight * observation_count * native_cell_count,
        "dense_observation_covariance_bytes": eight * observation_count * observation_count,
        "full_spectrum_basis_bytes": eight * observation_count * numerical_rank,
    }
    records = []
    for requested_rank in dict.fromkeys((*diagnostic_ranks, numerical_rank)):
        rank = min(requested_rank, numerical_rank)
        source_projected_bytes = eight * (
            source_sample_count * rank
            + 2 * observation_count
            + observation_count * rank
            + 2 * native_cell_count
            + 1
        )
        compressed_bytes = eight * (
            2 * observation_count
            + observation_count * numerical_rank
            + numerical_rank
            + component_count
            + component_count * rank
            + component_count * rank * rank
            + component_count * rank
            + component_count * rank * rank
            + component_count
            + 3
        )
        records.append(
            {
                "requested_mixture_rank": requested_rank,
                "mixture_rank": rank,
                "optimized_projected_source_persistent_bytes": source_projected_bytes,
                "compressed_full_spectrum_persistent_bytes": compressed_bytes,
            }
        )
    return {
        "source_sample_count": source_sample_count,
        "component_count": component_count,
        "current_builder_full_rank_source_persistent_bytes": (source_full_rank_bytes),
        "current_sobol_share_array_bytes": share_bytes,
        "current_sobol_largest_uniform_block_dimension": (largest_sobol_dimension),
        "current_sobol_largest_uniform_block_bytes": uniform_block_bytes,
        "current_sobol_known_simultaneous_lower_bound_bytes": (
            share_bytes + uniform_block_bytes + source_full_rank_bytes
        ),
        "spectrum_construction_array_sizes": spectrum_work_arrays,
        "known_lower_bound_excludes": [
            "active balanced-tree mass arrays",
            "SciPy inverse-Beta temporaries",
            "clustering arrays",
            "loaded NetCDF arrays and physical-mass design ownership",
            "source summary-design arrays",
            "Python and numerical-library workspaces",
        ],
        "rank_scenarios": records,
    }


def _peak_rss_bytes() -> int:
    """Return the lifetime process high-water RSS in bytes.

    Returns:
        Process high-water memory up to this call using the platform's
        ``ru_maxrss`` convention. It includes interpreter, input, adapter,
        spectrum, and numerical-library allocations. It is neither
        spectrum-stage memory nor a substitute for Slurm job-step ``MaxRSS``.
    """
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else 1024 * raw


def run_probe(
    *,
    input_path: Path,
    expected_input_sha256: str,
    input_id: str,
    source_revision: str,
    concentration: float,
    expected_shape: tuple[int, int, int] = (
        PARIS_OBSERVATION_COUNT,
        *PARIS_GRID_SHAPE,
    ),
    expected_outer_labels: tuple[str, ...] = PARIS_OUTER_LABELS,
    expected_schema: str | None = PARIS_INPUT_SCHEMA,
    variance_fractions: Sequence[float] = DEFAULT_VARIANCE_FRACTIONS,
    diagnostic_ranks: Sequence[int] = DEFAULT_RANKS,
    source_sample_count: int = DEFAULT_SOURCE_SAMPLE_COUNT,
    component_count: int = DEFAULT_COMPONENT_COUNT,
    engine: str | None = None,
    include_timings: bool = False,
) -> dict[str, Any]:
    r"""Build one authenticated observation-blind PARIS root-spectrum report.

    Args:
        input_path: Frozen NetCDF input. It must be a real regular file.
        expected_input_sha256: Required whole-file SHA-256.
        input_id: Non-empty caller-owned frozen-input identifier.
        source_revision: Complete Git SHA of this driver checkout.
        concentration: Dimensionless common global additive Dirichlet total
            \(\eta\).
        expected_shape: Exact ``(nmeasure, lat, lon)`` profile.
        expected_outer_labels: Exact ordered outer-region labels.
        expected_schema: Required dataset ``schema_id`` or ``None``.
        variance_fractions: Predeclared cumulative trace fractions.
        diagnostic_ranks: Predeclared projection/mixture rank scenarios.
        source_sample_count: Source-bank size used only for byte estimates.
        component_count: Compressed component count used only for estimates.
        engine: Optional xarray NetCDF backend.
        include_timings: Include volatile wall times and self peak RSS.

    Returns:
        Strictly JSON-compatible report containing input identities, closure,
        the complete analytic eigenvalue spectrum, rank/tail diagnostics, and
        deterministic resource estimates. The physical-mass design has shape
        ``(nmeasure, lat * lon)``; eigenvalues have squared noise-whitened
        observation units per squared root mass.

    Raises:
        FileNotFoundError: If the frozen input does not exist.
        TypeError: If scalar controls have invalid types.
        ValueError: If identity, profile, numerical, or closure checks fail.
        RuntimeError: If a validated adapter, bridge, or eigensolver
            dependency cannot complete.

    Notes:
        The function eagerly reads the explicit frozen variables and can
        allocate several arrays the size of ``H`` plus a dense
        observation-space covariance. Reported process RSS is the lifetime
        high-water mark, not incremental spectrum-only memory. It writes
        nothing.
    """
    revision = _full_revision(source_revision)
    expected_digest = _sha256(
        expected_input_sha256,
        name="expected_input_sha256",
    )
    eta = _positive_float(concentration, name="concentration")
    if not isinstance(input_id, str) or not input_id.strip() or input_id != input_id.strip():
        raise ValueError("input_id must be a non-empty stripped string")
    fractions = _validated_fractions(variance_fractions)
    ranks = _validated_ranks(diagnostic_ranks)
    source_count = _positive_integer(
        source_sample_count,
        name="source_sample_count",
    )
    components = _positive_integer(component_count, name="component_count")
    if source_count & (source_count - 1):
        raise ValueError("source_sample_count must be a power of two")
    if components > source_count:
        raise ValueError("component_count cannot exceed source_sample_count")
    if len(expected_shape) != 3 or any(
        isinstance(value, bool) or int(value) != value or int(value) < 1 for value in expected_shape
    ):
        raise ValueError("expected_shape must contain three positive integers")
    normalized_shape = (
        int(expected_shape[0]),
        int(expected_shape[1]),
        int(expected_shape[2]),
    )
    if not expected_outer_labels or len(set(expected_outer_labels)) != len(expected_outer_labels):
        raise ValueError("expected_outer_labels must be non-empty and unique")
    path = Path(input_path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("input_path must be a real regular file")

    started = time.perf_counter()
    before_digest = _sha256_file(path)
    if before_digest != expected_digest:
        raise ValueError("frozen input SHA-256 does not match expected_input_sha256")
    hash_before_seconds = time.perf_counter() - started
    load_started = time.perf_counter()
    dataset = _load_frozen_subset(path, engine=engine)
    load_seconds = time.perf_counter() - load_started
    after_digest = _sha256_file(path)
    if after_digest != before_digest:
        raise ValueError("frozen input changed while it was being loaded")
    _require_profile(
        dataset,
        expected_shape=normalized_shape,
        expected_outer_labels=expected_outer_labels,
        expected_schema=expected_schema,
    )

    adapter_started = time.perf_counter()
    adapter = gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=dataset["nominal_weight"],
        k_min=1,
        k_max=1,
        concentration=eta,
        root_variance=0.25,
        normalize_weights=True,
        likelihood_power=0.0,
        sensitivity_name="fp_x_flux",
        observation_name="mf",
        observation_sd_name="mf_error",
        fixed_design_name="outer_design",
        fixed_offset_name="YaprioriBC",
        fixed_coefficient_prior_mean=1.0,
        fixed_coefficient_prior_sd=1.0,
    )
    problem = full_tiling_problem_from_gamma_beta_adapter(
        adapter,
        concentration=eta,
    )
    aggregation = aggregation_from_full_tiling_problem(
        problem,
        np.empty((normalized_shape[0], 0), dtype=np.float64),
    )
    adapter_seconds = time.perf_counter() - adapter_started

    raw_scaling = np.asarray(
        dataset["fp_x_flux"].transpose("nmeasure", "lat", "lon").values,
        dtype=np.float64,
    ).sum(axis=(1, 2))
    physical_mean = aggregation.design @ problem.normalized_nominal_mass.reshape(-1)
    closure_error = np.asarray(physical_mean - raw_scaling, dtype=np.float64)
    closure_scale = max(1.0, float(np.max(np.abs(raw_scaling), initial=0.0)))
    closure_tolerance = float(
        512.0 * np.finfo(np.float64).eps * max(1, problem.normalized_nominal_mass.size) * closure_scale
    )
    if float(np.max(np.abs(closure_error), initial=0.0)) > closure_tolerance:
        raise ValueError("physical-mass conversion failed the prior-mean closure audit")

    spectrum_started = time.perf_counter()
    spectrum = RootResidualSpectrum.from_aggregation(
        aggregation,
        retained_variance_fraction=1.0,
    )
    spectrum_seconds = time.perf_counter() - spectrum_started
    spectrum_report = _spectrum_diagnostics(
        spectrum,
        native_cell_count=normalized_shape[1] * normalized_shape[2],
        fractions=fractions,
        diagnostic_ranks=ranks,
    )
    resource_report = _resource_estimates(
        observation_count=normalized_shape[0],
        native_cell_count=normalized_shape[1] * normalized_shape[2],
        numerical_rank=spectrum.retained_rank,
        diagnostic_ranks=ranks,
        source_sample_count=source_count,
        component_count=components,
    )
    variable_hashes = {
        name: _array_sha256(np.asarray(dataset[name].values))
        for name in (
            "fp_x_flux",
            "mf",
            "mf_error",
            "nominal_weight",
            "outer_design",
            "YaprioriBC",
        )
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "source_revision": revision,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "xarray": xr.__version__,
        },
        "input": {
            "path": str(path),
            "input_id": input_id,
            "sha256": before_digest,
            "schema_id": dataset.attrs.get("schema_id"),
            "shape": {
                "observations": normalized_shape[0],
                "latitudes": normalized_shape[1],
                "longitudes": normalized_shape[2],
                "native_cells": normalized_shape[1] * normalized_shape[2],
                "outer_regions": len(expected_outer_labels),
            },
            "outer_labels": list(expected_outer_labels),
            "variable_sha256": variable_hashes,
        },
        "model": {
            "coordinate": "physical_native_cell_mass",
            "unit_scaling_to_mass_conversion": "fp_x_flux / normalized_nominal_weight",
            "native_flattening_order": "C",
            "concentration": eta,
            "normalized_nominal_weight_sum": float(np.sum(problem.normalized_nominal_mass)),
            "cell_alpha_sum": float(np.sum(aggregation.cell_alphas)),
            "cell_alphas_sha256": spectrum.cell_alphas_sha256,
            "physical_mass_design_sha256": spectrum.design_sha256,
            "noise_sd_sha256": spectrum.noise_sd_sha256,
            "whitening_covariance": "fixed diagonal mf_error squared",
        },
        "closure": {
            "maximum_absolute_error": float(np.max(np.abs(closure_error), initial=0.0)),
            "rmse": float(np.sqrt(np.mean(np.square(closure_error)))),
            "tolerance": closure_tolerance,
            "passed": True,
        },
        "spectrum": spectrum_report,
        "resources": resource_report,
        "observed_residual_used_for_spectrum_selection": False,
        "partition_used_for_spectrum_selection": False,
        "k_used_for_spectrum_selection": False,
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "structural_inference_licensed": False,
        "paris_posterior_inference_performed": False,
        "interpretation": {
            "exact": "continuous Gaussian location-mixture law and analytic conditional covariance spectrum",
            "approximate": [
                "any retained-rank projection",
                "any finite Sobol source bank",
                "any Gaussian complement for non-Gaussian directions",
                "any moment-preserving component compression",
            ],
            "next_gate": "source-bank construction resource design before any PARIS conditional posterior",
        },
    }
    report["protocol_sha256"] = hashlib.sha256(
        _canonical_json(
            {
                "schema": SCHEMA,
                "protocol": PROTOCOL,
                "expected_shape": list(normalized_shape),
                "expected_outer_labels": list(expected_outer_labels),
                "expected_schema": expected_schema,
                "concentration": eta,
                "variance_fractions": list(fractions),
                "diagnostic_ranks": list(ranks),
                "source_sample_count": source_count,
                "component_count": components,
            }
        ).encode("ascii")
    ).hexdigest()
    if include_timings:
        report["timings"] = {
            "hash_before_seconds": hash_before_seconds,
            "load_seconds": load_seconds,
            "adapter_and_bridge_seconds": adapter_seconds,
            "spectrum_seconds": spectrum_seconds,
            "total_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "process_peak_rss_is_authoritative": False,
            "authoritative_peak_rss_source": "Slurm sacct MaxRSS",
        }
    return report


def _validate_output_path(path: Path) -> None:
    """Validate a create-only non-production report path before computation.

    Args:
        path: Intended report path.

    Raises:
        ValueError: If the resolved path enters ``PARIS_inversions`` or the
            parent is not a real existing directory.
        FileExistsError: If the target already exists or is a symlink.
    """
    resolved = path.resolve(strict=False)
    if any(part.casefold() == "paris_inversions" for part in resolved.parts):
        raise ValueError("output must not be written under PARIS_inversions")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace output: {path}")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("output parent must be a real existing directory")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Create one canonical JSON report atomically without replacement.

    Args:
        path: Previously validated create-only destination.
        payload: Strictly JSON-compatible report.

    Raises:
        ValueError: If the resolved destination is a production path or its
            parent is invalid.
        FileExistsError: If the destination already exists.
        OSError: If temporary creation, synchronization, or hard linking
            fails.

    Notes:
        Creates and fsyncs one temporary file in the destination directory,
        hard-links it create-only to the final name, and removes the
        temporary file.
    """
    _validate_output_path(path)
    parent = path.parent
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            stream.write(_canonical_json(payload))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_floats(value: str) -> tuple[float, ...]:
    """Parse a comma-separated float tuple."""
    try:
        return tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error


def _csv_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer tuple."""
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def _parser() -> argparse.ArgumentParser:
    """Build the PARIS resource-probe CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--input-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--concentration", type=float, required=True)
    parser.add_argument("--variance-fractions", type=_csv_floats, default=DEFAULT_VARIANCE_FRACTIONS)
    parser.add_argument("--diagnostic-ranks", type=_csv_ints, default=DEFAULT_RANKS)
    parser.add_argument("--source-sample-count", type=int, default=DEFAULT_SOURCE_SAMPLE_COUNT)
    parser.add_argument("--component-count", type=int, default=DEFAULT_COMPONENT_COUNT)
    parser.add_argument("--engine")
    parser.add_argument("--include-timings", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the authenticated probe and create one canonical report.

    Args:
        argv: Optional CLI arguments. ``None`` reads process arguments.

    Returns:
        Zero after successful create-only publication.

    Raises:
        SystemExit: If argument parsing fails.
        FileNotFoundError: If the input is absent.
        TypeError: If a scientific scalar has an invalid type.
        ValueError: If identity, profile, numerical, closure, or output-policy
            validation fails.
        FileExistsError: If the output already exists.
        OSError: If input or output I/O fails.

    Notes:
        The only write side effect is one create-only canonical JSON report.
    """
    arguments = _parser().parse_args(argv)
    _validate_output_path(arguments.output)
    report = run_probe(
        input_path=arguments.input,
        expected_input_sha256=arguments.expected_input_sha256,
        input_id=arguments.input_id,
        source_revision=arguments.source_revision,
        concentration=arguments.concentration,
        variance_fractions=arguments.variance_fractions,
        diagnostic_ranks=arguments.diagnostic_ranks,
        source_sample_count=arguments.source_sample_count,
        component_count=arguments.component_count,
        engine=arguments.engine,
        include_timings=arguments.include_timings,
    )
    _atomic_write(arguments.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
