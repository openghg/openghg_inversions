#!/usr/bin/env python3
r"""Run the staged BP1 chunked projected-bank engineering gates.

This driver constructs only the single-root Gamma--Dirichlet marginal model.
The realized PARIS residual is authenticated with the frozen input but is not
used to select the analytic spectrum, projection rank, allocation chunk, or
projection microbatch.  The leading ``q`` residual coordinates are stored as
one fixed scrambled-Sobol bank; directions ``q+1:r`` remain the analytic
Gaussian moment-closure complement.

The driver publishes create-only canonical manifests and little-endian binary
arrays.  It refuses output paths under ``PARIS_inversions``.  It does not
cluster, run a posterior, inspect a protected catalogue, or license structural
weights from approximation differences.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
import scipy
import xarray as xr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import conditional_residual_image_exact_mixture_paris_probe as probe
from openghg_inversions.experimental.rjmcmc import (
    aggregation_error_conditional_mixture as conditional_mixture_module,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (
    ConditionalAllocationMixture,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_exact_mixture import (
    RootResidualSpectrum,
    build_chunked_projected_root_bank,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
    aggregation_from_full_tiling_problem,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    full_tiling_problem_from_gamma_beta_adapter,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    gamma_beta_problem_from_rhime_inputs,
)

FloatArray = NDArray[np.float64]


class SlurmRecord(TypedDict):
    """Terminal resource fields for one Slurm job."""

    state: str | None
    elapsed_seconds: int | None
    max_rss_bytes: int


SCHEMA = "rjmcmc-chunked-projected-bank-hpc-v1"
PARIS_INPUT_SCHEMA = "paris-may-2014-gamma-beta-native-v1"
PARIS_OBSERVATION_COUNT = 1_382
PARIS_GRID_SHAPE = (183, 128)
PARIS_OUTER_LABELS = tuple(f"intem_label_{index}" for index in range(6))
PARIS_INPUT_SHA256 = "24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044"
SCIENCE_CALIBRATION_SCHEMA = "rjmcmc-modeled-european-domain-gbr-cv-calibration-v1"
MODELED_EUROPEAN_DOMAIN_TARGET_CV = 0.2
GBR_TARGET_CV = 0.5
SCIENTIFIC_ROOT_VARIANCE = 0.022861001527515423
SCIENTIFIC_CONCENTRATION = 528.618161317525
SOURCE_SEEDS = (731, 1_877, 4_099, 8_317)
SOURCE_SAMPLE_COUNT = 65_536
SOURCE_RANK = 128
PREFIX_SAMPLE_COUNT = 256
PREFIX_RANK = 32
P_LADDER = (64, 128, 256)
PREFIX_C_LADDER = (64, 128, 256)
RESOURCE_C_LADDER = (1_024, 2_048, 4_096, 8_192)
PARITY_FACTOR = 32.0


def _canonical_json(value: object) -> str:
    """Return strict canonical ASCII JSON."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _full_revision(value: str) -> str:
    """Validate and return one full lower-case Git revision."""
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("source revision must be a 40-character lower-case Git SHA")
    return value


def _sha256_file(path: Path) -> str:
    """Return the raw SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: NDArray[Any]) -> str:
    """Return a shape-sensitive little-endian numeric array digest."""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        canonical = np.ascontiguousarray(array, dtype="<f8")
    elif np.issubdtype(array.dtype, np.integer):
        canonical = np.ascontiguousarray(array, dtype="<i8")
    else:
        raise TypeError("only floating and integer arrays can be fingerprinted")
    header = _canonical_json({"dtype": canonical.dtype.str, "shape": list(canonical.shape)})
    digest = hashlib.sha256(header.encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _validate_output(path: Path) -> None:
    """Require a create-only non-production output with a real parent."""
    resolved = path.resolve(strict=False)
    if any(part.casefold() == "paris_inversions" for part in resolved.parts):
        raise ValueError("output must not be written under PARIS_inversions")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace output: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("output parent must be a real existing directory")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publish canonical JSON plus one newline without replacement."""
    _validate_output(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish ASCII text without replacement."""
    _validate_output(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_npy(path: Path, values: NDArray[Any]) -> None:
    """Publish one uncompressed non-pickle NumPy array without replacement."""
    _validate_output(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.lib.format.write_array(
                stream,
                np.ascontiguousarray(values),
                allow_pickle=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one canonical, newline-terminated, non-symlink JSON object."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"manifest must be a real regular file: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON manifest: {path}") from error
    if not isinstance(payload, dict) or raw != (_canonical_json(payload) + "\n").encode("ascii"):
        raise ValueError(f"manifest is not canonical JSON plus one newline: {path}")
    return payload


def _read_npy(path: Path, *, expected_sha256: str) -> NDArray[Any]:
    """Read and authenticate one real non-pickle NumPy array."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"array must be a real regular file: {path}")
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"array file SHA-256 mismatch: {path}")
    result = np.load(path, allow_pickle=False)
    if not isinstance(result, np.ndarray):
        raise ValueError(f"array payload is not an ndarray: {path}")
    return result


def _peak_rss_bytes() -> int:
    """Return the current process lifetime high-water RSS in bytes."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1_024


def _runtime() -> dict[str, object]:
    """Return the numerical runtime identity used by every stage."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "xarray": xr.__version__,
    }


def _positive_power_of_two(value: object, *, name: str) -> int:
    """Validate one strictly positive power-of-two integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1 or result & (result - 1):
        raise ValueError(f"{name} must be a positive power of two")
    return result


def _parity_tolerance(reference: FloatArray, *, native_cells: int) -> float:
    """Return the frozen scale-aware float64 v2/v3 parity tolerance."""
    scale = max(1.0, float(np.max(np.abs(reference), initial=0.0)))
    return float(PARITY_FACTOR * np.finfo(np.float64).eps * max(1, native_cells) * scale)


def _maximum_ulp_difference(first: FloatArray, second: FloatArray) -> int:
    """Return the largest representable-float distance between finite arrays."""
    if first.shape != second.shape:
        raise ValueError("ULP arrays must have the same shape")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("ULP arrays must be finite")
    first_bits = np.ascontiguousarray(first, dtype="<f8").view("<u8")
    second_bits = np.ascontiguousarray(second, dtype="<f8").view("<u8")
    sign = np.uint64(1 << 63)
    first_ordered = np.where(
        (first_bits & sign) != 0,
        ~first_bits,
        first_bits | sign,
    )
    second_ordered = np.where(
        (second_bits & sign) != 0,
        ~second_bits,
        second_bits | sign,
    )
    difference = np.maximum(first_ordered, second_ordered) - np.minimum(first_ordered, second_ordered)
    return int(np.max(difference, initial=np.uint64(0)))


def _aggregate_variance_factor(
    nominal_weight: FloatArray,
    physical_weight: FloatArray,
) -> tuple[float, float]:
    """Return the Dirichlet variance factor and physical aggregate total."""
    if (
        nominal_weight.shape != physical_weight.shape
        or not np.all(np.isfinite(nominal_weight))
        or not np.all(nominal_weight > 0.0)
        or not np.all(np.isfinite(physical_weight))
        or np.any(physical_weight < 0.0)
    ):
        raise ValueError("aggregate calibration weights are invalid")
    total = float(np.sum(physical_weight))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("physical aggregate total must be finite and positive")
    factor = float(
        np.sum(
            np.square(physical_weight) / (nominal_weight * total * total),
        )
        - 1.0
    )
    if not math.isfinite(factor) or factor < 0.0:
        raise ValueError("aggregate Dirichlet variance factor is invalid")
    return factor, total


def _solve_two_aggregate_calibration(
    *,
    broad_factor: float,
    local_factor: float,
    broad_target_cv: float,
    local_target_cv: float,
) -> tuple[float, float]:
    """Solve root variance and common concentration from two aggregate CVs."""
    if not (
        math.isfinite(broad_factor)
        and math.isfinite(local_factor)
        and 0.0 <= broad_factor < local_factor
        and math.isfinite(broad_target_cv)
        and math.isfinite(local_target_cv)
        and 0.0 < broad_target_cv < local_target_cv
    ):
        raise ValueError("two-aggregate calibration inputs are invalid")
    broad_variance = broad_target_cv * broad_target_cv
    local_variance = local_target_cv * local_target_cv
    root_variance = broad_variance - (
        broad_factor * (local_variance - broad_variance) / (local_factor - broad_factor)
    )
    if not 0.0 <= root_variance < broad_variance:
        raise ValueError("two-aggregate targets imply no non-negative finite root variance")
    concentration = (1.0 + root_variance) * broad_factor / (broad_variance - root_variance) - 1.0
    if not math.isfinite(concentration) or concentration <= 0.0:
        raise ValueError("two-aggregate targets imply no finite positive concentration")
    return root_variance, concentration


def _scientific_calibration(dataset: xr.Dataset) -> dict[str, object]:
    """Authenticate the observation-blind modeled-domain/GBR prior calibration."""
    required_dims = {
        "nominal_weight": ("lat", "lon"),
        "prior_flux": ("lat", "lon"),
        "grid_cell_area": ("lat", "lon"),
        "country_fraction": ("country", "lat", "lon"),
    }
    for name, dims in required_dims.items():
        if name not in dataset or dataset[name].dims != dims:
            raise ValueError(f"frozen calibration variable {name!r} has wrong dimensions")
    try:
        nominal_data, flux_data, area_data, gbr_data = xr.align(
            dataset["nominal_weight"],
            dataset["prior_flux"],
            dataset["grid_cell_area"],
            dataset["country_fraction"].sel(country="GBR"),
            join="exact",
            copy=False,
        )
    except (KeyError, ValueError) as error:
        raise ValueError("frozen input has no exactly aligned GBR calibration fields") from error
    nominal_weight = np.asarray(nominal_data.values, dtype=np.float64)
    prior_flux = np.asarray(flux_data.values, dtype=np.float64)
    grid_cell_area = np.asarray(area_data.values, dtype=np.float64)
    gbr_fraction = np.asarray(gbr_data.values, dtype=np.float64)
    if not np.all(np.isfinite(gbr_fraction)) or np.any(gbr_fraction < 0.0) or np.any(gbr_fraction > 1.0):
        raise ValueError("frozen GBR country fractions are invalid")
    if not math.isclose(
        float(np.sum(nominal_weight)),
        1.0,
        rel_tol=0.0,
        abs_tol=128.0 * np.finfo(np.float64).eps * nominal_weight.size,
    ):
        raise ValueError("frozen nominal weights are not normalized")
    domain_weight = prior_flux * grid_cell_area
    gbr_weight = domain_weight * gbr_fraction
    domain_factor, domain_total = _aggregate_variance_factor(
        nominal_weight,
        domain_weight,
    )
    gbr_factor, gbr_total = _aggregate_variance_factor(
        nominal_weight,
        gbr_weight,
    )
    root_variance, concentration = _solve_two_aggregate_calibration(
        broad_factor=domain_factor,
        local_factor=gbr_factor,
        broad_target_cv=MODELED_EUROPEAN_DOMAIN_TARGET_CV,
        local_target_cv=GBR_TARGET_CV,
    )
    tolerance = 64.0 * np.finfo(np.float64).eps
    if not math.isclose(
        root_variance,
        SCIENTIFIC_ROOT_VARIANCE,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise ValueError("frozen input no longer reproduces the scientific root variance")
    if not math.isclose(
        concentration,
        SCIENTIFIC_CONCENTRATION,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise ValueError("frozen input no longer reproduces the scientific concentration")

    def achieved_cv(factor: float) -> float:
        return math.sqrt(root_variance + (1.0 + root_variance) * factor / (concentration + 1.0))

    return {
        "schema": SCIENCE_CALIBRATION_SCHEMA,
        "aggregate_definition": (
            "sum(prior_flux*grid_cell_area*country_fraction*simulated_scaling); "
            "country_fraction=1 for the complete modeled European inner domain"
        ),
        "modeled_european_domain": {
            "target_cv": MODELED_EUROPEAN_DOMAIN_TARGET_CV,
            "achieved_cv": achieved_cv(domain_factor),
            "dirichlet_variance_factor": domain_factor,
            "prior_total_mol_s": domain_total,
            "physical_weight_sha256": _array_sha256(domain_weight),
        },
        "GBR": {
            "target_cv": GBR_TARGET_CV,
            "achieved_cv": achieved_cv(gbr_factor),
            "dirichlet_variance_factor": gbr_factor,
            "prior_total_mol_s": gbr_total,
            "physical_weight_sha256": _array_sha256(gbr_weight),
            "country_fraction_sha256": _array_sha256(gbr_fraction),
        },
        "nominal_weight_sha256": _array_sha256(nominal_weight),
        "prior_flux_sha256": _array_sha256(prior_flux),
        "grid_cell_area_sha256": _array_sha256(grid_cell_area),
        "country_labels_sha256": hashlib.sha256(
            _canonical_json([str(value) for value in dataset["country"].values]).encode("ascii")
        ).hexdigest(),
        "root_variance": root_variance,
        "root_cv": math.sqrt(root_variance),
        "root_gamma_shape": 1.0 / root_variance,
        "root_gamma_rate": 1.0 / root_variance,
        "common_native_concentration": concentration,
        "observed_mole_fraction_used": False,
        "modeled_domain_not_political_eu_membership": True,
    }


def _load_scientific_calibration_fields(
    input_path: Path,
    *,
    expected_input_sha256: str,
) -> xr.Dataset:
    """Load only the authenticated prior fields used by the CV calibration."""
    if input_path.is_symlink() or not input_path.is_file():
        raise FileNotFoundError("input must be a real regular file")
    before = _sha256_file(input_path)
    if before != expected_input_sha256:
        raise ValueError("frozen input SHA-256 mismatch")
    names = (
        "nominal_weight",
        "prior_flux",
        "grid_cell_area",
        "country_fraction",
    )
    with xr.open_dataset(input_path) as opened:
        missing = sorted(set(names).difference(opened.data_vars))
        if missing:
            raise ValueError(f"frozen input is missing calibration variables: {missing}")
        dataset = opened[list(names)].load()
    if _sha256_file(input_path) != before:
        raise ValueError("frozen input changed while calibration fields were read")
    return dataset


def _build_paris_aggregation(
    input_path: Path,
    *,
    expected_input_sha256: str,
    concentration: float,
) -> tuple[xr.Dataset, AdditiveDirichletAggregation]:
    """Authenticate the frozen input and build the one-root physical model."""
    if not math.isclose(
        concentration,
        SCIENTIFIC_CONCENTRATION,
        rel_tol=0.0,
        abs_tol=64.0 * np.finfo(np.float64).eps,
    ):
        raise ValueError("PARIS aggregation concentration must match the scientific lock")
    if input_path.is_symlink() or not input_path.is_file():
        raise FileNotFoundError("input must be a real regular file")
    before = _sha256_file(input_path)
    if before != expected_input_sha256:
        raise ValueError("frozen input SHA-256 mismatch")
    dataset = probe._load_frozen_subset(input_path, engine=None)
    if _sha256_file(input_path) != before:
        raise ValueError("frozen input changed while being read")
    probe._require_profile(
        dataset,
        expected_shape=(PARIS_OBSERVATION_COUNT, *PARIS_GRID_SHAPE),
        expected_outer_labels=PARIS_OUTER_LABELS,
        expected_schema=PARIS_INPUT_SCHEMA,
    )
    adapter = gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=dataset["nominal_weight"],
        k_min=1,
        k_max=1,
        concentration=concentration,
        root_variance=SCIENTIFIC_ROOT_VARIANCE,
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
        concentration=concentration,
    )
    aggregation = aggregation_from_full_tiling_problem(
        problem,
        np.empty((PARIS_OBSERVATION_COUNT, 0), dtype=np.float64),
    )
    return dataset, aggregation


def run_scientific_calibration(
    output: Path,
    *,
    input_path: Path,
    expected_input_sha256: str,
    source_revision: str,
) -> dict[str, object]:
    """Recalculate and publish the observation-blind two-aggregate prior lock."""
    calibration = _scientific_calibration(
        _load_scientific_calibration_fields(
            input_path,
            expected_input_sha256=expected_input_sha256,
        )
    )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "stage": "scientific-calibration",
        "source_revision": source_revision,
        "input_sha256": expected_input_sha256,
        "calibration": calibration,
        "realized_mf_used": False,
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "passed": True,
    }
    _atomic_write_json(output, report)
    return report


def _spectrum_scalars(spectrum: RootResidualSpectrum) -> dict[str, object]:
    """Return complete scalar spectrum metadata."""
    return {
        "total_variance": spectrum.total_variance,
        "discarded_variance": spectrum.discarded_variance,
        "requested_retained_variance_fraction": (spectrum.requested_retained_variance_fraction),
        "retained_variance_fraction": spectrum.retained_variance_fraction,
        "eigenvalue_tolerance": spectrum.eigenvalue_tolerance,
        "retained_rank": spectrum.retained_rank,
        "cell_alphas_sha256": spectrum.cell_alphas_sha256,
        "design_sha256": spectrum.design_sha256,
        "noise_sd_sha256": spectrum.noise_sd_sha256,
        "canonical_eigenvector_sign_rule": ("largest-absolute-entry-in-each-column-is-nonnegative"),
    }


def _write_spectrum_bundle(
    output_dir: Path,
    spectrum: RootResidualSpectrum,
    *,
    source_revision: str,
    input_path: Path,
    input_sha256: str,
    concentration: float,
    elapsed_seconds: float,
    scientific_calibration: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Publish the authoritative spectrum arrays followed by its manifest."""
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("spectrum output directory must be a real directory")
    arrays: dict[str, FloatArray] = {
        "observation_mean_design": spectrum.observation_mean_design,
        "noise_sd": spectrum.noise_sd,
        "basis": spectrum.basis,
        "eigenvalues": spectrum.eigenvalues,
    }
    records: dict[str, object] = {}
    for name, values in arrays.items():
        path = output_dir / f"{name}.npy"
        little_endian = np.ascontiguousarray(values, dtype="<f8")
        _atomic_write_npy(path, little_endian)
        records[name] = {
            "file": path.name,
            "dtype": "<f8",
            "shape": list(little_endian.shape),
            "array_sha256": _array_sha256(little_endian),
            "file_sha256": _sha256_file(path),
        }
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "stage": "G2",
        "source_revision": source_revision,
        "input": {
            "path": str(input_path),
            "sha256": input_sha256,
            "schema": PARIS_INPUT_SCHEMA,
        },
        "model": {
            "retained_roots": 1,
            "native_concentration": concentration,
            "root_variance": SCIENTIFIC_ROOT_VARIANCE,
            "concentration_status": (
                "scientific-modeled-domain-and-GBR-aggregate-CV-lock"
                if scientific_calibration is not None
                else "synthetic-test-or-unlocked"
            ),
            "scientific_calibration": scientific_calibration,
            "observed_residual_used": False,
            "partition_or_k_used": False,
        },
        "spectrum": _spectrum_scalars(spectrum),
        "arrays": records,
        "runtime": _runtime(),
        "elapsed_seconds": elapsed_seconds,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "protected_catalogue_accessed": False,
        "production_output_written": False,
    }
    _atomic_write_json(output_dir / "spectrum_manifest.json", manifest)
    return manifest


def _load_spectrum_bundle(manifest_path: Path) -> RootResidualSpectrum:
    """Load and strictly authenticate one authoritative spectrum bundle."""
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != SCHEMA or manifest.get("stage") != "G2":
        raise ValueError("spectrum manifest schema or stage mismatch")
    records = manifest.get("arrays")
    scalars = manifest.get("spectrum")
    if not isinstance(records, dict) or not isinstance(scalars, dict):
        raise ValueError("spectrum manifest is missing arrays or scalars")
    arrays: dict[str, FloatArray] = {}
    for name in ("observation_mean_design", "noise_sd", "basis", "eigenvalues"):
        record = records.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"missing spectrum array record: {name}")
        filename = record.get("file")
        file_sha256 = record.get("file_sha256")
        if not isinstance(filename, str) or not isinstance(file_sha256, str):
            raise ValueError(f"invalid spectrum array record: {name}")
        values = _read_npy(
            manifest_path.parent / filename,
            expected_sha256=file_sha256,
        )
        if values.dtype != np.dtype("<f8") or list(values.shape) != record.get("shape"):
            raise ValueError(f"spectrum array dtype or shape mismatch: {name}")
        if _array_sha256(values) != record.get("array_sha256"):
            raise ValueError(f"spectrum array value digest mismatch: {name}")
        arrays[name] = np.asarray(values, dtype=np.float64)
    spectrum = RootResidualSpectrum(
        arrays["observation_mean_design"],
        arrays["noise_sd"],
        arrays["basis"],
        arrays["eigenvalues"],
        total_variance=float(scalars["total_variance"]),
        discarded_variance=float(scalars["discarded_variance"]),
        requested_retained_variance_fraction=float(scalars["requested_retained_variance_fraction"]),
        eigenvalue_tolerance=float(scalars["eigenvalue_tolerance"]),
        cell_alphas_sha256=str(scalars["cell_alphas_sha256"]),
        design_sha256=str(scalars["design_sha256"]),
        noise_sd_sha256=str(scalars["noise_sd_sha256"]),
    )
    if _spectrum_scalars(spectrum) != scalars:
        raise ValueError("replayed spectrum scalars do not match manifest")
    return spectrum


def _authenticate_scientific_spectrum_manifest(
    manifest_path: Path,
    *,
    source_revision: str,
    input_sha256: str,
    concentration: float,
) -> dict[str, Any]:
    """Require a spectrum built from the exact committed scientific calibration."""
    manifest = _read_json(manifest_path)
    model = manifest.get("model")
    input_record = manifest.get("input")
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("stage") != "G2"
        or manifest.get("source_revision") != source_revision
        or not isinstance(model, dict)
        or not isinstance(input_record, dict)
        or input_record.get("sha256") != input_sha256
        or model.get("concentration_status") != "scientific-modeled-domain-and-GBR-aggregate-CV-lock"
        or model.get("root_variance") != SCIENTIFIC_ROOT_VARIANCE
        or model.get("native_concentration") != concentration
    ):
        raise ValueError("spectrum manifest does not match the scientific model lock")
    calibration = model.get("scientific_calibration")
    if (
        not isinstance(calibration, dict)
        or calibration.get("schema") != SCIENCE_CALIBRATION_SCHEMA
        or not isinstance(calibration.get("root_variance"), Real)
        or not math.isclose(
            float(calibration["root_variance"]),
            SCIENTIFIC_ROOT_VARIANCE,
            rel_tol=64.0 * np.finfo(np.float64).eps,
            abs_tol=64.0 * np.finfo(np.float64).eps,
        )
        or not isinstance(calibration.get("common_native_concentration"), Real)
        or not math.isclose(
            float(calibration["common_native_concentration"]),
            SCIENTIFIC_CONCENTRATION,
            rel_tol=64.0 * np.finfo(np.float64).eps,
            abs_tol=64.0 * np.finfo(np.float64).eps,
        )
    ):
        raise ValueError("spectrum manifest scientific calibration is missing or invalid")
    return manifest


def _all_at_once_source(
    aggregation: AdditiveDirichletAggregation,
    spectrum: RootResidualSpectrum,
    *,
    rank: int,
    sample_count: int,
    source_seed: int,
    source_provenance: str,
) -> ConditionalAllocationMixture:
    """Build the v2 all-at-once catalogue in a leading spectrum basis."""
    projected = AdditiveDirichletAggregation(
        aggregation.cell_alphas,
        aggregation.design,
        aggregation.noise_sd,
        spectrum.basis[:, :rank],
    )
    return ConditionalAllocationMixture.from_aggregation(
        projected,
        np.zeros(aggregation.cell_shape, dtype=np.int64),
        sample_count=sample_count,
        source_seed=source_seed,
        source_provenance=source_provenance,
        construction_method="scrambled_sobol_balanced_dirichlet",
    )


def _sobol_metadata(bank: ConditionalAllocationMixture) -> dict[str, object]:
    """Return compact Sobol catalogue metadata without expanding bank arrays."""
    node_count = conditional_mixture_module._sobol_dimension_count(bank.labels)
    return {
        "sobol_catalogue_sha256": (
            conditional_mixture_module._sobol_catalogue_sha256(
                bank.labels,
                bank.cell_ids,
            )
        ),
        "sobol_block_dimensions": (conditional_mixture_module._sobol_block_dimensions(node_count)),
    }


def _parity_record(
    reference: FloatArray,
    candidate: FloatArray,
    *,
    native_cells: int,
) -> dict[str, object]:
    """Return and enforce the predeclared v2/v3 float64 parity gate."""
    if reference.shape != candidate.shape:
        raise ValueError("v2/v3 projected array shapes differ")
    maximum_absolute = float(np.max(np.abs(candidate - reference), initial=0.0))
    tolerance = _parity_tolerance(reference, native_cells=native_cells)
    passed = maximum_absolute <= tolerance
    if not passed:
        raise ValueError(f"v2/v3 parity failed: {maximum_absolute} exceeds {tolerance}")
    return {
        "maximum_absolute_difference": maximum_absolute,
        "maximum_ulp_difference": _maximum_ulp_difference(reference, candidate),
        "absolute_tolerance": tolerance,
        "tolerance_formula": ("32*float64_epsilon*native_cell_count*max(1,max_abs(v2))"),
        "passed": True,
    }


def run_tiny(
    output_dir: Path,
    *,
    source_revision: str,
) -> dict[str, object]:
    """Run the G0 tiny v3 construction, JSON replay, and binary roundtrip."""
    alphas = np.array([0.7, 1.1, 1.6, 0.9], dtype=np.float64)
    design = np.array(
        [[1.8, -0.5, 0.3, 0.9], [0.2, 1.4, -0.7, 0.1], [0.5, -0.2, 1.1, 0.8]],
        dtype=np.float64,
    )
    aggregation = AdditiveDirichletAggregation(alphas, design, np.array([0.35, 0.8, 0.6]), np.eye(3))
    spectrum = RootResidualSpectrum.from_aggregation(aggregation)
    bank = build_chunked_projected_root_bank(
        aggregation,
        spectrum,
        mixture_rank=2,
        sample_count=64,
        sample_chunk_size=8,
        projection_chunk_size=4,
        source_seed=731,
        source_provenance="G0 tiny v3 construction",
    )
    replay = ConditionalAllocationMixture.from_json(bank.to_json(), expected_sha256=bank.sha256)
    if replay.to_json() != bank.to_json() or replay.sha256 != bank.sha256:
        raise ValueError("tiny v3 JSON replay failed")
    array = np.asarray(bank.projected_unit_mass_residual_factors[:, :, 0], dtype="<f8")
    array_path = output_dir / "tiny_bank.npy"
    _atomic_write_npy(array_path, array)
    loaded = _read_npy(array_path, expected_sha256=_sha256_file(array_path))
    if not np.array_equal(loaded, array):
        raise ValueError("tiny v3 binary roundtrip failed")
    report: dict[str, object] = {
        "schema": SCHEMA,
        "stage": "G0-tiny",
        "source_revision": source_revision,
        "bank_sha256": bank.sha256,
        "binary_file_sha256": _sha256_file(array_path),
        "projected_array_sha256": _array_sha256(array),
        "json_replay_exact": True,
        "binary_roundtrip_exact": True,
    }
    _atomic_write_json(output_dir / "tiny_report.json", report)
    return report


def _synthetic_aggregation(
    *,
    cells: int,
    observations: int,
    alpha_mode: str,
) -> AdditiveDirichletAggregation:
    """Return one deterministic observation-blind synthetic root operator."""
    index = np.arange(cells, dtype=np.float64)
    if alpha_mode == "singleton":
        alphas = np.ones(cells, dtype=np.float64)
    elif alpha_mode == "small":
        alphas = np.geomspace(1.0e-8, 1.0, cells, dtype=np.float64)
    elif alpha_mode == "heterogeneous":
        alphas = 0.2 + np.square(1.0 + np.mod(index, 13.0)) / 17.0
    else:
        raise ValueError("unknown synthetic alpha mode")
    rows = np.arange(1, observations + 1, dtype=np.float64)[:, np.newaxis]
    design = np.sin(rows * (index[np.newaxis, :] + 1.0) * 0.017) + 0.3 * np.cos(
        rows * (index[np.newaxis, :] + 2.0) * 0.011
    )
    noise = 0.7 + 0.05 * np.arange(observations, dtype=np.float64)
    return AdditiveDirichletAggregation(alphas, design, noise, np.eye(observations))


def run_g1(output: Path, *, source_revision: str) -> dict[str, object]:
    """Run the synthetic algorithm matrix and lock one projection microbatch."""
    started = time.perf_counter()
    benchmark = _synthetic_aggregation(cells=2_049, observations=8, alpha_mode="heterogeneous")
    benchmark_spectrum = RootResidualSpectrum.from_aggregation(benchmark)
    benchmark_outputs: dict[int, FloatArray] = {}
    p_medians: dict[int, float] = {}
    p_records: list[dict[str, object]] = []
    for projection_chunk in P_LADDER:
        elapsed: list[float] = []
        output_values: FloatArray | None = None
        for repeat in range(3):
            repeat_started = time.perf_counter()
            bank = build_chunked_projected_root_bank(
                benchmark,
                benchmark_spectrum,
                mixture_rank=min(4, benchmark_spectrum.retained_rank),
                sample_count=1_024,
                sample_chunk_size=1_024,
                projection_chunk_size=projection_chunk,
                source_seed=731,
                source_provenance=f"G1 P benchmark repeat {repeat}",
            )
            elapsed.append(time.perf_counter() - repeat_started)
            current = np.asarray(bank.projected_unit_mass_residual_factors[:, :, 0])
            if output_values is not None and not np.array_equal(current, output_values):
                raise ValueError("P benchmark did not replay exactly")
            output_values = current
        assert output_values is not None
        benchmark_outputs[projection_chunk] = output_values
        median_elapsed = float(np.median(elapsed))
        p_medians[projection_chunk] = median_elapsed
        p_records.append(
            {
                "projection_chunk_size": projection_chunk,
                "elapsed_seconds": elapsed,
                "median_elapsed_seconds": median_elapsed,
                "projected_array_sha256": _array_sha256(output_values),
            }
        )
    first_output = benchmark_outputs[P_LADDER[0]]
    benchmark_parity = {
        str(projection_chunk): _parity_record(
            first_output,
            benchmark_outputs[projection_chunk],
            native_cells=benchmark.cell_alphas.size,
        )
        for projection_chunk in P_LADDER
    }
    locked_p = min(
        P_LADDER,
        key=lambda value: (p_medians[value], value),
    )

    small = _synthetic_aggregation(cells=7, observations=5, alpha_mode="heterogeneous")
    small_spectrum = RootResidualSpectrum.from_aggregation(small)
    ranks = tuple(dict.fromkeys((0, 1, min(3, small_spectrum.retained_rank), small_spectrum.retained_rank)))
    small_records: list[dict[str, object]] = []
    for seed in SOURCE_SEEDS:
        for sample_count in (8, 64, 1_024, 65_536):
            reference = _all_at_once_source(
                small,
                small_spectrum,
                rank=small_spectrum.retained_rank,
                sample_count=sample_count,
                source_seed=seed,
                source_provenance="G1 small v2 reference",
            )
            effective_p = min(locked_p, sample_count)
            chunk_sizes = tuple(dict.fromkeys((effective_p, sample_count)))
            for rank in ranks:
                candidate_arrays: list[FloatArray] = []
                parity: dict[str, object] | None = None
                for chunk_size in chunk_sizes:
                    candidate = build_chunked_projected_root_bank(
                        small,
                        small_spectrum,
                        mixture_rank=rank,
                        sample_count=sample_count,
                        sample_chunk_size=chunk_size,
                        projection_chunk_size=effective_p,
                        source_seed=seed,
                        source_provenance="G1 small v3 candidate",
                    )
                    values = np.asarray(candidate.projected_unit_mass_residual_factors[:, :, 0])
                    candidate_arrays.append(values)
                    parity = _parity_record(
                        np.asarray(reference.projected_unit_mass_residual_factors[:, :rank, 0]),
                        values,
                        native_cells=small.cell_alphas.size,
                    )
                if any(not np.array_equal(candidate_arrays[0], values) for values in candidate_arrays[1:]):
                    raise ValueError("small v3 arrays changed with allocation chunk")
                small_records.append(
                    {
                        "seed": seed,
                        "sample_count": sample_count,
                        "rank": rank,
                        "projection_chunk_size": effective_p,
                        "sample_chunk_sizes": list(chunk_sizes),
                        "projected_array_sha256": _array_sha256(candidate_arrays[0]),
                        "parity": parity,
                    }
                )

    edge_records: list[dict[str, object]] = []
    for cells, mode in ((1, "singleton"), (9, "small")):
        aggregation = _synthetic_aggregation(cells=cells, observations=4, alpha_mode=mode)
        spectrum = RootResidualSpectrum.from_aggregation(aggregation)
        rank = spectrum.retained_rank
        bank = build_chunked_projected_root_bank(
            aggregation,
            spectrum,
            mixture_rank=rank,
            sample_count=64,
            sample_chunk_size=64,
            projection_chunk_size=min(locked_p, 64),
            source_seed=731,
            source_provenance=f"G1 {mode} edge",
        )
        edge_records.append(
            {
                "alpha_mode": mode,
                "cells": cells,
                "rank": rank,
                "finite": bool(np.all(np.isfinite(bank.projected_unit_mass_residual_factors))),
                "projected_array_sha256": _array_sha256(bank.projected_unit_mass_residual_factors),
            }
        )

    multiblock = _synthetic_aggregation(cells=21_203, observations=3, alpha_mode="heterogeneous")
    multiblock_spectrum = RootResidualSpectrum.from_aggregation(multiblock)
    multiblock_records: list[dict[str, object]] = []
    for sample_count in (8, 64, 1_024):
        rank = min(2, multiblock_spectrum.retained_rank)
        reference = _all_at_once_source(
            multiblock,
            multiblock_spectrum,
            rank=rank,
            sample_count=sample_count,
            source_seed=731,
            source_provenance="G1 forced-multiblock v2 reference",
        )
        effective_p = min(locked_p, sample_count)
        candidate = build_chunked_projected_root_bank(
            multiblock,
            multiblock_spectrum,
            mixture_rank=rank,
            sample_count=sample_count,
            sample_chunk_size=sample_count,
            projection_chunk_size=effective_p,
            source_seed=731,
            source_provenance="G1 forced-multiblock v3 candidate",
        )
        parity = _parity_record(
            np.asarray(reference.projected_unit_mass_residual_factors[:, :, 0]),
            np.asarray(candidate.projected_unit_mass_residual_factors[:, :, 0]),
            native_cells=multiblock.cell_alphas.size,
        )
        payload = candidate.payload
        if payload["sobol_block_dimensions"] != [21_201, 1]:
            raise ValueError("forced-multiblock Sobol dimensions changed")
        multiblock_records.append(
            {
                "sample_count": sample_count,
                "rank": rank,
                "sobol_block_dimensions": payload["sobol_block_dimensions"],
                "parity": parity,
            }
        )

    report: dict[str, object] = {
        "schema": SCHEMA,
        "stage": "G1",
        "source_revision": source_revision,
        "runtime": _runtime(),
        "projection_microbatch_selection": {
            "ladder": list(P_LADDER),
            "criterion": (
                "lowest median of three exact-replay, frozen-parity elapsed times; smaller P breaks exact ties"
            ),
            "records": p_records,
            "cross_candidate_reference_projection_chunk_size": P_LADDER[0],
            "cross_candidate_parity": benchmark_parity,
            "every_candidate_replays_bitwise_at_fixed_p": True,
            "all_candidate_outputs_within_frozen_parity_tolerance": True,
            "locked_projection_chunk_size": locked_p,
        },
        "small_matrix": small_records,
        "edge_cases": edge_records,
        "forced_multiblock_matrix": multiblock_records,
        "existing_v1_v2_and_failure_gates": ("enforced by focused pytest in the committed G1 launcher"),
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "passed": True,
    }
    _atomic_write_json(output, report)
    return report


def run_g2(
    output_dir: Path,
    *,
    input_path: Path,
    expected_input_sha256: str,
    source_revision: str,
    concentration: float,
) -> dict[str, object]:
    """Construct and publish the authoritative observation-blind spectrum."""
    started = time.perf_counter()
    _, aggregation = _build_paris_aggregation(
        input_path,
        expected_input_sha256=expected_input_sha256,
        concentration=concentration,
    )
    calibration = _scientific_calibration(
        _load_scientific_calibration_fields(
            input_path,
            expected_input_sha256=expected_input_sha256,
        )
    )
    calibrated_concentration = calibration["common_native_concentration"]
    if not isinstance(calibrated_concentration, Real):
        raise TypeError("scientific calibration concentration is not numeric")
    if not math.isclose(
        concentration,
        float(calibrated_concentration),
        rel_tol=0.0,
        abs_tol=64.0 * np.finfo(np.float64).eps,
    ):
        raise ValueError("G2 concentration does not match the scientific calibration")
    spectrum = RootResidualSpectrum.from_aggregation(aggregation, retained_variance_fraction=1.0)
    if spectrum.retained_rank != PARIS_OBSERVATION_COUNT - 1:
        raise ValueError("unexpected PARIS root spectrum rank")
    return _write_spectrum_bundle(
        output_dir,
        spectrum,
        source_revision=source_revision,
        input_path=input_path,
        input_sha256=expected_input_sha256,
        concentration=concentration,
        elapsed_seconds=time.perf_counter() - started,
        scientific_calibration=calibration,
    )


def run_g2_audit(
    output: Path,
    *,
    authoritative_manifest: Path,
    audit_manifest: Path,
    source_revision: str,
) -> dict[str, object]:
    """Compare a second-node spectrum with the authoritative G2 bundle."""
    authoritative = _load_spectrum_bundle(authoritative_manifest)
    audit = _load_spectrum_bundle(audit_manifest)
    authoritative_metadata = _read_json(authoritative_manifest)
    audit_metadata = _read_json(audit_manifest)
    if (
        authoritative_metadata.get("source_revision") != source_revision
        or audit_metadata.get("source_revision") != source_revision
    ):
        raise ValueError("G2 audit source revision mismatch")
    exact_context = (
        np.array_equal(
            authoritative.observation_mean_design,
            audit.observation_mean_design,
        )
        and np.array_equal(authoritative.noise_sd, audit.noise_sd)
        and authoritative.cell_alphas_sha256 == audit.cell_alphas_sha256
        and authoritative.design_sha256 == audit.design_sha256
        and authoritative.noise_sd_sha256 == audit.noise_sd_sha256
        and authoritative.retained_rank == audit.retained_rank
    )
    eigenvalue_scale = max(
        1.0,
        float(np.max(np.abs(authoritative.eigenvalues), initial=0.0)),
    )
    eigenvalue_tolerance = float(
        128.0 * np.finfo(np.float64).eps * max(1, authoritative.retained_rank) * eigenvalue_scale
    )
    maximum_eigenvalue_difference = float(
        np.max(
            np.abs(audit.eigenvalues - authoritative.eigenvalues),
            initial=0.0,
        )
    )
    authoritative_factor = authoritative.basis * np.sqrt(authoritative.eigenvalues)[np.newaxis, :]
    audit_factor = audit.basis * np.sqrt(audit.eigenvalues)[np.newaxis, :]
    authoritative_covariance = authoritative_factor @ authoritative_factor.T
    audit_covariance = audit_factor @ audit_factor.T
    covariance_scale = max(
        1.0,
        float(np.max(np.abs(authoritative_covariance), initial=0.0)),
    )
    covariance_tolerance = float(
        256.0
        * np.finfo(np.float64).eps
        * max(1, authoritative.observation_mean_design.size)
        * covariance_scale
    )
    maximum_covariance_difference = float(
        np.max(
            np.abs(audit_covariance - authoritative_covariance),
            initial=0.0,
        )
    )
    passed = (
        exact_context
        and maximum_eigenvalue_difference <= eigenvalue_tolerance
        and maximum_covariance_difference <= covariance_tolerance
    )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "stage": "G2-audit",
        "source_revision": source_revision,
        "authoritative_manifest_sha256": _sha256_file(authoritative_manifest),
        "audit_manifest_sha256": _sha256_file(audit_manifest),
        "exact_context_identity": exact_context,
        "maximum_eigenvalue_difference": maximum_eigenvalue_difference,
        "eigenvalue_tolerance": eigenvalue_tolerance,
        "maximum_reconstructed_covariance_difference": (maximum_covariance_difference),
        "reconstructed_covariance_tolerance": covariance_tolerance,
        "basis_array_identity_required": False,
        "audit_is_authoritative": False,
        "passed": passed,
    }
    _atomic_write_json(output, report)
    return report


def _locked_p(g1_manifest: Path) -> int:
    """Read the G1 lock and return its selected projection microbatch."""
    payload = _read_json(g1_manifest)
    if payload.get("schema") != SCHEMA or payload.get("stage") != "G1":
        raise ValueError("G1 manifest schema or stage mismatch")
    selection = payload.get("projection_microbatch_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("all_candidate_outputs_within_frozen_parity_tolerance") is not True
    ):
        raise ValueError("G1 manifest contains no valid frozen-parity P lock")
    locked = int(selection["locked_projection_chunk_size"])
    if locked not in P_LADDER:
        raise ValueError("G1 locked P is outside the predeclared ladder")
    return locked


def run_g3_prefix(
    output: Path,
    *,
    input_path: Path,
    expected_input_sha256: str,
    source_revision: str,
    concentration: float,
    spectrum_manifest: Path,
    g1_manifest: Path,
) -> dict[str, object]:
    """Run the actual-input v2/v3 prefix parity gate."""
    started = time.perf_counter()
    _, aggregation = _build_paris_aggregation(
        input_path,
        expected_input_sha256=expected_input_sha256,
        concentration=concentration,
    )
    _authenticate_scientific_spectrum_manifest(
        spectrum_manifest,
        source_revision=source_revision,
        input_sha256=expected_input_sha256,
        concentration=concentration,
    )
    spectrum = _load_spectrum_bundle(spectrum_manifest)
    locked_p = _locked_p(g1_manifest)
    reference = _all_at_once_source(
        aggregation,
        spectrum,
        rank=PREFIX_RANK,
        sample_count=PREFIX_SAMPLE_COUNT,
        source_seed=SOURCE_SEEDS[0],
        source_provenance="G3a PARIS v2 prefix reference",
    )
    reference_values = np.asarray(reference.projected_unit_mass_residual_factors[:, :, 0])
    records: list[dict[str, object]] = []
    candidate_values: list[FloatArray] = []
    legal_chunk_sizes = tuple(chunk_size for chunk_size in PREFIX_C_LADDER if chunk_size >= locked_p)
    if not legal_chunk_sizes:
        raise ValueError("locked P leaves no legal predeclared G3a allocation chunk")
    for chunk_size in legal_chunk_sizes:
        candidate = build_chunked_projected_root_bank(
            aggregation,
            spectrum,
            mixture_rank=PREFIX_RANK,
            sample_count=PREFIX_SAMPLE_COUNT,
            sample_chunk_size=chunk_size,
            projection_chunk_size=locked_p,
            source_seed=SOURCE_SEEDS[0],
            source_provenance="G3a PARIS v3 prefix candidate",
        )
        values = np.asarray(candidate.projected_unit_mass_residual_factors[:, :, 0])
        candidate_values.append(values)
        sobol = _sobol_metadata(candidate)
        if sobol["sobol_block_dimensions"] != [21_201, 2_222]:
            raise ValueError("frozen PARIS Sobol block identity changed")
        records.append(
            {
                "sample_chunk_size": chunk_size,
                "projection_chunk_size": locked_p,
                "bank_sha256": candidate.sha256,
                "projected_array_sha256": _array_sha256(values),
                "sobol_catalogue_sha256": sobol["sobol_catalogue_sha256"],
                "sobol_block_dimensions": sobol["sobol_block_dimensions"],
                "parity": _parity_record(
                    reference_values,
                    values,
                    native_cells=aggregation.cell_alphas.size,
                ),
            }
        )
    if any(not np.array_equal(candidate_values[0], values) for values in candidate_values[1:]):
        raise ValueError("G3a projected arrays changed with allocation chunk")
    report: dict[str, object] = {
        "schema": SCHEMA,
        "stage": "G3a",
        "source_revision": source_revision,
        "input_sha256": expected_input_sha256,
        "spectrum_manifest_sha256": _sha256_file(spectrum_manifest),
        "native_concentration": concentration,
        "root_variance": SCIENTIFIC_ROOT_VARIANCE,
        "science_calibration_schema": SCIENCE_CALIBRATION_SCHEMA,
        "sample_count": PREFIX_SAMPLE_COUNT,
        "mixture_rank": PREFIX_RANK,
        "source_seed": SOURCE_SEEDS[0],
        "predeclared_sample_chunk_ladder": list(PREFIX_C_LADDER),
        "legal_sample_chunks_for_locked_p": list(legal_chunk_sizes),
        "records": records,
        "all_v3_chunks_bitwise_identical": True,
        "observed_residual_used_by_builder": False,
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "passed": True,
    }
    _atomic_write_json(output, report)
    return report


def run_g3_bank(
    output_dir: Path,
    *,
    input_path: Path,
    expected_input_sha256: str,
    source_revision: str,
    concentration: float,
    spectrum_manifest: Path,
    g1_manifest: Path,
    sample_chunk_size: int,
    repeat: int,
) -> dict[str, object]:
    """Build one full projected source bank and publish binary then manifest."""
    chunk = _positive_power_of_two(sample_chunk_size, name="sample chunk size")
    if chunk not in RESOURCE_C_LADDER:
        raise ValueError("sample chunk size is outside the predeclared G3b ladder")
    if repeat not in (0, 1, 2):
        raise ValueError("repeat must be 0, 1, or 2")
    full_started = time.perf_counter()
    _, aggregation = _build_paris_aggregation(
        input_path,
        expected_input_sha256=expected_input_sha256,
        concentration=concentration,
    )
    _authenticate_scientific_spectrum_manifest(
        spectrum_manifest,
        source_revision=source_revision,
        input_sha256=expected_input_sha256,
        concentration=concentration,
    )
    spectrum = _load_spectrum_bundle(spectrum_manifest)
    locked_p = _locked_p(g1_manifest)
    constructor_started = time.perf_counter()
    bank = build_chunked_projected_root_bank(
        aggregation,
        spectrum,
        mixture_rank=SOURCE_RANK,
        sample_count=SOURCE_SAMPLE_COUNT,
        sample_chunk_size=chunk,
        projection_chunk_size=locked_p,
        source_seed=SOURCE_SEEDS[0],
        source_provenance=f"G3b PARIS resource C={chunk} repeat={repeat}",
    )
    constructor_seconds = time.perf_counter() - constructor_started
    values = np.ascontiguousarray(bank.projected_unit_mass_residual_factors[:, :, 0], dtype="<f8")
    bank_path = output_dir / "projected_locations.npy"
    _atomic_write_npy(bank_path, values)
    binary_sha256 = _sha256_file(bank_path)
    replay = _read_npy(bank_path, expected_sha256=binary_sha256)
    if not np.array_equal(values, replay):
        raise ValueError("published projected bank failed binary replay")
    sobol = _sobol_metadata(bank)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "stage": "G3b-candidate",
        "source_revision": source_revision,
        "input_sha256": expected_input_sha256,
        "spectrum_manifest_sha256": _sha256_file(spectrum_manifest),
        "g1_manifest_sha256": _sha256_file(g1_manifest),
        "native_concentration": concentration,
        "root_variance": SCIENTIFIC_ROOT_VARIANCE,
        "science_calibration_schema": SCIENCE_CALIBRATION_SCHEMA,
        "sample_count": SOURCE_SAMPLE_COUNT,
        "mixture_rank": SOURCE_RANK,
        "source_seed": SOURCE_SEEDS[0],
        "sample_chunk_size": chunk,
        "projection_chunk_size": locked_p,
        "repeat": repeat,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "bank_sha256": bank.sha256,
        "sobol_catalogue_sha256": sobol["sobol_catalogue_sha256"],
        "sobol_block_dimensions": sobol["sobol_block_dimensions"],
        "projected_array": {
            "file": bank_path.name,
            "dtype": "<f8",
            "shape": list(values.shape),
            "array_sha256": _array_sha256(values),
            "file_sha256": binary_sha256,
            "file_size_bytes": bank_path.stat().st_size,
            "binary_roundtrip_exact": True,
        },
        "constructor_seconds": constructor_seconds,
        "full_process_seconds": time.perf_counter() - full_started,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "constructor_rss_separately_available": False,
        "authoritative_full_process_rss": "Slurm sacct MaxRSS",
        "observed_residual_used_by_builder": False,
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "passed_internal_checks": True,
    }
    _atomic_write_json(output_dir / "bank_manifest.json", report)
    return report


def run_g3_warmup(
    output: Path,
    *,
    input_path: Path,
    expected_input_sha256: str,
    source_revision: str,
    concentration: float,
    spectrum_manifest: Path,
    g1_manifest: Path,
) -> dict[str, object]:
    """Run the single excluded warm-up before the timed G3b matrix."""
    started = time.perf_counter()
    _, aggregation = _build_paris_aggregation(
        input_path,
        expected_input_sha256=expected_input_sha256,
        concentration=concentration,
    )
    _authenticate_scientific_spectrum_manifest(
        spectrum_manifest,
        source_revision=source_revision,
        input_sha256=expected_input_sha256,
        concentration=concentration,
    )
    spectrum = _load_spectrum_bundle(spectrum_manifest)
    locked_p = _locked_p(g1_manifest)
    bank = build_chunked_projected_root_bank(
        aggregation,
        spectrum,
        mixture_rank=SOURCE_RANK,
        sample_count=4_096,
        sample_chunk_size=1_024,
        projection_chunk_size=locked_p,
        source_seed=SOURCE_SEEDS[0],
        source_provenance="G3b excluded warm-up",
    )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "stage": "G3b-warmup",
        "source_revision": source_revision,
        "sample_count": 4_096,
        "mixture_rank": SOURCE_RANK,
        "sample_chunk_size": 1_024,
        "projection_chunk_size": locked_p,
        "projected_array_sha256": _array_sha256(bank.projected_unit_mass_residual_factors),
        "excluded_from_resource_selection": True,
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "passed": True,
    }
    _atomic_write_json(output, report)
    return report


def _slurm_bytes(value: str) -> int:
    """Parse a Slurm memory quantity using binary suffixes."""
    normalized = value.strip()
    if not normalized:
        return 0
    suffixes = {
        "K": 1 << 10,
        "M": 1 << 20,
        "G": 1 << 30,
        "T": 1 << 40,
    }
    suffix = normalized[-1].upper()
    if suffix in suffixes:
        return int(math.ceil(float(normalized[:-1]) * suffixes[suffix]))
    return int(normalized)


def _sacct_records(job_ids: Sequence[str]) -> dict[str, SlurmRecord]:
    """Query terminal Slurm state and maximum step RSS for candidate jobs."""
    if not job_ids or any(
        not (
            job_id.isdigit() or (job_id.count("_") == 1 and all(part.isdigit() for part in job_id.split("_")))
        )
        for job_id in job_ids
    ):
        raise ValueError("candidate Slurm IDs must be decimal jobs or array_job_task IDs")
    command = [
        "sacct",
        "-n",
        "-P",
        "-j",
        ",".join(job_ids),
        # JobID preserves the logical array_job_task identity requested above.
        # JobIDRaw is the physical per-task allocation ID on BP1 and therefore
        # cannot be joined back to candidate manifests keyed by array_job_task.
        "--format=JobID,State,ElapsedRaw,MaxRSS",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    records: dict[str, SlurmRecord] = {
        job_id: {"state": None, "elapsed_seconds": None, "max_rss_bytes": 0} for job_id in job_ids
    }
    for raw_line in completed.stdout.splitlines():
        fields = raw_line.split("|")
        if len(fields) < 4:
            continue
        raw_id, state, elapsed, max_rss = fields[:4]
        base = raw_id.split(".", maxsplit=1)[0]
        if base not in records:
            continue
        if raw_id == base:
            records[base]["state"] = state.split(maxsplit=1)[0]
            records[base]["elapsed_seconds"] = int(elapsed) if elapsed else None
        records[base]["max_rss_bytes"] = max(
            int(records[base]["max_rss_bytes"]),
            _slurm_bytes(max_rss),
        )
    return records


def _time_verbose_swaps(path: Path) -> int:
    """Read the GNU time voluntary swap count from one completed job log."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"GNU time report must be a real file: {path}")
    prefix = "Swaps: "
    matches = [
        line.strip()[len(prefix) :]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(prefix)
    ]
    if len(matches) != 1 or not matches[0].isdigit():
        raise ValueError(f"GNU time report has no unique swap count: {path}")
    return int(matches[0])


def run_g3_certify(
    output: Path,
    completion_marker: Path,
    *,
    prefix_manifest: Path,
    candidate_manifests: Sequence[Path],
    source_revision: str,
) -> dict[str, object]:
    """Merge prefix/resource evidence and apply the predeclared G3 hard gates."""
    prefix = _read_json(prefix_manifest)
    if (
        prefix.get("schema") != SCHEMA
        or prefix.get("stage") != "G3a"
        or prefix.get("source_revision") != source_revision
        or prefix.get("native_concentration") != SCIENTIFIC_CONCENTRATION
        or prefix.get("root_variance") != SCIENTIFIC_ROOT_VARIANCE
        or prefix.get("science_calibration_schema") != SCIENCE_CALIBRATION_SCHEMA
        or prefix.get("passed") is not True
    ):
        raise ValueError("G3a prefix manifest is not a passing artifact")
    expected_count = len(RESOURCE_C_LADDER) * 3
    if len(candidate_manifests) != expected_count:
        raise ValueError(f"G3b requires exactly {expected_count} candidate manifests")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    accounting_ids: list[str] = []
    array_job_ids: set[str] = set()
    array_task_ids: set[int] = set()
    for path in candidate_manifests:
        payload = _read_json(path)
        if (
            payload.get("schema") != SCHEMA
            or payload.get("stage") != "G3b-candidate"
            or payload.get("source_revision") != source_revision
        ):
            raise ValueError(f"candidate manifest identity mismatch: {path}")
        chunk = int(payload["sample_chunk_size"])
        repeat = int(payload["repeat"])
        key = (chunk, repeat)
        if chunk not in RESOURCE_C_LADDER or repeat not in (0, 1, 2) or key in seen:
            raise ValueError(f"candidate matrix key is invalid or repeated: {key}")
        seen.add(key)
        job_id = payload.get("slurm_job_id")
        if not isinstance(job_id, str) or not job_id.isdigit():
            raise ValueError(f"candidate has no valid Slurm job ID: {path}")
        array_job_id = payload.get("slurm_array_job_id")
        array_task_id = payload.get("slurm_array_task_id")
        if (
            not isinstance(array_job_id, str)
            or not array_job_id.isdigit()
            or not isinstance(array_task_id, str)
            or not array_task_id.isdigit()
            or int(array_task_id) != RESOURCE_C_LADDER.index(chunk) * 3 + repeat
        ):
            raise ValueError(f"candidate has invalid Slurm array identity: {path}")
        array_job_ids.add(array_job_id)
        array_task_ids.add(int(array_task_id))
        accounting_id = f"{array_job_id}_{array_task_id}"
        accounting_ids.append(accounting_id)
        payload["_accounting_id"] = accounting_id
        if (
            payload.get("native_concentration") != SCIENTIFIC_CONCENTRATION
            or payload.get("root_variance") != SCIENTIFIC_ROOT_VARIANCE
            or payload.get("science_calibration_schema") != SCIENCE_CALIBRATION_SCHEMA
        ):
            raise ValueError(f"candidate scientific model identity mismatch: {path}")
        payload["_manifest_path"] = str(path)
        payload["_manifest_sha256"] = _sha256_file(path)
        candidates.append(payload)
    expected_keys = {(chunk, repeat) for chunk in RESOURCE_C_LADDER for repeat in (0, 1, 2)}
    if seen != expected_keys:
        raise ValueError("candidate matrix is incomplete")
    if len(array_job_ids) != 1 or array_task_ids != set(range(expected_count)):
        raise ValueError("candidate matrix is not one complete Slurm array")
    for field in (
        "input_sha256",
        "spectrum_manifest_sha256",
        "g1_manifest_sha256",
    ):
        values = {str(candidate[field]) for candidate in candidates}
        if len(values) != 1:
            raise ValueError(f"candidate matrix disagrees on {field}")
    if any(
        candidate["input_sha256"] != prefix["input_sha256"]
        or candidate["spectrum_manifest_sha256"] != prefix["spectrum_manifest_sha256"]
        for candidate in candidates
    ):
        raise ValueError("G3a and G3b scientific inputs disagree")
    projection_microbatches = {int(candidate["projection_chunk_size"]) for candidate in candidates}
    prefix_projection_microbatches = {
        int(record["projection_chunk_size"])
        for record in prefix.get("records", [])
        if isinstance(record, dict) and "projection_chunk_size" in record
    }
    if (
        len(projection_microbatches) != 1
        or projection_microbatches != prefix_projection_microbatches
        or next(iter(projection_microbatches)) not in P_LADDER
    ):
        raise ValueError("G3a and G3b projection-microbatch locks disagree")
    selected_projection_microbatch = next(iter(projection_microbatches))

    accounting = _sacct_records(accounting_ids)
    array_digests = {str(candidate["projected_array"]["array_sha256"]) for candidate in candidates}
    file_digests = {str(candidate["projected_array"]["file_sha256"]) for candidate in candidates}
    identical = len(array_digests) == 1 and len(file_digests) == 1
    resource_records: list[dict[str, object]] = []
    medians: dict[int, float] = {}
    passing_chunks: list[int] = []
    for chunk in RESOURCE_C_LADDER:
        chunk_candidates = sorted(
            (candidate for candidate in candidates if int(candidate["sample_chunk_size"]) == chunk),
            key=lambda candidate: int(candidate["repeat"]),
        )
        elapsed = [float(candidate["constructor_seconds"]) for candidate in chunk_candidates]
        swaps = [
            _time_verbose_swaps(Path(str(candidate["_manifest_path"])).parent / "resource.time")
            for candidate in chunk_candidates
        ]
        median_elapsed = float(np.median(elapsed))
        medians[chunk] = median_elapsed
        job_records = [accounting[str(candidate["_accounting_id"])] for candidate in chunk_candidates]
        max_rss = max(int(record["max_rss_bytes"]) for record in job_records)
        states = [record["state"] for record in job_records]
        internal = all(candidate.get("passed_internal_checks") is True for candidate in chunk_candidates)
        passed = (
            identical
            and internal
            and states == ["COMPLETED", "COMPLETED", "COMPLETED"]
            and max_rss > 0
            and max_rss <= 12 * (1 << 30)
            and swaps == [0, 0, 0]
            and all(value <= 45 * 60 for value in elapsed)
        )
        if passed:
            passing_chunks.append(chunk)
        resource_records.append(
            {
                "sample_chunk_size": chunk,
                "constructor_seconds": elapsed,
                "median_constructor_seconds": median_elapsed,
                "slurm_states": states,
                "slurm_max_rss_bytes": max_rss,
                "full_process_swaps": swaps,
                "internal_checks_passed": internal,
                "passed": passed,
            }
        )
    selected_chunk = (
        min(passing_chunks, key=lambda chunk: (medians[chunk], chunk)) if passing_chunks else None
    )
    passed = identical and bool(passing_chunks)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "stage": "G3",
        "source_revision": source_revision,
        "native_concentration": SCIENTIFIC_CONCENTRATION,
        "root_variance": SCIENTIFIC_ROOT_VARIANCE,
        "science_calibration_schema": SCIENCE_CALIBRATION_SCHEMA,
        "prefix_manifest": {
            "path": str(prefix_manifest),
            "sha256": _sha256_file(prefix_manifest),
            "passed": True,
        },
        "candidate_manifest_sha256": {
            str(candidate["_manifest_path"]): candidate["_manifest_sha256"] for candidate in candidates
        },
        "resource_gates": {
            "maximum_slurm_rss_bytes": 12 * (1 << 30),
            "maximum_constructor_seconds": 45 * 60,
            "required_terminal_state": "COMPLETED",
            "all_candidate_projected_array_digests_identical": identical,
            "records": resource_records,
            "slurm_array_job_id": next(iter(array_job_ids)),
        },
        "projected_array_sha256": next(iter(array_digests)) if identical else None,
        "binary_file_sha256": next(iter(file_digests)) if identical else None,
        "selection_rule": (
            "lowest median of three constructor times among resource-passing chunks; smaller C breaks exact ties"
        ),
        "selected_sample_chunk_size": selected_chunk,
        "selected_projection_microbatch": selected_projection_microbatch,
        "passed": passed,
        "next_gate": (
            "G4-scientific-source-validation-predeclared"
            if passed
            else "terminal-G3-resource-or-parity-hard-stop"
        ),
    }
    _atomic_write_json(output, report)
    if passed:
        _atomic_write_text(
            completion_marker,
            f"G3 complete for {source_revision}; selected C={selected_chunk}\n",
        )
    return report


def _parser() -> argparse.ArgumentParser:
    """Build the staged engineering CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tiny = subparsers.add_parser("tiny")
    tiny.add_argument("--output-dir", type=Path, required=True)
    tiny.add_argument("--source-revision", required=True)

    g1 = subparsers.add_parser("g1")
    g1.add_argument("--output", type=Path, required=True)
    g1.add_argument("--source-revision", required=True)

    calibration = subparsers.add_parser("scientific-calibration")
    calibration.add_argument("--input", type=Path, required=True)
    calibration.add_argument("--expected-input-sha256", default=PARIS_INPUT_SHA256)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--source-revision", required=True)

    for command in ("g2", "g3-prefix", "g3-bank", "g3-warmup"):
        stage = subparsers.add_parser(command)
        stage.add_argument("--input", type=Path, required=True)
        stage.add_argument("--expected-input-sha256", default=PARIS_INPUT_SHA256)
        stage.add_argument("--source-revision", required=True)
        stage.add_argument(
            "--concentration",
            type=float,
            default=SCIENTIFIC_CONCENTRATION,
        )
        if command == "g2":
            stage.add_argument("--output-dir", type=Path, required=True)
        else:
            stage.add_argument("--spectrum-manifest", type=Path, required=True)
            stage.add_argument("--g1-manifest", type=Path, required=True)
            if command == "g3-prefix":
                stage.add_argument("--output", type=Path, required=True)
            elif command == "g3-bank":
                stage.add_argument("--output-dir", type=Path, required=True)
                stage.add_argument("--sample-chunk-size", type=int, required=True)
                stage.add_argument("--repeat", type=int, required=True)
            else:
                stage.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("g2-audit")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--authoritative-manifest", type=Path, required=True)
    audit.add_argument("--audit-manifest", type=Path, required=True)
    audit.add_argument("--source-revision", required=True)
    certify = subparsers.add_parser("g3-certify")
    certify.add_argument("--output", type=Path, required=True)
    certify.add_argument("--completion-marker", type=Path, required=True)
    certify.add_argument("--prefix-manifest", type=Path, required=True)
    certify.add_argument(
        "--candidate-manifest",
        action="append",
        type=Path,
        required=True,
    )
    certify.add_argument("--source-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one create-only staged engineering action."""
    arguments = _parser().parse_args(argv)
    source_revision = _full_revision(arguments.source_revision)
    if arguments.command == "tiny":
        run_tiny(arguments.output_dir, source_revision=source_revision)
    elif arguments.command == "g1":
        run_g1(arguments.output, source_revision=source_revision)
    elif arguments.command == "scientific-calibration":
        run_scientific_calibration(
            arguments.output,
            input_path=arguments.input,
            expected_input_sha256=arguments.expected_input_sha256,
            source_revision=source_revision,
        )
    elif arguments.command == "g2":
        run_g2(
            arguments.output_dir,
            input_path=arguments.input,
            expected_input_sha256=arguments.expected_input_sha256,
            source_revision=source_revision,
            concentration=arguments.concentration,
        )
    elif arguments.command == "g2-audit":
        report = run_g2_audit(
            arguments.output,
            authoritative_manifest=arguments.authoritative_manifest,
            audit_manifest=arguments.audit_manifest,
            source_revision=source_revision,
        )
        return 0 if report["passed"] is True else 3
    elif arguments.command == "g3-prefix":
        run_g3_prefix(
            arguments.output,
            input_path=arguments.input,
            expected_input_sha256=arguments.expected_input_sha256,
            source_revision=source_revision,
            concentration=arguments.concentration,
            spectrum_manifest=arguments.spectrum_manifest,
            g1_manifest=arguments.g1_manifest,
        )
    elif arguments.command == "g3-bank":
        run_g3_bank(
            arguments.output_dir,
            input_path=arguments.input,
            expected_input_sha256=arguments.expected_input_sha256,
            source_revision=source_revision,
            concentration=arguments.concentration,
            spectrum_manifest=arguments.spectrum_manifest,
            g1_manifest=arguments.g1_manifest,
            sample_chunk_size=arguments.sample_chunk_size,
            repeat=arguments.repeat,
        )
    elif arguments.command == "g3-warmup":
        run_g3_warmup(
            arguments.output,
            input_path=arguments.input,
            expected_input_sha256=arguments.expected_input_sha256,
            source_revision=source_revision,
            concentration=arguments.concentration,
            spectrum_manifest=arguments.spectrum_manifest,
            g1_manifest=arguments.g1_manifest,
        )
    else:
        report = run_g3_certify(
            arguments.output,
            arguments.completion_marker,
            prefix_manifest=arguments.prefix_manifest,
            candidate_manifests=arguments.candidate_manifest,
            source_revision=source_revision,
        )
        return 0 if report["passed"] is True else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
