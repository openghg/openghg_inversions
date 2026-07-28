#!/usr/bin/env python3
r"""Run the observation-blind BP1 G4 projected-source scientific gate.

The executable consumes the scientifically calibrated single-root
Gamma--Dirichlet model and the authoritative G2 spectrum.  It never reads the
realized ``mf`` values.  A separately scrambled, held-out prior-predictive grid
is frozen once, then every candidate source seed is assessed on nested
``S``/``q`` prefixes.  Leading coordinates use the direct equal-weight finite
source likelihood; spectrum directions ``q:r`` use the analytic Gaussian
moment closure.  No singleton ``CompressedRootMixture`` is constructed.

All outputs are create-only, numeric arrays are little-endian float64 C-order,
and completion/lock markers are published last.  This is a fixed-root
approximation gate, not an RJ likelihood or a source of structural weights.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import itertools
import math
from numbers import Real
from pathlib import Path
import sys
import time
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats
from scipy.stats import qmc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import conditional_residual_image_chunked_projected_bank_hpc as hpc
from openghg_inversions.experimental.rjmcmc.aggregation_error_exact_mixture import (
    RootResidualSpectrum,
    build_chunked_projected_root_bank,
)

FloatArray = NDArray[np.float64]

SCHEMA = "rjmcmc-chunked-projected-bank-g4-v1"
GRID_SCHEMA = "rjmcmc-chunked-projected-bank-g4-grid-v1"
GRID_SIZE = 256
GRID_ALLOCATION_SEED = 12_947
GRID_NOISE_SEED = 12_953
GRID_OUTER_COEFFICIENT_SEED = 12_959
GRID_SOBOL_BITS = 52
SOURCE_SEEDS = (731, 1_877, 4_099, 8_317)
S_LADDER = (16_384, 32_768, 65_536)
Q_LADDER = (16, 32, 64, 128)
JOINT_PAIRS_ONE_BASED = (
    (1, 2),
    (15, 16),
    (16, 17),
    (31, 32),
    (32, 33),
    (63, 64),
    (64, 65),
    (127, 128),
)
RADIAL_DIMENSIONS = (2, 16, 32, 64, 128)
ROOT_GAMMA_SHAPE = 43.742615510366136
ROOT_GAMMA_RATE = 43.742615510366136
MOMENT_MEAN_LIMIT = 0.02
MOMENT_COVARIANCE_LIMIT = 0.06
KS_LIMIT = 0.020
TAIL_2_LIMIT = 0.005
TAIL_3_LIMIT = 0.002
JOINT_MAX_LIMIT = 0.005
JOINT_SIGNED_LIMIT = 0.002
RADIAL_99_LIMIT = 0.005
RADIAL_999_LIMIT = 0.002
LIKELIHOOD_MEDIAN_LIMIT_NAT = 0.05
LIKELIHOOD_P99_LIMIT_NAT = 0.20
TRANSLATION_FACTOR = 4_096.0
MINIMUM_SUFFIX_LENGTH = 2
THRESHOLD_SUPPLEMENT = (
    Path(__file__).resolve().parents[2]
    / "docs/plans/rjmcmc_chunked_projected_bank_g4_threshold_supplement.md"
)


def _threshold_supplement_record() -> dict[str, str]:
    """Return the immutable in-source identity of the pre-result G4 rules."""
    if THRESHOLD_SUPPLEMENT.is_symlink() or not THRESHOLD_SUPPLEMENT.is_file():
        raise FileNotFoundError("G4 threshold supplement is missing or is a symlink")
    return {
        "path": str(THRESHOLD_SUPPLEMENT),
        "sha256": hpc._sha256_file(THRESHOLD_SUPPLEMENT),
    }


def _strict_spectrum(
    manifest_path: Path,
    *,
    source_revision: str,
) -> tuple[RootResidualSpectrum, dict[str, Any]]:
    """Load a G2 spectrum after authenticating its scientific identity."""
    payload = hpc._read_json(manifest_path)
    model = payload.get("model")
    input_record = payload.get("input")
    calibration: object = model.get("scientific_calibration") if isinstance(model, dict) else None
    if (
        payload.get("schema") != hpc.SCHEMA
        or payload.get("stage") != "G2"
        or payload.get("source_revision") != source_revision
        or not isinstance(input_record, dict)
        or input_record.get("sha256") != hpc.PARIS_INPUT_SHA256
        or not isinstance(model, dict)
        or model.get("concentration_status") != "scientific-modeled-domain-and-GBR-aggregate-CV-lock"
        or not math.isclose(
            float(model.get("native_concentration", math.nan)),
            hpc.SCIENTIFIC_CONCENTRATION,
            rel_tol=0.0,
            abs_tol=64.0 * np.finfo(np.float64).eps,
        )
        or not math.isclose(
            float(model.get("root_variance", math.nan)),
            hpc.SCIENTIFIC_ROOT_VARIANCE,
            rel_tol=0.0,
            abs_tol=64.0 * np.finfo(np.float64).eps,
        )
        or not isinstance(calibration, dict)
        or calibration.get("schema") != hpc.SCIENCE_CALIBRATION_SCHEMA
        or not isinstance(calibration.get("common_native_concentration"), Real)
        or not math.isclose(
            float(calibration["common_native_concentration"]),
            hpc.SCIENTIFIC_CONCENTRATION,
            rel_tol=64.0 * np.finfo(np.float64).eps,
            abs_tol=64.0 * np.finfo(np.float64).eps,
        )
        or not isinstance(calibration.get("root_variance"), Real)
        or not math.isclose(
            float(calibration["root_variance"]),
            hpc.SCIENTIFIC_ROOT_VARIANCE,
            rel_tol=64.0 * np.finfo(np.float64).eps,
            abs_tol=64.0 * np.finfo(np.float64).eps,
        )
        or calibration.get("observed_mole_fraction_used") is not False
    ):
        raise ValueError("spectrum does not carry the locked scientific calibration")
    return hpc._load_spectrum_bundle(manifest_path), payload


def _array_record(path: Path, values: FloatArray) -> dict[str, object]:
    """Publish one float64 array and return its strict identity record."""
    little_endian = np.ascontiguousarray(values, dtype="<f8")
    hpc._atomic_write_npy(path, little_endian)
    return {
        "file": path.name,
        "dtype": "<f8",
        "shape": list(little_endian.shape),
        "array_sha256": hpc._array_sha256(little_endian),
        "file_sha256": hpc._sha256_file(path),
        "file_size_bytes": path.stat().st_size,
    }


def _load_record(parent: Path, record: object, *, name: str) -> FloatArray:
    """Strictly load one array described by a canonical manifest record."""
    if not isinstance(record, dict):
        raise ValueError(f"missing array record {name!r}")
    filename = record.get("file")
    digest = record.get("file_sha256")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"unsafe array filename for {name!r}")
    if not isinstance(digest, str):
        raise ValueError(f"missing file digest for {name!r}")
    values = hpc._read_npy(parent / filename, expected_sha256=digest)
    if (
        values.dtype != np.dtype("<f8")
        or list(values.shape) != record.get("shape")
        or hpc._array_sha256(values) != record.get("array_sha256")
    ):
        raise ValueError(f"array identity mismatch for {name!r}")
    return np.asarray(values, dtype=np.float64)


def _require_output_directory(path: Path) -> None:
    """Require an existing real output directory outside production output."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError("output directory must be an existing real directory")
    probe = path / ".g4-output-probe"
    hpc._validate_output(probe)


def _grid_masses() -> FloatArray:
    """Return the frozen permuted-midpoint Gamma root-mass grid."""
    index = np.arange(GRID_SIZE, dtype=np.int64)
    probability = ((73 * index) % GRID_SIZE + 0.5) / GRID_SIZE
    masses = stats.gamma.ppf(
        probability,
        a=ROOT_GAMMA_SHAPE,
        scale=1.0 / ROOT_GAMMA_RATE,
    )
    result = np.asarray(masses, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("frozen Gamma root-mass grid is invalid")
    return result


def _held_out_noise(observation_count: int) -> FloatArray:
    """Return the frozen scrambled-Sobol standard-normal noise grid."""
    engine = qmc.Sobol(
        d=observation_count,
        scramble=True,
        bits=GRID_SOBOL_BITS,
        rng=GRID_NOISE_SEED,
        optimization=None,
    )
    uniforms = engine.random_base2(m=int(math.log2(GRID_SIZE)))
    clipped = np.clip(uniforms, 2.0**-53, 1.0 - 2.0**-53)
    result = np.asarray(special.ndtri(clipped), dtype=np.float64)
    if result.shape != (GRID_SIZE, observation_count) or not np.all(np.isfinite(result)):
        raise ValueError("held-out Sobol noise grid is invalid")
    return result


def _outer_coefficients(outer_count: int) -> FloatArray:
    """Return frozen positive mean-one/SD-one prior-predictive coefficients."""
    engine = qmc.Sobol(
        d=outer_count,
        scramble=True,
        bits=GRID_SOBOL_BITS,
        rng=GRID_OUTER_COEFFICIENT_SEED,
        optimization=None,
    )
    uniforms = engine.random_base2(m=int(math.log2(GRID_SIZE)))
    clipped = np.clip(uniforms, 2.0**-53, 1.0 - 2.0**-53)
    log_variance = math.log(2.0)
    result = np.exp(-0.5 * log_variance + math.sqrt(log_variance) * special.ndtri(clipped))
    coefficients = np.asarray(result, dtype=np.float64)
    if coefficients.shape != (GRID_SIZE, outer_count) or not np.all(np.isfinite(coefficients)):
        raise ValueError("held-out outer-coefficient grid is invalid")
    return coefficients


def _offset_from_dataset(dataset: Any) -> tuple[FloatArray, FloatArray]:
    """Return prior-predictive outer coefficients and observation offsets."""
    boundary = np.asarray(dataset["YaprioriBC"].values, dtype=np.float64)
    outer = np.asarray(dataset["outer_design"].values, dtype=np.float64)
    coefficients = _outer_coefficients(outer.shape[1])
    result = boundary[np.newaxis, :] + coefficients @ outer.T
    if result.shape != (GRID_SIZE, boundary.size) or not np.all(np.isfinite(result)):
        raise ValueError("prior-predictive observation-blind offsets are invalid")
    return coefficients, result


def run_grid(
    output_dir: Path,
    *,
    input_path: Path,
    expected_input_sha256: str,
    spectrum_manifest: Path,
    source_revision: str,
) -> dict[str, object]:
    """Construct and publish the untouched 256-state prior-predictive grid."""
    _require_output_directory(output_dir)
    started = time.perf_counter()
    spectrum, spectrum_payload = _strict_spectrum(
        spectrum_manifest,
        source_revision=source_revision,
    )
    dataset, aggregation = hpc._build_paris_aggregation(
        input_path,
        expected_input_sha256=expected_input_sha256,
        concentration=hpc.SCIENTIFIC_CONCENTRATION,
    )
    held_out = hpc._all_at_once_source(
        aggregation,
        spectrum,
        rank=spectrum.retained_rank,
        sample_count=GRID_SIZE,
        source_seed=GRID_ALLOCATION_SEED,
        source_provenance="G4 held-out observation-blind v2 allocation grid",
    )
    allocation_coordinates = np.asarray(
        held_out.projected_unit_mass_residual_factors[:, :, 0],
        dtype=np.float64,
    )
    if allocation_coordinates.shape != (GRID_SIZE, spectrum.retained_rank):
        raise ValueError("held-out allocation grid has the wrong shape")
    observation_count = int(spectrum.observation_mean_design.size)
    noise = _held_out_noise(observation_count)
    masses = _grid_masses()
    outer_coefficients, offset = _offset_from_dataset(dataset)
    whitened_operator_response = allocation_coordinates @ spectrum.basis.T
    operator_response = spectrum.observation_mean_design[np.newaxis, :] + (
        spectrum.noise_sd[np.newaxis, :] * whitened_operator_response
    )
    observations = (
        offset + masses[:, np.newaxis] * operator_response + spectrum.noise_sd[np.newaxis, :] * noise
    )
    if not np.all(np.isfinite(observations)):
        raise ValueError("prior-predictive observation grid is non-finite")

    arrays = {
        "root_mass": _array_record(output_dir / "root_mass.npy", masses),
        "offset": _array_record(output_dir / "offset.npy", offset),
        "outer_coefficients": _array_record(
            output_dir / "outer_coefficients.npy",
            outer_coefficients,
        ),
        "allocation_coordinates": _array_record(
            output_dir / "allocation_coordinates.npy",
            allocation_coordinates,
        ),
        "standard_normal_noise": _array_record(
            output_dir / "standard_normal_noise.npy",
            noise,
        ),
        "operator_response": _array_record(
            output_dir / "operator_response.npy",
            operator_response,
        ),
        "observation": _array_record(
            output_dir / "observation.npy",
            observations,
        ),
    }
    report: dict[str, object] = {
        "schema": GRID_SCHEMA,
        "stage": "G4-grid",
        "source_revision": source_revision,
        "input_sha256": expected_input_sha256,
        "spectrum_manifest": {
            "path": str(spectrum_manifest),
            "sha256": hpc._sha256_file(spectrum_manifest),
        },
        "scientific_calibration": spectrum_payload["model"]["scientific_calibration"],
        "threshold_supplement": _threshold_supplement_record(),
        "grid_size": GRID_SIZE,
        "root_mass_rule": {
            "formula": "GammaPPF(((73*i mod 256)+0.5)/256)",
            "shape": ROOT_GAMMA_SHAPE,
            "rate": ROOT_GAMMA_RATE,
        },
        "allocation_rule": {
            "construction": "existing-v2-scrambled-Sobol-balanced-Dirichlet",
            "seed": GRID_ALLOCATION_SEED,
            "bank_sha256": held_out.sha256,
            **hpc._sobol_metadata(held_out),
        },
        "noise_rule": {
            "engine": "scipy.stats.qmc.Sobol",
            "scramble": True,
            "bits": GRID_SOBOL_BITS,
            "seed": GRID_NOISE_SEED,
            "inverse": "scipy.special.ndtri",
            "endpoint_clip": "2^-53",
        },
        "outer_coefficient_rule": {
            "engine": "scipy.stats.qmc.Sobol",
            "scramble": True,
            "bits": GRID_SOBOL_BITS,
            "seed": GRID_OUTER_COEFFICIENT_SEED,
            "distribution": "lognormal",
            "arithmetic_mean": 1.0,
            "arithmetic_sd": 1.0,
            "log_mu": -0.5 * math.log(2.0),
            "log_sigma": math.sqrt(math.log(2.0)),
            "endpoint_clip": "2^-53",
        },
        "offset_rule": "YaprioriBC+outer_design@outer_coefficients[i]",
        "observation_rule": "offset+root_mass*operator_response+noise_sd*standard_normal_noise",
        "arrays": arrays,
        "realized_mf_used": False,
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": hpc._peak_rss_bytes(),
        "passed": True,
    }
    hpc._atomic_write_json(output_dir / "grid_manifest.json", report)
    hpc._atomic_write_text(
        output_dir / "G4_GRID_COMPLETE.txt",
        f"G4 observation-blind grid complete for {source_revision}\n",
    )
    return report


def _load_grid(
    manifest_path: Path,
    *,
    source_revision: str,
    spectrum_manifest: Path,
) -> tuple[dict[str, FloatArray], dict[str, Any]]:
    """Load and strictly authenticate the frozen prior-predictive grid."""
    payload = hpc._read_json(manifest_path)
    spectrum_record = payload.get("spectrum_manifest")
    threshold_record = payload.get("threshold_supplement")
    if (
        payload.get("schema") != GRID_SCHEMA
        or payload.get("stage") != "G4-grid"
        or payload.get("source_revision") != source_revision
        or payload.get("grid_size") != GRID_SIZE
        or payload.get("realized_mf_used") is not False
        or not isinstance(spectrum_record, dict)
        or spectrum_record.get("sha256") != hpc._sha256_file(spectrum_manifest)
        or threshold_record != _threshold_supplement_record()
    ):
        raise ValueError("G4 grid identity mismatch")
    arrays_payload = payload.get("arrays")
    if not isinstance(arrays_payload, dict):
        raise ValueError("G4 grid has no array records")
    arrays = {
        name: _load_record(manifest_path.parent, arrays_payload.get(name), name=name)
        for name in (
            "root_mass",
            "offset",
            "outer_coefficients",
            "allocation_coordinates",
            "standard_normal_noise",
            "operator_response",
            "observation",
        )
    }
    if (
        arrays["root_mass"].shape != (GRID_SIZE,)
        or arrays["offset"].ndim != 2
        or arrays["offset"].shape[0] != GRID_SIZE
        or arrays["observation"].shape != arrays["offset"].shape
        or arrays["allocation_coordinates"].shape != (GRID_SIZE, hpc.PARIS_OBSERVATION_COUNT - 1)
        or arrays["standard_normal_noise"].shape != (GRID_SIZE, hpc.PARIS_OBSERVATION_COUNT)
        or arrays["outer_coefficients"].shape != (GRID_SIZE, len(hpc.PARIS_OUTER_LABELS))
        or not np.array_equal(arrays["root_mass"], _grid_masses())
        or not np.array_equal(
            arrays["standard_normal_noise"],
            _held_out_noise(arrays["offset"].shape[1]),
        )
        or not np.array_equal(
            arrays["outer_coefficients"],
            _outer_coefficients(len(hpc.PARIS_OUTER_LABELS)),
        )
    ):
        raise ValueError("G4 grid controls or dimensions do not replay")
    return arrays, payload


def _eigenvalue_threshold(spectrum: RootResidualSpectrum) -> float:
    """Return the frozen floor used only for relative moment diagnostics."""
    leading = float(spectrum.eigenvalues[0]) if spectrum.retained_rank else 0.0
    return float(
        max(
            float(spectrum.eigenvalue_tolerance),
            1_024.0 * np.finfo(np.float64).eps * max(1.0, leading),
        )
    )


def _moment_metrics(
    locations: FloatArray,
    eigenvalues: FloatArray,
    *,
    sample_count: int,
    rank: int,
    eigenvalue_threshold: float,
    maximum_sample_count: int = S_LADDER[-1],
) -> dict[str, object]:
    """Apply the frozen analytic-coordinate mean and covariance gates."""
    values = np.asarray(locations[:sample_count, :rank], dtype=np.float64)
    lambdas = np.asarray(eigenvalues[:rank], dtype=np.float64)
    if (
        values.shape != (sample_count, rank)
        or lambdas.shape != (rank,)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(lambdas))
        or np.any(lambdas < 0.0)
    ):
        raise ValueError("moment diagnostic inputs are invalid")
    scale = math.sqrt(maximum_sample_count / sample_count)
    active = lambdas > eigenvalue_threshold
    tiny = ~active

    if np.any(active):
        standardized = values[:, active] / np.sqrt(lambdas[active])[np.newaxis, :]
        active_mean = np.mean(standardized, axis=0)
        centered = standardized - active_mean[np.newaxis, :]
        active_covariance = centered.T @ centered / sample_count
        active_covariance_error = float(
            np.linalg.norm(
                active_covariance - np.eye(active_covariance.shape[0]),
                ord="fro",
            )
            / math.sqrt(active_covariance.shape[0])
        )
        active_mean_error = float(np.max(np.abs(active_mean), initial=0.0))
    else:
        active_mean_error = 0.0
        active_covariance_error = 0.0

    if np.any(tiny):
        tiny_values = values[:, tiny]
        tiny_mean = np.mean(tiny_values, axis=0)
        full_mean = np.mean(values, axis=0)
        full_centered = values - full_mean[np.newaxis, :]
        full_covariance_error = full_centered.T @ full_centered / sample_count - np.diag(lambdas)
        tiny_involved = tiny[:, np.newaxis] | tiny[np.newaxis, :]
        tiny_covariance_error = float(np.max(np.abs(full_covariance_error[tiny_involved]), initial=0.0))
        tiny_mean_error = float(np.max(np.abs(tiny_mean), initial=0.0))
    else:
        tiny_mean_error = 0.0
        tiny_covariance_error = 0.0

    active_mean_limit = MOMENT_MEAN_LIMIT * scale
    active_covariance_limit = MOMENT_COVARIANCE_LIMIT * scale
    tiny_mean_limit = MOMENT_MEAN_LIMIT * scale * math.sqrt(eigenvalue_threshold)
    tiny_covariance_limit = MOMENT_COVARIANCE_LIMIT * scale * eigenvalue_threshold
    checks = {
        "active_normalized_mean": active_mean_error <= active_mean_limit,
        "active_relative_covariance": active_covariance_error <= active_covariance_limit,
        "tiny_absolute_mean": tiny_mean_error <= tiny_mean_limit,
        "tiny_absolute_covariance": tiny_covariance_error <= tiny_covariance_limit,
    }
    return {
        "sample_count": sample_count,
        "rank": rank,
        "h_N": scale,
        "eigenvalue_threshold": eigenvalue_threshold,
        "active_coordinate_count": int(np.count_nonzero(active)),
        "tiny_coordinate_count": int(np.count_nonzero(tiny)),
        "active_maximum_absolute_normalized_mean": active_mean_error,
        "active_normalized_mean_limit": active_mean_limit,
        "active_relative_covariance_frobenius_per_sqrt_coordinate": (active_covariance_error),
        "active_relative_covariance_limit": active_covariance_limit,
        "tiny_maximum_absolute_mean": tiny_mean_error,
        "tiny_absolute_mean_limit": tiny_mean_limit,
        "tiny_maximum_absolute_covariance_error_including_active_tiny_cross_terms": (tiny_covariance_error),
        "tiny_absolute_covariance_limit": tiny_covariance_limit,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _ks_maximum(first: FloatArray, second: FloatArray) -> tuple[float, int]:
    """Return the maximum coordinatewise two-sample KS statistic."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("KS inputs must be sample-by-coordinate matrices")
    values = np.asarray(
        [
            float(
                cast(
                    Any,
                    stats.ks_2samp(
                        first[:, index],
                        second[:, index],
                        alternative="two-sided",
                        method="asymp",
                    ),
                ).statistic
            )
            for index in range(first.shape[1])
        ],
        dtype=np.float64,
    )
    worst = int(np.argmax(values)) if values.size else -1
    return float(values[worst]) if values.size else 0.0, worst


def _maximum_one_sided_difference(
    first: FloatArray,
    second: FloatArray,
    threshold: float,
) -> tuple[float, tuple[int, str]]:
    """Return the maximum coordinatewise upper/lower tail probability change."""
    upper = np.abs(np.mean(first >= threshold, axis=0) - np.mean(second >= threshold, axis=0))
    lower = np.abs(np.mean(first <= -threshold, axis=0) - np.mean(second <= -threshold, axis=0))
    combined = np.stack((upper, lower), axis=0)
    flat = int(np.argmax(combined)) if combined.size else 0
    sign, coordinate = np.unravel_index(flat, combined.shape)
    return float(combined[sign, coordinate]), (int(coordinate), "upper" if sign == 0 else "lower")


def _tail_metrics(
    first_locations: FloatArray,
    second_locations: FloatArray,
    eigenvalues: FloatArray,
    *,
    rank: int,
    eigenvalue_threshold: float,
) -> dict[str, object]:
    """Compare every frozen marginal, joint, and radial tail diagnostic."""
    lambdas = np.asarray(eigenvalues[:rank], dtype=np.float64)
    active_indices = np.flatnonzero(lambdas > eigenvalue_threshold)
    if active_indices.size == 0:
        raise ValueError("tail diagnostics require at least one active coordinate")
    first = np.asarray(first_locations[:, :rank], dtype=np.float64)
    second = np.asarray(second_locations[:, :rank], dtype=np.float64)
    first_x = first[:, active_indices] / np.sqrt(lambdas[active_indices])[np.newaxis, :]
    second_x = second[:, active_indices] / np.sqrt(lambdas[active_indices])[np.newaxis, :]
    ks_value, ks_index = _ks_maximum(first_x, second_x)
    tail_2, tail_2_where = _maximum_one_sided_difference(first_x, second_x, 2.0)
    tail_3, tail_3_where = _maximum_one_sided_difference(first_x, second_x, 3.0)

    active_lookup = {int(original): compact for compact, original in enumerate(active_indices)}
    pair_records: list[dict[str, object]] = []
    joint_maximum = 0.0
    joint_signed = 0.0
    for one_based in JOINT_PAIRS_ONE_BASED:
        original = (one_based[0] - 1, one_based[1] - 1)
        if original[1] >= rank or any(index not in active_lookup for index in original):
            continue
        pair = (active_lookup[original[0]], active_lookup[original[1]])
        first_pair = first_x[:, pair]
        second_pair = second_x[:, pair]
        maximum_event = abs(
            float(np.mean(np.max(np.abs(first_pair), axis=1) >= 2.0))
            - float(np.mean(np.max(np.abs(second_pair), axis=1) >= 2.0))
        )
        signed_differences = []
        for first_sign, second_sign in itertools.product((-1.0, 1.0), repeat=2):
            first_event = (first_sign * first_pair[:, 0] >= 2.0) & (second_sign * first_pair[:, 1] >= 2.0)
            second_event = (first_sign * second_pair[:, 0] >= 2.0) & (second_sign * second_pair[:, 1] >= 2.0)
            signed_differences.append(abs(float(np.mean(first_event)) - float(np.mean(second_event))))
        signed_maximum = max(signed_differences)
        joint_maximum = max(joint_maximum, maximum_event)
        joint_signed = max(joint_signed, signed_maximum)
        pair_records.append(
            {
                "coordinates_one_based": list(one_based),
                "max_abs_at_least_2_probability_difference": maximum_event,
                "maximum_signed_threshold_2_probability_difference": signed_maximum,
            }
        )

    radial_records: list[dict[str, object]] = []
    radial_ks = 0.0
    radial_99 = 0.0
    radial_999 = 0.0
    for dimension in RADIAL_DIMENSIONS:
        if dimension > rank or not np.all(lambdas[:dimension] > eigenvalue_threshold):
            continue
        first_radius = np.sum(np.square(first[:, :dimension] / np.sqrt(lambdas[:dimension])), axis=1)
        second_radius = np.sum(
            np.square(second[:, :dimension] / np.sqrt(lambdas[:dimension])),
            axis=1,
        )
        ks = float(
            cast(
                Any,
                stats.ks_2samp(
                    first_radius,
                    second_radius,
                    alternative="two-sided",
                    method="asymp",
                ),
            ).statistic
        )
        threshold_99 = float(stats.chi2.ppf(0.99, dimension))
        threshold_999 = float(stats.chi2.ppf(0.999, dimension))
        difference_99 = abs(
            float(np.mean(first_radius > threshold_99)) - float(np.mean(second_radius > threshold_99))
        )
        difference_999 = abs(
            float(np.mean(first_radius > threshold_999)) - float(np.mean(second_radius > threshold_999))
        )
        radial_ks = max(radial_ks, ks)
        radial_99 = max(radial_99, difference_99)
        radial_999 = max(radial_999, difference_999)
        radial_records.append(
            {
                "dimension": dimension,
                "ks": ks,
                "chi_square_99_probability_difference": difference_99,
                "chi_square_999_probability_difference": difference_999,
            }
        )

    checks = {
        "coordinatewise_ks": ks_value <= KS_LIMIT,
        "one_sided_tail_2": tail_2 <= TAIL_2_LIMIT,
        "one_sided_tail_3": tail_3 <= TAIL_3_LIMIT,
        "joint_max_abs_tail": joint_maximum <= JOINT_MAX_LIMIT,
        "joint_signed_tail": joint_signed <= JOINT_SIGNED_LIMIT,
        "radial_ks": radial_ks <= KS_LIMIT,
        "radial_chi_square_99": radial_99 <= RADIAL_99_LIMIT,
        "radial_chi_square_999": radial_999 <= RADIAL_999_LIMIT,
    }
    return {
        "rank": rank,
        "active_coordinate_count": int(active_indices.size),
        "coordinatewise_ks_maximum": ks_value,
        "coordinatewise_ks_worst_one_based": (int(active_indices[ks_index]) + 1 if ks_index >= 0 else None),
        "one_sided_tail_2_maximum_probability_difference": tail_2,
        "one_sided_tail_2_worst": {
            "coordinate_one_based": int(active_indices[tail_2_where[0]]) + 1,
            "side": tail_2_where[1],
        },
        "one_sided_tail_3_maximum_probability_difference": tail_3,
        "one_sided_tail_3_worst": {
            "coordinate_one_based": int(active_indices[tail_3_where[0]]) + 1,
            "side": tail_3_where[1],
        },
        "joint_max_abs_at_least_2_maximum_probability_difference": joint_maximum,
        "joint_signed_threshold_2_maximum_probability_difference": joint_signed,
        "joint_pairs": pair_records,
        "radial_ks_maximum": radial_ks,
        "radial_chi_square_99_maximum_probability_difference": radial_99,
        "radial_chi_square_999_maximum_probability_difference": radial_999,
        "radial_records": radial_records,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _direct_hybrid_log_likelihoods(
    centered_observations: FloatArray,
    root_masses: FloatArray,
    locations: FloatArray,
    spectrum: RootResidualSpectrum,
    *,
    sample_counts: Sequence[int] = S_LADDER,
    ranks: Sequence[int] = Q_LADDER,
) -> FloatArray:
    """Evaluate direct finite-source plus analytic-complement log likelihoods.

    The result has axes ``(rank, sample_count, grid_state)``.  The leading
    empirical mixture is normalized by its exact equal-weight sample count.
    No component covariance tensor is allocated.
    """
    observations = np.asarray(centered_observations, dtype=np.float64)
    masses = np.asarray(root_masses, dtype=np.float64)
    source = np.asarray(locations, dtype=np.float64)
    counts = tuple(int(value) for value in sample_counts)
    retained_ranks = tuple(int(value) for value in ranks)
    if (
        observations.ndim != 2
        or masses.shape != (observations.shape[0],)
        or source.ndim != 2
        or not counts
        or not retained_ranks
        or tuple(sorted(set(counts))) != counts
        or tuple(sorted(set(retained_ranks))) != retained_ranks
        or counts[-1] > source.shape[0]
        or retained_ranks[-1] > source.shape[1]
        or retained_ranks[-1] > spectrum.retained_rank
        or observations.shape[1] != spectrum.observation_mean_design.size
        or np.any(masses <= 0.0)
        or not np.all(np.isfinite(observations))
        or not np.all(np.isfinite(source))
    ):
        raise ValueError("direct hybrid likelihood inputs are invalid")

    whitened = (
        observations - masses[:, np.newaxis] * spectrum.observation_mean_design[np.newaxis, :]
    ) / spectrum.noise_sd[np.newaxis, :]
    coordinates = whitened @ spectrum.basis
    orthogonal = whitened - coordinates @ spectrum.basis.T
    common = -float(np.sum(np.log(spectrum.noise_sd)))
    common_values = common - 0.5 * (
        (spectrum.observation_mean_design.size - spectrum.retained_rank) * math.log(2.0 * math.pi)
        + np.sum(np.square(orthogonal), axis=1)
    )
    result = np.empty(
        (len(retained_ranks), len(counts), observations.shape[0]),
        dtype=np.float64,
    )
    maximum_count = counts[-1]
    for grid_index, mass in enumerate(masses):
        squared_distance = np.zeros(maximum_count, dtype=np.float64)
        previous_rank = 0
        for rank_index, rank in enumerate(retained_ranks):
            block = source[:maximum_count, previous_rank:rank]
            target = coordinates[grid_index, previous_rank:rank]
            squared_distance += np.sum(
                np.square(mass * block - target[np.newaxis, :]),
                axis=1,
            )
            log_kernel = -0.5 * (rank * math.log(2.0 * math.pi) + squared_distance)
            tail_variances = 1.0 + mass * mass * spectrum.eigenvalues[rank:]
            tail_coordinates = coordinates[grid_index, rank:]
            tail = -0.5 * float(
                np.sum(np.log(2.0 * math.pi * tail_variances) + np.square(tail_coordinates) / tail_variances)
            )
            for count_index, count in enumerate(counts):
                leading = float(cast(Real, special.logsumexp(log_kernel[:count]))) - math.log(count)
                result[rank_index, count_index, grid_index] = common_values[grid_index] + leading + tail
            previous_rank = rank
    if not np.all(np.isfinite(result)):
        raise ValueError("direct hybrid likelihood produced non-finite values")
    return result


def _direct_hybrid_log_likelihoods_with_offset(
    observations: FloatArray,
    offsets: FloatArray,
    root_masses: FloatArray,
    locations: FloatArray,
    spectrum: RootResidualSpectrum,
    *,
    sample_counts: Sequence[int] = S_LADDER,
    ranks: Sequence[int] = Q_LADDER,
) -> FloatArray:
    """Evaluate the direct hybrid density through the explicit offset path."""
    observed = np.asarray(observations, dtype=np.float64)
    offset = np.asarray(offsets, dtype=np.float64)
    if observed.shape != offset.shape:
        raise ValueError("observations and offsets must have identical shapes")
    return _direct_hybrid_log_likelihoods(
        observed - offset,
        root_masses,
        locations,
        spectrum,
        sample_counts=sample_counts,
        ranks=ranks,
    )


def _difference_metrics(
    first: FloatArray,
    second: FloatArray,
) -> dict[str, object]:
    """Return the frozen median/P99 absolute likelihood-difference gate."""
    difference = np.abs(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64))
    if difference.shape != (GRID_SIZE,) or not np.all(np.isfinite(difference)):
        raise ValueError("likelihood difference grid is invalid")
    median = float(np.median(difference))
    p99 = float(np.quantile(difference, 0.99))
    return {
        "median_absolute_difference_nat": median,
        "p99_absolute_difference_nat": p99,
        "median_limit_nat": LIKELIHOOD_MEDIAN_LIMIT_NAT,
        "p99_limit_nat": LIKELIHOOD_P99_LIMIT_NAT,
        "passed": (median <= LIKELIHOOD_MEDIAN_LIMIT_NAT and p99 <= LIKELIHOOD_P99_LIMIT_NAT),
    }


def _strict_development_reference(
    manifest_path: Path,
    *,
    source_revision: str,
    spectrum_manifest: Path,
    allocation_chunk: int,
    projection_microbatch: int,
) -> tuple[dict[str, Any], FloatArray]:
    """Load the selected G3 scientific development-bank reference."""
    payload = hpc._read_json(manifest_path)
    projected = payload.get("projected_array")
    if (
        payload.get("schema") != hpc.SCHEMA
        or payload.get("stage") != "G3b-candidate"
        or payload.get("source_revision") != source_revision
        or payload.get("source_seed") != SOURCE_SEEDS[0]
        or payload.get("sample_count") != S_LADDER[-1]
        or payload.get("mixture_rank") != Q_LADDER[-1]
        or payload.get("sample_chunk_size") != allocation_chunk
        or payload.get("projection_chunk_size") != projection_microbatch
        or not math.isclose(
            float(payload.get("native_concentration", math.nan)),
            hpc.SCIENTIFIC_CONCENTRATION,
            rel_tol=0.0,
            abs_tol=64.0 * np.finfo(np.float64).eps,
        )
        or payload.get("spectrum_manifest_sha256") != hpc._sha256_file(spectrum_manifest)
        or not isinstance(projected, dict)
        or payload.get("passed_internal_checks") is not True
    ):
        raise ValueError("development reference is not the selected scientific G3 bank")
    return payload, _load_record(manifest_path.parent, projected, name="G3 projected array")


def _strict_g3_controls(
    decision_path: Path,
    *,
    source_revision: str,
) -> tuple[int, int, dict[str, Any]]:
    """Authenticate the passing G3 decision and return its selected C and P."""
    payload = hpc._read_json(decision_path)
    chunk = payload.get("selected_sample_chunk_size")
    microbatch = payload.get("selected_projection_microbatch")
    if (
        payload.get("schema") != hpc.SCHEMA
        or payload.get("stage") != "G3"
        or payload.get("source_revision") != source_revision
        or payload.get("passed") is not True
        or payload.get("native_concentration") != hpc.SCIENTIFIC_CONCENTRATION
        or payload.get("root_variance") != hpc.SCIENTIFIC_ROOT_VARIANCE
        or payload.get("science_calibration_schema") != hpc.SCIENCE_CALIBRATION_SCHEMA
        or isinstance(chunk, bool)
        or not isinstance(chunk, int)
        or chunk not in hpc.RESOURCE_C_LADDER
        or isinstance(microbatch, bool)
        or not isinstance(microbatch, int)
        or microbatch not in hpc.P_LADDER
    ):
        raise ValueError("G3 decision does not carry passing selected C/P controls")
    return chunk, microbatch, payload


def _rank_key(rank: int) -> str:
    return str(rank)


def _sample_key(sample_count: int) -> str:
    return str(sample_count)


def run_seed(
    output_dir: Path,
    *,
    input_path: Path,
    expected_input_sha256: str,
    spectrum_manifest: Path,
    grid_manifest: Path,
    g3_decision: Path,
    source_revision: str,
    source_seed: int,
    development_reference_manifest: Path | None,
) -> dict[str, object]:
    """Build, publish, and score one complete G4 source-seed shard."""
    _require_output_directory(output_dir)
    if source_seed not in SOURCE_SEEDS:
        raise ValueError("source seed is outside the frozen G4 seed set")
    if (source_seed == SOURCE_SEEDS[0]) != (development_reference_manifest is not None):
        raise ValueError("only the development seed requires a G3 reference manifest")
    started = time.perf_counter()
    spectrum, _ = _strict_spectrum(spectrum_manifest, source_revision=source_revision)
    allocation_chunk, projection_microbatch, g3_payload = _strict_g3_controls(
        g3_decision,
        source_revision=source_revision,
    )
    grid, _ = _load_grid(
        grid_manifest,
        source_revision=source_revision,
        spectrum_manifest=spectrum_manifest,
    )
    _, aggregation = hpc._build_paris_aggregation(
        input_path,
        expected_input_sha256=expected_input_sha256,
        concentration=hpc.SCIENTIFIC_CONCENTRATION,
    )

    constructor_started = time.perf_counter()
    bank = build_chunked_projected_root_bank(
        aggregation,
        spectrum,
        mixture_rank=Q_LADDER[-1],
        sample_count=S_LADDER[-1],
        sample_chunk_size=allocation_chunk,
        projection_chunk_size=projection_microbatch,
        source_seed=source_seed,
        source_provenance=f"G4 scientific source seed {source_seed}",
    )
    constructor_seconds = time.perf_counter() - constructor_started
    locations = np.asarray(
        bank.projected_unit_mass_residual_factors[:, :, 0],
        dtype=np.float64,
    )
    source_record = _array_record(output_dir / "projected_locations.npy", locations)

    prefix_records: dict[str, object] = {}
    prefix_identity_passed = True
    for sample_count in S_LADDER[:-1]:
        prefix_bank = build_chunked_projected_root_bank(
            aggregation,
            spectrum,
            mixture_rank=Q_LADDER[-1],
            sample_count=sample_count,
            sample_chunk_size=min(allocation_chunk, sample_count),
            projection_chunk_size=projection_microbatch,
            source_seed=source_seed,
            source_provenance=f"G4 scientific prefix source seed {source_seed} S={sample_count}",
        )
        prefix_values = np.asarray(
            prefix_bank.projected_unit_mass_residual_factors[:, :, 0],
            dtype=np.float64,
        )
        identical = np.array_equal(prefix_values, locations[:sample_count])
        prefix_identity_passed &= identical
        prefix_records[_sample_key(sample_count)] = {
            "bank_sha256": prefix_bank.sha256,
            "projected_array_sha256": hpc._array_sha256(prefix_values),
            "identical_to_maximum_bank_prefix": identical,
        }
    if not prefix_identity_passed:
        raise ValueError("independently reconstructed source prefixes are not exact")

    development_rebuild: dict[str, object] | None = None
    if development_reference_manifest is not None:
        reference_payload, reference_values = _strict_development_reference(
            development_reference_manifest,
            source_revision=source_revision,
            spectrum_manifest=spectrum_manifest,
            allocation_chunk=allocation_chunk,
            projection_microbatch=projection_microbatch,
        )
        reference_array = reference_payload["projected_array"]
        if reference_array["array_sha256"] != g3_payload.get("projected_array_sha256") or reference_array[
            "file_sha256"
        ] != g3_payload.get("binary_file_sha256"):
            raise ValueError("development reference does not match the selected G3 digest")
        array_identical = np.array_equal(reference_values, locations)
        digest_identical = source_record["array_sha256"] == reference_array["array_sha256"]
        file_identical = source_record["file_sha256"] == reference_array["file_sha256"]
        development_rebuild = {
            "reference_manifest": str(development_reference_manifest),
            "reference_manifest_sha256": hpc._sha256_file(development_reference_manifest),
            "array_bitwise_identical": array_identical,
            "array_digest_identical": digest_identical,
            "binary_file_digest_identical": file_identical,
            "passed": array_identical and digest_identical and file_identical,
        }
        if development_rebuild["passed"] is not True:
            raise ValueError("development source rebuild does not match the selected G3 bank")

    threshold = _eigenvalue_threshold(spectrum)
    moments: dict[str, object] = {}
    for rank in Q_LADDER:
        rank_records = {}
        for sample_count in S_LADDER:
            rank_records[_sample_key(sample_count)] = _moment_metrics(
                locations,
                spectrum.eigenvalues,
                sample_count=sample_count,
                rank=rank,
                eigenvalue_threshold=threshold,
            )
        moments[_rank_key(rank)] = rank_records

    nested_tails: dict[str, object] = {}
    for rank in Q_LADDER:
        comparisons: dict[str, object] = {}
        for lower, upper in zip(S_LADDER[:-1], S_LADDER[1:], strict=True):
            comparisons[f"{lower}-vs-{upper}"] = _tail_metrics(
                locations[:lower],
                locations[:upper],
                spectrum.eigenvalues,
                rank=rank,
                eigenvalue_threshold=threshold,
            )
        nested_tails[_rank_key(rank)] = comparisons

    likelihood = _direct_hybrid_log_likelihoods_with_offset(
        grid["observation"],
        grid["offset"],
        grid["root_mass"],
        locations,
        spectrum,
    )
    translated_likelihood = _direct_hybrid_log_likelihoods_with_offset(
        grid["observation"] - grid["offset"],
        np.zeros_like(grid["offset"]),
        grid["root_mass"],
        locations,
        spectrum,
    )
    translation_difference = float(np.max(np.abs(likelihood - translated_likelihood), initial=0.0))
    translation_scale = max(
        1.0,
        float(np.max(np.abs(likelihood), initial=0.0)),
        float(np.max(np.abs(translated_likelihood), initial=0.0)),
    )
    translation_tolerance = TRANSLATION_FACTOR * np.finfo(np.float64).eps * translation_scale
    translation_passed = translation_difference <= translation_tolerance
    likelihood_record = _array_record(output_dir / "log_likelihood.npy", likelihood)

    nested_likelihood: dict[str, object] = {}
    rank_vs_maximum: dict[str, object] = {}
    finite_support: dict[str, object] = {}
    normalization_error = abs(
        float(
            cast(
                Real,
                special.logsumexp(np.zeros(S_LADDER[-1], dtype=np.float64)),
            )
        )
        - math.log(S_LADDER[-1])
    )
    normalization_limit = TRANSLATION_FACTOR * np.finfo(np.float64).eps
    for rank_index, rank in enumerate(Q_LADDER):
        comparisons = {}
        for count_index, sample_count in enumerate(S_LADDER[:-1]):
            comparisons[f"{sample_count}-vs-{S_LADDER[-1]}"] = _difference_metrics(
                likelihood[rank_index, count_index],
                likelihood[rank_index, -1],
            )
        nested_likelihood[_rank_key(rank)] = comparisons
        rank_vs_maximum[_rank_key(rank)] = _difference_metrics(
            likelihood[rank_index, -1],
            likelihood[-1, -1],
        )
        finite_support[_rank_key(rank)] = {
            "all_likelihoods_finite": bool(np.all(np.isfinite(likelihood[rank_index]))),
            "all_root_masses_positive": bool(np.all(grid["root_mass"] > 0.0)),
            "equal_weight_normalization_error": normalization_error,
            "equal_weight_normalization_limit": normalization_limit,
            "passed": bool(
                np.all(np.isfinite(likelihood[rank_index]))
                and np.all(grid["root_mass"] > 0.0)
                and normalization_error <= normalization_limit
            ),
        }

    rank_decisions: dict[str, object] = {}
    for rank in Q_LADDER:
        key = _rank_key(rank)
        all_moments = all(
            bool(record["passed"])
            for record in moments[key].values()  # type: ignore[union-attr]
        )
        maximum_moment = bool(moments[key][_sample_key(S_LADDER[-1])]["passed"])  # type: ignore[index]
        all_nested_tails = all(
            bool(record["passed"])
            for record in nested_tails[key].values()  # type: ignore[union-attr]
        )
        all_nested_likelihood = all(
            bool(record["passed"])
            for record in nested_likelihood[key].values()  # type: ignore[union-attr]
        )
        common = bool(
            all_nested_tails
            and all_nested_likelihood
            and rank_vs_maximum[key]["passed"]  # type: ignore[index]
            and finite_support[key]["passed"]  # type: ignore[index]
            and translation_passed
            and prefix_identity_passed
        )
        rank_decisions[key] = {
            "all_sample_count_moments_passed": all_moments,
            "maximum_sample_count_moments_passed": maximum_moment,
            "all_adjacent_nested_tail_gates_passed": all_nested_tails,
            "all_nested_likelihood_gates_passed": all_nested_likelihood,
            "rank_vs_128_likelihood_gate_passed": bool(
                rank_vs_maximum[key]["passed"]  # type: ignore[index]
            ),
            "development_passed": common and all_moments,
            "within_seed_passed": common and all_moments,
        }

    report: dict[str, object] = {
        "schema": SCHEMA,
        "stage": "G4-seed",
        "source_revision": source_revision,
        "input_sha256": expected_input_sha256,
        "source_seed": source_seed,
        "sample_count_ladder": list(S_LADDER),
        "rank_ladder": list(Q_LADDER),
        "projection_microbatch": projection_microbatch,
        "allocation_chunk": allocation_chunk,
        "g3_decision": {
            "path": str(g3_decision),
            "sha256": hpc._sha256_file(g3_decision),
        },
        "threshold_supplement": _threshold_supplement_record(),
        "scientific_concentration": hpc.SCIENTIFIC_CONCENTRATION,
        "scientific_root_variance": hpc.SCIENTIFIC_ROOT_VARIANCE,
        "spectrum_manifest": {
            "path": str(spectrum_manifest),
            "sha256": hpc._sha256_file(spectrum_manifest),
        },
        "grid_manifest": {
            "path": str(grid_manifest),
            "sha256": hpc._sha256_file(grid_manifest),
        },
        "source_bank_sha256": bank.sha256,
        "source_array": source_record,
        "likelihood_array": likelihood_record,
        "independent_prefix_reconstructions": prefix_records,
        "all_prefix_identities_exact": prefix_identity_passed,
        "development_rebuild": development_rebuild,
        "eigenvalue_threshold": {
            "value": threshold,
            "formula": "max(spectrum.eigenvalue_tolerance,1024*eps64*max(1,lambda_1))",
            "tiny_coordinates_excluded_only_from_relative_division": True,
        },
        "moment_metrics": moments,
        "nested_tail_metrics": nested_tails,
        "nested_likelihood_metrics": nested_likelihood,
        "rank_vs_128_likelihood_metrics": rank_vs_maximum,
        "finite_normalization_support": finite_support,
        "translation_parity": {
            "maximum_log_likelihood_difference_nat": translation_difference,
            "scale": translation_scale,
            "tolerance": translation_tolerance,
            "formula": "4096*eps64*max(1,max_abs(logp(y,b)),max_abs(logp(y-b,0)))",
            "passed": translation_passed,
        },
        "rank_decisions": rank_decisions,
        "direct_likelihood": {
            "leading_coordinates": "equal-weight finite source location mixture",
            "complement": "analytic Gaussian moment closure for q:r",
            "beyond_spectrum": "measurement noise",
            "singleton_compressed_root_mixture_constructed": False,
            "operational_leakage_thresholds_not_exact_accuracy": True,
        },
        "constructor_seconds": constructor_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": hpc._peak_rss_bytes(),
        "realized_mf_used": False,
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "passed_internal_checks": True,
    }
    hpc._atomic_write_json(output_dir / "seed_report.json", report)
    hpc._atomic_write_text(
        output_dir / "G4_SEED_COMPLETE.txt",
        f"G4 seed {source_seed} complete for {source_revision}\n",
    )
    return report


def _strict_seed_report(
    path: Path,
    *,
    source_revision: str,
    spectrum_manifest: Path,
    grid_manifest: Path,
) -> tuple[dict[str, Any], FloatArray, FloatArray]:
    """Load one complete seed report and its authenticated numeric artifacts."""
    payload = hpc._read_json(path)
    spectrum_record = payload.get("spectrum_manifest")
    grid_record = payload.get("grid_manifest")
    g3_record = payload.get("g3_decision")
    threshold_record = payload.get("threshold_supplement")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("stage") != "G4-seed"
        or payload.get("source_revision") != source_revision
        or payload.get("source_seed") not in SOURCE_SEEDS
        or payload.get("sample_count_ladder") != list(S_LADDER)
        or payload.get("rank_ladder") != list(Q_LADDER)
        or payload.get("passed_internal_checks") is not True
        or not isinstance(spectrum_record, dict)
        or spectrum_record.get("sha256") != hpc._sha256_file(spectrum_manifest)
        or not isinstance(grid_record, dict)
        or grid_record.get("sha256") != hpc._sha256_file(grid_manifest)
        or not isinstance(g3_record, dict)
        or not isinstance(g3_record.get("path"), str)
        or not isinstance(g3_record.get("sha256"), str)
        or threshold_record != _threshold_supplement_record()
    ):
        raise ValueError(f"G4 seed report identity mismatch: {path}")
    g3_path = Path(g3_record["path"])
    allocation_chunk, projection_microbatch, _ = _strict_g3_controls(
        g3_path,
        source_revision=source_revision,
    )
    if (
        hpc._sha256_file(g3_path) != g3_record["sha256"]
        or payload.get("allocation_chunk") != allocation_chunk
        or payload.get("projection_microbatch") != projection_microbatch
    ):
        raise ValueError(f"G4 seed report G3 control mismatch: {path}")
    locations = _load_record(path.parent, payload.get("source_array"), name="source array")
    likelihood = _load_record(
        path.parent,
        payload.get("likelihood_array"),
        name="likelihood array",
    )
    if locations.shape != (S_LADDER[-1], Q_LADDER[-1]) or likelihood.shape != (
        len(Q_LADDER),
        len(S_LADDER),
        GRID_SIZE,
    ):
        raise ValueError(f"G4 seed report array shape mismatch: {path}")
    return payload, locations, likelihood


def _common_suffix(passing: Mapping[int, bool]) -> tuple[int, ...]:
    """Return the smallest all-passing q suffix with at least two members."""
    if set(passing) != set(Q_LADDER):
        raise ValueError("rank decision map does not cover the frozen q ladder")
    for start in range(len(Q_LADDER) - MINIMUM_SUFFIX_LENGTH + 1):
        suffix = Q_LADDER[start:]
        if len(suffix) >= MINIMUM_SUFFIX_LENGTH and all(passing[rank] for rank in suffix):
            return suffix
    return ()


def run_development_certify(
    output: Path,
    completion_marker: Path,
    *,
    seed_report: Path,
    source_revision: str,
    spectrum_manifest: Path,
    grid_manifest: Path,
) -> dict[str, object]:
    """Select the smallest predeclared common passing q suffix for seed 731."""
    payload, _, _ = _strict_seed_report(
        seed_report,
        source_revision=source_revision,
        spectrum_manifest=spectrum_manifest,
        grid_manifest=grid_manifest,
    )
    if payload.get("source_seed") != SOURCE_SEEDS[0]:
        raise ValueError("development certifier requires source seed 731")
    rebuild = payload.get("development_rebuild")
    decisions = payload.get("rank_decisions")
    if not isinstance(rebuild, dict) or rebuild.get("passed") is not True:
        raise ValueError("development seed has no exact selected-G3 rebuild")
    if not isinstance(decisions, dict):
        raise ValueError("development seed has no rank decisions")
    passing = {rank: bool(decisions.get(_rank_key(rank), {}).get("development_passed")) for rank in Q_LADDER}
    suffix = _common_suffix(passing)
    passed = bool(suffix)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "stage": "G4-development",
        "source_revision": source_revision,
        "seed_report": {
            "path": str(seed_report),
            "sha256": hpc._sha256_file(seed_report),
            "source_seed": SOURCE_SEEDS[0],
        },
        "rank_passes": {_rank_key(rank): passing[rank] for rank in Q_LADDER},
        "suffix_rule": (
            "smallest q starting an all-larger passing suffix with at least "
            f"{MINIMUM_SUFFIX_LENGTH} consecutive q values"
        ),
        "passing_suffix": list(suffix),
        "selected_rank": suffix[0] if suffix else None,
        "passed": passed,
        "next_gate": ("G4-all-seed-confirmation" if passed else "terminal-G4-development-hard-stop"),
    }
    hpc._atomic_write_json(output, report)
    if passed:
        hpc._atomic_write_text(
            completion_marker,
            f"G4 development passed for {source_revision}; q={suffix[0]}\n",
        )
    return report


def run_all_seed_certify(
    output: Path,
    lock_marker: Path,
    *,
    development_report: Path,
    seed_reports: Sequence[Path],
    source_revision: str,
    spectrum_manifest: Path,
    grid_manifest: Path,
) -> dict[str, object]:
    """Apply every confirmation, between-seed tail, and likelihood gate."""
    development = hpc._read_json(development_report)
    if (
        development.get("schema") != SCHEMA
        or development.get("stage") != "G4-development"
        or development.get("source_revision") != source_revision
        or development.get("passed") is not True
    ):
        raise ValueError("G4 development certificate is not passing")
    raw_suffix = development.get("passing_suffix")
    if not isinstance(raw_suffix, list) or tuple(raw_suffix) not in tuple(
        Q_LADDER[index:] for index in range(len(Q_LADDER) - 1)
    ):
        raise ValueError("G4 development certificate has an invalid suffix")
    suffix = tuple(int(value) for value in raw_suffix)
    if len(seed_reports) != len(SOURCE_SEEDS):
        raise ValueError("all-seed certifier requires exactly four seed reports")

    reports: dict[int, dict[str, Any]] = {}
    locations: dict[int, FloatArray] = {}
    likelihoods: dict[int, FloatArray] = {}
    report_identities: dict[str, object] = {}
    for path in seed_reports:
        payload, source, likelihood = _strict_seed_report(
            path,
            source_revision=source_revision,
            spectrum_manifest=spectrum_manifest,
            grid_manifest=grid_manifest,
        )
        seed = int(payload["source_seed"])
        if seed in reports:
            raise ValueError("G4 seed report is duplicated")
        reports[seed] = payload
        locations[seed] = source
        likelihoods[seed] = likelihood
        report_identities[str(seed)] = {
            "path": str(path),
            "sha256": hpc._sha256_file(path),
        }
    if set(reports) != set(SOURCE_SEEDS):
        raise ValueError("G4 seed report set is incomplete")
    g3_digests = {str(reports[seed]["g3_decision"]["sha256"]) for seed in SOURCE_SEEDS}
    if len(g3_digests) != 1:
        raise ValueError("G4 seed reports do not share one G3 decision identity")

    within_seed: dict[str, object] = {}
    all_within = True
    for seed in SOURCE_SEEDS:
        decisions = reports[seed].get("rank_decisions")
        if not isinstance(decisions, dict):
            raise ValueError("seed report has no rank decisions")
        rank_results = {
            _rank_key(rank): bool(decisions.get(_rank_key(rank), {}).get("within_seed_passed"))
            for rank in suffix
        }
        seed_passed = all(rank_results.values())
        all_within &= seed_passed
        within_seed[str(seed)] = {
            "rank_passes": rank_results,
            "passed": seed_passed,
        }

    spectrum, _ = _strict_spectrum(spectrum_manifest, source_revision=source_revision)
    threshold = _eigenvalue_threshold(spectrum)
    pairwise_tails: dict[str, object] = {}
    all_pairwise = True
    for first_seed, second_seed in itertools.combinations(SOURCE_SEEDS, 2):
        rank_records = {}
        for rank in suffix:
            record = _tail_metrics(
                locations[first_seed],
                locations[second_seed],
                spectrum.eigenvalues,
                rank=rank,
                eigenvalue_threshold=threshold,
            )
            rank_records[_rank_key(rank)] = record
            all_pairwise &= bool(record["passed"])
        pairwise_tails[f"{first_seed}-vs-{second_seed}"] = rank_records

    cross_seed_likelihood: dict[str, object] = {}
    all_cross_seed = True
    for rank in suffix:
        rank_index = Q_LADDER.index(rank)
        stacked = np.stack(
            [likelihoods[seed][rank_index, -1] for seed in SOURCE_SEEDS],
            axis=0,
        )
        record = _difference_metrics(np.max(stacked, axis=0), np.min(stacked, axis=0))
        cross_seed_likelihood[_rank_key(rank)] = record
        all_cross_seed &= bool(record["passed"])

    passed = all_within and all_pairwise and all_cross_seed
    report: dict[str, object] = {
        "schema": SCHEMA,
        "stage": "G4-all-seed",
        "source_revision": source_revision,
        "development_report": {
            "path": str(development_report),
            "sha256": hpc._sha256_file(development_report),
        },
        "seed_reports": report_identities,
        "common_g3_decision_sha256": next(iter(g3_digests)),
        "development_suffix": list(suffix),
        "selected_rank": suffix[0],
        "within_seed_confirmation": within_seed,
        "all_within_seed_gates_passed": all_within,
        "pairwise_maximum_sample_tail_metrics": pairwise_tails,
        "all_pairwise_tail_gates_passed": all_pairwise,
        "cross_seed_likelihood_range_metrics": cross_seed_likelihood,
        "all_cross_seed_likelihood_gates_passed": all_cross_seed,
        "all_seeds_must_pass_no_averaging": True,
        "passed": passed,
        "next_gate": "G5-clustering" if passed else "terminal-G4-confirmation-hard-stop",
    }
    hpc._atomic_write_json(output, report)
    if passed:
        hpc._atomic_write_text(
            lock_marker,
            f"G4 source lock for {source_revision}; q={suffix[0]}; seeds={SOURCE_SEEDS}\n",
        )
    return report


def _parser() -> argparse.ArgumentParser:
    """Build the four create-only G4 commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    grid = subparsers.add_parser("grid")
    grid.add_argument("--output-dir", type=Path, required=True)
    grid.add_argument("--input", type=Path, required=True)
    grid.add_argument("--expected-input-sha256", default=hpc.PARIS_INPUT_SHA256)
    grid.add_argument("--spectrum-manifest", type=Path, required=True)
    grid.add_argument("--source-revision", required=True)

    seed = subparsers.add_parser("seed")
    seed.add_argument("--output-dir", type=Path, required=True)
    seed.add_argument("--input", type=Path, required=True)
    seed.add_argument("--expected-input-sha256", default=hpc.PARIS_INPUT_SHA256)
    seed.add_argument("--spectrum-manifest", type=Path, required=True)
    seed.add_argument("--grid-manifest", type=Path, required=True)
    seed.add_argument("--g3-decision", type=Path, required=True)
    seed.add_argument("--source-revision", required=True)
    seed.add_argument("--source-seed", type=int, required=True)
    seed.add_argument("--development-reference-manifest", type=Path)

    development = subparsers.add_parser("development-certify")
    development.add_argument("--output", type=Path, required=True)
    development.add_argument("--completion-marker", type=Path, required=True)
    development.add_argument("--seed-report", type=Path, required=True)
    development.add_argument("--spectrum-manifest", type=Path, required=True)
    development.add_argument("--grid-manifest", type=Path, required=True)
    development.add_argument("--source-revision", required=True)

    confirmation = subparsers.add_parser("all-seed-certify")
    confirmation.add_argument("--output", type=Path, required=True)
    confirmation.add_argument("--lock-marker", type=Path, required=True)
    confirmation.add_argument("--development-report", type=Path, required=True)
    confirmation.add_argument("--seed-report", action="append", type=Path, required=True)
    confirmation.add_argument("--spectrum-manifest", type=Path, required=True)
    confirmation.add_argument("--grid-manifest", type=Path, required=True)
    confirmation.add_argument("--source-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one create-only G4 action."""
    arguments = _parser().parse_args(argv)
    source_revision = hpc._full_revision(arguments.source_revision)
    if arguments.command == "grid":
        run_grid(
            arguments.output_dir,
            input_path=arguments.input,
            expected_input_sha256=arguments.expected_input_sha256,
            spectrum_manifest=arguments.spectrum_manifest,
            source_revision=source_revision,
        )
    elif arguments.command == "seed":
        run_seed(
            arguments.output_dir,
            input_path=arguments.input,
            expected_input_sha256=arguments.expected_input_sha256,
            spectrum_manifest=arguments.spectrum_manifest,
            grid_manifest=arguments.grid_manifest,
            g3_decision=arguments.g3_decision,
            source_revision=source_revision,
            source_seed=arguments.source_seed,
            development_reference_manifest=arguments.development_reference_manifest,
        )
    elif arguments.command == "development-certify":
        report = run_development_certify(
            arguments.output,
            arguments.completion_marker,
            seed_report=arguments.seed_report,
            source_revision=source_revision,
            spectrum_manifest=arguments.spectrum_manifest,
            grid_manifest=arguments.grid_manifest,
        )
        if report["passed"] is not True:
            return 3
    else:
        report = run_all_seed_certify(
            arguments.output,
            arguments.lock_marker,
            development_report=arguments.development_report,
            seed_reports=arguments.seed_report,
            source_revision=source_revision,
            spectrum_manifest=arguments.spectrum_manifest,
            grid_manifest=arguments.grid_manifest,
        )
        if report["passed"] is not True:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
