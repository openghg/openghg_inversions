#!/usr/bin/env python3
"""Run the checksum-pinned C1 conditional-allocation tiny screen.

This driver compares exact two- and four-cell conditional-allocation
quadrature with a deterministic frozen Monte Carlo bank.  Every screen uses a
full-rank identity summary basis, so it isolates finite-bank Monte Carlo error
and contains no Gaussian-complement approximation.

The driver deliberately has no held-out operator/partition mode.  That
catalogue is represented here only by an opaque identifier and digest.  In
particular, this program cannot use an observed residual, approximate
evidence, or protected result to choose a basis, partition, dimension, or
bank size.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Literal, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from openghg_inversions.experimental.rjmcmc.aggregation_error import (
    FourCellAggregationOracle,
    TwoCellAggregationOracle,
    beta_quadrature,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (
    ConditionalAllocationMixture,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
Family = Literal["two_cell", "four_cell"]
Tiling = Literal["root", "row", "fine"]

SCHEMA = "rjmcmc-conditional-allocation-c1-tiny-screen-v1"
PROTOCOL = "conditional-allocation-c1-full-rank-bank-v1"
GRADIENT_STEP = 2.0**-14
A1_SOURCE_REVISION = "5bb41399e45b78954488e286da3f40371dcb956e"
A1_NUMERICAL_SOURCE_SHA256 = "eaa5df1afdac1404470f80ae334cc6f6d9c31efd57fc809349e4a10ac4fee4fb"
A1_DEFINITIONS_SHA256 = "aba3fda80ce34fe75066c7b4fb37775438640b0fe65119dfcaa7076529711636"

THRESHOLDS = {
    "median_absolute_conditional_log_likelihood_error_nat": 0.05,
    "p99_absolute_conditional_log_likelihood_error_nat": 0.2,
    "scaled_coordinate_gradient_error": 0.05,
    "absolute_log_evidence_error_nat": 0.05,
    "posterior_mean_error_reference_sd": 0.05,
    "posterior_sd_relative_error": 0.02,
    "interval_endpoint_error_reference_sd": 0.05,
    "between_bank_log_evidence_range_nat": 0.05,
}
MERGER_THRESHOLDS = {
    "exact_log_evidence_tower_range_nat": 1.0e-6,
    "approximate_log_evidence_tower_range_nat": 0.05,
    "structural_weight_total_variation": 0.01,
}

SMOKE_SAMPLE_COUNTS = (4_096,)
SMOKE_REPEAT_SEEDS = (731,)
DEVELOPMENT_SAMPLE_COUNTS = (64, 256, 1_024, 4_096, 16_384)
DEVELOPMENT_REPEAT_SEEDS = (731, 1_877, 4_099, 8_317)
DEVELOPMENT_SELECTION_SEED = DEVELOPMENT_REPEAT_SEEDS[0]
CONFIRMATION_SEEDS = DEVELOPMENT_REPEAT_SEEDS[1:]
HELD_OUT_CATALOGUE_ID = "a1-held-out-operator-partition-catalogue-v1"
HELD_OUT_CATALOGUE_SHA256 = "da6f9aeeec7d12b1b2915f7ba31291e05ec298fcfd1499c2e03450f02ff160b8"


@dataclass(frozen=True)
class Regime:
    """One exact A1 numerical regime, copied without numeric modification."""

    name: str
    shapes2: tuple[float, float]
    gamma_rate2: float
    design2: tuple[tuple[float, float], ...]
    observation2: tuple[float, ...]
    noise2: tuple[float, ...]
    projection2: tuple[float, float]
    shapes4: tuple[float, float, float, float]
    gamma_rate4: float
    design4: tuple[tuple[float, float, float, float], ...]
    observation4: tuple[float, ...]
    noise4: tuple[float, ...]
    projection4: tuple[float, float, float, float]
    total_order: int
    fraction_order: int


REGIMES = (
    Regime(
        name="near_gaussian",
        shapes2=(45.0, 55.0),
        gamma_rate2=100.0,
        design2=((1.0, 0.7), (0.2, 1.1), (-0.5, 0.3)),
        observation2=(0.93, 0.71, -0.08),
        noise2=(0.42, 0.55, 0.48),
        projection2=(1.0, -0.65),
        shapes4=(40.0, 35.0, 45.0, 30.0),
        gamma_rate4=150.0,
        design4=(
            (1.00, 0.82, 0.45, 0.30),
            (0.15, 0.42, 0.90, 1.10),
            (-0.50, -0.10, 0.35, 0.55),
            (0.70, 0.62, 0.78, 0.85),
        ),
        observation4=(0.72, 0.64, 0.04, 0.79),
        noise4=(0.40, 0.48, 0.45, 0.52),
        projection4=(1.0, -0.5, 0.25, -0.75),
        total_order=40,
        fraction_order=24,
    ),
    Regime(
        name="skewed",
        shapes2=(0.35, 4.0),
        gamma_rate2=4.35,
        design2=((1.8, 0.1), (-0.4, 1.2), (0.8, -0.3)),
        observation2=(0.44, 0.91, -0.08),
        noise2=(0.25, 0.32, 0.28),
        projection2=(1.0, -0.65),
        shapes4=(0.35, 4.0, 1.2, 8.0),
        gamma_rate4=13.55,
        design4=(
            (1.80, 0.10, 0.50, -0.20),
            (-0.40, 1.20, 0.20, 0.85),
            (0.80, -0.30, 1.45, 0.10),
            (0.20, 0.35, -0.15, 1.60),
        ),
        observation4=(0.23, 0.83, 0.36, 1.12),
        noise4=(0.22, 0.30, 0.26, 0.34),
        projection4=(1.0, -0.8, 0.6, -0.3),
        total_order=56,
        fraction_order=32,
    ),
    Regime(
        name="boundary_heavy",
        shapes2=(0.12, 0.18),
        gamma_rate2=0.30,
        design2=((2.0, 0.0), (0.0, 1.7), (1.0, -1.0)),
        observation2=(1.75, 0.08, 0.94),
        noise2=(0.12, 0.14, 0.13),
        projection2=(1.0, -0.65),
        shapes4=(0.15, 0.18, 0.20, 0.12),
        gamma_rate4=0.65,
        design4=(
            (2.00, 0.00, 0.10, 0.00),
            (0.00, 1.70, 0.00, 0.10),
            (0.05, 0.00, 1.90, 0.00),
            (0.00, 0.10, 0.00, 2.10),
        ),
        observation4=(1.62, 0.08, 0.13, 0.06),
        noise4=(0.12, 0.14, 0.13, 0.15),
        projection4=(1.0, -1.0, 0.5, -0.5),
        total_order=64,
        fraction_order=40,
    ),
    Regime(
        name="equal_footprint",
        shapes2=(0.7, 3.1),
        gamma_rate2=3.8,
        design2=((0.8, 0.8), (-0.3, -0.3), (1.2, 1.2)),
        observation2=(0.75, -0.22, 1.05),
        noise2=(0.30, 0.34, 0.38),
        projection2=(1.0, -0.65),
        shapes4=(0.7, 1.3, 2.1, 3.4),
        gamma_rate4=7.5,
        design4=(
            (0.80, 0.80, 0.80, 0.80),
            (-0.30, -0.30, -0.30, -0.30),
            (1.20, 1.20, 1.20, 1.20),
            (0.45, 0.45, 0.45, 0.45),
        ),
        observation4=(0.75, -0.22, 1.05, 0.39),
        noise4=(0.30, 0.34, 0.38, 0.32),
        projection4=(1.0, -0.5, 0.25, -0.75),
        total_order=40,
        fraction_order=24,
    ),
)

# This is the preserved development-only diagonal catalogue.
DEVELOPMENT_MATRIX = tuple(
    (regime, family, tiling)
    for regime in ("near_gaussian", "skewed", "boundary_heavy")
    for family, tiling in (
        ("two_cell", "root"),
        ("four_cell", "root"),
        ("four_cell", "row"),
    )
)
SMOKE_MATRIX = (("near_gaussian", "two_cell", "root"),)
CONTROL_CATALOGUE = (
    ("equal_footprint", "two_cell", "root"),
    ("equal_footprint", "four_cell", "root"),
    ("near_gaussian", "two_cell", "fine"),
    ("near_gaussian", "four_cell", "fine"),
)
RUNNABLE_CONTROL_MATRIX = (
    ("equal_footprint", "two_cell", "root"),
    ("equal_footprint", "four_cell", "root"),
    ("near_gaussian", "two_cell", "fine"),
)


def _canonical_json(payload: object) -> str:
    """Return strict canonical JSON."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(payload: object) -> str:
    """Return the SHA-256 identity of one canonical JSON value."""
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def a1_definitions_sha256() -> str:
    """Return the development-only A1 numerical-definition identity."""
    development_names = {case[0] for case in DEVELOPMENT_MATRIX}
    return _sha256_json(
        {
            "gradient_step": GRADIENT_STEP,
            "regimes": [asdict(regime) for regime in REGIMES if regime.name in development_names],
            "development_matrix": DEVELOPMENT_MATRIX,
        }
    )


def _regime(name: str) -> Regime:
    """Return one named regime or fail closed."""
    for regime in REGIMES:
        if regime.name == name:
            return regime
    raise ValueError(f"unknown regime: {name}")


def _case_arrays(
    regime: Regime,
    family: Family,
) -> tuple[FloatArray, float, FloatArray, FloatArray, FloatArray]:
    """Return native shapes, rate, design, observation, and noise."""
    if family == "two_cell":
        return (
            np.asarray(regime.shapes2, dtype=np.float64),
            regime.gamma_rate2,
            np.asarray(regime.design2, dtype=np.float64),
            np.asarray(regime.observation2, dtype=np.float64),
            np.asarray(regime.noise2, dtype=np.float64),
        )
    return (
        np.asarray(regime.shapes4, dtype=np.float64),
        regime.gamma_rate4,
        np.asarray(regime.design4, dtype=np.float64),
        np.asarray(regime.observation4, dtype=np.float64),
        np.asarray(regime.noise4, dtype=np.float64),
    )


def labels_for_tiling(family: Family, tiling: Tiling) -> IntArray:
    """Return A1's contiguous labels for one projection."""
    if family == "two_cell":
        if tiling == "root":
            return np.asarray([0, 0], dtype=np.int64)
        if tiling == "fine":
            return np.asarray([0, 1], dtype=np.int64)
    else:
        labels = {
            "root": [0, 0, 0, 0],
            "row": [0, 0, 1, 1],
            "fine": [0, 1, 2, 3],
        }
        if tiling in labels:
            return np.asarray(labels[tiling], dtype=np.int64)
    raise ValueError(f"{family} does not support tiling {tiling}")


def _stable_logsumexp(values: FloatArray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def _log_weights(weights: FloatArray) -> FloatArray:
    result = np.full(weights.shape, -np.inf, dtype=np.float64)
    positive = weights > 0.0
    result[positive] = np.log(weights[positive])
    return result


def _mass_grid(
    *,
    shapes: FloatArray,
    rate: float,
    family: Family,
    tiling: Tiling,
    total_order: int,
    fraction_order: int,
) -> tuple[FloatArray, FloatArray]:
    """Return A1 quadrature masses and normalized log prior weights."""
    if family == "two_cell":
        oracle = TwoCellAggregationOracle(
            gamma_shape=float(shapes.sum()),
            gamma_rate=rate,
            beta_first_shape=float(shapes[0]),
            beta_second_shape=float(shapes[1]),
            total_order=total_order,
            fraction_order=fraction_order,
        )
        total_rule = oracle.total_rule()
        if tiling == "root":
            return total_rule.nodes[:, None], _log_weights(total_rule.weights)
        share_rule = oracle.fraction_rule()
    else:
        oracle4 = FourCellAggregationOracle(
            native_shapes=shapes,
            gamma_rate=rate,
            total_order=total_order,
            fraction_order=fraction_order,
        )
        total_rule = oracle4.total_rule()
        if tiling == "root":
            return total_rule.nodes[:, None], _log_weights(total_rule.weights)
        labels = labels_for_tiling(family, tiling)
        totals = np.bincount(labels, weights=shapes)
        if totals.size != 2:
            raise ValueError("this screen evaluates only one- and two-region mass charts")
        share_rule = beta_quadrature(
            float(totals[0]),
            float(totals[1]),
            fraction_order,
        )
    total = np.repeat(total_rule.nodes, share_rule.nodes.size)
    share = np.tile(share_rule.nodes, total_rule.nodes.size)
    masses = np.column_stack((total * share, total * (1.0 - share)))
    log_prior = np.repeat(_log_weights(total_rule.weights), share_rule.nodes.size)
    log_prior += np.tile(_log_weights(share_rule.weights), total_rule.nodes.size)
    return masses, log_prior


def _exact_log_likelihood(
    *,
    masses: FloatArray,
    shapes: FloatArray,
    rate: float,
    design: FloatArray,
    observation: FloatArray,
    noise: FloatArray,
    family: Family,
    tiling: Tiling,
    total_order: int,
    fraction_order: int,
) -> FloatArray:
    """Evaluate exact hidden-allocation quadrature at every retained state."""
    if family == "two_cell":
        oracle = TwoCellAggregationOracle(
            gamma_shape=float(shapes.sum()),
            gamma_rate=rate,
            beta_first_shape=float(shapes[0]),
            beta_second_shape=float(shapes[1]),
            total_order=total_order,
            fraction_order=fraction_order,
        )
        if tiling == "root":
            values = oracle.coarse_conditional_log_likelihood(
                masses[:, 0],
                observation,
                design,
                noise,
            )
        else:
            mean = masses @ design.T
            residual = (observation[None, :] - mean) / noise[None, :]
            values = -0.5 * (
                observation.size * math.log(2.0 * math.pi)
                + 2.0 * float(np.log(noise).sum())
                + np.square(residual).sum(axis=1)
            )
    else:
        oracle4 = FourCellAggregationOracle(
            native_shapes=shapes,
            gamma_rate=rate,
            total_order=total_order,
            fraction_order=fraction_order,
        )
        values = oracle4.conditional_log_likelihood(
            masses[:, 0] if tiling == "root" else masses,
            observation,
            design,
            noise,
            tiling=tiling,
        )
    return np.asarray(values, dtype=np.float64)


def coordinate_to_masses(coordinate: FloatArray) -> FloatArray:
    """Map log-total and optional logit-share coordinates to masses."""
    total = math.exp(float(coordinate[0]))
    if coordinate.size == 1:
        return np.asarray([total], dtype=np.float64)
    if coordinate.size != 2:
        raise ValueError("coordinate must contain log total and at most one logit share")
    share = 1.0 / (1.0 + math.exp(-float(coordinate[1])))
    return np.asarray([total * share, total * (1.0 - share)], dtype=np.float64)


def masses_to_coordinate(masses: FloatArray) -> FloatArray:
    """Map one positive one-/two-region mass vector to the A1 chart."""
    if masses.ndim != 1 or masses.size not in (1, 2) or np.any(masses <= 0.0):
        raise ValueError("masses must contain one or two strictly positive values")
    total = float(masses.sum())
    if masses.size == 1:
        return np.asarray([math.log(total)], dtype=np.float64)
    return np.asarray(
        [math.log(total), math.log(float(masses[0] / masses[1]))],
        dtype=np.float64,
    )


def mass_gradient_to_coordinate_gradient(
    masses: FloatArray,
    mass_gradient: FloatArray,
) -> FloatArray:
    """Apply the exact log-total/logit-share chain rule."""
    if masses.shape != mass_gradient.shape or masses.ndim != 1:
        raise ValueError("masses and mass_gradient must be aligned vectors")
    if masses.size == 1:
        return np.asarray([masses[0] * mass_gradient[0]], dtype=np.float64)
    if masses.size != 2:
        raise ValueError("the C1 coordinate chart supports one or two regions")
    total = float(masses.sum())
    if total <= 0.0:
        raise ValueError("two-region masses must have positive total")
    share = float(masses[0] / total)
    contrast_derivative = total * share * (1.0 - share)
    return np.asarray(
        [
            float(masses @ mass_gradient),
            contrast_derivative * (mass_gradient[0] - mass_gradient[1]),
        ],
        dtype=np.float64,
    )


def _anchor_coordinate(
    shapes: FloatArray,
    rate: float,
    labels: IntArray,
) -> FloatArray:
    alpha_totals = np.bincount(labels, weights=shapes)
    total = float(shapes.sum() / rate)
    if alpha_totals.size == 1:
        return np.asarray([math.log(total)], dtype=np.float64)
    return np.asarray(
        [math.log(total), math.log(float(alpha_totals[0] / alpha_totals[1]))],
        dtype=np.float64,
    )


def _weighted_state_index(
    values: FloatArray,
    normalized_weights: FloatArray,
    probability: float,
) -> int:
    """Return the stable state index at one weighted coordinate quantile."""
    target = _weighted_quantile(values, normalized_weights, probability)
    return int(np.argmin(np.abs(values - target)))


def _gradient_state_coordinates(
    *,
    masses: FloatArray,
    log_prior: FloatArray,
    exact_log_likelihood: FloatArray,
    prior_mean_coordinate: FloatArray,
) -> list[tuple[str, FloatArray]]:
    """Return predeclared finite interior, tail, and posterior-relevant states."""
    prior_weights = np.exp(log_prior - _stable_logsumexp(log_prior))
    total = masses.sum(axis=1)
    lower_total_coordinate = prior_mean_coordinate.copy()
    lower_total_coordinate[0] = math.log(float(total[_weighted_state_index(total, prior_weights, 0.05)]))
    upper_total_coordinate = prior_mean_coordinate.copy()
    upper_total_coordinate[0] = math.log(float(total[_weighted_state_index(total, prior_weights, 0.95)]))
    selected: list[tuple[str, FloatArray]] = [
        ("prior_mean_interior", prior_mean_coordinate),
        ("prior_lower_total_at_mean_share", lower_total_coordinate),
        ("prior_upper_total_at_mean_share", upper_total_coordinate),
        (
            "highest_exact_posterior_weight_grid_state",
            masses_to_coordinate(masses[int(np.argmax(log_prior + exact_log_likelihood))]),
        ),
    ]
    if masses.shape[1] == 2:
        share = masses[:, 0] / total
        lower_share_state = masses[
            _weighted_state_index(
                share,
                prior_weights,
                0.05,
            )
        ]
        upper_share_state = masses[
            _weighted_state_index(
                share,
                prior_weights,
                0.95,
            )
        ]
        lower_share_coordinate = prior_mean_coordinate.copy()
        lower_share_coordinate[1] = math.log(float(lower_share_state[0] / lower_share_state[1]))
        upper_share_coordinate = prior_mean_coordinate.copy()
        upper_share_coordinate[1] = math.log(float(upper_share_state[0] / upper_share_state[1]))
        selected.extend(
            (
                (
                    "prior_lower_share_at_mean_total",
                    lower_share_coordinate,
                ),
                (
                    "prior_upper_share_at_mean_total",
                    upper_share_coordinate,
                ),
            )
        )
    unique: list[tuple[str, FloatArray]] = []
    seen: set[bytes] = set()
    for name, coordinate in selected:
        identity = np.ascontiguousarray(coordinate, dtype="<f8").tobytes()
        if identity not in seen:
            seen.add(identity)
            unique.append((name, coordinate))
    return unique


def _centered_gradient(function: Any, coordinate: FloatArray) -> FloatArray:
    """Return A1's exact centered finite-difference coordinate gradient."""
    result = np.empty_like(coordinate)
    for index in range(coordinate.size):
        high = coordinate.copy()
        low = coordinate.copy()
        high[index] += GRADIENT_STEP
        low[index] -= GRADIENT_STEP
        result[index] = (function(high) - function(low)) / (2.0 * GRADIENT_STEP)
    return result


def _development_validation_state_mask(
    masses: FloatArray,
    *,
    total_order: int,
    fraction_order: int,
) -> NDArray[np.bool_]:
    """Return a frozen checkerboard split covering every mass-coordinate axis."""
    if masses.shape[0] < 2:
        raise ValueError("the mass grid must contain at least two states")
    if masses.shape[1] == 1:
        if masses.shape[0] != total_order:
            raise ValueError("root mass-grid shape does not match total order")
        mask = np.arange(masses.shape[0], dtype=np.int64) % 2 == 1
    elif masses.shape[1] == 2:
        if masses.shape[0] != total_order * fraction_order:
            raise ValueError("pair mass-grid shape does not match quadrature orders")
        total_index = np.repeat(
            np.arange(total_order, dtype=np.int64),
            fraction_order,
        )
        share_index = np.tile(
            np.arange(fraction_order, dtype=np.int64),
            total_order,
        )
        mask = (total_index + share_index) % 2 == 1
    else:
        raise ValueError("development-validation C1 state split supports one or two regions")
    if not np.any(mask) or np.all(mask):
        raise ValueError("development-validation C1 state split must be non-empty and proper")
    return mask


def _weighted_quantile(
    values: FloatArray,
    weights: FloatArray,
    probability: float,
) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, probability, side="left"))
    return float(values[order[min(index, values.size - 1)]])


def _stable_lock_sample_count(
    sample_counts: Sequence[int],
    pass_pattern: Sequence[bool],
    *,
    minimum_suffix_length: int,
) -> int | None:
    """Return the smallest count starting a sufficiently long passing suffix."""
    if len(sample_counts) != len(pass_pattern) or minimum_suffix_length < 1:
        raise ValueError("lock inputs must align and use a positive suffix")
    for index, sample_count in enumerate(sample_counts):
        suffix = pass_pattern[index:]
        if len(suffix) >= minimum_suffix_length and all(suffix):
            return int(sample_count)
    return None


def _posterior_summary(
    masses: FloatArray,
    log_prior: FloatArray,
    log_likelihood: FloatArray,
) -> dict[str, Any]:
    log_joint = log_prior + log_likelihood
    log_evidence = _stable_logsumexp(log_joint)
    weights = np.exp(log_joint - log_evidence)
    total = masses.sum(axis=1)
    coordinate_values: dict[str, FloatArray] = {"total_mass": total}
    for index in range(masses.shape[1]):
        coordinate_values[f"region_mass_{index}"] = masses[:, index]
    if masses.shape[1] == 2:
        coordinate_values["first_region_share"] = masses[:, 0] / total
        coordinate_values["log_first_to_second_region_mass_ratio"] = np.log(masses[:, 0] / masses[:, 1])
    coordinates: dict[str, dict[str, float]] = {}
    for name, values in coordinate_values.items():
        mean = float(weights @ values)
        variance = float(weights @ np.square(values - mean))
        coordinates[name] = {
            "mean": mean,
            "sd": math.sqrt(max(variance, 0.0)),
            "lower_0.025": _weighted_quantile(values, weights, 0.025),
            "upper_0.975": _weighted_quantile(values, weights, 0.975),
        }
    return {
        "log_evidence": log_evidence,
        "coordinates": coordinates,
    }


def _summary_errors(
    exact: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    exact_coordinates = exact["coordinates"]
    candidate_coordinates = candidate["coordinates"]
    if exact_coordinates.keys() != candidate_coordinates.keys():
        raise ValueError("posterior coordinate catalogues do not agree")
    by_coordinate: dict[str, dict[str, float]] = {}
    for name in exact_coordinates:
        reference = exact_coordinates[name]
        observed = candidate_coordinates[name]
        reference_sd = max(
            float(reference["sd"]),
            float(np.finfo(np.float64).tiny),
        )
        by_coordinate[name] = {
            "mean_error_reference_sd": abs(observed["mean"] - reference["mean"]) / reference_sd,
            "sd_relative_error": abs(observed["sd"] - reference["sd"]) / reference_sd,
            "interval_endpoint_error_reference_sd": max(
                abs(observed["lower_0.025"] - reference["lower_0.025"]),
                abs(observed["upper_0.975"] - reference["upper_0.975"]),
            )
            / reference_sd,
        }
    aggregate = {
        "posterior_mean_error_reference_sd": max(
            values["mean_error_reference_sd"] for values in by_coordinate.values()
        ),
        "posterior_sd_relative_error": max(values["sd_relative_error"] for values in by_coordinate.values()),
        "interval_endpoint_error_reference_sd": max(
            values["interval_endpoint_error_reference_sd"] for values in by_coordinate.values()
        ),
    }
    return aggregate, by_coordinate


def _case_input_sha256(
    regime: Regime,
    family: Family,
    tiling: Tiling,
    total_order: int,
    fraction_order: int,
) -> str:
    return _sha256_json(
        {
            "regime": asdict(regime),
            "family": family,
            "tiling": tiling,
            "total_order": total_order,
            "fraction_order": fraction_order,
            "summary_basis": "identity_full_rank",
        }
    )


def _evaluate_bank(
    *,
    artifact: ConditionalAllocationMixture,
    observation: FloatArray,
    masses: FloatArray,
    log_prior: FloatArray,
    exact_log_likelihood: FloatArray,
    exact_summary: dict[str, Any],
    gradient_states: Sequence[dict[str, Any]],
    validation_state_mask: NDArray[np.bool_],
    include_timings: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    bank_log_likelihood = np.asarray(
        [artifact.log_likelihood(observation, state) for state in masses],
        dtype=np.float64,
    )
    evaluation_seconds = time.perf_counter() - started if include_timings else None
    error = np.abs(bank_log_likelihood - exact_log_likelihood)
    bank_summary = _posterior_summary(masses, log_prior, bank_log_likelihood)
    summary_errors, summary_errors_by_coordinate = _summary_errors(
        exact_summary,
        bank_summary,
    )
    validation_prior_log_weights = log_prior[validation_state_mask]
    validation_prior_weights = np.exp(
        validation_prior_log_weights - _stable_logsumexp(validation_prior_log_weights)
    )
    validation_posterior_log_weights = (
        validation_prior_log_weights + exact_log_likelihood[validation_state_mask]
    )
    validation_posterior_weights = np.exp(
        validation_posterior_log_weights - _stable_logsumexp(validation_posterior_log_weights)
    )
    validation_error = error[validation_state_mask]
    gradient_audits: list[dict[str, Any]] = []
    for state in gradient_states:
        coordinate = np.asarray(state["coordinate"], dtype=np.float64)
        state_masses = coordinate_to_masses(coordinate)
        _, mass_gradient = artifact.log_likelihood_and_mass_gradient(
            observation,
            state_masses,
        )
        coordinate_gradient = mass_gradient_to_coordinate_gradient(
            state_masses,
            mass_gradient,
        )
        exact_gradient = np.asarray(
            state["exact_coordinate_gradient"],
            dtype=np.float64,
        )
        scaled_error = float(
            np.max(np.abs(coordinate_gradient - exact_gradient) / (1.0 + np.abs(exact_gradient)))
        )
        gradient_audits.append(
            {
                "state_id": state["state_id"],
                "coordinate": coordinate.tolist(),
                "exact_coordinate_gradient": exact_gradient.tolist(),
                "bank_coordinate_gradient": coordinate_gradient.tolist(),
                "scaled_error": scaled_error,
            }
        )
    gradient_error = max(audit["scaled_error"] for audit in gradient_audits)
    evidence_error = abs(bank_summary["log_evidence"] - exact_summary["log_evidence"])
    metrics = {
        "median_absolute_conditional_log_likelihood_error_nat": (
            _weighted_quantile(
                validation_error,
                validation_prior_weights,
                0.5,
            )
        ),
        "p99_absolute_conditional_log_likelihood_error_nat": (
            _weighted_quantile(
                validation_error,
                validation_posterior_weights,
                0.99,
            )
        ),
        "scaled_coordinate_gradient_error": gradient_error,
        "absolute_log_evidence_error_nat": evidence_error,
        **summary_errors,
    }
    diagnostics = {
        "unweighted_full_grid_median_absolute_conditional_log_likelihood_error_nat": float(np.median(error)),
        "unweighted_full_grid_p99_absolute_conditional_log_likelihood_error_nat": float(
            np.quantile(error, 0.99)
        ),
        "unweighted_full_grid_maximum_absolute_conditional_log_likelihood_error_nat": float(np.max(error)),
        "development_validation_prior_weighted_p95_absolute_conditional_log_likelihood_error_nat": (
            _weighted_quantile(
                validation_error,
                validation_prior_weights,
                0.95,
            )
        ),
        "development_validation_prior_weighted_p99_absolute_conditional_log_likelihood_error_nat": (
            _weighted_quantile(
                validation_error,
                validation_prior_weights,
                0.99,
            )
        ),
        "development_validation_exact_posterior_weighted_p99_absolute_conditional_log_likelihood_error_nat": (
            _weighted_quantile(
                validation_error,
                validation_posterior_weights,
                0.99,
            )
        ),
        "pointwise_gate_weighting": {
            "median": ("normalized quadrature prior weights on the C1 development validation view"),
            "p99": ("normalized exact-posterior quadrature weights on the C1 development validation view"),
        },
    }
    checks = {
        name: bool(metrics[name] <= threshold) for name, threshold in THRESHOLDS.items() if name in metrics
    }
    return {
        "artifact_sha256": artifact.sha256,
        "source_seed": artifact.source_seed,
        "sample_count": artifact.sample_count,
        "storage_nbytes": artifact.storage_nbytes,
        "evaluation_seconds": evaluation_seconds,
        "evaluation_states_per_second": (
            None
            if evaluation_seconds is None or evaluation_seconds == 0.0
            else masses.shape[0] / evaluation_seconds
        ),
        "metrics": metrics,
        "checks": checks,
        "scientific_pass_without_repeat_evidence_gate": all(checks.values()),
        "posterior_summary": bank_summary,
        "posterior_errors_by_coordinate": summary_errors_by_coordinate,
        "gradient_audits": gradient_audits,
        "diagnostics": diagnostics,
    }


def _structural_prior_weight(family: Family, tiling: Tiling) -> float:
    """Return the pinned A1 prior weight for one executable frontier."""
    weights = (
        {"root": 0.73, "fine": 0.27} if family == "two_cell" else {"root": 0.41, "row": 0.27, "fine": 0.13}
    )
    if tiling not in weights:
        raise ValueError("held-out structural-prior entries are unavailable")
    return weights[tiling]


def run_case(
    *,
    regime_name: str,
    family: Family,
    tiling: Tiling,
    sample_counts: Sequence[int],
    repeat_seeds: Sequence[int],
    profile: Literal["smoke", "development", "control"],
    include_timings: bool = True,
) -> dict[str, Any]:
    """Run one exact-versus-bank case and return aggregate canonical metrics."""
    if profile not in ("smoke", "development", "control"):
        raise ValueError("profile must be 'smoke', 'development', or 'control'")
    if not sample_counts or any(
        isinstance(value, bool) or int(value) != value or value < 1 for value in sample_counts
    ):
        raise ValueError("sample_counts must contain positive integers")
    if len(set(sample_counts)) != len(sample_counts):
        raise ValueError("sample_counts must be unique")
    if tuple(int(value) for value in sample_counts) != tuple(sorted(int(value) for value in sample_counts)):
        raise ValueError("sample_counts must be strictly increasing")
    if not repeat_seeds or any(
        isinstance(value, bool) or int(value) != value or not 0 <= value < 2**64 for value in repeat_seeds
    ):
        raise ValueError("repeat_seeds must contain unsigned integer seeds")
    if len(set(repeat_seeds)) != len(repeat_seeds):
        raise ValueError("repeat_seeds must be unique")
    if profile == "development" and (
        tuple(int(value) for value in sample_counts) != DEVELOPMENT_SAMPLE_COUNTS
        or tuple(int(value) for value in repeat_seeds) != DEVELOPMENT_REPEAT_SEEDS
    ):
        raise ValueError("development uses the source-pinned sample counts and seeds")

    case_key = (regime_name, family, tiling)
    if profile == "smoke":
        allowed = SMOKE_MATRIX
    elif profile == "development":
        allowed = DEVELOPMENT_MATRIX
    else:
        allowed = RUNNABLE_CONTROL_MATRIX
    if case_key not in allowed:
        raise ValueError(f"case {case_key!r} is not available in {profile}")
    regime = _regime(regime_name)
    shapes, rate, design, observation, noise = _case_arrays(regime, family)
    common_projection = np.asarray(
        regime.projection2 if family == "two_cell" else regime.projection4,
        dtype=np.float64,
    )
    labels = labels_for_tiling(family, tiling)
    if profile in ("smoke", "control"):
        total_order = 8
        fraction_order = 6
    else:
        total_order = regime.total_order
        fraction_order = regime.fraction_order
    masses, log_prior = _mass_grid(
        shapes=shapes,
        rate=rate,
        family=family,
        tiling=tiling,
        total_order=total_order,
        fraction_order=fraction_order,
    )
    exact_log_likelihood = _exact_log_likelihood(
        masses=masses,
        shapes=shapes,
        rate=rate,
        design=design,
        observation=observation,
        noise=noise,
        family=family,
        tiling=tiling,
        total_order=total_order,
        fraction_order=fraction_order,
    )
    exact_summary = _posterior_summary(
        masses,
        log_prior,
        exact_log_likelihood,
    )
    prior_mean_coordinate = _anchor_coordinate(shapes, rate, labels)

    def exact_function(value: FloatArray) -> float:
        state = coordinate_to_masses(value)
        return float(
            _exact_log_likelihood(
                masses=state[None, :],
                shapes=shapes,
                rate=rate,
                design=design,
                observation=observation,
                noise=noise,
                family=family,
                tiling=tiling,
                total_order=total_order,
                fraction_order=fraction_order,
            )[0]
        )

    gradient_states = [
        {
            "state_id": state_id,
            "coordinate": state_coordinate.tolist(),
            "exact_coordinate_gradient": _centered_gradient(
                exact_function,
                state_coordinate,
            ).tolist(),
        }
        for state_id, state_coordinate in _gradient_state_coordinates(
            masses=masses,
            log_prior=log_prior,
            exact_log_likelihood=exact_log_likelihood,
            prior_mean_coordinate=prior_mean_coordinate,
        )
    ]
    validation_state_mask = _development_validation_state_mask(
        masses,
        total_order=total_order,
        fraction_order=fraction_order,
    )
    aggregation = AdditiveDirichletAggregation(
        shapes,
        design,
        noise,
        np.eye(observation.size, dtype=np.float64),
    )

    def evaluate(sample_count: int, seed: int) -> dict[str, Any]:
        started = time.perf_counter()
        artifact = ConditionalAllocationMixture.from_aggregation(
            aggregation,
            labels,
            sample_count=sample_count,
            source_seed=seed,
            source_provenance=(f"{PROTOCOL}:{regime_name}:{family}:{tiling}:S={sample_count}:seed={seed}"),
        )
        build_seconds = time.perf_counter() - started if include_timings else None
        result = _evaluate_bank(
            artifact=artifact,
            observation=observation,
            masses=masses,
            log_prior=log_prior,
            exact_log_likelihood=exact_log_likelihood,
            exact_summary=exact_summary,
            gradient_states=gradient_states,
            validation_state_mask=validation_state_mask,
            include_timings=include_timings,
        )
        result["build_seconds"] = build_seconds
        return result

    development_seed = int(repeat_seeds[0])
    confirmation_seeds = [int(seed) for seed in repeat_seeds[1:]]
    development_evaluations: list[dict[str, Any]] = []
    for sample_count in sample_counts:
        development_evaluations.append(evaluate(int(sample_count), development_seed))
    development_pass_pattern = [
        {
            "sample_count": int(result["sample_count"]),
            "pass": bool(result["scientific_pass_without_repeat_evidence_gate"]),
        }
        for result in development_evaluations
    ]
    minimum_passing_suffix_length = 1 if profile in ("smoke", "control") else 2
    locked_sample_count = _stable_lock_sample_count(
        [int(result["sample_count"]) for result in development_evaluations],
        [bool(result["scientific_pass_without_repeat_evidence_gate"]) for result in development_evaluations],
        minimum_suffix_length=minimum_passing_suffix_length,
    )
    confirmation_evaluations = (
        [evaluate(locked_sample_count, seed) for seed in confirmation_seeds]
        if locked_sample_count is not None and profile != "control"
        else []
    )
    locked_development = next(
        (result for result in development_evaluations if result["sample_count"] == locked_sample_count),
        None,
    )
    locked_results = [locked_development, *confirmation_evaluations] if locked_development is not None else []
    locked_evidence = [result["posterior_summary"]["log_evidence"] for result in locked_results]
    evidence_range = float(max(locked_evidence) - min(locked_evidence)) if locked_evidence else None
    evidence_check = (
        bool(evidence_range <= THRESHOLDS["between_bank_log_evidence_range_nat"])
        if evidence_range is not None
        else False
    )
    confirmation_checks = [
        bool(result["scientific_pass_without_repeat_evidence_gate"]) for result in confirmation_evaluations
    ]
    confirmation_pass = (
        bool(all(confirmation_checks) and evidence_check) if confirmation_evaluations else None
    )
    development_lock_eligible = bool(locked_sample_count is not None and profile != "control")
    if profile == "smoke":
        case_pass = bool(development_lock_eligible and evidence_check)
    elif profile == "development":
        case_pass = bool(development_lock_eligible and confirmation_pass is True and evidence_check)
    else:
        case_pass = False
    return {
        "case_id": f"{regime_name}__{family}__{tiling}",
        "profile": profile,
        "input_sha256": _case_input_sha256(
            regime,
            family,
            tiling,
            total_order,
            fraction_order,
        ),
        "regime": regime_name,
        "family": family,
        "tiling": tiling,
        "summary_basis": {
            "kind": "identity",
            "rank": int(observation.size),
            "observation_count": int(observation.size),
            "selection": "fixed_full_rank_independent_of_observed_residual",
        },
        "quadrature": {
            "total_order": total_order,
            "fraction_order": fraction_order,
            "mass_state_count": int(masses.shape[0]),
        },
        "mass_grid": {
            "integration_role": "complete_pinned_quadrature",
            "sha256": hashlib.sha256(np.ascontiguousarray(masses, dtype="<f8").tobytes()).hexdigest(),
            "pointwise_gate_split": {
                "scheme": "c1-checkerboard-by-total-and-share-index-v1",
                "new_in_c1_not_in_a1_or_t2": True,
                "used_for_development_pointwise_scoring": True,
                "is_protected_operator_or_partition_data": False,
                "validation_state_count": int(np.count_nonzero(validation_state_mask)),
                "validation_mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(
                        validation_state_mask,
                        dtype=np.uint8,
                    ).tobytes()
                ).hexdigest(),
                "alters_evidence_or_posterior_quadrature": False,
            },
        },
        "coordinate_names": (
            ["log_total"]
            if prior_mean_coordinate.size == 1
            else ["log_total", "log_first_to_second_region_mass_ratio"]
        ),
        "gradient_state_catalogue": gradient_states,
        "common_native_projection": {
            "partition_invariant": True,
            "bank_posterior_summary_available": False,
            "definition_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    common_projection,
                    dtype="<f8",
                ).tobytes()
            ).hexdigest(),
            "status": (
                "deferred: the frozen observation bank does not retain the "
                "underlying allocation shares or projection factors"
            ),
        },
        "declared_structural_prior_weight": _structural_prior_weight(
            family,
            tiling,
        ),
        "evidence_merger_group_id": f"{regime_name}__{family}",
        "evidence_merger_thresholds": MERGER_THRESHOLDS,
        "independent_evidence_merger": {
            "status": "pending_not_implemented",
            "emitted_values_are_inputs_not_a_certificate": True,
        },
        "structural_evidence_use": ("diagnostic merger only; must not update partition or dimension"),
        "exact_posterior_summary": exact_summary,
        "development_seed": development_seed,
        "confirmation_seeds": confirmation_seeds,
        "development_evaluations": development_evaluations,
        "development_pass_pattern": development_pass_pattern,
        "minimum_passing_suffix_length": minimum_passing_suffix_length,
        "confirmation_evaluations": confirmation_evaluations,
        "locked_sample_count": locked_sample_count,
        "development_lock_eligible": development_lock_eligible,
        "lock_selection_rule": (
            "smallest predeclared S for which it and every larger S pass all "
            "development gates under the single development seed"
        ),
        "confirmation_can_retune_lock": False,
        "lock_certificate": {
            "schema": "conditional-allocation-c1-bank-lock-v1",
            "eligible": development_lock_eligible,
            "locked_sample_count": locked_sample_count,
            "development_seed": development_seed,
            "selection_rule_satisfied": development_lock_eligible,
            "minimum_passing_suffix_length": minimum_passing_suffix_length,
            "confirmation_requested": bool(confirmation_seeds),
            "confirmation_complete": bool(confirmation_evaluations)
            and len(confirmation_evaluations) == len(confirmation_seeds),
            "confirmation_pass": confirmation_pass,
            "confirmation_can_retune": False,
            "full_c1_promotion_licensed": False,
        },
        "between_bank_log_evidence_range_nat": evidence_range,
        "between_bank_log_evidence_range_pass": evidence_check,
        "confirmation_pass": confirmation_pass,
        "scientific_pass": case_pass,
    }


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _driver_sha256() -> str:
    """Return the exact identity of this executable source file."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def matrix_catalogue() -> dict[str, Any]:
    """Return the immutable development and inaccessible held-out catalogue."""
    return {
        "schema": SCHEMA,
        "development": [list(case) for case in DEVELOPMENT_MATRIX],
        "smoke": [list(case) for case in SMOKE_MATRIX],
        "runnable_controls_not_for_selection": [list(case) for case in RUNNABLE_CONTROL_MATRIX],
        "controls_not_for_bank_selection": [list(case) for case in CONTROL_CATALOGUE],
        "held_out_catalogue": {
            "id": HELD_OUT_CATALOGUE_ID,
            "sha256": HELD_OUT_CATALOGUE_SHA256,
            "numerical_values_present": False,
            "executable_here": False,
        },
        "held_out_information_read": False,
    }


def run_screen(
    *,
    profile: Literal["smoke", "development", "control"],
    sample_counts: Sequence[int] | None = None,
    repeat_seeds: Sequence[int] | None = None,
    case_id: str | None = None,
    include_timings: bool = True,
) -> dict[str, Any]:
    """Run a smoke or predeclared development matrix."""
    observed_definitions_sha = a1_definitions_sha256()
    if observed_definitions_sha != A1_DEFINITIONS_SHA256:
        raise RuntimeError("copied A1 numerical definitions no longer match their pin")
    if profile == "development" and (sample_counts is not None or repeat_seeds is not None):
        raise ValueError(
            "development uses the source-pinned sample counts and seeds; "
            "overrides require a new reviewed protocol revision"
        )
    if profile == "smoke":
        matrix = SMOKE_MATRIX
        selected_counts = SMOKE_SAMPLE_COUNTS if sample_counts is None else sample_counts
        selected_seeds = SMOKE_REPEAT_SEEDS if repeat_seeds is None else repeat_seeds
    elif profile == "development":
        matrix = DEVELOPMENT_MATRIX
        selected_counts = DEVELOPMENT_SAMPLE_COUNTS if sample_counts is None else sample_counts
        selected_seeds = DEVELOPMENT_REPEAT_SEEDS if repeat_seeds is None else repeat_seeds
    elif profile == "control":
        matrix = RUNNABLE_CONTROL_MATRIX
        selected_counts = SMOKE_SAMPLE_COUNTS if sample_counts is None else sample_counts
        selected_seeds = SMOKE_REPEAT_SEEDS if repeat_seeds is None else repeat_seeds
    else:
        raise ValueError("held-out execution is deliberately unavailable")
    if case_id is not None:
        matches = [case for case in matrix if "__".join(case) == case_id]
        if len(matches) != 1:
            raise ValueError(f"case_id {case_id!r} is not available in profile {profile}")
        matrix = tuple(matches)
    cases = [
        run_case(
            regime_name=regime,
            family=cast(Family, family),
            tiling=cast(Tiling, tiling),
            sample_counts=selected_counts,
            repeat_seeds=selected_seeds,
            profile=profile,
            include_timings=include_timings,
        )
        for regime, family, tiling in matrix
    ]
    return {
        "schema": SCHEMA,
        "completion_scope": "smoke_and_development_only_not_full_c1",
        "protocol": PROTOCOL,
        "profile": profile,
        "selected_case_id": case_id,
        "per_case_atomic_output": case_id is not None,
        "source_git_revision": _git_revision(),
        "driver_sha256": _driver_sha256(),
        "a1_source_revision": A1_SOURCE_REVISION,
        "a1_numerical_source_sha256": A1_NUMERICAL_SOURCE_SHA256,
        "a1_definitions_sha256": observed_definitions_sha,
        "protocol_sha256": _sha256_json(
            {
                "schema": SCHEMA,
                "protocol": PROTOCOL,
                "thresholds": THRESHOLDS,
                "merger_thresholds": MERGER_THRESHOLDS,
                "gradient_step": GRADIENT_STEP,
                "sample_counts": list(selected_counts),
                "repeat_seeds": list(selected_seeds),
                "matrix": matrix,
            }
        ),
        "held_out_information_read": False,
        "held_out_operator_partition_information_read": False,
        "c1_pointwise_validation_subset_evaluated": True,
        "held_out_execution_available": False,
        "observed_residual_used_for_basis_selection": False,
        "structural_inference_licensed": False,
        "full_c1_pass": False,
        "full_c1_pass_reason": (
            "held-out operators, partitions, retained-mass grids, and "
            "independent held-out invocation are intentionally not implemented"
        ),
        "independent_evidence_merger_status": "pending_not_implemented",
        "sample_counts": list(selected_counts),
        "repeat_seeds": list(selected_seeds),
        "bank_lock_protocol": {
            "development_seed": int(selected_seeds[0]),
            "confirmation_seeds": [int(seed) for seed in selected_seeds[1:]],
            "selection_rule": ("smallest S passing development gates under development seed"),
            "confirmation_can_retune": False,
        },
        "thresholds": THRESHOLDS,
        "merger_thresholds": MERGER_THRESHOLDS,
        "matrix_catalogue": matrix_catalogue(),
        "cases": cases,
        "scientific_pass": all(case["scientific_pass"] for case in cases),
    }


def _positive_csv(value: str, *, name: str, upper_bound: int) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list") from error
    if (
        not parsed
        or any(item < 1 or item > upper_bound for item in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise argparse.ArgumentTypeError(f"{name} must contain unique integers in [1, {upper_bound}]")
    return parsed


def _write_atomic_json(path: Path, payload: object) -> None:
    """Publish canonical JSON once, without partial or overwritten output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical_json(payload))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "development", "control"),
        default="smoke",
    )
    parser.add_argument(
        "--case-id",
        help=(
            "Run one profile case as REGIME__FAMILY__TILING; required for "
            "independent Slurm array outputs but optional for a full profile"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--sample-counts",
        type=lambda value: _positive_csv(
            value,
            name="sample-counts",
            upper_bound=1_000_000,
        ),
    )
    parser.add_argument(
        "--repeat-seeds",
        type=lambda value: _positive_csv(
            value,
            name="repeat-seeds",
            upper_bound=2**63 - 1,
        ),
    )
    parser.add_argument("--list-matrix", action="store_true")
    parser.add_argument("--no-timings", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the CLI, run the selected screen, and publish only on success."""
    args = _parser().parse_args(argv)
    if args.list_matrix:
        if args.output is not None or args.sample_counts or args.repeat_seeds or args.case_id:
            raise SystemExit("--list-matrix cannot be combined with run options")
        print(_canonical_json(matrix_catalogue()))
        return 0
    if args.output is None:
        raise SystemExit("--output is required unless --list-matrix is used")
    if args.profile == "development" and (args.sample_counts is not None or args.repeat_seeds is not None):
        raise SystemExit("development sample counts and seeds are source-pinned and cannot be overridden")
    report = run_screen(
        profile=args.profile,
        sample_counts=args.sample_counts,
        repeat_seeds=args.repeat_seeds,
        case_id=args.case_id,
        include_timings=not args.no_timings,
    )
    _write_atomic_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
