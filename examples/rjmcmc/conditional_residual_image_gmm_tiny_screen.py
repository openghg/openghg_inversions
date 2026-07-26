#!/usr/bin/env python3
"""Train and score the root-only conditional residual-image GMM baseline.

This is the first learned-density fallback after the frozen allocation-bank
screen.  It deliberately covers only one-region (``root``) partitions, for
which the residual-image density is unconditional.  A deterministic
full-covariance Gaussian mixture is fitted in the exact residual-image
coordinates and exported through the portable zero-input MDN evaluator.

Training, model selection, reporting, and the unavailable protected holdout
use separately domain-keyed scrambled-Sobol streams.  The development
function cannot open the protected holdout catalogue.  As in the C1 screen,
the resulting likelihood is a fixed-partition approximation and must not be
used to infer a partition or dimension.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Literal, Sequence, cast

import numpy as np
from numpy import __version__ as numpy_version
from numpy.typing import NDArray
from scipy import __version__ as scipy_version

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mdn import (
    ConditionalResidualImageMDN,
    RESIDUAL_IMAGE_BASIS_RULE,
    RESIDUAL_IMAGE_CONTEXT_SCHEMA,
    ResidualImageContext,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (
    ConditionalAllocationMixture,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)

FloatArray = NDArray[np.float64]
Profile = Literal["smoke", "development"]

SCHEMA = "rjmcmc-conditional-residual-image-gmm-tiny-screen-v1"
PROTOCOL = "conditional-residual-image-root-full-covariance-gmm-v1"
CONSTRUCTION_METHOD = "scrambled_sobol_balanced_dirichlet"
COMPONENT_COUNT = 8
INITIALIZATION_COUNT = 3
MINIMUM_VALID_INITIALIZATIONS = 2
COVARIANCE_REGULARIZATION = 1.0e-8
CHOLESKY_DIAGONAL_FLOOR = 1.0e-4
MAXIMUM_EM_ITERATIONS = 2_000
CONVERGENCE_NAT_PER_DRAW = 1.0e-7
CONVERGENCE_STREAK = 10
GENERALIZATION_NAT_PER_DIMENSION = 0.02
GENERALIZATION_MCSE_MULTIPLIER = 5.0
DEVELOPMENT_NUMPY_VERSION = "2.2.6"
DEVELOPMENT_SCIPY_VERSION = "1.15.2"

DEVELOPMENT_MATRIX = tuple(
    (regime, family, "root")
    for regime in ("near_gaussian", "skewed", "boundary_heavy")
    for family in ("two_cell", "four_cell")
)
SMOKE_MATRIX = (("near_gaussian", "two_cell", "root"),)
DEVELOPMENT_SAMPLE_COUNTS = (4_096, 16_384, 65_536, 262_144)
SMOKE_SAMPLE_COUNTS = (4_096,)
VALIDATION_SAMPLE_COUNT = 65_536
TEST_SAMPLE_COUNT = 131_072
PROTECTED_HOLDOUT_SAMPLE_COUNT = 131_072
DEVELOPMENT_SELECTION_SEED = 731
CONFIRMATION_SEEDS = (1_877, 4_099, 8_317)
SMOKE_REPEAT_SEEDS = (731,)

TRAINING_DOMAIN = "training"
VALIDATION_DOMAIN = "model-selection-validation"
TEST_DOMAIN = "development-reporting-test"
PROTECTED_HOLDOUT_CATALOGUE_ID = "conditional-residual-image-protected-density-holdout-v1"
PROTECTED_HOLDOUT_CATALOGUE_SHA256 = "83bec3945ebc90d5e25d0888b440fe56f761f9059cf01537fbb2227b81510b66"
DEVELOPMENT_PROTOCOL_SHA256 = "51ae6ce153a92091967c1f09fa4d5a3342ab793cb54d78254d3f162a81923fee"


def _protocol_sha256() -> str:
    """Return the frozen six-case development protocol identity."""
    return c1._sha256_json(
        {
            "schema": SCHEMA,
            "protocol": PROTOCOL,
            "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
            "residual_image_context_schema": RESIDUAL_IMAGE_CONTEXT_SCHEMA,
            "residual_image_basis_rule": RESIDUAL_IMAGE_BASIS_RULE,
            "matrix": DEVELOPMENT_MATRIX,
            "training_sample_counts": DEVELOPMENT_SAMPLE_COUNTS,
            "validation_sample_count": VALIDATION_SAMPLE_COUNT,
            "test_sample_count": TEST_SAMPLE_COUNT,
            "protected_holdout_catalogue_sha256": (PROTECTED_HOLDOUT_CATALOGUE_SHA256),
            "protected_holdout_contract": {
                "catalogue_id": PROTECTED_HOLDOUT_CATALOGUE_ID,
                "object": ("new residual draws from the same six source-pinned exact contexts"),
                "promoted_artifact": ("development seed 731 at the common locked training size"),
                "retrain": False,
                "retune_after_reveal": False,
                "acceptance": (
                    "unchanged likelihood/gradient/evidence/posterior gates plus "
                    "the frozen validation-versus-protected-density NLL gate"
                ),
            },
            "development_numpy_version": DEVELOPMENT_NUMPY_VERSION,
            "development_scipy_version": DEVELOPMENT_SCIPY_VERSION,
            "development_selection_seed": DEVELOPMENT_SELECTION_SEED,
            "confirmation_seeds": CONFIRMATION_SEEDS,
            "domains": [TRAINING_DOMAIN, VALIDATION_DOMAIN, TEST_DOMAIN],
            "construction_method": CONSTRUCTION_METHOD,
            "component_count": COMPONENT_COUNT,
            "initialization_count": INITIALIZATION_COUNT,
            "minimum_valid_initializations": MINIMUM_VALID_INITIALIZATIONS,
            "covariance_regularization": COVARIANCE_REGULARIZATION,
            "cholesky_diagonal_floor": CHOLESKY_DIAGONAL_FLOOR,
            "maximum_em_iterations": MAXIMUM_EM_ITERATIONS,
            "convergence_nat_per_draw": CONVERGENCE_NAT_PER_DRAW,
            "convergence_streak": CONVERGENCE_STREAK,
            "simulator_test_generalization": {
                "absolute_nll_gap_threshold_nat": ("max(0.02 * residual_dimension, 5 * pooled_mcse)"),
                "nat_per_dimension": GENERALIZATION_NAT_PER_DIMENSION,
                "mcse_multiplier": GENERALIZATION_MCSE_MULTIPLIER,
            },
            "fitted_bundle_envelope_schema": ("conditional-residual-image-fitted-bundle-v1"),
            "selection": "minimum_validation_nll",
            "training_size_lock": (
                "smallest common six-case size starting an all-larger passing suffix of length at least two"
            ),
            "confirmation": ("all three confirmation seeds at the single common locked size"),
            "thresholds": c1.THRESHOLDS,
            "gradient_step": c1.GRADIENT_STEP,
        }
    )


def _validate_development_protocol() -> None:
    """Fail closed if a source-pinned development setting has drifted."""
    if _protocol_sha256() != DEVELOPMENT_PROTOCOL_SHA256:
        raise RuntimeError("the frozen learned-density development protocol identity changed")
    if numpy_version != DEVELOPMENT_NUMPY_VERSION:
        raise RuntimeError("NumPy does not match the frozen development version")
    if scipy_version != DEVELOPMENT_SCIPY_VERSION:
        raise RuntimeError("SciPy does not match the frozen development version")


@dataclass(frozen=True)
class GaussianMixtureFit:
    """One converged deterministic full-covariance Gaussian-mixture fit."""

    weights: FloatArray
    means: FloatArray
    covariances: FloatArray
    initialization: int
    iterations: int
    training_mean_log_likelihood: float
    validation_mean_log_likelihood: float
    validation_nll: float
    convergence_streak: int
    objective_history: tuple[float, ...]


def _domain_seed(base_seed: int, *, case_id: str, domain: str) -> int:
    """Derive a stable unsigned Sobol seed for one disjoint data domain."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or not 0 <= base_seed < 2**64:
        raise ValueError("base_seed must be an unsigned 64-bit integer")
    if domain not in (TRAINING_DOMAIN, VALIDATION_DOMAIN, TEST_DOMAIN):
        raise ValueError("protected or unknown sample domains cannot be opened")
    digest = hashlib.sha256(PROTOCOL.encode("ascii"))
    digest.update(base_seed.to_bytes(8, byteorder="little", signed=False))
    digest.update(case_id.encode("ascii"))
    digest.update(domain.encode("ascii"))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)


def _stable_logsumexp_rows(values: FloatArray) -> FloatArray:
    """Return a stable log-sum-exp across the final matrix dimension."""
    maximum = np.max(values, axis=1)
    if not np.all(np.isfinite(maximum)):
        raise FloatingPointError("component log densities must contain a finite row maximum")
    return maximum + np.log(np.sum(np.exp(values - maximum[:, np.newaxis]), axis=1))


def _component_log_densities(
    samples: FloatArray,
    weights: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
) -> FloatArray:
    """Evaluate normalized weighted component log densities."""
    sample_count, dimension = samples.shape
    component_count = weights.size
    if (
        means.shape != (component_count, dimension)
        or covariances.shape != (component_count, dimension, dimension)
        or np.any(weights <= 0.0)
        or not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError("Gaussian-mixture parameters have incompatible shapes or weights")
    result = np.empty((sample_count, component_count), dtype=np.float64)
    normalization = dimension * math.log(2.0 * math.pi)
    for component in range(component_count):
        try:
            cholesky = np.linalg.cholesky(covariances[component])
        except np.linalg.LinAlgError as error:
            raise FloatingPointError("a Gaussian-mixture covariance is not positive definite") from error
        difference = samples - means[component]
        solved = np.linalg.solve(cholesky, difference.T)
        log_determinant = 2.0 * float(np.log(np.diag(cholesky)).sum())
        result[:, component] = math.log(float(weights[component])) - 0.5 * (
            normalization + log_determinant + np.square(solved).sum(axis=0)
        )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Gaussian-mixture component log densities became non-finite")
    return result


def _mean_log_likelihood(
    samples: FloatArray,
    weights: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
) -> float:
    """Return the mean normalized mixture log likelihood."""
    values = _mixture_log_likelihood_values(
        samples,
        weights,
        means,
        covariances,
    )
    result = float(np.mean(values))
    if not np.isfinite(result):
        raise FloatingPointError("Gaussian-mixture objective became non-finite")
    return result


def _mixture_log_likelihood_values(
    samples: FloatArray,
    weights: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
) -> FloatArray:
    """Return one normalized mixture log density per simulator draw."""
    values = _stable_logsumexp_rows(_component_log_densities(samples, weights, means, covariances))
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Gaussian-mixture log densities became non-finite")
    return np.asarray(values, dtype=np.float64)


def _simulator_test_generalization(
    validation_samples: FloatArray,
    test_samples: FloatArray,
    fit: GaussianMixtureFit,
) -> dict[str, float | int | bool]:
    """Apply the frozen validation-versus-test NLL generalization gate."""
    validation_nll_values = -_mixture_log_likelihood_values(
        validation_samples,
        fit.weights,
        fit.means,
        fit.covariances,
    )
    test_nll_values = -_mixture_log_likelihood_values(
        test_samples,
        fit.weights,
        fit.means,
        fit.covariances,
    )
    dimension = validation_samples.shape[1]
    if dimension < 1 or validation_nll_values.size < 2 or test_nll_values.size < 2:
        raise ValueError("generalization gating requires dimension and at least two draws")
    validation_nll = float(np.mean(validation_nll_values))
    test_nll = float(np.mean(test_nll_values))
    validation_mcse = math.sqrt(float(np.var(validation_nll_values, ddof=1)) / validation_nll_values.size)
    test_mcse = math.sqrt(float(np.var(test_nll_values, ddof=1)) / test_nll_values.size)
    pooled_mcse = math.hypot(validation_mcse, test_mcse)
    absolute_gap = abs(test_nll - validation_nll)
    fixed_floor = GENERALIZATION_NAT_PER_DIMENSION * dimension
    threshold = max(
        fixed_floor,
        GENERALIZATION_MCSE_MULTIPLIER * pooled_mcse,
    )
    return {
        "residual_dimension": dimension,
        "validation_nll_nat_per_draw": validation_nll,
        "simulator_test_nll_nat_per_draw": test_nll,
        "absolute_nll_gap_nat_per_draw": absolute_gap,
        "validation_nll_mcse_nat_per_draw": validation_mcse,
        "simulator_test_nll_mcse_nat_per_draw": test_mcse,
        "pooled_nll_mcse_nat_per_draw": pooled_mcse,
        "fixed_floor_nat_per_draw": fixed_floor,
        "threshold_nat_per_draw": threshold,
        "pass": bool(absolute_gap <= threshold),
    }


def _global_covariance(samples: FloatArray) -> FloatArray:
    """Return the regularized population covariance of the training sample."""
    centered = samples - np.mean(samples, axis=0)
    covariance = centered.T @ centered / float(samples.shape[0])
    covariance += COVARIANCE_REGULARIZATION * np.eye(samples.shape[1], dtype=np.float64)
    if not np.all(np.isfinite(covariance)):
        raise FloatingPointError("global training covariance became non-finite")
    np.linalg.cholesky(covariance)
    return np.asarray(covariance, dtype=np.float64)


def _initial_mean_indices(
    samples: FloatArray,
    *,
    initialization: int,
    component_count: int,
) -> NDArray[np.int64]:
    """Return deterministic, distinct initial-centre indices."""
    sample_count, dimension = samples.shape
    if component_count > sample_count:
        raise ValueError("component_count cannot exceed the training sample count")
    if initialization not in range(INITIALIZATION_COUNT):
        raise ValueError("unknown deterministic initialization")
    centered = samples - np.mean(samples, axis=0)
    if initialization == 0:
        key = centered[:, 0]
        order = np.argsort(key, kind="stable")
        positions = np.linspace(0, sample_count - 1, component_count, dtype=np.int64)
        return np.asarray(order[positions], dtype=np.int64)
    if initialization == 1:
        first = int(np.argmin(np.square(centered).sum(axis=1)))
    else:
        first = int(np.argmax(np.square(centered).sum(axis=1)))
    selected = [first]
    minimum_squared_distance = np.square(samples - samples[first]).sum(axis=1)
    for _ in range(1, component_count):
        minimum_squared_distance[np.asarray(selected, dtype=np.int64)] = -1.0
        next_index = int(np.argmax(minimum_squared_distance))
        if next_index in selected or minimum_squared_distance[next_index] < 0.0:
            raise FloatingPointError("deterministic initialization could not select distinct centres")
        selected.append(next_index)
        squared_distance = np.square(samples - samples[next_index]).sum(axis=1)
        minimum_squared_distance = np.minimum(minimum_squared_distance, squared_distance)
    result = np.asarray(selected, dtype=np.int64)
    if result.size != component_count or np.unique(result).size != component_count:
        raise FloatingPointError("deterministic initialization returned duplicate centres")
    return result


def _fit_one_initialization(
    training_samples: FloatArray,
    validation_samples: FloatArray,
    *,
    initialization: int,
) -> GaussianMixtureFit:
    """Fit one deterministic EM initialization, failing invalid components."""
    if (
        training_samples.ndim != 2
        or validation_samples.ndim != 2
        or training_samples.shape[1] != validation_samples.shape[1]
        or training_samples.shape[0] < COMPONENT_COUNT
        or validation_samples.shape[0] == 0
        or training_samples.shape[1] == 0
        or not np.all(np.isfinite(training_samples))
        or not np.all(np.isfinite(validation_samples))
    ):
        raise ValueError("training and validation samples must be finite aligned non-empty matrices")
    initial_indices = _initial_mean_indices(
        training_samples,
        initialization=initialization,
        component_count=COMPONENT_COUNT,
    )
    weights = np.full(COMPONENT_COUNT, 1.0 / COMPONENT_COUNT, dtype=np.float64)
    means = np.array(training_samples[initial_indices], dtype=np.float64, copy=True)
    global_covariance = _global_covariance(training_samples)
    covariances = np.repeat(global_covariance[np.newaxis, :, :], COMPONENT_COUNT, axis=0)
    previous_objective: float | None = None
    streak = 0
    objective = -math.inf
    objective_history: list[float] = []
    for iteration in range(1, MAXIMUM_EM_ITERATIONS + 1):
        component_log = _component_log_densities(
            training_samples,
            weights,
            means,
            covariances,
        )
        row_log = _stable_logsumexp_rows(component_log)
        responsibilities = np.exp(component_log - row_log[:, np.newaxis])
        effective_counts = np.sum(responsibilities, axis=0)
        if np.any(effective_counts <= 0.0) or not np.all(np.isfinite(effective_counts)):
            raise FloatingPointError("EM produced an empty or non-finite component")
        new_weights = effective_counts / float(training_samples.shape[0])
        new_means = (responsibilities.T @ training_samples) / effective_counts[:, np.newaxis]
        new_covariances = np.empty_like(covariances)
        for component in range(COMPONENT_COUNT):
            difference = training_samples - new_means[component]
            weighted = difference * responsibilities[:, component, np.newaxis]
            covariance = weighted.T @ difference / float(effective_counts[component])
            covariance += COVARIANCE_REGULARIZATION * np.eye(
                training_samples.shape[1],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(covariance)):
                raise FloatingPointError("EM produced a non-finite component covariance")
            np.linalg.cholesky(covariance)
            new_covariances[component] = covariance
        if (
            not np.all(np.isfinite(new_weights))
            or not np.all(np.isfinite(new_means))
            or np.any(new_weights <= 0.0)
        ):
            raise FloatingPointError("EM produced invalid mixture parameters")
        weights = new_weights
        means = new_means
        covariances = new_covariances
        objective = _mean_log_likelihood(
            training_samples,
            weights,
            means,
            covariances,
        )
        objective_history.append(objective)
        if previous_objective is not None and abs(objective - previous_objective) < (
            CONVERGENCE_NAT_PER_DRAW
        ):
            streak += 1
        else:
            streak = 0
        previous_objective = objective
        if streak >= CONVERGENCE_STREAK:
            break
    else:
        raise RuntimeError(
            f"EM initialization {initialization} did not converge in {MAXIMUM_EM_ITERATIONS} iterations"
        )
    validation_mean = _mean_log_likelihood(
        validation_samples,
        weights,
        means,
        covariances,
    )
    return GaussianMixtureFit(
        np.array(weights, copy=True),
        np.array(means, copy=True),
        np.array(covariances, copy=True),
        initialization,
        iteration,
        objective,
        validation_mean,
        -validation_mean,
        streak,
        tuple(objective_history),
    )


def fit_gaussian_mixture(
    training_samples: FloatArray,
    validation_samples: FloatArray,
) -> tuple[GaussianMixtureFit, list[dict[str, Any]]]:
    """Fit three deterministic starts and select minimum validation NLL."""
    fits: list[GaussianMixtureFit] = []
    attempts: list[dict[str, Any]] = []
    for initialization in range(INITIALIZATION_COUNT):
        try:
            fit = _fit_one_initialization(
                training_samples,
                validation_samples,
                initialization=initialization,
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            attempts.append(
                {
                    "initialization": initialization,
                    "status": "failed",
                    "reason": str(error),
                }
            )
            continue
        fits.append(fit)
        attempts.append(
            {
                "initialization": initialization,
                "status": "converged",
                "iterations": fit.iterations,
                "training_mean_log_likelihood": fit.training_mean_log_likelihood,
                "validation_nll": fit.validation_nll,
                "objective_history": list(fit.objective_history),
            }
        )
    if not fits:
        raise RuntimeError("all three deterministic EM initializations failed")
    selected = min(fits, key=lambda fit: (fit.validation_nll, fit.initialization))
    return selected, attempts


def _inverse_softplus(value: float) -> float:
    """Return a stable inverse softplus for one strictly positive value."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("inverse-softplus input must be finite and positive")
    return value + math.log(-math.expm1(-value))


def _fit_as_zero_input_mdn(
    context: ResidualImageContext,
    fit: GaussianMixtureFit,
    *,
    source_provenance: str,
) -> ConditionalResidualImageMDN:
    """Encode a fitted constant GMM as a zero-input portable MDN."""
    component_count, dimension = fit.means.shape
    packed_count = dimension * (dimension + 1) // 2
    output_size = component_count * (1 + dimension + packed_count)
    output_bias = np.empty(output_size, dtype=np.float64)
    output_bias[:component_count] = np.log(fit.weights)
    mean_start = component_count
    mean_stop = mean_start + component_count * dimension
    output_bias[mean_start:mean_stop] = fit.means.reshape(-1)
    packed = np.empty((component_count, packed_count), dtype=np.float64)
    lower_rows, lower_columns = np.tril_indices(dimension)
    for component in range(component_count):
        cholesky = np.linalg.cholesky(fit.covariances[component])
        values = np.asarray(cholesky[lower_rows, lower_columns], dtype=np.float64)
        for packed_index, (row, column) in enumerate(zip(lower_rows, lower_columns, strict=True)):
            if row == column:
                adjusted = float(values[packed_index] - CHOLESKY_DIAGONAL_FLOOR)
                values[packed_index] = _inverse_softplus(adjusted)
        packed[component] = values
    output_bias[mean_stop:] = packed.reshape(-1)
    hidden_size = 1
    return ConditionalResidualImageMDN(
        context,
        np.zeros((hidden_size, 0), dtype=np.float64),
        np.zeros(hidden_size, dtype=np.float64),
        np.zeros((hidden_size, hidden_size), dtype=np.float64),
        np.zeros(hidden_size, dtype=np.float64),
        np.zeros((output_size, hidden_size), dtype=np.float64),
        output_bias,
        component_count=component_count,
        cholesky_diagonal_floor=CHOLESKY_DIAGONAL_FLOOR,
        input_center=np.empty(0, dtype=np.float64),
        input_scale=np.empty(0, dtype=np.float64),
        source_provenance=source_provenance,
    )


def _residual_image_draws(
    aggregation: AdditiveDirichletAggregation,
    labels: c1.IntArray,
    context: ResidualImageContext,
    *,
    sample_count: int,
    source_seed: int,
    source_provenance: str,
) -> tuple[FloatArray, str]:
    """Draw one scrambled-Sobol bank in the context's complete residual image."""
    projected = AdditiveDirichletAggregation(
        aggregation.cell_alphas,
        aggregation.design,
        aggregation.noise_sd,
        context.residual_basis,
    )
    bank = ConditionalAllocationMixture.from_aggregation(
        projected,
        labels,
        sample_count=sample_count,
        source_seed=source_seed,
        source_provenance=source_provenance,
        cell_ids=context.cell_ids,
        construction_method=CONSTRUCTION_METHOD,
    )
    if bank.region_count != 1:
        raise RuntimeError("the root-only GMM trainer received a non-root allocation bank")
    draws = np.asarray(
        bank.projected_unit_mass_residual_factors[:, :, 0],
        dtype=np.float64,
    )
    if draws.shape != (sample_count, context.residual_rank):
        raise RuntimeError("residual-image bank shape does not match its context")
    return draws, bank.sha256


def _array_sha256(values: FloatArray) -> str:
    """Return a shape-aware little-endian float64 array identity."""
    canonical = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256(
        c1._canonical_json(
            {
                "dtype": "<f8",
                "shape": list(canonical.shape),
            }
        ).encode("ascii")
    )
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _domain_draw_bundle(
    aggregation: AdditiveDirichletAggregation,
    labels: c1.IntArray,
    context: ResidualImageContext,
    *,
    case_id: str,
    training_sample_count: int,
    validation_sample_count: int,
    test_sample_count: int,
    base_seed: int,
) -> tuple[dict[str, FloatArray], dict[str, dict[str, Any]]]:
    """Build three independent draw domains, each exactly once."""
    domain_draws: dict[str, FloatArray] = {}
    domain_artifacts: dict[str, dict[str, Any]] = {}
    domain_counts = {
        TRAINING_DOMAIN: training_sample_count,
        VALIDATION_DOMAIN: validation_sample_count,
        TEST_DOMAIN: test_sample_count,
    }
    for domain, domain_sample_count in domain_counts.items():
        domain_seed = _domain_seed(base_seed, case_id=case_id, domain=domain)
        draws, artifact_sha256 = _residual_image_draws(
            aggregation,
            labels,
            context,
            sample_count=domain_sample_count,
            source_seed=domain_seed,
            source_provenance=(
                f"{PROTOCOL}:{case_id}:S={domain_sample_count}:base={base_seed}:domain={domain}"
            ),
        )
        domain_draws[domain] = draws
        domain_artifacts[domain] = {
            "sample_count": domain_sample_count,
            "source_seed": domain_seed,
            "artifact_sha256": artifact_sha256,
            "draws_sha256": _array_sha256(draws),
        }
    return domain_draws, domain_artifacts


def _fitted_bundle_envelope(
    artifact: ConditionalResidualImageMDN,
    *,
    case_id: str,
    context_sha256: str,
    source_git_revision: str,
    driver_sha256: str,
    domain_artifacts: dict[str, dict[str, Any]],
    training_prefix_sha256: str,
    training_sample_count: int,
    attempts: list[dict[str, Any]],
    selected_initialization: int,
    generalization: dict[str, float | int | bool],
) -> dict[str, Any]:
    """Return one canonical, self-authenticating fitted-bundle envelope."""
    payload = {
        "schema": "conditional-residual-image-fitted-bundle-v1",
        "artifact": {
            "payload": artifact.payload,
            "sha256": artifact.artifact_sha256,
        },
        "protocol": {
            "name": PROTOCOL,
            "sha256": _protocol_sha256(),
            "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
            "protected_holdout_catalogue_sha256": (PROTECTED_HOLDOUT_CATALOGUE_SHA256),
        },
        "source": {
            "git_revision": source_git_revision,
            "driver_sha256": driver_sha256,
        },
        "case": {
            "case_id": case_id,
            "context_sha256": context_sha256,
        },
        "domains": domain_artifacts,
        "training": {
            "dtype": "<f8",
            "training_sample_count": training_sample_count,
            "training_prefix_sha256": training_prefix_sha256,
            "initialization_attempts": attempts,
            "selected_initialization": selected_initialization,
            "valid_initialization_count": sum(attempt["status"] == "converged" for attempt in attempts),
            "minimum_valid_initializations": (MINIMUM_VALID_INITIALIZATIONS),
        },
        "runtime": {
            "numpy_version": numpy_version,
            "scipy_version": scipy_version,
        },
        "generalization": generalization,
    }
    return {
        "payload": payload,
        "sha256": c1._sha256_json(payload),
    }


def validate_fitted_bundle_envelope(
    envelope: object,
    *,
    expected_sha256: str | None = None,
    expected_source_git_revision: str | None = None,
    expected_driver_sha256: str | None = None,
) -> ConditionalResidualImageMDN:
    """Authenticate and reconstruct one fitted-bundle envelope."""
    if not isinstance(envelope, dict) or set(envelope) != {
        "payload",
        "sha256",
    }:
        raise ValueError("fitted-bundle envelope has an unexpected schema")
    payload = envelope["payload"]
    observed_sha256 = c1._sha256_json(payload)
    if envelope["sha256"] != observed_sha256:
        raise ValueError("fitted-bundle envelope digest does not match its payload")
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("fitted-bundle envelope does not match expected_sha256")
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "artifact",
        "protocol",
        "source",
        "case",
        "domains",
        "training",
        "runtime",
        "generalization",
    }:
        raise ValueError("fitted-bundle payload has an unexpected schema")
    if payload["schema"] != "conditional-residual-image-fitted-bundle-v1":
        raise ValueError("fitted-bundle payload uses an unexpected schema version")
    protocol = payload["protocol"]
    if not isinstance(protocol, dict) or protocol != {
        "name": PROTOCOL,
        "sha256": _protocol_sha256(),
        "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
        "protected_holdout_catalogue_sha256": (PROTECTED_HOLDOUT_CATALOGUE_SHA256),
    }:
        raise ValueError("fitted-bundle protocol identity does not match")
    source = payload["source"]
    if not isinstance(source, dict) or set(source) != {
        "git_revision",
        "driver_sha256",
    }:
        raise ValueError("fitted-bundle source identity is malformed")
    git_revision = source["git_revision"]
    driver_sha256 = source["driver_sha256"]
    if (
        not isinstance(git_revision, str)
        or len(git_revision) != 40
        or any(character not in "0123456789abcdef" for character in git_revision)
        or not isinstance(driver_sha256, str)
        or len(driver_sha256) != 64
        or any(character not in "0123456789abcdef" for character in driver_sha256)
    ):
        raise ValueError("fitted-bundle source identities are not canonical digests")
    if expected_source_git_revision is not None and git_revision != expected_source_git_revision:
        raise ValueError("fitted-bundle source Git revision does not match")
    if expected_driver_sha256 is not None and driver_sha256 != expected_driver_sha256:
        raise ValueError("fitted-bundle driver digest does not match")
    artifact_record = payload["artifact"]
    case = payload["case"]
    if (
        not isinstance(artifact_record, dict)
        or set(artifact_record) != {"payload", "sha256"}
        or not isinstance(case, dict)
        or set(case) != {"case_id", "context_sha256"}
    ):
        raise ValueError("fitted-bundle artifact or case identity is malformed")
    serialized = c1._canonical_json(artifact_record["payload"])
    artifact = ConditionalResidualImageMDN.from_json(
        serialized,
        expected_sha256=cast(str, artifact_record["sha256"]),
    )
    if artifact.context.artifact_sha256 != case["context_sha256"]:
        raise ValueError("fitted-bundle artifact context does not match its case")
    training = payload["training"]
    if (
        not isinstance(training, dict)
        or set(training)
        != {
            "dtype",
            "training_sample_count",
            "training_prefix_sha256",
            "initialization_attempts",
            "selected_initialization",
            "valid_initialization_count",
            "minimum_valid_initializations",
        }
        or training.get("dtype") != "<f8"
    ):
        raise ValueError("fitted-bundle training dtype is not canonical float64")
    attempts = training.get("initialization_attempts")
    selected = training.get("selected_initialization")
    if (
        not isinstance(attempts, list)
        or len(attempts) != INITIALIZATION_COUNT
        or not isinstance(selected, int)
        or not any(
            isinstance(attempt, dict)
            and attempt.get("status") == "converged"
            and attempt.get("initialization") == selected
            for attempt in attempts
        )
    ):
        raise ValueError("fitted-bundle initialization evidence is insufficient")
    valid_count = sum(
        isinstance(attempt, dict) and attempt.get("status") == "converged" for attempt in attempts
    )
    if (
        training["valid_initialization_count"] != valid_count
        or training["minimum_valid_initializations"] != MINIMUM_VALID_INITIALIZATIONS
    ):
        raise ValueError("fitted-bundle valid-initialization count is inconsistent")
    domains = payload["domains"]
    if not isinstance(domains, dict) or set(domains) != {
        TRAINING_DOMAIN,
        VALIDATION_DOMAIN,
        TEST_DOMAIN,
    }:
        raise ValueError("fitted-bundle simulator domains are malformed")
    for domain in domains.values():
        if (
            not isinstance(domain, dict)
            or set(domain)
            != {
                "sample_count",
                "source_seed",
                "artifact_sha256",
                "draws_sha256",
            }
            or not isinstance(domain["sample_count"], int)
            or domain["sample_count"] < 1
            or not isinstance(domain["source_seed"], int)
            or not 0 <= domain["source_seed"] < 2**64
            or any(
                not isinstance(domain[name], str)
                or len(domain[name]) != 64
                or any(character not in "0123456789abcdef" for character in domain[name])
                for name in ("artifact_sha256", "draws_sha256")
            )
        ):
            raise ValueError("fitted-bundle simulator-domain evidence is malformed")
    runtime = payload["runtime"]
    if not isinstance(runtime, dict) or runtime != {
        "numpy_version": DEVELOPMENT_NUMPY_VERSION,
        "scipy_version": DEVELOPMENT_SCIPY_VERSION,
    }:
        raise ValueError("fitted-bundle runtime identity does not match")
    generalization = payload["generalization"]
    if not isinstance(generalization, dict) or not isinstance(generalization.get("pass"), bool):
        raise ValueError("fitted-bundle simulator-test generalization evidence is malformed")
    return artifact


def _fit_training_bundle(
    context: ResidualImageContext,
    *,
    case_id: str,
    training_draws: FloatArray,
    validation_draws: FloatArray,
    test_draws: FloatArray,
    domain_artifacts: dict[str, dict[str, Any]],
    sample_count: int,
    base_seed: int,
    source_git_revision: str,
    driver_sha256: str,
) -> tuple[ConditionalResidualImageMDN, dict[str, Any]]:
    """Fit one prefix and score it on fixed independent draw domains."""
    if sample_count > training_draws.shape[0]:
        raise ValueError("training prefix exceeds the authenticated training bank")
    training_prefix = np.asarray(training_draws[:sample_count], dtype=np.float64)
    fit, attempts = fit_gaussian_mixture(
        training_prefix,
        validation_draws,
    )
    generalization = _simulator_test_generalization(
        validation_draws,
        test_draws,
        fit,
    )
    valid_initialization_count = sum(attempt["status"] == "converged" for attempt in attempts)
    valid_initialization_pass = valid_initialization_count >= MINIMUM_VALID_INITIALIZATIONS
    provenance = f"{PROTOCOL}:{case_id}:S={sample_count}:base={base_seed}"
    artifact = _fit_as_zero_input_mdn(
        context,
        fit,
        source_provenance=provenance,
    )
    training_prefix_sha256 = _array_sha256(training_prefix)
    envelope = _fitted_bundle_envelope(
        artifact,
        case_id=case_id,
        context_sha256=context.artifact_sha256,
        source_git_revision=source_git_revision,
        driver_sha256=driver_sha256,
        domain_artifacts=domain_artifacts,
        training_prefix_sha256=training_prefix_sha256,
        training_sample_count=sample_count,
        attempts=attempts,
        selected_initialization=fit.initialization,
        generalization=generalization,
    )
    validate_fitted_bundle_envelope(
        envelope,
        expected_sha256=cast(str, envelope["sha256"]),
        expected_source_git_revision=source_git_revision,
        expected_driver_sha256=driver_sha256,
    )
    fit_development_pass = bool(valid_initialization_pass and generalization["pass"])
    return artifact, {
        "training_sample_count": sample_count,
        "training_prefix_sha256": training_prefix_sha256,
        "validation_sample_count": int(validation_draws.shape[0]),
        "test_sample_count": int(test_draws.shape[0]),
        "base_seed": base_seed,
        "domain_artifacts": domain_artifacts,
        "initialization_attempts": attempts,
        "valid_initialization_count": valid_initialization_count,
        "minimum_valid_initializations": (MINIMUM_VALID_INITIALIZATIONS),
        "valid_initialization_pass": valid_initialization_pass,
        "selected_initialization": fit.initialization,
        "iterations": fit.iterations,
        "convergence_streak": fit.convergence_streak,
        "training_mean_log_likelihood": fit.training_mean_log_likelihood,
        "validation_nll": fit.validation_nll,
        "test_nll": generalization["simulator_test_nll_nat_per_draw"],
        "simulator_test_generalization": generalization,
        "fit_development_pass": fit_development_pass,
        "artifact_sha256": artifact.artifact_sha256,
        "artifact_payload": artifact.payload,
        "fitted_bundle_envelope": envelope,
    }


def _evaluate_artifact(
    *,
    artifact: ConditionalResidualImageMDN,
    observation: FloatArray,
    masses: FloatArray,
    log_prior: FloatArray,
    exact_log_likelihood: FloatArray,
    exact_summary: dict[str, Any],
    gradient_states: Sequence[dict[str, Any]],
    validation_state_mask: NDArray[np.bool_],
) -> dict[str, Any]:
    """Apply the unchanged C1 pointwise, posterior, evidence, and FD gates."""
    learned_log_likelihood = np.asarray(
        [artifact.log_likelihood(observation, state) for state in masses],
        dtype=np.float64,
    )
    error = np.abs(learned_log_likelihood - exact_log_likelihood)
    learned_summary = c1._posterior_summary(masses, log_prior, learned_log_likelihood)
    summary_errors, by_coordinate = c1._summary_errors(exact_summary, learned_summary)
    validation_prior_log_weights = log_prior[validation_state_mask]
    validation_prior_weights = np.exp(
        validation_prior_log_weights - c1._stable_logsumexp(validation_prior_log_weights)
    )
    validation_posterior_log_weights = (
        validation_prior_log_weights + exact_log_likelihood[validation_state_mask]
    )
    validation_posterior_weights = np.exp(
        validation_posterior_log_weights - c1._stable_logsumexp(validation_posterior_log_weights)
    )
    validation_error = error[validation_state_mask]
    gradient_audits: list[dict[str, Any]] = []
    for state in gradient_states:
        coordinate = np.asarray(state["coordinate"], dtype=np.float64)

        def learned_function(value: FloatArray) -> float:
            return artifact.log_likelihood(
                observation,
                c1.coordinate_to_masses(value),
            )

        learned_gradient = c1._centered_gradient(learned_function, coordinate)
        exact_gradient = np.asarray(state["exact_coordinate_gradient"], dtype=np.float64)
        scaled_error = float(
            np.max(np.abs(learned_gradient - exact_gradient) / (1.0 + np.abs(exact_gradient)))
        )
        gradient_audits.append(
            {
                "state_id": state["state_id"],
                "coordinate": coordinate.tolist(),
                "exact_coordinate_gradient": exact_gradient.tolist(),
                "learned_coordinate_gradient": learned_gradient.tolist(),
                "scaled_error": scaled_error,
            }
        )
    metrics = {
        "median_absolute_conditional_log_likelihood_error_nat": c1._weighted_quantile(
            validation_error,
            validation_prior_weights,
            0.5,
        ),
        "p99_absolute_conditional_log_likelihood_error_nat": c1._weighted_quantile(
            validation_error,
            validation_posterior_weights,
            0.99,
        ),
        "scaled_coordinate_gradient_error": max(audit["scaled_error"] for audit in gradient_audits),
        "absolute_log_evidence_error_nat": abs(
            learned_summary["log_evidence"] - exact_summary["log_evidence"]
        ),
        **summary_errors,
    }
    checks = {
        name: bool(metrics[name] <= threshold) for name, threshold in c1.THRESHOLDS.items() if name in metrics
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "scientific_pass": bool(all(checks.values())),
        "posterior_summary": learned_summary,
        "posterior_errors_by_coordinate": by_coordinate,
        "gradient_audits": gradient_audits,
        "diagnostics": {
            "full_grid_median_absolute_log_likelihood_error_nat": float(np.median(error)),
            "full_grid_maximum_absolute_log_likelihood_error_nat": float(np.max(error)),
        },
    }


def _repeat_evidence_gate(
    evaluations: Sequence[dict[str, Any]],
) -> tuple[float | None, bool]:
    """Apply the unchanged C1 evidence-stability gate to repeat banks."""
    if not evaluations:
        return None, False
    evidence = [float(evaluation["posterior_summary"]["log_evidence"]) for evaluation in evaluations]
    if not all(np.isfinite(value) for value in evidence):
        raise ValueError("repeat-bank log evidence must be finite")
    evidence_range = float(max(evidence) - min(evidence))
    return (
        evidence_range,
        bool(evidence_range <= c1.THRESHOLDS["between_bank_log_evidence_range_nat"]),
    )


def run_case(
    *,
    regime_name: str,
    family: c1.Family,
    sample_counts: Sequence[int],
    repeat_seeds: Sequence[int],
    profile: Profile,
    source_git_revision: str,
    driver_sha256: str,
    run_development_ladder: bool = True,
    development_sample_count: int | None = None,
    confirmation_sample_count: int | None = None,
    confirmation_seed: int | None = None,
) -> dict[str, Any]:
    """Train one case's ladder and/or a preselected common confirmation size."""
    case_key = (regime_name, family, "root")
    allowed = SMOKE_MATRIX if profile == "smoke" else DEVELOPMENT_MATRIX
    if case_key not in allowed:
        raise ValueError(f"case {case_key!r} is not available in {profile}")
    normalized_counts = tuple(int(value) for value in sample_counts)
    normalized_seeds = tuple(int(value) for value in repeat_seeds)
    if (
        not normalized_counts
        or normalized_counts != tuple(sorted(set(normalized_counts)))
        or any(value < COMPONENT_COUNT or value & (value - 1) for value in normalized_counts)
    ):
        raise ValueError("sample counts must be unique increasing powers of two")
    if (
        not normalized_seeds
        or len(set(normalized_seeds)) != len(normalized_seeds)
        or any(value < 0 or value >= 2**64 for value in normalized_seeds)
    ):
        raise ValueError("repeat seeds must be unique unsigned 64-bit integers")
    if profile == "development" and (
        normalized_counts != DEVELOPMENT_SAMPLE_COUNTS
        or normalized_seeds != (DEVELOPMENT_SELECTION_SEED, *CONFIRMATION_SEEDS)
    ):
        raise ValueError("development sample counts and seeds are source-pinned")
    if not run_development_ladder and confirmation_sample_count is None:
        raise ValueError("confirmation-only execution requires a preselected sample count")
    if development_sample_count is not None and (
        not run_development_ladder or development_sample_count not in normalized_counts
    ):
        raise ValueError("development_sample_count must select one source-pinned ladder size")
    if confirmation_sample_count is not None and confirmation_sample_count not in normalized_counts:
        raise ValueError("confirmation_sample_count must lie on the source-pinned ladder")
    if confirmation_seed is not None and (
        confirmation_sample_count is None or confirmation_seed not in normalized_seeds[1:]
    ):
        raise ValueError("confirmation_seed must select one source-pinned confirmation stream")
    regime = c1._regime(regime_name)
    shapes, rate, design, observation, noise = c1._case_arrays(regime, family)
    labels = c1.labels_for_tiling(family, "root")
    total_order = 8 if profile == "smoke" else regime.total_order
    fraction_order = 6 if profile == "smoke" else regime.fraction_order
    masses, log_prior = c1._mass_grid(
        shapes=shapes,
        rate=rate,
        family=family,
        tiling="root",
        total_order=total_order,
        fraction_order=fraction_order,
    )
    exact_log_likelihood = c1._exact_log_likelihood(
        masses=masses,
        shapes=shapes,
        rate=rate,
        design=design,
        observation=observation,
        noise=noise,
        family=family,
        tiling="root",
        total_order=total_order,
        fraction_order=fraction_order,
    )
    exact_summary = c1._posterior_summary(masses, log_prior, exact_log_likelihood)
    prior_mean_coordinate = c1._anchor_coordinate(shapes, rate, labels)

    def exact_function(value: FloatArray) -> float:
        return float(
            c1._exact_log_likelihood(
                masses=c1.coordinate_to_masses(value)[None, :],
                shapes=shapes,
                rate=rate,
                design=design,
                observation=observation,
                noise=noise,
                family=family,
                tiling="root",
                total_order=total_order,
                fraction_order=fraction_order,
            )[0]
        )

    gradient_states = [
        {
            "state_id": state_id,
            "coordinate": coordinate.tolist(),
            "exact_coordinate_gradient": c1._centered_gradient(
                exact_function,
                coordinate,
            ).tolist(),
        }
        for state_id, coordinate in c1._gradient_state_coordinates(
            masses=masses,
            log_prior=log_prior,
            exact_log_likelihood=exact_log_likelihood,
            prior_mean_coordinate=prior_mean_coordinate,
        )
    ]
    validation_state_mask = c1._development_validation_state_mask(
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
    cell_ids = np.arange(shapes.size, dtype=np.int64)
    case_id = f"{regime_name}__{family}__root"
    context = ResidualImageContext.from_aggregation(
        aggregation,
        labels,
        cell_ids,
        source_provenance=f"{PROTOCOL}:{case_id}:residual-image-context",
    )
    if context.residual_rank == 0:
        raise RuntimeError("the learned-density matrix excludes zero-rank controls")

    validation_sample_count = SMOKE_SAMPLE_COUNTS[0] if profile == "smoke" else VALIDATION_SAMPLE_COUNT
    test_sample_count = SMOKE_SAMPLE_COUNTS[0] if profile == "smoke" else TEST_SAMPLE_COUNT

    def fit_and_evaluate(
        sample_count: int,
        base_seed: int,
        domain_draws: dict[str, FloatArray],
        domain_artifacts: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        artifact, training = _fit_training_bundle(
            context,
            case_id=case_id,
            training_draws=domain_draws[TRAINING_DOMAIN],
            validation_draws=domain_draws[VALIDATION_DOMAIN],
            test_draws=domain_draws[TEST_DOMAIN],
            domain_artifacts=domain_artifacts,
            sample_count=sample_count,
            base_seed=base_seed,
            source_git_revision=source_git_revision,
            driver_sha256=driver_sha256,
        )
        fit_seconds = time.perf_counter() - started
        result = _evaluate_artifact(
            artifact=artifact,
            observation=observation,
            masses=masses,
            log_prior=log_prior,
            exact_log_likelihood=exact_log_likelihood,
            exact_summary=exact_summary,
            gradient_states=gradient_states,
            validation_state_mask=validation_state_mask,
        )
        result["scientific_model_gates_pass"] = result["scientific_pass"]
        result["fit_development_pass"] = training["fit_development_pass"]
        result["scientific_pass"] = bool(
            result["scientific_model_gates_pass"] and result["fit_development_pass"]
        )
        result["training"] = training
        result["sample_count"] = sample_count
        result["base_seed"] = base_seed
        result["fit_and_evaluate_seconds"] = fit_seconds
        return result

    development_evaluations: list[dict[str, Any]] = []
    development_nested_training_bank: dict[str, Any] | None = None
    if run_development_ladder:
        development_draws, development_domain_artifacts = _domain_draw_bundle(
            aggregation,
            labels,
            context,
            case_id=case_id,
            training_sample_count=max(normalized_counts),
            validation_sample_count=validation_sample_count,
            test_sample_count=test_sample_count,
            base_seed=normalized_seeds[0],
        )
        evaluation_counts = (
            (development_sample_count,) if development_sample_count is not None else normalized_counts
        )
        development_evaluations = [
            fit_and_evaluate(
                sample_count,
                normalized_seeds[0],
                development_draws,
                development_domain_artifacts,
            )
            for sample_count in evaluation_counts
        ]
        development_nested_training_bank = {
            "largest_sample_count": max(normalized_counts),
            "artifact_sha256": development_domain_artifacts[TRAINING_DOMAIN]["artifact_sha256"],
            "full_draws_sha256": development_domain_artifacts[TRAINING_DOMAIN]["draws_sha256"],
            "prefixes": {
                str(evaluation["sample_count"]): evaluation["training"]["training_prefix_sha256"]
                for evaluation in development_evaluations
            },
        }
    minimum_suffix_length = 1 if profile == "smoke" else 2
    diagnostic_locked_sample_count = (
        c1._stable_lock_sample_count(
            normalized_counts,
            [bool(result["scientific_pass"]) for result in development_evaluations],
            minimum_suffix_length=minimum_suffix_length,
        )
        if development_evaluations and development_sample_count is None
        else None
    )
    confirmation_evaluations: list[dict[str, Any]] = []
    if confirmation_sample_count is not None:
        confirmation_base_seeds = (
            (confirmation_seed,) if confirmation_seed is not None else normalized_seeds[1:]
        )
        for base_seed in confirmation_base_seeds:
            assert base_seed is not None
            confirmation_draws, confirmation_domain_artifacts = _domain_draw_bundle(
                aggregation,
                labels,
                context,
                case_id=case_id,
                training_sample_count=confirmation_sample_count,
                validation_sample_count=validation_sample_count,
                test_sample_count=test_sample_count,
                base_seed=base_seed,
            )
            confirmation_evaluations.append(
                fit_and_evaluate(
                    confirmation_sample_count,
                    base_seed,
                    confirmation_draws,
                    confirmation_domain_artifacts,
                )
            )
    locked_development = next(
        (
            result
            for result in development_evaluations
            if result["sample_count"] == diagnostic_locked_sample_count
        ),
        None,
    )
    repeat_evaluations = (
        [locked_development, *confirmation_evaluations]
        if locked_development is not None
        else confirmation_evaluations
    )
    evidence_range, evidence_range_pass = _repeat_evidence_gate(repeat_evaluations)
    confirmation_pass_without_repeat_evidence_gate = (
        bool(all(result["scientific_pass"] for result in confirmation_evaluations))
        if confirmation_evaluations
        else None
    )
    confirmation_pass = (
        bool(confirmation_pass_without_repeat_evidence_gate is True and evidence_range_pass)
        if confirmation_evaluations
        else None
    )
    scientific_pass = bool(
        profile == "smoke"
        and diagnostic_locked_sample_count is not None
        and evidence_range_pass
        or (
            profile == "development"
            and diagnostic_locked_sample_count == confirmation_sample_count
            and confirmation_pass is True
        )
    )
    return {
        "case_id": case_id,
        "profile": profile,
        "input_sha256": c1._case_input_sha256(
            regime,
            family,
            "root",
            total_order,
            fraction_order,
        ),
        "context_sha256": context.artifact_sha256,
        "residual_image_rank": context.residual_rank,
        "quadrature": {
            "total_order": total_order,
            "fraction_order": fraction_order,
            "mass_state_count": int(masses.shape[0]),
        },
        "exact_posterior_summary": exact_summary,
        "development_seed": normalized_seeds[0],
        "confirmation_seeds": list(normalized_seeds[1:]),
        "executed_development_sample_count": development_sample_count,
        "executed_confirmation_seed": confirmation_seed,
        "development_evaluations": development_evaluations,
        "development_nested_training_bank": development_nested_training_bank,
        "minimum_passing_suffix_length": minimum_suffix_length,
        "locked_sample_count": diagnostic_locked_sample_count,
        "diagnostic_per_case_lock_only": profile == "development",
        "confirmation_sample_count": confirmation_sample_count,
        "confirmation_evaluations": confirmation_evaluations,
        "confirmation_pass_without_repeat_evidence_gate": (confirmation_pass_without_repeat_evidence_gate),
        "between_bank_log_evidence_range_nat": evidence_range,
        "between_bank_log_evidence_range_pass": evidence_range_pass,
        "confirmation_pass": confirmation_pass,
        "scientific_pass": scientific_pass,
        "structural_inference_licensed": False,
    }


def matrix_catalogue() -> dict[str, Any]:
    """Return the executable matrix and opaque protected catalogue identity."""
    return {
        "schema": SCHEMA,
        "development": [list(case) for case in DEVELOPMENT_MATRIX],
        "smoke": [list(case) for case in SMOKE_MATRIX],
        "protected_holdout": {
            "id": PROTECTED_HOLDOUT_CATALOGUE_ID,
            "sha256": PROTECTED_HOLDOUT_CATALOGUE_SHA256,
            "sample_count": PROTECTED_HOLDOUT_SAMPLE_COUNT,
            "object": "new residual draws from the same six exact contexts",
            "promoted_artifact": ("development seed 731 at the common locked training size"),
            "retrain": False,
            "retune_after_reveal": False,
            "numerical_values_present": False,
            "executable_here": False,
        },
        "held_out_information_read": False,
    }


def _read_confirmation_lock(
    path: Path,
    *,
    expected_raw_sha256: str,
    expected_source_revision: str,
    expected_driver_sha256: str,
    selected_case_id: str,
) -> tuple[dict[str, Any], str, str]:
    """Authenticate a common development lock before confirmation fitting."""
    if len(expected_raw_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_raw_sha256
    ):
        raise ValueError("expected confirmation-lock SHA-256 is not canonical")
    if path.is_symlink() or not path.is_file():
        raise ValueError("confirmation lock must be one regular non-symlink file")
    raw = path.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != expected_raw_sha256:
        raise ValueError("confirmation lock raw SHA-256 does not match")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"confirmation lock contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        text = raw.decode("ascii")
        envelope = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"confirmation lock contains non-finite value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("confirmation lock is not strict ASCII JSON") from error
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "sha256"}:
        raise ValueError("confirmation lock has an unexpected envelope schema")
    if text != f"{c1._canonical_json(envelope)}\n":
        raise ValueError("confirmation lock is not newline-terminated canonical JSON")
    payload = envelope["payload"]
    internal_sha256 = c1._sha256_json(payload)
    if envelope["sha256"] != internal_sha256:
        raise ValueError("confirmation lock internal digest does not match")
    expected_payload_fields = {
        "schema",
        "certification_protocol",
        "certification_protocol_sha256",
        "source_git_revision",
        "scientific_driver_sha256",
        "frozen_development_protocol_sha256",
        "a1_definitions_sha256",
        "matrix_catalogue",
        "sample_counts",
        "development_selection_seed",
        "confirmation_seeds",
        "minimum_passing_suffix_length",
        "development_pass_pattern",
        "locked_sample_count",
        "cases",
    }
    if not isinstance(payload, dict) or set(payload) != expected_payload_fields:
        raise ValueError("confirmation lock payload has an unexpected schema")
    if (
        payload["schema"] != "conditional-residual-image-gmm-common-lock-v1"
        or not isinstance(payload["certification_protocol"], str)
        or not payload["certification_protocol"]
        or not isinstance(payload["certification_protocol_sha256"], str)
        or len(payload["certification_protocol_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in payload["certification_protocol_sha256"])
        or payload["source_git_revision"] != expected_source_revision
        or payload["scientific_driver_sha256"] != expected_driver_sha256
        or payload["frozen_development_protocol_sha256"] != DEVELOPMENT_PROTOCOL_SHA256
        or payload["a1_definitions_sha256"] != c1.A1_DEFINITIONS_SHA256
        or payload["matrix_catalogue"] != matrix_catalogue()
        or payload["sample_counts"] != list(DEVELOPMENT_SAMPLE_COUNTS)
        or payload["development_selection_seed"] != DEVELOPMENT_SELECTION_SEED
        or payload["confirmation_seeds"] != list(CONFIRMATION_SEEDS)
        or payload["minimum_passing_suffix_length"] != 2
    ):
        raise ValueError("confirmation lock does not match the frozen protocol")
    locked_sample_count = payload["locked_sample_count"]
    if not isinstance(locked_sample_count, int) or locked_sample_count not in DEVELOPMENT_SAMPLE_COUNTS:
        raise ValueError("confirmation lock has no valid common sample count")
    cases = payload["cases"]
    expected_case_ids = {"__".join(case) for case in DEVELOPMENT_MATRIX}
    if not isinstance(cases, dict) or set(cases) != expected_case_ids:
        raise ValueError("confirmation lock does not contain exactly six cases")
    case = cases[selected_case_id]
    if not isinstance(case, dict) or set(case) != {
        "development_input_raw_sha256",
        "input_sha256",
        "context_sha256",
        "nominated_fitted_bundle_sha256",
        "nominated_artifact_sha256",
    }:
        raise ValueError("confirmation lock selected-case record is malformed")
    return payload, internal_sha256, raw_sha256


def run_screen(
    *,
    profile: Profile,
    sample_counts: Sequence[int] | None = None,
    repeat_seeds: Sequence[int] | None = None,
    case_id: str | None = None,
    source_revision: str | None = None,
    development_sample_count: int | None = None,
    confirmation_lock: Path | None = None,
    expected_confirmation_lock_sha256: str | None = None,
    confirmation_seed: int | None = None,
) -> dict[str, Any]:
    """Run the smoke profile or the source-pinned six-case development screen."""
    if profile not in ("smoke", "development"):
        raise ValueError("protected held-out execution is deliberately unavailable")
    observed_a1_definitions = c1.a1_definitions_sha256()
    if observed_a1_definitions != c1.A1_DEFINITIONS_SHA256:
        raise RuntimeError("shared C1 exact numerical definitions no longer match their pin")
    resolved_revision = c1._source_revision(source_revision)
    observed_driver_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    confirmation_mode = confirmation_lock is not None
    if confirmation_mode != (expected_confirmation_lock_sha256 is not None):
        raise ValueError("confirmation lock and expected raw SHA-256 must be supplied together")
    if confirmation_mode and (profile != "development" or case_id is None):
        raise ValueError("locked confirmation requires development profile and one case_id")
    development_shard_mode = development_sample_count is not None
    if development_shard_mode and (
        profile != "development"
        or case_id is None
        or confirmation_mode
        or development_sample_count not in DEVELOPMENT_SAMPLE_COUNTS
    ):
        raise ValueError("development shard requires one source-pinned development case and size")
    if confirmation_seed is not None and (
        not confirmation_mode or confirmation_seed not in CONFIRMATION_SEEDS
    ):
        raise ValueError("confirmation_seed requires a signed lock and source-pinned seed")
    if profile == "development":
        _validate_development_protocol()
        if sample_counts is not None or repeat_seeds is not None:
            raise ValueError("development counts and seeds are source-pinned")
        counts = DEVELOPMENT_SAMPLE_COUNTS
        seeds = (DEVELOPMENT_SELECTION_SEED, *CONFIRMATION_SEEDS)
        matrix = DEVELOPMENT_MATRIX
    else:
        counts = SMOKE_SAMPLE_COUNTS if sample_counts is None else tuple(sample_counts)
        seeds = SMOKE_REPEAT_SEEDS if repeat_seeds is None else tuple(repeat_seeds)
        matrix = SMOKE_MATRIX
    if case_id is not None:
        matches = [case for case in matrix if "__".join(case) == case_id]
        if len(matches) != 1:
            raise ValueError(f"case_id {case_id!r} is not available in profile {profile}")
        matrix = tuple(matches)
    started = time.perf_counter()
    confirmation_lock_payload: dict[str, Any] | None = None
    confirmation_lock_internal_sha256: str | None = None
    confirmation_lock_raw_sha256: str | None = None
    confirmation_sample_count: int | None = None
    if confirmation_mode:
        assert confirmation_lock is not None
        assert expected_confirmation_lock_sha256 is not None
        assert case_id is not None
        (
            confirmation_lock_payload,
            confirmation_lock_internal_sha256,
            confirmation_lock_raw_sha256,
        ) = _read_confirmation_lock(
            confirmation_lock,
            expected_raw_sha256=expected_confirmation_lock_sha256,
            expected_source_revision=resolved_revision,
            expected_driver_sha256=observed_driver_sha256,
            selected_case_id=case_id,
        )
        confirmation_sample_count = cast(int, confirmation_lock_payload["locked_sample_count"])
    cases = [
        run_case(
            regime_name=regime_name,
            family=cast(c1.Family, family),
            sample_counts=counts,
            repeat_seeds=seeds,
            profile=profile,
            source_git_revision=resolved_revision,
            driver_sha256=observed_driver_sha256,
            run_development_ladder=not confirmation_mode,
            development_sample_count=development_sample_count,
            confirmation_sample_count=confirmation_sample_count,
            confirmation_seed=confirmation_seed,
        )
        for regime_name, family, _ in matrix
    ]
    if confirmation_mode:
        return {
            "schema": SCHEMA,
            "protocol": PROTOCOL,
            "profile": profile,
            "execution_mode": (
                "confirmation_seed_shard" if confirmation_seed is not None else "confirmation_case"
            ),
            "selected_case_id": case_id,
            "per_case_atomic_output": True,
            "executed_confirmation_seed": confirmation_seed,
            "source_git_revision": resolved_revision,
            "driver_sha256": observed_driver_sha256,
            "a1_definitions_sha256": observed_a1_definitions,
            "protocol_sha256": _protocol_sha256(),
            "frozen_development_protocol_sha256": DEVELOPMENT_PROTOCOL_SHA256,
            "matrix_catalogue": matrix_catalogue(),
            "sample_counts": list(counts),
            "repeat_seeds": list(seeds),
            "confirmation_lock_internal_sha256": (confirmation_lock_internal_sha256),
            "confirmation_lock_raw_sha256": confirmation_lock_raw_sha256,
            "confirmation_locked_sample_count": confirmation_sample_count,
            "cases": cases,
            "development_pass": False,
            "eligible_for_protected_holdout": False,
            "protected_holdout_pass": None,
            "scientific_pass": False,
            "scientific_pass_available": False,
            "structural_inference_licensed": False,
            "held_out_information_read": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
    common_lock_status = "not_applicable_smoke"
    common_locked_sample_count: int | None = None
    common_development_pass_pattern: list[dict[str, Any]] = []
    common_confirmation_evidence: dict[str, Any] = {
        "requested": False,
        "complete": False,
        "all_cases_pass": None,
        "cases": {},
    }
    if profile == "development" and not development_shard_mode:
        common_development_pass_pattern = [
            {
                "sample_count": sample_count,
                "pass": bool(
                    all(
                        case["development_evaluations"][sample_index]["sample_count"] == sample_count
                        and case["development_evaluations"][sample_index]["scientific_pass"]
                        for case in cases
                    )
                ),
            }
            for sample_index, sample_count in enumerate(counts)
        ]
        if case_id is not None:
            common_lock_status = "unavailable_partial_matrix"
        else:
            common_locked_sample_count = c1._stable_lock_sample_count(
                counts,
                [bool(entry["pass"]) for entry in common_development_pass_pattern],
                minimum_suffix_length=2,
            )
            if common_locked_sample_count is None:
                common_lock_status = (
                    "hard_stop_isolated_largest_size_pass"
                    if common_development_pass_pattern[-1]["pass"]
                    else "hard_stop_no_common_two_size_passing_suffix"
                )
            else:
                common_lock_status = "locked_and_confirmed"
                confirmation_cases = [
                    run_case(
                        regime_name=regime_name,
                        family=cast(c1.Family, family),
                        sample_counts=counts,
                        repeat_seeds=seeds,
                        profile="development",
                        source_git_revision=resolved_revision,
                        driver_sha256=observed_driver_sha256,
                        run_development_ladder=False,
                        confirmation_sample_count=common_locked_sample_count,
                    )
                    for regime_name, family, _ in matrix
                ]
                confirmation_by_id = {case["case_id"]: case for case in confirmation_cases}
                for case in cases:
                    confirmation = confirmation_by_id[case["case_id"]]
                    if (
                        confirmation["context_sha256"] != case["context_sha256"]
                        or confirmation["input_sha256"] != case["input_sha256"]
                    ):
                        raise RuntimeError("confirmation case does not match its development context")
                    evaluations = confirmation["confirmation_evaluations"]
                    locked_development = next(
                        (
                            evaluation
                            for evaluation in case["development_evaluations"]
                            if evaluation["sample_count"] == common_locked_sample_count
                        ),
                        None,
                    )
                    if locked_development is None:
                        raise RuntimeError("common locked development evaluation is absent")
                    evidence_range, evidence_range_pass = _repeat_evidence_gate(
                        [locked_development, *evaluations]
                    )
                    confirmation_pass_without_repeat_evidence_gate = bool(
                        len(evaluations) == len(CONFIRMATION_SEEDS)
                        and all(evaluation["scientific_pass"] for evaluation in evaluations)
                    )
                    confirmation_pass = bool(
                        confirmation_pass_without_repeat_evidence_gate and evidence_range_pass
                    )
                    case["confirmation_sample_count"] = common_locked_sample_count
                    case["confirmation_evaluations"] = evaluations
                    case["confirmation_pass_without_repeat_evidence_gate"] = (
                        confirmation_pass_without_repeat_evidence_gate
                    )
                    case["between_bank_log_evidence_range_nat"] = evidence_range
                    case["between_bank_log_evidence_range_pass"] = evidence_range_pass
                    case["confirmation_pass"] = confirmation_pass
                    case["scientific_pass"] = confirmation_pass
                    common_confirmation_evidence["cases"][case["case_id"]] = {
                        "pass": confirmation_pass,
                        "pass_without_repeat_evidence_gate": (confirmation_pass_without_repeat_evidence_gate),
                        "between_bank_log_evidence_range_nat": evidence_range,
                        "between_bank_log_evidence_range_pass": (evidence_range_pass),
                        "base_seeds": [evaluation["base_seed"] for evaluation in evaluations],
                        "artifact_sha256": [
                            evaluation["training"]["artifact_sha256"] for evaluation in evaluations
                        ],
                    }
                common_confirmation_evidence["requested"] = True
                common_confirmation_evidence["complete"] = bool(
                    all(len(case["confirmation_evaluations"]) == len(CONFIRMATION_SEEDS) for case in cases)
                )
                common_confirmation_evidence["all_cases_pass"] = bool(
                    all(case["confirmation_pass"] is True for case in cases)
                )
                if not common_confirmation_evidence["all_cases_pass"]:
                    common_lock_status = "hard_stop_common_confirmation_failure"
    elif development_shard_mode:
        common_lock_status = "unavailable_development_size_shard"
    profile_pass = bool(
        all(case["scientific_pass"] for case in cases)
        if profile == "smoke"
        else (
            case_id is None
            and common_locked_sample_count is not None
            and common_confirmation_evidence["complete"] is True
            and common_confirmation_evidence["all_cases_pass"] is True
        )
    )
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "profile": profile,
        "execution_mode": (
            ("development_size_shard" if development_shard_mode else "development_ladder")
            if profile == "development" and case_id is not None
            else profile
        ),
        "selected_case_id": case_id,
        "executed_development_sample_count": development_sample_count,
        "per_case_atomic_output": bool(profile == "development" and case_id is not None),
        "source_git_revision": resolved_revision,
        "driver_sha256": observed_driver_sha256,
        "a1_definitions_sha256": observed_a1_definitions,
        "protocol_sha256": _protocol_sha256(),
        "frozen_development_protocol_sha256": DEVELOPMENT_PROTOCOL_SHA256,
        "sample_counts": list(counts),
        "repeat_seeds": list(seeds),
        "training_protocol": {
            "component_count": COMPONENT_COUNT,
            "deterministic_initialization_count": INITIALIZATION_COUNT,
            "minimum_valid_initializations": (MINIMUM_VALID_INITIALIZATIONS),
            "covariance_regularization": COVARIANCE_REGULARIZATION,
            "maximum_em_iterations": MAXIMUM_EM_ITERATIONS,
            "convergence_nat_per_draw": CONVERGENCE_NAT_PER_DRAW,
            "convergence_required_consecutive_iterations": CONVERGENCE_STREAK,
            "selection": "minimum_validation_nll",
            "simulator_test_generalization_gate": {
                "nat_per_residual_dimension": (GENERALIZATION_NAT_PER_DIMENSION),
                "pooled_mcse_multiplier": (GENERALIZATION_MCSE_MULTIPLIER),
                "rule": (
                    "absolute validation/test NLL gap <= max(0.02 * residual_dimension, 5 * pooled MCSE)"
                ),
            },
            "domains": [TRAINING_DOMAIN, VALIDATION_DOMAIN, TEST_DOMAIN],
            "validation_sample_count": (
                SMOKE_SAMPLE_COUNTS[0] if profile == "smoke" else VALIDATION_SAMPLE_COUNT
            ),
            "test_sample_count": (SMOKE_SAMPLE_COUNTS[0] if profile == "smoke" else TEST_SAMPLE_COUNT),
            "protected_holdout_sample_count": PROTECTED_HOLDOUT_SAMPLE_COUNT,
            "protected_holdout_accessible": False,
        },
        "thresholds": c1.THRESHOLDS,
        "matrix_catalogue": matrix_catalogue(),
        "common_training_lock": {
            "status": common_lock_status,
            "requires_complete_six_case_matrix": True,
            "minimum_passing_suffix_length": 2,
            "development_pass_pattern": common_development_pass_pattern,
            "locked_sample_count": common_locked_sample_count,
            "isolated_largest_size_pass_is_hard_stop": True,
        },
        "common_confirmation_evidence": common_confirmation_evidence,
        "cases": cases,
        "smoke_pass": profile_pass if profile == "smoke" else None,
        "development_pass": (profile_pass if profile == "development" else None),
        "eligible_for_protected_holdout": bool(profile == "development" and profile_pass),
        "protected_holdout_pass": None,
        "scientific_pass": False,
        "scientific_pass_available": False,
        "scientific_pass_reason": ("the separately sealed protected holdout is not executable here"),
        "structural_inference_licensed": False,
        "held_out_information_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _power_of_two_csv(value: str) -> tuple[int, ...]:
    """Parse a strict increasing sequence of power-of-two counts."""
    parsed = c1._positive_csv(
        value,
        name="sample-counts",
        upper_bound=1_048_576,
    )
    if any(item & (item - 1) for item in parsed):
        raise argparse.ArgumentTypeError("sample-counts must contain powers of two")
    return parsed


def _write_atomic_json(path: Path, payload: object) -> None:
    """Publish one canonical JSON output without partial replacement."""
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
            stream.write(c1._canonical_json(payload))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "development"), default="smoke")
    parser.add_argument("--case-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--development-sample-count",
        type=int,
        choices=DEVELOPMENT_SAMPLE_COUNTS,
    )
    parser.add_argument("--confirmation-lock", type=Path)
    parser.add_argument("--expected-confirmation-lock-sha256")
    parser.add_argument("--confirmation-seed", type=int, choices=CONFIRMATION_SEEDS)
    parser.add_argument("--sample-counts", type=_power_of_two_csv)
    parser.add_argument(
        "--repeat-seeds",
        type=lambda value: c1._positive_csv(
            value,
            name="repeat-seeds",
            upper_bound=2**63 - 1,
        ),
    )
    parser.add_argument("--list-matrix", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested profile and publish its immutable JSON result."""
    args = _parser().parse_args(argv)
    if args.list_matrix:
        if any(
            value is not None
            for value in (
                args.output,
                args.case_id,
                args.source_revision,
                args.development_sample_count,
                args.confirmation_lock,
                args.expected_confirmation_lock_sha256,
                args.confirmation_seed,
                args.sample_counts,
                args.repeat_seeds,
            )
        ):
            raise SystemExit("--list-matrix cannot be combined with run options")
        print(c1._canonical_json(matrix_catalogue()))
        return 0
    if args.output is None:
        raise SystemExit("--output is required unless --list-matrix is used")
    if args.profile == "development" and (args.sample_counts is not None or args.repeat_seeds is not None):
        raise SystemExit("development sample counts and seeds are source-pinned")
    if (args.confirmation_lock is None) != (args.expected_confirmation_lock_sha256 is None):
        raise SystemExit(
            "--confirmation-lock and --expected-confirmation-lock-sha256 must be supplied together"
        )
    if args.confirmation_seed is not None and args.confirmation_lock is None:
        raise SystemExit("--confirmation-seed requires --confirmation-lock")
    report = run_screen(
        profile=args.profile,
        sample_counts=args.sample_counts,
        repeat_seeds=args.repeat_seeds,
        case_id=args.case_id,
        source_revision=args.source_revision,
        development_sample_count=args.development_sample_count,
        confirmation_lock=args.confirmation_lock,
        expected_confirmation_lock_sha256=(args.expected_confirmation_lock_sha256),
        confirmation_seed=args.confirmation_seed,
    )
    _write_atomic_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
