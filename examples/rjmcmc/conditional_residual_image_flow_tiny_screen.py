#!/usr/bin/env python3
"""Train and score the BP1 direct noisy-residual neural likelihood.

The trainer constructs independent, domain-keyed simulations of the complete
projected observation residual.  It whitens each draw by the exact conditional
covariance, then fits the source-pinned conditional triangular spline flow.
The selected flow is exported through the authenticated normalized likelihood
and simulator in ``aggregation_error_conditional_flow``.

This driver cannot address the protected holdout domain.  It handles exactly
one root case and one training size per invocation so Slurm array tasks remain
independent and preserve failures.
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

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
from numpy import __version__ as numpy_version  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy import __version__ as scipy_version  # noqa: E402
from scipy import special  # noqa: E402
from scipy.stats import qmc  # noqa: E402

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1  # noqa: E402
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_flow import (  # noqa: E402
    EQUINOX_VERSION,
    FLOWJAX_VERSION,
    FLOW_ARCHITECTURE,
    FLOW_INVERT,
    FLOW_KNOTS,
    FLOW_LAYERS,
    FLOW_TANH_MAX,
    JAXLIB_VERSION,
    JAX_VERSION,
    OPTAX_VERSION,
    PARAMAX_VERSION,
    ConditionalResidualImageFlow,
    conditional_residual_unit_covariances,
    make_conditional_residual_flow,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mdn import (  # noqa: E402
    RESIDUAL_IMAGE_BASIS_RULE,
    RESIDUAL_IMAGE_CONTEXT_SCHEMA,
    ResidualImageContext,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (  # noqa: E402
    ConditionalAllocationMixture,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (  # noqa: E402
    AdditiveDirichletAggregation,
)

from flowjax.train import fit_to_data  # noqa: E402

FloatArray = NDArray[np.float64]
Profile = Literal["smoke", "development"]

SCHEMA = "rjmcmc-conditional-residual-image-flow-tiny-screen-v1"
PROTOCOL = "conditional-residual-image-direct-noisy-triangular-spline-flow-v1"
CONSTRUCTION_METHOD = "scrambled_sobol_balanced_dirichlet"
DEVELOPMENT_PROTOCOL_SHA256 = (
    "b4c548bcb9b83dcd2837a1a5ae88f716b3cf61a32c5acedd77ef75e1f5efcaf2"
)

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
SMOKE_VALIDATION_SAMPLE_COUNT = 4_096
SMOKE_TEST_SAMPLE_COUNT = 4_096
DEVELOPMENT_SELECTION_SEED = 731
CONFIRMATION_SEEDS = (1_877, 4_099, 8_317)
SMOKE_BASE_SEED = 731

INITIALIZATION_COUNT = 2
LEARNING_RATE = 5.0e-4
BATCH_SIZE = 1_024
MAXIMUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
INTERNAL_VALIDATION_PROPORTION = 0.1
SMOKE_INITIALIZATION_COUNT = 1
SMOKE_MAXIMUM_EPOCHS = 5
SMOKE_EARLY_STOPPING_PATIENCE = 2

LOG_TOTAL_MARGIN = 0.5
GENERALIZATION_NAT_PER_DIMENSION = 0.02
GENERALIZATION_MCSE_MULTIPLIER = 5.0

TRAINING_DOMAIN = "training"
VALIDATION_DOMAIN = "model-selection-validation"
TEST_DOMAIN = "development-reporting-test"
PUBLIC_DOMAINS = (TRAINING_DOMAIN, VALIDATION_DOMAIN, TEST_DOMAIN)
PROTECTED_HOLDOUT_CATALOGUE_SHA256 = (
    "83bec3945ebc90d5e25d0888b440fe56f761f9059cf01537fbb2227b81510b66"
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
    """Return the digest of canonical JSON."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _array_sha256(values: FloatArray) -> str:
    """Return a shape-aware little-endian float64 array digest."""
    canonical = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256(
        _canonical_json(
            {
                "dtype": "<f8",
                "shape": list(canonical.shape),
            }
        ).encode("ascii")
    )
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _protocol_payload() -> dict[str, Any]:
    """Return the complete frozen development protocol payload."""
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
        "residual_image_context_schema": RESIDUAL_IMAGE_CONTEXT_SCHEMA,
        "residual_image_basis_rule": RESIDUAL_IMAGE_BASIS_RULE,
        "matrix": DEVELOPMENT_MATRIX,
        "training_sample_counts": DEVELOPMENT_SAMPLE_COUNTS,
        "validation_sample_count": VALIDATION_SAMPLE_COUNT,
        "test_sample_count": TEST_SAMPLE_COUNT,
        "development_selection_seed": DEVELOPMENT_SELECTION_SEED,
        "confirmation_seeds": CONFIRMATION_SEEDS,
        "domains": PUBLIC_DOMAINS,
        "construction_method": CONSTRUCTION_METHOD,
        "log_total_margin": LOG_TOTAL_MARGIN,
        "flow": {
            "architecture": FLOW_ARCHITECTURE,
            "layers": FLOW_LAYERS,
            "knots": FLOW_KNOTS,
            "tanh_max": FLOW_TANH_MAX,
            "invert": FLOW_INVERT,
        },
        "fit": {
            "initialization_count": INITIALIZATION_COUNT,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "maximum_epochs": MAXIMUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "internal_validation_proportion": INTERNAL_VALIDATION_PROPORTION,
            "selection": "minimum independent validation NLL then initialization index",
        },
        "generalization": {
            "nat_per_dimension": GENERALIZATION_NAT_PER_DIMENSION,
            "mcse_multiplier": GENERALIZATION_MCSE_MULTIPLIER,
        },
        "thresholds": c1.THRESHOLDS,
        "gradient_step": c1.GRADIENT_STEP,
        "protected_holdout_catalogue_sha256": PROTECTED_HOLDOUT_CATALOGUE_SHA256,
        "training_size_lock": (
            "smallest common six-case size starting an all-larger passing "
            "suffix of length at least two"
        ),
        "runtime": {
            "numpy": numpy_version,
            "scipy": scipy_version,
            "jax": JAX_VERSION,
            "jaxlib": JAXLIB_VERSION,
            "flowjax": FLOWJAX_VERSION,
            "equinox": EQUINOX_VERSION,
            "optax": OPTAX_VERSION,
            "paramax": PARAMAX_VERSION,
        },
    }


def _protocol_sha256() -> str:
    """Return the complete development protocol identity."""
    return _sha256_json(_protocol_payload())


def _validate_development_protocol() -> None:
    """Fail closed if the source-pinned protocol has drifted."""
    if not DEVELOPMENT_PROTOCOL_SHA256:
        raise RuntimeError("the development protocol identity has not been frozen")
    if _protocol_sha256() != DEVELOPMENT_PROTOCOL_SHA256:
        raise RuntimeError("the frozen direct NLE development protocol identity changed")
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("JAX float64 mode is required")


def _domain_seed(
    base_seed: int,
    *,
    case_id: str,
    domain: str,
    stream: str,
) -> int:
    """Derive one stable unsigned 32-bit public-domain seed."""
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or not 0 <= base_seed < 2**64:
        raise ValueError("base_seed must be an unsigned 64-bit integer")
    if domain not in PUBLIC_DOMAINS:
        raise ValueError("protected or unknown sample domains cannot be opened")
    digest = hashlib.sha256(PROTOCOL.encode("ascii"))
    digest.update(base_seed.to_bytes(8, byteorder="little", signed=False))
    digest.update(case_id.encode("ascii"))
    digest.update(domain.encode("ascii"))
    digest.update(stream.encode("ascii"))
    return int.from_bytes(digest.digest()[:4], byteorder="little", signed=False)


@dataclass(frozen=True)
class SimulatedDomain:
    """One authenticated conditional-flow training or scoring domain."""

    targets: FloatArray
    conditions: FloatArray
    evidence: dict[str, Any]


def _simulated_domain(
    aggregation: AdditiveDirichletAggregation,
    labels: c1.IntArray,
    context: ResidualImageContext,
    unit_covariances: FloatArray,
    *,
    case_id: str,
    domain: str,
    sample_count: int,
    base_seed: int,
    log_total_minimum: float,
    log_total_maximum: float,
) -> SimulatedDomain:
    """Construct one independent direct noisy-residual simulation domain."""
    if sample_count < 1 or sample_count & (sample_count - 1):
        raise ValueError("domain sample_count must be a power of two")
    if not log_total_minimum < log_total_maximum:
        raise ValueError("log-total bounds must be increasing")
    residual_seed = _domain_seed(
        base_seed,
        case_id=case_id,
        domain=domain,
        stream="conditional-dirichlet-residual",
    )
    observation_seed = _domain_seed(
        base_seed,
        case_id=case_id,
        domain=domain,
        stream="log-total-and-projected-noise",
    )
    projected = AdditiveDirichletAggregation(
        aggregation.cell_alphas,
        aggregation.design,
        aggregation.noise_sd,
        context.residual_basis,
    )
    residual_bank = ConditionalAllocationMixture.from_aggregation(
        projected,
        labels,
        sample_count=sample_count,
        source_seed=residual_seed,
        source_provenance=(
            f"{PROTOCOL}:{case_id}:{domain}:conditional-dirichlet-residual"
        ),
        cell_ids=context.cell_ids,
        construction_method=CONSTRUCTION_METHOD,
    )
    if residual_bank.region_count != 1:
        raise RuntimeError("the BP1 direct NLE trainer currently accepts root cases only")
    unit_residual = np.asarray(
        residual_bank.projected_unit_mass_residual_factors[:, :, 0],
        dtype=np.float64,
    )
    if unit_residual.shape != (sample_count, context.residual_rank):
        raise RuntimeError("conditional residual bank has an unexpected shape")

    sobol = qmc.Sobol(
        d=1 + context.residual_rank,
        scramble=True,
        bits=52,
        seed=observation_seed,  # pyright: ignore[reportCallIssue]
        optimization=None,
    )
    points = np.asarray(
        sobol.random_base2(int(math.log2(sample_count))),
        dtype=np.float64,
    )
    log_totals = (
        log_total_minimum
        + (log_total_maximum - log_total_minimum) * points[:, 0]
    )
    totals = np.exp(log_totals)
    open_lower = np.nextafter(0.0, 1.0)
    open_upper = np.nextafter(1.0, 0.0)
    gaussian = special.ndtri(
        np.clip(
            points[:, 1:],
            open_lower,
            open_upper,
        )
    )
    conditional_covariance = (
        np.eye(context.residual_rank, dtype=np.float64)[np.newaxis, :, :]
        + totals[:, np.newaxis, np.newaxis] ** 2 * unit_covariances[0]
    )
    cholesky = np.linalg.cholesky(conditional_covariance)
    projected_noisy_residual = totals[:, np.newaxis] * unit_residual + gaussian
    targets = np.linalg.solve(
        cholesky,
        projected_noisy_residual[:, :, np.newaxis],
    )[:, :, 0]
    conditioner_center = 0.5 * (log_total_minimum + log_total_maximum)
    conditioner_scale = 0.5 * (log_total_maximum - log_total_minimum)
    conditions = (
        (log_totals - conditioner_center) / conditioner_scale
    )[:, np.newaxis]
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(conditions)):
        raise RuntimeError("direct noisy-residual simulations are non-finite")
    return SimulatedDomain(
        cast(FloatArray, targets),
        cast(FloatArray, conditions),
        {
            "domain": domain,
            "sample_count": sample_count,
            "residual_seed": residual_seed,
            "observation_seed": observation_seed,
            "residual_bank_sha256": residual_bank.sha256,
            "unit_residual_sha256": _array_sha256(unit_residual),
            "sobol_points_sha256": _array_sha256(points),
            "targets_sha256": _array_sha256(targets),
            "conditions_sha256": _array_sha256(conditions),
        },
    )


def _log_probabilities(
    flow: Any,
    domain: SimulatedDomain,
) -> FloatArray:
    """Evaluate one fitted flow over a complete independent domain."""
    result = np.asarray(
        flow.log_prob(
            jnp.asarray(domain.targets, dtype=jnp.float64),
            jnp.asarray(domain.conditions, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )
    if result.shape != (domain.targets.shape[0],) or not np.all(np.isfinite(result)):
        raise RuntimeError("fitted flow produced invalid domain log probabilities")
    return cast(FloatArray, result)


def _nll_summary(log_probabilities: FloatArray) -> dict[str, float]:
    """Return mean NLL and its independent-draw diagnostic MCSE."""
    negative = -np.asarray(log_probabilities, dtype=np.float64)
    return {
        "nll_nat_per_draw": float(np.mean(negative)),
        "nll_mcse_nat_per_draw": float(
            np.std(negative, ddof=1) / math.sqrt(negative.size)
        ),
    }


def _fit_attempt(
    context: ResidualImageContext,
    unit_covariances: FloatArray,
    domains: dict[str, SimulatedDomain],
    *,
    case_id: str,
    base_seed: int,
    initialization: int,
    profile: Profile,
    conditioner_center: FloatArray,
    conditioner_scale: FloatArray,
    source_git_revision: str,
) -> tuple[ConditionalResidualImageFlow, dict[str, Any]]:
    """Fit and independently score one deterministic initialization."""
    initialization_seed = _domain_seed(
        base_seed,
        case_id=case_id,
        domain=TRAINING_DOMAIN,
        stream=f"flow-initialization-{initialization}",
    )
    optimizer_seed = _domain_seed(
        base_seed,
        case_id=case_id,
        domain=TRAINING_DOMAIN,
        stream=f"optimizer-{initialization}",
    )
    flow = make_conditional_residual_flow(
        context.residual_rank,
        context.region_count,
        source_seed=initialization_seed,
    )
    started = time.perf_counter()
    fitted, losses = fit_to_data(
        jr.key(optimizer_seed),
        flow,
        data=(
            jnp.asarray(
                domains[TRAINING_DOMAIN].targets,
                dtype=jnp.float64,
            ),
            jnp.asarray(
                domains[TRAINING_DOMAIN].conditions,
                dtype=jnp.float64,
            ),
        ),
        learning_rate=LEARNING_RATE,
        max_epochs=(
            SMOKE_MAXIMUM_EPOCHS
            if profile == "smoke"
            else MAXIMUM_EPOCHS
        ),
        max_patience=(
            SMOKE_EARLY_STOPPING_PATIENCE
            if profile == "smoke"
            else EARLY_STOPPING_PATIENCE
        ),
        batch_size=BATCH_SIZE,
        val_prop=INTERNAL_VALIDATION_PROPORTION,
        return_best=True,
        show_progress=False,
    )
    fit_seconds = time.perf_counter() - started
    if set(losses) != {"train", "val"} or not losses["train"] or not losses["val"]:
        raise RuntimeError("FlowJAX returned malformed optimizer losses")
    train_losses = np.asarray(losses["train"], dtype=np.float64)
    internal_validation_losses = np.asarray(losses["val"], dtype=np.float64)
    if not np.all(np.isfinite(train_losses)) or not np.all(
        np.isfinite(internal_validation_losses)
    ):
        raise RuntimeError("FlowJAX optimizer losses are non-finite")
    validation_log_probabilities = _log_probabilities(
        fitted,
        domains[VALIDATION_DOMAIN],
    )
    test_log_probabilities = _log_probabilities(
        fitted,
        domains[TEST_DOMAIN],
    )
    validation = _nll_summary(validation_log_probabilities)
    test = _nll_summary(test_log_probabilities)
    gap = abs(test["nll_nat_per_draw"] - validation["nll_nat_per_draw"])
    pooled_mcse = math.hypot(
        test["nll_mcse_nat_per_draw"],
        validation["nll_mcse_nat_per_draw"],
    )
    gap_threshold = max(
        GENERALIZATION_NAT_PER_DIMENSION * context.residual_rank,
        GENERALIZATION_MCSE_MULTIPLIER * pooled_mcse,
    )
    artifact = ConditionalResidualImageFlow(
        context,
        unit_covariances,
        conditioner_center,
        conditioner_scale,
        fitted,
        initialization_seed=initialization_seed,
        source_provenance=(
            f"{PROTOCOL}:{case_id}:base={base_seed}:initialization={initialization}:"
            f"git={source_git_revision}"
        ),
    )
    return artifact, {
        "initialization": initialization,
        "initialization_seed": initialization_seed,
        "optimizer_seed": optimizer_seed,
        "fit_seconds": fit_seconds,
        "epochs": int(train_losses.size),
        "training_loss_history": train_losses.tolist(),
        "internal_validation_loss_history": internal_validation_losses.tolist(),
        "validation": validation,
        "test": test,
        "absolute_validation_test_nll_gap_nat_per_draw": gap,
        "pooled_nll_mcse_nat_per_draw": pooled_mcse,
        "generalization_threshold_nat_per_draw": gap_threshold,
        "generalization_pass": bool(gap <= gap_threshold),
        "artifact_sha256": artifact.artifact_sha256,
    }


def _evaluate_artifact(
    *,
    artifact: ConditionalResidualImageFlow,
    observation: FloatArray,
    masses: FloatArray,
    log_prior: FloatArray,
    exact_log_likelihood: FloatArray,
    exact_summary: dict[str, Any],
    gradient_states: Sequence[dict[str, Any]],
    validation_state_mask: NDArray[np.bool_],
) -> dict[str, Any]:
    """Apply the unchanged C1 likelihood, gradient, and posterior gates."""
    learned_log_likelihood = artifact.log_likelihood_batch(
        observation,
        masses,
    )
    error = np.abs(learned_log_likelihood - exact_log_likelihood)
    learned_summary = c1._posterior_summary(
        masses,
        log_prior,
        learned_log_likelihood,
    )
    summary_errors, by_coordinate = c1._summary_errors(
        exact_summary,
        learned_summary,
    )
    validation_prior_log_weights = log_prior[validation_state_mask]
    validation_prior_weights = np.exp(
        validation_prior_log_weights
        - c1._stable_logsumexp(validation_prior_log_weights)
    )
    validation_posterior_log_weights = (
        validation_prior_log_weights
        + exact_log_likelihood[validation_state_mask]
    )
    validation_posterior_weights = np.exp(
        validation_posterior_log_weights
        - c1._stable_logsumexp(validation_posterior_log_weights)
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

        learned_gradient = c1._centered_gradient(
            learned_function,
            coordinate,
        )
        exact_gradient = np.asarray(
            state["exact_coordinate_gradient"],
            dtype=np.float64,
        )
        scaled_error = float(
            np.max(
                np.abs(learned_gradient - exact_gradient)
                / (1.0 + np.abs(exact_gradient))
            )
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
        "median_absolute_conditional_log_likelihood_error_nat": (
            c1._weighted_quantile(
                validation_error,
                validation_prior_weights,
                0.5,
            )
        ),
        "p99_absolute_conditional_log_likelihood_error_nat": (
            c1._weighted_quantile(
                validation_error,
                validation_posterior_weights,
                0.99,
            )
        ),
        "scaled_coordinate_gradient_error": max(
            audit["scaled_error"] for audit in gradient_audits
        ),
        "absolute_log_evidence_error_nat": abs(
            learned_summary["log_evidence"]
            - exact_summary["log_evidence"]
        ),
        **summary_errors,
    }
    checks = {
        name: bool(metrics[name] <= threshold)
        for name, threshold in c1.THRESHOLDS.items()
        if name in metrics
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "scientific_pass": bool(all(checks.values())),
        "posterior_summary": learned_summary,
        "posterior_errors_by_coordinate": by_coordinate,
        "gradient_audits": gradient_audits,
        "diagnostics": {
            "full_grid_median_absolute_log_likelihood_error_nat": float(
                np.median(error)
            ),
            "full_grid_maximum_absolute_log_likelihood_error_nat": float(
                np.max(error)
            ),
        },
    }


def run_case(
    *,
    regime_name: str,
    family: c1.Family,
    training_sample_count: int,
    base_seed: int,
    profile: Profile,
    source_git_revision: str,
    driver_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Fit and score one source-pinned root case and training size."""
    case_key = (regime_name, family, "root")
    allowed = SMOKE_MATRIX if profile == "smoke" else DEVELOPMENT_MATRIX
    if case_key not in allowed:
        raise ValueError(f"case {case_key!r} is not available in {profile}")
    allowed_counts = (
        SMOKE_SAMPLE_COUNTS
        if profile == "smoke"
        else DEVELOPMENT_SAMPLE_COUNTS
    )
    if training_sample_count not in allowed_counts:
        raise ValueError("training_sample_count is not source-pinned")
    if profile == "development":
        _validate_development_protocol()
    if (
        not isinstance(source_git_revision, str)
        or len(source_git_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_git_revision)
    ):
        raise ValueError("source_git_revision must be a full lower-case Git SHA")
    if (
        not isinstance(driver_sha256, str)
        or len(driver_sha256) != 64
        or any(character not in "0123456789abcdef" for character in driver_sha256)
    ):
        raise ValueError("driver_sha256 must be a lower-case SHA-256 digest")
    regime = c1._regime(regime_name)
    shapes, rate, design, observation, noise = c1._case_arrays(
        regime,
        family,
    )
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
    exact_summary = c1._posterior_summary(
        masses,
        log_prior,
        exact_log_likelihood,
    )
    prior_mean_coordinate = c1._anchor_coordinate(shapes, rate, labels)

    def exact_function(value: FloatArray) -> float:
        return float(
            c1._exact_log_likelihood(
                masses=c1.coordinate_to_masses(value)[np.newaxis, :],
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
    case_id = f"{regime_name}__{family}__root"
    context = ResidualImageContext.from_aggregation(
        aggregation,
        labels,
        np.arange(shapes.size, dtype=np.int64),
        source_provenance=f"{PROTOCOL}:{case_id}:residual-image-context",
    )
    if context.residual_rank == 0:
        raise RuntimeError("the NLE development matrix excludes zero-rank controls")
    unit_covariances = conditional_residual_unit_covariances(
        aggregation,
        context,
    )
    totals = np.sum(masses, axis=1)
    log_total_minimum = float(np.min(np.log(totals)) - LOG_TOTAL_MARGIN)
    log_total_maximum = float(np.max(np.log(totals)) + LOG_TOTAL_MARGIN)
    conditioner_center = np.asarray(
        [0.5 * (log_total_minimum + log_total_maximum)],
        dtype=np.float64,
    )
    conditioner_scale = np.asarray(
        [0.5 * (log_total_maximum - log_total_minimum)],
        dtype=np.float64,
    )
    validation_sample_count = (
        SMOKE_VALIDATION_SAMPLE_COUNT
        if profile == "smoke"
        else VALIDATION_SAMPLE_COUNT
    )
    test_sample_count = (
        SMOKE_TEST_SAMPLE_COUNT
        if profile == "smoke"
        else TEST_SAMPLE_COUNT
    )
    domain_counts = {
        TRAINING_DOMAIN: training_sample_count,
        VALIDATION_DOMAIN: validation_sample_count,
        TEST_DOMAIN: test_sample_count,
    }
    domains = {
        domain: _simulated_domain(
            aggregation,
            labels,
            context,
            unit_covariances,
            case_id=case_id,
            domain=domain,
            sample_count=sample_count,
            base_seed=base_seed,
            log_total_minimum=log_total_minimum,
            log_total_maximum=log_total_maximum,
        )
        for domain, sample_count in domain_counts.items()
    }
    attempt_count = (
        SMOKE_INITIALIZATION_COUNT
        if profile == "smoke"
        else INITIALIZATION_COUNT
    )
    attempts: list[dict[str, Any]] = []
    artifacts: list[ConditionalResidualImageFlow] = []
    for initialization in range(attempt_count):
        artifact, attempt = _fit_attempt(
            context,
            unit_covariances,
            domains,
            case_id=case_id,
            base_seed=base_seed,
            initialization=initialization,
            profile=profile,
            conditioner_center=conditioner_center,
            conditioner_scale=conditioner_scale,
            source_git_revision=source_git_revision,
        )
        artifacts.append(artifact)
        attempts.append(attempt)
    selected_index = min(
        range(attempt_count),
        key=lambda index: (
            attempts[index]["validation"]["nll_nat_per_draw"],
            index,
        ),
    )
    selected = artifacts[selected_index]
    selected_attempt = attempts[selected_index]
    selected_bytes = selected.to_bytes()
    replay = ConditionalResidualImageFlow.from_bytes(
        selected_bytes,
        expected_sha256=selected.artifact_sha256,
    )
    artifact_replay_pass = bool(
        replay.to_bytes() == selected_bytes
        and replay.artifact_sha256 == selected.artifact_sha256
    )
    evaluation = _evaluate_artifact(
        artifact=selected,
        observation=observation,
        masses=masses,
        log_prior=log_prior,
        exact_log_likelihood=exact_log_likelihood,
        exact_summary=exact_summary,
        gradient_states=gradient_states,
        validation_state_mask=validation_state_mask,
    )
    fit_pass = bool(len(attempts) == attempt_count)
    development_task_pass = bool(
        fit_pass
        and selected_attempt["generalization_pass"]
        and artifact_replay_pass
        and evaluation["scientific_pass"]
    )
    smoke_task_pass = bool(
        fit_pass
        and selected_attempt["generalization_pass"]
        and artifact_replay_pass
    )
    result = {
        "schema": SCHEMA,
        "protocol": {
            "name": PROTOCOL,
            "sha256": _protocol_sha256(),
            "payload": _protocol_payload(),
        },
        "profile": profile,
        "source": {
            "git_revision": source_git_revision,
            "driver_sha256": driver_sha256,
        },
        "case_id": case_id,
        "training_sample_count": training_sample_count,
        "base_seed": base_seed,
        "context_sha256": context.artifact_sha256,
        "unit_covariances_sha256": _array_sha256(unit_covariances),
        "log_total_bounds": [
            log_total_minimum,
            log_total_maximum,
        ],
        "conditioner_center": conditioner_center.tolist(),
        "conditioner_scale": conditioner_scale.tolist(),
        "domains": {
            name: domain.evidence
            for name, domain in domains.items()
        },
        "attempts": attempts,
        "selected_initialization": selected_index,
        "selected_artifact_sha256": selected.artifact_sha256,
        "artifact_replay_pass": artifact_replay_pass,
        "fit_development_pass": fit_pass,
        "selected_generalization_pass": selected_attempt[
            "generalization_pass"
        ],
        "evaluation": evaluation,
        "task_pass": (
            smoke_task_pass
            if profile == "smoke"
            else development_task_pass
        ),
    }
    return result, selected_bytes


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write bytes atomically within an existing output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_result(
    output_directory: Path,
    result: dict[str, Any],
    artifact_bytes: bytes,
) -> dict[str, str]:
    """Publish artifact, report envelope, and completion marker in order."""
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{result['case_id']}__S{result['training_sample_count']}"
        f"__base{result['base_seed']}"
    )
    artifact_path = output_directory / f"{stem}.flow"
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if artifact_sha256 != result["selected_artifact_sha256"]:
        raise RuntimeError("selected artifact bytes do not match the result identity")
    _atomic_write(artifact_path, artifact_bytes)
    envelope_payload = {
        "result": result,
        "artifact": {
            "path": artifact_path.name,
            "sha256": artifact_sha256,
        },
    }
    envelope = {
        "payload": envelope_payload,
        "sha256": _sha256_json(envelope_payload),
    }
    report_path = output_directory / f"{stem}.json"
    report_bytes = (_canonical_json(envelope) + "\n").encode("utf-8")
    _atomic_write(report_path, report_bytes)
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    marker_payload = {
        "schema": "rjmcmc-conditional-residual-image-flow-task-complete-v1",
        "case_id": result["case_id"],
        "training_sample_count": result["training_sample_count"],
        "base_seed": result["base_seed"],
        "task_pass": result["task_pass"],
        "artifact_sha256": artifact_sha256,
        "report_sha256": report_sha256,
    }
    marker_path = output_directory / f"{stem}.complete.json"
    _atomic_write(
        marker_path,
        (_canonical_json(marker_payload) + "\n").encode("utf-8"),
    )
    return {
        "artifact": str(artifact_path),
        "report": str(report_path),
        "completion_marker": str(marker_path),
    }


def main() -> None:
    """Run one independent smoke or development task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "development"),
        required=True,
    )
    parser.add_argument(
        "--regime",
        choices=("near_gaussian", "skewed", "boundary_heavy"),
        required=True,
    )
    parser.add_argument(
        "--family",
        choices=("two_cell", "four_cell"),
        required=True,
    )
    parser.add_argument(
        "--training-sample-count",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEVELOPMENT_SELECTION_SEED,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-git-revision",
        required=True,
    )
    parser.add_argument(
        "--driver-sha256",
        required=True,
    )
    parser.add_argument(
        "--print-protocol-sha256",
        action="store_true",
    )
    args = parser.parse_args()
    if args.print_protocol_sha256:
        print(_protocol_sha256())
        return
    result, artifact = run_case(
        regime_name=args.regime,
        family=cast(c1.Family, args.family),
        training_sample_count=args.training_sample_count,
        base_seed=args.base_seed,
        profile=cast(Profile, args.profile),
        source_git_revision=args.source_git_revision,
        driver_sha256=args.driver_sha256,
    )
    paths = _write_result(args.output_directory, result, artifact)
    print(
        _canonical_json(
            {
                "task_pass": result["task_pass"],
                "paths": paths,
            }
        )
    )


if __name__ == "__main__":
    main()
