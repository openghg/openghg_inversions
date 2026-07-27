#!/usr/bin/env python
"""Train and score the predeclared BP1 conditional sbi-NSF likelihood.

The driver generates ancestral root-model pairs, applies exact conditional
moment whitening, fits the pinned ``sbi`` autoregressive neural spline, and
scores the selected authenticated artifact against the unchanged C1 oracle.
Protected sample domains are unreachable from this program.
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
from typing import Any, Literal, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy import special
from scipy.stats import qmc
import torch

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import conditional_residual_image_flow_tiny_screen as flow_screen
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mdn import (
    ResidualImageContext,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (
    ConditionalAllocationMixture,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_sbi_nsf import (
    ConditionalResidualImageSbiNsf,
    NFLOWS_VERSION,
    NSF_HIDDEN_FEATURES,
    NSF_NUM_BINS,
    NSF_NUM_TRANSFORMS,
    SBI_VERSION,
    TORCH_VERSION,
    conditional_residual_unit_covariances,
    make_conditional_residual_nsf,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)

FloatArray: TypeAlias = NDArray[np.float64]
Profile = Literal["smoke", "development"]

SCHEMA = "rjmcmc-conditional-residual-image-sbi-nsf-tiny-screen-v1"
PROTOCOL = "conditional-residual-image-prior-predictive-sbi-nsf-v1"
CONSTRUCTION_METHOD = "scrambled_sobol_balanced_dirichlet"
PROTECTED_HOLDOUT_CATALOGUE_ID = "conditional-residual-image-sbi-nsf-protected-v1"
DEVELOPMENT_PROTOCOL_SHA256 = (
    "ef8441560ac107e377cebe7785259bff0ff288d5e84ea013908f5aa52c752f27"
)

DEVELOPMENT_MATRIX = tuple(
    (regime, family, "root")
    for regime in ("near_gaussian", "skewed", "boundary_heavy")
    for family in ("two_cell", "four_cell")
)
SMOKE_MATRIX = (("near_gaussian", "two_cell", "root"),)
DEVELOPMENT_SAMPLE_COUNTS = (4_096, 16_384, 65_536, 262_144)
SMOKE_SAMPLE_COUNTS = (4_096,)
OPTIMIZER_VALIDATION_SAMPLE_COUNT = 65_536
MODEL_SELECTION_VALIDATION_SAMPLE_COUNT = 65_536
TEST_SAMPLE_COUNT = 131_072
SMOKE_DOMAIN_SAMPLE_COUNT = 4_096
DEVELOPMENT_SELECTION_SEED = 731
CONFIRMATION_SEEDS = (1_877, 4_099, 8_317)

INITIALIZATION_COUNT = 2
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-6
BATCH_SIZE = 2_048
MAXIMUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
SMOKE_INITIALIZATION_COUNT = 1
SMOKE_MAXIMUM_EPOCHS = 4
SMOKE_EARLY_STOPPING_PATIENCE = 2
GENERALIZATION_NAT_PER_DIMENSION = 0.02
GENERALIZATION_MCSE_MULTIPLIER = 5.0

TRAINING_DOMAIN = "training"
OPTIMIZER_VALIDATION_DOMAIN = "optimizer-validation"
MODEL_SELECTION_VALIDATION_DOMAIN = "model-selection-validation"
TEST_DOMAIN = "development-test"
PUBLIC_DOMAINS = (
    TRAINING_DOMAIN,
    OPTIMIZER_VALIDATION_DOMAIN,
    MODEL_SELECTION_VALIDATION_DOMAIN,
    TEST_DOMAIN,
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
    """Return the SHA-256 of canonical JSON."""
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _array_sha256(values: FloatArray) -> str:
    """Return a shape-aware canonical float64 array digest."""
    array = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256(
        _canonical_json(
            {
                "dtype": "<f8",
                "shape": list(array.shape),
            }
        ).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _protocol_payload() -> dict[str, Any]:
    """Return the complete source-pinned development declaration."""
    return {
        "protocol": PROTOCOL,
        "schema": SCHEMA,
        "architecture": {
            "backend": "sbi",
            "model": "nsf",
            "hidden_features": NSF_HIDDEN_FEATURES,
            "num_transforms": NSF_NUM_TRANSFORMS,
            "num_bins": NSF_NUM_BINS,
            "dtype": "float64",
            "device": "cpu",
            "deterministic_algorithms": True,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
        },
        "runtime": {
            "torch": TORCH_VERSION,
            "sbi": SBI_VERSION,
            "nflows": NFLOWS_VERSION,
        },
        "development_matrix": [list(case) for case in DEVELOPMENT_MATRIX],
        "training_sample_counts": list(DEVELOPMENT_SAMPLE_COUNTS),
        "domain_sample_counts": {
            OPTIMIZER_VALIDATION_DOMAIN: OPTIMIZER_VALIDATION_SAMPLE_COUNT,
            MODEL_SELECTION_VALIDATION_DOMAIN: (
                MODEL_SELECTION_VALIDATION_SAMPLE_COUNT
            ),
            TEST_DOMAIN: TEST_SAMPLE_COUNT,
        },
        "fitting": {
            "initialization_count": INITIALIZATION_COUNT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "maximum_epochs": MAXIMUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "selection": "minimum independent model-selection validation NLL",
        },
        "generalization": {
            "nat_per_residual_dimension": GENERALIZATION_NAT_PER_DIMENSION,
            "pooled_mcse_multiplier": GENERALIZATION_MCSE_MULTIPLIER,
        },
        "simulator": {
            "allocation_construction": CONSTRUCTION_METHOD,
            "total_mass": "exact root Gamma inverse CDF",
            "measurement_noise": "binary64 normal inverse CDF",
            "conditioner": "analytic-standardized log root mass",
            "whitening": "exact conditional residual covariance",
        },
        "scientific_thresholds": c1.THRESHOLDS,
        "common_lock": (
            "smallest common six-case all-larger passing suffix "
            "of length at least two"
        ),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "protected": {
            "catalogue_id": PROTECTED_HOLDOUT_CATALOGUE_ID,
            "protected_action_authorized": False,
            "g3_requires_passing_g2_holdout_eligible_certificate": True,
        },
    }


def _protocol_sha256() -> str:
    """Return the development declaration digest."""
    return _sha256_json(_protocol_payload())


def _validate_development_protocol() -> None:
    """Fail closed until the committed protocol digest is frozen."""
    if not DEVELOPMENT_PROTOCOL_SHA256:
        raise RuntimeError("the sbi-NSF development protocol has not been frozen")
    if _protocol_sha256() != DEVELOPMENT_PROTOCOL_SHA256:
        raise RuntimeError("the frozen sbi-NSF development protocol changed")
    if torch.get_default_dtype() != torch.float64:
        raise RuntimeError("Torch float64 default dtype is required")
    if torch.cuda.is_available():
        raise RuntimeError("the BP1 sbi-NSF protocol requires CPU-only Torch")


def _configure_torch() -> None:
    """Apply the source-pinned deterministic CPU runtime."""
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def _domain_seed(
    base_seed: int,
    *,
    case_id: str,
    domain: str,
    stream: str,
) -> int:
    """Derive one stable public-domain unsigned 32-bit seed."""
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise TypeError("base_seed must be an integer")
    if not 0 <= base_seed < 2**64:
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
    """One independent prior-predictive NLE domain."""

    targets: FloatArray
    conditions: FloatArray
    evidence: dict[str, Any]


def _simulated_domain(
    aggregation: AdditiveDirichletAggregation,
    labels: NDArray[np.int64],
    context: ResidualImageContext,
    unit_covariances: FloatArray,
    *,
    case_id: str,
    domain: str,
    sample_count: int,
    base_seed: int,
    total_shape: float,
    rate: float,
    conditioner_center: FloatArray,
    conditioner_scale: FloatArray,
) -> SimulatedDomain:
    """Generate one exact ancestral `(mass, noisy residual)` domain."""
    if sample_count < 1 or sample_count & (sample_count - 1):
        raise ValueError("domain sample_count must be a power of two")
    residual_seed = _domain_seed(
        base_seed,
        case_id=case_id,
        domain=domain,
        stream="conditional-dirichlet-residual",
    )
    joint_seed = _domain_seed(
        base_seed,
        case_id=case_id,
        domain=domain,
        stream="gamma-total-and-projected-noise",
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
        raise RuntimeError("the BP1 sbi-NSF trainer currently accepts root cases only")
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
        seed=joint_seed,  # pyright: ignore[reportCallIssue]
        optimization=None,
    )
    points = np.asarray(
        sobol.random_base2(int(math.log2(sample_count))),
        dtype=np.float64,
    )
    open_lower = np.nextafter(0.0, 1.0)
    open_upper = np.nextafter(1.0, 0.0)
    open_points = np.clip(points, open_lower, open_upper)
    totals = special.gammaincinv(total_shape, open_points[:, 0]) / rate
    gaussian = special.ndtri(open_points[:, 1:])
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
    conditions = (
        np.log(totals)[:, np.newaxis]
        - conditioner_center[np.newaxis, :]
    ) / conditioner_scale[np.newaxis, :]
    if (
        not np.all(np.isfinite(totals))
        or np.any(totals <= 0.0)
        or not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(conditions))
    ):
        raise RuntimeError("prior-predictive sbi-NSF simulations are invalid")
    return SimulatedDomain(
        cast(FloatArray, targets),
        cast(FloatArray, conditions),
        {
            "domain": domain,
            "sample_count": sample_count,
            "residual_seed": residual_seed,
            "joint_seed": joint_seed,
            "residual_bank_sha256": residual_bank.sha256,
            "unit_residual_sha256": _array_sha256(
                cast(FloatArray, unit_residual)
            ),
            "sobol_points_sha256": _array_sha256(
                cast(FloatArray, points)
            ),
            "totals_sha256": _array_sha256(cast(FloatArray, totals)),
            "targets_sha256": _array_sha256(cast(FloatArray, targets)),
            "conditions_sha256": _array_sha256(cast(FloatArray, conditions)),
        },
    )


def _log_probabilities(
    model: Any,
    domain: SimulatedDomain,
    *,
    batch_size: int = 8_192,
) -> FloatArray:
    """Evaluate one model over a complete independent domain."""
    values: list[NDArray[np.float64]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, domain.targets.shape[0], batch_size):
            stop = min(start + batch_size, domain.targets.shape[0])
            target = torch.as_tensor(
                domain.targets[start:stop],
                dtype=torch.float64,
            )
            condition = torch.as_tensor(
                domain.conditions[start:stop],
                dtype=torch.float64,
            )
            values.append(
                np.asarray(
                    model.log_prob(
                        target[None, :, :],
                        condition=condition,
                    )[0].cpu(),
                    dtype=np.float64,
                )
            )
    result = np.concatenate(values)
    if result.shape != (domain.targets.shape[0],) or not np.all(np.isfinite(result)):
        raise RuntimeError("sbi-NSF produced invalid domain log probabilities")
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
) -> tuple[ConditionalResidualImageSbiNsf, dict[str, Any]]:
    """Fit and independently score one deterministic initialization."""
    initialization_seed = _domain_seed(
        base_seed,
        case_id=case_id,
        domain=TRAINING_DOMAIN,
        stream=f"nsf-initialization-{initialization}",
    )
    optimizer_seed = _domain_seed(
        base_seed,
        case_id=case_id,
        domain=TRAINING_DOMAIN,
        stream=f"optimizer-{initialization}",
    )
    model = make_conditional_residual_nsf(
        context.residual_rank,
        context.region_count,
        source_seed=initialization_seed,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    training_targets = torch.as_tensor(
        domains[TRAINING_DOMAIN].targets,
        dtype=torch.float64,
    )
    training_conditions = torch.as_tensor(
        domains[TRAINING_DOMAIN].conditions,
        dtype=torch.float64,
    )
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(optimizer_seed)
    maximum_epochs = (
        SMOKE_MAXIMUM_EPOCHS
        if profile == "smoke"
        else MAXIMUM_EPOCHS
    )
    patience_limit = (
        SMOKE_EARLY_STOPPING_PATIENCE
        if profile == "smoke"
        else EARLY_STOPPING_PATIENCE
    )
    best_validation_nll = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    training_history: list[float] = []
    optimizer_validation_history: list[float] = []
    started = time.perf_counter()
    for _epoch in range(maximum_epochs):
        model.train()
        permutation = torch.randperm(
            training_targets.shape[0],
            generator=permutation_generator,
        )
        total_loss = 0.0
        total_count = 0
        for start in range(0, training_targets.shape[0], BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(
                training_targets[indices],
                condition=training_conditions[indices],
            )
            loss = torch.mean(losses)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("sbi-NSF training loss became non-finite")
            loss.backward()
            optimizer.step()
            count = int(indices.numel())
            total_loss += float(loss.detach()) * count
            total_count += count
        training_history.append(total_loss / total_count)
        optimizer_validation = _nll_summary(
            _log_probabilities(
                model,
                domains[OPTIMIZER_VALIDATION_DOMAIN],
            )
        )["nll_nat_per_draw"]
        optimizer_validation_history.append(optimizer_validation)
        if optimizer_validation < best_validation_nll:
            best_validation_nll = optimizer_validation
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                break
    fit_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("sbi-NSF optimization did not publish a finite state")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    artifact = ConditionalResidualImageSbiNsf(
        context,
        unit_covariances,
        conditioner_center,
        conditioner_scale,
        model,
        initialization_seed=initialization_seed,
        source_provenance=(
            f"{PROTOCOL}:{case_id}:base{base_seed}:init{initialization}:"
            f"source-{source_git_revision}"
        ),
    )
    validation = _nll_summary(
        _log_probabilities(
            artifact.model,
            domains[MODEL_SELECTION_VALIDATION_DOMAIN],
        )
    )
    test = _nll_summary(
        _log_probabilities(artifact.model, domains[TEST_DOMAIN])
    )
    gap = abs(validation["nll_nat_per_draw"] - test["nll_nat_per_draw"])
    pooled_mcse = math.hypot(
        validation["nll_mcse_nat_per_draw"],
        test["nll_mcse_nat_per_draw"],
    )
    gap_threshold = max(
        GENERALIZATION_NAT_PER_DIMENSION * context.residual_rank,
        GENERALIZATION_MCSE_MULTIPLIER * pooled_mcse,
    )
    return artifact, {
        "initialization": initialization,
        "initialization_seed": initialization_seed,
        "optimizer_seed": optimizer_seed,
        "fit_seconds": fit_seconds,
        "epochs": len(training_history),
        "training_loss_history": training_history,
        "optimizer_validation_loss_history": optimizer_validation_history,
        "best_optimizer_validation_nll_nat_per_draw": best_validation_nll,
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
    artifact: ConditionalResidualImageSbiNsf,
    observation: FloatArray,
    masses: FloatArray,
    log_prior: FloatArray,
    exact_log_likelihood: FloatArray,
    exact_summary: dict[str, Any],
    gradient_states: Sequence[dict[str, Any]],
    validation_state_mask: NDArray[np.bool_],
) -> dict[str, Any]:
    """Apply the unchanged C1 likelihood, gradient, and posterior gates."""
    return flow_screen._evaluate_artifact(  # pyright: ignore[reportPrivateUsage]
        artifact=cast(Any, artifact),
        observation=observation,
        masses=masses,
        log_prior=log_prior,
        exact_log_likelihood=exact_log_likelihood,
        exact_summary=exact_summary,
        gradient_states=gradient_states,
        validation_state_mask=validation_state_mask,
    )


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
    _configure_torch()
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
        or any(
            character not in "0123456789abcdef"
            for character in source_git_revision
        )
    ):
        raise ValueError("source_git_revision must be a full lower-case Git SHA")
    if (
        not isinstance(driver_sha256, str)
        or len(driver_sha256) != 64
        or any(character not in "0123456789abcdef" for character in driver_sha256)
    ):
        raise ValueError("driver_sha256 must be a lower-case SHA-256 digest")
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
        raise RuntimeError("the sbi-NSF development matrix excludes zero rank")
    unit_covariances = conditional_residual_unit_covariances(
        aggregation,
        context,
    )
    total_shape = float(np.sum(shapes))
    conditioner_center = np.asarray(
        [special.digamma(total_shape) - math.log(rate)],
        dtype=np.float64,
    )
    conditioner_scale = np.asarray(
        [math.sqrt(float(special.polygamma(1, total_shape)))],
        dtype=np.float64,
    )
    public_sample_count = (
        SMOKE_DOMAIN_SAMPLE_COUNT
        if profile == "smoke"
        else None
    )
    domain_counts = {
        TRAINING_DOMAIN: training_sample_count,
        OPTIMIZER_VALIDATION_DOMAIN: (
            public_sample_count or OPTIMIZER_VALIDATION_SAMPLE_COUNT
        ),
        MODEL_SELECTION_VALIDATION_DOMAIN: (
            public_sample_count or MODEL_SELECTION_VALIDATION_SAMPLE_COUNT
        ),
        TEST_DOMAIN: public_sample_count or TEST_SAMPLE_COUNT,
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
            total_shape=total_shape,
            rate=rate,
            conditioner_center=conditioner_center,
            conditioner_scale=conditioner_scale,
        )
        for domain, sample_count in domain_counts.items()
    }
    attempt_count = (
        SMOKE_INITIALIZATION_COUNT
        if profile == "smoke"
        else INITIALIZATION_COUNT
    )
    attempts: list[dict[str, Any]] = []
    artifacts: list[ConditionalResidualImageSbiNsf] = []
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
    replay = ConditionalResidualImageSbiNsf.from_bytes(
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
    return {
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
        "selected_generalization_pass": selected_attempt["generalization_pass"],
        "evaluation": evaluation,
        "task_pass": (
            smoke_task_pass
            if profile == "smoke"
            else development_task_pass
        ),
    }, selected_bytes


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
    """Publish artifact, report, and completion marker in that order."""
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{result['case_id']}__S{result['training_sample_count']}"
        f"__base{result['base_seed']}"
    )
    artifact_path = output_directory / f"{stem}.nsf"
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
        "schema": "rjmcmc-conditional-residual-image-sbi-nsf-task-complete-v1",
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
    parser.add_argument("--training-sample-count", type=int, required=True)
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEVELOPMENT_SELECTION_SEED,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--print-protocol-sha256", action="store_true")
    args = parser.parse_args()
    if args.print_protocol_sha256:
        _configure_torch()
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
    print(_canonical_json({"task_pass": result["task_pass"], "paths": paths}))


if __name__ == "__main__":
    main()
