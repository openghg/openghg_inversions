#!/usr/bin/env python3
"""Fit and score one exact-oracle score-regularized projected-root flow.

The development profile is one shard of the frozen N1 matrix.  It constructs
three observation-blind, domain-separated simulator catalogues, fits both
predeclared initializations with deterministic score-microbatch Adam, selects
on the independent model-selection composite loss, and then applies the
unchanged C1 likelihood, posterior, evidence, and centered-gradient gates.

The smoke profile is a separate, deliberately small N0 engineering identity.
Neither profile can derive a protected-domain seed or write below a protected
or ``PARIS_inversions`` path.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Sequence, TypeAlias, cast

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1  # noqa: E402
from examples.rjmcmc import score_regularized_flow_tiny_domains as tiny_domains  # noqa: E402
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_artifact import (  # noqa: E402
    GAMMA_LOG_MASS_CONDITIONING_RULE,
    ScoreRegularizedRootFlow,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_optimizer import (  # noqa: E402
    fit_score_regularized_flow,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_training import (  # noqa: E402
    FLOW_INVERT,
    FLOW_LAYERS,
    FLOW_NN_DEPTH,
    FLOW_NN_WIDTH,
    FLOW_SPLINE_INTERVAL,
    FLOW_SPLINE_KNOTS,
    make_score_regularized_conditional_flow,
    raw_log_mass_condition_score,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_regularized_flow import (  # noqa: E402
    standardization_scale,
)

FloatArray: TypeAlias = NDArray[np.float64]
Profile = Literal["smoke", "development"]

SCHEMA = "rjmcmc-score-regularized-flow-tiny-screen-v1"
PROTOCOL = tiny_domains.PROTOCOL
DEVELOPMENT_PROTOCOL_SHA256 = (
    "f03b5c3ba9f22bc12992807631fc14660ddfe59cbcefdd1ce40f8bfc4a9a8f0a"
)

DEVELOPMENT_MATRIX = tiny_domains.DEVELOPMENT_MATRIX
SMOKE_MATRIX = (("near_gaussian", "two_cell", "root"),)
DEVELOPMENT_SAMPLE_COUNTS = (4_096, 16_384, 65_536, 262_144)
SMOKE_SAMPLE_COUNTS = (64,)
DEVELOPMENT_BASE_SEED = 731
CONFIRMATION_BASE_SEEDS = (1_877, 4_099, 8_317)
SMOKE_BASE_SEED = 731

MODEL_SELECTION_SAMPLE_COUNT = 65_536
REPORTING_TEST_SAMPLE_COUNT = 131_072
SMOKE_MODEL_SELECTION_SAMPLE_COUNT = 64
SMOKE_REPORTING_TEST_SAMPLE_COUNT = 128

INITIALIZATION_COUNT = 2
LEARNING_RATE = 5.0e-4
BATCH_SIZE = 1_024
SCORE_MICROBATCH_SIZE = 64
MAXIMUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
INTERNAL_VALIDATION_PROPORTION = 0.1

SMOKE_INITIALIZATION_COUNT = 1
SMOKE_BATCH_SIZE = 32
SMOKE_SCORE_MICROBATCH_SIZE = 8
SMOKE_MAXIMUM_EPOCHS = 1
SMOKE_EARLY_STOPPING_PATIENCE = 0
SMOKE_INTERNAL_VALIDATION_PROPORTION = 0.25

SCORING_CHUNK_SIZE = 2_048
SMOKE_SCORING_CHUNK_SIZE = 64
GENERALIZATION_NAT_PER_DIMENSION = 0.02
GENERALIZATION_MCSE_MULTIPLIER = 5.0

FLOW_INITIALIZATION_STREAMS = (
    "flow-initialization-0",
    "flow-initialization-1",
)
OPTIMIZER_STREAMS = ("optimizer-0", "optimizer-1")
PRIVATE_TASK_STREAMS = FLOW_INITIALIZATION_STREAMS + OPTIMIZER_STREAMS

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
    """Return the digest of strict canonical JSON."""
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _driver_sha256() -> str:
    """Return the identity of this exact driver source."""
    return _sha256_file(Path(__file__).resolve())


def _runtime_versions() -> dict[str, str]:
    """Return the dependency versions entering artifact and fit identity."""
    return {
        package: version(package)
        for package in (
            "equinox",
            "flowjax",
            "jax",
            "jaxlib",
            "numpy",
            "optax",
            "paramax",
            "scipy",
        )
    }


def _protocol_payload() -> dict[str, Any]:
    """Return the complete frozen N1 development protocol."""
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
        "matrix": DEVELOPMENT_MATRIX,
        "training_sample_counts": DEVELOPMENT_SAMPLE_COUNTS,
        "development_base_seed": DEVELOPMENT_BASE_SEED,
        "confirmation_base_seeds": CONFIRMATION_BASE_SEEDS,
        "model_selection_sample_count": MODEL_SELECTION_SAMPLE_COUNT,
        "reporting_test_sample_count": REPORTING_TEST_SAMPLE_COUNT,
        "public_domains": tiny_domains.PUBLIC_DOMAINS,
        "simulator_streams": tiny_domains.SIMULATOR_STREAMS,
        "initialization_streams": FLOW_INITIALIZATION_STREAMS,
        "optimizer_streams": OPTIMIZER_STREAMS,
        "stream_contract": (
            "uint32_le_first4_sha256(protocol||uint64_le(base_seed)||"
            "case_id||public_domain||stream_name)"
        ),
        "flow": {
            "layers": FLOW_LAYERS,
            "spline_knots": FLOW_SPLINE_KNOTS,
            "spline_interval": FLOW_SPLINE_INTERVAL,
            "nn_width": FLOW_NN_WIDTH,
            "nn_depth": FLOW_NN_DEPTH,
            "activation": "tanh",
            "invert": FLOW_INVERT,
            "one_dimensional_specialization": (
                "conditional-masked-autoregressive-rational-quadratic-spline"
            ),
        },
        "fit": {
            "initialization_count": INITIALIZATION_COUNT,
            "optimizer": "adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "score_microbatch_size": SCORE_MICROBATCH_SIZE,
            "maximum_epochs": MAXIMUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "internal_validation_proportion": (
                INTERNAL_VALIDATION_PROPORTION
            ),
            "objective": "nll_per_q_plus_raw_log_mass_score_mse_per_q",
            "selection": (
                "minimum independent model-selection composite loss then "
                "initialization index"
            ),
        },
        "scoring_chunk_size": SCORING_CHUNK_SIZE,
        "generalization": {
            "nat_per_dimension": GENERALIZATION_NAT_PER_DIMENSION,
            "mcse_multiplier": GENERALIZATION_MCSE_MULTIPLIER,
        },
        "c1_thresholds": c1.THRESHOLDS,
        "c1_gradient_step": c1.GRADIENT_STEP,
        "conditioning_rule": GAMMA_LOG_MASS_CONDITIONING_RULE,
        "protected_holdout_catalogue_sha256": (
            PROTECTED_HOLDOUT_CATALOGUE_SHA256
        ),
        "training_size_lock": (
            "smallest common six-case size starting an all-larger passing "
            "suffix of length at least two"
        ),
        "runtime": _runtime_versions(),
    }


def _protocol_sha256() -> str:
    """Return the canonical development-protocol identity."""
    return _sha256_json(_protocol_payload())


def _validate_development_protocol() -> None:
    """Fail closed until the complete protocol identity is frozen."""
    if not DEVELOPMENT_PROTOCOL_SHA256:
        raise RuntimeError("the development protocol identity has not been frozen")
    if _protocol_sha256() != DEVELOPMENT_PROTOCOL_SHA256:
        raise RuntimeError("the frozen score-regularized protocol identity changed")
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("JAX float64 mode is required")


def _case_id(regime_name: str, family: str) -> str:
    """Return and validate one frozen root-case identity."""
    key = (regime_name, family, "root")
    if key not in DEVELOPMENT_MATRIX:
        raise ValueError("case is not one of the six frozen root cases")
    return f"{regime_name}__{family}__root"


def _task_stream_seed(
    base_seed: int,
    *,
    case_id: str,
    stream_name: str,
) -> int:
    """Derive an initialization/optimizer seed from the frozen byte contract."""
    if (
        not isinstance(base_seed, int)
        or isinstance(base_seed, bool)
        or not 0 <= base_seed < 2**64
    ):
        raise ValueError("base_seed must be an unsigned 64-bit integer")
    if case_id not in tiny_domains.CASE_IDS:
        raise ValueError("case_id is not one of the six frozen root cases")
    if stream_name not in PRIVATE_TASK_STREAMS:
        raise ValueError("unknown private task stream")
    digest = hashlib.sha256(PROTOCOL.encode("ascii"))
    digest.update(base_seed.to_bytes(8, byteorder="little", signed=False))
    digest.update(case_id.encode("ascii"))
    digest.update(tiny_domains.TRAINING_DOMAIN.encode("ascii"))
    digest.update(stream_name.encode("ascii"))
    return int.from_bytes(digest.digest()[:4], byteorder="little", signed=False)


def _mean_mcse(values: FloatArray) -> tuple[float, float]:
    """Return a finite mean and independent-draw diagnostic MCSE."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("diagnostic values must be a finite vector of length at least two")
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1) / math.sqrt(array.size)),
    )


def _score_domain(
    artifact: ScoreRegularizedRootFlow,
    domain: tiny_domains.TinyScoreDomain,
    *,
    chunk_size: int,
) -> dict[str, float]:
    """Score one independent simulator domain in bounded deterministic chunks."""
    if artifact.leading_rank != domain.spectrum.retained_rank:
        raise ValueError("tiny exact-oracle artifacts must learn the full spectrum")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    count, dimension = domain.standardized_draw.shape
    if dimension != artifact.leading_rank:
        raise ValueError("domain dimension does not match artifact")

    def observation_score_one(
        target: jax.Array,
        raw_tau: jax.Array,
    ) -> jax.Array:
        condition = jnp.asarray(
            [
                (raw_tau - artifact.condition_center)
                / artifact.condition_scale
            ],
            dtype=jnp.float64,
        )
        return jax.grad(
            lambda value: artifact.flow.log_prob(value, condition)
        )(target)

    batched_observation_score = jax.jit(
        jax.vmap(observation_score_one)
    )
    negative_log_likelihood_parts: list[FloatArray] = []
    mass_risk_parts: list[FloatArray] = []
    observation_risk_parts: list[FloatArray] = []
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        projected = jnp.asarray(
            domain.standardized_draw[start:stop],
            dtype=jnp.float64,
        )
        raw_tau = jnp.asarray(
            domain.raw_log_mass[start:stop],
            dtype=jnp.float64,
        )
        conditions = jnp.expand_dims(
            (raw_tau - artifact.condition_center) / artifact.condition_scale,
            axis=-1,
        )
        log_probabilities = np.asarray(
            artifact.flow.log_prob(projected, conditions),
            dtype=np.float64,
        )
        predicted_mass_score = np.asarray(
            raw_log_mass_condition_score(
                artifact.flow,
                projected,
                raw_tau,
                condition_center=artifact.condition_center,
                condition_scale=artifact.condition_scale,
            ),
            dtype=np.float64,
        )
        predicted_observation_score = np.asarray(
            batched_observation_score(projected, raw_tau),
            dtype=np.float64,
        )
        scales = np.asarray(
            standardization_scale(
                domain.total_mass[start:stop],
                domain.spectrum.eigenvalues,
            ),
            dtype=np.float64,
        )
        mass_error = (
            predicted_mass_score - domain.mass_score_target[start:stop]
        )
        observation_error = (
            predicted_observation_score
            - domain.observation_score_target[start:stop]
        ) / scales
        negative_log_likelihood_parts.append(
            cast(FloatArray, -log_probabilities)
        )
        mass_risk_parts.append(
            cast(FloatArray, np.square(mass_error) / dimension)
        )
        observation_risk_parts.append(
            cast(
                FloatArray,
                np.mean(np.square(observation_error), axis=1),
            )
        )
    negative_log_likelihood = np.concatenate(negative_log_likelihood_parts)
    mass_risk = np.concatenate(mass_risk_parts)
    observation_risk = np.concatenate(observation_risk_parts)
    nll_mean, nll_mcse = _mean_mcse(negative_log_likelihood)
    mass_mean, mass_mcse = _mean_mcse(mass_risk)
    observation_mean, observation_mcse = _mean_mcse(observation_risk)
    return {
        "sample_count": count,
        "nll_nat_per_draw": nll_mean,
        "nll_mcse_nat_per_draw": nll_mcse,
        "nll_nat_per_dimension": nll_mean / dimension,
        "mass_score_risk_per_dimension": mass_mean,
        "mass_score_risk_mcse_per_dimension": mass_mcse,
        "observation_score_risk_per_dimension": observation_mean,
        "observation_score_risk_mcse_per_dimension": observation_mcse,
        "composite_loss": nll_mean / dimension + mass_mean,
    }


def _fit_controls(profile: Profile) -> dict[str, int | float]:
    """Return frozen development controls or separate small smoke controls."""
    if profile == "development":
        return {
            "batch_size": BATCH_SIZE,
            "score_microbatch_size": SCORE_MICROBATCH_SIZE,
            "maximum_epochs": MAXIMUM_EPOCHS,
            "patience": EARLY_STOPPING_PATIENCE,
            "validation_proportion": INTERNAL_VALIDATION_PROPORTION,
            "scoring_chunk_size": SCORING_CHUNK_SIZE,
        }
    return {
        "batch_size": SMOKE_BATCH_SIZE,
        "score_microbatch_size": SMOKE_SCORE_MICROBATCH_SIZE,
        "maximum_epochs": SMOKE_MAXIMUM_EPOCHS,
        "patience": SMOKE_EARLY_STOPPING_PATIENCE,
        "validation_proportion": SMOKE_INTERNAL_VALIDATION_PROPORTION,
        "scoring_chunk_size": SMOKE_SCORING_CHUNK_SIZE,
    }


def _fit_attempt(
    training: tiny_domains.TinyScoreDomain,
    model_selection: tiny_domains.TinyScoreDomain,
    reporting_test: tiny_domains.TinyScoreDomain,
    *,
    case_id: str,
    base_seed: int,
    initialization: int,
    profile: Profile,
    source_git_revision: str,
) -> tuple[ScoreRegularizedRootFlow, dict[str, Any]]:
    """Fit and independently score one deterministic initialization."""
    initialization_seed = _task_stream_seed(
        base_seed,
        case_id=case_id,
        stream_name=f"flow-initialization-{initialization}",
    )
    optimizer_seed = _task_stream_seed(
        base_seed,
        case_id=case_id,
        stream_name=f"optimizer-{initialization}",
    )
    model = make_score_regularized_conditional_flow(
        training.spectrum.retained_rank,
        source_seed=initialization_seed,
    )
    controls = _fit_controls(profile)
    fitted, history = fit_score_regularized_flow(
        jr.key(optimizer_seed),
        model,
        jnp.asarray(training.standardized_draw, dtype=jnp.float64),
        jnp.asarray(training.raw_log_mass, dtype=jnp.float64),
        jnp.asarray(training.mass_score_target, dtype=jnp.float64),
        condition_center=training.evidence.conditioning_center,
        condition_scale=training.evidence.conditioning_scale,
        learning_rate=LEARNING_RATE,
        batch_size=cast(int, controls["batch_size"]),
        score_microbatch_size=cast(
            int,
            controls["score_microbatch_size"],
        ),
        val_prop=cast(float, controls["validation_proportion"]),
        max_epochs=cast(int, controls["maximum_epochs"]),
        patience=cast(int, controls["patience"]),
    )
    if not history.train or not history.validation:
        raise RuntimeError("optimizer returned empty histories")
    if not np.all(np.isfinite(history.train)) or not np.all(
        np.isfinite(history.validation)
    ):
        raise RuntimeError("optimizer returned non-finite histories")
    artifact = ScoreRegularizedRootFlow(
        training.spectrum,
        training.spectrum.retained_rank,
        training.evidence.gamma_shape,
        training.evidence.gamma_rate,
        fitted,
        conditioning_rule_id=GAMMA_LOG_MASS_CONDITIONING_RULE,
        initialization_seed=initialization_seed,
        source_provenance=(
            f"{PROTOCOL}:{case_id}:base={base_seed}:"
            f"initialization={initialization}:git={source_git_revision}"
        ),
    )
    scoring_chunk_size = cast(int, controls["scoring_chunk_size"])
    validation_summary = _score_domain(
        artifact,
        model_selection,
        chunk_size=scoring_chunk_size,
    )
    test_summary = _score_domain(
        artifact,
        reporting_test,
        chunk_size=scoring_chunk_size,
    )
    nll_gap = abs(
        validation_summary["nll_nat_per_draw"]
        - test_summary["nll_nat_per_draw"]
    )
    pooled_nll_mcse = math.hypot(
        validation_summary["nll_mcse_nat_per_draw"],
        test_summary["nll_mcse_nat_per_draw"],
    )
    generalization_threshold = max(
        GENERALIZATION_NAT_PER_DIMENSION
        * training.spectrum.retained_rank,
        GENERALIZATION_MCSE_MULTIPLIER * pooled_nll_mcse,
    )
    mass_score_gap = abs(
        validation_summary["mass_score_risk_per_dimension"]
        - test_summary["mass_score_risk_per_dimension"]
    )
    pooled_mass_score_mcse = math.hypot(
        validation_summary["mass_score_risk_mcse_per_dimension"],
        test_summary["mass_score_risk_mcse_per_dimension"],
    )
    observation_score_gap = abs(
        validation_summary["observation_score_risk_per_dimension"]
        - test_summary["observation_score_risk_per_dimension"]
    )
    pooled_observation_score_mcse = math.hypot(
        validation_summary["observation_score_risk_mcse_per_dimension"],
        test_summary["observation_score_risk_mcse_per_dimension"],
    )
    return artifact, {
        "initialization": initialization,
        "initialization_seed": initialization_seed,
        "optimizer_seed": optimizer_seed,
        "epochs": len(history.train),
        "best_epoch": history.best_epoch,
        "stopped_early": history.stopped_early,
        "training_composite_loss_history": list(history.train),
        "internal_validation_composite_loss_history": list(
            history.validation
        ),
        "model_selection": validation_summary,
        "reporting_test": test_summary,
        "absolute_model_selection_test_nll_gap_nat_per_draw": nll_gap,
        "pooled_nll_mcse_nat_per_draw": pooled_nll_mcse,
        "generalization_threshold_nat_per_draw": generalization_threshold,
        "generalization_pass": bool(nll_gap <= generalization_threshold),
        "absolute_model_selection_test_mass_score_risk_gap": (
            mass_score_gap
        ),
        "pooled_mass_score_risk_mcse": pooled_mass_score_mcse,
        "mass_score_five_mcse_agreement": bool(
            mass_score_gap
            <= GENERALIZATION_MCSE_MULTIPLIER * pooled_mass_score_mcse
        ),
        "absolute_model_selection_test_observation_score_risk_gap": (
            observation_score_gap
        ),
        "pooled_observation_score_risk_mcse": (
            pooled_observation_score_mcse
        ),
        "observation_score_five_mcse_agreement": bool(
            observation_score_gap
            <= GENERALIZATION_MCSE_MULTIPLIER
            * pooled_observation_score_mcse
        ),
        "artifact_sha256": artifact.artifact_sha256,
    }


def _select_initialization(attempts: Sequence[dict[str, Any]]) -> int:
    """Select minimum external composite loss with index as the exact tie rule."""
    if not attempts:
        raise ValueError("at least one fit attempt is required")
    for index, attempt in enumerate(attempts):
        if attempt.get("initialization") != index:
            raise ValueError("fit attempts must use contiguous initialization indices")
        try:
            composite = float(
                attempt["model_selection"]["composite_loss"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "fit attempt has no valid model-selection composite loss"
            ) from error
        if not math.isfinite(composite):
            raise ValueError(
                "model-selection composite loss must be finite"
            )
    return min(
        range(len(attempts)),
        key=lambda index: (
            attempts[index]["model_selection"]["composite_loss"],
            index,
        ),
    )


def _evaluate_artifact(
    *,
    artifact: ScoreRegularizedRootFlow,
    observation: FloatArray,
    masses: FloatArray,
    log_prior: FloatArray,
    exact_log_likelihood: FloatArray,
    exact_summary: dict[str, Any],
    gradient_states: Sequence[dict[str, Any]],
    validation_state_mask: NDArray[np.bool_],
) -> dict[str, Any]:
    """Apply the unchanged C1 likelihood, gradient, posterior, and evidence gates."""
    totals = np.sum(masses, axis=1)
    learned_log_likelihood = artifact.log_likelihood_batch(
        observation,
        totals,
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
            root_total = float(np.sum(c1.coordinate_to_masses(value)))
            return artifact.log_likelihood(observation, root_total)

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
    finite_likelihood_pass = bool(
        np.all(np.isfinite(learned_log_likelihood))
    )
    checks["finite_normalized_likelihood"] = finite_likelihood_pass
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
            "normalized_density_by_construction": True,
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
    """Fit and exact-oracle score one source-pinned root case."""
    if profile not in ("smoke", "development"):
        raise ValueError("profile must be smoke or development")
    case_id = _case_id(regime_name, family)
    allowed_matrix = SMOKE_MATRIX if profile == "smoke" else DEVELOPMENT_MATRIX
    if (regime_name, family, "root") not in allowed_matrix:
        raise ValueError(f"case is not available in {profile} profile")
    allowed_counts = (
        SMOKE_SAMPLE_COUNTS
        if profile == "smoke"
        else DEVELOPMENT_SAMPLE_COUNTS
    )
    if training_sample_count not in allowed_counts:
        raise ValueError("training_sample_count is not source-pinned")
    expected_base_seed = (
        SMOKE_BASE_SEED if profile == "smoke" else DEVELOPMENT_BASE_SEED
    )
    if base_seed != expected_base_seed:
        raise ValueError("base_seed is not source-pinned for this profile")
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
    if driver_sha256 != _driver_sha256():
        raise ValueError("driver_sha256 does not match this exact source file")

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
    model_selection_count = (
        SMOKE_MODEL_SELECTION_SAMPLE_COUNT
        if profile == "smoke"
        else MODEL_SELECTION_SAMPLE_COUNT
    )
    reporting_test_count = (
        SMOKE_REPORTING_TEST_SAMPLE_COUNT
        if profile == "smoke"
        else REPORTING_TEST_SAMPLE_COUNT
    )
    domain_counts = {
        tiny_domains.TRAINING_DOMAIN: training_sample_count,
        tiny_domains.MODEL_SELECTION_VALIDATION_DOMAIN: (
            model_selection_count
        ),
        tiny_domains.DEVELOPMENT_REPORTING_TEST_DOMAIN: reporting_test_count,
    }
    constructed = {
        domain: tiny_domains.simulate_tiny_score_domain(
            case_id,
            domain=domain,
            sample_count=sample_count,
            base_seed=base_seed,
        )
        for domain, sample_count in domain_counts.items()
    }
    training = constructed[tiny_domains.TRAINING_DOMAIN]
    model_selection = constructed[
        tiny_domains.MODEL_SELECTION_VALIDATION_DOMAIN
    ]
    reporting_test = constructed[
        tiny_domains.DEVELOPMENT_REPORTING_TEST_DOMAIN
    ]
    spectrum_identities = {
        domain.evidence.spectrum_sha256 for domain in constructed.values()
    }
    scientific_input_identities = {
        domain.evidence.scientific_input_sha256
        for domain in constructed.values()
    }
    if len(spectrum_identities) != 1 or len(scientific_input_identities) != 1:
        raise RuntimeError("independent domains do not share one scientific model")

    attempt_count = (
        SMOKE_INITIALIZATION_COUNT
        if profile == "smoke"
        else INITIALIZATION_COUNT
    )
    attempts: list[dict[str, Any]] = []
    artifacts: list[ScoreRegularizedRootFlow] = []
    for initialization in range(attempt_count):
        artifact, attempt = _fit_attempt(
            training,
            model_selection,
            reporting_test,
            case_id=case_id,
            base_seed=base_seed,
            initialization=initialization,
            profile=profile,
            source_git_revision=source_git_revision,
        )
        artifacts.append(artifact)
        attempts.append(attempt)
    selected_index = _select_initialization(attempts)
    selected = artifacts[selected_index]
    selected_attempt = attempts[selected_index]
    selected_bytes = selected.to_bytes()
    replay = ScoreRegularizedRootFlow.from_bytes(
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
    finite_score_pass = bool(
        all(
            math.isfinite(
                float(
                    attempt[domain][metric]
                )
            )
            for attempt in attempts
            for domain in ("model_selection", "reporting_test")
            for metric in (
                "mass_score_risk_per_dimension",
                "observation_score_risk_per_dimension",
            )
        )
    )
    fit_pass = bool(len(attempts) == attempt_count and finite_score_pass)
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
        "runtime": _runtime_versions(),
        "case_id": case_id,
        "training_sample_count": training_sample_count,
        "base_seed": base_seed,
        "leading_rank": training.spectrum.retained_rank,
        "spectrum_sha256": training.evidence.spectrum_sha256,
        "scientific_input_sha256": (
            training.evidence.scientific_input_sha256
        ),
        "domain_evidence": {
            name: domain.evidence.payload()
            for name, domain in constructed.items()
        },
        "fit_controls": _fit_controls(profile),
        "attempts": attempts,
        "selected_initialization": selected_index,
        "selection_rule": (
            "minimum independent model-selection composite loss then "
            "initialization index"
        ),
        "selected_artifact_sha256": selected.artifact_sha256,
        "artifact_replay_pass": artifact_replay_pass,
        "finite_score_pass": finite_score_pass,
        "fit_pass": fit_pass,
        "selected_generalization_pass": selected_attempt[
            "generalization_pass"
        ],
        "evaluation": evaluation,
        "access_audit": {
            "realized_mf_accessed": False,
            "protected_catalogue_accessed": False,
            "paris_inversions_written": False,
        },
        "task_pass": (
            smoke_task_pass
            if profile == "smoke"
            else development_task_pass
        ),
    }
    return result, selected_bytes


def _validate_output_directory(path: Path) -> Path:
    """Reject protected and production output paths before any write."""
    if path.is_symlink():
        raise ValueError("output_directory must be a real directory, not a symlink")
    resolved = path.resolve(strict=False)
    lowered = tuple(part.lower() for part in resolved.parts)
    if "paris_inversions" in lowered or any(
        "protected" in part for part in lowered
    ):
        raise ValueError(
            "output_directory must not be protected or below PARIS_inversions"
        )
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("output_directory exists and is not a directory")
    return resolved


def _publish_create_only(path: Path, payload: bytes) -> None:
    """Atomically create one final file, refusing any existing target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
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
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_result(
    output_directory: Path,
    result: dict[str, Any],
    artifact_bytes: bytes,
) -> dict[str, str]:
    """Publish the artifact, report, and completion marker in that order."""
    output_directory = _validate_output_directory(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{result['case_id']}__S{result['training_sample_count']}"
        f"__base{result['base_seed']}"
    )
    artifact_path = output_directory / f"{stem}.score-flow"
    report_path = output_directory / f"{stem}.json"
    marker_path = output_directory / f"{stem}.complete.json"
    for path in (artifact_path, report_path, marker_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to replace existing output: {path}")
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if artifact_sha256 != result["selected_artifact_sha256"]:
        raise RuntimeError("artifact bytes do not match selected identity")
    _publish_create_only(artifact_path, artifact_bytes)
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
    report_bytes = (_canonical_json(envelope) + "\n").encode("utf-8")
    _publish_create_only(report_path, report_bytes)
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    marker_payload = {
        "schema": "rjmcmc-score-regularized-flow-task-complete-v1",
        "case_id": result["case_id"],
        "training_sample_count": result["training_sample_count"],
        "base_seed": result["base_seed"],
        "task_pass": result["task_pass"],
        "artifact_sha256": artifact_sha256,
        "report_sha256": report_sha256,
    }
    _publish_create_only(
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
        default=DEVELOPMENT_BASE_SEED,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--print-protocol-sha256", action="store_true")
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
