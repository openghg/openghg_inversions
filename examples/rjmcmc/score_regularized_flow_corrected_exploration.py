#!/usr/bin/env python3
"""Run one corrected, lightweight score-flow exploration attempt.

One invocation is one array-friendly combination of case, catalogue size,
loss ablation, and initialization.  It never selects a structural basis from
an approximate evidence and never opens protected or PARIS inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any, cast

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import equinox as eqx  # noqa: E402
import numpy as np  # noqa: E402
from scipy import special, stats  # noqa: E402

from examples.rjmcmc import score_regularized_flow_tiny_domains as domains  # noqa: E402
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_artifact import (  # noqa: E402
    GAMMA_LOG_MASS_CONDITIONING_RULE,
    ScoreRegularizedRootFlow,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_exploration import (  # noqa: E402
    ExplorationFitHistory,
    ExplorationLossConfig,
    fit_exploratory_score_flow,
    initialization_loss_diagnostics,
    loss_scale_diagnostics,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_training import (  # noqa: E402
    conditional_log_prob_and_observation_score,
    make_score_regularized_conditional_flow,
    raw_log_mass_condition_log_prob_and_score,
)
from openghg_inversions.experimental.rjmcmc import aggregation_error_tiny_oracle  # noqa: E402

SCHEMA = "rjmcmc-score-nle-corrected-exploration-attempt-v1"
PROTOCOL = "rjmcmc-score-nle-corrected-pcg64-exploration-v1"
LEGACY_ORACLE_BUNDLE_SCHEMA = "rjmcmc-score-nle-corrected-oracle-bundle-v1"
PROMOTION_ORACLE_BUNDLE_SCHEMA = "rjmcmc-score-nle-corrected-oracle-bundle-v2"
ORACLE_BUNDLE_SCHEMA = PROMOTION_ORACLE_BUNDLE_SCHEMA
SELECTED_CASES = (
    "near_gaussian__two_cell__root",
    "skewed__four_cell__root",
    "boundary_heavy__two_cell__root",
)
PROMOTION_CASES = aggregation_error_tiny_oracle.CASE_IDS
OVERFIT_CASES = (
    "near_gaussian__two_cell__root",
    "skewed__four_cell__root",
)
STANDARD_SAMPLE_COUNTS = (4_096, 16_384)
PROMOTION_SAMPLE_COUNTS = STANDARD_SAMPLE_COUNTS
PROMOTION_ORACLE_ORDER_LADDERS = {
    "near_gaussian__two_cell__root": (16, 32),
    "near_gaussian__four_cell__root": (8, 12, 16),
    "skewed__two_cell__root": (8, 16, 32),
    "skewed__four_cell__root": (8, 12, 16),
    "boundary_heavy__two_cell__root": (16, 32, 64),
    "boundary_heavy__four_cell__root": (12, 16, 24),
}
PROMOTION_EXECUTION_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "JAX_ENABLE_X64": "True",
    "JAX_PLATFORM_NAME": "cpu",
    "XLA_FLAGS": (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 "
        "--xla_cpu_parallel_codegen_split_count=1"
    ),
}
OVERFIT_SAMPLE_COUNT = 256
CONFIG_IDS = (
    "nll_only",
    "fisher_partial_joint",
    "nll_pretrain_then_partial",
    "fisher_observation_joint",
)
GRID_COUNTS = (1_024, 2_048, 4_096, 8_192)
GRID_EVIDENCE_TOLERANCE_NAT = 0.005
GRID_POINTWISE_TOLERANCE_NAT = 0.005
GRID_POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD = 0.005
GRID_POSTERIOR_SD_RELATIVE_TOLERANCE = 0.002


def _canonical_json(payload: object, *, pretty: bool) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=True,
    )


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload, pretty=False).encode("ascii")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_replay_diagnostics(
    trained: ScoreRegularizedRootFlow,
    canonical: ScoreRegularizedRootFlow,
    observation: np.ndarray,
    totals: np.ndarray,
) -> dict[str, object]:
    """Authenticate parameters and bound trained-to-replay roundoff."""
    if jax.tree_util.tree_structure(trained.flow) != jax.tree_util.tree_structure(canonical.flow):
        raise RuntimeError("canonical replay changed the fitted-flow tree structure.")
    trained_leaves = [
        np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(trained.flow) if eqx.is_inexact_array(leaf)
    ]
    canonical_leaves = [
        np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(canonical.flow) if eqx.is_inexact_array(leaf)
    ]
    if len(trained_leaves) != len(canonical_leaves):
        raise RuntimeError("canonical replay changed the fitted-flow tree.")
    for trained_leaf, canonical_leaf in zip(
        trained_leaves,
        canonical_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(trained_leaf, canonical_leaf)
    for name in (
        "observation_mean_design",
        "noise_sd",
        "basis",
        "eigenvalues",
    ):
        np.testing.assert_array_equal(
            getattr(trained.spectrum, name),
            getattr(canonical.spectrum, name),
        )

    trained_values = np.asarray(
        trained.log_likelihood_batch(observation, totals),
        dtype=np.float64,
    )
    canonical_values = np.asarray(
        canonical.log_likelihood_batch(observation, totals),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(canonical_values)):
        raise ValueError("canonical artifact likelihood diagnostic is non-finite.")
    trained_values_finite = bool(np.all(np.isfinite(trained_values)))
    exact_identity = {
        "flow_tree_structure_identical": True,
        "flow_float_leaf_count": len(trained_leaves),
        "flow_float_leaves_bitwise_identical": True,
        "spectrum_arrays_bitwise_identical": True,
    }
    if not trained_values_finite:
        return {
            "diagnostic": "non_authoritative_layout_roundoff_diagnostic",
            "gating": False,
            "canonical_replay_used_for_scientific_evaluation": True,
            **exact_identity,
            "canonical_values_finite": True,
            "trained_values_finite": False,
            "trained_to_canonical_likelihood_max_absolute_error_nat": None,
            "trained_to_canonical_likelihood_max_relative_error": None,
            "trained_to_canonical_likelihood_max_output_ulp_error": None,
            "advisory_scale_aware_epsilon_multiplier": 256.0,
            "maximum_fraction_of_advisory_roundoff_range": None,
            "within_advisory_roundoff_range": False,
        }
    absolute_error = np.abs(trained_values - canonical_values)
    denominator = np.maximum(
        np.maximum(np.abs(trained_values), np.abs(canonical_values)),
        np.finfo(np.float64).tiny,
    )
    relative_error = absolute_error / denominator
    epsilon = float(np.finfo(np.float64).eps)
    advisory_multiplier = 256.0
    advisory_scale = np.maximum(
        1.0,
        np.maximum(np.abs(trained_values), np.abs(canonical_values)),
    )
    advisory_bound = advisory_multiplier * epsilon * advisory_scale
    advisory_fraction = absolute_error / advisory_bound
    output_spacing = np.maximum(
        np.abs(np.spacing(trained_values)),
        np.abs(np.spacing(canonical_values)),
    )
    output_ulp_error = absolute_error / output_spacing
    return {
        "diagnostic": "non_authoritative_layout_roundoff_diagnostic",
        "gating": False,
        "canonical_replay_used_for_scientific_evaluation": True,
        **exact_identity,
        "canonical_values_finite": True,
        "trained_values_finite": True,
        "trained_to_canonical_likelihood_max_absolute_error_nat": float(np.max(absolute_error, initial=0.0)),
        "trained_to_canonical_likelihood_max_relative_error": float(np.max(relative_error, initial=0.0)),
        "trained_to_canonical_likelihood_max_output_ulp_error": float(np.max(output_ulp_error, initial=0.0)),
        "advisory_scale_aware_epsilon_multiplier": advisory_multiplier,
        "maximum_fraction_of_advisory_roundoff_range": float(np.max(advisory_fraction, initial=0.0)),
        "within_advisory_roundoff_range": bool(np.all(advisory_fraction <= 1.0)),
    }


def _json_bytes(payload: object) -> bytes:
    return f"{_canonical_json(payload, pretty=True)}\n".encode("ascii")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(_canonical_json(payload, pretty=True))
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _private_stream_seed(
    base_seed: int,
    *,
    case_id: str,
    stream_name: str,
) -> int:
    digest = hashlib.sha256(PROTOCOL.encode("ascii"))
    digest.update(base_seed.to_bytes(8, byteorder="little", signed=False))
    digest.update(case_id.encode("ascii"))
    digest.update(domains.TRAINING_DOMAIN.encode("ascii"))
    digest.update(stream_name.encode("ascii"))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)


def _jax_seed_from_pcg64(source_seed: int) -> int:
    generator = np.random.Generator(np.random.PCG64(source_seed))
    return int(generator.integers(0, 2**32, dtype=np.uint32))


def _stage_plan(
    config_id: str,
    maximum_epochs: int,
) -> tuple[tuple[str, int], ...]:
    if maximum_epochs < 1:
        raise ValueError("maximum epochs must be positive.")
    if config_id == "nll_pretrain_then_partial":
        if maximum_epochs < 2:
            raise ValueError("pretraining requires at least two maximum total epochs.")
        pretrain_epochs = maximum_epochs // 2
        return (
            ("nll_pretrain", pretrain_epochs),
            ("partial_score_finetune", maximum_epochs - pretrain_epochs),
        )
    return (("joint", maximum_epochs),)


def _private_stream_plan(
    base_seed: int,
    *,
    case_id: str,
    init_index: int,
    stage_count: int,
) -> tuple[dict[str, int | str], tuple[dict[str, int | str], ...]]:
    initialization_role = f"flow-initialization-{init_index}"
    initialization_source_seed = _private_stream_seed(
        base_seed,
        case_id=case_id,
        stream_name=initialization_role,
    )
    initialization = {
        "role": initialization_role,
        "pcg64_source_seed": initialization_source_seed,
        "derived_jax_seed": _jax_seed_from_pcg64(initialization_source_seed),
    }
    optimizer = tuple(
        {
            "role": f"optimizer-stage-{stage_index}-init-{init_index}",
            "pcg64_source_seed": source_seed,
            "derived_jax_seed": _jax_seed_from_pcg64(source_seed),
        }
        for stage_index in range(stage_count)
        for source_seed in (
            _private_stream_seed(
                base_seed,
                case_id=case_id,
                stream_name=(f"optimizer-stage-{stage_index}-init-{init_index}"),
            ),
        )
    )
    return initialization, optimizer


def _validate_private_stream_plan(
    initialization: dict[str, int | str],
    optimizer: tuple[dict[str, int | str], ...],
    constructed: dict[str, domains.TinyScoreDomain],
) -> dict[str, object]:
    simulator_source_seeds = {
        int(seed) for domain in constructed.values() for _, seed in domain.evidence.stream_seeds
    }
    private_records = (initialization, *optimizer)
    private_source_seeds = [int(record["pcg64_source_seed"]) for record in private_records]
    private_jax_seeds = [int(record["derived_jax_seed"]) for record in private_records]
    checks = {
        "private_pcg64_source_seeds_unique": (len(set(private_source_seeds)) == len(private_source_seeds)),
        "private_pcg64_source_seeds_disjoint_from_simulator": (
            simulator_source_seeds.isdisjoint(private_source_seeds)
        ),
        "private_derived_jax_seeds_unique": (len(set(private_jax_seeds)) == len(private_jax_seeds)),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "private initializer/optimizer streams are not fail-closed "
            "separated from simulator or one another."
        )
    return {
        "checks": checks,
        "simulator_source_seed_count": len(simulator_source_seeds),
        "private_source_seed_count": len(private_source_seeds),
    }


def _mean_iid_mcse(values: np.ndarray) -> tuple[float, float]:
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("IID diagnostic values must be a finite vector.")
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1) / math.sqrt(values.size)),
    )


def _data(
    domain: domains.TinyScoreDomain,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        jnp.asarray(domain.standardized_draw, dtype=jnp.float64),
        jnp.asarray(domain.raw_log_mass, dtype=jnp.float64),
        jnp.asarray(domain.mass_score_target, dtype=jnp.float64),
        jnp.asarray(domain.observation_score_target, dtype=jnp.float64),
    )


def _loss_config(
    config_id: str,
    scale_diagnostics: dict[str, Any],
    *,
    dimension: int,
) -> ExplorationLossConfig:
    partial_scale = float(scale_diagnostics["partial_score_rms"])
    observation_scales = tuple(
        float(value) for value in scale_diagnostics["observation_score_rms_by_coordinate"]
    )
    if len(observation_scales) != dimension:
        raise RuntimeError("training observation-score scales have wrong rank.")
    if config_id == "nll_only":
        return ExplorationLossConfig(
            include_partial_score=False,
            include_observation_score=False,
            partial_score_weight=0.0,
            observation_score_weight=0.0,
            partial_score_scale=partial_scale,
            observation_score_scales=observation_scales,
        )
    if config_id in (
        "fisher_partial_joint",
        "nll_pretrain_then_partial",
    ):
        return ExplorationLossConfig(
            include_partial_score=True,
            include_observation_score=False,
            partial_score_weight=1.0,
            observation_score_weight=0.0,
            partial_score_scale=partial_scale,
            observation_score_scales=observation_scales,
        )
    if config_id == "fisher_observation_joint":
        return ExplorationLossConfig(
            include_partial_score=False,
            include_observation_score=True,
            partial_score_weight=0.0,
            observation_score_weight=1.0,
            partial_score_scale=partial_scale,
            observation_score_scales=observation_scales,
        )
    raise ValueError("unknown corrected exploration config.")


def _nll_config(
    scale_diagnostics: dict[str, Any],
    *,
    dimension: int,
) -> ExplorationLossConfig:
    return _loss_config(
        "nll_only",
        scale_diagnostics,
        dimension=dimension,
    )


def _candidate_loss_rule(config_id: str) -> dict[str, object]:
    active_terms = ["nll_per_dimension"]
    stages = ["joint"]
    if config_id == "fisher_partial_joint":
        active_terms.append("fisher_scaled_partial_score")
    elif config_id == "nll_pretrain_then_partial":
        active_terms.append("fisher_scaled_partial_score")
        stages = ["nll_pretrain", "partial_score_finetune"]
    elif config_id == "fisher_observation_joint":
        active_terms.append("fisher_scaled_observation_score")
    elif config_id != "nll_only":
        raise ValueError("unknown corrected exploration config.")
    return {
        "active_terms": active_terms,
        "stage_sequence": stages,
        "nll_rule": "mean negative log likelihood divided by retained rank",
        "partial_score_scale_rule": ("training-target root-mean-square with fixed 1e-8 floor"),
        "observation_score_scale_rule": (
            "per-coordinate training-target root-mean-square with fixed "
            "1e-8 floor, then mean across coordinates"
        ),
        "active_auxiliary_weight": 1.0,
        "inactive_auxiliary_weight": 0.0,
    }


def _history_payload(history: ExplorationFitHistory) -> dict[str, object]:
    return {
        "training": [asdict(metrics) for metrics in history.training],
        "validation": [asdict(metrics) for metrics in history.validation],
        "best_epoch": history.best_epoch,
        "stopped_early": history.stopped_early,
        "stop_reason": history.stop_reason,
        "optimizer_state_reset_at_start": (history.optimizer_state_reset_at_start),
    }


def _score_domain(
    flow: Any,
    domain: domains.TinyScoreDomain,
    *,
    partial_scale: float,
    observation_scales: tuple[float, ...],
    chunk_size: int = 512,
) -> dict[str, object]:
    nll_rows: list[np.ndarray] = []
    partial_rows: list[np.ndarray] = []
    observation_rows: list[np.ndarray] = []
    for start in range(0, domain.total_mass.size, chunk_size):
        stop = min(start + chunk_size, domain.total_mass.size)
        projected = jnp.asarray(
            domain.standardized_draw[start:stop],
            dtype=jnp.float64,
        )
        tau = jnp.asarray(domain.raw_log_mass[start:stop], dtype=jnp.float64)
        log_probabilities, partial_score = raw_log_mass_condition_log_prob_and_score(
            flow,
            projected,
            tau,
            condition_center=domain.evidence.conditioning_center,
            condition_scale=domain.evidence.conditioning_scale,
        )
        _, observation_score = conditional_log_prob_and_observation_score(
            flow,
            projected,
            tau,
            condition_center=domain.evidence.conditioning_center,
            condition_scale=domain.evidence.conditioning_scale,
        )
        nll_rows.append(-np.asarray(log_probabilities, dtype=np.float64))
        partial_rows.append(
            np.square(np.asarray(partial_score, dtype=np.float64) - domain.mass_score_target[start:stop])
        )
        observation_rows.append(
            np.mean(
                np.square(
                    (
                        np.asarray(observation_score, dtype=np.float64)
                        - domain.observation_score_target[start:stop]
                    )
                    / np.asarray(observation_scales)[None, :]
                ),
                axis=1,
            )
        )
    nll = np.concatenate(nll_rows)
    partial = np.concatenate(partial_rows)
    observation = np.concatenate(observation_rows)
    nll_mean, nll_mcse = _mean_iid_mcse(nll)
    partial_mean, partial_mcse = _mean_iid_mcse(partial / partial_scale**2)
    observation_mean, observation_mcse = _mean_iid_mcse(observation)
    return {
        "iid_method": "independent PCG64 rows; ddof=1 standard deviation",
        "sample_count": int(nll.size),
        "nll_nat_per_draw": nll_mean,
        "nll_iid_mcse_nat_per_draw": nll_mcse,
        "nll_nat_per_dimension": nll_mean / domain.spectrum.retained_rank,
        "fisher_scaled_partial_score_risk": partial_mean,
        "fisher_scaled_partial_score_risk_iid_mcse": partial_mcse,
        "fisher_scaled_observation_score_risk": observation_mean,
        "fisher_scaled_observation_score_risk_iid_mcse": observation_mcse,
    }


def _vectorized_artifact_log_likelihood(
    artifact: ScoreRegularizedRootFlow,
    observation: np.ndarray,
    totals: np.ndarray,
) -> np.ndarray:
    residual = (
        observation[None, :] - totals[:, None] * artifact.spectrum.observation_mean_design[None, :]
    ) / artifact.spectrum.noise_sd[None, :]
    coordinates = residual @ artifact.spectrum.basis
    orthogonal = residual - coordinates @ artifact.spectrum.basis.T
    scales = np.sqrt(1.0 + totals[:, None] ** 2 * artifact.spectrum.eigenvalues[None, :])
    log_likelihood = -float(np.log(artifact.spectrum.noise_sd).sum()) - 0.5 * (
        (artifact.observation_count - artifact.retained_rank) * math.log(2.0 * math.pi)
        + np.sum(np.square(orthogonal), axis=1)
    )
    standardized = coordinates[:, : artifact.leading_rank] / scales[:, : artifact.leading_rank]
    conditions = ((np.log(totals) - artifact.condition_center) / artifact.condition_scale)[:, None]
    log_likelihood += np.asarray(
        artifact.flow.log_prob(
            jnp.asarray(standardized, dtype=jnp.float64),
            jnp.asarray(conditions, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )
    log_likelihood -= np.log(scales[:, : artifact.leading_rank]).sum(axis=1)
    if artifact.leading_rank < artifact.retained_rank:
        complement = coordinates[:, artifact.leading_rank :] / scales[:, artifact.leading_rank :]
        log_likelihood -= 0.5 * (
            (artifact.retained_rank - artifact.leading_rank) * math.log(2.0 * math.pi)
            + np.sum(np.square(complement), axis=1)
        )
        log_likelihood -= np.log(scales[:, artifact.leading_rank :]).sum(axis=1)
    if not np.all(np.isfinite(log_likelihood)):
        raise FloatingPointError("vectorized artifact likelihood is non-finite.")
    return log_likelihood


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, probability, side="left"))
    return float(values[order[min(index, values.size - 1)]])


def _logsumexp_scalar(values: np.ndarray) -> float:
    return float(cast(Any, special.logsumexp)(values))


def _posterior_summary(
    totals: np.ndarray,
    log_likelihood: np.ndarray,
    *,
    prior_cdf_bin_gamma: tuple[float, float] | None = None,
) -> dict[str, float]:
    log_weights = log_likelihood - _logsumexp_scalar(log_likelihood)
    weights = np.exp(log_weights)
    mean = float(weights @ totals)
    sd = math.sqrt(float(weights @ np.square(totals - mean)))
    if prior_cdf_bin_gamma is None:

        def quantile(probability: float) -> float:
            return _weighted_quantile(totals, weights, probability)

    else:
        gamma_shape, gamma_rate = prior_cdf_bin_gamma

        def quantile(probability: float) -> float:
            """Interpolate within the crossing equal-prior-probability bin."""
            cumulative = np.cumsum(weights)
            index = min(
                int(np.searchsorted(cumulative, probability, side="left")),
                weights.size - 1,
            )
            previous = 0.0 if index == 0 else float(cumulative[index - 1])
            fraction = (probability - previous) / float(weights[index])
            prior_probability = (index + min(max(fraction, 0.0), 1.0)) / weights.size
            return float(
                stats.gamma.ppf(
                    prior_probability,
                    a=gamma_shape,
                    scale=1.0 / gamma_rate,
                )
            )

    return {
        "log_evidence": (_logsumexp_scalar(log_likelihood) - math.log(log_likelihood.size)),
        "mean_total": mean,
        "sd_total": sd,
        "lower_0_025_total": quantile(0.025),
        "median_total": quantile(0.5),
        "upper_0_975_total": quantile(0.975),
    }


def _scientific_grid_evaluation(
    artifact: ScoreRegularizedRootFlow,
    case_id: str,
    oracle_reference: dict[str, Any],
    *,
    interpolate_posterior_quantiles: bool = False,
    exact_grid_cache: dict[int, np.ndarray] | None = None,
    promotion_oracle_case: dict[str, Any] | None = None,
) -> dict[str, object]:
    case = aggregation_error_tiny_oracle.tiny_root_case(case_id)
    shapes, rate, _, observation, _ = case.arrays()
    gamma_shape = float(shapes.sum())
    fraction_order = int(oracle_reference["fraction_order"])
    reference_sd = float(oracle_reference["posterior_sd_total"])
    expected_grid_rows: dict[int, dict[str, Any]] = {}
    gradient_preflight: dict[str, Any] | None = None
    oracle_certificate_valid = True
    if promotion_oracle_case is not None:
        if (
            promotion_oracle_case.get("reference") != oracle_reference
            or promotion_oracle_case.get("pass") is not True
        ):
            raise ValueError("promotion oracle case/reference is not passing.")
        metric_preflight = promotion_oracle_case.get("metric_grid_preflight")
        gradient_preflight = promotion_oracle_case.get("gradient_preflight")
        if (
            not isinstance(metric_preflight, dict)
            or metric_preflight.get("pass") is not True
            or not isinstance(gradient_preflight, dict)
            or gradient_preflight.get("pass") is not True
        ):
            raise ValueError("promotion metric/gradient preflight is not passing.")
        expected_grid_rows = {int(row["count"]): row for row in metric_preflight["rows"]}
        oracle_certificate_valid = True
    ladder: list[dict[str, Any]] = []
    final_totals: np.ndarray | None = None
    final_exact: np.ndarray | None = None
    final_learned: np.ndarray | None = None
    for count in GRID_COUNTS:
        probabilities = (np.arange(count, dtype=np.float64) + 0.5) / count
        totals = np.asarray(
            stats.gamma.ppf(
                probabilities,
                a=gamma_shape,
                scale=1.0 / rate,
            ),
            dtype=np.float64,
        )
        if exact_grid_cache is not None and count in exact_grid_cache:
            exact = np.asarray(
                exact_grid_cache[count],
                dtype=np.float64,
            )
            if exact.shape != totals.shape:
                raise ValueError("cached exact metric grid has the wrong shape.")
        else:
            exact = np.asarray(
                aggregation_error_tiny_oracle.root_conditional_log_likelihood(
                    case_id,
                    totals,
                    fraction_order=fraction_order,
                ),
                dtype=np.float64,
            )
            if exact_grid_cache is not None:
                exact_grid_cache[count] = exact
        expected_grid_row = expected_grid_rows.get(count)
        if expected_grid_row is not None:
            observed_total_sha256 = _sha256_bytes(np.ascontiguousarray(totals, dtype="<f8").tobytes())
            observed_exact_sha256 = _sha256_bytes(np.ascontiguousarray(exact, dtype="<f8").tobytes())
            if (
                expected_grid_row.get("total_grid_sha256") != observed_total_sha256
                or expected_grid_row.get("exact_log_likelihood_sha256") != observed_exact_sha256
            ):
                raise ValueError("scientific exact grid differs from the oracle preflight.")
        quantile_rule = (gamma_shape, rate) if interpolate_posterior_quantiles else None
        exact_summary = _posterior_summary(
            totals,
            exact,
            prior_cdf_bin_gamma=quantile_rule,
        )
        learned = _vectorized_artifact_log_likelihood(
            artifact,
            observation,
            totals,
        )
        learned_summary = _posterior_summary(
            totals,
            learned,
            prior_cdf_bin_gamma=quantile_rule,
        )
        exact_weights = np.exp(exact - _logsumexp_scalar(exact))
        error = np.abs(learned - exact)
        prior_weights = np.full(totals.size, 1.0 / totals.size)
        ladder.append(
            {
                "count": count,
                "exact_posterior": exact_summary,
                "learned_posterior": learned_summary,
                "exact_log_evidence_error_from_adaptive_reference_nat": abs(
                    exact_summary["log_evidence"] - float(oracle_reference["log_evidence"])
                ),
                "learned_log_evidence_error_from_adaptive_reference_nat": abs(
                    learned_summary["log_evidence"] - float(oracle_reference["log_evidence"])
                ),
                "exact_posterior_errors_from_adaptive_reference": {
                    "mean_error_reference_sd": abs(
                        exact_summary["mean_total"] - float(oracle_reference["posterior_mean_total"])
                    )
                    / reference_sd,
                    "sd_relative_error": abs(exact_summary["sd_total"] - reference_sd) / reference_sd,
                    "interval_endpoint_error_reference_sd": max(
                        abs(
                            exact_summary["lower_0_025_total"]
                            - float(oracle_reference["posterior_lower_0_025"])
                        ),
                        abs(
                            exact_summary["upper_0_975_total"]
                            - float(oracle_reference["posterior_upper_0_975"])
                        ),
                    )
                    / reference_sd,
                    "median_error_reference_sd": abs(
                        exact_summary["median_total"] - float(oracle_reference["posterior_median"])
                    )
                    / reference_sd,
                },
                "prior_weighted_median_absolute_log_likelihood_error_nat": (
                    _weighted_quantile(error, prior_weights, 0.5)
                ),
                "exact_posterior_weighted_p99_absolute_log_likelihood_error_nat": (
                    _weighted_quantile(error, exact_weights, 0.99)
                ),
            }
        )
        final_totals = totals
        final_exact = exact
        final_learned = learned
    if final_totals is None or final_exact is None or final_learned is None:  # pragma: no cover
        raise RuntimeError("scientific grid ladder is empty.")
    quantile_rule = (gamma_shape, rate) if interpolate_posterior_quantiles else None
    exact_summary = _posterior_summary(
        final_totals,
        final_exact,
        prior_cdf_bin_gamma=quantile_rule,
    )
    learned_summary = _posterior_summary(
        final_totals,
        final_learned,
        prior_cdf_bin_gamma=quantile_rule,
    )
    exact_weights = np.exp(final_exact - _logsumexp_scalar(final_exact))
    error = np.abs(final_learned - final_exact)
    prior_weights = np.full(final_totals.size, 1.0 / final_totals.size)
    parity_indices = np.unique(
        np.asarray(
            (
                0,
                1,
                final_totals.size // 4,
                final_totals.size // 2,
                3 * final_totals.size // 4,
                final_totals.size - 2,
                final_totals.size - 1,
            ),
            dtype=np.int64,
        )
    )
    public_values = np.asarray(
        artifact.log_likelihood_batch(
            observation,
            final_totals[parity_indices],
        ),
        dtype=np.float64,
    )
    parity_error = float(np.max(np.abs(public_values - final_learned[parity_indices])))
    if parity_error > 2.0e-10:
        raise RuntimeError("vectorized scientific likelihood differs from the public artifact.")
    mode_total = float(oracle_reference["posterior_mode_total"])
    mode_probability = float(
        stats.gamma.cdf(
            mode_total,
            a=gamma_shape,
            scale=1.0 / rate,
        )
    )
    mode_bin = min(
        int(math.floor(mode_probability * final_totals.size)),
        final_totals.size - 1,
    )
    mode_representative = float(final_totals[mode_bin])
    gradient_steps = (2.0**-12, 2.0**-13, 2.0**-14)

    def log_total_gradient(function: Any, step: float) -> float:
        return (
            float(function(mode_total * math.exp(step))) - float(function(mode_total * math.exp(-step)))
        ) / (2.0 * step)

    gradient_ladder = []
    for gradient_step in gradient_steps:
        exact_gradient_at_step = log_total_gradient(
            lambda total: aggregation_error_tiny_oracle.root_conditional_log_likelihood(
                case_id,
                total,
                fraction_order=fraction_order,
            ),
            gradient_step,
        )
        learned_gradient_at_step = log_total_gradient(
            lambda total: artifact.log_likelihood(observation, total),
            gradient_step,
        )
        gradient_ladder.append(
            {
                "log_total_step": gradient_step,
                "exact_log_total_gradient": exact_gradient_at_step,
                "learned_log_total_gradient": learned_gradient_at_step,
                "scaled_error": (
                    abs(learned_gradient_at_step - exact_gradient_at_step)
                    / (1.0 + abs(exact_gradient_at_step))
                ),
            }
        )
    exact_gradient = float(gradient_ladder[-1]["exact_log_total_gradient"])
    learned_gradient = float(gradient_ladder[-1]["learned_log_total_gradient"])
    gradient_scaled_error_delta = abs(
        float(gradient_ladder[-1]["scaled_error"]) - float(gradient_ladder[-2]["scaled_error"])
    )
    gradient_numerically_interpretable = True
    if gradient_preflight is not None:
        gradient_numerically_interpretable = (
            abs(exact_gradient - float(gradient_preflight["reference_gradient"]))
            <= 1.0e-12 * (1.0 + abs(exact_gradient))
        ) and gradient_scaled_error_delta <= 0.005
    previous = ladder[-2]
    final = ladder[-1]
    previous_exact = previous["exact_posterior"]
    final_exact_summary = final["exact_posterior"]
    previous_learned = previous["learned_posterior"]
    final_learned_summary = final["learned_posterior"]
    convergence = {
        "exact_log_evidence_delta_nat": abs(
            float(final_exact_summary["log_evidence"]) - float(previous_exact["log_evidence"])
        ),
        "learned_log_evidence_delta_nat": abs(
            float(final_learned_summary["log_evidence"]) - float(previous_learned["log_evidence"])
        ),
        "exact_posterior_mean_delta_reference_sd": abs(
            float(final_exact_summary["mean_total"]) - float(previous_exact["mean_total"])
        )
        / reference_sd,
        "learned_posterior_mean_delta_reference_sd": abs(
            float(final_learned_summary["mean_total"]) - float(previous_learned["mean_total"])
        )
        / reference_sd,
        "exact_posterior_sd_relative_delta": abs(
            float(final_exact_summary["sd_total"]) - float(previous_exact["sd_total"])
        )
        / reference_sd,
        "learned_posterior_sd_relative_delta": abs(
            float(final_learned_summary["sd_total"]) - float(previous_learned["sd_total"])
        )
        / reference_sd,
        "exact_posterior_endpoint_delta_reference_sd": max(
            abs(float(final_exact_summary["lower_0_025_total"]) - float(previous_exact["lower_0_025_total"])),
            abs(float(final_exact_summary["upper_0_975_total"]) - float(previous_exact["upper_0_975_total"])),
            abs(float(final_exact_summary["median_total"]) - float(previous_exact["median_total"])),
        )
        / reference_sd,
        "learned_posterior_endpoint_delta_reference_sd": max(
            abs(
                float(final_learned_summary["lower_0_025_total"])
                - float(previous_learned["lower_0_025_total"])
            ),
            abs(
                float(final_learned_summary["upper_0_975_total"])
                - float(previous_learned["upper_0_975_total"])
            ),
            abs(float(final_learned_summary["median_total"]) - float(previous_learned["median_total"])),
        )
        / reference_sd,
        "prior_median_error_delta_nat": abs(
            float(final["prior_weighted_median_absolute_log_likelihood_error_nat"])
            - float(previous["prior_weighted_median_absolute_log_likelihood_error_nat"])
        ),
        "posterior_p99_error_delta_nat": abs(
            float(final["exact_posterior_weighted_p99_absolute_log_likelihood_error_nat"])
            - float(previous["exact_posterior_weighted_p99_absolute_log_likelihood_error_nat"])
        ),
    }
    convergence_checks = {
        "exact_log_evidence": (convergence["exact_log_evidence_delta_nat"] <= GRID_EVIDENCE_TOLERANCE_NAT),
        "learned_log_evidence": (
            convergence["learned_log_evidence_delta_nat"] <= GRID_EVIDENCE_TOLERANCE_NAT
        ),
        "exact_posterior_mean": (
            convergence["exact_posterior_mean_delta_reference_sd"]
            <= GRID_POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "learned_posterior_mean": (
            convergence["learned_posterior_mean_delta_reference_sd"]
            <= GRID_POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "exact_posterior_sd": (
            convergence["exact_posterior_sd_relative_delta"] <= GRID_POSTERIOR_SD_RELATIVE_TOLERANCE
        ),
        "learned_posterior_sd": (
            convergence["learned_posterior_sd_relative_delta"] <= GRID_POSTERIOR_SD_RELATIVE_TOLERANCE
        ),
        "exact_posterior_endpoints": (
            convergence["exact_posterior_endpoint_delta_reference_sd"]
            <= GRID_POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "learned_posterior_endpoints": (
            convergence["learned_posterior_endpoint_delta_reference_sd"]
            <= GRID_POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "prior_median_error": (convergence["prior_median_error_delta_nat"] <= GRID_POINTWISE_TOLERANCE_NAT),
        "posterior_p99_error": (convergence["posterior_p99_error_delta_nat"] <= GRID_POINTWISE_TOLERANCE_NAT),
        "final_exact_evidence_matches_adaptive": (
            float(final["exact_log_evidence_error_from_adaptive_reference_nat"])
            <= GRID_EVIDENCE_TOLERANCE_NAT
        ),
        "final_exact_posterior_mean_matches_adaptive": (
            float(final["exact_posterior_errors_from_adaptive_reference"]["mean_error_reference_sd"])
            <= GRID_POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "final_exact_posterior_sd_matches_adaptive": (
            float(final["exact_posterior_errors_from_adaptive_reference"]["sd_relative_error"])
            <= GRID_POSTERIOR_SD_RELATIVE_TOLERANCE
        ),
        "final_exact_posterior_quantiles_match_adaptive": all(
            (
                float(
                    final["exact_posterior_errors_from_adaptive_reference"][
                        "interval_endpoint_error_reference_sd"
                    ]
                )
                <= GRID_POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD,
                float(final["exact_posterior_errors_from_adaptive_reference"]["median_error_reference_sd"])
                <= GRID_POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD,
            )
        ),
    }
    metric_interpretability = {
        "prior_weighted_median_log_likelihood_error": (convergence_checks["prior_median_error"]),
        "posterior_weighted_p99_log_likelihood_error": (convergence_checks["posterior_p99_error"]),
        "log_evidence": all(
            convergence_checks[key]
            for key in (
                "exact_log_evidence",
                "learned_log_evidence",
                "final_exact_evidence_matches_adaptive",
            )
        ),
        "posterior_moments": all(
            convergence_checks[key]
            for key in (
                "exact_posterior_mean",
                "learned_posterior_mean",
                "exact_posterior_sd",
                "learned_posterior_sd",
                "final_exact_posterior_mean_matches_adaptive",
                "final_exact_posterior_sd_matches_adaptive",
            )
        ),
        "posterior_quantiles": all(
            convergence_checks[key]
            for key in (
                "exact_posterior_endpoints",
                "learned_posterior_endpoints",
                "final_exact_posterior_quantiles_match_adaptive",
            )
        ),
        "retained_mass_gradient": gradient_numerically_interpretable,
    }
    grid_converged = all(metric_interpretability.values())
    exact_grid_errors = final["exact_posterior_errors_from_adaptive_reference"]
    learned_posterior_errors = {
        "mean_error_reference_sd": abs(
            learned_summary["mean_total"] - float(oracle_reference["posterior_mean_total"])
        )
        / reference_sd,
        "sd_relative_error": abs(learned_summary["sd_total"] - reference_sd) / reference_sd,
        "interval_endpoint_error_reference_sd": max(
            abs(learned_summary["lower_0_025_total"] - float(oracle_reference["posterior_lower_0_025"])),
            abs(learned_summary["upper_0_975_total"] - float(oracle_reference["posterior_upper_0_975"])),
        )
        / reference_sd,
        "median_error_reference_sd": abs(
            learned_summary["median_total"] - float(oracle_reference["posterior_median"])
        )
        / reference_sd,
    }
    return {
        "grid": {
            "construction": "midpoint prior-CDF strata",
            "posterior_quantile_rule": (
                "within-bin interpolation under piecewise-constant likelihood"
                if interpolate_posterior_quantiles
                else "crossing-bin midpoint"
            ),
            "tail_bin_convention": (
                "first and last midpoint represent their complete prior "
                "probability bins; no conditional subset renormalization"
            ),
            "ladder": ladder,
            "last_two_convergence": convergence,
            "convergence_checks": convergence_checks,
            "grid_converged": grid_converged,
            "finite_grid_rows_retained_fraction": 1.0,
            "continuous_support_from_adaptive_reference": {
                "represented_prior_mass": float(oracle_reference["represented_prior_mass"]),
                "represented_posterior_mass_lower_bound": float(
                    oracle_reference["represented_posterior_mass"]
                ),
                "posterior_mode_in_continuous_support": bool(oracle_reference["mode_included"]),
                "posterior_mode_prior_probability_bin": mode_bin,
                "posterior_mode_bin_representative_total": (mode_representative),
                "posterior_mode_bin_representative_distance_reference_sd": (
                    abs(mode_representative - mode_total) / reference_sd
                ),
            },
        },
        "pointwise": {
            "prior_weighted_median_absolute_log_likelihood_error_nat": (
                _weighted_quantile(error, prior_weights, 0.5)
            ),
            "exact_posterior_weighted_p99_absolute_log_likelihood_error_nat": (
                _weighted_quantile(error, exact_weights, 0.99)
            ),
            "maximum_absolute_log_likelihood_error_nat": float(np.max(error)),
        },
        "evidence": {
            "adaptive_exact_log_evidence": float(oracle_reference["log_evidence"]),
            "grid_exact_log_evidence": exact_summary["log_evidence"],
            "grid_learned_log_evidence": learned_summary["log_evidence"],
            "absolute_learned_error_from_adaptive_reference_nat": abs(
                learned_summary["log_evidence"] - float(oracle_reference["log_evidence"])
            ),
            "absolute_exact_grid_error_from_adaptive_reference_nat": abs(
                exact_summary["log_evidence"] - float(oracle_reference["log_evidence"])
            ),
        },
        "posterior": {
            "grid_exact": exact_summary,
            "grid_learned": learned_summary,
            "exact_grid_errors_from_adaptive_reference": exact_grid_errors,
            **learned_posterior_errors,
        },
        "gradient": {
            "state": "adaptive exact posterior mode total",
            "total_mass": mode_total,
            "exact_log_total_gradient": exact_gradient,
            "learned_log_total_gradient": learned_gradient,
            "scaled_error": abs(learned_gradient - exact_gradient) / (1.0 + abs(exact_gradient)),
            "step_ladder": gradient_ladder,
            "last_two_scaled_error_delta": gradient_scaled_error_delta,
            "numerical_tolerance": 0.005,
        },
        "normalization": {
            "normalized_density_by_flow_and_jacobian_construction": True,
            "finite_on_complete_metric_grid": bool(np.all(np.isfinite(final_learned))),
        },
        "scientific_metrics_interpretable": grid_converged,
        "metric_interpretability": metric_interpretability,
        "oracle_certificate_valid": oracle_certificate_valid,
        "vectorized_public_likelihood_parity": {
            "evaluated_indices": parity_indices.tolist(),
            "maximum_absolute_error_nat": parity_error,
            "pass": True,
        },
        "learned_log_likelihood_sha256": _sha256_bytes(
            np.ascontiguousarray(final_learned, dtype="<f8").tobytes()
        ),
    }


def _load_oracle_bundle(
    path: Path,
    source_git_revision: str,
    *,
    promotion: bool = False,
) -> dict[str, Any]:
    report_bytes = path.read_bytes()
    payload = json.loads(report_bytes.decode("ascii"))
    expected_schema = PROMOTION_ORACLE_BUNDLE_SCHEMA if promotion else LEGACY_ORACLE_BUNDLE_SCHEMA
    expected_cases = PROMOTION_CASES if promotion else SELECTED_CASES
    if payload.get("schema") != expected_schema:
        raise ValueError("oracle bundle has the wrong schema.")
    if payload.get("source_git_revision") != source_git_revision:
        raise ValueError("oracle bundle Git revision differs from the attempt.")
    if payload.get("tiny_root_definitions_sha256") != (aggregation_error_tiny_oracle.definitions_sha256()):
        raise ValueError("oracle bundle tiny definitions differ.")
    without_sha = dict(payload)
    observed_sha = without_sha.pop("sha256", None)
    if _sha256_json(without_sha) != observed_sha:
        raise ValueError("oracle bundle SHA-256 does not replay.")
    if not payload.get("pass"):
        raise ValueError("oracle bundle is not passing.")
    completion_path = path.parent / "COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="ascii"))
    if completion != {
        "schema": expected_schema,
        "source_git_revision": source_git_revision,
        "report_path": str(path),
        "oracle_bundle_payload_sha256": observed_sha,
        "oracle_bundle_file_sha256": _sha256_bytes(report_bytes),
        "completion_marker_published_last": True,
    }:
        raise ValueError("oracle completion marker does not bind the bundle.")
    selected = payload.get("selected_cases")
    if not isinstance(selected, dict) or set(selected) != set(expected_cases):
        raise ValueError("oracle bundle selected-case catalogue is incomplete.")
    for case_id, raw_case in selected.items():
        if not isinstance(raw_case, dict) or raw_case.get("pass") is not True:
            raise ValueError("oracle case is malformed or not passing.")
        ladder = raw_case.get("order_ladder")
        reference = raw_case.get("reference")
        if (
            not isinstance(ladder, list)
            or len(ladder) < 2
            or not isinstance(reference, dict)
            or reference != ladder[-1]
        ):
            raise ValueError("oracle case ladder/reference is malformed.")
        for raw_summary in ladder:
            if not isinstance(raw_summary, dict):
                raise ValueError("oracle summary must be a mapping.")
            summary_without_sha = dict(raw_summary)
            summary_sha = summary_without_sha.pop("sha256", None)
            if (
                raw_summary.get("schema") != aggregation_error_tiny_oracle.SCHEMA
                or raw_summary.get("case_id") != case_id
                or raw_summary.get("definitions_sha256") != aggregation_error_tiny_oracle.definitions_sha256()
                or _sha256_json(summary_without_sha) != summary_sha
            ):
                raise ValueError("nested oracle summary identity does not replay.")
        if promotion:
            observed_orders = tuple(int(raw_summary["fraction_order"]) for raw_summary in ladder)
            if observed_orders != PROMOTION_ORACLE_ORDER_LADDERS[case_id]:
                raise ValueError("promotion oracle uses the wrong allocation-order ladder.")
            grid = raw_case.get("metric_grid_preflight")
            if not isinstance(grid, dict) or grid.get("pass") is not True:
                raise ValueError("promotion metric-grid preflight is absent or failing.")
            grid_without_sha = dict(grid)
            grid_sha = grid_without_sha.pop("sha256", None)
            grid_rows = grid.get("rows")
            if (
                grid.get("case_id") != case_id
                or grid.get("counts") != list(GRID_COUNTS[-2:])
                or not isinstance(grid_rows, list)
                or [row.get("count") for row in grid_rows if isinstance(row, dict)] != list(GRID_COUNTS[-2:])
                or not all(isinstance(row, dict) and row.get("finite") is True for row in grid_rows)
                or _sha256_json(grid_without_sha) != grid_sha
                or not grid.get("checks")
                or not all(grid.get("checks", {}).values())
            ):
                raise ValueError("promotion metric-grid preflight identity differs.")
            gradient = raw_case.get("gradient_preflight")
            if not isinstance(gradient, dict) or gradient.get("pass") is not True:
                raise ValueError("promotion gradient preflight is absent or failing.")
            gradient_without_sha = dict(gradient)
            gradient_sha = gradient_without_sha.pop("sha256", None)
            if (
                gradient.get("case_id") != case_id
                or gradient.get("final_fraction_order") != PROMOTION_ORACLE_ORDER_LADDERS[case_id][-1]
                or gradient.get("previous_fraction_order") != PROMOTION_ORACLE_ORDER_LADDERS[case_id][-2]
                or [row.get("log_total_step") for row in gradient.get("final_order_step_ladder", [])]
                != [2.0**-12, 2.0**-13, 2.0**-14]
                or _sha256_json(gradient_without_sha) != gradient_sha
                or not gradient.get("checks")
                or not all(gradient.get("checks", {}).values())
            ):
                raise ValueError("promotion gradient preflight identity differs.")
    boundary = payload.get("boundary_independent_certificate")
    if not isinstance(boundary, dict) or boundary.get("pass") is not True:
        raise ValueError("boundary independent certificate is absent or failing.")
    boundary_without_sha = dict(boundary)
    boundary_sha = boundary_without_sha.pop("sha256", None)
    if _sha256_json(boundary_without_sha) != boundary_sha:
        raise ValueError("boundary certificate SHA-256 does not replay.")
    if promotion:
        certificates = payload.get("independent_certificates")
        expected_certificate_cases = {
            "near_gaussian__four_cell__root",
            "skewed__two_cell__root",
            "skewed__four_cell__root",
            "boundary_heavy__four_cell__root",
        }
        if not isinstance(certificates, dict) or set(certificates) != expected_certificate_cases:
            raise ValueError("promotion independent-certificate catalogue differs.")
        for case_id, certificate in certificates.items():
            if not isinstance(certificate, dict) or certificate.get("pass") is not True:
                raise ValueError("promotion independent certificate is failing.")
            certificate_without_sha = dict(certificate)
            certificate_sha = certificate_without_sha.pop("sha256", None)
            if _sha256_json(certificate_without_sha) != certificate_sha:
                raise ValueError("promotion independent certificate SHA-256 differs.")
            selected_reference = selected[case_id]["reference"]
            if (
                certificate.get("case_id") != case_id
                or not certificate.get("checks")
                or not all(certificate.get("checks", {}).values())
            ):
                raise ValueError("promotion independent certificate semantics differ.")
            if case_id.endswith("__four_cell__root"):
                column = certificate.get("column_summary")
                if not isinstance(column, dict):
                    raise ValueError("four-cell column-chart summary is absent.")
                column_without_sha = dict(column)
                column_sha = column_without_sha.pop("sha256", None)
                if (
                    certificate.get("fraction_order") != selected_reference["fraction_order"]
                    or certificate.get("row_reference_sha256") != selected_reference["sha256"]
                    or column.get("schema") != aggregation_error_tiny_oracle.SCHEMA
                    or column.get("definitions_sha256") != aggregation_error_tiny_oracle.definitions_sha256()
                    or column.get("case_id") != case_id
                    or column.get("fraction_order") != selected_reference["fraction_order"]
                    or "column_first" not in str(column.get("method"))
                    or _sha256_json(column_without_sha) != column_sha
                ):
                    raise ValueError("four-cell column-chart summary identity differs.")
            else:
                summaries = certificate.get("summaries")
                if (
                    case_id != "skewed__two_cell__root"
                    or certificate.get("lower_log_mass_ladder") != [-80.0, -120.0]
                    or not isinstance(summaries, list)
                    or len(summaries) != 2
                ):
                    raise ValueError("skewed native-log-mass certificate differs.")
                for summary_index, summary in enumerate(summaries):
                    summary_without_sha = dict(summary)
                    summary_sha = summary_without_sha.pop("sha256", None)
                    if (
                        summary.get("schema") != aggregation_error_tiny_oracle.SCHEMA
                        or summary.get("definitions_sha256")
                        != aggregation_error_tiny_oracle.definitions_sha256()
                        or summary.get("case_id") != "skewed__two_cell__root"
                        or summary.get("method") != "adaptive_native_two_dimensional_log_masses"
                        or summary.get("lower_log_mass")
                        != certificate["lower_log_mass_ladder"][summary_index]
                        or _sha256_json(summary_without_sha) != summary_sha
                    ):
                        raise ValueError("native-log-mass summary SHA-256 differs.")
        boundary_four = certificates["boundary_heavy__four_cell__root"].get("fixed_log_total_column_chart")
        if (
            not isinstance(boundary_four, dict)
            or boundary_four.get("method") != "fixed Gauss-Legendre in log(total)"
            or boundary_four.get("chart") != "column-first"
            or boundary_four.get("fraction_order") != 24
            or boundary_four.get("prior_tail_probability") != 1.0e-15
            or boundary_four.get("total_order_ladder") != [512, 1024, 2048]
            or boundary_four.get("pass") is not True
            or not boundary_four.get("checks")
            or not all(boundary_four.get("checks", {}).values())
        ):
            raise ValueError("boundary four-cell fixed-log-total certificate differs.")
        checks = payload.get("checks")
        if not isinstance(checks, dict) or not checks or not all(checks.values()):
            raise ValueError("promotion oracle top-level checks are not all passing.")
        from examples.rjmcmc import (
            score_regularized_flow_corrected_oracle as promotion_oracle,
        )

        promotion_oracle.validate_bundle_semantics(payload)
    return payload


def _runtime_versions() -> dict[str, str]:
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


def _runtime_identity() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    lock_path = repository_root / "pixi.lock"
    if not lock_path.is_file() or lock_path.is_symlink():
        raise ValueError("runtime lock file must be a regular file.")
    without_sha: dict[str, Any] = {
        "schema": "rjmcmc-score-nle-runtime-identity-v1",
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "packages": _runtime_versions(),
        "pixi_lock_sha256": _sha256_bytes(lock_path.read_bytes()),
    }
    return {**without_sha, "sha256": _sha256_json(without_sha)}


def _execution_identity(*, promotion: bool) -> dict[str, Any]:
    observed_environment = {key: os.environ.get(key) for key in PROMOTION_EXECUTION_ENVIRONMENT}
    if promotion and observed_environment != PROMOTION_EXECUTION_ENVIRONMENT:
        raise ValueError("promotion compiler/thread environment differs from the frozen identity.")
    without_sha: dict[str, Any] = {
        "schema": "rjmcmc-score-nle-execution-identity-v1",
        "environment": observed_environment,
        "jax_x64_enabled": bool(cast(Any, jax.config).x64_enabled),
        "autodiff_schedule": {
            "observation_score_parameter_gradient": ("outer_reverse_parameter_gradient_over_forward_jvp"),
            "partial_score_parameter_gradient": ("outer_reverse_parameter_gradient_over_forward_jvp"),
        },
        "artifact_evaluation_policy": "canonical_deserialized_artifact",
        "scheduler_policy": "shared_node_nonexclusive_single_cpu_array_task",
        "promotion_array_required": promotion,
    }
    return {**without_sha, "sha256": _sha256_json(without_sha)}


def run_attempt(arguments: argparse.Namespace, attempt_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if arguments.mode == "standard":
        if arguments.case_id not in SELECTED_CASES:
            raise ValueError("standard mode case is not in the E2 catalogue.")
        if arguments.sample_count not in STANDARD_SAMPLE_COUNTS:
            raise ValueError("standard mode sample count is not in the E2 ladder.")
    elif arguments.mode == "promotion":
        if arguments.case_id not in PROMOTION_CASES:
            raise ValueError("promotion mode case is not in the all-six catalogue.")
        if arguments.sample_count not in PROMOTION_SAMPLE_COUNTS:
            raise ValueError("promotion mode sample count is not in the frozen ladder.")
    elif arguments.mode == "overfit":
        if arguments.case_id not in OVERFIT_CASES:
            raise ValueError("overfit mode case is not in the E2 catalogue.")
        if arguments.sample_count != OVERFIT_SAMPLE_COUNT:
            raise ValueError("overfit mode requires the frozen small catalogue.")
    else:
        raise ValueError("mode must be standard, promotion, or overfit.")
    if arguments.config_id not in CONFIG_IDS:
        raise ValueError("config_id is not in the corrected E2 catalogue.")
    if arguments.init_index < 0:
        raise ValueError("init_index must be non-negative.")
    if not 0 <= arguments.base_seed < 2**64:
        raise ValueError("base_seed must be unsigned 64-bit.")
    frozen_matrix_row = (
        arguments.mode,
        arguments.case_id,
        arguments.sample_count,
        arguments.config_id,
        arguments.init_index,
    )
    matrix_id = str(getattr(arguments, "matrix_id", "direct"))
    matrix_task_id = getattr(arguments, "matrix_task_id", None)
    matrix_task_count = getattr(arguments, "matrix_task_count", None)
    declared_matrix_row = tuple(getattr(arguments, "matrix_row", frozen_matrix_row))
    if declared_matrix_row != frozen_matrix_row:
        raise ValueError("declared matrix row differs from attempt arguments.")
    legacy_matrix_identity = {
        "matrix_id": matrix_id,
        "array_task_id": matrix_task_id,
        "array_task_count": matrix_task_count,
        "row": list(frozen_matrix_row),
    }
    declared_matrix_identity = getattr(arguments, "matrix_identity", None)
    if declared_matrix_identity is None:
        matrix_identity = legacy_matrix_identity
    else:
        matrix_identity = cast(dict[str, Any], declared_matrix_identity)
        if matrix_identity.get("matrix_id") != matrix_id:
            raise ValueError("declared matrix identity has the wrong matrix.")
        if matrix_identity.get("array_task_id") != matrix_task_id:
            raise ValueError("declared matrix identity has the wrong task.")
        if matrix_identity.get("array_task_count") != matrix_task_count:
            raise ValueError("declared matrix identity has the wrong task count.")
        if matrix_identity.get("row") != list(frozen_matrix_row):
            raise ValueError("declared matrix identity has the wrong row.")
    runtime_identity = _runtime_identity()
    execution_identity = _execution_identity(
        promotion=arguments.mode == "promotion",
    )
    oracle_bundle = _load_oracle_bundle(
        arguments.oracle_bundle,
        arguments.source_git_revision,
        promotion=arguments.mode == "promotion",
    )
    constructed = {
        domain: domains.simulate_tiny_score_domain(
            arguments.case_id,
            domain=domain,
            sample_count=arguments.sample_count,
            base_seed=arguments.base_seed,
        )
        for domain in domains.PUBLIC_DOMAINS
    }
    training = constructed[domains.TRAINING_DOMAIN]
    validation = constructed[domains.MODEL_SELECTION_VALIDATION_DOMAIN]
    reporting = constructed[domains.DEVELOPMENT_REPORTING_TEST_DOMAIN]
    scale_diagnostics = cast(
        dict[str, Any],
        loss_scale_diagnostics(
            training.mass_score_target,
            training.observation_score_target,
        ),
    )
    dimension = training.spectrum.retained_rank
    stage_plan = _stage_plan(arguments.config_id, arguments.max_epochs)
    initialization_stream, optimizer_stream_plan = _private_stream_plan(
        arguments.base_seed,
        case_id=arguments.case_id,
        init_index=arguments.init_index,
        stage_count=len(stage_plan),
    )
    stream_separation = _validate_private_stream_plan(
        initialization_stream,
        optimizer_stream_plan,
        constructed,
    )
    initialization_jax_seed = int(initialization_stream["derived_jax_seed"])
    model = make_score_regularized_conditional_flow(
        dimension,
        source_seed=initialization_jax_seed,
    )
    partial_scale = float(scale_diagnostics["partial_score_rms"])
    observation_scales = tuple(
        float(value) for value in scale_diagnostics["observation_score_rms_by_coordinate"]
    )
    initialization_diagnostics = initialization_loss_diagnostics(
        model,
        _data(training),
        condition_center=training.evidence.conditioning_center,
        condition_scale=training.evidence.conditioning_scale,
        partial_score_scale=partial_scale,
        observation_score_scales=observation_scales,
    )
    loss_config = _loss_config(
        arguments.config_id,
        scale_diagnostics,
        dimension=dimension,
    )
    histories: list[dict[str, object]] = []
    optimizer_streams: list[dict[str, object]] = []
    validation_data = _data(training) if arguments.mode == "overfit" else _data(validation)

    def fit_stage(
        stage_id: str,
        stage_index: int,
        current_model: Any,
        config: ExplorationLossConfig,
        epochs: int,
    ) -> Any:
        stream = optimizer_stream_plan[stage_index]
        jax_seed = int(stream["derived_jax_seed"])
        fitted, history = fit_exploratory_score_flow(
            jr.key(jax_seed),
            current_model,
            _data(training),
            validation_data,
            condition_center=training.evidence.conditioning_center,
            condition_scale=training.evidence.conditioning_scale,
            loss_config=config,
            learning_rate=arguments.learning_rate,
            batch_size=arguments.batch_size,
            microbatch_size=arguments.microbatch_size,
            max_epochs=epochs,
            patience=arguments.patience,
        )
        optimizer_streams.append(
            {
                **stream,
                "stage_id": stage_id,
                "stage_index": stage_index,
                "jax_internal_generator": "JAX PRNG seeded from PCG64",
            }
        )
        histories.append(
            {
                "stage_id": stage_id,
                "loss_config": config.payload(),
                "history": _history_payload(history),
            }
        )
        return fitted

    if len(stage_plan) == 2:
        model = fit_stage(
            stage_plan[0][0],
            0,
            model,
            _nll_config(scale_diagnostics, dimension=dimension),
            stage_plan[0][1],
        )
        model = fit_stage(
            stage_plan[1][0],
            1,
            model,
            loss_config,
            stage_plan[1][1],
        )
    else:
        model = fit_stage(
            stage_plan[0][0],
            0,
            model,
            loss_config,
            stage_plan[0][1],
        )
    trained_artifact = ScoreRegularizedRootFlow(
        training.spectrum,
        dimension,
        training.evidence.gamma_shape,
        training.evidence.gamma_rate,
        model,
        conditioning_rule_id=GAMMA_LOG_MASS_CONDITIONING_RULE,
        initialization_seed=initialization_jax_seed,
        source_provenance=(
            f"{PROTOCOL}:{arguments.case_id}:S={arguments.sample_count}:"
            f"config={arguments.config_id}:init={arguments.init_index}:"
            f"git={arguments.source_git_revision}"
        ),
    )
    artifact_bytes = trained_artifact.to_bytes()
    serialized_sha256 = _sha256_bytes(artifact_bytes)
    artifact = ScoreRegularizedRootFlow.from_bytes(
        artifact_bytes,
        expected_sha256=serialized_sha256,
    )
    if artifact.to_bytes() != artifact_bytes:
        raise RuntimeError("authenticated artifact did not replay exact bytes.")
    oracle_case = cast(
        dict[str, Any],
        oracle_bundle["selected_cases"][arguments.case_id],
    )
    case = aggregation_error_tiny_oracle.tiny_root_case(arguments.case_id)
    _, _, _, observation, _ = case.arrays()
    replay_totals = np.asarray((0.5, 1.0, 1.5), dtype=np.float64)
    replay_diagnostics = _canonical_replay_diagnostics(
        trained_artifact,
        artifact,
        observation,
        replay_totals,
    )
    model_selection = _score_domain(
        artifact.flow,
        validation,
        partial_scale=float(scale_diagnostics["partial_score_rms"]),
        observation_scales=tuple(
            float(value) for value in scale_diagnostics["observation_score_rms_by_coordinate"]
        ),
    )
    if arguments.mode == "promotion":
        reporting_test: dict[str, object] = {
            "status": "deferred_until_model_selection_only_start_selection",
            "selection_uses_reporting_data": False,
        }
        scientific: dict[str, object] = {
            "status": "deferred_until_model_selection_only_start_selection",
            "selection_uses_oracle_evidence": False,
        }
    else:
        reporting_test = _score_domain(
            artifact.flow,
            reporting,
            partial_scale=float(scale_diagnostics["partial_score_rms"]),
            observation_scales=tuple(
                float(value) for value in scale_diagnostics["observation_score_rms_by_coordinate"]
            ),
        )
        scientific = _scientific_grid_evaluation(
            artifact,
            arguments.case_id,
            oracle_case["reference"],
        )
    candidate_payload = {
        "config_id": arguments.config_id,
        "architecture_rank": dimension,
        "architecture_family": ("q1_masked_autoregressive" if dimension == 1 else "q3_coupling"),
        "loss_rule": _candidate_loss_rule(arguments.config_id),
        "learning_rate": arguments.learning_rate,
        "batch_size": arguments.batch_size,
        "microbatch_size": arguments.microbatch_size,
        "maximum_total_epochs": arguments.max_epochs,
        "patience": arguments.patience,
        "pretrain_optimizer_state_policy": (
            "reset_between_stages" if arguments.config_id == "nll_pretrain_then_partial" else "single_stage"
        ),
    }
    target_payload = {
        "case_id": arguments.case_id,
        "sample_count": arguments.sample_count,
        "mode": arguments.mode,
        "tiny_root_definitions_sha256": (aggregation_error_tiny_oracle.definitions_sha256()),
        "spectrum_sha256": training.evidence.spectrum_sha256,
        "scientific_input_sha256": (training.evidence.scientific_input_sha256),
        "oracle_reference_sha256": oracle_case["reference"]["sha256"],
    }
    scientific_target_payload = {
        key: target_payload[key]
        for key in (
            "case_id",
            "tiny_root_definitions_sha256",
            "spectrum_sha256",
            "scientific_input_sha256",
            "oracle_reference_sha256",
        )
    }
    catalogue_identity_payload = {
        "mode": arguments.mode,
        "sample_count": arguments.sample_count,
        "domain_evidence_sha256": {name: value.evidence.sha256 for name, value in constructed.items()},
    }
    attempt_identity = {
        "schema": SCHEMA,
        "source_git_revision": arguments.source_git_revision,
        "target": target_payload,
        "scientific_target_sha256": _sha256_json(scientific_target_payload),
        "catalogue_identity_sha256": _sha256_json(catalogue_identity_payload),
        "candidate_sha256": _sha256_json(candidate_payload),
        "domain_evidence_sha256": catalogue_identity_payload["domain_evidence_sha256"],
        "runtime_identity_sha256": runtime_identity["sha256"],
        "execution_identity_sha256": execution_identity["sha256"],
        "initialization_index": arguments.init_index,
        "attempt_tag": arguments.attempt_tag,
        "matrix_identity": matrix_identity,
    }
    attempt_id = _sha256_json(attempt_identity)
    report_without_sha: dict[str, Any] = {
        **attempt_identity,
        "attempt_id": attempt_id,
        "candidate": candidate_payload,
        "scientific_target": scientific_target_payload,
        "catalogue_identity": catalogue_identity_payload,
        "runtime_identity": runtime_identity,
        "execution_identity": execution_identity,
        "streams": {
            "flow_initialization": {
                **initialization_stream,
                "jax_internal_generator": "JAX PRNG seeded from PCG64",
            },
            "optimizer_stages": optimizer_streams,
            "separation": stream_separation,
        },
        "domain_evidence": {name: value.evidence.payload() for name, value in constructed.items()},
        "training_loss_scale_diagnostics": scale_diagnostics,
        "initialization_loss_diagnostics": initialization_diagnostics,
        "stages": histories,
        "model_selection": model_selection,
        "reporting_test": reporting_test,
        "scientific_evaluation": scientific,
        "artifact": {
            "artifact_sha256": artifact.artifact_sha256,
            "serialized_sha256": serialized_sha256,
            "serialized_artifact_file_sha256": serialized_sha256,
            "serialized_size_bytes": len(artifact_bytes),
            "byte_replay_pass": (artifact.artifact_sha256 == trained_artifact.artifact_sha256),
            "canonical_replay": replay_diagnostics,
        },
        "execution": {
            "runtime_identity_sha256": runtime_identity["sha256"],
            "execution_identity_sha256": execution_identity["sha256"],
            "runtime_seconds": time.perf_counter() - started,
            "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "output_root": str(attempt_root),
        },
        "interpretation": {
            "status": (
                "promotion_attempt_pending_independent_merger"
                if arguments.mode == "promotion"
                else "exploratory_result_not_promotion"
            ),
            "overfit_validation_role": (
                "same catalogue optimizer diagnostic" if arguments.mode == "overfit" else None
            ),
            "approximate_evidence_is_structural_information": False,
            "protected_or_paris_inputs_opened": False,
            "direct_fixed_raw_observation_score_ablation": (
                "not run: existing helper holds projected residual fixed and "
                "does not provide the required raw-observation mean chain term"
            ),
        },
    }
    report = {
        **report_without_sha,
        "sha256": _sha256_json(report_without_sha),
    }
    report_bytes = _json_bytes(report)
    report_file_sha256 = _sha256_bytes(report_bytes)
    _atomic_bytes(attempt_root / "artifact.bin", artifact_bytes)
    _atomic_bytes(attempt_root / "report.json", report_bytes)
    completion_payload: dict[str, Any] = {
        "schema": SCHEMA,
        "attempt_id": attempt_id,
        "source_git_revision": arguments.source_git_revision,
        "report": str(attempt_root / "report.json"),
        "report_payload_sha256": report["sha256"],
        "report_file_sha256": report_file_sha256,
        "artifact_metadata_sha256": artifact.artifact_sha256,
        "serialized_artifact_file_sha256": serialized_sha256,
        "completion_marker_published_last": True,
    }
    if arguments.mode == "promotion":
        completion_payload.update(
            {
                "runtime_identity_sha256": runtime_identity["sha256"],
                "execution_identity_sha256": execution_identity["sha256"],
            }
        )
    _atomic_json(
        attempt_root / "COMPLETE.json",
        completion_payload,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("standard", "promotion", "overfit"), required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--config-id", choices=CONFIG_IDS, required=True)
    parser.add_argument("--init-index", type=int, required=True)
    parser.add_argument("--attempt-tag", required=True)
    parser.add_argument("--base-seed", type=int, default=731)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--oracle-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    arguments = parser.parse_args()
    if len(arguments.source_git_revision) != 40 or any(
        character not in "0123456789abcdef" for character in arguments.source_git_revision
    ):
        raise ValueError("source_git_revision must be a full lower-case Git SHA.")
    slug = (
        f"{arguments.mode}__{arguments.case_id}__S{arguments.sample_count}"
        f"__{arguments.config_id}__init{arguments.init_index}"
        f"__{arguments.attempt_tag}"
    )
    attempt_root = arguments.output_root / "attempts" / slug
    if attempt_root.exists():
        raise FileExistsError("attempt root already exists; use a new attempt tag to preserve it.")
    attempt_root.mkdir(parents=True)
    try:
        report = run_attempt(arguments, attempt_root)
    except Exception as error:
        _atomic_json(
            attempt_root / "FAILURE.json",
            {
                "schema": SCHEMA,
                "source_git_revision": arguments.source_git_revision,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "completion_marker_published": False,
            },
        )
        raise
    print(
        _canonical_json(
            {
                "attempt_id": report["attempt_id"],
                "report": str(attempt_root / "report.json"),
                "sha256": report["sha256"],
            },
            pretty=False,
        )
    )


if __name__ == "__main__":
    main()
