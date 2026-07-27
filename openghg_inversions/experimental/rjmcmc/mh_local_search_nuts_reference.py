"""NumPyro NUTS reference helpers for the bounded synthetic local-search screen.

This module deliberately supports only the five representative fixed-topology
cells predeclared for each synthetic experiment stage.  It reuses
the existing :mod:`fixed_basis_nuts` model and sampler without changing their
public APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from .fixed_basis_nuts import (
    FixedBasisNUTSData,
    build_fixed_basis_pymc_model,
    preflight_fixed_basis_nuts,
)
from .full_tiling import TilingState
from .full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    build_full_tiling_posterior_state,
)
from .mh_local_search_synthetic import (
    SyntheticEvaluationArtifact,
    SyntheticTrainingArtifact,
    common_native_totals,
    prepare_fixed_basis_reference,
    reconstruct_native_fields,
    validate_artifact_pair,
)

if TYPE_CHECKING:
    from arviz import InferenceData
    from pymc import Model

FloatArray: TypeAlias = NDArray[np.float64]
TopologyRole = Literal["p0", "pstar"]
ProfileName = Literal["primary", "retry1"]

REFERENCE_CELLS = (
    "aligned-p0",
    "edge-one-p0",
    "edge-one-pstar",
    "relocation-one-p0",
    "relocation-one-pstar",
)
PROJECTION_NAMES = (
    "whole_domain",
    "top_half",
    "bottom_half",
    "left_half",
    "right_half",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
)
_START_SEEDS = {
    "s0": (None, 64101, 64102, 64103),
    "s1": (None, 74101, 74102, 74103),
}
_SAMPLER_SEEDS = {"s0": 64100, "s1": 74100}
# Backwards-compatible S0 constants used by archived S0 validation.
START_SEEDS = _START_SEEDS["s0"]
SAMPLER_SEED = _SAMPLER_SEEDS["s0"]


@dataclass(frozen=True, slots=True)
class NUTSReferenceProfile:
    """One frozen NumPyro NUTS execution profile."""

    name: ProfileName
    tune: int
    draws: int
    target_accept: float
    max_tree_depth: int
    dense_mass: bool = False


PRIMARY_PROFILE = NUTSReferenceProfile(
    name="primary",
    tune=1_000,
    draws=1_000,
    target_accept=0.90,
    max_tree_depth=10,
)
RETRY_PROFILE = NUTSReferenceProfile(
    name="retry1",
    tune=2_000,
    draws=2_000,
    target_accept=0.95,
    max_tree_depth=12,
)


@dataclass(frozen=True, slots=True)
class NUTSReferenceStart:
    """One audited constrained start and its independent target state."""

    profile: Literal["prior-mean", "prior-draw"]
    seed: int | None
    state: FullTilingPosteriorState
    initvals: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class S0NUTSReferenceSetup:
    """Complete backend-independent description of one supported reference cell."""

    stage: str
    cell_name: str
    topology_role: TopologyRole
    problem: FullTilingProblem
    prior_mean_state: FullTilingPosteriorState
    data: FixedBasisNUTSData
    starts: tuple[NUTSReferenceStart, ...]


def reference_profile(name: ProfileName) -> NUTSReferenceProfile:
    """Return one of the two and only two predeclared execution profiles."""
    if name == "primary":
        return PRIMARY_PROFILE
    if name == "retry1":
        return RETRY_PROFILE
    raise ValueError("profile must be 'primary' or 'retry1'")


def reference_seeds(stage: str) -> tuple[tuple[int | None, ...], int]:
    """Return the frozen dispersed-start and sampler seeds for one stage."""
    try:
        return _START_SEEDS[stage], _SAMPLER_SEEDS[stage]
    except KeyError as error:
        raise ValueError("stage must be 's0' or 's1'") from error


def _cell_name(
    training: SyntheticTrainingArtifact,
    evaluation: SyntheticEvaluationArtifact,
    topology_role: TopologyRole,
) -> str:
    validate_artifact_pair(training, evaluation)
    if training.stage not in ("s0", "s1") or training.replicate != 0:
        raise ValueError("the NUTS reference supports only replicate zero in S0 or S1")
    if topology_role not in ("p0", "pstar"):
        raise ValueError("topology_role must be 'p0' or 'pstar'")
    if evaluation.scenario == "aligned" and topology_role != "p0":
        raise ValueError("aligned/Pstar duplicates aligned/P0 and is not a reference cell")
    name = f"{evaluation.scenario}-{topology_role}"
    if name not in REFERENCE_CELLS:
        raise ValueError("unsupported NUTS reference cell")
    return name


def _prior_draw_state(
    problem: FullTilingProblem,
    prior_mean_state: FullTilingPosteriorState,
    *,
    seed: int,
) -> FullTilingPosteriorState:
    generator = np.random.Generator(np.random.PCG64(seed))
    prior = problem.base.prior
    root_total = float(
        generator.gamma(
            shape=prior.root_shape,
            scale=1.0 / prior.root_rate,
        )
    )
    alphas = problem.allocation_prior.leaf_alphas(
        prior_mean_state.allocation.tiling,
    )
    masses = root_total * generator.dirichlet(alphas)
    return build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(
            prior_mean_state.allocation.tiling,
            masses,
        ),
    )


def prepare_s0_nuts_reference(
    training: SyntheticTrainingArtifact,
    evaluation: SyntheticEvaluationArtifact,
    *,
    topology_role: TopologyRole,
) -> S0NUTSReferenceSetup:
    """Prepare one supported target and its four frozen constrained starts."""
    cell_name = _cell_name(training, evaluation, topology_role)
    start_seeds, _ = reference_seeds(training.stage)
    bounds = training.p0_bounds if topology_role == "p0" else evaluation.pstar_bounds
    problem, prior_mean_state, data = prepare_fixed_basis_reference(
        training,
        bounds,
    )
    if data.n_fixed_coefficients != 0:
        raise RuntimeError("the S0 reference must not contain fixed coefficients")
    states = (
        prior_mean_state,
        *(
            _prior_draw_state(
                problem,
                prior_mean_state,
                seed=seed,
            )
            for seed in start_seeds[1:]
            if seed is not None
        ),
    )
    starts = tuple(
        NUTSReferenceStart(
            profile="prior-mean" if seed is None else "prior-draw",
            seed=seed,
            state=state,
            initvals={
                "root_total": float(state.root_total),
                "leaf_share": np.array(
                    state.leaf_masses / state.root_total,
                    dtype=np.float64,
                    copy=True,
                ),
            },
        )
        for seed, state in zip(start_seeds, states, strict=True)
    )
    return S0NUTSReferenceSetup(
        stage=training.stage,
        cell_name=cell_name,
        topology_role=topology_role,
        problem=problem,
        prior_mean_state=prior_mean_state,
        data=data,
        starts=starts,
    )


def preflight_s0_nuts_reference(
    setup: S0NUTSReferenceSetup,
    model: Model | None = None,
) -> tuple[Model, tuple[dict[str, float | int | str | bool], ...]]:
    """Require float64 CPU density parity independently at all four starts."""
    selected_model = build_fixed_basis_pymc_model(setup.data) if model is None else model
    audits = tuple(
        preflight_fixed_basis_nuts(
            setup.data,
            selected_model,
            initvals=start.initvals,
            expected_log_target=start.state.log_target,
        )
        for start in setup.starts
    )
    return selected_model, audits


def _require_variable(
    dataset: xr.Dataset,
    *,
    group: str,
    name: str,
    dims: tuple[str, ...],
) -> FloatArray:
    if name not in dataset:
        raise RuntimeError(f"trace group {group!r} is missing {name!r}")
    variable = dataset[name]
    if variable.dims != dims:
        raise RuntimeError(
            f"trace variable {group}.{name} must have dimensions {dims}; found {variable.dims}"
        )
    values = np.asarray(variable.values)
    if values.dtype != np.dtype(np.float64) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"trace variable {group}.{name} must be finite float64")
    return values


def _require_coordinates(
    dataset: xr.Dataset,
    *,
    expected_chains: int,
    expected_draws: int,
) -> None:
    expected = {
        "chain": np.arange(expected_chains, dtype=np.int64),
        "draw": np.arange(expected_draws, dtype=np.int64),
    }
    for name, values in expected.items():
        if name not in dataset.coords or not np.array_equal(
            np.asarray(dataset.coords[name].values),
            values,
        ):
            raise RuntimeError(f"trace has incompatible {name!r} coordinates")


def _require_exact_coordinate(
    dataset: xr.Dataset,
    *,
    group: str,
    name: str,
    expected: NDArray[Any],
) -> None:
    if name not in dataset.coords or not np.array_equal(
        np.asarray(dataset.coords[name].values),
        expected,
    ):
        raise RuntimeError(f"trace group {group!r} has incompatible {name!r} coordinates")


def validate_reference_trace(
    inference_data: InferenceData,
    *,
    data: FixedBasisNUTSData,
    expected_draws: int,
    expected_chains: int = 4,
) -> dict[str, object]:
    """Validate the scientific and diagnostic identities of a four-chain trace."""
    import arviz as az

    if not isinstance(inference_data, az.InferenceData):
        raise RuntimeError("sampler output must be an ArviZ InferenceData")
    required_groups = ("posterior", "sample_stats", "observed_data", "log_likelihood")
    if any(group not in inference_data.groups() for group in required_groups):
        raise RuntimeError("trace is missing a required InferenceData group")
    if expected_chains != 4:
        raise ValueError("the S0 reference requires exactly four chains")
    if expected_draws < 1:
        raise ValueError("expected_draws must be positive")

    posterior = cast(xr.Dataset, getattr(inference_data, "posterior"))
    _require_coordinates(
        posterior,
        expected_chains=expected_chains,
        expected_draws=expected_draws,
    )
    _require_exact_coordinate(
        posterior,
        group="posterior",
        name="leaf",
        expected=np.asarray(data.leaf_labels, dtype=np.str_),
    )
    _require_exact_coordinate(
        posterior,
        group="posterior",
        name="fixed",
        expected=np.asarray([], dtype=np.str_),
    )
    _require_exact_coordinate(
        posterior,
        group="posterior",
        name="observation",
        expected=np.arange(data.observations.size, dtype=np.int64),
    )
    root = _require_variable(
        posterior,
        group="posterior",
        name="root_total",
        dims=("chain", "draw"),
    )
    shares = _require_variable(
        posterior,
        group="posterior",
        name="leaf_share",
        dims=("chain", "draw", "leaf"),
    )
    masses = _require_variable(
        posterior,
        group="posterior",
        name="leaf_mass",
        dims=("chain", "draw", "leaf"),
    )
    scaling = _require_variable(
        posterior,
        group="posterior",
        name="leaf_scaling",
        dims=("chain", "draw", "leaf"),
    )
    mean = _require_variable(
        posterior,
        group="posterior",
        name="mean_observation",
        dims=("chain", "draw", "observation"),
    )
    fixed = _require_variable(
        posterior,
        group="posterior",
        name="fixed_coefficient",
        dims=("chain", "draw", "fixed"),
    )
    if (
        shares.shape != (expected_chains, expected_draws, data.k)
        or masses.shape != shares.shape
        or scaling.shape != shares.shape
        or mean.shape != (expected_chains, expected_draws, data.observations.size)
        or fixed.shape != (expected_chains, expected_draws, 0)
        or np.any(root <= 0.0)
        or np.any(shares <= 0.0)
        or np.any(masses <= 0.0)
        or np.any(scaling <= 0.0)
    ):
        raise RuntimeError("posterior scientific coordinate shapes or supports are invalid")
    simplex_error = float(
        np.max(
            np.abs(np.sum(shares, axis=-1) - 1.0),
            initial=0.0,
        )
    )
    if simplex_error > 5.0e-12:
        raise RuntimeError("posterior leaf shares do not lie on the simplex")
    mass_error = float(np.max(np.abs(masses - root[..., None] * shares), initial=0.0))
    scaling_error = float(
        np.max(
            np.abs(scaling - masses / data.nominal_leaf_share),
            initial=0.0,
        )
    )
    expected_mean = data.fixed_offset[None, None, :] + masses @ data.dynamic_design.T
    mean_error = float(np.max(np.abs(mean - expected_mean), initial=0.0))
    if (
        not np.allclose(masses, root[..., None] * shares, rtol=5.0e-12, atol=5.0e-10)
        or not np.allclose(
            scaling,
            masses / data.nominal_leaf_share,
            rtol=5.0e-12,
            atol=5.0e-10,
        )
        or not np.allclose(mean, expected_mean, rtol=5.0e-12, atol=5.0e-10)
    ):
        raise RuntimeError("posterior deterministic identity failed")

    sample_stats = cast(xr.Dataset, getattr(inference_data, "sample_stats"))
    _require_coordinates(
        sample_stats,
        expected_chains=expected_chains,
        expected_draws=expected_draws,
    )
    if "diverging" not in sample_stats:
        raise RuntimeError("trace sample_stats is missing 'diverging'")
    diverging = np.asarray(sample_stats["diverging"].values)
    if diverging.dtype != np.dtype(bool) or diverging.shape != (expected_chains, expected_draws):
        raise RuntimeError("trace divergences have incompatible dtype or shape")
    for name in ("acceptance_rate", "energy", "lp", "step_size"):
        values = _require_variable(
            sample_stats,
            group="sample_stats",
            name=name,
            dims=("chain", "draw"),
        )
        if name == "acceptance_rate" and np.any((values < 0.0) | (values > 1.0)):
            raise RuntimeError("acceptance rates must lie in [0, 1]")
        if name == "step_size" and np.any(values <= 0.0):
            raise RuntimeError("step sizes must be positive")
    for name in ("n_steps", "tree_depth"):
        if name not in sample_stats:
            raise RuntimeError(f"trace sample_stats is missing {name!r}")
        values = np.asarray(sample_stats[name].values)
        if (
            sample_stats[name].dims != ("chain", "draw")
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 1)
        ):
            raise RuntimeError(f"trace sample_stats.{name} is incompatible")

    observed_data = cast(xr.Dataset, getattr(inference_data, "observed_data"))
    _require_exact_coordinate(
        observed_data,
        group="observed_data",
        name="observation",
        expected=np.arange(data.observations.size, dtype=np.int64),
    )
    if "observed" not in observed_data or observed_data["observed"].dims != ("observation",):
        raise RuntimeError("trace observed data has incompatible dimensions")
    observed = np.asarray(observed_data["observed"].values)
    if (
        observed.dtype != np.dtype(np.float64)
        or not np.all(np.isfinite(observed))
        or not np.array_equal(observed, data.observations)
    ):
        raise RuntimeError("trace observed data differs from the reference input")
    log_likelihood = cast(xr.Dataset, getattr(inference_data, "log_likelihood"))
    _require_coordinates(
        log_likelihood,
        expected_chains=expected_chains,
        expected_draws=expected_draws,
    )
    _require_exact_coordinate(
        log_likelihood,
        group="log_likelihood",
        name="observation",
        expected=np.arange(data.observations.size, dtype=np.int64),
    )
    pointwise = _require_variable(
        log_likelihood,
        group="log_likelihood",
        name="observed",
        dims=("chain", "draw", "observation"),
    )
    residual = (data.observations[None, None, :] - mean) / data.observation_sd
    expected_pointwise = (
        -0.5 * residual * residual - np.log(data.observation_sd) - 0.5 * math.log(2.0 * math.pi)
    )
    log_likelihood_error = float(np.max(np.abs(pointwise - expected_pointwise), initial=0.0))
    if not np.allclose(
        pointwise,
        expected_pointwise,
        rtol=5.0e-12,
        atol=5.0e-10,
    ):
        raise RuntimeError("trace pointwise log likelihood identity failed")
    return {
        "groups": list(required_groups),
        "chains": expected_chains,
        "draws": expected_draws,
        "maximum_leaf_share_simplex_error": simplex_error,
        "maximum_leaf_mass_identity_error": mass_error,
        "maximum_leaf_scaling_identity_error": scaling_error,
        "maximum_mean_observation_identity_error": mean_error,
        "maximum_pointwise_log_likelihood_error": log_likelihood_error,
    }


def _diagnostic_value(dataset: xr.Dataset, name: str, index: int | None) -> float:
    values = np.asarray(dataset[name].values, dtype=np.float64)
    return float(values.item() if index is None else values[index])


def summarize_reference_trace(
    inference_data: InferenceData,
    *,
    data: FixedBasisNUTSData,
    nominal_weight: FloatArray,
) -> dict[str, object]:
    """Return exact root/leaf diagnostics and common-projection moments."""
    import arviz as az

    posterior = cast(xr.Dataset, getattr(inference_data, "posterior"))
    diagnostic_input = posterior[["root_total", "leaf_mass"]]
    rank_rhat = cast(xr.Dataset, az.rhat(diagnostic_input, method="rank"))
    bulk_ess = cast(xr.Dataset, az.ess(diagnostic_input, method="bulk"))
    tail_ess = cast(xr.Dataset, az.ess(diagnostic_input, method="tail"))
    diagnostics: dict[str, dict[str, float]] = {}
    labels = ("root_total", *tuple(f"leaf_mass[{label}]" for label in data.leaf_labels))
    diagnostics["root_total"] = {
        "rank_normalized_rhat": _diagnostic_value(rank_rhat, "root_total", None),
        "bulk_ess": _diagnostic_value(bulk_ess, "root_total", None),
        "tail_ess": _diagnostic_value(tail_ess, "root_total", None),
    }
    for index, label in enumerate(data.leaf_labels):
        diagnostics[f"leaf_mass[{label}]"] = {
            "rank_normalized_rhat": _diagnostic_value(rank_rhat, "leaf_mass", index),
            "bulk_ess": _diagnostic_value(bulk_ess, "leaf_mass", index),
            "tail_ess": _diagnostic_value(tail_ess, "leaf_mass", index),
        }
    if tuple(diagnostics) != labels:
        raise RuntimeError("diagnostic variable ordering drifted")
    for variable, values in diagnostics.items():
        if any(not math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"non-finite convergence diagnostic for {variable}")
    worst_rhat_variable = max(
        diagnostics,
        key=lambda name: diagnostics[name]["rank_normalized_rhat"],
    )
    minimum_bulk_variable = min(
        diagnostics,
        key=lambda name: diagnostics[name]["bulk_ess"],
    )
    minimum_tail_variable = min(
        diagnostics,
        key=lambda name: diagnostics[name]["tail_ess"],
    )

    masses = np.asarray(posterior["leaf_mass"].values, dtype=np.float64)
    chains, draws, k = masses.shape
    repeated_bounds = np.broadcast_to(
        data.rectangle_bounds,
        (chains * draws, k, 4),
    )
    fields = reconstruct_native_fields(
        repeated_bounds,
        masses.reshape(chains * draws, k),
        nominal_weight,
    )
    projections = common_native_totals(
        fields,
        nominal_weight,
    ).reshape(chains, draws, len(PROJECTION_NAMES))
    projection_summary: dict[str, dict[str, float]] = {}
    midpoint = draws // 2
    for index, name in enumerate(PROJECTION_NAMES):
        values = projections[:, :, index]
        mcse_result = az.mcse(
            xr.DataArray(
                values,
                dims=("chain", "draw"),
                name="projection",
            ),
            method="mean",
        )
        mcse = float(np.asarray(mcse_result["projection"].values).item())
        item = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "mcse_mean": mcse,
            "first_half_mean": float(np.mean(values[:, :midpoint])),
            "second_half_mean": float(np.mean(values[:, midpoint:])),
        }
        if any(not math.isfinite(value) for value in item.values()):
            raise RuntimeError(f"non-finite projection summary for {name}")
        projection_summary[name] = item

    divergences = int(
        np.count_nonzero(
            np.asarray(cast(xr.Dataset, getattr(inference_data, "sample_stats"))["diverging"].values)
        )
    )
    worst_rhat = diagnostics[worst_rhat_variable]["rank_normalized_rhat"]
    minimum_bulk = diagnostics[minimum_bulk_variable]["bulk_ess"]
    minimum_tail = diagnostics[minimum_tail_variable]["tail_ess"]
    first_failed_gate: str | None = None
    if divergences:
        first_failed_gate = "zero_divergences"
    elif worst_rhat > 1.01:
        first_failed_gate = "rank_normalized_rhat"
    elif minimum_bulk < 200.0:
        first_failed_gate = "bulk_ess"
    elif minimum_tail < 200.0:
        first_failed_gate = "tail_ess"
    return {
        "divergences": divergences,
        "root_leaf_diagnostics": diagnostics,
        "worst_rhat_variable": worst_rhat_variable,
        "worst_rhat_value": worst_rhat,
        "minimum_bulk_ess_variable": minimum_bulk_variable,
        "minimum_bulk_ess_value": minimum_bulk,
        "minimum_tail_ess_variable": minimum_tail_variable,
        "minimum_tail_ess_value": minimum_tail,
        "projections": projection_summary,
        "first_failed_gate": first_failed_gate,
    }


__all__ = [
    "NUTSReferenceProfile",
    "NUTSReferenceStart",
    "PRIMARY_PROFILE",
    "PROJECTION_NAMES",
    "REFERENCE_CELLS",
    "RETRY_PROFILE",
    "SAMPLER_SEED",
    "S0NUTSReferenceSetup",
    "START_SEEDS",
    "preflight_s0_nuts_reference",
    "prepare_s0_nuts_reference",
    "reference_profile",
    "reference_seeds",
    "summarize_reference_trace",
    "validate_reference_trace",
]
