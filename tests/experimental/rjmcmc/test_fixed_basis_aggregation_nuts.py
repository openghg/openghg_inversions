"""Focused tests for fixed-partition aggregation-aware PyMC/NumPyro NUTS."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    PartitionSummaryFactors,
)
from openghg_inversions.experimental.rjmcmc.fixed_basis_aggregation_nuts import (
    FixedBasisAggregationNUTSData,
    build_fixed_basis_aggregation_pymc_model,
    compile_fixed_basis_aggregation_pytensor_evaluator,
    fixed_basis_aggregation_numpy_logp_and_gradient,
    prepare_fixed_basis_aggregation_nuts,
    sample_fixed_basis_aggregation_nuts,
    validate_fixed_basis_aggregation_inference_data,
)
from openghg_inversions.experimental.rjmcmc.fixed_basis_nuts import (
    build_fixed_basis_pymc_model,
    fixed_basis_nuts_initvals,
    prepare_fixed_basis_nuts,
)
from openghg_inversions.experimental.rjmcmc.full_tiling import TilingState
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    build_full_tiling_posterior_state,
    full_tiling_problem_from_gamma_beta_adapter,
    initialize_full_tiling_posterior_state,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    gamma_beta_problem_from_rhime_inputs,
)

_THIS_FILE = Path(__file__).resolve()
_LOG_TWO_PI = math.log(2.0 * math.pi)


def _pytensor_flags_with_float64(flags: str, *, cache: Path) -> str:
    """Return uncontaminated float64 flags with one writable compile cache."""
    retained = []
    for item in flags.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name not in {"floatX", "warn_float64", "base_compiledir"}:
            retained.append(stripped)
    return ",".join(
        (
            "floatX=float64",
            "warn_float64=ignore",
            f"base_compiledir={cache}",
            *retained,
        )
    )


def _run_x64_case(case: str) -> None:
    """Run one PyMC/PyTensor assertion in a fresh float64 process."""
    cache = Path("/tmp/openghg-fixed-basis-aggregation-nuts-tests")
    environment = os.environ.copy()
    environment["PYTENSOR_FLAGS"] = _pytensor_flags_with_float64(
        environment.get("PYTENSOR_FLAGS", ""),
        cache=cache / "pytensor",
    )
    environment["JAX_ENABLE_X64"] = "1"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XDG_CACHE_HOME"] = str(cache / "xdg")
    environment["MPLCONFIGDIR"] = str(cache / "matplotlib")
    environment["NUMBA_CACHE_DIR"] = str(cache / "numba")
    completed = subprocess.run(
        [sys.executable, str(_THIS_FILE), case],
        cwd=_THIS_FILE.parents[3],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, (
        f"isolated aggregation-aware NUTS case {case!r} failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _problem_state(
    *,
    k: int = 4,
    with_fixed: bool = True,
) -> tuple[FullTilingProblem, FullTilingPosteriorState]:
    """Return a small target with heterogeneous native weights and fixed terms."""
    sensitivity = np.arange(1.0, 49.0).reshape(3, 4, 4)
    outer = np.arange(6.0).reshape(3, 2) / 5.0
    boundary = np.array([2.0, 3.0, 5.0])
    fixed_mean = np.array([0.8, 1.2])
    observations = boundary + sensitivity.sum(axis=(1, 2))
    if with_fixed:
        observations = observations + outer @ fixed_mean
    dataset = xr.Dataset(
        {
            "fp_x_flux": (
                ("lon", "nmeasure", "lat"),
                sensitivity.transpose(2, 0, 1),
            ),
            "mf": ("nmeasure", observations),
            "mf_error": ("nmeasure", np.array([0.7, 0.8, 0.9])),
            "outer": (("fixed", "nmeasure"), outer.T),
            "boundary": ("nmeasure", boundary),
        },
        coords={
            "nmeasure": ["a", "b", "c"],
            "lat": np.arange(4) + 50.0,
            "lon": np.arange(4) - 3.0,
            "fixed": ["north", "south"],
        },
    )
    nominal_weight = xr.DataArray(
        np.arange(1.0, 17.0).reshape(4, 4).T,
        dims=("lon", "lat"),
        coords={"lon": dataset.lon, "lat": dataset.lat},
    )
    fixed_arguments: dict[str, object] = {}
    if with_fixed:
        fixed_arguments = {
            "fixed_design_name": "outer",
            "fixed_coefficient_prior_mean": fixed_mean,
            "fixed_coefficient_prior_sd": np.array([0.3, 0.5]),
        }
    adapter = gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=nominal_weight,
        k_min=4,
        k_max=4,
        concentration=5.0,
        root_variance=0.25,
        likelihood_power=1.0,
        sensitivity_name="fp_x_flux",
        observation_name="mf",
        observation_sd_name="mf_error",
        fixed_offset_name="boundary",
        **fixed_arguments,  # type: ignore[arg-type]
    )
    problem = full_tiling_problem_from_gamma_beta_adapter(
        adapter,
        concentration=7.0,
    )
    return problem, initialize_full_tiling_posterior_state(problem, k=k)


def _basis(rank: int) -> np.ndarray:
    """Return a fixed orthonormal observation-space basis."""
    if rank == 0:
        return np.empty((3, 0))
    raw = np.array([[1.0, 0.2], [0.1, 1.0], [0.3, -0.2]])
    basis, _ = np.linalg.qr(raw)
    return basis[:, :rank]


def _alpha_field(problem: FullTilingProblem) -> np.ndarray:
    """Return one K-independent native-cell alpha field."""
    return problem.concentration * problem.normalized_nominal_mass


def _data(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
    *,
    rank: int,
) -> FixedBasisAggregationNUTSData:
    """Prepare one fixture bridge."""
    return prepare_fixed_basis_aggregation_nuts(
        problem,
        state,
        summary_basis=_basis(rank),
        native_cell_alphas=_alpha_field(problem),
        native_alpha_id="fixture-native-alpha-v1",
    )


def _interior_state(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
) -> FullTilingPosteriorState:
    """Return one asymmetric state on the fixture topology."""
    shares = np.array([0.11, 0.19, 0.27, 0.43])
    return build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(source.allocation.tiling, 1.37 * shares),
        fixed_coefficients=np.array([0.72, 1.41]),
    )


def _points(
    data: FixedBasisAggregationNUTSData,
) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Return nominal, seeded prior-draw, and extreme valid coordinates."""
    base = data.fixed_basis
    rng = np.random.Generator(np.random.PCG64(104729))
    fixed_mu, fixed_sigma = base.fixed_lognormal_mu_sigma
    prior = (
        float(rng.gamma(base.root_shape, 1.0 / base.root_rate)),
        rng.dirichlet(base.dirichlet_alpha),
        rng.lognormal(fixed_mu, fixed_sigma),
    )
    extreme_share = np.array([1.0e-7, 0.0999999, 0.2, 0.7])
    assert float(extreme_share.sum()) == 1.0
    return {
        "nominal": (
            base.initial_root_total,
            np.asarray(base.initial_leaf_share),
            np.asarray(base.initial_fixed_coefficient),
        ),
        "prior_draw": prior,
        "extreme": (
            1.0e-4,
            extreme_share,
            np.array([1.0e-3, 12.0]),
        ),
    }


def _valid_inference_data(
    az: Any,
    data: FixedBasisAggregationNUTSData,
    *,
    chains: int = 1,
    draws: int = 2,
) -> Any:
    """Return a complete scientifically consistent posterior fixture."""
    base = data.fixed_basis
    root = np.empty((chains, draws), dtype=np.float64)
    share = np.empty((chains, draws, data.k), dtype=np.float64)
    fixed = np.empty(
        (chains, draws, base.n_fixed_coefficients),
        dtype=np.float64,
    )
    for chain in range(chains):
        for draw in range(draws):
            displacement = 1.0 + 0.01 * (chain * draws + draw)
            root[chain, draw] = base.initial_root_total * displacement
            share[chain, draw] = base.initial_leaf_share
            fixed[chain, draw] = base.initial_fixed_coefficient * displacement
    mass = root[..., None] * share
    scaling = mass / base.nominal_leaf_share
    mean = (
        base.fixed_offset
        + np.einsum(
            "...k,nk->...n",
            mass,
            base.dynamic_design,
            optimize=False,
        )
        + np.einsum(
            "...f,nf->...n",
            fixed,
            base.fixed_design,
            optimize=False,
        )
    )
    likelihood = np.empty((chains, draws), dtype=np.float64)
    for index in np.ndindex((chains, draws)):
        likelihood[index] = _independent_dense_joint_likelihood(
            data,
            (
                float(root[index]),
                np.asarray(share[index]),
                np.asarray(fixed[index]),
            ),
        )
    posterior = xr.Dataset(
        data_vars={
            "root_total": (("chain", "draw"), root),
            "leaf_share": (("chain", "draw", "leaf"), share),
            "leaf_mass": (("chain", "draw", "leaf"), mass),
            "leaf_scaling": (("chain", "draw", "leaf"), scaling),
            "fixed_coefficient": (("chain", "draw", "fixed"), fixed),
            "mean_observation": (
                ("chain", "draw", "observation"),
                mean,
            ),
            "aggregation_joint_log_likelihood": (
                ("chain", "draw"),
                likelihood,
            ),
        },
        coords={
            "chain": np.arange(chains, dtype=np.int64),
            "draw": np.arange(draws, dtype=np.int64),
            "leaf": np.asarray(base.leaf_labels),
            "fixed": np.asarray(tuple(f"fixed_{position}" for position in range(base.n_fixed_coefficients))),
            "observation": np.arange(
                base.observations.size,
                dtype=np.int64,
            ),
        },
    )
    return az.InferenceData(posterior=posterior)


def _attach_test_manifest(
    data: FixedBasisAggregationNUTSData,
    result: Any,
) -> None:
    """Attach the production-format manifest to one output fixture."""
    manifest_json = json.dumps(
        data.target_manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    result.attrs["fixed_basis_aggregation_manifest_json"] = manifest_json
    result.attrs["fixed_basis_aggregation_manifest_sha256"] = hashlib.sha256(
        manifest_json.encode("utf-8")
    ).hexdigest()


def _alpha_sha256(values: np.ndarray) -> str:
    """Independently reproduce the public native-alpha content identity."""
    digest = hashlib.sha256()
    digest.update(b"fixed-native-alpha-field-v1\0")
    digest.update(b"native_cell_alphas\0")
    digest.update(b"<f8\0")
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def test_bridge_freezes_one_native_alpha_identity_across_k() -> None:
    """The same explicit native field must retain total and hash across bases."""
    problem, state_k4 = _problem_state(k=4)
    state_k2 = initialize_full_tiling_posterior_state(problem, k=2)
    alpha = _alpha_field(problem)
    data_k4 = _data(problem, state_k4, rank=2)
    data_k2 = _data(problem, state_k2, rank=2)

    assert data_k4.native_alpha_id == "fixture-native-alpha-v1"
    assert data_k4.native_alpha_sha256 == _alpha_sha256(alpha)
    assert data_k2.native_alpha_sha256 == data_k4.native_alpha_sha256
    assert data_k2.native_alpha_total == data_k4.native_alpha_total == 7.0
    assert data_k2.k == 2
    assert data_k4.k == 4
    assert np.array_equal(
        data_k4.factors.observation_mean_design,
        data_k4.fixed_basis.dynamic_design,
    )
    assert np.array_equal(
        data_k4.factors.alpha_totals,
        data_k4.fixed_basis.dirichlet_alpha,
    )
    assert not data_k4.summary_basis.flags.writeable
    assert not data_k4.native_cell_alphas.flags.writeable


def test_rank_zero_cache_is_exact_existing_target_identity() -> None:
    """Canonical cached means and priors must be exact when aggregation rank is zero."""
    problem, state = _problem_state()
    data = _data(problem, state, rank=0)

    assert data.summary_dimension == 0
    assert data.factors.summary_mean_design.shape == (0, 4)
    assert data.factors.summary_covariance_factors.shape == (4, 0, 0)
    assert np.array_equal(
        data.factors.observation_mean_design,
        data.fixed_basis.dynamic_design,
    )
    assert np.array_equal(
        data.factors.alpha_totals,
        data.fixed_basis.dirichlet_alpha,
    )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"native_alpha_id": " bad "}, "non-empty stripped"),
        ({"native_cell_alphas": np.ones((2, 8))}, "native shape"),
        (
            {"native_cell_alphas": np.full((4, 4), np.nan)},
            "finite",
        ),
        (
            {"native_cell_alphas": np.ones((4, 4))},
            "region sums|conditional means",
        ),
        (
            {"summary_basis": np.ones((3, 1))},
            "orthonormal",
        ),
    ],
)
def test_bridge_fails_closed_on_identity_shape_and_finite_drift(
    replacement: dict[str, object],
    message: str,
) -> None:
    """Malformed scientific identity fields must fail before model building."""
    problem, state = _problem_state()
    arguments: dict[str, object] = {
        "summary_basis": _basis(2),
        "native_cell_alphas": _alpha_field(problem),
        "native_alpha_id": "fixture-native-alpha-v1",
    }
    arguments.update(replacement)

    with pytest.raises(ValueError, match=message):
        prepare_fixed_basis_aggregation_nuts(
            problem,
            state,
            **arguments,  # type: ignore[arg-type]
        )


def test_bridge_rejects_cache_order_and_target_drift() -> None:
    """Cache arrays must remain aligned with canonical leaves and base priors."""
    problem, state = _problem_state()
    data = _data(problem, state, rank=2)
    factors = data.factors
    changed_design = np.array(factors.observation_mean_design, copy=True)
    changed_design[0, 0] = np.nextafter(changed_design[0, 0], np.inf)
    changed_factors = PartitionSummaryFactors(
        factors.labels,
        factors.alpha_totals,
        changed_design,
        factors.summary_mean_design,
        factors.summary_covariance_factors,
    )

    with pytest.raises(ValueError, match="authoritative fixed-basis design exactly"):
        replace(data, factors=changed_factors)

    changed_alpha = np.array(factors.alpha_totals, copy=True)
    changed_alpha[0] = np.nextafter(changed_alpha[0], np.inf)
    changed_factors = PartitionSummaryFactors(
        factors.labels,
        changed_alpha,
        factors.observation_mean_design,
        factors.summary_mean_design,
        factors.summary_covariance_factors,
    )
    with pytest.raises(ValueError, match="Dirichlet shapes exactly"):
        replace(data, factors=changed_factors)


def test_bridge_authenticates_covariance_factors_and_hidden_native_allocation() -> None:
    """PSD cache substitution and within-region alpha rearrangement must fail."""
    problem, state = _problem_state()
    data = _data(problem, state, rank=2)
    factors = data.factors
    assert data.partition_factors_sha256 == data.reconstructed_factors_sha256

    changed_covariance = np.array(
        factors.summary_covariance_factors,
        copy=True,
    )
    changed_covariance[0] += 1.0e-6 * np.eye(data.summary_dimension)
    changed_factors = PartitionSummaryFactors(
        factors.labels,
        factors.alpha_totals,
        factors.observation_mean_design,
        factors.summary_mean_design,
        changed_covariance,
    )
    with pytest.raises(ValueError, match="do not reconstruct"):
        replace(data, factors=changed_factors)

    labels = factors.labels.reshape(-1)
    selected = np.flatnonzero(labels == 0)
    assert selected.size >= 2
    rearranged = np.array(data.native_cell_alphas, copy=True).reshape(-1)
    first, second = int(selected[0]), int(selected[-1])
    rearranged[first], rearranged[second] = (
        rearranged[second],
        rearranged[first],
    )
    rearranged = rearranged.reshape(data.native_cell_alphas.shape)
    np.testing.assert_allclose(
        [rearranged[labels.reshape(data.native_cell_alphas.shape) == label].sum() for label in range(data.k)],
        factors.alpha_totals,
        rtol=0.0,
        atol=3.0e-16,
    )
    with pytest.raises(ValueError, match="do not reconstruct|conditional means"):
        replace(data, native_cell_alphas=rearranged)

    changed_sensitivity = np.array(data.native_sensitivity, copy=True)
    changed_sensitivity[0, first] += 1.0e-3
    with pytest.raises(ValueError, match="do not reconstruct|conditional means"):
        replace(data, native_sensitivity=changed_sensitivity)


def test_target_manifest_authenticates_full_bridge_content() -> None:
    """The manifest must expose all source, target, topology, and cache hashes."""
    problem, state = _problem_state()
    data = _data(problem, state, rank=2)
    manifest = data.target_manifest

    assert manifest["schema"] == "fixed-basis-aggregation-nuts-manifest-v1"
    assert manifest["native_alpha_id"] == "fixture-native-alpha-v1"
    assert manifest["native_alpha_sha256"] == data.native_alpha_sha256
    assert manifest["native_alpha_total"] == 7.0
    assert manifest["summary_basis_sha256"] == data.summary_basis_sha256
    assert manifest["partition_factors_sha256"] == data.partition_factors_sha256
    assert manifest["reconstructed_factors_sha256"] == data.partition_factors_sha256
    assert manifest["topology_sha256"] == data.topology_sha256
    assert manifest["native_sensitivity_sha256"] == data.native_sensitivity_sha256
    assert manifest["fixed_target_sha256"] == data.fixed_target_sha256
    assert manifest["model_identity_sha256"] == data.model_identity_sha256
    assert manifest["source_implementation_sha256"] == data.source_implementation_sha256
    for name in (
        "native_alpha_sha256",
        "summary_basis_sha256",
        "partition_factors_sha256",
        "topology_sha256",
        "native_sensitivity_sha256",
        "fixed_target_sha256",
        "aggregation_bridge_sha256",
        "model_identity_sha256",
        "source_implementation_sha256",
    ):
        assert isinstance(manifest[name], str)
        assert len(str(manifest[name])) == 64


def _independent_dense_joint_likelihood(
    data: FixedBasisAggregationNUTSData,
    point: tuple[float, np.ndarray, np.ndarray],
) -> float:
    """Evaluate the normalized likelihood through an independent dense oracle."""
    root, share, fixed = point
    base = data.fixed_basis
    masses = root * share
    mean = base.fixed_offset + base.dynamic_design @ masses + base.fixed_design @ fixed
    summary_covariance = np.einsum(
        "k,kij->ij",
        np.square(masses),
        data.factors.summary_covariance_factors,
    )
    lifted = base.observation_sd[:, None] * data.summary_basis
    covariance = np.diag(np.square(base.observation_sd))
    covariance += lifted @ summary_covariance @ lifted.T
    sign, log_determinant = np.linalg.slogdet(covariance)
    assert sign == 1.0
    residual = base.observations - mean
    likelihood = -0.5 * (
        base.observations.size * _LOG_TWO_PI
        + log_determinant
        + residual @ np.linalg.solve(covariance, residual)
    )
    return float(likelihood)


def _independent_dense_log_target(
    data: FixedBasisAggregationNUTSData,
    point: tuple[float, np.ndarray, np.ndarray],
) -> float:
    """Evaluate the target through an independent dense covariance oracle."""
    root, share, fixed = point
    base = data.fixed_basis
    likelihood = _independent_dense_joint_likelihood(data, point)
    root_prior = (
        base.root_shape * math.log(base.root_rate)
        - math.lgamma(base.root_shape)
        + (base.root_shape - 1.0) * math.log(root)
        - base.root_rate * root
    )
    share_prior = (
        math.lgamma(float(base.dirichlet_alpha.sum()))
        - sum(math.lgamma(float(alpha)) for alpha in base.dirichlet_alpha)
        + np.dot(base.dirichlet_alpha - 1.0, np.log(share))
    )
    fixed_mu, fixed_sigma = base.fixed_lognormal_mu_sigma
    fixed_prior = np.sum(
        -0.5 * _LOG_TWO_PI
        - np.log(fixed_sigma)
        - np.log(fixed)
        - 0.5 * np.square((np.log(fixed) - fixed_mu) / fixed_sigma)
    )
    return float(likelihood + root_prior + share_prior + fixed_prior)


def _independent_diagonal_logp_and_gradient(
    data: FixedBasisAggregationNUTSData,
    point: tuple[float, np.ndarray, np.ndarray],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return an independent normalized diagonal target and analytic gradient."""
    root, share, fixed = point
    base = data.fixed_basis
    masses = root * share
    mean = base.fixed_offset + base.dynamic_design @ masses + base.fixed_design @ fixed
    residual = (base.observations - mean) / base.observation_sd
    likelihood = -0.5 * (
        base.observations.size * _LOG_TWO_PI + 2.0 * np.log(base.observation_sd).sum() + residual @ residual
    )
    root_prior = (
        base.root_shape * math.log(base.root_rate)
        - math.lgamma(base.root_shape)
        + (base.root_shape - 1.0) * math.log(root)
        - base.root_rate * root
    )
    share_prior = (
        math.lgamma(float(base.dirichlet_alpha.sum()))
        - sum(math.lgamma(float(alpha)) for alpha in base.dirichlet_alpha)
        + np.dot(base.dirichlet_alpha - 1.0, np.log(share))
    )
    fixed_mu, fixed_sigma = base.fixed_lognormal_mu_sigma
    fixed_prior = np.sum(
        -0.5 * _LOG_TWO_PI
        - np.log(fixed_sigma)
        - np.log(fixed)
        - 0.5 * np.square((np.log(fixed) - fixed_mu) / fixed_sigma)
    )
    mean_gradient = residual / base.observation_sd
    mass_gradient = base.dynamic_design.T @ mean_gradient
    root_gradient = mass_gradient @ share + (base.root_shape - 1.0) / root - base.root_rate
    share_gradient = root * mass_gradient + (base.dirichlet_alpha - 1.0) / share
    fixed_gradient = (
        base.fixed_design.T @ mean_gradient
        - 1.0 / fixed
        - (np.log(fixed) - fixed_mu) / (np.square(fixed_sigma) * fixed)
    )
    return (
        float(likelihood + root_prior + share_prior + fixed_prior),
        float(root_gradient),
        np.asarray(share_gradient),
        np.asarray(fixed_gradient),
    )


def test_numpy_oracle_matches_independent_dense_normalized_target() -> None:
    """The low-rank oracle must retain the dense Gaussian normalizer."""
    problem, state = _problem_state()
    data = _data(problem, state, rank=2)

    for point in _points(data).values():
        actual = fixed_basis_aggregation_numpy_logp_and_gradient(data, *point)[0]
        expected = _independent_dense_log_target(data, point)
        assert actual == pytest.approx(expected, rel=2.0e-12, abs=2.0e-12)


def test_numpy_oracle_gradient_matches_dense_directional_differences() -> None:
    """Analytic gradients must match dense-oracle finite differences."""
    problem, state = _problem_state()
    data = _data(problem, state, rank=2)
    point = _points(data)["prior_draw"]
    root, share, fixed = point
    _, root_gradient, share_gradient, fixed_gradient = fixed_basis_aggregation_numpy_logp_and_gradient(
        data, *point
    )

    root_step = 1.0e-6 * root
    root_difference = (
        _independent_dense_log_target(data, (root + root_step, share, fixed))
        - _independent_dense_log_target(data, (root - root_step, share, fixed))
    ) / (2.0 * root_step)
    assert root_gradient == pytest.approx(root_difference, rel=2.0e-6, abs=2.0e-6)

    direction = np.array([0.3, -0.2, 0.4, -0.5])
    assert direction.sum() == pytest.approx(0.0)
    share_step = 1.0e-7
    share_difference = (
        _independent_dense_log_target(
            data,
            (root, share + share_step * direction, fixed),
        )
        - _independent_dense_log_target(
            data,
            (root, share - share_step * direction, fixed),
        )
    ) / (2.0 * share_step)
    assert float(share_gradient @ direction) == pytest.approx(
        share_difference,
        rel=2.0e-6,
        abs=2.0e-6,
    )

    fixed_direction = np.array([0.4, -0.6])
    fixed_step = 1.0e-7
    fixed_difference = (
        _independent_dense_log_target(
            data,
            (root, share, fixed + fixed_step * fixed_direction),
        )
        - _independent_dense_log_target(
            data,
            (root, share, fixed - fixed_step * fixed_direction),
        )
    ) / (2.0 * fixed_step)
    assert float(fixed_gradient @ fixed_direction) == pytest.approx(
        fixed_difference,
        rel=2.0e-6,
        abs=2.0e-6,
    )


def test_q0_pymc_target_is_exact_existing_fixed_basis_target() -> None:
    """The scalar-Potential q=0 target must equal the existing reference."""
    _run_x64_case("q0")


def _assert_q0_pymc_target_is_exact_existing_fixed_basis_target() -> None:
    """Compare both constrained model densities inside one float64 process."""
    import pymc as pm

    pm_runtime: Any = pm

    def constrained_logp(model: Any, initvals: dict[str, object]) -> float:
        point_fn = pm_runtime.initial_point.make_initial_point_fn(
            model=model,
            overrides=initvals,
            jitter_rvs=set(),
            default_strategy="support_point",
            return_transformed=True,
        )
        return float(model.compile_logp(jacobian=False)(point_fn(0)))

    for with_fixed in (True, False):
        problem, prior_state = _problem_state(with_fixed=with_fixed)
        state = _interior_state(problem, prior_state) if with_fixed else prior_state
        existing_data = prepare_fixed_basis_nuts(problem, state)
        aggregation_data = _data(problem, state, rank=0)
        existing = build_fixed_basis_pymc_model(existing_data)
        aggregation = build_fixed_basis_aggregation_pymc_model(aggregation_data)
        if with_fixed:
            points = tuple(_points(aggregation_data).values())
        else:
            base = aggregation_data.fixed_basis
            points = (
                (
                    base.initial_root_total,
                    np.asarray(base.initial_leaf_share),
                    np.empty(0),
                ),
                (
                    1.0e-4,
                    np.array([1.0e-7, 0.0999999, 0.2, 0.7]),
                    np.empty(0),
                ),
            )
        for root, share, fixed in points:
            initvals: dict[str, object] = {
                "root_total": root,
                "leaf_share": share,
            }
            if with_fixed:
                initvals["fixed_coefficient"] = fixed
            assert constrained_logp(aggregation, initvals) == constrained_logp(
                existing,
                initvals,
            )


def test_q0_pytensor_density_and_gradient_match_independent_diagonal_oracle() -> None:
    """Rank-zero PyTensor density and gradients must retain diagonal parity."""
    _run_x64_case("q0_gradient")


def _assert_q0_pytensor_density_and_gradient_match_independent_diagonal_oracle() -> None:
    """Check fixed and zero-fixed rank-zero targets in scientific coordinates."""
    for with_fixed in (True, False):
        problem, state = _problem_state(with_fixed=with_fixed)
        data = _data(problem, state, rank=0)
        evaluator = compile_fixed_basis_aggregation_pytensor_evaluator(data)
        if with_fixed:
            points = tuple(_points(data).values())
        else:
            base = data.fixed_basis
            points = (
                (
                    base.initial_root_total,
                    np.asarray(base.initial_leaf_share),
                    np.empty(0),
                ),
                (
                    1.0e-4,
                    np.array([1.0e-7, 0.0999999, 0.2, 0.7]),
                    np.empty(0),
                ),
            )
        for point in points:
            expected = _independent_diagonal_logp_and_gradient(data, point)
            actual = evaluator(*point)
            assert actual[0] == pytest.approx(expected[0], rel=2.0e-12, abs=2.0e-12)
            assert actual[1] == pytest.approx(expected[1], rel=5.0e-10, abs=5.0e-9)
            np.testing.assert_allclose(
                actual[2],
                expected[2],
                rtol=5.0e-10,
                atol=5.0e-9,
            )
            np.testing.assert_allclose(
                actual[3],
                expected[3],
                rtol=5.0e-10,
                atol=5.0e-9,
            )


def test_pytensor_log_density_and_gradient_match_numpy_oracle() -> None:
    """PyTensor must match NumPy at nominal, prior-draw, and extreme states."""
    _run_x64_case("gradient")


def _assert_pytensor_log_density_and_gradient_match_numpy_oracle() -> None:
    """Evaluate all predeclared points in one compiled float64 graph."""
    import pymc as pm

    pm_runtime: Any = pm
    problem, state = _problem_state()
    data = _data(problem, state, rank=2)
    pytensor_evaluator = compile_fixed_basis_aggregation_pytensor_evaluator(data)
    model = build_fixed_basis_aggregation_pymc_model(data)
    compiled_model_logp = model.compile_logp(jacobian=False)

    for point in _points(data).values():
        expected = fixed_basis_aggregation_numpy_logp_and_gradient(data, *point)
        actual = pytensor_evaluator(*point)
        root, share, fixed = point
        point_fn = pm_runtime.initial_point.make_initial_point_fn(
            model=model,
            overrides={
                "root_total": root,
                "leaf_share": share,
                "fixed_coefficient": fixed,
            },
            jitter_rvs=set(),
            default_strategy="support_point",
            return_transformed=True,
        )
        model_logp = float(compiled_model_logp(point_fn(0)))
        assert model_logp == pytest.approx(expected[0], rel=2.0e-12, abs=2.0e-12)
        assert actual[0] == pytest.approx(expected[0], rel=2.0e-12, abs=2.0e-12)
        assert actual[1] == pytest.approx(expected[1], rel=5.0e-10, abs=5.0e-9)
        np.testing.assert_allclose(
            actual[2],
            expected[2],
            rtol=5.0e-10,
            atol=5.0e-9,
        )
        np.testing.assert_allclose(
            actual[3],
            expected[3],
            rtol=5.0e-10,
            atol=5.0e-9,
        )


def test_model_uses_only_scalar_joint_potential() -> None:
    """No fake observed Normal or pointwise likelihood may enter the model."""
    _run_x64_case("model")


def _assert_model_uses_only_scalar_joint_potential() -> None:
    """Inspect model variables inside the isolated float64 runtime."""
    problem, state = _problem_state()
    data = _data(problem, state, rank=2)
    model = build_fixed_basis_aggregation_pymc_model(data)

    assert model.observed_RVs == []
    assert {variable.name for variable in model.free_RVs} == {
        "root_total",
        "leaf_share",
        "fixed_coefficient",
    }
    assert {variable.name for variable in model.potentials} == {"aggregation_joint_likelihood_potential"}
    assert model["aggregation_joint_likelihood_potential"].ndim == 0
    assert model["aggregation_joint_log_likelihood"].ndim == 0
    assert "observed" not in model.named_vars
    assert all(str(variable.dtype) == "float64" for variable in model.value_vars)


def test_sampler_disables_pointwise_log_likelihood() -> None:
    """The NUTS wrapper must prohibit a misleading ArviZ likelihood group."""
    _run_x64_case("sampler")


def _assert_sampler_disables_pointwise_log_likelihood() -> None:
    """Capture the exact PyMC call inside the isolated runtime."""
    import arviz as az

    problem, state = _problem_state()
    data = _data(problem, state, rank=2)
    model = build_fixed_basis_aggregation_pymc_model(data)
    captured: dict[str, Any] = {}
    expected = _valid_inference_data(az, data, chains=1, draws=2)

    def fake_sample(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return expected

    monkeypatch = pytest.MonkeyPatch()
    numpy_seed: Any = np.int64(901)
    try:
        monkeypatch.setattr("pymc.sample", fake_sample)
        actual = sample_fixed_basis_aggregation_nuts(
            model,
            data,
            draws=7,
            tune=11,
            seed=numpy_seed,
            target_accept=0.93,
            chains=1,
            cores=1,
            chain_method="parallel",
            progressbar=False,
            max_tree_depth=9,
            dense_mass=True,
            initvals=fixed_basis_nuts_initvals(data.fixed_basis),
        )
    finally:
        monkeypatch.undo()

    assert actual is expected
    assert captured["idata_kwargs"] == {"log_likelihood": False}
    assert captured["nuts_sampler"] == "numpyro"
    assert captured["return_inferencedata"] is True
    assert captured["nuts_sampler_kwargs"] == {
        "jitter": False,
        "chain_method": "parallel",
        "nuts_kwargs": {
            "max_tree_depth": 9,
            "dense_mass": True,
        },
    }
    assert "log_likelihood" not in actual.groups()
    manifest = validate_fixed_basis_aggregation_inference_data(data, actual)
    assert manifest == data.target_manifest
    assert actual.attrs["fixed_basis_aggregation_manifest_json"]
    assert len(actual.attrs["fixed_basis_aggregation_manifest_sha256"]) == 64

    original_manifest = actual.attrs["fixed_basis_aggregation_manifest_json"]
    actual.attrs["fixed_basis_aggregation_manifest_json"] = "{}"
    with pytest.raises(ValueError, match="manifest"):
        validate_fixed_basis_aggregation_inference_data(data, actual)
    actual.attrs["fixed_basis_aggregation_manifest_json"] = original_manifest
    actual.attrs["fixed_basis_aggregation_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        validate_fixed_basis_aggregation_inference_data(data, actual)


def test_output_validation_requires_complete_scientific_schema() -> None:
    """Missing, malformed, nonfinite, or inconsistent output must fail closed."""
    _run_x64_case("output_schema")


def _assert_output_validation_requires_complete_scientific_schema() -> None:
    """Exercise exact fixed and zero-fixed posterior validation."""
    import arviz as az

    results: list[tuple[FixedBasisAggregationNUTSData, Any]] = []
    for with_fixed in (True, False):
        problem, state = _problem_state(with_fixed=with_fixed)
        data = _data(problem, state, rank=2)
        result = _valid_inference_data(az, data)
        _attach_test_manifest(data, result)
        assert validate_fixed_basis_aggregation_inference_data(data, result) == data.target_manifest
        results.append((data, result))

    data, pristine = results[0]

    missing = copy.deepcopy(pristine)
    missing.posterior = missing.posterior.drop_vars("leaf_mass")
    with pytest.raises(ValueError, match="posterior variables.*missing"):
        validate_fixed_basis_aggregation_inference_data(data, missing)

    extra = copy.deepcopy(pristine)
    extra.posterior["unplanned_output"] = extra.posterior["root_total"]
    with pytest.raises(ValueError, match="posterior variables.*extra"):
        validate_fixed_basis_aggregation_inference_data(data, extra)

    wrong_dims = copy.deepcopy(pristine)
    wrong_dims.posterior["leaf_share"] = wrong_dims.posterior["leaf_share"].transpose(
        "chain",
        "leaf",
        "draw",
    )
    with pytest.raises(ValueError, match="leaf_share.*dimensions"):
        validate_fixed_basis_aggregation_inference_data(data, wrong_dims)

    wrong_coordinate = copy.deepcopy(pristine)
    changed_leaf = np.asarray(wrong_coordinate.posterior.coords["leaf"].values).copy()
    changed_leaf[0] = "wrong_leaf"
    wrong_coordinate.posterior = wrong_coordinate.posterior.assign_coords(
        leaf=changed_leaf,
    )
    with pytest.raises(ValueError, match="coordinate 'leaf'.*wrong"):
        validate_fixed_basis_aggregation_inference_data(data, wrong_coordinate)

    wrong_dtype = copy.deepcopy(pristine)
    wrong_dtype.posterior["root_total"] = wrong_dtype.posterior["root_total"].astype(np.float32)
    with pytest.raises(ValueError, match="root_total.*dtype float64"):
        validate_fixed_basis_aggregation_inference_data(data, wrong_dtype)

    nonfinite = copy.deepcopy(pristine)
    nonfinite.posterior["mean_observation"].values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="mean_observation.*finite"):
        validate_fixed_basis_aggregation_inference_data(data, nonfinite)

    bad_share = copy.deepcopy(pristine)
    bad_share.posterior["leaf_share"].values[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="leaf_share must sum to one"):
        validate_fixed_basis_aggregation_inference_data(data, bad_share)

    for variable, message in (
        ("leaf_mass", "leaf_mass = root_total"),
        ("leaf_scaling", "leaf_scaling = leaf_mass"),
        ("mean_observation", "mean_observation"),
    ):
        corrupted = copy.deepcopy(pristine)
        corrupted.posterior[variable].values.flat[0] += 0.1
        with pytest.raises(ValueError, match=message):
            validate_fixed_basis_aggregation_inference_data(data, corrupted)

    wrong_likelihood = copy.deepcopy(pristine)
    wrong_likelihood.posterior["aggregation_joint_log_likelihood"].values[0, 0] += 0.1
    with pytest.raises(ValueError, match="does not reproduce.*every retained draw"):
        validate_fixed_basis_aggregation_inference_data(data, wrong_likelihood)


def test_sampler_rejects_unbound_or_mutated_models_and_missing_output() -> None:
    """Target graph binding must fail before an unrelated graph can sample."""
    _run_x64_case("binding")


def _assert_sampler_rejects_unbound_or_mutated_models_and_missing_output() -> None:
    """Exercise strict model and output authentication in a float64 runtime."""
    import arviz as az
    import pymc as pm

    pm_runtime: Any = pm
    problem, state = _problem_state()
    data = _data(problem, state, rank=2)

    def controls(model: Any) -> dict[str, Any]:
        return {
            "model": model,
            "data": data,
            "draws": 2,
            "tune": 2,
            "seed": 19,
            "target_accept": 0.8,
            "chains": 1,
            "cores": 1,
            "chain_method": "parallel",
            "progressbar": False,
        }

    unrelated = build_fixed_basis_pymc_model(data.fixed_basis)
    with pytest.raises(ValueError, match="not bound"):
        sample_fixed_basis_aggregation_nuts(**controls(unrelated))

    observed_model = build_fixed_basis_aggregation_pymc_model(data)
    with observed_model:
        pm.Normal("forbidden_observed", observed=0.0)
    with pytest.raises(ValueError, match="observed RVs|diagonal likelihood"):
        sample_fixed_basis_aggregation_nuts(**controls(observed_model))

    missing_potential = build_fixed_basis_aggregation_pymc_model(data)
    missing_potential.potentials.clear()
    with pytest.raises(ValueError, match="exactly one scalar Potential"):
        sample_fixed_basis_aggregation_nuts(**controls(missing_potential))

    wrong_potential = build_fixed_basis_aggregation_pymc_model(data)
    wrong_potential.potentials.clear()
    with wrong_potential:
        pm.Potential("wrong_potential", pm_runtime.math.constant(0.0))
    with pytest.raises(ValueError, match="wrong scalar Potential"):
        sample_fixed_basis_aggregation_nuts(**controls(wrong_potential))

    missing_output = az.from_dict(posterior={"root_total": np.ones((1, 2))})
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("pymc.sample", lambda **_: missing_output)
        with pytest.raises(ValueError, match="posterior variables.*missing"):
            sample_fixed_basis_aggregation_nuts(**controls(build_fixed_basis_aggregation_pymc_model(data)))
    finally:
        monkeypatch.undo()


def test_real_two_draw_numpyro_smoke_persists_only_joint_likelihood() -> None:
    """The actual backend must compile the correlated Potential and persist it."""
    _run_x64_case("real_sampler")


def _assert_real_two_draw_numpyro_smoke_persists_only_joint_likelihood() -> None:
    """Run a compilation smoke, not a convergence or calibration experiment."""
    problem, state = _problem_state()
    data = _data(problem, state, rank=2)
    model = build_fixed_basis_aggregation_pymc_model(data)
    result = sample_fixed_basis_aggregation_nuts(
        model,
        data,
        draws=2,
        tune=2,
        seed=1301,
        target_accept=0.8,
        chains=1,
        cores=1,
        chain_method="parallel",
        progressbar=False,
        max_tree_depth=3,
        dense_mass=False,
    )

    assert "log_likelihood" not in result.groups()
    assert "observed_data" not in result.groups()
    result_as_any: Any = result
    assert "aggregation_joint_log_likelihood" in result_as_any.posterior
    likelihood = np.asarray(
        result_as_any.posterior["aggregation_joint_log_likelihood"],
        dtype=np.float64,
    )
    assert likelihood.shape == (1, 2)
    assert np.all(np.isfinite(likelihood))
    assert (
        validate_fixed_basis_aggregation_inference_data(
            data,
            result,
        )
        == data.target_manifest
    )


_ISOLATED_CASES = {
    "q0": _assert_q0_pymc_target_is_exact_existing_fixed_basis_target,
    "q0_gradient": _assert_q0_pytensor_density_and_gradient_match_independent_diagonal_oracle,
    "gradient": _assert_pytensor_log_density_and_gradient_match_numpy_oracle,
    "model": _assert_model_uses_only_scalar_joint_potential,
    "sampler": _assert_sampler_disables_pointwise_log_likelihood,
    "output_schema": _assert_output_validation_requires_complete_scientific_schema,
    "binding": _assert_sampler_rejects_unbound_or_mutated_models_and_missing_output,
    "real_sampler": _assert_real_two_draw_numpyro_smoke_persists_only_joint_likelihood,
}


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in _ISOLATED_CASES:
        raise SystemExit("expected one isolated aggregation-aware NUTS test case")
    _ISOLATED_CASES[sys.argv[1]]()
