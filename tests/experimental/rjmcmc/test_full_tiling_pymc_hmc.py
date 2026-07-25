"""Tests for the mobile full-tiling structural plus static PyMC HMC kernel."""

from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Literal

import numpy as np
import pytest
import xarray as xr

import openghg_inversions.experimental.rjmcmc.full_tiling_pymc_hmc as sampling
from openghg_inversions.experimental.rjmcmc.core import lognormal_mu_sigma
from openghg_inversions.experimental.rjmcmc.full_tiling import (
    LeafTiling,
    Rectangle,
    TilingState,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    FullTilingProblem,
    FullTilingPosteriorState,
    PosteriorTransitionTerms,
    build_full_tiling_posterior_state,
    full_tiling_problem_from_gamma_beta_adapter,
    initialize_full_tiling_posterior_state,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    gamma_beta_problem_from_rhime_inputs,
)

_THIS_FILE = Path(__file__).resolve()
_X64_CHILD_ENV = "OPENGHG_INVERSIONS_PYMC_HMC_X64_CHILD"
_IS_X64_CHILD = os.environ.get(_X64_CHILD_ENV) == "1"
_requires_x64_child = pytest.mark.skipif(
    not _IS_X64_CHILD,
    reason="PyMC runtime assertion executes in the isolated float64 child",
)


def _pytensor_flags_with_float64(flags: str) -> str:
    """Return PyTensor flags with an unambiguous float64 configuration."""
    retained = []
    for item in flags.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name not in {"floatX", "warn_float64"}:
            retained.append(stripped)
    return ",".join(("floatX=float64", "warn_float64=ignore", *retained))


def _run_x64_test_file() -> None:
    """Run all PyMC assertions in one fresh float64 subprocess."""
    environment = os.environ.copy()
    environment[_X64_CHILD_ENV] = "1"
    environment["PYTENSOR_FLAGS"] = _pytensor_flags_with_float64(
        environment.get("PYTENSOR_FLAGS", ""),
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(_THIS_FILE)],
        cwd=_THIS_FILE.parents[3],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        "isolated full-tiling PyMC HMC tests failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.skipif(_IS_X64_CHILD, reason="parent-only subprocess dispatch")
def test_pymc_runtime_cases_use_a_fresh_float64_subprocess() -> None:
    """PyMC cases cannot inherit process-global float32 from RHIME tests."""
    _run_x64_test_file()


def _problem_state(
    *,
    k: int = 4,
) -> tuple[FullTilingProblem, FullTilingPosteriorState]:
    """Return a tiny non-degenerate problem with three fixed coefficients."""
    sensitivity = np.arange(1.0, 49.0).reshape(3, 4, 4) / 40.0
    fixed_design = np.arange(9.0).reshape(3, 3) / 20.0
    fixed_offset = np.array([0.1, 0.2, 0.3])
    dataset = xr.Dataset(
        {
            "fp_x_flux": (("nmeasure", "lat", "lon"), sensitivity),
            "mf": ("nmeasure", np.array([1.2, 1.4, 1.8])),
            "mf_error": ("nmeasure", np.full(3, 0.8)),
            "outer": (("nmeasure", "outer_region"), fixed_design),
            "boundary": ("nmeasure", fixed_offset),
        },
        coords={
            "nmeasure": np.arange(3),
            "lat": np.arange(4, dtype=float),
            "lon": np.arange(4, dtype=float),
            "outer_region": np.arange(3),
        },
    )
    adapter = gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=np.arange(1.0, 17.0).reshape(4, 4),
        k_min=1,
        k_max=16,
        concentration=4.0,
        root_variance=0.25,
        fixed_design_name="outer",
        fixed_offset_name="boundary",
        fixed_coefficient_prior_mean=np.array([0.8, 1.1, 1.5]),
        fixed_coefficient_prior_sd=np.array([0.3, 0.4, 0.5]),
    )
    problem = full_tiling_problem_from_gamma_beta_adapter(
        adapter,
        concentration=4.0,
    )
    return problem, initialize_full_tiling_posterior_state(problem, k=k)


def _nonuniform_state(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
) -> FullTilingPosteriorState:
    """Return an arbitrary supported point with non-unit total mass."""
    masses = np.linspace(0.3, 1.2, state.k, dtype=np.float64)
    fixed = np.array([0.7, 1.3, 1.8], dtype=np.float64)
    return build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(state.tiling_state.tiling, masses),
        fixed_coefficients=fixed,
    )


def _assert_states_equal(
    actual: FullTilingPosteriorState,
    expected: FullTilingPosteriorState,
) -> None:
    """Assert exact topology, coordinates, caches, and target equality."""
    assert actual.problem is expected.problem
    assert actual.tiling_state.tiling == expected.tiling_state.tiling
    for name in (
        "leaf_masses",
        "fixed_coefficients",
        "dynamic_prediction",
        "fixed_prediction",
        "prediction",
        "residual",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in (
        "log_gaussian_likelihood",
        "log_likelihood",
        "log_root_prior",
        "log_allocation_prior",
        "log_fixed_coefficient_prior",
        "log_target",
    ):
        assert getattr(actual, name) == getattr(expected, name)


def _assert_results_equal(
    actual: sampling.FullTilingPyMCHMCSamplingResult,
    expected: sampling.FullTilingPyMCHMCSamplingResult,
) -> None:
    """Assert exact trace, state, settings, and PCG64 continuation equality."""
    for name in actual.trace.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(actual.trace, name),
            getattr(expected.trace, name),
        )
    np.testing.assert_array_equal(actual.trace.hmc_seed, expected.trace.hmc_seed)
    assert actual.trace.hmc_seed.dtype == np.dtype(np.uint64)
    _assert_states_equal(actual.final_state, expected.final_state)
    np.testing.assert_array_equal(
        actual.checkpoint.log_leaf_mass,
        expected.checkpoint.log_leaf_mass,
    )
    np.testing.assert_array_equal(
        actual.checkpoint.log_fixed_coefficient,
        expected.checkpoint.log_fixed_coefficient,
    )
    np.testing.assert_array_equal(
        actual.trace.log_leaf_mass,
        expected.trace.log_leaf_mass,
    )
    np.testing.assert_array_equal(
        actual.trace.log_fixed_coefficient,
        expected.trace.log_fixed_coefficient,
    )
    np.testing.assert_array_equal(
        actual.trace.log_leaf_mass[-1],
        actual.checkpoint.log_leaf_mass,
    )
    np.testing.assert_array_equal(
        actual.trace.log_fixed_coefficient[-1],
        actual.checkpoint.log_fixed_coefficient,
    )
    np.testing.assert_array_equal(
        np.exp(actual.checkpoint.log_leaf_mass),
        actual.checkpoint.state.leaf_masses,
    )
    np.testing.assert_array_equal(
        np.exp(actual.checkpoint.log_fixed_coefficient),
        actual.checkpoint.state.fixed_coefficients,
    )
    assert actual.checkpoint.rng_state == expected.checkpoint.rng_state
    assert actual.checkpoint.sweeps_completed == expected.checkpoint.sweeps_completed
    assert actual.checkpoint.kernel_settings == expected.checkpoint.kernel_settings
    assert actual.checkpoint.runtime_identity == expected.checkpoint.runtime_identity
    assert actual.checkpoint.runtime_identity == sampling.full_tiling_pymc_hmc_runtime_identity()


def _config(*, iterations: int, seed: int = 761) -> sampling.FullTilingPyMCHMCConfig:
    """Return a short frozen-HMC configuration used throughout the tests."""
    return sampling.FullTilingPyMCHMCConfig(
        iterations=iterations,
        step_size=0.002,
        leapfrog_steps=2,
        leaf_position_scale=1.7,
        fixed_coefficient_position_scale=(0.6, 1.1, 2.4),
        seed=seed,
    )


@_requires_x64_child
def test_compiled_transformed_target_includes_both_coordinate_jacobians() -> None:
    """PyMC logp equals the scientific target plus mass and fixed Jacobians."""
    problem, initial = _problem_state(k=4)
    state = _nonuniform_state(problem, initial)
    model = sampling.build_full_tiling_pymc_hmc_model(problem, state)
    point = model.initial_point()
    compiled = float(model.compile_logp()(point))
    log_mass = np.log(state.leaf_masses)
    log_total = float(np.logaddexp.reduce(log_mass))
    mass_jacobian = float(log_mass.sum() - (state.k - 1) * log_total)
    fixed_jacobian = float(np.log(state.fixed_coefficients).sum())

    assert compiled - state.log_target == pytest.approx(
        mass_jacobian + fixed_jacobian,
        abs=2.0e-10,
    )
    assert compiled == pytest.approx(
        state.log_target + mass_jacobian + fixed_jacobian,
        abs=2.0e-10,
    )


@_requires_x64_child
def test_compiled_target_matches_independent_closed_form_at_arbitrary_xy() -> None:
    """An independent normalized density matches PyMC away from encoded state."""
    problem, state = _problem_state(k=4)
    model = sampling.build_full_tiling_pymc_hmc_model(problem, state)
    point = model.initial_point()
    x = np.array([-1.1, -0.6, -0.9, -0.35])
    y = np.log(np.array([0.65, 1.25, 1.7]))
    point["x"] = x
    point["y"] = y

    log_total = float(np.logaddexp.reduce(x))
    total = math.exp(log_total)
    log_share = x - log_total
    prior = problem.base.prior
    log_gamma = (
        prior.root_shape * math.log(prior.root_rate)
        - math.lgamma(prior.root_shape)
        + (prior.root_shape - 1.0) * log_total
        - prior.root_rate * total
    )
    alpha = problem.allocation_prior.leaf_alphas(state.tiling_state.tiling)
    log_dirichlet = (
        math.lgamma(float(alpha.sum()))
        - sum(math.lgamma(float(value)) for value in alpha)
        + float(np.dot(alpha - 1.0, log_share))
    )
    dynamic_design = np.column_stack(
        [problem.design_column(leaf) for leaf in state.tiling_state.tiling.leaves],
    )
    fixed_block = problem.base.fixed_block
    fixed_offset = problem.base.fixed_offset
    assert fixed_block is not None
    assert fixed_offset is not None
    prediction = fixed_offset + dynamic_design @ np.exp(x) + fixed_block.design @ np.exp(y)
    residual = (prediction - problem.observations) / problem.observation_sd
    log_gaussian = (
        -0.5 * float(np.dot(residual, residual))
        - float(np.log(problem.observation_sd).sum())
        - 0.5 * problem.observations.size * math.log(2.0 * math.pi)
    )
    log_likelihood = problem.base.likelihood_power * log_gaussian
    log_fixed = 0.0
    for coordinate, mean, sd in zip(
        y,
        fixed_block.coefficient_prior_mean,
        fixed_block.coefficient_prior_sd,
        strict=True,
    ):
        mu, sigma = lognormal_mu_sigma(float(mean), float(sd))
        log_fixed += (
            -0.5 * ((float(coordinate) - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)
        )
    log_x_jacobian = float(x.sum() - (state.k - 1) * log_total)
    expected = log_likelihood + log_gamma + log_dirichlet + log_fixed + log_x_jacobian

    assert float(model.compile_logp()(point)) == pytest.approx(
        expected,
        rel=2.0e-12,
        abs=2.0e-12,
    )


@_requires_x64_child
def test_canonical_leaf_permutation_realigns_target_and_scalar_metric() -> None:
    """Aligned topology arrays and log coordinates preserve target and metric."""
    problem, initial = _problem_state(k=4)
    state = _nonuniform_state(problem, initial)
    model = sampling.build_full_tiling_pymc_hmc_model(problem, state)
    point = model.initial_point()
    logp = model.compile_logp()
    canonical_logp = float(logp(point))
    design, alpha, log_normalizer = sampling._topology_arrays(problem, state)
    leaves = state.tiling_state.tiling.leaves
    permutation = np.array([2, 0, 3, 1])
    reordered_leaves = tuple(leaves[index] for index in permutation)
    reordered_design = design[:, permutation]
    reordered_alpha = alpha[permutation]
    reordered_point = {name: np.array(value, copy=True) for name, value in point.items()}
    reordered_point["x"] = reordered_point["x"][permutation]

    assert reordered_leaves != leaves
    for index, leaf in enumerate(reordered_leaves):
        np.testing.assert_array_equal(
            reordered_design[:, index],
            problem.design_column(leaf),
        )
        assert reordered_alpha[index] == problem.allocation_prior.alpha(leaf)
        assert np.exp(reordered_point["x"][index]) == state.tiling_state.mass(leaf)

    mass_by_leaf = {
        leaf: mass
        for leaf, mass in zip(
            reordered_leaves,
            state.leaf_masses[permutation],
            strict=True,
        )
    }
    rebuilt = build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(
            state.tiling_state.tiling,
            np.asarray([mass_by_leaf[leaf] for leaf in leaves]),
        ),
        fixed_coefficients=state.fixed_coefficients,
    )
    _assert_states_equal(rebuilt, state)

    sampling._set_topology_data_atomically(
        model,
        reordered_design,
        reordered_alpha,
        log_normalizer,
    )
    assert float(logp(reordered_point)) == pytest.approx(
        canonical_logp,
        rel=0.0,
        abs=2.0e-12,
    )

    settings = sampling.FullTilingPyMCHMCKernelSettings(
        fixed_k=state.k,
        step_size=0.01,
        leapfrog_steps=2,
        leaf_position_scale=2.5,
        fixed_coefficient_position_scale=(0.4, 0.8, 1.6),
    )
    leaf_metric = settings.position_scale_diagonal[: state.k]
    np.testing.assert_array_equal(leaf_metric[permutation], leaf_metric)


@_requires_x64_child
def test_compiled_gradient_matches_central_directional_difference() -> None:
    """PyMC's gradient matches a stable finite difference inside support."""
    problem, state = _problem_state(k=4)
    model = sampling.build_full_tiling_pymc_hmc_model(problem, state)
    point = model.initial_point()
    point["x"] = np.array([-1.1, -0.6, -0.9, -0.35])
    point["y"] = np.log(np.array([0.65, 1.25, 1.7]))
    direction = np.array([0.3, -0.4, 0.2, 0.1, -0.25, 0.35, -0.15])
    direction /= np.linalg.norm(direction)
    gradient = np.asarray(
        model.compile_dlogp(vars=[model["x"], model["y"]])(point),
        dtype=np.float64,
    )
    step = 2.0e-6
    plus = {name: np.array(value, copy=True) for name, value in point.items()}
    minus = {name: np.array(value, copy=True) for name, value in point.items()}
    plus["x"] += step * direction[: state.k]
    minus["x"] -= step * direction[: state.k]
    plus["y"] += step * direction[state.k :]
    minus["y"] -= step * direction[state.k :]
    logp = model.compile_logp()
    finite_difference = (float(logp(plus)) - float(logp(minus))) / (2.0 * step)

    assert float(np.dot(gradient, direction)) == pytest.approx(
        finite_difference,
        rel=2.0e-7,
        abs=2.0e-7,
    )


@_requires_x64_child
def test_model_rejects_unrepresentable_log_coordinates() -> None:
    """Overflowing or underflowing log coordinates have negative-infinite logp."""
    problem, state = _problem_state(k=4)
    model = sampling.build_full_tiling_pymc_hmc_model(problem, state)
    logp = model.compile_logp()
    initial = model.initial_point()

    assert math.isfinite(float(logp(initial)))
    for name, value in (
        ("x", -1000.0),
        ("x", 1000.0),
        ("y", -1000.0),
        ("y", 1000.0),
    ):
        point = {key: np.array(item, copy=True) for key, item in initial.items()}
        point[name][...] = value
        assert float(logp(point)) == -math.inf


@_requires_x64_child
@pytest.mark.parametrize("outcome", ["accepted", "rejected", "invalid"])
def test_every_structural_outcome_is_followed_by_one_hmc_transition(
    monkeypatch: pytest.MonkeyPatch,
    outcome: Literal["accepted", "rejected", "invalid"],
) -> None:
    """Accepted, rejected, and invalid structural attempts all run HMC once."""
    problem, initial = _problem_state(k=4)
    original_draw = sampling._draw_structural_transition
    valid_transition = None
    search_rng = np.random.default_rng(923)
    for _ in range(100):
        candidate, _ = original_draw(problem, initial, rng=search_rng)
        if candidate.valid:
            valid_transition = candidate
            break
    assert valid_transition is not None
    transition = (
        PosteriorTransitionTerms(
            candidate=initial,
            move="edge_flip",
            delta_log_likelihood=0.0,
            valid=False,
            reason="forced invalid structural attempt",
        )
        if outcome == "invalid"
        else valid_transition
    )

    monkeypatch.setattr(
        sampling,
        "_draw_structural_transition",
        lambda *args, **kwargs: (transition, None),
    )
    original_accept = sampling.accept_or_reject

    def controlled_accept(source, terms, *, log_uniform):
        """Force the selected valid outcome while retaining invalid semantics."""
        if outcome == "accepted":
            return terms.candidate
        if outcome == "rejected":
            return source
        return original_accept(source, terms, log_uniform=log_uniform)

    monkeypatch.setattr(sampling, "accept_or_reject", controlled_accept)
    result = sampling.sample_full_tiling_pymc_hmc(
        problem,
        initial,
        _config(iterations=1),
    )

    assert result.trace.global_sweep.tolist() == [1]
    assert result.trace.structural_valid.tolist() == [outcome != "invalid"]
    assert result.trace.structural_accepted.tolist() == [outcome == "accepted"]
    assert result.trace.hmc_n_steps.tolist() == [2]
    assert result.trace.hmc_step_size.tolist() == pytest.approx([0.002])
    expected_tiling = (
        valid_transition.candidate.tiling_state.tiling
        if outcome == "accepted"
        else initial.tiling_state.tiling
    )
    assert result.final_state.tiling_state.tiling == expected_tiling


@_requires_x64_child
def test_hmc_preserves_topology_and_rebuilds_all_scientific_caches() -> None:
    """Every HMC endpoint keeps its input tiling and closes through a full rebuild."""
    problem, initial = _problem_state(k=4)
    result = sampling.sample_full_tiling_pymc_hmc(
        problem,
        initial,
        _config(iterations=3, seed=812),
    )

    np.testing.assert_array_equal(result.trace.state_sweep, np.arange(4))
    np.testing.assert_array_equal(result.trace.global_sweep, np.arange(1, 4))
    for before, after, accepted in zip(
        result.trace.rectangle_bounds[:-1],
        result.trace.rectangle_bounds[1:],
        result.trace.structural_accepted,
        strict=True,
    ):
        if not accepted:
            np.testing.assert_array_equal(after, before)
    for bounds, masses, fixed, log_target in zip(
        result.trace.rectangle_bounds,
        result.trace.leaf_masses,
        result.trace.fixed_coefficients,
        result.trace.log_target,
        strict=True,
    ):
        tiling = LeafTiling(
            problem.shape,
            tuple(Rectangle(*(int(value) for value in row)) for row in bounds),
        )
        rebuilt_draw = build_full_tiling_posterior_state(
            problem,
            allocation=TilingState(tiling, masses),
            fixed_coefficients=fixed,
        )
        assert rebuilt_draw.log_target == log_target
    rebuilt = build_full_tiling_posterior_state(
        problem,
        allocation=result.final_state.tiling_state,
        fixed_coefficients=result.final_state.fixed_coefficients,
    )
    _assert_states_equal(result.final_state, rebuilt)


@_requires_x64_child
def test_trace_preserves_authoritative_hmc_coordinates_read_only() -> None:
    """Trace coordinates retain exact boundary points with fixed immutable shapes."""
    problem, initial = _problem_state(k=4)
    result = sampling.sample_full_tiling_pymc_hmc(
        problem,
        initial,
        _config(iterations=2, seed=268),
    )

    assert result.trace.log_leaf_mass.shape == (3, 4)
    assert result.trace.log_fixed_coefficient.shape == (3, 3)
    np.testing.assert_array_equal(
        np.exp(result.trace.log_leaf_mass),
        result.trace.leaf_masses,
    )
    np.testing.assert_array_equal(
        np.exp(result.trace.log_fixed_coefficient),
        result.trace.fixed_coefficients,
    )
    np.testing.assert_array_equal(
        result.trace.log_leaf_mass[-1],
        result.checkpoint.log_leaf_mass,
    )
    np.testing.assert_array_equal(
        result.trace.log_fixed_coefficient[-1],
        result.checkpoint.log_fixed_coefficient,
    )
    assert math.isfinite(result.kernel_setup_seconds)
    assert result.kernel_setup_seconds >= 0.0
    assert math.isfinite(result.transition_seconds)
    assert result.transition_seconds >= 0.0
    for coordinates in (
        result.trace.log_leaf_mass,
        result.trace.log_fixed_coefficient,
    ):
        assert not coordinates.flags.writeable
        with pytest.raises(ValueError):
            coordinates[0, 0] = 0.0

    with pytest.raises(ValueError, match="log_leaf_mass must have shape"):
        replace(
            result.trace,
            log_leaf_mass=result.trace.log_leaf_mass[:, :-1],
        )
    with pytest.raises(
        ValueError,
        match="log_fixed_coefficient must match",
    ):
        replace(
            result.trace,
            log_fixed_coefficient=result.trace.log_fixed_coefficient[:, :-1],
        )
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        replace(result, kernel_setup_seconds=-1.0)


@_requires_x64_child
def test_seeded_replay_is_exact_including_pcg64_checkpoint() -> None:
    """The same seed reproduces every state, diagnostic, and RNG bit exactly."""
    problem, initial = _problem_state(k=4)
    config = _config(iterations=3, seed=20260725)

    first = sampling.sample_full_tiling_pymc_hmc(problem, initial, config)
    replay = sampling.sample_full_tiling_pymc_hmc(problem, initial, config)

    _assert_results_equal(first, replay)
    np.testing.assert_array_equal(
        first.trace.log_leaf_mass,
        replay.trace.log_leaf_mass,
    )
    np.testing.assert_array_equal(
        first.trace.log_fixed_coefficient,
        replay.trace.log_fixed_coefficient,
    )


@_requires_x64_child
def test_awkward_split_continuation_matches_uninterrupted_sampling() -> None:
    """A one-plus-two split exactly matches a direct three-sweep execution."""
    problem, initial = _problem_state(k=4)
    direct = sampling.sample_full_tiling_pymc_hmc(
        problem,
        initial,
        _config(iterations=3, seed=988),
    )
    first = sampling.sample_full_tiling_pymc_hmc(
        problem,
        initial,
        _config(iterations=1, seed=988),
    )
    second = sampling.continue_full_tiling_pymc_hmc(
        problem,
        first.checkpoint,
        iterations=2,
    )

    assert math.isfinite(second.kernel_setup_seconds)
    assert second.kernel_setup_seconds >= 0.0
    assert math.isfinite(second.transition_seconds)
    assert second.transition_seconds >= 0.0
    for name in direct.trace.__dataclass_fields__:
        first_values = getattr(first.trace, name)
        second_values = getattr(second.trace, name)
        combined = (
            np.concatenate((first_values, second_values[1:]), axis=0)
            if name
            in {
                "state_sweep",
                "rectangle_bounds",
                "leaf_masses",
                "fixed_coefficients",
                "log_leaf_mass",
                "log_fixed_coefficient",
                "log_target",
            }
            else np.concatenate((first_values, second_values), axis=0)
        )
        np.testing.assert_array_equal(combined, getattr(direct.trace, name))
    for name in ("log_leaf_mass", "log_fixed_coefficient"):
        np.testing.assert_array_equal(
            getattr(first.trace, name)[-1],
            getattr(second.trace, name)[0],
        )
        np.testing.assert_array_equal(
            np.concatenate(
                (
                    getattr(first.trace, name),
                    getattr(second.trace, name)[1:],
                ),
                axis=0,
            ),
            getattr(direct.trace, name),
        )
    np.testing.assert_array_equal(
        np.concatenate((first.trace.hmc_seed, second.trace.hmc_seed)),
        direct.trace.hmc_seed,
    )
    _assert_states_equal(second.final_state, direct.final_state)
    np.testing.assert_array_equal(
        second.checkpoint.log_leaf_mass,
        direct.checkpoint.log_leaf_mass,
    )
    np.testing.assert_array_equal(
        second.checkpoint.log_fixed_coefficient,
        direct.checkpoint.log_fixed_coefficient,
    )
    np.testing.assert_array_equal(
        np.exp(second.checkpoint.log_leaf_mass),
        second.final_state.leaf_masses,
    )
    np.testing.assert_array_equal(
        np.exp(second.checkpoint.log_fixed_coefficient),
        second.final_state.fixed_coefficients,
    )
    assert second.checkpoint.rng_state == direct.checkpoint.rng_state
    assert second.checkpoint.sweeps_completed == direct.checkpoint.sweeps_completed == 3
    assert second.checkpoint.runtime_identity == direct.checkpoint.runtime_identity


def test_resolved_position_scale_is_fixed_and_topology_neutral() -> None:
    """Leaf positions share one scale while fixed identities retain their entries."""
    settings = sampling.FullTilingPyMCHMCKernelSettings(
        fixed_k=4,
        step_size=0.01,
        leapfrog_steps=3,
        leaf_position_scale=2.5,
        fixed_coefficient_position_scale=(0.4, 0.8, 1.6),
    )

    np.testing.assert_array_equal(
        settings.position_scale_diagonal,
        np.array([2.5, 2.5, 2.5, 2.5, 0.4, 0.8, 1.6]),
    )
    assert not settings.position_scale_diagonal.flags.writeable


@_requires_x64_child
def test_hmc_is_frozen_with_exact_leapfrog_count_and_no_tuning() -> None:
    """The constructed HMC step is static and reports the fixed path each sweep."""
    problem, initial = _problem_state(k=4)
    settings = sampling.FullTilingPyMCHMCKernelSettings(
        fixed_k=4,
        step_size=0.003,
        leapfrog_steps=3,
        leaf_position_scale=1.0,
        fixed_coefficient_position_scale=(1.0, 1.0, 1.0),
    )
    _, compound, _, _ = sampling._build_compound_kernel(
        problem,
        initial,
        settings,
        np.random.default_rng(44),
    )
    hmc = compound.methods[1]

    assert compound.tune is False
    assert hmc.tune is False
    assert hmc.adapt_step_size is False
    assert hmc._step_rand is None
    assert hmc.max_steps == 3
    np.testing.assert_array_equal(
        hmc.potential.v,
        settings.position_scale_diagonal,
    )
    momentum = np.arange(1.0, 8.0)
    np.testing.assert_array_equal(
        hmc.potential.velocity(momentum),
        settings.position_scale_diagonal * momentum,
    )
    assert hmc.potential.energy(momentum) == pytest.approx(
        0.5
        * float(
            np.dot(
                momentum,
                settings.position_scale_diagonal * momentum,
            )
        ),
    )
    result = sampling.sample_full_tiling_pymc_hmc(
        problem,
        initial,
        sampling.FullTilingPyMCHMCConfig(
            iterations=3,
            step_size=0.003,
            leapfrog_steps=3,
            seed=44,
        ),
    )
    np.testing.assert_array_equal(result.trace.hmc_n_steps, np.full(3, 3))
    np.testing.assert_allclose(
        result.trace.hmc_step_size,
        np.full(3, 0.003),
        rtol=0.0,
        atol=1.0e-18,
    )


@_requires_x64_child
def test_divergent_hmc_returns_rejected_state_and_valid_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-finite divergent trajectory is retained as a restartable rejection."""
    problem, initial = _problem_state(k=4)
    invalid = PosteriorTransitionTerms(
        candidate=initial,
        move="edge_flip",
        delta_log_likelihood=0.0,
        valid=False,
        reason="forced invalid structural attempt",
    )
    monkeypatch.setattr(
        sampling,
        "_draw_structural_transition",
        lambda *args, **kwargs: (invalid, None),
    )

    result = sampling.sample_full_tiling_pymc_hmc(
        problem,
        initial,
        sampling.FullTilingPyMCHMCConfig(
            iterations=1,
            step_size=10.0,
            leapfrog_steps=2,
            seed=4,
        ),
    )

    assert result.trace.hmc_diverging.tolist() == [True]
    assert result.trace.hmc_accepted.tolist() == [False]
    _assert_states_equal(result.final_state, initial)
    np.testing.assert_array_equal(
        np.exp(result.checkpoint.log_leaf_mass),
        result.checkpoint.state.leaf_masses,
    )
    np.testing.assert_array_equal(
        np.exp(result.checkpoint.log_fixed_coefficient),
        result.checkpoint.state.fixed_coefficients,
    )
    assert isinstance(
        result.checkpoint.rng_state.generator(),
        np.random.Generator,
    )


@_requires_x64_child
def test_leapfrog_count_survives_pymc_step_size_rounding() -> None:
    """One-ULP step-size reconstruction still runs the requested path length."""
    problem, initial = _problem_state(k=4)
    result = sampling.sample_full_tiling_pymc_hmc(
        problem,
        initial,
        sampling.FullTilingPyMCHMCConfig(
            iterations=1,
            step_size=0.005227487401508368,
            leapfrog_steps=88,
            seed=41,
        ),
    )

    assert result.trace.hmc_n_steps.tolist() == [88]


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"iterations": True}, TypeError, "iterations must be an integer"),
        ({"iterations": 0}, ValueError, "iterations must be positive"),
        ({"step_size": True}, TypeError, "step_size must be a real number"),
        ({"step_size": math.inf}, ValueError, "step_size must be finite"),
        ({"leapfrog_steps": 1.5}, TypeError, "leapfrog_steps must be an integer"),
        ({"leapfrog_steps": 0}, ValueError, "leapfrog_steps must be positive"),
        (
            {"leaf_position_scale": 0.0},
            ValueError,
            "leaf_position_scale must be finite",
        ),
        (
            {"fixed_coefficient_position_scale": (1.0, math.nan, 1.0)},
            ValueError,
            "fixed_coefficient_position_scale must be finite",
        ),
        ({"seed": -1}, ValueError, "seed must be non-negative"),
    ],
)
def test_config_rejects_invalid_static_hmc_settings(
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    """Invalid scalar, count, mass, and seed settings fail at construction."""
    values = {
        "iterations": 2,
        "step_size": 0.01,
        "leapfrog_steps": 3,
        "leaf_position_scale": 1.0,
        "fixed_coefficient_position_scale": (1.0, 1.0, 1.0),
        "seed": 4,
    }
    values.update(changes)

    with pytest.raises(error, match=message):
        sampling.FullTilingPyMCHMCConfig(**values)


def test_sampling_rejects_fixed_position_scale_length_mismatch() -> None:
    """Problem resolution rejects a fixed-scale vector of the wrong length."""
    problem, initial = _problem_state(k=4)
    config = sampling.FullTilingPyMCHMCConfig(
        iterations=1,
        step_size=0.01,
        leapfrog_steps=2,
        fixed_coefficient_position_scale=(1.0, 2.0),
    )

    with pytest.raises(ValueError, match="one entry per fixed coefficient"):
        sampling.sample_full_tiling_pymc_hmc(problem, initial, config)


@_requires_x64_child
def test_checkpoint_rejects_problem_k_settings_and_schedule_mismatches() -> None:
    """Continuation boundaries fail closed on all kernel-defining identities."""
    problem, initial = _problem_state(k=4)
    result = sampling.sample_full_tiling_pymc_hmc(
        problem,
        initial,
        _config(iterations=1, seed=66),
    )
    checkpoint = result.checkpoint
    other_problem, _ = _problem_state(k=4)

    with pytest.raises(ValueError, match="exact checkpoint problem"):
        sampling.continue_full_tiling_pymc_hmc(
            other_problem,
            checkpoint,
            iterations=1,
        )
    with pytest.raises(ValueError, match="state K must match"):
        replace(
            checkpoint,
            kernel_settings=replace(checkpoint.kernel_settings, fixed_k=3),
        )
    with pytest.raises(ValueError, match="fixed block must match"):
        replace(
            checkpoint,
            kernel_settings=replace(
                checkpoint.kernel_settings,
                fixed_coefficient_position_scale=(1.0, 1.0),
            ),
        )
    with pytest.raises(ValueError, match="log_leaf_mass must exactly encode"):
        replace(
            checkpoint,
            log_leaf_mass=checkpoint.log_leaf_mass + 1.0e-6,
        )
    with pytest.raises(
        ValueError,
        match="log_fixed_coefficient must exactly encode",
    ):
        replace(
            checkpoint,
            log_fixed_coefficient=checkpoint.log_fixed_coefficient + 1.0e-6,
        )
    with pytest.raises(ValueError, match="runtime identity is incompatible"):
        replace(
            checkpoint,
            runtime_identity=replace(
                checkpoint.runtime_identity,
                metric_semantics_id="future_metric_v2",
            ),
        )
    with pytest.raises(ValueError, match="schedule is incompatible"):
        replace(checkpoint, schedule_id="future_compound_hmc_v2")


@_requires_x64_child
def test_checkpoint_freezes_hmc_settings_for_continuation() -> None:
    """Continuation retains the exact step, path, and position-scale settings."""
    problem, initial = _problem_state(k=4)
    first = sampling.sample_full_tiling_pymc_hmc(
        problem,
        initial,
        _config(iterations=1, seed=17),
    )
    second = sampling.continue_full_tiling_pymc_hmc(
        problem,
        first.checkpoint,
        iterations=1,
    )

    assert second.checkpoint.kernel_settings == first.checkpoint.kernel_settings
    np.testing.assert_allclose(
        second.trace.hmc_step_size,
        np.array([first.checkpoint.kernel_settings.step_size]),
        rtol=0.0,
        atol=1.0e-18,
    )
    np.testing.assert_array_equal(
        second.trace.hmc_n_steps,
        np.array([first.checkpoint.kernel_settings.leapfrog_steps]),
    )
