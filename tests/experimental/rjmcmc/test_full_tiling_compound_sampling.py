"""Tests for the fixed-K full-tiling compound reference sampler."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import openghg_inversions.experimental.rjmcmc.full_tiling as full_tiling_geometry
import openghg_inversions.experimental.rjmcmc.full_tiling_compound_sampling as sampling
from openghg_inversions.experimental.rjmcmc.full_tiling_compound_sampling import (
    FullTilingCompoundConfig,
    FullTilingCompoundSamplingResult,
    continue_full_tiling_compound,
    sample_full_tiling_compound,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    FullTilingPosteriorState,
    full_tiling_problem_from_gamma_beta_adapter,
    initialize_full_tiling_posterior_state,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    gamma_beta_problem_from_rhime_inputs,
)


def _problem_state(*, k: int = 1):
    """Return a tiny problem with six fixed coefficients and one initial state."""
    sensitivity = np.arange(1.0, 49.0).reshape(3, 4, 4)
    fixed_design = np.arange(18.0).reshape(3, 6) / 20.0
    fixed_offset = np.array([1.0, 2.0, 3.0])
    dataset = xr.Dataset(
        {
            "fp_x_flux": (("nmeasure", "lat", "lon"), sensitivity),
            "mf": ("nmeasure", np.zeros(3)),
            "mf_error": ("nmeasure", np.ones(3)),
            "outer": (("nmeasure", "outer_region"), fixed_design),
            "boundary": ("nmeasure", fixed_offset),
        },
        coords={
            "nmeasure": np.arange(3),
            "lat": np.arange(4, dtype=float),
            "lon": np.arange(4, dtype=float),
            "outer_region": np.arange(6),
        },
    )
    adapter = gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=np.ones((4, 4)),
        k_min=1,
        k_max=16,
        concentration=4.0,
        root_variance=0.25,
        fixed_design_name="outer",
        fixed_offset_name="boundary",
        fixed_coefficient_prior_mean=np.ones(6),
        fixed_coefficient_prior_sd=np.full(6, 0.5),
        likelihood_power=0.0,
    )
    problem = full_tiling_problem_from_gamma_beta_adapter(adapter, concentration=4.0)
    return problem, initialize_full_tiling_posterior_state(problem, k=k)


def _assert_states_equal(
    actual: FullTilingPosteriorState,
    expected: FullTilingPosteriorState,
) -> None:
    """Assert exact equality of coordinates and posterior caches."""
    assert actual.problem is expected.problem
    assert actual.allocation.tiling == expected.allocation.tiling
    for name in (
        "leaf_masses",
        "fixed_coefficients",
        "dynamic_prediction",
        "fixed_prediction",
        "prediction",
        "residual",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.log_target == expected.log_target


def _assert_results_equal(
    actual: FullTilingCompoundSamplingResult,
    expected: FullTilingCompoundSamplingResult,
) -> None:
    """Assert exact equality of complete traces, states, and RNG checkpoints."""
    for name in actual.trace.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual.trace, name), getattr(expected.trace, name))
    _assert_states_equal(actual.final_state, expected.final_state)
    assert actual.checkpoint.rng_state == expected.checkpoint.rng_state
    assert actual.checkpoint.transitions_completed == expected.checkpoint.transitions_completed
    assert actual.checkpoint.schedule_phase == expected.checkpoint.schedule_phase


def test_six_fixed_coefficients_and_five_pairs_make_exactly_fourteen_slots() -> None:
    """The reviewed six-outer schedule has exactly fourteen atomic attempts."""
    problem, initial = _problem_state(k=1)
    result = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(iterations=14, seed=9),
    )

    assert result.trace.slot.tolist() == [
        "structural",
        "structural",
        "root",
        "pair_allocation",
        "pair_allocation",
        "pair_allocation",
        "pair_allocation",
        "pair_allocation",
        "fixed",
        "fixed",
        "fixed",
        "fixed",
        "fixed",
        "fixed",
    ]
    assert result.trace.move[2:].tolist() == [
        "root_total_refresh",
        *(["pair_allocation_refresh"] * 5),
        *(["fixed_coefficient"] * 6),
    ]
    assert result.trace.valid[:2].tolist() == [False, False]
    assert result.trace.valid[3:8].tolist() == [False] * 5
    assert result.trace.global_transition.tolist() == list(range(1, 15))
    assert result.trace.state_transition.tolist() == [0, 14]
    assert result.checkpoint.schedule_phase == 0


def test_seeded_replay_covers_invalid_slots_and_every_acceptance_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeded replay is exact while invalid attempts still receive uniforms."""
    import openghg_inversions.experimental.rjmcmc.full_tiling_compound_sampling as sampling

    problem, initial = _problem_state(k=1)
    observed: list[tuple[bool, float]] = []
    original = sampling.accept_or_reject

    def recording_accept(source, transition, *, log_uniform):
        """Record the supplied uniform without changing sampler behavior."""
        observed.append((transition.valid, log_uniform))
        return original(source, transition, log_uniform=log_uniform)

    monkeypatch.setattr(sampling, "accept_or_reject", recording_accept)
    config = FullTilingCompoundConfig(iterations=31, seed=20260724)
    first = sample_full_tiling_compound(problem, initial, config)
    first_uniforms = tuple(observed)
    observed.clear()
    replay = sample_full_tiling_compound(problem, initial, config)

    _assert_results_equal(first, replay)
    assert tuple(observed) == first_uniforms
    assert len(observed) == 31
    assert sum(not valid for valid, _ in observed) == int(np.count_nonzero(~first.trace.valid))
    assert all(np.isfinite(value) and value <= 0.0 for _, value in observed)


def test_awkward_phase_continuation_is_identical_to_uninterrupted_sampling() -> None:
    """Continuation after five attempts reproduces a two-cycle direct run."""
    problem, initial = _problem_state(k=4)
    direct = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(iterations=28, seed=177),
    )
    first = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(iterations=5, seed=177),
    )
    second = continue_full_tiling_compound(problem, first.checkpoint, iterations=23)

    for name in direct.trace.__dataclass_fields__:
        combined = np.concatenate((getattr(first.trace, name), getattr(second.trace, name)), axis=0)
        np.testing.assert_array_equal(combined, getattr(direct.trace, name))
    _assert_states_equal(second.final_state, direct.final_state)
    assert second.checkpoint.rng_state == direct.checkpoint.rng_state
    assert first.checkpoint.schedule_phase == 5
    assert second.checkpoint.schedule_phase == 0


def test_sampler_never_uses_exhaustive_geometry_oracles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling remains local when every exhaustive geometry oracle raises."""

    def forbidden(*args, **kwargs):
        """Fail immediately if an exhaustive oracle is called."""
        raise AssertionError("exhaustive geometry oracle was called")

    for name in ("edge_flip_paths", "relocation_paths", "enumerate_tilings"):
        monkeypatch.setattr(full_tiling_geometry, name, forbidden)
        monkeypatch.setattr(sampling, name, forbidden, raising=False)
    problem, initial = _problem_state(k=4)

    result = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(iterations=42, seed=81),
    )

    assert result.trace.global_transition.size == 42
    assert result.final_state.k == 4
