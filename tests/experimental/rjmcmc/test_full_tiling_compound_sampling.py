"""Tests for the fixed-K full-tiling compound reference sampler."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

import openghg_inversions.experimental.rjmcmc.full_tiling as full_tiling_geometry
import openghg_inversions.experimental.rjmcmc.full_tiling_compound_sampling as sampling
from openghg_inversions.experimental.rjmcmc.full_tiling_compound_sampling import (
    FIXED_BASIS_COMPOUND_SCHEDULE_ID,
    FULL_TILING_COMPOUND_SCHEDULE_ID,
    FullTilingCompoundConfig,
    FullTilingCompoundSamplingResult,
    continue_full_tiling_compound,
    full_tiling_compound_cycle_length,
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
        "root_total_slice",
        *(["pair_allocation_refresh"] * 5),
        *(["fixed_coefficient"] * 6),
    ]
    assert result.trace.valid[:2].tolist() == [False, False]
    assert result.trace.valid[3:8].tolist() == [False] * 5
    assert result.trace.global_transition.tolist() == list(range(1, 15))
    assert result.trace.state_transition.tolist() == [0, 14]
    assert result.checkpoint.schedule_phase == 0


def test_seeded_replay_covers_invalid_slots_and_non_slice_acceptance_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeded replay is exact and only non-slice attempts receive MH uniforms."""
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
    assert len(observed) == 28
    assert sum(not valid for valid, _ in observed) == int(np.count_nonzero(~first.trace.valid))
    assert all(np.isfinite(value) and value <= 0.0 for _, value in observed)


@pytest.mark.parametrize("endpoint", [0.0, 1.0])
def test_numerical_beta_endpoints_are_explicit_self_attempts(
    endpoint: float,
) -> None:
    """A binary64 Beta endpoint is rejected without clipping or redrawing."""

    class EndpointRng:
        """Return one selected pair and a forced numerical Beta endpoint."""

        def integers(self, high: int) -> int:
            """Select the first catalogue entry."""
            assert high > 0
            return 0

        def beta(self, first: float, second: float) -> float:
            """Return the requested endpoint after validating positive shapes."""
            assert first > 0.0
            assert second > 0.0
            return endpoint

    problem, initial = _problem_state(k=2)
    transition, statistics = sampling._draw_pair_refresh_transition(
        problem,
        initial,
        rng=EndpointRng(),
    )

    assert not transition.valid
    assert transition.candidate is initial
    assert transition.reason == "new fraction lies outside support"
    assert transition.log_acceptance_ratio == -np.inf
    assert statistics.pair_catalogue_size == 1


@pytest.mark.parametrize("first_iterations", [2, 3, 5])
def test_awkward_phase_continuation_is_identical_to_uninterrupted_sampling(
    first_iterations: int,
) -> None:
    """Continuation around variable-draw root slots reproduces a direct run."""
    problem, initial = _problem_state(k=4)
    direct = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(iterations=28, seed=177),
    )
    first = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(iterations=first_iterations, seed=177),
    )
    second = continue_full_tiling_compound(
        problem,
        first.checkpoint,
        iterations=28 - first_iterations,
    )

    for name in direct.trace.__dataclass_fields__:
        combined = np.concatenate((getattr(first.trace, name), getattr(second.trace, name)), axis=0)
        np.testing.assert_array_equal(combined, getattr(direct.trace, name))
    _assert_states_equal(second.final_state, direct.final_state)
    assert second.checkpoint.rng_state == direct.checkpoint.rng_state
    assert first.checkpoint.schedule_phase == first_iterations
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


def test_structure_mode_defaults_to_mobile_and_rejects_unknown_values() -> None:
    """The new control preserves the historical default and validates modes."""
    assert FullTilingCompoundConfig(iterations=1).structure_mode == "mobile"

    with pytest.raises(TypeError, match="structure_mode must be a string"):
        FullTilingCompoundConfig(iterations=1, structure_mode=True)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match="structure_mode must be 'mobile' or 'fixed_basis'",
    ):
        FullTilingCompoundConfig(iterations=1, structure_mode="unknown")  # type: ignore[arg-type]


def test_fixed_basis_cycle_has_exactly_twelve_non_structural_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A six-outer fixed basis cycle matches twelve continuous opportunities."""
    problem, initial = _problem_state(k=4)

    def fail_structural_draw(*args: object, **kwargs: object) -> None:
        """Fail if fixed-basis dispatch enters structural proposal code."""
        raise AssertionError("fixed-basis schedule entered structural proposal code")

    monkeypatch.setattr(sampling, "_draw_structural_transition", fail_structural_draw)
    result = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(
            iterations=12,
            seed=91,
            structure_mode="fixed_basis",
        ),
    )

    assert result.checkpoint.schedule_id == FIXED_BASIS_COMPOUND_SCHEDULE_ID
    assert (
        full_tiling_compound_cycle_length(
            result.checkpoint.kernel_settings,
            schedule_id=result.checkpoint.schedule_id,
        )
        == 12
    )
    assert result.trace.slot.tolist() == [
        "root",
        *(["pair_allocation"] * 5),
        *(["fixed"] * 6),
    ]
    assert result.trace.move.tolist() == [
        "root_total_slice",
        *(["pair_allocation_refresh"] * 5),
        *(["fixed_coefficient"] * 6),
    ]
    assert result.trace.state_transition.tolist() == [0, 12]
    assert result.checkpoint.schedule_phase == 0


def test_fixed_basis_preserves_topology_but_updates_the_same_target() -> None:
    """Fixed mode holds rectangles constant without changing target semantics."""
    problem, initial = _problem_state(k=4)
    mobile = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(iterations=14, seed=19),
    )
    fixed = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(
            iterations=24,
            seed=19,
            structure_mode="fixed_basis",
        ),
    )
    initial_bounds = fixed.trace.rectangle_bounds[0]

    np.testing.assert_array_equal(
        mobile.trace.rectangle_bounds[0],
        initial_bounds,
    )
    np.testing.assert_array_equal(mobile.trace.leaf_masses[0], fixed.trace.leaf_masses[0])
    assert mobile.trace.log_target[0] == fixed.trace.log_target[0] == initial.log_target
    assert not np.any(fixed.trace.slot == "structural")
    assert not np.any(np.isin(fixed.trace.move, ("edge_flip", "resolution_relocation")))
    for bounds in fixed.trace.rectangle_bounds:
        np.testing.assert_array_equal(bounds, initial_bounds)
    assert fixed.final_state.allocation.tiling == initial.allocation.tiling


def test_fixed_basis_seeded_replay_is_exact() -> None:
    """A fixed-basis seed reproduces states, traces, and the PCG64 checkpoint."""
    problem, initial = _problem_state(k=4)
    config = FullTilingCompoundConfig(
        iterations=37,
        seed=734,
        structure_mode="fixed_basis",
    )

    first = sample_full_tiling_compound(problem, initial, config)
    replay = sample_full_tiling_compound(problem, initial, config)

    _assert_results_equal(first, replay)


def test_fixed_basis_awkward_phase_continuation_matches_direct_cycle() -> None:
    """A five-plus-seven fixed segment exactly reproduces one direct cycle."""
    problem, initial = _problem_state(k=4)
    direct = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(
            iterations=12,
            seed=284,
            structure_mode="fixed_basis",
        ),
    )
    first = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(
            iterations=5,
            seed=284,
            structure_mode="fixed_basis",
        ),
    )
    second = continue_full_tiling_compound(
        problem,
        first.checkpoint,
        iterations=7,
    )

    for name in direct.trace.__dataclass_fields__:
        combined = np.concatenate((getattr(first.trace, name), getattr(second.trace, name)), axis=0)
        np.testing.assert_array_equal(combined, getattr(direct.trace, name))
    _assert_states_equal(second.final_state, direct.final_state)
    assert second.checkpoint.rng_state == direct.checkpoint.rng_state
    assert first.checkpoint.schedule_phase == 5
    assert second.checkpoint.schedule_phase == 0
    assert first.checkpoint.schedule_id == FIXED_BASIS_COMPOUND_SCHEDULE_ID


def test_fixed_basis_checkpoint_rejects_wrong_schedule_phase() -> None:
    """Fixed-basis checkpoints fail closed when their cycle phase is altered."""
    problem, initial = _problem_state(k=4)
    first = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(
            iterations=5,
            seed=1,
            structure_mode="fixed_basis",
        ),
    )

    with pytest.raises(
        ValueError,
        match="schedule_phase is inconsistent",
    ):
        replace(first.checkpoint, schedule_phase=6)


def test_checkpoint_rejects_cross_mode_relabelling_at_incompatible_phase() -> None:
    """Schedule relabelling fails when the transition phase would disagree."""
    problem, initial = _problem_state(k=4)
    fixed = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(
            iterations=13,
            seed=2,
            structure_mode="fixed_basis",
        ),
    )
    assert fixed.checkpoint.schedule_phase == 1

    with pytest.raises(
        ValueError,
        match="schedule_phase is inconsistent",
    ):
        replace(
            fixed.checkpoint,
            schedule_id=FULL_TILING_COMPOUND_SCHEDULE_ID,
        )
