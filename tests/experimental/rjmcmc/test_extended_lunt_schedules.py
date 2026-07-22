"""Exact tests for inferred-OU and shared-hierarchy Lunt schedules."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc import sampling
from openghg_inversions.experimental.rjmcmc.checkpoint_io import load_checkpoint, save_checkpoint
from openghg_inversions.experimental.rjmcmc.core import (
    FixedDesignBlock,
    InferredOUErrorModel,
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
    uniform_log_k_prior,
)
from openghg_inversions.experimental.rjmcmc.hierarchy import SharedLognormalHierarchy
from openghg_inversions.experimental.rjmcmc.likelihood import IndependentSiteOUData
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings
from openghg_inversions.experimental.rjmcmc.sampling import (
    LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_ID,
    LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_PROFILE,
    LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_ID,
    LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_PROFILE,
    KernelSettings,
    SamplerConfig,
    SamplingResult,
    SamplingTrace,
    continue_sample,
    sample,
)


def _problem(
    *,
    hierarchy: bool = False,
    inferred_ou: bool = True,
    n_fixed: int = 6,
) -> TransDimensionalProblem:
    """Build a small six-fixed-column target with optional OU and hierarchy."""
    observation_sd = np.array([0.4, 0.5, 0.6, 0.45])
    error_model = None
    if inferred_ou:
        data = IndependentSiteOUData(
            observation_sd=observation_sd,
            observation_time=np.array([0.0, 1.0, 0.0, 2.0]),
            site_index=np.array([0, 0, 1, 1]),
            mismatch_group_index=np.array([0, 1, 0, 1]),
            site_tau_index=np.array([0, 1]),
        )
        error_model = InferredOUErrorModel(
            data=data,
            mismatch_sd_prior_lower=np.array([0.1, 0.1]),
            mismatch_sd_prior_upper=np.array([2.0, 2.0]),
            correlation_timescale_prior_lower=np.array([0.2, 0.2]),
            correlation_timescale_prior_upper=np.array([5.0, 5.0]),
        )
    coefficient_hierarchy = None
    if hierarchy:
        coefficient_hierarchy = SharedLognormalHierarchy(
            mean_hyperprior_median=1.0,
            mean_hyperprior_log_sd=0.5,
            sd_hyperprior_median=0.7,
            sd_hyperprior_log_sd=0.4,
        )
    return TransDimensionalProblem(
        observations=np.array([2.1, 1.4, 2.7, 1.8]),
        observation_sd=observation_sd,
        sensitivities=np.array(
            [
                [0.8, 0.2, 0.4, 0.1, 0.3],
                [0.1, 0.6, 0.2, 0.4, 0.5],
                [0.3, 0.2, 0.9, 0.2, 0.1],
                [0.4, 0.3, 0.1, 0.7, 0.2],
            ]
        ),
        grid_coordinates=np.arange(5, dtype=float)[
            :,
            np.newaxis,
        ],
        k_min=1,
        k_max=4,
        log_k_prior=uniform_log_k_prior(1, 4),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.7,
        fixed_block=FixedDesignBlock(
            design=np.linspace(0.02, 0.24, 4 * n_fixed, dtype=np.float64).reshape(4, n_fixed),
            coefficient_prior_mean=np.ones(n_fixed),
            coefficient_prior_sd=np.full(n_fixed, 0.4),
        ),
        error_model=error_model,
        coefficient_hierarchy=coefficient_hierarchy,
    )


def _initial_state(problem: TransDimensionalProblem) -> TransDimensionalState:
    """Return a valid state for either extended schedule."""
    return build_state(
        problem,
        [0, 3],
        [0.9, 1.1],
        fixed_coefficients=np.linspace(0.8, 1.2, problem.n_fixed_coefficients),
        mismatch_sd=np.array([0.6, 0.8]) if problem.error_model is not None else None,
        correlation_timescale=np.array([1.2, 1.8]) if problem.error_model is not None else None,
    )


def _config(
    iterations: int,
    *,
    hierarchy: bool = False,
    backend: str = "numpy",
) -> SamplerConfig:
    """Return deterministic settings for one extended schedule."""
    kwargs: dict[str, object] = {
        "iterations": iterations,
        "coefficient_proposal_sd": 0.08,
        "fixed_coefficient_proposal_sd": 0.06,
        "birth_proposal_sd": 0.12,
        "mismatch_sd_proposal_sd": 0.05,
        "correlation_timescale_proposal_sd": 0.09,
        "seed": 7341,
        "backend": backend,
        "schedule_profile": LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_PROFILE,
    }
    if hierarchy:
        kwargs.update(
            schedule_profile=LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_PROFILE,
            eta_proposal_sd=0.04,
            zeta_proposal_sd=0.03,
        )
    return SamplerConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(("hierarchy", "cycle_length"), [(False, 16), (True, 17)])
def test_extended_cycle_appends_optional_slots_after_unchanged_lunt_prefix(
    hierarchy: bool,
    cycle_length: int,
) -> None:
    """Each cycle must append OU and hierarchy moves after the old 14 slots."""
    problem = _problem(hierarchy=hierarchy)
    result = sample(problem, _initial_state(problem), _config(cycle_length * 2, hierarchy=hierarchy))

    for cycle in range(2):
        moves = result.trace.moves[cycle * cycle_length : (cycle + 1) * cycle_length]
        assert set(moves[:2]).issubset({"birth", "death"})
        assert moves[2] == "global_move"
        assert moves[3:9].tolist() == ["fixed_coefficient"] * 6
        assert moves[9:14].tolist() == ["coefficient"] * 5
        assert moves[14:16].tolist() == ["mismatch_sd", "correlation_timescale"]
        if hierarchy:
            assert moves[16] == "shared_hierarchy"
    assert result.trace.moves.dtype == np.dtype("<U21")
    assert result.checkpoint.schedule_id == (
        LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_ID
        if hierarchy
        else LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_ID
    )


def test_extended_slots_dispatch_with_stable_scalar_draw_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """OU slots draw position then Normal; hierarchy draws eta then zeta."""
    problem = _problem(hierarchy=True)
    state = _initial_state(problem)
    calls: list[tuple[str, dict[str, Any]]] = []

    def recorder(name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> Any:
            calls.append((name, kwargs))
            return sampling.TransitionTerms(
                candidate=state,
                log_target_delta=0.0,
                log_q_forward=0.0,
                log_q_reverse=0.0,
                log_jacobian=0.0,
                move=name,
            )

        return record

    monkeypatch.setattr(sampling, "propose_mismatch_sd", recorder("mismatch_sd"))
    monkeypatch.setattr(sampling, "propose_correlation_timescale", recorder("correlation_timescale"))
    monkeypatch.setattr(sampling, "propose_shared_hierarchy", recorder("shared_hierarchy"))
    rng = np.random.default_rng(991)
    expected = np.random.default_rng(991)
    mismatch_position = int(expected.integers(2))
    mismatch_delta = float(expected.normal(scale=0.05))
    timescale_position = int(expected.integers(2))
    timescale_delta = float(expected.normal(scale=0.09))
    eta_delta = float(expected.normal(scale=0.04))
    zeta_delta = float(expected.normal(scale=0.03))

    sampling._draw_transition(problem, state, _config(1, hierarchy=True), rng, "mismatch_sd")
    sampling._draw_transition(problem, state, _config(1, hierarchy=True), rng, "correlation_timescale")
    sampling._draw_transition(problem, state, _config(1, hierarchy=True), rng, "shared_hierarchy")

    assert [name for name, _ in calls] == ["mismatch_sd", "correlation_timescale", "shared_hierarchy"]
    assert calls[0][1]["mismatch_sd_position"] == mismatch_position
    assert calls[0][1]["proposed_mismatch_sd"] == state.mismatch_sd[mismatch_position] + mismatch_delta
    assert calls[1][1]["correlation_timescale_position"] == timescale_position
    assert (
        calls[1][1]["proposed_correlation_timescale"]
        == state.correlation_timescale[timescale_position] + timescale_delta
    )
    assert calls[2][1]["proposed_eta"] == state.eta + eta_delta
    assert calls[2][1]["proposed_zeta"] == state.zeta + zeta_delta
    assert rng.bit_generator.state == expected.bit_generator.state


@pytest.mark.parametrize(
    "updates",
    [
        {"mismatch_sd_proposal_sd": None},
        {"correlation_timescale_proposal_sd": 0.0},
        {"eta_proposal_sd": 0.1},
        {"zeta_proposal_sd": np.inf},
    ],
)
def test_profile_scales_are_required_exactly_when_used(updates: dict[str, object]) -> None:
    """Missing, extra, and nonpositive optional-kernel scales must fail early."""
    kwargs: dict[str, object] = {
        "iterations": 1,
        "coefficient_proposal_sd": 0.1,
        "birth_proposal_sd": 0.1,
        "mismatch_sd_proposal_sd": 0.1,
        "correlation_timescale_proposal_sd": 0.1,
        "schedule_profile": LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_PROFILE,
    }
    kwargs.update(updates)
    with pytest.raises(ValueError):
        SamplerConfig(**kwargs)  # type: ignore[arg-type]


def test_profiles_reject_targets_whose_optional_parameters_would_be_frozen() -> None:
    """A target and profile must agree exactly on OU and hierarchy activation."""
    ou_problem = _problem()
    hierarchy_problem = _problem(hierarchy=True)
    fixed_problem = _problem(inferred_ou=False)
    old_config = SamplerConfig(
        iterations=1,
        coefficient_proposal_sd=0.1,
        birth_proposal_sd=0.1,
    )
    with pytest.raises(ValueError, match="frozen"):
        sample(ou_problem, _initial_state(ou_problem), old_config)
    with pytest.raises(ValueError, match="hierarchy activation"):
        sample(hierarchy_problem, _initial_state(hierarchy_problem), _config(1))
    with pytest.raises(ValueError, match="inferred OU"):
        sample(fixed_problem, _initial_state(fixed_problem), _config(1))
    wrong_fixed_count = _problem(n_fixed=5)
    with pytest.raises(ValueError, match="exactly six"):
        sample(wrong_fixed_count, _initial_state(wrong_fixed_count), _config(1))


def test_kernel_settings_enforce_the_same_profile_scale_contract() -> None:
    """Direct immutable kernel construction cannot bypass optional-scale checks."""
    with pytest.raises(ValueError, match="mismatch_sd_proposal_sd is required"):
        KernelSettings(
            coefficient_proposal_sd=0.1,
            birth_proposal_sd=0.1,
            backend="numpy",
            nucleus_move="global",
            local_move_scale=None,
            schedule_profile=LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_PROFILE,
        )


def _assert_split_matches_full(
    full: SamplingResult,
    first: SamplingResult,
    continued: SamplingResult,
) -> None:
    """Assert exact concatenated trace and final-state equality."""
    for name in (
        "state_transition",
        "k",
        "nuclei",
        "coefficients",
        "fixed_coefficients",
        "mismatch_sd",
        "correlation_timescale",
        "eta",
        "zeta",
        "log_target",
        "moves",
        "accepted",
        "log_acceptance_ratio",
    ):
        actual = np.concatenate((getattr(first.trace, name), getattr(continued.trace, name)))
        np.testing.assert_array_equal(actual, getattr(full.trace, name))
    assert first.trace.coefficient_hierarchy_active == full.trace.coefficient_hierarchy_active
    assert continued.trace.coefficient_hierarchy_active == full.trace.coefficient_hierarchy_active
    for name in full.final_state.__dataclass_fields__:
        actual = getattr(continued.final_state, name)
        expected = getattr(full.final_state, name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected
    assert continued.checkpoint.rng_state == full.checkpoint.rng_state


def test_awkward_phase_in_memory_restart_retains_every_optional_state() -> None:
    """Splitting inside the appended slots must preserve schedule and RNG phase."""
    problem = _problem(hierarchy=True)
    retention = RetentionSettings(warmup_transitions=4, thin=3)
    full = sample(problem, _initial_state(problem), _config(53, hierarchy=True), retention)
    first = sample(problem, _initial_state(problem), _config(16, hierarchy=True), retention)
    continued = continue_sample(problem, first.checkpoint, iterations=37)

    _assert_split_matches_full(full, first, continued)


def test_ou_only_durable_restart_preserves_appended_schedule_phase(tmp_path: Path) -> None:
    """Checkpoint v3 must exactly continue the 16-slot OU-only profile."""
    problem = _problem()
    retention = RetentionSettings(warmup_transitions=3, thin=4)
    full = sample(problem, _initial_state(problem), _config(45), retention)
    first = sample(problem, _initial_state(problem), _config(15), retention)
    path = tmp_path / "ou-only-v3.npz"
    save_checkpoint(path, first.checkpoint)

    loaded_problem = _problem()
    loaded = load_checkpoint(path, loaded_problem)
    continued = continue_sample(loaded_problem, loaded, iterations=30)

    _assert_split_matches_full(full, first, continued)
    assert loaded.schedule_id == LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_ID


def test_extended_hierarchy_schedule_has_numpy_numba_parity() -> None:
    """Both numerical backends must follow the same seeded extended chain."""
    problem = _problem(hierarchy=True)
    numpy_result = sample(
        problem,
        _initial_state(problem),
        _config(34, hierarchy=True, backend="numpy"),
    )
    numba_initial = build_state(
        problem,
        [0, 3],
        [0.9, 1.1],
        fixed_coefficients=np.linspace(0.8, 1.2, 6),
        mismatch_sd=np.array([0.6, 0.8]),
        correlation_timescale=np.array([1.2, 1.8]),
        backend="numba",
    )
    numba_result = sample(
        problem,
        numba_initial,
        _config(34, hierarchy=True, backend="numba"),
    )

    for name in (
        "k",
        "nuclei",
        "coefficients",
        "fixed_coefficients",
        "mismatch_sd",
        "correlation_timescale",
        "eta",
        "zeta",
        "moves",
        "accepted",
    ):
        np.testing.assert_array_equal(getattr(numpy_result.trace, name), getattr(numba_result.trace, name))
    np.testing.assert_allclose(
        numpy_result.trace.log_target,
        numba_result.trace.log_target,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        numpy_result.trace.log_acceptance_ratio,
        numba_result.trace.log_acceptance_ratio,
        rtol=0.0,
        atol=1e-12,
    )


def test_trace_legacy_defaults_normalize_optional_state_storage() -> None:
    """Legacy manual traces receive empty OU blocks and inactive NaN hierarchy rows."""
    trace = SamplingTrace(
        k=np.array([1, 1]),
        nuclei=np.array([[0], [0]]),
        coefficients=np.array([[1.0], [1.0]]),
        log_target=np.array([0.0, 0.0]),
        moves=np.array(["coefficient"]),
        accepted=np.array([True]),
        log_acceptance_ratio=np.array([0.0]),
    )

    assert trace.mismatch_sd.shape == (2, 0)
    assert trace.correlation_timescale.shape == (2, 0)
    np.testing.assert_array_equal(np.isnan(trace.eta), [True, True])
    np.testing.assert_array_equal(np.isnan(trace.zeta), [True, True])
    assert not trace.coefficient_hierarchy_active


def test_start_state_validation_checks_optional_shapes_and_hierarchy_coordinates() -> None:
    """Malformed optional state arrays and active hierarchy coordinates are rejected."""
    problem = _problem(hierarchy=True)
    state = _initial_state(problem)
    config = _config(1, hierarchy=True)
    with pytest.raises(ValueError, match="mismatch_sd"):
        sample(problem, replace(state, mismatch_sd=np.array([0.5])), config)
    with pytest.raises(ValueError, match="hierarchy coordinates"):
        sample(problem, replace(state, eta=np.nan), config)
