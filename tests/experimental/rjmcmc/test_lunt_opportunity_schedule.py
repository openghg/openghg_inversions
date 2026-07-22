"""Exact tests for the opt-in Lunt opportunity-matched transition schedule."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc import sampling
from openghg_inversions.experimental.rjmcmc.checkpoint_io import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    save_checkpoint,
)
from openghg_inversions.experimental.rjmcmc.core import (
    FixedDesignBlock,
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
    uniform_log_k_prior,
)
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings
from openghg_inversions.experimental.rjmcmc.sampling import (
    LUNT_OPPORTUNITY_MATCHED_FIXED_BLOCK_SCHEDULE_ID,
    LUNT_OPPORTUNITY_MATCHED_SCHEDULE_PROFILE,
    SamplerConfig,
    SamplingResult,
    _draw_transition,
    _retained_transition_numbers,
    continue_sample,
    sample,
)


def _problem(*, n_fixed: int = 6) -> TransDimensionalProblem:
    """Build a small positive-support problem with a configurable fixed block."""
    observations = np.array([2.1, 1.4, 2.7, 1.8])
    return TransDimensionalProblem(
        observations=observations,
        observation_sd=np.array([0.5, 0.7, 0.6, 0.8]),
        sensitivities=np.array(
            [
                [0.8, 0.2, 0.4, 0.1, 0.3],
                [0.1, 0.6, 0.2, 0.4, 0.5],
                [0.3, 0.2, 0.9, 0.2, 0.1],
                [0.4, 0.3, 0.1, 0.7, 0.2],
            ]
        ),
        grid_coordinates=np.arange(5, dtype=float)[:, np.newaxis],
        k_min=1,
        k_max=4,
        log_k_prior=uniform_log_k_prior(1, 4),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.5,
        fixed_block=FixedDesignBlock(
            design=np.linspace(0.02, 0.24, observations.size * n_fixed).reshape(observations.size, n_fixed),
            coefficient_prior_mean=np.ones(n_fixed),
            coefficient_prior_sd=np.full(n_fixed, 0.4),
        ),
    )


def _initial_state(problem: TransDimensionalProblem) -> TransDimensionalState:
    """Return a reproducible valid state for schedule tests."""
    return build_state(
        problem,
        [0, 3],
        [0.9, 1.1],
        fixed_coefficients=np.linspace(0.8, 1.2, problem.n_fixed_coefficients),
    )


def _config(iterations: int, *, nucleus_move: str = "global") -> SamplerConfig:
    """Return the fixed settings used by all opportunity-profile chains."""
    return SamplerConfig(
        iterations=iterations,
        coefficient_proposal_sd=0.08,
        fixed_coefficient_proposal_sd=0.06,
        birth_proposal_sd=0.12,
        seed=7341,
        nucleus_move=nucleus_move,
        local_move_scale=1.2 if nucleus_move == "local" else None,
        schedule_profile=LUNT_OPPORTUNITY_MATCHED_SCHEDULE_PROFILE,
    )


def _assert_split_matches_full(
    full: SamplingResult,
    first: SamplingResult,
    continued: SamplingResult,
) -> None:
    """Assert bitwise-equivalent trace, state, and random-stream continuation."""
    for name in (
        "state_transition",
        "k",
        "nuclei",
        "coefficients",
        "fixed_coefficients",
        "log_target",
        "moves",
        "accepted",
        "log_acceptance_ratio",
    ):
        expected = getattr(full.trace, name)
        actual = np.concatenate((getattr(first.trace, name), getattr(continued.trace, name)))
        np.testing.assert_array_equal(actual, expected)
    for name in full.final_state.__dataclass_fields__:
        actual = getattr(continued.final_state, name)
        expected = getattr(full.final_state, name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected
    assert continued.checkpoint.rng_state == full.checkpoint.rng_state


@pytest.mark.parametrize(
    ("nucleus_move", "expected_move"),
    [("global", "global_move"), ("local", "local_move")],
)
def test_fourteen_slot_schedule_has_exact_opportunities_and_fixed_positions(
    monkeypatch: pytest.MonkeyPatch,
    nucleus_move: str,
    expected_move: str,
) -> None:
    """Every cycle must expose the intended moves and each fixed position once."""
    problem = _problem()
    fixed_positions: list[int] = []
    deterministic_flags: list[bool] = []
    original = sampling.propose_fixed_coefficient

    def record_fixed(*args: Any, **kwargs: Any) -> Any:
        fixed_positions.append(kwargs["coefficient_position"])
        deterministic_flags.append(kwargs["position_selected_deterministically"])
        return original(*args, **kwargs)

    monkeypatch.setattr(sampling, "propose_fixed_coefficient", record_fixed)
    result = sample(problem, _initial_state(problem), _config(42, nucleus_move=nucleus_move))

    for cycle in range(3):
        moves = result.trace.moves[14 * cycle : 14 * (cycle + 1)]
        assert set(moves[:2]).issubset({"birth", "death"})
        assert moves[2] == expected_move
        assert moves[3:9].tolist() == ["fixed_coefficient"] * 6
        assert moves[9:].tolist() == ["coefficient"] * 5
    assert fixed_positions == list(range(6)) * 3
    assert deterministic_flags == [True] * 18
    assert np.count_nonzero(result.trace.moves == "coefficient") == 15
    assert np.count_nonzero(result.trace.moves == "fixed_coefficient") == 18
    assert np.count_nonzero(np.isin(result.trace.moves, ["birth", "death"])) == 6
    assert np.count_nonzero(result.trace.moves == expected_move) == 3
    assert result.checkpoint.schedule_id == LUNT_OPPORTUNITY_MATCHED_FIXED_BLOCK_SCHEDULE_ID


class _CoefficientRNG:
    """Minimal RNG double exposing controlled with-replacement position draws."""

    def __init__(self, positions: list[int]) -> None:
        self.positions = iter(positions)
        self.integer_bounds: list[int] = []

    def integers(self, high: int) -> int:
        self.integer_bounds.append(high)
        return next(self.positions)

    def normal(self, *, scale: float) -> float:
        del scale
        return 0.0


def test_dynamic_slots_draw_active_positions_independently_with_replacement() -> None:
    """Each dynamic opportunity must make its own active-position RNG draw."""
    problem = _problem()
    state = _initial_state(problem)
    rng = _CoefficientRNG([1, 0, 1, 1, 0])
    positions: list[int] = []
    original = sampling.propose_coefficient

    def record_position(*args: Any, **kwargs: Any) -> Any:
        positions.append(kwargs["coefficient_position"])
        return original(*args, **kwargs)

    sampling.propose_coefficient = record_position
    try:
        for _ in range(5):
            _draw_transition(problem, state, _config(1), rng, "coefficient")  # type: ignore[arg-type]
    finally:
        sampling.propose_coefficient = original

    assert positions == [1, 0, 1, 1, 0]
    assert rng.integer_bounds == [state.k] * 5


class _FixedRNG:
    """Minimal RNG double that fails if a deterministic slot draws a position."""

    def integers(self, high: int) -> int:
        raise AssertionError(f"unexpected fixed-position RNG draw with upper bound {high}")

    def normal(self, *, scale: float) -> float:
        return scale / 2.0


def test_deterministic_fixed_slot_omits_uniform_position_density() -> None:
    """A fixed slot is conditional on its position and consumes only a Gaussian draw."""
    problem = _problem()
    state = _initial_state(problem)
    scale = _config(1).fixed_coefficient_proposal_sd
    assert scale is not None

    transition = _draw_transition(
        problem,
        state,
        _config(1),
        _FixedRNG(),  # type: ignore[arg-type]
        "fixed_coefficient",
        fixed_coefficient_position=4,
    )

    expected_log_density = -math.log(scale * math.sqrt(2.0 * math.pi)) - 0.5 * (0.5**2)
    assert transition.log_q_forward == pytest.approx(expected_log_density)
    assert transition.log_q_reverse == pytest.approx(expected_log_density)


def test_profile_rejects_any_problem_without_exactly_six_fixed_coefficients() -> None:
    """The versioned opportunity profile is defined only for the six-column block."""
    problem = _problem(n_fixed=5)
    with pytest.raises(ValueError, match="exactly six fixed coefficients"):
        sample(problem, _initial_state(problem), _config(1))


def test_unaligned_durable_restart_preserves_schedule_retention_and_rng(tmp_path: Path) -> None:
    """A split inside the fixed block must exactly match an uninterrupted chain."""
    problem = _problem()
    retention = RetentionSettings(warmup_transitions=5, thin=3)
    full = sample(problem, _initial_state(problem), _config(47), retention)
    first = sample(problem, _initial_state(problem), _config(7), retention)
    checkpoint_path = tmp_path / "opportunity-profile.npz"
    save_checkpoint(checkpoint_path, first.checkpoint)

    loaded_problem = _problem()
    loaded = load_checkpoint(checkpoint_path, loaded_problem)
    continued = continue_sample(loaded_problem, loaded, iterations=40)

    assert loaded.kernel_settings.schedule_profile == LUNT_OPPORTUNITY_MATCHED_SCHEDULE_PROFILE
    assert loaded.schedule_id == LUNT_OPPORTUNITY_MATCHED_FIXED_BLOCK_SCHEDULE_ID
    _assert_split_matches_full(full, first, continued)
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
    assert metadata["schema_version"] == CHECKPOINT_SCHEMA_VERSION == 3
    assert metadata["kernel"]["schedule_profile"] == LUNT_OPPORTUNITY_MATCHED_SCHEDULE_PROFILE


def test_production_budget_matches_lunt_opportunities_and_retains_5000_states() -> None:
    """The production atomic budget must exactly reproduce Lunt's move opportunities."""
    total = 1_680_000
    cycles, remainder = divmod(total, 14)
    assert (cycles, remainder) == (120_000, 0)
    assert {
        "dynamic_coefficient": 5 * cycles,
        "fixed_coefficient": 6 * cycles,
        "mixed_dimension": 2 * cycles,
        "nucleus": cycles,
    } == {
        "dynamic_coefficient": 600_000,
        "fixed_coefficient": 720_000,
        "mixed_dimension": 240_000,
        "nucleus": 120_000,
    }

    retained = _retained_transition_numbers(
        transitions_completed=0,
        iterations=total,
        retention=RetentionSettings(warmup_transitions=280_280, thin=280),
        include_initial=True,
    )
    assert retained.size == 5_000
    assert retained[0] == 280_280
    assert retained[-1] == total
    np.testing.assert_array_equal(np.diff(retained), np.full(4_999, 280))
