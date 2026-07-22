"""Focused tests for the first seeded four-slot TDMCMC sampler."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.core import (
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
    uniform_log_k_prior,
)
from openghg_inversions.experimental.rjmcmc.proposals import propose_local_move
from openghg_inversions.experimental.rjmcmc.sampling import (
    SamplerConfig,
    SamplingResult,
    _draw_transition,
    sample,
)


def _problem(*, k_min: int = 1, k_max: int = 3) -> TransDimensionalProblem:
    """Return a tiny in-memory problem with configurable model-size bounds."""
    return TransDimensionalProblem(
        observations=np.array([5.0, 1.0, -0.5]),
        observation_sd=np.array([0.8, 1.2, 0.5]),
        sensitivities=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.5, 0.0, 1.0, 0.0],
                [-1.0, 2.0, 0.0, 1.0],
            ]
        ),
        grid_coordinates=np.arange(4, dtype=float)[:, np.newaxis],
        k_min=k_min,
        k_max=k_max,
        log_k_prior=uniform_log_k_prior(k_min, k_max),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.4,
    )


def _assert_results_equal(actual: SamplingResult, expected: SamplingResult) -> None:
    """Assert exact replay equality for every trace and final-state field."""
    for actual_array, expected_array in (
        (actual.trace.k, expected.trace.k),
        (actual.trace.nuclei, expected.trace.nuclei),
        (actual.trace.coefficients, expected.trace.coefficients),
        (actual.trace.log_target, expected.trace.log_target),
        (actual.trace.moves, expected.trace.moves),
        (actual.trace.accepted, expected.trace.accepted),
        (actual.trace.log_acceptance_ratio, expected.trace.log_acceptance_ratio),
        (actual.final_state.nuclei, expected.final_state.nuclei),
        (actual.final_state.coefficients, expected.final_state.coefficients),
        (actual.final_state.labels, expected.final_state.labels),
        (actual.final_state.design, expected.final_state.design),
        (actual.final_state.prediction, expected.final_state.prediction),
        (actual.final_state.residual, expected.final_state.residual),
    ):
        np.testing.assert_array_equal(actual_array, expected_array)
    assert actual.final_state.log_target == expected.final_state.log_target


def _assert_state_matches_trace_row(
    state: TransDimensionalState,
    result: SamplingResult,
    row: int,
) -> None:
    """Compare a rebuilt state with one fixed-capacity trace row."""
    k = int(result.trace.k[row])
    np.testing.assert_array_equal(result.trace.nuclei[row, k:], -1)
    np.testing.assert_array_equal(result.trace.coefficients[row, k:], 0.0)
    np.testing.assert_array_equal(state.nuclei, result.trace.nuclei[row])
    np.testing.assert_array_equal(state.coefficients, result.trace.coefficients[row])
    assert state.log_target == pytest.approx(result.trace.log_target[row], rel=0.0, abs=1e-12)


def _assert_four_slot_move_pattern(moves: np.ndarray, *, nucleus_move: str) -> None:
    """Check fixed slot roles while allowing independent dimension-move draws."""
    np.testing.assert_array_equal(moves[0::4], "coefficient")
    assert np.all(np.isin(moves[1::4], ["birth", "death"]))
    assert np.all(np.isin(moves[2::4], ["birth", "death"]))
    np.testing.assert_array_equal(moves[3::4], nucleus_move)


def test_fixed_seed_replays_the_complete_sampler_trace() -> None:
    """The same seed and initial state should reproduce every attempted transition."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    config = SamplerConfig(
        iterations=24,
        coefficient_proposal_sd=0.15,
        birth_proposal_sd=0.25,
        seed=481,
    )

    first = sample(problem, initial, config)
    second = sample(problem, initial, config)

    _assert_results_equal(first, second)
    _assert_four_slot_move_pattern(first.trace.moves, nucleus_move="global_move")
    np.testing.assert_array_equal(
        first.trace.accepted,
        [
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
        ],
    )
    np.testing.assert_array_equal(
        first.trace.k,
        [2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3],
    )
    assert first.checkpoint.rng_state.state == 251418900833954616027504699521200364421
    assert first.checkpoint.rng_state.increment == 182842302344849351119389596454276712109
    assert first.checkpoint.rng_state.has_uint32 == 1
    assert first.checkpoint.rng_state.uinteger == 3481975001
    assert 0.0 <= first.trace.acceptance_rate <= 1.0


@pytest.mark.parametrize(("seed", "expected_move"), [(0, "death"), (2, "birth")])
def test_dimension_slot_uses_seeded_equal_probability_choice(
    seed: int,
    expected_move: str,
) -> None:
    """A dimension slot should map a seeded uniform draw to birth or death."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    config = SamplerConfig(
        iterations=1,
        coefficient_proposal_sd=0.15,
        birth_proposal_sd=0.25,
    )

    transition = _draw_transition(
        problem,
        initial,
        config,
        np.random.default_rng(seed),
        "dimension",
    )

    assert transition.move == expected_move


def test_default_global_mode_matches_explicit_global_configuration() -> None:
    """Adding the local option should not change default fixed-seed replay."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    common = {
        "iterations": 24,
        "coefficient_proposal_sd": 0.15,
        "birth_proposal_sd": 0.25,
        "seed": 481,
    }

    default = sample(problem, initial, SamplerConfig(**common))
    explicit = sample(problem, initial, SamplerConfig(**common, nucleus_move="global"))

    _assert_results_equal(default, explicit)


def test_local_destination_draw_uses_normalized_coordinate_distance_weights() -> None:
    """The sampler should draw from the exact discrete-Gaussian cell weights."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    config = SamplerConfig(
        iterations=1,
        coefficient_proposal_sd=0.15,
        birth_proposal_sd=0.25,
        seed=52,
        nucleus_move="local",
        local_move_scale=0.7,
    )
    actual_rng = np.random.default_rng(config.seed)
    expected_rng = np.random.default_rng(config.seed)

    position = int(expected_rng.integers(initial.k))
    empty = np.array([1, 2], dtype=np.int64)
    origin = int(initial.active_nuclei[position])
    differences = (problem.grid_coordinates[empty] - problem.grid_coordinates[origin]) / 0.7
    log_weights = -0.5 * np.einsum("ij,ij->i", differences, differences)
    weights = np.exp(log_weights - np.max(log_weights))
    cell = int(expected_rng.choice(empty, p=weights / weights.sum()))
    expected = propose_local_move(
        problem,
        initial,
        move_position=position,
        new_nucleus=cell,
        proposal_scale=0.7,
    )

    actual = _draw_transition(problem, initial, config, actual_rng, "local_move")

    assert actual.move == "local_move"
    assert actual.log_q_forward == pytest.approx(expected.log_q_forward)
    assert actual.log_q_reverse == pytest.approx(expected.log_q_reverse)
    np.testing.assert_array_equal(actual.candidate.nuclei, expected.candidate.nuclei)
    np.testing.assert_array_equal(actual.candidate.coefficients, expected.candidate.coefficients)


def test_fixed_seed_replays_local_nucleus_moves() -> None:
    """Local mode should replay the complete trace under a fixed seed."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    config = SamplerConfig(
        iterations=24,
        coefficient_proposal_sd=0.15,
        birth_proposal_sd=0.25,
        seed=481,
        nucleus_move="local",
        local_move_scale=0.8,
    )

    first = sample(problem, initial, config)
    second = sample(problem, initial, config)

    _assert_results_equal(first, second)
    _assert_four_slot_move_pattern(first.trace.moves, nucleus_move="local_move")


def test_local_sampler_has_numpy_numba_backend_parity() -> None:
    """Both state builders should follow the same seeded local-move chain."""
    problem = _problem()
    numpy_initial = build_state(problem, [0, 3], [0.8, 1.2], backend="numpy")
    numba_initial = build_state(problem, [0, 3], [0.8, 1.2], backend="numba")
    common = {
        "iterations": 24,
        "coefficient_proposal_sd": 0.15,
        "birth_proposal_sd": 0.25,
        "seed": 901,
        "nucleus_move": "local",
        "local_move_scale": 0.8,
    }

    numpy_result = sample(problem, numpy_initial, SamplerConfig(**common, backend="numpy"))
    numba_result = sample(problem, numba_initial, SamplerConfig(**common, backend="numba"))

    for numpy_array, numba_array in (
        (numpy_result.trace.k, numba_result.trace.k),
        (numpy_result.trace.nuclei, numba_result.trace.nuclei),
        (numpy_result.trace.coefficients, numba_result.trace.coefficients),
        (numpy_result.trace.moves, numba_result.trace.moves),
        (numpy_result.trace.accepted, numba_result.trace.accepted),
    ):
        np.testing.assert_array_equal(numpy_array, numba_array)
    np.testing.assert_allclose(
        numpy_result.trace.log_acceptance_ratio,
        numba_result.trace.log_acceptance_ratio,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        numpy_result.trace.log_target,
        numba_result.trace.log_target,
        rtol=0.0,
        atol=1e-12,
    )


def test_fixed_k_sampler_records_birth_and_death_boundary_self_transitions() -> None:
    """Impossible structural moves should remain explicit rejected attempts."""
    problem = _problem(k_min=1, k_max=1)
    initial = build_state(problem, [0], [1.0])
    config = SamplerConfig(
        iterations=4,
        coefficient_proposal_sd=0.1,
        birth_proposal_sd=0.2,
        seed=7,
    )

    result = sample(problem, initial, config)

    np.testing.assert_array_equal(result.trace.k, np.ones(5, dtype=np.int64))
    _assert_four_slot_move_pattern(result.trace.moves, nucleus_move="global_move")
    np.testing.assert_array_equal(result.trace.accepted[1:3], [False, False])
    np.testing.assert_array_equal(result.trace.log_acceptance_ratio[1:3], [-np.inf, -np.inf])
    np.testing.assert_array_equal(result.trace.nuclei[1], result.trace.nuclei[2])
    np.testing.assert_array_equal(result.trace.nuclei[2], result.trace.nuclei[3])
    np.testing.assert_array_equal(result.trace.coefficients[1], result.trace.coefficients[2])
    np.testing.assert_array_equal(result.trace.coefficients[2], result.trace.coefficients[3])


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_trace_padding_and_cached_targets_reconstruct(backend: str) -> None:
    """Every saved padded row should rebuild the same normalized target caches."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2], backend=backend)  # type: ignore[arg-type]
    result = sample(
        problem,
        initial,
        SamplerConfig(
            iterations=20,
            coefficient_proposal_sd=0.15,
            birth_proposal_sd=0.25,
            seed=91,
            backend=backend,  # type: ignore[arg-type]
        ),
    )

    for row, k_value in enumerate(result.trace.k):
        k = int(k_value)
        rebuilt = build_state(
            problem,
            result.trace.nuclei[row, :k],
            result.trace.coefficients[row, :k],
            backend=backend,  # type: ignore[arg-type]
        )
        _assert_state_matches_trace_row(rebuilt, result, row)

    _assert_state_matches_trace_row(result.final_state, result, -1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"iterations": True},
        {"coefficient_proposal_sd": 0.0},
        {"coefficient_proposal_sd": np.inf},
        {"birth_proposal_sd": -1.0},
        {"birth_proposal_sd": np.nan},
        {"backend": "jax"},
        {"nucleus_move": "nearby"},
        {"nucleus_move": "local"},
        {"nucleus_move": "local", "local_move_scale": 0.0},
        {"nucleus_move": "local", "local_move_scale": np.inf},
        {"nucleus_move": "local", "local_move_scale": np.nan},
    ],
)
def test_sampler_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    """Invalid iteration, proposal-scale and backend settings should fail early."""
    config_kwargs: dict[str, object] = {
        "iterations": 4,
        "coefficient_proposal_sd": 0.1,
        "birth_proposal_sd": 0.2,
        "seed": 5,
    }
    config_kwargs.update(kwargs)

    with pytest.raises(ValueError):
        SamplerConfig(**config_kwargs)  # type: ignore[arg-type]


def test_sampler_rejects_incompatible_or_zero_density_initial_states() -> None:
    """Sampling should require matching capacity and finite initial target density."""
    problem = _problem(k_max=2)
    other_problem = _problem(k_max=3)
    incompatible = build_state(other_problem, [0, 3], [0.8, 1.2])
    config = SamplerConfig(
        iterations=1,
        coefficient_proposal_sd=0.1,
        birth_proposal_sd=0.2,
    )

    with pytest.raises(ValueError, match="capacity"):
        sample(problem, incompatible, config)

    zero_mass_problem = TransDimensionalProblem(
        observations=problem.observations,
        observation_sd=problem.observation_sd,
        sensitivities=problem.sensitivities,
        grid_coordinates=problem.grid_coordinates,
        k_min=1,
        k_max=2,
        log_k_prior=np.array([-np.inf, 0.0]),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.4,
    )
    zero_density = build_state(zero_mass_problem, [0], [1.0])
    with pytest.raises(ValueError, match="finite target"):
        sample(zero_mass_problem, zero_density, config)
