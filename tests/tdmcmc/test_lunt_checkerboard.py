"""Local checkerboard benchmark inspired by Lunt et al. (2016), Sect. 4.

This is an implementation benchmark, not a reproduction of the publication.
It preserves the paper's alternating 0.5/1.5 scaling truth, independent 5 ppb
noise, arithmetic lognormal prior moments of mean 1 and standard deviation 1,
and native-grid posterior reconstruction. The following scientific inputs are
deliberately synthetic or reduced:

- an 8 by 8 grid replaces the paper's 56 by 48 NAME grid;
- 96 analytic Gaussian footprints replace 942 six-hour NAME footprints from
  four DECC sites during May--June 2014;
- a smooth artificial prior-flux field replaces regridded EDGAR methane;
- the 4 by 4 checkerboard uses regular 2 by 2 blocks rather than the archived
  paper mask, which is not locally available;
- sensitivity magnitudes are arbitrary ppb-like amplitudes, not NAME units;
- ``k`` spans 8--28 and starts at 16 instead of spanning 5--500 and starting
  at 40; adaptive-geometry chains share one seeded non-oracle nucleus layout
  with all coefficients at the prior mean, rather than paper-era random nuclei;
- the local run uses 40,000 transitions, a 15,000-row burn cutoff, thinning by
  10, and three seeded comparator runs, not a paper-era production/convergence
  analysis;
- the current four-slot non-hierarchical sampler, with two invariant 50/50
  birth/death slots and a corrected finite-grid local move, replaces the exact
  historical proposal implementation.

Consequently, posterior ``k`` and published RMSE values are intentionally not
asserted. The slow seeded benchmark compares the trans-dimensional chain with
an oracle fixed partition, a movable fixed-``k`` partition, and several random
fixed partitions. It is a local implementation comparison, not evidence that
trans-dimensional inference outperforms a production RHIME or PyMC inversion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import log
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
import pytest

from openghg_inversions.tdmcmc.core import (
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
    uniform_log_k_prior,
)
from openghg_inversions.tdmcmc.postprocessing import summarize_fine_grid_posterior
from openghg_inversions.tdmcmc.proposals import accept_or_reject, propose_coefficient
from openghg_inversions.tdmcmc.sampling import SamplerConfig, SamplingTrace, sample

FloatArray = NDArray[np.float64]

_GRID_SHAPE = (8, 8)
_OBSERVATION_COUNT = 96
_OBSERVATION_SD = 5.0
_TRUTH_NUCLEI = np.array(
    [0, 2, 4, 6, 16, 18, 20, 22, 32, 34, 36, 38, 48, 50, 52, 54],
    dtype=np.int64,
)
_INITIAL_NUCLEI = np.array(
    [4, 7, 8, 9, 10, 13, 14, 18, 24, 30, 37, 39, 49, 55, 57, 59],
    dtype=np.int64,
)
_BENCHMARK_ITERATIONS = 40_000
_BENCHMARK_START = 15_000
_BENCHMARK_THIN = 10
_BENCHMARK_SEEDS = (481, 917, 1601)
_RANDOM_LAYOUT_SEEDS = (2401, 2402, 2403)


@dataclass(frozen=True, slots=True)
class LuntCheckerboardCase:
    """Complete deterministic local checkerboard inversion benchmark."""

    problem: TransDimensionalProblem
    truth: FloatArray
    noise: FloatArray
    noiseless_observations: FloatArray
    footprint_amplitudes: FloatArray


@dataclass(frozen=True, slots=True)
class _BenchmarkMetrics:
    """Scientific recovery metrics and elapsed time for one seeded chain."""

    prediction_rmse: float
    grid_rmse: float
    spatial_correlation: float
    contrast: float
    runtime_seconds: float
    visited_k: tuple[int, ...]


def _checkerboard_truth() -> FloatArray:
    """Return alternating 0.5/1.5 values on sixteen regular 2 by 2 blocks."""
    rows, columns = np.indices(_GRID_SHAPE)
    block_parity = (rows // 2 + columns // 2) % 2
    return np.where(block_parity == 0, 0.5, 1.5).reshape(-1).astype(np.float64)


def _build_lunt_checkerboard_case() -> LuntCheckerboardCase:
    """Build deterministic smooth sensitivities and seeded noisy observations."""
    rows, columns = _GRID_SHAPE
    grid_coordinates: FloatArray = np.array(
        [(row, column) for row in range(rows) for column in range(columns)],
        dtype=np.float64,
    )
    row_offsets = grid_coordinates[:, 0] - 3.5
    column_offsets = grid_coordinates[:, 1] - 3.5
    radial_distance: FloatArray = row_offsets * row_offsets + column_offsets * column_offsets
    prior_flux = 0.75 + 0.35 * np.exp(-radial_distance / 12.0) + 0.15 * grid_coordinates[:, 1] / 7.0

    footprint_rng = np.random.default_rng(2016)
    centres = np.column_stack(
        (
            footprint_rng.uniform(-0.25, 7.25, _OBSERVATION_COUNT),
            footprint_rng.uniform(-0.25, 7.25, _OBSERVATION_COUNT),
        )
    )
    widths = footprint_rng.uniform(0.55, 1.35, _OBSERVATION_COUNT)
    amplitudes = footprint_rng.uniform(50.0, 130.0, _OBSERVATION_COUNT)
    sensitivities = np.empty((_OBSERVATION_COUNT, rows * columns), dtype=np.float64)
    for observation in range(_OBSERVATION_COUNT):
        squared_distance = np.sum(
            np.square(grid_coordinates - centres[observation]),
            axis=1,
        )
        weights = prior_flux * np.exp(-squared_distance / (2.0 * widths[observation] ** 2))
        sensitivities[observation] = amplitudes[observation] * weights / weights.sum()

    truth = _checkerboard_truth()
    noiseless_observations = sensitivities @ truth
    noise = np.random.default_rng(2017).normal(
        scale=_OBSERVATION_SD,
        size=_OBSERVATION_COUNT,
    )
    problem = TransDimensionalProblem(
        observations=noiseless_observations + noise,
        observation_sd=np.full(_OBSERVATION_COUNT, _OBSERVATION_SD),
        sensitivities=sensitivities,
        grid_coordinates=grid_coordinates,
        k_min=8,
        k_max=28,
        log_k_prior=uniform_log_k_prior(8, 28),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=1.0,
    )
    return LuntCheckerboardCase(
        problem=problem,
        truth=truth,
        noise=noise,
        noiseless_observations=noiseless_observations,
        footprint_amplitudes=amplitudes,
    )


def _true_state(case: LuntCheckerboardCase, *, backend: str = "numpy") -> TransDimensionalState:
    """Build the exact sixteen-region representation of the checkerboard truth."""
    coefficients = case.truth[_TRUTH_NUCLEI]
    return build_state(
        case.problem,
        _TRUTH_NUCLEI,
        coefficients,
        backend=backend,  # type: ignore[arg-type]
    )


def _single_state_trace(state: TransDimensionalState) -> SamplingTrace:
    """Wrap one complete state as the smallest valid reconstruction trace."""
    return SamplingTrace(
        k=np.array([state.k], dtype=np.int64),
        nuclei=state.nuclei[np.newaxis, :],
        coefficients=state.coefficients[np.newaxis, :],
        log_target=np.array([state.log_target]),
        moves=np.empty(0, dtype="U16"),
        accepted=np.empty(0, dtype=np.bool_),
        log_acceptance_ratio=np.empty(0, dtype=np.float64),
    )


def _fixed_k_problem(case: LuntCheckerboardCase) -> TransDimensionalProblem:
    """Return the same target inputs conditioned on exactly sixteen regions."""
    return replace(
        case.problem,
        k_min=_TRUTH_NUCLEI.size,
        k_max=_TRUTH_NUCLEI.size,
        log_k_prior=uniform_log_k_prior(_TRUTH_NUCLEI.size, _TRUTH_NUCLEI.size),
    )


def _sample_fixed_nuclei_coefficients(
    problem: TransDimensionalProblem,
    nuclei: NDArray[np.int64],
    *,
    seed: int,
    iterations: int = _BENCHMARK_ITERATIONS,
    proposal_sd: float = 0.1,
) -> tuple[SamplingTrace, float]:
    """Run seeded coefficient-only MH while holding a supplied partition fixed.

    This intentionally small test helper uses the production coefficient
    proposal and accept/reject functions, so its normalized Gaussian likelihood
    and lognormal coefficient prior are exactly those used by the full sampler.
    It attempts a coefficient update in every fourth trace slot and holds the
    other slots fixed. Thus 40,000 saved transitions provide the same 10,000
    coefficient opportunities and retained-row logic as a full four-slot run;
    the wall-clock timings are not comparable with production implementations.
    """
    if iterations % 4:
        raise ValueError("iterations must contain complete four-slot budgets.")
    rng = np.random.default_rng(seed)
    state = build_state(problem, nuclei, np.ones(nuclei.size), backend="numba")
    k_trace = np.full(iterations + 1, state.k, dtype=np.int64)
    nuclei_trace = np.repeat(state.nuclei[np.newaxis, :], iterations + 1, axis=0)
    coefficient_trace = np.empty((iterations + 1, problem.k_max), dtype=np.float64)
    log_target_trace = np.empty(iterations + 1, dtype=np.float64)
    moves = np.full(iterations, "fixed", dtype="U16")
    accepted = np.zeros(iterations, dtype=np.bool_)
    log_acceptance_ratio = np.empty(iterations, dtype=np.float64)
    coefficient_trace[0] = state.coefficients
    log_target_trace[0] = state.log_target

    started = perf_counter()
    for iteration in range(iterations):
        if iteration % 4 == 0:
            position = int(rng.integers(state.k))
            transition = propose_coefficient(
                problem,
                state,
                coefficient_position=position,
                proposed_coefficient=float(
                    state.active_coefficients[position] + rng.normal(scale=proposal_sd)
                ),
                proposal_stdev=proposal_sd,
                backend="numba",
            )
            uniform = float(rng.random())
            next_state = accept_or_reject(
                state,
                transition,
                log_uniform=log(uniform) if uniform > 0.0 else -np.inf,
            )
            accepted[iteration] = transition.valid and next_state is transition.candidate
            log_acceptance_ratio[iteration] = transition.log_acceptance_ratio
            moves[iteration] = "coefficient"
            state = next_state
        else:
            log_acceptance_ratio[iteration] = -np.inf
        coefficient_trace[iteration + 1] = state.coefficients
        log_target_trace[iteration + 1] = state.log_target
    runtime_seconds = perf_counter() - started

    return (
        SamplingTrace(
            k=k_trace,
            nuclei=nuclei_trace,
            coefficients=coefficient_trace,
            log_target=log_target_trace,
            moves=moves,
            accepted=accepted,
            log_acceptance_ratio=log_acceptance_ratio,
        ),
        runtime_seconds,
    )


def _summarize_benchmark(
    case: LuntCheckerboardCase,
    problem: TransDimensionalProblem,
    trace: SamplingTrace,
    runtime_seconds: float,
) -> _BenchmarkMetrics:
    """Calculate common prediction and native-grid metrics for one trace."""
    summary = summarize_fine_grid_posterior(
        problem,
        trace,
        case.noiseless_observations,
        start=_BENCHMARK_START,
        thin=_BENCHMARK_THIN,
        backend="numba",
    )
    low_mean = float(np.mean(summary.mean[case.truth == 0.5]))
    high_mean = float(np.mean(summary.mean[case.truth == 1.5]))
    return _BenchmarkMetrics(
        prediction_rmse=summary.rmse,
        grid_rmse=float(np.sqrt(np.mean(np.square(summary.mean - case.truth)))),
        spatial_correlation=float(np.corrcoef(summary.mean, case.truth)[0, 1]),
        contrast=high_mean - low_mean,
        runtime_seconds=runtime_seconds,
        visited_k=tuple(int(value) for value in np.unique(trace.k)),
    )


def _run_full_sampler_benchmark(
    case: LuntCheckerboardCase,
    problem: TransDimensionalProblem,
    *,
    seed: int,
) -> _BenchmarkMetrics:
    """Run the common full sampler configuration and summarize recovery."""
    initial_state = build_state(
        problem,
        _INITIAL_NUCLEI,
        np.ones(_INITIAL_NUCLEI.size),
        backend="numba",
    )
    started = perf_counter()
    result = sample(
        problem,
        initial_state,
        SamplerConfig(
            iterations=_BENCHMARK_ITERATIONS,
            coefficient_proposal_sd=0.1,
            birth_proposal_sd=0.35,
            seed=seed,
            backend="numba",
            nucleus_move="local",
            local_move_scale=1.4,
        ),
    )
    runtime_seconds = perf_counter() - started
    return _summarize_benchmark(case, problem, result.trace, runtime_seconds)


def test_checkerboard_problem_has_paper_like_declared_structure() -> None:
    """The local case should close its grid, smooth design, prior, and noise contracts."""
    case = _build_lunt_checkerboard_case()

    assert case.problem.grid_coordinates.shape == (64, 2)
    assert case.problem.sensitivities.shape == (96, 64)
    assert np.all(case.problem.sensitivities > 0.0)
    np.testing.assert_allclose(
        case.problem.sensitivities.sum(axis=1),
        case.footprint_amplitudes,
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_array_equal(np.unique(case.truth), [0.5, 1.5])
    np.testing.assert_array_equal(case.truth.reshape(_GRID_SHAPE)[:2, :4], [[0.5, 0.5, 1.5, 1.5]] * 2)
    np.testing.assert_allclose(
        case.problem.observations,
        case.noiseless_observations + case.noise,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(case.problem.observation_sd, np.full(96, 5.0))
    assert case.problem.coefficient_prior_mean == 1.0
    assert case.problem.coefficient_prior_sd == 1.0


def test_true_voronoi_state_and_native_grid_reconstruction_close() -> None:
    """The exact sixteen-region state should recover truth and noise-free observations."""
    case = _build_lunt_checkerboard_case()
    state = _true_state(case)
    summary = summarize_fine_grid_posterior(
        case.problem,
        _single_state_trace(state),
        case.noiseless_observations,
    )

    np.testing.assert_array_equal(state.active_nuclei, _TRUTH_NUCLEI)
    np.testing.assert_array_equal(summary.samples, case.truth[np.newaxis, :])
    np.testing.assert_array_equal(summary.mean, case.truth)
    np.testing.assert_allclose(
        summary.predicted_observations,
        case.noiseless_observations,
        rtol=0.0,
        atol=2e-14,
    )
    assert summary.rmse < 1e-14


@pytest.mark.slow
def test_fair_fixed_and_trans_dimensional_checkerboard_comparison() -> None:
    """Seeded alternatives should show the expected oracle-to-random ordering.

    The oracle is advantaged by knowing the true partition. The movable fixed-k
    and trans-dimensional chains instead share one seeded non-oracle partition
    with all-one coefficients. Random fixed layouts measure sensitivity to a
    deliberately misspecified partition. This test is not a comparison with a
    full production RHIME/PyMC inversion: its fixed-basis comparator is only
    coefficient-only MH, and the reference implementations rebuild complete
    states rather than representing production runtime.
    """
    case = _build_lunt_checkerboard_case()
    fixed_problem = _fixed_k_problem(case)

    oracle_metrics: list[_BenchmarkMetrics] = []
    movable_fixed_k_metrics: list[_BenchmarkMetrics] = []
    random_fixed_metrics: list[_BenchmarkMetrics] = []
    trans_dimensional_metrics: list[_BenchmarkMetrics] = []
    for chain_seed, layout_seed in zip(_BENCHMARK_SEEDS, _RANDOM_LAYOUT_SEEDS, strict=True):
        oracle_trace, oracle_runtime = _sample_fixed_nuclei_coefficients(
            fixed_problem,
            _TRUTH_NUCLEI,
            seed=chain_seed,
        )
        oracle_metrics.append(_summarize_benchmark(case, fixed_problem, oracle_trace, oracle_runtime))

        random_nuclei = np.sort(
            np.random.default_rng(layout_seed).choice(
                case.problem.n_grid_cells,
                size=_TRUTH_NUCLEI.size,
                replace=False,
            )
        ).astype(np.int64)
        random_trace, random_runtime = _sample_fixed_nuclei_coefficients(
            fixed_problem,
            random_nuclei,
            seed=chain_seed,
        )
        random_fixed_metrics.append(_summarize_benchmark(case, fixed_problem, random_trace, random_runtime))

        movable_fixed_k_metrics.append(_run_full_sampler_benchmark(case, fixed_problem, seed=chain_seed))
        trans_dimensional_metrics.append(_run_full_sampler_benchmark(case, case.problem, seed=chain_seed))

    prior_prediction = case.problem.sensitivities @ np.ones(case.problem.n_grid_cells)
    prior_rmse = float(np.sqrt(np.mean(np.square(prior_prediction - case.noiseless_observations))))
    oracle_prediction_rmse = np.array([metric.prediction_rmse for metric in oracle_metrics])
    movable_prediction_rmse = np.array([metric.prediction_rmse for metric in movable_fixed_k_metrics])
    random_prediction_rmse = np.array([metric.prediction_rmse for metric in random_fixed_metrics])
    trans_dimensional_prediction_rmse = np.array(
        [metric.prediction_rmse for metric in trans_dimensional_metrics]
    )
    # Knowing the correct partition is an oracle advantage. Random layouts are
    # independent replicates rather than pooled draws, and their median should
    # be worse than the oracle fixed basis.
    assert np.median(random_prediction_rmse) > np.median(oracle_prediction_rmse)

    for metrics in (oracle_metrics, movable_fixed_k_metrics, trans_dimensional_metrics):
        assert max(metric.prediction_rmse for metric in metrics) < 0.5 * prior_rmse
        assert min(metric.spatial_correlation for metric in metrics) > 0.55
        assert min(metric.contrast for metric in metrics) > 0.5

    assert all(metric.visited_k == (16,) for metric in oracle_metrics)
    assert all(metric.visited_k == (16,) for metric in random_fixed_metrics)
    assert all(metric.visited_k == (16,) for metric in movable_fixed_k_metrics)
    assert all(len(metric.visited_k) > 1 for metric in trans_dimensional_metrics)

    # Guard broad calibrated ranges without asserting which adaptive method is
    # superior: that conclusion would require a production-quality fixed-basis
    # comparator and considerably longer convergence diagnostics.
    assert np.all((1.5 < oracle_prediction_rmse) & (oracle_prediction_rmse < 2.5))
    assert np.all((2.0 < movable_prediction_rmse) & (movable_prediction_rmse < 5.0))
    assert np.all((2.0 < trans_dimensional_prediction_rmse) & (trans_dimensional_prediction_rmse < 7.0))
    assert all(metric.grid_rmse < 0.4 for metric in movable_fixed_k_metrics)
    assert all(metric.grid_rmse < 0.5 for metric in trans_dimensional_metrics)
