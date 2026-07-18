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
  at 40; the local start uses the regular checkerboard nucleus layout with all
  coefficients set to the prior mean, rather than paper-era random nuclei;
- the local run uses 40,000 transitions, a 15,000-row burn cutoff, thinning by
  10, and one seeded chain, not a paper-era production/convergence run;
- the current four-slot non-hierarchical sampler, with two invariant 50/50
  birth/death slots and a corrected finite-grid local move, replaces the exact
  historical proposal implementation.

Consequently, posterior ``k`` and published RMSE values are intentionally not
asserted. The slow seeded recovery smoke test only requires improvement relative
to the all-ones prior, the correct checkerboard contrast direction, and positive
spatial correlation with the known native-grid truth.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from openghg_inversions.tdmcmc.sampling import SamplerConfig, SamplingTrace, sample

FloatArray = NDArray[np.float64]

_GRID_SHAPE = (8, 8)
_OBSERVATION_COUNT = 96
_OBSERVATION_SD = 5.0
_TRUTH_NUCLEI = np.array(
    [0, 2, 4, 6, 16, 18, 20, 22, 32, 34, 36, 38, 48, 50, 52, 54],
    dtype=np.int64,
)
_INITIAL_NUCLEI = _TRUTH_NUCLEI.copy()


@dataclass(frozen=True, slots=True)
class LuntCheckerboardCase:
    """Complete deterministic local checkerboard inversion benchmark."""

    problem: TransDimensionalProblem
    truth: FloatArray
    noise: FloatArray
    noiseless_observations: FloatArray
    footprint_amplitudes: FloatArray


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
def test_seeded_numba_sampler_recovers_checkerboard_signal() -> None:
    """A local-move chain should improve prediction and recover spatial contrast."""
    case = _build_lunt_checkerboard_case()
    initial_state = build_state(
        case.problem,
        _INITIAL_NUCLEI,
        np.ones(_INITIAL_NUCLEI.size),
        backend="numba",
    )
    result = sample(
        case.problem,
        initial_state,
        SamplerConfig(
            iterations=40_000,
            coefficient_proposal_sd=0.1,
            birth_proposal_sd=0.35,
            seed=481,
            backend="numba",
            nucleus_move="local",
            local_move_scale=1.4,
        ),
    )
    summary = summarize_fine_grid_posterior(
        case.problem,
        result.trace,
        case.noiseless_observations,
        start=15_000,
        thin=10,
        backend="numba",
    )

    prior_prediction = case.problem.sensitivities @ np.ones(case.problem.n_grid_cells)
    prior_rmse = float(np.sqrt(np.mean(np.square(prior_prediction - case.noiseless_observations))))
    spatial_correlation = float(np.corrcoef(summary.mean, case.truth)[0, 1])
    low_mean = float(np.mean(summary.mean[case.truth == 0.5]))
    high_mean = float(np.mean(summary.mean[case.truth == 1.5]))

    assert summary.trace_rows.size == 2_501
    assert summary.rmse < 0.4 * prior_rmse
    assert spatial_correlation > 0.6
    assert high_mean - low_mean > 0.55
