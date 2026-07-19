"""Tests for fine-grid TD-MCMC posterior reconstruction and summaries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from math import sqrt

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.core import TransDimensionalProblem, uniform_log_k_prior
from openghg_inversions.experimental.rjmcmc.postprocessing import (
    FineGridPosteriorSummary,
    posterior_mean_prediction,
    reconstruct_fine_grid_samples,
    summarize_fine_grid_posterior,
)
from openghg_inversions.experimental.rjmcmc.sampling import SamplingTrace


def _problem() -> TransDimensionalProblem:
    """Return a four-cell problem with an exact observation-space projection."""
    return TransDimensionalProblem(
        observations=np.zeros(2),
        observation_sd=np.ones(2),
        sensitivities=np.array(
            [
                [1.0, 0.0, 2.0, -1.0],
                [0.5, 1.0, 0.0, 2.0],
            ]
        ),
        grid_coordinates=np.arange(4, dtype=np.float64)[:, np.newaxis],
        k_min=1,
        k_max=2,
        log_k_prior=uniform_log_k_prior(1, 2),
        coefficient_prior_mean=2.0,
        coefficient_prior_sd=1.0,
    )


def _trace() -> SamplingTrace:
    """Return three padded states with different native-grid partitions."""
    return SamplingTrace(
        k=np.array([1, 2, 2], dtype=np.int64),
        nuclei=np.array(
            [
                [0, -1],
                [0, 3],
                [1, 3],
            ],
            dtype=np.int64,
        ),
        coefficients=np.array(
            [
                [1.0, 0.0],
                [2.0, 4.0],
                [6.0, 8.0],
            ]
        ),
        log_target=np.zeros(3),
        moves=np.array(["birth", "global_move"]),
        accepted=np.ones(2, dtype=np.bool_),
        log_acceptance_ratio=np.zeros(2),
    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_reconstruction_uses_each_retained_rows_partition(backend: str) -> None:
    """Each row's nuclei and aligned coefficients should define its fine-grid field."""
    samples = reconstruct_fine_grid_samples(
        _problem(),
        _trace(),
        backend=backend,  # type: ignore[arg-type]
    )

    np.testing.assert_array_equal(
        samples,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 4.0, 4.0],
                [6.0, 6.0, 6.0, 8.0],
            ]
        ),
    )
    assert not samples.flags.writeable


def test_start_and_thin_select_saved_state_rows() -> None:
    """Trace slicing should count saved states, including the initial row zero."""
    samples = reconstruct_fine_grid_samples(_problem(), _trace(), start=0, thin=2)

    np.testing.assert_array_equal(
        samples,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [6.0, 6.0, 6.0, 8.0],
            ]
        ),
    )


def test_summary_calculates_default_quantiles_and_noise_free_rmse() -> None:
    """A mapped posterior mean should reproduce a noise-free comparison exactly."""
    problem = _problem()
    comparison = np.array([6.0, 14.25])

    summary = summarize_fine_grid_posterior(
        problem,
        _trace(),
        comparison,
        start=0,
        thin=2,
    )

    assert isinstance(summary, FineGridPosteriorSummary)
    np.testing.assert_array_equal(summary.trace_rows, [0, 2])
    np.testing.assert_array_equal(summary.mean, [3.5, 3.5, 3.5, 4.5])
    np.testing.assert_array_equal(summary.quantile_levels, [0.05, 0.5, 0.95])
    np.testing.assert_allclose(
        summary.quantiles,
        np.array(
            [
                [1.25, 1.25, 1.25, 1.35],
                [3.5, 3.5, 3.5, 4.5],
                [5.75, 5.75, 5.75, 7.65],
            ]
        ),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(summary.predicted_observations, comparison)
    assert summary.rmse == 0.0
    for array in (
        summary.trace_rows,
        summary.samples,
        summary.mean,
        summary.quantile_levels,
        summary.quantiles,
        summary.predicted_observations,
    ):
        assert not array.flags.writeable
    with pytest.raises(FrozenInstanceError):
        summary.rmse = 1.0  # type: ignore[misc]


def test_summary_accepts_custom_quantiles() -> None:
    """Requested ordered probabilities should control the summary's quantile axis."""
    summary = summarize_fine_grid_posterior(
        _problem(),
        _trace(),
        np.zeros(2),
        quantiles=(0.25, 0.75),
    )

    expected_samples = reconstruct_fine_grid_samples(_problem(), _trace())
    np.testing.assert_array_equal(summary.quantile_levels, [0.25, 0.75])
    np.testing.assert_allclose(
        summary.quantiles,
        np.quantile(expected_samples, [0.25, 0.75], axis=0),
    )


def test_posterior_mean_prediction_reports_nonzero_rmse() -> None:
    """Observation prediction and RMSE should follow their direct definitions."""
    predicted, rmse = posterior_mean_prediction(
        _problem(),
        posterior_mean=np.array([1.0, 2.0, 3.0, 4.0]),
        comparison=np.array([1.0, 9.5]),
    )

    np.testing.assert_array_equal(predicted, [3.0, 10.5])
    assert rmse == pytest.approx(sqrt(2.5))
    assert not predicted.flags.writeable


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start": -1}, "start"),
        ({"start": 3}, "start"),
        ({"start": 0.5}, "start"),
        ({"start": True}, "start"),
        ({"thin": 0}, "thin"),
        ({"thin": -1}, "thin"),
        ({"thin": 1.5}, "thin"),
        ({"thin": True}, "thin"),
        ({"backend": "jax"}, "backend"),
    ],
)
def test_reconstruction_rejects_invalid_slicing_and_backend(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Only existing saved rows, positive integer thinning, and known backends are valid."""
    with pytest.raises(ValueError, match=message):
        reconstruct_fine_grid_samples(_problem(), _trace(), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "quantiles",
    [
        (),
        0.5,
        (-0.1, 0.5),
        (0.5, 1.1),
        (0.5, 0.5),
        (0.9, 0.1),
        (0.1, np.nan),
    ],
)
def test_summary_rejects_invalid_quantiles(quantiles: object) -> None:
    """Quantile levels must be finite, one-dimensional, ordered, unique probabilities."""
    with pytest.raises(ValueError, match="quantiles"):
        summarize_fine_grid_posterior(
            _problem(),
            _trace(),
            np.zeros(2),
            quantiles=quantiles,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("posterior_mean", "comparison", "message"),
    [
        (np.ones((2, 2)), np.ones(2), "posterior_mean"),
        (np.ones(4), np.ones(3), "comparison"),
        (np.array([1.0, 2.0, np.nan, 4.0]), np.ones(2), "posterior_mean"),
        (np.ones(4), np.array([1.0, np.inf]), "comparison"),
    ],
)
def test_prediction_rejects_invalid_vectors(
    posterior_mean: np.ndarray,
    comparison: np.ndarray,
    message: str,
) -> None:
    """Prediction inputs must be finite vectors aligned with the numerical problem."""
    with pytest.raises(ValueError, match=message):
        posterior_mean_prediction(_problem(), posterior_mean, comparison)


@pytest.mark.parametrize(
    ("trace", "message"),
    [
        (replace(_trace(), k=np.array([[1, 2, 2]])), "trace.k"),
        (replace(_trace(), nuclei=np.array([[0], [0], [1]])), "trace.nuclei"),
        (replace(_trace(), coefficients=np.ones((3, 1))), "trace.coefficients"),
        (replace(_trace(), k=np.array([0, 2, 2])), "supported range"),
        (
            replace(
                _trace(),
                nuclei=np.array([[0, -1], [0, 4], [1, 3]]),
            ),
            "invalid active grid index",
        ),
        (
            replace(
                _trace(),
                nuclei=np.array([[0, -1], [0, 0], [1, 3]]),
            ),
            "unique active indices",
        ),
        (
            replace(
                _trace(),
                nuclei=np.array([[0, 1], [0, 3], [1, 3]]),
            ),
            "inactive padding",
        ),
        (
            replace(
                _trace(),
                coefficients=np.array([[1.0, 9.0], [2.0, 4.0], [6.0, 8.0]]),
            ),
            "inactive padding",
        ),
    ],
)
def test_reconstruction_rejects_malformed_trace_fields(
    trace: SamplingTrace,
    message: str,
) -> None:
    """Padded trace arrays must be shape-compatible and contain valid active indices."""
    with pytest.raises(ValueError, match=message):
        reconstruct_fine_grid_samples(_problem(), trace)
