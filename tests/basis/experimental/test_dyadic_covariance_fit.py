"""Tests for conventional distance-covariance reference fits."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic import (
    fit_projected_exponential_length_scale,
    fit_separable_exponential_length_scale,
    grouped_exponential_covariance,
    projected_exponential_covariance,
    separable_exponential_correlation,
)


def test_fit_recovers_synthetic_length_scale_with_fixed_variances() -> None:
    """The covariance fit recovers a known scale with unequal marginals."""
    latitude = np.array([40.0, 44.0, 51.0, 58.0])
    longitude = np.array([-8.0, 3.0, 12.0, 21.0])
    sigma = np.array([0.5, 1.0, 2.0, 3.0])
    expected_scale = 7.5
    correlation = separable_exponential_correlation(latitude, longitude, expected_scale)
    covariance = sigma[:, None] * sigma[None, :] * correlation

    fit = fit_separable_exponential_length_scale(
        covariance,
        latitude,
        longitude,
        standard_deviation=sigma,
    )

    assert fit.converged
    assert fit.length_scale == pytest.approx(expected_scale, rel=1.0e-6)
    assert fit.rmse < 1.0e-8
    assert fit.relative_rmse < 5.0e-8
    assert fit.target_model_correlation == pytest.approx(1.0)
    assert fit.pair_count == 6


def test_fit_uses_target_diagonal_when_sigma_is_omitted() -> None:
    """Omitted standard deviations are inferred from the target diagonal."""
    latitude = np.array([45.0, 50.0, 55.0])
    longitude = np.array([0.0, 4.0, 9.0])
    sigma = np.array([0.75, 1.25, 2.5])
    correlation = separable_exponential_correlation(latitude, longitude, 11.0)
    covariance = sigma[:, None] * sigma[None, :] * correlation

    fit = fit_separable_exponential_length_scale(covariance, latitude, longitude)

    assert fit.length_scale == pytest.approx(11.0, rel=1.0e-6)


def test_grouped_covariance_preserves_diagonal_and_zeros_cross_group_pairs() -> None:
    """Hard-class covariance keeps requested variances and group boundaries."""
    sigma = np.array([0.5, 1.0, 2.0, 3.0])
    latitude = np.array([40.0, 42.0, 50.0, 52.0])
    longitude = np.array([0.0, 1.0, 10.0, 11.0])
    groups = np.array([0, 0, 1, 1])

    covariance = grouped_exponential_covariance(
        sigma,
        latitude,
        longitude,
        groups,
        length_scale=5.0,
    )

    np.testing.assert_allclose(np.diag(covariance), np.square(sigma))
    np.testing.assert_array_equal(covariance[:2, 2:], 0.0)
    assert covariance[0, 1] > 0.0
    assert covariance[2, 3] > 0.0
    np.testing.assert_allclose(covariance, covariance.T)


def test_projected_covariance_matches_explicit_pbp_transformation() -> None:
    """The matrix-free regional result equals an explicit native covariance."""
    latitude = np.array([40.0, 45.0])
    longitude = np.array([-5.0, 0.0, 8.0])
    projection = np.array(
        [
            [0.7, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.6, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.8],
        ]
    )
    length_scale = 6.0
    latitude_factor = np.exp(-np.abs(latitude[:, None] - latitude[None, :]) / length_scale)
    longitude_factor = np.exp(-np.abs(longitude[:, None] - longitude[None, :]) / length_scale)
    native_covariance = np.kron(latitude_factor, longitude_factor)

    projected = projected_exponential_covariance(
        latitude,
        longitude,
        projection,
        length_scale,
    )

    np.testing.assert_allclose(projected, projection.T @ native_covariance @ projection, atol=1.0e-13)


def test_projected_fit_recovers_scale_after_regional_normalization() -> None:
    """The projected fit recovers a known native scale on fixed regions."""
    latitude = np.array([40.0, 45.0])
    longitude = np.array([-5.0, 0.0, 8.0])
    projection = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.75],
        ]
    )
    sigma = np.array([0.5, 1.0, 2.0])
    expected_scale = 9.0
    target = projected_exponential_covariance(
        latitude,
        longitude,
        projection,
        expected_scale,
        regional_standard_deviation=sigma,
    )

    fit = fit_projected_exponential_length_scale(
        target,
        latitude,
        longitude,
        projection,
        standard_deviation=sigma,
    )

    assert fit.converged
    assert fit.length_scale == pytest.approx(expected_scale, rel=2.0e-5)
    assert fit.relative_rmse < 1.0e-5


@pytest.mark.parametrize(
    ("target", "latitude", "longitude", "message"),
    [
        (np.eye(2), np.array([1.0]), np.array([1.0]), "same regions"),
        (np.eye(1), np.array([1.0]), np.array([1.0]), "At least two"),
        (np.array([[1.0, 0.1], [0.2, 1.0]]), np.arange(2), np.arange(2), "symmetric"),
    ],
)
def test_fit_rejects_inconsistent_inputs(
    target: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    message: str,
) -> None:
    """The fit rejects inconsistent dimensions and non-covariance matrices."""
    with pytest.raises(ValueError, match=message):
        fit_separable_exponential_length_scale(target, latitude, longitude)
