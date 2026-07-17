"""Tests for the matrix-free separable dyadic grid covariance operator."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.grid_covariance import (
    SeparableGridCovariance,
)


@pytest.fixture
def covariance_problem() -> tuple[SeparableGridCovariance, np.ndarray]:
    """Return a small operator and its explicit row-major covariance."""
    latitude = np.array([50.0, 51.5])
    longitude = np.array([-2.0, 0.5, 3.0])
    operator = SeparableGridCovariance(
        latitude,
        longitude,
        latitude_length_scale=2.25,
        longitude_length_scale=1.75,
    )
    explicit = np.kron(operator.latitude_factor, operator.longitude_factor)
    return operator, explicit


def test_apply_matches_explicit_kronecker_for_vector_and_batch(
    covariance_problem: tuple[SeparableGridCovariance, np.ndarray],
) -> None:
    """Vector and batched applications should match row-major ``np.kron``."""
    operator, explicit = covariance_problem
    vector = np.array([0.5, -1.2, 0.3, 2.1, -0.7, 1.4])
    batch = np.column_stack((vector, vector[::-1], np.arange(operator.size)))

    np.testing.assert_allclose(operator.apply(vector), explicit @ vector, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(operator.apply(batch), explicit @ batch, rtol=1e-13, atol=1e-13)
    assert operator.apply(vector).shape == vector.shape
    assert operator.apply(batch).shape == batch.shape


def test_marginal_standard_deviations_scale_both_covariance_sides(
    covariance_problem: tuple[SeparableGridCovariance, np.ndarray],
) -> None:
    """Marginal standard deviations should produce ``diag(s) K diag(s)``."""
    base_operator, base_covariance = covariance_problem
    marginal_sds = np.array([[0.4, 1.1, 0.0], [1.8, 0.7, 2.2]])
    operator = SeparableGridCovariance(
        base_operator.latitude,
        base_operator.longitude,
        latitude_length_scale=base_operator.latitude_length_scale,
        longitude_length_scale=base_operator.longitude_length_scale,
        marginal_standard_deviations=marginal_sds,
    )
    vector = np.linspace(-1.0, 1.0, operator.size)
    flattened_sds = marginal_sds.reshape(-1)
    expected = (flattened_sds[:, None] * base_covariance) * flattened_sds[None, :]

    np.testing.assert_allclose(operator.apply(vector), expected @ vector, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.diag(expected), flattened_sds**2, rtol=1e-13, atol=1e-13)


def test_class_labels_block_land_ocean_cross_covariance(
    covariance_problem: tuple[SeparableGridCovariance, np.ndarray],
) -> None:
    """Land and ocean classes should retain within-class and zero cross-class covariance."""
    base_operator, base_covariance = covariance_problem
    land_ocean = np.array([[0, 1, 1], [0, 0, 1]])
    operator = SeparableGridCovariance(
        base_operator.latitude,
        base_operator.longitude,
        latitude_length_scale=base_operator.latitude_length_scale,
        longitude_length_scale=base_operator.longitude_length_scale,
        class_labels=land_ocean,
    )
    flattened_classes = land_ocean.reshape(-1)
    explicit = base_covariance * (flattened_classes[:, None] == flattened_classes[None, :])
    batch = np.arange(18.0).reshape(operator.size, 3) - 4.0

    np.testing.assert_allclose(operator.apply(batch), explicit @ batch, rtol=1e-13, atol=1e-13)
    assert np.all(explicit[flattened_classes[:, None] != flattened_classes[None, :]] == 0.0)


def test_missing_classes_are_rejected_only_at_active_inputs() -> None:
    """Missing or infinite classes should fail when their input rows become active."""
    operator = SeparableGridCovariance(
        [0.0, 1.0],
        [10.0, 11.0],
        latitude_length_scale=1.0,
        longitude_length_scale=1.0,
        class_labels=np.array([[0.0, np.nan], [1.0, np.inf]]),
    )
    classified_only = np.array([2.0, 0.0, -1.0, 0.0])
    result = operator.apply(classified_only)

    assert result[1] == 0.0
    assert result[3] == 0.0
    for invalid_index in (1, 3):
        active = classified_only.copy()
        active[invalid_index] = 1.0
        with pytest.raises(ValueError, match="finite at every active input"):
            operator.apply(active)


def test_projected_and_observation_covariances_match_explicit_algebra(
    covariance_problem: tuple[SeparableGridCovariance, np.ndarray],
) -> None:
    """Projection, signal, and cross-covariance helpers should match dense products."""
    operator, explicit = covariance_problem
    projection = np.array(
        [
            [0.5, 0.0],
            [0.5, 0.0],
            [0.0, 0.2],
            [0.0, 0.3],
            [0.0, 0.1],
            [0.0, 0.4],
        ]
    )
    observation_operator = np.array(
        [
            [1.0, 0.0, -0.2, 0.4, 0.0, 0.3],
            [0.0, 0.5, 0.5, 0.0, -0.1, 0.2],
        ]
    )

    np.testing.assert_allclose(
        operator.projected_covariance(projection),
        projection.T @ explicit @ projection,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        operator.observation_cross_covariance(observation_operator),
        explicit @ observation_operator.T,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        operator.observation_signal_covariance(observation_operator),
        observation_operator @ explicit @ observation_operator.T,
        rtol=1e-13,
        atol=1e-13,
    )


def test_scaled_class_blocked_batch_and_helpers_match_one_dense_covariance(
    covariance_problem: tuple[SeparableGridCovariance, np.ndarray],
) -> None:
    """All operator features should compose under one explicit dense oracle."""
    base_operator, base_covariance = covariance_problem
    marginal_sds = np.array([0.4, 1.1, 0.8, 1.8, 0.7, 2.2])
    classes = np.array([0, 1, 1, 0, 0, 1])
    operator = SeparableGridCovariance(
        base_operator.latitude,
        base_operator.longitude,
        latitude_length_scale=base_operator.latitude_length_scale,
        longitude_length_scale=base_operator.longitude_length_scale,
        marginal_standard_deviations=marginal_sds,
        class_labels=classes,
    )
    class_block = classes[:, None] == classes[None, :]
    explicit = (
        marginal_sds[:, None]
        * (base_covariance * class_block)
        * marginal_sds[None, :]
    )
    batch = np.arange(18.0).reshape(operator.size, 3) / 7.0
    projection = np.arange(12.0).reshape(operator.size, 2) / 9.0
    observation_operator = np.arange(18.0).reshape(3, operator.size) / 11.0

    np.testing.assert_allclose(operator.apply(batch), explicit @ batch, atol=1e-13)
    np.testing.assert_allclose(
        operator.projected_covariance(projection),
        projection.T @ explicit @ projection,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        operator.observation_cross_covariance(observation_operator),
        explicit @ observation_operator.T,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        operator.observation_signal_covariance(observation_operator),
        observation_operator @ explicit @ observation_operator.T,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("marginal_standard_deviations", np.ones((3, 2)), "must have shape"),
        ("marginal_standard_deviations", [1.0, 1.0, 1.0, np.nan], "must be finite"),
        ("marginal_standard_deviations", [1.0, -0.1, 1.0, 1.0], "non-negative"),
        ("class_labels", np.ones((1, 4)), "must have shape"),
    ],
)
def test_constructor_rejects_invalid_grid_metadata(
    keyword: str,
    value: np.ndarray | list[float],
    message: str,
) -> None:
    """Grid metadata should have compatible shapes and valid standard deviations."""
    with pytest.raises(ValueError, match=message):
        SeparableGridCovariance(
            [0.0, 1.0],
            [10.0, 11.0],
            latitude_length_scale=1.0,
            longitude_length_scale=1.0,
            **{keyword: value},
        )


@pytest.mark.parametrize(
    ("latitude", "longitude", "latitude_scale", "longitude_scale", "message"),
    [
        (np.ones((1, 2)), [0.0], 1.0, 1.0, "latitude must be one-dimensional"),
        ([], [0.0], 1.0, 1.0, "latitude must not be empty"),
        ([np.nan], [0.0], 1.0, 1.0, "latitude must be finite"),
        ([0.0], [np.inf], 1.0, 1.0, "longitude must be finite"),
        ([0.0], [1.0], 0.0, 1.0, "latitude_length_scale must be positive"),
        ([0.0], [1.0], 1.0, np.inf, "longitude_length_scale must be positive"),
    ],
)
def test_constructor_rejects_invalid_coordinates_and_scales(
    latitude: np.ndarray | list[float],
    longitude: np.ndarray | list[float],
    latitude_scale: float,
    longitude_scale: float,
    message: str,
) -> None:
    """Coordinates and length scales should be finite and dimensionally valid."""
    with pytest.raises(ValueError, match=message):
        SeparableGridCovariance(
            latitude,
            longitude,
            latitude_length_scale=latitude_scale,
            longitude_length_scale=longitude_scale,
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.ones((2, 2, 1)), r"shape \(M,\) or \(M, n_rhs\)"),
        (np.ones(3), "must have 4 rows"),
        (np.empty((4, 0)), "at least one right-hand side"),
        (np.array([1.0, np.nan, 2.0, 3.0]), "must be finite"),
        (np.array([1.0 + 1.0j, 0.0, 0.0, 0.0]), "must be real"),
    ],
)
def test_apply_rejects_invalid_dimensions_and_values(values: np.ndarray, message: str) -> None:
    """Application should reject incompatible, non-finite, and complex inputs."""
    operator = SeparableGridCovariance(
        [0.0, 1.0],
        [10.0, 11.0],
        latitude_length_scale=1.0,
        longitude_length_scale=1.0,
    )

    with pytest.raises(ValueError, match=message):
        operator.apply(values)


def test_projection_and_observation_helpers_validate_dense_inputs(
    covariance_problem: tuple[SeparableGridCovariance, np.ndarray],
) -> None:
    """Dense helper matrices should be finite, two-dimensional, and compatible."""
    operator, _ = covariance_problem

    with pytest.raises(ValueError, match="projection must be two-dimensional"):
        operator.projected_covariance(np.ones(operator.size))
    with pytest.raises(ValueError, match=f"projection must have {operator.size} rows"):
        operator.projected_covariance(np.ones((operator.size - 1, 2)))
    invalid_projection = np.ones((operator.size, 2))
    invalid_projection[0, 0] = np.nan
    with pytest.raises(ValueError, match="projection must be finite"):
        operator.projected_covariance(invalid_projection)
    with pytest.raises(ValueError, match="observation_operator must be two-dimensional"):
        operator.observation_cross_covariance(np.ones(operator.size))
    with pytest.raises(ValueError, match=f"observation_operator must have {operator.size} columns"):
        operator.observation_signal_covariance(np.ones((2, operator.size - 1)))
    invalid_observation_operator = np.ones((2, operator.size))
    invalid_observation_operator[0, 0] = np.inf
    with pytest.raises(ValueError, match="observation_operator must be finite"):
        operator.observation_signal_covariance(invalid_observation_operator)


def test_operator_arrays_are_immutable_and_detached() -> None:
    """Stored coordinates and metadata should be read-only copies of caller arrays."""
    latitude = np.array([0.0, 1.0])
    longitude = np.array([10.0, 11.0])
    marginal_sds = np.ones((2, 2))
    class_labels = np.array([[0.0, 0.0], [1.0, 1.0]])
    operator = SeparableGridCovariance(
        latitude,
        longitude,
        latitude_length_scale=1.0,
        longitude_length_scale=1.0,
        marginal_standard_deviations=marginal_sds,
        class_labels=class_labels,
    )
    latitude[0] = 99.0
    marginal_sds[0, 0] = 99.0
    class_labels[0, 0] = 99.0

    assert operator.latitude[0] == 0.0
    assert operator.marginal_standard_deviations[0] == 1.0
    assert operator.class_labels is not None
    assert operator.class_labels[0] == 0.0
    for array in (
        operator.latitude,
        operator.longitude,
        operator.latitude_factor,
        operator.longitude_factor,
        operator.marginal_standard_deviations,
        operator.class_labels,
    ):
        assert not array.flags.writeable
