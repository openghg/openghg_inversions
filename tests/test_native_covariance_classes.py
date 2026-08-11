"""Tests for optional class blocking in native covariance actions."""

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.native_covariance import SeparableExponentialCovariance


@pytest.fixture
def coordinates() -> tuple[xr.DataArray, xr.DataArray]:
    """Return a small irregular native grid with degree metadata."""
    latitude = xr.DataArray(
        [-1.0, 0.5, 2.0],
        dims="lat",
        coords={"lat": [-1.0, 0.5, 2.0]},
        attrs={"units": "degrees_north"},
    )
    longitude = xr.DataArray(
        [10.0, 11.0],
        dims="lon",
        coords={"lon": [10.0, 11.0]},
        attrs={"units": "degrees_east"},
    )
    return latitude, longitude


def _dense_blocked_covariance(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    class_labels: xr.DataArray,
    sigma: float,
    correlation_length: float,
) -> np.ndarray:
    """Build the small dense class-blocked oracle used only by tests."""
    lat_values = np.asarray(latitude.values)
    lon_values = np.asarray(longitude.values)
    lat_factor = np.exp(-np.abs(lat_values[:, None] - lat_values[None, :]) / correlation_length)
    lon_factor = np.exp(-np.abs(lon_values[:, None] - lon_values[None, :]) / correlation_length)
    unblocked = sigma**2 * np.kron(lat_factor, lon_factor)
    labels = np.asarray(class_labels.transpose("lat", "lon").values).reshape(-1)
    return unblocked * (labels[:, None] == labels[None, :])


def test_class_blocked_apply_and_solve_match_dense_oracle(
    coordinates: tuple[xr.DataArray, xr.DataArray],
) -> None:
    """Class-blocked actions match a dense oracle for multiple RHS layouts."""
    latitude, longitude = coordinates
    labels = xr.DataArray(
        [["land", "sea"], ["land", "sea"], ["sea", "sea"]],
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
        name="region_class",
    )
    covariance = SeparableExponentialCovariance(
        latitude,
        longitude,
        sigma=1.7,
        correlation_length=1.2,
        class_labels=labels,
    )
    rhs = xr.DataArray(
        np.arange(12, dtype=float).reshape(2, 2, 3) / 7.0,
        dims=("realisation", "lon", "lat"),
        coords={"realisation": ["a", "b"], "lat": latitude, "lon": longitude},
        name="rhs",
        attrs={"units": "1"},
    )

    dense = _dense_blocked_covariance(latitude, longitude, labels, 1.7, 1.2)
    native_rhs = rhs.transpose("lat", "lon", "realisation").values.reshape(6, 2)
    expected_apply = dense @ native_rhs
    expected_solve = np.linalg.solve(dense, native_rhs)

    applied = covariance.apply(rhs)
    solved = covariance.solve(rhs)

    assert applied.dims == rhs.dims
    assert solved.dims == rhs.dims
    assert applied.name == rhs.name
    assert applied.attrs == rhs.attrs
    xr.testing.assert_identical(applied.coords.to_dataset(), rhs.coords.to_dataset())
    np.testing.assert_allclose(
        applied.transpose("lat", "lon", "realisation").values.reshape(6, 2),
        expected_apply,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        solved.transpose("lat", "lon", "realisation").values.reshape(6, 2),
        expected_solve,
        rtol=1e-10,
        atol=1e-11,
    )
    xr.testing.assert_allclose(covariance.apply(solved), rhs, rtol=1e-10, atol=1e-11)


def test_class_labels_round_trip_through_dataset(
    coordinates: tuple[xr.DataArray, xr.DataArray],
) -> None:
    """Serialization retains class labels and reproduces the blocked action."""
    latitude, longitude = coordinates
    labels = xr.DataArray(
        [[0, 1], [0, 1], [1, 1]],
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
        attrs={"meaning": "surface class"},
    )
    original = SeparableExponentialCovariance(latitude, longitude, class_labels=labels)

    restored = SeparableExponentialCovariance.from_dataset(original.to_dataset())

    assert restored.class_labels is not None
    xr.testing.assert_identical(restored.class_labels, labels)
    rhs = xr.ones_like(labels, dtype=float)
    xr.testing.assert_allclose(restored.apply(rhs), original.apply(rhs))


def test_single_class_uses_unblocked_covariance(
    coordinates: tuple[xr.DataArray, xr.DataArray],
) -> None:
    """A single class is algebraically identical to the separable covariance."""
    latitude, longitude = coordinates
    labels = xr.DataArray(
        np.full((3, 2), "all"),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )
    blocked = SeparableExponentialCovariance(latitude, longitude, class_labels=labels)
    unblocked = SeparableExponentialCovariance(latitude, longitude)
    rhs = xr.DataArray(
        np.arange(6, dtype=float).reshape(3, 2),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )

    xr.testing.assert_allclose(blocked.apply(rhs), unblocked.apply(rhs))
    xr.testing.assert_allclose(blocked.solve(rhs), unblocked.solve(rhs))


@pytest.mark.parametrize("invalid_value", [None, np.nan])
def test_class_labels_reject_unassigned_cells(
    coordinates: tuple[xr.DataArray, xr.DataArray],
    invalid_value: object,
) -> None:
    """Every native cell must have a non-null class assignment."""
    latitude, longitude = coordinates
    values = np.full((3, 2), "land", dtype=object)
    values[0, 0] = invalid_value
    labels = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )

    with pytest.raises(ValueError, match="non-null class"):
        SeparableExponentialCovariance(latitude, longitude, class_labels=labels)


def test_class_labels_require_exact_grid_alignment(
    coordinates: tuple[xr.DataArray, xr.DataArray],
) -> None:
    """Class labels cannot be silently reordered against the covariance grid."""
    latitude, longitude = coordinates
    labels = xr.DataArray(
        np.zeros((3, 2), dtype=int),
        dims=("lat", "lon"),
        coords={"lat": latitude[::-1], "lon": longitude},
    )

    with pytest.raises(ValueError, match="does not align"):
        SeparableExponentialCovariance(latitude, longitude, class_labels=labels)
