"""Tests for labelled, matrix-free native-grid covariance actions."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.native_covariance import SeparableExponentialCovariance


def _coordinates() -> tuple[xr.DataArray, xr.DataArray]:
    """Return a small, irregular labelled latitude-longitude grid."""
    latitude = xr.DataArray(
        [-1.0, 0.25, 2.0],
        dims="lat",
        coords={"lat": [-1.0, 0.25, 2.0]},
        attrs={"units": "degrees_north"},
    )
    longitude = xr.DataArray(
        [10.0, 11.5],
        dims="lon",
        coords={"lon": [10.0, 11.5]},
        attrs={"units": "degrees_east"},
    )
    return latitude, longitude


def _dense_covariance(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    *,
    sigma: float,
    correlation_length: float,
) -> np.ndarray:
    """Construct the small-grid Kronecker covariance used only as a test oracle."""
    k_lat = np.exp(
        -np.abs(latitude.values[:, np.newaxis] - latitude.values[np.newaxis, :]) / correlation_length
    )
    k_lon = np.exp(
        -np.abs(longitude.values[:, np.newaxis] - longitude.values[np.newaxis, :]) / correlation_length
    )
    return sigma**2 * np.kron(k_lat, k_lon)


def _covariance(*, sigma: float = 1.7, correlation_length: float = 0.8):
    """Construct the covariance action shared by focused tests."""
    latitude, longitude = _coordinates()
    return SeparableExponentialCovariance(
        latitude=latitude,
        longitude=longitude,
        sigma=sigma,
        correlation_length=correlation_length,
    )


def test_apply_and_solve_match_dense_oracle_for_multiple_rhs() -> None:
    """Apply and solve preserve labels while matching a dense multiple-RHS oracle."""
    latitude, longitude = _coordinates()
    covariance = _covariance()
    rhs = xr.DataArray(
        np.arange(24, dtype=float).reshape(2, 2, 3, 2) / 7.0,
        dims=("draw", "lon", "lat", "component"),
        coords={
            "draw": ["a", "b"],
            "lon": longitude,
            "lat": latitude,
            "component": ["east", "west"],
        },
        name="right_hand_side",
        attrs={"units": "ppm"},
    )
    dense = _dense_covariance(latitude, longitude, sigma=1.7, correlation_length=0.8)
    rhs_matrix = rhs.transpose("lat", "lon", "draw", "component").values.reshape(6, 4)

    applied = covariance.apply(rhs)
    solved = covariance.solve(rhs)

    expected_applied = (dense @ rhs_matrix).reshape(3, 2, 2, 2)
    expected_solved = np.linalg.solve(dense, rhs_matrix).reshape(3, 2, 2, 2)
    assert applied.dims == rhs.dims
    assert solved.dims == rhs.dims
    assert applied.name == rhs.name
    assert solved.name == rhs.name
    assert applied.attrs == rhs.attrs
    assert solved.attrs == rhs.attrs
    xr.testing.assert_equal(applied.coords.to_dataset(), rhs.coords.to_dataset())
    xr.testing.assert_equal(solved.coords.to_dataset(), rhs.coords.to_dataset())
    np.testing.assert_allclose(
        applied.transpose("lat", "lon", "draw", "component"),
        expected_applied,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        solved.transpose("lat", "lon", "draw", "component"),
        expected_solved,
        rtol=1e-11,
        atol=1e-11,
    )


def test_single_rhs_round_trips_through_apply_and_solve() -> None:
    """A labelled grid field round-trips without requiring a trailing RHS dimension."""
    latitude, longitude = _coordinates()
    covariance = _covariance()
    rhs = xr.DataArray(
        np.arange(6, dtype=float).reshape(3, 2),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )

    restored = covariance.solve(covariance.apply(rhs))

    xr.testing.assert_allclose(restored, rhs, rtol=1e-11, atol=1e-11)


def test_anisotropic_correlation_lengths_match_dense_oracle() -> None:
    """Latitude and longitude length scales are independently configurable."""
    latitude, longitude = _coordinates()
    covariance = SeparableExponentialCovariance(
        latitude,
        longitude,
        sigma=1.3,
        latitude_correlation_length=0.6,
        longitude_correlation_length=2.1,
    )
    rhs = xr.DataArray(
        np.arange(6, dtype=float).reshape(3, 2),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )
    k_lat = np.exp(-np.abs(latitude.values[:, None] - latitude.values[None, :]) / 0.6)
    k_lon = np.exp(-np.abs(longitude.values[:, None] - longitude.values[None, :]) / 2.1)
    dense = 1.3**2 * np.kron(k_lat, k_lon)

    np.testing.assert_allclose(covariance.apply(rhs).values.reshape(-1), dense @ rhs.values.reshape(-1))
    np.testing.assert_allclose(
        covariance.solve(rhs).values.reshape(-1), np.linalg.solve(dense, rhs.values.reshape(-1))
    )


def test_default_configuration_round_trips_through_dataset() -> None:
    """Serialization records the resolved default and reconstructs an equivalent action."""
    latitude, longitude = _coordinates()
    covariance = SeparableExponentialCovariance(
        latitude=latitude,
        longitude=longitude,
        sigma=2.5,
    )
    rhs = xr.DataArray(
        np.arange(6, dtype=float).reshape(3, 2),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )

    restored = SeparableExponentialCovariance.from_dataset(covariance.to_dataset())

    assert covariance.correlation_length == 1.5
    assert restored.correlation_length == 1.5
    assert restored.sigma == 2.5
    assert restored.native_dims == ("lat", "lon")
    np.testing.assert_array_equal(restored.latitude, covariance.latitude)
    np.testing.assert_array_equal(restored.longitude, covariance.longitude)
    assert restored.latitude.dims == covariance.latitude.dims
    assert restored.longitude.dims == covariance.longitude.dims
    assert restored.latitude.attrs == covariance.latitude.attrs
    assert restored.longitude.attrs == covariance.longitude.attrs
    xr.testing.assert_allclose(restored.apply(rhs), covariance.apply(rhs))


@pytest.mark.parametrize(
    ("bad_rhs", "message"),
    [
        pytest.param(
            lambda rhs: rhs.assign_coords(lon=[11.5, 10.0]),
            "coordinate|align|longitude|lon",
            id="reordered-coordinate-labels",
        ),
        pytest.param(
            lambda rhs: rhs.drop_indexes("lon").drop_vars("lon"),
            "lon",
            id="missing-coordinate",
        ),
    ],
)
def test_action_requires_labelled_aligned_rhs(bad_rhs, message: str) -> None:
    """The public action requires its configured labelled native grid."""
    latitude, longitude = _coordinates()
    covariance = _covariance()
    rhs = xr.DataArray(
        np.arange(6, dtype=float).reshape(3, 2),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )

    with pytest.raises(ValueError, match=message):
        covariance.apply(bad_rhs(rhs))


@pytest.mark.parametrize(
    ("coordinate", "message"),
    [
        pytest.param([0.0, 0.0, 1.0], "unique|duplicate", id="duplicate"),
        pytest.param([0.0, np.nan, 1.0], "finite", id="non-finite"),
    ],
)
def test_constructor_rejects_invalid_native_coordinates(coordinate, message: str) -> None:
    """Native coordinates must be finite and unique before factors are constructed."""
    latitude, longitude = _coordinates()
    latitude = latitude.assign_coords(lat=coordinate).copy(data=coordinate)

    with pytest.raises(ValueError, match=message):
        SeparableExponentialCovariance(latitude=latitude, longitude=longitude, sigma=1.0)


def test_constructor_rejects_coordinate_data_that_disagree_with_dimension_labels() -> None:
    """A coordinate cannot advertise labels that differ from its numeric grid data."""
    latitude, longitude = _coordinates()
    latitude = latitude.assign_coords(lat=[40.0, 50.0, 60.0])

    with pytest.raises(ValueError, match="data.*dimension labels"):
        SeparableExponentialCovariance(latitude=latitude, longitude=longitude)


def test_constructor_rejects_complex_degree_coordinates() -> None:
    """Complex coordinates are not silently truncated when constructing degree distances."""
    latitude, longitude = _coordinates()
    values = np.asarray(latitude.values, dtype=np.complex128) + 1j
    latitude = latitude.copy(data=values).assign_coords(lat=values)

    with pytest.raises(ValueError, match="real numeric"):
        SeparableExponentialCovariance(latitude=latitude, longitude=longitude)


@pytest.mark.parametrize("axis_name", ["latitude", "longitude"])
def test_constructor_rejects_empty_native_axes(axis_name: str) -> None:
    """Each native coordinate must contain at least one grid point."""
    latitude, longitude = _coordinates()
    if axis_name == "latitude":
        latitude = latitude.isel(lat=slice(0, 0))
    else:
        longitude = longitude.isel(lon=slice(0, 0))

    with pytest.raises(ValueError, match=rf"{axis_name}.*at least one"):
        SeparableExponentialCovariance(latitude=latitude, longitude=longitude)


@pytest.mark.parametrize(
    ("sigma", "correlation_length", "message"),
    [
        pytest.param(0.0, 1.5, "sigma|positive", id="zero-sigma"),
        pytest.param(-1.0, 1.5, "sigma|positive", id="negative-sigma"),
        pytest.param(1.0, 0.0, "correlation_length|positive", id="zero-length"),
        pytest.param(1.0, np.inf, "correlation_length|finite", id="infinite-length"),
    ],
)
def test_constructor_rejects_invalid_covariance_parameters(
    sigma: float,
    correlation_length: float,
    message: str,
) -> None:
    """Covariance scale and correlation length must be finite and strictly positive."""
    latitude, longitude = _coordinates()

    with pytest.raises(ValueError, match=message):
        SeparableExponentialCovariance(
            latitude=latitude,
            longitude=longitude,
            sigma=sigma,
            correlation_length=correlation_length,
        )


def test_actions_do_not_construct_a_native_kronecker_matrix(monkeypatch) -> None:
    """Production apply and solve use separable factors without calling ``numpy.kron``."""
    latitude, longitude = _coordinates()
    rhs = xr.DataArray(
        np.ones((3, 2, 2)),
        dims=("lat", "lon", "rhs"),
        coords={"lat": latitude, "lon": longitude, "rhs": [0, 1]},
    )

    def fail_kron(*args, **kwargs):
        """Fail if the matrix-free implementation constructs a Kronecker matrix."""
        raise AssertionError("the structured action must not call numpy.kron")

    monkeypatch.setattr(np, "kron", fail_kron)

    covariance = _covariance()
    applied = covariance.apply(rhs)
    solved = covariance.solve(rhs)

    assert applied.shape == rhs.shape
    assert solved.shape == rhs.shape
    assert np.isfinite(applied).all()
    assert np.isfinite(solved).all()


def test_constructor_owns_coordinates_and_properties_are_borrowed() -> None:
    """Construction owns coordinate copies, properties borrow them, and identity is retained."""
    latitude, longitude = _coordinates()
    covariance = SeparableExponentialCovariance(latitude, longitude)
    equivalent = SeparableExponentialCovariance(latitude, longitude)

    latitude.values[1] = 777.0

    np.testing.assert_array_equal(covariance.latitude.values, [-1.0, 0.25, 2.0])
    assert covariance.latitude is covariance.latitude
    assert covariance.longitude is covariance.longitude
    assert covariance.latitude is not latitude
    assert covariance.longitude is not longitude
    assert covariance != equivalent
    assert isinstance(hash(covariance), int)


def test_constructor_materializes_lazy_coordinates_once() -> None:
    """Lazy coordinate vectors become owned eager arrays at construction."""
    latitude, longitude = _coordinates()
    lazy_latitude = latitude.copy(data=da.from_array(latitude.to_numpy(), chunks=2))
    lazy_longitude = longitude.copy(data=da.from_array(longitude.to_numpy(), chunks=1))

    covariance = SeparableExponentialCovariance(lazy_latitude, lazy_longitude)

    assert isinstance(covariance.latitude.data, np.ndarray)
    assert isinstance(covariance.longitude.data, np.ndarray)
    assert covariance.latitude is covariance.latitude
    assert covariance.longitude is covariance.longitude


def test_configuration_is_read_only_and_reconstruction_rebuilds_cached_factors() -> None:
    """Reconstruction changes parameters while the original slotted configuration stays read-only."""
    latitude, longitude = _coordinates()
    covariance = SeparableExponentialCovariance(latitude, longitude, sigma=1.0)
    scaled = SeparableExponentialCovariance(covariance.latitude, covariance.longitude, sigma=2.0)
    longer = SeparableExponentialCovariance(
        covariance.latitude,
        covariance.longitude,
        correlation_length=3.0,
    )
    rhs = xr.DataArray(
        np.ones((3, 2)),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )

    with pytest.raises(AttributeError):
        covariance.correlation_length = 3.0  # type: ignore[misc]

    assert not hasattr(covariance, "__dict__")
    xr.testing.assert_allclose(scaled.apply(rhs), 4.0 * covariance.apply(rhs))
    assert longer.latitude_correlation_length == 3.0
    assert longer.longitude_correlation_length == 3.0
    assert not np.allclose(longer.apply(rhs), covariance.apply(rhs))


def test_zero_length_rhs_axis_is_preserved() -> None:
    """An empty non-native RHS axis remains a valid labelled batch."""
    latitude, longitude = _coordinates()
    covariance = _covariance()
    rhs = xr.DataArray(
        np.empty((3, 2, 0)),
        dims=("lat", "lon", "rhs"),
        coords={"lat": latitude, "lon": longitude, "rhs": []},
    )

    applied = covariance.apply(rhs)

    assert applied.shape == rhs.shape
