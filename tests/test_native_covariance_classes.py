"""Tests for optional class blocking in native covariance actions."""

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import openghg_inversions.native_covariance as native_covariance_module
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
    same_class = np.fromiter(
        (left == right for left in labels for right in labels),
        dtype=bool,
        count=labels.size**2,
    ).reshape(labels.size, labels.size)
    return unblocked * same_class


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


def test_class_blocked_solve_supports_legacy_cg_tolerance_keyword(
    coordinates: tuple[xr.DataArray, xr.DataArray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compatibility dispatch uses ``tol`` with pre-1.12 SciPy CG signatures."""
    latitude, longitude = coordinates
    labels = xr.DataArray(
        [["land", "sea"], ["land", "sea"], ["sea", "sea"]],
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )
    covariance = SeparableExponentialCovariance(latitude, longitude, class_labels=labels)
    rhs = xr.ones_like(labels, dtype=float)
    current_cg = native_covariance_module.cg

    def legacy_cg(operator, vector, *, tol, atol, maxiter):
        """Expose the legacy keyword while delegating to the installed SciPy."""
        return current_cg(operator, vector, rtol=tol, atol=atol, maxiter=maxiter)

    monkeypatch.setattr(native_covariance_module, "cg", legacy_cg)
    monkeypatch.setattr(native_covariance_module, "_CG_RELATIVE_TOLERANCE", "tol")

    solved = covariance.solve(rhs)

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


def test_tuple_class_labels_match_dense_oracle_and_round_trip_through_netcdf(
    coordinates: tuple[xr.DataArray, xr.DataArray],
    tmp_path: Path,
) -> None:
    """Composite scalar labels define correct masks and use NetCDF-safe tagged encoding."""
    latitude, longitude = coordinates
    values = np.empty((3, 2), dtype=object)
    tuple_labels = [
        ("biosphere", 1),
        ("ocean", 2),
        ("biosphere", 1),
        ("ocean", 2),
        ("ocean", 2),
        ("ocean", 2),
    ]
    for index, label in enumerate(tuple_labels):
        values.flat[index] = label
    labels = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
        name="surface_class",
        attrs={"version": np.int64(2)},
    )
    covariance = SeparableExponentialCovariance(
        latitude,
        longitude,
        sigma=1.4,
        correlation_length=1.1,
        class_labels=labels,
    )
    rhs = xr.DataArray(
        np.arange(6, dtype=float).reshape(3, 2),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )
    dense = _dense_blocked_covariance(latitude, longitude, labels, 1.4, 1.1)
    path = tmp_path / "tuple-label-covariance.nc"

    np.testing.assert_allclose(covariance.apply(rhs).values.reshape(-1), dense @ rhs.values.reshape(-1))
    np.testing.assert_allclose(
        covariance.solve(rhs).values.reshape(-1),
        np.linalg.solve(dense, rhs.values.reshape(-1)),
    )
    covariance.to_dataset().to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as stored:
        restored = SeparableExponentialCovariance.from_dataset(stored.load())

    restored_labels = restored.class_labels
    assert restored_labels is not None
    xr.testing.assert_identical(restored_labels, labels.assign_attrs(version=2))


@pytest.mark.parametrize("class_blocked", [False, True])
def test_covariance_round_trips_through_h5netcdf(
    coordinates: tuple[xr.DataArray, xr.DataArray],
    class_blocked: bool,
    tmp_path: Path,
) -> None:
    """NetCDF persistence preserves blocked and unblocked covariance configuration."""
    latitude, longitude = coordinates
    class_labels = None
    if class_blocked:
        class_labels = xr.DataArray(
            [[0, 1], [0, 1], [1, 1]],
            dims=("lat", "lon"),
            coords={"lat": latitude, "lon": longitude},
            name="surface_class",
        )
    original = SeparableExponentialCovariance(
        latitude,
        longitude,
        sigma=1.3,
        class_labels=class_labels,
    )
    path = tmp_path / "covariance.nc"

    original.to_dataset().to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as stored:
        loaded = stored.load()
    restored = SeparableExponentialCovariance.from_dataset(loaded)

    assert loaded.attrs["class_blocked"] == int(class_blocked)
    assert (restored.class_labels is not None) is class_blocked
    if class_labels is not None:
        xr.testing.assert_identical(restored.class_labels, class_labels)
    replaced = SeparableExponentialCovariance(
        restored.latitude,
        restored.longitude,
        sigma=restored.sigma,
        correlation_length=3.0,
        class_labels=restored.class_labels,
    )
    assert replaced.latitude_correlation_length == 3.0
    assert replaced.longitude_correlation_length == 3.0


def test_single_class_uses_unblocked_covariance(
    coordinates: tuple[xr.DataArray, xr.DataArray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single class uses the allocation-free separable path."""
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

    def reject_blocked_result_allocation(*args: object, **kwargs: object) -> None:
        """Fail if apply enters the multi-class masked accumulation path."""
        raise AssertionError("single-class covariance allocated a blocked result")

    monkeypatch.setattr(native_covariance_module.np, "zeros_like", reject_blocked_result_allocation)
    xr.testing.assert_allclose(blocked.apply(rhs), unblocked.apply(rhs))
    xr.testing.assert_allclose(blocked.solve(rhs), unblocked.solve(rhs))


def test_constructor_copies_class_labels_and_property_borrows_owned_labels(
    coordinates: tuple[xr.DataArray, xr.DataArray],
) -> None:
    """Construction isolates labels while ordinary access returns the owned array."""
    latitude, longitude = coordinates
    labels = xr.DataArray(
        [["land", "ocean"], ["land", "ocean"], ["land", "ocean"]],
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )
    covariance = SeparableExponentialCovariance(latitude, longitude, class_labels=labels)
    exposed = covariance.class_labels

    assert exposed is not None
    labels.values[1, 0] = "ocean"
    assert covariance.class_labels is exposed
    np.testing.assert_array_equal(
        exposed.values,
        [["land", "ocean"], ["land", "ocean"], ["land", "ocean"]],
    )

    unblocked = SeparableExponentialCovariance(
        covariance.latitude,
        covariance.longitude,
        sigma=covariance.sigma,
        correlation_length=covariance.correlation_length,
        class_labels=None,
    )
    assert unblocked.class_labels is None


def test_constructor_materializes_lazy_class_labels_once(
    coordinates: tuple[xr.DataArray, xr.DataArray],
) -> None:
    """Lazy labels become one owned eager array used by cached masks."""
    latitude, longitude = coordinates
    labels = xr.DataArray(
        da.from_array(
            np.asarray([["land", "ocean"], ["land", "ocean"], ["land", "ocean"]]),
            chunks=(2, 1),
        ),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )

    covariance = SeparableExponentialCovariance(latitude, longitude, class_labels=labels)

    assert covariance.class_labels is not None
    assert isinstance(covariance.class_labels.data, np.ndarray)
    assert covariance.class_labels is covariance.class_labels


def test_native_dimension_cannot_collide_with_serialized_class_labels() -> None:
    """The component schema rejects its reserved class-label variable name."""
    latitude = xr.DataArray([0.0, 1.0], dims="class_label_encoded")
    longitude = xr.DataArray([10.0, 11.0], dims="lon")

    with pytest.raises(ValueError, match="reserved by the serialized schema"):
        SeparableExponentialCovariance(latitude, longitude)


def test_deserialization_rejects_contradictory_implicit_length(
    coordinates: tuple[xr.DataArray, xr.DataArray],
) -> None:
    """A fallback-derived axis length must match the serialized fallback."""
    latitude, longitude = coordinates
    dataset = SeparableExponentialCovariance(latitude, longitude).to_dataset()
    dataset.attrs["latitude_correlation_length_degrees"] = 99.0

    with pytest.raises(ValueError, match="implicit latitude.*contradicts"):
        SeparableExponentialCovariance.from_dataset(dataset)


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


@pytest.mark.parametrize(
    ("class_blocked", "include_labels"),
    [(1, False), (0, True)],
)
def test_deserialization_rejects_contradictory_blocked_state(
    coordinates: tuple[xr.DataArray, xr.DataArray],
    class_blocked: int,
    include_labels: bool,
) -> None:
    """The serialized blocked flag must agree with the presence of encoded labels."""
    latitude, longitude = coordinates
    labels = xr.DataArray(
        np.zeros((3, 2), dtype=int),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )
    covariance = SeparableExponentialCovariance(
        latitude,
        longitude,
        class_labels=labels if include_labels else None,
    )
    dataset = covariance.to_dataset()
    dataset.attrs["class_blocked"] = class_blocked

    with pytest.raises(ValueError, match="contradicts"):
        SeparableExponentialCovariance.from_dataset(dataset)
