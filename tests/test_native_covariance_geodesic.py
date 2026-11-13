"""Tests for class-mixed native covariance with dense geodesic class blocks."""

import numpy as np
import pytest
import xarray as xr

import openghg_inversions.native_covariance as native_covariance_module
from openghg_inversions.native_covariance import (
    DEFAULT_EARTH_RADIUS_KM,
    ClassMixedExponentialCovariance,
    GeodesicExponentialKernel,
    SeparableExponentialCovariance,
)
from openghg_inversions.source_covariance import IndependentSourceCovariance


LATITUDES = [36.0, 41.5, 47.0, 52.5, 58.0, 63.5, 69.0]
LONGITUDES = [-12.0, -6.0, 0.0, 6.0, 12.0, 18.0]


def _grid() -> tuple[xr.DataArray, xr.DataArray]:
    """Return a small European-latitude grid with degree metadata."""
    latitude = xr.DataArray(
        LATITUDES, dims="lat", coords={"lat": LATITUDES}, attrs={"units": "degrees_north"}
    )
    longitude = xr.DataArray(
        LONGITUDES, dims="lon", coords={"lon": LONGITUDES}, attrs={"units": "degrees_east"}
    )
    return latitude, longitude


def _labels(latitude: xr.DataArray, longitude: xr.DataArray) -> xr.DataArray:
    """Return sea (2), inner land (1) and two outer classes, one non-rectangular."""
    values = np.full((latitude.size, longitude.size), 2)
    values[1:6, 1:5] = 1
    values[2, 1] = 2
    values[0, :2] = 401
    values[6, 4:] = 402
    values[5, 5] = 402
    return xr.DataArray(
        values, dims=("lat", "lon"), coords={"lat": latitude, "lon": longitude}, name="basis_class"
    )


def _great_circle_correlation(latitude: np.ndarray, longitude: np.ndarray, length_km: float) -> np.ndarray:
    """Independent dense geodesic oracle using unit vectors and atan2."""
    phi = np.deg2rad(latitude)
    lam = np.deg2rad(longitude)
    unit = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)
    cross = np.linalg.norm(np.cross(unit[:, None, :], unit[None, :, :]), axis=-1)
    dot = unit @ unit.T
    distance = DEFAULT_EARTH_RADIUS_KM * np.arctan2(cross, dot)
    return np.exp(-distance / length_km)


def _dense_oracle(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    labels: xr.DataArray,
    *,
    sigma: float,
    latitude_length: float,
    longitude_length: float,
    geodesic: dict[object, float],
) -> np.ndarray:
    """Dense ``B``: class-blocked separable with geodesic class blocks replaced."""
    lat = np.asarray(latitude.values, dtype=float)
    lon = np.asarray(longitude.values, dtype=float)
    k_lat = np.exp(-np.abs(lat[:, None] - lat[None, :]) / latitude_length)
    k_lon = np.exp(-np.abs(lon[:, None] - lon[None, :]) / longitude_length)
    flat_labels = np.asarray(labels.transpose("lat", "lon").values).reshape(-1)
    dense = np.kron(k_lat, k_lon) * (flat_labels[:, None] == flat_labels[None, :])
    cell_lat = np.repeat(lat, lon.size)
    cell_lon = np.tile(lon, lat.size)
    for label, length in geodesic.items():
        index = np.flatnonzero(flat_labels == label)
        dense[np.ix_(index, index)] = _great_circle_correlation(cell_lat[index], cell_lon[index], length)
    return sigma**2 * dense


def _covariance(**overrides) -> ClassMixedExponentialCovariance:
    """Return the standard mixed test covariance."""
    latitude, longitude = _grid()
    kwargs = dict(
        class_labels=_labels(latitude, longitude),
        class_kernels={1: GeodesicExponentialKernel(500.0)},
        sigma=0.8,
        correlation_length=4.0,
        latitude_correlation_length=500.0 / 111.0,
        longitude_correlation_length=500.0 / (111.0 * np.cos(np.deg2rad(50.0))),
    )
    kwargs.update(overrides)
    return ClassMixedExponentialCovariance(latitude, longitude, **kwargs)


def _oracle(covariance: ClassMixedExponentialCovariance) -> np.ndarray:
    """Dense oracle for a mixed covariance's own configuration."""
    return _dense_oracle(
        covariance.latitude,
        covariance.longitude,
        covariance.class_labels,
        sigma=covariance.sigma,
        latitude_length=covariance.latitude_correlation_length,
        longitude_length=covariance.longitude_correlation_length,
        geodesic={label: kernel.correlation_length_km for label, kernel in covariance.class_kernels.items()},
    )


def test_apply_solve_and_variance_match_dense_oracle() -> None:
    """Apply, solve and functional variances match the dense mixed covariance."""
    covariance = _covariance()
    latitude, longitude = _grid()
    rng = np.random.default_rng(188)
    rhs = xr.DataArray(
        rng.normal(size=(2, longitude.size, 3, latitude.size)),
        dims=("draw", "lon", "member", "lat"),
        coords={"draw": ["a", "b"], "lat": latitude, "lon": longitude, "member": [3, 1, 2]},
        name="rhs",
        attrs={"units": "1"},
    )
    dense = _oracle(covariance)
    matrix = rhs.transpose("lat", "lon", "draw", "member").values.reshape(dense.shape[0], -1)

    applied = covariance.apply(rhs)
    solved = covariance.solve(rhs)

    assert applied.dims == rhs.dims and applied.name == "rhs" and applied.attrs == {"units": "1"}
    xr.testing.assert_identical(applied.coords.to_dataset(), rhs.coords.to_dataset())
    np.testing.assert_allclose(
        applied.transpose("lat", "lon", "draw", "member").values.reshape(dense.shape[0], -1),
        dense @ matrix,
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        solved.transpose("lat", "lon", "draw", "member").values.reshape(dense.shape[0], -1),
        np.linalg.solve(dense, matrix),
        rtol=1e-8,
        atol=1e-9,
    )
    weights = xr.DataArray(
        rng.uniform(0.0, 2.0, size=(4, latitude.size, longitude.size)),
        dims=("region", "lat", "lon"),
        coords={"region": list("wxyz"), "lat": latitude, "lon": longitude},
    )
    variance = (weights * covariance.apply(weights)).sum(("lat", "lon"))
    flat_weights = weights.values.reshape(4, -1)
    np.testing.assert_allclose(variance.values, np.einsum("ri,ij,rj->r", flat_weights, dense, flat_weights))
    np.testing.assert_allclose(np.diag(dense), covariance.sigma**2)


def test_separable_classes_match_class_blocked_separable_covariance() -> None:
    """Outside dense classes the action equals the class-blocked separable one."""
    covariance = _covariance()
    separable = SeparableExponentialCovariance(
        covariance.latitude,
        covariance.longitude,
        sigma=covariance.sigma,
        correlation_length=covariance.correlation_length,
        latitude_correlation_length=covariance.latitude_correlation_length,
        longitude_correlation_length=covariance.longitude_correlation_length,
        class_labels=covariance.class_labels,
    )
    rng = np.random.default_rng(5)
    field = xr.DataArray(
        rng.normal(size=(len(LATITUDES), len(LONGITUDES))),
        dims=("lat", "lon"),
        coords={"lat": covariance.latitude, "lon": covariance.longitude},
    ).where(covariance.class_labels != 1, 0.0)

    np.testing.assert_allclose(covariance.apply(field), separable.apply(field), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(covariance.solve(field), separable.solve(field), rtol=1e-8, atol=1e-10)


def test_geodesic_only_covariance_solves_by_cholesky() -> None:
    """A grid with one dense class is applied and solved without iterations."""
    latitude, longitude = _grid()
    labels = xr.zeros_like(_labels(latitude, longitude))
    covariance = ClassMixedExponentialCovariance(
        latitude,
        longitude,
        class_labels=labels,
        class_kernels={0: GeodesicExponentialKernel(200.0)},
        sigma=1.3,
    )
    dense = _oracle(covariance)
    rhs = xr.DataArray(
        np.arange(labels.size, dtype=float).reshape(labels.shape) / 10.0,
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )

    np.testing.assert_allclose(covariance.apply(rhs).values.reshape(-1), dense @ rhs.values.reshape(-1))
    np.testing.assert_allclose(
        covariance.solve(rhs).values.reshape(-1), np.linalg.solve(dense, rhs.values.reshape(-1)), rtol=1e-11
    )


def test_geodesic_kernel_keeps_111_km_per_degree_and_widens_east_west() -> None:
    """N-S correlations equal the separable 111 km/degree convention; E-W follow cos(latitude)."""
    kernel = GeodesicExponentialKernel(500.0)

    north_south = kernel.correlation(np.array(50.0), np.array(4.0), np.array(53.0), np.array(4.0))
    east_west = [
        kernel.correlation(np.array(lat), np.array(0.0), np.array(lat), np.array(1.0)) for lat in (40.0, 65.0)
    ]

    np.testing.assert_allclose(north_south, np.exp(-3.0 * 111.0 / 500.0), rtol=1e-14)
    for value, lat in zip(east_west, (40.0, 65.0), strict=True):
        np.testing.assert_allclose(value, np.exp(-111.0 * np.cos(np.deg2rad(lat)) / 500.0), rtol=1e-4)


@pytest.mark.parametrize("storage", ["memory", "netcdf"])
def test_round_trip_preserves_configuration_and_action(tmp_path, storage) -> None:
    """Serialization restores kernels, typed labels and an identical action."""
    latitude, longitude = _grid()
    labels = _labels(latitude, longitude).astype(str).where(lambda value: value != "1", "inner")
    labels.attrs = {"description": "test classes"}
    covariance = _covariance(
        class_labels=labels,
        class_kernels={"inner": GeodesicExponentialKernel(350.0, earth_radius_km=6371.0)},
    )
    dataset = covariance.to_dataset()
    if storage == "netcdf":
        path = tmp_path / "covariance.nc"
        dataset.to_netcdf(path)
        with xr.open_dataset(path) as opened:
            dataset = opened.load()

    restored = ClassMixedExponentialCovariance.from_dataset(dataset)
    rhs = xr.DataArray(
        np.random.default_rng(2).normal(size=(len(LATITUDES), len(LONGITUDES))),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )

    assert dict(restored.class_kernels) == {"inner": GeodesicExponentialKernel(350.0, earth_radius_km=6371.0)}
    assert restored.sigma == covariance.sigma
    assert restored.latitude_correlation_length == covariance.latitude_correlation_length
    xr.testing.assert_identical(restored.class_labels, covariance.class_labels)
    np.testing.assert_array_equal(restored.apply(rhs), covariance.apply(rhs))
    xr.testing.assert_identical(restored.to_dataset(), covariance.to_dataset())
    with pytest.raises(ValueError, match="schema"):
        SeparableExponentialCovariance.from_dataset(dataset)
    with pytest.raises(ValueError, match="missing variables"):
        ClassMixedExponentialCovariance.from_dataset(dataset.drop_vars("geodesic_class_kernel"))
    with pytest.raises(ValueError, match="Unsupported serialized class kernel"):
        ClassMixedExponentialCovariance.from_dataset(
            dataset.assign(geodesic_class_kernel=dataset.geodesic_class_kernel.copy(data=["anisotropic"]))
        )


def test_dataset_identifies_the_unit_kernel_and_with_sigma_shares_factors() -> None:
    """Datasets differ only in sigma for rescaled kernels, which share dense storage."""
    covariance = _covariance()
    rescaled = covariance.with_sigma(2.5)
    rhs = covariance.class_labels.astype(float)

    first, second = covariance.to_dataset(), rescaled.to_dataset()
    assert first.attrs.pop("sigma") == 0.8 and second.attrs.pop("sigma") == 2.5
    xr.testing.assert_identical(first, second)
    assert rescaled._blocks[0].storage is covariance._blocks[0].storage
    np.testing.assert_allclose(rescaled.apply(rhs), covariance.apply(rhs) * (2.5 / 0.8) ** 2, rtol=1e-14)
    np.testing.assert_allclose(rescaled.solve(rhs), covariance.solve(rhs) * (0.8 / 2.5) ** 2, rtol=1e-10)
    with pytest.raises(ValueError, match="sigma"):
        covariance.with_sigma(0.0)


@pytest.mark.parametrize("dense_eigen_limit", [64, 0])
def test_positive_definiteness_report_matches_dense_eigenvalues(monkeypatch, dense_eigen_limit) -> None:
    """Condition estimates and minimum eigenvalues describe each dense block."""
    monkeypatch.setattr(native_covariance_module, "_GEODESIC_DENSE_EIGEN_LIMIT", dense_eigen_limit)
    covariance = _covariance()
    flat_labels = covariance.class_labels.values.reshape(-1)
    index = np.flatnonzero(flat_labels == 1)
    block = _oracle(covariance)[np.ix_(index, index)] / covariance.sigma**2

    (report,) = covariance.geodesic_diagnostics
    minimum = covariance.minimum_eigenvalues()

    assert report.class_label == 1 and report.size == index.size
    assert report.nbytes == 8 * index.size * (index.size + 1)
    np.testing.assert_allclose(report.one_norm, np.abs(block).sum(axis=0).max())
    assert 0.1 < report.reciprocal_condition * np.linalg.cond(block, 1) < 10.0
    np.testing.assert_allclose(minimum[1], np.linalg.eigvalsh(block)[0], rtol=1e-8)


@pytest.mark.parametrize(
    ("length_km", "message"),
    [(1.0e20, "not positive definite: Cholesky failed at leading minor 2"), (1.0e15, "numerically singular")],
)
def test_numerically_singular_block_is_rejected(length_km, message) -> None:
    """Constant kernels fail Cholesky; nearly constant ones fail the condition check."""
    with pytest.raises(np.linalg.LinAlgError, match=f"class 1 .*{message}"):
        _covariance(class_kernels={1: GeodesicExponentialKernel(length_km)})


def test_invalid_class_kernel_configuration_is_rejected() -> None:
    """Class kernels must be non-empty, typed and name present classes once."""
    latitude, longitude = _grid()
    with pytest.raises(ValueError, match="at least one class"):
        _covariance(class_kernels={})
    with pytest.raises(ValueError, match="absent"):
        _covariance(class_kernels={7: GeodesicExponentialKernel(500.0)})
    with pytest.raises(TypeError, match="GeodesicExponentialKernel"):
        _covariance(class_kernels={1: 500.0})
    with pytest.raises(ValueError, match="requires class_labels"):
        ClassMixedExponentialCovariance(
            latitude, longitude, class_labels=None, class_kernels={1: GeodesicExponentialKernel(500.0)}
        )
    with pytest.raises(ValueError, match="correlation_length_km"):
        GeodesicExponentialKernel(-1.0)
    with pytest.raises(ValueError, match="earth_radius_km"):
        GeodesicExponentialKernel(100.0, earth_radius_km=np.inf)


def _sources() -> IndependentSourceCovariance:
    """Per-source configuration: dense GPP/FF inner land with their own lengths, separable ocean."""
    latitude, longitude = _grid()
    labels = _labels(latitude, longitude)
    lat_length = 3.0
    return IndependentSourceCovariance(
        {
            "GPP": _covariance(sigma=0.5, class_kernels={1: GeodesicExponentialKernel(500.0)}),
            "FF": _covariance(
                sigma=0.7,
                class_kernels={1: GeodesicExponentialKernel(200.0), 402: GeodesicExponentialKernel(300.0)},
            ),
            "ocean": SeparableExponentialCovariance(
                latitude, longitude, sigma=0.2, correlation_length=lat_length, class_labels=labels
            ),
        }
    )


def test_independent_sources_accept_per_source_class_kernels() -> None:
    """Mixed and separable source blocks apply and solve as a dense block diagonal."""
    covariance = _sources()
    latitude, longitude = _grid()
    rhs = xr.DataArray(
        np.random.default_rng(9).normal(size=(3, len(LATITUDES), len(LONGITUDES), 2)),
        dims=("native_source", "lat", "lon", "obs"),
        coords={"native_source": ["GPP", "FF", "ocean"], "lat": latitude, "lon": longitude, "obs": [0, 1]},
    )
    applied = covariance.apply(rhs)
    solved = covariance.solve(rhs)
    for source in covariance.source_labels:
        component = covariance.source_covariances[source]
        if isinstance(component, ClassMixedExponentialCovariance):
            dense = _oracle(component)
        else:
            dense = _dense_oracle(
                latitude,
                longitude,
                _labels(latitude, longitude),
                sigma=component.sigma,
                latitude_length=component.latitude_correlation_length,
                longitude_length=component.longitude_correlation_length,
                geodesic={},
            )
        matrix = rhs.sel(native_source=source).values.reshape(-1, 2)
        np.testing.assert_allclose(
            applied.sel(native_source=source).values.reshape(-1, 2), dense @ matrix, rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            solved.sel(native_source=source).values.reshape(-1, 2),
            np.linalg.solve(dense, matrix),
            rtol=1e-8,
            atol=1e-8,
        )


def test_independent_source_serialization_records_class_kernels(tmp_path) -> None:
    """Class-mixed sources write schema version 2; separable-only sources stay version 1."""
    covariance = _sources()
    path = tmp_path / "sources.nc"
    covariance.to_dataset().to_netcdf(path)
    with xr.open_dataset(path) as opened:
        dataset = opened.load()

    restored = IndependentSourceCovariance.from_dataset(dataset)
    rhs = xr.ones_like(dataset.class_label_encoded, dtype=float)

    assert dataset.attrs["schema_version"] == 2
    assert dataset.spatial_kernel.values.tolist() == [
        "class_mixed_exponential",
        "class_mixed_exponential",
        "separable_exponential",
    ]
    assert dict(restored.source_covariances["FF"].class_kernels) == {
        1: GeodesicExponentialKernel(200.0),
        402: GeodesicExponentialKernel(300.0),
    }
    assert isinstance(restored.source_covariances["ocean"], SeparableExponentialCovariance)
    np.testing.assert_array_equal(restored.apply(rhs), covariance.apply(rhs))

    separable_only = IndependentSourceCovariance({"ocean": covariance.source_covariances["ocean"]})
    separable_dataset = separable_only.to_dataset()
    assert separable_dataset.attrs["schema_version"] == 1
    assert "spatial_kernel" not in separable_dataset and "geodesic_class" not in separable_dataset.dims

    contradictory = dataset.assign(
        spatial_kernel=dataset.spatial_kernel.copy(data=["separable_exponential"] * 3)
    )
    with pytest.raises(ValueError, match="separable source 'GPP' has class kernels"):
        IndependentSourceCovariance.from_dataset(contradictory)
    with pytest.raises(ValueError, match="missing labels or variables"):
        IndependentSourceCovariance.from_dataset(dataset.drop_vars("spatial_kernel"))
    with pytest.raises(ValueError, match="reserved"):
        IndependentSourceCovariance(dict(covariance.source_covariances), source_dim="geodesic_class")
