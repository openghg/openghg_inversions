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
    same_unit_kernel,
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


def test_restored_sources_share_identical_dense_blocks(tmp_path) -> None:
    """Sources sharing a unit kernel before saving share one dense factor after loading."""
    gpp = _covariance(sigma=0.5)
    sources = IndependentSourceCovariance(
        {
            "GPP": gpp,
            "TER": gpp.with_sigma(0.6),
            "FF": _covariance(class_kernels={1: GeodesicExponentialKernel(200.0)}),
        }
    )
    path = tmp_path / "sources.nc"
    sources.to_dataset().to_netcdf(path)
    with xr.open_dataset(path) as opened:
        restored = IndependentSourceCovariance.from_dataset(opened.load())

    blocks = {name: restored.source_covariances[name]._blocks[0].storage for name in ("GPP", "TER", "FF")}
    assert blocks["TER"] is blocks["GPP"]
    assert blocks["FF"] is not blocks["GPP"]
    assert restored.source_covariances["TER"].sigma == 0.6
    rhs = xr.ones_like(restored.to_dataset().class_label_encoded, dtype=float)
    np.testing.assert_array_equal(restored.apply(rhs), sources.apply(rhs))


def test_class_kernel_keys_are_canonical_and_identity_ignores_sigma_and_label_metadata() -> None:
    """Key spelling and order do not change the stored kernel; same_unit_kernel compares the kernel only."""
    k1, k2 = GeodesicExponentialKernel(500.0), GeodesicExponentialKernel(300.0)
    first = _covariance(class_kernels={1: k1, 402: k2})
    reordered = _covariance(class_kernels={402.0: k2, np.int64(1): k1}, sigma=3.0)

    assert list(reordered.class_kernels.items()) == [(1, k1), (402, k2)]
    assert all(type(key) is int for key in reordered.class_kernels)
    first_dataset, reordered_dataset = first.to_dataset(), reordered.to_dataset()
    first_dataset.attrs.pop("sigma"), reordered_dataset.attrs.pop("sigma")
    xr.testing.assert_identical(first_dataset, reordered_dataset)
    assert first.same_unit_kernel(reordered) and same_unit_kernel(reordered, first)

    renamed = _covariance(
        class_labels=first.class_labels.rename("other").assign_attrs(note="ignored"),
        class_kernels={1: k1, 402: k2},
    )
    assert first.same_unit_kernel(renamed)
    assert not first.same_unit_kernel(_covariance(class_kernels={1: k1}))
    assert not first.same_unit_kernel(
        _covariance(class_kernels={1: k1, 402: GeodesicExponentialKernel(301.0)})
    )
    assert not first.same_unit_kernel(
        _covariance(class_kernels={1: k1, 402: k2}, latitude_correlation_length=4.0)
    )
    shifted = first.class_labels.where(first.class_labels != 2, 3)
    assert not first.same_unit_kernel(_covariance(class_labels=shifted, class_kernels={1: k1, 402: k2}))
    as_float = first.class_labels.astype(float)
    assert not first.same_unit_kernel(_covariance(class_labels=as_float, class_kernels={1: k1, 402: k2}))
    latitude, longitude = _grid()
    moved = ClassMixedExponentialCovariance(
        latitude.copy(data=latitude.values + 0.5).assign_coords(lat=latitude.values + 0.5),
        longitude,
        class_labels=first.class_labels.assign_coords(lat=latitude.values + 0.5),
        class_kernels={1: k1, 402: k2},
        correlation_length=4.0,
        latitude_correlation_length=first.latitude_correlation_length,
        longitude_correlation_length=first.longitude_correlation_length,
    )
    assert not first.same_unit_kernel(moved)
    separable = SeparableExponentialCovariance(
        latitude, longitude, correlation_length=2.0, class_labels=first.class_labels
    )
    assert not first.same_unit_kernel(separable) and not same_unit_kernel(separable, first)
    assert same_unit_kernel(
        separable,
        SeparableExponentialCovariance(
            latitude,
            longitude,
            sigma=4.0,
            correlation_length=2.0,
            class_labels=shifted.where(shifted != 3, 2),
        ),
    )
    assert not same_unit_kernel(separable, object())

    with pytest.raises(ValueError, match="Boolean"):
        _covariance(class_kernels={True: k1})


def test_coincident_polar_cells_are_named_in_the_error() -> None:
    """Several cells at a pole are one point on the sphere, and the error says so."""
    latitude = xr.DataArray(
        [80.0, 90.0], dims="lat", coords={"lat": [80.0, 90.0]}, attrs={"units": "degrees_north"}
    )
    longitude = xr.DataArray(
        [0.0, 30.0], dims="lon", coords={"lon": [0.0, 30.0]}, attrs={"units": "degrees_east"}
    )
    labels = xr.DataArray(
        np.ones((2, 2), int), dims=("lat", "lon"), coords={"lat": latitude, "lon": longitude}
    )

    with pytest.raises(
        np.linalg.LinAlgError, match=r"lat 90, lon 0\) and \(lat 90, lon 30\) coincide on the sphere"
    ):
        ClassMixedExponentialCovariance(
            latitude, longitude, class_labels=labels, class_kernels={1: GeodesicExponentialKernel(500.0)}
        )


def test_longitudes_across_the_antimeridian_and_in_0_360_form() -> None:
    """Great-circle distances wrap in longitude, so both conventions give one covariance."""
    latitude = xr.DataArray(
        [50.0, 55.0], dims="lat", coords={"lat": [50.0, 55.0]}, attrs={"units": "degrees_north"}
    )
    signed = [170.0, 175.0, -180.0 + 1.0, -175.0]
    unsigned = [170.0, 175.0, 181.0, 185.0]
    actions = []
    for values in (signed, unsigned):
        longitude = xr.DataArray(values, dims="lon", coords={"lon": values}, attrs={"units": "degrees_east"})
        labels = xr.DataArray(
            np.zeros((2, 4), int), dims=("lat", "lon"), coords={"lat": latitude, "lon": longitude}
        )
        actions.append(
            ClassMixedExponentialCovariance(
                latitude, longitude, class_labels=labels, class_kernels={0: GeodesicExponentialKernel(400.0)}
            )
        )
    field = np.random.default_rng(3).normal(size=(2, 4))
    results = [
        action.apply(
            xr.DataArray(field, dims=("lat", "lon"), coords={"lat": latitude, "lon": action.longitude})
        )
        for action in actions
    ]

    dense = _great_circle_correlation(np.repeat([50.0, 55.0], 4), np.tile(signed, 2), 400.0)
    np.testing.assert_allclose(results[0].values.reshape(-1), dense @ field.reshape(-1), rtol=1e-13)
    np.testing.assert_allclose(results[1].values, results[0].values, rtol=1e-13)
    # 175E to 179W is 6 degrees of longitude across the antimeridian, not 354.
    np.testing.assert_allclose(
        dense[1, 2], np.exp(-6.0 * 111.0 * np.cos(np.deg2rad(50.0)) / 400.0), rtol=1e-3
    )


def test_kernel_matches_the_reference_g_formula() -> None:
    """The kernel equals the OPE-188 geodesic-session G formula (geokern.py)."""
    rng = np.random.default_rng(11)
    lat1, lat2 = rng.uniform(33.0, 71.0, size=(2, 200))
    lon1, lon2 = rng.uniform(-30.0, 40.0, size=(2, 200))
    radius = 111.0 * 180.0 / np.pi
    for length in (200.0, 500.0):
        p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
        h = (
            np.sin(0.5 * (p1 - p2)) ** 2
            + np.cos(p1) * np.cos(p2) * np.sin(0.5 * np.deg2rad(lon1 - lon2)) ** 2
        )
        reference = np.exp(-2 * radius * np.arcsin(np.sqrt(np.clip(h, 0, 1))) / length)
        np.testing.assert_allclose(
            GeodesicExponentialKernel(length).correlation(lat1, lon1, lat2, lon2),
            reference,
            rtol=1e-15,
            atol=0.0,
        )


# IndependentSourceCovariance.to_dataset() output of origin/devel (b2bd9ec8), from ``Dataset.to_dict()``.
_DEVEL_V1_DATASET = {
    "coords": {
        "native_source": {"dims": ("native_source",), "attrs": {}, "data": ["z-source", "a-source"]},
        "lat": {"dims": ("lat",), "attrs": {"units": "degrees_north"}, "data": [-1.0, 0.5]},
        "lon": {"dims": ("lon",), "attrs": {"units": "degrees_east"}, "data": [10.0, 11.0]},
    },
    "attrs": {
        "schema": "openghg_inversions.independent_source_covariance",
        "schema_version": 1,
        "source_dim": "native_source",
        "latitude_dim": "lat",
        "longitude_dim": "lon",
        "class_label_encoding": "tagged_json_v1",
    },
    "dims": {"native_source": 2, "lat": 2, "lon": 2},
    "data_vars": {
        "sigma": {"dims": ("native_source",), "attrs": {}, "data": [1.6, 0.7]},
        "correlation_length": {"dims": ("native_source",), "attrs": {}, "data": [1.5, 1.5]},
        "latitude_correlation_length": {"dims": ("native_source",), "attrs": {}, "data": [0.7, 1.5]},
        "longitude_correlation_length": {"dims": ("native_source",), "attrs": {}, "data": [2.2, 1.5]},
        "latitude_correlation_length_explicit": {
            "dims": ("native_source",),
            "attrs": {},
            "data": [True, False],
        },
        "longitude_correlation_length_explicit": {
            "dims": ("native_source",),
            "attrs": {},
            "data": [True, False],
        },
        "class_blocked": {"dims": ("native_source",), "attrs": {}, "data": [True, False]},
        "class_label_encoded": {
            "dims": ("native_source", "lat", "lon"),
            "attrs": {},
            "data": [[['["int",1]', '["int",2]'], ['["int",1]', '["int",2]']], [["", ""], ["", ""]]],
        },
        "class_label_name": {"dims": ("native_source",), "attrs": {}, "data": ["surface_class", ""]},
        "class_label_attrs": {
            "dims": ("native_source",),
            "attrs": {},
            "data": ['{"description": "golden"}', "{}"],
        },
    },
}
# Devel's apply and solve of np.arange(8.0) on (native_source, lat, lon) for that covariance.
_DEVEL_V1_APPLY = [
    [[0.6006741304025641, 3.461011195603847], [5.120000000000001, 7.980337065201284]],
    [[4.947280816406728, 5.2734183088643585], [5.884809649929843, 6.210947142387475]],
]
_DEVEL_V1_SOLVE = [
    [[-0.09293473234725412, 0.2566744141292048], [0.7921530253001721, 1.1417621717766309]],
    [[1.7556444252000047, 4.821835903108392], [6.022179871335041, 9.088371349243415]],
]


def test_devel_version_1_dataset_still_loads_and_is_still_written() -> None:
    """A separable-only dataset written by devel loads, and the same configuration writes it unchanged."""
    golden = xr.Dataset.from_dict(_DEVEL_V1_DATASET)
    restored = IndependentSourceCovariance.from_dataset(golden)
    rhs = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("native_source", "lat", "lon"),
        coords={"native_source": ["z-source", "a-source"], "lat": golden.lat, "lon": golden.lon},
    )

    xr.testing.assert_identical(restored.to_dataset(), golden)
    assert restored.to_dataset().attrs["schema_version"] == 1
    assert IndependentSourceCovariance.schema_version == 2
    np.testing.assert_array_equal(restored.apply(rhs).values, _DEVEL_V1_APPLY)
    np.testing.assert_allclose(restored.solve(rhs).values, _DEVEL_V1_SOLVE, rtol=1e-13)


def test_string_class_labels_round_trip_through_independent_sources(tmp_path) -> None:
    """Typed string labels and their kernels survive the version-2 NetCDF round trip."""
    latitude, longitude = _grid()
    labels = _labels(latitude, longitude).astype(str)
    labels = labels.where(labels != "1", "inner").where(labels != "2", "sea")
    mixed = _covariance(class_labels=labels, class_kernels={"inner": GeodesicExponentialKernel(500.0)})
    ocean = SeparableExponentialCovariance(latitude, longitude, sigma=0.2, class_labels=labels)
    sources = IndependentSourceCovariance({"GPP": mixed, "ocean": ocean})
    path = tmp_path / "sources.nc"
    sources.to_dataset().to_netcdf(path)
    with xr.open_dataset(path) as opened:
        restored = IndependentSourceCovariance.from_dataset(opened.load())

    restored_mixed = restored.source_covariances["GPP"]
    assert isinstance(restored_mixed, ClassMixedExponentialCovariance)
    assert dict(restored_mixed.class_kernels) == {"inner": GeodesicExponentialKernel(500.0)}
    assert restored_mixed.same_unit_kernel(mixed)
    np.testing.assert_array_equal(restored_mixed.class_labels.values, labels.values)


def test_reserved_spatial_dimension_is_rejected_for_class_mixed_sources() -> None:
    """A spatial dimension may not reuse a class-mixed serialization name."""
    latitude, longitude = _grid()
    renamed = longitude.rename(lon="spatial_kernel")
    labels = _labels(latitude, longitude).rename(lon="spatial_kernel")
    mixed = ClassMixedExponentialCovariance(
        latitude, renamed, class_labels=labels, class_kernels={1: GeodesicExponentialKernel(500.0)}
    )

    with pytest.raises(ValueError, match="spatial dimension 'spatial_kernel' is reserved"):
        IndependentSourceCovariance({"GPP": mixed})


def test_separable_class_solve_reports_non_convergence(monkeypatch) -> None:
    """A conjugate-gradient failure on the separable classes raises LinAlgError."""
    covariance = _covariance()
    monkeypatch.setattr(
        native_covariance_module, "cg", lambda operator, rhs, **kwargs: (np.zeros_like(rhs), 7)
    )

    with pytest.raises(np.linalg.LinAlgError, match="Separable-class covariance solve did not converge"):
        covariance.solve(covariance.class_labels.astype(float))
