"""Tests for labelled affine native-scaling and flux reconstruction."""

import dask.array as da
from dask import delayed
import numpy as np
import pytest
import xarray as xr
from sparse import COO

from openghg_inversions.basis import AffineFluxMap
from openghg_inversions.basis.operators import BucketBasisOperator, MultiSourceBucketBasisOperator


def _arrays() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    native_mean = xr.DataArray(
        [[0.7, 1.2], [1.5, 0.4]],
        dims=("lat", "lon"),
        coords={"lat": [50.0, 51.0], "lon": [-2.0, -1.0]},
        attrs={"units": "1"},
        name="native_mean",
    )
    prolongation = xr.DataArray(
        [
            [[0.8, 0.2], [0.1, 0.9]],
            [[-0.3, 1.3], [0.6, 0.4]],
        ],
        dims=("lat", "lon", "state"),
        coords={"lat": native_mean.lat, "lon": native_mean.lon, "state": ["west", "east"]},
        attrs={"units": "1"},
        name="prolongation",
    )
    flux = xr.DataArray(
        [
            [[2.0, -1.0], [0.5, 3.0]],
            [[4.0, -2.0], [1.0, 6.0]],
        ],
        dims=("time", "lat", "lon"),
        coords={"time": ["2020-01", "2020-02"], "lat": native_mean.lat, "lon": native_mean.lon},
        attrs={"units": "mol m-2 s-1"},
        name="flux",
    )
    reference = xr.DataArray(
        [0.4, 1.7],
        dims="state",
        coords={"state": prolongation.state},
        attrs={"units": "1"},
        name="reference_state",
    )
    state = xr.DataArray(
        [
            [[0.8, 1.1], [1.5, 2.2], [0.2, 1.9]],
            [[0.6, 1.4], [1.0, 1.7], [1.8, 0.5]],
        ],
        dims=("chain", "draw", "state"),
        coords={"chain": [0, 1], "draw": [0, 1, 2], "state": prolongation.state},
        attrs={"units": "1"},
        name="alpha",
    )
    return native_mean, flux, prolongation, reference, state


def test_explicit_reconstruction_matches_independent_dense_oracle() -> None:
    """Arbitrary means, reference states, and sample axes obey the affine equations."""
    native_mean, flux, prolongation, reference, state = _arrays()
    affine_map = AffineFluxMap(
        native_mean=native_mean,
        flux=flux,
        prolongation=prolongation,
        state_dim="state",
    )

    native = affine_map.state_to_native(state, reference_state=reference)
    reconstructed_flux = affine_map.state_to_flux(state, reference_state=reference)

    u = prolongation.values.reshape(4, 2)
    centred = state.values.reshape(6, 2) - reference.values
    expected_native = native_mean.values.reshape(4, 1) + u @ centred.T
    expected_native = expected_native.reshape(2, 2, 2, 3)
    expected_flux = flux.values[:, :, :, None, None] * expected_native[None, :, :, :, :]
    np.testing.assert_allclose(
        native.transpose("lat", "lon", "chain", "draw"),
        expected_native,
    )
    np.testing.assert_allclose(
        reconstructed_flux.transpose("time", "lat", "lon", "chain", "draw"),
        expected_flux,
    )
    assert native.attrs == {"units": "1", "uncertainty_scope": "retained_state_conditional"}
    assert reconstructed_flux.attrs == {
        "units": "mol m-2 s-1",
        "uncertainty_scope": "retained_state_conditional",
    }


def test_bucket_and_equivalent_explicit_prolongations_match() -> None:
    """The two closed representations share one reconstruction contract."""
    native_mean, flux, _, _, state = _arrays()
    basis_flat = xr.DataArray(
        [[1, 1], [2, 2]],
        dims=("lat", "lon"),
        coords={"lat": native_mean.lat, "lon": native_mean.lon},
    )
    operator = BucketBasisOperator(basis_flat, state_dim="state")
    state = state.assign_coords(state=operator.basis_matrix.state)
    reference = xr.DataArray(
        [0.4, 1.7],
        dims="state",
        coords={"state": operator.basis_matrix.state},
        attrs={"units": "1"},
    )
    explicit = operator._native_prolongation(
        native_mean,
        native_dims=("lat", "lon"),
    ).assign_attrs(units="1")
    bucket_map = AffineFluxMap(
        native_mean,
        flux,
        operator,
        state_dim="state",
    )
    explicit_map = AffineFluxMap(
        native_mean,
        flux,
        explicit,
        state_dim="state",
    )

    xr.testing.assert_allclose(
        bucket_map.state_to_native(state, reference_state=reference),
        explicit_map.state_to_native(state, reference_state=reference),
    )
    xr.testing.assert_allclose(
        bucket_map.state_to_flux(state, reference_state=reference),
        explicit_map.state_to_flux(state, reference_state=reference),
    )
    assert bucket_map.representation == "bucket"
    assert explicit_map.representation == "explicit"
    assert bucket_map.prolongation is operator


def test_country_contraction_precedes_sample_axes() -> None:
    """Native ingredients form a compact country action before applying draws."""
    mean, flux, prolongation, reference, state = _arrays()
    affine_map = AffineFluxMap(mean, flux, prolongation, state_dim="state")
    membership = xr.DataArray(
        [
            [[1.0, 0.5], [0.0, 0.0]],
            [[0.0, 0.5], [1.0, 1.0]],
        ],
        dims=("country", "lat", "lon"),
        coords={"country": ["A", "B"], "lat": mean.lat, "lon": mean.lon},
    )
    area = xr.DataArray(
        [[2.0, 3.0], [4.0, 5.0]],
        dims=("lat", "lon"),
        coords={"lat": mean.lat, "lon": mean.lon},
    )
    conversion = 7.0
    weights = membership * area * conversion

    reference_country = xr.dot(weights, flux * mean, dim=affine_map.native_dims)
    country_to_state = xr.dot(
        weights * flux,
        prolongation,
        dim=affine_map.native_dims,
    )
    assert set(country_to_state.dims) == {"country", "time", "state"}
    assert "chain" not in country_to_state.dims
    assert "draw" not in country_to_state.dims

    country_draws = reference_country + xr.dot(country_to_state, state - reference, dim="state")
    expected_grid = (
        mean.values[None, :, :, None, None]
        + np.einsum("ijk,cdk->ijcd", prolongation.values, state.values - reference.values)[None, :, :, :, :]
    )
    expected_country = np.einsum("kij,tij,tijcd->ktcd", weights.values, flux.values, expected_grid)
    np.testing.assert_allclose(
        country_draws.transpose("country", "time", "chain", "draw"),
        expected_country,
    )


def test_multisource_bucket_preserves_native_source_order_and_gathered_state() -> None:
    """Native sources stay explicit while the retained state remains one ragged axis."""
    native_mean, _, _, _, _ = _arrays()
    basis_a = xr.DataArray(
        [[1, 1], [2, 2]],
        dims=("lat", "lon"),
        coords={"lat": native_mean.lat, "lon": native_mean.lon},
    )
    basis_b = xr.DataArray(
        [[1, 2], [1, 2]],
        dims=("lat", "lon"),
        coords={"lat": native_mean.lat, "lon": native_mean.lon},
    )
    operator = MultiSourceBucketBasisOperator({"fossil": basis_a, "bio": basis_b})
    native_mean = xr.concat(
        [native_mean, native_mean + 0.3],
        dim=xr.IndexVariable("native_source", ["fossil", "bio"]),
    ).assign_attrs(units="1")
    flux = xr.ones_like(native_mean).assign_attrs(units="kg m-2 s-1")
    reference = xr.DataArray(
        [0.2, 0.4, 1.3, 0.8],
        dims="state",
        coords={"state": operator.basis_matrix.state},
        attrs={"units": "1"},
    )
    state = (reference + [0.5, -0.1, 0.2, 0.7]).assign_attrs(units="1")
    affine_map = AffineFluxMap(
        native_mean,
        flux,
        operator,
        state_dim="state",
    )
    explicit = operator._native_prolongation(
        native_mean,
        native_dims=("native_source", "lat", "lon"),
    ).assign_attrs(units="1")
    explicit_map = AffineFluxMap(
        native_mean,
        flux,
        explicit,
        state_dim="state",
    )

    xr.testing.assert_allclose(
        affine_map.state_to_native(state, reference_state=reference),
        explicit_map.state_to_native(state, reference_state=reference),
    )
    assert affine_map.state_to_native(state, reference_state=reference).native_source.values.tolist() == [
        "fossil",
        "bio",
    ]
    assert affine_map.state_to_native(state, reference_state=reference).dims == (
        "native_source",
        "lat",
        "lon",
    )

    with pytest.raises(ValueError, match="flux 'native_source' labels must exactly match"):
        AffineFluxMap(
            native_mean,
            flux.isel(native_source=[1, 0]),
            operator,
            state_dim="state",
        )


def test_multisource_reconstruction_matches_independent_source_oracle() -> None:
    """Source order, gathered regions, centring, and signed flux have independent values."""
    native_mean, _, _, _, _ = _arrays()
    grid = {"lat": native_mean.lat, "lon": native_mean.lon}
    bio_basis = xr.DataArray([[1, 2], [1, 2]], dims=("lat", "lon"), coords=grid)
    fossil_basis = xr.DataArray([[1, 1], [2, 2]], dims=("lat", "lon"), coords=grid)
    operator = MultiSourceBucketBasisOperator({"fossil": fossil_basis, "bio": bio_basis})
    mean = xr.DataArray(
        [[[0.4, 0.7], [1.3, 1.6]], [[0.9, 1.1], [0.8, 1.4]]],
        dims=("native_source", "lat", "lon"),
        coords={"native_source": ["fossil", "bio"], **grid},
        attrs={"units": "1"},
    )
    flux = xr.DataArray(
        [[[-2.0, -3.0], [4.0, 5.0]], [[7.0, -8.0], [9.0, -10.0]]],
        dims=mean.dims,
        coords=mean.coords,
        attrs={"units": "kg m-2 s-1"},
    )
    reference = xr.DataArray(
        [0.2, 1.4, 0.5, 1.8],
        dims="state",
        coords={"state": operator.basis_matrix.state},
        attrs={"units": "1"},
    )
    state = xr.DataArray(
        [[0.6, 1.1, 1.7, 0.9], [1.2, 0.3, 0.4, 2.3]],
        dims=("draw", "state"),
        coords={"draw": [0, 1], "state": operator.basis_matrix.state},
        attrs={"units": "1"},
    )
    affine_map = AffineFluxMap(mean, flux, operator, state_dim="state")
    assert affine_map.native_dims == mean.dims
    assert reference.indexes["state"].tolist() == [("fossil", 0), ("fossil", 1), ("bio", 0), ("bio", 1)]

    delta = state.values - reference.values
    expected = np.empty((2, 2, 2, 2))
    for source_position, region_indices in enumerate(([[0, 0], [1, 1]], [[2, 3], [2, 3]])):
        for latitude in range(2):
            for longitude in range(2):
                expected[source_position, latitude, longitude] = (
                    mean.values[source_position, latitude, longitude]
                    + delta[:, region_indices[latitude][longitude]]
                )
    native = affine_map.state_to_native(state, reference_state=reference)
    reconstructed_flux = affine_map.state_to_flux(state, reference_state=reference)
    np.testing.assert_allclose(native.transpose("native_source", "lat", "lon", "draw"), expected)
    np.testing.assert_allclose(
        reconstructed_flux.transpose("native_source", "lat", "lon", "draw"),
        expected * flux.values[:, :, :, None],
    )


def test_bucket_construction_does_not_expand_multisource_prolongation(monkeypatch) -> None:
    """Construction checks labels using the retained matrix without building native U*."""
    native_mean, _, _, _, _ = _arrays()
    basis = xr.DataArray(
        [[1, 1], [2, 2]],
        dims=("lat", "lon"),
        coords={"lat": native_mean.lat, "lon": native_mean.lon},
    )
    operator = MultiSourceBucketBasisOperator({"bio": basis, "fossil": basis})
    mean = xr.concat(
        [native_mean, native_mean],
        dim=xr.IndexVariable("native_source", ["bio", "fossil"]),
    ).assign_attrs(units="1")
    flux = xr.ones_like(mean).assign_attrs(units="kg m-2 s-1")

    def unexpected_expansion(*args, **kwargs):
        raise AssertionError("native prolongation expanded during construction")

    monkeypatch.setattr(operator, "_native_prolongation", unexpected_expansion)
    AffineFluxMap(mean, flux, operator, state_dim="state")


@pytest.mark.parametrize(
    ("sample_dim", "sample_coord"),
    [
        ("lat", [50.0, 51.0]),
        ("time", ["2020-01", "2020-02"]),
        ("time", ["sample-a", "sample-b"]),
    ],
)
def test_sample_axis_name_collision_keeps_independent_axes(sample_dim: str, sample_coord: list) -> None:
    """Native and flux coordinates never pair with same-named sample coordinates."""
    mean, flux, prolongation, reference, _ = _arrays()
    sample = xr.DataArray(
        [[0.8, 1.1], [1.5, 2.2]],
        dims=(sample_dim, "state"),
        coords={sample_dim: sample_coord, "state": reference.state},
        attrs={"units": "1"},
    )
    affine_map = AffineFluxMap(mean, flux, prolongation, state_dim="state")
    native = affine_map.state_to_native(sample, reference_state=reference)
    result = affine_map.state_to_flux(sample, reference_state=reference)

    expected = mean.values[:, :, None] + (
        prolongation.values.reshape(4, 2) @ (sample.values - reference.values).T
    ).reshape(2, 2, 2)
    sample_axis = f"state_{sample_dim}"
    assert native.sizes[sample_axis] == 2
    assert result.sizes[sample_axis] == 2
    np.testing.assert_allclose(native.transpose("lat", "lon", sample_axis), expected)
    np.testing.assert_allclose(
        result.transpose("time", "lat", "lon", sample_axis),
        flux.values[:, :, :, None] * expected[None, :, :, :],
    )


@pytest.mark.parametrize(
    ("auxiliary_names", "renamed_axis"),
    [
        (("state_time",), "state_time_2"),
        (("state_time", "state_time_2"), "state_time_3"),
    ],
)
def test_colliding_state_axis_avoids_auxiliary_coordinate_names(
    auxiliary_names: tuple[str, ...], renamed_axis: str
) -> None:
    """An auxiliary coordinate cannot shadow the renamed independent axis."""
    mean, flux, prolongation, reference, _ = _arrays()
    sample = xr.DataArray(
        [[0.8, 1.1], [1.5, 2.2]],
        dims=("time", "state"),
        coords={"time": ["2020-01", "2020-02"], "state": reference.state},
        attrs={"units": "1"},
    ).assign_coords({name: ("time", [10, 20]) for name in auxiliary_names})
    affine_map = AffineFluxMap(mean, flux, prolongation, state_dim="state")

    native = affine_map.state_to_native(sample, reference_state=reference)
    reconstructed_flux = affine_map.state_to_flux(sample, reference_state=reference)

    expected = mean.values[:, :, None] + (
        prolongation.values.reshape(4, 2) @ (sample.values - reference.values).T
    ).reshape(2, 2, 2)
    assert renamed_axis in native.dims
    assert renamed_axis in reconstructed_flux.dims
    assert all(name in native.coords for name in auxiliary_names)
    np.testing.assert_allclose(native.transpose("lat", "lon", renamed_axis), expected)
    np.testing.assert_allclose(
        reconstructed_flux.transpose("time", "lat", "lon", renamed_axis),
        flux.values[:, :, :, None] * expected[None, :, :, :],
    )


def test_already_prefixed_colliding_state_axis_uses_numeric_suffix() -> None:
    """A state axis already named ``state_*`` is not prefixed twice."""
    mean, flux, prolongation, reference, _ = _arrays()
    flux = flux.rename(time="state_time")
    sample = xr.DataArray(
        [[0.8, 1.1], [1.5, 2.2]],
        dims=("state_time", "state"),
        coords={"state_time": ["2020-01", "2020-02"], "state": reference.state},
        attrs={"units": "1"},
    )
    affine_map = AffineFluxMap(mean, flux, prolongation, state_dim="state")

    native = affine_map.state_to_native(sample, reference_state=reference)
    reconstructed_flux = affine_map.state_to_flux(sample, reference_state=reference)

    assert "state_time_2" in native.dims
    assert "state_time_2" in reconstructed_flux.dims
    assert "state_state_time" not in reconstructed_flux.dims
    assert reconstructed_flux.sizes["state_time"] == 2
    assert reconstructed_flux.sizes["state_time_2"] == 2


def test_compatible_scaled_units_are_converted_without_changing_inputs() -> None:
    """Percent and dimensionless input conventions produce the same physical grids."""
    mean, flux, prolongation, reference, state = _arrays()
    expected = AffineFluxMap(mean, flux, prolongation, "state")
    scaled_map = AffineFluxMap(
        mean.assign_attrs(units="percent").copy(data=mean.data * 100),
        flux,
        prolongation.assign_attrs(units="percent").copy(data=prolongation.data * 100),
        "state",
    )
    percent_state = state.assign_attrs(units="percent").copy(data=state.data * 100)
    percent_reference = reference.assign_attrs(units="percent").copy(data=reference.data * 100)

    xr.testing.assert_allclose(
        scaled_map.state_to_native(percent_state, reference_state=percent_reference),
        expected.state_to_native(state, reference_state=reference),
    )
    xr.testing.assert_allclose(
        scaled_map.state_to_flux(percent_state, reference_state=percent_reference),
        expected.state_to_flux(state, reference_state=reference),
    )
    assert scaled_map.native_mean.attrs["units"] == "percent"
    assert percent_state.attrs["units"] == "percent"


def test_invalid_flux_unit_is_rejected() -> None:
    mean, flux, prolongation, _, _ = _arrays()
    with pytest.raises(ValueError, match="flux units.*invalid"):
        AffineFluxMap(mean, flux.assign_attrs(units="not_a_real_unit"), prolongation, "state")


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        (
            "native_mean",
            lambda value: value.isel(lat=[1, 0]),
            "flux 'lat' labels must exactly match",
        ),
        (
            "flux",
            lambda value: value.assign_coords(lon=[-2.0, -2.0]),
            "flux 'lon' labels must be unique",
        ),
        (
            "prolongation",
            lambda value: value.isel(lon=[1, 0]),
            "prolongation 'lon' labels must exactly match",
        ),
        (
            "prolongation",
            lambda value: value.drop_indexes("state").drop_vars("state"),
            "prolongation requires a labelled 'state' dimension",
        ),
        (
            "native_mean",
            lambda value: value.assign_attrs(units="m"),
            "native_mean units.*incompatible",
        ),
    ],
)
def test_construction_rejects_incompatible_native_inputs(field, replacement, match) -> None:
    """Independent native ingredients must align exactly before multiplication."""
    native_mean, flux, prolongation, _, _ = _arrays()
    values = {
        "native_mean": native_mean,
        "flux": flux,
        "prolongation": prolongation,
    }
    values[field] = replacement(values[field])

    with pytest.raises(ValueError, match=match):
        AffineFluxMap(
            **values,
            state_dim="state",
        )


@pytest.mark.parametrize(
    ("which", "replacement", "match"),
    [
        ("state", lambda value: value.isel(state=[1, 0]), "state 'state' labels must exactly match"),
        (
            "reference",
            lambda value: value.assign_coords(state=["west", "west"]),
            "reference_state 'state' labels must be unique",
        ),
        (
            "reference",
            lambda value: value.rename(state="region"),
            "reference_state must have dimensions",
        ),
        ("state", lambda value: value.assign_attrs(units="kg"), "state units.*incompatible"),
    ],
)
def test_application_rejects_incompatible_state_inputs(which, replacement, match) -> None:
    """State data never falls back to positional alignment or unitless broadcasting."""
    native_mean, flux, prolongation, reference, state = _arrays()
    affine_map = AffineFluxMap(
        native_mean,
        flux,
        prolongation,
        state_dim="state",
    )
    if which == "state":
        state = replacement(state)
    else:
        reference = replacement(reference)

    with pytest.raises(ValueError, match=match):
        affine_map.state_to_native(state, reference_state=reference)


def test_construction_and_application_preserve_borrowed_dask_ownership() -> None:
    """Constructor and application retain payloads without executing their graph."""
    native_mean, flux, prolongation, reference, state = _arrays()
    executions: list[str] = []

    def read_payload(name: str, values: np.ndarray) -> np.ndarray:
        executions.append(name)
        return values

    def lazy(array: xr.DataArray, name: str) -> xr.DataArray:
        payload = da.from_delayed(
            delayed(read_payload)(name, array.data), shape=array.shape, dtype=array.dtype
        )
        return array.copy(data=payload)

    lazy_mean = lazy(native_mean, "mean")
    lazy_flux = lazy(flux, "flux")
    lazy_prolongation = lazy(prolongation, "prolongation")
    lazy_state = lazy(state, "state")

    affine_map = AffineFluxMap(
        lazy_mean,
        lazy_flux,
        lazy_prolongation,
        state_dim="state",
    )
    native = affine_map.state_to_native(lazy_state, reference_state=reference)
    reconstructed_flux = affine_map.state_to_flux(lazy_state, reference_state=reference)

    assert affine_map.native_mean.data is lazy_mean.data
    assert affine_map.flux.data is lazy_flux.data
    assert affine_map.prolongation.data is lazy_prolongation.data
    assert isinstance(native.data, da.Array)
    assert isinstance(reconstructed_flux.data, da.Array)
    assert executions == []


def test_bucket_prolongation_stays_sparse_until_reconstruction() -> None:
    """Construction retains the operator rather than flattening its sparse matrix."""
    native_mean, flux, _, reference, state = _arrays()
    operator = BucketBasisOperator(
        xr.DataArray(
            [[1, 1], [2, 2]],
            dims=("lat", "lon"),
            coords={"lat": native_mean.lat, "lon": native_mean.lon},
        ),
        state_dim="state",
    )
    reference = reference.assign_coords(state=operator.basis_matrix.state)
    state = state.assign_coords(state=operator.basis_matrix.state)

    affine_map = AffineFluxMap(
        native_mean,
        flux,
        operator,
        state_dim="state",
    )

    assert affine_map.prolongation is operator
    assert isinstance(operator.basis_matrix.data, da.Array)
    assert operator.basis_matrix.data._meta.__class__.__module__.startswith("sparse")
    native = affine_map.state_to_native(state, reference_state=reference)
    assert isinstance(native.data, da.Array)
    assert isinstance(operator.basis_matrix.data._meta, COO)


def test_invalid_prolongation_type_is_rejected() -> None:
    """The representation set remains closed rather than accepting arbitrary operators."""
    native_mean, flux, _, _, _ = _arrays()

    with pytest.raises(TypeError, match="prolongation must be"):
        AffineFluxMap(
            native_mean,
            flux,
            object(),  # type: ignore[arg-type]
            state_dim="state",
        )
