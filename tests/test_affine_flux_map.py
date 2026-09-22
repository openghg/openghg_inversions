"""Tests for labelled affine native-scaling and flux reconstruction."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

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
        native_dims=("lat", "lon"),
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
    explicit = operator.native_prolongation(
        native_mean,
        native_dims=("lat", "lon"),
    ).assign_attrs(units="1")
    bucket_map = AffineFluxMap(
        native_mean,
        flux,
        operator,
        native_dims=("lat", "lon"),
        state_dim="state",
    )
    explicit_map = AffineFluxMap(
        native_mean,
        flux,
        explicit,
        native_dims=("lat", "lon"),
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
        native_dims=("native_source", "lat", "lon"),
        state_dim="state",
    )
    explicit = operator.native_prolongation(
        native_mean,
        native_dims=("native_source", "lat", "lon"),
    ).assign_attrs(units="1")
    explicit_map = AffineFluxMap(
        native_mean,
        flux,
        explicit,
        native_dims=("native_source", "lat", "lon"),
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
            native_dims=("native_source", "lat", "lon"),
            state_dim="state",
        )


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
            native_dims=("lat", "lon"),
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
        native_dims=("lat", "lon"),
        state_dim="state",
    )
    if which == "state":
        state = replacement(state)
    else:
        reference = replacement(reference)

    with pytest.raises(ValueError, match=match):
        affine_map.state_to_native(state, reference_state=reference)


def test_construction_and_application_preserve_borrowed_dask_ownership() -> None:
    """The value neither executes nor replaces borrowed lazy payloads."""
    native_mean, flux, prolongation, reference, state = _arrays()
    lazy_mean = native_mean.copy(data=da.from_array(native_mean.data, chunks=(1, 2)))
    lazy_flux = flux.copy(data=da.from_array(flux.data, chunks=(1, 1, 2)))
    lazy_prolongation = prolongation.copy(
        data=da.from_array(prolongation.data, chunks=(1, 2, 2))
    )
    lazy_state = state.copy(data=da.from_array(state.data, chunks=(1, 2, 2)))

    affine_map = AffineFluxMap(
        lazy_mean,
        lazy_flux,
        lazy_prolongation,
        native_dims=("lat", "lon"),
        state_dim="state",
    )
    native = affine_map.state_to_native(lazy_state, reference_state=reference)
    reconstructed_flux = affine_map.state_to_flux(lazy_state, reference_state=reference)

    assert affine_map.native_mean.data is lazy_mean.data
    assert affine_map.flux.data is lazy_flux.data
    assert affine_map.prolongation.data is lazy_prolongation.data
    assert isinstance(native.data, da.Array)
    assert isinstance(reconstructed_flux.data, da.Array)


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
        native_dims=("lat", "lon"),
        state_dim="state",
    )

    assert affine_map.prolongation is operator
    assert isinstance(operator.basis_matrix.data, da.Array)
    assert operator.basis_matrix.data._meta.__class__.__module__.startswith("sparse")
    assert isinstance(
        affine_map.state_to_native(state, reference_state=reference).data,
        da.Array,
    )


def test_invalid_prolongation_type_is_rejected() -> None:
    """The representation set remains closed rather than accepting arbitrary operators."""
    native_mean, flux, _, _, _ = _arrays()

    with pytest.raises(TypeError, match="prolongation must be"):
        AffineFluxMap(
            native_mean,
            flux,
            object(),  # type: ignore[arg-type]
            native_dims=("lat", "lon"),
            state_dim="state",
        )
