import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.operators import (
    BucketBasisOperator,
    MultiSourceBucketBasisOperator,
    BasisOperator,
)

from helpers import basis_function


@pytest.fixture
def basis_func():
    nlat, nlon = 10, 12
    nbasis = 3
    return basis_function(nlat, nlon, nbasis)


@pytest.fixture
def basis_func2():
    nlat, nlon = 10, 12
    nbasis = 7
    return basis_function(nlat, nlon, nbasis)


def test_bucket_basis_operator_roundtrip_datatree(basis_func):
    op = BucketBasisOperator(basis_func, state_dim="state")
    dt = op.to_datatree()
    op2 = BasisOperator.decode_datatree(dt)

    assert isinstance(op2, BucketBasisOperator)
    xr.testing.assert_identical(op2.basis_flat, op.basis_flat)
    xr.testing.assert_identical(op2.basis_matrix, op.basis_matrix)


def test_multisource_basis_operator_roundtrip_datatree(basis_func, basis_func2):
    basis = {"a": basis_func, "b": basis_func2}
    op = MultiSourceBucketBasisOperator(basis, state_dim="state")
    dt = op.to_datatree()
    op2 = BasisOperator.decode_datatree(dt)

    assert isinstance(op2, MultiSourceBucketBasisOperator)
    assert set(op2.basis_flat) == set(op.basis_flat)

    xr.testing.assert_identical(op2.basis_matrix, op.basis_matrix)

    for k in op.basis_flat:
        xr.testing.assert_identical(op2.basis_flat[k], op.basis_flat[k])


# --------------------------------------------------------------------------------------
# Canonical dim name test: do not bake in "region"
# --------------------------------------------------------------------------------------
def test_bucket_basis_operator_roundtrip_preserves_default_state_dim():
    basis = basis_function(6, 5, 3)
    op = BucketBasisOperator(basis, state_dim="state")
    dt = op.to_datatree()
    op2 = BasisOperator.decode_datatree(dt)

    assert op2.meta.state_dim == "state"
    assert "state" in op2.basis_matrix.dims


# --------------------------------------------------------------------------------------
# Test interpolation for MultiSourceBucketBasisOperator
# --------------------------------------------------------------------------------------

def _make_multisource_operator_for_broadcast_tests(state_dim: str = "state") -> MultiSourceBucketBasisOperator:
    """Create a tiny multisource operator with ragged per-source region counts.

    Source A has 2 regions; source B has 1 region.
    """
    basis_a = xr.DataArray(
        np.array([[1, 2], [1, 2]], dtype=int),
        dims=("lat", "lon"),
        coords={"lat": [0, 1], "lon": [0, 1]},
        name="basis_flat",
    )
    basis_b = xr.DataArray(
        np.array([[1, 1], [1, 1]], dtype=int),
        dims=("lat", "lon"),
        coords={"lat": [0, 1], "lon": [0, 1]},
        name="basis_flat",
    )
    return MultiSourceBucketBasisOperator({"A": basis_a, "B": basis_b}, state_dim=state_dim)


def _state_for_operator(op: MultiSourceBucketBasisOperator, values: list[float], name: str = "state") -> xr.DataArray:
    """Create a state(state) vector matching the operator's gathered MultiIndex order."""
    state_index = op.basis_matrix[op.meta.state_dim]
    if len(values) != state_index.size:
        raise ValueError("values must match gathered state size")

    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=(op.meta.state_dim,),
        coords={op.meta.state_dim: state_index},
        name=name,
    )


def test_multisource_interpolate_broadcasts_weights_source_to_state():
    """weights(source, lat, lon) should broadcast onto gathered state(source, region_in_source)."""
    op = _make_multisource_operator_for_broadcast_tests(state_dim="state")

    # State order is source-major: (A, r1), (A, r2), (B, r1)
    # For basis defined above:
    # - A r1 covers left column, A r2 covers right column, B r1 covers entire grid.
    state = _state_for_operator(op, [10.0, 100.0, 1000.0], name="x")

    # weights differ by source, so broadcasting/alignment matters
    weights = xr.DataArray(
        np.stack([np.ones((2, 2), dtype=float) * 2.0, np.ones((2, 2), dtype=float) * 3.0], axis=0),
        dims=("source", "lat", "lon"),
        coords={"source": ["A", "B"], "lat": [0, 1], "lon": [0, 1]},
        name="weights",
    )

    out = op.interpolate(state, weights=weights)
    assert set(out.dims) == {"lat", "lon"}

    expected = xr.DataArray(
        np.array([[10.0 * 2.0 + 1000.0 * 3.0, 100.0 * 2.0 + 1000.0 * 3.0],
                  [10.0 * 2.0 + 1000.0 * 3.0, 100.0 * 2.0 + 1000.0 * 3.0]], dtype=float),
        dims=("lat", "lon"),
        coords={"lat": [0, 1], "lon": [0, 1]},
        name=out.name,
    )

    xr.testing.assert_allclose(out, expected)


def test_multisource_interpolate_broadcasts_weights_source_with_nontrivial_values():
    """Second check: per-source weights and per-region states combine correctly."""
    op = _make_multisource_operator_for_broadcast_tests(state_dim="state")

    # Make A regions different, B different, to ensure ordering is respected.
    state_array = _state_for_operator(op, [1.0, 10.0, 100.0], name="x")

    # A weights=5, B weights=7
    weights = xr.DataArray(
        np.stack([np.ones((2, 2), dtype=float) * 5.0, np.ones((2, 2), dtype=float) * 7.0], axis=0),
        dims=("source", "lat", "lon"),
        coords={"source": ["A", "B"], "lat": [0, 1], "lon": [0, 1]},
        name="weights",
    )

    out = op.interpolate(state_array, weights=weights)

    expected = xr.DataArray(
        np.array(
            [
                [1.0 * 5.0 + 100.0 * 7.0, 10.0 * 5.0 + 100.0 * 7.0],
                [1.0 * 5.0 + 100.0 * 7.0, 10.0 * 5.0 + 100.0 * 7.0],
            ],
            dtype=float,
        ),
        dims=("lat", "lon"),
        coords={"lat": [0, 1], "lon": [0, 1]},
        name=out.name,
    )

    xr.testing.assert_allclose(out, expected)
