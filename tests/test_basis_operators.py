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
