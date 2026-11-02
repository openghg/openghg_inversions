"""Small test-data helpers remain independent of other fixture calls."""

import xarray as xr

from tests.helpers import basis_function


def test_basis_function_repeats_across_interleaved_calls():
    expected = basis_function(6, 5, 3)
    xr.testing.assert_identical(basis_function(6, 5, 3), expected)
    basis_function(10, 12, 7)
    xr.testing.assert_identical(basis_function(6, 5, 3), expected)
