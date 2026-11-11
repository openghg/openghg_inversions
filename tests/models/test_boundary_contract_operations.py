"""Boundary domain operations preserve labels and own eager checks explicitly."""

import dask
import numpy as np
import pytensor
import pytest
import xarray as xr

from openghg_inversions.models.boundary import (
    BoundAdditiveSensitivity,
    BoundCenteredSensitivity,
    RawCenteredSensitivity,
    normalize_centering_weights,
)
from test_boundary_contracts import _forbid_compute, _observations, _raw_on_nmeasure, bound_example


def test_replacing_weights_keeps_the_supplied_calendar_until_binding():
    """An old Dataset calendar must not truncate a replacement's extra period."""
    raw = RawCenteredSensitivity.validate(_raw_on_nmeasure(lazy=True))
    original = raw.copy(deep=False)
    old_weights = raw.bc_centering_weights
    extra = old_weights.assign_coords(bc_period=old_weights.bc_period + np.timedelta64(1, "D"))
    weights = xr.concat([old_weights, extra], dim="bc_period")
    with dask.config.set(scheduler=_forbid_compute), pytensor.config.change_flags(floatX="float64"):
        replaced = RawCenteredSensitivity.with_centering_weights(raw, weights)
        xr.testing.assert_identical(replaced.bc_centering_weights, weights)
        bound = BoundCenteredSensitivity.bind(replaced, _observations())
        with pytest.raises(ValueError, match="align"):
            BoundAdditiveSensitivity.bind_weights(bound, replaced.bc_centering_weights)
    xr.testing.assert_identical(raw, original)


def test_weights_with_the_bound_labels_in_another_order_bind_by_label():
    """Weight order is not part of the contract; binding selects by curtain and period label."""
    raw = RawCenteredSensitivity.validate(_raw_on_nmeasure(lazy=True))
    weights = raw.bc_centering_weights
    permuted = weights.isel(bc_curtain=slice(None, None, -1), bc_period=slice(None, None, -1))
    with dask.config.set(scheduler=_forbid_compute), pytensor.config.change_flags(floatX="float64"):
        bound = BoundCenteredSensitivity.bind(raw, _observations())
        expected = BoundAdditiveSensitivity.bind_weights(bound, weights)
        reordered = BoundAdditiveSensitivity.bind_weights(bound, permuted)
    xr.testing.assert_identical(reordered.bc_centering_weights, expected.bc_centering_weights)


@pytest.mark.parametrize("coordinate", [False, True])
def test_replacing_weights_rejects_axes_shared_with_other_data(coordinate):
    """Replacement must not silently delete extra variables or auxiliary coordinates."""
    raw = _raw_on_nmeasure()
    other = xr.ones_like(raw.bc_centering_weights)
    raw = raw.assign_coords(other=other) if coordinate else raw.assign(other=other)
    with pytest.raises(ValueError, match="shared.*other"):
        RawCenteredSensitivity.with_centering_weights(RawCenteredSensitivity.validate(raw), raw.bc_centering_weights)


def test_weight_preparation_is_structural_and_normalization_owns_values():
    """Preparation neither computes lazy weights nor repeats the later value scan."""
    raw = RawCenteredSensitivity.validate(_raw_on_nmeasure())
    invalid = xr.full_like(raw.bc_centering_weights, -1.0)
    for weights in (invalid, invalid.chunk()):
        with dask.config.set(scheduler=_forbid_compute):
            RawCenteredSensitivity.with_centering_weights(raw, weights)
    with pytest.raises(ValueError, match="finite and strictly positive"):
        normalize_centering_weights(invalid)


def test_numerical_operations_do_not_materialize_individual_variables():
    """Explicit eager operations reject lazy inputs before touching .values."""
    with dask.config.set(scheduler=_forbid_compute):
        with pytest.raises(ValueError, match="Materialize boundary data jointly"):
            BoundCenteredSensitivity.reference_shift(BoundCenteredSensitivity.validate(bound_example(lazy=True)))
        with pytest.raises(ValueError, match="Materialize bc_centering_weights"):
            normalize_centering_weights(bound_example(lazy=True, weights=True).bc_centering_weights)


def test_reference_shift_rejects_zero_mass_after_valid_row_selection():
    """A nonzero raw mass promise cannot survive arbitrary observation subsetting."""
    raw = _raw_on_nmeasure(weights=False)
    raw["H_bc_mass"] = raw.H_bc_mass.where(raw.nmeasure > 1, 0.0)
    with pytensor.config.change_flags(floatX="float64"):
        bound = BoundCenteredSensitivity.bind(RawCenteredSensitivity.validate(raw), _observations().isel(nmeasure=[0, 1]))
    assert BoundCenteredSensitivity.validate(bound) is bound
    with pytest.raises(ValueError, match="not all zero"):
        BoundCenteredSensitivity.reference_shift(bound)


def test_numerical_results_keep_units_and_stable_weight_normalization():
    """The eager operations return reusable scientific results without graph construction."""
    data = BoundCenteredSensitivity.validate(bound_example())
    shift = BoundCenteredSensitivity.reference_shift(data)
    xr.testing.assert_equal(shift, (data.bc_reference * data.H_bc_mass.sum("bc_region")).assign_attrs(units="ppm"))
    weights = xr.DataArray([1e308, 5e307], dims="bc_region", coords={"bc_region": ["a", "b"]})
    normalized = normalize_centering_weights(weights)
    np.testing.assert_allclose(normalized, [2 / 3, 1 / 3])
    assert normalized.attrs["units"] == "1"
