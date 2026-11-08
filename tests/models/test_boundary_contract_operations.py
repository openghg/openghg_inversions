"""Boundary domain operations preserve labels and own eager checks explicitly."""

import dask
import numpy as np
import pytensor
import pytest
import xarray as xr

from openghg_inversions.models.boundary import BoundAdditiveTransport, BoundCenteredTransport, RawCenteredTransport
from openghg_inversions.rhime.co2.co2_o2_preparation import _validate_boundary_corrections
from test_boundary_contracts import _forbid_compute, _observations, _raw_on_nmeasure, bound_example


def test_replacing_weights_keeps_the_supplied_calendar_until_binding():
    """An old Dataset calendar must not truncate a replacement's extra period."""
    raw = RawCenteredTransport.validate(_raw_on_nmeasure(lazy=True))
    original = raw.copy(deep=False)
    old_weights = raw.bc_centering_weights
    extra = old_weights.assign_coords(bc_period=old_weights.bc_period + np.timedelta64(1, "D"))
    weights = xr.concat([old_weights, extra], dim="bc_period")
    with dask.config.set(scheduler=_forbid_compute), pytensor.config.change_flags(floatX="float64"):
        replaced = RawCenteredTransport.with_centering_weights(raw, weights)
        xr.testing.assert_identical(replaced.bc_centering_weights, weights)
        bound = BoundCenteredTransport.bind(replaced, _observations())
        with pytest.raises(ValueError, match="align"):
            BoundAdditiveTransport.bind_weights(bound, replaced.bc_centering_weights)
    xr.testing.assert_identical(raw, original)


@pytest.mark.parametrize("coordinate", [False, True])
def test_replacing_weights_rejects_axes_shared_with_other_data(coordinate):
    """Replacement must not silently delete extra variables or auxiliary coordinates."""
    raw = _raw_on_nmeasure()
    other = xr.ones_like(raw.bc_centering_weights)
    raw = raw.assign_coords(other=other) if coordinate else raw.assign(other=other)
    with pytest.raises(ValueError, match="shared.*other"):
        RawCenteredTransport.with_centering_weights(RawCenteredTransport.validate(raw), raw.bc_centering_weights)


def test_weight_preparation_is_structural_and_normalization_owns_values():
    """Preparation neither computes lazy weights nor repeats the later value scan."""
    raw = RawCenteredTransport.validate(_raw_on_nmeasure())
    invalid = xr.full_like(raw.bc_centering_weights, -1.0)
    for weights in (invalid, invalid.chunk()):
        with dask.config.set(scheduler=_forbid_compute):
            _validate_boundary_corrections({"co2": weights}, {"co2": raw})
    with pytest.raises(ValueError, match="finite and strictly positive"):
        BoundAdditiveTransport.normalized_weights(invalid)


def test_numerical_operations_do_not_materialize_individual_variables():
    """Explicit eager operations reject lazy inputs before touching .values."""
    with dask.config.set(scheduler=_forbid_compute):
        with pytest.raises(ValueError, match="Materialize boundary data jointly"):
            BoundCenteredTransport.reference_shift(BoundCenteredTransport.validate(bound_example(lazy=True)))
        with pytest.raises(ValueError, match="Materialize bc_centering_weights"):
            BoundAdditiveTransport.normalized_weights(bound_example(lazy=True, weights=True).bc_centering_weights)


def test_reference_shift_rejects_zero_mass_after_valid_row_selection():
    """A nonzero raw mass promise cannot survive arbitrary observation subsetting."""
    raw = _raw_on_nmeasure(weights=False)
    raw["H_bc_mass"] = raw.H_bc_mass.where(raw.nmeasure > 1, 0.0)
    with pytensor.config.change_flags(floatX="float64"):
        bound = BoundCenteredTransport.bind(RawCenteredTransport.validate(raw), _observations().isel(nmeasure=[0, 1]))
    assert BoundCenteredTransport.validate(bound) is bound
    with pytest.raises(ValueError, match="not all zero"):
        BoundCenteredTransport.reference_shift(bound)


def test_numerical_results_keep_units_and_stable_weight_normalization():
    """The eager operations return reusable scientific results without graph construction."""
    data = BoundCenteredTransport.validate(bound_example())
    shift = BoundCenteredTransport.reference_shift(data)
    xr.testing.assert_equal(shift, (data.bc_reference * data.H_bc_mass.sum("bc_region")).assign_attrs(units="ppm"))
    weights = xr.DataArray([1e308, 5e307], dims="bc_region", coords={"bc_region": ["a", "b"]})
    normalized = BoundAdditiveTransport.normalized_weights(weights)
    np.testing.assert_allclose(normalized, [2 / 3, 1 / 3])
    assert normalized.attrs["units"] == "1"


def test_linked_runner_does_not_hide_a_supplied_weight_calendar_mismatch():
    """The normal linked handoff rejects extra periods even with embedded weights."""
    from dataclasses import replace

    from test_co2_o2_additive_boundary import _additive_inputs, _linked_choices
    from test_co2_o2_fixed_ou import _prepared
    from openghg_inversions.rhime.co2.co2_o2_runner import _bind_co2_o2_boundaries

    prepared = _prepared()
    with pytensor.config.change_flags(floatX="float64"):
        arguments = _additive_inputs(prepared)
        original_weights = arguments["bc_centering_weights"]["co2"]
        extra = original_weights.assign_coords(bc_period=original_weights.bc_period + np.timedelta64(1, "D"))
        replacement = xr.concat([original_weights, extra], dim="bc_period")
        raw = {
            channel: RawCenteredTransport.validate(data)
            for channel, data in arguments["boundary_operands"].items()
        }
        raw["co2"] = RawCenteredTransport.with_centering_weights(raw["co2"], original_weights)
        weights = {**arguments["bc_centering_weights"], "co2": replacement}
        _validate_boundary_corrections(weights, raw)
        prepared = replace(prepared, boundary_operands=raw, bc_centering_weights=weights)
        with pytest.raises(ValueError, match="align"):
            _bind_co2_o2_boundaries(prepared, _linked_choices(arguments))
