"""Sensitivity preparation borrows full matrices and preserves pruning semantics."""

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.models.state_activity import prepare_linear_sensitivity


def _sensitivity() -> xr.DataArray:
    return xr.DataArray(
        np.arange(1.0, 7.0).reshape(2, 3),
        dims=("nmeasure", "region"),
        coords={"nmeasure": [0, 1], "region": ["west", "east", "north"],
                "basis_group": ("region", ["inner", "outer", "inner"])},
        name="H", attrs={"units": "ppm"},
    )


@pytest.mark.parametrize("transposed", [False, True])
def test_all_retained_sensitivity_shares_numeric_storage(transposed):
    sensitivity = _sensitivity()
    if transposed:
        sensitivity = sensitivity.transpose("region", "nmeasure")
    sensitivity.data.flags.writeable = False
    original = sensitivity.copy(deep=True)

    prepared = prepare_linear_sensitivity(sensitivity)

    assert np.shares_memory(prepared.sensitivity.data, sensitivity.data)
    xr.testing.assert_identical(prepared.sensitivity, original.transpose("nmeasure", "region"))
    np.testing.assert_array_equal(prepared.removed, [False, False, False])
    np.testing.assert_array_equal(prepared.retained_indices, [0, 1, 2])
    xr.testing.assert_identical(sensitivity, original)


def test_all_retained_sensitivity_preserves_lazy_payload_and_inspects_once():
    sensitivity = _sensitivity().chunk({"nmeasure": 1, "region": 2})
    payload = sensitivity.data
    inspections = []

    with Callback(start=lambda graph: inspections.append(graph)):
        prepared = prepare_linear_sensitivity(sensitivity)

    assert len(inspections) == 1
    assert isinstance(prepared.sensitivity.data, da.Array)
    assert prepared.sensitivity.data is payload
    assert sensitivity.data is payload
    assert not isinstance(prepared.removed.data, da.Array)


def test_pruning_keeps_full_state_mapping_and_leaves_input_unchanged():
    sensitivity = _sensitivity()
    sensitivity.data[:, 1] = 0.0
    sensitivity.data[:, 2] = [1.0e-12, -1.0e-12]
    original = sensitivity.copy(deep=True)
    sensitivity.data.flags.writeable = False

    prepared = prepare_linear_sensitivity(sensitivity)

    expected = original.isel(region=[0, 2]).rename({
        "region": "region_retained", "basis_group": "basis_group_retained",
    })
    xr.testing.assert_identical(prepared.sensitivity, expected)
    np.testing.assert_array_equal(prepared.removed, [False, True, False])
    np.testing.assert_array_equal(prepared.removed.basis_group, original.basis_group)
    np.testing.assert_array_equal(prepared.retained_indices, [0, 2])
    xr.testing.assert_identical(sensitivity, original)
