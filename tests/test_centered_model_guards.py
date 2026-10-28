"""Model runners enforce the supported centered-boundary integration scope."""

from types import SimpleNamespace

import pytest
import xarray as xr


@pytest.mark.parametrize("family", ["standard", "multisector"])
def test_unimplemented_model_binding_rejects_centered_operands(family):
    from openghg_inversions.rhime.standard import build_standard_model
    from openghg_inversions.rhime.multisector import build_multisector_model

    builder = build_standard_model if family == "standard" else build_multisector_model
    inputs = xr.Dataset({"bc_reference": 209300.0})
    with pytest.raises(ValueError, match="CO2-family pilot model binding"):
        builder(inputs, SimpleNamespace(use_bc=True), priors={})
