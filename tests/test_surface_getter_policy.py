"""Explicit surface acquisition must not disguise an older inventory as current."""

from unittest.mock import Mock
from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr
from openghg.types import SearchError

from openghg_inversions.inversion_data import getters


@pytest.mark.parametrize("empty", [False, True])
def test_explicit_surface_getter_does_not_relabel_older_flux(monkeypatch, empty):
    monkeypatch.setattr(getters, "adjust_flux_start_date", lambda *args: pd.Timestamp("2024-01-01"))
    retrieve = Mock()
    if empty:
        retrieve.return_value = Mock(data=xr.Dataset(coords={"time": pd.DatetimeIndex([])}))
    else:
        retrieve.side_effect = SearchError("No matching inventory")
    monkeypatch.setattr(getters, "get_flux", retrieve)

    with pytest.raises(SearchError, match="older-inventory fallback is disabled"):
        getters.get_flux_data(
            ["biosphere"], "co2", "EUROPE", "2024-01-01", "2024-02-01", allow_older_flux=False
        )

    assert retrieve.call_count == 1
    assert retrieve.call_args.kwargs["start_date"] == pd.Timestamp("2024-01-01")


@pytest.mark.parametrize("family", ["standard", "multisector"])
def test_unimplemented_model_binding_rejects_centered_operands(family):
    from openghg_inversions.rhime.standard import build_standard_model
    from openghg_inversions.rhime.multisector import build_multisector_model

    builder = build_standard_model if family == "standard" else build_multisector_model
    inputs = xr.Dataset({"bc_reference": 209300.0})
    with pytest.raises(ValueError, match="CO2-family pilot model binding"):
        builder(inputs, SimpleNamespace(use_bc=True), priors={})


@pytest.mark.parametrize("split", [False, True])
def test_surface_native_reporting_converts_units_without_source_time_policy(monkeypatch, split):
    import numpy as np
    from openghg_inversions import flux_sanitization
    from openghg_inversions.native_flux import resolve_runtime_flux

    clock = Mock()
    clock.now.return_value = pd.Timestamp("2020-01-01", tz="UTC")
    monkeypatch.setattr(flux_sanitization, "datetime", clock)

    times = pd.date_range("2020-01-01", periods=3, freq="h")
    first = xr.Dataset({"flux": ("time", [1., 2., 3.], {"units": "mol / m2 / s"})}, coords={"time": times})
    other = xr.Dataset({"flux": ("time", [-1.e6, -2.e6, -3.e6], {"units": "umol / m2 / s"})}, coords={"time": times})
    result = resolve_runtime_flux({"a": first, "b": other}, split_by_sectors=split, explicit_surface=True)
    reversed_result = resolve_runtime_flux({"b": other, "a": first}, split_by_sectors=split, explicit_surface=True)
    xr.testing.assert_identical(result, reversed_result)
    np.testing.assert_allclose(result.sum("source") if split else result, 0, atol=1e-15)
    with pytest.raises(ValueError, match="identical source axes"):
        resolve_runtime_flux({"a": first, "b": other.isel(time=[0])}, explicit_surface=True)
