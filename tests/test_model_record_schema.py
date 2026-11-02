"""Small-record validation must leave scientific arrays with their owner."""

from dataclasses import replace

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from openghg_inversions.rhime._model_records import ModelRecord
from openghg_inversions.rhime.specs import (
    AdditiveSigmaSettings, FixedErrorSettings, PollutionEventSettings, RhimeModelSpec, SectorSpec,
)

jsonschema = pytest.importorskip("jsonschema")


@pytest.mark.parametrize("likelihood", [
    None, FixedErrorSettings(), AdditiveSigmaSettings(), PollutionEventSettings(),
])
def test_nested_model_schema_round_trip(likelihood):
    model = RhimeModelSpec("ch4", "EUROPE", (SectorSpec("total", "inventory", {}, "total"),),
                           likelihood=likelihood, use_bc=False)
    record = model.to_record(priors={"x": {"pdf": "normal", "mu": 1., "sigma": {"data": "x_prior_sigma"}}})
    jsonschema.Draft202012Validator(ModelRecord.model_json_schema()).validate(record)
    trace = xr.DataTree.from_dict({"constant_data": xr.Dataset({"x_prior_sigma": ("region", [0.1])})})
    decoded = RhimeModelSpec.from_record(record, trace=trace)
    assert decoded.likelihood == likelihood
    xr.testing.assert_identical(decoded.sectors[0].x_prior["sigma"], trace["constant_data"]["x_prior_sigma"])
    assert decoded.sectors[0].x_prior["sigma"].data is trace["constant_data"]["x_prior_sigma"].data
    with pytest.raises(ValueError, match="Missing registered prior data 'x_prior_sigma'"):
        RhimeModelSpec.from_record(record)
    assert record["sectors"][0]["x_prior"]["sigma"] == {"data": "x_prior_sigma"}


@pytest.mark.parametrize("change,location", [
    ({"extra": 1}, ("extra",)),
    ({"use_bc": "false"}, ("use_bc",)),
    ({"likelihood": {"kind": "unknown", "options": {}}}, ("likelihood",)),
    ({"likelihood": {"kind": "fixed_error", "options": {"sigma_prior": {}}}},
     ("likelihood", "fixed_error", "options", "sigma_prior")),
    ({"offset_args": {"nested": [float("nan")]}}, ("offset_args",)),
])
def test_nested_errors_include_field_paths(change, location):
    model = RhimeModelSpec("ch4", "EUROPE", (SectorSpec("total", "inventory", {}, "total"),))
    with pytest.raises(ValidationError) as error:
        RhimeModelSpec.from_record(model.to_record() | change)
    assert any(item["loc"][:len(location)] == location for item in error.value.errors())


def test_temporal_projection_and_unused_lazy_priors_do_not_compute():
    calls = []

    @dask.delayed
    def payload():
        calls.append(1)
        return np.array([1.])

    lazy = xr.DataArray(da.from_delayed(payload(), shape=(1,), dtype=float), dims="region")
    instant = np.datetime64("2026-10-14T12:00:00.123456789", "ns")
    model = RhimeModelSpec("ch4", "EUROPE", (SectorSpec("total", "inventory", {}, "total"),),
                           bc_prior={"mu": lazy}, likelihood=AdditiveSigmaSettings(sigma_freq_anchor=instant))
    record = model.to_record()
    assert record["bc_prior"] is None
    assert record["likelihood"]["options"]["sigma_freq_anchor"] == str(instant)
    # An accidental array in a declared JSON field must fail without executing it.
    with pytest.raises(ValidationError):
        replace(model, offset_args={"unexpected": lazy}).to_record()
    assert calls == []
