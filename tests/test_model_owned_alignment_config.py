"""Model alignment and error-floor policy stay independent of raw preparation."""

from collections import UserDict
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.boundary_sensitivity import BoundaryAlignment
from openghg_inversions.rhime.params import DataConfig, RhimeConfig, group_params, resolve_model_options
from openghg_inversions.rhime.requests import load_params
from openghg_inversions.rhime.specs import (
    AdditiveSigmaSettings,
    PollutionEventSettings,
    RhimeModelSpec,
)


def request(**options):
    return {
        "species": "ch4", "domain": "EUROPE", "sites": ["TAC"],
        "averaging_period": "1h", "start_date": "2020-01-01", "end_date": "2020-02-01",
        "flux_sources": ["inventory"], "output_name": "configuration", "output_format": "none",
        "mismatch_model": "pollution_event", **options,
    }


@pytest.mark.parametrize("name,value", [
    ("bc_freq", "monthly"), ("min_error", "residual"), ("min_error_options", {"by_site": True}),
])
def test_model_choices_are_rejected_in_data_mapping(name, value):
    data = group_params(request())["data"]
    assert name not in DataConfig.option_names()
    with pytest.raises(ValueError, match=name):
        DataConfig.from_params({**data, name: value})
    with pytest.raises(ValueError, match=f"data.{name}"):
        group_params({"data": {**data, name: value}})


def test_flat_ini_and_saved_model_phase_route_choices_to_model(tmp_path):
    path = tmp_path / "request.ini"
    path.write_text("[RHIME]\nbc_freq = '24h'\nmin_error = 'residual'\nmin_error_options = {'by_site': True}\n")
    expected = {"bc_freq": "24h", "min_error": "residual", "min_error_options": {"by_site": True}}
    for phases in (None, ("model",)):
        groups = load_params(config_file=path, phase_groups=phases)
        assert groups["model"] == expected
        assert groups["data"] == {}


def test_bc_anchor_defaults_to_request_start_and_can_be_replaced():
    data = DataConfig.from_params(group_params(request())["data"])
    model = resolve_model_options({"bc_freq": "24h"}, data_config=data)
    assert model.bc_freq_anchor == data.start_date
    changed = resolve_model_options({"bc_freq": "monthly", "bc_freq_anchor": "2019-12-03"}, data_config=data)
    assert changed.bc_freq == "monthly"
    assert changed.bc_freq_anchor == "2019-12-03"
    with pytest.raises(ValueError, match="BC inputs"):
        resolve_model_options({"use_bc": True}, data_config=replace(data, use_bc=False))


@pytest.mark.parametrize("options,expected_anchor,expected_sensitivity", [
    ({}, "2020-01-01", [[2.0, 0.0], [0.0, 3.0]]),
    ({"bc_freq_anchor": None}, None, [[2.0, 3.0]]),
])
def test_bc_anchor_omission_and_none_have_distinct_retained_periods(
    options, expected_anchor, expected_sensitivity,
):
    config = RhimeConfig.from_params(request(bc_freq="8D", **options), multisector=False)
    assert config.model.bc_freq_anchor == expected_anchor
    trace = xr.DataTree()
    trace.attrs["rhime_priors"] = json.dumps({})
    record = config.model.to_record(trace=trace, data_config=config.data)
    assert record["bc_freq_anchor"] == expected_anchor
    restored = RhimeModelSpec.from_record(record, data_config=config.data, trace=trace)
    assert restored.bc_freq_anchor == expected_anchor

    sensitivity = xr.DataArray(
        [[2.0, 3.0]], dims=("bc_region", "time"),
        coords={"bc_region": ["north"], "time": np.array(["2020-01-08", "2020-01-09"], dtype="datetime64[ns]")},
    )
    for model in (config.model, restored):
        aligned = BoundaryAlignment.prepare(
            sensitivity, frequency=model.bc_freq, anchor_time=model.bc_freq_anchor,
        ).data
        np.testing.assert_array_equal(aligned, expected_sensitivity)


@pytest.mark.parametrize("anchor,expected", [
    ("2020/01/01", "2020-01-01"),
    ("Jan 1 2020", "2020-01-01"),
    ("2020-01-01T06:30:00.123456789", "2020-01-01T06:30:00.123456789"),
    ("2020-01-01T01:00:00+01:00", "2020-01-01"),
])
def test_bc_anchor_is_normalised_for_boundary_alignment_and_record_roundtrip(anchor, expected):
    config = RhimeConfig.from_params(request(bc_freq="8D", bc_freq_anchor=anchor), multisector=False)
    direct = replace(config.model, bc_freq_anchor=anchor)
    assert direct.bc_freq_anchor == config.model.bc_freq_anchor == expected
    trace = xr.DataTree()
    trace.attrs["rhime_priors"] = json.dumps({})
    record = config.model.to_record(trace=trace, data_config=config.data)
    assert record["bc_freq_anchor"] == expected
    restored = RhimeModelSpec.from_record(record, data_config=config.data, trace=trace)
    assert restored.bc_freq_anchor == expected

    sensitivity = xr.DataArray(
        [[2.0, 3.0]], dims=("bc_region", "time"),
        coords={"bc_region": ["north"], "time": np.array(["2020-01-08", "2020-01-10"], dtype="datetime64[ns]")},
    )
    aligned = BoundaryAlignment.prepare(
        sensitivity, frequency=restored.bc_freq, anchor_time=restored.bc_freq_anchor,
    ).data
    np.testing.assert_array_equal(aligned, [[2.0, 0.0], [0.0, 3.0]])


@pytest.mark.parametrize("settings_type", [PollutionEventSettings, AdditiveSigmaSettings])
def test_floor_settings_own_mappings_and_normalise_defaults(settings_type):
    values = UserDict({"TAC": 0.5})
    options = UserDict({"by_site": True})
    settings = settings_type(min_error=values, min_error_options=options)
    values["TAC"] = 42
    options["by_site"] = False
    assert settings.min_error == {"TAC": 0.5}
    assert settings.min_error_options == {"by_site": True}
    assert settings_type(min_error=None).min_error == 0.0
    assert settings_type().min_error_options == {"by_site": False}
    assert settings.required_prepared_inputs == ()


@pytest.mark.parametrize("settings_type", [PollutionEventSettings, AdditiveSigmaSettings])
@pytest.mark.parametrize("options", [
    {"min_error": -1}, {"min_error": "invalid"}, {"min_error": {"TAC": True}},
    {"min_error_options": []}, {"min_error_options": {"by_site": 1}},
    {"min_error_options": {"robust": True}},
])
def test_floor_validation_also_applies_to_direct_settings(settings_type, options):
    with pytest.raises(ValueError, match="min_error"):
        settings_type(**options)


@pytest.mark.parametrize("kind", [None, "fixed_error"])
@pytest.mark.parametrize("options", [{"min_error": 0.0}, {"min_error_options": {"by_site": False}}])
def test_fixed_and_custom_likelihoods_reject_builtin_floor_options(kind, options):
    with pytest.raises(ValueError, match="min_error"):
        RhimeConfig.from_params(request(mismatch_model=kind, **options), multisector=False)


@pytest.mark.parametrize("kind", ["pollution_event", "additive_sigma"])
def test_model_record_roundtrip_preserves_floor_alignment_and_trace_priors(kind):
    config = RhimeConfig.from_params(request(
        mismatch_model=kind, bc_freq="24h", bc_freq_anchor="2019-12-31",
        min_error="residual", min_error_options={"by_site": True},
    ), multisector=False)
    priors = {"x": {"pdf": "normal", "mu": {"data": "prior_mu"}, "sigma": 1.0}}
    trace = xr.DataTree.from_dict({"constant_data": xr.Dataset({"prior_mu": ("state", [1.0])})})
    trace.attrs["rhime_priors"] = json.dumps(priors)
    record = config.model.to_record(trace=trace, data_config=config.data)
    assert record["bc_freq"] == "24h"
    assert record["bc_freq_anchor"] == "2019-12-31"
    assert record["likelihood"]["options"]["min_error"] == "residual"
    assert record["likelihood"]["options"]["min_error_options"] == {"by_site": True}
    restored = RhimeModelSpec.from_record(record, data_config=config.data, trace=trace)
    assert restored.bc_freq == config.model.bc_freq
    assert restored.bc_freq_anchor == config.model.bc_freq_anchor
    assert restored.likelihood.min_error == "residual"
    assert restored.likelihood.min_error_options == {"by_site": True}
    assert restored.sectors[0].x_prior["mu"].identical(trace["constant_data"]["prior_mu"])
    for name, value in (("bc_freq", "invalid"), ("bc_freq_anchor", "NaT")):
        invalid = {**record, name: value}
        with pytest.raises(ValueError, match=name):
            RhimeModelSpec.from_record(invalid, data_config=config.data, trace=trace)
    invalid = deepcopy(record)
    invalid["likelihood"]["options"]["min_error"] = -1
    with pytest.raises(ValueError, match="min_error"):
        RhimeModelSpec.from_record(invalid, data_config=config.data, trace=trace)
    invalid = deepcopy(record)
    invalid.pop("bc_freq_anchor")
    with pytest.raises(ValueError, match="exactly"):
        RhimeModelSpec.from_record(invalid, data_config=config.data, trace=trace)


@pytest.mark.parametrize("frequency", ["invalid", "0h", "-1h", 12])
def test_direct_model_spec_validates_frequency(frequency):
    with pytest.raises(ValueError, match="bc_freq"):
        RhimeModelSpec(species="ch4", domain="EUROPE", sectors=(), bc_freq=frequency)


@pytest.mark.parametrize("anchor", ["not-a-date", "NaT", []])
def test_direct_model_spec_validates_anchor(anchor):
    with pytest.raises(ValueError, match="bc_freq_anchor"):
        RhimeModelSpec(species="ch4", domain="EUROPE", sectors=(), bc_freq_anchor=anchor)
