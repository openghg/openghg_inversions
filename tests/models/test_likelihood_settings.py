"""Component-owned settings validate scalars without executing numerical priors."""

from dataclasses import asdict, replace
import json

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pydantic import TypeAdapter, ValidationError

from openghg_inversions.models._likelihood_settings import LikelihoodSettings
from openghg_inversions.models.additive_sigma import AdditiveSigmaRecord, AdditiveSigmaSettings
from openghg_inversions.models.fixed_error import FixedErrorRecord, FixedErrorSettings
from openghg_inversions.models.pollution_event import (
    PollutionEventOptions, PollutionEventRecord, PollutionEventSettings,
)
from openghg_inversions.models.priors import PriorRecord, trace_prior_records


@pytest.mark.parametrize("settings,record_type", [
    (FixedErrorSettings(), FixedErrorRecord),
    (AdditiveSigmaSettings(), AdditiveSigmaRecord),
    (PollutionEventSettings(), PollutionEventRecord),
])
def test_components_own_settings_and_bidirectional_records(settings, record_type):
    record = settings.to_record(priors={})
    assert isinstance(record, record_type)
    assert record.to_runtime() == settings
    assert type(settings).__module__ == record_type.__module__
    assert replace(settings) == settings
    assert asdict(settings)["kind"] == settings.kind
    with pytest.raises(ValidationError):
        type(settings)(unknown_option=True)


def test_discriminated_factory_resolves_defaults_and_consumes_only_its_options():
    remaining = {"sigma_per_site": False, "unrelated": object()}
    settings = LikelihoodSettings.from_params(remaining, mismatch_model="additive_sigma", start_date="2026-01-01")
    assert settings == AdditiveSigmaSettings(
        sigma_per_site=False, sigma_prior={"pdf": "halfnormal", "sigma": 1.0}, sigma_freq_anchor="2026-01-01",
    )
    assert set(remaining) == {"unrelated"}
    assert "kind" not in LikelihoodSettings.option_names()
    for options in ({"sigma_per_site": "false"}, {"power": 2.0}):
        with pytest.raises(ValidationError) as error:
            LikelihoodSettings.from_params(options, mismatch_model="additive_sigma", start_date="2026-01-01")
        assert error.value.errors()[0]["loc"][0] == "additive_sigma"
    anchor = np.datetime64("2026-01-01T00:00:00.123456789", "ns")
    assert AdditiveSigmaSettings(sigma_freq_anchor=anchor).sigma_freq_anchor == str(anchor)


@pytest.mark.parametrize("kind,settings_type", [
    ("additive_sigma", AdditiveSigmaSettings),
    ("pollution_event", PollutionEventSettings),
])
def test_request_defaults_are_explicit_and_do_not_depend_on_validation_context(kind, settings_type):
    direct = settings_type()
    assert TypeAdapter(settings_type).validate_python({}, context={"unrelated": True}) == direct
    assert direct.sigma_prior is None and direct.sigma_freq_anchor is None

    omitted = LikelihoodSettings.from_params({}, mismatch_model=kind, start_date="2026-01-01")
    explicit_null = LikelihoodSettings.from_params(
        {"sigma_prior": None, "sigma_freq_anchor": None}, mismatch_model=kind, start_date="2026-01-01",
    )
    assert omitted.sigma_freq_anchor == "2026-01-01"
    assert explicit_null.sigma_freq_anchor is None
    assert omitted.sigma_prior == explicit_null.sigma_prior == settings_type.DEFAULT_SIGMA_PRIOR
    omitted.sigma_prior["pdf"] = "normal"
    assert explicit_null.sigma_prior == settings_type.DEFAULT_SIGMA_PRIOR


def test_request_discriminator_validation_does_not_execute_a_lazy_scalar():
    @dask.delayed
    def forbidden():
        raise AssertionError("Validating a likelihood tag computed a lazy scalar")

    tag = da.from_delayed(forbidden(), shape=(), dtype=object)
    with pytest.raises(ValidationError):
        LikelihoodSettings.from_params({}, mismatch_model=tag, start_date="2026-01-01")


def test_prior_ownership_capture_and_numeric_rejection_never_compute():
    calls = []

    @dask.delayed
    def source():
        calls.append(1)
        return np.array([1.0])

    lazy = xr.DataArray(da.from_delayed(source(), shape=(1,), dtype=float), dims="region")
    configured = {"pdf": "normal", "mu": lazy, "nested": {"labels": ["a"]}}
    settings = PollutionEventSettings(sigma_prior=configured, power=configured)
    assert settings.sigma_prior["mu"] is lazy and settings.power["mu"] is lazy
    configured["nested"]["labels"].append("b")
    assert settings.sigma_prior["nested"]["labels"] == ["a"]
    assert settings.power["nested"]["labels"] == ["a"]
    absent = settings.to_record(priors={})
    assert absent.options.sigma_prior is None and absent.options.power is None
    record = settings.to_record(priors={
        "sigma": {"pdf": "normal", "mu": {"data": "sigma_prior_mu"}},
        "power": {"pdf": "normal", "mu": {"data": "power_prior_mu"}},
    })
    trace = xr.DataTree.from_dict({"constant_data": xr.Dataset({"sigma_prior_mu": lazy, "power_prior_mu": lazy})})
    restored = record.to_runtime(trace=trace)
    assert restored.sigma_prior["mu"].data is trace["constant_data"]["sigma_prior_mu"].data
    assert restored.power["mu"].data is trace["constant_data"]["power_prior_mu"].data
    for validate in (
        lambda: PriorRecord.model_validate({"mu": lazy}),
        lambda: PollutionEventOptions.model_validate(record.options.model_dump() | {"power": lazy}),
        lambda: PollutionEventSettings(power=lazy),
        lambda: PollutionEventSettings(power=True),
    ):
        with pytest.raises(ValidationError):
            validate()
    assert calls == []


@pytest.mark.parametrize("metadata", [
    [], {"x": 1}, {"x": {"mu": {"data": "parameter", "unexpected": True}}}, {"x": {"mu": float("nan")}},
])
def test_trace_prior_metadata_schema_rejects_malformed_surrounding_json(metadata):
    trace = xr.DataTree(xr.Dataset(attrs={"rhime_priors": json.dumps(metadata)}))
    with pytest.raises(ValidationError):
        trace_prior_records(trace)
