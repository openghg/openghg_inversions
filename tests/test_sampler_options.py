"""Resolved choices cannot leak mutable sampler state between invocations."""

import numpy as np
import pytest
from openghg_inversions.rhime.sampling import SamplerOptions


def test_nested_keywords_and_invocation_overrides_are_isolated():
    supplied = {"idata_kwargs": {"log_likelihood": True}, "nuts": {"target_accept": 0.9}}
    options = SamplerOptions(sample_kwargs=supplied)
    supplied["idata_kwargs"]["log_likelihood"] = False
    with pytest.raises(TypeError):
        options.sample_kwargs["idata_kwargs"]["log_likelihood"] = False
    first = options.create_sampler(draws=12)
    first.sample_kwargs["idata_kwargs"]["log_likelihood"] = False
    first.sample_kwargs["nuts"]["target_accept"] = 0.1
    second = options.create_sampler()
    assert second.draws == 1000
    assert second.sample_kwargs["idata_kwargs"]["log_likelihood"] is True
    assert second.sample_kwargs["nuts"]["target_accept"] == 0.9


def test_nested_invocation_override_lists_are_independent():
    supplied = {"custom": [{"choice": 1}]}
    options = SamplerOptions()
    first = options.create_sampler(sample_kwargs=supplied)
    first.sample_kwargs["custom"][0]["choice"] = 2
    assert supplied["custom"][0]["choice"] == 1
    assert options.create_sampler(sample_kwargs=supplied).sample_kwargs["custom"][0]["choice"] == 1


def test_numpy_initial_values_are_isolated_from_sources_invocations_and_overrides():
    supplied = np.array([1.0])
    options = SamplerOptions(sample_kwargs={"initvals": {"x": supplied}})
    supplied[0] = 2.0
    assert options.sample_kwargs["initvals"]["x"][0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        options.sample_kwargs["initvals"]["x"][0] = 3.0
    first = options.create_sampler()
    first.sample_kwargs["initvals"]["x"][0] = 4.0
    assert options.create_sampler().sample_kwargs["initvals"]["x"][0] == 1.0
    override = np.array([5.0])
    changed = options.create_sampler(sample_kwargs={"initvals": {"x": override}})
    changed.sample_kwargs["initvals"]["x"][0] = 6.0
    assert override[0] == 5.0
    override[0] = 7.0
    assert changed.sample_kwargs["initvals"]["x"][0] == 6.0
    assert options.create_sampler().sample_kwargs["initvals"]["x"][0] == 1.0
