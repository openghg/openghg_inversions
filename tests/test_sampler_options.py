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


def test_resolved_keywords_restore_nested_list_and_tuple_types_without_aliasing():
    supplied = {"initvals": [{"x": [1.0]}, {"x": [2.0]}], "nested": ([{"values": (3.0, [4.0])}], (), [])}
    options = SamplerOptions(sample_kwargs=supplied)
    supplied["initvals"][0]["x"][0] = 99.0
    with pytest.raises(AttributeError):
        options.sample_kwargs["initvals"].append({})
    # Re-resolving frozen settings must preserve the list marker too.
    resolved = SamplerOptions(sample_kwargs=options.sample_kwargs)
    for runtime in (
        options.create_sampler().sample_kwargs,
        options.as_dict()["sample_kwargs"],
        resolved.create_sampler().sample_kwargs,
    ):
        assert type(runtime["initvals"]) is list
        assert type(runtime["initvals"][0]["x"]) is list
        assert runtime["initvals"][0]["x"][0] == 1.0
        assert type(runtime["nested"]) is tuple
        assert type(runtime["nested"][0]) is list
        assert type(runtime["nested"][0][0]["values"]) is tuple
        assert type(runtime["nested"][0][0]["values"][1]) is list
        assert type(runtime["nested"][1]) is tuple
        assert type(runtime["nested"][2]) is list
        runtime["initvals"][0]["x"][0] = 88.0
        runtime["nested"][0][0]["values"][1].append(5.0)
    assert options.create_sampler().sample_kwargs["initvals"][0]["x"][0] == 1.0
    assert options.create_sampler().sample_kwargs["nested"][0][0]["values"][1] == [4.0]


@pytest.mark.parametrize("resolved", [False, True])
@pytest.mark.parametrize("name", ["coords", "dims"])
def test_sampler_rejects_conversion_coordinate_overrides_before_pymc(monkeypatch, resolved, name):
    import pymc as pm
    from openghg_inversions.rhime.sampling import RhimeSampler

    kwargs = {"idata_kwargs": {name: {}}}
    if resolved:
        sampler = SamplerOptions(sample_kwargs=kwargs).create_sampler()
    else:
        sampler = RhimeSampler()
        # Direct runtime settings can change after creation.
        sampler.sample_kwargs = kwargs

    def forbidden(**kwargs):
        pytest.fail("Coordinate conversion overrides reached pm.sample")

    monkeypatch.setattr("openghg_inversions.rhime.sampling.pm.sample", forbidden)
    with pytest.raises(ValueError, match="register labelled coordinates and dimensions"):
        sampler.sample(pm.Model())


@pytest.mark.parametrize("log_likelihood", [False, True])
def test_sampler_forwards_resolved_chain_lists_and_allowed_conversion_options(monkeypatch, log_likelihood):
    import pymc as pm
    import xarray as xr

    seen = {}

    def sample(**kwargs):
        seen.update(kwargs)
        return xr.DataTree.from_dict({"posterior": xr.Dataset({"x": (("chain", "draw"), [[1.0], [2.0]])})})

    monkeypatch.setattr("openghg_inversions.rhime.sampling.pm.sample", sample)
    options = SamplerOptions(
        chains=2,
        draws=1,
        tune=0,
        sample_kwargs={
            "initvals": [{"x": 1.0}, {"x": 2.0}],
            "compute_convergence_checks": False,
            "idata_kwargs": {"log_likelihood": log_likelihood, "include_transformed": True},
        },
        sample_prior_predictive=False,
        sample_posterior_predictive=False,
    )
    options.create_sampler().sample(pm.Model())
    assert type(seen["initvals"]) is list
    assert seen["initvals"] == [{"x": 1.0}, {"x": 2.0}]
    assert seen["compute_convergence_checks"] is False
    assert seen["idata_kwargs"] == {"log_likelihood": log_likelihood, "include_transformed": True}
