"""Portable current configuration has no executable scientific side effects."""

from dataclasses import asdict, fields, replace
import json

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from openghg_inversions._record_schema import JsonFloat, portable_json
from openghg_inversions.inference.sampling import RhimeSampler, resolve_sampler
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.specs import PollutionEventSettings
from tests.test_rhime_configuration import _request


@pytest.mark.parametrize("scalar", [np.int32(7), np.int64(7), np.uint64(7)])
def test_eager_numpy_integer_runtime_options(scalar):
    sampler = RhimeSampler(draws=scalar, burn=scalar, tune=scalar, chains=scalar,
                           sample_prior_predictive=scalar)
    assert sampler.to_dict()["draws"] == 7
    assert type(sampler.draws) is int
    config = RhimeConfig.from_params(_request(nbasis=scalar, max_level=scalar), multisector=False)
    assert config.data.nbasis == 7
    assert config.data.site_options.max_level == (7, 7)
    assert replace(config.data, nbasis=scalar).nbasis == 7
    assert replace(config.data.site_options, max_level=(scalar, None)).max_level == (7, None)


@pytest.mark.parametrize("scalar", [np.float32(.5), np.float64(.5), np.int64(1)])
def test_eager_numpy_float_runtime_options(scalar):
    assert type(PollutionEventSettings(power=scalar).power) is float
    config = RhimeConfig.from_params(_request(power=scalar, min_error=scalar), multisector=False)
    assert config.model.likelihood.power == scalar.item()
    assert config.data.min_error == scalar.item()
    assert replace(config.data, min_error={"TAC": scalar}).min_error == {"TAC": scalar.item()}
    sampler = RhimeSampler(sample_kwargs={"target_accept": scalar})
    assert type(sampler.to_dict()["sample_kwargs"]["target_accept"]) in (float, int)
    # Runtime support does not weaken the persisted number primitive.
    with pytest.raises(ValueError):
        TypeAdapter(JsonFloat).validate_python(scalar)


@pytest.mark.parametrize("value", [np.array(2), np.array([2]), np.bool_(True), True, "2"])
def test_integer_runtime_options_reject_array_or_ambiguous_values(value):
    with pytest.raises(ValidationError):
        RhimeSampler(draws=value)


def test_runtime_scalar_validation_never_calls_opaque_conversion():
    converted = []

    class OpaqueNumber:
        def __int__(self):
            converted.append("int")
            return 2

        def __float__(self):
            converted.append("float")
            return 2.

    opaque = OpaqueNumber()
    config = RhimeConfig.from_params(_request(), multisector=False)
    for construct in (lambda: RhimeSampler(draws=opaque),
                      lambda: PollutionEventSettings(power=opaque),
                      lambda: replace(config.data, min_error=opaque)):
        with pytest.raises(ValueError):
            construct()
    assert converted == []


@pytest.mark.parametrize("backend", ["pymc", "nutpie", "numpyro", "blackjax"])
@pytest.mark.parametrize("selection", [False, True, ["concentration", "flux_scale"]])
def test_sampler_and_full_configuration_roundtrip_without_execution(tmp_path, monkeypatch, backend, selection):
    import openghg_inversions.inference.sampling as sampling

    def unexpected(*args, **kwargs):
        pytest.fail("configuration must not sample")

    monkeypatch.setattr(sampling.pm, "sample", unexpected)
    sampler = RhimeSampler(
        draws=23, tune=11, burn=2, chains=2, nuts_sampler=backend,
        sample_kwargs={"target_accept": .95, "random_seed": [10, 11]},
        sample_prior_predictive=5, sample_posterior_predictive=selection,
        posterior_predictive_kwargs={"progressbar": False},
    )
    config = RhimeConfig.from_params(_request(
        basis_path=tmp_path / "unopened.nc", inlet=slice("10m", "100m"), **sampler.to_dict(),
    ), multisector=False)
    assert config.sampler == sampler
    adapter = TypeAdapter(RhimeConfig)
    wire = config.to_dict()
    schema = RhimeConfig.json_schema()
    assert schema["$defs"]["DataConfig"]["properties"]["nbasis"]["exclusiveMinimum"] == 0
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(wire)
    assert RhimeSampler.from_dict(sampler.to_dict()) == sampler
    assert RhimeConfig.from_dict(wire) == config
    assert adapter.validate_json(adapter.dump_json(config)) == config
    assert resolve_sampler(RhimeConfig.from_dict(wire).sampler) == sampler
    assert isinstance(asdict(config)["sampler"], dict)
    assert wire["data"]["site_options"]["inlet"][0] == {
        "type": "slice", "start": "10m", "stop": "100m", "step": None,
    }
    assert not (tmp_path / "unopened.nc").exists()


def test_none_or_omitted_sampler_is_portable_but_not_executable():
    config = RhimeConfig.from_params(_request(), multisector=False)
    for no_sampler in (replace(config, sampler=None),
                       RhimeConfig(data=config.data, model=config.model, output=config.output)):
        wire = no_sampler.to_dict()
        assert wire["sampler"] is None
        del wire["sampler"]
        restored = RhimeConfig.from_dict(wire)
        assert restored.sampler is None
        with pytest.raises(TypeError, match="sampler"):
            resolve_sampler(restored.sampler)


def test_sampler_schema_declares_all_settings_and_predictive_types():
    schema = RhimeSampler.json_schema()
    assert set(schema["properties"]) == {field.name for field in fields(RhimeSampler)}
    for selection in (False, ["concentration"]):
        Draft202012Validator(schema).validate(RhimeSampler(sample_posterior_predictive=selection).to_dict())
    with pytest.raises(ValidationError, match="sample_posterior_predictive"):
        RhimeSampler(sample_posterior_predictive=[1])
    with pytest.raises(ValidationError, match="unknown"):
        RhimeSampler(unknown=1)


@pytest.mark.parametrize("bad", [float("inf"), np.array([2]), (1, 2), object()])
def test_sampler_export_rejects_unportable_options_without_losing_evidence(bad):
    sampler = RhimeSampler(sample_kwargs={"custom": bad})
    with pytest.raises(PydanticSerializationError, match="sampler.sample_kwargs"):
        sampler.to_dict()


def test_portable_configuration_checks_raw_keys_before_json_projection():
    collision = {1: "integer evidence", "1": "string evidence"}
    sampler = RhimeSampler(sample_kwargs={})
    sampler.sample_kwargs["collision"] = collision
    with pytest.raises(PydanticSerializationError, match="keys must be strings"):
        sampler.to_dict()
    config = RhimeConfig.from_params(_request(min_error={"TAC": 1.}), multisector=False)
    config.data.min_error[1] = 2.
    with pytest.raises(PydanticSerializationError, match="keys must be strings"):
        config.to_dict()
    wire = RhimeConfig.from_params(_request(), multisector=False).to_dict()
    wire["sampler"]["sample_kwargs"] = {"collision": collision}
    with pytest.raises(ValueError, match="keys must be strings"):
        RhimeConfig.from_dict(wire)


def test_mutated_sampler_scalars_are_revalidated_at_export():
    sampler = RhimeSampler()
    sampler.draws = True
    with pytest.raises(PydanticSerializationError, match="draws"):
        sampler.to_dict()


def test_portable_export_rejects_lazy_prior_sampler_and_policy_without_execution():
    executed = []

    @dask.delayed
    def payload():
        executed.append("executed")
        return np.array(.5)

    lazy = xr.DataArray(da.from_delayed(payload(), shape=(), dtype=float))
    config = RhimeConfig.from_params(_request(), multisector=False)
    policy = StateActivity(active=lazy)
    bad_prior = replace(config.model.sectors[0], x_prior={"pdf": "normal", "mu": lazy, "sigma": 1.})
    configurations = (
        replace(config, sampler=RhimeSampler(sample_kwargs={"initvals": {"x": lazy}})),
        replace(config, model=replace(config.model, sectors=(bad_prior,))),
        replace(config, model=replace(config.model, state_activity=policy)),
    )
    for value in configurations:
        with pytest.raises(PydanticSerializationError):
            value.to_dict()
    assert executed == []


def test_portable_json_rejects_cycles_and_strict_wire_numpy():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic"):
        portable_json(cyclic)
    with pytest.raises(ValueError, match="portable JSON"):
        RhimeSampler.from_dict({"draws": np.int64(3)})
    with pytest.raises(ValueError, match="finite"):
        RhimeConfig.from_dict({"sampler": {"sample_kwargs": {"bad": float("nan")}}})
    assert json.loads(json.dumps(portable_json({"a": [None, True, 1, .5]}))) == {"a": [None, True, 1, .5]}


def test_negative_prior_draws_fail_at_constructor_read_and_schema():
    for construct in (lambda: RhimeSampler(sample_prior_predictive=-1),
                      lambda: RhimeSampler.from_dict({"sample_prior_predictive": -1})):
        with pytest.raises(ValidationError, match="sample_prior_predictive"):
            construct()
    schema = RhimeSampler.json_schema()
    wire = RhimeSampler().to_dict()
    wire["sample_prior_predictive"] = -1
    assert list(Draft202012Validator(schema).iter_errors(wire))


@pytest.mark.parametrize("multisector", [False, True])
def test_live_configured_runners_reject_no_sampler_before_preparation(multisector):
    from openghg_inversions.rhime import run_multisector, run_standard

    sources = ["inventory", "ocean"] if multisector else ["inventory"]
    config = RhimeConfig.from_params(_request(flux_sources=sources), multisector=multisector)
    runner = run_multisector if multisector else run_standard
    with pytest.raises(TypeError, match="sampler"):
        runner(config=replace(config, sampler=None), prepared=object())


def test_minimum_error_options_are_resolved_at_their_constructor_owner():
    config = RhimeConfig.from_params(_request(), multisector=False)
    assert replace(config.data, min_error_options={}).min_error_options == {"by_site": False}
    wire = config.to_dict()
    wire["data"]["min_error_options"] = {}
    assert RhimeConfig.from_dict(wire).data.min_error_options == {"by_site": False}
    for bad in ({"unexpected": True}, {"by_site": "false"}):
        with pytest.raises(ValidationError, match="min_error_options"):
            replace(config.data, min_error_options=bad)
        wire["data"]["min_error_options"] = bad
        with pytest.raises(ValidationError, match="min_error_options"):
            RhimeConfig.from_dict(wire)
