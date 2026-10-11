"""Live sampler selection is explicit and leaves caller-owned options intact."""

from importlib import import_module
from inspect import Parameter, signature
from typing import Any, cast

import dask.array as da
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.inference.sampling import RhimeSampler, resolve_sampler


RUNNERS = (
    ("rhime.prepared", "run_rhime_from_prepared_inputs"),
    ("rhime.nested", "run_rhime_nested_from_prepared_inputs"),
    ("rhime.co2.co2_runner", "run_rhime_co2"),
    ("rhime.co2.co2_cached_sigma_runner", "run_rhime_co2_cached_sigma"),
    ("rhime.co2.co2_o2_runner", "run_rhime_co2_o2_from_prepared_inputs"),
    ("rhime.co2.co2_o2_cached_sigma_runner", "run_rhime_co2_o2_cached_sigma_from_prepared_inputs"),
    ("experimental.ramsden2022.model", "run_ramsden_from_prepared_inputs"),
)


@pytest.mark.parametrize("module_name,function_name", RUNNERS)
@pytest.mark.parametrize("invalid", [None, "unknown"])
def test_live_runners_reject_invalid_sampler_before_touching_inputs(
    module_name: str, function_name: str, invalid: Any,
) -> None:
    """Every live selection boundary rejects invalid requests before data access."""
    runner = getattr(import_module("openghg_inversions." + module_name), function_name)
    required = {
        name: object()
        for name, parameter in signature(runner).parameters.items()
        if parameter.default is Parameter.empty
    }
    with pytest.raises(TypeError, match="Omit the optional sampler argument"):
        runner(**required, sampler=invalid)


def test_resolved_sampler_owns_options_and_borrows_runtime_objects() -> None:
    """Defaults are fresh; option shape/type and opaque object identity survive."""
    lazy = da.ones(2, chunks=1)
    labelled = xr.DataArray(lazy, dims="state")
    step = object()
    sampler = RhimeSampler(
        draws=17,
        sample_kwargs={
            "random_seed": np.array([3, 4], dtype=np.int32),
            "initvals": [{"x": (np.array([1.0]), [2.0])}],
            "step": step,
            "borrowed": labelled,
            "lazy": lazy,
        },
        posterior_predictive_kwargs={"compile_kwargs": {"updates": []}},
    )
    resolved = resolve_sampler(sampler)
    assert resolved is not sampler
    assert resolved == sampler
    assert resolved.draws == 17
    kwargs = cast(dict[str, Any], resolved.sample_kwargs)
    original = cast(dict[str, Any], sampler.sample_kwargs)
    assert kwargs["step"] is step
    assert kwargs["borrowed"] is labelled
    assert kwargs["lazy"] is lazy
    assert isinstance(kwargs["initvals"], list)
    assert isinstance(kwargs["initvals"][0]["x"], tuple)
    assert kwargs["random_seed"].dtype == np.int32
    kwargs["random_seed"][0] = 99
    kwargs["initvals"][0]["x"][0][0] = 99
    kwargs["initvals"][0]["x"][1].append(99)
    cast(dict[str, Any], resolved.posterior_predictive_kwargs)["compile_kwargs"]["updates"].append(99)
    np.testing.assert_array_equal(original["random_seed"], [3, 4])
    np.testing.assert_array_equal(original["initvals"][0]["x"][0], [1.0])
    assert original["initvals"][0]["x"][1] == [2.0]
    assert sampler.posterior_predictive_kwargs == {"compile_kwargs": {"updates": []}}
    assert resolved != sampler
    first = resolve_sampler("default")
    second = resolve_sampler("default")
    assert first is not second
    assert first == second == RhimeSampler()
    assert resolve_sampler(sampler, draws=99).draws == 17


@pytest.mark.parametrize(
    "different",
    [
        {"x": [np.array([3, 5])]},
        {"x": [np.array([[3, 4]])]},
        {"x": (np.array([3, 4]),)},
        {"x": [np.array([3, 4]), 0]},
        {"other": [np.array([3, 4])]},
    ],
)
def test_sampler_equality_distinguishes_nested_array_options(different: dict[str, Any]) -> None:
    """Array values, shapes and surrounding containers remain meaningful settings."""
    sampler = RhimeSampler(sample_kwargs={"initvals": {"x": [np.array([3, 4])]}})
    assert sampler == resolve_sampler(sampler)
    assert sampler != RhimeSampler(sample_kwargs={"initvals": different})


@pytest.mark.parametrize("resolve", [False, True])
def test_predictive_idata_none_executes_with_pymc_defaults(
    monkeypatch: pytest.MonkeyPatch, resolve: bool,
) -> None:
    """The PyMC-supported None value survives validation and real prediction."""
    sampler = RhimeSampler(
        sample_prior_predictive=False,
        posterior_predictive_kwargs={"idata_kwargs": None, "random_seed": 12},
    )
    if resolve:
        sampler = resolve_sampler(sampler)
    trace = xr.DataTree.from_dict({
        "posterior": xr.Dataset({"x": (("chain", "draw"), [[1.0, 2.0]])}),
    })
    monkeypatch.setattr(pm, "sample", lambda **kwargs: trace)
    with pm.Model() as model:
        x = pm.Normal("x")
        pm.Normal("y", mu=x, observed=0.0)
    result = sampler.sample(model)
    assert result["posterior_predictive"]["y"].shape == (1, 2)
    assert np.isfinite(result["posterior_predictive"]["y"]).all()
    assert sampler.posterior_predictive_kwargs["idata_kwargs"] is None


@pytest.mark.parametrize("key", ["coords", "dims"])
@pytest.mark.parametrize("container", ["sample_kwargs", "posterior_predictive_kwargs"])
def test_model_coordinate_options_rejected_at_resolution_and_direct_sampling(key: str, container: str) -> None:
    """Neither initial settings nor later sampler mutations override model labels."""
    with pytest.raises(ValueError, match="register coordinates and dimensions"):
        RhimeSampler(**{container: {"idata_kwargs": {key: {}}}})
    sampler = RhimeSampler()
    setattr(sampler, container, {"idata_kwargs": {key: {}}})
    with pytest.raises(ValueError, match="register coordinates and dimensions"):
        resolve_sampler(sampler)
    with pytest.raises(ValueError, match="register coordinates and dimensions"):
        sampler.sample(cast(pm.Model, object()))


def test_direct_sampling_protects_reused_sampler_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend mutations cannot alter a sampler reused for another inference."""
    sampler = RhimeSampler(
        sample_kwargs={"random_seed": [2, 3], "initvals": [{"x": np.array([1.0])}]},
        sample_prior_predictive=False,
        sample_posterior_predictive=True,
        posterior_predictive_kwargs={"compile_kwargs": {"nested": []}},
    )

    def sample(**kwargs: Any) -> xr.DataTree:
        assert kwargs["random_seed"] == [2, 3]
        np.testing.assert_array_equal(kwargs["initvals"][0]["x"], [1.0])
        kwargs["random_seed"].append(4)
        kwargs["initvals"][0]["x"][0] = 9
        return xr.DataTree.from_dict({"posterior": xr.Dataset({"x": (("chain", "draw"), [[1.0]])})})

    def predictive(trace: xr.DataTree, **kwargs: Any) -> xr.DataTree:
        assert kwargs["compile_kwargs"]["nested"] == []
        kwargs["compile_kwargs"]["nested"].append(9)
        return xr.DataTree()

    monkeypatch.setattr(pm, "sample", sample)
    monkeypatch.setattr(pm, "sample_posterior_predictive", predictive)
    model = pm.Model()
    sampler.sample(model)
    sampler.sample(model)
    assert sampler.sample_kwargs["random_seed"] == [2, 3]
    assert sampler.posterior_predictive_kwargs == {"compile_kwargs": {"nested": []}}
