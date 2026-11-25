"""Models with NumPy/SciPy callback Ops compile with the C/VM linker, not Numba's object-mode callback."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from pytensor.compile.mode import get_mode
from pytensor.link.numba.linker import NumbaLinker

from openghg_inversions.inference.sampling import RhimeSampler, sample_prior_predictive
from openghg_inversions.models import registered_model
from openghg_inversions.models.additive_sigma import add_additive_sigma_likelihood
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood

STEPS = ("sample", "compute_log_likelihood", "sample_prior_predictive", "sample_posterior_predictive")


def _observations() -> tuple[xr.DataArray, xr.DataArray]:
    times = np.datetime64("2021-07-01T00") + np.arange(6).astype("timedelta64[h]")
    coords = {
        "nmeasure": np.arange(6),
        "site": ("nmeasure", ["A"] * 3 + ["B"] * 3),
        "time": ("nmeasure", np.r_[times[:3], times[:3]]),
    }
    observations = xr.DataArray([1.0, 1.2, 0.9, 2.0, 2.1, 1.8], dims="nmeasure", coords=coords)
    return observations, observations.copy(data=np.full(6, 0.3))


def _model(kind: str) -> pm.Model:
    observations, error = _observations()
    with registered_model() as model:
        level = pm.Normal("level", 1.5, 1.0)
        mean = level * pm.math.ones(6)
        if kind == "fixed_ou":
            add_fixed_ou_gaussian_likelihood(
                observations=observations,
                observation_error=error,
                mean=mean,
                tau_hours=2.0,
                fixed_site_amplitudes=0.5,
                output_dim="nmeasure",
            )
        else:
            add_additive_sigma_likelihood(
                observations=observations, observation_error=error, mean=mean, output_dim="nmeasure"
            )
    return model


def _is_numba(compile_kwargs) -> bool:
    return isinstance(get_mode((compile_kwargs or {}).get("mode")).linker, NumbaLinker)


@pytest.fixture
def recorded(monkeypatch):
    """Record the compile options PyMC receives at each step, then run it."""
    calls: dict[str, list] = {step: [] for step in (*STEPS, "compute_deterministics")}
    for step in calls:
        original = getattr(pm, step)

        def wrapper(*args, _original=original, _step=step, **kwargs):
            calls[_step].append(kwargs.get("compile_kwargs"))
            return _original(*args, **kwargs)

        monkeypatch.setattr(pm, step, wrapper)
    return calls


def test_staged_prior_predictive_selects_the_c_linker(recorded):
    """The standalone inference operation selects C/VM for its own model."""
    sample_prior_predictive(_model("fixed_ou"), draws=2, random_seed=1104)
    (selected,) = recorded["sample_prior_predictive"]
    assert get_mode(selected["mode"]).linker.use_cloop

    sample_prior_predictive(_model("additive"), draws=2, random_seed=1104)
    assert recorded["sample_prior_predictive"][-1] is None


def _sample(model: pm.Model):
    sampler = RhimeSampler(draws=20, tune=20, chains=1, sample_kwargs={"random_seed": 1104, "cores": 1})
    return sampler.sample(model)


def test_fixed_ou_sampling_never_uses_the_numba_callback(recorded):
    trace = _sample(_model("fixed_ou"))
    for step in STEPS:
        assert recorded[step], step
        assert all(kwargs is not None and not _is_numba(kwargs) for kwargs in recorded[step]), step
    # The log likelihood is still saved, now computed with the same compile options.
    assert "y" in trace["log_likelihood"]
    assert trace["log_likelihood"]["y"].sizes["draw"] == 20


@pytest.mark.parametrize(
    "kind,mode",
    [("fixed_ou", None), ("fixed_ou", "FAST_COMPILE"), ("normal", "FAST_COMPILE")],
)
def test_deferred_evaluation_uses_the_resolved_compiler(monkeypatch, recorded, kind, mode):
    """Use the same selected compiler for likelihoods and deferred deterministics."""
    if kind == "fixed_ou":
        model = _model(kind)
    else:
        with pm.Model() as model:
            level = pm.Normal("level")
            pm.Normal("y", level, observed=1.0)
    with model:
        pm.Deterministic("signal", 2 * model["level"])
    posterior = xr.Dataset(
        {"level": (("chain", "draw"), [[1.0, 1.5, 2.0]])},
        coords={"chain": [0], "draw": [0, 1, 2]},
    )
    monkeypatch.setattr(pm, "sample", lambda **kwargs: xr.DataTree.from_dict({"posterior": posterior}))
    compile_override = {"compile_kwargs": {"mode": mode}} if mode is not None else {}
    trace = RhimeSampler(
        draws=3, burn=1, chains=1, sample_kwargs=compile_override,
        sample_prior_predictive=False, sample_posterior_predictive=False,
    )._sample(model, defer_deterministics=True)

    for step in ("compute_log_likelihood", "compute_deterministics"):
        (compiler,) = recorded[step]
        if mode is None:
            assert get_mode(compiler["mode"]).linker.use_cloop
        else:
            assert compiler["mode"] == mode
    assert trace.log_likelihood.y.sizes["draw"] == 2
    np.testing.assert_allclose(trace.posterior.signal, [[3.0, 4.0]])


def test_ordinary_models_keep_the_default_compile_mode(recorded):
    trace = _sample(_model("additive"))
    assert recorded["sample"] == [None]
    # PyMC computes the log likelihood itself, as before.
    assert recorded["compute_log_likelihood"] == []
    assert recorded["sample_prior_predictive"] == [None]
    assert recorded["sample_posterior_predictive"] == [None]
    assert "y" in trace["log_likelihood"]


@pytest.mark.parametrize("overrides", [{"compile_kwargs": {"mode": "FAST_COMPILE"}}, {"backend": "c"}])
def test_explicit_caller_compile_options_win(recorded, overrides):
    """A caller can select PyMC's mode or its backend shorthand."""
    RhimeSampler(draws=5, tune=5, chains=1, sample_kwargs={"random_seed": 1, "cores": 1, **overrides}).sample(
        _model("fixed_ou")
    )
    (selected,) = recorded["sample"]
    if "compile_kwargs" in overrides:
        assert selected["mode"] == "FAST_COMPILE"
    else:
        assert selected is None


def test_posterior_prediction_uses_its_own_model_for_compile_options(recorded, monkeypatch):
    """An alternate prediction model can require C/VM after ordinary sampling."""
    trace = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset({"level": (("chain", "draw"), [[1.0, 1.5]])}),
        }
    )
    monkeypatch.setattr(pm, "sample", lambda **kwargs: trace)

    RhimeSampler(
        sample_prior_predictive=False,
        posterior_predictive_kwargs={"model": _model("fixed_ou"), "random_seed": 1104},
    ).sample(_model("additive"))

    (selected,) = recorded["sample_posterior_predictive"]
    assert get_mode(selected["mode"]).linker.use_cloop
