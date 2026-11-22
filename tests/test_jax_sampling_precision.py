"""JAX NUTS backends sample at the precision of the model they are given."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

import numpy as np
import pymc as pm
import pytensor
import pytest

from openghg_inversions.inference.sampling import RhimeSampler, jax_sampling_precision

pytest.importorskip("numpyro")
jax = pytest.importorskip("jax")


def _tiny_model() -> pm.Model:
    """One location parameter whose posterior SD sits far below float32 resolution at -700."""
    observed = np.full(4, -700.0) + np.array([1e-4, -1e-4, 2e-4, -2e-4])
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=-700.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1e-3, observed=observed)
    return model


def _sample(model: pm.Model) -> object:
    sampler = RhimeSampler(
        draws=20,
        tune=20,
        chains=1,
        nuts_sampler="numpyro",
        sample_kwargs={"random_seed": 1},
        sample_prior_predictive=False,
        sample_posterior_predictive=False,
    )
    return sampler.sample(model)


@pytest.fixture
def jax_x64_pinned_off(monkeypatch: pytest.MonkeyPatch):
    """Reproduce a session whose PyTensor JAX backend was first imported at float32."""
    import pytensor.link.jax.dispatch  # noqa: F401

    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    previous = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.usefixtures("jax_x64_pinned_off")
def test_numpyro_samples_float64_model_in_double_precision() -> None:
    """A float64 graph is not truncated to float32 by an earlier PyTensor pin."""
    with pytensor.config.change_flags(floatX="float64"):
        model = _tiny_model()
        trace = _sample(model)

    assert trace.posterior["mu"].dtype == np.float64
    assert json.loads(trace.attrs["sampler_precision"]) == {
        "floatX": "float64",
        "model_dtype": "float64",
        "jax_enable_x64": True,
    }
    # The trace holds JAX arrays, so x64 stays on for later diagnostics and outputs.
    assert jax.config.read("jax_enable_x64") is True


@pytest.mark.usefixtures("jax_x64_pinned_off")
def test_numpyro_warns_for_float32_model(caplog: pytest.LogCaptureFixture) -> None:
    """The float32 default still runs, but says so and records it."""
    with pytensor.config.change_flags(floatX="float32"):
        model = _tiny_model()
        with caplog.at_level(logging.WARNING, logger="openghg_inversions.inference.sampling"):
            trace = _sample(model)

    assert "single precision" in caplog.text
    assert json.loads(trace.attrs["sampler_precision"])["jax_enable_x64"] is False


@pytest.mark.usefixtures("jax_x64_pinned_off")
def test_environment_request_enables_x64_for_float32_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """JAX_ENABLE_X64 keeps its meaning after PyTensor's import-time pin."""
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    with pytensor.config.change_flags(floatX="float32"):
        model = _tiny_model()
        with jax_sampling_precision(model, "numpyro") as precision:
            assert jax.config.read("jax_enable_x64") is True

    assert precision["jax_enable_x64"] is True


def test_non_jax_backend_records_nothing() -> None:
    """PyMC, nutpie and other compiled backends keep their own precision handling."""
    with jax_sampling_precision(_tiny_model(), "pymc") as precision:
        assert precision == {}


@pytest.mark.fresh_process
def test_jax_enable_x64_environment_selects_float64_graphs() -> None:
    """Without an explicit floatX, JAX_ENABLE_X64=1 survives PyTensor's JAX import."""
    env = os.environ.copy()
    flags = [
        setting
        for setting in env.get("PYTENSOR_FLAGS", "").split(",")
        if setting and setting.partition("=")[0].strip() not in {"floatX", "warn_float64"}
    ]
    env["PYTENSOR_FLAGS"] = ",".join(flags)
    env["JAX_ENABLE_X64"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import openghg_inversions.rhime, pytensor, jax; "
            "import pytensor.link.jax.dispatch; "
            "print(pytensor.config.floatX, jax.config.read('jax_enable_x64'))",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.stdout.split() == ["float64", "True"]
