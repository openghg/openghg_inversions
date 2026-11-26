"""Check the process-start precision of JAX sampling."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pymc as pm
import pytest
from pytensor import config
import xarray as xr

from openghg_inversions.inference.sampling import RhimeSampler

pytest.importorskip("numpyro")
jax = pytest.importorskip("jax")


def _gaussian_model() -> pm.Model:
    """One-parameter Gaussian built at the current floatX."""
    with pm.Model() as model:
        mean = pm.Normal("mean")
        pm.Normal("y", mu=mean, observed=[0.1, 0.2])
    return model


def _upcast_model() -> pm.Model:
    """Float32 value variables whose log density is float64.

    A float64 NumPy error vector upcasts the likelihood, as the CO2 fixed-error
    likelihood does under float32. NumPyro fails with "cond branches must have
    equal output types" if this model is sampled with JAX x64 switched on.
    """
    with pm.Model() as model:
        mean = pm.Normal("mean", shape=3)
        pm.Normal("y", mu=mean, sigma=np.array([0.5, 1.0, 2.0]), observed=np.array([0.1, 0.2, 0.3]))
    return model


def _sample(model: pm.Model | None = None) -> xr.DataTree:
    """Sample a model through the public RHIME sampler."""
    if model is None:
        model = _gaussian_model()
    sampler = RhimeSampler(
        draws=10,
        tune=10,
        chains=1,
        nuts_sampler="numpyro",
        sample_kwargs={"random_seed": 1},
        sample_prior_predictive=False,
        sample_posterior_predictive=False,
    )
    return sampler.sample(model)


@pytest.mark.fresh_process
def test_explicit_float64_startup_preserves_numpyro_posterior() -> None:
    """Select float64 before PyTensor and its JAX dispatcher initialize."""
    env = os.environ.copy()
    flags = [
        setting
        for setting in env.get("PYTENSOR_FLAGS", "").split(",")
        if setting and setting.partition("=")[0].strip() != "floatX"
    ]
    env["PYTENSOR_FLAGS"] = ",".join([*flags, "floatX=float64"])
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_jax_sampling_precision import _sample; "
            "trace = _sample(); "
            "print(trace.posterior['mean'].dtype); print(trace.attrs['sampler_precision'])",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    dtype, precision = completed.stdout.splitlines()[-2:]
    assert dtype == "float64"
    assert json.loads(precision) == {"floatX": "float64", "jax_enable_x64": True}


@pytest.fixture
def jax_x64_off():
    """A float32 session whose JAX dispatcher has set x64 off."""
    import pytensor.link.jax.dispatch  # noqa: F401

    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.usefixtures("jax_x64_off")
def test_float32_model_with_float64_log_density_samples_in_float32() -> None:
    """An upcast log density does not switch JAX to x64 for float32 value variables."""
    with config.change_flags(floatX="float32"):
        model = _upcast_model()
        assert model.logp().dtype == "float64"
        assert [variable.dtype for variable in model.value_vars] == ["float32"]
        trace = _sample(model)

    assert trace.posterior["mean"].dtype == np.float32
    assert np.isfinite(trace.posterior["mean"].values).all()
    assert json.loads(trace.attrs["sampler_precision"]) == {"floatX": "float32", "jax_enable_x64": False}
    assert not jax.config.x64_enabled


def test_x64_runtime_rejects_float32_model() -> None:
    """JAX x64 switched on in-process with floatX=float32 fails before NumPyro runs."""
    import pytensor.link.jax.dispatch  # noqa: F401

    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        with (
            config.change_flags(floatX="float32"),
            pytest.raises(ValueError, match="Start a fresh Python process"),
        ):
            _sample(_upcast_model())
        assert jax.config.x64_enabled
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.fresh_process
def test_jax_enable_x64_with_explicit_float32_samples_in_float32() -> None:
    """JAX_ENABLE_X64=1 with explicit floatX=float32 samples consistently and warns."""
    env = os.environ.copy()
    flags = [
        setting
        for setting in env.get("PYTENSOR_FLAGS", "").split(",")
        if setting and setting.partition("=")[0].strip() != "floatX"
    ]
    env["PYTENSOR_FLAGS"] = ",".join([*flags, "floatX=float32"])
    env["JAX_ENABLE_X64"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import jax; "
            "from tests.test_jax_sampling_precision import _sample, _upcast_model; "
            "trace = _sample(_upcast_model()); "
            "print(trace.posterior['mean'].dtype); print(trace.attrs['sampler_precision'])",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    dtype, precision = completed.stdout.splitlines()[-2:]
    assert dtype == "float32"
    assert json.loads(precision) == {"floatX": "float32", "jax_enable_x64": False}
    assert "PyTensor floatX selects float32" in completed.stderr


def test_incompatible_jax_runtime_fails_before_sampling() -> None:
    """Explain a precision mismatch when a backend was imported at float32."""
    import pytensor.link.jax.dispatch  # noqa: F401

    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        with (
            config.change_flags(floatX="float64"),
            pytest.raises(ValueError, match="Start a fresh Python process"),
        ):
            _sample()
        assert not jax.config.x64_enabled
    finally:
        jax.config.update("jax_enable_x64", previous)
