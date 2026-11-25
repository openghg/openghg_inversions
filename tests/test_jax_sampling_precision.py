"""Check the process-start precision of JAX sampling."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pymc as pm
import pytest
from pytensor import config
import xarray as xr

from openghg_inversions.inference.sampling import RhimeSampler

pytest.importorskip("numpyro")
jax = pytest.importorskip("jax")


def _sample() -> xr.DataTree:
    """Sample a one-parameter Gaussian through the public RHIME sampler."""
    with pm.Model() as model:
        mean = pm.Normal("mean")
        pm.Normal("y", mu=mean, observed=[0.1, 0.2])
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
