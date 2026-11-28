"""Check the install hint for JAX sampling without TensorFlow Probability."""

from __future__ import annotations

import sys

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.models._gamma_quantile import _GammaStandardQuantile

pytest.importorskip("numpyro")
jax = pytest.importorskip("jax")


@pytest.fixture
def without_tfp(monkeypatch):
    """Make TensorFlow Probability unimportable, as without the ``jax`` extra."""
    for name in [name for name in sys.modules if name.partition(".")[0] == "tensorflow_probability"]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "tensorflow_probability", None)


@pytest.fixture
def jax_precision():
    """Keep JAX precision consistent with floatX and restore it afterwards."""
    import pytensor
    import pytensor.link.jax.dispatch  # noqa: F401

    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", str(pytensor.config.floatX) == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _sample(model: pm.Model, nuts_sampler: str = "numpyro"):
    """Sample a small model through the public RHIME sampler."""
    sampler = RhimeSampler(
        draws=5,
        tune=5,
        chains=1,
        nuts_sampler=nuts_sampler,
        sample_kwargs={"random_seed": 1},
        sample_prior_predictive=False,
        sample_posterior_predictive=False,
    )
    return sampler.sample(model)


def _gamma_quantile_model() -> pm.Model:
    """Gamma copula quantile, whose JAX implementation imports TensorFlow Probability."""
    with pm.Model() as model:
        latent = pm.Normal("latent", shape=2)
        shape = pt.as_tensor_variable(np.full(2, 2.0, dtype=latent.dtype))
        pm.Potential("gamma", -_GammaStandardQuantile(latent.type, shape.type)(latent, shape).sum())
    return model


def _gammaincinv_model() -> pm.Model:
    """Inverse incomplete gamma, which PyTensor's JAX dispatch takes from TensorFlow Probability."""
    with pm.Model() as model:
        latent = pm.Normal("latent")
        pm.Potential("gamma", -pt.gammaincinv(2.0, pm.math.sigmoid(latent)))
    return model


@pytest.mark.parametrize("build", [_gamma_quantile_model, _gammaincinv_model])
@pytest.mark.usefixtures("without_tfp", "jax_precision")
def test_missing_tfp_names_the_jax_extra(build) -> None:
    """A model needing TensorFlow Probability fails before NUTS with the install hint."""
    with pytest.raises(ImportError, match=r"openghg-inversions\[jax\]") as raised:
        _sample(build())
    assert isinstance(raised.value.__cause__, NotImplementedError | ModuleNotFoundError)


@pytest.mark.usefixtures("without_tfp", "jax_precision")
def test_models_without_tfp_ops_sample_without_tfp() -> None:
    """Gaussian models still sample under NumPyro without the extra."""
    with pm.Model() as model:
        mean = pm.Normal("mean")
        pm.Normal("y", mu=mean, observed=np.array([0.1, 0.2]))
    trace = _sample(model)
    assert np.isfinite(trace.posterior["mean"].values).all()


@pytest.mark.usefixtures("without_tfp")
def test_pymc_sampler_does_not_need_tfp() -> None:
    """The PyMC backend samples the same Ops without the extra."""
    trace = _sample(_gammaincinv_model(), nuts_sampler="pymc")
    assert np.isfinite(trace.posterior["latent"].values).all()
