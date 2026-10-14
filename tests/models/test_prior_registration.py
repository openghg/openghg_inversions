"""Numerical and persistence checks for construction-owned prior inputs."""

import json

import dask
import dask.array as da
import numpy as np
import pymc as pm
import pytest
import xarray as xr
from scipy.stats import lognorm, norm

from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.models.components import add_state_vector
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.priors import parse_prior, prior_records, trace_prior_records
from openghg_inversions.models.state_activity import StateActivity, resolve_state_activity
from openghg_inversions.serialization import load_trace, save_trace


@pytest.mark.parametrize("family,reparameterise", [("normal", False), ("lognormal", False), ("lognormal", True)])
def test_lazy_prior_values_coordinates_and_density_survive_sampling(tmp_path, family, reparameterise):
    calls = []

    @dask.delayed
    def source():
        calls.append(1)
        return np.array([2.0, 1.0])

    raw = da.from_delayed(source(), shape=(2,), dtype=float)
    mean = xr.DataArray(raw, dims="region", coords={
        "region": ["b", "a"], "group": ("region", raw.astype(int)),
    })
    stdev = xr.DataArray(raw.sum() / 10)
    borrowed_graph = mean.data
    params = ({"pdf": family, "mu": mean, "sigma": stdev} if family == "normal"
              else {"pdf": family, "mean": mean, "stdev": stdev, "reparameterise": reparameterise})
    with registered_model(coords={"region": ["a", "b"]}) as model:
        state = parse_prior("x", params, dims="region")
    assert calls == [1]
    record = prior_records(model)["x"]
    scalar_key = "sigma" if family == "normal" else "stdev"
    array_key = "mu" if family == "normal" else "mean"
    assert record[scalar_key] == pytest.approx(0.3)
    assert record[array_key] == {"data": f"x_prior_{array_key}"}
    assert f"x_prior_{scalar_key}" not in model

    point = np.array([1.1, 2.2])
    if family == "normal":
        expected = norm.logpdf(point, [1, 2], 0.3).sum()
        actual = pm.logp(state, point).eval()
    else:
        variance_log = np.log1p((0.3 / np.array([1., 2.])) ** 2)
        mu = np.log([1., 2.]) - variance_log / 2
        sigma = np.sqrt(variance_log)
        if reparameterise:
            z = np.array([-0.2, 0.4], dtype=model["x_latent"].dtype)
            np.testing.assert_allclose(state.eval({model["x_latent"]: z}), np.exp(mu + sigma * z), rtol=1e-6)
            expected = norm.logpdf(z).sum()
            actual = pm.logp(model["x_latent"], z).eval()
        else:
            expected = lognorm.logpdf(point, s=sigma, scale=np.exp(mu)).sum()
            actual = pm.logp(state, point).eval()
    np.testing.assert_allclose(actual.sum(), expected, rtol=1e-6, atol=1e-5)

    trace = RhimeSampler(draws=4, tune=4, chains=1, sample_prior_predictive=False,
                        sample_posterior_predictive=False,
                        sample_kwargs={"cores": 1, "random_seed": 42}).sample(model)
    save_trace(trace, tmp_path / "prior.nc")
    loaded = load_trace(tmp_path / "prior.nc")
    values = loaded["constant_data"][f"x_prior_{array_key}"]
    np.testing.assert_allclose(values, [1, 2])
    assert values.region.values.tolist() == ["a", "b"]
    assert values.group.values.tolist() == [1, 2]
    assert trace_prior_records(loaded)["x"][scalar_key] == pytest.approx(0.3)
    assert calls == [1]
    assert mean.data is borrowed_graph and dask.is_dask_collection(stdev)
    assert params[array_key] is mean


def test_full_parameters_registered_before_active_selection_and_all_fixed_skips_prior():
    zero = xr.DataArray([False, False, True], dims="region", coords={"region": ["a", "b", "c"]})
    activity = resolve_state_activity(zero, StateActivity(active=[True, False, True]))
    mean = xr.DataArray([30., 20., 10.], dims="region", coords={"region": ["c", "b", "a"]})
    with registered_model() as model:
        vector = add_state_vector(activity, {"pdf": "normal", "mu": mean, "sigma": .2}, "x")
    np.testing.assert_array_equal(model["x_prior_mu"].get_value(), [10., 20., 30.])
    np.testing.assert_allclose(pm.logp(vector.latent, [10.]).eval(), norm.logpdf([10.], 10., .2), rtol=1e-6)
    assert prior_records(model)["x"]["mu"] == {"data": "x_prior_mu"}
    calls = []

    @dask.delayed
    def unused():
        calls.append(1)
        return np.zeros(3)

    lazy = xr.DataArray(da.from_delayed(unused(), shape=(3,), dtype=float), dims="region", coords=zero.coords)
    fixed = resolve_state_activity(zero, StateActivity(active=False))
    with registered_model() as model:
        add_state_vector(fixed, {"pdf": "normal", "mu": lazy, "sigma": .2}, "x")
    assert calls == [] and prior_records(model) == {}


def test_scalar_symbolic_and_length_one_parameters_have_distinct_representations():
    with registered_model(coords={"region": ["a"]}) as model:
        scale = parse_prior("scale", {"pdf": "halfnormal", "sigma": .2})
        x = parse_prior("x", {"pdf": "normal", "mu": np.array([1.]), "sigma": scale}, dims="region")
    assert prior_records(model)["scale"] == {"pdf": "halfnormal", "sigma": .2}
    assert prior_records(model)["x"]["sigma"] == {"variable": "scale"}
    assert model["x_prior_mu"].ndim == 1
    assert "scale_prior_sigma" not in model and "x_prior_sigma" not in model
    np.testing.assert_allclose(pm.logp(x, [1.]).eval({scale: .5}), norm.logpdf([1.], 1., .5))


def test_missing_required_array_reference_is_reported():
    trace = xr.DataTree(xr.Dataset(attrs={"rhime_priors": json.dumps({"x": {"pdf": "normal", "mu": {"data": "missing"}}})}))
    with pytest.raises(ValueError, match="Missing registered prior data.*missing"):
        trace_prior_records(trace)


def test_labelled_prior_axes_broadcast_by_dimension_and_named_model_references():
    sigma = xr.DataArray([1., 2.], dims="site", coords={"site": ["a", "b"]})
    mean = xr.DataArray(np.arange(6.).reshape(3, 2), dims=("time", "site"),
                        coords={"site": ["a", "b"], "time": [0, 1, 2]})
    with registered_model(name="recipe", coords={"site": ["a", "b"], "time": [0, 1, 2]}) as model:
        state = parse_prior("x", {"pdf": "normal", "mu": mean, "sigma": sigma}, dims=("site", "time"))
    expected = norm.logpdf(np.ones((2, 3)), mean.values.T, sigma.values[:, None])
    np.testing.assert_allclose(pm.logp(state, np.ones((2, 3))).eval(), expected, rtol=1e-6)
    assert prior_records(model)["x"]["sigma"] == {"data": "recipe::x_prior_sigma"}

    with registered_model(coords={"site": ["a", "b"]}):
        with pytest.raises(ValueError, match="dimensions must be a subset"):
            parse_prior("wrong", {"pdf": "normal", "mu": xr.DataArray([1., 2.], dims="wrong_axis"),
                                  "sigma": .2}, dims="site")


def test_plain_model_positional_singletons_preserve_broadcasting():
    with pm.Model(coords={"site": ["a", "b"], "time": [0, 1, 2]}) as model:
        parse_prior("x", {"pdf": "normal", "mu": 0., "sigma": np.array([[1.], [2.]])},
                    dims=("site", "time"))
        trace = pm.sample_prior_predictive(draws=1, random_seed=42)
    assert trace.prior["x"].shape == (1, 1, 2, 3)
    assert model["x_prior_sigma"].get_value().shape == (2, 1)


@pytest.mark.parametrize("model_context", [registered_model, pm.Model])
def test_named_prior_dimensions_validate_length_without_indexes(model_context):
    with model_context(coords={"site": ["a", "b"]}) as model:
        with pytest.raises(ValueError, match="incompatible 'site' length"):
            parse_prior("wrong", {"pdf": "normal", "mu": xr.DataArray([1.], dims="site"),
                                  "sigma": 1.}, dims="site")
        assert "wrong_prior_mu" not in model
        state = parse_prior("x", {"pdf": "normal", "mu": xr.DataArray([1., 2.], dims="site"),
                                  "sigma": 1.}, dims="site")
        trace = pm.sample_prior_predictive(draws=1, random_seed=42)
    assert trace.prior[state.name].sizes["site"] == 2
    np.testing.assert_array_equal(trace.constant_data["x_prior_mu"], [1., 2.])
