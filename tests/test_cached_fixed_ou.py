"""Fixed fixed-OU amplitudes are a specialisation of the cached fixed-OU sampler.

With fixed amplitudes the state quadratic is built once, there is no amplitude
step and no refresh, and stock state NUTS samples the model priors plus that
quadratic. These tests check its density and gradient against the plain
fixed-OU model and a dense oracle, the step structure, the sampled posterior,
routing of configured and prepared-input runs, and validation.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from pymc.blocking import DictToArrayBijection
from pymc.step_methods import CompoundStep
from scipy.stats import multivariate_normal

from boundary_test_utils import bound_linked_model_arguments
from examples.co2_o2_handoffs import synthetic_inputs
from openghg_inversions.correlated_state import CorrelatedPrior
from openghg_inversions.inference.cached_sigma import (
    PymcCachedSigmaNutsStep,
    PytensorMarginalQuadraticCache,
    make_cached_sigma_compound_step,
)
from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.models.cached_sigma import FixedOuCachedSigmaTarget
from openghg_inversions.models.fixed_ou import (
    FixedOuLowRank,
    add_fixed_ou_gaussian_likelihood,
    prepare_fixed_ou_low_rank,
)
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import co2_cached_sigma_model, co2_cached_sigma_runner, inference
from openghg_inversions.rhime.co2 import co2_o2_handoffs
from openghg_inversions.rhime.co2.co2_model import build_co2_model
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_runner import (
    run_rhime_co2_o2_cached_sigma_from_prepared_inputs,
)
from openghg_inversions.rhime.co2.co2_o2_fixed_ou import linked_fixed_ou_alignment
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.rhime.co2.co2_o2_runner import run_rhime_co2_o2_from_prepared_inputs
from openghg_inversions.rhime.co2.configuration import Co2ModelConfig, _resolve_co2_sampling
from openghg_inversions.serialization import load_trace

from test_co2_cached_sigma import _build, _dense_fixed_ou_covariance, _inputs
from test_co2_o2_fixed_ou import _dense, _kwargs, _per_meg_prepared, _prepared
from test_rhime_co2_o2 import _independent_error


AMPLITUDES = {"AAA": 0.3, "BBB": 0.4}
TAU = {"AAA": 3.0, "BBB": 7.0}
LINKED_TAU = {"co2:A": 6.0, "o2:A": 8.0, "o2:B": 2.0}
# Mixed-unit amplitudes: CO2 in ppm, O2 in per meg.
LINKED_AMPLITUDES = {"co2:A": 0.4, "o2:A": 6.0, "o2:B": 3.0}


def _fixed_co2(**kwargs: Any):
    """Build the CO2 cached graph with fixed site amplitudes."""
    options = {
        "site_amplitude_prior": None,
        "initial_site_amplitudes": None,
        "fixed_site_amplitudes": AMPLITUDES,
    }
    return _build(**{**options, **kwargs})


def _plain_co2():
    inputs = _inputs()
    return build_co2_model(
        inputs["H"],
        retained_prior=CorrelatedPrior(
            inputs["alpha_prior_mean"],
            inputs["alpha_prior_covariance"],
            covariance_dim="region_cov",
        ),
        fixed_prior_contribution=inputs["fixed_prior_contribution"],
        observations=inputs["mf"],
        observation_error=inputs["mf_error"],
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        likelihood_builder=add_fixed_ou_gaussian_likelihood,
        likelihood_kwargs={"tau_hours": TAU, "fixed_site_amplitudes": AMPLITUDES},
    )


def _state_step(cached):
    """Return the only method of the fixed-amplitude CompoundStep."""
    compound = make_cached_sigma_compound_step(
        model=cached.model,
        sigma=None,
        states=cached.states,
        modelled_mean=cached.modelled_mean,
        target=cached.target,
        shared_cache=cached.shared_cache,
        initial_cache=cached.initial_cache,
        cached_state_logp=cached.cached_state_logp,
        rng=np.random.default_rng(31),
    )
    assert isinstance(compound, CompoundStep)
    (step,) = compound.methods
    return step


def _assert_state_cost_matches(cached, plain, dense_logp, points):
    """Compare the cached state cost and gradient to the plain model and a dense oracle."""
    step = _state_step(cached)
    plain_logp = plain.compile_logp()
    plain_dlogp = plain.compile_dlogp(vars=plain.free_RVs)
    prior_logp = plain.compile_logp(vars=plain.free_RVs, jacobian=True)
    mean_fn = plain.compile_fn(
        plain.replace_rvs_by_values([plain["modelled_concentration"]])[0],
        inputs=plain.value_vars,
    )
    assert [value.name for value in cached.model.value_vars] == [value.name for value in plain.value_vars]

    def dense_cost(point):
        return prior_logp(point) + dense_logp(mean_fn(point))

    for point in points:
        step._logp_dlogp_func.set_extra_values(point)
        actual, gradient = step._logp_dlogp_func(DictToArrayBijection.map(point))
        np.testing.assert_allclose(actual, plain_logp(point), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(gradient, plain_dlogp(point), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(actual, dense_cost(point), rtol=1e-10, atol=1e-10)
        flat = DictToArrayBijection.map(point)
        expected_gradient = np.empty(flat.data.size)
        for index in range(flat.data.size):
            plus, minus = flat.data.copy(), flat.data.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            expected_gradient[index] = (
                dense_cost(DictToArrayBijection.rmap(flat._replace(data=plus)))
                - dense_cost(DictToArrayBijection.rmap(flat._replace(data=minus)))
            ) / 2e-6
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.usefixtures("double_precision")
def test_single_species_fixed_state_cost_matches_plain_model_and_dense_oracle():
    cached = _fixed_co2()
    plain = _plain_co2()
    inputs = _inputs()
    covariance = _dense_fixed_ou_covariance(inputs, np.array([AMPLITUDES["AAA"], AMPLITUDES["BBB"]]), TAU)

    def dense_logp(mean):
        return multivariate_normal.logpdf(inputs["mf"].values, mean=mean, cov=covariance)

    assert cached.fixed_amplitudes
    assert "ou_site_amplitude" not in {rv.name for rv in cached.model.free_RVs}
    assert "ou_site_amplitude" in {data.name for data in cached.model.data_vars}
    np.testing.assert_array_equal(cached.initial_cache.sigma, [0.3, 0.4])
    points = [
        {"flux_scaling_latent": np.asarray(value, dtype=float)}
        for value in ([0.0, 0.0], [0.7, -0.4], [-1.3, 0.9])
    ]
    _assert_state_cost_matches(cached, plain, dense_logp, points)


@pytest.mark.usefixtures("double_precision")
def test_linked_mixed_unit_fixed_state_cost_matches_plain_model_and_dense_oracle():
    kwargs = _kwargs(_per_meg_prepared())
    assert set(kwargs["observations"].observation_units.values.tolist()) == {"ppm", "per meg"}
    options = {**kwargs, "tau_hours": LINKED_TAU, "fixed_site_amplitudes": LINKED_AMPLITUDES}
    cached = build_co2_o2_cached_sigma_model(**bound_linked_model_arguments(options))
    plain = build_co2_o2_model(**bound_linked_model_arguments(options))
    labels = linked_fixed_ou_alignment(kwargs["observations"]).site_labels.values
    covariance = _dense(kwargs, np.array([LINKED_AMPLITUDES[label] for label in labels]), LINKED_TAU)

    def dense_logp(mean):
        return multivariate_normal.logpdf(kwargs["observations"].values, mean=mean, cov=covariance)

    assert cached.fixed_amplitudes
    names = [value.name for value in plain.value_vars]
    rng = np.random.default_rng(5)
    points = [plain.initial_point()] + [
        {name: rng.normal(scale=0.5, size=np.shape(value)) for name, value in plain.initial_point().items()}
        for _ in range(2)
    ]
    assert all(list(point) == names for point in points)
    _assert_state_cost_matches(cached, plain, dense_logp, points)


@pytest.mark.parametrize("full_width", [False, True])
def test_fixed_amplitudes_refresh_once_and_state_steps_never_refactorize(monkeypatch, full_width):
    refreshes = []
    original_refresh = FixedOuCachedSigmaTarget.refresh

    def counted_refresh(self, sigma):
        refreshes.append(np.asarray(sigma).copy())
        return original_refresh(self, sigma)

    monkeypatch.setattr(FixedOuCachedSigmaTarget, "refresh", counted_refresh)
    inputs = _inputs()
    if full_width:
        # A full-rank aggregation factor takes the covariance-solver refresh branch.
        inputs["aggregation_error_covariance"] = inputs["aggregation_error_covariance"] + 0.01 * xr.DataArray(
            np.eye(4),
            dims=("nmeasure", "nmeasure_cov"),
        )
    cached = _fixed_co2(inputs=inputs)
    assert len(refreshes) == 1
    np.testing.assert_array_equal(refreshes[0], [0.3, 0.4])
    sampler = co2_cached_sigma_runner._sampler_for_cached_graph(
        RhimeSampler(nuts_sampler="pymc", sample_kwargs={"target_accept": 0.91}),
        cached_model=cached,
    )
    compound = sampler.sample_kwargs["step"]
    assert "target_accept" not in sampler.sample_kwargs
    assert len(compound.methods) == 1
    (state_step,) = compound.methods
    assert isinstance(state_step, pm.NUTS)
    assert not any(isinstance(method, PymcCachedSigmaNutsStep) for method in compound.methods)
    assert state_step.target_accept == 0.91
    assert len(refreshes) == 1

    def forbidden(*args, **kwargs):
        pytest.fail("Fixed-amplitude state steps must not refresh or factor the observation covariance.")

    monkeypatch.setattr(FixedOuCachedSigmaTarget, "refresh", forbidden)
    monkeypatch.setattr(FixedOuLowRank, "evaluate", forbidden)
    monkeypatch.setattr(FixedOuLowRank, "_evaluate", forbidden)
    monkeypatch.setattr(FixedOuLowRank, "_solve_transformed", forbidden)
    for name in ("cholesky", "qr", "svd", "eigh"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    point = cached.model.initial_point()
    for _ in range(3):
        point, stats = state_step.step(point)
        assert stats[0]["tree_size"] > 0
    for name in ("linear", "precision"):
        installed = getattr(cached.shared_cache, name).get_value()
        np.testing.assert_array_equal(installed, getattr(cached.initial_cache, name).astype(installed.dtype))


@pytest.mark.usefixtures("double_precision")
def test_fixed_route_matches_independent_dense_posterior_moments():
    """Sampled state moments agree with quadrature of the fixed-amplitude posterior."""
    import arviz as az
    from scipy.integrate import trapezoid

    observations = np.array([0.9, -0.1, 0.7])
    design = np.array([[0.8], [0.3], [1.0]])
    factor = np.array([[0.25], [-0.1], [0.2]])
    diagonal_variance = np.array([0.15, 0.2, 0.12])
    times = np.array([0.0, 1.0, 2.0])
    tau_hours = 4.0
    amplitude = 0.4
    prepared = prepare_fixed_ou_low_rank(
        factor=factor,
        diagonal_variance=diagonal_variance,
        observation_times=times,
        site_index=np.zeros(3, dtype=int),
        tau_hours=np.array([tau_hours]),
        site_labels=("AAA",),
    )
    target = FixedOuCachedSigmaTarget(
        prepared=prepared,
        observations=observations,
        fixed_contribution=np.zeros(3),
        design=design,
    )
    state_grid = np.linspace(-5.0, 3.0, 4000)
    covariance = (
        factor @ factor.T
        + np.diag(diagonal_variance)
        + amplitude**2 * np.exp(-np.abs(times[:, None] - times[None, :]) / tau_hours)
    )
    residual = observations[:, None] - design @ np.exp(state_grid)[None, :]
    log_weights = -0.5 * (
        np.square(state_grid) + np.sum(residual * np.linalg.solve(covariance, residual), axis=0)
    )
    weights = np.exp(log_weights - log_weights.max())
    expected_state = trapezoid(weights * np.exp(state_grid), state_grid) / trapezoid(weights, state_grid)

    initial_cache = target.refresh(np.array([amplitude]))
    shared = PytensorMarginalQuadraticCache(initial_cache)
    with pm.Model() as model:
        state_white = pm.Normal("state_white", shape=1)
        state = pm.Deterministic("state", pt.exp(state_white))
        mean = pm.Deterministic("modelled_mean", target.fixed_contribution + pt.dot(target.design, state))
        fixed_amplitude = pm.Data("site_amplitude", np.array([amplitude]))
        pm.CustomDist(
            "y",
            mean,
            fixed_amplitude,
            logp=prepared.logp,
            random=prepared.random,
            signature="(n),(s)->(n)",
            observed=observations,
        )
        step = make_cached_sigma_compound_step(
            model=model,
            sigma=None,
            states=[state_white],
            modelled_mean=mean,
            target=target,
            shared_cache=shared,
            initial_cache=initial_cache,
            cached_state_logp=shared.log_likelihood(state),
            state_target_accept=0.9,
            rng=np.random.default_rng(710),
        )
        assert len(step.methods) == 1
        idata = pm.sample(
            draws=3000,
            tune=600,
            chains=2,
            cores=1,
            step=step,
            random_seed=20261124,
            progressbar=False,
            compute_convergence_checks=False,
        )

    mcse = float(az.mcse(idata, var_names=["state"])["state"].item())
    assert np.isfinite(mcse) and mcse <= 0.0125, mcse
    assert float(az.rhat(idata, var_names=["state"])["state"].item()) < 1.01
    error = abs(float(idata.posterior["state"].mean()) - expected_state)
    assert error <= min(0.05, 4.0 * mcse + 0.0002), (error, mcse)
    assert int(np.asarray(idata.sample_stats["diverging"]).sum()) == 0
    assert not any(name.startswith("sigma_") or name == "cache_refreshes" for name in idata.sample_stats)


@pytest.mark.parametrize(
    "likelihood",
    [
        {"kind": "fixed_ou", "tau_hours": 4.0, "fixed_site_amplitudes": 0.5},
        {"kind": "fixed_ou", "tau_hours": 4.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0}},
    ],
)
def test_fixed_ou_likelihood_always_uses_cached_route(likelihood):
    model = Co2ModelConfig.from_params({"likelihood": likelihood}).model_arguments()
    assert inference.uses_cached_ou(model)
    assert inference.uses_fixed_ou_amplitudes(model) is ("fixed_site_amplitudes" in likelihood)


def test_configured_co2_fixed_ou_maps_target_accept_to_the_state_step():
    likelihood = {"kind": "fixed_ou", "tau_hours": 4.0, "fixed_site_amplitudes": 0.5}
    model = Co2ModelConfig.from_params({"likelihood": likelihood}).model_arguments()
    previous = _resolve_co2_sampling({"sampling": {"target_accept": 0.93}}, model=model)
    assert previous.sampler.sample_kwargs["target_accept"] == 0.93
    assert previous.sigma_target_accept is None and previous.state_target_accept is None
    assert co2_cached_sigma_runner.resolve_cached_target_accept(
        previous.sampler,
        fixed_amplitudes=True,
    ) == (None, 0.93)
    explicit = _resolve_co2_sampling({"sampling": {"state_target_accept": 0.91}}, model=model)
    assert explicit.state_target_accept == 0.91 and explicit.sigma_target_accept is None
    default = _resolve_co2_sampling({}, model=model)
    assert co2_cached_sigma_runner.resolve_cached_target_accept(default.sampler, fixed_amplitudes=True) == (
        None,
        co2_cached_sigma_runner.FIXED_AMPLITUDE_STATE_TARGET_ACCEPT,
    )
    with pytest.raises(ValueError, match="sigma_target_accept.*no amplitude step"):
        _resolve_co2_sampling({"sampling": {"sigma_target_accept": 0.8}}, model=model)
    with pytest.raises(ValueError, match="not both"):
        _resolve_co2_sampling({"sampling": {"target_accept": 0.9, "state_target_accept": 0.9}}, model=model)


@pytest.fixture
def captured_steps(monkeypatch):
    """Record every cached CompoundStep the CO2 cached runner constructs."""
    calls: list[dict[str, Any]] = []
    original = co2_cached_sigma_runner.make_cached_sigma_compound_step

    def capture(**kwargs):
        step = original(**kwargs)
        calls.append({**kwargs, "step": step})
        return step

    monkeypatch.setattr(co2_cached_sigma_runner, "make_cached_sigma_compound_step", capture)
    return calls


def _short_sampler(**sample_kwargs):
    return RhimeSampler(
        nuts_sampler="pymc",
        draws=4,
        tune=4,
        chains=1,
        sample_kwargs={
            "cores": 1,
            "random_seed": 23,
            "progressbar": False,
            "compute_convergence_checks": False,
            **sample_kwargs,
        },
    )


def _assert_state_only_trace(trace, calls, *, target_accept):
    assert len(calls) == 1
    assert calls[0]["sigma"] is None
    assert calls[0]["state_target_accept"] == target_accept
    (state_step,) = calls[0]["step"].methods
    assert isinstance(state_step, pm.NUTS)
    assert "ou_site_amplitude" not in trace.posterior
    assert "ou_site_amplitude" in trace.constant_data
    assert not any(name.startswith("sigma_") or name == "cache_refreshes" for name in trace.sample_stats)
    actual = json.loads(trace.attrs["rhime_actual_inference"])
    assert actual["algorithm"] == "cached_fixed_ou_state_nuts"
    assert actual["state_target_accept"] == target_accept
    assert "sigma_target_accept" not in actual


def test_configured_linked_fixed_ou_sample_uses_cached_state_step(tmp_path, captured_steps):
    manifest = co2_o2_handoffs.save_preparation(synthetic_inputs(), output_dir=tmp_path / "prepare")
    model = {
        "channels": {"co2": {"units": "ppm"}, "o2": {"units": "ppm"}},
        "likelihood": {"kind": "fixed_ou", "tau_hours": 12.0, "fixed_site_amplitudes": 0.3},
    }
    sample_manifest = co2_o2_handoffs.sample(
        manifest,
        model=model,
        output_dir=tmp_path / "sample",
        sampler=_short_sampler(target_accept=0.95),
    )
    record = json.loads(sample_manifest.read_text())
    assert record["actual_inference"]["algorithm"] == "cached_fixed_ou_state_nuts"
    assert record["actual_inference"]["state_target_accept"] == 0.95
    trace = load_trace(sample_manifest.parent / record["posterior"])
    _assert_state_only_trace(trace, captured_steps, target_accept=0.95)
    assert trace.attrs["rhime_recipe"] == "co2_o2"
    amplitude = trace.constant_data.ou_site_amplitude
    np.testing.assert_allclose(amplitude, 0.3)
    assert amplitude.ou_units.values.tolist() == ["ppm", "ppm"]
    assert amplitude.attrs["units"] == "ppm"
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    assert roles["fixed_ou_site_amplitude"] == "ou_site_amplitude"
    assert json.loads(amplitude.attrs["rhime_scientific_roles"]) == ["fixed_ou_site_amplitude"]
    assert trace.log_likelihood.y.dims == ("chain", "draw")
    assert "ou_site_amplitude" in co2_o2_handoffs.load_samples(sample_manifest).idata.constant_data


@pytest.mark.usefixtures("double_precision")
def test_prepared_input_linked_runner_samples_mixed_unit_fixed_ou_with_cached_state_step(captured_steps):
    prepared = _per_meg_prepared()
    trace = run_rhime_co2_o2_from_prepared_inputs(
        prepared_inputs=prepared,
        independent_error_sd=_independent_error(prepared),
        tau_hours=LINKED_TAU,
        fixed_site_amplitudes=LINKED_AMPLITUDES,
        sampler=_short_sampler(target_accept=0.95),
    )
    _assert_state_only_trace(trace, captured_steps, target_accept=0.95)
    assert trace.attrs["rhime_recipe"] == "co2_o2"
    amplitude = trace.constant_data.ou_site_amplitude
    assert amplitude.ou_site.values.tolist() == ["co2:A", "o2:A", "o2:B"]
    assert amplitude.ou_units.values.tolist() == ["ppm", "per meg", "per meg"]
    assert amplitude.attrs["units"] == "mixed; see ou_units coordinate"
    np.testing.assert_allclose(amplitude, [0.4, 6.0, 3.0])
    assert trace.constant_data.error.attrs["units"] == "mixed; see observation_units coordinate"


def test_named_linked_cached_runner_accepts_fixed_amplitudes(captured_steps):
    prepared = _prepared()
    trace = run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
        prepared_inputs=prepared,
        independent_error_sd=_independent_error(prepared),
        tau_hours=6.0,
        fixed_site_amplitudes=0.5,
        sampler=_short_sampler(),
        state_target_accept=0.85,
    )
    _assert_state_only_trace(trace, captured_steps, target_accept=0.85)
    assert trace.attrs["rhime_recipe"] == "co2_o2_cached_sigma_fixed_ou"


def test_co2_fixed_ou_inference_records_cached_state_step_and_constant_amplitudes(captured_steps):
    from test_co2_affine_output import _prepared as co2_prepared

    prepared, _ = co2_prepared()
    site = str(prepared.inv_inputs.site.values[0])
    request = {
        "likelihood": {"kind": "fixed_ou", "tau_hours": 4.0, "fixed_site_amplitudes": {site: 0.3}},
        "sampling": {
            "target_accept": 0.9,
            "draws": 3,
            "tune": 3,
            "chains": 1,
            "random_seed": 14,
            "progressbar": False,
        },
    }
    model = Co2ModelConfig.from_params(request).model_arguments()
    sampling = _resolve_co2_sampling(request, model=model)
    trace, actual = inference.sample_co2_inference(
        prepared,
        model,
        sampling.sampler,
        sigma_target_accept=sampling.sigma_target_accept,
        state_target_accept=sampling.state_target_accept,
    )
    assert actual == json.loads(trace.attrs["rhime_actual_inference"])
    _assert_state_only_trace(trace, captured_steps, target_accept=0.9)
    amplitude = trace.constant_data.ou_site_amplitude
    np.testing.assert_allclose(amplitude, 0.3)
    assert amplitude.attrs["units"] == prepared.inv_inputs.mf.attrs["units"]
    assert json.loads(amplitude.attrs["rhime_scientific_roles"]) == ["fixed_ou_site_amplitude"]
    assert (
        json.loads(trace.attrs["rhime_likelihood_builder"])["qualname"] == "add_fixed_ou_gaussian_likelihood"
    )
    assert json.loads(trace.attrs["rhime_likelihood_kwargs"])["fixed_site_amplitudes"] == {site: 0.3}


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75}, "fixed_site_amplitudes": AMPLITUDES},
    ],
)
def test_cached_builders_require_exactly_one_amplitude_option(options):
    with pytest.raises(ValueError, match="exactly one"):
        _build(**{"site_amplitude_prior": None, "initial_site_amplitudes": None, **options})
    kwargs = _kwargs(_prepared())
    with pytest.raises(ValueError, match="exactly one"):
        build_co2_o2_cached_sigma_model(
            **bound_linked_model_arguments({**kwargs, "tau_hours": 6.0, **options})
        )


def test_initial_amplitudes_require_an_amplitude_prior():
    with pytest.raises(ValueError, match="initial_site_amplitudes apply only to inferred"):
        _fixed_co2(initial_site_amplitudes=0.3)


@pytest.mark.parametrize("value", [-0.1, float("nan"), {"AAA": 0.3}])
def test_fixed_amplitudes_must_be_finite_non_negative_and_cover_sites(value):
    with pytest.raises(ValueError):
        _fixed_co2(fixed_site_amplitudes=value)


def test_singular_fixed_amplitude_covariance_uses_the_plain_message():
    prepared = prepare_fixed_ou_low_rank(
        factor=np.empty((2, 0)),
        diagonal_variance=np.zeros(2),
        observation_times=np.array([0.0, 1.0]),
        site_index=np.zeros(2, dtype=int),
        tau_hours=np.array([4.0]),
        site_labels=("AAA",),
    )
    target = FixedOuCachedSigmaTarget(
        prepared=prepared,
        observations=np.zeros(2),
        fixed_contribution=np.zeros(2),
        design=np.ones((2, 1)),
    )
    with pytest.raises(ValueError, match="Fixed OU amplitudes must produce a positive-definite"):
        co2_cached_sigma_model._fixed_amplitude_cache(target, np.zeros(1))
    assert np.isfinite(co2_cached_sigma_model._fixed_amplitude_cache(target, np.ones(1)).constant)


def test_mixed_units_require_a_species_site_mapping_for_cached_fixed_amplitudes():
    kwargs = _kwargs(_per_meg_prepared())
    base = {**kwargs, "tau_hours": LINKED_TAU}
    with pytest.raises(ValueError, match="must map"):
        build_co2_o2_cached_sigma_model(
            **bound_linked_model_arguments({**base, "fixed_site_amplitudes": 0.5})
        )
    with pytest.raises(ValueError, match="cannot hold mixed"):
        build_co2_o2_cached_sigma_model(
            **bound_linked_model_arguments(
                {**base, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.8}}
            )
        )
    cached = build_co2_o2_cached_sigma_model(
        **bound_linked_model_arguments({**base, "fixed_site_amplitudes": LINKED_AMPLITUDES})
    )
    assert cached.fixed_amplitudes
    np.testing.assert_allclose(cached.initial_cache.sigma, [0.4, 6.0, 3.0])
    zero = build_co2_o2_cached_sigma_model(
        **bound_linked_model_arguments({**base, "fixed_site_amplitudes": 0.0})
    )
    assert zero.fixed_amplitudes


def test_sigma_target_accept_is_rejected_with_fixed_amplitudes(monkeypatch):
    cached = _fixed_co2()
    with pytest.raises(ValueError, match="no amplitude step"):
        co2_cached_sigma_runner._sampler_for_cached_graph(
            RhimeSampler(nuts_sampler="pymc"),
            cached_model=cached,
            sigma_target_accept=0.8,
        )
    with pytest.raises(ValueError, match="not both"):
        co2_cached_sigma_runner._sampler_for_cached_graph(
            RhimeSampler(nuts_sampler="pymc", sample_kwargs={"target_accept": 0.9}),
            cached_model=cached,
            state_target_accept=0.9,
        )
    prepared = _prepared()
    from openghg_inversions.rhime.co2 import co2_o2_cached_sigma_runner

    monkeypatch.setattr(
        co2_o2_cached_sigma_runner,
        "_materialize_co2_o2_replay_inputs",
        lambda *args, **kwargs: pytest.fail("validated after loading inputs"),
    )
    with pytest.raises(ValueError, match="no amplitude step"):
        run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
            prepared_inputs=prepared,
            independent_error_sd=_independent_error(prepared),
            tau_hours=6.0,
            fixed_site_amplitudes=0.5,
            sigma_target_accept=0.8,
        )


def test_fixed_amplitudes_without_active_states_are_rejected():
    inputs = _inputs()
    cached = _fixed_co2(
        state_activity=StateActivity(
            active=xr.zeros_like(inputs["alpha_prior_mean"], dtype=bool),
        )
    )
    assert cached.fixed_amplitudes and not cached.states
    with pytest.raises(ValueError, match="no active states"):
        co2_cached_sigma_runner._sampler_for_cached_graph(
            RhimeSampler(nuts_sampler="pymc"), cached_model=cached
        )
    with pytest.raises(ValueError, match="no active states"):
        make_cached_sigma_compound_step(
            model=cached.model,
            sigma=None,
            states=(),
            modelled_mean=cached.modelled_mean,
            target=cached.target,
            shared_cache=cached.shared_cache,
            initial_cache=cached.initial_cache,
            cached_state_logp=cached.cached_state_logp,
        )
