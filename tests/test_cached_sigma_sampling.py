"""Tests for the cached sigma-then-state PyMC compound sampler."""

from __future__ import annotations

from copy import deepcopy
import pickle
import warnings

import numpy as np
import pymc as pm
from pymc.pytensorf import floatX
import pytensor
import pytensor.tensor as pt
import pytest
from pymc.blocking import DictToArrayBijection
from pytensor.compile.mode import Mode, get_mode
from scipy.stats import multivariate_normal

from openghg_inversions.models.cached_sigma import FixedOuCachedSigmaTarget
from openghg_inversions.models.fixed_ou import FixedOuLowRank, prepare_fixed_ou_low_rank
from openghg_inversions.inference.cached_sigma import (
    PymcCachedSigmaNutsStep,
    PytensorMarginalQuadraticCache,
    _PytensorSigmaLikelihoodOp,
    make_cached_sigma_compound_step,
)


@pytest.fixture(params=["float32", "float64"])
def graph_precision(request):
    with pytensor.config.change_flags(floatX=request.param):
        yield request.param


def _target() -> FixedOuCachedSigmaTarget:
    prepared = prepare_fixed_ou_low_rank(
        factor=np.array([[0.20], [-0.10], [0.15], [0.05]]),
        diagonal_variance=np.array([0.12, 0.18, 0.16, 0.20]),
        observation_times=np.array(
            [
                "2020-01-01T00",
                "2020-01-01T00",
                "2020-01-01T01",
                "2020-01-01T02",
            ],
            dtype="datetime64[h]",
        ),
        site_index=np.array([0, 1, 0, 1]),
        tau_hours=np.array([5.0, 7.0]),
        site_labels=("MHD", "TAC"),
    )
    return FixedOuCachedSigmaTarget(
        prepared=prepared,
        observations=np.array([0.8, -0.2, 1.1, 0.4]),
        fixed_contribution=np.array([0.1, 0.0, -0.1, 0.2]),
        design=np.array([[0.7], [0.2], [0.9], [-0.3]]),
    )


def _dense_covariance(target, amplitude):
    times = np.array([0.0, 0.0, 1.0, 2.0])
    covariance = target.prepared.factor @ target.prepared.factor.T + np.diag(target.prepared.diagonal_variance)
    for site, tau in enumerate([5.0, 7.0]):
        rows = np.flatnonzero(target.prepared.site_index == site)
        covariance[np.ix_(rows, rows)] += amplitude[site] ** 2 * np.exp(
            -np.abs(times[rows, None] - times[None, rows]) / tau
        )
    return covariance


def _step_context(
    *, constructor_seed: int = 101, forbid_constructor_refresh: bool = False
):
    target = _target()
    with pm.Model() as model:
        state = pm.Normal("state", shape=1)
        sigma = pm.HalfNormal(
            "sigma_site", sigma=0.75, shape=2, initval=floatX([0.4, 0.4])
        )
        initial_cache = target.refresh(np.array([0.4, 0.4]))
        shared = PytensorMarginalQuadraticCache(initial_cache)
        coefficients = pt.exp(state)
        modelled_mean = pm.Deterministic(
            "modelled_mean",
            target.fixed_contribution + pt.dot(target.design, coefficients),
        )
        pm.Potential("cached_likelihood", shared.log_likelihood(coefficients))
    if forbid_constructor_refresh:
        target.refresh = lambda _: (_ for _ in ()).throw(
            AssertionError("Step construction must reuse the installed initial cache.")
        )
    point = model.initial_point()
    point["state"] = np.array([0.3])
    step = PymcCachedSigmaNutsStep(
        sigma,
        target=target,
        shared_cache=shared,
        initial_cache=initial_cache,
        modelled_mean=modelled_mean,
        initial_point=point,
        model=model,
        max_treedepth=4,
        early_max_treedepth=3,
        rng=np.random.default_rng(constructor_seed),
    )
    return target, shared, initial_cache, model, state, sigma, point, step


def _nuts_stats() -> list[dict[str, object]]:
    return [
        {
            "tree_size": 1,
            "depth": 1,
            "diverging": False,
            "divergences": 0,
            "energy_error": 0.0,
            "max_energy_error": 0.0,
            "reached_max_treedepth": False,
            "index_in_trajectory": 1,
            "mean_tree_accept": 1.0,
            "step_size": 0.1,
        }
    ]


def _observed_context(*, observed=True):
    target = _target()
    initial_cache = target.refresh([0.4, 0.4])
    shared = PytensorMarginalQuadraticCache(initial_cache)
    with pm.Model() as model:
        state = pm.Normal("state", shape=1)
        sigma = pm.HalfNormal("sigma_site", sigma=0.75, shape=2)
        coefficients = pt.exp(state)
        mean = pm.Deterministic("modelled_mean", target.fixed_contribution + target.design @ coefficients)
        if observed:
            pm.CustomDist(
                "y", mean, sigma, logp=target.prepared.logp, random=target.prepared.random,
                signature="(n),(s)->(n)", observed=target.observations,
            )
        else:
            pm.Potential("cached_likelihood", shared.log_likelihood(coefficients))
    point = model.initial_point()
    point["sigma_site_log__"] = np.asarray(np.log([0.4, 0.4]), dtype=point["sigma_site_log__"].dtype)
    arguments = dict(
        model=model, sigma=sigma, states=[state], modelled_mean=mean, target=target,
        shared_cache=shared, initial_cache=initial_cache,
        initial_point=point, rng=np.random.default_rng(769),
    )
    return target, shared, model, point, shared.log_likelihood(coefficients), arguments


def test_observed_graph_requires_explicit_cached_expression():
    *_, arguments = _observed_context()
    with pytest.raises(ValueError, match="explicit cached_state_logp"):
        make_cached_sigma_compound_step(**arguments)


@pytest.mark.compiled_backend
def test_amplitude_backend_preserves_state_and_runtime_configuration():
    configuration = (pytensor.config.mode, pytensor.config.linker, pytensor.config.floatX)
    *_, quadratic, arguments = _observed_context()
    compound = make_cached_sigma_compound_step(**arguments, cached_state_logp=quadratic)
    amplitude_step, state_step = compound.methods
    amplitude_linker = amplitude_step.nuts_step._logp_dlogp_func._pytensor_function.maker.mode.linker
    state_linker = state_step._logp_dlogp_func._pytensor_function.maker.mode.linker

    assert type(amplitude_linker) is type(Mode(linker="cvm", optimizer="fast_run").linker)
    assert amplitude_linker.use_cloop
    assert type(state_linker) is type(get_mode().linker)
    assert configuration == (pytensor.config.mode, pytensor.config.linker, pytensor.config.floatX)


@pytest.mark.compiled_backend
def test_supplied_state_function_matches_public_density_without_covariance_work(monkeypatch):
    target, shared, model, point, quadratic, arguments = _observed_context()
    compound = make_cached_sigma_compound_step(**arguments, cached_state_logp=quadratic)
    sigma_step, state_step = compound.methods
    exact = model.compile_logp()
    gradient = model.compile_dlogp(vars=arguments["states"])
    # Cached float32 algebra and the public Gaussian gradient round in different orders.
    gradient_rtol = max(1e-7, 2 * np.finfo(point["state"].dtype).eps)
    for state, amplitudes in ((-0.3, [0.2, 0.7]), (0.4, [0.6, 1.2])):
        point["state"] = np.asarray([state], dtype=point["state"].dtype)
        point["sigma_site_log__"] = np.asarray(np.log(amplitudes), dtype=point["sigma_site_log__"].dtype)
        shared.update(target.refresh(sigma_step.physical_sigma_fn(point["sigma_site_log__"])))
        state_step._logp_dlogp_func.set_extra_values(point)
        actual, actual_gradient = state_step._logp_dlogp_func(DictToArrayBijection.map({"state": point["state"]}))
        np.testing.assert_allclose(actual, exact(point), atol=1e-10)
        np.testing.assert_allclose(actual_gradient, gradient(point), rtol=gradient_rtol, atol=1e-10)

    def forbidden(*args, **kwargs):
        pytest.fail("A cached state transition must not evaluate or factor observation covariance.")

    monkeypatch.setattr(FixedOuLowRank, "evaluate", forbidden)
    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    updated, stats = state_step.step(point)
    assert np.isfinite(updated["state"]).all()
    assert stats[0]["tree_size"] > 0


def _sample_observed_spawn_chains(*, observed=True):
    target, shared, model, point, quadratic, arguments = _observed_context(observed=observed)
    step = make_cached_sigma_compound_step(**arguments, cached_state_logp=quadratic if observed else None)
    starts = [
        {
            "state": np.asarray([state], dtype=point["state"].dtype),
            "sigma_site_log__": np.asarray(np.log(amplitudes), dtype=point["sigma_site_log__"].dtype),
        }
        for state, amplitudes in ((-0.2, [0.3, 0.5]), (0.3, [0.6, 0.7]))
    ]
    with model:
        trace = pm.sample(
            draws=4, tune=4, chains=2, cores=2, mp_ctx="spawn", step=step,
            initvals=starts, random_seed=769, progressbar=False, compute_convergence_checks=False,
            idata_kwargs={"log_likelihood": observed},
        )
    return target, shared, model, point, trace


@pytest.mark.compiled_backend
def test_observed_spawn_chains_preserve_shared_cache_and_generic_joint_density():
    target, shared, model, point, first = _sample_observed_spawn_chains()
    *_, second = _sample_observed_spawn_chains()
    *_, baseline = _sample_observed_spawn_chains(observed=False)
    for name in ("state", "sigma_site", "modelled_mean"):
        np.testing.assert_array_equal(first.posterior[name], second.posterior[name])
        np.testing.assert_allclose(first.posterior[name], baseline.posterior[name], atol=1e-8, rtol=1e-8)
    assert not np.array_equal(first.posterior.state[0], first.posterior.state[1])
    assert int(first.sample_stats.cache_refreshes.sum()) > 0
    shared.update(target.refresh([3.0, 3.0]))
    exact = model.compile_logp(vars=[model["y"]])
    expected_values = np.empty((2, 4))
    for chain in range(2):
        for draw in range(4):
            amplitude = first.posterior.sigma_site.values[chain, draw]
            covariance = _dense_covariance(target, amplitude)
            expected = multivariate_normal.logpdf(
                target.observations, mean=first.posterior.modelled_mean.values[chain, draw], cov=covariance,
            )
            expected_values[chain, draw] = expected
            point["state"] = first.posterior.state.values[chain, draw]
            point["sigma_site_log__"] = np.log(amplitude)
            np.testing.assert_allclose(exact(point), expected, atol=1e-9)
    assert first.log_likelihood.y.dims == ("chain", "draw")
    np.testing.assert_allclose(first.log_likelihood.y, expected_values, atol=1e-9)
    del first["log_likelihood"]
    pm.compute_log_likelihood(first, model=model, progressbar=False)
    np.testing.assert_allclose(first.log_likelihood.y, expected_values, atol=1e-9)


def test_sigma_step_reuses_passed_initial_cache() -> None:
    _, _, initial_cache, _, _, _, _, step = _step_context(
        forbid_constructor_refresh=True
    )

    assert step.current_cache is initial_cache


def test_sigma_likelihood_transforms_once_per_installed_residual(monkeypatch) -> None:
    target = _target()
    op = _PytensorSigmaLikelihoodOp(target)
    sigma = pt.dvector("sigma")
    evaluate = pytensor.function([sigma], op(sigma, op.residual))
    residuals = [target.observations - target.fixed_contribution - target.design[:, 0] * state
                 for state in (0.7, 1.3)]
    amplitudes = [np.array([0.3, 0.8]), np.array([0.5, 0.2])]
    expected = [[target.evaluate_from_residual(residual, amplitude) for amplitude in amplitudes]
                for residual in residuals]
    forward_calls = 0
    backward_calls = 0
    transform_type = type(target.prepared.mode_transform)
    original_transform = transform_type.__matmul__

    def counted_transform(matrix, value):
        nonlocal forward_calls
        forward_calls += 1
        return original_transform(matrix, value)

    def unexpected_backward(*args, **kwargs):
        nonlocal backward_calls
        backward_calls += 1
        pytest.fail("Amplitude trajectories must not compute the observation-residual gradient.")

    monkeypatch.setattr(transform_type, "__matmul__", counted_transform)
    monkeypatch.setattr(type(target.prepared.mode_transform.T), "__matmul__", unexpected_backward)
    for index, (residual, references) in enumerate(zip(residuals, expected, strict=True), start=1):
        borrowed = residual.copy()
        op.install_residual(borrowed)
        borrowed[:] = 0.0
        for amplitude, reference in zip(amplitudes, references, strict=True):
            logp, gradient = evaluate(amplitude)
            assert logp == reference.log_likelihood
            np.testing.assert_array_equal(gradient, reference.gradient_site_amplitude)
        assert forward_calls == index
        assert backward_calls == 0


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, np.finfo(np.float64).max])
def test_sigma_likelihood_install_rejects_nonfinite_or_overflowing_residual_safely(invalid) -> None:
    target = _target()
    op = _PytensorSigmaLikelihoodOp(target)
    residual = np.array([invalid, -invalid, -invalid, invalid])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        op.install_residual(residual)
        outputs = [[None], [None]]
        op.perform(None, [np.array([0.3, 0.8]), op.residual.get_value()], outputs)
    assert outputs[0][0] == -np.inf
    np.testing.assert_array_equal(outputs[1][0], np.zeros(target.n_group))


@pytest.mark.parametrize("clone", [deepcopy, lambda op: pickle.loads(pickle.dumps(op))])
def test_copied_sigma_likelihood_keeps_mode_residual_chain_local(clone) -> None:
    target = _target()
    original = _PytensorSigmaLikelihoodOp(target)
    original.install_residual(np.array([0.1, 0.2, 0.3, 0.4]))
    original_modes = original.residual.get_value().copy()
    copied = clone(original)
    copied.install_residual(np.array([0.4, 0.3, 0.2, 0.1]))

    np.testing.assert_array_equal(original.residual.get_value(), original_modes)
    assert not np.shares_memory(original.residual.get_value(borrow=True), copied.residual.get_value(borrow=True))
    sigma = pt.dvector("sigma")
    for op, residual in ((original, [0.1, 0.2, 0.3, 0.4]), (copied, [0.4, 0.3, 0.2, 0.1])):
        evaluate = pytensor.function([sigma], op(sigma, op.residual))
        logp, gradient = evaluate(np.array([0.3, 0.8]))
        expected = target.evaluate_from_residual(residual, [0.3, 0.8])
        assert logp == expected.log_likelihood
        np.testing.assert_array_equal(gradient, expected.gradient_site_amplitude)


def test_sigma_step_reinstalls_modes_when_conditioning_state_changes(monkeypatch) -> None:
    target, _, _, _, _, _, point, step = _step_context()
    sigma = np.exp(point[step.sigma_value_name])
    recorded = []

    def unchanged_transition(conditional_point):
        outputs = [[None], [None]]
        step.likelihood_op.perform(None, [sigma, step.likelihood_op.residual.get_value()], outputs)
        recorded.append(float(outputs[0][0]))
        return conditional_point, _nuts_stats()

    monkeypatch.setattr(step.nuts_step, "step", unchanged_transition)
    for state in (0.3, -0.4):
        point = {**point, "state": np.array([state])}
        _, stats = step.step(point)
        expected = target.log_likelihood(np.exp([state]), sigma)
        assert recorded[-1] == pytest.approx(expected, rel=1e-14, abs=1e-14)
        assert stats[0]["cache_refreshes"] == 0
    assert recorded[0] != recorded[1]


def test_sigma_step_refreshes_once_only_when_the_returned_value_changes(
    monkeypatch,
) -> None:
    target, shared, _, _, _, _, point, step = _step_context()
    proposed = point["sigma_site_log__"] + np.array([0.05, -0.03])
    original_refresh = target.refresh
    refreshes = 0

    def counted_refresh(sigma):
        nonlocal refreshes
        refreshes += 1
        return original_refresh(sigma)

    monkeypatch.setattr(target, "refresh", counted_refresh)
    monkeypatch.setattr(
        step.nuts_step,
        "step",
        lambda _: ({"sigma_site_log__": proposed.copy()}, _nuts_stats()),
    )

    updated, stats = step.step(point)

    assert refreshes == 1
    assert stats[0]["cache_refreshes"] == 1
    expected_point = np.asarray(
        proposed,
        dtype=point["sigma_site_log__"].dtype,
    )
    np.testing.assert_array_equal(updated["sigma_site_log__"], expected_point)
    expected = original_refresh(np.exp(proposed))
    np.testing.assert_allclose(shared.linear.get_value(), expected.linear, rtol=2e-6)

    monkeypatch.setattr(
        step.nuts_step,
        "step",
        lambda point: ({"sigma_site_log__": point["sigma_site_log__"].copy()}, _nuts_stats()),
    )
    _, unchanged_stats = step.step(updated)
    assert refreshes == 1
    assert unchanged_stats[0]["cache_refreshes"] == 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.compiled_backend
def test_shared_quadratic_uses_active_pytensor_dtype(dtype: str) -> None:
    target = _target()
    cache = target.refresh(np.array([0.4, 0.4]))

    with pytensor.config.change_flags(floatX=dtype):
        shared = PytensorMarginalQuadraticCache(cache)
        state = pt.vector("state", dtype=dtype)
        evaluate = pytensor.function([state], shared.log_likelihood(state))
        value = evaluate(np.array([0.3], dtype=dtype))

    assert shared.constant.dtype == dtype
    assert shared.linear.dtype == dtype
    assert shared.precision.dtype == dtype
    assert np.asarray(value).dtype == np.dtype(dtype)
    assert float(value) == pytest.approx(
        cache.log_likelihood(np.array([0.3])),
        rel=2e-6 if dtype == "float32" else 1e-12,
    )


def test_setup_chain_reseeds_the_delegated_nuts_and_repeats_one_transition() -> None:
    """PyMC chain setup gives the wrapper and delegated NUTS independent RNGs."""
    *_, point1, step1 = _step_context(constructor_seed=102)
    *_, point2, step2 = _step_context(constructor_seed=999)

    step1.setup_chain(np.random.default_rng(20260913), tune=10, draws=20)
    step2.setup_chain(np.random.default_rng(20260913), tune=10, draws=20)
    updated1, stats1 = step1.step(point1)
    updated2, stats2 = step2.step(point2)

    np.testing.assert_array_equal(
        updated1["sigma_site_log__"],
        updated2["sigma_site_log__"],
    )
    assert stats1[0]["sigma_nuts_tree_steps"] == stats2[0]["sigma_nuts_tree_steps"]
    assert step1.nuts_step.rng is not step1.rng
    assert step2.nuts_step.rng is not step2.rng


def test_sigma_sampler_mutable_state_is_chain_local() -> None:
    *_, step1 = _step_context(constructor_seed=104)
    *_, step2 = _step_context(constructor_seed=105)
    step1.likelihood_op.install_residual(np.array([1.0, 0.0, 0.0, 0.0]))

    assert step1.likelihood_op is not step2.likelihood_op
    assert step1.nuts_step is not step2.nuts_step
    assert step1.conditional_model is not step2.conditional_model
    assert step1.nuts_step.potential is not step2.nuts_step.potential
    assert not np.array_equal(
        step1.likelihood_op.residual.get_value(),
        step2.likelihood_op.residual.get_value(),
    )


def test_compound_orders_sigma_before_state_without_state_refactorization(
    monkeypatch,
) -> None:
    target, shared, _, model, state, sigma, point, step = _step_context()
    compound = make_cached_sigma_compound_step(
        model=model,
        sigma=sigma,
        states=[state],
        modelled_mean=model["modelled_mean"],
        target=target,
        shared_cache=shared,
        initial_cache=step.current_cache,
        initial_point=point,
        rng=np.random.default_rng(703),
    )

    assert isinstance(compound.methods[0], PymcCachedSigmaNutsStep)
    monkeypatch.setattr(
        target,
        "refresh",
        lambda _: (_ for _ in ()).throw(
            AssertionError("State NUTS must not refresh the sigma cache.")
        ),
    )
    monkeypatch.setattr(
        target,
        "evaluate_from_residual",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("State NUTS must not evaluate the exact covariance.")
        ),
    )

    state_point = point.copy()
    state_point["state"] = state_point["state"].astype(
        model.rvs_to_values[state].dtype
    )
    compound.methods[1].step(state_point)


def _sample_two_spawn_chains(seed: int):
    target = _target()
    with pm.Model() as model:
        state = pm.Normal("state", shape=1)
        sigma = pm.HalfNormal(
            "sigma_site", sigma=0.75, shape=2, initval=floatX([0.4, 0.4])
        )
        initial_cache = target.refresh(np.array([0.4, 0.4]))
        shared = PytensorMarginalQuadraticCache(initial_cache)
        coefficients = pt.exp(state)
        modelled_mean = pm.Deterministic(
            "modelled_mean",
            target.fixed_contribution + pt.dot(target.design, coefficients),
        )
        pm.Potential("cached_likelihood", shared.log_likelihood(coefficients))
        step = make_cached_sigma_compound_step(
            model=model,
            sigma=sigma,
            states=[state],
            modelled_mean=modelled_mean,
            target=target,
            shared_cache=shared,
            initial_cache=initial_cache,
            rng=np.random.default_rng(700),
        )
        return pm.sample(
            draws=4,
            tune=4,
            chains=2,
            cores=2,
            mp_ctx="spawn",
            step=step,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )


def test_two_spawn_chains_are_reproducible_and_emit_cached_diagnostics() -> None:
    first = _sample_two_spawn_chains(20260913)
    second = _sample_two_spawn_chains(20260913)

    np.testing.assert_array_equal(first.posterior["state"], second.posterior["state"])
    np.testing.assert_array_equal(
        first.posterior["sigma_site"],
        second.posterior["sigma_site"],
    )
    assert not np.array_equal(
        first.posterior["sigma_site"].isel(chain=0),
        first.posterior["sigma_site"].isel(chain=1),
    )
    assert first.posterior.sizes == {
        "chain": 2,
        "draw": 4,
        "state_dim_0": 1,
        "sigma_site_dim_0": 2,
        "modelled_mean_dim_0": 4,
    }
    assert "sigma_nuts_tree_steps" in first.sample_stats
    assert "cache_refreshes" in first.sample_stats
    assert np.isfinite(first.posterior["sigma_site"]).all()


@pytest.mark.parametrize("family", ["halfnormal", "gamma", "uniform"])
def test_compound_sampler_matches_independent_dense_posterior_moments(family) -> None:
    """The composed amplitude/state transition targets the intended posterior."""
    import arviz as az
    from scipy.integrate import trapezoid
    from scipy import stats

    observations = np.array([0.9, -0.1, 0.7])
    design = np.array([[0.8], [0.3], [1.0]])
    factor = np.array([[0.25], [-0.1], [0.2]])
    diagonal_variance = np.array([0.15, 0.2, 0.12])
    times = np.array([0.0, 1.0, 2.0])
    tau_hours = 4.0
    prior_scale = 0.75
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

    state_grid = np.linspace(-4.0, 3.0, 700)
    amplitude_grid = np.linspace(0.1, 1.5, 600) if family == "uniform" else np.linspace(1.0e-4, 3.5, 600)
    distribution = {
        "halfnormal": stats.halfnorm(scale=prior_scale),
        "gamma": stats.gamma(a=2.0, scale=0.4),
        "uniform": stats.uniform(loc=0.1, scale=1.4),
    }[family]
    physical_state_grid = np.exp(state_grid)
    log_weights = np.empty((amplitude_grid.size, state_grid.size))
    correlation = np.exp(-np.abs(times[:, None] - times[None, :]) / tau_hours)
    fixed_covariance = factor @ factor.T + np.diag(diagonal_variance)
    residual = observations[:, None] - design @ physical_state_grid[None, :]
    for index, amplitude in enumerate(amplitude_grid):
        covariance = fixed_covariance + amplitude**2 * correlation
        cholesky = np.linalg.cholesky(covariance)
        solved = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, residual))
        log_weights[index] = -0.5 * (
            np.square(state_grid)
            - 2.0 * distribution.logpdf(amplitude)
            + 2.0 * np.log(np.diag(cholesky)).sum()
            + np.sum(residual * solved, axis=0)
        )
    weights = np.exp(log_weights - log_weights.max())
    normalization = trapezoid(
        trapezoid(weights, state_grid, axis=1),
        amplitude_grid,
    )
    expected_state = trapezoid(
        trapezoid(weights * physical_state_grid, state_grid, axis=1),
        amplitude_grid,
    ) / normalization
    expected_amplitude = trapezoid(
        trapezoid(weights * amplitude_grid[:, None], state_grid, axis=1),
        amplitude_grid,
    ) / normalization

    initial_cache = target.refresh(np.array([0.4]))
    shared = PytensorMarginalQuadraticCache(initial_cache)
    with pm.Model() as model:
        state_white = pm.Normal("state_white", shape=1)
        state = pm.Deterministic("state", pt.exp(state_white))
        from openghg_inversions.models.priors import parse_prior
        prior = {
            "halfnormal": {"pdf": "halfnormal", "sigma": prior_scale},
            "gamma": {"pdf": "gamma", "alpha": 2.0, "beta": 2.5},
            "uniform": {"pdf": "uniform", "lower": 0.1, "upper": 1.5},
        }[family]
        amplitude = parse_prior("site_amplitude", prior, shape=1, initval=floatX([0.4]))
        pm.Potential("cached_likelihood", shared.log_likelihood(state))
        modelled_mean = pm.Deterministic(
            "modelled_mean",
            target.fixed_contribution + pt.dot(target.design, state),
        )
        step = make_cached_sigma_compound_step(
            model=model,
            sigma=amplitude,
            states=[state_white],
            modelled_mean=modelled_mean,
            target=target,
            shared_cache=shared,
            initial_cache=initial_cache,
            sigma_target_accept=0.9,
            state_target_accept=0.9,
            rng=np.random.default_rng(710),
        )
        idata = pm.sample(
            # The posterior means need enough effective samples to resolve 0.05.
            draws=3200,
            tune=800,
            chains=2,
            cores=1,
            step=step,
            random_seed=20260914,
            progressbar=False,
            compute_convergence_checks=False,
        )

    for name, expected in [("state", expected_state), ("site_amplitude", expected_amplitude)]:
        mcse = float(az.mcse(idata, var_names=[name])[name].item())
        assert np.isfinite(mcse) and mcse <= 0.0125, (name, mcse)
        assert float(az.rhat(idata, var_names=[name])[name].item()) < 1.01
        error = abs(float(idata.posterior[name].mean()) - expected)
        # Finer, expanded quadrature changes these oracle means by < 0.0002.
        assert error <= min(0.05, 4.0 * mcse + 0.0002), (name, error, mcse)
    assert int(np.asarray(idata.sample_stats["diverging"]).sum()) == 0


@pytest.mark.parametrize("prior", [
    {"pdf": "halfnormal", "sigma": 0.75},
    {"pdf": "halfstudentt", "nu": 4.0, "sigma": 0.75},
    {"pdf": "exponential", "lam": 1.5},
    {"pdf": "gamma", "alpha": 2.0, "beta": 3.0},
    {"pdf": "lognormal", "mean": 0.7, "stdev": 0.3, "reparameterise": True},
    {"pdf": "uniform", "lower": 0.1, "upper": 1.5},
])
def test_conditional_density_and_gradient_match_independent_positive_prior(prior, graph_precision):
    from scipy import stats
    from scipy.special import expit
    from openghg_inversions.models.priors import parse_prior

    target = _target()
    initial = np.array([0.4, 0.7], dtype=graph_precision)
    initial_cache = target.refresh(initial)
    shared = PytensorMarginalQuadraticCache(initial_cache)
    prior = dict(prior)
    # The matched recipe uses the original LogNormal RV and its log transform.
    prior["reparameterise"] = False
    # Keep scalar parameter constants at the selected precision, including folded log normalizers.
    prior = {key: pt.constant(value, dtype=graph_precision) if key not in {"pdf", "reparameterise"} else value
             for key, value in prior.items()}
    with pm.Model() as model:
        state = pm.Normal("state", shape=1)
        amplitude = parse_prior("sigma_site", prior, shape=2)
        model.set_initval(amplitude, initial)
        mean = target.fixed_contribution + target.design @ pt.exp(state)
    point = model.initial_point()
    step = PymcCachedSigmaNutsStep(
        amplitude, target=target, shared_cache=shared, initial_cache=initial_cache,
        modelled_mean=mean, initial_point=point, model=model,
    )
    residual = target.observations - target.fixed_contribution - target.design[:, 0]
    step.likelihood_op.install_residual(residual)
    value_name = step.sigma_value_name
    assert point[value_name].dtype == graph_precision
    assert step.conditional_sigma.dtype == graph_precision
    density = step.nuts_step._logp_dlogp_func
    density.set_extra_values(point)
    family = prior["pdf"]
    if family == "halfnormal":
        distribution = stats.halfnorm(scale=0.75)
    elif family == "halfstudentt":
        distribution = stats.t(df=4.0, scale=0.75)
    elif family == "exponential":
        distribution = stats.expon(scale=1 / 1.5)
    elif family == "gamma":
        distribution = stats.gamma(a=2.0, scale=1 / 3.0)
    elif family == "lognormal":
        variance_log = np.log1p((0.3 / 0.7) ** 2)
        distribution = stats.lognorm(s=np.sqrt(variance_log), scale=np.exp(np.log(0.7) - variance_log / 2))
    else:
        distribution = stats.uniform(loc=0.1, scale=1.4)

    def expected(eta):
        eta = np.asarray(eta, dtype=np.float64)
        if family == "uniform":
            physical = 0.1 + 1.4 * expit(eta)
            jacobian = np.log(1.4) + np.log(expit(eta)) + np.log(expit(-eta))
        else:
            physical = np.exp(eta)
            jacobian = eta
        prior_density = distribution.logpdf(physical).sum()
        if family == "halfstudentt":
            prior_density += 2 * np.log(2.0)
        return multivariate_normal.logpdf(residual, cov=_dense_covariance(target, physical)) + prior_density + jacobian.sum()

    tolerance = max(1e-10, 16 * np.finfo(graph_precision).eps)
    for eta in (point[value_name], np.array([-0.7, 0.2]), np.array([1.0, -1.5])):
        eta = np.asarray(eta, dtype=graph_precision)
        actual, actual_gradient = density(DictToArrayBijection.map({value_name: eta}))
        np.testing.assert_allclose(actual, expected(eta), rtol=max(tolerance, 1e-7), atol=tolerance)
        gradient = np.array([
            (expected(eta + np.eye(2)[axis] * 1e-5) - expected(eta - np.eye(2)[axis] * 1e-5)) / 2e-5
            for axis in range(2)
        ])
        np.testing.assert_allclose(actual_gradient, gradient, rtol=max(tolerance, 1e-7), atol=max(tolerance, 1e-7))


@pytest.mark.parametrize("bounded", [False, True])
def test_prior_cache_uses_inverse_transform(monkeypatch, graph_precision, bounded):
    from scipy.special import expit

    target = _target()
    initial_cache = target.refresh([0.8, 0.8])
    shared = PytensorMarginalQuadraticCache(initial_cache)
    with pm.Model() as model:
        state = pm.Normal("state", shape=1)
        amplitude = (pm.Uniform("sigma_site", lower=0.1, upper=1.5, shape=2) if bounded else
                     pm.HalfNormal("sigma_site", sigma=0.75, shape=2, initval=floatX([0.8, 0.8])))
        mean = target.fixed_contribution + target.design @ state
    point = model.initial_point()
    step = PymcCachedSigmaNutsStep(
        amplitude, target=target, shared_cache=shared, initial_cache=initial_cache,
        modelled_mean=mean, initial_point=point, model=model,
    )
    name = step.sigma_value_name
    proposed = np.array([-2.0, 1.0], dtype=graph_precision)
    monkeypatch.setattr(step.nuts_step, "step", lambda _: ({name: proposed}, _nuts_stats()))
    updated, stats = step.step(point)
    physical = 0.1 + 1.4 * expit(proposed.astype(float)) if bounded else np.exp(proposed.astype(float))
    tolerance = max(1e-12, 8 * np.finfo(graph_precision).eps)
    np.testing.assert_allclose(step.current_cache.sigma, physical, rtol=tolerance)
    shared_logp = pytensor.function([state], shared.log_likelihood(state))
    for state_value in (-0.5, 1.2):
        coefficients = np.array([state_value], dtype=graph_precision)
        expected = multivariate_normal.logpdf(
            target.observations, mean=target.fixed_contribution + target.design @ coefficients,
            cov=_dense_covariance(target, physical),
        )
        np.testing.assert_allclose(step.current_cache.log_likelihood(coefficients), expected, rtol=tolerance)
        np.testing.assert_allclose(shared_logp(coefficients), expected, rtol=tolerance)
    np.testing.assert_array_equal(updated[name], proposed)
    assert updated[name].dtype == graph_precision
    assert stats[0]["cache_refreshes"] == 1
    _, stats = step.step(updated)
    assert stats[0]["cache_refreshes"] == 0


def test_conditional_rejects_hierarchical_amplitude_prior():
    target = _target()
    initial_cache = target.refresh([0.4, 0.4])
    with pm.Model() as model:
        state = pm.HalfNormal("scale")
        amplitude = pm.HalfNormal("sigma_site", sigma=state, shape=2)
        model.set_initval(amplitude, np.asarray([0.4, 0.4], dtype=amplitude.dtype))
        mean = target.fixed_contribution + target.design[:, 0] * state
    with pytest.raises(ValueError, match="independent priors"):
        PymcCachedSigmaNutsStep(
            amplitude, target=target, shared_cache=PytensorMarginalQuadraticCache(initial_cache),
            initial_cache=initial_cache, modelled_mean=mean, model=model,
        )
