"""Tests for the accepted-sigma cached PyMC compound sampler."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from openghg_inversions.models.cached_sigma import FixedOuCachedSigmaTarget
from openghg_inversions.models.fixed_ou import prepare_fixed_ou_low_rank
from openghg_inversions.rhime.cached_sigma import (
    PymcCachedSigmaNutsStep,
    PytensorMarginalQuadraticCache,
    make_cached_sigma_compound_step,
)


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


def _step_context(
    *, constructor_seed: int = 101, forbid_constructor_refresh: bool = False
):
    target = _target()
    with pm.Model() as model:
        state = pm.Normal("state", shape=1)
        sigma = pm.HalfNormal(
            "sigma_site", sigma=0.75, shape=2, initval=pm.floatX([0.4, 0.4])
        )
        initial_cache = target.refresh(np.array([0.4, 0.4]))
        shared = PytensorMarginalQuadraticCache(initial_cache)
        pm.Potential("cached_likelihood", shared.log_likelihood(pt.exp(state)))
    if forbid_constructor_refresh:
        target.refresh = lambda _: (_ for _ in ()).throw(
            AssertionError("Step construction must reuse the installed initial cache.")
        )
    point = model.initial_point()
    point["state"] = np.array([0.3])
    step = PymcCachedSigmaNutsStep(
        [sigma],
        target=target,
        shared_cache=shared,
        initial_cache=initial_cache,
        state_value_name="state",
        state_location=np.zeros(1),
        state_cholesky=np.eye(1),
        prior_scale=0.75,
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


def test_sigma_step_reuses_passed_initial_cache() -> None:
    _, _, initial_cache, _, _, _, _, step = _step_context(
        forbid_constructor_refresh=True
    )

    assert step.current_cache is initial_cache


def test_sigma_step_refreshes_once_only_when_the_accepted_value_changes(
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


def test_set_rng_reseeds_the_delegated_nuts_and_repeats_one_transition() -> None:
    *_, point1, step1 = _step_context(constructor_seed=102)
    *_, point2, step2 = _step_context(constructor_seed=999)

    step1.set_rng(np.random.default_rng(20260913))
    step2.set_rng(np.random.default_rng(20260913))
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
        state=state,
        target=target,
        shared_cache=shared,
        initial_cache=step.current_cache,
        state_value_name="state",
        state_location=np.zeros(1),
        state_cholesky=np.eye(1),
        prior_scale=0.75,
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
            "sigma_site", sigma=0.75, shape=2, initval=pm.floatX([0.4, 0.4])
        )
        initial_cache = target.refresh(np.array([0.4, 0.4]))
        shared = PytensorMarginalQuadraticCache(initial_cache)
        pm.Potential("cached_likelihood", shared.log_likelihood(pt.exp(state)))
        step = make_cached_sigma_compound_step(
            model=model,
            sigma=sigma,
            state=state,
            target=target,
            shared_cache=shared,
            initial_cache=initial_cache,
            state_value_name="state",
            state_location=np.zeros(1),
            state_cholesky=np.eye(1),
            prior_scale=0.75,
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
    assert first.posterior.sizes == {"chain": 2, "draw": 4, "state_dim_0": 1, "sigma_site_dim_0": 2}
    assert "sigma_nuts_tree_steps" in first.sample_stats
    assert "cache_refreshes" in first.sample_stats
    assert np.isfinite(first.posterior["sigma_site"]).all()
