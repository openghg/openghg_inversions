"""Per-state copula marginals on linked CO2/O2 prepared inputs and runners (#1099)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pymc as pm
import pytensor
import pytest
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedPrior
from openghg_inversions.models import StateActivity
from openghg_inversions.rhime.co2 import Co2O2PreparedInputs, build_co2_o2_model, prepare_co2_o2_inputs
from openghg_inversions.rhime.co2 import co2_o2_runner
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_runner import run_rhime_co2_o2_cached_sigma_from_prepared_inputs
from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2, run_rhime_co2_o2_from_prepared_inputs
from openghg_inversions.rhime.sampling import RhimeSampler

from test_co2_o2_fixed_ou import _prepared
from test_rhime_co2_o2 import _independent_error
from test_rhime_co2_o2_preparation import _durable_inputs

FAMILIES = ["weibull", "gamma", "weibull", "lognormal", "gamma"]


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _marginal(prior, families=FAMILIES) -> xr.DataArray:
    return prior.mean.copy(data=np.array(families, dtype=object))


def _with_marginals(prepared: Co2O2PreparedInputs, floor: float | None = None) -> Co2O2PreparedInputs:
    """The same preparation with Weibull/gamma marginals, validated as preparation would."""
    marginal = _marginal(prepared.retained_prior)
    CorrelatedPrior(
        prepared.retained_prior.mean, prepared.retained_prior.arithmetic_covariance,
        marginal=marginal, latent_correlation_floor=floor,
    )
    return replace(
        prepared,
        retained_prior_marginal=marginal.astype(str).rename("retained_prior_marginal"),
        latent_correlation_floor=floor,
    )


def _capture_prior(monkeypatch) -> list[CorrelatedPrior]:
    captured: list[CorrelatedPrior] = []
    original = co2_o2_runner._build_co2_o2_model

    def capture(**kwargs):
        captured.append(kwargs["retained_prior"])
        return original(**kwargs)

    monkeypatch.setattr(co2_o2_runner, "_build_co2_o2_model", capture)
    return captured


def test_default_linked_prior_graph_and_draws_are_unchanged(monkeypatch, tmp_path: Path) -> None:
    """Without marginals the runner's prior, graph and draws are the former from_lognormal ones."""
    prepared = prepare_co2_o2_inputs(**_durable_inputs())
    assert prepared.retained_prior_marginal is None and prepared.latent_correlation_floor is None
    tree = prepared.to_datatree()
    assert "marginal" not in tree["retained_prior"].dataset
    assert "latent_correlation_floor" not in tree["retained_prior"].attrs
    path = tmp_path / "linked.nc"
    prepared.save(path)
    restored = Co2O2PreparedInputs.load(path)
    assert restored.retained_prior_marginal is None and restored.latent_correlation_floor is None

    captured = _capture_prior(monkeypatch)
    error = _independent_error(prepared)
    built = build_rhime_co2_o2(prepared_inputs=prepared, independent_error_sd=error)
    (prior,) = captured
    former = CorrelatedPrior.from_lognormal(prepared.retained_prior)
    assert prior.is_lognormal
    for name in ("latent_mean", "latent_covariance", "latent_cholesky"):
        np.testing.assert_array_equal(getattr(prior.lognormal, name), getattr(former.lognormal, name))
    for name in ("latent_correlation_cholesky", "marginal_shape", "marginal_scale", "marginal"):
        np.testing.assert_array_equal(getattr(prior, name), getattr(former, name))

    reference = build_co2_o2_model(
        observations=prepared.observations,
        fixed_prior_contribution=prepared.fixed_prior_contribution,
        co2_sensitivity=prepared.co2_sensitivity,
        o2_sensitivity=prepared.o2_sensitivity,
        aggregation_error=prepared.aggregation_error,
        retained_prior=former,
        independent_error_sd=error,
    )
    graphs = [pytensor.dprint(model.logp(), file="str") for model in (built.model, reference)]
    assert graphs[0] == graphs[1]
    point = built.model.initial_point()
    rng = np.random.default_rng(1099)
    for _ in range(3):
        point = {name: value + rng.normal(scale=0.3, size=value.shape) for name, value in point.items()}
        for function in ("compile_logp", "compile_dlogp"):
            np.testing.assert_array_equal(
                getattr(built.model, function)()(point), getattr(reference, function)()(point)
            )
    draws = []
    for model in (built.model, reference):
        with model:
            draws.append(pm.sample_prior_predictive(draws=20, random_seed=1099, var_names=["flux_scaling"]))
    np.testing.assert_array_equal(draws[0].prior["flux_scaling"], draws[1].prior["flux_scaling"])
    metadata = built.output_contract.metadata
    assert metadata["prior"] == "correlated arithmetic-moment lognormal"
    assert metadata["prior_marginals"] == {"lognormal": 5}
    assert metadata["prior_latent_correlation_repair"] is None


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
@pytest.mark.parametrize("floor", [None, 1e-6])
def test_weibull_and_gamma_marginals_and_floor_survive_save_and_load(tmp_path: Path, suffix, floor) -> None:
    inputs = _durable_inputs()
    marginal = _marginal(inputs["retained_prior"])
    prepared = prepare_co2_o2_inputs(**inputs, prior_marginal=marginal, latent_correlation_floor=floor)
    assert list(prepared.retained_prior_marginal.values) == FAMILIES
    assert prepared.retained_prior_marginal.indexes["state"].equals(prepared.retained_prior.mean.indexes["state"])
    assert prepared.latent_correlation_floor == floor
    path = tmp_path / f"linked{suffix}"
    prepared.save(path)
    restored = Co2O2PreparedInputs.load(path)

    xr.testing.assert_identical(restored.retained_prior_marginal, prepared.retained_prior_marginal)
    assert restored.latent_correlation_floor == floor
    assert restored.retained_prior.mean.indexes["state"].names == ["source", "tracer_scope", "region_in_source"]
    xr.testing.assert_identical(restored.retained_prior.mean, prepared.retained_prior.mean)
    if floor is None:
        assert "latent_correlation_floor" not in prepared.to_datatree()["retained_prior"].attrs


def test_preparation_rejects_invalid_marginals_and_floor() -> None:
    inputs = _durable_inputs()
    prior = inputs["retained_prior"]
    with pytest.raises(ValueError, match="latent_correlation_floor applies only to copula priors"):
        prepare_co2_o2_inputs(**inputs, latent_correlation_floor=1e-6)
    with pytest.raises(ValueError, match="Unsupported state marginal"):
        prepare_co2_o2_inputs(**inputs, prior_marginal=_marginal(prior, ["weibull", "beta", *FAMILIES[2:]]))
    with pytest.raises(ValueError, match="state labels in the same order"):
        prepare_co2_o2_inputs(**inputs, prior_marginal=_marginal(prior).isel(state=[1, 0, 2, 3, 4]))
    with pytest.raises(ValueError, match=r"latent_correlation_floor must be in \(0, 0.1\]"):
        prepare_co2_o2_inputs(**inputs, prior_marginal=_marginal(prior), latent_correlation_floor=0.2)
    # A lognormal-feasible pair covariance that no Weibull pair reaches fails at preparation, not at sampling.
    sd = np.array([0.3, 1.0, 0.1, 0.1, 0.1]) * prior.mean.to_numpy()
    correlation = np.eye(5)
    correlation[0, 1] = correlation[1, 0] = 0.9
    inputs["retained_prior"] = type(prior)(prior.mean, correlation * np.outer(sd, sd))
    with pytest.raises(ValueError, match="No latent correlation reproduces"):
        prepare_co2_o2_inputs(**inputs, prior_marginal=_marginal(prior, ["weibull"] * 5))


def test_loader_rejects_a_floor_without_marginals() -> None:
    tree = prepare_co2_o2_inputs(**_durable_inputs()).to_datatree()
    tree["retained_prior"].attrs["latent_correlation_floor"] = 1e-6
    with pytest.raises(ValueError, match="latent_correlation_floor applies only to copula priors"):
        Co2O2PreparedInputs.from_datatree(tree)


@pytest.mark.parametrize("partial", [False, True])
def test_linked_runner_builds_the_copula_prior_with_partial_activity(monkeypatch, partial) -> None:
    prepared = _with_marginals(prepare_co2_o2_inputs(**_durable_inputs()), floor=1e-6)
    state_dim = prepared.retained_prior.state_dim
    activity = None
    if partial:
        activity = StateActivity(
            active=xr.DataArray(
                [True, False, True, True, True], dims=state_dim, coords={state_dim: prepared.retained_prior.mean[state_dim]}
            ),
            fixed_value=0.75,
        )
    captured = _capture_prior(monkeypatch)
    built = build_rhime_co2_o2(
        prepared_inputs=prepared, independent_error_sd=_independent_error(prepared), state_activity=activity
    )
    (prior,) = captured
    assert not prior.is_lognormal
    assert list(prior.marginal.values) == FAMILIES
    assert prior.latent_correlation_repair["floor"] == 1e-6
    metadata = built.output_contract.metadata
    assert metadata["prior"] == "correlated arithmetic-moment Gaussian copula"
    assert metadata["prior_marginals"] == {"gamma": 2, "lognormal": 1, "weibull": 2}
    assert metadata["prior_latent_correlation_repair"] == prior.latent_correlation_repair
    with built.model:
        draws = pm.sample_prior_predictive(draws=200, random_seed=7, var_names=["flux_scaling"]).prior["flux_scaling"]
    values = draws.values.reshape(-1, 5)
    assert np.all(values > 0)
    if partial:
        np.testing.assert_array_equal(values[:, 1], 0.75)
        assert [rv.name for rv in built.model.free_RVs] == ["flux_scaling_active_latent"]
        assert built.model.initial_point()["flux_scaling_active_latent"].shape == (4,)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_saved_marginals_sample_through_ordinary_and_cached_linked_runners(tmp_path: Path, cached, partial) -> None:
    """Smoke test: a reloaded Weibull/gamma preparation samples through both linked runners."""
    prepared = _with_marginals(_prepared())
    prepared = replace(prepared, independent_error_sd=_independent_error(prepared))
    path = tmp_path / "prepared.nc"
    prepared.save(path)
    prepared = Co2O2PreparedInputs.load(path)
    assert list(prepared.retained_prior_marginal.values) == FAMILIES
    state_dim = prepared.retained_prior.state_dim
    activity = None
    if partial:
        activity = StateActivity(
            active=xr.DataArray(
                [True, True, False, True, True], dims=state_dim, coords={state_dim: prepared.retained_prior.mean[state_dim]}
            ),
            fixed_value=0.75,
        )
    runner = run_rhime_co2_o2_cached_sigma_from_prepared_inputs if cached else run_rhime_co2_o2_from_prepared_inputs
    trace = runner(
        prepared_inputs=prepared,
        state_activity=activity,
        tau_hours={"co2:A": 6.0, "o2:A": 8.0, "o2:B": 2.0},
        site_amplitude_prior={"pdf": "uniform", "lower": 0.1, "upper": 1.5},
        sampler=RhimeSampler(
            nuts_sampler="pymc",
            draws=5,
            tune=5,
            chains=1,
            sample_kwargs={"cores": 1, "random_seed": 1099, "progressbar": False, "compute_convergence_checks": False},
        ),
    )
    flux = trace.posterior["flux_scaling"].values
    assert flux.shape[-1] == 5 and np.isfinite(flux).all() and (flux > 0).all()
    if partial:
        np.testing.assert_array_equal(flux[..., 2], 0.75)
    metadata = json.loads(trace.attrs["rhime_model_metadata"])
    assert metadata["prior"] == "correlated arithmetic-moment Gaussian copula"
    assert metadata["prior_marginals"] == {"gamma": 2, "lognormal": 1, "weibull": 2}
