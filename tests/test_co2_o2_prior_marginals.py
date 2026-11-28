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

from openghg_inversions import correlated_state
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


def _copula(prior: CorrelatedPrior, floor: float | None = None) -> CorrelatedPrior:
    """The same moments with Weibull/gamma marginals."""
    return CorrelatedPrior(
        prior.mean, prior.arithmetic_covariance, marginal=_marginal(prior), latent_correlation_floor=floor
    )


def _with_marginals(prepared: Co2O2PreparedInputs, floor: float | None = None) -> Co2O2PreparedInputs:
    """The same preparation with Weibull/gamma marginals."""
    return replace(prepared, retained_prior=_copula(prepared.retained_prior, floor))


def _capture_prior(monkeypatch) -> list[CorrelatedPrior]:
    captured: list[CorrelatedPrior] = []
    original = co2_o2_runner._build_co2_o2_model

    def capture(**kwargs):
        captured.append(kwargs["retained_prior"])
        return original(**kwargs)

    monkeypatch.setattr(co2_o2_runner, "_build_co2_o2_model", capture)
    return captured


def test_default_linked_prior_graph_and_draws_are_unchanged(monkeypatch, tmp_path: Path) -> None:
    """Without marginals the runner samples the prepared all-lognormal prior with the direct graph."""
    prepared = prepare_co2_o2_inputs(**_durable_inputs())
    assert prepared.retained_prior.is_lognormal
    tree = prepared.to_datatree()
    assert {"marginal", "latent_correlation"}.isdisjoint(tree["retained_prior"].dataset)
    path = tmp_path / "linked.nc"
    prepared.save(path)
    restored = Co2O2PreparedInputs.load(path)
    assert restored.retained_prior.is_lognormal

    captured = _capture_prior(monkeypatch)
    error = _independent_error(prepared)
    built = build_rhime_co2_o2(prepared_inputs=prepared, independent_error_sd=error)
    (prior,) = captured
    former = prepared.retained_prior
    assert prior is former

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
def test_weibull_and_gamma_marginals_and_floor_survive_save_and_load(
    monkeypatch, tmp_path: Path, suffix, floor
) -> None:
    """The saved copula prior reloads with its solved latent correlation, without solving it again."""
    inputs = _durable_inputs()
    inputs["retained_prior"] = _copula(inputs["retained_prior"], floor)
    prepared = prepare_co2_o2_inputs(**inputs)
    assert prepared.retained_prior is inputs["retained_prior"]
    path = tmp_path / f"linked{suffix}"
    prepared.save(path)

    def unexpected_solve(*args, **kwargs):
        raise AssertionError("Loading must reuse the saved latent correlation.")

    monkeypatch.setattr(correlated_state, "_solve_latent_correlation", unexpected_solve)
    restored = Co2O2PreparedInputs.load(path)

    original, prior = prepared.retained_prior, restored.retained_prior
    assert list(prior.marginal.values) == FAMILIES
    assert prior.mean.indexes["state"].names == ["source", "tracer_scope", "region_in_source"]
    xr.testing.assert_identical(prior.mean, original.mean)
    np.testing.assert_array_equal(prior.arithmetic_covariance, original.arithmetic_covariance)
    np.testing.assert_array_equal(prior.latent_correlation, original.latent_correlation)
    np.testing.assert_array_equal(prior.latent_correlation_cholesky, original.latent_correlation_cholesky)
    assert prior.latent_correlation_repair == original.latent_correlation_repair
    assert (prior.latent_correlation_repair is None) == (floor is None)


@pytest.mark.parametrize(
    ("tamper", "match"),
    [("drop", "lacks its latent correlation; prepare again"), ("perturb", "does not reproduce")],
)
def test_loader_rejects_a_missing_or_inconsistent_latent_correlation(tamper, match) -> None:
    inputs = _durable_inputs()
    inputs["retained_prior"] = _copula(inputs["retained_prior"])
    tree = prepare_co2_o2_inputs(**inputs).to_datatree()
    node = tree["retained_prior"].to_dataset()
    if tamper == "drop":
        node = node.drop_vars("latent_correlation")
    else:
        correlation = node["latent_correlation"].values.copy()
        correlation[0, 1] = correlation[1, 0] = correlation[0, 1] + 0.05
        node["latent_correlation"] = node["latent_correlation"].copy(data=correlation)
    tree["retained_prior"] = xr.DataTree(node)
    with pytest.raises(ValueError, match=match):
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


@pytest.mark.parametrize("route", ["ordinary", "cached", "fixed_cached"])
@pytest.mark.parametrize("partial", [False, True])
def test_saved_marginals_sample_through_ordinary_and_cached_linked_runners(tmp_path: Path, route, partial) -> None:
    """Smoke test: a reloaded Weibull/gamma preparation samples through both linked runners."""
    prepared = _with_marginals(_prepared(), floor=1e-6)
    prepared = replace(prepared, independent_error_sd=_independent_error(prepared))
    path = tmp_path / "prepared.nc"
    prepared.save(path)
    prepared = Co2O2PreparedInputs.load(path)
    assert list(prepared.retained_prior.marginal.values) == FAMILIES
    state_dim = prepared.retained_prior.state_dim
    activity = None
    if partial:
        activity = StateActivity(
            active=xr.DataArray(
                [True, True, False, True, True], dims=state_dim, coords={state_dim: prepared.retained_prior.mean[state_dim]}
            ),
            fixed_value=0.75,
        )
    if route == "cached":
        runner = run_rhime_co2_o2_cached_sigma_from_prepared_inputs
    else:
        runner = run_rhime_co2_o2_from_prepared_inputs
    amplitudes = (
        {"fixed_site_amplitudes": 0.5}
        if route == "fixed_cached"
        else {"site_amplitude_prior": {"pdf": "uniform", "lower": 0.1, "upper": 1.5}}
    )
    trace = runner(
        prepared_inputs=prepared,
        state_activity=activity,
        tau_hours={"co2:A": 6.0, "o2:A": 8.0, "o2:B": 2.0},
        sampler=RhimeSampler(
            nuts_sampler="pymc",
            draws=5,
            tune=5,
            chains=1,
            sample_kwargs={"cores": 1, "random_seed": 1099, "progressbar": False, "compute_convergence_checks": False},
        ),
        **amplitudes,
    )
    flux = trace.posterior["flux_scaling"].values
    assert flux.shape[-1] == 5 and np.isfinite(flux).all() and (flux > 0).all()
    if partial:
        np.testing.assert_array_equal(flux[..., 2], 0.75)
    metadata = json.loads(trace.attrs["rhime_model_metadata"])
    assert metadata["prior"] == "correlated arithmetic-moment Gaussian copula"
    assert metadata["prior_marginals"] == {"gamma": 2, "lognormal": 1, "weibull": 2}
    assert metadata["prior_latent_correlation_repair"]["floor"] == 1e-6
    assert metadata["recipe"] == ("co2_o2_cached_sigma_fixed_ou" if route == "cached" else "co2_o2")
    if route == "fixed_cached":
        actual = json.loads(trace.attrs["rhime_actual_inference"])
        assert actual["algorithm"] == "cached_fixed_ou_state_nuts"
