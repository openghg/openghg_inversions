"""Deterministic CO2 backgrounds leave residual likelihoods and gradients unchanged."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr
from scipy.stats import multivariate_normal

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2.co2_model import build_co2_model
from openghg_inversions.rhime.co2.co2_cached_sigma_model import build_co2_cached_sigma_model


@pytest.fixture
def inputs():
    ds = xr.Dataset(
        {
            "H": (("nmeasure", "region"), [[.8, .1], [.4, .3], [.2, .7], [.5, .2]]),
            "fixed_prior_contribution": ("nmeasure", [.1, .2, -.1, .2]),
            "mf": ("nmeasure", [401.1, 402.0, 400.8, 401.4]),
            "mf_error": ("nmeasure", [.1, .2, .15, .12]),
            "supplied_baseline": ("nmeasure", [400., 401., 400., 400.5]),
            "aggregation_error_covariance": (("nmeasure", "nmeasure_cov"), .01*np.eye(4)+.002),
        },
        coords={"nmeasure": np.arange(4), "nmeasure_cov": np.arange(4),
                "region": ["inner_a", "inner_b"],
                "site": ("nmeasure", ["AAA", "AAA", "BBB", "BBB"]),
                "time": ("nmeasure", np.array(["2021-01-01T00", "2021-01-01T03", "2021-01-01T01", "2021-01-01T05"], dtype="datetime64[h]"))},
    )
    for name in ["mf", "mf_error", "supplied_baseline", "fixed_prior_contribution"]:
        ds[name].attrs["units"] = "ppm"
    return ds


def _build(inputs, *, cached, baseline=True, **kwargs):
    prior = CorrelatedLognormalPrior(
        xr.DataArray([1., 1.], dims="region", coords={"region": inputs.region}),
        xr.DataArray([[.08, .01], [.01, .06]], dims=("region", "region_cov"),
                     coords={"region": inputs.region, "region_cov": inputs.region.values}),
        covariance_dim="region_cov",
    )
    options = dict(retained_prior=prior, fixed_prior_contribution=inputs.fixed_prior_contribution,
                   observations=inputs.mf, observation_error=inputs.mf_error,
                   aggregation_error=resolve_aggregation_error(inputs, "dense"), **kwargs)
    if baseline:
        options["supplied_baseline"] = inputs.supplied_baseline
    ou = dict(tau_hours=3., site_amplitude_prior={"pdf": "halfnormal", "sigma": .75})
    if cached:
        return build_co2_cached_sigma_model(inputs.H, **options, **ou)
    builder = options.pop("likelihood_builder", add_fixed_ou_gaussian_likelihood)
    likelihood_options = options.pop("likelihood_kwargs", ou)
    return build_co2_model(inputs.H, **options,
                           likelihood_builder=builder, likelihood_kwargs=likelihood_options)


def _expressions(model, outputs):
    return model.compile_fn(model.replace_rvs_by_values(outputs), inputs=model.value_vars,
                            on_unused_input="ignore")


def test_absolute_subtracted_stock_cached_density_and_gradients(inputs):
    """Identical errors yield identical targets away from unit flux scalings."""
    with pytensor.config.change_flags(floatX="float64"):
        absolute = _build(inputs, cached=False)
        residual_inputs = inputs.assign(mf=(inputs.mf-inputs.supplied_baseline).assign_attrs(units="ppm"))
        subtracted = _build(residual_inputs, cached=False, baseline=False)
        cached = _build(inputs, cached=True)
        models = [absolute, subtracted, cached.model]
        likelihoods = [model.compile_logp(vars=model.observed_RVs) for model in models]
        gradients = [model.compile_dlogp() for model in models]
        mean = _expressions(absolute, [absolute["modelled_concentration"], absolute["co2_flux_contribution"]])
        cache_outputs = _expressions(cached.model, [cached.cached_state_logp])
        cache_gradient = _expressions(cached.model, [pt.grad(
            cached.model.replace_rvs_by_values([cached.cached_state_logp])[0],
            cached.model.rvs_to_values[cached.states[0]],
        )])
        residual_gradient = _expressions(subtracted, [pt.grad(
            subtracted.logp(vars=subtracted.observed_RVs),
            subtracted.rvs_to_values[subtracted["flux_scaling_latent"]],
        )])
        for latent, amplitude in [([-.3, .2], [.3, .4]), ([.25, -.1], [.7, .25])]:
            point = absolute.initial_point()
            point["flux_scaling_latent"] = np.asarray(latent)
            point["ou_site_amplitude_log__"] = np.log(amplitude)
            cached.shared_cache.update(cached.target.refresh(np.asarray(amplitude)))
            density = [fn(point) for fn in likelihoods]
            public_mean, flux_mean = mean(point)
            np.testing.assert_allclose(public_mean, inputs.supplied_baseline.values+flux_mean, atol=1e-12)
            np.testing.assert_allclose(density, density[0], atol=1e-10)
            np.testing.assert_allclose(cache_outputs(point)[0], density[0], atol=1e-10)
            np.testing.assert_allclose(gradients[0](point), gradients[1](point), atol=1e-9)
            np.testing.assert_allclose(gradients[0](point), gradients[2](point), atol=1e-9)
            # Independent dense OU oracle uses the inherited covariance unchanged.
            covariance = inputs.aggregation_error_covariance.values + np.diag(inputs.mf_error.values**2)
            for rows, sigma in [(np.array([0, 1]), amplitude[0]), (np.array([2, 3]), amplitude[1])]:
                times = inputs.time.values[rows].astype("datetime64[h]").astype(float)
                covariance[np.ix_(rows, rows)] += sigma**2*np.exp(-np.abs(times[:, None]-times)/3.)
            oracle = multivariate_normal.logpdf(inputs.mf.values, mean=public_mean, cov=covariance)
            np.testing.assert_allclose(density[0], oracle, atol=1e-10)
            np.testing.assert_allclose(cache_gradient(point)[0], residual_gradient(point)[0], atol=1e-9)
            numerical = []
            for index in range(2):
                plus = {key: value.copy() for key, value in point.items()}
                minus = {key: value.copy() for key, value in point.items()}
                plus["flux_scaling_latent"][index] += 1e-5
                minus["flux_scaling_latent"][index] -= 1e-5
                numerical.append((
                    multivariate_normal.logpdf(inputs.mf.values, mean=mean(plus)[0], cov=covariance)
                    - multivariate_normal.logpdf(inputs.mf.values, mean=mean(minus)[0], cov=covariance)
                ) / 2e-5)
            np.testing.assert_allclose(cache_gradient(point)[0], numerical, atol=2e-7, rtol=2e-7)
        assert len(cached.states) == 1
        assert "bc" not in cached.model and "offset" not in cached.model
        assert "supplied_baseline" not in [rv.name for rv in cached.model.free_RVs]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("option", ["boundary_sensitivity", "bc_prior", "offset_prior"])
def test_supplied_baseline_rejects_double_counted_components(inputs, cached, option):
    value = inputs.H.rename(region="bc_region") if option == "boundary_sensitivity" else {"pdf": "normal", "mu": 1., "sigma": .1}
    with pytest.raises(ValueError, match="supplied_baseline.*boundary or offset"):
        _build(inputs, cached=cached, **{option: value})


@pytest.mark.parametrize("cached", [False, True])
def test_supplied_baseline_requires_finite_concentrations(inputs, cached):
    inputs.supplied_baseline.values[1] = np.nan
    with pytest.raises(ValueError, match="supplied_baseline must be finite"):
        _build(inputs, cached=cached)


@pytest.mark.parametrize("cached", [False, True])
def test_runners_select_materialize_and_describe_supplied_background(inputs, monkeypatch, cached):
    """The recipe forwards the selected series and records its fixed scientific role."""
    from types import SimpleNamespace
    from openghg_inversions.rhime.co2 import co2_runner, co2_cached_sigma_runner

    inputs = inputs.assign(
        alpha_prior_mean=("region", [1., 1.]),
        alpha_prior_covariance=(("region", "region_cov"), [[.08, .01], [.01, .06]]),
    ).assign_coords(region_cov=inputs.region.values)
    prepared = SimpleNamespace(inv_inputs=inputs, rhime_inputs=object(), aggregation_error_mode="dense")
    module = co2_cached_sigma_runner if cached else co2_runner
    # Full contract validation belongs to the preparation helper, tested separately.
    monkeypatch.setattr(module, "co2_supplied_baseline_input_names", lambda *a, **k: ("supplied_baseline",))
    selected = []

    def materialize(_prepared, *, variable_names, **kwargs):
        selected.append(variable_names)
        if "prior_parameters" in kwargs:
            return inputs, kwargs["prior_parameters"]
        return inputs

    if cached:
        monkeypatch.setattr(module, "materialize_pymc_inputs", materialize)
        model = module.build_rhime_co2_cached_sigma(
            prepared_inputs=prepared, tau_hours=3.,
            site_amplitude_prior={"pdf": "halfnormal", "sigma": .75},
        )
        built = module.co2_cached_sigma_build_result(model, prepared)
    else:
        monkeypatch.setattr(module, "_materialize_pymc_inputs_and_priors", materialize)
        eager, priors = module.materialize_co2_inputs(prepared_inputs=prepared, no_model_error=True)
        built = module.build_co2_from_materialized(
            eager, prepared_inputs=prepared, priors=priors, no_model_error=True,
        )
    assert selected[0].count("supplied_baseline") == 1
    assert built.output_contract.variable_roles["baseline_concentration"] == "supplied_baseline_contribution"
    assert "bc" not in built.model and "offset" not in built.model
    evaluate = _expressions(built.model, [built.model["modelled_concentration"], built.model["co2_flux_contribution"]])
    absolute, flux = evaluate(built.model.initial_point())
    np.testing.assert_allclose(absolute-flux, inputs.supplied_baseline.values, atol=1e-10)


@pytest.mark.parametrize("cached", [False, True])
def test_supplied_baseline_draws_reconstruct_after_trace_reload(inputs, tmp_path, cached):
    """Saved prior/posterior components preserve the absolute concentration equation."""
    import json
    from types import SimpleNamespace
    import pymc as pm
    from openghg_inversions.rhime.co2.co2_outputs import reconstruct_co2_concentrations
    from openghg_inversions.serialization import save_trace, load_trace
    from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords

    result = _build(inputs, cached=cached)
    model = result.model if cached else result
    with model:
        trace = pm.sample_prior_predictive(
            draws=4, random_seed=139,
            var_names=["modelled_concentration", "co2_flux_contribution", "flux_scaling", "supplied_baseline_contribution"],
        )
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    # Different prior/posterior draw counts exercise reconstruction's sample axes.
    trace["posterior"] = xr.DataTree(trace.prior.to_dataset().isel(draw=slice(0, 2)))
    trace.attrs["rhime_variable_roles"] = json.dumps({
        "model_mean": "modelled_concentration", "pollution_concentration": "co2_flux_contribution",
        "flux_scale": "flux_scaling", "baseline_concentration": "supplied_baseline_contribution",
        "supplied_baseline_concentration": "supplied_baseline_contribution",
    })
    prepared = SimpleNamespace(inv_inputs=inputs, provenance={})
    path = tmp_path / "co2-supplied-baseline.nc"
    save_trace(trace, path)
    for saved in [trace, load_trace(path)]:
        components = reconstruct_co2_concentrations(saved, prepared)
        for group in ["prior", "posterior"]:
            state, design = xr.align(components[f"flux_scale_{group}"], inputs.H, join="exact")
            flux = xr.dot(design, state, dim="region") + inputs.fixed_prior_contribution
            np.testing.assert_allclose(components[f"pollution_{group}"].transpose(*flux.dims), flux)
            np.testing.assert_allclose(
                components[f"modelled_{group}"],
                components[f"pollution_{group}"] + components[f"baseline_{group}"],
            )
            np.testing.assert_allclose(components[f"baseline_{group}"],
                                      xr.broadcast(components[f"baseline_{group}"], inputs.supplied_baseline)[1])
        assert components.sizes["prior_draw"] == 4
        assert components.sizes["draw"] == 2


def test_scalar_sigma_cache_materializes_the_supplied_baseline(inputs, monkeypatch):
    from types import SimpleNamespace
    from openghg_inversions.rhime.co2 import co2_runner

    monkeypatch.setattr(co2_runner, "co2_supplied_baseline_input_names", lambda *a, **k: ("supplied_baseline",))
    selected = []

    def materialize(_prepared, *, variable_names):
        assert "supplied_baseline" in variable_names
        selected.append(variable_names)
        return inputs[list(variable_names)]

    monkeypatch.setattr(co2_runner, "materialize_pymc_inputs", materialize)
    prepared = SimpleNamespace(inv_inputs=inputs, rhime_inputs=None, aggregation_error_mode="dense")
    basis = co2_runner.prepare_co2_scalar_sigma_eigenbasis(prepared)
    assert basis is not None
    assert len(selected) == 1


def test_scalar_sigma_absolute_subtracted_target_parity(inputs):
    from openghg_inversions.models.scalar_sigma import (
        prepare_scalar_sigma_eigenbasis, add_scalar_sigma_eigen_likelihood,
    )

    options = dict(
        eigenbasis=prepare_scalar_sigma_eigenbasis(
            observations=inputs.mf, observation_error=inputs.mf_error,
            aggregation_error=resolve_aggregation_error(inputs, "dense"),
        ),
        sigma_prior={"pdf": "halfnormal", "sigma": .75},
    )
    absolute = _build(inputs, cached=False, likelihood_builder=add_scalar_sigma_eigen_likelihood,
                      likelihood_kwargs=options)
    residual_inputs = inputs.assign(mf=(inputs.mf-inputs.supplied_baseline).assign_attrs(units="ppm"))
    subtracted = _build(residual_inputs, cached=False, baseline=False,
                        likelihood_builder=add_scalar_sigma_eigen_likelihood, likelihood_kwargs=options)
    point = absolute.initial_point()
    point["flux_scaling_latent"] = np.array([-.2, .35])
    point["sigma_global_log__"] = np.log(.6)
    np.testing.assert_allclose(absolute.compile_logp()(point), subtracted.compile_logp()(point), atol=1e-10)
    np.testing.assert_allclose(absolute.compile_dlogp()(point), subtracted.compile_dlogp()(point), atol=1e-9)
