"""Linked additive BC widths and exact centered fixed-OU cache equations."""

from dataclasses import replace
import json

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from openghg_inversions.models import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.models.boundary import AdditiveChoice, MultiplicativeChoice
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2, annotate_co2_o2_trace
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR
from test_co2_o2_fixed_ou import _prepared, _kwargs
from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.models.boundary import bind_centered_transport
from boundary_test_utils import boundary_choice, bound_linked_model_arguments
from boundary_test_utils import raw_transport


@pytest.fixture(autouse=True)
def double_precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _additive_inputs(prepared=None):
    kwargs = _kwargs(_prepared() if prepared is None else prepared)
    weights = xr.DataArray(
        [2.0, 3.0, 5.0],
        dims="bc_region",
        coords={"bc_region": ["north", "south", "unobserved"]},
        attrs={"units": "1"},
    )
    operands = {}
    for channel, reference in (("co2", 400.0), ("o2", 200000.0)):
        sensitivity = kwargs[f"{channel}_sensitivity"]
        dim = str(sensitivity.dims[0])
        count = sensitivity.sizes[dim]
        mass = xr.DataArray(
            np.tile([0.7, 0.3, 0.0], (count, 1)),
            dims=(dim, "bc_region"),
            coords={dim: sensitivity[dim], "bc_region": weights.bc_region},
            attrs={"units": "1"},
        )
        anomaly = mass.copy(data=np.tile([0.001, -0.002, 0.0], (count, 1))).assign_attrs(units="ppm")
        operands[channel] = xr.Dataset(
            {
                "H_bc_anomaly": anomaly.rename({dim: "nmeasure"}),
                "H_bc_mass": mass.rename({dim: "nmeasure"}),
                "bc_reference": xr.DataArray(reference, attrs={"units": "ppm"}),
            },
            attrs={"boundary_representation": "centered-v1"},
        )
        native_observations = select_gathered_data_array(
            kwargs["observations"],
            key=channel,
            key_dim="species",
            ragged_dim="channel_observation",
            stack_dim="observation",
        ).rename({"observation": dim})
        bound = bind_centered_transport(raw_transport(operands[channel]), native_observations)
        channel_weights = xr.DataArray(
            weights.values, dims="bc_region", coords={"bc_region": bound.bc_region}, attrs={"units": "1"}
        )
        kwargs.setdefault("bc_centering_weights", {})[channel] = channel_weights.unstack("bc_region")
        kwargs.setdefault("bc_mean_shift_prior", {})[channel] = {"pdf": "normal", "mu": 0.0, "sigma": 0.02}
        kwargs.setdefault("bc_anomaly_scale", {})[channel] = 0.04
    kwargs["boundary_sensitivity"] = {}
    kwargs["boundary_operands"] = operands
    kwargs["observations"] = kwargs["observations"] + xr.DataArray(
        [400.0, 400.0, 200000.0, 200000.0, 200000.0],
        dims="observation",
        coords={"observation": kwargs["observations"].observation},
    )
    return kwargs


def _linked_choices(kwargs):
    """Translate flat channel fixture options into the runners' boundary choices."""
    choices = {channel: boundary_choice(kwargs, channel) for channel in ("co2", "o2")}
    return {channel: choice for channel, choice in choices.items() if choice is not None}


def test_linked_additive_o2_has_narrow_ppm_prior_and_weighted_closure():
    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs()
        model = build_co2_o2_model(**bound_linked_model_arguments({**kwargs}))
        assert "o2_bc" not in model and "co2_bc" not in model
        with model:
            trace = pm.sample_prior_predictive(
                draws=400,
                random_seed=44,
                var_names=[var.name for var in model.free_RVs + model.deterministics],
            )
        trace = restore_inferencedata_coords(trace, get_coord_registry(model))
        for channel in ("co2", "o2"):
            anomalies = trace.prior[f"{channel}_bc_anomaly"].values
            np.testing.assert_allclose(anomalies @ np.array([0.2, 0.3, 0.5]), 0.0, atol=1e-16)
            np.testing.assert_allclose(
                trace.prior[f"{channel}_mu_bc"],
                trace.constant_data[f"{channel}_mu_bc_reference"].broadcast_like(
                    trace.prior[f"{channel}_mu_bc"]
                )
                + trace.prior[f"{channel}_mu_bc_mean_shift"]
                + trace.prior[f"{channel}_mu_bc_anomaly"],
            )
            other = slice(2, 5) if channel == "co2" else slice(0, 2)
            np.testing.assert_array_equal(trace.prior[f"{channel}_mu_bc"][..., other], 0.0)
        widths = trace.prior.o2_mu_bc.values[..., 2:].std(axis=(0, 1))
        assert np.all((widths > 0.01) & (widths < 0.1))
        co2 = trace.prior.co2_bc_mean_shift.values.ravel()
        o2 = trace.prior.o2_bc_mean_shift.values.ravel()
        assert abs(np.corrcoef(co2, o2)[0, 1]) < 0.15
        expected_labels = [
            (name, kwargs["bc_centering_weights"]["o2"].bc_period.values[0])
            for name in ("north", "south", "unobserved")
        ]
        assert trace.prior.o2_bc_anomaly.o2_bc_region.values.tolist() == expected_labels


@pytest.mark.parametrize("scales", [(0.04, 0.04), (0.0, 0.04), (0.04, 0.0), (0.0, 0.0)])
def test_linked_additive_inferred_ou_cache_matches_stock_density_and_gradients(scales):
    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs()
        for channel, scale in zip(("co2", "o2"), scales):
            kwargs.setdefault("bc_anomaly_scale", {})[channel] = scale
        ou = {"tau_hours": 6.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.8}}
        cached = build_co2_o2_cached_sigma_model(**bound_linked_model_arguments({**kwargs, **ou}))
        stock = build_co2_o2_model(**bound_linked_model_arguments({**kwargs, **ou}))
        model = cached.model
        point = model.initial_point()
        point["co2_bc_mean_shift"] = np.asarray(0.013)
        point["o2_bc_mean_shift"] = np.asarray(-0.027)
        for channel, scale in zip(("co2", "o2"), scales):
            name = f"{channel}_bc_anomaly_latent"
            assert f"{channel}_bc_mean_shift" in model
            assert (name in model) == (scale > 0)
            assert (name in {state.name for state in cached.states}) == (scale > 0)
            assert (name in stock) == (scale > 0)
            if scale > 0:
                point[name] = np.array([0.01, -0.02])
            else:
                assert f"{channel}_bc_anomaly" not in model
                assert f"{channel}_mu_bc_anomaly" not in model
        sigma = np.array([0.3, 0.4, 0.5])
        point["ou_site_amplitude_log__"] = np.log(sigma)
        cached.shared_cache.update(cached.target.refresh(sigma))
        expression = model.replace_rvs_by_values(
            [cached.cached_state_logp, cached.modelled_mean, model["modelled_concentration"]]
        )
        evaluate = model.compile_fn(expression, inputs=model.value_vars, on_unused_input="ignore")
        cache_logp, centered_mean, absolute_mean = evaluate(point)
        np.testing.assert_allclose(cache_logp, model.compile_logp(vars=model.observed_RVs)(point), atol=1e-8)
        np.testing.assert_allclose(model.compile_logp()(point), stock.compile_logp()(point), atol=1e-8)
        np.testing.assert_allclose(
            model.compile_dlogp()(point), stock.compile_dlogp()(point), rtol=1e-8, atol=1e-8
        )
        state_values = [model.rvs_to_values[rv] for rv in cached.states]
        cache_gradient = model.compile_fn(
            pt.grad(expression[0], state_values), inputs=model.value_vars, on_unused_input="ignore"
        )(point)
        observed = model.logp(vars=model.observed_RVs)
        stock_gradient = model.compile_fn(
            pt.grad(observed, state_values), inputs=model.value_vars, on_unused_input="ignore"
        )(point)
        stock_gradient = np.concatenate([np.atleast_1d(value) for value in stock_gradient])
        np.testing.assert_allclose(
            np.concatenate([np.atleast_1d(value) for value in cache_gradient]), stock_gradient, atol=1e-8
        )
        shift = np.array([400.0, 400.0, 200000.0, 200000.0, 200000.0])
        np.testing.assert_allclose(absolute_mean, centered_mean + shift, atol=1e-10)
        assert cached.target.design.shape == (5, 7 + 2 * sum(scale > 0 for scale in scales))


@pytest.mark.parametrize("o2_anomaly_scale", [0.0, 0.04])
def test_linked_additive_runner_restores_absolute_predictives_and_annotations(tmp_path, o2_anomaly_scale):
    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs()
        kwargs["bc_anomaly_scale"]["o2"] = o2_anomaly_scale
        prepared = _prepared()
        prepared = replace(
            prepared, observations=kwargs["observations"], boundary_operands=kwargs["boundary_operands"],
            bc_centering_weights=kwargs["bc_centering_weights"],
        )
        built = build_rhime_co2_o2(
            prepared_inputs=prepared, independent_error_sd=kwargs["independent_error_sd"],
            boundary=_linked_choices(kwargs),
        )
        with built.model:
            trace = pm.sample_prior_predictive(draws=3, random_seed=18)
        trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
        annotate_co2_o2_trace(trace, built=built)
        original = trace.prior_predictive.y.copy()
        annotate_co2_o2_trace(trace, built=built)
        xr.testing.assert_equal(trace.prior_predictive.y, original)
        np.testing.assert_allclose(trace.observed_data.y, prepared.observations)
        assert trace.prior_predictive.y.values[..., 2:].mean() > 199900.0
        roles = json.loads(trace.attrs["rhime_variable_roles"])
        assert "o2_boundary_scale" not in roles
        assert roles["o2_boundary_mean_shift"] == "o2_bc_mean_shift"
        assert trace.prior.o2_bc_correction.attrs["units"] == "ppm"
        assert trace.constant_data.o2_H_bc_mass.attrs["units"] == "1"
        from openghg_inversions.rhime.co2.co2_o2_outputs import (
            Co2O2OutputOptions,
            make_co2_o2_output,
            calculate_co2_o2_products,
        )

        from openghg_inversions.serialization import save_trace, load_trace

        path = tmp_path / "linked-additive.nc"
        save_trace(trace, path)
        trace = load_trace(path)
        assert ("o2_bc_anomaly_latent" in trace.prior) == (o2_anomaly_scale > 0)
        if o2_anomaly_scale == 0:
            assert "o2_bc_anomaly" not in trace.prior
            assert "o2_mu_bc_anomaly" not in trace.prior
            assert "o2_boundary_correction_anomaly" not in roles
        trace["posterior"] = trace["prior"].dataset.copy(deep=False)
        result = make_co2_o2_output(
            prepared=prepared, trace=trace, start_date="2021-01-01", end_date="2021-01-02", domain="EUROPE"
        )
        calculate_co2_o2_products(result=result, output=Co2O2OutputOptions())
        from openghg_inversions.rhime.co2.co2_o2_outputs import save_co2_o2_products

        output = Co2O2OutputOptions(output_path=tmp_path / "products", save_concentrations=True)
        save_co2_o2_products(result, output=output)
        saved_path = next((tmp_path / "products").glob("*_o2_concentration_components*.nc"))
        with xr.open_dataset(saved_path) as saved:
            assert "boundary_reference_posterior" in saved
            assert "boundary_mean_shift_posterior" in saved
            assert "boundary_anomaly_posterior" in saved
        product = result.outputs["o2_concentration_components"]
        np.testing.assert_allclose(
            product.boundary_posterior,
            product.boundary_reference_posterior
            + product.boundary_mean_shift_posterior
            + product.boundary_anomaly_posterior,
        )


@pytest.mark.parametrize("anomaly_scale", [0.0, 0.04])
def test_standalone_co2_matches_linked_co2_boundary_equation_and_marginal_density(anomaly_scale):
    from scipy.stats import multivariate_normal
    from openghg_inversions.models import StateActivity
    from openghg_inversions.observation_error import AggregationError
    from openghg_inversions.rhime.co2.co2_model import build_co2_model

    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs()
        kwargs["bc_anomaly_scale"]["co2"] = anomaly_scale
        kwargs["state_activity"] = StateActivity(active=False)
        linked = build_co2_o2_model(**bound_linked_model_arguments({**kwargs}))

        def native(array):
            return select_gathered_data_array(
                array, key="co2", key_dim="species", ragged_dim="channel_observation", stack_dim="observation"
            ).rename(observation="nmeasure")

        observations = native(kwargs["observations"])
        covariance = kwargs["aggregation_error"].covariance.values[:2, :2]
        aggregation = AggregationError(
            mode="dense",
            marginal_variance=np.diag(covariance),
            covariance=xr.DataArray(
                covariance,
                dims=("nmeasure", "nmeasure_cov"),
                coords={"nmeasure": observations.nmeasure, "nmeasure_cov": observations.nmeasure.values},
            ),
        )
        single = build_co2_model(
            kwargs["co2_sensitivity"].rename(co2_measure="nmeasure"),
            retained_prior=kwargs["retained_prior"],
            fixed_prior_contribution=native(kwargs["fixed_prior_contribution"]),
            observations=observations,
            observation_error=native(kwargs["independent_error_sd"]),
            aggregation_error=aggregation,
            state_activity=StateActivity(active=False),
            boundary=AdditiveChoice(
                kwargs["bc_mean_shift_prior"]["co2"], kwargs["bc_anomaly_scale"]["co2"],
            ).bind(
                observations,
                raw_transport(kwargs["boundary_operands"]["co2"], weights=kwargs["bc_centering_weights"]["co2"],
                ),
            ),
        )
        outputs = []
        for model, prefix in ((single, ""), (linked, "co2_")):
            point = model.initial_point()
            point[f"{prefix}bc_mean_shift"] = np.asarray(0.013)
            if anomaly_scale > 0:
                point[f"{prefix}bc_anomaly_latent"] = np.array([0.01, -0.02])
            value = model.compile_fn(
                model.replace_rvs_by_values([model[f"{prefix}mu_bc"], model["modelled_concentration"]]),
                inputs=model.value_vars,
                on_unused_input="ignore",
            )(point)
            outputs.append(value)
            if model is single:
                single_logp = model.compile_logp(vars=model.observed_RVs)(point)
        np.testing.assert_allclose(outputs[0][0], outputs[1][0][:2], atol=1e-12)
        np.testing.assert_allclose(outputs[0][1], outputs[1][1][:2], atol=1e-12)
        expected = multivariate_normal.logpdf(
            observations.values,
            mean=outputs[1][1][:2],
            cov=covariance + np.diag(native(kwargs["independent_error_sd"]).values ** 2),
        )
        np.testing.assert_allclose(single_logp, expected, atol=1e-10)


def test_linked_additive_rejects_permuted_weight_labels():
    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs()
        kwargs["bc_centering_weights"]["o2"] = kwargs["bc_centering_weights"]["o2"].isel(bc_curtain=[2, 0, 1])
        with pytest.raises(ValueError, match="align"):
            build_co2_o2_model(**bound_linked_model_arguments({**kwargs}))


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("scales", [(0.04, 0.04), (0.0, 0.04), (0.04, 0.0), (0.0, 0.0)])
def test_public_linked_runners_forward_prepared_additive_boundary_options(monkeypatch, cached, scales):
    from openghg_inversions.rhime.co2 import co2_o2_runner, co2_o2_cached_sigma_runner
    from openghg_inversions.rhime.sampling import RhimeSampler

    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs()
        for channel, scale in zip(("co2", "o2"), scales):
            kwargs.setdefault("bc_anomaly_scale", {})[channel] = scale
        prepared = replace(
            _prepared(),
            observations=kwargs["observations"],
            boundary_operands=kwargs["boundary_operands"],
            bc_centering_weights={
                channel: kwargs["bc_centering_weights"][channel] for channel in ("co2", "o2")
            },
        )
        options = {"boundary": _linked_choices(kwargs)}

        def prior_sample(built, sampler):
            with built.model:
                trace = pm.sample_prior_predictive(draws=2, random_seed=81)
            return restore_inferencedata_coords(trace, get_coord_registry(built.model))

        module = co2_o2_cached_sigma_runner if cached else co2_o2_runner
        monkeypatch.setattr(module, "sample_rhime_model", prior_sample)
        runner = (
            module.run_rhime_co2_o2_cached_sigma_from_prepared_inputs
            if cached
            else module.run_rhime_co2_o2_from_prepared_inputs
        )
        if cached:
            options.update(tau_hours=6.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8})
        trace = runner(
            prepared_inputs=prepared,
            independent_error_sd=kwargs["independent_error_sd"],
            sampler=RhimeSampler(draws=2, tune=0, chains=1, burn=0, nuts_sampler="pymc"),
            **options,
        )
        np.testing.assert_allclose(trace.observed_data.y, prepared.observations)
        assert trace.prior.o2_bc_correction.attrs["units"] == "ppm"
        for channel, scale in zip(("co2", "o2"), scales):
            assert trace.constant_data[f"{channel}_bc_anomaly_scale"].item() == scale
            assert (f"{channel}_bc_anomaly_latent" in trace.prior) == (scale > 0)
            if scale == 0:
                assert f"{channel}_bc_anomaly" not in trace.prior
                assert f"{channel}_mu_bc_anomaly" not in trace.prior
        roles = json.loads(trace.attrs["rhime_variable_roles"])
        assert roles["o2_boundary_anomaly_scale"] == "o2_bc_anomaly_scale"
        assert "o2_boundary_scale" not in roles


@pytest.mark.parametrize("cached", [False, True])
def test_linked_additive_requires_native_centered_operands(cached):
    from openghg_inversions.rhime.co2 import co2_o2_runner, co2_o2_cached_sigma_runner
    from openghg_inversions.rhime.sampling import RhimeSampler

    kwargs = _additive_inputs()
    absolute = {}
    for channel, operands in kwargs["boundary_operands"].items():
        dim = str(kwargs[f"{channel}_sensitivity"].dims[0])
        absolute[channel] = (operands.H_bc_anomaly + operands.bc_reference * operands.H_bc_mass).rename(
            nmeasure=dim
        )
    prepared = replace(
        _prepared(), observations=kwargs["observations"], boundary_sensitivity=absolute, boundary_operands={},
    )
    options = {"boundary": _linked_choices(kwargs)}
    if cached:
        options.update(tau_hours=6.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8})
    runner = (
        co2_o2_cached_sigma_runner.run_rhime_co2_o2_cached_sigma_from_prepared_inputs
        if cached
        else co2_o2_runner.run_rhime_co2_o2_from_prepared_inputs
    )
    with pytest.raises(ValueError, match="co2 AdditiveChoice requires centered boundary operands"):
        runner(
            prepared_inputs=prepared,
            independent_error_sd=kwargs["independent_error_sd"],
            sampler=RhimeSampler(draws=1, tune=0, chains=1, burn=0, nuts_sampler="pymc"),
            **options,
        )


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("mode", ["multiplicative", "disabled", "co2_additive"])
def test_linked_runners_do_not_compute_passive_additive_arrays(monkeypatch, cached, mode):
    from dask import delayed
    import dask.array as da
    from openghg_inversions.rhime.co2 import co2_o2_runner, co2_o2_cached_sigma_runner
    from openghg_inversions.rhime.sampling import RhimeSampler

    def unexpected_computation():
        raise AssertionError("Computed an unused additive boundary graph")

    def passive(array):
        payload = da.from_delayed(delayed(unexpected_computation)(), shape=array.shape, dtype=array.dtype)
        return array.copy(data=payload)

    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs()
        weights = {}
        boundary = {}
        for channel in ("co2", "o2"):
            channel_weights = kwargs["bc_centering_weights"][channel]
            if mode == "co2_additive" and channel == "co2":
                boundary["co2"] = AdditiveChoice(kwargs["bc_mean_shift_prior"]["co2"], kwargs["bc_anomaly_scale"]["co2"])
            else:
                channel_weights = passive(channel_weights)
                if mode != "disabled":
                    boundary[channel] = MultiplicativeChoice(dict(DEFAULT_BC_PRIOR))
            weights[channel] = channel_weights
        prepared = replace(
            _prepared(),
            observations=kwargs["observations"],
            boundary_sensitivity=kwargs["boundary_sensitivity"],
            boundary_operands=kwargs["boundary_operands"],
            bc_centering_weights=weights,
        )
        options = {"boundary": boundary}

        def prior_sample(built, sampler):
            with built.model:
                trace = pm.sample_prior_predictive(draws=1, random_seed=83)
            return restore_inferencedata_coords(trace, get_coord_registry(built.model))

        module = co2_o2_cached_sigma_runner if cached else co2_o2_runner
        monkeypatch.setattr(module, "sample_rhime_model", prior_sample)
        if cached:
            monkeypatch.setattr(module, "_sampler_for_cached_graph", lambda sampler, **_: sampler)
            options.update(tau_hours=6.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8})
        runner = (
            module.run_rhime_co2_o2_cached_sigma_from_prepared_inputs
            if cached
            else module.run_rhime_co2_o2_from_prepared_inputs
        )
        trace = runner(
            prepared_inputs=prepared,
            independent_error_sd=kwargs["independent_error_sd"],
            sampler=RhimeSampler(draws=1, tune=0, chains=1, burn=0, nuts_sampler="pymc"),
            **options,
        )
        assert ("co2_bc_mean_shift" in trace.prior) == (mode == "co2_additive")
        assert "o2_bc_mean_shift" not in trace.prior
        assert "o2_bc_centering_weights" not in trace.constant_data


def _mixed_centered_boundary_inputs():
    kwargs = _additive_inputs()
    kwargs["bc_mean_shift_prior"].pop("co2")
    kwargs["bc_anomaly_scale"].pop("co2")
    kwargs["bc_centering_weights"].pop("co2")
    kwargs["bc_prior"] = {"co2": {"pdf": "normal", "mu": 1.0, "sigma": 0.01}}
    return kwargs


@pytest.mark.parametrize("o2_anomaly_scale", [0.0, 0.04])
def test_mixed_centered_boundary_ou_density_gradients_and_absolute_closure(o2_anomaly_scale):
    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _mixed_centered_boundary_inputs()
        kwargs["bc_anomaly_scale"]["o2"] = o2_anomaly_scale
        ou = {"tau_hours": 6.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.8}}
        stock = build_co2_o2_model(**bound_linked_model_arguments({**kwargs, **ou}))
        cached = build_co2_o2_cached_sigma_model(**bound_linked_model_arguments({**kwargs, **ou}))
        absolute_kwargs = {**kwargs, "boundary_operands": {}, "boundary_sensitivity": {}}
        for channel, operands in kwargs["boundary_operands"].items():
            sensitivity = kwargs[f"{channel}_sensitivity"]
            dim = str(sensitivity.dims[0])
            observations = xr.DataArray(
                np.zeros(sensitivity.sizes[dim]), dims=dim, coords={dim: sensitivity[dim]}
            )
            bound = bind_centered_transport(raw_transport(operands), observations)
            if channel == "co2":
                absolute_kwargs["boundary_sensitivity"][channel] = (
                    bound.H_bc_anomaly + bound.bc_reference * bound.H_bc_mass
                )
            else:
                absolute_kwargs["boundary_operands"][channel] = operands
        absolute = build_co2_o2_model(**bound_linked_model_arguments({**absolute_kwargs, **ou}))
        point = cached.model.initial_point()
        co2_rv = next(rv for rv in cached.model.free_RVs if rv.name.startswith("co2_bc"))
        co2_value = cached.model.rvs_to_values[co2_rv].name
        point[co2_value] = np.full_like(point[co2_value], 1.0001)
        point["o2_bc_mean_shift"] = np.asarray(-0.027)
        if o2_anomaly_scale > 0:
            point["o2_bc_anomaly_latent"] = np.array([-0.03, 0.015])
        sigma = np.array([0.3, 0.4, 0.5])
        point["ou_site_amplitude_log__"] = np.log(sigma)
        cached.shared_cache.update(cached.target.refresh(sigma))
        expression = cached.model.replace_rvs_by_values(
            [cached.cached_state_logp, cached.modelled_mean, cached.model["modelled_concentration"]]
        )
        cache_logp, centered_mean, absolute_mean = cached.model.compile_fn(
            expression, inputs=cached.model.value_vars, on_unused_input="ignore"
        )(point)
        np.testing.assert_allclose(cache_logp, stock.compile_logp(vars=stock.observed_RVs)(point), atol=1e-8)
        np.testing.assert_allclose(cached.model.compile_logp()(point), stock.compile_logp()(point), atol=1e-8)
        np.testing.assert_allclose(
            cached.model.compile_dlogp()(point), stock.compile_dlogp()(point), rtol=1e-8, atol=1e-8
        )
        state_values = [cached.model.rvs_to_values[rv] for rv in cached.states]
        cache_gradient = cached.model.compile_fn(
            pt.grad(expression[0], state_values), inputs=cached.model.value_vars, on_unused_input="ignore"
        )(point)
        observed_gradient = cached.model.compile_fn(
            pt.grad(cached.model.logp(vars=cached.model.observed_RVs), state_values),
            inputs=cached.model.value_vars,
            on_unused_input="ignore",
        )(point)
        np.testing.assert_allclose(
            np.concatenate([np.atleast_1d(value) for value in cache_gradient]),
            np.concatenate([np.atleast_1d(value) for value in observed_gradient]),
            atol=1e-8,
        )
        absolute_prediction = absolute.compile_fn(
            absolute.replace_rvs_by_values([absolute["modelled_concentration"]]),
            inputs=absolute.value_vars,
            on_unused_input="ignore",
        )(point)[0]
        shift = np.array([400.0, 400.0, 200000.0, 200000.0, 200000.0])
        np.testing.assert_allclose(absolute_mean, centered_mean + shift, atol=1e-10)
        np.testing.assert_allclose(absolute_mean, absolute_prediction, atol=1e-10)
        np.testing.assert_allclose(stock.compile_logp()(point), absolute.compile_logp()(point), atol=1e-8)
        np.testing.assert_allclose(
            stock.compile_dlogp()(point), absolute.compile_dlogp()(point), rtol=1e-7, atol=1e-7
        )


@pytest.mark.parametrize("cached", [False, True])
def test_public_linked_ou_restores_mixed_centered_boundary_predictions(monkeypatch, cached):
    from openghg_inversions.rhime.co2 import co2_o2_runner, co2_o2_cached_sigma_runner
    from openghg_inversions.rhime.sampling import RhimeSampler

    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _mixed_centered_boundary_inputs()
        prepared = replace(
            _prepared(),
            observations=kwargs["observations"],
            boundary_operands=kwargs["boundary_operands"],
            bc_centering_weights={"o2": kwargs["bc_centering_weights"]["o2"]},
        )
        options = dict(
            boundary=_linked_choices(kwargs),
            tau_hours=6.0,
            site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8},
        )

        def prior_sample(built, sampler):
            with built.model:
                trace = pm.sample_prior_predictive(draws=3, random_seed=93)
            return restore_inferencedata_coords(trace, get_coord_registry(built.model))

        module = co2_o2_cached_sigma_runner if cached else co2_o2_runner
        monkeypatch.setattr(module, "sample_rhime_model", prior_sample)
        if cached:
            monkeypatch.setattr(module, "_sampler_for_cached_graph", lambda sampler, **_: sampler)
        runner = (
            module.run_rhime_co2_o2_cached_sigma_from_prepared_inputs
            if cached
            else module.run_rhime_co2_o2_from_prepared_inputs
        )
        trace = runner(
            prepared_inputs=prepared,
            independent_error_sd=kwargs["independent_error_sd"],
            sampler=RhimeSampler(draws=3, tune=0, chains=1, burn=0, nuts_sampler="pymc"),
            **options,
        )
        assert "co2_bc" in trace.prior and "co2_bc_mean_shift" not in trace.prior
        assert "o2_bc_mean_shift" in trace.prior and "o2_bc" not in trace.prior
        np.testing.assert_allclose(trace.observed_data.y, prepared.observations)
        assert trace.prior_predictive.y.attrs["concentration_representation"] == "absolute-v1"
        assert trace.prior_predictive.y.values[..., 2:].mean() > 199900.0
        np.testing.assert_allclose(
            trace.prior.modelled_concentration,
            trace.prior.co2_o2_flux_contribution
            + trace.constant_data.fixed_prior_contribution
            + trace.prior.co2_mu_bc
            + trace.prior.o2_mu_bc,
            atol=1e-10,
        )


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("channel", ["co2", "o2"])
@pytest.mark.parametrize("drop_first", [False, True])
@pytest.mark.parametrize("anomaly_scale", [0.0, 0.04])
def test_linked_public_additive_builders_warn_on_native_channel_rows(
    cached, channel, drop_first, anomaly_scale
):
    import warnings
    from test_rhime_co2_o2_baselines import _prepared as prepared_with_two_sites

    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs(prepared_with_two_sites())
        other = "o2" if channel == "co2" else "co2"
        kwargs["bc_mean_shift_prior"].pop(other)
        kwargs["bc_anomaly_scale"].pop(other)
        kwargs.setdefault("bc_anomaly_scale", {})[channel] = anomaly_scale
        kwargs["offset_prior"] = {channel: {"pdf": "normal", "mu": 0.0, "sigma": 0.1}}
        kwargs["offset_args"] = {channel: {"drop_first": drop_first}}
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            if cached:
                kwargs.update(tau_hours=4.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.2})
                model = build_co2_o2_cached_sigma_model(**bound_linked_model_arguments({**kwargs})).model
            else:
                model = build_co2_o2_model(**bound_linked_model_arguments({**kwargs}))
        confounding = [
            warning
            for warning in recorded
            if "unrestricted observation offsets share a constant observation response" in str(warning.message)
        ]
        assert len(confounding) == (0 if drop_first else 1)
        if confounding:
            assert str(confounding[0].message).startswith(f"{channel}:")
        assert f"{channel}_bc_mean_shift" in model and f"{channel}_offset" in model
        assert (f"{channel}_bc_anomaly_latent" in model) == (anomaly_scale > 0)
