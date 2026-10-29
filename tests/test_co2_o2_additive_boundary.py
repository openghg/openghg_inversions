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
from openghg_inversions.rhime.co2.centered_boundary import bind_boundary_operands
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2, annotate_co2_o2_trace
from test_co2_o2_fixed_ou import _prepared, _kwargs


def _additive_inputs(centered):
    kwargs = _kwargs(_prepared())
    weights = xr.DataArray(
        [2.0, 3.0, 5.0],
        dims="bc_region",
        coords={"bc_region": ["north", "south", "unobserved"]},
        attrs={"units": "1"},
    )
    boundaries = {}
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
        if centered:
            operands[channel] = xr.Dataset(
                {
                    "H_bc_anomaly": anomaly.rename({dim: "nmeasure"}),
                    "H_bc_mass": mass.rename({dim: "nmeasure"}),
                    "bc_reference": xr.DataArray(reference, attrs={"units": "ppm"}),
                },
                attrs={"boundary_representation": "centered-v1"},
            )
        else:
            boundaries[channel] = reference * mass + anomaly
            kwargs[f"boundary_correction_sensitivity_{channel}"] = mass
        if centered:
            native_observations = xr.DataArray(np.zeros(count), dims=dim, coords={dim: sensitivity[dim]})
            bound = bind_boundary_operands(operands[channel], native_observations)
            channel_weights = xr.DataArray(
                weights.values, dims="bc_region", coords={"bc_region": bound.bc_region}, attrs={"units": "1"}
            )
        else:
            channel_weights = weights
        kwargs[f"bc_centering_weights_{channel}"] = channel_weights
        kwargs[f"bc_mean_shift_prior_{channel}"] = {"pdf": "normal", "mu": 0.0, "sigma": 0.02}
        kwargs[f"bc_anomaly_scale_{channel}"] = 0.04
    kwargs["boundary_sensitivity"] = boundaries
    kwargs["boundary_operands"] = operands
    kwargs["observations"] = kwargs["observations"] + xr.DataArray(
        [400.0, 400.0, 200000.0, 200000.0, 200000.0],
        dims="observation",
        coords={"observation": kwargs["observations"].observation},
    )
    return kwargs


@pytest.mark.parametrize("centered", [False, True])
def test_linked_additive_o2_has_narrow_ppm_prior_and_weighted_closure(centered):
    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs(centered)
        model = build_co2_o2_model(**kwargs)
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
                trace.prior[f"{channel}_mu_bc_reference"]
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
        expected_labels = (
            [(name, 0) for name in ("north", "south", "unobserved")]
            if centered
            else ["north", "south", "unobserved"]
        )
        assert trace.prior.o2_bc_anomaly.o2_bc_region.values.tolist() == expected_labels


@pytest.mark.parametrize("centered", [False, True])
def test_linked_additive_inferred_ou_cache_matches_stock_density_and_gradients(centered):
    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs(centered)
        ou = {"tau_hours": 6.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.8}}
        cached = build_co2_o2_cached_sigma_model(**kwargs, **ou)
        stock = build_co2_o2_model(**kwargs, **ou)
        model = cached.model
        point = model.initial_point()
        point["co2_bc_mean_shift"] = np.asarray(0.013)
        point["o2_bc_mean_shift"] = np.asarray(-0.027)
        point["co2_bc_anomaly_latent"] = np.array([0.01, -0.02])
        point["o2_bc_anomaly_latent"] = np.array([-0.03, 0.015])
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
        shift = np.array([400.0, 400.0, 200000.0, 200000.0, 200000.0]) if centered else 0.0
        np.testing.assert_allclose(absolute_mean, centered_mean + shift, atol=1e-10)
        assert cached.target.design.shape == (5, 11)


def test_linked_additive_runner_restores_absolute_predictives_and_annotations(tmp_path):
    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs(True)
        prepared = _prepared()
        prepared = replace(
            prepared, observations=kwargs["observations"], boundary_operands=kwargs["boundary_operands"]
        )
        options = {
            key: value
            for key, value in kwargs.items()
            if key.startswith(("bc_mean_", "bc_anomaly_", "bc_centering_"))
        }
        built = build_rhime_co2_o2(
            prepared_inputs=prepared, independent_error_sd=kwargs["independent_error_sd"], **options
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
        assert trace.prior.o2_bc_anomaly.attrs["units"] == "ppm"
        assert trace.constant_data.o2_G_bc.attrs["units"] == "1"
        from openghg_inversions.rhime.co2.co2_o2_outputs import (
            Co2O2OutputOptions,
            make_co2_o2_output,
            calculate_co2_o2_products,
        )

        from openghg_inversions.serialization import save_trace, load_trace

        path = tmp_path / "linked-additive.nc"
        save_trace(trace, path)
        trace = load_trace(path)
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


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("conflicting", ["bc_prior", "bc_state_activity"])
def test_linked_additive_rejects_multiplicative_options_early(cached, conflicting):
    from openghg_inversions.models import StateActivity

    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs(True)
        kwargs[conflicting] = {
            "o2": {"pdf": "normal", "mu": 1.0, "sigma": 0.01}
            if conflicting == "bc_prior"
            else StateActivity(active=False)
        }
        builder = build_co2_o2_cached_sigma_model if cached else build_co2_o2_model
        if cached:
            kwargs.update(tau_hours=6.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8})
        with pytest.raises(ValueError, match="o2 additive boundaries replace"):
            builder(**kwargs)


def test_standalone_co2_matches_linked_co2_boundary_equation_and_marginal_density():
    from scipy.stats import multivariate_normal
    from openghg_inversions.array_ops import select_gathered_data_array
    from openghg_inversions.models import StateActivity
    from openghg_inversions.observation_error import AggregationError
    from openghg_inversions.rhime.co2.co2_model import build_co2_model

    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs(True)
        kwargs["state_activity"] = StateActivity(active=False)
        linked = build_co2_o2_model(**kwargs)

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
            boundary_operands=kwargs["boundary_operands"]["co2"],
            bc_mean_shift_prior=kwargs["bc_mean_shift_prior_co2"],
            bc_anomaly_scale=kwargs["bc_anomaly_scale_co2"],
            bc_centering_weights=kwargs["bc_centering_weights_co2"],
        )
        outputs = []
        for model, prefix in ((single, ""), (linked, "co2_")):
            point = model.initial_point()
            point[f"{prefix}bc_mean_shift"] = np.asarray(0.013)
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
        kwargs = _additive_inputs(True)
        kwargs["bc_centering_weights_o2"] = kwargs["bc_centering_weights_o2"].isel(bc_region=[2, 0, 1])
        with pytest.raises(ValueError, match="align"):
            build_co2_o2_model(**kwargs)


@pytest.mark.parametrize("cached", [False, True])
def test_public_linked_runners_forward_prepared_additive_boundary_options(monkeypatch, cached):
    from openghg_inversions.rhime.co2 import co2_o2_runner, co2_o2_cached_sigma_runner
    from openghg_inversions.rhime.sampling import RhimeSampler

    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs(True)
        prepared = replace(
            _prepared(),
            observations=kwargs["observations"],
            boundary_operands=kwargs["boundary_operands"],
            bc_centering_weights={
                channel: kwargs[f"bc_centering_weights_{channel}"] for channel in ("co2", "o2")
            },
            boundary_correction_sensitivity={
                channel: operands.H_bc_mass for channel, operands in kwargs["boundary_operands"].items()
            },
        )
        options = {key: value for key, value in kwargs.items() if key.startswith(("bc_mean_", "bc_anomaly_"))}

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
        assert trace.prior.o2_bc_anomaly.attrs["units"] == "ppm"
        assert trace.constant_data.o2_bc_anomaly_scale.item() == 0.04
        roles = json.loads(trace.attrs["rhime_variable_roles"])
        assert roles["o2_boundary_anomaly_scale"] == "o2_bc_anomaly_scale"
        assert "o2_boundary_scale" not in roles


@pytest.mark.parametrize("cached", [False, True])
def test_passive_correction_arrays_do_not_change_multiplicative_period_binding(cached):
    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs(False)
        for channel in ("co2", "o2"):
            kwargs.pop(f"bc_mean_shift_prior_{channel}")
            kwargs.pop(f"bc_anomaly_scale_{channel}")
            kwargs[f"boundary_correction_sensitivity_{channel}"] = kwargs[
                f"boundary_correction_sensitivity_{channel}"
            ].assign_attrs(boundary_sensitivity_preparation="sampled-period-expansion")
        kwargs["bc_freq"] = "1D"
        if cached:
            kwargs.update(tau_hours=6.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8})
            model = build_co2_o2_cached_sigma_model(**kwargs).model
        else:
            model = build_co2_o2_model(**kwargs)
        assert "o2_bc" in model and "co2_bc" in model
        assert "o2_bc_mean_shift" not in model and "co2_bc_mean_shift" not in model


@pytest.mark.parametrize("cached", [False, True])
def test_linked_canonical_raw_spatial_absolute_inputs_bind_one_period(cached):
    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _additive_inputs(False)
        expanded = _additive_inputs(True)
        for channel in ("co2", "o2"):
            kwargs["boundary_sensitivity"][channel] = kwargs["boundary_sensitivity"][channel].assign_attrs(
                boundary_sensitivity_preparation="raw-spatial"
            )
            kwargs[f"boundary_correction_sensitivity_{channel}"] = kwargs[
                f"boundary_correction_sensitivity_{channel}"
            ].assign_attrs(boundary_sensitivity_preparation="raw-spatial")
            kwargs[f"bc_centering_weights_{channel}"] = expanded[f"bc_centering_weights_{channel}"]
        if cached:
            kwargs.update(tau_hours=6.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8})
            model = build_co2_o2_cached_sigma_model(**kwargs).model
        else:
            model = build_co2_o2_model(**kwargs)
        assert "o2_bc_mean_shift" in model and "co2_bc_mean_shift" in model
