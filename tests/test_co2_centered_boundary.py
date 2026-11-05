"""Scientific invariants of additive, centred boundary corrections."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.stats import multivariate_normal
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import build_co2_cached_sigma_model, build_co2_model
from openghg_inversions.boundary_sensitivity import validate_boundary_operands
from openghg_inversions.models.boundary import AdditiveChoice, CenteredOperands
from boundary_test_utils import bound_model_arguments, boundary_transport


@pytest.fixture(autouse=True)
def double_precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _inputs(n_boundary: int = 3) -> xr.Dataset:
    response = np.array([[0.2, 0.3, 0.1], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.5]])
    response = response[:, :n_boundary]
    boundary = response * np.array([400.0, 410.0, 390.0])[:n_boundary]
    inputs = xr.Dataset(
        {
            "H": (("nmeasure", "region"), np.ones((4, 1))),
            "alpha_prior_mean": ("region", [1.0]),
            "alpha_prior_covariance": (("region", "region_cov"), [[0.04]]),
            "fixed_prior_contribution": ("nmeasure", np.full(4, 0.1)),
            "mf": ("nmeasure", boundary.sum(axis=1) + np.array([1.15, 1.06, 1.12, 1.07])),
            "mf_error": ("nmeasure", np.full(4, 0.1)),
            "aggregation_error_covariance": (("nmeasure", "nmeasure_cov"), np.eye(4) * 0.02),
            "H_bc": (("nmeasure", "bc_region"), boundary),
            "H_bc_mass": (("nmeasure", "bc_region"), response),
            "bc_centering_weights": ("bc_region", np.array([1.0, 2.0, 4.0])[:n_boundary]),
        },
        coords={
            "nmeasure": np.arange(4),
            "region": ["flux"],
            "bc_region": ["north", "east", "west"][:n_boundary],
            "site": ("nmeasure", ["AAA", "AAA", "BBB", "BBB"]),
            "time": (
                "nmeasure",
                np.array(
                    ["2021-01-01T00", "2021-01-01T03", "2021-01-01T01", "2021-01-01T05"],
                    dtype="datetime64[h]",
                ),
            ),
        },
    )
    for name in ("mf", "mf_error", "H_bc", "fixed_prior_contribution"):
        inputs[name].attrs["units"] = "ppm"
    for name in ("H_bc_mass", "bc_centering_weights"):
        inputs[name].attrs["units"] = "1"
    return inputs


def _period_weights(values, curtains, periods) -> xr.DataArray:
    """Centering weights on the prepared ``(bc_curtain, bc_period)`` layout."""
    return xr.DataArray(
        np.asarray(values, dtype=float).reshape(len(curtains), len(periods)),
        dims=("bc_curtain", "bc_period"),
        coords={"bc_curtain": np.asarray(curtains), "bc_period": np.asarray(periods)},
        attrs={"units": "1"},
    )


def _whole_window_weights(inputs: xr.Dataset) -> xr.DataArray:
    return _period_weights(
        inputs.bc_centering_weights.values,
        inputs.bc_region.values,
        [inputs.time.values.min().astype("datetime64[ns]")],
    )


def _kwargs(inputs: xr.Dataset) -> dict[str, Any]:
    return dict(
        retained_prior=CorrelatedLognormalPrior(
            inputs.alpha_prior_mean, inputs.alpha_prior_covariance, covariance_dim="region_cov"
        ),
        fixed_prior_contribution=inputs.fixed_prior_contribution,
        observations=inputs.mf,
        observation_error=inputs.mf_error,
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        state_activity=StateActivity(active=np.zeros(1, dtype=bool), fixed_value=1.0),
        boundary_operands=xr.Dataset(
            {
                "H_bc_anomaly": (inputs.H_bc - 400.0 * inputs.H_bc_mass).assign_attrs(units="ppm"),
                "H_bc_mass": inputs.H_bc_mass,
                "bc_reference": xr.DataArray(400.0, attrs={"units": "ppm"}),
            },
            attrs={"boundary_representation": "centered-v1"},
        ),
        bc_centering_weights=_whole_window_weights(inputs),
        bc_mean_shift_prior={"pdf": "normal", "mu": 0.0, "sigma": 0.5},
        bc_anomaly_scale=0.3,
    )


def _anomaly_rv(model: pm.Model) -> Any:
    return next(rv for rv in model.free_RVs if rv.name != "bc_mean_shift" and "sigma" not in rv.name)


def _anomaly_design(model: pm.Model) -> np.ndarray:
    anomaly = model.replace_rvs_by_values([model["bc_anomaly"]])[0]
    coefficients = model.rvs_to_values[_anomaly_rv(model)]
    jacobian = pt.jacobian(anomaly, coefficients)
    evaluate = model.compile_fn(jacobian, inputs=model.value_vars, on_unused_input="ignore")
    return np.asarray(evaluate(model.initial_point()))


def test_weighted_anomaly_prior_preserves_contrasts_and_permutation() -> None:
    inputs = _inputs()
    model = build_co2_model(inputs.H, **bound_model_arguments(_kwargs(inputs)))
    assert "bc" not in model.named_vars
    assert len(model.free_RVs) == 2
    design = _anomaly_design(model)
    latent = _anomaly_rv(model)
    point = model.initial_point()
    point[model.rvs_to_values[latent].name] = np.array([0.2, -0.4])
    prior_logp = model.compile_fn(
        model.logp(vars=[latent]), inputs=model.value_vars, on_unused_input="ignore"
    )
    assert prior_logp(point) == pytest.approx(
        multivariate_normal.logpdf([0.2, -0.4], mean=np.zeros(2), cov=np.eye(2) * 0.3**2)
    )
    weights = inputs.bc_centering_weights.values / inputs.bc_centering_weights.sum().item()
    projection = np.eye(3) - np.ones((3, 1)) * weights
    np.testing.assert_allclose(weights @ design, 0.0, atol=1e-15)
    np.testing.assert_allclose(design @ design.T, projection @ projection.T, atol=1e-14)
    contrasts = design[[0, 0, 1]] - design[[1, 2, 2]]
    np.testing.assert_allclose(np.sum(contrasts**2, axis=1), 2.0)

    order = np.array([2, 0, 1])
    permuted = inputs.isel(bc_region=order)
    other = build_co2_model(permuted.H, **bound_model_arguments(_kwargs(permuted)))
    other_design = _anomaly_design(other)
    np.testing.assert_allclose(
        other_design @ other_design.T, (design @ design.T)[np.ix_(order, order)], atol=1e-14
    )


@pytest.mark.parametrize("n_boundary", [1, 3])
@pytest.mark.parametrize("anomaly_scale", [0.0, 0.3])
def test_forward_equations_and_cached_likelihood_gradient(n_boundary: int, anomaly_scale: float) -> None:
    inputs = _inputs(n_boundary)
    kwargs = _kwargs(inputs)
    kwargs["bc_anomaly_scale"] = anomaly_scale
    ordinary = build_co2_model(inputs.H, **bound_model_arguments(kwargs))
    cached = build_co2_cached_sigma_model(
        inputs.H,
        **bound_model_arguments(
            {
                **kwargs,
                "tau_hours": {"AAA": 3.0, "BBB": 7.0},
                "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75},
                "initial_site_amplitudes": 0.4,
            }
        ),
    )
    has_contrasts = n_boundary > 1 and anomaly_scale > 0
    assert cached.target.design.shape == (4, n_boundary if has_contrasts else 1)
    assert [state.name for state in cached.states] == (
        ["bc_mean_shift", "bc_anomaly_latent"] if has_contrasts else ["bc_mean_shift"]
    )
    covariance = np.eye(4) * 0.03
    for rows, lag, tau in (([0, 1], 3.0, 3.0), ([2, 3], 4.0, 7.0)):
        covariance[np.ix_(rows, rows)] += 0.16 * np.array(
            [[1.0, np.exp(-lag / tau)], [np.exp(-lag / tau), 1.0]]
        )
    predictions = []
    for model in (ordinary, cached.model):
        assert "bc" not in model.named_vars
        if not has_contrasts:
            assert not any("anomaly" in rv.name for rv in model.free_RVs)
            assert "bc_anomaly_latent" not in model.named_vars
            assert "bc_anomaly_basis" not in model.named_vars
            assert "bc_anomaly" not in model.named_vars
            assert "mu_bc_anomaly" not in model.named_vars
        point = model.initial_point()
        if model is cached.model:
            point["ou_site_amplitude_log__"] = np.log(np.full(2, 0.4))
        point["bc_mean_shift"] = np.asarray(0.7)
        if has_contrasts:
            rv = _anomaly_rv(model)
            point[model.rvs_to_values[rv].name] = np.array([0.2, -0.4])
        outputs = model.replace_rvs_by_values(
            [
                model[name]
                if name in model
                else pt.zeros_like(model["bc_correction" if name == "bc_anomaly" else "mu_bc_mean_shift"])
                for name in (
                    "bc_anomaly",
                    "bc_correction",
                    "mu_bc_reference",
                    "mu_bc_mean_shift",
                    "mu_bc_anomaly",
                    "mu_bc",
                    "modelled_concentration",
                )
            ]
        )
        if model is cached.model:
            likelihood = model.logp(vars=model.observed_RVs)
            state_values = [model.rvs_to_values[model["bc_mean_shift"]]]
            if has_contrasts:
                state_values.append(model.rvs_to_values[_anomaly_rv(model)])
            outputs.extend([likelihood, *pt.grad(likelihood, state_values)])
        evaluate = model.compile_fn(outputs, inputs=model.value_vars, on_unused_input="ignore")
        values = evaluate(point)
        anomaly, correction, reference, shift, variation, boundary, mean = values[:7]
        assert anomaly.shape == (n_boundary,)
        if not has_contrasts:
            np.testing.assert_array_equal(anomaly, np.zeros(n_boundary))
            np.testing.assert_array_equal(variation, np.zeros(4))
        response = inputs.H_bc_mass.values
        weights = inputs.bc_centering_weights.values / inputs.bc_centering_weights.sum().item()
        np.testing.assert_allclose(weights @ anomaly, 0.0, atol=1e-14)
        np.testing.assert_allclose(weights @ correction, 0.7)
        np.testing.assert_allclose(reference, inputs.H_bc.sum("bc_region"))
        np.testing.assert_allclose(shift, response.sum(axis=1) * 0.7)
        np.testing.assert_allclose(variation, response @ anomaly)
        np.testing.assert_allclose(boundary, reference + response @ correction)
        np.testing.assert_allclose(mean, boundary + 1.1)
        predictions.append(mean)
        if model is cached.model:
            np.testing.assert_allclose(
                values[7], multivariate_normal.logpdf(inputs.mf, mean=mean, cov=covariance)
            )
            residual_precision = np.linalg.solve(covariance, inputs.mf.values - mean)
            np.testing.assert_allclose(
                values[8], response.sum(axis=1) @ residual_precision, rtol=1e-6, atol=1e-8
            )
            if has_contrasts:
                np.testing.assert_allclose(
                    values[9],
                    (response @ _anomaly_design(model)).T @ residual_precision,
                    rtol=1e-6,
                    atol=1e-8,
                )
    np.testing.assert_allclose(*predictions)


@pytest.mark.parametrize(
    "override",
    [
        {"bc_anomaly_scale": None},
        {"bc_anomaly_scale": -0.1},
        {"bc_anomaly_scale": np.nan},
        {"bc_anomaly_scale": np.inf},
        {"bc_anomaly_scale": -np.inf},
        {"bc_centering_weights": None},
        {"boundary_operands": None, "boundary_sensitivity": "absolute"},
    ],
)
def test_centered_boundary_rejects_incomplete_options(override: dict[str, Any]) -> None:
    inputs = _inputs()
    kwargs = _kwargs(inputs)
    kwargs.update(override)
    if kwargs.get("boundary_sensitivity") == "absolute":
        kwargs["boundary_sensitivity"] = inputs.H_bc
    with pytest.raises((ValueError, TypeError)):
        build_co2_model(inputs.H, **bound_model_arguments(kwargs))


@pytest.mark.parametrize(
    "invalid",
    [
        "negative_weight",
        "zero_weight",
        "nan_weight",
        "negative_response",
        "zero_response",
        "dimensional_response",
        "misaligned_labels",
    ],
)
def test_centered_boundary_validates_scientific_inputs(invalid: str) -> None:
    inputs = _inputs()
    kwargs = _kwargs(inputs)
    if invalid.endswith("weight"):
        weights = kwargs["bc_centering_weights"].copy(deep=True)
        weights[0, 0] = {"negative_weight": -1.0, "zero_weight": 0.0, "nan_weight": np.nan}[invalid]
        kwargs["bc_centering_weights"] = weights
    elif invalid == "misaligned_labels":
        kwargs["bc_centering_weights"] = kwargs["bc_centering_weights"].assign_coords(
            bc_curtain=["wrong", "east", "west"]
        )
    else:
        response = inputs.H_bc_mass.copy(deep=True)
        if invalid == "negative_response":
            response[0, 0] = -0.1
        elif invalid == "zero_response":
            response[:] = 0.0
        else:
            # Units are owned by preparation and artifact loading, not by binding.
            response.attrs["units"] = "m"
            kwargs["boundary_operands"]["H_bc_mass"] = response
            with pytest.raises(ValueError, match="mass units must be '1'"):
                validate_boundary_operands(kwargs["boundary_operands"])
            return
        kwargs["boundary_operands"]["H_bc_mass"] = response
    with pytest.raises(ValueError):
        build_co2_model(inputs.H, **bound_model_arguments(kwargs))


def test_additive_term_keeps_sampled_boundary_and_observation_offsets_in_mean() -> None:
    from openghg_inversions.models import ConcentrationTerm, likelihood_representation, registered_model

    inputs = _inputs()
    mass = inputs.H_bc_mass
    operands = xr.Dataset(
        {
            "H_bc_anomaly": (mass * 0.001).assign_attrs(units="ppm"),
            "H_bc_mass": mass,
            "bc_reference": xr.DataArray(200000.0, attrs={"units": "ppm"}),
        }
    )
    prior = {"pdf": "normal", "mu": 0.0, "sigma": 0.2}
    transport = CenteredOperands(
        operands.H_bc_anomaly, operands.H_bc_mass, operands.bc_reference, weights=_whole_window_weights(inputs)
    )
    prepared = AdditiveChoice(prior, 0.3).bind(inputs.mf, transport)
    with registered_model() as model:
        result = prepared.add()
        observation_offset = pm.Normal("observation_offset", mu=0.0, sigma=0.1)
        offset = pt.ones(inputs.sizes["nmeasure"]) * observation_offset
        absolute = result.output + offset
        mean, observations, shift = likelihood_representation(
            [result, ConcentrationTerm.absolute(offset)],
            inputs.mf,
            absolute,
        )
    expected_shift = 200000.0 * mass.sum("bc_region")
    xr.testing.assert_allclose(shift, expected_shift)
    xr.testing.assert_allclose(observations, inputs.mf - expected_shift)
    evaluate = model.compile_fn(
        model.replace_rvs_by_values([absolute, mean, model["bc_correction"]]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    for boundary_mean, instrument_offset in ((-0.25, 0.07), (0.17, -0.03)):
        point = model.initial_point()
        point["bc_mean_shift"] = np.asarray(boundary_mean)
        point["bc_anomaly_latent"] = np.array([0.1, -0.05])
        point["observation_offset"] = np.asarray(instrument_offset)
        absolute_values, centered_values, correction = evaluate(point)
        expected = (
            operands.H_bc_anomaly.sum("bc_region").values + mass.values @ correction + instrument_offset
        )
        np.testing.assert_allclose(centered_values, expected, atol=1e-12)
        np.testing.assert_allclose(absolute_values, centered_values + expected_shift, atol=1e-10)
        xr.testing.assert_allclose(shift, expected_shift)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("drop_first", [False, True])
@pytest.mark.parametrize("anomaly_scale", [0.0, 0.3])
def test_public_additive_builders_warn_for_unanchored_observation_offsets(cached, drop_first, anomaly_scale):
    import warnings

    inputs = _inputs()
    response = inputs.H_bc_mass.copy(data=np.tile([0.7, 0.3, 0.0], (4, 1)))
    inputs["H_bc_mass"] = response
    inputs["H_bc"] = (response * 400.0).assign_attrs(units="ppm")
    inputs["mf"] = (response.sum("bc_region") * 400.0 + 1.0).assign_attrs(units="ppm")
    kwargs = _kwargs(inputs)
    kwargs["bc_anomaly_scale"] = anomaly_scale
    kwargs["offset_prior"] = {"pdf": "normal", "mu": 0.0, "sigma": 0.1}
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        if cached:
            kwargs.update(
                tau_hours=4.0,
                site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.2},
                offset_drop_first=drop_first,
            )
            model = build_co2_cached_sigma_model(inputs.H, **bound_model_arguments(kwargs)).model
        else:
            model = build_co2_model(
                inputs.H, **bound_model_arguments({**kwargs, "offset_args": {"drop_first": drop_first}})
            )
    confounding = [
        warning
        for warning in recorded
        if "A model coefficient and unrestricted observation offsets share a constant observation response"
        in str(warning.message)
    ]
    assert len(confounding) == (0 if drop_first else 1)
    assert "bc_mean_shift" in model and "offset" in model
    assert ("bc_anomaly_latent" in model) == (anomaly_scale > 0)


def test_nonseparable_region_period_weights_preserve_whole_window_prior():
    """A period may have nonzero mean while the full-window weighted sum is zero."""
    from openghg_inversions.models import registered_model

    mass = xr.DataArray(
        np.tile([0.6, 0.4], (4, 1)),
        dims=("nmeasure", "bc_region"),
        coords={"nmeasure": np.arange(4), "bc_region": ["north", "south"]},
        attrs={"units": "1"},
    )
    raw = xr.Dataset(
        {
            "H_bc_mass": mass,
            "H_bc_anomaly": (mass * 0.01).assign_attrs(units="ppm"),
            "bc_reference": xr.DataArray(400.0, attrs={"units": "ppm"}),
        },
        attrs={"boundary_representation": "centered-v1"},
    )
    observations = mass.sum("bc_region").assign_coords(
        time=("nmeasure", pd.to_datetime(["2020-01-01", "2020-01-20", "2020-02-01", "2020-02-20"]))
    )
    periods = CenteredOperands.from_dataset(raw).bind(observations, frequency="monthly")
    weights = xr.DataArray(
        [1.0, 2.0, 3.0, 7.0], dims="bc_region", coords=periods.bc_region.coords, attrs={"units": "1"}
    )
    prior = {"pdf": "uniform", "lower": -1.0, "upper": 2.0}
    transport = replace(CenteredOperands.from_dataset(raw), weights=weights.unstack("bc_region"))
    prepared = AdditiveChoice(prior, 0.3, frequency="monthly").bind(observations, transport)
    w = weights.values / weights.sum().item()
    projection = np.eye(4) - np.ones((4, 1)) * w
    with registered_model() as model:
        result = prepared.add()
    basis = model["bc_anomaly_basis"].get_value()
    np.testing.assert_allclose(basis @ basis.T, projection @ projection.T, atol=1e-14)
    anomaly = basis @ np.array([0.2, -0.3, 0.4])
    np.testing.assert_allclose(w @ anomaly, 0.0, atol=1e-15)
    assert abs(weights.values[::2] @ anomaly[::2]) > 1e-3
    assert result.sampled_rvs[0].name == "bc_mean_shift"
    assert "bc_mean_shift_interval__" in model.initial_point()


def test_additive_boundary_materializes_shared_operands_at_explicit_model_boundary():
    import dask.array as da
    from dask import delayed
    calls = []

    @delayed
    def payload():
        calls.append("computed")
        return np.array([[0.7, 0.3], [0.2, 0.8]])

    mass = xr.DataArray(
        da.from_delayed(payload(), shape=(2, 2), dtype=float),
        dims=("nmeasure", "bc_region"),
        coords={"nmeasure": [0, 1], "bc_region": ["north", "south"]},
        attrs={"units": "1"},
    )
    operands = xr.Dataset(
        {
            "H_bc_mass": mass,
            "H_bc_anomaly": (mass * 0.001).assign_attrs(units="ppm"),
            "bc_reference": (mass.sum() * 200.0).assign_attrs(units="ppm"),
        }
    )
    weights = _period_weights([1.0, 2.0], mass.bc_region.values, ["whole_window"])
    observations = xr.DataArray([400.0, 400.0], dims="nmeasure", coords={"nmeasure": [0, 1]})
    prior = {"pdf": "normal", "mu": 0.0, "sigma": 0.2}
    transport = replace(CenteredOperands.from_dataset(operands), weights=weights)
    prepared = AdditiveChoice(prior, 0.3).bind(observations, transport)
    assert calls == []
    assert isinstance(prepared.data.H_bc_mass.data, da.Array)
    prepared = replace(prepared, data=prepared.data.compute())
    assert calls == ["computed"]
    assert isinstance(operands.H_bc_mass.data, da.Array)
    assert isinstance(prepared.data.H_bc_mass.data, np.ndarray)
    assert isinstance(prepared.data.bc_reference.data, np.ndarray)


@pytest.mark.parametrize("anomaly_scale", [0.0, 0.3])
def test_bound_additive_affine_result_equals_likelihood_and_keeps_resolved_scale(anomaly_scale):
    """The component owns its coefficients, transforms, and mean-only result."""
    from openghg_inversions.models import registered_model

    inputs = _inputs()
    boundary = AdditiveChoice({"pdf": "uniform", "lower": -0.5, "upper": 0.8}, anomaly_scale).bind(
        inputs.mf, boundary_transport(_kwargs(inputs))
    )
    assert boundary.choice.anomaly_scale == anomaly_scale
    boundary = replace(boundary, data=boundary.data.compute())
    with registered_model() as model:
        result = boundary.add()
    affine = pt.as_tensor_variable(result.fixed_contribution.values)
    for block in result.blocks:
        affine = affine + pt.dot(block.design.values, block.coefficients)
    evaluate = model.compile_fn(
        model.replace_rvs_by_values([result.likelihood_contribution, affine]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    assert len(result.blocks) == (2 if anomaly_scale else 1)
    assert len(result.sampled_rvs) == (2 if anomaly_scale else 1)
    assert "bc_mean_shift_interval__" in model.initial_point()
    assert ("bc_anomaly" in model) == bool(anomaly_scale)
    for transformed_mean in (-1.2, 0.0, 1.1):
        point = model.initial_point()
        point["bc_mean_shift_interval__"] = np.asarray(transformed_mean)
        if anomaly_scale:
            point["bc_anomaly_latent"] = np.array([0.1, -0.2])
        direct, lowered = evaluate(point)
        np.testing.assert_allclose(direct, lowered, rtol=1e-13, atol=1e-13)


def test_boundary_calendar_mismatch_rejected_before_lazy_payload_computation():
    from dask import delayed
    import dask.array as da

    inputs = _inputs().assign_coords(time=("nmeasure", np.array(
        ["2021-01-01T01", "2021-01-01T04", "2021-01-02T01", "2021-01-02T04"],
        dtype="datetime64[ns]",
    )))
    options = _kwargs(inputs)
    executions = []

    @delayed
    def payload(values):
        executions.append("computed")
        return values

    mass = inputs.H_bc_mass.copy(data=da.from_delayed(
        payload(inputs.H_bc_mass.values), shape=inputs.H_bc_mass.shape, dtype=float,
    ))
    operands = options["boundary_operands"].assign(
        H_bc_mass=mass, H_bc_anomaly=(mass * .01).assign_attrs(units="ppm"),
    )
    weights = xr.DataArray(
        da.from_delayed(payload(np.array([[1., 7.], [3., 2.], [5., 4.]])), shape=(3, 2), dtype=float),
        dims=("bc_curtain", "bc_period"),
        coords={"bc_curtain": inputs.bc_region.values,
                "bc_period": np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[ns]")},
        attrs={"units": "1"},
    )
    transport = replace(CenteredOperands.from_dataset(operands), weights=weights)

    def choice(anchor_time):
        return AdditiveChoice(options["bc_mean_shift_prior"], .3, frequency="1D", anchor_time=anchor_time)

    bound = choice("2021-01-01").bind(inputs.mf, transport)
    assert isinstance(bound.data.H_bc_mass.data, da.Array)
    assert executions == []
    # Both anchors have membership [0,0,1,1], but represent different physical intervals.
    with pytest.raises(ValueError, match="join='exact'"):
        choice("2021-01-01T12").bind(inputs.mf, transport)
    assert executions == []
