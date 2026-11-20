"""Retained-state calibration equations and scientific boundary failures."""

import json

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import pytensor
import xarray as xr

from openghg_inversions.biosphere_prior import BIOSPHERE_PRIOR_SCOPE, BiospherePrior, calibrate_biosphere_prior
from openghg_inversions.correlated_state import CorrelatedPrior
from openghg_inversions.models.biosphere import annotate_biosphere_prior, resolve_retained_flux_prior
from openghg_inversions.models.components import add_correlated_state_with_activity
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.state_activity import resolve_state_activity
from openghg_inversions.serialization import save_trace, load_trace


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def case(order=None, units="kg / year", scale=1):
    labels = ["g-a", "t-a", "g-b", "fossil", "outer"]
    mean = xr.DataArray([1.1, 1, 0.9, 1.2, 1], dims="state", coords={"state": labels})
    latent = np.array(
        [
            [0.15, 0.02, 0.01, 0.015, 0.005],
            [0.02, 0.12, 0.01, 0.005, 0.01],
            [0.01, 0.01, 0.1, 0.008, 0.004],
            [0.015, 0.005, 0.008, 0.08, 0.01],
            [0.005, 0.01, 0.004, 0.01, 0.07],
        ]
    )
    covariance = np.outer(mean, mean) * np.expm1(latent)
    w = xr.DataArray(
        np.array([[-10, 0, -6, 0, 0], [0, 14, 0, 0, 0]]) * scale,
        dims=("process", "state"),
        coords={"process": ["GPP", "TER"], "state": labels},
        attrs={"units": units},
    )
    ref = xr.DataArray(
        np.array([-17.0, 13.0]) * scale,
        dims="process",
        coords={"process": ["GPP", "TER"]},
        attrs={"units": units},
    )
    bio = mean.copy(data=[True, True, True, False, False])
    if order is not None:
        mean, w, bio = (a.isel(state=order) for a in (mean, w, bio))
        covariance = covariance[np.ix_(order, order)]
    return CorrelatedPrior(mean, covariance), dict(biosphere=bio, weights=w, reference=ref)


def calibrate(factor=0.5, **kwargs):
    prior, inputs = case()
    return calibrate_biosphere_prior(prior, **inputs, residual_sd_multiplier=factor, **kwargs)


def sampled(result):
    """Return the prior a model samples for a calibration of ``case()``."""
    return resolve_retained_flux_prior(case()[0], result)


def test_identity_is_same_joint_lognormal_not_only_moments():
    baseline, _ = case()
    resolved = sampled(calibrate(1))
    np.testing.assert_array_equal(resolved.arithmetic_covariance, baseline.arithmetic_covariance)
    np.testing.assert_array_equal(resolved.lognormal.latent_mean, baseline.lognormal.latent_mean)
    np.testing.assert_array_equal(resolved.lognormal.latent_covariance, baseline.lognormal.latent_covariance)


def test_rank_one_update_control_identities_and_analytic_functionals():
    result = calibrate()
    data = result.data
    b, new = data.baseline_covariance.to_numpy(), data.covariance.to_numpy()
    gpp, ter = data.weights.to_numpy()
    net, gross = gpp + ter, -gpp + ter
    controls = np.vstack([net, np.eye(5)[~data.biosphere.to_numpy()]])
    r = data.residual_weights.to_numpy()
    np.testing.assert_allclose(controls @ new, controls @ b, atol=1e-14)
    np.testing.assert_allclose(r @ new @ r, 0.25 * (r @ b @ r), rtol=1e-13)
    np.testing.assert_allclose(new - b, -0.75 * np.outer(b @ r, b @ r) / (r @ b @ r), atol=1e-16)
    functionals = result.functionals()
    np.testing.assert_allclose(functionals.functional_mean, [-17, 13, -4, 30, 0])
    for name, weight in (("GPP", gpp), ("TER", ter), ("net", net), ("gross", gross), ("residual_gross", r)):
        np.testing.assert_allclose(
            functionals.functional_sd.sel(functional=name) ** 2, [weight @ b @ weight, weight @ new @ weight]
        )
    # Net/gross covariance is retained as part of C B, never set independent.
    assert abs(net @ b @ gross) > 0
    np.testing.assert_allclose(net @ new @ gross, net @ b @ gross)
    assert np.linalg.matrix_rank(new - b, tol=1e-12) == 1


def test_factor_residual_target_marginal_target_and_unit_conversion_agree():
    result = calibrate()
    baseline, inputs = case()
    for name, key in (("residual_gross", "target_residual_sd"), ("gross", "target_gross_sd")):
        target = xr.DataArray(
            float(result.functionals().functional_sd.sel(prior="calibrated", functional=name)) * 1000,
            attrs={"units": "g / year"},
        )
        other = calibrate_biosphere_prior(baseline, **inputs, **{key: target})
        np.testing.assert_allclose(result.data.covariance, other.data.covariance, atol=1e-15)
    prior_scaled, inputs_scaled = case(units="g / year", scale=1000)
    scaled = calibrate_biosphere_prior(prior_scaled, **inputs_scaled, residual_sd_multiplier=0.5)
    np.testing.assert_allclose(result.data.covariance, scaled.data.covariance, atol=1e-15)
    np.testing.assert_allclose(scaled.functionals().functional_sd, result.functionals().functional_sd * 1000)


def test_label_permutation_and_unequal_process_counts():
    original = calibrate()
    permutation = [4, 2, 0, 3, 1]
    baseline, inputs = case(permutation)
    result = calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=0.5)
    np.testing.assert_allclose(
        result.data.covariance, original.data.covariance.to_numpy()[np.ix_(permutation, permutation)]
    )
    np.testing.assert_allclose(result.functionals().functional_sd, original.functionals().functional_sd)


@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(residual_sd_multiplier=0.5, target_residual_sd=xr.DataArray(1)),
        dict(residual_sd_multiplier=-1),
        dict(residual_sd_multiplier=float("nan")),
    ],
)
def test_rejects_ambiguous_or_invalid_policy(options):
    baseline, inputs = case()
    with pytest.raises(ValueError):
        calibrate_biosphere_prior(baseline, **inputs, **options)


def test_infeasible_marginal_target_and_invalid_labels_and_units():
    baseline, inputs = case()
    with pytest.raises(ValueError, match="below explained variance"):
        calibrate_biosphere_prior(
            baseline, **inputs, target_gross_sd=xr.DataArray(0, attrs={"units": "kg/year"})
        )
    with pytest.raises(ValueError, match="align|index|coordinate"):
        calibrate_biosphere_prior(
            baseline,
            **{**inputs, "biosphere": inputs["biosphere"].isel(state=[1, 0, 2, 3, 4])},
            residual_sd_multiplier=0.5,
        )
    with pytest.raises(ValueError, match="declared flux units"):
        calibrate_biosphere_prior(baseline, **inputs, target_residual_sd=xr.DataArray(1))


def test_redundant_and_zero_residual_controls():
    baseline, inputs = case()
    # Net lives entirely in the protected block, so it is redundant with selectors.
    inputs["biosphere"] = inputs["biosphere"].copy(data=[True, False, True, False, False])
    inputs["weights"] = inputs["weights"].copy(data=[[0, 0, 0, -3, 0], [0, 0, 0, 0, 2]])
    with pytest.raises(ValueError, match="zero residual variance"):
        calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=0.5)
    # A net with zero weights is explicitly redundant, with a nonzero gross mode.
    inputs["weights"] = inputs["weights"].copy(data=[[-3, 0, 0, 0, 0], [3, 0, 0, 0, 0]])
    result = calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=0.8)
    change = result.data.covariance.values - result.data.baseline_covariance.values
    assert np.linalg.matrix_rank(change, tol=1e-12) == 1


def test_invalid_lognormal_is_rejected_without_covariance_projection():
    # Large gross inflation eventually makes the GPP/TER arithmetic
    # anticorrelation incompatible with a lognormal latent covariance.
    baseline, inputs = case()
    invalid = None
    for factor in (0.01, 3, 10, 30):
        try:
            calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=factor)
        except ValueError as exc:
            if "invalid for the prior's marginal families" in str(exc):
                invalid = factor
                break
    assert invalid is not None


def test_serialization_with_gathered_labels_reproduces_the_sampled_prior(tmp_path):
    result = calibrate()
    # Exercise gathered production state labels and the shared trace codec.
    index = pd.MultiIndex.from_arrays(
        [["GPP", "TER", "GPP", "FF", "outer"], [0, 0, 1, 0, 0]], names=["source", "region"]
    )
    data = result.data.drop_vars("state").assign_coords(xr.Coordinates.from_pandas_multiindex(index, "state"))
    # Rebuild the serialized second-axis tokens using the canonical constructor.
    base = CorrelatedPrior(data.baseline_mean, data.baseline_covariance.drop_vars("state_covariance_label"))
    data = data.assign_coords(state_covariance_label=base.arithmetic_covariance.state_covariance_label)
    path = tmp_path / "calibration.nc"
    save_trace(xr.DataTree.from_dict({"biosphere_prior": data}), path)
    reloaded = BiospherePrior.from_dataset(load_trace(path)["biosphere_prior"].dataset)
    np.testing.assert_array_equal(reloaded.data.covariance, result.data.covariance)
    before = resolve_retained_flux_prior(base, BiospherePrior(data)).lognormal
    after = resolve_retained_flux_prior(base, reloaded).lognormal
    for name in ("latent_mean", "latent_cholesky"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))


def test_saved_covariance_column_identity_is_checked_at_load_and_binding(tmp_path):
    data = calibrate().data
    labels = data.state_covariance_label
    reversed_columns = data.assign_coords(state_covariance_label=labels.copy(data=labels.values[::-1]))
    path = tmp_path / "calibration.nc"
    save_trace(xr.DataTree.from_dict({"biosphere_prior": reversed_columns}), path)
    with pytest.raises(ValueError, match="column identity"):
        BiospherePrior.from_dataset(load_trace(path)["biosphere_prior"].dataset)
    with pytest.raises(ValueError, match="state_covariance_label column identity"):
        BiospherePrior.from_dataset(data.drop_vars("state_covariance_label"))
    with pytest.raises(ValueError, match="column identity"):
        sampled(BiospherePrior(reversed_columns))


def test_prior_draws_match_joint_moments_and_annotation_records_scope():
    result = calibrate()
    prior = sampled(result)
    activity = resolve_state_activity(xr.zeros_like(prior.mean, dtype=bool), None)
    with registered_model() as model:
        add_correlated_state_with_activity(activity, prior, var_name="flux_scaling")
    # Vectorized compiled draws are secondary to the analytic checks above.
    with model:
        draws = pm.draw(model.flux_scaling, draws=30000, random_seed=719)
    np.testing.assert_allclose(draws.mean(0), prior.mean, atol=0.012)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), result.data.covariance, rtol=0.08, atol=0.005)
    assert np.all(draws > 0)
    trace = annotate_biosphere_prior(xr.DataTree(), result)
    xr.testing.assert_identical(trace["biosphere_prior"].to_dataset(), result.to_dataset())
    assert json.loads(trace.attrs["rhime_model_metadata"]) == {"biosphere_prior_scope": BIOSPHERE_PRIOR_SCOPE}


def test_identity_without_residual_mode_and_explicit_zero_target():
    baseline, inputs = case()
    inputs["weights"] = inputs["weights"].copy(data=[[0, 0, 0, -3, 0], [0, 0, 0, 0, 2]])
    identity = calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=1)
    np.testing.assert_array_equal(identity.data.covariance, baseline.arithmetic_covariance)
    assert identity.functionals().functional_sd.sel(functional="residual_gross").max() == 0
    xr.testing.assert_identical(BiospherePrior.from_dataset(identity.to_dataset()).data, identity.data)
    with pytest.raises(ValueError, match="singular"):
        calibrate(0)


def test_shared_lazy_functional_inputs_are_computed_once_and_not_mutated():
    from dask import delayed
    import dask.array as da

    baseline, inputs = case()
    calls = []

    @delayed
    def payload():
        calls.append(1)
        return np.concatenate([inputs["weights"].to_numpy().ravel(), inputs["reference"].to_numpy()])

    lazy = da.from_delayed(payload(), shape=(12,), dtype=float)
    weights = inputs["weights"].copy(data=lazy[:10].reshape((2, 5)))
    reference = inputs["reference"].copy(data=lazy[10:])
    result = calibrate_biosphere_prior(
        baseline,
        biosphere=inputs["biosphere"],
        weights=weights,
        reference=reference,
        residual_sd_multiplier=0.5,
    )
    assert calls == [1]
    assert weights.chunks is not None and reference.chunks is not None
    np.testing.assert_array_equal(result.data.weights, inputs["weights"])
    np.testing.assert_array_equal(result.data.reference, inputs["reference"])


def test_absolute_policy_roundtrip_and_corrupted_moments_rejected():
    baseline, inputs = case()
    result = calibrate()
    target = result.functionals().functional_sd.sel(prior="calibrated", functional="residual_gross").drop_vars(
        ["prior", "functional"]
    )
    absolute = calibrate_biosphere_prior(baseline, **inputs, target_residual_sd=target)
    restored = BiospherePrior.from_dataset(absolute.to_dataset())
    assert restored.data.attrs["policy"] == "target_residual_sd"
    corrupted = absolute.data.assign(covariance=-absolute.data.covariance)
    with pytest.raises(ValueError, match="residual-direction change"):
        BiospherePrior.from_dataset(corrupted)


def test_loading_frozen_calibration_does_not_resolve_policy_again(monkeypatch):
    prepared = calibrate()
    import openghg_inversions.biosphere_prior as module

    def unexpected_calibration(*args, **kwargs):
        raise AssertionError("Loading must consume saved calibrated moments")

    monkeypatch.setattr(module, "calibrate_biosphere_prior", unexpected_calibration)
    restored = BiospherePrior.from_dataset(prepared.to_dataset())
    xr.testing.assert_identical(restored.data, prepared.data)


def with_residual(data, residual):
    """Replace the residual weights and rebuild a matching covariance change."""
    data = data.copy(deep=True)
    b = data.baseline_covariance.values
    direction = b @ residual
    factor = data.attrs["residual_sd_multiplier"]
    data["covariance"].values = b + (factor**2 - 1) * np.outer(direction, direction) / (residual @ direction)
    data["residual_weights"].values = residual
    return data


def test_loader_rejects_asymmetric_or_scaled_covariance():
    data = calibrate().to_dataset().copy(deep=True)
    data.covariance.values[0, 1] += 0.01
    with pytest.raises(ValueError, match="residual-direction change"):
        BiospherePrior.from_dataset(data)
    data = calibrate().to_dataset()
    with pytest.raises(ValueError, match="residual-direction change"):
        BiospherePrior.from_dataset(data.assign(covariance=1.5 * data.covariance))


def test_loader_rejects_residual_weights_that_change_protected_covariance():
    data = calibrate().to_dataset()
    gross = data.weights.sel(process="TER").values - data.weights.sel(process="GPP").values
    with pytest.raises(ValueError, match="preserve protected"):
        BiospherePrior.from_dataset(with_residual(data, gross))


@pytest.mark.parametrize("units,scale", [("kg/year", 1), ("petagram/year", 1e-12), ("nanogram/year", 1e12)])
def test_loader_net_preservation_is_independent_of_reporting_units(units, scale):
    baseline, inputs = case(units=units, scale=scale)
    # With every state selected, only net is protected.
    inputs["biosphere"] = xr.ones_like(inputs["biosphere"])
    calibrated = calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=.5)
    restored = BiospherePrior.from_dataset(calibrated.to_dataset())
    xr.testing.assert_identical(restored.data, calibrated.data)
    weights = calibrated.data.weights
    gross = weights.sel(process="TER").values - weights.sel(process="GPP").values
    with pytest.raises(ValueError, match="preserve net-functional"):
        BiospherePrior.from_dataset(with_residual(calibrated.to_dataset(), gross))


def test_loader_accepts_frozen_calibration_with_exactly_zero_net_functional():
    baseline, inputs = case()
    weights = inputs["weights"].copy(deep=True)
    weights.loc[{"process": "TER"}] = -weights.sel(process="GPP")
    reference = inputs["reference"].copy(deep=True)
    reference.loc[{"process": "TER"}] = -reference.sel(process="GPP")
    inputs.update(weights=weights, reference=reference)
    calibrated = calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=.5)
    restored = BiospherePrior.from_dataset(calibrated.to_dataset())
    xr.testing.assert_identical(restored.data, calibrated.data)


def test_resolution_keeps_marginals_and_rejects_different_baseline_moments():
    baseline, _ = case()
    result = calibrate()
    resolved = resolve_retained_flux_prior(baseline, result)
    np.testing.assert_array_equal(resolved.arithmetic_covariance, result.data.covariance)
    xr.testing.assert_identical(resolved.mean, baseline.mean)
    xr.testing.assert_identical(resolved.marginal, baseline.marginal)
    assert resolve_retained_flux_prior(baseline, None) is baseline
    for other in (
        CorrelatedPrior(baseline.mean * 1.01, baseline.arithmetic_covariance),
        CorrelatedPrior(baseline.mean, baseline.arithmetic_covariance * 1.01),
    ):
        with pytest.raises(ValueError, match="baseline arithmetic moments"):
            resolve_retained_flux_prior(other, result)


def test_correlated_prior_accepts_only_the_lognormal_marginal():
    baseline, _ = case()
    marginal = baseline.marginal.copy(data=["lognormal", "lognormal", "weibull", "lognormal", "lognormal"])
    with pytest.raises(ValueError, match="Unsupported state marginal"):
        CorrelatedPrior(baseline.mean, baseline.arithmetic_covariance, marginal=marginal)
    with pytest.raises(ValueError, match="same order"):
        CorrelatedPrior(baseline.mean, baseline.arithmetic_covariance, marginal=baseline.marginal[::-1])


def test_all_fixed_calibration_has_no_sampled_state():
    from openghg_inversions.models import StateActivity

    prior = sampled(calibrate())
    activity = resolve_state_activity(
        xr.zeros_like(prior.mean, dtype=bool), StateActivity(active=False, fixed_value=1.7)
    )
    with registered_model() as model:
        state = add_correlated_state_with_activity(activity, prior, var_name="flux_scaling")
    assert state.latent is None
    assert not model.free_RVs
    np.testing.assert_array_equal(model.compile_fn(state.state)({}), np.full(5, 1.7))


def test_partial_activity_samples_the_exact_active_lognormal_marginal():
    from openghg_inversions.models import StateActivity

    active = np.array([True, False, True, False, True])
    result = calibrate()
    prior = sampled(result)
    activity = resolve_state_activity(
        xr.zeros_like(prior.mean, dtype=bool), StateActivity(active=active, fixed_value=1.7)
    )
    z = np.array([0.3, -0.2, 0.4])
    with registered_model() as model:
        state = add_correlated_state_with_activity(activity, prior, var_name="flux_scaling")
    m = result.data.baseline_mean.to_numpy()[active]
    b = result.data.covariance.to_numpy()[np.ix_(active, active)]
    latent_covariance = np.log1p(b / np.outer(m, m))
    expected = np.full(5, 1.7)
    expected[active] = np.exp(np.log(m) - np.diag(latent_covariance) / 2 + np.linalg.cholesky(latent_covariance) @ z)
    actual = model.compile_fn(
        model.replace_rvs_by_values([state.state]), inputs=model.value_vars
    )({"flux_scaling_active_latent": z})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    assert [rv.name for rv in model.free_RVs] == ["flux_scaling_active_latent"]


@pytest.mark.parametrize(("target_name", "functional"), [("target_residual_sd", "residual_gross"), ("target_gross_sd", "gross")])
def test_lazy_absolute_target_shares_one_materialization_with_functionals(target_name, functional):
    from dask import delayed
    import dask.array as da

    baseline, inputs = case()
    expected = calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=0.5)
    target_value = expected.functionals().functional_sd.sel(functional=functional, prior="calibrated").item()
    calls = []

    @delayed
    def payload():
        calls.append(1)
        return np.r_[inputs["weights"].values.ravel(), inputs["reference"].values, target_value]

    shared = da.from_delayed(payload(), shape=(13,), dtype=float)
    weights = inputs["weights"].copy(data=shared[:10].reshape(2, 5))
    reference = inputs["reference"].copy(data=shared[10:12])
    target = xr.DataArray(shared[12], attrs={"units": inputs["weights"].attrs["units"]})
    borrowed = [array.data for array in (weights, reference, target)]
    attributes = [dict(array.attrs) for array in (weights, reference, target)]
    actual = calibrate_biosphere_prior(
        baseline, biosphere=inputs["biosphere"], weights=weights, reference=reference, **{target_name: target},
    )
    assert calls == [1]
    for array, original, attrs in zip((weights, reference, target), borrowed, attributes, strict=True):
        assert array.data is original and array.chunks is not None
        assert array.attrs == attrs
    np.testing.assert_allclose(actual.data.covariance, expected.data.covariance, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(actual.functionals().functional_sd, expected.functionals().functional_sd, rtol=1e-13)


@pytest.mark.parametrize("target", [xr.DataArray(1.0), xr.DataArray([1.0], attrs={"units": "kg/year"})])
def test_invalid_target_metadata_fails_before_functional_payloads(target):
    from dask import delayed
    import dask.array as da

    baseline, inputs = case()

    @delayed
    def unexpected_compute():
        raise AssertionError("Invalid target computed reporting weights")

    weights = inputs["weights"].copy(data=da.from_delayed(unexpected_compute(), shape=(2, 5), dtype=float))
    with pytest.raises(ValueError, match="scalar with declared flux units"):
        calibrate_biosphere_prior(baseline, **dict(inputs, weights=weights), target_residual_sd=target)
