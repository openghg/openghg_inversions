"""Retained-state calibration equations and scientific boundary failures."""

import json

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import pytensor
import xarray as xr

from openghg_inversions.biosphere_prior import BiospherePrior, calibrate_biosphere_prior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.biosphere import add_biosphere_state, annotate_biosphere_prior
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
    return CorrelatedLognormalPrior(mean, covariance), dict(biosphere=bio, weights=w, reference=ref)


def calibrate(factor=0.5, **kwargs):
    prior, inputs = case()
    return calibrate_biosphere_prior(prior, **inputs, residual_sd_multiplier=factor, **kwargs)


def test_identity_is_same_joint_lognormal_not_only_moments():
    result = calibrate(1)
    assert result.positive_prior is result.baseline
    np.testing.assert_array_equal(result.data.covariance, result.baseline.arithmetic_covariance)
    np.testing.assert_array_equal(result.positive_prior.latent_mean, result.baseline.latent_mean)
    np.testing.assert_array_equal(result.positive_prior.latent_covariance, result.baseline.latent_covariance)


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
    np.testing.assert_allclose(
        data.functional_mean.sel(prior="requested"), data.functional_mean.sel(prior="baseline")
    )
    for name, weight in (("GPP", gpp), ("TER", ter), ("net", net), ("gross", gross), ("residual_gross", r)):
        np.testing.assert_allclose(
            data.functional_sd.sel(functional=name) ** 2, [weight @ b @ weight, weight @ new @ weight]
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
            float(result.data.functional_sd.sel(prior="requested", functional=name)) * 1000,
            attrs={"units": "g / year"},
        )
        other = calibrate_biosphere_prior(baseline, **inputs, **{key: target})
        np.testing.assert_allclose(result.data.covariance, other.data.covariance, atol=1e-15)
    prior_scaled, inputs_scaled = case(units="g / year", scale=1000)
    scaled = calibrate_biosphere_prior(prior_scaled, **inputs_scaled, residual_sd_multiplier=0.5)
    np.testing.assert_allclose(result.data.covariance, scaled.data.covariance, atol=1e-15)
    np.testing.assert_allclose(scaled.data.functional_sd, result.data.functional_sd * 1000)


def test_label_permutation_and_unequal_process_counts():
    original = calibrate()
    permutation = [4, 2, 0, 3, 1]
    baseline, inputs = case(permutation)
    result = calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=0.5)
    np.testing.assert_allclose(
        result.data.covariance, original.data.covariance.to_numpy()[np.ix_(permutation, permutation)]
    )
    np.testing.assert_allclose(result.data.functional_sd, original.data.functional_sd)


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
    report = json.loads(result.data.attrs["calibration_report"])
    assert report["redundant_control_rows"] == 1


def test_invalid_lognormal_is_rejected_without_covariance_projection():
    # Large gross inflation eventually makes the GPP/TER arithmetic
    # anticorrelation incompatible with a lognormal latent covariance.
    baseline, inputs = case()
    invalid = None
    for factor in (0.01, 3, 10, 30):
        try:
            calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=factor)
        except ValueError as exc:
            if "invalid lognormal" in str(exc):
                invalid = factor
                break
    assert invalid is not None
    gaussian = calibrate_biosphere_prior(
        baseline, **inputs, residual_sd_multiplier=invalid, family="gaussian"
    )
    assert not json.loads(gaussian.data.attrs["calibration_report"])["lognormal_valid"]


def test_exact_enhancement_marginal_serialization_and_reconstruction(tmp_path):
    result = calibrate()
    labels = ["g-b", "g-a", "t-a", "fossil"]
    marginal = result.lognormal_marginal(labels)
    positions = [2, 0, 1, 3]
    np.testing.assert_array_equal(
        marginal.arithmetic_covariance, result.data.covariance.to_numpy()[np.ix_(positions, positions)]
    )
    np.testing.assert_allclose(
        marginal.latent_covariance,
        result.positive_prior.latent_covariance.to_numpy()[np.ix_(positions, positions)],
    )
    # Exercise gathered production state labels and the shared trace codec.
    index = pd.MultiIndex.from_arrays(
        [["GPP", "TER", "GPP", "FF", "outer"], [0, 0, 1, 0, 0]], names=["source", "region"]
    )
    data = result.data.drop_vars("state").assign_coords(xr.Coordinates.from_pandas_multiindex(index, "state"))
    # Rebuild the serialized second-axis tokens using the canonical constructor.
    base = CorrelatedLognormalPrior(
        data.baseline_mean, data.baseline_covariance.drop_vars("state_covariance_label")
    )
    data = data.assign_coords(state_covariance_label=base.arithmetic_covariance.state_covariance_label)
    path = tmp_path / "calibration.nc"
    save_trace(xr.DataTree.from_dict({"biosphere_prior": data}), path)
    reloaded = BiospherePrior.from_dataset(load_trace(path)["biosphere_prior"].dataset)
    np.testing.assert_array_equal(reloaded.data.covariance, result.data.covariance)
    z = np.array([0.4, -0.8, 0.2, 0.5, -0.1])
    before = result.data.reference.to_numpy() + result.data.weights.to_numpy() @ (
        np.exp(
            result.positive_prior.latent_mean.to_numpy()
            + result.positive_prior.latent_cholesky.to_numpy() @ z
        )
        - result.baseline.mean.to_numpy()
    )
    after = reloaded.data.reference.to_numpy() + reloaded.data.weights.to_numpy() @ (
        np.exp(
            reloaded.positive_prior.latent_mean.to_numpy()
            + reloaded.positive_prior.latent_cholesky.to_numpy() @ z
        )
        - reloaded.baseline.mean.to_numpy()
    )
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("family", ["lognormal", "gaussian"])
def test_prior_draws_match_joint_moments_and_report_negative_gaussian(family):
    result = calibrate(family=family)
    activity = resolve_state_activity(xr.zeros_like(result.baseline.mean, dtype=bool), None)
    with registered_model() as model:
        add_biosphere_state(activity, result.baseline, result)
    # Vectorized compiled draws are secondary to the analytic checks above.
    state_fn = model.compile_fn(model.flux_scaling, inputs=[model.flux_scaling_latent], point_fn=False)
    with model:
        draws = pm.draw(model.flux_scaling, draws=30000, random_seed=719)
    np.testing.assert_allclose(draws.mean(0), result.baseline.mean, atol=0.012)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), result.data.covariance, rtol=0.08, atol=0.005)
    bio = result.data.biosphere.to_numpy()
    assert np.all(draws[:, ~bio] > 0)
    if family == "gaussian":
        # Force a legitimate signed draw and retain both GPP and TER templates.
        z = np.zeros(5)
        z[0] = -20
        assert np.any(state_fn(z)[bio] < 0)
        assert np.max(abs(result.data.baseline_covariance.to_numpy()[np.ix_(bio, ~bio)])) > 0
    else:
        assert np.all(draws > 0)
    trace = annotate_biosphere_prior(xr.DataTree(), result)
    assert "biosphere_prior" in trace.children
    assert json.loads(trace.attrs["biosphere_prior"])["family"] == family


def test_identity_without_residual_mode_and_explicit_zero_target():
    baseline, inputs = case()
    inputs["weights"] = inputs["weights"].copy(data=[[0, 0, 0, -3, 0], [0, 0, 0, 0, 2]])
    identity = calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=1)
    assert identity.positive_prior is baseline
    assert identity.data.functional_sd.sel(functional="residual_gross").max() == 0
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
    target = result.data.functional_sd.sel(prior="requested", functional="residual_gross").drop_vars(
        ["prior", "functional"]
    )
    absolute = calibrate_biosphere_prior(baseline, **inputs, target_residual_sd=target)
    restored = BiospherePrior.from_dataset(absolute.to_dataset())
    assert json.loads(restored.data.attrs["calibration_report"])["policy"] == "target_residual_sd"
    corrupted = absolute.data.assign(covariance=absolute.data.covariance * 1.01)
    with pytest.raises(ValueError, match="inconsistent"):
        BiospherePrior.from_dataset(corrupted)


def test_sampling_comparison_uses_summed_mcse_and_frozen_comparator(tmp_path):
    from examples.biosphere_prior.compare_sampling import compare

    accepted = tmp_path / "accepted.csv"
    accepted.write_text("functional,posterior_mean,mcse_mean\nGPP,-10,2\n")
    reports = [
        dict(
            family=family,
            functionals={"GPP": {"mean": mean, "mcse_mean": 1}},
            max_rhat=1.01,
            min_bulk_ess=400,
            events={"diverging": 0},
            negative_biosphere_coefficient_fraction=0,
        )
        for family, mean in [("lognormal_original", -9), ("lognormal_shrunk", -13)]
    ]
    result = compare(reports, {"lognormal_original": accepted})
    assert result["accepted_comparisons"][0]["difference_over_summed_mcse"] == pytest.approx(1 / 3)
    assert result["treatment_comparisons"][0]["difference_over_summed_mcse"] == -2
