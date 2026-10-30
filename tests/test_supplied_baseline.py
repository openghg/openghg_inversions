"""Deterministic backgrounds preserve the inner operator and absolute mean."""

from dataclasses import replace

import dask.array as da
from dask import is_dask_collection
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr

from openghg_inversions.inversion_data import PreparedInputs
from openghg_inversions.rhime.baseline import prepare_supplied_baseline
from openghg_inversions.rhime.specs import (
    AdditiveSigmaSettings, FixedErrorSettings, PollutionEventSettings, RhimeModelSpec, SectorSpec,
)
from openghg_inversions.rhime.standard import build_standard_model, materialize_standard_inputs
from test_prepared_inputs_serialisation import _prepared_inputs


def _case(*, intem=False, **overrides):
    original = _prepared_inputs()
    observations = original.inv_inputs.mf
    baseline = xr.full_like(observations, .008).assign_attrs(units="mmol/mol", species="ch4")
    if intem:
        baseline = xr.Dataset({
            "boundary": baseline * .75, "exterior_flux": baseline * .25,
        }, attrs={"species": "ch4"})
        for value in baseline.data_vars.values():
            value.attrs["units"] = "mmol/mol"
    native = xr.DataArray(
        da.from_array(np.array([[[1., 2.], [30., 40.]]])),
        dims=("lat", "lon", "nmeasure"),
        coords={**original.basis_functions.flux.coords, **observations.coords},
        attrs={"units": "ppm"},
    )
    mask = xr.DataArray([[True, False]], dims=("lat", "lon"), coords=original.basis_functions.flux.coords)
    args = dict(
        native_response=native, inner_support=mask,
        inner_fixed_contribution=xr.full_like(observations, .5),
        baseline_site_metadata=original.site_metadata, species="ch4",
        provenance="synthetic independent background", constituents=("boundary", "exterior_flux"),
    )
    args.update(overrides)
    prepared = prepare_supplied_baseline(original, baseline, **args)
    spec = RhimeModelSpec(
        species="ch4", domain="EUROPE", use_bc=False, likelihood=FixedErrorSettings(),
        sectors=(SectorSpec("total", "inventory", {"pdf": "normal", "mu": 1., "sigma": .2}, "total"),),
    )
    return original, prepared, spec


def _build(prepared, spec, **kwargs):
    inputs, priors = materialize_standard_inputs(prepared, spec)
    return build_standard_model(inputs, spec, priors=priors, **kwargs)


def test_wur_style_inner_operator_units_and_borrowed_inputs():
    original, prepared, spec = _case()
    assert is_dask_collection(prepared.inv_inputs.H)
    np.testing.assert_allclose(prepared.inv_inputs.H.transpose("region", "nmeasure"), [[1, 2], [0, 0]])
    np.testing.assert_allclose(prepared.inv_inputs.supplied_baseline, 8)
    np.testing.assert_allclose(original.inv_inputs.H, [[1, 2], [3, 4]])
    np.testing.assert_allclose(original.basis_functions.flux, [[1.5, 2.5]])
    np.testing.assert_allclose(prepared.basis_functions.flux, [[1.5, 0]])
    built = _build(prepared, spec)
    model = built.model
    np.testing.assert_allclose(model["mu_inner"].eval({model["x"]: [2., 1.]}), [2.5, 4.5])
    np.testing.assert_allclose(model["mu_baseline"].eval(), [8, 8])
    assert built.output_contract.variable_roles["baseline"] == "mu_baseline"
    assert "legacy" not in built.output_contract.supported_output_formats


@pytest.mark.parametrize("likelihood", [FixedErrorSettings(), AdditiveSigmaSettings()])
def test_fixed_background_equals_subtracted_observations_under_same_error_model(likelihood):
    _, prepared, spec = _case()
    spec = replace(spec, likelihood=likelihood)
    absolute = _build(prepared, spec).model
    residual_inputs = prepared.inv_inputs.drop_vars(["supplied_baseline", "inner_fixed_contribution"])
    residual_inputs["mf"] = prepared.inv_inputs.mf - prepared.inv_inputs.supplied_baseline - .5
    residual = PreparedInputs(residual_inputs, prepared.basis_functions, prepared.site_metadata)
    subtracted = _build(residual, spec).model
    logp_absolute, logp_subtracted = absolute.compile_logp(), subtracted.compile_logp()
    for scale in (.3, 1., 2.):
        point = absolute.initial_point()
        point["x_active"][:] = scale
        assert logp_absolute(point) == pytest.approx(logp_subtracted(point))


def test_custom_likelihood_receives_complete_mean():
    _, prepared, spec = _case()
    def gaussian(*, mean, observations, observation_error, output_dim):
        pm.Deterministic("epsilon", pt.as_tensor_variable(observation_error.values), dims=output_dim)
        return pm.Normal("y", mean, observation_error.values, observed=observations.values, dims=output_dim)
    built = _build(prepared, replace(spec, likelihood=None), likelihood_builder=gaussian)
    assert np.isfinite(built.model.compile_logp()(built.model.initial_point()))


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_supplied_baseline_round_trip(tmp_path, suffix):
    _, prepared, spec = _case(intem=True)
    path = tmp_path / ("prepared" + suffix)
    prepared.save(path)
    restored = PreparedInputs.load(path)
    for name in ("supplied_baseline", "inner_fixed_contribution", "inner_domain_support",
                 "supplied_baseline_boundary", "supplied_baseline_exterior_flux"):
        xr.testing.assert_identical(restored.inv_inputs[name], prepared.inv_inputs[name])
    built = _build(restored, spec)
    with built.model:
        draws = pm.sample_prior_predictive(draws=2, random_seed=7)
    for group in ("prior",):
        ds = draws[group]
        np.testing.assert_allclose(ds.mu_total, ds.mu + .5 + 8)
        np.testing.assert_allclose(ds.mu_baseline, 8)


@pytest.mark.parametrize("overrides, message", [
    ({"constituents": ("boundary",)}, "constituents"),
    ({"constituents": ("boundary", "exterior_flux", "inner_fixed_contribution")}, "constituents"),
    ({"species": "co2"}, "species"),
])
def test_reject_ambiguous_accounting(overrides, message):
    with pytest.raises(ValueError, match=message):
        _case(**overrides)


def test_reject_double_boundary_and_nonfinite():
    _, prepared, spec = _case()
    inputs = prepared.inv_inputs.assign(H_bc=(("bc_region", "nmeasure"), [[1., 1.]]))
    prepared = PreparedInputs(inputs, prepared.basis_functions, prepared.site_metadata)
    with pytest.raises(ValueError, match="double-count"):
        _build(prepared, replace(spec, use_bc=True))
    bad = prepared.inv_inputs.copy()
    bad["supplied_baseline"] = bad.supplied_baseline.copy(data=[np.nan, 8.])
    with pytest.raises(ValueError, match="finite"):
        _build(PreparedInputs(bad, prepared.basis_functions, prepared.site_metadata), spec)


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("offset", [False, True])
def test_intem_sampled_correction_and_absolute_output_roundtrip(tmp_path, sampled, offset):
    from openghg_inversions.models.state_activity import StateActivity
    from openghg_inversions.postprocessing.inversion_output import InversionOutput
    from openghg_inversions.postprocessing.output_views import make_inversion_output
    from openghg_inversions.postprocessing.make_outputs import make_concentration_outputs
    _, prepared, spec = _case(intem=True, sampled_boundary_is_correction=True)
    inputs = prepared.inv_inputs.assign(H_bc=(("bc_region", "nmeasure"), [[.2, .3]]))
    prepared = PreparedInputs(inputs, prepared.basis_functions, prepared.site_metadata)
    spec = replace(spec, use_bc=sampled, add_offset=offset, bc_state_activity=StateActivity(active=False))
    built = _build(prepared, spec)
    with built.model:
        draws = pm.sample_prior_predictive(draws=3, random_seed=3)
    ds = draws.prior.to_dataset()
    expected_baseline = 8 + ds.mu_bc if sampled else xr.full_like(ds.mu_baseline, 8)
    np.testing.assert_allclose(ds.mu_baseline, expected_baseline)
    expected_offset = ds.offset if offset else 0
    np.testing.assert_allclose(ds.mu_total, ds.mu + .5 + ds.mu_baseline + expected_offset)
    trace = xr.DataTree.from_dict({"prior": ds, "posterior": ds})
    output = make_inversion_output(
        prepared=prepared, trace=trace, contract=built.output_contract,
        run_metadata={}, model_metadata={}, output_metadata={}, provenance={},
    )
    path = tmp_path / "output.nc"
    output.save(path)
    restored = InversionOutput.load(path)
    xr.testing.assert_identical(restored.trace_group("posterior").mu_total, ds.mu_total)
    np.testing.assert_allclose(restored.inv_inputs.supplied_baseline_exterior_flux, 2)
    assert restored.trace_dataset(var_roles="baseline").mu_baseline_posterior.attrs["units"] == "ppm"
    concentration = make_concentration_outputs(restored, stats=["mean"], combine_bc_and_offset=True)
    np.testing.assert_allclose(
        concentration.mu_bc_posterior_mean, (expected_baseline + expected_offset).mean(("chain", "draw")),
    )


def test_missing_observations_metadata_and_grid_are_rejected():
    original = _prepared_inputs()
    bad_metadata = original.site_metadata.assign(averaging_period=("site", ["2h", "2h"]))
    with pytest.raises(ValueError, match="metadata"):
        _case(baseline_site_metadata=bad_metadata)
    with pytest.raises(ValueError, match="missing retained"):
        _case(inner_fixed_contribution=original.inv_inputs.mf.isel(nmeasure=[0]))
    mask = xr.DataArray([[True, False]], dims=("lat", "lon"), coords={"lat": [52.], "lon": [-2., -1.]})
    with pytest.raises(ValueError, match="align"):
        _case(inner_support=mask)


def test_observation_derived_pollution_scaling_excludes_offset():
    _, prepared, spec = _case()
    spec = replace(spec, add_offset=True, likelihood=PollutionEventSettings(
        pollution_events_from_obs=True, power=2., sigma_per_site=False,
    ))
    model = _build(prepared, spec).model
    # Evaluation through RV replacement makes the offset independent of PEFO scaling.
    epsilon = model["epsilon"].eval({model["sigma"]: np.array([[.2]], dtype=model["sigma"].dtype)})
    np.testing.assert_allclose(epsilon, np.sqrt(prepared.inv_inputs.mf_error.values ** 2 +
                                             ((prepared.inv_inputs.mf.values - 8) * .2) ** 2))
