"""Real coherent CO2 full/staged/replay products and independent equations."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal
import xarray as xr

from examples.co2_handoffs import END_DATE, START_DATE, configuration, synthetic_inputs
from openghg_inversions.basis.affine_flux_map_io import save as save_affine
from openghg_inversions.rhime.co2 import co2_runner
from openghg_inversions.rhime.co2.co2_affine_output import prepared_inputs_content_id
from openghg_inversions.rhime.specs import RhimeOutputSpec


@pytest.mark.parametrize("imported", [False, True], ids=["new-pair", "imported-pair"])
def test_configured_full_saved_and_replay_preserve_coherent_products(tmp_path, monkeypatch, imported):
    """Every product route uses retained scientific meaning, including signed flux."""
    from openghg_inversions.rhime.co2 import handoffs, run_co2_from_config
    from openghg_inversions.rhime.co2 import execution

    canonical, reduction, prepared, affine, countries = synthetic_inputs(tmp_path)
    params = configuration(tmp_path / "full-products", countries)
    setup = execution.resolve_co2_phase_options(params, phase="sample")["model"]
    if imported:
        source = tmp_path / "source.nc"
        companion = tmp_path / "source-affine.nc"
        prepared.save(source)
        save_affine(replace(affine, prepared_inputs_id=prepared_inputs_content_id(source)), companion)
        manifest = handoffs.prepare(
            output_dir=tmp_path / "prepared", prepared_path=source, affine_path=companion,
            start_date=START_DATE, end_date=END_DATE, domain="EUROPE",
        )
    else:
        manifest = handoffs.save_preparation(
            prepared, output_dir=tmp_path / "prepared", affine_artifact=affine,
            start_date=START_DATE, end_date=END_DATE, domain="EUROPE",
        )
    before = manifest.read_bytes()
    samples = handoffs.sample(manifest, model=setup, sampler=setup.sampler, output_dir=tmp_path / "samples")
    staged = handoffs.postprocess(
        samples, output=RhimeOutputSpec(
            output_format="basic", output_name="co2", save_inversion_output=False,
            country_file=str(countries), output_path=str(tmp_path / "staged-products"),
        ),
    )
    def forbidden(*args, **kwargs):
        raise AssertionError("Full execution and replay must not use checkpoint preparation or sampling")

    with monkeypatch.context() as full_checks:
        full_checks.setattr(handoffs, "load_preparation", forbidden)
        full_checks.setattr(handoffs, "save_preparation", forbidden)
        full_checks.setattr(handoffs, "sample", forbidden)
        full = run_co2_from_config(
            params=params, canonical_inputs=canonical, reduction=reduction,
            aggregation_error_rank=1, affine_artifact=affine,
        )
    for name in ("basic", "concentration_components", "native_flux", "country_flux"):
        xr.testing.assert_allclose(full.outputs[name], staged.outputs[name])
        if name != "concentration_components":
            assert Path(full.output_metadata[f"{name}_path"]).is_file()
            assert Path(staged.output_metadata[f"{name}_path"]).is_file()
    assert full.sampler is not None
    assert staged.sampler is None
    assert not hasattr(staged, "model")
    assert not hasattr(full, "run_spec")
    assert not hasattr(full, "model_spec")
    assert staged.start_date == full.start_date == START_DATE
    assert staged.end_date == full.end_date == END_DATE
    assert str(prepared.inv_inputs.time.min().values)[:10] == "2020-01-02"
    assert str(prepared.inv_inputs.time.max().values)[:10] == "2020-01-04"
    assert manifest.read_bytes() == before

    trace = staged.idata
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    posterior = trace.posterior.to_dataset()
    state = posterior[roles["flux_scale"]]
    pollution = xr.dot(prepared.inv_inputs.H, state, dim="region") + prepared.inv_inputs.fixed_prior_contribution
    np.testing.assert_allclose(
        posterior[roles["pollution_concentration"]],
        pollution.transpose(*posterior[roles["pollution_concentration"]].dims), rtol=2e-6,
    )
    mean = pollution + posterior[roles["boundary_concentration"]] + posterior[roles["offset_concentration"]]
    np.testing.assert_allclose(
        posterior[roles["model_mean"]], mean.transpose(*posterior[roles["model_mean"]].dims), rtol=2e-6,
    )
    # Independent bucket lift: the first column belongs to state 0, the second to state 1.
    shift = (state - prepared.inv_inputs.alpha_prior_mean).transpose("chain", "draw", "region").values
    native_draws = affine.affine_map.flux.values * (
        affine.affine_map.native_mean.values + shift[:, :, None, :]
    )
    native = staged.outputs["native_flux"]
    np.testing.assert_allclose(native.flux_total_posterior_mean.squeeze(), native_draws.mean((0, 1)), rtol=2e-6)
    np.testing.assert_allclose(native.flux_total_posterior_stdev.squeeze(), native_draws.std((0, 1)), rtol=2e-6)
    assert native.attrs["uncertainty_scope"] == "retained_state_conditional"
    assert staged.outputs["concentration_components"].sizes["prior_draw"] == 8
    assert staged.outputs["concentration_components"].sizes["draw"] == 12
    assert staged.outputs["concentration_components"].modelled_posterior.attrs["units"] == "ppm"

    built = co2_runner.build_rhime_co2(prepared_inputs=prepared, **setup.runner_kwargs)
    point = built.model.initial_point()
    mean_function = built.model.compile_fn(
        built.model.replace_rvs_by_values([built.model[roles["model_mean"]]])[0],
        inputs=built.model.value_vars, on_unused_input="ignore",
    )
    likelihood = float(built.model.compile_logp(vars=built.model.observed_RVs)(point))
    factor = prepared.inv_inputs.low_rank_factor.values
    covariance = factor @ factor.T + np.diag(
        prepared.inv_inputs.diagonal_residual_variance.values + prepared.inv_inputs.mf_error.values**2
    )
    reference = multivariate_normal.logpdf(prepared.inv_inputs.mf.values, mean=mean_function(point), cov=covariance)
    assert likelihood == pytest.approx(reference, rel=1e-5, abs=1e-4)

    monkeypatch.setattr(co2_runner, "build_rhime_co2", forbidden)
    monkeypatch.setattr(co2_runner, "build_co2_model", forbidden)
    monkeypatch.setattr(co2_runner, "sample_rhime_model", forbidden)
    monkeypatch.setattr(co2_runner, "materialize_pymc_inputs", forbidden)
    monkeypatch.setattr(handoffs, "prepare_co2_inputs", forbidden)
    monkeypatch.setattr(handoffs, "materialize_co2_inputs", forbidden)
    monkeypatch.setattr(handoffs, "build_co2_from_materialized", forbidden)
    monkeypatch.setattr(handoffs, "sample_rhime_model", forbidden)
    monkeypatch.setattr(execution, "run_co2_from_config", forbidden)
    replay = handoffs.postprocess(
        samples, output=RhimeOutputSpec(
            output_format="paris", output_name="replay", save_inversion_output=False,
            country_file=str(countries), output_path=str(tmp_path / "replay-products"),
        ),
    )
    assert replay.sampler is None
    assert replay.output_metadata["sampler"] == staged.output_metadata["sampler"]
    for name in ("basic", "concentration_components", "native_flux", "country_flux"):
        xr.testing.assert_identical(replay.outputs[name], staged.outputs[name])
    for name in ("paris_concentration", "paris_flux"):
        assert Path(replay.output_metadata[f"{name}_path"]).is_file()
    assert replay.outputs["paris_flux"].attrs["uncertainty_scope"] == "retained_state_conditional"
    requested_bounds = (
        np.asarray([[START_DATE, END_DATE]], dtype="datetime64[D]") - np.datetime64("1970-01-01")
    ) / np.timedelta64(1, "D")
    np.testing.assert_allclose(replay.outputs["paris_flux"].time_bnds, requested_bounds)


def test_configured_full_uses_output_policy_before_sampling(tmp_path, monkeypatch):
    """Unsupported output choices fail before expensive scientific execution."""
    from openghg_inversions.rhime.co2 import run_co2_from_config
    from openghg_inversions.rhime.co2 import execution

    _, _, prepared, affine, countries = synthetic_inputs(tmp_path)
    params = configuration(tmp_path / "unsupported", countries)
    params["outputs"]["output_format"] = "inv_out"

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported output must fail before model construction")

    monkeypatch.setattr(execution, "materialize_co2_inputs", forbidden)
    monkeypatch.setattr(execution, "build_co2_from_materialized", forbidden)
    with pytest.raises(ValueError, match="support"):
        run_co2_from_config(params=params, prepared=prepared, affine_artifact=affine)
    assert not (tmp_path / "unsupported").exists()


def test_full_rejects_supplied_preparation_with_an_unrelated_saved_affine_pair(tmp_path, monkeypatch):
    """The sampled preparation must own the reconstruction, even with output off."""
    from openghg_inversions.rhime.co2 import execution
    from openghg_inversions.rhime.co2.co2_preparation import Co2PreparedInputs
    from openghg_inversions.inversion_data import RhimePreparedInputs

    _, _, prepared, affine, countries = synthetic_inputs(tmp_path)
    source = tmp_path / "source.nc"
    companion = tmp_path / "source-affine.nc"
    prepared.save(source)
    save_affine(replace(affine, prepared_inputs_id=prepared_inputs_content_id(source)), companion)
    changed = prepared.inv_inputs.assign(alpha_prior_mean=prepared.inv_inputs.alpha_prior_mean * 1.1)
    changed.alpha_prior_mean.attrs = dict(prepared.inv_inputs.alpha_prior_mean.attrs)
    supplied = Co2PreparedInputs(
        RhimePreparedInputs(changed, prepared.basis_functions, prepared.site_metadata),
        prepared.aggregation_error_mode, prepared.provenance,
    )
    params = configuration(tmp_path / "rejected", countries)
    params["prepared_inputs"]["path"] = str(source)
    params["outputs"].update(output_format="none", reconstruction_path=str(companion))

    def forbidden(*args, **kwargs):
        raise AssertionError("A conflicting prepared/affine pair must fail before inference")

    monkeypatch.setattr(execution, "sample_rhime_model", forbidden)
    monkeypatch.setattr(execution, "materialize_co2_inputs", forbidden)
    with pytest.raises(ValueError, match="prepar|affine|source"):
        execution.run_co2_from_config(params=params, prepared=supplied)
    assert not (tmp_path / "rejected").exists()


def test_full_rejects_bucket_affine_flux_mismatch_before_inference(tmp_path, monkeypatch):
    """In-memory full execution applies the same bucket flux check as saved pairs."""
    from openghg_inversions.rhime.co2 import execution

    _, _, prepared, affine, countries = synthetic_inputs(tmp_path)
    changed_flux = (affine.affine_map.flux * 2).assign_attrs(affine.affine_map.flux.attrs)
    altered = replace(affine, affine_map=replace(affine.affine_map, flux=changed_flux))
    params = configuration(tmp_path / "rejected", countries)
    params["outputs"]["output_format"] = "none"

    def forbidden(*args, **kwargs):
        raise AssertionError("An incompatible bucket flux must fail before inference")

    monkeypatch.setattr(execution, "sample_rhime_model", forbidden)
    monkeypatch.setattr(execution, "materialize_co2_inputs", forbidden)
    with pytest.raises(ValueError, match="flux"):
        execution.run_co2_from_config(params=params, prepared=prepared, affine_artifact=altered)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("invalid", ["observation-dimension", "state-dimension", "state-units"])
def test_no_output_replay_rejects_inconsistent_scientific_role_meaning(tmp_path, invalid):
    """Disabled products cannot bypass saved scientific dimensions or units."""
    from openghg_inversions.rhime.co2 import handoffs
    from test_co2_staged_outputs import _components_fixture

    prepared, trace = _components_fixture()
    for group in ("prior", "posterior"):
        for name in ("mean_custom", "emission_custom", "bc_custom", "offset_custom", "sd_custom"):
            trace[group][name].attrs["units"] = "ppm"
    if invalid == "observation-dimension":
        trace.posterior["mean_custom"] = trace.posterior.mean_custom.isel(nmeasure=0, drop=True)
    elif invalid == "state-dimension":
        trace.posterior["state_custom"] = trace.posterior.state_custom.isel(region=0, drop=True)
    else:
        trace.posterior.state_custom.attrs["units"] = "ppm"
    prep = handoffs.save_preparation(
        prepared, output_dir=tmp_path / "prep", start_date=START_DATE, end_date=END_DATE, domain="EUROPE",
    )
    output = RhimeOutputSpec(output_format="none", save_inversion_output=False)
    # A reader owns external trace validation; live PyMC traces need no repeated check.
    with pytest.raises(ValueError, match="role|dimension|units|flux_scale"):
        sample = handoffs.save_samples(
            prep, trace, actual_inference={"draws": 3},
            preparation_generation=handoffs.load_preparation(prep).generation, output_dir=tmp_path / "sample",
        )
        handoffs.load_samples(sample, output=output)
