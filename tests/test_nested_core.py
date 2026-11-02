"""Equation, execution-boundary and real inference checks for the nested recipe."""

from dataclasses import replace

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from scipy.stats import norm

from openghg_inversions.inversion_data import PreparedInputs
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.rhime.nested import (
    build_nested_rhime_model,
    combine_nested_rhime_inputs,
    materialize_nested_inputs,
    run_nested,
)
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.sampling import RhimeSampler
from tests.test_nested_rhime import _basis, _prepared, _model_spec


def _nested():
    basis = _basis([50., 51.], [-2., -1.], np.array([[1, 2], [1, 2]]))
    outer = _prepared(times=["2019-01-01T00:00", "2019-01-01T01:00"],
                      sensitivity=np.array([[1., 2.], [3., 4.]]), basis=basis)
    inner = _prepared(times=["2019-01-01T00:00", "2019-01-01T01:00"],
                      sensitivity=np.array([[5., 6.], [7., 8.]]),
                      basis=_basis([50., 50.5], [-1.75, -1.25], np.array([[1, 2], [1, 2]])))
    for prepared in (outer, inner):
        prepared.inv_inputs["mf_repeatability"] = prepared.inv_inputs.mf_error.variable * .5
        prepared.inv_inputs["mf_variability"] = prepared.inv_inputs.mf_error.variable * .25
    return combine_nested_rhime_inputs(outer, inner, outer_overlap_masked=True, inner_domain_label="EUROPE-6km")


def test_nested_logp_matches_independent_two_grid_and_baseline_equation():
    nested = _nested()
    inputs = nested.combined.inv_inputs.copy(deep=False)
    inputs["H_bc"] = xr.DataArray([[.3, .4]], dims=("bc_region", "nmeasure"), coords={"bc_region": [0]})
    prepared = PreparedInputs(inputs, nested.combined.basis_functions, nested.combined.site_metadata)
    model_spec = replace(
        _model_spec(), use_bc=True, add_offset=True,
        bc_prior={"pdf": "normal", "mu": 1., "sigma": .5},
        offset_prior={"pdf": "normal", "mu": 0., "sigma": 2.},
        offset_args={"per_site": False},
    )
    materialized, priors = materialize_nested_inputs(prepared, model_spec)
    model = build_nested_rhime_model(materialized, model_spec, priors=priors).model
    point = {"x_outer": np.array([.8, 1.2]), "x_inner": np.array([1.4, .6]),
             "bc": np.array([.9]), "offset_latent": np.array(.7)}
    expected_mean = np.array([1., 3.]) @ point["x_outer"] + np.array([5., 7.]) @ point["x_inner"]
    expected_mean = np.array([expected_mean,
        np.array([2., 4.]) @ point["x_outer"] + np.array([6., 8.]) @ point["x_inner"]])
    expected_mean += np.array([.3, .4]) * .9 + .7
    expected_logp = (norm.logpdf(point["x_outer"], 1., .2).sum()
                     + norm.logpdf(point["x_inner"], 1., .2).sum()
                     + norm.logpdf(.9, 1., .5) + norm.logpdf(.7, 0., 2.)
                     + norm.logpdf(inputs.mf.values, expected_mean, inputs.mf_error.values).sum())
    np.testing.assert_allclose(model.compile_logp()(point), expected_logp, rtol=1e-6)
    values = model.compile_fn(model.replace_rvs_by_values([model["mu_outer"], model["mu_inner"], model["mu_bc"], model["offset"]]),
                              inputs=model.value_vars, on_unused_input="ignore")(point)
    np.testing.assert_allclose(np.sum(values, axis=0), expected_mean)


def test_nested_inputs_and_priors_compute_shared_graph_once():
    nested = _nested()
    calls = []

    @dask.delayed
    def payload():
        calls.append(1)
        return np.array([[1., 2.], [3., 4.]])

    shared = da.from_delayed(payload(), shape=(2, 2), dtype=float)
    inputs = nested.combined.inv_inputs.copy(deep=False)
    inputs["H"] = inputs.H.variable.copy(data=shared)
    inputs["H_inner"] = inputs.H_inner.variable.copy(data=shared + 4.)
    inputs["extension"] = xr.DataArray(shared, dims=("region", "nmeasure"))
    borrowed = PreparedInputs(inputs, nested.combined.basis_functions, nested.combined.site_metadata)
    prior = {"pdf": "normal", "mu": xr.DataArray(shared[:, 0], dims="region", coords={"region": inputs.region}), "sigma": .2}
    spec = replace(_model_spec(), sectors=(replace(_model_spec().sectors[0], x_prior=prior),))
    inner_prior = {**prior, "mu": prior["mu"].rename(region="inner_region")}
    eager, priors = materialize_nested_inputs(borrowed, spec, inner_x_prior=inner_prior)
    assert calls == [1]
    assert isinstance(inputs.H.data, da.Array)
    assert isinstance(eager.extension.data, da.Array)
    assert eager.extension.data is inputs.extension.data
    assert isinstance(priors["x_outer"]["mu"].data, np.ndarray)
    assert isinstance(priors["x_inner"]["mu"].data, np.ndarray)
    built = build_nested_rhime_model(eager, spec, priors=priors)
    assert built.model is not None
    assert calls == [1]
    with pytest.raises(ValueError, match="materialized"):
        build_nested_rhime_model(inputs, spec, priors=priors)


def test_nested_prepared_runner_samples_and_none_writes_nothing(monkeypatch, tmp_path):
    # The small synthetic grid is independent of the full OpenGHG domain.
    monkeypatch.setattr(
        "openghg_inversions.postprocessing.nested_paris_outputs.get_domain_info", lambda: {},
    )
    nested = _nested()
    config = RhimeConfig.from_params({
        "species": "ch4", "domain": "EUROPE", "sites": ["TAC"], "averaging_period": ["1h"],
        "start_date": "2019-01-01", "end_date": "2019-01-02", "flux_sources": ["inventory"],
        "use_bc": False, "mismatch_model": "fixed_error", "output_format": "none",
        "output_name": "nested", "output_path": str(tmp_path), "save_trace": True,
        "save_inversion_output": True,
        "x_prior": {"pdf": "normal", "mu": 1., "sigma": .2},
    }, multisector=False)
    sampler = RhimeSampler(draws=3, tune=3, burn=0, chains=1, nuts_sampler="pymc",
        sample_prior_predictive=3, sample_kwargs={"cores": 1, "random_seed": 42})
    config = replace(
        config, sampler=sampler,
        model=replace(config.model, sectors=(
            replace(config.model.sectors[0], state_activity=StateActivity(fixed_value=11.)),
        )),
    )
    inner_policy = StateActivity(fixed_value=7.)

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared execution must bypass acquisition/preparation and all output writers")

    monkeypatch.setattr("openghg_inversions.rhime.nested.acquire_merged_data", forbidden)
    monkeypatch.setattr("openghg_inversions.rhime.nested.prepare_nested_rhime_inputs", forbidden)
    monkeypatch.setattr("openghg_inversions.rhime.nested._write_nested_paris_outputs", forbidden)
    monkeypatch.setattr("openghg_inversions.rhime.nested.save_inversion_output", forbidden)
    output = run_nested(config=config, prepared=nested, inner_state_activity=inner_policy)
    assert output.outer.trace is output.inner.trace
    assert output.outer.trace["posterior"].sizes["draw"] == 3
    assert output.outer.trace["posterior_predictive"]["y"].sizes["nmeasure"] == 2
    assert output.outer.model_metadata["sectors"][0]["state_activity"]["fixed_value"] == 11.
    assert output.inner.model_metadata["sectors"][0]["state_activity"]["fixed_value"] == 7.
    assert output.inner.model_metadata["state_activity"]["fixed_value"] == 7.
    assert list(tmp_path.iterdir()) == []
    path = tmp_path / "inner.nc"
    output.inner.save(path)
    from openghg_inversions.postprocessing.inversion_output import InversionOutput
    restored = InversionOutput.load(path)
    assert restored.variable_name("flux_scale") == "x_inner"
    assert restored.output_metadata["nested_domain"]["inner_domain"] == "EUROPE-6km"
    xr.testing.assert_equal(restored.trace["posterior"]["x_inner"], output.inner.trace["posterior"]["x_inner"])

    country_file = tmp_path / "country_outer.nc"
    xr.Dataset(
        {"country": (("lat", "lon"), [[0, 1], [0, 1]]),
         "name": ("ncountries", ["OCEAN", "UNITEDKINGDOM"]),
         "country_code": ("ncountries", ["OCEAN", "GBR"])},
        coords={"lat": [50., 51.], "lon": [-2., -1.], "ncountries": [0, 1]},
    ).to_netcdf(country_file)
    # Reuse that posterior to check the requested PARIS/export route separately.
    monkeypatch.undo()
    monkeypatch.setattr(
        "openghg_inversions.postprocessing.nested_paris_outputs.get_domain_info", lambda: {},
    )
    monkeypatch.setattr("openghg_inversions.rhime.nested.sample_rhime_model", lambda built, sampler: output.outer.trace)
    paris_dir = tmp_path / "paris"
    outer_basis_path = tmp_path / "new-bases" / "outer.nc"
    inner_basis_path = tmp_path / "new-bases" / "inner.zarr"
    paris_config = replace(
        config, data=replace(config.data, basis_output_path=outer_basis_path),
        output=replace(
            config.output, output_format="paris", output_path=paris_dir, country_file=country_file,
            paris_postprocessing_kwargs={"template_version": "latest", "inversion_grid": False,
                                         "country_selections": None, "country_file_cache_dir": tmp_path / "countries"},
        ),
    )
    written = run_nested(
        config=paris_config, prepared=nested, inner_basis_output_path=inner_basis_path,
        inner_state_activity=inner_policy,
    )
    outer_flux = written.products["paris_flux"]
    inner_flux = written.products["paris_flux_inner"]
    np.testing.assert_array_equal(outer_flux.latitude, [50., 51.])
    np.testing.assert_array_equal(inner_flux.latitude, [50., 50.5])
    np.testing.assert_array_equal(inner_flux.longitude, [-1.75, -1.25])
    assert inner_flux.attrs["inner_domain"] == "EUROPE-6km"
    assert len(list(paris_dir.glob("*_flux_*.nc"))) == 2
    assert len(list(paris_dir.glob("*_conc_*.nc"))) == 1
    assert len(list(paris_dir.glob("*inversion_output*.nc"))) == 2
    assert len(list(paris_dir.glob("*_trace.nc"))) == 1
    restored_outer = InversionOutput.load(paris_dir / "nested2019-01-01_inversion_output.nc")
    restored_inner = InversionOutput.load(paris_dir / "nested2019-01-01_inversion_output_inner.nc")
    xr.testing.assert_equal(restored_outer.trace["posterior"], written.outer.trace["posterior"])
    xr.testing.assert_equal(restored_inner.trace["posterior"], written.inner.trace["posterior"])
    np.testing.assert_array_equal(restored_inner.flux.lat, [50., 50.5])
    assert restored_outer.basis_functions.basis_artifact_path == str(outer_basis_path)
    assert restored_inner.basis_functions.basis_artifact_path == str(inner_basis_path)
    assert restored_outer.model_metadata["sectors"][0]["state_activity"]["fixed_value"] == 11.
    assert restored_inner.model_metadata["sectors"][0]["state_activity"]["fixed_value"] == 7.
    assert restored_inner.model_metadata["state_activity"]["fixed_value"] == 7.
    assert nested.outer.basis_functions.basis_artifact_path is None
    assert nested.inner.basis_functions.basis_artifact_path is None
    assert set(written.products) == {"paris_flux", "paris_flux_inner", "paris_concentration"}
    assert set(written.paths) == {
        "paris_flux_path", "paris_flux_inner_path", "paris_concentration_path",
        "inversion_output_path", "inversion_output_inner_path", "trace_path",
    }
    assert all(path.exists() for path in written.paths.values())
    assert written.products["paris_concentration"].sizes["index"] == 2
    assert set(restored_outer.provenance) == {"version", "revision", "dirty"}
    assert not {"output_format", "output_path", "output_name", "save_trace", "save_inversion_output"} & (
        restored_outer.output_metadata.keys()
    )


@pytest.mark.parametrize("destination", [True, "custom.nc", "custom.zarr"])
def test_nested_artifact_suffix_uses_shared_writer_paths(tmp_path, destination):
    from types import SimpleNamespace
    from openghg_inversions.rhime.outputs import save_inversion_output
    from openghg_inversions.rhime.specs import RhimeOutputSpec

    saved = []
    output = SimpleNamespace(run_metadata={"start_date": "2019-01-01"}, save=saved.append)
    selected = destination if destination is True else tmp_path / destination
    spec = RhimeOutputSpec(
        output_format="inv_out", output_name="nested", output_path=tmp_path,
        save_inversion_output=selected, save_trace=False,
    )
    outer = save_inversion_output(output, spec)
    inner = save_inversion_output(output, spec, suffix="_inner")
    path = outer["inversion_output_path"]
    assert inner["inversion_output_inner_path"] == path.with_name(f"{path.stem}_inner{path.suffix}")
    assert saved == [path, inner["inversion_output_inner_path"]]


@pytest.mark.parametrize("override", [{}, {"inner_x_prior": None}])
def test_nested_explicit_prior_override_replaces_ini_value(monkeypatch, override):
    """Explicit None restores prior inheritance instead of reusing the file prior."""
    from tests.test_staged_workflow import _params

    file_prior = {"pdf": "normal", "mu": 5., "sigma": .2}
    params = _params(output_format="none", use_bc=False, inner_x_prior=file_prior)
    monkeypatch.setattr("openghg_inversions.rhime.nested.read_rhime_ini", lambda path: params)

    class ReachedMaterialization(Exception):
        pass

    def materialize(prepared, model_spec, *, inner_x_prior):
        assert inner_x_prior == override.get("inner_x_prior", file_prior)
        raise ReachedMaterialization

    monkeypatch.setattr("openghg_inversions.rhime.nested.materialize_nested_inputs", materialize)
    with pytest.raises(ReachedMaterialization):
        run_nested(config_file="nested.ini", prepared=_nested(), **override)


def test_nested_explicit_none_domain_replaces_ini_before_acquisition(monkeypatch):
    """CLI/Python overrides use the same merge, including an explicit missing domain."""
    from tests.test_staged_workflow import _params

    params = _params(output_format="none", inner_domain="6km")
    monkeypatch.setattr("openghg_inversions.rhime.nested.read_rhime_ini", lambda path: params)
    with pytest.raises(ValueError, match="requires `inner_domain`"):
        run_nested(config_file="nested.ini", inner_domain=None)


def test_nested_rejects_preparation_domain_with_supplied_inputs():
    """Prepared native identities cannot be replaced by an acquisition selector."""
    from tests.test_staged_workflow import _params

    config = RhimeConfig.from_params(_params(output_format="none"), multisector=False)
    with pytest.raises(ValueError, match="preparation options"):
        run_nested(config=config, prepared=_nested(), inner_domain="AFRICA")


def test_nested_no_basis_export_borrows_validated_preparations(monkeypatch):
    """Skipping exports must not reconstruct canonical prepared records."""
    from openghg_inversions.rhime.nested import NestedRhimePreparedInputs
    from tests.test_staged_workflow import _params

    canonical = _nested().validated()
    supplied = _nested()
    monkeypatch.setattr(NestedRhimePreparedInputs, "validated", lambda self: canonical)
    config = RhimeConfig.from_params(_params(output_format="none"), multisector=False)

    class ReachedMaterialization(Exception):
        pass

    def materialize(prepared, *args, **kwargs):
        assert prepared is canonical.combined
        raise ReachedMaterialization

    monkeypatch.setattr("openghg_inversions.rhime.nested.materialize_nested_inputs", materialize)
    # Exporting no destination returns the original basis. Reconstructing any
    # prepared record here would trigger canonicalization and break borrowing.
    def forbidden(*args, **kwargs):
        pytest.fail("No basis export should rebuild prepared inputs")

    monkeypatch.setattr(PreparedInputs, "__init__", forbidden)
    with pytest.raises(ReachedMaterialization):
        run_nested(config=config, prepared=supplied)
