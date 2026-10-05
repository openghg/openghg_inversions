"""Canonical recipe boundaries retain labels, science, and borrowed arrays."""

from dataclasses import fields

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_data import preparation as acquisition
from openghg_inversions.rhime import multisector, standard
from openghg_inversions.rhime.params import resolve_rhime_options
from openghg_inversions.rhime.preparation import filter_rhime_observations, retrieve_or_reload_rhime_data


def _site_data():
    time = pd.date_range("2019-01-01", periods=2, freq="h")
    return xr.Dataset(
        {
            "fp_x_flux": (("time", "lat", "lon"), da.from_array([[[2.0, 3.0]], [[4.0, 5.0]]])),
            "mf": ("time", [5.0, 9.0], {"units": "1e-9"}),
            "mf_error": ("time", [1.0, 2.0], {"units": "1e-9"}),
            "mf_repeatability": ("time", [1.0, 2.0]),
            "mf_variability": ("time", [0.0, 0.0]),
        },
        coords={"time": time, "lat": [0.0], "lon": [0.0, 1.0]},
    )


def _config(multisector=False, **overrides):
    options = dict(
        species="ch4",
        domain="EUROPE",
        sites=["AAA", "BBB"],
        averaging_period=["1h", "2h"],
        inlet=["10m", "20m"],
        fp_height=["100m", "200m"],
        instrument=["first", "second"],
        platform=["surface", "surface"],
        obs_data_level=["1", "2"],
        met_model=["met-a", "met-b"],
        max_level=[1, 2],
        time_resolved=[False, True],
        start_date="2019-01-01",
        end_date="2019-01-02",
        output_name="recipe",
        flux_sources=["ff", "ocean"] if multisector else ["ff"],
        mismatch_model="fixed_error",
        use_bc=False,
        output_format="none",
    )
    options.update(overrides)
    return resolve_rhime_options(params=options, multisector=multisector)


@pytest.mark.parametrize("family", [standard, multisector])
@pytest.mark.parametrize("retention", ["acquisition", "cache", "filter"])
def test_preparation_retained_subsets_share_real_science(monkeypatch, family, retention):
    """IO subsets and real empty-site filtering yield the same projected values."""
    split = family is multisector
    setup = _config(split, reload_merged_data=retention == "cache", merged_data_dir="unused")
    data = _site_data()
    data.attrs["openghg_inversions_time_resolved"] = "true"
    if split:
        data["fp_x_flux_sectoral"] = xr.concat(
            [data.fp_x_flux, 2 * data.fp_x_flux],
            dim=xr.IndexVariable("source", ["ff", "ocean"]),
        )
    fp_all = {"BBB": data, ".species": "CH4", ".split_by_sectors": split}
    if retention == "filter":
        fp_all["AAA"] = data.isel(time=slice(0, 0))
        retained = ["AAA", "BBB"]
    else:
        retained = ["BBB"]

    def gather(**kwargs):
        assert kwargs["sites"] == ["AAA", "BBB"]
        assert kwargs["time_resolved"] == [False, True]
        values = setup.data_args
        indices = [values["sites"].index(site) for site in retained]
        metadata = [
            [values[name][index] for index in indices]
            for name in ("inlet", "fp_height", "instrument", "averaging_period")
        ]
        return fp_all, retained, *metadata

    monkeypatch.setattr(acquisition, "data_processing_surface_notracer", gather)
    monkeypatch.setattr(acquisition, "load_merged_data", lambda *args: dict(fp_all))
    basis_flat = xr.DataArray([[1, 1]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0, 1.0]})
    basis = BasisFunctions.from_flat_basis(
        basis_flat=basis_flat, flux=xr.ones_like(basis_flat), operator_kwargs={"state_dim": "region"}
    )
    monkeypatch.setattr(family, "build_rhime_basis", lambda merged, options: basis)
    merged = retrieve_or_reload_rhime_data(setup.data_args, multisector=split)
    filtered = filter_rhime_observations(merged, setup.data_args)
    requested = acquisition._SiteOptions.from_inputs(
        **{field.name: setup.data_args[field.name] for field in fields(acquisition._SiteOptions)}
    )
    assert filtered.site_options == requested.select_indices([1])
    operation = family.prepare_multisector_rhime_inputs if split else family.prepare_standard_rhime_inputs
    prepared = operation(merged, setup.data_args)
    assert prepared.sites == ("BBB",)
    assert prepared.averaging_period == ("2h",)
    expected = [[5.0, 9.0], [10.0, 18.0]] if split else [[5.0, 9.0]]
    np.testing.assert_allclose(
        prepared.inv_inputs.H.transpose(..., "nmeasure").values.reshape(-1, 2), expected
    )
    np.testing.assert_allclose(prepared.inv_inputs.mf, [5.0, 9.0])
    assert isinstance(data.fp_x_flux.data, da.Array)
    assert "H" not in data
    assert setup.run_spec.sites == ("AAA", "BBB")


@pytest.mark.parametrize("family", [standard, multisector])
def test_empty_preparation_fails_before_basis(monkeypatch, family):
    setup = _config(family is multisector)
    options = acquisition._SiteOptions.from_inputs(
        **{field.name: setup.data_args[field.name] for field in fields(acquisition._SiteOptions)}
    )
    merged = acquisition.RhimeMergedData(
        {site: _site_data().isel(time=slice(0, 0)) for site in options.sites},
        options,
    )
    monkeypatch.setattr(family, "build_rhime_basis", lambda *args: pytest.fail("empty input reached basis"))
    operation = (
        family.prepare_multisector_rhime_inputs
        if family is multisector
        else family.prepare_standard_rhime_inputs
    )
    with pytest.raises(ValueError, match="No sites remain after filtering"):
        operation(merged, setup.data_args)


def test_resolved_configuration_choices_cannot_leak_invocation_changes():
    config = _config(
        sample_kwargs={"nuts": {"target_accept": 0.9}},
        x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.5},
        paris_postprocessing_kwargs={"optional": {"enabled": False}},
    )
    sampler = config.sampler
    sampler.sample_kwargs["nuts"]["target_accept"] = 0.5
    config.run_spec.model.sectors[0].x_prior["mu"] = 99.0
    config.run_spec.output.paris_postprocessing_kwargs["optional"]["enabled"] = True
    config.data_args["averaging_period"][1] = "changed"
    assert config.sampler.sample_kwargs["nuts"]["target_accept"] == 0.9
    assert config.run_spec.model.sectors[0].x_prior["mu"] == 1.0
    assert config.run_spec.output.paris_postprocessing_kwargs["optional"]["enabled"] is False
    assert config.data_args["averaging_period"] == ["1h", "2h"]


def test_config_access_leaves_supplied_arrays_borrowed():
    array = xr.DataArray(da.ones((2,)), dims="time")
    config = _config(bc_input=array)
    assert config.data_args["bc_input"] is array
    assert config.data_args["bc_input"].data is array.data


@pytest.mark.parametrize("returned_sites", [["UNKNOWN"], ["BBB", "BBB"], []])
def test_acquisition_rejects_malformed_retained_labels(monkeypatch, returned_sites):
    config = _config()
    monkeypatch.setattr(
        acquisition,
        "data_processing_surface_notracer",
        lambda **kwargs: ({"BBB": _site_data()}, returned_sites, [], [], [], []),
    )
    with pytest.raises(ValueError):
        retrieve_or_reload_rhime_data(config.data_args, multisector=False)


@pytest.mark.parametrize("family", [standard, multisector])
def test_full_and_prepared_models_agree_without_checkpoint_roundtrip(monkeypatch, family):
    """Real construction agrees with the Normal reference at controlled states."""
    from scipy.stats import norm

    from openghg_inversions.rhime import _stage_artifacts
    from openghg_inversions.rhime.preparation import with_prepared_rhime_sites
    from openghg_inversions.rhime.prepared import run_rhime_from_prepared_inputs
    from openghg_inversions.rhime.sampling import RhimeSampler

    split = family is multisector
    setup = _config(split, x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.5})
    data = _site_data()
    data.attrs["openghg_inversions_time_resolved"] = "true"
    if split:
        data["fp_x_flux_sectoral"] = xr.concat(
            [data.fp_x_flux, 2 * data.fp_x_flux],
            dim=xr.IndexVariable("source", ["ff", "ocean"]),
        )
    options = acquisition._SiteOptions.from_inputs(
        **{field.name: setup.data_args[field.name] for field in fields(acquisition._SiteOptions)}
    ).select_indices([1])
    merged = acquisition.RhimeMergedData(
        {"BBB": data, ".species": "CH4", ".split_by_sectors": split},
        options,
    )
    flat = xr.DataArray([[1, 1]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0, 1.0]})
    basis = BasisFunctions.from_flat_basis(
        basis_flat=flat, flux=xr.ones_like(flat), operator_kwargs={"state_dim": "region"}
    )
    monkeypatch.setattr(family, "build_rhime_basis", lambda *args: basis)

    def forbidden(*args, **kwargs):
        pytest.fail("in-memory execution attempted intermediate persistence")

    monkeypatch.setattr(acquisition.RhimePreparedInputs, "save", forbidden)
    monkeypatch.setattr(_stage_artifacts, "write_json", forbidden)
    from openghg_inversions.inversion_data import get_data

    monkeypatch.setattr(get_data, "_save_merged_data", forbidden)
    logps = []

    def sample(self, model, **kwargs):
        point = model.initial_point()
        logps.append(model.compile_logp()(point))
        return xr.DataTree.from_dict(
            {
                "posterior": xr.Dataset(
                    {
                        name: (("chain", "draw", f"state_{name}"), np.ones((1, 1, len(value))))
                        for name, value in point.items()
                    }
                )
            }
        )

    monkeypatch.setattr(RhimeSampler, "sample", sample)
    raw = {
        **setup.data_args,
        "output_format": "none",
        "mismatch_model": "fixed_error",
        "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
    }
    raw.pop("split_by_sectors", None)
    run = family.run_rhime_multisector if split else family.run_rhime
    acquisition_calls = []

    def gather(**kwargs):
        acquisition_calls.append(True)
        assert kwargs["save_merged_data"] is False
        return merged.fp_all, ["BBB"], ["20m"], ["200m"], ["second"], ["2h"]

    monkeypatch.setattr(acquisition, "data_processing_surface_notracer", gather)
    full = run(**raw)
    supplied_merged = run(merged_data=merged, **raw)
    assert acquisition_calls == [True]
    xr.testing.assert_identical(supplied_merged.inv_inputs, full.inv_inputs)
    prepare = family.prepare_multisector_rhime_inputs if split else family.prepare_standard_rhime_inputs
    prepared = prepare(merged, setup.data_args)
    xr.testing.assert_identical(full.inv_inputs, prepared.inv_inputs)
    run_spec = with_prepared_rhime_sites(setup.run_spec, prepared)
    monkeypatch.setattr(
        family, "prepare_multisector_rhime_inputs" if split else "prepare_standard_rhime_inputs", forbidden
    )
    replay = run_rhime_from_prepared_inputs(
        prepared_inputs=prepared, run_spec=run_spec, sampler=RhimeSampler()
    )
    xr.testing.assert_identical(replay.inv_inputs, full.inv_inputs)
    prediction = np.array([15.0, 27.0]) if split else np.array([5.0, 9.0])
    expected = norm.logpdf([5.0, 9.0], loc=prediction, scale=[1.0, 2.0]).sum()
    expected += (2 if split else 1) * norm.logpdf(1.0, loc=1.0, scale=0.5)
    np.testing.assert_allclose(logps, [expected, expected, expected])
    assert isinstance(data.fp_x_flux.data, da.Array)
    assert "H" not in data


@pytest.mark.parametrize("family", [standard, multisector])
def test_real_prefilter_cache_reload_enters_canonical_stage_preparation(monkeypatch, tmp_path, family):
    """A real saved cache retains pre-filter observations and reloads without IO."""
    from openghg_inversions.inversion_data.serialise import _save_merged_data, load_merged_data
    from openghg_inversions.rhime import _multisector_stages, _standard_stages

    split = family is multisector
    setup = _config(
        split,
        save_merged_data=True,
        merged_data_dir=str(tmp_path / "cache"),
        merged_data_name="acquisition.nc",
        filters=["daily_median"],
    )
    data = _site_data()
    data.attrs["openghg_inversions_time_resolved"] = "true"
    if split:
        data["fp_x_flux_sectoral"] = xr.concat(
            [data.fp_x_flux, 2 * data.fp_x_flux],
            dim=xr.IndexVariable("source", ["ff", "ocean"]),
        )
    fp_all = {"BBB": data, ".split_by_sectors": split}
    acquisitions = []

    def gather(**kwargs):
        acquisitions.append(True)
        assert kwargs["save_merged_data"] is True
        _save_merged_data(fp_all, kwargs["merged_data_dir"], merged_data_name=kwargs["merged_data_name"])
        return fp_all, ["BBB"], ["20m"], ["200m"], ["second"], ["2h"]

    monkeypatch.setattr(acquisition, "data_processing_surface_notracer", gather)
    flat = xr.DataArray([[1, 1]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0, 1.0]})
    basis = BasisFunctions.from_flat_basis(
        basis_flat=flat, flux=xr.ones_like(flat), operator_kwargs={"state_dim": "region"}
    )
    monkeypatch.setattr(family, "build_rhime_basis", lambda *args: basis)
    merged = retrieve_or_reload_rhime_data(setup.data_args, multisector=split)
    prepare = family.prepare_multisector_rhime_inputs if split else family.prepare_standard_rhime_inputs
    expected = prepare(merged, setup.data_args)
    assert expected.inv_inputs.sizes["nmeasure"] == 1
    cached = load_merged_data(tmp_path / "cache", merged_data_name="acquisition.nc")
    xr.testing.assert_identical(cached["BBB"], data.compute())
    assert cached["BBB"].sizes["time"] == 2
    reload_setup = _config(
        split,
        reload_merged_data=True,
        merged_data_dir=str(tmp_path / "cache"),
        merged_data_name="acquisition.nc",
        filters=["daily_median"],
    )
    stages = _multisector_stages if split else _standard_stages
    manifest = stages.prepare(setup=reload_setup, output_dir=tmp_path / "stage")
    restored = acquisition.RhimePreparedInputs.load(tmp_path / "stage" / "prepared-inputs.nc")
    xr.testing.assert_identical(restored.inv_inputs, expected.inv_inputs)
    xr.testing.assert_identical(restored.site_metadata, expected.site_metadata)
    assert acquisitions == [True]
    assert "merged_data" not in manifest["artifacts"]
    assert "merged_data" not in manifest["artifact_identities"]
    assert not (tmp_path / "stage" / "merged-data").exists()


def test_public_setup_positional_sampler_constructor_preserves_choices():
    from openghg_inversions.rhime.params import RhimeRunnerSetup
    from openghg_inversions.rhime.sampling import RhimeSampler

    original = _config()
    runtime = RhimeSampler(draws=11, sample_kwargs={"nuts": {"target_accept": 0.85}})
    config = RhimeRunnerSetup(original.run_spec, runtime, original.data_args)
    runtime.sample_kwargs["nuts"]["target_accept"] = 0.5
    assert config.sampler.draws == 11
    assert config.sampler.sample_kwargs["nuts"]["target_accept"] == 0.85
    assert config.sampler is not config.sampler
