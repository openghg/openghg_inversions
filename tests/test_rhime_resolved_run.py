"""Resolved configuration reaches execution without another parsing pass."""

import pytest
import xarray as xr

from openghg_inversions.rhime import RhimeConfig, multisector, standard


def _config(multisector_mode, **overrides):
    return RhimeConfig.from_params(
        {
            "species": "ch4",
            "domain": "EUROPE",
            "sites": ["tac"],
            "averaging_period": "1h",
            "start_date": "2019-01-01",
            "end_date": "2019-02-01",
            "output_name": "resolved",
            "output_format": "none",
            "flux_sources": ["inventory", "ocean"] if multisector_mode else ["inventory"],
            "mismatch_model": "fixed_error",
            **overrides,
        },
        multisector=multisector_mode,
    )


@pytest.mark.parametrize("multisector_mode", [False, True])
@pytest.mark.parametrize("save", [False, True])
def test_resolved_request_reaches_acquisition_without_resolution(monkeypatch, multisector_mode, save):
    config = _config(multisector_mode, save_merged_data=save)
    recipe = multisector if multisector_mode else standard
    runner = recipe.run_rhime_multisector if multisector_mode else recipe.run_rhime

    def unexpected_resolution(*args, **kwargs):
        pytest.fail("A resolved request must not be parsed or resolved again")

    class ReachedAcquisition(Exception):
        pass

    def acquire(**kwargs):
        assert kwargs["site_options"] is config.site_options
        assert kwargs["flux_sources"] == config.flux_sources
        assert kwargs["save_merged_data"] is save
        raise ReachedAcquisition

    monkeypatch.setattr(RhimeConfig, "from_params", unexpected_resolution)
    monkeypatch.setattr(recipe.RhimeMergedData, "from_options", acquire)
    with pytest.raises(ReachedAcquisition):
        runner(config=config)


@pytest.mark.parametrize("multisector_mode", [False, True])
@pytest.mark.parametrize("overrides", [{"config_file": "missing.ini"}, {"species": "co2"}])
def test_resolved_request_rejects_ambiguous_overrides(monkeypatch, multisector_mode, overrides):
    recipe = multisector if multisector_mode else standard
    runner = recipe.run_rhime_multisector if multisector_mode else recipe.run_rhime
    config = _config(multisector_mode)
    monkeypatch.setattr(recipe.RhimeMergedData, "from_options", lambda **kwargs: pytest.fail("Unexpected acquisition"))
    with pytest.raises(ValueError, match="either resolved"):
        runner(config=config, **overrides)


@pytest.mark.parametrize("multisector_mode", [False, True])
def test_resolved_request_checks_recipe_and_likelihood_before_acquisition(monkeypatch, multisector_mode):
    recipe = multisector if multisector_mode else standard
    runner = recipe.run_rhime_multisector if multisector_mode else recipe.run_rhime
    monkeypatch.setattr(recipe.RhimeMergedData, "from_options", lambda **kwargs: pytest.fail("Unexpected acquisition"))
    with pytest.raises(ValueError, match="different sector layout"):
        runner(config=_config(not multisector_mode))
    with pytest.raises(ValueError, match="custom likelihood cannot be combined"):
        runner(config=_config(multisector_mode), likelihood_builder=lambda **kwargs: None)


@pytest.mark.parametrize("multisector_mode", [False, True])
def test_ini_runner_merges_before_one_resolution_and_alias_translation(monkeypatch, tmp_path, multisector_mode):
    """Raw file aliases and winning overrides reach the single factory boundary."""
    recipe = multisector if multisector_mode else standard
    runner = recipe.run_rhime_multisector if multisector_mode else recipe.run_rhime
    sources = ["inventory", "ocean"] if multisector_mode else ["inventory"]
    path = tmp_path / "request.ini"
    path.write_text(
        "[RHIME]\n" + "\n".join(f"{key} = {value!r}" for key, value in {
            "species": "ch4", "domain": "EUROPE", "sites": ["TAC"],
            "averaging_period": "1h", "start_date": "2019-01-01", "end_date": "2019-02-01",
            "outputname": "file-name", "output_format": "none", "emissions_name": sources,
            "mismatch_model": "fixed_error",
        }.items()),
        encoding="utf-8",
    )
    original_factory = RhimeConfig.from_params
    configs = []

    def resolve(cls, params, *, multisector):
        assert params["outputname"] == "file-name"
        assert params["emissions_name"] == sources
        resolved = original_factory(params, multisector=multisector)
        configs.append(resolved)
        return resolved

    class ReachedAcquisition(Exception):
        pass

    def acquire(**kwargs):
        assert len(configs) == 1
        assert kwargs["site_options"].sites == ("TAC", "MHD")
        assert kwargs["site_options"].averaging_period == ("1h", "1h")
        assert kwargs["output_name"] == "winning-name"
        raise ReachedAcquisition

    monkeypatch.setattr(RhimeConfig, "from_params", classmethod(resolve))
    monkeypatch.setattr(recipe.RhimeMergedData, "from_options", acquire)
    with pytest.warns(DeprecationWarning, match="deprecated"), pytest.raises(ReachedAcquisition):
        runner(config_file=path, sites=["TAC", "MHD"], output_name="winning-name")


@pytest.mark.parametrize("multisector_mode", [False, True])
def test_supplied_data_reaches_filtering_without_acquisition(monkeypatch, multisector_mode):
    from openghg_inversions.inversion_data import RhimeMergedData
    config = _config(multisector_mode, reload_merged_data=True, save_merged_data=True)
    supplied = RhimeMergedData.from_legacy_fp_all({"TAC": xr.Dataset(), ".split_by_sectors": multisector_mode}, config.site_options)
    recipe = multisector if multisector_mode else standard
    runner = recipe.run_rhime_multisector if multisector_mode else recipe.run_rhime
    def forbidden(*args, **kwargs):
        pytest.fail("supplied data must bypass both acquisition factories")
    class ReachedFiltering(Exception):
        pass
    def filter_supplied(merged, **kwargs):
        assert merged is supplied
        raise ReachedFiltering
    monkeypatch.setattr(RhimeMergedData, "load", forbidden)
    monkeypatch.setattr(RhimeMergedData, "from_options", forbidden)
    monkeypatch.setattr(recipe, "filter_rhime_observations", filter_supplied)
    with pytest.raises(ReachedFiltering):
        runner(config=config, merged_data=supplied)


@pytest.mark.parametrize("multisector_mode", [False, True])
def test_supplied_layout_is_checked_before_filtering(monkeypatch, multisector_mode):
    from openghg_inversions.inversion_data import RhimeMergedData
    config = _config(multisector_mode)
    supplied = RhimeMergedData.from_legacy_fp_all({"TAC": xr.Dataset(), ".split_by_sectors": not multisector_mode}, config.site_options)
    recipe = multisector if multisector_mode else standard
    runner = recipe.run_rhime_multisector if multisector_mode else recipe.run_rhime
    monkeypatch.setattr(recipe, "filter_rhime_observations", lambda *a, **kw: pytest.fail("layout not checked"))
    monkeypatch.setattr(supplied, "close", lambda: pytest.fail("Supplied record is borrowed"))
    with pytest.raises(ValueError, match="incompatible.*layout"):
        runner(config=config, merged_data=supplied)


@pytest.mark.parametrize("multisector_mode", [False, True])
def test_loaded_layout_rejection_closes_owned_record(monkeypatch, multisector_mode):
    from openghg_inversions.inversion_data import RhimeMergedData
    config = _config(multisector_mode, reload_merged_data=True, merged_data_dir="unused")
    loaded = RhimeMergedData(site_data={"TAC": xr.Dataset()}, flux_data={}, site_options=config.site_options, split_by_sectors=not multisector_mode)
    closed = []
    monkeypatch.setattr(loaded, "close", lambda: closed.append(True))
    monkeypatch.setattr(RhimeMergedData, "load", lambda **kw: loaded)
    recipe = multisector if multisector_mode else standard
    runner = recipe.run_rhime_multisector if multisector_mode else recipe.run_rhime
    with pytest.raises(ValueError, match="incompatible.*layout"):
        runner(config=config)
    assert closed == [True]


@pytest.mark.parametrize("multisector_mode", [False, True])
@pytest.mark.parametrize("source", ["supplied", "nc", "zarr"])
def test_acquired_data_binding_preserves_selectors_and_rejects_conflicts(
    monkeypatch, tmp_path, multisector_mode, source
):
    """Both runner paths bind actual reopened or supplied data before filtering."""
    import dask.array as da
    from dask.callbacks import Callback
    import numpy as np

    from openghg_inversions.inversion_data import RhimeMergedData, SiteOptions

    options = SiteOptions.from_inputs(sites=["TAC"], averaging_period="4h", inlet="100m")
    data = xr.Dataset(
        {"mf": ("time", da.from_array([1.0, 2.0], chunks=1))},
        coords={"time": np.array(["2019-01-01", "2019-01-02"], dtype="datetime64[ns]")},
    )
    supplied = RhimeMergedData(
        site_data={"TAC": data}, flux_data={}, site_options=options,
        split_by_sectors=multisector_mode,
        acquisition={"stage": "acquired", "species": "CH4", "domain": "europe",
                     "start_date": "2019-01-01T00:00:00", "end_date": "2019-02-01"},
    )
    config_options = {"inlet": "10m", "nbasis": 3, "output_name": "different-model-choice"}
    if source != "supplied":
        supplied.save(tmp_path, merged_data_name=f"acquired.{source}")
        config_options.update(reload_merged_data=True, merged_data_dir=tmp_path,
                              merged_data_name=f"acquired.{source}")
    recipe = multisector if multisector_mode else standard
    runner = recipe.run_rhime_multisector if multisector_mode else recipe.run_rhime
    reached = []

    class ReachedFiltering(Exception):
        pass

    def filtering(merged, **kwargs):
        reached.append(merged)
        assert merged.site_options == options
        assert merged.acquisition == supplied.acquisition
        assert merged.provenance == supplied.provenance
        assert isinstance(merged.site_data["TAC"].mf.data, da.Array)
        if source == "supplied":
            assert merged is supplied
            assert merged.site_data["TAC"].mf.data is data.mf.data
        raise ReachedFiltering

    monkeypatch.setattr(recipe, "filter_rhime_observations", filtering)
    monkeypatch.setattr(RhimeMergedData, "from_options", lambda **kw: pytest.fail("Unexpected acquisition"))
    calls = []
    try:
        with Callback(pretask=lambda *args: calls.append(args)):
            with pytest.raises(ReachedFiltering):
                runner(config=_config(multisector_mode, **config_options),
                       **({"merged_data": supplied} if source == "supplied" else {}))
            for key, value in [("species", "co2"), ("domain", "USA"),
                               ("start_date", "2019-01-02"), ("end_date", "2019-01-31")]:
                with pytest.raises(ValueError, match=key):
                    runner(config=_config(multisector_mode, **{**config_options, key: value}),
                           **({"merged_data": supplied} if source == "supplied" else {}))
        assert len(reached) == 1
        assert not calls
        assert supplied.acquisition["species"] == "CH4"
        assert supplied.site_options == options
    finally:
        if source != "supplied":
            for merged in reached:
                merged.close()
