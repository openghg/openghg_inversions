"""Resolved configuration reaches execution without another parsing pass."""

import pytest

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
    monkeypatch.setattr(recipe, "load_rhime_data", acquire)
    with pytest.raises(ReachedAcquisition):
        runner(config=config)


@pytest.mark.parametrize("multisector_mode", [False, True])
@pytest.mark.parametrize("overrides", [{"config_file": "missing.ini"}, {"species": "co2"}])
def test_resolved_request_rejects_ambiguous_overrides(monkeypatch, multisector_mode, overrides):
    recipe = multisector if multisector_mode else standard
    runner = recipe.run_rhime_multisector if multisector_mode else recipe.run_rhime
    config = _config(multisector_mode)
    monkeypatch.setattr(recipe, "load_rhime_data", lambda **kwargs: pytest.fail("Unexpected acquisition"))
    with pytest.raises(ValueError, match="either resolved"):
        runner(config=config, **overrides)


@pytest.mark.parametrize("multisector_mode", [False, True])
def test_resolved_request_checks_recipe_and_likelihood_before_acquisition(monkeypatch, multisector_mode):
    recipe = multisector if multisector_mode else standard
    runner = recipe.run_rhime_multisector if multisector_mode else recipe.run_rhime
    monkeypatch.setattr(recipe, "load_rhime_data", lambda **kwargs: pytest.fail("Unexpected acquisition"))
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
    monkeypatch.setattr(recipe, "load_rhime_data", acquire)
    with pytest.warns(DeprecationWarning, match="deprecated"), pytest.raises(ReachedAcquisition):
        runner(config_file=path, sites=["TAC", "MHD"], output_name="winning-name")
