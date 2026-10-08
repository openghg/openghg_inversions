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

    monkeypatch.setattr(recipe, "resolve_rhime_config", unexpected_resolution)
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
