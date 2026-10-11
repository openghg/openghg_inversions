"""Resolved requests keep external shorthand separate from retained run facts."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import dask.array as da
from dask.callbacks import Callback
import pytest
import xarray as xr

import openghg_inversions.rhime as rhime
from openghg_inversions.hbmcmc.compatibility import params_from_config
from openghg_inversions.rhime.ini import read_rhime_ini
from openghg_inversions.hbmcmc.legacy_basis import basis_functions
from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.inversion_data import RhimeMergedData, SiteOptions
from openghg_inversions.rhime import multisector, standard
from openghg_inversions.rhime import params as rhime_params
from openghg_inversions.rhime import specs as rhime_specs
from openghg_inversions.hbmcmc.legacy_data import (
    from_legacy_fp_all,
    to_legacy_fp_all,
)


def _request(**overrides):
    return {
        "species": "ch4",
        "domain": "EUROPE",
        "sites": ["tac", "MHD"],
        "averaging_period": "1h",
        "start_date": "2019-01-01",
        "end_date": "2019-02-01",
        "output_name": "configuration",
        "output_format": "none",
        "flux_sources": ["inventory"],
        "mismatch_model": "pollution_event",
        **overrides,
    }


@pytest.mark.parametrize("multisector", [False, True])
def test_scalar_and_expanded_requests_resolve_equally(multisector):
    """All site selectors share the same scalar/sequence convention."""
    selectors = {
        "averaging_period": "1h",
        "inlet": "100m",
        "fp_height": "110m",
        "instrument": "instrument",
        "platform": "surface",
        "obs_data_level": "level",
        "met_model": "met",
        "max_level": 17,
        "time_resolved": None,
    }
    sources = ["inventory", "ocean"] if multisector else ["inventory"]
    scalar = rhime_params.RhimeConfig.from_params(
        _request(flux_sources=sources, **selectors), multisector=multisector
    )
    expanded = rhime_params.RhimeConfig.from_params(
        _request(flux_sources=sources, **{name: [value, value] for name, value in selectors.items()}),
        multisector=multisector,
    )

    assert scalar == expanded
    assert scalar.site_options.sites == ("TAC", "MHD")
    assert scalar.site_options.time_resolved == (None, None)
    assert isinstance(scalar.sampler, RhimeSampler)
    assert scalar.sampler.draws == 1000
    assert scalar.sampler.sample_posterior_predictive == ("y",)
    assert not hasattr(scalar, "run_spec")


@pytest.mark.parametrize(
    ("overrides", "option"),
    [
        ({"averaging_period": ["1h"]}, "averaging_period"),
        ({"instrument": ["one"]}, "instrument"),
        ({"max_level": [True, None]}, "max_level"),
        ({"sites": ["tac", "TAC"]}, "unique"),
        ({"unknown_option": True}, "unknown_option"),
    ],
)
def test_invalid_effective_requests_fail_without_acquisition(monkeypatch, overrides, option):
    def fail(*args, **kwargs):
        pytest.fail("option resolution must not acquire or sample")

    monkeypatch.setattr(rhime.RhimeMergedData, "from_options", fail)
    monkeypatch.setattr(RhimeSampler, "sample", fail)
    with pytest.raises(ValueError, match=option):
        rhime_params.RhimeConfig.from_params(_request(**overrides), multisector=False)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("flux_non_finite_check", "typo"),
        ("aggregation_error_mode", "typo"),
        ("basis_algorithm", "typo"),
        ("basis_algorithm", None),
        ("min_error", "typo"),
    ],
)
@pytest.mark.parametrize("multisector_mode", [False, True])
@pytest.mark.parametrize("from_ini", [False, True])
def test_invalid_active_choices_fail_before_acquisition(
    monkeypatch, tmp_path, option, value, multisector_mode, from_ini
):
    request = _request(
        flux_sources=["inventory", "ocean"] if multisector_mode else ["inventory"],
        **{option: value},
    )
    with pytest.raises(ValueError, match=option):
        rhime_params.RhimeConfig.from_params(request, multisector=multisector_mode)

    recipe = multisector if multisector_mode else standard
    runner = recipe.run_rhime_multisector if multisector_mode else recipe.run_rhime
    acquire = Mock(side_effect=AssertionError("invalid configuration must fail before acquisition"))
    monkeypatch.setattr(recipe.RhimeMergedData, "from_options", acquire)
    if from_ini:
        path = tmp_path / "invalid.ini"
        path.write_text(
            "[RHIME]\n" + "\n".join(f"{name} = {value!r}" for name, value in request.items()),
            encoding="utf-8",
        )
        request = {"config_file": path}
    with pytest.raises(ValueError, match=option):
        runner(**request)
    acquire.assert_not_called()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("flux_non_finite_check", "lazy"),
        ("flux_non_finite_check", "count"),
        ("aggregation_error_mode", "auto"),
        ("aggregation_error_mode", "none"),
        ("aggregation_error_mode", "dense"),
        ("aggregation_error_mode", "low_rank"),
        ("aggregation_error_mode", "diagonal"),
        ("min_error", "residual"),
        ("min_error", "percentile"),
    ],
)
def test_valid_finite_choices_are_preserved(option, value):
    config = rhime_params.RhimeConfig.from_params(_request(**{option: value}), multisector=False)
    owner = config.model if option == "aggregation_error_mode" else config
    assert getattr(owner, option) == value


@pytest.mark.parametrize("algorithm", ["typo", None])
def test_saved_basis_case_takes_precedence_during_resolution(algorithm):
    config = rhime_params.RhimeConfig.from_params(
        _request(fp_basis_case="saved_case", basis_algorithm=algorithm), multisector=False
    )
    assert config.fp_basis_case == "saved_case"
    assert config.basis_algorithm == algorithm


@pytest.mark.parametrize("algorithm", ("quadtree", "weighted", "region_constrained"))
def test_builtin_basis_algorithms_resolve(algorithm):
    config = rhime_params.RhimeConfig.from_params(_request(basis_algorithm=algorithm), multisector=False)
    assert config.basis_algorithm == algorithm


def test_basis_resolution_is_independent_of_legacy_registry(monkeypatch):
    monkeypatch.setitem(basis_functions, "project_algorithm", object())
    with pytest.raises(ValueError, match="basis_algorithm.*project_algorithm"):
        rhime_params.RhimeConfig.from_params(
            _request(basis_algorithm="project_algorithm"), multisector=False
        )


@pytest.mark.parametrize("replace_periods", [False, True])
def test_file_overrides_precede_site_expansion(tmp_path: Path, replace_periods):
    """A replaced list is not validated against either the old or new request."""
    configured = _request(averaging_period=["invalid-length"] if replace_periods else "1h")
    path = tmp_path / "configuration.ini"
    path.write_text(
        "[RHIME.OPTIONS]\n" + "\n".join(f"{name} = {value!r}" for name, value in configured.items()),
        encoding="utf-8",
    )
    raw = rhime.read_rhime_ini(path)
    assert raw["sites"] == ["tac", "MHD"]
    assert raw["averaging_period"] == configured["averaging_period"]
    overrides = {"sites": ["TAC", "MHD", "BSD"]}
    if replace_periods:
        overrides["averaging_period"] = "1h"

    config = rhime_params.RhimeConfig.from_params({**raw, **overrides}, multisector=False)
    direct = rhime_params.RhimeConfig.from_params({**configured, **overrides}, multisector=False)

    assert config == direct
    assert config.site_options.averaging_period == ("1h", "1h", "1h")
    assert raw["sites"] == ["tac", "MHD"]
    assert raw["averaging_period"] == configured["averaging_period"]


@pytest.mark.parametrize("multisector", [False, True])
def test_ini_decoding_precedes_recipe_defaults_and_overrides(tmp_path: Path, multisector):
    configured = _request(flux_sources=["inventory", "ocean"] if multisector else ["inventory"])
    path = tmp_path / "configuration.ini"
    path.write_text(
        "[REQUEST]\n"
        + "\n".join(f"{name} = {value!r}" for name, value in configured.items())
        + "\n[OTHER.SECTION]\nstart_date = '2001-01-01'\n",
        encoding="utf-8",
    )
    raw = rhime.read_rhime_ini(path)
    initial = rhime_params.RhimeConfig.from_params(raw, multisector=multisector)
    overrides = {"start_date": "2020-02-01"}
    updated = rhime_params.RhimeConfig.from_params({**raw, **overrides}, multisector=multisector)

    assert initial.start_date == "2019-01-01"
    assert updated.start_date == updated.model.likelihood.sigma_freq_anchor == "2020-02-01"
    assert updated == rhime_params.RhimeConfig.from_params({**configured, **overrides}, multisector=multisector)
    assert overrides == {"start_date": "2020-02-01"}


def test_ini_dictionary_adapter_retains_normalization_controls(tmp_path: Path):
    path = tmp_path / "adapter.ini"
    path.write_text("[RHIME]\noutputname = 'old-name'\ndraws = '7'\n", encoding="utf-8")
    with pytest.warns(DeprecationWarning, match="params_from_config"):
        raw = params_from_config(path, normalise=False)
    assert raw == {"outputname": "old-name", "draws": "7"}
    with pytest.warns(DeprecationWarning, match="params_from_config"), pytest.warns(DeprecationWarning, match="outputname"):
        normalized = params_from_config(
            path, output_path="cli-output", extra_kwargs={"output_path": "winning-output"}
        )
    assert normalized == {"output_name": "old-name", "draws": 7, "output_path": "winning-output"}


@pytest.mark.parametrize("invalid", [False, True])
def test_resolution_preserves_caller_containers_and_lazy_keywords(monkeypatch, invalid):
    """Configuration ownership never materializes opaque sampler payloads."""
    original = _request(
        xprior={"pdf": "normal", "mu": 1.0, "sigma": 0.5},
        inlet=[slice("10m", "100m"), None],
        sample_kwargs={"idata_kwargs": {"log_likelihood": False}, "random_seed": [2, 3]},
        posterior_predictive_kwargs={"compile_kwargs": {"mode": "FAST_COMPILE"}},
    )
    if invalid:
        original["instrument"] = ["wrong-length"]
    before = deepcopy(original)
    lazy = da.arange(3, chunks=2)
    original["sample_kwargs"]["opaque_value"] = lazy

    def fail(*args, **kwargs):
        pytest.fail("resolution must not execute sampling")

    monkeypatch.setattr(RhimeSampler, "sample", fail)
    with Callback(pretask=fail), pytest.warns(DeprecationWarning, match="xprior"):
        if invalid:
            with pytest.raises(ValueError, match="instrument"):
                rhime_params.RhimeConfig.from_params(original, multisector=False)
        else:
            config = rhime_params.RhimeConfig.from_params(original, multisector=False)
            assert config.sampler.sample_kwargs["opaque_value"] is lazy
            assert config.site_options.inlet == (slice("10m", "100m"), None)
            assert config.model.sectors[0].x_prior == before["xprior"]
            assert config.sampler.sample_kwargs["idata_kwargs"] is not original["sample_kwargs"]["idata_kwargs"]
            assert config.sampler.sample_kwargs["random_seed"] is not original["sample_kwargs"]["random_seed"]
            assert config.sampler.posterior_predictive_kwargs["compile_kwargs"] is not original["posterior_predictive_kwargs"]["compile_kwargs"]

    assert original["sample_kwargs"]["opaque_value"] is lazy
    without_opaque = {**original, "sample_kwargs": dict(original["sample_kwargs"])}
    del without_opaque["sample_kwargs"]["opaque_value"]
    assert without_opaque == before


def test_retained_run_uses_prepared_metadata_without_changing_request():
    config = rhime_params.RhimeConfig.from_params(
        _request(averaging_period=["1h", "2h"]), multisector=False
    )
    prepared = SimpleNamespace(sites=("MHD",), averaging_period=("2h",))

    run = config.retained_run_spec(prepared)

    assert config.site_options.sites == ("TAC", "MHD")
    assert config.site_options.averaging_period == ("1h", "2h")
    assert run.sites == ("MHD",)
    assert run.averaging_period == ("2h",)
    assert (run.start_date, run.end_date) == ("2019-01-01", "2019-02-01")
    assert run.model is config.model
    assert run.output is config.output


def test_resolved_choices_are_available_without_a_preparation_record():
    request = _request(use_bc=False, mismatch_model=None, draws="7", time_resolved=True)
    config = rhime_params.RhimeConfig.from_params(request, multisector=False)

    assert config.use_bc is config.model.use_bc is False
    assert config.sampler.draws == 7
    assert config.site_options.time_resolved == (True, True)
    assert config.obs_store == "user"
    assert config.basis_algorithm == "weighted"
    assert not hasattr(config, "preparation")
    assert not hasattr(config, "as_data_args")
    assert not hasattr(config, "use_tracer")


def test_canonical_configuration_exposes_active_prior_defaults():
    config = rhime_params.RhimeConfig.from_params(_request(add_offset=True), multisector=False)

    assert config.model.bc_prior == rhime_specs.DEFAULT_BC_PRIOR
    assert config.model.offset_prior == rhime_specs.DEFAULT_OFFSET_PRIOR
    assert config.model.likelihood.sigma_prior == rhime_specs.DEFAULT_POLLUTION_EVENT_SIGMA_PRIOR


def test_minimum_error_none_resolves_to_numeric_default():
    unspecified = rhime_params.RhimeConfig.from_params(_request(min_error=None), multisector=False)
    explicit = rhime_params.RhimeConfig.from_params(_request(min_error=0.0), multisector=False)

    assert unspecified == explicit
    assert unspecified.min_error == 0.0


def test_filtering_preserves_typed_filter_request():
    """The legacy filtering function can normalize a copy of owned choices."""
    request = _request(filters={"TAC": "six_hr_mean", "MHD": None})
    config = rhime_params.RhimeConfig.from_params(request, multisector=False)
    before = deepcopy(config.filters)
    data = xr.Dataset(
        {"mf": ("time", [1.0, 3.0])},
        coords={"time": [datetime(2019, 1, 1, 12), datetime(2019, 1, 1, 13)]},
    )
    merged = from_legacy_fp_all(
        fp_all={"TAC": data, "MHD": data}, site_options=config.site_options
    )

    filtered = rhime.filter_observations(merged, filters=config.filters)

    assert to_legacy_fp_all(filtered)["TAC"].sizes["time"] == 1
    assert to_legacy_fp_all(filtered)["TAC"].mf.item() == 2.0
    assert to_legacy_fp_all(filtered)["MHD"].sizes["time"] == 2
    assert config.filters == before
    assert request["filters"] == before
    assert to_legacy_fp_all(merged)["TAC"].sizes["time"] == 2


def test_configuration_api_is_public():
    assert rhime.RhimeConfig is rhime_params.RhimeConfig
    assert rhime.read_rhime_ini is read_rhime_ini
    config = rhime_params.RhimeConfig.from_params(_request(), multisector=False)
    assert isinstance(config.site_options, SiteOptions)
    for name in (
        "RhimePreparationConfig", "RhimeRunnerSetup", "make_rhime_runner_setup",
        "resolve_rhime_options", "load_rhime_config", "resolve_rhime_config",
    ):
        assert not hasattr(rhime_params, name)
        assert not hasattr(rhime, name)


def test_multisector_preparation_selects_sources_from_typed_request():
    """Tuple configuration labels remain a list selection at the xarray boundary."""
    config = rhime_params.RhimeConfig.from_params(
        _request(sites=["TAC"], flux_sources=["second", "first"], use_bc=False),
        multisector=True,
    )
    sensitivity = xr.DataArray(
        [[[1.0]], [[2.0]]],
        dims=("source", "region", "time"),
        coords={"source": ["first", "second"], "region": [0], "time": [datetime(2019, 1, 1)]},
    )
    merged = from_legacy_fp_all(
        {"TAC": sensitivity.sum("region").rename("fp_x_flux_sectoral").to_dataset()},
        config.site_options,
    )
    basis = SimpleNamespace(sensitivity=lambda values: sensitivity)

    site_data = rhime.build_sensitivities(
        merged, basis, domain=config.domain, flux_sources=config.flux_sources,
        use_bc=config.use_bc, bc_basis_case=config.bc_basis_case,
        bc_basis_directory=config.bc_basis_directory, multisector=True,
    )

    # Assignment preserves the merged dataset's labelled source axis.
    assert site_data["TAC"].H.source.values.tolist() == ["first", "second"]
    assert site_data["TAC"].H.sel(source="second").item() == 2.0
    assert config.flux_sources == ("second", "first")


def test_class_factory_preserves_inputs():
    request = _request(filters={"TAC": ["six_hr_mean"], "MHD": None})
    before = deepcopy(request)

    resolved = rhime_params.RhimeConfig.from_params(request, multisector=False)
    assert request == before
    assert resolved.filters is not request["filters"]
    assert resolved.filters["TAC"] is not request["filters"]["TAC"]


def test_config_selection_borrows_values_and_rejects_unknown_attributes():
    config = rhime_params.RhimeConfig.from_params(
        _request(filters={"TAC": ["six_hr_mean"]}), multisector=False,
    )

    selected = config.select("site_options", "species", "filters")

    assert tuple(selected) == ("site_options", "species", "filters")
    assert selected["site_options"] is config.site_options
    assert selected["filters"] is config.filters
    selected["species"] = "co2"
    assert config.species == "ch4"
    with pytest.raises(AttributeError, match="misspelled"):
        config.select("misspelled")


def test_configuration_uses_the_sampler_owning_defaults(monkeypatch):
    original_init = RhimeSampler.__init__

    def init_with_a_new_default(self, **kwargs):
        kwargs.setdefault("draws", 17)
        original_init(self, **kwargs)

    monkeypatch.setattr(RhimeSampler, "__init__", init_with_a_new_default)
    config = rhime_params.RhimeConfig.from_params(_request(), multisector=False)
    overridden = rhime_params.RhimeConfig.from_params(_request(draws=23), multisector=False)

    assert config.sampler.draws == 17
    assert overridden.sampler.draws == 23


@pytest.mark.parametrize("format_name", ["inv_out", "INV_OUT", "paris", "none"])
def test_output_owner_resolves_the_active_format_save_default(format_name, tmp_path):
    params = {"output_format": format_name, "output_path": str(tmp_path)}
    before = dict(params)
    output = rhime_specs.RhimeOutputSpec.from_params(params, multisector=False)

    assert output.output_format == format_name.lower()
    assert output.save_inversion_output is (format_name.lower() == "inv_out")
    assert params == before


def test_sector_priors_are_owned_independently_and_borrow_numerical_values():
    opaque = da.arange(3, chunks=2)
    prior = {"pdf": "normal", "mu": opaque, "parameters": {"labels": ["a"]}}
    config = rhime_params.RhimeConfig.from_params(
        _request(flux_sources=["first", "second"], x_prior=prior), multisector=True,
    )
    first, second = config.model.sectors

    assert first.x_prior["mu"] is second.x_prior["mu"] is opaque
    first.x_prior["parameters"]["labels"].append("b")
    assert second.x_prior["parameters"]["labels"] == ["a"]
    assert prior["parameters"]["labels"] == ["a"]


@pytest.mark.parametrize("invalid", [[], {}, 1, False, "unknown"])
def test_malformed_likelihood_selection_reports_the_option(invalid):
    with pytest.raises(ValueError, match="mismatch_model"):
        rhime_params.RhimeConfig.from_params(_request(mismatch_model=invalid), multisector=False)


def test_ini_reader_decodes_recipe_options_without_semantic_resolution(tmp_path, monkeypatch):
    path = tmp_path / "project.ini"
    path.write_text(
        "[PROJECT]\nproject_basis_path = 'basis.nc'\nbasis_algorithm = None\n"
        "sites = ['tac', 'MHD']\naveraging_period = '1h'\noutputname = 'legacy'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rhime_params.RhimeConfig, "from_params",
        lambda *args, **kwargs: pytest.fail("INI decoding must not resolve a recipe"),
    )

    assert rhime.read_rhime_ini(path) == {
        "project_basis_path": "basis.nc", "basis_algorithm": None,
        "sites": ["tac", "MHD"], "averaging_period": "1h", "outputname": "legacy",
    }


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), {"TAC": -1}, {"TAC": True}])
@pytest.mark.parametrize("multisector_mode", [False, True])
def test_invalid_minimum_error_fails_before_acquisition(monkeypatch, value, multisector_mode):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid minimum error reached acquisition")
    monkeypatch.setattr(RhimeMergedData, "from_options", forbidden)
    recipe = multisector if multisector_mode else standard
    runner = recipe.run_rhime_multisector if multisector_mode else recipe.run_rhime
    with pytest.raises(ValueError, match="min_error"):
        runner(**_request(min_error=value, flux_sources=["one", "two"] if multisector_mode else ["one"]))


def test_minimum_error_owns_any_mapping_without_requiring_dropped_sites():
    from collections import UserDict
    values = UserDict({"TAC": 0.5})
    config = rhime_params.RhimeConfig.from_params(_request(min_error=values), multisector=False)
    values["TAC"] = 99.0
    assert config.min_error == {"TAC": 0.5}
