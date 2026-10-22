"""Configuration contracts for the concrete CO2-family recipes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, cast

import pytest

from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.scalar_sigma import add_scalar_sigma_eigen_likelihood
from openghg_inversions.models.site_sigma import add_site_sigma_gaussian_likelihood
from openghg_inversions.rhime.co2.configuration import (
    co2_config_templates,
    load_co2_family_config,
    Co2ModelConfig,
    _resolve_co2_sampling,
    resolve_co2_prior_seed,
    Co2PreparationConfig,
    validate_co2_config,
)
from openghg_inversions.rhime.outputs import annotate_likelihood_trace
from openghg_inversions.rhime.sampling import RhimeSampler
from tests.helpers import make_trace


def _ordinary() -> dict[str, object]:
    return {
        "format_version": 1,
        "recipe": "co2",
        "prepared_inputs": {"path": "prepared.zarr"},
        "likelihood": {
            "kind": "additive_sigma",
            "sigma_prior": {"pdf": "Half-Normal", "sigma": 0.75},
        },
        "sampling": {
            "draws": 20,
            "tune": 10,
            "chains": 2,
            "target_accept": 0.95,
            "random_seed": 42,
        },
    }


def _linked() -> dict[str, object]:
    return {
        "format_version": 1,
        "recipe": "co2_o2",
        "variant": "linked",
        "channels": {
            "co2": {"units": "ppm", "independent_error_sd": 1.0},
            "o2": {"units": "ppm", "independent_error_sd": 2.0},
        },
        "sampling": {"nuts_sampler": "numpyro"},
    }


def test_loader_only_parses_toml(tmp_path: Path) -> None:
    path = tmp_path / "co2.toml"
    path.write_text(
        'format_version = 1\nrecipe = "co2"\n',
        encoding="utf-8",
    )

    assert load_co2_family_config(path) == {
        "format_version": 1,
        "recipe": "co2",
    }


def test_installed_templates_pass_the_semantic_resolver() -> None:
    from openghg_inversions.rhime.co2.co2_o2_configuration import (
        load_co2_o2_params, Co2O2ModelConfig,
        Co2O2PreparationConfig,
        resolve_co2_o2_sampler,
    )

    templates = co2_config_templates()

    assert tuple(templates) == (
        "co2.toml",
        "co2_fixed_ou.toml",
        "co2_o2.toml",
    )
    for resource in templates.values():
        config = load_co2_family_config(resource)
        if config.get("recipe") == "co2_o2":
            load_co2_o2_params(params=config)
            Co2O2PreparationConfig.from_params(config, require_path=True).preparation_arguments()
            resolve_co2_o2_sampler(config, model=Co2O2ModelConfig.from_params(config))
        elif config["recipe"] == "co2":
            validate_co2_config(config)
            Co2PreparationConfig.from_params(config, require_path=True).preparation_arguments()
            Co2ModelConfig.from_params(config).model_arguments()
            _resolve_co2_sampling(config, model=Co2ModelConfig.from_params(config).model_arguments()).sampler


def test_ordinary_configuration_matches_explicit_scientific_arguments() -> None:
    config = _ordinary()
    assert Co2ModelConfig.from_params(config).model_arguments() == {
        "no_model_error": False,
        "sigma_prior": {"pdf": "halfnormal", "sigma": 0.75},
    }
    assert Co2PreparationConfig.from_params(config, require_path=True).preparation_arguments()["prepared_path"] == Path("prepared.zarr").resolve()
    assert _resolve_co2_sampling(config, model=Co2ModelConfig.from_params(_ordinary()).model_arguments()).sampler == RhimeSampler(
        draws=20, tune=10, chains=2, nuts_sampler="numpyro",
        sample_kwargs={"target_accept": 0.95, "random_seed": 42},
        posterior_predictive_kwargs={"random_seed": 42},
    )




@pytest.mark.parametrize(
    ("kind", "options", "builder", "expected"),
    [
        (
            "site_sigma",
            {"site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0}},
            add_site_sigma_gaussian_likelihood,
            {"site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0}},
        ),
        (
            "fixed_ou",
            {"tau_hours": 24.0, "fixed_site_amplitudes": {"MHD": 0.5}},
            add_fixed_ou_gaussian_likelihood,
            {"tau_hours": 24.0, "fixed_site_amplitudes": {"MHD": 0.5}},
        ),
        (
            "scalar_sigma",
            {
                "eigenbasis_path": "sigma.nc",
                "sigma_prior": {"pdf": "gamma", "alpha": 2.0, "beta": 1.0},
            },
            add_scalar_sigma_eigen_likelihood,
            {
                "eigenbasis_path": Path("sigma.nc"),
                "sigma_prior": {"pdf": "gamma", "alpha": 2.0, "beta": 1.0},
            },
        ),
    ],
)
def test_ordinary_package_likelihoods_lower_explicitly(
    kind: str,
    options: dict[str, object],
    builder: object,
    expected: dict[str, object],
) -> None:
    config = _ordinary()
    config["likelihood"] = {"kind": kind, **options}

    if kind == "scalar_sigma":
        expected = {**expected, "eigenbasis_path": Path("sigma.nc").resolve()}
    assert Co2ModelConfig.from_params(config).model_arguments() == {
        "likelihood_builder": builder,
        "likelihood_kwargs": expected,
    }


def test_resolved_nested_options_are_json_serializable_for_provenance() -> None:
    config = _ordinary()
    config["likelihood"] = {
        "kind": "site_sigma",
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0},
    }
    model = Co2ModelConfig.from_params(config).model_arguments()
    trace = make_trace()

    annotate_likelihood_trace(
        trace,
        builder_identity={"module": "test", "qualname": "builder"},
        likelihood_kwargs=cast(Mapping[str, Any], model["likelihood_kwargs"]),
    )

    assert json.loads(trace.attrs["rhime_likelihood_kwargs"]) == {
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0}
    }


def test_boundary_offset_and_inferred_ou_choices_lower_to_component_arguments() -> None:
    config = _ordinary()
    config["model"] = {
        "boundary": {
            "enabled": True,
            "prior": {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.1, "lower": 0.0},
        },
        "offset": {
            "prior": {"pdf": "normal", "mu": 0.0, "sigma": 0.1},
            "frequency": "monthly",
            "per_site": True,
            "drop_first": True,
        },
    }
    config["likelihood"] = {
        "kind": "fixed_ou",
        "tau_hours": {"MHD": 24.0},
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.5},
        "initial_site_amplitudes": 0.5,
    }
    config["sampling"] = {"chains": 2, "sigma_target_accept": 0.82, "state_target_accept": 0.91}

    model = Co2ModelConfig.from_params(config).model_arguments()
    sampling = _resolve_co2_sampling(config, model=model)
    assert sampling.sigma_target_accept == 0.82
    assert sampling.state_target_accept == 0.91
    assert model.pop("likelihood_builder") is add_fixed_ou_gaussian_likelihood
    likelihood = model.pop("likelihood_kwargs")
    assert {**model, **likelihood} == {
        "use_bc": True,
        "bc_prior": {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.1, "lower": 0.0},
        "offset_prior": {"pdf": "normal", "mu": 0.0, "sigma": 0.1},
        "offset_args": {"offset_freq": "monthly", "per_site": True, "drop_first": True},
        "tau_hours": {"MHD": 24.0},
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.5},
        "initial_site_amplitudes": 0.5,
    }










@pytest.mark.parametrize(
    ("likelihood", "message"),
    [
        ({"kind": "module:callable"}, "likelihood.kind"),
        (
            {
                "kind": "additive_sigma",
                "no_model_error": True,
                "sigma_prior": {"pdf": "halfnormal", "sigma": 1.0},
            },
            "cannot be combined",
        ),
        (
            {"kind": "site_sigma", "fixed_site_amplitudes": {"MHD": -1.0}},
            "non-negative",
        ),
        (
            {"kind": "site_sigma", "fixed_site_amplitudes": 1.0},
            "site mapping",
        ),
        ({"kind": "fixed_ou", "tau_hours": 0.0}, "positive"),
        (
            {"kind": "scalar_sigma", "eigenbasis_path": "cache.nc"},
            "sigma_prior",
        ),
    ],
)
def test_invalid_likelihoods_fail_during_resolution(
    likelihood: dict[str, object], message: str
) -> None:
    config = _ordinary()
    config["likelihood"] = likelihood

    with pytest.raises((TypeError, ValueError), match=message):
        Co2ModelConfig.from_params(config).model_arguments()


@pytest.mark.parametrize(
    ("prior", "message"),
    [
        ({"pdf": "halfnormal", "sigam": 1.0}, "sigma"),
        ({"pdf": "halfnormal", "sigma": float("inf")}, "finite and positive"),
        ({"pdf": "uniform", "lower": 2.0, "upper": 1.0}, "lower must be less"),
    ],
)
def test_invalid_prior_parameters_fail_during_resolution(
    prior: dict[str, object], message: str
) -> None:
    config = _ordinary()
    config["likelihood"] = {"kind": "additive_sigma", "sigma_prior": prior}

    with pytest.raises((TypeError, ValueError), match=message):
        Co2ModelConfig.from_params(config).model_arguments()


def test_lognormal_prior_accepts_supported_moment_parameterization() -> None:
    config = _ordinary()
    config["likelihood"] = {
        "kind": "additive_sigma",
        "sigma_prior": {
            "pdf": "lognormal",
            "mean": 1.0,
            "stdev": 0.25,
            "reparameterise": True,
        },
    }

    model = Co2ModelConfig.from_params(config).model_arguments()

    assert model["sigma_prior"] == {
        "pdf": "lognormal",
        "mean": 1.0,
        "stdev": 0.25,
        "reparameterise": True,
    }


def test_cached_variant_rejects_generic_sampler_controls() -> None:
    config = _ordinary()
    config["likelihood"] = {
        "kind": "fixed_ou",
        "tau_hours": 24.0,
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0},
    }
    config["sampling"] = {"target_accept": 0.9}

    with pytest.raises(ValueError, match="sigma_target_accept"):
        _resolve_co2_sampling(config, model=Co2ModelConfig.from_params(config).model_arguments())


@pytest.mark.parametrize(
    "sampling",
    [
        {"draws": 1, "burn": 1},
        {"burn": 1000},
    ],
)
def test_sampling_rejects_burn_that_removes_every_draw(
    sampling: dict[str, object],
) -> None:
    config = _ordinary()
    config["sampling"] = sampling

    with pytest.raises(ValueError, match="burn must be less than"):
        _resolve_co2_sampling(config, model=Co2ModelConfig.from_params(_ordinary()).model_arguments()).sampler


def test_cached_variant_rejects_unsupported_posterior_predictive_names() -> None:
    config = _ordinary()
    config["likelihood"] = {
        "kind": "fixed_ou",
        "tau_hours": 24.0,
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0},
    }
    config["sampling"] = {
        "sample_posterior_predictive": ["y", "flux_scaling"],
    }

    with pytest.raises(ValueError, match="supports only"):
        _resolve_co2_sampling(config, model=Co2ModelConfig.from_params(config).model_arguments())


@pytest.mark.parametrize(
    ("offset_options", "message"),
    [
        ({"per_site": False, "frequency": "monthly"}, "does not accept frequency"),
        ({"per_site": False, "drop_first": True}, "does not support drop_first"),
    ],
)
def test_global_offset_rejects_site_specific_options_during_resolution(
    offset_options: dict[str, object], message: str
) -> None:
    config = _ordinary()
    config["model"] = {
        "offset": {
            "prior": {"pdf": "normal", "mu": 0.0, "sigma": 0.1},
            **offset_options,
        }
    }

    with pytest.raises(ValueError, match=message):
        Co2ModelConfig.from_params(config).model_arguments()












def test_imported_preparation_facts_are_explicit_and_optional() -> None:
    config = _ordinary()
    config["prepared_inputs"] = {
        "path": "prepared.zarr", "start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE",
        "reconstruction_path": "affine.nc",
    }
    assert Co2PreparationConfig.from_params(config).preparation_arguments() == {
        "prepared_path": Path("prepared.zarr").resolve(), "affine_path": Path("affine.nc").resolve(),
        "start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE",
    }
    assert Co2PreparationConfig.from_params({}).preparation_arguments() == {
        "prepared_path": None, "affine_path": None, "start_date": None, "end_date": None, "domain": None,
    }
    with pytest.raises(ValueError, match="prepared_inputs.path"):
        Co2PreparationConfig.from_params({}, require_path=True).preparation_arguments()


def test_model_resolution_consumes_only_owned_choices_without_an_envelope() -> None:
    config = _ordinary()
    for name in ("format_version", "recipe", "prepared_inputs"):
        del config[name]
    config["outputs"] = object()
    config["sampling"] = object()
    model = Co2ModelConfig.from_params(config).model_arguments()
    assert model == {"no_model_error": False, "sigma_prior": {"pdf": "halfnormal", "sigma": 0.75}}


def test_sampler_resolution_consumes_only_owned_choices_without_an_envelope() -> None:
    config = {"sampling": cast(dict, _ordinary()["sampling"]), "model": object(), "outputs": object()}
    assert _resolve_co2_sampling(config, model=Co2ModelConfig.from_params(_ordinary()).model_arguments()).sampler.draws == 20
    assert _resolve_co2_sampling({}, model=Co2ModelConfig.from_params(_ordinary()).model_arguments()).sampler is not _resolve_co2_sampling({}, model=Co2ModelConfig.from_params(_ordinary()).model_arguments()).sampler
    config["sampling"] = object()
    with pytest.raises(TypeError, match="sampling must be a table"):
        _resolve_co2_sampling(config, model=Co2ModelConfig.from_params(_ordinary()).model_arguments()).sampler


def test_model_resolution_owns_eigenbasis_path_from_invocation_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = {
        "likelihood": {
            "kind": "scalar_sigma", "eigenbasis_path": "cache/eigenbasis.nc",
            "sigma_prior": {"pdf": "halfnormal", "sigma": 0.5},
        },
    }
    before = deepcopy(config)
    model = Co2ModelConfig.from_params(config).model_arguments()
    assert model["likelihood_builder"] is add_scalar_sigma_eigen_likelihood
    assert model["likelihood_kwargs"]["eigenbasis_path"] == tmp_path / "cache" / "eigenbasis.nc"
    assert config == before
    assert not (tmp_path / "cache").exists()


def test_component_resolution_preserves_selected_caller_values() -> None:
    config = _ordinary()
    config["model"] = {
        "boundary": {"enabled": True, "prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.1}},
        "offset": {"prior": {"pdf": "normal", "mu": 0.0, "sigma": 0.2}, "per_site": True},
    }
    before = deepcopy(config)
    model = Co2ModelConfig.from_params(config).model_arguments()
    sampler = _resolve_co2_sampling(config, model=Co2ModelConfig.from_params(_ordinary()).model_arguments()).sampler
    assert model["use_bc"] is True
    assert model["bc_prior"] == {"pdf": "normal", "mu": 1.0, "sigma": 0.1}
    assert model["offset_prior"] == {"pdf": "normal", "mu": 0.0, "sigma": 0.2}
    assert sampler.draws == 20
    assert config == before


def test_prior_seed_consumes_only_its_owned_sampling_option() -> None:
    config = {"sampling": {"random_seed": 42, "draws": -1, "nuts_sampler": object()}, "model": object()}
    assert resolve_co2_prior_seed(config) == 42
    assert resolve_co2_prior_seed({}) is None
    assert resolve_co2_prior_seed({"sampling": {"draws": -1}}) is None
    with pytest.raises(ValueError, match="sampling.nuts_sampler"):
        _resolve_co2_sampling(config, model=Co2ModelConfig.from_params(_ordinary()).model_arguments()).sampler


@pytest.mark.parametrize("seed", [True, -1, 0.5, "42", None])
def test_prior_and_posterior_share_random_seed_validation(seed: object) -> None:
    for resolve in (resolve_co2_prior_seed, lambda config: _resolve_co2_sampling(config, model=Co2ModelConfig.from_params(_ordinary()).model_arguments())):
        with pytest.raises(ValueError, match="random_seed must be a non-negative integer"):
            resolve({"sampling": {"random_seed": seed}})


def test_envelope_validation_does_not_resolve_unowned_sections() -> None:
    config = {"model": object(), "sampling": object(), "outputs": object()}
    assert validate_co2_config(config) == config
    assert validate_co2_config({}) == {}
    with pytest.raises(ValueError, match="config.recipe"):
        validate_co2_config({"recipe": "co2_o2"})


@pytest.mark.parametrize("prepared", [{"path": False}, {"path": ""}, {"reconstruction_path": []}, {"surprise": True}])
def test_preparation_owner_rejects_invalid_or_unknown_options(prepared: object) -> None:
    with pytest.raises((TypeError, ValueError), match="prepared_inputs"):
        Co2PreparationConfig.from_params({"prepared_inputs": prepared}).preparation_arguments()


@pytest.mark.parametrize("prior", [
    {"pdf": "halfnormal", "sigma": 0.8},
    {"pdf": "halfstudentt", "nu": 4.0, "sigma": 0.8},
    {"pdf": "exponential", "lam": 1.5},
    {"pdf": "gamma", "alpha": 2.0, "beta": 3.0},
    {"pdf": "lognormal", "mean": 0.7, "stdev": 0.3},
    {"pdf": "uniform", "lower": 0.1, "upper": 1.5},
])
def test_cached_configuration_accepts_positive_prior_families(prior):
    config = _ordinary()
    config["likelihood"] = {"kind": "fixed_ou", "tau_hours": 24.0, "site_amplitude_prior": prior}
    model = Co2ModelConfig.from_params(config).model_arguments()
    assert model["likelihood_kwargs"]["site_amplitude_prior"] == prior
