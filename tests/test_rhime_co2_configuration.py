"""Configuration contracts for the concrete CO2-family recipes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.scalar_sigma import add_scalar_sigma_eigen_likelihood
from openghg_inversions.models.site_sigma import add_site_sigma_gaussian_likelihood
from openghg_inversions.rhime.co2.configuration import (
    Co2O2RunSetup,
    Co2RunSetup,
    co2_config_templates,
    load_co2_family_config,
    resolve_co2_family_config,
    resolve_co2_model,
    resolve_co2_sampler,
    resolve_co2_prior_seed,
    resolve_co2_preparation,
    validate_co2_config,
)
from openghg_inversions.rhime.co2.co2_cached_sigma_runner import (
    run_rhime_co2_cached_sigma,
)
from openghg_inversions.rhime.co2.co2_o2_preparation import Co2O2PreparedInputs
from openghg_inversions.rhime.co2.co2_o2_runner import (
    run_rhime_co2_o2_from_prepared_inputs,
)
from openghg_inversions.rhime.outputs import annotate_likelihood_trace
from openghg_inversions.rhime.sampling import RhimeSampler
from tests.helpers import make_trace


def _ordinary() -> dict[str, object]:
    return {
        "format_version": 1,
        "recipe": "co2",
        "variant": "ordinary",
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
        'format_version = 1\nrecipe = "co2"\nvariant = "ordinary"\n',
        encoding="utf-8",
    )

    assert load_co2_family_config(path) == {
        "format_version": 1,
        "recipe": "co2",
        "variant": "ordinary",
    }


def test_installed_templates_pass_the_semantic_resolver() -> None:
    templates = co2_config_templates()

    assert tuple(templates) == (
        "co2.toml",
        "co2_cached_sigma.toml",
        "co2_o2.toml",
    )
    for resource in templates.values():
        config = load_co2_family_config(resource)
        if config.get("variant") == "ordinary":
            validate_co2_config(config)
            resolve_co2_preparation(config, require_path=True)
            resolve_co2_model(config)
            resolve_co2_sampler(config)
        else:
            assert resolve_co2_family_config(config)


def test_ordinary_configuration_matches_explicit_scientific_arguments() -> None:
    config = _ordinary()
    assert resolve_co2_model(config) == {
        "no_model_error": False,
        "sigma_prior": {"pdf": "halfnormal", "sigma": 0.75},
    }
    assert resolve_co2_preparation(config, require_path=True)["prepared_path"] == Path("prepared.zarr").resolve()
    assert resolve_co2_sampler(config) == RhimeSampler(
        draws=20, tune=10, chains=2, nuts_sampler="pymc",
        sample_kwargs={"target_accept": 0.95, "random_seed": 42},
        posterior_predictive_kwargs={"random_seed": 42},
    )


def test_ordinary_legacy_family_facade_is_retired() -> None:
    with pytest.raises(ValueError, match="Ordinary CO2 no longer uses Co2RunSetup"):
        resolve_co2_family_config(_ordinary())


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
    assert resolve_co2_model(config) == {
        "likelihood_builder": builder,
        "likelihood_kwargs": expected,
    }


def test_resolved_nested_options_are_json_serializable_for_provenance() -> None:
    config = _ordinary()
    config["likelihood"] = {
        "kind": "site_sigma",
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0},
    }
    model = resolve_co2_model(config)
    trace = make_trace()

    annotate_likelihood_trace(
        trace,
        builder_identity={"module": "test", "qualname": "builder"},
        likelihood_kwargs=cast(Mapping[str, Any], model["likelihood_kwargs"]),
    )

    assert json.loads(trace.attrs["rhime_likelihood_kwargs"]) == {
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0}
    }


def test_boundary_offset_and_cached_variant_lower_to_runner_arguments() -> None:
    config = _ordinary()
    config["variant"] = "cached_fixed_ou"
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
        "sigma_target_accept": 0.82,
        "state_target_accept": 0.91,
    }
    config["sampling"] = {"chains": 2, "nuts_sampler": "pymc"}

    setup = cast(Co2RunSetup, resolve_co2_family_config(config))

    assert setup.runner is run_rhime_co2_cached_sigma
    assert setup.runner_kwargs == {
        "use_bc": True,
        "bc_prior": {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.1, "lower": 0.0},
        "offset_prior": {"pdf": "normal", "mu": 0.0, "sigma": 0.1},
        "offset_args": {"offset_freq": "monthly", "per_site": True, "drop_first": True},
        "tau_hours": {"MHD": 24.0},
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.5},
        "initial_site_amplitudes": 0.5,
        "sigma_target_accept": 0.82,
        "state_target_accept": 0.91,
    }


def test_linked_channels_remain_distinct_and_bind_to_labelled_errors() -> None:
    setup = resolve_co2_family_config(_linked())
    assert isinstance(setup, Co2O2RunSetup)
    assert setup.runner is run_rhime_co2_o2_from_prepared_inputs
    assert setup.preparation_kwargs == {
        "co2_units": "ppm",
        "o2_units": "ppm",
    }
    observations = xr.DataArray(
        [400.0, 401.0, -120.0],
        dims="observation",
        coords={
            "observation": ["co2:a", "co2:b", "o2:a"],
            "species": ("observation", ["co2", "co2", "o2"]),
            "observation_units": ("observation", ["ppm", "ppm", "ppm"]),
        },
    )
    prepared = cast(
        Co2O2PreparedInputs,
        type("Prepared", (), {"observations": observations})(),
    )

    arguments = setup.runner_arguments(prepared)

    assert arguments["prepared_inputs"] is prepared
    assert arguments["sampler"] is setup.sampler
    error = cast(xr.DataArray, arguments["independent_error_sd"])
    np.testing.assert_allclose(error, [1.0, 1.0, 2.0])
    assert error.coords.to_index().equals(observations.coords.to_index())


def test_linked_omitted_channel_errors_bind_saved_vector_without_computing() -> None:
    """Omitting both channel errors requires and preserves the lazy saved vector."""
    config = _linked()
    channels = cast(dict[str, dict[str, object]], config["channels"])
    for channel in channels.values():
        channel.pop("independent_error_sd")
    setup = cast(Co2O2RunSetup, resolve_co2_family_config(config))
    observations = xr.DataArray(
        [400.0, 401.0, -120.0],
        dims="observation",
        coords={
            "observation": ["co2:a", "co2:b", "o2:a"],
            "species": ("observation", ["co2", "co2", "o2"]),
            "observation_units": ("observation", ["ppm", "ppm", "ppm"]),
        },
    )
    error = observations.copy(data=da.from_array([0.5, 0.75, 1.25], chunks=2))
    prepared = cast(
        Co2O2PreparedInputs,
        type("Prepared", (), {"observations": observations, "independent_error_sd": error})(),
    )

    assert setup.runner_arguments(prepared)["independent_error_sd"] is error
    assert isinstance(error.data, da.Array)

    prepared.independent_error_sd = None
    with pytest.raises(ValueError, match="requires errors in the prepared artifact"):
        setup.runner_arguments(prepared)


@pytest.mark.parametrize("channel", ["co2", "o2"])
def test_linked_partial_channel_error_omission_is_rejected(channel: str) -> None:
    """Configuration rejects mixing one channel error with implicit saved errors."""
    config = _linked()
    channels = cast(dict[str, dict[str, object]], config["channels"])
    channels[channel].pop("independent_error_sd")

    with pytest.raises(ValueError, match="both linked channels or omit it for both"):
        resolve_co2_family_config(config)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"format_version": 2}, "format_version"),
        ({"recipe": "co2_o2", "variant": "ordinary"}, "variant='linked'"),
        ({"surprise": True}, "config.surprise"),
    ],
)
def test_top_level_invalid_options_fail_during_resolution(
    change: dict[str, object], message: str
) -> None:
    config = _ordinary()
    config.update(change)

    with pytest.raises(ValueError, match=message):
        if config["recipe"] == "co2_o2":
            resolve_co2_family_config(config)
        else:
            validate_co2_config(config)


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
        resolve_co2_model(config)


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
        resolve_co2_model(config)


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

    model = resolve_co2_model(config)

    assert model["sigma_prior"] == {
        "pdf": "lognormal",
        "mean": 1.0,
        "stdev": 0.25,
        "reparameterise": True,
    }


def test_cached_variant_rejects_generic_sampler_controls() -> None:
    config = _ordinary()
    config["variant"] = "cached_fixed_ou"
    config["likelihood"] = {
        "kind": "fixed_ou",
        "tau_hours": 24.0,
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0},
    }
    config["sampling"] = {"nuts_sampler": "pymc", "target_accept": 0.9}

    with pytest.raises(ValueError, match="sigma_target_accept"):
        resolve_co2_family_config(config)


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
        resolve_co2_sampler(config)


def test_cached_variant_rejects_unsupported_posterior_predictive_names() -> None:
    config = _ordinary()
    config["variant"] = "cached_fixed_ou"
    config["likelihood"] = {
        "kind": "fixed_ou",
        "tau_hours": 24.0,
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0},
    }
    config["sampling"] = {
        "nuts_sampler": "pymc",
        "sample_posterior_predictive": ["y", "flux_scaling"],
    }

    with pytest.raises(ValueError, match="supports only"):
        resolve_co2_family_config(config)


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
        resolve_co2_model(config)


@pytest.mark.parametrize(
    ("sampling", "expected"),
    [
        (None, RhimeSampler(nuts_sampler="numpyro", sample_kwargs={"target_accept": 0.95})),
        (
            {"draws": 20},
            RhimeSampler(
                draws=20,
                nuts_sampler="numpyro",
                sample_kwargs={"target_accept": 0.95},
            ),
        ),
    ],
)
def test_linked_sampling_preserves_recipe_defaults(
    sampling: dict[str, object] | None, expected: RhimeSampler
) -> None:
    config = _linked()
    if sampling is None:
        config.pop("sampling")
    else:
        config["sampling"] = sampling

    setup = cast(Co2O2RunSetup, resolve_co2_family_config(config))

    assert setup.sampler == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("model", {"boundary": {"enabled": False, "prior": {"pdf": "normal"}}}), "config.model"),
        (("model", {"offset": {"frequency": "monthly"}}), "config.model"),
        (("likelihood", {"kind": "additive_sigma"}), "config.likelihood"),
    ],
)
def test_linked_recipe_rejects_inapplicable_component_tables(
    mutation: tuple[str, object], message: str
) -> None:
    config = deepcopy(_linked())
    config[mutation[0]] = mutation[1]

    with pytest.raises(ValueError, match=message):
        resolve_co2_family_config(config)


def test_linked_recipe_rejects_heterogeneous_units_before_binding() -> None:
    config = _linked()
    channels = cast(dict[str, dict[str, object]], config["channels"])
    channels["o2"]["units"] = "per meg"

    with pytest.raises(ValueError, match="identical"):
        resolve_co2_family_config(config)


def test_linked_recipe_rejects_non_mole_fraction_units() -> None:
    config = _linked()
    channels = cast(dict[str, dict[str, object]], config["channels"])
    channels["co2"]["units"] = "bananas"
    channels["o2"]["units"] = "bananas"

    with pytest.raises(ValueError, match="mol/mol"):
        resolve_co2_family_config(config)


def test_linked_binding_rechecks_units_against_prepared_inputs() -> None:
    setup = cast(Co2O2RunSetup, resolve_co2_family_config(_linked()))
    observations = xr.DataArray(
        [1.0, 2.0],
        dims="observation",
        coords={
            "species": ("observation", ["co2", "o2"]),
            "observation_units": ("observation", ["ppm", "per meg"]),
        },
    )
    prepared = cast(
        Co2O2PreparedInputs,
        type("Prepared", (), {"observations": observations})(),
    )

    with pytest.raises(ValueError, match="channels.o2.units"):
        setup.runner_arguments(prepared)


def test_imported_preparation_facts_are_explicit_and_optional() -> None:
    config = _ordinary()
    config["prepared_inputs"] = {
        "path": "prepared.zarr", "start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE",
        "reconstruction_path": "affine.nc",
    }
    assert resolve_co2_preparation(config) == {
        "prepared_path": Path("prepared.zarr").resolve(), "affine_path": Path("affine.nc").resolve(),
        "start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE",
    }
    assert resolve_co2_preparation({}) == {
        "prepared_path": None, "affine_path": None, "start_date": None, "end_date": None, "domain": None,
    }
    with pytest.raises(ValueError, match="prepared_inputs.path"):
        resolve_co2_preparation({}, require_path=True)


def test_model_resolution_consumes_only_owned_choices_without_an_envelope() -> None:
    config = _ordinary()
    for name in ("format_version", "recipe", "variant", "prepared_inputs"):
        del config[name]
    config["outputs"] = object()
    config["sampling"] = object()
    model = resolve_co2_model(config)
    assert model == {"no_model_error": False, "sigma_prior": {"pdf": "halfnormal", "sigma": 0.75}}


def test_sampler_resolution_consumes_only_owned_choices_without_an_envelope() -> None:
    config = {"sampling": cast(dict, _ordinary()["sampling"]), "model": object(), "outputs": object()}
    assert resolve_co2_sampler(config).draws == 20
    assert resolve_co2_sampler({}) is not resolve_co2_sampler({})
    config["sampling"] = object()
    with pytest.raises(TypeError, match="sampling must be a table"):
        resolve_co2_sampler(config)


def test_model_resolution_owns_eigenbasis_path_from_invocation_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = {
        "likelihood": {
            "kind": "scalar_sigma", "eigenbasis_path": "cache/eigenbasis.nc",
            "sigma_prior": {"pdf": "halfnormal", "sigma": 0.5},
        },
    }
    before = deepcopy(config)
    model = resolve_co2_model(config)
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
    model = resolve_co2_model(config)
    sampler = resolve_co2_sampler(config)
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
        resolve_co2_sampler(config)


@pytest.mark.parametrize("seed", [True, -1, 0.5, "42", None])
def test_prior_and_posterior_share_random_seed_validation(seed: object) -> None:
    for resolve in (resolve_co2_prior_seed, resolve_co2_sampler):
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
        resolve_co2_preparation({"prepared_inputs": prepared})


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
    config["variant"] = "cached_fixed_ou"
    config["sampling"] = {"nuts_sampler": "pymc"}
    config["likelihood"] = {"kind": "fixed_ou", "tau_hours": 24.0, "site_amplitude_prior": prior}
    setup = resolve_co2_family_config(config)
    assert setup.runner_kwargs["site_amplitude_prior"] == prior
