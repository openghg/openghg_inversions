"""Configuration contracts for the concrete CO2-family recipes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, cast

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
)
from openghg_inversions.rhime.co2.co2_cached_sigma_runner import (
    run_rhime_co2_cached_sigma,
)
from openghg_inversions.rhime.co2.co2_o2_preparation import Co2O2PreparedInputs
from openghg_inversions.rhime.co2.co2_o2_runner import (
    run_rhime_co2_o2_from_prepared_inputs,
)
from openghg_inversions.rhime.co2.co2_runner import run_rhime_co2
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
            "nuts_sampler": "numpyro",
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
        assert resolve_co2_family_config(config)


def test_ordinary_configuration_matches_direct_python_arguments() -> None:
    setup = resolve_co2_family_config(_ordinary())

    assert isinstance(setup, Co2RunSetup)
    assert setup.runner is run_rhime_co2
    assert setup.preparation_kwargs == {"path": Path("prepared.zarr")}
    assert setup.runner_kwargs == {
        "no_model_error": False,
        "sigma_prior": {"pdf": "halfnormal", "sigma": 0.75},
    }
    assert setup.sampler == RhimeSampler(
        draws=20,
        tune=10,
        chains=2,
        nuts_sampler="numpyro",
        sample_kwargs={"target_accept": 0.95, "random_seed": 42},
        posterior_predictive_kwargs={"random_seed": 42},
    )

    prepared = cast(Any, object())
    assert setup.runner_arguments(prepared) == {
        "prepared_inputs": prepared,
        **setup.runner_kwargs,
        "sampler": setup.sampler,
    }


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

    setup = cast(Co2RunSetup, resolve_co2_family_config(config))

    assert setup.runner_kwargs == {
        "likelihood_builder": builder,
        "likelihood_kwargs": expected,
    }


def test_resolved_nested_options_are_json_serializable_for_provenance() -> None:
    config = _ordinary()
    config["likelihood"] = {
        "kind": "site_sigma",
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0},
    }
    setup = cast(Co2RunSetup, resolve_co2_family_config(config))
    trace = make_trace()

    annotate_likelihood_trace(
        trace,
        builder_identity={"module": "test", "qualname": "builder"},
        likelihood_kwargs=cast(Mapping[str, Any], setup.runner_kwargs["likelihood_kwargs"]),
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
        "site_amplitude_prior_scale": 1.5,
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
        "site_amplitude_prior_scale": 1.5,
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
        resolve_co2_family_config(config)


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
        resolve_co2_family_config(config)


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
        resolve_co2_family_config(config)


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

    setup = cast(Co2RunSetup, resolve_co2_family_config(config))

    assert setup.runner_kwargs["sigma_prior"] == {
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
        "site_amplitude_prior_scale": 1.0,
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
        resolve_co2_family_config(config)


def test_cached_variant_rejects_unsupported_posterior_predictive_names() -> None:
    config = _ordinary()
    config["variant"] = "cached_fixed_ou"
    config["likelihood"] = {
        "kind": "fixed_ou",
        "tau_hours": 24.0,
        "site_amplitude_prior_scale": 1.0,
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
        resolve_co2_family_config(config)


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
