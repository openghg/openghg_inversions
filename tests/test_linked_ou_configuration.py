"""Linked OU configuration selects the same labelled arguments as Python."""

from typing import cast

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.rhime.co2 import (
    Co2O2PreparedInputs,
    Co2O2RunSetup,
    resolve_co2_family_config,
    run_rhime_co2_o2_cached_sigma_from_prepared_inputs,
    run_rhime_co2_o2_from_prepared_inputs,
)


def _config(*, cached: bool = False) -> dict:
    return {
        "format_version": 1,
        "recipe": "co2_o2",
        "variant": "cached_fixed_ou" if cached else "linked",
        "channels": {
            "co2": {"units": "ppm", "independent_error_sd": 0.2},
            "o2": {"units": "ppm", "independent_error_sd": 0.3},
        },
        "likelihood": {
            "kind": "fixed_ou",
            "tau_hours": {"co2:MHD": 24.0, "o2:MHD": 12.0},
            **(
                {"site_amplitude_prior_scale": 0.5, "initial_site_amplitudes": 0.2}
                if cached
                else {"fixed_site_amplitudes": {"co2:MHD": 0.4, "o2:MHD": 0.6}}
            ),
        },
    }


@pytest.mark.parametrize("cached", [False, True])
def test_linked_ou_config_preserves_groups_and_binds_reported_error(cached: bool) -> None:
    config = _config(cached=cached)
    config["channels"]["co2"]["boundary"] = {"enabled": True}
    config["channels"]["o2"]["offset"] = {
        "prior": {"pdf": "normal", "mu": 0.0, "sigma": 0.2}, "per_site": False,
    }
    setup = cast(Co2O2RunSetup, resolve_co2_family_config(config))
    observations = xr.DataArray(
        [400.0, -120.0, 401.0],
        dims="observation",
        coords={
            "observation": ["a", "b", "c"],
            "species": ("observation", ["co2", "o2", "co2"]),
            "observation_units": ("observation", ["ppm"] * 3),
        },
    )
    prepared = cast(
        Co2O2PreparedInputs,
        type("Prepared", (), {"observations": observations})(),
    )
    arguments = setup.runner_arguments(prepared)
    assert setup.runner is (
        run_rhime_co2_o2_cached_sigma_from_prepared_inputs
        if cached
        else run_rhime_co2_o2_from_prepared_inputs
    )
    assert setup.sampler.nuts_sampler == "pymc"
    assert arguments["tau_hours"] == {"co2:MHD": 24.0, "o2:MHD": 12.0}
    np.testing.assert_allclose(arguments["independent_error_sd"], [0.2, 0.3, 0.2])
    for name, value in config["likelihood"].items():
        if name != "kind":
            assert arguments[name] == value
    assert arguments["prepared_inputs"] is prepared
    assert arguments["use_bc"] == {"co2": True}
    assert arguments["offset_prior"] == {"o2": {"pdf": "normal", "mu": 0.0, "sigma": 0.2}}
    assert arguments["offset_args"] == {"o2": {"per_site": False}}


def test_linked_ou_config_resolves_positive_amplitude_prior() -> None:
    config = _config()
    del config["likelihood"]["fixed_site_amplitudes"]
    config["likelihood"]["site_amplitude_prior"] = {"pdf": "Half-Normal", "sigma": 0.5}
    setup = cast(Co2O2RunSetup, resolve_co2_family_config(config))
    assert setup.runner_kwargs["site_amplitude_prior"] == {"pdf": "halfnormal", "sigma": 0.5}


@pytest.mark.parametrize("cached", [False, True])
def test_linked_ou_config_rejects_unsupported_backend(cached: bool) -> None:
    config = _config(cached=cached)
    config["sampling"] = {"nuts_sampler": "numpyro"}
    with pytest.raises(ValueError, match="pymc"):
        resolve_co2_family_config(config)


@pytest.mark.parametrize(
    "likelihood",
    [
        {"kind": "site_sigma", "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1}},
        {"kind": "fixed_ou", "tau_hours": 24.0},
        {"kind": "fixed_ou", "tau_hours": 0.0, "fixed_site_amplitudes": 1.0},
        {
            "kind": "fixed_ou",
            "tau_hours": 24.0,
            "fixed_site_amplitudes": 1.0,
            "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0},
        },
    ],
)
def test_linked_ou_config_rejects_ambiguous_or_unsupported_target(likelihood: dict) -> None:
    config = _config()
    config["likelihood"] = likelihood
    with pytest.raises(ValueError):
        resolve_co2_family_config(config)
