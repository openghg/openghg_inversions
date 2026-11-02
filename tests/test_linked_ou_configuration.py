"""Concrete linked configuration preserves channel-specific scientific choices."""

import numpy as np
import pytest

from openghg_inversions.rhime.co2 import Co2O2ModelConfig
from openghg_inversions.rhime.co2.co2_o2_configuration import resolve_co2_o2_sampler
from test_co2_o2_fixed_ou import _prepared


def _config():
    return {
        "channels": {
            "co2": {"units": "ppm", "independent_error_sd": 0.2, "boundary": {"enabled": True}},
            "o2": {"units": "ppm", "independent_error_sd": 0.3,
                   "offset": {"prior": {"pdf": "normal", "mu": 0, "sigma": 0.2}, "per_site": False}},
        },
        "likelihood": {"kind": "fixed_ou", "tau_hours": {"co2:A": 24, "o2:A": 12, "o2:B": 6},
                       "fixed_site_amplitudes": {"co2:A": 0.4, "o2:A": 0.6, "o2:B": 0.2}},
    }


def test_linked_ou_owner_preserves_groups_and_channel_baselines():
    config = _config()
    model = Co2O2ModelConfig.from_params(config)
    arguments = model.bind(_prepared())
    np.testing.assert_allclose(arguments["independent_error_sd"], [0.2, 0.2, 0.3, 0.3, 0.3])
    assert arguments["use_bc"] == {"co2": True}
    assert arguments["offset_args"] == {"o2": {"per_site": False}}
    for name in ("tau_hours", "fixed_site_amplitudes"):
        assert arguments[name] == config["likelihood"][name]
    assert resolve_co2_o2_sampler(config, model=model).nuts_sampler == "pymc"
    config["channels"]["o2"]["offset"]["prior"]["sigma"] = 99
    assert arguments["offset_prior"]["o2"]["sigma"] == .2


@pytest.mark.parametrize("likelihood", [
    {"kind": "site_sigma", "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1}},
    {"kind": "fixed_ou", "tau_hours": 24},
    {"kind": "fixed_ou", "tau_hours": 0, "fixed_site_amplitudes": 1},
    {"kind": "fixed_ou", "tau_hours": 24, "fixed_site_amplitudes": 1,
     "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1}},
])
def test_linked_owner_rejects_unsupported_science(likelihood):
    with pytest.raises(ValueError):
        Co2O2ModelConfig.from_params({**_config(), "likelihood": likelihood})
