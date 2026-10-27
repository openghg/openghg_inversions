"""Observation roles use fixed aliases without changing extension variables."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from openghg_inversions.inversion_data import prepare_surface_site, surface


@pytest.fixture
def role_inputs():
    time = pd.date_range("2020-01-01", periods=3, freq="h")
    obs = xr.Dataset(
        {"co2": ("time", [400., 401., 402.], {"units": "ppm"}),
         "co2_error": ("time", [.1, .2, .3]),
         "co2_repeatability": ("time", [.01, .02, .03]),
         "co2_variability": ("time", [.3, .4, .5]),
         "co2_number_of_observations": ("time", [2, 3, 4]),
         "co2_quality_flag": ("time", [0, 1, 0])},
        coords={"time": time}, attrs={"sampling_period": 3600},
    )
    fp = xr.Dataset({"fp": (("lat", "lon", "time"), np.ones((1, 1, 3)), {"units": "s m2/mol"})},
                    coords={"lat": [0.], "lon": [0.], "time": time})
    flux = (fp.rename(fp="flux") * 1e-6)
    flux.flux.attrs["units"] = "mol/m2/s"
    return obs, fp, flux


def prepare(obs, fp, flux, **kwargs):
    return prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "integrated"}, **kwargs)


@pytest.mark.parametrize("convention", ["species", "canonical", "mixed", "custom"])
@pytest.mark.parametrize("lazy", [False, True])
def test_supported_observation_roles_preserve_values_extensions_and_borrowing(role_inputs, convention, lazy):
    obs, fp, flux = role_inputs
    aliases = {name: "mf" + name[3:] for name in obs if name != "co2_quality_flag"}
    kwargs = {}
    if convention == "canonical":
        obs = obs.rename(aliases)
    elif convention == "mixed":
        obs = obs.rename({"co2_error": "mf_error", "co2_variability": "mf_variability"})
    elif convention == "custom":
        obs = obs.rename(co2="measured_concentration")
        kwargs["observation_variable"] = "measured_concentration"
    if lazy:
        obs, fp, flux = (value.chunk({"time": 2}) for value in (obs, fp, flux))
    before = obs.copy(deep=True)
    with Callback(pretask=lambda *args: pytest.fail("Preparation computed a payload")):
        result = prepare(obs, fp, flux, **kwargs)
    if lazy:
        assert isinstance(result.mf.data, da.Array)
    np.testing.assert_allclose(result.mf, [400., 401., 402.])
    np.testing.assert_allclose(result.mf_error, [.1, .2, .3])
    np.testing.assert_allclose(result.mf_repeatability, [.01, .02, .03])
    np.testing.assert_allclose(result.mf_variability, [.3, .4, .5])
    np.testing.assert_array_equal(result.mf_number_of_observations.values, [2, 3, 4])
    xr.testing.assert_identical(result.co2_quality_flag, obs.co2_quality_flag)
    assert "mf_quality_flag" not in result
    xr.testing.assert_identical(obs, before)


@pytest.mark.parametrize("suffix", ["", "_error", "_repeatability", "_variability", "_number_of_observations"])
def test_duplicate_role_aliases_fail_with_context_before_provider(role_inputs, monkeypatch, suffix):
    obs, fp, flux = role_inputs
    obs["mf" + suffix] = obs["co2" + suffix]
    monkeypatch.setattr(surface, "_surface_operations", lambda: pytest.fail("Called provider"))
    with pytest.raises(ValueError, match=rf"Observation role.*duplicate aliases.*co2{suffix}.*mf{suffix}"):
        prepare(obs, fp, flux)


def test_explicit_primary_rejects_conflicting_canonical_alias(role_inputs):
    obs, fp, flux = role_inputs
    obs = obs.rename(co2="custom")
    obs["mf"] = obs.custom
    with pytest.raises(ValueError, match="primary mole fraction.*custom, mf"):
        prepare(obs, fp, flux, observation_variable="custom")
    with pytest.raises(ValueError, match="primary variable 'absent' is missing"):
        prepare(obs.drop_vars("mf"), fp, flux, observation_variable="absent")


def test_primary_selector_cannot_reuse_an_uncertainty_role(role_inputs):
    obs, fp, flux = role_inputs
    with pytest.raises(ValueError, match="selected for both the primary.*error"):
        prepare(obs.drop_vars("co2"), fp, flux, observation_variable="co2_error")
