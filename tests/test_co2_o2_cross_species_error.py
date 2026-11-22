"""Native cross-species error correlation in the linked CO2/O2 likelihood."""

from __future__ import annotations

import json

import numpy as np
import xarray as xr
import pytest
from scipy.stats import multivariate_normal

from boundary_test_utils import bound_linked_model_arguments
from openghg_inversions.rhime.co2.co2_o2_configuration import Co2O2ModelConfig
from openghg_inversions.rhime.co2.co2_o2_error_correlation import linked_cross_species_pairs
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2

from test_co2_o2_fixed_ou import _dense, _kwargs, _per_meg_prepared, _prepared

# The fixture pairs (co2:A, o2:A) at hours 4 and 0; o2:B at hour 2 is unpaired.
RHO = -0.9


def _channel_errors(kwargs):
    """Independent errors of 0.5 ppm for CO2 and 5 per meg for O2."""
    species = kwargs["observations"].species.values
    error = kwargs["independent_error_sd"]
    return {**kwargs, "independent_error_sd": error.copy(data=np.where(species == "co2", 0.5, 5.0))}


def _cross(kwargs, correlation):
    observations = kwargs["observations"]
    sd = kwargs["independent_error_sd"].values
    pairs = linked_cross_species_pairs(observations, correlation)
    return pairs.covariance(sd)


def _logp(model):
    point = model.initial_point()
    mean = model.compile_fn(model.replace_rvs_by_values([model["modelled_concentration"]]))(point)[0]
    return model.compile_logp(vars=[model["y"]])(point), mean


def test_pairs_follow_site_and_time_and_check_site_mappings():
    observations = _prepared().observations
    pairs = linked_cross_species_pairs(observations, RHO)
    species = observations.species.values[np.concatenate([pairs.first, pairs.second])]
    assert species.tolist() == ["co2", "co2", "o2", "o2"]
    for first, second in zip(pairs.first, pairs.second, strict=True):
        assert observations.site.values[first] == observations.site.values[second] == "A"
        assert observations.time.values[first] == observations.time.values[second]
    assert linked_cross_species_pairs(observations, {"A": -0.5}).correlation.tolist() == [-0.5, -0.5]
    for correlation, message in (
        ({"A": -0.5, "C": 0.1}, "unobserved sites"),
        ({"B": 0.1}, "lacks paired sites"),
        (1.0, "strictly between"),
        (True, "strictly between"),
    ):
        with pytest.raises(ValueError, match=message):
            linked_cross_species_pairs(observations, correlation)
    # Pairing is keyed by channel labels, not specific to CO2/O2.
    relabelled = xr.DataArray(
        observations.values,
        dims="row",
        coords={
            "species": ("row", ["co2", "co2", "c14", "c14", "c14"]),
            "site": ("row", observations.site.values),
            "time": ("row", observations.time.values),
        },
    )
    generic = linked_cross_species_pairs(relabelled, RHO, channels=("co2", "c14"))
    np.testing.assert_array_equal(generic.first, pairs.first)
    np.testing.assert_array_equal(generic.second, pairs.second)
    with pytest.raises(ValueError, match="co2/o2 pairs"):
        linked_cross_species_pairs(relabelled, RHO)
    unpaired = observations.assign_coords(site=("observation", ["A", "A", "B", "B", "B"]))
    with pytest.raises(ValueError, match="no same-site"):
        linked_cross_species_pairs(unpaired, RHO)


@pytest.mark.usefixtures("double_precision")
@pytest.mark.parametrize("units", ["ppm", "per meg"])
def test_linked_fixed_error_likelihood_matches_dense_oracle(units):
    prepared = _prepared() if units == "ppm" else _per_meg_prepared()
    kwargs = _channel_errors(_kwargs(prepared))
    model = build_co2_o2_model(
        **bound_linked_model_arguments({**kwargs, "cross_species_error_correlation": RHO})
    )
    actual, mean = _logp(model)
    covariance = (
        kwargs["aggregation_error"].covariance.values
        + np.diag(kwargs["independent_error_sd"].values ** 2)
        + _cross(kwargs, RHO)
    )
    expected = multivariate_normal.logpdf(kwargs["observations"].values, mean=mean, cov=covariance)
    np.testing.assert_allclose(actual, expected, rtol=1e-8)
    without = build_co2_o2_model(**bound_linked_model_arguments(kwargs))
    assert not np.isclose(_logp(without)[0], actual)


@pytest.mark.usefixtures("double_precision")
def test_linked_fixed_ou_with_mixed_units_and_site_rho_matches_dense_oracle():
    kwargs = _channel_errors(_kwargs(_per_meg_prepared()))
    tau = {"co2:A": 6.0, "o2:A": 8.0, "o2:B": 2.0}
    amplitudes = {"co2:A": 0.4, "o2:A": 6.0, "o2:B": 3.0}
    model = build_co2_o2_model(
        **bound_linked_model_arguments(
            {
                **kwargs,
                "tau_hours": tau,
                "fixed_site_amplitudes": amplitudes,
                "cross_species_error_correlation": {"A": RHO},
            }
        )
    )
    actual, mean = _logp(model)
    covariance = _dense(kwargs, np.array([0.4, 6.0, 3.0]), tau) + _cross(kwargs, {"A": RHO})
    expected = multivariate_normal.logpdf(kwargs["observations"].values, mean=mean, cov=covariance)
    np.testing.assert_allclose(actual, expected, rtol=1e-8)


def test_configured_correlation_reaches_the_built_model_and_metadata():
    prepared = _per_meg_prepared()
    config = {
        "channels": {
            "co2": {"units": "ppm", "independent_error_sd": 0.5},
            "o2": {"units": "ppm", "independent_error_sd": 5.0},
        },
        "likelihood": {"cross_species_error_correlation": {"A": RHO}},
    }
    model_config = Co2O2ModelConfig.from_params(config)
    assert model_config.cross_species_error_correlation == {"A": RHO}
    ou = {
        **config,
        "likelihood": {
            "kind": "fixed_ou",
            "tau_hours": 6.0,
            "fixed_site_amplitudes": 0.0,
            "cross_species_error_correlation": RHO,
        },
    }
    assert Co2O2ModelConfig.from_params(ou).cross_species_error_correlation == RHO
    with pytest.raises(ValueError, match="strictly between"):
        Co2O2ModelConfig.from_params({**config, "likelihood": {"cross_species_error_correlation": -1.0}})

    same_units = _prepared()
    arguments = model_config.bind(same_units)
    assert arguments["cross_species_error_correlation"] == {"A": RHO}
    built = build_rhime_co2_o2(prepared_inputs=same_units, **arguments)
    roles = built.output_contract.variable_roles
    assert roles["cross_species_error_correlation"] == "paired_error_correlation"
    np.testing.assert_allclose(built.model["paired_error_correlation"].get_value(), [RHO, RHO, RHO, 0.0, RHO])
    np.testing.assert_array_equal(built.model["paired_error_partner"].get_value(), [2, 4, 0, -1, 1])
    metadata = json.loads(json.dumps(built.output_contract.metadata))
    assert metadata["cross_species_error_correlation"]["value"] == {"A": RHO}
    assert prepared.observations.observation_units.values.tolist()[-1] == "per meg"
