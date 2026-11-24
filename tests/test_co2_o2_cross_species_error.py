"""Native cross-species error correlation in the linked CO2/O2 likelihood."""

from __future__ import annotations

import json
from dataclasses import replace

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


def test_missing_times_are_never_paired():
    """Two NaT rows at one site have no known common time, so they stay independent."""
    observations = _prepared().observations
    times = observations.time.values.copy()
    # Rows 0 (co2:A, hour 4) and 2 (o2:A, hour 4) would otherwise pair.
    times[[0, 2]] = np.datetime64("NaT")
    missing = observations.assign_coords(time=("observation", times))
    pairs = linked_cross_species_pairs(missing, RHO)
    assert pairs.first.tolist() == [1]
    assert pairs.second.tolist() == [4]
    both_missing = observations.assign_coords(
        time=("observation", np.full(5, np.datetime64("NaT"), "M8[ns]"))
    )
    with pytest.raises(ValueError, match="no same-site"):
        linked_cross_species_pairs(both_missing, RHO)


@pytest.mark.parametrize("missing", [None, np.nan], ids=["None", "NaN"])
def test_missing_sites_are_never_paired(missing):
    """Two unknown sites are not the same site, and a valid pair still matches beside them."""
    observations = _prepared().observations
    sites = observations.site.values.astype(object)
    # Rows 0 (co2, hour 4) and 2 (o2, hour 4) share a time; only rows 1 and 4 keep site A.
    sites[[0, 2]] = missing
    unknown = observations.assign_coords(site=("observation", sites))
    pairs = linked_cross_species_pairs(unknown, RHO)
    assert pairs.first.tolist() == [1]
    assert pairs.second.tolist() == [4]
    assert linked_cross_species_pairs(unknown, {"A": RHO}).first.tolist() == [1]
    with pytest.raises(ValueError, match="no same-site"):
        linked_cross_species_pairs(
            observations.assign_coords(site=("observation", np.full(5, missing, dtype=object))), RHO
        )


@pytest.mark.usefixtures("double_precision")
def test_built_model_leaves_missing_time_rows_unpaired():
    from openghg_inversions.rhime.co2 import prepare_co2_o2_inputs
    from test_rhime_co2_o2 import _inputs

    inputs = _inputs()
    inputs["o2_units"] = "ppm"
    # co2:A and o2:A share hour 0 (a real pair) and both miss their other time (NaT).
    for channel, sites, hours in [("co2", ["A", "A"], [None, 0]), ("o2", ["A", "B", "A"], [None, 2, 0])]:
        observed = inputs[f"{channel}_observations"]
        times = [
            np.datetime64("NaT", "ns")
            if hour is None
            else np.datetime64("2021-01-01T00", "ns") + np.timedelta64(hour, "h")
            for hour in hours
        ]
        inputs[f"{channel}_observations"] = observed.assign_coords(
            site=(observed.dims[0], sites), time=(observed.dims[0], np.array(times, dtype="M8[ns]"))
        )
    kwargs = _channel_errors(_kwargs(prepare_co2_o2_inputs(**inputs)))
    # The model coordinate registry compares NaT as unequal, so keep time only on the row arrays.
    aggregation = kwargs["aggregation_error"]
    covariance = aggregation.covariance
    kwargs["aggregation_error"] = replace(
        aggregation,
        covariance=covariance.drop_vars([name for name in covariance.coords if str(name).startswith("time")]),
    )
    model = build_co2_o2_model(
        **bound_linked_model_arguments({**kwargs, "cross_species_error_correlation": RHO})
    )
    np.testing.assert_array_equal(model["paired_error_partner"].get_value(), [-1, 4, -1, -1, 1])
    actual, mean = _logp(model)
    covariance = (
        kwargs["aggregation_error"].covariance.values
        + np.diag(kwargs["independent_error_sd"].values ** 2)
        + _cross(kwargs, RHO)
    )
    assert covariance[0, 2] == kwargs["aggregation_error"].covariance.values[0, 2]
    expected = multivariate_normal.logpdf(kwargs["observations"].values, mean=mean, cov=covariance)
    np.testing.assert_allclose(actual, expected, rtol=1e-8)


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


@pytest.mark.parametrize("correlation", [np.float32(-0.9), {"A": np.float32(-0.9)}], ids=["scalar", "site"])
def test_runner_normalizes_numpy_correlations_for_pairing_and_metadata(correlation):
    """NumPy scalars are normalized once, so the stored metadata is plain JSON and matches the model."""
    arguments = Co2O2ModelConfig.from_params(
        {
            "channels": {
                "co2": {"units": "ppm", "independent_error_sd": 0.5},
                "o2": {"units": "ppm", "independent_error_sd": 5.0},
            }
        }
    ).bind(_prepared())
    built = build_rhime_co2_o2(
        prepared_inputs=_prepared(), **{**arguments, "cross_species_error_correlation": correlation}
    )
    stored = json.loads(json.dumps(built.output_contract.metadata))["cross_species_error_correlation"][
        "value"
    ]
    value = float(np.float32(-0.9))
    assert stored == (value if not isinstance(correlation, dict) else {"A": value})
    assert type(stored if not isinstance(stored, dict) else stored["A"]) is float
    np.testing.assert_allclose(
        built.model["paired_error_correlation"].get_value(), [value, value, value, 0.0, value], rtol=1e-6
    )
