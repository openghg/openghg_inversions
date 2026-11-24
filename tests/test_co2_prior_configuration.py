"""Configured priors retain the explicit Python target; boundary choices bind at construction."""

import json
from dataclasses import replace
import tomllib

import numpy as np
import pytest
import pytensor

from openghg_inversions.correlated_state import CorrelatedLognormalPrior, CorrelatedPrior
from openghg_inversions.models.boundary import AnomalyScalingChoice, RawCenteredSensitivity
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.outer_flux import OuterFluxChoice
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    Co2ModelConfig,
    Co2O2ModelConfig,
    Co2O2PreparedInputs,
    build_co2_model,
    build_rhime_co2_o2,
    prepare_co2_o2_inputs,
)
from openghg_inversions.rhime.co2.co2_preparation import Co2PreparedInputs, prepare_co2_inputs
from openghg_inversions.rhime.co2.co2_o2_configuration import load_co2_o2_params
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.sigma import SigmaAlignment
from openghg_inversions.rhime.co2.execution import load_co2_params
from openghg_inversions.rhime.co2.inference import build_co2_inference
from test_co2_bound_frontends import _native, _operands
from test_co2_o2_fixed_ou import _prepared, _kwargs
from test_centered_handoffs import _ordinary


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64", cast_policy="numpy+floatX"):
        yield


def test_public_configuration_annotations_resolve_at_runtime():
    from typing import get_type_hints

    assert get_type_hints(Co2ModelConfig.model_arguments)


def _parity(left, right):
    assert [rv.name for rv in left.free_RVs] == [rv.name for rv in right.free_RVs]
    point = left.initial_point()
    for name, value in point.items():
        point[name] = value + np.linspace(0.1, 0.3, value.size).reshape(value.shape)
    np.testing.assert_allclose(left.compile_logp()(point), right.compile_logp()(point), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        left.compile_dlogp()(point), right.compile_dlogp()(point), rtol=1e-11, atol=1e-9
    )

    def prediction(model):
        return model.compile_fn(
            model.replace_rvs_by_values([model.modelled_concentration]),
            inputs=model.value_vars,
            on_unused_input="ignore",
        )(point)[0]

    np.testing.assert_allclose(prediction(left), prediction(right), rtol=1e-13, atol=1e-11)


@pytest.mark.parametrize("cached", [False, True])
def test_configured_co2_inference_uses_the_python_boundary_choice(cached):
    prepared = _ordinary()
    inv = prepared.inv_inputs
    inv = inv.assign(
        alpha_prior_mean=inv.alpha_prior_mean.copy(data=np.ones(2)),
        alpha_prior_covariance=inv.alpha_prior_covariance.copy(data=np.array([[0.08, 0.01], [0.01, 0.06]])),
    )
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inv))
    prior = CorrelatedPrior(
        inv.alpha_prior_mean, inv.alpha_prior_covariance, covariance_dim="region_cov"
    )
    mean_shift_prior = {"pdf": "normal", "mu": 0.0, "sigma": 0.2}
    boundary = AnomalyScalingChoice(mean_shift_prior, 0.3)
    likelihood = (
        {"kind": "fixed_ou", "tau_hours": 5.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75}}
        if cached
        else {"kind": "additive_sigma", "fixed_model_mismatch": 0.3}
    )
    request = {
        "model": {
            "boundary": {"type": "anomaly_scaling", "scale_sd": 0.3, "mean_shift_prior": mean_shift_prior},
        },
        "likelihood": likelihood,
    }
    choices = Co2ModelConfig.from_params(load_co2_params(params=request)).model_arguments()
    assert choices["boundary"] == boundary
    built, _ = build_co2_inference(prepared, choices)
    if cached:
        expected = build_co2_model(
            inv.H,
            retained_prior=prior,
            fixed_prior_contribution=inv.fixed_prior_contribution,
            observations=inv.mf,
            observation_error=inv.mf_error,
            aggregation_error=resolve_aggregation_error(inv, prepared.aggregation_error_mode),
            boundary=boundary.bind(inv.mf, RawCenteredSensitivity.validate(inv)),
            likelihood_builder=add_fixed_ou_gaussian_likelihood,
            likelihood_kwargs={k: v for k, v in likelihood.items() if k != "kind"},
        )
    else:
        python_choices = {**choices, "boundary": boundary}
        expected = build_co2_inference(prepared, python_choices)[0].model
    _parity(built.model, expected)


class _Built(Exception):
    pass


@pytest.mark.parametrize("cached", [False, True])
def test_linked_config_independent_species_corrections_and_one_flux_state(monkeypatch, cached):
    from openghg_inversions.rhime.co2 import co2_o2_cached_sigma_runner

    prepared = _prepared()
    prepared = replace(
        prepared, boundary_operands={channel: _operands(_native(prepared, channel)) for channel in ("co2", "o2")},
    )
    kwargs = _kwargs(prepared)
    prior = CorrelatedPrior.from_lognormal(prepared.retained_prior)
    expected_boundary, channels = {}, {}
    for channel, sd in (("co2", 0.2), ("o2", 0.003)):
        mean_shift_prior = {"pdf": "normal", "mu": 0.0, "sigma": sd}
        native_dim = str(kwargs[f"{channel}_sensitivity"].dims[0])
        expected_boundary[channel] = AnomalyScalingChoice(mean_shift_prior, 0.3).bind(
            _native(prepared, channel).rename(observation=native_dim),
            prepared.boundary_operands[channel],
        )
        channels[channel] = {
            "units": "ppm",
            "independent_error_sd": 0.1,
            "boundary": {"type": "anomaly_scaling", "scale_sd": 0.3, "mean_shift_prior": mean_shift_prior},
        }
    request = {"channels": channels}
    choices = Co2O2ModelConfig.from_params(load_co2_o2_params(params=request)).bind(prepared)
    ou = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    if cached:
        built = []
        builder = co2_o2_cached_sigma_runner._build_co2_o2_cached_sigma_model

        def capture(**model_kwargs):
            built.append(builder(**model_kwargs).model)
            raise _Built

        monkeypatch.setattr(co2_o2_cached_sigma_runner, "_build_co2_o2_cached_sigma_model", capture)
        with pytest.raises(_Built):
            co2_o2_cached_sigma_runner.run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
                prepared_inputs=prepared, **choices, **ou,
            )
        configured = built[0]
    else:
        configured = build_rhime_co2_o2(prepared_inputs=prepared, **choices).model
    expected = build_co2_o2_model(
        **{**kwargs, "retained_prior": prior},
        boundary=expected_boundary, **(ou if cached else {})
    )
    _parity(configured, expected)
    assert len([rv for rv in configured.free_RVs if rv.name.startswith("flux_scaling")]) == 1


def _with_groups(prepared, groups):
    """Label the prepared retained states with basis groups and sources."""
    inv = prepared.inv_inputs
    state = str(inv.alpha_prior_mean.dims[0])
    labels = {"basis_group": (state, groups), "source": (state, ["GPP"] * len(groups))}
    inv = inv.assign_coords(labels)
    return replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inv))


@pytest.mark.parametrize("cached", [False, True])
def test_configured_outer_flux_matches_python_option(cached):
    prepared = _with_groups(_ordinary(), ["inner", "outer"])
    inv = prepared.inv_inputs
    mean_shift_prior = {"pdf": "normal", "mu": 0.0, "sigma": 0.2}
    likelihood = (
        {"kind": "fixed_ou", "tau_hours": 5.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75}}
        if cached
        else {"kind": "additive_sigma", "fixed_model_mismatch": 0.3}
    )
    request = {
        "model": {
            "boundary": {"type": "anomaly_scaling", "scale_sd": 0.3, "mean_shift_prior": mean_shift_prior},
            "outer_flux": {"mode": "fixed", "outer_groups": ["outer"]},
        },
        "likelihood": likelihood,
    }
    choices = Co2ModelConfig.from_params(load_co2_params(params=request)).model_arguments()
    assert choices["outer_flux"] == OuterFluxChoice("fixed", ("outer",))
    built, _ = build_co2_inference(prepared, choices)
    assert json.loads(json.dumps(built.output_contract.metadata))["outer_flux_mode"] == "fixed"

    prior = CorrelatedPrior(inv.alpha_prior_mean, inv.alpha_prior_covariance, covariance_dim="region_cov")
    expected = build_co2_model(
        inv.H,
        retained_prior=prior,
        fixed_prior_contribution=inv.fixed_prior_contribution,
        observations=inv.mf,
        observation_error=inv.mf_error,
        aggregation_error=resolve_aggregation_error(inv, prepared.aggregation_error_mode),
        boundary=AnomalyScalingChoice(mean_shift_prior, 0.3).bind(inv.mf, RawCenteredSensitivity.validate(inv)),
        outer_flux=OuterFluxChoice("fixed", ("outer",)),
        **(
            {"likelihood_builder": add_fixed_ou_gaussian_likelihood,
             "likelihood_kwargs": {k: v for k, v in likelihood.items() if k != "kind"}}
            if cached else
            {"sigma_alignment": SigmaAlignment.from_observations(inv.mf), "fixed_model_mismatch": 0.3}
        ),
    )
    _parity(built.model, expected)


@pytest.mark.parametrize("cached", [False, True])
def test_linked_configured_outer_flux_matches_python_option(monkeypatch, cached):
    from openghg_inversions.rhime.co2 import co2_o2_cached_sigma_runner

    prepared = _prepared()
    prepared = replace(
        prepared,
        boundary_operands={channel: _operands(_native(prepared, channel)) for channel in ("co2", "o2")},
        retained_prior=CorrelatedLognormalPrior(
            prepared.retained_prior.mean.assign_coords(
                basis_group=(prepared.retained_prior.state_dim, ["inner", "inner", "outer", "outer", "outer"]),
            ),
            prepared.retained_prior.arithmetic_covariance,
        ),
    )
    kwargs = _kwargs(prepared)
    expected_boundary, channels = {}, {}
    for channel, sd in (("co2", 0.2), ("o2", 0.003)):
        mean_shift_prior = {"pdf": "normal", "mu": 0.0, "sigma": sd}
        native_dim = str(kwargs[f"{channel}_sensitivity"].dims[0])
        expected_boundary[channel] = AnomalyScalingChoice(mean_shift_prior, 0.3).bind(
            _native(prepared, channel).rename(observation=native_dim), prepared.boundary_operands[channel],
        )
        channels[channel] = {
            "units": "ppm",
            "independent_error_sd": 0.1,
            "boundary": {"type": "anomaly_scaling", "scale_sd": 0.3, "mean_shift_prior": mean_shift_prior},
        }
    outer = {"mode": "fixed", "outer_groups": ["outer"], "fixed_sources": ["ocean"]}
    request = {"channels": channels, "outer_flux": outer}
    choices = Co2O2ModelConfig.from_params(load_co2_o2_params(params=request)).bind(prepared)
    ou = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    if cached:
        built = []
        builder = co2_o2_cached_sigma_runner._build_co2_o2_cached_sigma_model

        def capture(**model_kwargs):
            built.append(builder(**model_kwargs).model)
            raise _Built

        monkeypatch.setattr(co2_o2_cached_sigma_runner, "_build_co2_o2_cached_sigma_model", capture)
        with pytest.raises(_Built):
            co2_o2_cached_sigma_runner.run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
                prepared_inputs=prepared, **choices, **ou,
            )
        configured = built[0]
    else:
        configured = build_rhime_co2_o2(prepared_inputs=prepared, **choices).model
    outer_flux = OuterFluxChoice("fixed", ("outer",), ("ocean",))
    expected = build_co2_o2_model(**kwargs, boundary=expected_boundary, outer_flux=outer_flux, **(ou if cached else {}))
    _parity(configured, expected)


@pytest.mark.parametrize(
    "outer, match",
    [
        ({"mode": "multiplicative", "outer_groups": ["outer"]}, 'must be "fixed"'),
        ({"mode": "fixed"}, "outer_groups"),
        ({"mode": "fixed", "outer_groups": "outer"}, "list of strings"),
        ({"mode": "fixed", "outer_groups": ["outer"], "level_sd": 1.0}, "Unknown"),
    ],
)
def test_invalid_outer_flux_options_fail(outer, match):
    with pytest.raises(ValueError, match=match):
        Co2ModelConfig.from_params(load_co2_params(params={
            "model": {"outer_flux": outer}, "likelihood": {"kind": "additive_sigma"},
        }))


@pytest.mark.parametrize("labelled", [True, False])
def test_outer_flux_selectors_survive_co2_preparation_and_saving(tmp_path, labelled):
    from test_co2_preparation import _canonical_inputs, _reduction

    canonical = _canonical_inputs()
    reduction = _reduction(canonical)
    mean = reduction.retained_mean
    state = str(mean.dims[0])
    if labelled:
        groups = ["inner"] * (mean.sizes[state] - 2) + ["outer", "outer"]
        sources = ["GPP"] * (mean.sizes[state] - 1) + ["ocean"]
        reduction = replace(reduction, retained_mean=mean.assign_coords(
            basis_group=(state, groups), source=(state, sources),
        ))
    prepare_co2_inputs(canonical, reduction).save(tmp_path / "prepared.nc")
    inv = Co2PreparedInputs.load(tmp_path / "prepared.nc").inv_inputs
    prior = CorrelatedPrior(inv.alpha_prior_mean, inv.alpha_prior_covariance, covariance_dim="region_cov")
    # The fixture's states are all correlated, so only the selector labels are checked here.
    choice = OuterFluxChoice("fixed", ("outer",))
    if not labelled:
        with pytest.raises(ValueError, match="basis_group"):
            choice.resolve_states(prior)
        return
    assert inv.alpha_prior_mean["source"].values.tolist() == sources
    assert inv.alpha_prior_mean["basis_group"].values.tolist() == groups


def test_outer_flux_selectors_survive_linked_preparation_and_saving():
    from test_rhime_co2_o2_preparation import _inputs

    inputs = _inputs()
    prior = inputs["retained_prior"]
    groups = ["inner", "inner", "outer", "outer", "outer"]
    inputs["retained_prior"] = CorrelatedLognormalPrior(
        prior.mean.assign_coords(basis_group=(prior.state_dim, groups)), prior.arithmetic_covariance,
    )
    prepared = Co2O2PreparedInputs.from_datatree(prepare_co2_o2_inputs(**inputs).to_datatree())
    states = OuterFluxChoice("fixed", ("outer",), ("ocean",)).resolve_states(
        CorrelatedPrior.from_lognormal(prepared.retained_prior)
    )
    ocean = np.char.lower(prepared.retained_prior.mean["source"].values.astype(str)) == "ocean"
    outer = np.array(groups) == "outer"
    np.testing.assert_array_equal(states.fixed.values, outer & ocean)


def test_toml_outer_flux_choice_is_the_python_choice():
    request = tomllib.loads("""
        [model.outer_flux]
        mode = "fixed"
        outer_groups = ["outer"]
        fixed_sources = ["ocean"]
        [likelihood]
        kind = "additive_sigma"
    """)
    config = Co2ModelConfig.from_params(load_co2_params(params=request))
    assert config.outer_flux == OuterFluxChoice("fixed", ("outer",), ("ocean",))


def test_partial_outer_flux_overrides_merge_with_the_base_table():
    base = {"mode": "fixed", "outer_groups": ["outer"]}
    expected = OuterFluxChoice("fixed", ("outer",), ("ocean",))
    single = load_co2_params(
        params={"model": {"outer_flux": dict(base)}, "likelihood": {"kind": "additive_sigma"}},
        overrides={"model": {"outer_flux": {"fixed_sources": ["ocean"]}}},
    )
    assert Co2ModelConfig.from_params(single).outer_flux == expected
    linked = load_co2_o2_params(
        params={"outer_flux": dict(base), "channels": {"co2": {"units": "ppm"}, "o2": {"units": "ppm"}}},
        overrides={"outer_flux": {"fixed_sources": ["ocean"]}},
    )
    assert linked["outer_flux"] == {**base, "fixed_sources": ["ocean"]}
    assert Co2O2ModelConfig.from_params(linked).outer_flux == expected


@pytest.mark.parametrize("request_kind", ["params_file", "config_file"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_full_run_protects_consumed_request_before_model_resolution(tmp_path, monkeypatch, request_kind, alias):
    import json
    from openghg_inversions.rhime.co2 import execution

    prepared = _ordinary()
    source = tmp_path / "request.nc"
    destination = source if alias == "direct" else tmp_path / "trace.nc"
    facts = {"start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE"}
    request = {
        "prepared_inputs": facts,
        "likelihood": {"kind": "additive_sigma"},
        "outputs": {"output_format": "basic", "save_trace": str(destination)},
    }
    if request_kind == "params_file":
        source.write_text(json.dumps(request), encoding="utf-8")
    else:
        source.write_text(
            "[prepared_inputs]\n" + "\n".join(f"{key} = {json.dumps(value)}" for key, value in facts.items())
            + '\n[likelihood]\nkind = "additive_sigma"\n[outputs]\noutput_format = "basic"\nsave_trace = '
            + json.dumps(str(destination)) + "\n", encoding="utf-8",
        )
    if alias == "symlink":
        destination.symlink_to(source)
    elif alias == "hardlink":
        destination.hardlink_to(source)
    before = source.read_bytes()
    monkeypatch.setattr(Co2ModelConfig, "model_arguments", lambda *a: pytest.fail("model resolution reached"))
    monkeypatch.setattr(execution, "sample_co2_inference", lambda *a, **kw: pytest.fail("sampling reached"))
    with pytest.raises(ValueError, match="overlaps an input or output artifact"):
        execution.run_co2(**{request_kind: source}, prepared=prepared)
    assert source.read_bytes() == before
    assert destination.read_bytes() == before


def test_full_run_mapping_request_adds_no_file_to_protected_inputs(tmp_path, monkeypatch):
    from openghg_inversions.rhime.co2 import execution

    request = {
        "prepared_inputs": {"start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE"},
        "likelihood": {"kind": "additive_sigma"},
        "outputs": {"output_format": "basic", "save_trace": str(tmp_path / "trace.nc")},
    }
    def reached_model(*args):
        raise RuntimeError("request passed preflight")
    monkeypatch.setattr(Co2ModelConfig, "model_arguments", reached_model)
    monkeypatch.setattr(execution, "sample_co2_inference", lambda *a, **kw: pytest.fail("sampling reached"))
    with pytest.raises(RuntimeError, match="request passed preflight"):
        execution.run_co2(params=request, prepared=_ordinary())
    assert not (tmp_path / "trace.nc").exists()


def test_ordinary_linked_configuration_binds_declared_fields():
    prepared = _prepared()
    request = {"channels": {name: {"units": "ppm", "independent_error_sd": .1}
                             for name in ("co2", "o2")}}
    bound = Co2O2ModelConfig.from_params(request).bind(prepared)
    np.testing.assert_array_equal(bound["independent_error_sd"], .1)
    assert "site_amplitude_prior" not in bound
