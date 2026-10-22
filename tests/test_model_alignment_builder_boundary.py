"""Eager builders require model-owned BC alignment before graph construction."""

from dataclasses import replace
from types import SimpleNamespace

import dask.array as da
from dask import delayed
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis import BasisFunctions
from openghg_inversions.rhime.materialization import materialize_pymc_inputs
from openghg_inversions.rhime.multisector import build_multisector_model, materialize_multisector_inputs
from openghg_inversions.rhime.nested import build_nested_rhime_model, materialize_nested_inputs
from openghg_inversions.rhime.params import RhimeConfig, group_params
from openghg_inversions.rhime.standard import build_standard_model, materialize_standard_inputs
from tests.test_model_owned_alignment_config import request
from tests.test_standard_builder import _inputs, _model
from tests.test_staged_workflow import _prepared


def _preparation(inputs):
    basis = _prepared().basis_functions
    if "source" in inputs.H.dims:
        basis = BasisFunctions.from_flat_basis(
            basis.flat_basis(), flux=xr.concat([basis.flux, basis.flux * 2], dim=inputs.source),
            operator_kwargs={"state_dim": "region"},
        )
    return SimpleNamespace(inv_inputs=inputs, basis_functions=basis)


@pytest.fixture(params=["standard", "multisector", "nested"])
def family(request):
    """Extend the existing two-observation fixture for each concrete recipe."""
    inputs = _inputs().assign_coords(
        site=("nmeasure", ["TAC", "TAC"]),
        time=("nmeasure", np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[ns]")),
        bc_region=["north"], region=[0],
    ).assign(H_bc=(("bc_region", "nmeasure"), [[2., 3.]]))
    inputs.H_bc.attrs["boundary_sensitivity_preparation"] = "raw-spatial"
    spec = replace(_model(), use_bc=True)
    if request.param == "multisector":
        inputs["H"] = xr.concat([inputs.H, inputs.H * 2], dim=xr.IndexVariable("source", ["inventory", "other"]))
        sector = replace(spec.sectors[0], name="other", flux_source="other", variable_suffix="other")
        spec = replace(spec, sectors=(*spec.sectors, sector))
        return inputs, spec, build_multisector_model, materialize_multisector_inputs
    if request.param == "nested":
        inputs["H_inner"] = inputs.H.rename(region="inner_region")
        return inputs, spec, build_nested_rhime_model, materialize_nested_inputs
    return inputs, spec, build_standard_model, materialize_standard_inputs


def test_one_eager_raw_input_builds_two_frequencies_via_family_materializer(family):
    inputs, spec, build, materialize = family
    original = inputs.copy(deep=True)
    for variable in inputs.data_vars.values():
        variable.data.flags.writeable = False
    for frequency, expected in ((None, [[2., 3.]]), ("monthly", [[2., 0.], [0., 3.]])):
        model_spec = replace(spec, bc_freq=frequency)
        eager, priors = materialize(_preparation(inputs), model_spec)
        np.testing.assert_allclose(eager.H_bc, expected)
        built = build(eager, model_spec, priors=priors)
        np.testing.assert_allclose(built.model["hbc"].get_value(), np.asarray(expected).T)
    xr.testing.assert_identical(inputs, original)


@pytest.mark.parametrize("frequency", [None, "monthly", "1D"])
def test_marked_raw_bc_rejected_before_graph_build(family, monkeypatch, frequency):
    inputs, spec, build, materialize = family
    spec = replace(spec, bc_freq=frequency)
    original = inputs.copy(deep=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Raw BC reached graph construction")

    monkeypatch.setattr("openghg_inversions.rhime.standard.registered_model", forbidden)
    monkeypatch.setattr("openghg_inversions.rhime.nested.registered_model", forbidden)
    # Generic materialization cannot establish model-owned period semantics.
    eager = materialize_pymc_inputs(_preparation(inputs), variable_names=tuple(inputs.data_vars))
    with pytest.raises(ValueError, match=materialize.__name__):
        build(eager, spec, priors={})
    xr.testing.assert_identical(inputs, original)


@pytest.mark.parametrize("frequency", ["monthly", "1D"])
def test_unmarked_matrix_cannot_silently_ignore_requested_frequency(family, frequency):
    inputs, spec, build, materialize = family
    inputs.H_bc.attrs.clear()
    original = inputs.copy(deep=True)
    with pytest.raises(ValueError, match=materialize.__name__):
        build(inputs, replace(spec, bc_freq=frequency), priors={})
    xr.testing.assert_identical(inputs, original)


def test_grouped_monthly_request_rejects_unmarked_raw_matrix():
    config = RhimeConfig.from_params(
        group_params(request(mismatch_model="fixed_error", bc_freq="monthly")), multisector=False,
    )
    inputs = _inputs().assign(H_bc=(("bc_region", "nmeasure"), [[2., 3.]]))
    assert config.model.bc_freq == "monthly"
    with pytest.raises(ValueError, match="materialize_standard_inputs"):
        build_standard_model(inputs, config.model, priors={})


def test_unmarked_final_constant_matrix_remains_supported(family):
    inputs, spec, build, materialize = family
    inputs.H_bc.attrs.clear()
    original = inputs.copy(deep=True)
    _, priors = materialize(_preparation(inputs), spec)
    built = build(inputs, spec, priors=priors)
    np.testing.assert_allclose(built.model["hbc"].get_value(), [[2.], [3.]])
    xr.testing.assert_identical(inputs, original)


def test_disabled_bc_ignores_marked_lazy_raw_operands(family):
    inputs, spec, build, materialize = family

    @delayed
    def unused():
        pytest.fail("Disabled raw BC operand was computed")

    inputs["H_bc"] = inputs.H_bc.copy(data=da.from_delayed(unused(), shape=(1, 2), dtype=float))
    original = inputs.copy(deep=True)
    spec = replace(spec, use_bc=False, bc_freq="monthly")
    eager, priors = materialize(_preparation(inputs), spec)
    built = build(eager, spec, priors=priors)
    assert "hbc" not in built.model
    assert isinstance(inputs.H_bc.data, da.Array)
    xr.testing.assert_identical(inputs, original)


@pytest.mark.parametrize(("producer", "consumer"), [
    ({"bc_freq": None}, {"bc_freq": "monthly"}),
    ({"bc_freq": "monthly"}, {"bc_freq": None}),
    ({"bc_freq": "1D", "bc_freq_anchor": "2020-01-01"},
     {"bc_freq": "24h", "bc_freq_anchor": "2020-01-02"}),
])
def test_changed_boundary_policy_rejected_before_graph_or_execution(family, monkeypatch, producer, consumer):
    inputs, spec, build, materialize = family
    eager, priors = materialize(_preparation(inputs), replace(spec, **producer))

    @delayed
    def forbidden():
        pytest.fail("Policy validation executed a lazy operand")

    eager["H"] = eager.H.copy(data=da.from_delayed(forbidden(), shape=eager.H.shape, dtype=float))
    monkeypatch.setattr("openghg_inversions.rhime.standard.registered_model", forbidden)
    monkeypatch.setattr("openghg_inversions.rhime.nested.registered_model", forbidden)
    with pytest.raises(ValueError, match="H_bc.*different boundary-period policy"):
        build(eager, replace(spec, **consumer), priors=priors)


def test_equivalent_boundary_duration_and_anchor_accepted(family):
    inputs, spec, build, materialize = family
    producer = replace(spec, bc_freq="1D", bc_freq_anchor="2020-01-01")
    eager, priors = materialize(_preparation(inputs), producer)
    consumer = replace(spec, bc_freq="24h", bc_freq_anchor="2020-01-01T00:00:00.000000000")
    assert "hbc" in build(eager, consumer, priors=priors).model


@pytest.mark.parametrize(("producer", "consumer"), [
    ({"min_error": 1.}, {"min_error": 50.}),
    ({"min_error": {"TAC": 1.}}, {"min_error": {"TAC": 50.}}),
    ({"min_error": "residual"}, {"min_error": "percentile"}),
    ({"min_error": "residual"}, {"min_error": "residual", "min_error_options": {"by_site": True}}),
])
def test_changed_floor_policy_rejected_before_graph_or_execution(family, monkeypatch, producer, consumer):
    from openghg_inversions.rhime.specs import AdditiveSigmaSettings

    inputs, spec, build, materialize = family
    inputs["mf_mod"] = inputs.mf * .5
    likelihood = AdditiveSigmaSettings(use_minimum_error_floor=True, **producer)
    spec = replace(spec, likelihood=likelihood)
    eager, priors = materialize(_preparation(inputs), spec)

    @delayed
    def forbidden():
        pytest.fail("Floor policy validation executed a lazy operand")

    eager["min_error"] = eager.min_error.copy(data=da.from_delayed(forbidden(), shape=(2,), dtype=float))
    monkeypatch.setattr("openghg_inversions.rhime.standard.registered_model", forbidden)
    monkeypatch.setattr("openghg_inversions.rhime.nested.registered_model", forbidden)
    with pytest.raises(ValueError, match="min_error.*different minimum-error policy"):
        build(eager, replace(spec, likelihood=replace(likelihood, **consumer)), priors=priors)


def test_retained_site_floor_policy_ignores_unused_choices_and_allows_graph_sigma_changes(family):
    from openghg_inversions.rhime.specs import AdditiveSigmaSettings

    inputs, spec, build, materialize = family
    likelihood = AdditiveSigmaSettings(
        use_minimum_error_floor=True, min_error={"TAC": 1., "DROPPED": 3.},
    )
    spec = replace(spec, likelihood=likelihood)
    eager, priors = materialize(_preparation(inputs), spec)
    consumer = replace(spec, likelihood=replace(
        likelihood, min_error={"TAC": 1., "DROPPED": 50.}, min_error_options={"by_site": True},
        sigma_freq="monthly", sigma_per_site=False,
    ))
    assert "sigma" in build(eager, consumer, priors=priors).model


def test_unused_floor_settings_and_operands_do_not_execute(family):
    from openghg_inversions.rhime.specs import AdditiveSigmaSettings

    inputs, spec, build, materialize = family

    @delayed
    def forbidden():
        pytest.fail("Disabled floor was computed")

    inputs["mf_mod"] = inputs.mf.copy(data=da.from_delayed(forbidden(), shape=(2,), dtype=float))
    spec = replace(spec, likelihood=AdditiveSigmaSettings(min_error="residual"))
    eager, priors = materialize(_preparation(inputs), spec)
    # A stale producer floor is irrelevant when the consuming likelihood has no floor.
    eager["min_error"] = inputs.mf.copy(data=da.from_delayed(forbidden(), shape=(2,), dtype=float))
    eager.min_error.attrs["minimum_error_source_policy"] = '{"scalar": 1.0}'
    consumer = replace(spec, likelihood=replace(spec.likelihood, min_error=50.))
    assert "sigma" in build(eager, consumer, priors=priors).model


def test_independently_supplied_final_floor_remains_authoritative(family):
    from openghg_inversions.rhime.specs import AdditiveSigmaSettings

    inputs, spec, build, materialize = family
    spec = replace(spec, likelihood=AdditiveSigmaSettings(use_minimum_error_floor=True, min_error=50.))
    eager, priors = materialize(_preparation(inputs), spec)
    eager["min_error"] = ("nmeasure", [1., 1.])
    assert "sigma" in build(eager, spec, priors=priors).model


@pytest.mark.parametrize("floor", [None, 1., "percentile"])
def test_floor_policy_ignores_irrelevant_by_site_and_normalizes_scalar(family, floor):
    from openghg_inversions.rhime.specs import AdditiveSigmaSettings

    inputs, spec, build, materialize = family
    likelihood = AdditiveSigmaSettings(use_minimum_error_floor=True, min_error=floor)
    spec = replace(spec, likelihood=likelihood)
    eager, priors = materialize(_preparation(inputs), spec)
    consumer = replace(spec, likelihood=replace(
        likelihood, min_error=0. if floor is None else floor, min_error_options={"by_site": True},
    ))
    assert "sigma" in build(eager, consumer, priors=priors).model
