"""Labelled boundary responses stay intact until the cached numerical boundary."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pytest
import xarray as xr
from dask import delayed
from dask import array as da

from openghg_inversions.models.boundary import AdditiveBoundary, MultiplicativeBoundary
from openghg_inversions.models.components import AffineBlock, AffineConcentrationTerm, ConcentrationTerm
from openghg_inversions.rhime.co2 import co2_cached_sigma_model, co2_cached_sigma_runner, co2_runner
from openghg_inversions.rhime.co2.co2_o2_model import _pad_channel_design
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR


@pytest.mark.parametrize("matrix", [False, True])
def test_padding_preserves_interleaved_rows_and_reversed_native_coefficient_labels(matrix):
    index = pd.MultiIndex.from_tuples(
        [("o2", "o1"), ("co2", "c2"), ("o2", "o2"), ("co2", "c1")],
        names=("species", "channel_observation"),
    )
    observations = xr.DataArray(
        np.zeros(4), dims="observation",
        coords=xr.Coordinates.from_pandas_multiindex(index, "observation"),
    )
    # Joint channel rows reverse native order; coefficient labels stay unsorted.
    design = xr.DataArray([10.0, 20.0], dims="native", coords={"native": ["c1", "c2"]})
    expected = np.array([0.0, 20.0, 0.0, 10.0])
    if matrix:
        design = xr.concat([design, 3 * design], dim="coefficient").assign_coords(
            coefficient=["south", "north"],
        ).transpose("native", "coefficient")
        expected = np.column_stack([expected, 3 * expected])
    padded = _pad_channel_design(design, observations, "co2")
    np.testing.assert_array_equal(padded.values, expected)
    assert padded.indexes["observation"].equals(index)
    assert padded.dims == ("observation", *design.dims[1:])
    if matrix:
        xr.testing.assert_identical(padded.coefficient, design.coefficient)


def _lower_until_target(monkeypatch, terms, observations):
    def capture_target(**kwargs):
        raise RuntimeError(kwargs)

    monkeypatch.setattr(co2_cached_sigma_model, "FixedOuCachedSigmaTarget", capture_target)
    co2_cached_sigma_model._add_cached_likelihood(
        terms, covariance=None, modelled_mean=pt.zeros(2), observations=observations,
        observation_error=xr.ones_like(observations), site_coord=None, site_index=None,
        initial_amplitudes=None, site_amplitude_prior={}, output_dim="observation",
    )


def _affine_term(fixed, *designs):
    return AffineConcentrationTerm(
        ConcentrationTerm.absolute(pt.zeros(2)), fixed,
        tuple(AffineBlock(design, pt.ones(design.shape[1])) for design in designs), (),
    )


@pytest.mark.parametrize("reversed_part", ["fixed", "block"])
def test_cached_lowering_rejects_reversed_rows_before_reading_payload(monkeypatch, reversed_part):
    observations = xr.DataArray([1.0, 2.0], dims="observation", coords={"observation": ["a", "b"]})
    fixed = xr.zeros_like(observations)
    design = xr.DataArray([[1.0], [2.0]], dims=("observation", "coefficient"),
                          coords={"observation": observations.observation, "coefficient": ["north"]})
    @delayed
    def unread_payload(shape):
        raise AssertionError("Mismatched labelled payload must not be read.")

    if reversed_part == "fixed":
        fixed = fixed.isel(observation=[1, 0]).copy(
            data=da.from_delayed(unread_payload((2,)), shape=(2,), dtype=float),
        )
    else:
        design = design.isel(observation=[1, 0]).copy(
            data=da.from_delayed(unread_payload((2, 1)), shape=(2, 1), dtype=float),
        )
    with pytest.raises(ValueError, match="align.*exact"):
        _lower_until_target(monkeypatch, (_affine_term(fixed, design),), observations)


def test_cached_lowering_keeps_independent_same_named_coefficient_axes(monkeypatch):
    observations = xr.DataArray([1.0, 2.0], dims="observation", coords={"observation": ["a", "b"]})
    design = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("observation", "coefficient"),
                          coords={"observation": observations.observation, "coefficient": ["south", "north"]})
    second = xr.DataArray([[5.0], [6.0]], dims=design.dims,
                          coords={"observation": observations.observation, "coefficient": ["level"]})
    terms = (_affine_term(xr.ones_like(observations), design),
             _affine_term(2 * xr.ones_like(observations), second))
    with pytest.raises(RuntimeError) as reached:
        _lower_until_target(monkeypatch, terms, observations)
    lowered = reached.value.args[0]
    np.testing.assert_array_equal(lowered["design"], [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])
    np.testing.assert_array_equal(lowered["fixed_contribution"], [3.0, 3.0])


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("mode", ["absent", "disabled", "default", "explicit", "additive"])
def test_materialization_runner_resolves_boundary_prior_once(monkeypatch, mode, cached):
    from test_co2_affine_output import _prepared

    prepared, _ = _prepared()
    inputs = prepared.inv_inputs.assign(
        H_bc=xr.DataArray([[0.5]], dims=("nmeasure", "bc_region"),
                          coords={"nmeasure": prepared.inv_inputs.nmeasure, "bc_region": ["north"]},
                          attrs={"units": "ppm"}),
    )
    choices = {"use_bc": mode != "disabled"}
    if mode == "explicit":
        choices["bc_prior"] = {"pdf": "normal", "mu": 1.2, "sigma": 0.3}
    if mode == "additive":
        inputs = inputs.drop_vars("H_bc").assign(
            H_bc_anomaly=xr.DataArray([[0.1]], dims=("nmeasure", "bc_region"), attrs={"units": "ppm"}),
            H_bc_mass=xr.DataArray([[1.0]], dims=("nmeasure", "bc_region"), attrs={"units": "1"}),
            bc_reference=xr.DataArray(400.0, attrs={"units": "ppm"}),
            bc_centering_weights=xr.DataArray([[1.0]], dims=("bc_curtain", "bc_period"),
                coords={"bc_curtain": ["north"], "bc_period": [inputs.time.values[0]]}, attrs={"units": "1"}),
        )
        inputs.attrs["boundary_representation"] = "centered-v1"
        choices.update(bc_mean_shift_prior={"pdf": "normal", "mu": 0.0, "sigma": 0.2}, bc_anomaly_scale=0.0)
    if mode == "absent":
        inputs = inputs.drop_vars("H_bc")
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
    calls = []
    prepare = co2_runner.prepare_boundary

    def record_boundary(*args, **kwargs):
        calls.append(kwargs)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(co2_runner, "prepare_boundary", record_boundary)
    def build_boundary():
        if cached:
            class BoundaryReached(Exception):
                pass

            def capture_boundary(*args, **kwargs):
                raise BoundaryReached(kwargs["boundary"])

            monkeypatch.setattr(co2_cached_sigma_runner, "build_co2_cached_sigma_model", capture_boundary)
            with pytest.raises(BoundaryReached) as reached:
                co2_cached_sigma_runner.build_rhime_co2_cached_sigma(
                    prepared_inputs=prepared, tau_hours=3.0,
                    site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.2}, **choices,
                )
            return reached.value.args[0]
        return co2_runner.materialize_co2_inputs(prepared_inputs=prepared, **choices)[2]

    if mode == "absent":
        with pytest.raises(ValueError, match="missing required"):
            build_boundary()
        assert calls == []
        return
    boundary = build_boundary()
    assert len(calls) == 1
    if mode == "disabled":
        assert boundary is None
        assert calls[0]["bc_prior"] is None
    elif mode == "additive":
        assert isinstance(boundary, AdditiveBoundary)
        assert calls[0]["bc_prior"] is None
        assert boundary.mean_prior == choices["bc_mean_shift_prior"]
    else:
        assert isinstance(boundary, MultiplicativeBoundary)
        expected = DEFAULT_BC_PRIOR if mode == "default" else choices["bc_prior"]
        assert calls[0]["bc_prior"] == boundary.prior == expected


def test_linked_runner_defaults_only_the_selected_boundary_once(monkeypatch):
    from openghg_inversions.rhime.co2 import co2_o2_runner
    from test_rhime_co2_o2 import _independent_error
    from test_rhime_co2_o2_baselines import _prepared

    prepared = _prepared(("co2", "o2"))
    calls, forwarded = [], []
    prepare = co2_o2_runner.prepare_boundary

    def record_boundary(*args, **kwargs):
        calls.append(kwargs)
        return prepare(*args, **kwargs)

    def capture_builder(**kwargs):
        forwarded.append(kwargs["boundary"])
        raise RuntimeError("boundaries reached")

    monkeypatch.setattr(co2_o2_runner, "prepare_boundary", record_boundary)
    monkeypatch.setattr(co2_o2_runner, "build_co2_o2_model", capture_builder)
    with pytest.raises(RuntimeError, match="boundaries reached"):
        co2_o2_runner.build_rhime_co2_o2(
            prepared_inputs=prepared, independent_error_sd=_independent_error(prepared),
            use_bc={"co2": True, "o2": False},
        )
    assert set(forwarded[0]) == {"co2"}
    assert forwarded[0]["co2"].prior == DEFAULT_BC_PRIOR
    selected = [call for call in calls if call["boundary_sensitivity"] is not None]
    assert len(selected) == 1
    assert selected[0]["bc_prior"] == DEFAULT_BC_PRIOR
    assert all(call["bc_prior"] is None for call in calls if call["boundary_sensitivity"] is None)
