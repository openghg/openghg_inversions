"""Linked CO2/O2 padding preserves bound boundary contracts."""

import dask
import numpy as np
import pytensor
import pytest
import xarray as xr

from openghg_inversions.models import registered_model
from openghg_inversions.models.boundary import (
    BoundAdditiveSensitivity, BoundCenteredSensitivity, MultiplicativeBoundary,
)
from openghg_inversions.rhime.co2.co2_o2_model import pad_linked_boundary
from boundary_test_utils import bound_linked_model_arguments
from test_co2_o2_additive_boundary import _additive_inputs
from test_co2_o2_fixed_ou import _kwargs, _prepared


@pytest.fixture(autouse=True)
def double_precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _forbid_compute(*_args, **_kwargs):
    raise AssertionError("Padding computed a Dask payload.")


def _bound_additive():
    kwargs = _additive_inputs()
    arguments = bound_linked_model_arguments(kwargs)
    return arguments["boundary"], arguments["observations"]


@pytest.mark.parametrize("channel", ["co2", "o2"])
def test_padding_preserves_the_additive_extension(channel):
    """Rows follow the joint labels; states, weights and reference are untouched."""
    boundaries, observations = _bound_additive()
    bound = boundaries[channel]
    padded = pad_linked_boundary(bound, observations, channel)

    assert BoundAdditiveSensitivity.validate(padded.data) is padded.data
    assert padded.observation_dim == observations.dims[0]
    assert padded.data.indexes[padded.observation_dim].equals(observations.indexes[observations.dims[0]])
    assert padded.data.indexes["bc_region"].equals(bound.data.indexes["bc_region"])
    xr.testing.assert_identical(padded.data.bc_centering_weights, bound.data.bc_centering_weights)
    xr.testing.assert_identical(padded.data.bc_reference, bound.data.bc_reference)
    assert padded.data.attrs == bound.data.attrs

    rows = observations.indexes[observations.dims[0]].get_level_values("species") == channel
    native = padded.data.H_bc_mass.isel({padded.observation_dim: rows})
    np.testing.assert_array_equal(native.values, bound.data.H_bc_mass.values)
    np.testing.assert_array_equal(padded.data.H_bc_mass.isel({padded.observation_dim: ~rows}).values, 0.0)


def test_padded_reference_shift_and_affine_blocks_match_native_rows():
    """The fixed shift and affine designs are the native ones, zero elsewhere."""
    boundaries, observations = _bound_additive()
    rows = observations.indexes[observations.dims[0]].get_level_values("species") == "o2"
    output_dim = str(observations.dims[0])
    bound = boundaries["o2"]
    with registered_model():
        native = bound.add(output_dim=bound.observation_dim, prefix="native_", owner="native")
    with registered_model():
        padded = pad_linked_boundary(bound, observations, "o2").add(output_dim=output_dim, prefix="o2_", owner="o2")

    shift = padded.observation_shift
    np.testing.assert_allclose(shift.isel({output_dim: rows}).values, native.observation_shift.values)
    np.testing.assert_array_equal(shift.isel({output_dim: ~rows}).values, 0.0)
    assert len(padded.blocks) == len(native.blocks)
    for padded_block, native_block in zip(padded.blocks, native.blocks, strict=True):
        design = padded_block.design.isel({output_dim: rows})
        np.testing.assert_allclose(design.values, native_block.design.values)
        np.testing.assert_array_equal(padded_block.design.isel({output_dim: ~rows}).values, 0.0)


def test_padding_keeps_lazy_centered_multiplicative_transport_lazy():
    """Padding is a labelled transformation; it does not compute payloads."""
    kwargs = _additive_inputs()
    kwargs["boundary_operands"] = {name: data.chunk() for name, data in kwargs["boundary_operands"].items()}
    for name in ("bc_centering_weights", "bc_mean_shift_prior", "bc_anomaly_scale"):
        kwargs.pop(name)
    kwargs["bc_prior"] = {"co2": {"pdf": "lognormal", "mu": 0.0, "sigma": 0.1}}
    arguments = bound_linked_model_arguments(kwargs)
    bound = arguments["boundary"]["co2"]
    with dask.config.set(scheduler=_forbid_compute):
        padded = pad_linked_boundary(bound, arguments["observations"], "co2")
        assert BoundCenteredSensitivity.validate(padded.data) is padded.data
    assert dask.is_dask_collection(padded.data.H_bc_anomaly)


def test_padding_supports_absolute_multiplicative_transport():
    """Absolute H_bc keeps its own state labels and is padded as rows only."""
    kwargs = _kwargs(_prepared())
    sensitivity = kwargs["co2_sensitivity"]
    dim = str(sensitivity.dims[0])
    absolute = xr.DataArray(
        np.ones((sensitivity.sizes[dim], 2)), dims=(dim, "bc_state"),
        coords={dim: sensitivity[dim], "bc_state": ["a", "b"]}, attrs={"units": "ppm"},
    )
    arguments = bound_linked_model_arguments({
        **kwargs, "boundary_sensitivity": {"co2": absolute}, "bc_prior": {"co2": {"pdf": "lognormal", "mu": 0.0, "sigma": 0.1}},
    })
    bound = arguments["boundary"]["co2"]
    assert isinstance(bound, MultiplicativeBoundary)
    padded = pad_linked_boundary(bound, arguments["observations"], "co2")
    assert padded.data.H_bc.dims == (arguments["observations"].dims[0], "bc_state")
    assert padded.data.indexes["bc_state"].equals(bound.data.indexes["bc_state"])


def test_consumers_trust_contracts_and_compute_numerical_results_once(monkeypatch):
    """Consumers trust structure and compute each numerical result once per channel."""
    from openghg_inversions.models.boundary import additive_anomalies, contracts

    kwargs = _additive_inputs()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A trusted consumer revalidated a contract.")

    arguments = bound_linked_model_arguments(kwargs)  # Tests validate the supplied raw fixtures here.
    for contract in (contracts.RawCenteredSensitivity, contracts.BoundCenteredSensitivity, contracts.BoundAdditiveSensitivity):
        monkeypatch.setattr(contract, "validate", staticmethod(forbidden))
    shift_calls = []
    weight_calls = []
    shift = contracts.BoundCenteredSensitivity.reference_shift
    normalize = contracts.normalize_centering_weights
    monkeypatch.setattr(
        contracts.BoundCenteredSensitivity, "reference_shift",
        staticmethod(lambda data: shift_calls.append(1) or shift(data)),
    )
    monkeypatch.setattr(
        additive_anomalies, "normalize_centering_weights",
        lambda weights: weight_calls.append(1) or normalize(weights),
    )

    observations = arguments["observations"]
    with registered_model():
        for channel, bound in arguments["boundary"].items():
            pad_linked_boundary(bound, observations, channel).add(
                output_dim=str(observations.dims[0]), prefix=f"{channel}_", owner=channel,
            )
    assert len(shift_calls) == len(arguments["boundary"])
    assert len(weight_calls) == len(arguments["boundary"])
