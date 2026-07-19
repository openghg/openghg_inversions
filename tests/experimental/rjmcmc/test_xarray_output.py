"""Tests for the experimental retained-trace xarray boundary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.experimental.rjmcmc.sampling import SamplingTrace
from openghg_inversions.experimental.rjmcmc.xarray_output import sampling_trace_to_dataset


def _trace(*, fixed_coefficients: np.ndarray | None = None) -> SamplingTrace:
    """Return two sparse retained states and four transition diagnostics."""
    kwargs = {}
    if fixed_coefficients is not None:
        kwargs["fixed_coefficients"] = fixed_coefficients
    return SamplingTrace(
        k=np.array([1, 2], dtype=np.int64),
        nuclei=np.array([[1, -1, -1], [0, 2, -1]], dtype=np.int64),
        coefficients=np.array([[1.5, 0.0, 0.0], [2.0, 3.0, 0.0]]),
        log_target=np.array([-4.0, -3.5]),
        moves=np.array(["coefficient", "birth", "death", "global_move"]),
        accepted=np.array([True, True, False, True]),
        log_acceptance_ratio=np.array([0.2, -0.3, -np.inf, 1.0]),
        state_transition=np.array([5, 9], dtype=np.int64),
        **kwargs,
    )


def test_converter_preserves_sparse_states_padding_and_explicit_metadata() -> None:
    """Retained rows should expose their global transitions and padded slots."""
    metadata = {"profile_name": "tiny-checkerboard", "schema_version": 1}

    dataset = sampling_trace_to_dataset(_trace(), metadata=metadata)

    assert isinstance(dataset, xr.Dataset)
    assert dict(dataset.sizes) == {"draw": 2, "region_slot": 3, "fixed_parameter": 0}
    np.testing.assert_array_equal(dataset.draw, [0, 1])
    np.testing.assert_array_equal(dataset.state_transition, [5, 9])
    np.testing.assert_array_equal(dataset.k, [1, 2])
    np.testing.assert_array_equal(dataset.nuclei, [[1, -1, -1], [0, 2, -1]])
    np.testing.assert_array_equal(dataset.coefficients, [[1.5, 0.0, 0.0], [2.0, 3.0, 0.0]])
    np.testing.assert_array_equal(
        dataset.active,
        [[True, False, False], [True, True, False]],
    )
    assert dataset.fixed_coefficients.dims == ("draw", "fixed_parameter")
    assert dataset.fixed_coefficients.shape == (2, 0)
    np.testing.assert_array_equal(dataset.log_target, [-4.0, -3.5])
    assert dataset.attrs == metadata


def test_converter_uses_separate_fixed_parameter_dimension() -> None:
    """Always-active coefficients should not be confused with dynamic slots."""
    fixed_coefficients = np.array([[0.8, 1.2], [0.9, 1.1]])

    dataset = sampling_trace_to_dataset(_trace(fixed_coefficients=fixed_coefficients))

    assert dict(dataset.sizes) == {"draw": 2, "region_slot": 3, "fixed_parameter": 2}
    np.testing.assert_array_equal(dataset.fixed_parameter, [0, 1])
    np.testing.assert_array_equal(dataset.fixed_coefficients, fixed_coefficients)
    assert dataset.attrs == {}


def test_converter_preserves_shaped_empty_retained_trace() -> None:
    """A warmup-only segment should retain its capacity and fixed width."""
    trace = SamplingTrace(
        k=np.empty(0, dtype=np.int64),
        nuclei=np.empty((0, 3), dtype=np.int64),
        coefficients=np.empty((0, 3), dtype=np.float64),
        fixed_coefficients=np.empty((0, 2), dtype=np.float64),
        log_target=np.empty(0, dtype=np.float64),
        moves=np.array(["coefficient", "fixed_coefficient"]),
        accepted=np.array([False, True]),
        log_acceptance_ratio=np.array([-np.inf, 0.25]),
        state_transition=np.empty(0, dtype=np.int64),
    )

    dataset = sampling_trace_to_dataset(trace)

    assert dict(dataset.sizes) == {"draw": 0, "region_slot": 3, "fixed_parameter": 2}
    assert dataset.k.shape == dataset.log_target.shape == dataset.state_transition.shape == (0,)
    assert dataset.nuclei.shape == dataset.coefficients.shape == dataset.active.shape == (0, 3)
    assert dataset.fixed_coefficients.shape == (0, 2)


@pytest.mark.parametrize(
    ("trace", "message"),
    [
        (replace(_trace(), k=np.array([[1, 2]])), "trace.k"),
        (replace(_trace(), nuclei=np.array([[1, -1], [0, 2], [3, -1]])), "one row"),
        (replace(_trace(), coefficients=np.ones((2, 2))), "same shape"),
        (replace(_trace(), log_target=np.array([-4.0])), "one value"),
        (replace(_trace(), state_transition=np.array([5, 5])), "strictly increasing"),
        (replace(_trace(), k=np.array([1, 4])), "region-slot capacity"),
        (
            replace(
                _trace(),
                nuclei=np.array([[1, 0, -1], [0, 2, -1]]),
            ),
            "Inactive trace.nuclei padding",
        ),
        (
            replace(
                _trace(),
                coefficients=np.array([[1.5, 9.0, 0.0], [2.0, 3.0, 0.0]]),
            ),
            "Inactive trace.coefficients padding",
        ),
        (
            replace(
                _trace(),
                nuclei=np.array([[1, -1, -1], [2, 0, -1]]),
            ),
            "strictly increasing",
        ),
        (replace(_trace(), accepted=np.array([True])), "same length"),
        (
            replace(
                _trace(),
                log_acceptance_ratio=np.array([0.2, np.nan, -np.inf, 1.0]),
            ),
            "must not contain NaN",
        ),
    ],
)
def test_converter_rejects_malformed_trace(trace: SamplingTrace, message: str) -> None:
    """Malformed retained arrays, padding, ordering, and diagnostics should fail closed."""
    with pytest.raises(ValueError, match=message):
        sampling_trace_to_dataset(trace)


def test_converter_rejects_wrong_input_and_metadata_types() -> None:
    """The public boundary should reject objects that are not its declared inputs."""
    with pytest.raises(TypeError, match="SamplingTrace"):
        sampling_trace_to_dataset(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metadata"):
        sampling_trace_to_dataset(_trace(), metadata=[("name", "value")])  # type: ignore[arg-type]
