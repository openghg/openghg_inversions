"""Tests for the experimental retained-trace xarray boundary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

import openghg_inversions.experimental.rjmcmc as rjmcmc
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
    assert dict(dataset.sizes) == {
        "draw": 2,
        "region_slot": 3,
        "fixed_parameter": 0,
        "mismatch_group": 0,
        "timescale_parameter": 0,
    }
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
    assert dataset.mismatch_sd.dims == ("draw", "mismatch_group")
    assert dataset.mismatch_sd.shape == (2, 0)
    assert dataset.correlation_timescale.dims == ("draw", "timescale_parameter")
    assert dataset.correlation_timescale.shape == (2, 0)
    assert np.all(np.isnan(dataset.eta))
    assert np.all(np.isnan(dataset.zeta))
    assert np.all(np.isnan(dataset.coefficient_prior_mean))
    assert np.all(np.isnan(dataset.coefficient_prior_sd))
    assert not dataset.coefficient_hierarchy_active.item()
    np.testing.assert_array_equal(dataset.log_target, [-4.0, -3.5])
    assert dataset.attrs == metadata
    assert "moves" not in dataset
    assert "accepted" not in dataset
    assert "log_acceptance_ratio" not in dataset


def test_converter_uses_separate_fixed_parameter_dimension() -> None:
    """Always-active coefficients should not be confused with dynamic slots."""
    fixed_coefficients = np.array([[0.8, 1.2], [0.9, 1.1]])

    dataset = sampling_trace_to_dataset(_trace(fixed_coefficients=fixed_coefficients))

    assert dict(dataset.sizes) == {
        "draw": 2,
        "region_slot": 3,
        "fixed_parameter": 2,
        "mismatch_group": 0,
        "timescale_parameter": 0,
    }
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

    assert dict(dataset.sizes) == {
        "draw": 0,
        "region_slot": 3,
        "fixed_parameter": 2,
        "mismatch_group": 0,
        "timescale_parameter": 0,
    }
    assert dataset.k.shape == dataset.log_target.shape == dataset.state_transition.shape == (0,)
    assert dataset.nuclei.shape == dataset.coefficients.shape == dataset.active.shape == (0, 3)
    assert dataset.fixed_coefficients.shape == (0, 2)


def test_converter_labels_inferred_ou_and_shared_hierarchy_parameters() -> None:
    """Optional target parameters should retain their scientific parameterization."""
    trace = replace(
        _trace(),
        mismatch_sd=np.array([[2.0, 3.0], [2.5, 3.5]]),
        correlation_timescale=np.array([[12.0], [18.0]]),
        eta=np.log(np.array([1.0, 1.5])),
        zeta=np.log(np.array([0.8, 1.2])),
        coefficient_hierarchy_active=True,
    )

    dataset = sampling_trace_to_dataset(trace)

    assert dict(dataset.sizes) == {
        "draw": 2,
        "region_slot": 3,
        "fixed_parameter": 0,
        "mismatch_group": 2,
        "timescale_parameter": 1,
    }
    np.testing.assert_array_equal(dataset.mismatch_group, [0, 1])
    np.testing.assert_array_equal(dataset.timescale_parameter, [0])
    np.testing.assert_allclose(dataset.mismatch_sd, [[2.0, 3.0], [2.5, 3.5]])
    np.testing.assert_allclose(dataset.correlation_timescale, [[12.0], [18.0]])
    np.testing.assert_allclose(dataset.eta, np.log([1.0, 1.5]))
    np.testing.assert_allclose(dataset.zeta, np.log([0.8, 1.2]))
    np.testing.assert_allclose(dataset.coefficient_prior_mean, [1.0, 1.5])
    np.testing.assert_allclose(dataset.coefficient_prior_sd, [0.8, 1.2])
    assert dataset.coefficient_hierarchy_active.item()
    assert "arithmetic coefficient-prior mean" in dataset.eta.attrs["long_name"]
    assert "arithmetic coefficient-prior standard deviation" in dataset.zeta.attrs["long_name"]


def test_converter_preserves_optional_widths_for_empty_retention() -> None:
    """An empty retained segment should preserve OU and hierarchy schema widths."""
    trace = SamplingTrace(
        k=np.empty(0, dtype=np.int64),
        nuclei=np.empty((0, 3), dtype=np.int64),
        coefficients=np.empty((0, 3), dtype=np.float64),
        fixed_coefficients=np.empty((0, 0), dtype=np.float64),
        mismatch_sd=np.empty((0, 2), dtype=np.float64),
        correlation_timescale=np.empty((0, 1), dtype=np.float64),
        eta=np.empty(0, dtype=np.float64),
        zeta=np.empty(0, dtype=np.float64),
        coefficient_hierarchy_active=True,
        log_target=np.empty(0, dtype=np.float64),
        moves=np.array(["mismatch_sd"]),
        accepted=np.array([True]),
        log_acceptance_ratio=np.array([0.1]),
        state_transition=np.empty(0, dtype=np.int64),
    )

    dataset = sampling_trace_to_dataset(trace)

    assert dataset.mismatch_sd.shape == (0, 2)
    assert dataset.correlation_timescale.shape == (0, 1)
    assert dataset.eta.shape == dataset.zeta.shape == (0,)
    assert dataset.coefficient_hierarchy_active.item()


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
        (
            replace(_trace(), mismatch_sd=np.array([[1.0], [0.0]])),
            "trace.mismatch_sd must contain finite positive values",
        ),
        (
            replace(_trace(), correlation_timescale=np.array([[1.0], [np.nan]])),
            "trace.correlation_timescale must contain finite positive values",
        ),
        (
            replace(_trace(), eta=np.array([np.inf, np.nan])),
            "trace.eta and trace.zeta must contain only NaN",
        ),
    ],
)
def test_converter_rejects_malformed_trace(trace: SamplingTrace, message: str) -> None:
    """Malformed retained arrays, padding, ordering, and diagnostics should fail closed."""
    with pytest.raises(ValueError, match=message):
        sampling_trace_to_dataset(trace)


def test_converter_rejects_nonfinite_active_hierarchy_coordinates() -> None:
    """An active hierarchy must expose finite eta and zeta at every draw."""
    trace = _trace()
    object.__setattr__(trace, "coefficient_hierarchy_active", True)

    with pytest.raises(ValueError, match="trace.eta must be finite"):
        sampling_trace_to_dataset(trace)


@pytest.mark.parametrize("eta", [1_000.0, -1_000.0])
def test_converter_rejects_unrepresentable_active_hierarchy_moments(eta: float) -> None:
    """Finite log coordinates must map to finite positive arithmetic moments."""
    trace = replace(
        _trace(),
        eta=np.array([eta, eta]),
        zeta=np.zeros(2),
        coefficient_hierarchy_active=True,
    )

    with pytest.raises(ValueError, match="finite positive arithmetic coefficient-prior moments"):
        sampling_trace_to_dataset(trace)


def test_converter_rejects_wrong_input_and_metadata_types() -> None:
    """The public boundary should reject objects that are not its declared inputs."""
    with pytest.raises(TypeError, match="SamplingTrace"):
        sampling_trace_to_dataset(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metadata"):
        sampling_trace_to_dataset(_trace(), metadata=[("name", "value")])  # type: ignore[arg-type]


def test_experimental_package_exports_optional_target_api() -> None:
    """The package boundary should expose the supported OU and hierarchy API."""
    expected = {
        "IndependentSiteOUData",
        "InferredOUErrorModel",
        "SharedLognormalHierarchy",
        "arithmetic_moments_to_log_state",
        "arithmetic_moments_to_lognormal_parameters",
        "log_moments_to_lognormal_parameters",
        "ou_log_likelihood_numba",
        "ou_log_likelihood_numpy",
        "propose_correlation_timescale",
        "propose_mismatch_sd",
        "propose_shared_hierarchy",
        "shared_coefficient_log_prior_numba",
        "shared_coefficient_log_prior_numpy",
        "shared_hyperprior_log_density_numba",
        "shared_hyperprior_log_density_numpy",
        "LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_ID",
        "LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_PROFILE",
        "LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_ID",
        "LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_PROFILE",
    }

    assert expected <= set(rjmcmc.__all__)
    assert all(getattr(rjmcmc, name) is not None for name in expected)
