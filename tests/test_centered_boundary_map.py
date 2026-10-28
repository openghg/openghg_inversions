"""Independent numerical oracles for centered and absolute boundary concentrations."""

from collections.abc import Iterator
from decimal import Decimal, localcontext

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from openghg_inversions.rhime.co2.centered_boundary import CenteredBoundaryMap


@pytest.fixture(autouse=True)
def double_precision() -> Iterator[None]:
    with pytensor.config.change_flags(floatX="float64"):
        yield


def test_boundary_map_preserves_multiplicative_model_and_state_derivatives() -> None:
    anomaly = np.array([[0.012, -0.017, 0.031], [-0.025, 0.005, 0.007], [0.011, 0.013, -0.019]])
    mass = np.array([[0.2, 0.1, 0.05], [0.1, 0.4, 0.15], [0.7, 0.05, 0.02]])
    reference = 400.0
    shift = reference * mass.sum(axis=1)
    state = pt.dvector("boundary_state")
    boundary = CenteredBoundaryMap(
        pt.constant(anomaly),
        pt.constant(mass),
        pt.constant(reference),
        pt.constant(shift),
    )
    centered = boundary.centered_concentration(state)
    evaluate = pytensor.function(
        [state],
        [centered, boundary.concentration(state), pt.jacobian(centered, state)],
    )

    # Physical coefficients can include fixed values different from one.
    coefficients = np.array([1.2, 0.7, 1.05])
    actual_centered, actual_absolute, jacobian = evaluate(coefficients)
    absolute_design = anomaly + reference * mass
    expected_absolute = absolute_design @ coefficients
    np.testing.assert_allclose(actual_absolute, expected_absolute, rtol=1e-14)
    np.testing.assert_allclose(actual_centered, expected_absolute - shift, rtol=1e-14, atol=5e-14)
    np.testing.assert_allclose(jacobian, absolute_design, rtol=1e-14)
    assert np.unique(shift).size == mass.shape[0]


@pytest.mark.parametrize("reference", [209300.0, 1e12])
def test_boundary_map_keeps_small_centered_signal_at_large_reference(reference: float) -> None:
    anomaly = np.array([[0.01, -0.02], [0.03, 0.004], [-0.005, 0.002]])
    mass = np.array([[0.2, 0.1], [0.4, 0.25], [0.05, 0.9]])
    coefficients = np.array([1.0 + 1e-12, 1.0 - 2e-13])
    shift = reference * mass.sum(axis=1)
    state = pt.dvector("boundary_state")
    boundary = CenteredBoundaryMap(
        pt.constant(anomaly),
        pt.constant(mass),
        pt.constant(reference),
        pt.constant(shift),
    )
    evaluate = pytensor.function([state], [boundary.centered_concentration(state), boundary.concentration(state)])
    centered, absolute = evaluate(coefficients)

    # Evaluate the original absolute multiplicative model at high precision,
    # then subtract its reference there. Decimal preserves the actual binary
    # input values, including the small deviations of physical states from one.
    expected_centered = []
    expected_absolute = []
    with localcontext() as context:
        context.prec = 80
        c = Decimal.from_float(reference)
        for anomaly_row, mass_row in zip(anomaly, mass, strict=True):
            absolute_row = sum(
                (Decimal.from_float(float(a)) + c * Decimal.from_float(float(m)))
                * Decimal.from_float(float(b))
                for a, m, b in zip(anomaly_row, mass_row, coefficients, strict=True)
            )
            reference_row = c * sum(Decimal.from_float(float(m)) for m in mass_row)
            expected_absolute.append(float(absolute_row))
            expected_centered.append(float(absolute_row - reference_row))

    np.testing.assert_allclose(centered, expected_centered, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(absolute, expected_absolute, rtol=3 * np.finfo(float).eps)
    # Forming the absolute concentration and then subtracting the large shift
    # cannot recover the same small signal; the factored arithmetic must stay.
    lost_signal = np.max(np.abs((absolute - shift) - expected_centered))
    assert lost_signal > (1e-6 if reference == 1e12 else 1e-12)
