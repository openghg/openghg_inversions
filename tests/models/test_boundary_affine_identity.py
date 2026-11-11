"""Every boundary family's affine form equals its likelihood contribution.

Cached likelihoods consume ``fixed_contribution`` and the ``blocks`` while
ordinary likelihoods consume ``likelihood_contribution``; the two must be the
same function of the sampled variables:

    likelihood_contribution = fixed_contribution + sum_j design_j @ coefficients_j
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from openghg_inversions.models import StateActivity, registered_model
from openghg_inversions.models.boundary import (
    AdditiveChoice,
    AnomalyScalingChoice,
    MultiplicativeChoice,
)
from boundary_test_utils import bind_fixture, centered_transport

TIMES = pd.to_datetime(["2020-01-03", "2020-01-20", "2020-02-03", "2020-02-05"])
ROWS = [10, 20, 30, 40]
REGIONS = ["n", "e", "s"]
PRIOR = {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.1, "lower": 0.0}
MEAN_PRIOR = {"pdf": "normal", "mu": 0.0, "sigma": 1.0}


@pytest.fixture(autouse=True)
def float64():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _observations():
    return xr.DataArray(
        [401.0, 402.0, 403.0, 404.0], dims="nmeasure",
        coords={"nmeasure": ROWS, "time": ("nmeasure", TIMES)}, attrs={"units": "ppm"},
    )


def _spatial(values, units):
    return xr.DataArray(
        values, dims=("nmeasure", "bc_region"), coords={"nmeasure": ROWS, "bc_region": REGIONS},
        attrs={"units": units},
    )


def _operands(weights=None):
    rng = np.random.default_rng(3)
    return centered_transport(
        _spatial(rng.normal(0.0, 2.0, (4, 3)), "ppm"),
        _spatial(rng.uniform(0.1, 0.5, (4, 3)), "1"),
        xr.DataArray(400.0, attrs={"units": "ppm"}),
        weights,
    )


def _weights(frequency):
    periods = [TIMES.min()] if frequency is None else pd.to_datetime(["2020-01-01", "2020-02-01"])
    return xr.DataArray(
        np.arange(1.0, 1.0 + 3 * len(periods)).reshape(3, len(periods)), dims=("bc_curtain", "bc_period"),
        coords={"bc_curtain": REGIONS, "bc_period": periods}, attrs={"units": "1"},
    )


CASES = {
    "absolute-multiplicative": lambda: (MultiplicativeChoice(PRIOR), _spatial(np.full((4, 3), 130.0), "ppm")),
    "centered-multiplicative": lambda: (MultiplicativeChoice(PRIOR, frequency="monthly"), _operands()),
    "centered-multiplicative-fixed-states": lambda: (
        # One whole-window period, so the positional mask covers the three curtains.
        MultiplicativeChoice(PRIOR, StateActivity(active=np.array([True, False, True]), fixed_value=1.3)),
        _operands(),
    ),
    "additive-anomalies": lambda: (AdditiveChoice(MEAN_PRIOR, 2.0, frequency="monthly"), _operands(_weights("monthly"))),
    "additive-mean-only": lambda: (AdditiveChoice(MEAN_PRIOR, 0.0), _operands(_weights(None))),
    "additive-period-contrasts-and-means": lambda: (
        AdditiveChoice(MEAN_PRIOR, 2.0, anomalies="period_contrasts", mean_by_period=True, frequency="monthly"),
        _operands(),
    ),
    "additive-independent": lambda: (AdditiveChoice(None, 2.0, anomalies="independent", frequency="monthly"), _operands()),
    "anomaly-scaling": lambda: (
        AnomalyScalingChoice(MEAN_PRIOR, 0.3, reference="whole_window_transport_mean", frequency="monthly"),
        _operands(),
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_affine_blocks_reproduce_likelihood_contribution(case):
    choice, transport = CASES[case]()
    bound = bind_fixture(choice, _observations(), transport)
    with registered_model():
        term = bound.add()
        affine = pt.as_tensor_variable(term.fixed_contribution.transpose("nmeasure").values)
        for block in term.blocks:
            affine = affine + pt.dot(block.design.transpose("nmeasure", ...).values, block.coefficients)
        likelihood, reconstructed = pm.draw([term.likelihood_contribution, affine], draws=5, random_seed=7)

    assert term.blocks
    np.testing.assert_allclose(reconstructed, likelihood, rtol=1e-12, atol=1e-9)
    if term.observation_shift is not None:
        assert term.observation_shift.dims == ("nmeasure",)
