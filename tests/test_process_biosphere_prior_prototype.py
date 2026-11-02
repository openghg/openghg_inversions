"""Process-prior equation and graph checks, without posterior sampling."""

import numpy as np
import pytest
import pytensor
from numpy.polynomial.hermite import hermgauss
import xarray as xr

from prototypes.process_biosphere_prior import (
    BASE_LOG_SD,
    MODES,
    build_process_biosphere_model,
    process_prior_covariance,
)


def _case():
    return xr.Dataset(
        {
            "H": (("nmeasure", "flux_state"), [
                [-3.0, -1.0, 2.0, 0.5, 0.7, 0.3],
                [-1.0, -2.0, 0.5, 1.0, 0.2, 0.4],
                [0.0, 0.0, 1.0, 0.6, 0.3, 0.2],
            ]),
            "mf": ("nmeasure", [-0.3, -0.9, 2.1]),
            "intercept": ("nmeasure", [0.2, -0.1, 0.0]),
        },
        coords={
            "nmeasure": [0, 1, 2],
            "flux_state": ["G-n", "G-s", "Ra-n", "Ra-s", "Rh-n", "Rh-s"],
            "source": ("flux_state", ["GPP", "GPP", "Ra", "Ra", "Rh", "Rh"]),
            "group": ("flux_state", ["north", "south"] * 3),
        },
    )


@pytest.fixture(autouse=True)
def double_precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def test_marginal_widths_pairing_and_signed_net_variance():
    data = _case()
    independent, linked, hierarchical = (
        process_prior_covariance(data.H, mode=mode) for mode in MODES
    )
    for covariance in (independent, linked, hierarchical):
        np.testing.assert_allclose(np.diag(covariance), 0.2**2, rtol=1e-12)
    for covariance in (linked, hierarchical):
        assert covariance[0, 2] > 0
        assert covariance[0, 3] == covariance[0, 4] == covariance[0, 1] == 0
        net = np.array([-100, -80, 55, 40, 45, 40])
        assert net @ covariance @ net < net @ independent @ net
        assert covariance[0, 2] / covariance[0, 0] != pytest.approx(0.6)

    # Independent quadrature over the HalfNormal scale verifies its mixture
    # covariance, rather than treating it as a single LogNormal distribution.
    nodes, weights = hermgauss(60)
    scale = np.sqrt(-np.expm1(-2 * BASE_LOG_SD**2) / 2)
    s = np.sqrt(2) * scale * np.abs(nodes)
    np.testing.assert_allclose(weights @ np.expm1(s**2) / np.sqrt(np.pi), hierarchical[0, 0])
    np.testing.assert_allclose(weights @ np.expm1(0.6 * s**2) / np.sqrt(np.pi), hierarchical[0, 2])


@pytest.mark.parametrize("mode", MODES)
def test_label_shuffle_forward_signs_and_finite_gradient(mode):
    data = _case().isel(flux_state=[4, 1, 2, 5, 0, 3])
    model, activity = build_process_biosphere_model(
        data.H, observations=data.mf, fixed_prior_contribution=data.intercept, mode=mode,
    )
    assert activity.active.all().item()
    assert not activity.structurally_removed.any().item()
    np.testing.assert_array_equal(activity.source, data.source)
    np.testing.assert_array_equal(activity.fixed_value, 1)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    assert np.isfinite(model.compile_dlogp()(model.initial_point())).all()

    z = np.array([0.4, -0.2, 0.5, 0.1, -0.6, 0.3])
    inputs = [model.flux_scaling_latent]
    arguments = [z]
    s = BASE_LOG_SD
    if mode == "hierarchical":
        inputs.append(model.shared_log_sd)
        s = 0.3
        arguments.append(s)
    state, prediction = model.compile_fn(
        [model.flux_scaling, model.modelled_concentration], inputs=inputs, point_fn=False,
    )(*arguments)
    log_covariance = np.log1p(process_prior_covariance(data.H, mode="linked" if mode == "hierarchical" else mode))
    correlation = log_covariance / BASE_LOG_SD**2
    np.testing.assert_allclose(state, np.exp(-s**2 / 2 + s * np.linalg.cholesky(correlation) @ z))
    np.testing.assert_allclose(prediction, data.intercept + data.H.values @ state)
    np.testing.assert_allclose(model.epsilon.eval(), 1)

    order = [4, 1, 2, 5, 0, 3]
    original = process_prior_covariance(_case().H, mode=mode)
    np.testing.assert_allclose(process_prior_covariance(data.H, mode=mode), original[np.ix_(order, order)])


def test_rejects_ambiguous_process_labels_and_invisible_states():
    data = _case()
    invalid = data.H.assign_coords(source=("flux_state", ["TER"] * 6))
    with pytest.raises(ValueError, match="TER"):
        process_prior_covariance(invalid)
    invalid = data.H.assign_coords(group=("flux_state", ["same"] * 6))
    with pytest.raises(ValueError, match="pair must be unique"):
        process_prior_covariance(invalid)
    invisible = data.H.copy(deep=True)
    invisible.loc[{"flux_state": "G-n"}] = 0
    with pytest.raises(ValueError, match="conditional prior reconstruction"):
        build_process_biosphere_model(
            invisible, observations=data.mf, fixed_prior_contribution=data.intercept,
        )
    with pytest.raises(ValueError, match="finite prior variance"):
        process_prior_covariance(data.H, mode="hierarchical", hyper_scale=1.0)
