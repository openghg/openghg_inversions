"""Graph checks of the exact Gaussian estimator: Normal priors and an affine mean."""

import numpy as np
import pymc as pm
import pytest

from openghg_inversions.inference.exact_gaussian import affine_maps, normal_variables, solve_linear_gaussian_model


def test_affine_map_and_posterior_follow_the_graph(double_precision):
    with pm.Model() as model:
        x = pm.Normal("x", mu=[1.0, -2.0], sigma=[0.5, 2.0])
        offset = pm.Normal("c", mu=3.0, sigma=1.0)
        mean = pm.Deterministic("mean", 4.0 + np.array([[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]]) @ x + offset)
        pm.Normal("y", mean, 0.3, observed=np.array([5.0, 1.0, 9.0]))
    variables = normal_variables(model)
    assert variables.names == ("x", "c")
    np.testing.assert_allclose(variables.prior_mean, [1.0, -2.0, 3.0])
    np.testing.assert_allclose(variables.prior_sd, [0.5, 2.0, 1.0])
    (mapped,) = affine_maps(model, [mean], variables, rng=np.random.default_rng(0))
    np.testing.assert_allclose(mapped.offset, [4.0, 4.0, 4.0])
    np.testing.assert_allclose(mapped.matrix, [[1.0, 2.0, 1.0], [0.0, 1.0, 1.0], [3.0, -1.0, 1.0]])

    y = np.array([5.0, 1.0, 9.0])
    _, posterior = solve_linear_gaussian_model(
        model, likelihood_mean=mean, observations=y, noise_factor=0.3 * np.eye(3), rng=np.random.default_rng(1),
    )
    design, prior = mapped.matrix, np.diag(variables.prior_sd**2)
    gain = np.linalg.solve(design @ prior @ design.T + 0.09 * np.eye(3), design @ prior).T
    np.testing.assert_allclose(posterior.mean, variables.prior_mean + gain @ (y - 4.0 - design @ variables.prior_mean))
    np.testing.assert_allclose(posterior.covariance, prior - gain @ design @ prior, atol=1e-14)


def test_non_normal_and_hierarchical_priors_are_named(double_precision):
    with pm.Model() as model:
        scale = pm.HalfNormal("scale", 1.0)
        pm.Normal("level", 0.0, scale)
        pm.TruncatedNormal("bc", mu=1.0, sigma=0.1, lower=0.0)
    with pytest.raises(ValueError, match=r"'scale'.*'level' \(hierarchical Normal\).*'bc'"):
        normal_variables(model)


def test_nonlinear_mean_is_refused(double_precision):
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 1.0, shape=2)
        mean = pm.Deterministic("mean", pm.math.exp(x))
    with pytest.raises(ValueError, match="not affine"):
        affine_maps(model, [mean], normal_variables(model), rng=np.random.default_rng(2))


@pytest.mark.parametrize("extra", ["potential", "observed"])
def test_potentials_and_extra_observations_are_refused(double_precision, extra):
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 1.0, shape=2)
        mean = 2.0 * x
        pm.Normal("y", mean, 1.0, observed=np.zeros(2))
        if extra == "potential":
            pm.Potential("penalty", -(x**2).sum())
        else:
            pm.Normal("z", x, 1.0, observed=np.zeros(2))
    with pytest.raises(ValueError, match="Potential|exactly one observed"):
        solve_linear_gaussian_model(model, likelihood_mean=mean, observations=np.zeros(2),
                                    noise_factor=np.eye(2), rng=np.random.default_rng(3))
