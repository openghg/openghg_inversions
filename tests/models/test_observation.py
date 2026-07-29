"""Tests for coherent transformed observation distributions."""

import numpy as np
import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.configdefaults import config
from pytensor.gradient import grad
from scipy.stats import johnsonsu, norm

from openghg_inversions.models._observation import (
    _mean_centered_johnson_su_logp,
    _mean_centered_johnson_su_random,
)


def _floatx(value: object) -> np.ndarray:
    """Convert numerical test inputs to the project's configured PyTensor dtype."""
    return np.asarray(value, dtype=config.floatX)


def _transformed_mean(u: np.ndarray, error: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return the transformed Normal location that mean-centres original-scale draws."""
    return np.arcsinh((sigma * u / error) * np.exp(-0.5 * sigma**2)) / sigma


def test_mean_centered_johnson_su_logp_matches_manual_change_of_variables() -> None:
    """The logp includes the transformed Normal density and exact Jacobian."""
    value = _floatx([-0.7, 2.1, 5.3])
    baseline = _floatx([-1.0, 0.4, 2.0])
    u = _floatx([0.2, 1.1, 2.5])
    error = _floatx([0.3, 0.5, 0.8])
    sigma = _floatx([0.2, 0.4, 0.7])
    enhancement = value - baseline
    transformed_value = np.arcsinh(sigma * enhancement / error) / sigma
    transformed_mean = _transformed_mean(u, error, sigma)
    expected = norm.logpdf(transformed_value, loc=transformed_mean) - 0.5 * np.log(
        error**2 + (sigma * enhancement) ** 2
    )

    actual = _mean_centered_johnson_su_logp(
        pt.as_tensor_variable(value),
        pt.as_tensor_variable(baseline),
        pt.as_tensor_variable(u),
        pt.as_tensor_variable(error),
        pt.as_tensor_variable(sigma),
    ).eval()

    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-6)


def test_mean_centered_johnson_su_matches_scipy_distribution() -> None:
    """The transformed density agrees with SciPy's normalized Johnson-SU family."""
    value = _floatx(np.linspace(-8.0, 10.0, 37))
    baseline = _floatx(0.7)
    u = _floatx(1.4)
    error = _floatx(0.6)
    sigma = _floatx(0.35)
    transformed_mean = _transformed_mean(
        u,
        error,
        sigma,
    )
    expected = johnsonsu.logpdf(
        value,
        a=-transformed_mean,
        b=1.0 / sigma,
        loc=baseline,
        scale=error / sigma,
    )

    actual = _mean_centered_johnson_su_logp(
        pt.as_tensor_variable(value),
        pt.as_tensor_variable(baseline),
        pt.as_tensor_variable(u),
        pt.as_tensor_variable(error),
        pt.as_tensor_variable(sigma),
    ).eval()

    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-6)


def test_mean_centered_johnson_su_random_has_expected_shape_and_mean() -> None:
    """Vector random draws preserve draw shape and target baseline-plus-model means."""
    baseline = _floatx([-0.5, 1.0, 2.0])
    u = _floatx([0.2, 1.3, -0.4])
    error = _floatx([0.4, 0.7, 0.3])
    sigma = _floatx([0.2, 0.35, 0.25])

    draws = _mean_centered_johnson_su_random(
        baseline,
        u,
        error,
        sigma,
        rng=np.random.default_rng(814),
        size=(150_000, 3),
    )

    assert draws.shape == (150_000, 3)
    np.testing.assert_allclose(draws.mean(axis=0), baseline + u, atol=0.01)


def test_mean_centered_johnson_su_approaches_normal_at_small_sigma() -> None:
    """Very small positive sigma approaches the Normal likelihood and generator."""
    value = _floatx([-0.3, 1.2, 4.0])
    baseline = _floatx([-0.5, 0.4, 1.5])
    u = _floatx([0.1, 1.0, 2.0])
    error = _floatx([0.2, 0.6, 0.8])
    sigma = _floatx(np.full(3, 1e-8))

    actual_logp = _mean_centered_johnson_su_logp(
        pt.as_tensor_variable(value),
        pt.as_tensor_variable(baseline),
        pt.as_tensor_variable(u),
        pt.as_tensor_variable(error),
        pt.as_tensor_variable(sigma),
    ).eval()
    expected_logp = norm.logpdf(value, loc=baseline + u, scale=error)
    np.testing.assert_allclose(actual_logp, expected_logp, rtol=1e-6, atol=1e-6)

    seed = 513
    actual_draws = _mean_centered_johnson_su_random(
        baseline,
        u,
        error,
        sigma,
        rng=np.random.default_rng(seed),
        size=(4, 3),
    )
    expected_draws = np.random.default_rng(seed).normal(
        loc=baseline + u,
        scale=error,
        size=(4, 3),
    )
    np.testing.assert_allclose(actual_draws, expected_draws, rtol=1e-7, atol=1e-7)


def test_mean_centered_johnson_su_logp_has_finite_gradients() -> None:
    """The transformed logp supplies finite gradients for NUTS parameters."""
    baseline = pt.scalar("baseline", dtype=config.floatX)
    u = pt.scalar("u", dtype=config.floatX)
    error = pt.scalar("error", dtype=config.floatX)
    sigma = pt.scalar("sigma", dtype=config.floatX)
    logp = _mean_centered_johnson_su_logp(
        pt.as_tensor_variable(1.7, dtype=config.floatX),
        baseline,
        u,
        error,
        sigma,
    )
    gradients = grad(logp, [baseline, u, error, sigma])
    gradient_fn = function([baseline, u, error, sigma], gradients)

    actual = gradient_fn(0.3, 1.1, 0.4, 0.35)

    assert np.all(np.isfinite(actual))
