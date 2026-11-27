"""Dense checks for inferred OU innovations with two kinds of cross covariance."""

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from openghg_inversions.models.paired_error import PairedErrorCorrelation
from openghg_inversions.models.paired_ou_innovations import prepare_paired_ou_innovations


@pytest.mark.parametrize(
    ("rank", "same_time", "persistent", "first_amplitude"),
    [(4, -0.97, 0.0, 0.7), (4, 0.0, 0.75, 0.0), (4, 0.83, -0.62, 0.7), (0, 0.0, 0.0, 0.7)],
)
def test_innovations_match_dense(rank, same_time, persistent, first_amplitude):
    """Compare normalized density, physical amplitude derivatives, and state cache."""
    rng = np.random.default_rng(52)
    # Scrambled, irregular sampling with an isolated group and more than one same-time pair.
    hour = np.array([7, 0, 2.5, 0.5, 0, 7, 1, 5, 2.5])
    group = np.array([1, 0, 1, 2, 1, 0, 1, 0, 0])
    tau = np.array([2.3, 2.3 if persistent else 4.8, 7.0])
    amplitude = np.array([first_amplitude, 1.1, 0.4])
    factor = rng.normal(scale=0.2, size=(len(hour), rank))
    raw_residual = rng.normal(size=len(hour))
    design = rng.normal(size=(len(hour), 2))
    state = rng.normal(size=design.shape[1])
    y = raw_residual - design @ state
    sd = rng.uniform(0.4, 0.9, size=len(hour))
    diagonal = sd**2 + 0.04
    pairs = PairedErrorCorrelation([4, 0, 2], [1, 5, 8], [same_time] * 3)
    prepared = prepare_paired_ou_innovations(
        factor,
        diagonal,
        group,
        hour,
        tau,
        paired_error=pairs,
        model_error_sd=sd,
        persistent_correlation={(1, 0): persistent} if persistent else None,
    )

    row_amplitude = amplitude[group]
    stationary = np.eye(len(tau))
    stationary[0, 1] = stationary[1, 0] = persistent
    lag = abs(hour[:, None] - hour)
    kernel = stationary[group[:, None], group] * np.exp(-lag / tau[group, None])
    covariance = (
        factor @ factor.T
        + np.diag(diagonal)
        + pairs.covariance(sd)
        + row_amplitude[:, None] * row_amplitude * kernel
    )
    chol = cho_factor(covariance, lower=True)
    solved = cho_solve(chol, y)
    precision = cho_solve(chol, np.eye(len(hour)))
    logdet = 2 * np.log(chol[0].diagonal()).sum()
    expected_logp = -0.5 * (len(y) * np.log(2.0 * np.pi) + logdet + y @ solved)
    score = 0.5 * (np.outer(solved, solved) - precision)
    grad = []
    for index in range(len(tau)):
        selected = group == index
        one_side = selected[:, None] * row_amplitude * kernel
        grad.append(np.sum(score * (one_side + one_side.T)))

    actual_logp, actual_grad = prepared.likelihood(y, amplitude)
    np.testing.assert_allclose(actual_logp, expected_logp, atol=1e-11, rtol=0.0)
    np.testing.assert_allclose(actual_grad, grad, atol=1e-11, rtol=0.0)
    assert prepared.likelihood(y, amplitude, gradients=False)[0] == pytest.approx(expected_logp)
    cache = prepared.state_cache(raw_residual, design, amplitude)
    assert cache.log_likelihood(state) == pytest.approx(expected_logp)
    np.testing.assert_allclose(cache.gradient(state), design.T @ solved, atol=1e-11, rtol=0.0)


def test_pairing_and_proposal_validation():
    """Reject incompatible site models, and return a rejection density for invalid proposals."""
    group = [0, 1, 2]
    times = [0.0, 0.5, 0.5]
    common = (np.ones((3, 1)), np.ones(3), group, times)
    with pytest.raises(ValueError, match="common decay"):
        prepare_paired_ou_innovations(*common, [1, 2, 3], persistent_correlation={(0, 1): 0.3})
    with pytest.raises(ValueError, match="at most two"):
        prepare_paired_ou_innovations(*common, 1.0, persistent_correlation={(0, 1): 0.3, (1, 2): 0.3})
    with pytest.raises(ValueError, match="same time"):
        prepare_paired_ou_innovations(
            *common, 1.0, paired_error=PairedErrorCorrelation([0], [1], [0.5]), model_error_sd=np.ones(3)
        )
    prepared = prepare_paired_ou_innovations(*common, 1.0)
    logp, gradient = prepared.likelihood(np.zeros(3), [1, 1, -1])
    assert logp == -np.inf
    np.testing.assert_array_equal(gradient, 0.0)
