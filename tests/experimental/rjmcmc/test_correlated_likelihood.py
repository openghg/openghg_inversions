"""Tests for the independent-site OU correlated likelihood."""

from __future__ import annotations

from math import log, pi
from typing import Callable

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.likelihood import (
    IndependentSiteOUData,
    ou_log_likelihood_numba,
    ou_log_likelihood_numpy,
)


def _dense_log_likelihood(
    residual: np.ndarray,
    observation_sd: np.ndarray,
    observation_time: np.ndarray,
    site_index: np.ndarray,
    mismatch_group_index: np.ndarray,
    site_tau_index: np.ndarray,
    mismatch_sd: np.ndarray,
    correlation_timescale: np.ndarray,
) -> float:
    """Evaluate the declared covariance directly as an independent oracle."""
    n_observations = residual.size
    covariance = np.diag(np.square(observation_sd))
    amplitude = mismatch_sd[mismatch_group_index]
    for row in range(n_observations):
        for column in range(n_observations):
            if site_index[row] == site_index[column]:
                tau_index = site_tau_index[site_index[row]]
                correlation = np.exp(
                    -abs(observation_time[row] - observation_time[column]) / correlation_timescale[tau_index]
                )
                covariance[row, column] += amplitude[row] * amplitude[column] * correlation
    sign, log_determinant = np.linalg.slogdet(covariance)
    assert sign == 1.0
    quadratic = float(residual @ np.linalg.solve(covariance, residual))
    return -0.5 * (n_observations * log(2.0 * pi) + log_determinant + quadratic)


def _interleaved_case() -> tuple[np.ndarray, ...]:
    """Return an irregular case with changing groups and a shared timescale."""
    return (
        np.array([0.4, -0.8, 1.1, 0.2, -0.5, 0.7]),
        np.array([0.3, 0.7, 0.4, 0.8, 0.5, 0.6]),
        np.array([2.0, 0.0, 0.0, 1.0, 3.0, 4.0]),
        np.array([1, 0, 1, 0, 0, 0]),
        np.array([1, 0, 0, 1, 0, 1]),
        np.array([0, 0]),
        np.array([0.9, 1.7]),
        np.array([1.3]),
    )


def _data_from_case(case: tuple[np.ndarray, ...]) -> IndependentSiteOUData:
    """Build static likelihood data from the common test tuple."""
    _, observation_sd, observation_time, site, group, site_tau, _, _ = case
    return IndependentSiteOUData(observation_sd, observation_time, site, group, site_tau)


def test_ou_likelihood_matches_dense_covariance_for_interleaved_irregular_data() -> None:
    """The O(N) recursion should equal a dense D + M Q M calculation."""
    case = _interleaved_case()
    residual, observation_sd, time, site, group, site_tau, mismatch_sd, tau = case
    data = _data_from_case(case)

    expected = _dense_log_likelihood(
        residual,
        observation_sd,
        time,
        site,
        group,
        site_tau,
        mismatch_sd,
        tau,
    )

    assert ou_log_likelihood_numpy(residual, data, mismatch_sd, tau) == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-12,
    )
    np.testing.assert_array_equal(data.observation_order, [1, 3, 4, 5, 2, 0])
    with pytest.raises(ValueError, match="read-only"):
        data.observation_order[0] = 0


def test_likelihood_is_invariant_to_observation_permutation() -> None:
    """Permuting interleaved inputs should not alter the Gaussian density."""
    case = _interleaved_case()
    residual, observation_sd, time, site, group, site_tau, mismatch_sd, tau = case
    expected = ou_log_likelihood_numpy(residual, _data_from_case(case), mismatch_sd, tau)
    permutation = np.array([4, 2, 0, 5, 1, 3])
    permuted_case = (
        residual[permutation],
        observation_sd[permutation],
        time[permutation],
        site[permutation],
        group[permutation],
        site_tau,
        mismatch_sd,
        tau,
    )

    actual = ou_log_likelihood_numpy(
        permuted_case[0],
        _data_from_case(permuted_case),
        mismatch_sd,
        tau,
    )

    assert actual == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_distinct_and_shared_site_timescale_indices_match_dense_covariance() -> None:
    """Several sites mapped to shared and distinct tau values need an independent oracle."""
    residual = np.array([0.2, -0.5, 0.8, 1.1, -0.7, 0.4, 0.9])
    observation_sd = np.array([0.3, 0.4, 0.25, 0.6, 0.5, 0.35, 0.45])
    time = np.array([0.0, 1.7, 0.2, 3.4, 2.1, 0.8, 4.9])
    site = np.array([0, 2, 1, 0, 2, 1, 2])
    group = np.array([0, 1, 2, 1, 0, 2, 1])
    site_tau = np.array([0, 1, 0])
    mismatch_sd = np.array([0.7, 1.4, 0.9])
    tau = np.array([2.3, 0.65])
    data = IndependentSiteOUData(observation_sd, time, site, group, site_tau)

    expected = _dense_log_likelihood(
        residual,
        observation_sd,
        time,
        site,
        group,
        site_tau,
        mismatch_sd,
        tau,
    )

    assert ou_log_likelihood_numpy(residual, data, mismatch_sd, tau) == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-12,
    )
    assert ou_log_likelihood_numba(residual, data, mismatch_sd, tau) == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-12,
    )


def test_mismatch_group_changes_do_not_reset_site_ou_state() -> None:
    """Changing OU amplitude must preserve the latent same-site correlation."""
    residual = np.array([0.3, -1.1, 0.8, 1.4])
    observation_sd = np.array([0.2, 0.4, 0.3, 0.5])
    time = np.array([0.0, 0.4, 1.1, 2.0])
    site = np.zeros(4, dtype=np.int64)
    group = np.array([0, 1, 0, 1])
    site_tau = np.array([0])
    mismatch_sd = np.array([0.25, 2.0])
    tau = np.array([3.0])
    data = IndependentSiteOUData(observation_sd, time, site, group, site_tau)

    expected = _dense_log_likelihood(
        residual,
        observation_sd,
        time,
        site,
        group,
        site_tau,
        mismatch_sd,
        tau,
    )
    actual = ou_log_likelihood_numpy(residual, data, mismatch_sd, tau)

    assert actual == pytest.approx(expected, rel=0.0, abs=1e-12)
    reset_covariance = np.diag(np.square(observation_sd) + np.square(mismatch_sd[group]))
    same_group = group[:, None] == group[None, :]
    correlation = np.exp(-np.abs(time[:, None] - time[None, :]) / tau[0])
    loading = mismatch_sd[group]
    reset_covariance += same_group * correlation * loading[:, None] * loading[None, :] - np.diag(
        np.square(loading)
    )
    reset_sign, reset_logdet = np.linalg.slogdet(reset_covariance)
    assert reset_sign == 1.0
    reset_value = -0.5 * (
        residual.size * log(2.0 * pi) + reset_logdet + residual @ np.linalg.solve(reset_covariance, residual)
    )
    assert abs(actual - reset_value) > 1e-3


def test_sites_are_independent_when_they_share_a_timescale() -> None:
    """Sites sharing tau should retain separate latent Kalman states."""
    case = _interleaved_case()
    residual, observation_sd, time, site, group, _, mismatch_sd, tau = case
    joint_data = _data_from_case(case)
    joint = ou_log_likelihood_numpy(residual, joint_data, mismatch_sd, tau)
    separate = 0.0
    for selected_site in (0, 1):
        selected = site == selected_site
        site_data = IndependentSiteOUData(
            observation_sd[selected],
            time[selected],
            np.zeros(int(selected.sum()), dtype=np.int64),
            group[selected],
            np.array([0]),
        )
        separate += ou_log_likelihood_numpy(residual[selected], site_data, mismatch_sd, tau)

    assert joint == pytest.approx(separate, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("time", "tau"),
    [
        (np.array([2.5]), np.array([0.7])),
        (np.array([0.0, 1_000.0]), np.array([0.1])),
    ],
)
def test_singleton_and_long_gap_cases_match_dense_covariance(
    time: np.ndarray,
    tau: np.ndarray,
) -> None:
    """Singleton sites and underflowed long-gap correlations remain normalized."""
    n_observations = time.size
    residual = np.linspace(-0.6, 0.9, n_observations)
    observation_sd = np.linspace(0.3, 0.5, n_observations)
    site = np.zeros(n_observations, dtype=np.int64)
    group = np.zeros(n_observations, dtype=np.int64)
    site_tau = np.array([0])
    mismatch_sd = np.array([1.2])
    data = IndependentSiteOUData(observation_sd, time, site, group, site_tau)

    expected = _dense_log_likelihood(
        residual,
        observation_sd,
        time,
        site,
        group,
        site_tau,
        mismatch_sd,
        tau,
    )

    assert ou_log_likelihood_numpy(residual, data, mismatch_sd, tau) == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ({"observation_sd": [0.2, 0.0]}, "observation_sd"),
        ({"observation_sd": [0.2, np.inf]}, "observation_sd"),
        ({"observation_time": [0.0, np.nan]}, "observation_time"),
        ({"site_index": [0, 0.5]}, "site_index"),
        ({"site_index": [0, 2]}, "site_index"),
        ({"mismatch_group_index": [0, -1]}, "mismatch_group_index"),
        ({"mismatch_group_index": [0, 2]}, "mismatch_group_index"),
        ({"site_tau_index": [0, 2]}, "site_tau_index"),
    ],
)
def test_static_data_rejects_invalid_values(
    replacement: dict[str, list[float]],
    match: str,
) -> None:
    """Static likelihood data should reject malformed numerical supports."""
    arguments: dict[str, object] = {
        "observation_sd": [0.2, 0.3],
        "observation_time": [0.0, 1.0],
        "site_index": [0, 1],
        "mismatch_group_index": [0, 1],
        "site_tau_index": [0, 1],
    }
    arguments.update(replacement)

    with pytest.raises(ValueError, match=match):
        IndependentSiteOUData(**arguments)


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        (([0.2], [0.0, 1.0], [0], [0], [0]), "one value per observation"),
        (([0.2, 0.3], [0.0, 0.0], [0, 0], [0, 0], [0]), "strictly increasing"),
        (([0.2, 0.3], [0.0, 1.0], [0, 1], [0, 1], [0]), "one value per site"),
        (([], [], [], [], []), "at least one observation"),
    ],
)
def test_static_data_rejects_invalid_shapes_and_repeated_site_times(
    arguments: tuple[list[float], ...],
    match: str,
) -> None:
    """Static data should require aligned observations and unique site times."""
    with pytest.raises(ValueError, match=match):
        IndependentSiteOUData(*arguments)


@pytest.mark.parametrize("backend", [ou_log_likelihood_numpy, ou_log_likelihood_numba])
@pytest.mark.parametrize(
    ("residual", "mismatch_sd", "tau", "match"),
    [
        ([0.2], [0.8, 1.1], [1.0], "residual"),
        ([0.2, np.nan], [0.8, 1.1], [1.0], "residual"),
        ([0.2, 0.3], [0.8], [1.0], "mismatch_sd"),
        ([0.2, 0.3], [0.8, 0.0], [1.0], "mismatch_sd"),
        ([0.2, 0.3], [0.8, 1.1], [np.inf], "correlation_timescale"),
        ([0.2, 0.3], [0.8, 1.1], [-1.0], "correlation_timescale"),
    ],
)
def test_likelihood_rejects_invalid_dynamic_values(
    backend: Callable[..., float],
    residual: list[float],
    mismatch_sd: list[float],
    tau: list[float],
    match: str,
) -> None:
    """Both public backends should enforce dynamic parameter support."""
    data = IndependentSiteOUData([0.3, 0.4], [0.0, 1.0], [0, 0], [0, 1], [0])

    with pytest.raises(ValueError, match=match):
        backend(residual, data, mismatch_sd, tau)


@pytest.mark.parametrize("seed", [5, 17, 103])
def test_ou_likelihood_backends_match_for_seeded_irregular_cases(seed: int) -> None:
    """NumPy and Numba should agree across varied shared-tau OU cases."""
    rng = np.random.default_rng(seed)
    n_sites = 4
    observations_per_site = 5
    site = np.repeat(np.arange(n_sites), observations_per_site)
    time = np.concatenate([np.cumsum(rng.uniform(0.1, 4.0, observations_per_site)) for _ in range(n_sites)])
    group = np.tile(np.array([0, 1, 2, 0, 1]), n_sites)
    permutation = rng.permutation(site.size)
    site = site[permutation]
    time = time[permutation]
    group = group[permutation]
    observation_sd = rng.uniform(0.05, 1.2, site.size)
    residual = rng.normal(size=site.size)
    site_tau = np.array([0, 1, 0, 1])
    mismatch_sd = np.array([0.3, 1.1, 2.4])
    tau = np.array([0.2, 8.0])
    data = IndependentSiteOUData(observation_sd, time, site, group, site_tau)

    numpy_value = ou_log_likelihood_numpy(residual, data, mismatch_sd, tau)
    numba_value = ou_log_likelihood_numba(residual, data, mismatch_sd, tau)

    assert numba_value == pytest.approx(numpy_value, rel=0.0, abs=1e-12)
