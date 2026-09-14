"""Tests for the labelled IID per-site mismatch likelihood.

The implementation contract is pinned to Verification Games
``src/verification_games/rhime_calibration/site_sigma.py`` at commit
``41d061aea153ddc56130694bfa18b7e801fcd9df``. The dense A/B/A fixture comes
from ``tests/test_rhime_site_sigma.py`` at the original model commit
``88f8d4cb21c7eb84b601c26fa51e806ff0bb3ed7``; the low-rank fixture comes from
that test at ``abdf4671e5d7364fabb1c98f42f6982f8d29d4c0``.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from scipy.stats import multivariate_normal

from openghg_inversions.models.coords import get_coord_registry, registered_model
from openghg_inversions.models.site_sigma import add_site_sigma_gaussian_likelihood
from openghg_inversions.observation_error import AggregationError, resolve_aggregation_error
from openghg_inversions.rhime.standard import build_standard_rhime_model


SITE_LABELS = np.array(["A", "B", "A"])
FIXED_SITE_SIGMA = {"A": 0.5, "B": 1.0}
VG_BASE_COVARIANCE = np.array(
    [
        [1.0, 0.2, 0.1],
        [0.2, 1.2, 0.3],
        [0.1, 0.3, 0.9],
    ]
)


def _observations(values: np.ndarray | None = None) -> xr.DataArray:
    """Return an observation vector with the interleaved VG site labels."""
    return xr.DataArray(
        np.array([1.3, 2.0, 0.9]) if values is None else values,
        dims="nmeasure",
        coords={"nmeasure": np.arange(3), "site": ("nmeasure", SITE_LABELS)},
        name="mf",
    )


def _aggregation_error(
    mode: str,
    *,
    covariance: np.ndarray | None = None,
    factor: np.ndarray | None = None,
    diagonal: np.ndarray | None = None,
) -> AggregationError:
    """Build a small validated-by-construction aggregation error fixture."""
    nmeasure = 3
    if mode == "dense":
        assert covariance is not None
        return AggregationError(
            mode="dense",
            marginal_variance=np.diag(covariance),
            covariance=xr.DataArray(
                covariance,
                dims=("nmeasure", "nmeasure_cov"),
                coords={"nmeasure": np.arange(nmeasure), "nmeasure_cov": np.arange(nmeasure)},
            ),
        )
    if mode == "low_rank":
        assert factor is not None and diagonal is not None
        return AggregationError(
            mode="low_rank",
            marginal_variance=np.sum(factor**2, axis=1) + diagonal,
            factor=xr.DataArray(
                factor,
                dims=("nmeasure", "agg_rank"),
                coords={"nmeasure": np.arange(nmeasure), "agg_rank": np.arange(factor.shape[1])},
            ),
            diagonal_variance=xr.DataArray(
                diagonal,
                dims="nmeasure",
                coords={"nmeasure": np.arange(nmeasure)},
            ),
        )
    if mode == "diagonal":
        assert diagonal is not None
        return AggregationError(
            mode="diagonal",
            marginal_variance=diagonal,
            diagonal_variance=xr.DataArray(
                diagonal,
                dims="nmeasure",
                coords={"nmeasure": np.arange(nmeasure)},
            ),
        )
    return AggregationError(mode="none", marginal_variance=np.zeros(nmeasure))


def _build_model(
    *,
    observations: xr.DataArray | None = None,
    observation_error: np.ndarray | None = None,
    aggregation_error: AggregationError | None = None,
    fixed_site_amplitudes: Mapping[str, float] | None = FIXED_SITE_SIGMA,
    site_amplitude_prior: Mapping[str, Any] | None = None,
) -> pm.Model:
    """Build the core likelihood around a fixed mean for direct logp tests."""
    observations = _observations() if observations is None else observations
    observation_error = np.array([0.3, 0.4, 0.2]) if observation_error is None else observation_error
    aggregation_error = _aggregation_error("none") if aggregation_error is None else aggregation_error
    error = xr.DataArray(
        observation_error,
        dims="nmeasure",
        coords={"nmeasure": observations.coords["nmeasure"]},
    )
    with registered_model(coords={"nmeasure": observations.coords["nmeasure"]}) as model:
        mean = pm.Data("mean", np.array([1.0, 1.2, 0.8]), dims="nmeasure")
        add_site_sigma_gaussian_likelihood(
            observations=observations,
            observation_error=error,
            aggregation_error=aggregation_error,
            mean=mean,
            fixed_site_amplitudes=fixed_site_amplitudes,
            site_amplitude_prior=site_amplitude_prior,
        )
    return model


def _likelihood_logp(model: pm.Model) -> float:
    """Evaluate only the observed likelihood contribution at the initial point."""
    return float(model.compile_logp(vars=model.observed_RVs)(model.initial_point()))


def test_fixed_site_sigma_matches_frozen_vg_dense_covariance() -> None:
    """Match the VG A/B/A covariance while preserving every dense off-diagonal."""
    observation_error = np.array([0.3, 0.4, 0.2])
    aggregation_covariance = VG_BASE_COVARIANCE - np.diag(observation_error**2)
    model = _build_model(
        observation_error=observation_error,
        aggregation_error=_aggregation_error("dense", covariance=aggregation_covariance),
    )
    expected_covariance = VG_BASE_COVARIANCE + np.diag([0.25, 1.0, 0.25])
    expected = multivariate_normal.logpdf(
        _observations().values,
        mean=np.array([1.0, 1.2, 0.8]),
        cov=expected_covariance,
    )

    assert _likelihood_logp(model) == pytest.approx(expected, rel=1e-6)
    np.testing.assert_allclose(model["sigma_observation"].eval(), [0.5, 1.0, 0.5])
    np.testing.assert_allclose(model["epsilon"].eval() ** 2, np.diag(expected_covariance))


@pytest.mark.parametrize("mode", ["diagonal", "low_rank"])
def test_aggregation_covariance_and_reported_error_are_counted_once(mode: str) -> None:
    """Match a dense oracle when aggregation error uses a compact representation."""
    observation_error = np.array([0.3, 0.4, 0.2])
    aggregation_diagonal = np.array([0.2, 0.35, 0.25])
    if mode == "diagonal":
        aggregation = _aggregation_error("diagonal", diagonal=aggregation_diagonal)
        aggregation_covariance = np.diag(aggregation_diagonal)
    else:
        factor = np.array([[0.4, 0.1], [0.2, -0.3], [0.5, 0.2]])
        aggregation = _aggregation_error("low_rank", factor=factor, diagonal=aggregation_diagonal)
        aggregation_covariance = factor @ factor.T + np.diag(aggregation_diagonal)
    site_variance = np.array([0.25, 1.0, 0.25])
    expected_covariance = aggregation_covariance + np.diag(observation_error**2 + site_variance)
    model = _build_model(
        observation_error=observation_error,
        aggregation_error=aggregation,
    )

    expected = multivariate_normal.logpdf(
        _observations().values,
        mean=np.array([1.0, 1.2, 0.8]),
        cov=expected_covariance,
    )

    assert _likelihood_logp(model) == pytest.approx(expected, rel=1e-6)
    np.testing.assert_allclose(model["epsilon"].eval() ** 2, np.diag(expected_covariance))


def test_low_rank_logp_matches_the_four_row_vg_fixture() -> None:
    """Match the VG low-rank fixture at commit ``abdf4671e5d7364fabb1c98f42f6982f8d29d4c0``."""
    factor = np.array([[0.4, 0.1], [0.2, -0.3], [0.5, 0.2], [-0.1, 0.6]])
    residual_variance = np.array([0.2, 0.35, 0.25, 0.4])
    mean_values = np.array([1.0, 1.2, 0.9, 1.1])
    observed_values = np.array([1.1, 0.8, 1.3, 1.0])
    observations = xr.DataArray(
        observed_values,
        dims="nmeasure",
        coords={
            "nmeasure": np.arange(4),
            "site": ("nmeasure", ["A", "B", "A", "B"]),
        },
    )
    reported_error = xr.zeros_like(observations)
    aggregation = AggregationError(
        mode="low_rank",
        marginal_variance=np.sum(factor**2, axis=1) + residual_variance,
        factor=xr.DataArray(factor, dims=("nmeasure", "agg_rank")),
        diagonal_variance=xr.DataArray(residual_variance, dims="nmeasure"),
    )
    with registered_model(coords={"nmeasure": np.arange(4)}) as model:
        mean = pm.Data("mean", mean_values, dims="nmeasure")
        add_site_sigma_gaussian_likelihood(
            observations=observations,
            observation_error=reported_error,
            aggregation_error=aggregation,
            mean=mean,
            fixed_site_amplitudes=FIXED_SITE_SIGMA,
        )

    expected_covariance = factor @ factor.T + np.diag(residual_variance + np.array([0.25, 1.0, 0.25, 1.0]))
    expected = multivariate_normal.logpdf(
        observed_values,
        mean=mean_values,
        cov=expected_covariance,
    )

    assert _likelihood_logp(model) == pytest.approx(expected, rel=1e-6)


def test_inferred_sigma_uses_labelled_first_occurrence_site_order() -> None:
    """Expose inferred sigma on stable VG-compatible A/B labels."""
    model = _build_model(
        fixed_site_amplitudes=None,
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
    )

    registry = get_coord_registry(model)
    assert registry is not None
    np.testing.assert_array_equal(registry.original_coords["sigma_site_dim"], ["A", "B"])
    assert model["sigma_site"] in model.free_RVs
    np.testing.assert_array_equal(model["sigma_site_index"].eval(), [0, 1, 0])
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_inferred_site_sigma_likelihood_matches_dense_vg_oracle() -> None:
    """Match the VG dense likelihood at an explicit inferred physical sigma point."""
    observation_error = np.array([0.3, 0.4, 0.2])
    aggregation_covariance = VG_BASE_COVARIANCE - np.diag(observation_error**2)
    model = _build_model(
        observation_error=observation_error,
        aggregation_error=_aggregation_error("dense", covariance=aggregation_covariance),
        fixed_site_amplitudes=None,
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
    )
    point = model.initial_point()
    point["sigma_site_log__"] = np.log([0.5, 1.0])
    expected = multivariate_normal.logpdf(
        _observations().values,
        mean=np.array([1.0, 1.2, 0.8]),
        cov=VG_BASE_COVARIANCE + np.diag([0.25, 1.0, 0.25]),
    )

    actual = float(model.compile_logp(vars=model.observed_RVs)(point))

    assert actual == pytest.approx(expected, rel=1e-6)


def test_stock_pymc_samples_state_and_site_sigma_together() -> None:
    """Exercise bounded joint state and site-sigma sampling without a cached backend."""
    observations = _observations()
    error = xr.DataArray(np.full(3, 0.2), dims="nmeasure")
    with registered_model(coords={"nmeasure": np.arange(3)}):
        state = pm.Normal("state", mu=1.0, sigma=0.2)
        mean = state * pm.Data("design", [1.0, 1.8, 0.7], dims="nmeasure")
        add_site_sigma_gaussian_likelihood(
            observations=observations,
            observation_error=error,
            aggregation_error=_aggregation_error("none"),
            mean=mean,
            site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
        )
        trace = pm.sample(
            draws=5,
            tune=5,
            chains=1,
            cores=1,
            random_seed=20260913,
            progressbar=False,
            compute_convergence_checks=False,
        )

    assert trace.posterior["state"].shape == (1, 5)
    assert trace.posterior["sigma_site"].shape == (1, 5, 2)
    assert np.isfinite(trace.posterior["sigma_site"]).all()


def test_fixed_sigma_is_labelled_constant_data() -> None:
    """Keep fixed site amplitudes inspectable without introducing a free RV."""
    model = _build_model()

    registry = get_coord_registry(model)
    assert registry is not None
    np.testing.assert_array_equal(registry.original_coords["sigma_site_dim"], ["A", "B"])
    assert model["sigma_site"] not in model.free_RVs
    np.testing.assert_allclose(model["sigma_site"].eval(), [0.5, 1.0])


def test_positive_site_sigma_rescues_a_zero_base_covariance() -> None:
    """Allow positive IID site variance to make an exact-zero base nonsingular."""
    model = _build_model(
        observation_error=np.zeros(3),
        fixed_site_amplitudes={"A": 0.5, "B": 1.0},
    )

    assert np.isfinite(_likelihood_logp(model))


def test_fixed_zero_sigma_is_valid_when_reported_error_is_positive() -> None:
    """Accept exact-zero site amplitudes when the complete covariance remains PD."""
    model = _build_model(fixed_site_amplitudes={"A": 0.0, "B": 0.0})

    assert np.isfinite(_likelihood_logp(model))


def test_low_rank_zero_diagonal_rescued_only_by_factor_is_rejected() -> None:
    """Reject the zero-diagonal boundary unsupported by the Woodbury evaluator."""
    factor = np.eye(3)
    aggregation = _aggregation_error("low_rank", factor=factor, diagonal=np.zeros(3))

    with pytest.raises(ValueError, match="low-rank|diagonal|strictly positive"):
        _build_model(
            observation_error=np.zeros(3),
            aggregation_error=aggregation,
            fixed_site_amplitudes={"A": 0.0, "B": 0.0},
        )


def test_standard_recipe_accepts_site_sigma_through_public_likelihood_seam() -> None:
    """Install the VG site-sigma component through the ordinary RHIME builder seam."""
    observations = _observations()
    data = xr.Dataset(
        {
            "mf": observations,
            "mf_error": ("nmeasure", [0.3, 0.4, 0.2]),
        }
    )
    design = xr.DataArray(
        np.ones((3, 1)),
        dims=("nmeasure", "region"),
        coords={"nmeasure": observations.nmeasure, "region": ["r1"]},
    )

    model = build_standard_rhime_model(
        design,
        observations=data["mf"],
        observation_error=data["mf_error"],
        aggregation_error=resolve_aggregation_error(data, "none"),
        use_bc=False,
        x_prior={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        likelihood_builder=add_site_sigma_gaussian_likelihood,
        likelihood_kwargs={"fixed_site_amplitudes": FIXED_SITE_SIGMA},
    )

    assert {"y", "epsilon", "sigma_site", "sigma_site_index"} <= set(model.named_vars)
    assert "sigma" not in model.named_vars


def test_fixed_mapping_requires_each_observation_site() -> None:
    """Reject a fixed mapping that cannot be selected onto observations."""
    with pytest.raises(ValueError, match="every coordinate label"):
        _build_model(fixed_site_amplitudes={"A": 0.5})


def test_fixed_mapping_ignores_unused_sites_and_uses_label_order() -> None:
    """Select observed sites by label, independent of mapping insertion order."""
    model = _build_model(fixed_site_amplitudes={"C": 0.2, "B": 1.0, "A": 0.5})

    np.testing.assert_allclose(model["sigma_site"].eval(), [0.5, 1.0])


def test_fixed_mapping_rejects_boolean_amplitudes() -> None:
    """Reject booleans rather than silently treating them as physical amplitudes."""
    with pytest.raises(ValueError, match="numeric|boolean"):
        _build_model(fixed_site_amplitudes={"A": True, "B": 1.0})


@pytest.mark.parametrize(
    "site_amplitude_prior",
    [
        {"pdf": "normal", "mu": 0.0, "sigma": 1.0},
        {"pdf": "uniform", "lower": -1.0, "upper": 1.0},
    ],
)
def test_inferred_sigma_rejects_priors_without_positive_support(
    site_amplitude_prior: Mapping[str, Any],
) -> None:
    """Reject inferred standard-deviation priors that permit negative values."""
    with pytest.raises(ValueError, match="positive|non-negative|support"):
        _build_model(
            fixed_site_amplitudes=None,
            site_amplitude_prior=site_amplitude_prior,
        )


@pytest.mark.parametrize(
    ("fixed_site_amplitudes", "site_amplitude_prior"),
    [
        (None, None),
        (FIXED_SITE_SIGMA, {"pdf": "halfnormal", "sigma": 0.75}),
    ],
)
def test_likelihood_requires_exactly_one_site_sigma_mode(
    fixed_site_amplitudes: Mapping[str, float] | None,
    site_amplitude_prior: Mapping[str, Any] | None,
) -> None:
    """Require callers to choose fixed or inferred site amplitudes explicitly."""
    with pytest.raises(ValueError, match="exactly one|fixed|prior"):
        _build_model(
            fixed_site_amplitudes=fixed_site_amplitudes,
            site_amplitude_prior=site_amplitude_prior,
        )
